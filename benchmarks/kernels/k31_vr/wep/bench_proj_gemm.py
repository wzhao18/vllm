# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3.1 projection GEMMs at the per-rank shapes of the wide-EP decode
topologies on VR (SM107): DEP (TP1), DCP4 hybrids (TP4 attention, SP MoE) and
DCP8 hybrids (TP8), BF16 vs MXFP8, with the activation quantization split out.

Per (shape, rows) it times, in a CUDA graph with rotating weight copies (> 2x
L2, so every call streams the weight from HBM):
  bf16       F.linear (cuBLAS)
  actq       FlashInfer mxfp8_quantize of the input (swizzled scales)
  cd_gemm    FlashInfer CuTe DSL mm_mxfp8 on a pre-quantized input
  lt_gemm    cuBLASLt torch._scaled_mm (MXFP8, 1x32 E8M0) on a pre-quantized input
  apply      the online-MXFP8 linear as served on SM107 (KimiK3Mxfp8Kernel:
             quant + CuTe DSL at <= 16 rows, cuBLASLt above)
and the weight-stream floor at 12.5 TB/s (the wide-EP model's T budget) and
13.9 TB/s (measured achievable HBM read).

Sets: tp1 / tp4 / tp8 attention + dense projections at the attention rows,
mlp (shared / latent MoE GEMMs at the SP rows, replicated or TP-sharded), and
pack (packed MoE GEMMs: [shared gate_up ; latent down] along N, [shared down |
latent up] along K).
Usage: python bench_proj_gemm.py --set tp1 [--rows ...] [--out f.json]
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_graph  # noqa: E402

# name: (N, K) per rank
SETS = {
    "tp1": {
        "kda_in_proj": (49376, 7168),
        "kda_o_proj": (7168, 12288),
        "dsa_qkv_a+wkv": (2272, 7168),
        "dsa_q_b+wq_index": (22528, 1536),
        "dsa_g_proj": (12288, 7168),
        "dsa_o_proj": (7168, 12288),
        "dense_gate_up": (67584, 7168),
        "dense_down": (7168, 33792),
    },
    "tp4": {
        "kda_in_proj": (12448, 7168),
        "kda_o_proj": (7168, 3072),
        "dsa_qkv_a+wkv": (2272, 7168),
        "dsa_q_b": (4608, 1536),
        "dsa_wq_index": (4096, 1536),
        "dsa_g_proj": (3072, 7168),
        "dsa_o_proj": (7168, 3072),
        "dense_gate_up": (16896, 7168),
        "dense_down": (7168, 8448),
    },
    "tp8": {
        "kda_in_proj": (6288, 7168),
        "kda_o_proj": (7168, 1536),
        "dsa_qkv_a+wkv": (2272, 7168),
        "dsa_q_b": (2304, 1536),
        "dsa_wq_index": (4096, 1536),
        "dsa_g_proj": (1536, 7168),
        "dsa_o_proj": (7168, 1536),
        "dense_gate_up": (8448, 7168),
        "dense_down": (7168, 4224),
    },
    # MoE-layer GEMMs: replicated (DEP and SP hybrids, rows = MoE tokens/rank)
    # and TP-sharded across the TP group (rows = all the group's tokens).
    "mlp": {
        "shared_gate_up": (12288, 7168),
        "shared_down": (7168, 6144),
        "latent_down": (3584, 7168),
        "latent_up": (7168, 3584),
        "shared_gate_up/tp4": (3072, 7168),
        "shared_down/tp4": (7168, 1536),
        "latent_down/tp4": (896, 7168),
        "latent_up/tp4": (7168, 896),
        "shared_gate_up/tp8": (1536, 7168),
        "shared_down/tp8": (7168, 768),
        "latent_down/tp8": (448, 7168),
        "latent_up/tp8": (7168, 448),
    },
    "pack": {
        "gate_up+latent_down (N-pack)": (12288 + 3584, 7168),
        "shared_down|latent_up (K-pack)": (7168, 6144 + 3584),
        "gate_up+latent_down/tp4": (3072 + 896, 7168),
        "shared_down|latent_up/tp4": (7168, 1536 + 896),
    },
}
ROWS = {
    "tp1": "16,32,64,96,128",
    "tp4": "64,128,256,384,512",
    "tp8": "128,256,512,768,1024",
    "mlp": "16,32,64,96,128,256,384,512,768",
    "pack": "16,32,64,96,128,256,384,512",
}


class _L(torch.nn.Module):
    pass


def make_mxfp8(w_bf16, kernel_cls):
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    N, K = w_bf16.shape
    kern = kernel_cls(Mxfp8LinearLayerConfig(weight_shape=(N, K)))
    layer = _L()
    wq, ws = mxfp8_e4m3_quantize(w_bf16.contiguous())
    layer.register_parameter("weight", torch.nn.Parameter(wq, requires_grad=False))
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(ws, requires_grad=False)
    )
    layer.params_dtype = torch.bfloat16
    layer.input_scale = None
    kern.process_weights_after_loading(layer)
    return kern, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tp1")
    ap.add_argument("--rows", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--rot_bytes", type=float, default=1.2e9)
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.kimi_k3.nvidia.mxfp8 import KimiK3Mxfp8Kernel
    from vllm.utils import flashinfer as vfi

    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(f"device {p.name} cc {p.major}.{p.minor} sms {p.multi_processor_count}",
          flush=True)  # fmt: skip
    rows = [int(r) for r in (args.rows or ROWS[args.set]).split(",")]
    out = []
    torch.manual_seed(0)
    for name, (N, K) in SETS[args.set].items():
        if args.only and args.only not in name:
            continue
        ncopy = max(2, min(16, math.ceil(args.rot_bytes / (N * K * 2))))
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
              for _ in range(ncopy)]  # fmt: skip
        mx = [make_mxfp8(w, KimiK3Mxfp8Kernel) for w in ws]
        e8 = torch.float8_e8m0fnu
        for M in rows:
            xs = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(ncopy)]
            qs = [mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True) for x in xs]
            row = dict(set=args.set, layer=name, N=N, K=K, M=M)
            mxb = N * K * (1 + 1 / 32)
            row["floor12.5"] = round(mxb / 12.5e6, 2)
            row["floor13.9"] = round(mxb / 13.9e6, 2)
            row["floor_bf16"] = round(N * K * 2 / 13.9e6, 2)
            cases = {
                "bf16": [lambda x=x, w=w: F.linear(x, w) for x, w in zip(xs, ws)],
                "actq": [lambda x=x: mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                         for x in xs],  # fmt: skip
                "cd_gemm": [
                    lambda q=q, kl=kl: vfi.mm_mxfp8(q[0], kl[1].weight, q[1],
                                                    kl[1].weight_scale,
                                                    out_dtype=torch.bfloat16,
                                                    backend="cute-dsl")
                    for q, kl in zip(qs, mx)],  # fmt: skip
                "lt_gemm": [
                    lambda q=q, kl=kl: torch._scaled_mm(
                        q[0], kl[1].weight, scale_a=q[1].view(e8),
                        scale_b=kl[1].weight_scale.view(e8), out_dtype=torch.bfloat16)
                    for q, kl in zip(qs, mx)],  # fmt: skip
                "apply": [lambda x=x, kl=kl: kl[0].apply_weights(kl[1], x)
                          for x, kl in zip(xs, mx)],  # fmt: skip
            }
            for b, fns in cases.items():
                try:
                    row[b] = round(time_graph(fns), 2)
                except Exception as e:  # noqa: BLE001
                    row[b] = f"ERR {type(e).__name__}: {str(e)[:100]}"
            g = [row[b] for b in ("cd_gemm", "lt_gemm") if isinstance(row.get(b), float)]
            if g:
                row["best_gemm"] = min(g)
                row["TBs_best"] = round(mxb / (min(g) * 1e-6) / 1e12, 2)
            print(json.dumps(row), flush=True)
            out.append(row)
        del mx, ws
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
