# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3.1 TP1 (DEP) projection GEMMs as NVFP4 W4A4 on VR (SM107): FlashInfer
mm_fp4 backends vs the served MXFP8 GEMM (best of CuTe DSL / cuBLASLt), at the
per-rank rows of the DEP owner points (DEP32 r6 q8 = 48, DEP16 r16 q8 = 128,
DEP16 r28 q8 = 224 rows).

Weights are quantized once (NVFP4: E2M1 + E4M3 per-16 scales + FP32 global);
the activation is pre-quantized (a fused producer writes it), so the times are
GEMM only, in a CUDA graph with rotating weight copies (> 2x L2). Also reports
the standalone NVFP4 activation quantization (vLLM scaled_fp4_quant).
Usage: python bench_nvfp4_gemm.py [--set tp1|mlp] [--rows 48,128,224] [--out f.json]
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_proj_gemm import SETS, make_mxfp8  # noqa: E402
from common import time_graph  # noqa: E402

BACKENDS = ("cutlass", "cudnn", "trtllm", "cute-dsl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tp1")
    ap.add_argument("--rows", default="16,48,128,224")
    ap.add_argument("--only", default="")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--out", default="")
    ap.add_argument("--rot_bytes", type=float, default=1.2e9)
    args = ap.parse_args()

    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.kimi_k3.nvidia.mxfp8 import KimiK3Mxfp8Kernel
    from vllm.utils import flashinfer as vfi

    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(f"device {p.name} cc {p.major}.{p.minor} sms {p.multi_processor_count}",
          flush=True)  # fmt: skip
    rows = [int(r) for r in args.rows.split(",")]
    backends = args.backends.split(",")
    e8 = torch.float8_e8m0fnu
    out = []
    torch.manual_seed(0)
    for name, (N, K) in SETS[args.set].items():
        if (args.only and args.only not in name) or "/tp" in name:
            continue
        ncopy = max(2, min(16, math.ceil(args.rot_bytes / (N * K * 2))))
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
              for _ in range(ncopy)]  # fmt: skip
        mx = [make_mxfp8(w, KimiK3Mxfp8Kernel) for w in ws]
        nv = []
        for w in ws:
            gs = (448.0 * 6.0 / w.abs().max().float()).reshape(1)
            wq, wsf = scaled_fp4_quant(w, gs, is_sf_swizzled_layout=True)
            nv.append((wq, wsf, gs))
        del ws
        for M in rows:
            xs = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(ncopy)]
            qs = [mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True) for x in xs]
            xgs = [(448.0 * 6.0 / x.abs().max().float()).reshape(1) for x in xs]
            fq = [scaled_fp4_quant(x, g, is_sf_swizzled_layout=True) for x, g in zip(xs, xgs)]
            alphas = [(1.0 / (g * w[2])).float() for g, w in zip(xgs, nv)]
            row = dict(set=args.set, layer=name, N=N, K=K, M=M)
            mxb = N * K * (1 + 1 / 32)
            nvb = N * K * (0.5 + 1 / 16)
            row["mx_floor12.5"] = round(mxb / 12.5e6, 2)
            row["nv_floor12.5"] = round(nvb / 12.5e6, 2)
            cases = {
                "mx_cd": [
                    lambda q=q, kl=kl: vfi.mm_mxfp8(q[0], kl[1].weight, q[1],
                                                    kl[1].weight_scale,
                                                    out_dtype=torch.bfloat16,
                                                    backend="cute-dsl")
                    for q, kl in zip(qs, mx)],  # fmt: skip
                "mx_lt": [
                    lambda q=q, kl=kl: torch._scaled_mm(
                        q[0], kl[1].weight, scale_a=q[1].view(e8),
                        scale_b=kl[1].weight_scale.view(e8), out_dtype=torch.bfloat16)
                    for q, kl in zip(qs, mx)],  # fmt: skip
                "nv_actq": [lambda x=x, g=g: scaled_fp4_quant(x, g, is_sf_swizzled_layout=True)
                            for x, g in zip(xs, xgs)],  # fmt: skip
            }
            for b in backends:
                cases[f"nv_{b}"] = [
                    lambda f=f, w=w, a=a, b=b: vfi.flashinfer_scaled_fp4_mm(
                        f[0], w[0], f[1], w[1], a, torch.bfloat16, backend=b)
                    for f, w, a in zip(fq, nv, alphas)]  # fmt: skip
            for b, fns in cases.items():
                try:
                    row[b] = round(time_graph(fns), 2)
                except Exception as e:  # noqa: BLE001
                    row[b] = f"ERR {type(e).__name__}: {str(e)[:120]}"
            # accuracy of each NVFP4 backend vs the dequantized reference (copy 0)
            try:
                ref = vfi.flashinfer_scaled_fp4_mm(
                    fq[0][0], nv[0][0], fq[0][1], nv[0][1], alphas[0], torch.bfloat16,
                    backend="cutlass").float()  # fmt: skip
                for b in backends:
                    if isinstance(row.get(f"nv_{b}"), float):
                        o = vfi.flashinfer_scaled_fp4_mm(
                            fq[0][0], nv[0][0], fq[0][1], nv[0][1], alphas[0],
                            torch.bfloat16, backend=b).float()  # fmt: skip
                        row[f"relerr_{b}"] = round(
                            ((o - ref).norm() / ref.norm()).item(), 5)
            except Exception as e:  # noqa: BLE001
                row["relerr"] = f"ERR {str(e)[:80]}"
            mxg = [row[b] for b in ("mx_cd", "mx_lt") if isinstance(row.get(b), float)]
            nvg = [row[f"nv_{b}"] for b in backends if isinstance(row.get(f"nv_{b}"), float)]
            if mxg:
                row["mx_best"] = min(mxg)
            if nvg:
                row["nv_best"] = min(nvg)
                row["nv_TBs"] = round(nvb / (min(nvg) * 1e-6) / 1e12, 2)
            print(json.dumps(row), flush=True)
            out.append(row)
        del mx, nv
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
