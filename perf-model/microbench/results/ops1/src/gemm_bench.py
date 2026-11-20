"""ops1 GEMM bench: every in-scope model GEMM case at its sharded (batch, M, N, K, dtype).

bf16 (batch 1): cuBLAS (F.linear, x[M,K] @ w[N,K]^T) and the Kimi K3 low-latency CuTe GEMM
  (vllm.models.kimi_k3.nvidia.low_latency_gemm.try_low_latency_gemm, used where its shape table applies).
bf16 (batch > 1, DSA w_uk / w_uv): torch.bmm [b,M,K] x [b,K,N].
fp8 (MXFP8, nvfp4_modelopt KDA / shared): pre-quantized activations (block 32, swizzled UE8M0) x MXFP8 weight;
  cuBLASLt torch._scaled_mm and FlashInfer CuTe DSL mm_mxfp8 (rows <= 256), i.e. both arms of
  KimiK3Mxfp8Kernel's dispatch; activation quant timed separately (quant_us).
Timing: CUPTI per call, cold L2 (weights stream from HBM, as the model assumes), median.
usage: gemm_bench.py cases.json out.jsonl i/n
"""
import json
import math
import sys
import traceback

import torch
import torch.nn.functional as F

from mbc import Out, env_info, log, shard, time_us

cases = shard([c for c in json.load(open(sys.argv[1])) if c["family"] == "gemm"], sys.argv[3])
out = Out(sys.argv[2])
torch.manual_seed(0)
dev = torch.device("cuda")
out.row(kind="env", **env_info())

try:
    from vllm.models.kimi_k3.nvidia.low_latency_gemm import try_low_latency_gemm
except Exception as e:  # noqa: BLE001
    log("no low_latency_gemm", repr(e)[:200])
    try_low_latency_gemm = None
try:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi
except Exception as e:  # noqa: BLE001
    log("no mxfp8 utils", repr(e)[:200])
    mxfp8_e4m3_quantize = None


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-6))


for c in sorted(cases, key=lambda c: c["shape"]["M"] * c["shape"]["N"]):
    s = c["shape"]
    B, M, N, K = s["batch"], s["M"], s["N"], s["K"]
    res = {}
    notes = []
    if M != int(M) or M < 1:
        notes.append(f"fractional M {M} -> {max(1, math.ceil(M))}")
    M = max(1, math.ceil(M))
    try:
        if c["dtype"] == "bf16" and B == 1:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            ref = F.linear(x, w)
            res["cublas"] = time_us(lambda: F.linear(x, w))
            if try_low_latency_gemm is not None:
                try:
                    y = try_low_latency_gemm(x, w)
                    if y is not None:
                        notes.append(f"lowlat rel_err {rel(y, ref):.2e}")
                        res["lowlat"] = time_us(lambda: try_low_latency_gemm(x, w))
                except Exception as e:  # noqa: BLE001
                    notes.append("lowlat err " + repr(e)[:120])
            del x, w, ref
        elif c["dtype"] == "bf16":
            x = torch.randn(B, M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(B, K, N, device=dev, dtype=torch.bfloat16) * 0.05
            res["bmm"] = time_us(lambda: torch.bmm(x, w))
            del x, w
        elif c["dtype"] == "fp8":
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
            res["quant"] = time_us(lambda: mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True))
            sa, sb = xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)
            wt = wq.t()
            ref = F.linear(x, w)
            if N % 16:
                Np = -(-N // 16) * 16
                notes.append(f"N {N} not /16: scaled_mm on N padded to {Np}")
                wp = torch.randn(Np, K, device=dev, dtype=torch.bfloat16) * 0.02
                wpq, wps = mxfp8_e4m3_quantize(wp, is_sf_swizzled_layout=True)
                wpt, spb = wpq.t(), wps.view(torch.float8_e8m0fnu)
                res["scaled_mm_padN"] = time_us(
                    lambda: torch._scaled_mm(xq, wpt, scale_a=sa, scale_b=spb, out_dtype=torch.bfloat16))
            try:
                y = torch._scaled_mm(xq, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                notes.append(f"scaled_mm rel_err {rel(y, ref):.2e}")
                res["scaled_mm"] = time_us(
                    lambda: torch._scaled_mm(xq, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            except Exception as e:  # noqa: BLE001
                notes.append("scaled_mm err " + repr(e)[:160])
            if M <= 256 or N % 16:
                try:
                    y = vfi.mm_mxfp8(xq, wt, xs, ws, out_dtype=torch.bfloat16, backend="cute-dsl")
                    notes.append(f"cutedsl rel_err {rel(y, ref):.2e}")
                    res["cutedsl"] = time_us(
                        lambda: vfi.mm_mxfp8(xq, wt, xs, ws, out_dtype=torch.bfloat16, backend="cute-dsl"))
                except Exception as e:  # noqa: BLE001
                    notes.append("cutedsl err " + repr(e)[:160])
            del x, w, xq, wq, ref
        gemm = {k: v for k, v in res.items() if k != "quant"}
        best = min(gemm.items(), key=lambda kv: kv[1][0]) if gemm else None
        flops = 2.0 * B * M * N * K
        out.row(kind="case", case=c["id"], ops=c["ops"], topologies=c["topo_ids"], ckpts=c["ckpts"],
                shape=s, dtype=c["dtype"], backends={k: round(v[0], 3) for k, v in res.items()},
                method=(best[1][1] if best else None), best=(best[0] if best else None),
                measured_us=(round(best[1][0], 3) if best else None),
                tflops=(round(flops / best[1][0] / 1e6, 1) if best else None),
                t_sol_us=c["t_sol_us"], t_ach_us=c["t_ach_us"], notes="; ".join(notes))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out.row(kind="case", case=c["id"], error=repr(e)[:300], shape=s, dtype=c["dtype"])
    torch.cuda.empty_cache()
log("gemm shard done")
