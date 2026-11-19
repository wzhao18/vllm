# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-GEMM microbench: Kimi K3.1 DEP16 prefill dense GEMMs at the deployed
per-GPU shapes (TP1, M = 16384 / 8192), every available backend, vs the perf
model's achievable time.

Times are CUPTI kernel time (torch profiler, sum of all device kernels of one
call, median over iterations), so launch gaps are excluded. The kernel names
are reported to identify the tactic.

    python benchmarks/kimi_k3/pgemm_bench.py --rows 16384,8192 --out res.json
"""

import argparse
import json
import statistics
import sys

import torch
from torch.profiler import ProfilerActivity, profile

# Perf-model achievable rates (out/optables.json nvfp4_modelopt|P3-16|prefill:
# g_proj 2.886 TFLOP / 790.7 us, KDA.in_proj 11.6 TFLOP / 1064 us, w_uk bytes /
# 160.2 us).
ACH_BF16 = 2.886e12 / 790.7e-6
ACH_FP8 = 11.6e12 / 1064.0e-6
ACH_HBM = (12.58e6 + 2.215e9) / 160.2e-6


def model_ach_us(m, n, k, dtype, batch=1, out_bytes=2):
    flops = 2.0 * m * n * k * batch
    w_bytes = n * k * batch * (1 if dtype == "fp8" else 2)
    a_bytes = m * k * batch * (1 if dtype == "fp8" else 2) + m * n * batch * out_bytes
    peak = ACH_FP8 if dtype == "fp8" else ACH_BF16
    return max(flops / peak, (w_bytes + a_bytes) / ACH_HBM) * 1e6


def kernel_times(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()
    evs = [
        e
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and "Memset" not in e.name
        and "Memcpy" not in e.name
    ]
    evs.sort(key=lambda e: e.time_range.start)
    # Group kernels per call: iterations are separated by synchronize, so split
    # into `iters` equal groups (each call launches the same kernel sequence).
    per = len(evs) // iters if iters else 0
    totals, names = [], {}
    for i in range(iters):
        grp = evs[i * per : (i + 1) * per]
        totals.append(sum(e.time_range.elapsed_us() for e in grp))
        for e in grp:
            names[e.name[:110]] = names.get(e.name[:110], 0) + e.time_range.elapsed_us()
    med = statistics.median(totals) if totals else float("nan")
    top = sorted(names.items(), key=lambda kv: -kv[1])[:3]
    return med, [(n, round(t / iters, 1)) for n, t in top]


# (name, N, K, dtype, batch) at TP1; dtype fp8 = MXFP8 W8A8 in nvfp4_modelopt.
SHAPES = [
    ("KDA.in_proj", 49376, 7168, "fp8", 1),
    ("KDA.o_proj", 7168, 12288, "fp8", 1),
    ("MOE.shared_gate_up", 12288, 7168, "fp8", 1),
    ("MOE.shared_down", 7168, 6144, "fp8", 1),
    ("MOE.router", 896, 7168, "bf16f32", 1),
    ("MOE.latent_down", 3584, 7168, "bf16", 1),
    ("MOE.latent_up", 7168, 3584, "bf16", 1),
    ("DSA.qkv_a_packed", 2272, 7168, "bf16", 1),
    ("DSA.qb_wq_packed", 22528, 1536, "bf16", 1),
    ("DSA.g_proj", 12288, 7168, "bf16", 1),
    ("DSA.o_proj", 7168, 12288, "bf16", 1),
    ("DSA.w_uk", 512, 128, "bf16", 96),
    ("DSA.w_uv", 128, 512, "bf16", 96),
    ("KDA.f_b_proj", 12288, 128, "bf16", 1),
    ("DENSE.gate_up", 67584, 7168, "bf16", 1),
    ("DENSE.down", 7168, 33792, "bf16", 1),
    ("KDA.in_proj.bf16", 49376, 7168, "bf16", 1),
    ("MOE.shared_gate_up.bf16", 12288, 7168, "bf16", 1),
]


def bench_fp8(m, n, k, args):
    import flashinfer
    from flashinfer import mxfp8_quantize

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    res = {}
    res["quant.fi"] = kernel_times(
        lambda: mxfp8_quantize(x, is_sf_swizzled_layout=True)
    )
    try:
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

        res["quant.vllm_act"] = kernel_times(lambda: mxfp8_act.quantize(x))
    except Exception as e:  # noqa: BLE001
        res["quant.vllm_act"] = (float("nan"), [repr(e)[:100]])
    wt = wq.t()  # [K, N] column-major view
    ref = None
    try:
        out = torch._scaled_mm(
            xq,
            wt,
            scale_a=xs.view(torch.float8_e8m0fnu),
            scale_b=ws.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )
        ref = out.float()
        res["cublaslt"] = kernel_times(
            lambda: torch._scaled_mm(
                xq,
                wt,
                scale_a=xs.view(torch.float8_e8m0fnu),
                scale_b=ws.view(torch.float8_e8m0fnu),
                out_dtype=torch.bfloat16,
            )
        )
    except Exception as e:  # noqa: BLE001
        res["cublaslt"] = (float("nan"), [repr(e)[:160]])
    try:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        wq_l, ws_l = mxfp8_quantize(w, is_sf_swizzled_layout=False)
        pn = (n + 127) // 128 * 128
        wpad = torch.zeros(pn, k, dtype=wq_l.dtype, device="cuda")
        wpad[:n] = wq_l
        spad = torch.zeros(pn, k // 32, dtype=torch.uint8, device="cuda")
        spad[:n] = ws_l.view(torch.uint8).reshape(n, k // 32)
        w_sh = shuffle_matrix_a(wpad.view(torch.uint8), 128).view(wq_l.dtype)
        s_sh = shuffle_matrix_sf_a(spad, 128, num_elts_per_sf=32).reshape(-1)

        def trt():
            return flashinfer.mm_mxfp8(
                xq, w_sh.reshape(pn, k).t(), xs, s_sh, out_dtype=torch.bfloat16,
                backend="trtllm", use_8x4_sf_layout=False,
            )

        o = trt()[:, :n]
        err = (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        t = kernel_times(trt)
        res["trtllm"] = (t[0], t[1], err)
    except Exception as e:  # noqa: BLE001
        res["trtllm"] = (float("nan"), [repr(e)[:160]])
    for backend in ("cute-dsl", "cutlass", "cudnn"):
        try:
            o = flashinfer.mm_mxfp8(xq, wt, xs, ws, backend=backend)
            err = (
                (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
                if ref is not None
                else float("nan")
            )
            t = kernel_times(lambda: flashinfer.mm_mxfp8(xq, wt, xs, ws, backend=backend))
            res[backend] = (t[0], t[1], err)
        except Exception as e:  # noqa: BLE001
            res[backend] = (float("nan"), [repr(e)[:160]])
    return res


def bench_bf16(m, n, k, batch, f32out, args):
    import flashinfer

    res = {}
    if batch > 1:
        x = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
        res["torch.bmm"] = kernel_times(lambda: torch.bmm(x, w))
        # The model's [m, batch, k] head-interleaved input (DSA layout).
        xi = torch.randn(m, batch, k, device="cuda", dtype=torch.bfloat16)
        res["torch.bmm.strided"] = kernel_times(lambda: torch.bmm(xi.transpose(0, 1), w))
        return res
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if f32out:
        res["torch.mm.f32out"] = kernel_times(
            lambda: torch.mm(x, w.T, out_dtype=torch.float32)
        )
    res["F.linear"] = kernel_times(lambda: torch.nn.functional.linear(x, w))
    for backend in ("cutlass", "cudnn", "cute-dsl"):
        try:
            flashinfer.mm_bf16(
                x, w.T, out_dtype=torch.float32 if f32out else torch.bfloat16,
                backend=backend,
            )
            res["fi." + backend] = kernel_times(
                lambda: flashinfer.mm_bf16(
                    x, w.T, out_dtype=torch.float32 if f32out else torch.bfloat16,
                    backend=backend,
                )
            )
        except Exception as e:  # noqa: BLE001
            res["fi." + backend] = (float("nan"), [repr(e)[:160]])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="16384,8192")
    p.add_argument("--only", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(), torch.__version__, file=sys.stderr)
    allres = []
    for m in [int(r) for r in args.rows.split(",")]:
        for name, n, k, dt, batch in SHAPES:
            if args.only and not any(s in name for s in args.only.split(",")):
                continue
            if dt == "fp8":
                r = bench_fp8(m, n, k, args)
                ach = model_ach_us(m, n, k, "fp8")
            else:
                r = bench_bf16(m, n, k, batch, dt == "bf16f32", args)
                ach = model_ach_us(m, n, k, "bf16", batch, 4 if dt == "bf16f32" else 2)
            for be, v in r.items():
                t = v[0]
                ratio = t / ach if t == t and not be.startswith("quant") else None
                row = {
                    "op": name, "M": m, "N": n, "K": k, "batch": batch, "dtype": dt,
                    "backend": be, "us": round(t, 2) if t == t else None,
                    "ach_us": round(ach, 2),
                    "x_ach": round(ratio, 3) if ratio else None,
                    "kernels": v[1],
                    "relerr": v[2] if len(v) > 2 else None,
                }
                allres.append(row)
                print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(allres, f, indent=1)


if __name__ == "__main__":
    main()
