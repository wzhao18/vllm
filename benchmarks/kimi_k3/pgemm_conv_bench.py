#!/usr/bin/env python3
"""KDA in_proj Q|K|V GEMM (+ short conv epilogue) vs cuBLASLt + qkv_conv.

Modes:
  base   plain BF16-out sm_107 block-scaled GEMM, tactic sweep, vs cuBLASLt.
  conv   fused conv-epilogue GEMM + fix-up vs cuBLASLt + qkv_conv (+ bitwise).
"""

import argparse
import json
import statistics

import torch
from torch.profiler import ProfilerActivity, profile


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
    per = len(evs) // iters if iters else 0
    totals, names = [], {}
    for i in range(iters):
        grp = evs[i * per : (i + 1) * per]
        totals.append(sum(e.time_range.elapsed_us() for e in grp))
        for e in grp:
            k = e.name[:90]
            names[k] = names.get(k, 0) + e.time_range.elapsed_us()
    med = statistics.median(totals) if totals else float("nan")
    top = sorted(names.items(), key=lambda kv: -kv[1])[:4]
    return round(med, 1), [(n, round(t / iters, 1)) for n, t in top]


def operands(m, n, k):
    from flashinfer import mxfp8_quantize

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    return xq, xs, wq, ws


def cublas(xq, xs, wq, ws):
    return torch._scaled_mm(
        xq,
        wq.t(),
        scale_a=xs.view(torch.float8_e8m0fnu),
        scale_b=ws.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )


TACTICS = [
    # (mma tile M, N), cluster, inst M, swizzle, raster
    ((256, 256), (2, 1), 256, 8, "n"),
    ((256, 256), (2, 2), 256, 8, "n"),
    ((256, 256), (4, 1), 256, 8, "n"),
    ((256, 256), (2, 4), 256, 8, "n"),
    ((256, 256), (4, 2), 256, 8, "n"),
    ((256, 256), (2, 1), 256, 8, "m"),
    ((256, 256), (2, 2), 256, 1, "m"),
    ((512, 256), (2, 1), 256, 8, "n"),
    ((512, 256), (2, 2), 256, 8, "n"),
    ((512, 256), (4, 1), 256, 8, "n"),
    ((256, 192), (2, 1), 256, 8, "n"),
    ((256, 128), (2, 2), 256, 8, "n"),
]


def mode_base(args):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        PlainGemm,
    )

    m, n, k = args.m, args.n, args.k
    xq, xs, wq, ws = operands(m, n, k)
    ref = cublas(xq, xs, wq, ws)
    res = {"shape": [m, n, k], "cublaslt": kernel_times(lambda: cublas(xq, xs, wq, ws))}
    print(json.dumps(res), flush=True)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    for tac in TACTICS:
        name = f"{tac[0][0]}x{tac[0][1]}_c{tac[1][0]}x{tac[1][1]}_s{tac[3]}{tac[4]}"
        try:
            g = PlainGemm(wq, ws, tac)
            out.zero_()
            g(xq, xs, out)
            torch.cuda.synchronize()
            eq = bool(torch.equal(out, ref))
            t = kernel_times(lambda: g(xq, xs, out))
            row = {"tactic": name, "us": t[0], "ratio": round(t[0] / res["cublaslt"][0], 3), "bitwise": eq}
        except Exception as e:  # noqa: BLE001
            row = {"tactic": name, "error": repr(e)[:300]}
        print(json.dumps(row), flush=True)


def mode_conv(args):
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
        reference_inv_norms,
    )

    m, p, k = args.m, args.p, args.k
    n = 3 * p
    xq, xs, wq, ws = operands(m, n, k)
    cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
    lens = [int(v) for v in args.seqlens.split(",")] if args.seqlens else [m]
    assert sum(lens) <= m
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    S = len(lens)
    lines = torch.arange(1, S + 1, dtype=torch.int32, device="cuda")
    has_init = torch.tensor([i % 2 == 1 for i in range(S)], device="cuda")
    state0 = torch.randn(S + 2, n, 4, device="cuda").to(torch.bfloat16).float()
    state0 = state0[:, :, :3]  # (lines, 3P, 3) strided view like the KDA cache

    proj = cublas(xq, xs, wq, ws)
    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, p)
    tot = int(qsl[-1])
    tac = tuple(tuple(t) if isinstance(t, list) else t for t in args.tactic) if args.tactic else None
    g = KdaConvGemm(wq, ws, cw, p, tactic=tac, drain=not args.nodrain)
    st = state0.clone()
    out = g(xq, xs, st, qsl, lines, has_init, with_norms=True)
    torch.cuda.synchronize()
    y, norms = out
    eq_out = bool(torch.equal(y[:, :tot], ref[:, :tot]))
    eq_st = bool(torch.equal(st, st_ref))
    nref = reference_inv_norms(ref[:, :tot], p)
    nerr = float(((norms[:, :tot] - nref).abs() / nref.abs()).max())
    res = {"shape": [m, p, k], "drain": g.drain, "seqs": S, "bitwise_out": eq_out, "bitwise_state": eq_st, "norm_max_rel": nerr}
    if not eq_out:
        bad = (y[:, :tot] != ref[:, :tot]).nonzero()
        res["mismatch"] = [int(bad.shape[0])] + bad[:6].tolist()
    print(json.dumps(res), flush=True)

    def unfused():
        o = cublas(xq, xs, wq, ws)
        kda_conv.qkv_conv(o, cw, state0.clone(), qsl, lines, has_init, p)

    t_ref = kernel_times(unfused)
    t_cub = kernel_times(lambda: cublas(xq, xs, wq, ws))
    t_f = kernel_times(lambda: g(xq, xs, st, qsl, lines, has_init, with_norms=False))
    t_fn = kernel_times(lambda: g(xq, xs, st, qsl, lines, has_init, with_norms=True))
    print(json.dumps({"unfused(cublas+conv+clone)": t_ref, "cublas": t_cub, "fused": t_f, "fused+norms": t_fn}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base")
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=36864)
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--seqlens", default="")
    ap.add_argument("--tactic", type=json.loads, default=None)
    ap.add_argument("--nodrain", action="store_true")
    args = ap.parse_args()
    {"base": mode_base, "conv": mode_conv}[args.mode](args)


if __name__ == "__main__":
    main()
