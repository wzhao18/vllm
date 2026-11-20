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
    import cutlass.utils as cu

    hw = cu.HardwareInfo()
    print(json.dumps({"max_active_clusters": {c: hw.get_max_active_clusters(c) for c in (2, 4, 8)}}), flush=True)
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        MixedGemm,
    )

    for tile, pref, fb in [((512, 256), (4, 1), (2, 1)), ((512, 256), (2, 4), (2, 1)),
                           ((512, 256), (2, 2), (2, 1)), ((512, 256), (4, 2), (2, 1))]:
        name = f"mixed{tile[0]}x{tile[1]}_p{pref[0]}x{pref[1]}_f{fb[0]}x{fb[1]}"
        try:
            g = MixedGemm(wq, ws, tile, pref, fb)
            out.zero_()
            g(xq, xs, out)
            torch.cuda.synchronize()
            eq = bool(torch.equal(out, ref))
            t = kernel_times(lambda: g(xq, xs, out))
            row = {"tactic": name, "pmac": g.pmac, "fmac": g.fmac, "us": t[0], "ratio": round(t[0] / res["cublaslt"][0], 3), "bitwise": eq}
        except Exception as e:  # noqa: BLE001
            row = {"tactic": name, "error": repr(e)[:400]}
        print(json.dumps(row), flush=True)
    for tac in TACTICS[7:10]:
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


SKEL_TACTICS = [
    ((256, 256), (2, 1), 256, 8, "n", 0),
    ((256, 256), (2, 1), 256, 8, "n", 1),
    ((256, 256), (2, 1), 256, 8, "m", 1),
    ((256, 256), (2, 2), 256, 8, "n", 1),
    ((256, 256), (4, 1), 256, 8, "n", 1),
    ((512, 256), (4, 1), 256, 8, "n", 0),
    ((512, 256), (4, 1), 256, 8, "n", 1),
    ((512, 256), (2, 1), 256, 8, "n", 1),
    ((512, 256), (2, 2), 256, 8, "n", 1),
    # swap-AB (A = W, tokens on the MMA N side, C column-major): the KDA conv
    # epilogue layout (one feature channel per TMEM lane).
    ((256, 256), (2, 1), 256, 8, "n", 1, 1),
    ((256, 256), (2, 1), 256, 8, "m", 1, 1),
    ((512, 256), (4, 1), 256, 8, "n", 1, 1),
    ((512, 256), (4, 1), 256, 8, "n", 0, 1),
]


def mode_skel(args):
    """Base GEMM skeleton (BF16 out = final bytes, no epilogue math) vs SOL."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        PlainGemm,
    )

    n, k = args.n, args.k
    for m in [int(v) for v in args.ms.split(",")]:
        xq, xs, wq, ws = operands(m, n, k)
        ref = cublas(xq, xs, wq, ws)
        sol = args.sol_us * m / 16384
        t_cub = kernel_times(lambda: cublas(xq, xs, wq, ws))[0]
        print(json.dumps({"m": m, "sol_us": sol, "cublaslt": t_cub, "x_sol": round(t_cub / sol, 3)}), flush=True)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        tacs = [tuple(tuple(t) if isinstance(t, list) else t for t in args.tactic)] if args.tactic else SKEL_TACTICS
        for tac in tacs:
            if args.swap_only and len(tac) < 7:
                continue
            name = (
                f"{tac[0][0]}x{tac[0][1]}_c{tac[1][0]}x{tac[1][1]}_{tac[4]}{tac[3]}_p{1 + tac[5]}"
                + ("_swap" if len(tac) > 6 and tac[6] else "")
                + (f"_{tac[7]}" if len(tac) > 7 and tac[7] else "")
            )
            try:
                g = PlainGemm(wq, ws, tac)
                out.zero_()
                g(xq, xs, out)
                torch.cuda.synchronize()
                eq = bool(torch.equal(out, ref))
                t = kernel_times(lambda: g(xq, xs, out))[0]
                row = {"m": m, "tactic": name, "us": t, "x_sol": round(t / sol, 3), "x_cublas": round(t / t_cub, 3), "bitwise": eq}
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc().strip().splitlines()[-6:]
                row = {"m": m, "tactic": name, "error": repr(e)[:300], "tb": tb}
            print(json.dumps(row), flush=True)
        del xq, xs, wq, ws, ref, out
        torch.cuda.empty_cache()


B3_TACTICS = [
    ((512, 256), (2, 1), False),
    ((512, 256), (2, 1), True),
    ((512, 256), (4, 1), True),
    ((256, 256), (2, 1), True),
    ((256, 256), (2, 2), True),
]


def mode_clk(args):
    """Sustained back-to-back timing with SM clock / power sampled by nvidia-smi."""
    import subprocess

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        PlainGemm,
    )

    m, n, k = int(args.ms.split(",")[0]), args.n, args.k
    xq, xs, wq, ws = operands(m, n, k)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    fns = {"cublaslt": lambda: cublas(xq, xs, wq, ws)}
    if args.tactic:
        tac = tuple(tuple(t) if isinstance(t, list) else t for t in args.tactic)
        g = PlainGemm(wq, ws, tac)
        fns["cute"] = lambda: g(xq, xs, out)
    dev = torch.cuda.current_device()
    for name, fn in fns.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        smi = subprocess.Popen(
            ["nvidia-smi", "-i", str(dev), "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits", "-lms", "25"],
            stdout=subprocess.PIPE, text=True,
        )  # fmt: skip
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        smi.terminate()
        rows = [r.split(",") for r in smi.stdout.read().strip().splitlines() if r.count(",") == 3]
        clk = sorted(float(r[0]) for r in rows) or [0.0]
        pw = sorted(float(r[1]) for r in rows) or [0.0]
        print(json.dumps({
            "m": m, "kernel": name, "us_sustained": round(s.elapsed_time(e) * 1000 / args.iters, 1),
            "sm_mhz_med": clk[len(clk) // 2], "sm_mhz_min": clk[0], "sm_mhz_max": clk[-1],
            "power_w_med": pw[len(pw) // 2], "throttle": sorted({r[3].strip() for r in rows}), "samples": len(rows),
        }), flush=True)  # fmt: skip


def mode_b3(args):
    """Packed MoE front pack_n [latent_down | shared gate_up] with the SiTU +
    MXFP8 epilogue vs cuBLASLt + situ_and_mul_quantize_fast (in-step path)."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        packed_situ_gemm as psg,
    )

    n_lat, inter, k = 3584, 6144, 7168
    n = n_lat + 2 * inter
    beta, lbeta = 4.0, 25.0
    dev = torch.device("cuda")
    for m in [int(v) for v in args.ms.split(",")]:
        torch.manual_seed(m)
        x = torch.randint(-1, 2, (m, k), device=dev).float().to(torch.float8_e4m3fn)
        w = torch.randint(-1, 2, (n, k), device=dev).float().to(torch.float8_e4m3fn)
        sx = torch.full((((m + 127) // 128) * 128 * k // 32,), 127, dtype=torch.uint8, device=dev)
        sw = torch.randint(119, 123, (n * k // 32,), dtype=torch.uint8, device=dev)

        def ref_path(out=None):
            y = torch._scaled_mm(x, w.t(), scale_a=sx.view(torch.float8_e8m0fnu),
                                 scale_b=sw.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)
            r = out if out is not None else mxfp8_act.Mxfp8Buffer(m, inter + n_lat, dev)
            mxfp8_act.situ_and_mul_quantize_fast(y[:, n_lat:], beta, lbeta, out=r)
            return y, r

        ref = mxfp8_act.Mxfp8Buffer(m, inter + n_lat, dev)
        ref.data.zero_()
        ref.scale.zero_()
        y, _ = ref_path(ref)
        sol = args.sol_us * (m * n) / (16384 * 36864)
        t_ref = kernel_times(lambda: ref_path(ref))
        t_gemm = kernel_times(lambda: torch._scaled_mm(
            x, w.t(), scale_a=sx.view(torch.float8_e8m0fnu),
            scale_b=sw.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16))[0]
        print(json.dumps({"m": m, "sol_us_gemm": round(sol, 1), "ref_total": t_ref[0], "ref_gemm": t_gemm, "ref_top": t_ref[1]}), flush=True)
        wi, swi = w.clone(), sw.clone()
        psg.interleave_shared_rows(wi, swi, n_lat, k)
        act = mxfp8_act.Mxfp8Buffer(m, inter + n_lat, dev)
        lat = torch.empty(m, n_lat, device=dev, dtype=torch.bfloat16)
        for tac in B3_TACTICS:
            name = f"{tac[0][0]}x{tac[0][1]}_c{tac[1][0]}x{tac[1][1]}_p{1 + int(tac[2])}"
            try:
                g = psg.PackedSituGemm(wi, swi, n_lat, beta, lbeta, tactic=tac)
                act.data.zero_()
                act.scale.zero_()
                g(x, sx, act, lat)
                torch.cuda.synchronize()
                eq = (bool(torch.equal(lat.view(torch.int16), y[:, :n_lat].view(torch.int16)))
                      and bool(torch.equal(act.data.view(torch.uint8), ref.data.view(torch.uint8))))
                t = kernel_times(lambda: g(x, sx, act, lat))[0]
                row = {"m": m, "tactic": name, "us": t, "x_sol_gemm": round(t / sol, 3), "x_ref_total": round(t / t_ref[0], 3), "bitwise": eq}
            except Exception as e:  # noqa: BLE001
                row = {"m": m, "tactic": name, "error": repr(e)[:300]}
            print(json.dumps(row), flush=True)
        del x, w, wi, sw, swi, y, ref, act, lat
        torch.cuda.empty_cache()


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


def mode_ncu(args):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    m, p, k = args.m, args.p, args.k
    xq, xs, wq, ws = operands(m, 3 * p, k)
    cw = (torch.randn(3 * p, 4, device="cuda") * 0.5).contiguous()
    qsl = torch.tensor([0, m], dtype=torch.int32, device="cuda")
    lines = torch.ones(1, dtype=torch.int32, device="cuda")
    has_init = torch.zeros(1, dtype=torch.bool, device="cuda")
    st = torch.zeros(4, 3 * p, 3, device="cuda")
    g = KdaConvGemm(wq, ws, cw, p, drain=not args.nodrain)
    for _ in range(2):
        g(xq, xs, st, qsl, lines, has_init)
    torch.cuda.synchronize()
    print("{\"ncu_done\": 1}")


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
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--ms", default="16384,8192")
    ap.add_argument("--swap-only", dest="swap_only", action="store_true")
    ap.add_argument("--sol-us", dest="sol_us", type=float, default=748.0)
    args = ap.parse_args()
    {"base": mode_base, "skel": mode_skel, "clk": mode_clk, "b3": mode_b3, "conv": mode_conv, "ncu": mode_ncu}[args.mode](args)


if __name__ == "__main__":
    main()
