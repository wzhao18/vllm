#!/usr/bin/env python3
"""B3: prefill shared-expert gate_up GEMM with the SiTU + MXFP8 epilogue vs
cuBLASLt gate_up + situ_and_mul_quantize_auto (the deployed path).

Per M: reference (GEMM + act) times, plain CuTe base tactics (BF16 out and
mainloop-only), and the fused SharedSituGemm tactics; bitwise check on
exact-input operands (small-integer FP8 with power-of-two scales, so every
summation order gives the same FP32 accumulator).
"""

import argparse
import json
import statistics

import torch
from torch.profiler import ProfilerActivity, profile


def kernel_times(fn, iters=20, warm=3):
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
            k = e.name[:60]
            names[k] = names.get(k, 0) + e.time_range.elapsed_us()
    med = statistics.median(totals) if totals else float("nan")
    top = sorted(names.items(), key=lambda kv: -kv[1])[:3]
    return round(med, 1), [(n, round(t / iters, 1)) for n, t in top]


def clock_power(fn, iters=300):
    """Mean SM clock (MHz) and board power (W) while ``fn`` runs back to back."""
    import threading
    import time

    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append((torch.cuda.clock_rate(), torch.cuda.power_draw() / 1000))
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.005)

    th = threading.Thread(target=sample)
    th.start()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    stop.set()
    th.join()
    if not samples:
        return None, None
    tail = samples[len(samples) // 4 :] or samples
    return (
        round(statistics.mean(c for c, _ in tail)),
        round(statistics.mean(w for _, w in tail)),
    )


def scaled_mm(x, sx, w, sw):
    return torch._scaled_mm(
        x, w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu), scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip


def exact_operands(m, n, k, dev):
    x = torch.randint(-2, 3, (m, k), device=dev).float().to(torch.float8_e4m3fn)
    w = torch.randint(-2, 3, (n, k), device=dev).float().to(torch.float8_e4m3fn)
    sx = torch.full((((m + 127) // 128) * 128 * k // 32,), 124, dtype=torch.uint8, device=dev)
    sw = torch.randint(117, 121, (n * k // 32,), dtype=torch.uint8, device=dev)
    return x, sx, w, sw


def real_operands(m, n, k, dev):
    from flashinfer import mxfp8_quantize

    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    return xq, xs.view(torch.uint8).reshape(-1), wq, ws.view(torch.uint8).reshape(-1)


FUSED = [
    ((512, 256), (2, 4), 256, 4, "m", 0, 0, "drain"),
    ((512, 256), (2, 1), 256, 8, "n", 0, 0, "drain"),
    ((512, 256), (4, 1), 256, 8, "n", 0, 0, "drain"),
    ((512, 256), (2, 4), 256, 4, "m", 0, 0, ""),
    ((512, 256), (2, 1), 256, 8, "n", 0, 0, ""),
]
PLAIN = [
    ((512, 256), (2, 4), 256, 4, "m", 0, 0, ""),
    ((512, 256), (2, 4), 256, 4, "m", 0, 0, "noepi"),
    ((512, 256), (2, 1), 256, 8, "n", 0, 0, ""),
    ((512, 256), (4, 1), 256, 8, "n", 0, 0, ""),
]


def tname(t):
    return f"{t[0][0]}x{t[0][1]}_c{t[1][0]}x{t[1][1]}_{t[4]}{t[3]}" + (f"_{t[7]}" if t[7] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="16384,8192")
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--lbeta", type=float, default=25.0)
    ap.add_argument("--fused", type=json.loads, default=None)
    ap.add_argument("--plain", type=json.loads, default=None)
    ap.add_argument("--noplain", action="store_true")
    ap.add_argument("--sol-us", dest="sol_us", type=float, default=0.0)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        PlainGemm,
    )

    dev = torch.device("cuda")
    inter, k = args.inter, args.k
    n = 2 * inter
    beta, lbeta = args.beta, args.lbeta
    fused = [tuple(tuple(v) if isinstance(v, list) else v for v in t) for t in args.fused] if args.fused else FUSED
    plain = [tuple(tuple(v) if isinstance(v, list) else v for v in t) for t in args.plain] if args.plain else PLAIN
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "n": n, "k": k, "beta": beta, "lbeta": lbeta}), flush=True)
    for m in [int(v) for v in args.ms.split(",")]:
        for data in ("exact", "real"):
            torch.manual_seed(m)
            ops = exact_operands if data == "exact" else real_operands
            x, sx, w, sw = ops(m, n, k, dev)
            ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)

            def ref_path():
                y = scaled_mm(x, sx, w, sw)
                mxfp8_act.situ_and_mul_quantize_auto(y, beta, lbeta, out=ref)

            ref.data.zero_()
            ref.scale.zero_()
            ref_path()
            torch.cuda.synchronize()
            t_ref = kernel_times(ref_path)
            t_gemm = kernel_times(lambda: scaled_mm(x, sx, w, sw))[0]
            mhz, watt = clock_power(ref_path)
            print(json.dumps({"m": m, "data": data, "ref_total": t_ref[0], "ref_gemm": t_gemm,
                              "ref_act": round(t_ref[0] - t_gemm, 1), "mhz": mhz, "w": watt,
                              "ref_top": t_ref[1]}), flush=True)
            if data == "real" and not args.noplain:
                out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
                for tac in plain:
                    try:
                        g = PlainGemm(w, sw, tac)
                        g(x, sx, out)
                        torch.cuda.synchronize()
                        t = kernel_times(lambda: g(x, sx, out))[0]
                        mhz, watt = clock_power(lambda: g(x, sx, out))
                        print(json.dumps({"m": m, "plain": tname(tac), "us": t, "mhz": mhz, "w": watt,
                                          "x_cublas": round(t / t_gemm, 3)}), flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(json.dumps({"m": m, "plain": tname(tac), "error": repr(e)[:300]}), flush=True)
                del out
            wi, swi = w.clone(), sw.clone()
            ssg.interleave_gate_up_rows(wi, swi)
            act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
            for tac in fused:
                try:
                    g = ssg.SharedSituGemm(wi, swi, beta, lbeta, tactic=tac)
                    act.data.view(torch.uint8).fill_(0x7F)
                    act.scale.zero_()
                    g(x, sx, act)
                    torch.cuda.synchronize()
                    dq = act.data.view(torch.uint8) != ref.data.view(torch.uint8)
                    eq_q = int(dq.sum())
                    eq_s = int((act.scale != ref.scale).sum())
                    t = kernel_times(lambda: g(x, sx, act))[0]
                    mhz, watt = clock_power(lambda: g(x, sx, act))
                    print(json.dumps({"m": m, "data": data, "fused": tname(tac), "us": t, "mhz": mhz, "w": watt,
                                      "x_ref_total": round(t / t_ref[0], 3), "save_us": round(t_ref[0] - t, 1),
                                      "q_mismatch": eq_q, "sf_mismatch": eq_s}), flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    tb = traceback.format_exc().strip().splitlines()[-8:]
                    print(json.dumps({"m": m, "fused": tname(tac), "error": repr(e)[:400], "tb": tb}), flush=True)
            del x, w, wi, sw, swi, ref, act
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
