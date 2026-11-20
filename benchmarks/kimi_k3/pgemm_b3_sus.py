#!/usr/bin/env python3
"""B3 sustained (power-capped) A/B: the in-step regime of the prefill shared
gate_up GEMM. The F16K prefill slice runs at the VR200 software power cap the
whole time, so a kernel's in-step time follows its energy, not its burst rate.
Each variant runs back to back for ``--secs`` seconds per round (randn-scaled
MXFP8 operands, like the online-MXFP8 step weights); reported per variant:
median ms/call over the last second of each round, NVML SM clock, board power
and energy per call (total-energy counter). Rounds interleave the variants.
"""

import argparse
import json
import statistics
import threading
import time

import torch

from pgemm_b3_bench import exact_operands, real_operands, scaled_mm, tname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--lbeta", type=float, default=25.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--fused", type=json.loads, default=[])
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--noref", action="store_true")
    args = ap.parse_args()

    import pynvml as N

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByUUID(
        "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    )
    samples, stop = [], [False]

    def poll():
        while not stop[0]:
            try:
                samples.append((
                    time.perf_counter(),
                    N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                    N.nvmlDeviceGetPowerUsage(h) / 1e3,
                    N.nvmlDeviceGetTotalEnergyConsumption(h),
                ))  # fmt: skip
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.004)

    threading.Thread(target=poll, daemon=True).start()

    def tac(t):
        return tuple(tuple(v) if isinstance(v, list) else v for v in t)

    dev = torch.device("cuda")
    m, inter, k = args.m, args.inter, args.k
    n = 2 * inter
    torch.manual_seed(m)
    x, sx, w, sw = (exact_operands if args.exact else real_operands)(m, n, k, dev)
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    variants = {}
    if not args.noref:
        variants["ref"] = lambda: mxfp8_act.situ_and_mul_quantize_auto(
            scaled_mm(x, sx, w, sw), args.beta, args.lbeta, out=ref
        )
        variants["cublas"] = lambda: scaled_mm(x, sx, w, sw)
    for t in args.fused:
        g = ssg.SharedSituGemm(wi, swi, args.beta, args.lbeta, tactic=tac(t))
        variants["fused_" + tname(tac(t))] = lambda g=g: g(x, sx, act)
    for fn in variants.values():
        fn()
    torch.cuda.synchronize()

    res = {name: [] for name in variants}
    for _ in range(args.rounds):
        for name, fn in variants.items():
            ev = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.secs:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                ev.append((a, b))
                if len(ev) % 8 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ms = [a.elapsed_time(b) for a, b in ev]
            last, cum = [], 0.0
            for v in reversed(ms):
                if cum > 1000.0:
                    break
                last.append(v)
                cum += v
            win = [s for s in samples if t1 - cum * 1e-3 <= s[0] <= t1]
            if len(win) >= 2:
                de = (win[-1][3] - win[0][3]) / 1e3  # J
                dt = win[-1][0] - win[0][0]
                calls = len(last) * dt / (cum * 1e-3)
                jpc = de / calls if calls else 0.0
                clk = statistics.median(s[1] for s in win)
                pw = statistics.median(s[2] for s in win)
            else:
                jpc, clk, pw = 0.0, 0, 0.0
            res[name].append((statistics.median(last) * 1e3, clk, pw, jpc))
    stop[0] = True
    base = statistics.median(r[0] for r in res[next(iter(res))])
    for name, rs in res.items():
        us = statistics.median(r[0] for r in rs)
        print(json.dumps({
            "v": name, "us": round(us, 1), "rounds": [round(r[0], 1) for r in rs],
            "x_first": round(us / base, 3),
            "mhz": round(statistics.median(r[1] for r in rs)),
            "w": round(statistics.median(r[2] for r in rs)),
            "mj_per_call": round(1e3 * statistics.median(r[3] for r in rs), 2),
        }), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
