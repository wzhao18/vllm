# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long power-capped run of the prefill shared gate_up (B3) kernel vs time under load.

Runs one variant back to back for ``--secs`` seconds (optionally duty-cycled with an idle
gap after every call) and prints one JSON line per ``--every`` seconds: median call time,
NVML SM clock, power, GPU temperature and energy per call (NVML energy counter). Shows
whether the kernel's time / energy per call drifts as the GPU heats (leakage) and how
energy per call depends on the clock (duty cycle).

  python3 pgemm_b3_heat.py --variant die --secs 40
  python3 pgemm_b3_heat.py --variant die --secs 10 --gap-us 300
"""

import argparse
import json
import statistics
import threading
import time

import torch

from pgemm_b3_bench import _die_rows, real_operands, scaled_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--variant", default="die", choices=["die", "c2x4", "cublas", "latent"])
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--gap-us", type=float, default=0.0)
    ap.add_argument(
        "--operands", default="randn",
        help="randn | uniform (vLLM dummy-like weights) | <file.pt> captured in-step",
    )  # fmt: skip
    ap.add_argument(
        "--corun", type=int, default=0,
        help="N of a BF16 x @ W[N, K]^T GEMM run beside every call on a 2nd stream",
    )  # fmt: skip
    ap.add_argument("--corun-seq", action="store_true", help="co-run GEMM after, same stream")
    ap.add_argument("--tactic", type=json.loads, default=None, help="tactic JSON override")
    ap.add_argument("--b3-time", action="store_true", help="time the B3 call alone in co-run mode")
    args = ap.parse_args()

    import pynvml as N

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    samples, stop = [], [False]

    def poll():
        while not stop[0]:
            try:
                samples.append((
                    time.perf_counter(),
                    N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                    N.nvmlDeviceGetPowerUsage(h) / 1e3,
                    N.nvmlDeviceGetTotalEnergyConsumption(h),
                    N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
                ))  # fmt: skip
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.004)

    threading.Thread(target=poll, daemon=True).start()
    dev = torch.device("cuda")
    m, inter, k = args.m, args.inter, args.k
    n = 2 * inter
    torch.manual_seed(m)
    if args.operands.endswith(".pt"):
        cap = torch.load(args.operands)
        x, sx = cap["x"].to(dev), cap["sx"].to(dev)
        # captured weights are already gate/up interleaved (PrefillSharedSitu.prepare)
        wi, swi = cap["w"].to(dev).contiguous(), cap["sw"].to(dev)
        w, sw = None, None
        m = x.shape[0]
    else:
        x, sx, w, sw = real_operands(m, n, k, dev)
        if args.operands == "uniform":
            from flashinfer import mxfp8_quantize

            wu = (torch.rand(n, k, device=dev, dtype=torch.bfloat16) * 2 - 1) * 1e-3
            wq, ws = mxfp8_quantize(wu, is_sf_swizzled_layout=True)
            w, sw = wq, ws.view(torch.uint8).reshape(-1)
        wi, swi = w.clone(), sw.clone()
        ssg.interleave_gate_up_rows(wi, swi)
    act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    if args.variant == "cublas":
        def fn0():
            scaled_mm(x, sx, w, sw)
    elif args.variant == "latent":
        xl = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        wlat = torch.randn(3584, k, device=dev, dtype=torch.bfloat16) * 0.02

        def fn0():
            torch.mm(xl, wlat.t())
    else:
        t = ssg.DIE_TACTIC if args.variant == "die" else ssg.DEFAULT_TACTIC
        if args.tactic:
            t = tuple(tuple(v) if isinstance(v, list) else v for v in args.tactic)
        g = ssg.SharedSituGemm(_die_rows(wi, t), swi, 4.0, 25.0, tactic=t)

        def fn0():
            g(x, sx, act)

    fn = fn0
    if args.corun:
        xb = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        wl = torch.randn(args.corun, k, device=dev, dtype=torch.bfloat16) * 0.02
        side = torch.cuda.Stream()
        b3ev = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
        ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()

        def fn():
            if args.corun_seq:
                fn0()
                torch.mm(xb, wl.t())
                return
            # B3 is launched first, the GEMM beside it takes the SMs B3 leaves
            # (the in-step order: B3 ahead of the latent down)
            ev0.record()
            if args.b3_time:
                b3ev[0].record()
            fn0()
            if args.b3_time:
                b3ev[1].record()
            with torch.cuda.stream(side):
                ev0.wait()
                torch.mm(xb, wl.t())
                ev1.record()
            ev1.wait()

    # idle gap on the GPU after every call: torch's 1-thread spin kernel (cycles at ~1.9 GHz)
    gap_cycles = int(args.gap_us * 1.9e3)
    fn()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    next_t = t_start + args.every
    ev = []
    while True:
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        if args.corun and args.b3_time:
            b3ev[0], b3ev[1] = a, b
            fn()
        else:
            a.record()
            fn()
            b.record()
        if gap_cycles:
            torch.cuda._sleep(gap_cycles)
        ev.append((a, b))
        if len(ev) % 8 == 0:
            torch.cuda.synchronize()
        now = time.perf_counter()
        if now >= next_t:
            torch.cuda.synchronize()
            now = time.perf_counter()
            ms = [x_.elapsed_time(y_) for x_, y_ in ev]
            win = [s for s in samples if next_t - args.every <= s[0] <= now]
            if len(win) >= 2:
                de = (win[-1][3] - win[0][3]) / 1e3
                dt = win[-1][0] - win[0][0]
                calls = len(ms) * dt / max(1e-9, (now - (next_t - args.every)))
                print(json.dumps({
                    "t": round(now - t_start, 1), "calls": len(ms),
                    "us": round(statistics.median(ms) * 1e3, 1),
                    "mhz": round(statistics.median(s[1] for s in win)),
                    "w": round(statistics.median(s[2] for s in win)),
                    "temp": round(statistics.median(s[4] for s in win)),
                    "mj_per_call": round(1e3 * de / calls, 1) if calls else 0,
                }), flush=True)  # fmt: skip
            ev = []
            next_t = now + args.every
            if now - t_start >= args.secs:
                break
    stop[0] = True


if __name__ == "__main__":
    main()
