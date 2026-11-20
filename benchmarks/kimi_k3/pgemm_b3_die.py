#!/usr/bin/env python3
"""B3 die-split probe (benchmark only): is a die-local N split of the prefill
shared gate_up GEMM worth building into the kernel scheduler?

Compares, sustained back to back at the power cap (randn MXFP8 operands):
  full      the deployed tactic on all SMs (B striped);
  seq       the two N halves one after the other on all SMs (B striped);
  die       half h on a green context of die h with its B half homed on die h,
            both halves concurrently (die-local B, each die's L2 holds its half);
  die_str   as ``die`` with striped B halves.
Each half writes its own MXFP8 buffer (bytes as the full kernel's halves).
"""

import argparse
import json
import statistics
import threading
import time

import torch

from pgemm_b3_bench import real_operands


class _Hw:
    """``cutlass.utils.HardwareInfo`` stand-in with a fixed cluster capacity."""

    def __init__(self, real, mac):
        self._real, self._mac = real, mac

    def __call__(self):
        return self

    def get_max_active_clusters(self, n):
        return self._mac if self._mac else self._real().get_max_active_clusters(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--half-swz", default="m6,m2")
    ap.add_argument("--die-mac", default="11,10")
    args = ap.parse_args()

    import cutlass.utils as cu
    import pynvml as N

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    samples, stop = [], [False]

    def poll():
        while not stop[0]:
            try:
                samples.append((time.perf_counter(), N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                                N.nvmlDeviceGetPowerUsage(h) / 1e3))  # fmt: skip
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.004)

    threading.Thread(target=poll, daemon=True).start()
    dev = torch.device("cuda")
    m, inter, k = args.m, 6144, 7168
    n = 2 * inter
    torch.manual_seed(m)
    x, sx, w, sw = real_operands(m, n, k, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    hn = n // 2
    w_h = [wi[:hn].contiguous(), wi[hn:].contiguous()]
    s_h = [swi[: swi.numel() // 2].contiguous(), swi[swi.numel() // 2 :].contiguous()]
    w_loc = [dies.localized_like(w_h[d], [d]) for d in range(2)]
    s_loc = [dies.localized_like(s_h[d], [d]) for d in range(2)]
    streams = [dies.domain_stream(dev.index or 0, d, coscheduled=8) for d in range(2)]
    real_hw = cu.HardwareInfo
    tac = lambda swz: ((512, 256), (2, 4), 256, int(swz[1:]), swz[0], 0, 0, "drain+roll+st8")  # noqa: E731

    cur = [None]

    def make(wt, st, tactic, mac=0):
        if cur[0] != mac:  # compiled kernels depend on (tactic, n, capacity) only
            ssg._COMPILED.clear()
            cur[0] = mac
        cu.HardwareInfo = _Hw(real_hw, mac)
        try:
            g = ssg.SharedSituGemm(wt, st, 4.0, 25.0, tactic=tactic)
            out = mxfp8_act.Mxfp8Buffer(m, wt.shape[0] // 2, dev)
            g(x, sx, out)  # compiles with this capacity
            torch.cuda.synchronize()
        finally:
            cu.HardwareInfo = real_hw
        return g, out

    def concurrent(hs2):
        def run():
            ev = torch.cuda.Event()
            ev.record()
            done = []
            for d in range(2):
                streams[d].wait_event(ev)
                with torch.cuda.stream(streams[d]):
                    hs2[d][0](x, sx, hs2[d][1])
                e = torch.cuda.Event()
                e.record(streams[d])
                done.append(e)
            for e in done:
                torch.cuda.current_stream().wait_event(e)

        return run

    variants = {}
    swzs = args.half_swz.split(",")
    g, o = make(wi, swi, tac("m4"))
    variants["full_m4"] = lambda g=g, o=o: g(x, sx, o)
    ok = []
    for swz in swzs:
        try:
            hs = [make(w_h[d], s_h[d], tac(swz)) for d in range(2)]
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"v": f"seq_{swz}", "error": repr(e)[:200]}), flush=True)
            continue
        ok.append(swz)
        variants[f"seq_{swz}"] = lambda hs=hs: [g(x, sx, o) for g, o in hs]
    for mac in [int(v) for v in args.die_mac.split(",")]:
        for swz in ok:
            for name, ws, ss in (("die", w_loc, s_loc), ("die_str", w_h, s_h)):
                hs2 = [make(ws[d], ss[d], tac(swz), mac) for d in range(2)]
                variants[f"{name}_{swz}_mac{mac}"] = concurrent(hs2)
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
            clk = statistics.median(s[1] for s in win) if win else 0
            pw = statistics.median(s[2] for s in win) if win else 0
            res[name].append((statistics.median(last) * 1e3, clk, pw))
    stop[0] = True
    base = statistics.median(r[0] for r in res["full_m4"])
    for name, rs in res.items():
        us = statistics.median(r[0] for r in rs)
        print(json.dumps({"v": name, "us": round(us, 1), "rounds": [round(r[0], 1) for r in rs],
                          "x_full": round(us / base, 3), "mhz": round(statistics.median(r[1] for r in rs)),
                          "w": round(statistics.median(r[2] for r in rs))}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
