# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE-front L2 touch microbench (one Rubin GPU): a cold-L2 producer stand-in
(``torch.cuda._sleep``, HBM idle) followed by the router | latent down ltpin GEMM
(4480 x 7168 BF16 -> FP32), with and without the side-stream touch of the weights
forked at the producer's start. Prints producer+GEMM time (median us) per rows and
checks the GEMM output is bitwise unchanged by the touch."""

import argparse
import statistics

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.touch import (
    compile_touch,
    sm_count,
    touch_table,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.ltpin import LtPin
from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute
import cuda.bindings.driver as cuda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,64,96")
    ap.add_argument("--sleep-us", default="2,4,6")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    w = (torch.randn(4480, 7168, device=dev) * 0.02).to(torch.bfloat16)
    rows = tuple(int(r) for r in args.rows.split(","))
    gemm = LtPin("router_latent", 4480, 7168, rows, out32=True, device=dev)
    table = touch_table([w], dev)
    touch = compile_touch(1, sm_count(dev), dev.index)
    views = to_cute(table, 16)
    _, high = torch.cuda.Stream.priority_range()
    side = torch.cuda.Stream(device=dev, priority=high)
    flush = torch.empty(1 << 29, dtype=torch.uint8, device=dev)
    main_s = torch.cuda.current_stream(dev)
    clk = 2.4e9  # SM clock under load (hwdb hw55)
    print(f"SMs {sm_count(dev)} clock {clk / 1e9:.2f} GHz")
    for us in (float(s) for s in args.sleep_us.split(",")):
        cycles = int(us * 1e-6 * clk)
        for n in rows:
            x = torch.randn(n, 7168, device=dev).to(torch.bfloat16)
            outs, times = {}, {}
            for mode in ("base", "touch", "base", "touch"):
                y = torch.empty(n, 4480, device=dev, dtype=torch.float32)
                ts = []
                for _ in range(args.iters):
                    flush.zero_()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record(main_s)
                    if mode == "touch":
                        side.wait_stream(main_s)
                        touch(views, cuda.CUstream(side.cuda_stream))
                    torch.cuda._sleep(cycles)
                    gemm.run(w, x, y)
                    b.record(main_s)
                    if mode == "touch":
                        main_s.wait_stream(side)
                    torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b) * 1e3)
                times.setdefault(mode, []).extend(ts)
                outs[mode] = y.clone()
            same = torch.equal(outs["base"], outs["touch"])
            mb, mt = statistics.median(times["base"]), statistics.median(times["touch"])
            print(f"producer {us:.0f} us n{n}: base {mb:.2f} touch {mt:.2f} us "
                  f"(delta {mt - mb:+.2f}) bitwise_equal={same}", flush=True)
    # touch alone (cold), for the stream rate
    ts = []
    for _ in range(args.iters):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(main_s)
        touch(views, cuda.CUstream(main_s.cuda_stream))
        b.record(main_s)
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    t = statistics.median(ts)
    print(f"touch alone 64.2 MB cold: {t:.2f} us = {w.numel() * 2 / t / 1e6:.1f} TB/s")


if __name__ == "__main__":
    main()
