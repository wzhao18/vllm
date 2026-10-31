# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared timing helper for the K3.1 wide-EP projection benches: replay a CUDA
graph of many calls (rotating inputs / weight copies) and report the median
microseconds per call."""

import torch


def time_graph(fns, calls=64, reps=5):
    """fns: zero-arg callables (one per rotating copy); us per call."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    n = len(fns)
    loops = max(1, calls // n)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(loops):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (4 * loops * n))
    ts.sort()
    return ts[len(ts) // 2]
