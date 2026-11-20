#!/usr/bin/env python3
"""Can a die-split GEMM (N halves with die-homed weights, one half per die
green-context stream) be captured into a CUDA graph forked from the capture
stream, and does the replay match eager bitwise?"""

import json
import time

import torch
import torch.nn.functional as F


def main():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    streams = [dies.domain_stream(dev.index or 0, d, coscheduled=8) for d in range(2)]
    m, n, k = 16384, 7168, 12288
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    h = n // 2
    w_loc = [dies.localized_like(w[d * h : (d + 1) * h].contiguous(), [d]) for d in range(2)]
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    ref = F.linear(x, w)

    def die_conc():
        ev = torch.cuda.current_stream().record_event()
        evs = []
        for d in range(2):
            streams[d].wait_event(ev)
            with torch.cuda.stream(streams[d]):
                torch.mm(x, w_loc[d].t(), out=out[:, d * h : (d + 1) * h])
                evs.append(streams[d].record_event())
        for e in evs:
            torch.cuda.current_stream().wait_event(e)

    die_conc()
    torch.cuda.synchronize()
    res = {"eager_bitwise": bool(torch.equal(out, ref))}
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            die_conc()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            die_conc()
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        res["graph_bitwise"] = bool(torch.equal(out, ref))
        for name, fn in (("eager", die_conc), ("graph", g.replay), ("full", lambda: F.linear(x, w))):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                fn()
            torch.cuda.synchronize()
            res[name + "_us"] = round((time.perf_counter() - t0) / 200 * 1e6, 1)
    except Exception as e:  # noqa: BLE001
        res["graph_error"] = repr(e)[:400]
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
