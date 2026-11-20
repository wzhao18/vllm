# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3.1 dense MLP (layer 0, replicated under SP: gate_up 67584 x 7168 and down
7168 x 33792 BF16, 1.45 GB per step) on one Rubin GPU: today's single-stream cuBLAS
chain (default striped weights) vs a die-split chain: each die's green-context stream
runs the gate / up rows of its half of the intermediate dim and the down rows of its
half of the hidden dim, from weights homed on that die (``common.dies``).

Chain per call: gate_up -> act -> down; the die split forks both dies after the
input, joins after the activation (each die's down reads the whole intermediate
row) and joins again at the end. CUDA graph, median us per call; each output is
checked against the stock chain."""

import argparse
import json
import statistics

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies


def act(gu, out=None):
    gate, up = gu.chunk(2, dim=-1)
    return torch.mul(torch.nn.functional.silu(gate), up, out=out)


def graph_time(fn, reps=10, samples=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(samples):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,64,96")
    ap.add_argument("--inter", type=int, default=33792)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--coscheduled", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    idx = dev.index
    n_dom = dies.domain_count(idx)
    assert n_dom == 2, f"needs a two-die device, found {n_dom}"
    torch.manual_seed(0)
    inter, hidden = args.inter, args.hidden
    half_i, half_h = inter // 2, hidden // 2
    w_gu = (torch.randn(2 * inter, hidden, device=dev) * 0.02).to(torch.bfloat16)
    w_dn = (torch.randn(hidden, inter, device=dev) * 0.02).to(torch.bfloat16)
    # Die d: gate rows [d*I/2, (d+1)*I/2) then the same up rows; down rows of hidden
    # half d. Every 2 MiB chunk of die d's copies is homed on die d.
    gu_d, dn_d = [], []
    for d in range(2):
        rows = torch.cat((w_gu[d * half_i:(d + 1) * half_i],
                          w_gu[inter + d * half_i:inter + (d + 1) * half_i]))  # fmt: skip
        gu_d.append(dies.localized_like(rows.contiguous(), d))
        dn_d.append(dies.localized_like(w_dn[d * half_h:(d + 1) * half_h].contiguous(), d))
    streams = [dies.domain_stream(idx, d, args.coscheduled) for d in range(2)]
    for n in (int(r) for r in args.rows.split(",")):
        x = torch.randn(n, hidden, device=dev).to(torch.bfloat16)
        gu = torch.empty(n, 2 * inter, device=dev, dtype=torch.bfloat16)
        h = torch.empty(n, inter, device=dev, dtype=torch.bfloat16)
        y = torch.empty(n, hidden, device=dev, dtype=torch.bfloat16)
        gu_s = [torch.empty(n, 2 * half_i, device=dev, dtype=torch.bfloat16) for _ in range(2)]
        y_d = [torch.empty(n, half_h, device=dev, dtype=torch.bfloat16) for _ in range(2)]
        h2 = torch.empty(n, inter, device=dev, dtype=torch.bfloat16)

        def stock():
            torch.mm(x, w_gu.t(), out=gu)
            act(gu, out=h)
            torch.mm(h, w_dn.t(), out=y)

        def split():
            main = torch.cuda.current_stream()
            fork = torch.cuda.Event()
            fork.record(main)
            mid = [torch.cuda.Event() for _ in range(2)]
            end = [torch.cuda.Event() for _ in range(2)]
            for d in range(2):
                with torch.cuda.stream(streams[d]):
                    streams[d].wait_event(fork)
                    torch.mm(x, gu_d[d].t(), out=gu_s[d])
                    act(gu_s[d], out=h2[:, d * half_i:(d + 1) * half_i])
                    mid[d].record(streams[d])
            for d in range(2):
                with torch.cuda.stream(streams[d]):
                    streams[d].wait_event(mid[1 - d])
                    torch.mm(h2, dn_d[d].t(), out=y_d[d])
                    end[d].record(streams[d])
            for d in range(2):
                main.wait_event(end[d])

        stock()
        split()
        torch.cuda.synchronize()
        y2 = torch.cat(y_d, dim=1)
        ref = (act(x.float() @ w_gu.float().t()).bfloat16().float() @ w_dn.float().t())
        scale = ref.abs().max().item()
        out = {
            "rows": n,
            "stock_err": (y.float() - ref).abs().max().item() / scale,
            "split_err": (y2.float() - ref).abs().max().item() / scale,
            "split_eq_stock_h": bool(torch.equal(h, h2)),
            "stock_us": round(graph_time(stock), 2),
            "split_us": round(graph_time(split), 2),
        }
        gb = (w_gu.numel() + w_dn.numel()) * 2 / 1e9
        out["stock_TBps"] = round(gb / out["stock_us"] * 1e3, 2)
        out["split_TBps"] = round(gb / out["split_us"] * 1e3, 2)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
