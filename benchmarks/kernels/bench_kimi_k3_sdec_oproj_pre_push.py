# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP o-projection "pre_push" GEMM (the vendored sm_107 MXFP8 pre GEMM with the
reduce-scatter push epilogue) against the same GEMM writing its local output, on one
GPU: the four row owners' receive slots are local buffers, so this measures the GEMM
and its epilogue without NVLink. Bitwise check of the pushed words against the
sanitized local output, then CUDA-graph timing with rotating cold weight copies.

  python bench_kimi_k3_sdec_oproj_pre_push.py --rows 128,256,384
"""

import argparse

import torch

WIDTH, K, TP = 7168, 3072, 4
SENTINEL = -0x80000000


def mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    buf = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=buf)
    return buf


def graph_time(fn, reps=16, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(warm):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--copies", type=int, default=8)
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import pre_tactic
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107.prewait import (
        Mxfp8PreGemm,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    w = mx(torch.randn(WIDTH, K, device=dev).bfloat16() * K**-0.5)
    copies = [(w.data.clone(), w.scale.clone()) for _ in range(args.copies)]
    for rows in [int(r) for r in args.rows.split(",")]:
        n = rows // TP
        cap = n
        tactic, prewait, stages = pre_tactic(WIDTH, K, rows)
        x = mx(torch.randn(rows, K, device=dev).bfloat16())
        local = Mxfp8PreGemm(WIDTH, K, rows, tactic, prewait=prewait, max_stages=stages)
        push = Mxfp8PreGemm(
            WIDTH, K, rows, tactic, prewait=prewait, max_stages=stages, push=True
        )
        out = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
        recv = torch.empty(TP, cap, WIDTH, dtype=torch.bfloat16, device=dev)
        recv.view(torch.int32).fill_(SENTINEL)
        peers = torch.tensor(
            [recv[o].data_ptr() for o in range(TP)], dtype=torch.int64, device=dev
        )
        # this rank = 0: owner o's slot [0] holds rows o n .. o n + n - 1
        pargs = Mxfp8PreGemm.push_args(peers, 0, n, cap, rows)
        local(x.data, x.scale, w.data, w.scale, out)
        push(x.data, x.scale, w.data, w.scale, out, pargs)
        torch.cuda.synchronize()
        words = out.view(torch.int32).clone()
        words[words == SENTINEL] = 0
        got = recv.view(torch.int32).reshape(TP * cap, WIDTH // 2)[:rows]
        same = (got == words).float().mean().item()
        t_local = graph_time(
            lambda i: local(x.data, x.scale, *copies[i % len(copies)], out)
        )
        t_push = graph_time(
            lambda i: push(x.data, x.scale, *copies[i % len(copies)], out, pargs)
        )
        print(
            f"R{rows} tactic {tactic}: pushed words bitwise {same * 100:.3f}%  "
            f"pre {t_local:.2f} us  pre_push {t_push:.2f} us/call"
        )


if __name__ == "__main__":
    main()
