# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GPU, the TP4 rank's Markov position (vocab_rank 40960): ``_vp_markov_keys``
block/warp variants (greedy / sampled) next to the same bytes as a plain BF16 add
(base + bias -> cache rows), each alone in a CUDA graph."""

import argparse

import torch

from vllm.triton_utils import triton

VOCAB_RANK, STEPS = 40960, 3


def bench(fn, reps=20, iters=30):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=96)
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    dev = torch.device("cuda")
    B = args.batch
    base = torch.randn((B * STEPS, VOCAB_RANK), device=dev).bfloat16()
    bias = torch.randn((B, VOCAB_RANK), device=dev).bfloat16()
    keys = torch.zeros(B, dtype=torch.int64, device=dev)
    temp = torch.ones(B, dtype=torch.float32, device=dev)
    seeds = torch.arange(B, dtype=torch.int64, device=dev)
    slots = torch.arange(B, device=dev).repeat_interleave(STEPS)
    pos = torch.arange(B * STEPS, dtype=torch.int64, device=dev) + 9
    cache = torch.zeros((B, STEPS, VOCAB_RANK * 4), dtype=torch.bfloat16, device=dev)
    rows = base.view(B, STEPS, VOCAB_RANK)[:, 1]
    out = cache[:, 1, :VOCAB_RANK]
    print(f"B{B} plain add (same bytes): {bench(lambda: torch.add(rows, bias, out=out)):.2f} us")
    for sampled in (False, True):
        for block, warps in ((1024, 4), (1024, 8), (2048, 8), (4096, 8), (512, 4),
                             (256, 2), (2048, 4)):  # fmt: skip
            grid = (B, triton.cdiv(VOCAB_RANK, block))

            def run():
                return draft._vp_markov_keys[grid](
                    keys, base, base.stride(0), bias, bias.stride(0), cache,
                    cache.stride(0), cache.stride(1), slots, pos, temp, seeds, 1,
                    STEPS, 0, VOCAB_RANK, keys, keys, 0, grid[1], SAMPLED=sampled,
                    BLOCK=block, TP=4, PUBLISH=False, num_warps=warps,
                )  # fmt: skip

            k = run()
            print(f"B{B} keys sampled={int(sampled)} block{block} w{warps}: "
                  f"{bench(run):.2f} us regs={getattr(k, 'n_regs', None)}",
                  flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
