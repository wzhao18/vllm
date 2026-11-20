# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GPU, the TP4 rank's Markov position (vocab_rank 40960, rank 512): the stock
cuBLAS bias GEMM, and ``_vp_markov_fused`` tile variants (greedy / sampled), each
timed alone in a CUDA graph; prints registers, spills and the MMA instruction."""

import argparse
import re

import torch
import torch.nn.functional as F

from vllm.triton_utils import triton

VOCAB_RANK, RANK, STEPS = 40960, 512, 3


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
    w1 = (torch.randn((VOCAB_RANK * 4, RANK), device=dev) * 0.05).bfloat16()
    w2 = (torch.randn((VOCAB_RANK, RANK), device=dev) * 0.05).bfloat16()
    base = torch.randn((B * STEPS, VOCAB_RANK), device=dev).bfloat16()
    anchors = torch.randint(0, VOCAB_RANK * 4, (B,), device=dev, dtype=torch.int32)
    emb = w1[anchors.long()]
    keys = torch.zeros(B, dtype=torch.int64, device=dev)
    temp = torch.ones(B, dtype=torch.float32, device=dev)
    seeds = torch.arange(B, dtype=torch.int64, device=dev)
    slots = torch.arange(B, device=dev).repeat_interleave(STEPS)
    pos = torch.arange(B * STEPS, dtype=torch.int64, device=dev) + 9
    cache = torch.zeros((B, STEPS, VOCAB_RANK * 4), dtype=torch.bfloat16, device=dev)
    print(f"B{B} cuBLAS bias GEMM: {bench(lambda: F.linear(emb, w2)):.2f} us", flush=True)
    for sampled in (False, True):
        for bm, bn, bk, warps, stages in (
            (128, 128, 64, 8, 3),
            (128, 64, 64, 4, 3),
            (128, 64, 64, 8, 2),
            (64, 128, 64, 4, 3),
            (128, 32, 64, 4, 3),
            (128, 256, 64, 8, 2),
        ):
            if bm < B and bm != 128:
                continue
            grid = (VOCAB_RANK // bn, triton.cdiv(B, bm))

            def run():
                return draft._vp_markov_fused[grid](
                    keys, anchors, 1, w1, w2, base, base.stride(0), cache,
                    cache.stride(0), cache.stride(1), slots, pos, temp, seeds,
                    B, 1, STEPS, 0, keys, keys, 0, grid[0], SAMPLED=sampled,
                    BM=bm, BN=bn, BK=bk, RANK=RANK, TP=4, PUBLISH=False,
                    num_warps=warps, num_stages=stages,
                )  # fmt: skip

            k = run()
            ptx = k.asm.get("ptx", "")
            mma = "tcgen05" if "tcgen05.mma" in ptx else (
                "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "?"))
            spills = getattr(k, "n_spills", None)
            regs = getattr(k, "n_regs", None)
            sass_local = len(re.findall(r"\bld\.local|\bst\.local", ptx))
            us = bench(run)
            print(f"B{B} fused sampled={int(sampled)} bm{bm} bn{bn} bk{bk} w{warps} "
                  f"s{stages}: {us:.2f} us mma={mma} regs={regs} spills={spills} "
                  f"ptx_local={sass_local}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
