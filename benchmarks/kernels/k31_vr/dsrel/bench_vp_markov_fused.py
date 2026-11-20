# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``runtime.draft.VocabParallelMarkov`` as deployed on the TP4 drafter (peer
mailbox, probabilistic drafting, k = 3), one process per GPU: checks the drafts and
the draft-logits cache against the stock loop (full-vocabulary bias GEMM + BF16 add +
``gumbel_sample``), then times ``run`` in a CUDA graph (median replay, max over
ranks). Run the same script on two overlays for an A/B.

    torchrun --nproc-per-node 4 bench_vp_markov_fused.py --tag A
"""

import argparse
import os

import torch
import torch.nn.functional as F

VOCAB, RANK, STEPS = 163840, 512, 3


def time_graph(fn, device, iters):
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        torch.accelerator.synchronize(device)
        torch.distributed.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    median = torch.tensor([times[len(times) // 2]], device=device)
    torch.distributed.all_reduce(median, op=torch.distributed.ReduceOp.MAX)
    del graph
    return median.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--batches", default="32,64,96,128")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--graph-reps", type=int, default=10)
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    torch.distributed.init_process_group("nccl")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(1)
    w1 = (torch.randn((VOCAB, RANK), generator=gen, device=device) * 0.05).bfloat16()
    w2 = (torch.randn((VOCAB, RANK), generator=gen, device=device) * 0.05).bfloat16()
    batches = [int(b) for b in args.batches.split(",")]
    group = torch.distributed.group.WORLD
    markov = draft.VocabParallelMarkov(
        w1, w2, world, rank, STEPS, group=group, cap=max(batches)
    )
    width = markov.vocab_rank
    lo = rank * width
    cols = torch.arange(STEPS, dtype=torch.int32, device=device)
    for batch in batches:
        markov.prepare(batch, STEPS, STEPS, True)
        full = torch.randn((batch * STEPS, VOCAB), generator=gen, device=device) * 2
        full = full.bfloat16()
        local = full[:, lo : lo + width].contiguous()
        anchors = torch.randint(0, VOCAB, (batch,), generator=gen, device=device,
                                dtype=torch.int32)  # fmt: skip
        temperature = torch.ones(batch, dtype=torch.float32, device=device)
        temperature[::5] = 0.0
        temperature[1::5] = 0.7
        seeds = torch.arange(batch, dtype=torch.int64, device=device) * 7919
        slots = torch.arange(batch, device=device).repeat_interleave(STEPS)
        positions = torch.arange(batch * STEPS, dtype=torch.int64, device=device) + 9
        cache = torch.zeros((batch, STEPS, VOCAB), dtype=torch.bfloat16, device=device)
        out = torch.zeros((batch, STEPS), dtype=torch.int64, device=device)
        sampling = (temperature, seeds, slots, positions, cache)

        # stock loop, on every rank (full vocabulary)
        ref_cache = torch.zeros_like(cache)
        ref = torch.zeros_like(out)
        base = full.view(batch, STEPS, VOCAB)
        idx, pos = slots.view(batch, STEPS), positions.view(batch, STEPS)
        prev = anchors.long()
        for i in range(STEPS):
            logits = base[:, i] + F.linear(w1[prev], w2)
            prev = gumbel_sample(
                logits, idx[:, i].contiguous(), temperature, seeds, pos[:, i] - 1,
                apply_temperature=True, is_drafting=True, logits_cache=ref_cache,
                logits_cache_col=cols[i],
            )  # fmt: skip
            ref[:, i] = prev
        markov.run(local, anchors, out, sampling)
        torch.accelerator.synchronize(device)
        mine = cache[:, :, lo : lo + width].float()
        theirs = ref_cache[:, :, lo : lo + width].float()
        diff = (mine - theirs).abs()
        # a draft that differs changes every later position's bias: compare rows
        # whose earlier drafts agree
        agree = (out == ref).all(dim=1)
        bad_first = (out[:, 0] != ref[:, 0]).sum().item()
        if rank == 0:
            print(
                f"[{args.tag}] B{batch}: drafts equal {agree.sum().item()}/{batch} "
                f"(first-position mismatches {bad_first}); cache max|d| "
                f"{diff[agree].max().item() if agree.any() else -1:.4g}, "
                f"elements differing {(diff[agree] != 0).float().mean().item():.2e}",
                flush=True,
            )

        def spec(local=local, anchors=anchors, out=out, sampling=sampling, n=args.graph_reps):
            for _ in range(n):
                markov.run(local, anchors, out, sampling)

        us = time_graph(spec, device, args.iters) / args.graph_reps
        if rank == 0:
            print(f"[{args.tag}] B{batch}: run {us:8.2f} us ({us / STEPS:.2f} per position)",
                  flush=True)  # fmt: skip
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
