# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP4 drafter's mailbox Markov sampling (``runtime.draft.VocabParallelMarkov``
with a peer mailbox, as deployed) per draft step set: the split launches (shard bias
GEMM, ``_vp_markov_keys``, select) against ``_vp_markov_fused`` (the bias tile inside
the keys kernel, then select) at several vocabulary columns per program. Probabilistic
drafting, Markov rank 512, vocabulary 163840, k = 3; median graph replay, max over
ranks. Also checks the fused drafts and cache rows against the split ones (bitwise
when the bias is order-exact: ``--grid``).

    torchrun --nproc-per-node 4 bench_draft_markov_fused.py [--grid]
"""

import argparse
import os

import torch
from bench_draft_markov import RANK, STEPS, VOCAB, time_graph

BATCHES = (8, 16, 32, 64, 96, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--bn", default="64,128,256")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    torch.distributed.init_process_group("nccl")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(1)

    def weight():
        if args.grid:
            w = torch.randint(-8, 9, (VOCAB, RANK), generator=gen, device=device)
            return (w.float() / 64).bfloat16()
        return (torch.randn((VOCAB, RANK), generator=gen, device=device) * 0.05).bfloat16()

    w1, w2 = weight(), weight()
    batches = [int(b) for b in args.batches.split(",")]
    group = torch.distributed.group.WORLD
    markov = draft.VocabParallelMarkov(
        w1, w2, world, rank, STEPS, group=group, cap=max(batches)
    )
    width = markov.vocab_rank
    for batch in batches:
        local = torch.randn((batch * STEPS, width), generator=gen, device=device)
        local = local.bfloat16()
        anchors = torch.randint(0, VOCAB, (batch,), generator=gen, device=device,
                                dtype=torch.int32)  # fmt: skip
        temperature = torch.rand(batch, generator=gen, device=device) + 0.5
        seeds = torch.arange(batch, dtype=torch.int64, device=device) * 7919
        slots = torch.arange(batch, device=device).repeat_interleave(STEPS)
        positions = torch.arange(batch * STEPS, dtype=torch.int64, device=device) + 9
        outs = {}
        line = []
        for arm in ["split"] + [f"bn{b}" for b in args.bn.split(",")]:
            if arm == "split":
                draft.VP_MARKOV_FUSED_ROWS = 0
            else:
                draft.VP_MARKOV_FUSED_ROWS = 1 << 30
                draft.VP_MARKOV_BN = int(arm[2:])
            cache = torch.zeros((batch, STEPS, VOCAB), dtype=torch.bfloat16,
                                device=device)  # fmt: skip
            out = torch.zeros((batch, STEPS), dtype=torch.int64, device=device)
            sampling = (temperature, seeds, slots, positions, cache)
            torch.distributed.barrier()
            markov.run(local, anchors, out, sampling)  # (compile, eager)
            torch.accelerator.synchronize(device)
            outs[arm] = (out.clone(), cache.clone())
            us = time_graph(
                lambda: markov.run(local, anchors, out, sampling), device
            )
            line.append(f"{arm} {us:7.1f}")
        ref_d, ref_c = outs["split"]
        diffs = []
        for arm, (d, c) in outs.items():
            if arm == "split":
                continue
            cols = slice(rank * width, (rank + 1) * width)
            nd = (d != ref_d).sum().item()
            nc = (c[:, :, cols] != ref_c[:, :, cols]).sum().item()
            diffs.append(f"{arm} drafts {nd} cache {nc}")
        if rank == 0:
            print(f"B{batch:>4}: " + ", ".join(line) + " us | " + "; ".join(diffs),
                  flush=True)  # fmt: skip
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
