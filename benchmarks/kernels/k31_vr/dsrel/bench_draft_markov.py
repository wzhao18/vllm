# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release drafter's Markov sampling on the TP4 drafter (C512's SP decode role:
``runtime.draft.VocabParallelMarkov``, one set of launches per draft position)
against the stock probabilistic loop, in CUDA graphs over the local GPUs (torchrun,
one process per GPU; Rubin):

  stock: the LM-head shard's logits all-gathered to the full vocabulary, then per
         position F.linear(markov_w1[prev], markov_w2) over the full vocabulary
         (the replicated [163840, 512] weight), the BF16 add and ``gumbel_sample``
         writing the whole draft-logits cache row;
  spec:  VocabParallelMarkov on this rank's vocabulary shard (no logits gather:
         this rank's cache columns, the keys all-gathered).

Markov rank 512, vocabulary 163840, k = 3 (C512), probabilistic drafting; per batch
the median replay time, the max over ranks. At 2 GPUs a rank owns twice TP4's
vocabulary rows (the spec arm's per-rank work is TP4's x2; the stock arm's is the
same at any TP).

    torchrun --nproc-per-node 2 bench_draft_markov.py --out DIR
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F

VOCAB, RANK, STEPS = 163840, 512, 3
BATCHES = (2, 8, 16, 32, 50, 64, 96, 128)


def time_graph(fn, device, iters=30):
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
    ap.add_argument("--out", required=True)
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

    def gather(x):
        out = torch.empty((world * x.shape[0],) + x.shape[1:], dtype=x.dtype,
                          device=x.device)  # fmt: skip
        torch.distributed.all_gather_into_tensor(out, x)
        return out

    markov = draft.VocabParallelMarkov(w1, w2, world, rank, STEPS, gather)
    width = markov.vocab_rank
    cols = torch.arange(STEPS, dtype=torch.int32, device=device)
    results = []
    for batch in BATCHES:
        markov.prepare(batch, STEPS, STEPS, True)
        local = torch.randn((batch * STEPS, width), generator=gen, device=device)
        local = local.bfloat16()
        anchors = torch.randint(0, VOCAB, (batch,), generator=gen, device=device,
                                dtype=torch.int32)  # fmt: skip
        temperature = torch.ones(batch, dtype=torch.float32, device=device)
        seeds = torch.arange(batch, dtype=torch.int64, device=device)
        slots = torch.arange(batch, device=device).repeat_interleave(STEPS)
        positions = torch.arange(batch * STEPS, dtype=torch.int64, device=device) + 9
        cache = torch.zeros((batch, STEPS, VOCAB), dtype=torch.bfloat16, device=device)
        draft_out = torch.zeros((batch, STEPS), dtype=torch.int64, device=device)
        sampling = (temperature, seeds, slots, positions, cache)
        idx = slots.view(batch, STEPS)
        pos = positions.view(batch, STEPS)

        def stock(local=local, anchors=anchors, batch=batch, idx=idx, pos=pos,
                  temperature=temperature, seeds=seeds, cache=cache,
                  draft_out=draft_out):  # fmt: skip
            parts = gather(local.view(batch * STEPS, width).t().contiguous())
            base = parts.view(world * width, batch * STEPS).t()
            base = base.reshape(batch, STEPS, VOCAB)
            prev = anchors.long()
            for i in range(STEPS):
                logits = base[:, i] + F.linear(w1[prev], w2)
                prev = gumbel_sample(
                    logits, idx[:, i].contiguous(), temperature, seeds,
                    pos[:, i] - 1, apply_temperature=True, is_drafting=True,
                    logits_cache=cache, logits_cache_col=cols[i],
                )  # fmt: skip
                draft_out[:, i] = prev

        def spec(local=local, anchors=anchors, draft_out=draft_out, sampling=sampling):
            markov.run(local, anchors, draft_out, sampling)

        stock_us = time_graph(stock, device)
        spec_us = time_graph(spec, device)
        results.append((batch, stock_us, spec_us, spec_us / stock_us))
        if rank == 0:
            print(
                f"B{batch:>4}: stock {stock_us:8.1f} us, vocabulary-parallel "
                f"{spec_us:8.1f} us ({spec_us / stock_us:.3f})",
                flush=True,
            )
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(
            os.path.join(args.out, f"markov_w{world}.csv"), "w", newline=""
        ) as fh:
            w = csv.writer(fh)
            w.writerow(["batch", "stock_us", "spec_us", "ratio"])
            w.writerows(results)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
