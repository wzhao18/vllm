# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's fused Markov drafting (cutedsl.draft.markov MarkovGreedy at <= 8
requests, markov_wide.MarkovWide above, consecutive launches over request slices
past one launch's width) at Markov rank 512 (kimi-k3.1-pro-dspark-preview) and 256
(kimi-k3-dspark: the previous reference) against the stock Markov loop, sampled
(production: probabilistic drafting) and greedy, draft lengths 3 (q4) and 7 (q8).

torchrun --nproc-per-node 4 bench_draft_markov.py --out DIR

Emulation (4 GPUs for TP16): each rank holds the production per-rank vocabulary
shard (10240 rows: the same CTAs per rank, 40 at rank 256 / 80 at rank 512), so the
kernel's per-rank work is the production one; the key exchange runs across 4 ranks
instead of 16 (fewer mailbox chunks to poll: a lower bound of the exchange).

Stock (one GPU, the replicated computation every TP16 rank runs; DSparkSpeculator
._sample_sequential): per position bias = F.linear(markov_w1[prev], markov_w2) over the
full 163840-token vocabulary (BF16), logits = base + bias, then torch.argmax (greedy)
or gumbel_sample (sampled, writing the draft-logits cache) -- plus, reported apart
(``ag_us``) and added into stock_us, the base-logits all-gather that the fused path
also removes (NCCL all_gather of a [rows, 163840 / 4] block per rank into the full
[4, rows, 40960] = production volume per receiving rank within 1/16, + the vocab
interleave copy).

Specialized timing: the kernels in a CUDA graph, each call preceded by a small plain
kernel (as the LM head separates them in serving; its own time measured and
subtracted). Every rank replays together after a barrier.

Cases, in priority order (a soft budget skips the rest):
  prod      the production configuration (runtime.draft FUSED_MARKOV / SETTLED_MARKOV:
            B1 gated + settled, B2 gated, B3-B8 gated + staged, B9+ MarkovWide, launch
            plan of markov.PreparedMarkov.plan at TP16: ``prod_plan16``, since the
            bench's PreparedMarkov has tp 4 and launch_width depends on tp) at every
            B, rank 512 and 256, sampled, q8 and q4
  variant   NQ-TUNE 1/2: plain / gated / gated+staged / gated+settled /
            gated+staged+settled at B in --variant-batches, rank 512 (and 256)
  width     NQ-TUNE 3: forced MarkovWide plans (launch widths 32 / 64 / the device's
            launch_width / 128) at B >= 9, rank 512 (and 256)
  greedy    the production configuration, greedy
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    NEW_RANK,
    OLD_RANK,
    TP,
    VOCAB,
    Deadline,
    Table,
    guarded,
    log,
    time_eager,
    time_graph,
    time_graph_agreed,
)

VOCAB_RANK = VOCAB // TP  # 10240
BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 48, 56, 64, 72, 96, 120, 128,
           130, 160, 192, 256]  # fmt: skip
VARIANTS = {
    "plain": dict(gated=False, staged=False, settled=False),
    "gated": dict(gated=True, staged=False, settled=False),
    "staged": dict(gated=True, staged=True, settled=False),
    "settled": dict(gated=True, staged=False, settled=True),
    "staged_settled": dict(gated=True, staged=True, settled=True),
}


def prod_variant(draft, batch, tp=TP):
    """runtime.draft's MarkovGreedy flags at ``batch`` (B <= 8)."""
    table = draft.batch_table(draft.closed(draft.FUSED_MARKOV.get(tp, ""), 256))
    settle = draft.batch_table(draft.SETTLED_MARKOV.get(tp, ""))
    v = table.get(batch)
    flags = dict(gated=v in (2, 3), staged=v == 3, settled=bool(settle.get(batch)))
    for name, f in VARIANTS.items():
        if f == flags:
            return name
    return "plain"


def wide_plan(batch, width):
    """A MarkovWide plan of request slices of at most ``width`` slots (balanced, as
    PreparedMarkov.plan)."""
    count = -(-batch // width)
    per = -(-batch // count)
    w = -(-per // 8) * 8
    return [(lo, min(lo + w, batch), w) for lo in range(0, batch, w)]


class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="prod,variant,width,greedy")
    ap.add_argument("--ranks", default=f"{NEW_RANK},{OLD_RANK}")
    ap.add_argument("--steps", default="7,3")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--variant-batches", default="1,2,3,4,8")
    ap.add_argument("--width-batches", default="9,16,64,96,120,128,130,192,256")
    ap.add_argument("--budget", type=float, default=0)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        markov,
        markov_wide,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    deadline = Deadline(args.budget)
    lead = rank == 0
    say = log if lead else (lambda m: None)

    def agree(ok):
        """Every rank's flag (MIN): a case runs its cross-rank kernels only when every
        rank compiled it."""
        t = torch.tensor([1 if ok else 0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def budget_ok():
        t = torch.tensor([0 if deadline.out() else 1], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    out_dir = args.out
    if lead:
        os.makedirs(out_dir, exist_ok=True)
    tab = Table(os.path.join(out_dir, "markov.csv") if lead else None, echo=lead,
                extra_cols=["markov_rank", "mode", "steps", "batch", "launches",
                            "loop_us", "ag_us", "sep_us"])  # fmt: skip
    vocab = world * VOCAB_RANK
    limits = markov.device_limits(dev)
    say(f"world {world}, per-rank vocab {VOCAB_RANK}, device limits {limits}; "
        f"launch widths (sampled) r512 "
        f"{markov_wide.launch_width(TP, 512, True, limits, VOCAB_RANK)} r256 "
        f"{markov_wide.launch_width(TP, 256, True, limits, VOCAB_RANK)}")  # fmt: skip

    ranks = [int(r) for r in args.ranks.split(",")]
    steps_list = [int(s) for s in args.steps.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    parts = args.parts.split(",")
    gen = torch.Generator(device=dev).manual_seed(4321)

    # separator kernel between calls (the LM head's place)
    scratch = torch.zeros(1024, device=dev)
    sep = lambda: scratch.add_(1.0)  # noqa: E731
    sep_us = time_graph([sep], calls=256)
    say(f"separator kernel {sep_us:.2f} us (subtracted)")

    # ------------------------------------------------------------ inputs
    def inputs(batch, steps, vocab_total, sampled, cache_vocab):
        base = torch.randn((batch * steps, vocab_total), device=dev,
                           generator=gen).bfloat16()  # fmt: skip
        anchors = torch.randint(0, vocab_total, (batch,), device=dev, generator=gen,
                                dtype=torch.int32)  # fmt: skip
        if not sampled:
            return base, anchors, None
        max_reqs = batch + 5
        temperature = torch.rand((max_reqs,), device=dev, generator=gen) + 0.5
        temperature[::5] = 0.0
        seeds = torch.randint(0, 1 << 62, (max_reqs,), device=dev, generator=gen,
                              dtype=torch.int64)  # fmt: skip
        slots = torch.randperm(max_reqs, device=dev, generator=gen)[:batch]
        start = torch.randint(1, 1 << 20, (batch,), device=dev, generator=gen,
                              dtype=torch.int64)  # fmt: skip
        positions = (start[:, None] + torch.arange(steps, device=dev)).reshape(-1)
        slots = slots.repeat_interleave(steps).contiguous()
        cache = torch.zeros((max_reqs, steps, cache_vocab), dtype=torch.bfloat16,
                            device=dev)  # fmt: skip
        return base, anchors, (temperature, seeds, slots, positions.contiguous(), cache)

    # ------------------------------------------------------------- stock
    stock_cache = {}
    full = {}

    def full_weights(mrank):
        if mrank not in full:
            g = torch.Generator(device=dev).manual_seed(77 + mrank)
            w1 = (torch.randn((VOCAB, mrank), device=dev, generator=g) * 0.05).bfloat16()
            w2 = (torch.randn((VOCAB, mrank), device=dev, generator=g) * 0.05).bfloat16()
            full[mrank] = (w1, w2)
        return full[mrank]

    def stock_loop(mrank, batch, steps, sampled):
        """us of the stock sequential loop over the full vocabulary (rank 0)."""
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        w1, w2 = full_weights(mrank)
        base, anchors, smp = inputs(batch, steps, VOCAB, sampled, VOCAB)
        base3 = base.view(batch, steps, VOCAB)
        cols = torch.arange(steps, dtype=torch.int32, device=dev)

        def loop():
            prev = anchors.long()
            for i in range(steps):
                logits = base3[:, i] + torch.nn.functional.linear(w1[prev], w2)
                if smp is None:
                    prev = torch.argmax(logits, dim=-1)
                else:
                    temperature, seeds, slots, positions, cache = smp
                    sl = slots.view(batch, steps)[:, i].contiguous()
                    pos = positions.view(batch, steps)[:, i] - 1
                    prev = gumbel_sample(
                        logits, sl, temperature, seeds, pos, apply_temperature=True,
                        is_drafting=True, logits_cache=cache, logits_cache_col=cols[i],
                    )  # fmt: skip

        return time_graph([loop], calls=16)

    def all_gather_us(rows):
        """NCCL all-gather of the base logits (production volume) + interleave copy;
        every rank together."""
        part = torch.randn((rows, VOCAB // world), device=dev).bfloat16()
        out = torch.empty((world, rows, VOCAB // world), dtype=torch.bfloat16,
                          device=dev)  # fmt: skip

        def ag():
            dist.all_gather_into_tensor(out, part, group=group)
            return out.permute(1, 0, 2).reshape(rows, VOCAB)

        # (eager, every rank together: a capture failing on one rank only would
        # desynchronize the collectives)
        return time_eager(ag, iters=30, sync=dist.barrier)

    def stock_us(mrank, batch, steps, sampled):
        key = mrank, batch, steps, sampled
        if key not in stock_cache:
            loop = None
            if lead:
                loop = guarded(log, f"stock loop r{mrank} B{batch} K{steps}", stock_loop,
                               mrank, batch, steps, sampled)  # fmt: skip
            ag = guarded(say, f"all-gather {batch * steps} rows", all_gather_us,
                         batch * steps)  # fmt: skip
            stock_cache[key] = (loop, ag)
            torch.cuda.empty_cache()
        return stock_cache[key]

    # -------------------------------------------------------- specialized
    preps, heads = {}, {}

    def prepared(mrank, steps, variant):
        """One PreparedMarkov per (rank, steps, variant): every batch <= 8 compiled
        with that variant's flags (the compile key is the batch); only "plain" holds
        the MarkovWide mailboxes (B > 8 always runs it). Collective: every rank
        constructs the same instances in the same order."""
        key = mrank, steps, variant
        if key not in preps:
            if mrank not in heads:
                g = torch.Generator(device=dev).manual_seed(1234 + mrank)
                shape = (vocab, mrank)
                heads[mrank] = tuple(
                    (torch.randn(shape, device=dev, generator=g) * 0.05).bfloat16()
                    for _ in range(2)
                )
            w1, w2 = heads[mrank]
            f = VARIANTS[variant]
            small = range(1, 9)
            preps[key] = markov.PreparedMarkov(
                w1, w2, world, rank, group, steps,
                gated=small if f["gated"] else (),
                staged=small if f["staged"] else (),
                settled=small if f["settled"] else (),
                wide=variant == "plain",
            )  # fmt: skip
        return preps[key]

    def spec_case(mrank, steps, variant, batch, sampled, plan=None):
        """(us, launches) of the fused drafting of ``batch`` requests."""
        key = batch, sampled
        saved = p = None
        try:
            p = prepared(mrank, steps, variant)
            saved = p.plans.get(key)
            if plan is not None:
                p.plans[key] = plan
            launches = p.plan(batch, sampled)
            if not launches:
                raise ValueError("not covered (no launch fits)")
            t0 = time.time()
            for index, (lo, hi, width) in enumerate(launches):
                chained = bool(width) and index > 0
                p.kernel(hi - lo, 1, steps, sampled, width, chained)
            compile_s = time.time() - t0
        except Exception:
            agree(False)
            if plan is not None and p is not None:
                p.plans.pop(key, None)
                if saved is not None:
                    p.plans[key] = saved
            raise
        if not agree(True):
            raise RuntimeError("another rank failed to compile")
        try:
            # inputs + sampling args on every rank agreed before any cross-rank launch
            # (a one-sided failure here would leave the others in the exchange)
            err = None
            try:
                base_full, anchors, smp = inputs(batch, steps, vocab, sampled, vocab)
                lo = rank * VOCAB_RANK
                base = base_full.view(batch, steps, vocab)[:, :, lo : lo + VOCAB_RANK]
                base = base.reshape(batch * steps, VOCAB_RANK).contiguous()
                del base_full
                draft_out = torch.zeros((batch, steps), dtype=torch.int64, device=dev)
                sampling = p.sampling_args(*smp) if sampled else None
            except Exception as e:  # noqa: BLE001
                err = e
            if not agree(err is None):
                if err is not None:
                    raise err
                raise RuntimeError("setup failed on another rank")
            fn = lambda: p.run(base, anchors, draft_out, sampling)  # noqa: E731
            dist.barrier()
            torch.cuda.synchronize()
            us = time_graph_agreed([fn], agree, dist.barrier, calls=32,
                                   between=sep) - sep_us  # fmt: skip
            dist.barrier()
        finally:
            if plan is not None:
                p.plans.pop(key, None)
                if saved is not None:
                    p.plans[key] = saved
        return us, launches, compile_s

    def report(case, mrank, steps, batch, sampled, variant, chosen, plan=None, note=""):
        if not budget_ok():
            say(f"budget out: skip {case} r{mrank} K{steps} B{batch}")
            return
        r = guarded(say if lead else log, f"{case} r{mrank} K{steps} B{batch} {chosen}",
                    spec_case, mrank, steps, variant, batch, sampled, plan)  # fmt: skip
        loop, ag = stock_us(mrank, batch, steps, sampled)
        if not lead:
            return
        spec = r[0] if r else None
        launches = r[1] if r else []
        stock = loop + (ag or 0.0) if loop is not None else None
        widths = "+".join(str(w or (hi - lo)) for lo, hi, w in launches)
        tab.add(f"markov_{case}_r{mrank}_{'s' if sampled else 'g'}_K{steps}", batch * steps,
                spec, stock, chosen, (note + f" compile {r[2]:.0f}s" if r else note).strip(),
                f"B{batch}", markov_rank=mrank, mode="sampled" if sampled else "greedy",
                steps=steps, batch=batch, launches=widths, loop_us=loop, ag_us=ag,
                sep_us=sep_us)  # fmt: skip

    def chosen_name(mrank, batch, sampled, variant=None, plan=None):
        if batch <= markov.MAX_WIDTH and plan is None:
            return f"greedy-kernel.{variant}"
        return "wide" + (".forced" if plan is not None else ".prod")

    def prod_plan16(mrank, batch, sampled):
        """PreparedMarkov.plan as production computes it at TP16 (launch_width depends
        on tp through wide_smem; the bench's instances have tp = world = 4). None at
        B <= MAX_WIDTH (one MarkovGreedy launch, no tp dependence)."""
        lw = markov_wide.launch_width(TP, mrank, sampled, limits, VOCAB_RANK)
        return wide_plan(batch, lw) if batch > markov.MAX_WIDTH and lw else None

    # ----------------------------------------------------------- cases
    if "prod" in parts:
        for steps in steps_list:
            for mrank in ranks:
                for batch in batches:
                    v = prod_variant(draft, batch) if batch <= 8 else "plain"
                    report("prod", mrank, steps, batch, True, v,
                           chosen_name(mrank, batch, True, v),
                           plan=prod_plan16(mrank, batch, True))  # fmt: skip
    if "variant" in parts:
        vb = [int(b) for b in args.variant_batches.split(",")]
        for mrank in ranks:
            for steps in steps_list:
                for batch in vb:
                    for v in VARIANTS:
                        tag = "*" if v == prod_variant(draft, batch) else ""
                        report("variant", mrank, steps, batch, True, v,
                               f"greedy-kernel.{v}{tag}")  # fmt: skip
    if "width" in parts:
        wb = [int(b) for b in args.width_batches.split(",")]
        for mrank in ranks:
            lw = markov_wide.launch_width(TP, mrank, True, limits, VOCAB_RANK)
            for steps in steps_list:
                for batch in wb:
                    prod_plan = prod_plan16(mrank, batch, True)
                    widths = sorted({w for w in (32, 64, lw, 128) if 8 <= w <= lw})
                    seen = []
                    for w in widths:
                        plan = wide_plan(batch, w)
                        if plan in seen:
                            continue
                        seen.append(plan)
                        mark = "*" if plan == prod_plan else ""
                        report("width", mrank, steps, batch, True, "plain",
                               f"wide.w{w}x{len(plan)}{mark}", plan=plan)  # fmt: skip
    if "greedy" in parts:
        for steps in steps_list:
            for mrank in ranks:
                for batch in batches:
                    v = prod_variant(draft, batch) if batch <= 8 else "plain"
                    report("prod", mrank, steps, batch, False, v,
                           chosen_name(mrank, batch, False, v),
                           plan=prod_plan16(mrank, batch, False))  # fmt: skip
    if lead:
        tab.finish("markov fused vs stock loop + all-gather")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
