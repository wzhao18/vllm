# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's 5-layer forward with its REAL neighbours: bench_draft_forward's
specialized chain (GqaSites: split-K qkv, ring / two-shot o_proj and down producers, SiLU-fused
gate/up, the all-reduce + norm sites) with the real GqaWindow attention core in place of the
stand-in (cutedsl.draft.gqa_attention state, full 2048-key windows at 150K-token contexts, each
layer its own cache). Graph replay time (median, max over ranks) per row count, optionally with
every replay cold (--flush-mb: weights and KV out of L2, the target forward's place).

``--gqa-debug N`` compiles GqaWindow with debug bits N (gqa_window.GqaWindow: 1 / 2 / 4 / 8 skip
the math / the TMA loads / the prep / the combine, 16 returns after the PDL wait, 32 globaltimer
phase stamps). With 32, one extra replay records every layer's per-CTA phase stamps and rank 0
prints, per row count, the median over layers of: lead (first CTA's wait return - first CTA
entry), and per phase k (the last CTA's phase k - the first CTA's wait return).

    torchrun --nproc-per-node 4 bench_draft_chain.py --out DIR --rows 7,14,28,56 \
        --flush-mb 512 --gqa-debug 32
"""

import argparse
import csv
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_draft_forward as bf  # noqa: E402
import bench_gqa_attention_int as gi  # noqa: E402

Q, CTX, PAGE = 7, 150000, 1024
PHASES = ("entry", "wait", "loads", "prep", "keys", "merged", "landed", "end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="7,14,28,56")
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--gqa-debug", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from types import SimpleNamespace

    from cutlass import Int64

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.allreduce import (
        PreparedDraftAr,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    rows = [int(r) for r in args.rows.split(",")]
    debug = args.gqa_debug
    if debug:
        original = ga.compile_window

        def compile_debug(*a, **k):
            k["debug"] = debug
            return original(*a, **k)

        ga.compile_window = compile_debug
    with set_current_vllm_config(config):
        init_distributed_environment(world, rank, local_rank=local, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        group = get_tp_group().device_group
        for name in dir(draft):
            table = getattr(draft, name)
            if name.startswith("GQA_") and isinstance(table, dict) and 16 in table:
                table[world] = table[16]
        act = SiluAndMul()
        gen = torch.Generator(device=device).manual_seed(100 + rank)
        layers = [bf.Layer(i, gen, device, act, RMSNorm) for i in range(5)]
        # the real attention core: TP16 per-rank geometry (6 query heads, 1 KV head)
        attn_layers = gi.make_layers(1, device)
        state = ga.install_gqa_attention(
            attn_layers,
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=Q),
                model_config=SimpleNamespace(max_model_len=gi.MAX_POS),
            ),
            rows, [r // Q * (Q + 1) for r in rows], True,
        )  # fmt: skip
        assert state is not None
        current = {"positions": None, "stamps": 0}

        for i, layer in enumerate(layers):

            def core(positions, hidden, _i=i, _layer=layer):
                qkv, _ = _layer.self_attn.qkv_proj(hidden)
                if debug & 32:
                    current["stamps"] = stamp_bufs[_i].data_ptr()
                out = state.forward(_i, qkv, current["positions"])
                assert out is not None
                return out

            layer.attention_core = core
        flush = None
        if args.flush_mb:
            flush = torch.empty(args.flush_mb << 20, dtype=torch.uint8, device=device)
        shared = torch.Generator(device=device).manual_seed(7)
        vocab = 1024 * world
        table = torch.randn((vocab, bf.HIDDEN), generator=shared, device=device).bfloat16()
        shard = vocab // world
        final = RMSNorm(bf.HIDDEN, eps=bf.EPS).to(device)
        model = type("Model", (), {})()
        model.layers, model.norm = layers, final
        model.has_separate_mask_embedding = False
        model.embed_tokens = type("Embed", (), {})()
        model.embed_tokens.weight = table[rank * shard : (rank + 1) * shard].contiguous()
        wrapper = type("Wrapper", (), {})()
        wrapper.model = type("Inner", (), {})()
        wrapper.model.layers = layers
        draft.install_gqa_projections(wrapper, world, rows, [], rank)
        top = max(rows)
        policy = lambda t, r: draft.covered(  # noqa: E731
            draft.closed(t.get(world, ""), top), r, 7
        )
        ar = PreparedDraftAr(
            group, device, world, rank, rows, vocab,
            {r for r in rows if policy(draft.GQA_FAST_AR, r) == 2},
            {r for r in rows if policy(draft.GQA_GATED_AR, r)},
            {r for r in rows if policy(draft.GQA_TWO_SHOT_EARLY, r)},
            {r for r in rows if policy(draft.GQA_VALID_AR, r)},
            {r for r in rows if policy(draft.GQA_FAST_AR, r) == 3},
        )  # fmt: skip
        draft.GqaDraftFastPath.prepare_producers(ar, layers, world, policy)
        sites = draft.GqaSites(ar)
        # KV caches: 5 layers, full windows, one disjoint page set per request
        L = CTX + Q
        lo = L - Q - (gi.WINDOW - 1)
        pages = list(range(lo // PAGE, (L - 1) // PAGE + 1))
        Bmax = max(rows) // Q
        blocks = Bmax * len(pages) + 2
        layer_page = PAGE * 2 * gi.HEAD
        stride = 5 * layer_page + 128
        buffer = torch.empty(blocks * stride, dtype=torch.bfloat16, device=device)
        buffer.normal_()
        for i, la in enumerate(attn_layers):
            la.attn.bind_kv_cache(
                buffer.as_strided((blocks, 1, PAGE, 2 * gi.HEAD),
                                  (stride, PAGE * 2 * gi.HEAD, 2 * gi.HEAD, 1),
                                  i * layer_page)
            )  # fmt: skip
        if debug & 32:
            for bound in state.bound.values():
                for shape, kernel in list(bound.kernels.items()):

                    def stamped(*a, _k=kernel):
                        return _k(*a[:-1], Int64(current["stamps"]), a[-1])

                    bound.kernels[shape] = stamped
        os.makedirs(args.out, exist_ok=True)
        out_rows = []
        for r in rows:
            B = r // Q
            perm = (torch.arange(B * len(pages), device=device).view(B, len(pages)) + 1).int()
            tab = torch.zeros((B, (L - 1) // PAGE + 2), dtype=torch.int32, device=device)
            tab[:, pages[0] : pages[-1] + 1] = perm
            seq = torch.full((B,), L, dtype=torch.int32, device=device)
            positions = torch.arange(L - Q, L, device=device).repeat(B)
            pos = positions.cpu()
            slots = (tab.cpu()[torch.arange(B).repeat_interleave(Q), pos // PAGE].long()
                     * PAGE + pos % PAGE).to(device)  # fmt: skip
            md = FlashAttentionMetadata(
                num_actual_tokens=r, max_query_len=Q,
                query_start_loc=torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32,
                                             device=device),
                max_seq_len=L, seq_lens=seq, block_table=tab, slot_mapping=slots,
                causal=True,
            )  # fmt: skip
            names = [la.attn.layer_name for la in attn_layers]
            fc = ForwardContext(
                no_compile_layers={},
                attn_metadata={n: md for n in names},
                slot_mapping={n: slots for n in names},
            )
            current["positions"] = positions
            ctas = 16 * B  # (splits x requests: every TP16 k7 shape up to B32 keeps 16 x B or fewer)
            stamp_bufs = [torch.zeros((ctas * 4, 8), dtype=torch.int64, device=device)
                          for _ in range(5)]  # fmt: skip
            ids = torch.randint(0, vocab, (r,), generator=shared, device=device,
                                dtype=torch.int32)  # fmt: skip

            def run(ids=ids, fc=fc):
                with override_forward_context(fc):
                    sites.run(model, ids, None)

            spec, phases = time_chain(run, device, args.iters, flush, stamp_bufs if debug & 32 else None)
            out_rows.append((r, spec, phases))
            line = f"rows {r:>4} (B{B}) debug {debug}: chain {spec:8.2f} us"
            if phases:
                line += "\n   gqa per layer (median of 5 layers): " + " ".join(
                    f"{k} {v:6.2f}" for k, v in phases.items()
                )
            bf.log(line)
        if rank == 0:
            path = os.path.join(args.out, f"chain_w{world}_d{debug}{args.tag}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["rows", "debug", "chain_us"] + [f"gqa_{p}" for p in PHASES])
                for r, spec, ph in out_rows:
                    w.writerow([r, debug, f"{spec:.2f}"] + [f"{ph.get(p, float('nan')):.3f}" for p in PHASES])
    torch.distributed.barrier()


def time_chain(fn, device, iters, flush, stamp_bufs):
    fn()
    torch.accelerator.synchronize(device)
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for it in range(iters + 1):
        torch.accelerator.synchronize(device)
        if flush is not None:
            flush.zero_()
            torch.accelerator.synchronize(device)
        if stamp_bufs is not None and it == iters:
            for s in stamp_bufs:
                s.zero_()
            torch.accelerator.synchronize(device)
        torch.distributed.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        if it < iters:
            times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    median = torch.tensor([times[len(times) // 2]], device=device)
    torch.distributed.all_reduce(median, op=torch.distributed.ReduceOp.MAX)
    phases = {}
    if stamp_bufs is not None:
        per = {p: [] for p in PHASES}
        per["lead"] = []
        for s in stamp_bufs:
            t = s.cpu()
            t = t[t[:, 0] > 0]
            if t.numel() == 0:
                continue
            w0 = t[:, 1][t[:, 1] > 0].min().item()
            per["lead"].append((w0 - t[:, 0].min().item()) / 1e3)
            for k, p in enumerate(PHASES):
                col = t[:, k][t[:, k] > 0]
                if col.numel():
                    per[p].append((col.max().item() - w0) / 1e3)
        phases = {k: statistics.median(v) for k, v in per.items() if v}
    del graph
    return median.item(), phases


if __name__ == "__main__":
    main()
