# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill top-k and DCP merge on AgentX 16K-token chunks.

The indexer splits every prefill chunk into sub-chunks of at most
``--max-logits-mb`` FP32 logits, budgeted on the per-request KV lengths that
``DeepseekV32IndexerMetadataBuilder`` plans with, and DeepGEMM writes each
sub-chunk's BF16 logits as wide as the longest of them. This script replays
that plan (with the builder code it runs against) on real DeepGEMM logits
(random fp8 inputs) and sums each path over the sub-chunks of a chunk:

- ``current``: FP32 upcast + ``top_k_per_row_prefill``, then (DCP) the FP32
  ``(score, id)`` pack, one all-gather and the stable select;
- ``bf16``: DeepSelect on the BF16 logits, then (DCP) the BF16 pack and the
  same all-gather and select.

Sections:

- ``tp``: every key on one rank (no DCP).
- ``dcp``: one GPU plays rank 0 of ``--dcp`` ranks; the other ranks' logits
  come from their own keys. Local work is measured (CUDA events after an L2
  flush, median); the all-gathers are pre-built, and their time is an
  estimate from bytes and call count. Also checks that both paths merge
  the same global top-k set on every row.
- ``dist``: launched with ``WORLD_SIZE == --dcp`` (one process per GPU, e.g.
  ``torchrun --nproc-per-node 8``): every rank runs its real local top-k and
  merge, NCCL all-gathers included; the max over ranks of the median is
  reported.

    python benchmarks/kernels/benchmark_kimi_k3_dsa_dcp_prefill_topk.py
"""

import argparse
import os
import statistics

import numpy as np
import torch
import torch.distributed as dist

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
    pack_dcp_topk_candidates_cutedsl,
    stable_topk_from_gathered_candidates_cutedsl,
)
from vllm.model_executor.layers.indexer_topk import deep_select_topk
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

TOPK = 2048
NUM_HEADS = 32
HEAD_DIM = 128
CHUNK = 16_384
MAX_MODEL_LEN = 1 << 20
# vLLM NCCL all-gather, 8 ranks over 2 GB300 nodes (MNNVL): t = a + MB / BW
# with MB received per rank; fit to measured 1-31 MB transfers.
AG_LATENCY_US = 26.9
AG_GBPS = 545.0  # measured 8-rank all-gather of 361.5 MB


# AgentX-like prefill chunks: (new tokens, prior context) per request.
def agentx_context(seed: int) -> int:
    rng = np.random.default_rng(seed)
    mu = np.log(72_000)
    ctx = float(np.exp(rng.normal(mu, (np.log(165_000) - mu) / 1.2816)))
    if rng.random() < 0.15:
        ctx = float(rng.integers(256, 8_192))
    return int(np.clip(ctx, 256, 1 << 20))


def scenarios() -> list[tuple[str, list[tuple[int, int]]]]:
    rng = np.random.default_rng(7)
    mixed, budget = [], CHUNK
    while budget > 0:
        new = int(min(budget, rng.integers(512, 8_192)))
        mixed.append((new, agentx_context(int(rng.integers(1 << 30)))))
        budget -= new
    return [
        ("fresh-16k", [(CHUNK, 0)]),
        ("ctx32k-16k", [(CHUNK, 32_768)]),
        ("ctx128k-16k", [(CHUNK, 131_072)]),
        ("agentx-mixed-16k", mixed),
    ]


def plan(requests, max_logits_mb: int, world: int):
    """The indexer's sub-chunks of one chunk under DCP ``world`` (interleave
    1): (requests, first row within them, rows, first token of the chunk,
    logits width) each. The KV workspace bound does not bind here."""
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.dcp_world_size = world
    builder.compressed_cp_interleave_size = 1
    seq_lens = builder._prefill_split_seq_lens(
        torch.tensor([new + prior for new, prior in requests])
    )
    query_lens = torch.tensor([new for new, _ in requests])
    specs = builder._split_indexer_prefill_chunks(
        seq_lens, query_lens, 1 << 40, max_logits_mb << 20
    )
    return [
        (
            requests[reqs],
            rows.start,
            rows.stop - rows.start,
            int(query_lens[: reqs.start].sum()) + rows.start,
            int(seq_lens[reqs].max()),
        )
        for reqs, rows in specs
    ]


def local_len(tokens: int, world: int, rank: int) -> int:
    return (tokens - rank + world - 1) // world if tokens > rank else 0


def row_bounds(requests, world: int, rank: int):
    """DeepGEMM (ks, ke) per row over this rank's gathered keys."""
    ks, ke, offset = [], [], 0
    for new, prior in requests:
        for i in range(new):
            ks.append(offset)
            ke.append(offset + local_len(prior + i + 1, world, rank))
        offset += local_len(prior + new, world, rank)
    as_t = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")  # noqa: E731
    return as_t(ks), as_t(ke), offset


class Inputs:
    """One chunk's indexer Q and weights (identical on every rank)."""

    def __init__(self, seed: int = 0) -> None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn((CHUNK, NUM_HEADS, HEAD_DIM), generator=gen, device="cuda")
        self.q = q.to(torch.float8_e4m3fn)
        self.weights = (
            torch.randn((CHUNK, NUM_HEADS), generator=gen, device="cuda") * 0.1
        )

    def logits(self, sub, world: int, rank: int):
        """This rank's bf16 logits of one sub-chunk and their row ends."""
        requests, first, rows, start, width = sub
        ks, ke, num_keys = row_bounds(requests, world, rank)
        ks, ke = ks[first : first + rows], ke[first : first + rows]
        q = self.q[start : start + rows]
        gen = torch.Generator(device="cuda").manual_seed(1000 * rank + first)
        k = torch.randn((max(num_keys, 1), HEAD_DIM), generator=gen, device="cuda")
        k = k.to(torch.float8_e4m3fn)
        k_scale = torch.full(
            (k.shape[0],), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
        )
        q_scale, w = mqa_logits_mx_inputs(q, None, self.weights[start : start + rows])
        row_ends = ke - ks
        if num_keys == 0:
            return torch.empty((rows, 0), device="cuda"), row_ends
        logits = fp8_fp4_mqa_logits((q, q_scale), (k, k_scale), w, ks, ke, width)
        return logits, row_ends


def timed(fn, iters: int, warmup: int = 3, barrier: bool = False) -> float:
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    times = []
    for i in range(warmup + iters):
        flush.zero_()
        if barrier:
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def topk_current(logits, row_ends, out):
    logits32 = sparse_indexer.as_fp32_logits(logits, row_ends)
    ops.top_k_per_row_prefill(
        logits32,
        torch.zeros_like(row_ends),
        row_ends,
        out,
        logits32.shape[0],
        logits32.stride(0),
        logits32.stride(1),
        TOPK,
    )
    return logits32


def topk_bf16(logits, row_ends, out):
    deep_select_topk(logits, TOPK, end=row_ends, output_idx=out)
    return logits


def local_topk(path, logits, row_ends, out):
    """This rank's local top-k; returns the logits the merge reads."""
    if logits.shape[1] == 0:
        out.fill_(-1)
        return logits
    return (topk_current if path == "current" else topk_bf16)(logits, row_ends, out)


def ag_estimate_us(mb: float, calls: int) -> float:
    return calls * AG_LATENCY_US + mb / AG_GBPS * 1e3


def new_idx(rows: int) -> torch.Tensor:
    return torch.empty((rows, TOPK), dtype=torch.int32, device="cuda")


def tp_subchunk(sub, inputs, iters: int) -> np.ndarray:
    """[current us, bf16 us, rows, rows with equal top-k values, rows with
    equal column sets]."""
    logits, ends = inputs.logits(sub, 1, 0)
    outs = new_idx(sub[2]), new_idx(sub[2])
    cur = timed(lambda: topk_current(logits, ends, outs[0]), iters)
    new = timed(lambda: topk_bf16(logits, ends, outs[1]), iters)
    # Both kernels select the exact top-k; only columns tied at the cutoff
    # may differ.
    sets = [o.sort(dim=1).values for o in outs]
    values = [
        torch.where(o >= 0, logits.gather(1, o.clamp(min=0).long()).float(), 0)
        .sort(dim=1)
        .values
        for o in outs
    ]
    same_values = int((values[0] == values[1]).all(dim=1).sum())
    same_set = int((sets[0] == sets[1]).all(dim=1).sum())
    return np.array([cur, new, sub[2], same_values, same_set])


def bench_tp(name, requests, inputs, args):
    subs = plan(requests, args.max_logits_mb, 1)
    cur, new, rows, same_values, same_set = sum(
        tp_subchunk(sub, inputs, args.iters) for sub in subs
    )
    print(
        f"| {name} | {len(subs)} | {cur:.0f} | {new:.0f} | {cur / new:.2f}x "
        f"| {same_values:.0f}/{rows:.0f} | {same_set:.0f}/{rows:.0f} |",
        flush=True,
    )


DCP_FIELDS = (
    "cur_topk",
    "pack",
    "reorg",
    "select",
    "dsel",
    "pack16",
    "rows",
    "same_set",
)


def dcp_subchunk(sub, inputs, world: int, iters: int) -> np.ndarray:
    """Rank 0's local times of one sub-chunk (see ``DCP_FIELDS``)."""
    m = sub[2]
    rank_logits, rank_ends = zip(*(inputs.logits(sub, world, r) for r in range(world)))
    topks = {"current": [], "bf16": []}
    pairs = {"current": [], "bf16": []}
    for r in range(world):
        for path in ("current", "bf16"):
            topk = new_idx(m)
            packed = torch.empty((m, TOPK, 2), device="cuda")
            scored = local_topk(path, rank_logits[r], rank_ends[r], topk)
            if scored.shape[1] == 0:
                packed[..., 0], packed[..., 1] = float("-inf"), -1
            else:
                pack_dcp_topk_candidates_cutedsl(
                    scored, topk, packed, r, world, 1, None
                )
            topks[path].append(topk)
            pairs[path].append(packed)
    gathered = {path: torch.cat(pairs[path], dim=1) for path in pairs}
    del pairs
    merged = {
        path: stable_topk_from_gathered_candidates_cutedsl(gathered[path], TOPK)
        .sort(dim=1)
        .values
        for path in gathered
    }
    same_set = int((merged["current"] == merged["bf16"]).all(dim=1).sum())

    logits, ends = rank_logits[0], rank_ends[0]
    topk0, packed = new_idx(m), torch.empty((m, TOPK, 2), device="cuda")
    flat = torch.empty((world, m, TOPK, 2), device="cuda")
    logits32 = sparse_indexer.as_fp32_logits(logits, ends)
    fns = {
        "cur_topk": lambda: topk_current(logits, ends, topk0),
        "pack": lambda: pack_dcp_topk_candidates_cutedsl(
            logits32, topks["current"][0], packed, 0, world, 1, None
        ),
        # GroupCoordinator.all_gather(dim=1) reorders the gathered ranks.
        "reorg": lambda: flat.movedim(0, 1).reshape(m, world * TOPK, 2),
        "select": lambda: stable_topk_from_gathered_candidates_cutedsl(
            gathered["current"], TOPK, out=topk0
        ),
        "dsel": lambda: topk_bf16(logits, ends, topk0),
        "pack16": lambda: pack_dcp_topk_candidates_cutedsl(
            logits, topks["bf16"][0], packed, 0, world, 1, None
        ),
    }
    times = [timed(fns[f], iters) for f in DCP_FIELDS[:6]]
    return np.array([*times, m, same_set])


def bench_dcp(name, requests, inputs, args):
    world = args.dcp
    subs = plan(requests, args.max_logits_mb, world)
    t = dict(
        zip(
            DCP_FIELDS,
            sum(dcp_subchunk(s, inputs, world, args.iters) for s in subs),
        )
    )
    torch.cuda.empty_cache()
    n, rows = len(subs), int(t["rows"])
    mb_pairs = (world - 1) * rows * TOPK * 8 / 1e6
    comm = ag_estimate_us(mb_pairs, n) + t["reorg"]
    local = {
        "current": t["cur_topk"] + t["pack"] + t["select"],
        "bf16": t["dsel"] + t["pack16"] + t["select"],
    }
    print(
        f"| {name} | {n} | {local['current']:.0f} | {local['bf16']:.0f} "
        f"| {local['current'] / local['bf16']:.2f}x | {mb_pairs:.0f} "
        f"| {comm:.0f} | {local['current'] + comm:.0f} "
        f"| {local['bf16'] + comm:.0f} "
        f"| {(local['current'] + comm) / (local['bf16'] + comm):.2f}x "
        f"| {t['same_set']:.0f}/{rows} |",
        flush=True,
    )


class NcclGroup:
    """``GroupCoordinator.all_gather`` over the default process group."""

    def __init__(self) -> None:
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        dim %= x.dim()
        out = torch.empty((self.world_size, *x.shape), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out.view(-1, *x.shape[1:]), x.contiguous())
        if dim == 0:
            return out.view(-1, *x.shape[1:])
        return out.movedim(0, dim).reshape(*x.shape[:dim], -1, *x.shape[dim + 1 :])


def dist_subchunk(sub, inputs, group, iters: int) -> np.ndarray:
    """This rank's [current, bf16] us of one sub-chunk, its rows and the rows
    where both paths merge the same global top-k set."""
    world, rank, m = group.world_size, group.rank_in_group, sub[2]
    logits, ends = inputs.logits(sub, world, rank)
    topk = new_idx(m)

    def generic(path):
        scored = local_topk(path, logits, ends, topk)
        if scored.shape[1] == 0:
            # No local keys: the op hands the merge empty FP32 logits.
            scored = torch.empty((m, 0), device="cuda")
        sparse_indexer._merge_dcp_topk_global(scored, topk, TOPK, rank, world, 1)
        return topk.sort(dim=1).values

    times = [
        timed(fn, iters, barrier=True)
        for fn in (lambda: generic("current"), lambda: generic("bf16"))
    ]
    same = int((generic("current") == generic("bf16")).all(dim=1).sum())
    return np.array([*times, m, same])


def bench_dist(name, requests, inputs, args):
    group = NcclGroup()
    sparse_indexer.get_dcp_group = lambda: group
    subs = plan(requests, args.max_logits_mb, group.world_size)
    stats = sum(dist_subchunk(sub, inputs, group, args.iters) for sub in subs)
    torch.cuda.empty_cache()
    times = torch.tensor(stats[:2], device="cuda")
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    counts = torch.tensor(stats[2:], device="cuda")
    dist.all_reduce(counts)
    if group.rank_in_group == 0:
        cur, b16 = times.tolist()
        rows, same = counts.tolist()
        print(
            f"| {name} | {len(subs)} | {cur:.0f} | {b16:.0f} | {cur / b16:.2f}x "
            f"| {same:.0f}/{rows:.0f} |",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--sections", default="tp,dcp")
    parser.add_argument("--only", default="", help="comma-separated scenarios")
    parser.add_argument(
        "--max-logits-mb",
        type=int,
        default=512,
        help="indexer logits budget (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)",
    )
    args = parser.parse_args()
    sections = args.sections.split(",")
    if "dist" in sections:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        assert dist.get_world_size() == args.dcp
    only = set(filter(None, args.only.split(",")))
    cases = [(n, r) for n, r in scenarios() if not only or n in only]
    inputs = Inputs()
    lead = not dist.is_initialized() or dist.get_rank() == 0

    if "tp" in sections:
        print("\n## Prefill top-k, all keys on one rank (us per 16K chunk)\n")
        print(
            "| scenario | sub-chunks | current | bf16 DeepSelect | speedup "
            "| rows with equal top-k values | rows with equal column sets |"
        )
        print("|" + "---|" * 7)
        for name, requests in cases:
            bench_tp(name, requests, inputs, args)
            torch.cuda.empty_cache()

    if "dcp" in sections:
        print(f"\n## DCP{args.dcp} rank 0: local top-k + merge (us per 16K chunk)\n")
        print(
            "| scenario | sub-chunks | current local | bf16 local | local speedup "
            "| comm MB/rank | comm us (e) | current | bf16 | speedup "
            "| bf16 == current rows |"
        )
        print("|" + "---|" * 11)
        for name, requests in cases:
            bench_dcp(name, requests, inputs, args)

    if "dist" in sections:
        if lead:
            print(
                f"\n## DCP{args.dcp}, {args.dcp} ranks: local top-k + merge with "
                "NCCL (us per 16K chunk, max over ranks)\n"
            )
            print(
                "| scenario | sub-chunks | current | bf16 | speedup "
                "| bf16 == current rows (all ranks) |"
            )
            print("|" + "---|" * 6)
        for name, requests in cases:
            bench_dist(name, requests, inputs, args)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
