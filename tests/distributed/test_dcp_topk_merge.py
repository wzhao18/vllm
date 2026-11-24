# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact one-round DCP decode top-k merge over a real DCP group.

One process per GPU (DCP = TP = world size), the production construction
gate (``init_dcp_topk_merge``: rank agreement, symmetric rescue workspace)
and the real all-gather. Every rank builds the same global bf16 logits and
keeps its interleave-1 shard (columns ``rank::W``); the reference is the
top-K of the union of all ranks' logits. Each world size spawns one group
and runs every case inside it.
"""

import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.network_utils import get_open_port

TOPK = 2048
# DeepSelect reads logits rows at a 1024-byte aligned stride.
_ROW_ALIGN = 512
_BLOCK = 64


def _has_deep_select() -> bool:
    try:
        import vllm._deepselect_C  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not has_cutedsl()
    or not _has_deep_select(),
    reason="The DCP top-k merge needs SM100, CuTe DSL and DeepSelect",
)


# --- Reference helpers -------------------------------------------------------


def _make_case(
    lengths: list[int], seed: int, levels: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global bf16 logits ``[rows, max_len]`` (identical on every rank) and
    per-row global ends. ``levels`` > 0 quantizes the scores (tie-heavy)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((len(lengths), max(lengths)), generator=gen, device="cuda")
    if levels:
        logits = torch.floor(logits * levels / 8) / levels
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return logits.to(torch.bfloat16), ends


def _boost(logits: torch.Tensor, ranks, rows, world: int) -> torch.Tensor:
    """Raise the given ranks' tokens on the given rows above all others."""
    out = logits.float()
    for rank in ranks:
        out[rows, rank::world] += 8.0
    return out.to(torch.bfloat16)


def _shard(
    logits: torch.Tensor, ends: torch.Tensor, world: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's local bf16 logits (DeepSelect-aligned rows) and row ends."""
    num_rows, width = logits.shape
    local_width = -(-width // world)
    padded = -(-local_width // _ROW_ALIGN) * _ROW_ALIGN
    local = torch.full(
        (num_rows, padded), float("-inf"), dtype=torch.bfloat16, device="cuda"
    )
    cols = logits[:, rank::world]
    local[:, : cols.shape[1]] = cols
    local_ends = ((ends - rank + world - 1) // world).clamp_min(0).int()
    return local, local_ends


def _as_lists(out: torch.Tensor) -> list[list[int]]:
    """Sorted selections per row, checking the selected-then--1 layout."""
    lists = []
    for row in out.tolist():
        valid = [i for i in row if i >= 0]
        assert row == valid + [-1] * (len(row) - len(valid)), "bad -1 tail"
        lists.append(sorted(valid))
    return lists


def _assert_exact(ids: torch.Tensor, logits: torch.Tensor, ends: torch.Tensor):
    """Repeat-free ids whose score multiset is the exact global top-K (ties
    at the cutoff resolved either way)."""
    for row, got in enumerate(_as_lists(ids)):
        assert len(set(got)) == len(got), f"row {row}: repeated ids"
        v = logits[row, : int(ends[row])].float()
        want = v.topk(min(TOPK, v.numel())).values
        sel = torch.tensor(got, dtype=torch.long, device="cuda")
        scores = logits[row].float()[sel].sort(descending=True).values
        assert torch.equal(scores, want), f"row {row}: not the exact top-K"


def _assert_same_on_every_rank(group, ids: torch.Tensor, what: str) -> None:
    every = group.all_gather(ids.sort(dim=1).values.unsqueeze(0), dim=0)
    for r in range(group.world_size):
        assert torch.equal(every[r], every[0]), f"{what}: rank {r} differs"


def _block_table(num_rows: int, max_local: int) -> torch.Tensor:
    blocks = -(-max(max_local, 1) // _BLOCK) + 1
    table = torch.arange(num_rows * blocks, dtype=torch.int32, device="cuda")
    return table.flip(0).view(num_rows, blocks).contiguous()


def _assert_rows(group, ids, rows_out, valid_out, req_id, block_table) -> None:
    """Rows and counts are the stock filter/convert of the merged ids, and the
    ranks' owned rows partition the merged set."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    ref_rows, ref_valid = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        ids,
        dcp_size=group.world_size,
        dcp_rank=group.rank_in_group,
        cp_kv_cache_interleave_size=1,
        BLOCK_SIZE=_BLOCK,
        BLOCK_STRIDE_ROWS=_BLOCK,
        NUM_TOPK_TOKENS=ids.shape[1],
        return_valid_counts=True,
    )
    assert torch.equal(valid_out, ref_valid)
    assert _as_lists(rows_out) == _as_lists(ref_rows)
    total = group.all_reduce(valid_out.clone())
    assert torch.equal(total, (ids >= 0).sum(1).int())


def _sat_masks(ws, num_rows: int) -> list[int]:
    return ((ws.sat_word[:num_rows] >> 16) & 0xFFFF).tolist()


# --- The worker --------------------------------------------------------------


def _check_cases(group, ws, n: int) -> None:
    """Normal, saturated (1 rank and the (K - 1) // N bound), mixed, short and
    tie-heavy rows through both entry points (ids; rows from the logits and
    from packed input)."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
        dcp_topk_pack,
    )

    world, rank = group.world_size, group.rank_in_group
    max_sat = (TOPK - 1) // n
    long = [100_000, 40_000, 9_000, 70_123, 150_000, 6_000]
    # Normal rows also cover rows below K, empty rows and local indices
    # straddling 2**16 (the bit plane).
    straddle = world * 66_000
    normal = [100_000, 40_000, 9_000, 70_123, 2_049, 1, 0, straddle]
    cases = [
        # (name, lengths, levels, saturated ranks, saturated rows)
        ("normal", normal, 0, [], None),
        ("sat1", long, 0, [world - 1], slice(None)),
        ("sat-max", long, 0, list(range(max_sat)), slice(None)),
        ("mixed", long, 0, [world // 2], slice(0, None, 2)),
        # Local lengths in (N, K): a saturated rank has fewer than K tokens.
        ("short", [300, 1_000, 3_000, 9_000, 20_000, 120], 0, [0], slice(None)),
        ("ties", [50_000, 30_000, 120_000], 16, [], None),
        ("ties-sat", [50_000, 30_000, 120_000], 16, [1], slice(None)),
    ]
    for seed, (name, lengths, levels, sat, sat_rows) in enumerate(cases):
        num_rows = len(lengths)
        logits, ends = _make_case(lengths, seed=seed, levels=levels)
        if name == "normal":
            boundary = world * 65_536
            logits[-1, boundary - world * 300 : boundary + world * 300] += 6
        if sat:
            logits = _boost(logits, sat, sat_rows, world)
        local, local_end = _shard(logits, ends, world, rank)
        if name == "mixed":
            # The per-row rescue sequence wraps like any int32.
            ws.seq2.fill_(2**31 - 1)

        ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
        dcp_topk_merge(local, local_end, ids, n)
        masks = _sat_masks(ws, num_rows)
        _assert_exact(ids, logits, ends)
        _assert_same_on_every_rank(group, ids, name)
        sat_mask = sum(1 << r for r in sat)
        assert all(bin(m).count("1") <= max_sat for m in masks), (name, masks)
        if levels == 0 and not sat:
            assert masks == [0] * num_rows, (name, masks)
        elif levels == 0:
            local_len = [-(-x // world) for x in lengths]
            for row, m in enumerate(masks):
                boosted = sat_rows == slice(None) or row % 2 == 0
                if boosted and local_len[row] > n and lengths[row] > TOPK:
                    assert m & sat_mask, (name, row, masks)
                if not boosted:
                    assert m == 0, (name, row, masks)
        if name == "mixed":
            want = [-(2**31) if m else 2**31 - 1 for m in masks]
            assert ws.seq2[:num_rows].tolist() == want, name

        req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        block_table = _block_table(num_rows, int(local_end.max()))
        packed = dcp_topk_pack(local, local_end, n)
        for mode in ("logits", "packed"):
            rows_out = torch.full_like(ids, -7)
            valid_out = torch.full((num_rows,), -7, dtype=torch.int32, device="cuda")
            ids2 = torch.full_like(ids, -7)
            dcp_topk_merge_rows(
                logits=local,
                row_ends=local_end,
                packed=packed if mode == "packed" else None,
                local_topk=n,
                topk=TOPK,
                req_id=req_id,
                block_table=block_table,
                block_size=_BLOCK,
                stride_rows=_BLOCK,
                dcp_rank=rank,
                rows_out=rows_out,
                valid_out=valid_out,
                ids_out=ids2,
            )
            _assert_exact(ids2, logits, ends)
            _assert_same_on_every_rank(group, ids2, f"{name}/{mode}")
            if levels == 0:
                assert _as_lists(ids2) == _as_lists(ids), (name, mode)
            _assert_rows(group, ids2, rows_out, valid_out, req_id, block_table)


def _check_nonfinite(group, n: int) -> None:
    """NaN / +-inf logits: ids stay in range, never select a NaN, agree on
    every rank, and rows without a NaN keep the exact top-K."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge

    world, rank = group.world_size, group.rank_in_group
    lengths = [30_000] * 6 + [1_500, 30_000]
    logits, ends = _make_case(lengths, seed=41)
    logits[0, ::7] = float("nan")
    logits[1] = float("nan")
    logits[2, ::5] = float("inf")
    logits[3] = float("-inf")
    logits[4] = float("inf")
    logits[5, (world - 1) :: world] = float("nan")  # NaN on one rank
    logits[6, ::3] = float("nan")
    logits[7, 0::world] = float("inf")  # +inf on one rank: it saturates
    local, local_end = _shard(logits, ends, world, rank)
    ids = torch.full((len(lengths), TOPK), -7, dtype=torch.int32, device="cuda")
    dcp_topk_merge(local, local_end, ids, n)
    _assert_same_on_every_rank(group, ids, "nonfinite")
    for row, got in enumerate(_as_lists(ids)):
        end = int(ends[row])
        assert all(0 <= g < end for g in got), row
        row_logits = logits[row, :end]
        if got:
            sel = torch.tensor(got, device="cuda")
            assert not logits[row, sel].isnan().any(), row
        if not row_logits.isnan().any():
            _assert_exact(ids[row : row + 1], logits[row : row + 1], ends[row:])
    assert _as_lists(ids)[1] == []


def _check_cuda_graph(group, ws, n: int) -> None:
    """One graph of both entry points, replayed over inputs alternating
    clean, one saturated row and every row saturated (1 rank, and the bound):
    each replay is exact, and the rescue sequence advances only on saturated
    rows."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [100_000, 40_000, 9_000, 70_123, 150_000, 6_000, 3_000, 220_000]
    num_rows = len(lengths)
    base, ends = _make_case(lengths, seed=29)
    variants = [
        base,
        _boost(base, [0], slice(7, 8), world),
        _boost(base, [world - 1], slice(None), world),
        _boost(base, list(range((TOPK - 1) // n)), slice(None), world),
    ]
    shards = [_shard(v, ends, world, rank) for v in variants]
    static = shards[0][0].clone()
    local_end = shards[0][1]
    ids = torch.empty((num_rows, TOPK), dtype=torch.int32, device="cuda")
    ids2, rows_out = torch.empty_like(ids), torch.empty_like(ids)
    valid_out = torch.empty(num_rows, dtype=torch.int32, device="cuda")
    req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    block_table = _block_table(num_rows, int(local_end.max()))

    def step() -> None:
        dcp_topk_merge(static, local_end, ids, n)
        dcp_topk_merge_rows(
            logits=static,
            row_ends=local_end,
            local_topk=n,
            topk=TOPK,
            req_id=req_id,
            block_table=block_table,
            block_size=_BLOCK,
            stride_rows=_BLOCK,
            dcp_rank=rank,
            rows_out=rows_out,
            valid_out=valid_out,
            ids_out=ids2,
        )

    step()  # compile outside capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        step()
    torch.accelerator.synchronize()
    seq = ws.seq2[:num_rows].clone()
    for i in range(2 * len(variants)):
        v = i % len(variants)
        static.copy_(shards[v][0])
        graph.replay()
        torch.accelerator.synchronize()
        _assert_exact(ids, variants[v], ends)
        _assert_exact(ids2, variants[v], ends)
        _assert_same_on_every_rank(group, ids, f"graph replay {i}")
        _assert_rows(group, ids2, rows_out, valid_out, req_id, block_table)
        triggered = torch.tensor(
            [int(m != 0) for m in _sat_masks(ws, num_rows)],
            dtype=torch.int32,
            device="cuda",
        )
        assert bool(triggered.any()) == (v != 0), (i, triggered.tolist())
        # Two merges per replay: each saturated row advances twice.
        assert torch.equal(ws.seq2[:num_rows], seq + 2 * triggered), i
        seq = ws.seq2[:num_rows].clone()


def _check_late_peer(group, ws, n: int) -> None:
    """The saturated rank starts its select ~100 ms after the all-gather,
    while the other ranks are already in their rescue waiting on its flag:
    they must wait for its push (and take that long), and the result is the
    exact top-K."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack,
        dcp_topk_select,
    )

    world, rank = group.world_size, group.rank_in_group
    late = world - 1
    lengths = [150_000, 100_000, 70_123]
    logits, ends = _make_case(lengths, seed=31)
    logits = _boost(logits, [late], slice(None), world)
    local, local_end = _shard(logits, ends, world, rank)
    ids = torch.full((len(lengths), TOPK), -7, dtype=torch.int32, device="cuda")
    packed = dcp_topk_pack(local, local_end, n)
    gathered = group.all_gather(packed, dim=0).view(world, *packed.shape)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    if rank == late:
        torch.cuda._sleep(200_000_000)
    dcp_topk_select(gathered, n, ids, exact=ws.args(local, local_end))
    stop.record()
    torch.accelerator.synchronize()
    assert _sat_masks(ws, len(lengths)) == [1 << late] * len(lengths)
    _assert_exact(ids, logits, ends)
    _assert_same_on_every_rank(group, ids, "late peer")
    if rank != late:
        assert start.elapsed_time(stop) > 20, "the rescue did not wait"


def _check_deterministic(group, ws, n: int, repeats: int = 8) -> None:
    """Repeated merges of the same logits are bit-identical, order included,
    through both entry points (ids; rows with and without ids), unsaturated
    and saturated (the rescue round): the sparse attention downstream sums
    the selected tokens in this order."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [150_000, 100_000, 70_123, 40_000, 220_000, 9_000, 120_000, 60_000]
    num_rows = len(lengths)
    base, ends = _make_case(lengths, seed=53)
    for sat in (False, True):
        logits = _boost(base, [world - 1], slice(None), world) if sat else base
        local, local_end = _shard(logits, ends, world, rank)
        req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        block_table = _block_table(num_rows, int(local_end.max()))
        runs: list[list[torch.Tensor]] = []
        for _ in range(repeats):
            ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
            dcp_topk_merge(local, local_end, ids, n)
            masks = _sat_masks(ws, num_rows)
            assert all(masks) if sat else not any(masks), (sat, masks)
            outs = [ids]
            for with_ids in (True, False):
                rows_out = torch.full_like(ids, -7)
                valid_out = torch.full(
                    (num_rows,), -7, dtype=torch.int32, device="cuda"
                )
                ids2 = torch.full_like(ids, -7) if with_ids else None
                dcp_topk_merge_rows(
                    logits=local,
                    row_ends=local_end,
                    local_topk=n,
                    topk=TOPK,
                    req_id=req_id,
                    block_table=block_table,
                    block_size=_BLOCK,
                    stride_rows=_BLOCK,
                    dcp_rank=rank,
                    rows_out=rows_out,
                    valid_out=valid_out,
                    ids_out=ids2,
                )
                outs += [rows_out, valid_out] + ([ids2] if with_ids else [])
            runs.append(outs)
        torch.accelerator.synchronize()
        _assert_exact(runs[0][0], logits, ends)
        _assert_rows(group, runs[0][3], runs[0][1], runs[0][2], req_id, block_table)
        what = ["ids", "rows", "valid", "rows ids", "rows (no ids)", "valid (no ids)"]
        for i, run in enumerate(runs[1:], 1):
            for name, got, want in zip(what, run, runs[0]):
                assert torch.equal(got, want), f"sat={sat} run {i}: {name} changed"
        # Without the ids the rows come out the same.
        assert torch.equal(runs[0][4], runs[0][1])


def _check_exchange_ab(group, ws, n: int) -> None:
    """Round 1 through the mailbox against the all-gather on the same inputs:
    ids, rows, counts and saturation words are bit-identical (order
    included) on clean, tie-heavy, saturated (1 rank on every third row, and
    the bound on every row) and non-finite rows, for row counts that change
    between merges (each row's mailbox parity advances on its own), through
    both entry points (rows from the logits and from packed input)."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
        dcp_topk_pack,
        set_dcp_topk_merge_exchange,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [150_000, 100_000, 70_123, 40_000, 220_000, 9_000, 2_049, 1, 0]
    lengths += [world * 66_000]
    base, ends = _make_case(lengths, seed=61)
    tied, _ = _make_case(lengths, seed=67, levels=16)
    # Words that are NaN or denormal as FP32 (+inf scores, the bit plane's
    # marked half-words) must cross the multicast store unchanged.
    nonfinite = base.clone()
    nonfinite[0, ::7] = float("inf")
    nonfinite[1] = float("-inf")
    nonfinite[2] = float("inf")
    nonfinite[3, ::5] = float("nan")
    nonfinite[4, 0::world] = float("inf")
    variants = [
        base,
        tied,
        _boost(base, [world - 1], slice(0, None, 3), world),
        _boost(base, list(range((TOPK - 1) // n)), slice(None), world),
        nonfinite,
    ]

    def merge(exchange: str, local, local_end, packed) -> list[torch.Tensor]:
        num_rows = local.shape[0]
        req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        block_table = _block_table(num_rows, int(local_end.max()))
        set_dcp_topk_merge_exchange(exchange)
        try:
            ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
            dcp_topk_merge(local, local_end, ids, n)
            outs = [ids, ws.sat_word[:num_rows].clone()]
            for mode in ("logits", "packed"):
                rows_out = torch.full_like(ids, -7)
                valid_out = torch.full(
                    (num_rows,), -7, dtype=torch.int32, device="cuda"
                )
                ids2 = torch.full_like(ids, -7)
                dcp_topk_merge_rows(
                    logits=local,
                    row_ends=local_end,
                    packed=packed if mode == "packed" else None,
                    local_topk=n,
                    topk=TOPK,
                    req_id=req_id,
                    block_table=block_table,
                    block_size=_BLOCK,
                    stride_rows=_BLOCK,
                    dcp_rank=rank,
                    rows_out=rows_out,
                    valid_out=valid_out,
                    ids_out=ids2,
                )
                outs += [rows_out, valid_out, ids2, ws.sat_word[:num_rows].clone()]
        finally:
            set_dcp_topk_merge_exchange("auto")
        torch.accelerator.synchronize()
        return outs

    what = ["ids", "sat", "rows", "valid", "rows ids", "rows sat"]
    what += ["packed rows", "packed valid", "packed ids", "packed sat"]
    for v, logits in enumerate(variants):
        local, local_end = _shard(logits, ends, world, rank)
        for num_rows in (len(lengths), 5, 1, 8, 3):
            args = local[:num_rows], local_end[:num_rows]
            packed = dcp_topk_pack(*args, n)
            ref = merge("all_gather", *args, packed)
            got = merge("auto", *args, packed)
            for name, a, b in zip(what, ref, got):
                # DeepSelect's ids on a NaN row are arbitrary: there only the
                # merges of the same packed rows compare.
                if logits is not nonfinite or name.startswith("packed"):
                    assert torch.equal(a, b), f"variant {v} rows {num_rows}: {name}"
            if logits is not nonfinite:
                _assert_exact(got[0], logits[:num_rows], ends[:num_rows])
            _assert_same_on_every_rank(group, got[0], f"mailbox {v}/{num_rows}")
        if v in (2, 3):
            assert bool((ws.sat_word[: len(lengths)] >> 16).any()), v


def _check_late_publisher(group, n: int) -> None:
    """One rank publishes its round-1 rows ~100 ms late: the other ranks wait
    in the select's poll (and take that long), and the result is exact."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge

    world, rank = group.world_size, group.rank_in_group
    late = world - 1
    lengths = [150_000, 100_000, 70_123]
    logits, ends = _make_case(lengths, seed=71)
    local, local_end = _shard(logits, ends, world, rank)
    ids = torch.full((len(lengths), TOPK), -7, dtype=torch.int32, device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    if rank == late:
        torch.cuda._sleep(200_000_000)
    dcp_topk_merge(local, local_end, ids, n)
    stop.record()
    torch.accelerator.synchronize()
    _assert_exact(ids, logits, ends)
    _assert_same_on_every_rank(group, ids, "late publisher")
    if rank != late:
        assert start.elapsed_time(stop) > 20, "the select did not wait"


def _check_graph_stress(group, ws, n: int, replays: int) -> None:
    """The mailbox under CUDA-graph replay: graphs of 24 merges at 64 and 8
    rows and of 3 merges at 5 rows, replayed in a seeded order shared by all
    ranks over clean and saturated (1 rank on every third row, the bound on
    every row) inputs. Before each replay the ids, saturation words and the
    rescue's round-1 rows are poisoned and a random rank starts up to ~2 ms
    late; every 10th replay is followed by an eager merge of 7 rows. Every
    merge is bit-identical to the all-gather's (cached, and rerun eagerly
    every 25th replay); afterwards every rank counted the same merges per
    row and every mailbox slot is empty again."""
    import random

    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        set_dcp_topk_merge_exchange,
    )

    world, rank = group.world_size, group.rank_in_group
    mailbox = ws.mailbox
    layers, max_rows = 24, 64
    gen = torch.Generator(device="cuda").manual_seed(83)
    glen = torch.randint(9_000, 178_001, (max_rows,), generator=gen, device="cuda")
    # Local ids past 2**16 (the bit plane), a short, an empty and a 1-token row.
    glen[0], glen[2], glen[3], glen[4] = world * 70_000, 2_049, 0, 1
    local_end = ((glen - rank + world - 1) // world).clamp_min(0).int()
    width = -(-(world * 70_000) // world // _ROW_ALIGN) * _ROW_ALIGN
    gen.manual_seed(89 + rank)
    clean = [
        torch.randn((layers, max_rows, width), generator=gen, device="cuda").to(
            torch.bfloat16
        )
        for _ in range(2)
    ]
    patterns = {
        None: ([], None),
        "third": ([world - 1], slice(0, None, 3)),
        "bound": (list(range((TOPK - 1) // n)), slice(None)),
    }
    variants = [(s, p) for s in range(2) for p in patterns]
    static = torch.empty_like(clean[0])

    def fill(v: int) -> None:
        s, pattern = variants[v]
        static.copy_(clean[s])
        ranks, rows = patterns[pattern]
        if rank in ranks:
            static[:, rows] = (static[:, rows].float() + 8.0).to(torch.bfloat16)

    ids = torch.empty((layers, max_rows, TOPK), dtype=torch.int32, device="cuda")
    sats = torch.empty((layers, max_rows), dtype=torch.int32, device="cuda")
    ids7 = torch.empty((7, TOPK), dtype=torch.int32, device="cuda")
    shapes = [(layers, 64), (layers, 8), (3, 5)]

    def run(g: int) -> None:
        count, t = shapes[g]
        for j in range(count):
            dcp_topk_merge(static[j, :t], local_end[:t], ids[j, :t], n)
            sats[j, :t].copy_(ws.sat_word[:t])

    def merge7() -> None:
        dcp_topk_merge(static[0, :7], local_end[:7], ids7, n)

    refs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    refs7: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    set_dcp_topk_merge_exchange("all_gather")
    try:
        for v in range(len(variants)):
            fill(v)
            for g, (count, t) in enumerate(shapes):
                run(g)
                refs[g, v] = ids[:count, :t].clone(), sats[:count, :t].clone()
            merge7()
            refs7[v] = ids7.clone(), ws.sat_word[:7].clone()
    finally:
        set_dcp_topk_merge_exchange("auto")
    torch.accelerator.synchronize()
    assert any(bool((refs[0, v][1] >> 16).any()) for v in range(len(variants)))

    graphs = []
    for g in range(len(shapes)):
        fill(0)
        run(g)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with group.graph_capture() as ctx, torch.cuda.graph(graph, stream=ctx.stream):
            run(g)
        graphs.append(graph)
    torch.accelerator.synchronize()

    # bad[i]: mismatching ids, saturation words and eager 7-row outputs of
    # replay i (counted on the device, so ranks may drift between replays).
    bad = torch.zeros((replays, 3), dtype=torch.int32, device="cuda")
    schedule = random.Random(97)
    for i in range(replays):
        g = schedule.randrange(len(shapes))
        pattern = schedule.randrange(1, 3) if schedule.random() < 1 / 3 else 0
        v = len(patterns) * schedule.randrange(2) + pattern
        late, cycles = schedule.randrange(world), schedule.randrange(4_000_000)
        count, t = shapes[g]
        fill(v)
        ids.fill_(-7)
        sats.fill_(-7)
        ws.sat_word.fill_(0x5A5A5A5A)
        mailbox.gathered.fill_(0x5A5A5A5A)
        if rank == late:
            torch.cuda._sleep(cycles)
        graphs[g].replay()
        want_ids, want_sats = refs[g, v]
        bad[i, 0] = (ids[:count, :t] != want_ids).sum()
        bad[i, 1] = (sats[:count, :t] != want_sats).sum()
        if i % 10 == 9:
            ids7.fill_(-7)
            merge7()
            want7, want7_sat = refs7[v]
            bad[i, 2] = (ids7 != want7).sum() + (ws.sat_word[:7] != want7_sat).sum()
        if i % 25 == 24:
            got_ids, got_sats = ids[:count, :t].clone(), sats[:count, :t].clone()
            set_dcp_topk_merge_exchange("all_gather")
            try:
                run(g)
            finally:
                set_dcp_topk_merge_exchange("auto")
            assert torch.equal(ids[:count, :t], got_ids), f"replay {i}: ids"
            assert torch.equal(sats[:count, :t], got_sats), f"replay {i}: sat"

    torch.accelerator.synchronize()
    torch.distributed.barrier(group=group.cpu_group)
    failed = group.all_reduce(bad.clone())
    first = int(failed.any(1).int().argmax())
    assert not bool(failed.any()), f"replay {first}: {failed[first].tolist()}"
    every = group.all_gather(mailbox.ordinals.unsqueeze(0), dim=0)
    for r in range(world):
        assert torch.equal(every[r], every[0]), f"ordinals of rank {r} differ"
    assert bool((mailbox.box == -1).all()), "a mailbox slot is not empty"


def _check_error_after_publish(group, n: int) -> None:
    """A host check of the merge that fails once round 1 is under way
    (packed rows with fewer logits rows) raises, and the next merges, one rank
    late each, still equal the all-gather's: a failed call must not leave an
    unconsumed round-1 publish behind. Clean rows (no saturated rank), so a
    stale round-1 row shows as a wrong selection, not as a rescue wait."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
        dcp_topk_pack,
        set_dcp_topk_merge_exchange,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [150_000, 100_000, 70_123, 40_000, 120_000, 60_000, 90_000, 30_000]
    num_rows = len(lengths)
    a, ends = _make_case(lengths, seed=101)
    b, _ = _make_case(lengths, seed=103)
    local_a, local_end = _shard(a, ends, world, rank)
    local_b, _ = _shard(b, ends, world, rank)
    # A warm merge first: the select is compiled, so the next merge polls as
    # soon as its own pack has published (a cold select's JIT compile would
    # give every peer time to overwrite a stale row).
    warm = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
    dcp_topk_merge(local_a, local_end, warm, n)
    packed = dcp_topk_pack(local_a, local_end, n)
    with pytest.raises(ValueError):
        dcp_topk_merge_rows(
            logits=local_a[: num_rows // 2],
            row_ends=local_end,
            packed=packed,
            local_topk=n,
            topk=TOPK,
            req_id=torch.arange(num_rows, dtype=torch.int32, device="cuda"),
            block_table=_block_table(num_rows, int(local_end.max())),
            block_size=_BLOCK,
            stride_rows=_BLOCK,
            dcp_rank=rank,
            rows_out=torch.empty((num_rows, TOPK), dtype=torch.int32, device="cuda"),
            valid_out=torch.empty(num_rows, dtype=torch.int32, device="cuda"),
        )
    torch.accelerator.synchronize()
    torch.distributed.barrier(group=group.cpu_group)
    same = []
    for k in range(4):
        local = local_b if k % 2 == 0 else local_a
        ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
        if rank == (world - 1 - k) % world:
            torch.cuda._sleep(100_000_000)
        dcp_topk_merge(local, local_end, ids, n)
        ref = torch.full_like(ids, -7)
        set_dcp_topk_merge_exchange("all_gather")
        try:
            dcp_topk_merge(local, local_end, ref, n)
        finally:
            set_dcp_topk_merge_exchange("auto")
        torch.accelerator.synchronize()
        same.append(torch.equal(ids, ref))
    every = group.all_gather(
        torch.tensor([same], dtype=torch.int32, device="cuda"), dim=0
    )
    assert bool(every.all()), (
        f"merges after the error equal (rank x merge): {every.tolist()}"
    )


def _check_hook(group, n: int) -> None:
    """VLLM_DSA_BPRIME_CHECK over a real DCP group: the stock indexer's
    reference merge next to the one-round merge, eager and inside a CUDA
    graph, finds no miss on clean, tie-heavy and saturated rows, counts the
    saturated ones, and flags a lost token with its (layer, step, row)."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _check_dcp_decode_topk_merge,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import check
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge

    world, rank = group.world_size, group.rank_in_group
    state = check.get_dcp_topk_merge_check()
    assert state is not None, "VLLM_DSA_BPRIME_CHECK did not turn the check on"
    lengths = [100_000, 40_000, 9_000, 70_123, 150_000, 6_000, 3_000, 220_000]
    num_rows = len(lengths)
    base, ends = _make_case(lengths, seed=31)
    tied, _ = _make_case(lengths, seed=37, levels=16)
    variants = [
        base,
        tied,
        _boost(base, [0], slice(7, 8), world),
        _boost(base, list(range((TOPK - 1) // n)), slice(None), world),
    ]
    shards = [_shard(v, ends, world, rank) for v in variants]
    static = shards[0][0].clone()
    local_end = shards[0][1]
    seq_lens = local_end.view(-1, 1)
    max_seq_len = int(local_end.max())
    max_model_len = world << 17
    ids = torch.empty((num_rows, TOPK), dtype=torch.int32, device="cuda")

    def step(layer: str, merged: torch.Tensor | None = None) -> None:
        dcp_topk_merge(static, local_end, ids, n)
        _check_dcp_decode_topk_merge(
            layer,
            static,
            local_end,
            seq_lens,
            1,
            ids if merged is None else merged,
            n,
            "deep_select",
            False,
            max_seq_len,
            max_model_len,
            rank,
            world,
            16,
        )

    step("eager")
    torch.accelerator.synchronize()
    totals = state.snapshot()["totals"]
    assert totals[check.MISS_ROWS] == 0 and totals[check.ROWS] == num_rows, totals
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        step("graph")
    for i in range(2 * len(variants)):
        static.copy_(shards[i % len(variants)][0])
        graph.replay()
        torch.accelerator.synchronize()
        _assert_exact(ids, variants[i % len(variants)], ends)
    snap = state.snapshot()
    totals = snap["totals"]
    assert totals[check.CALLS] == 1 + 2 * len(variants), totals
    assert totals[check.MISS_ROWS] == 0 and not snap["misses"], snap
    assert totals[check.NAN_ROWS] == 0, totals
    # The bound variant saturates (K - 1) // N ranks of each long row.
    assert totals[check.SAT_ROWS] > 0, totals
    assert totals[check.MAX_SAT_RANKS] == (TOPK - 1) // n, totals

    static.copy_(shards[0][0])
    dcp_topk_merge(static, local_end, ids, n)
    lost = ids.clone()
    row0 = set(lost[0].tolist())
    top = int(variants[0][0, : lengths[0]].float().argmax())
    assert top in row0
    lost[0][lost[0] == top] = next(g for g in range(lengths[0]) if g not in row0)
    step("lost", lost)
    torch.accelerator.synchronize()
    snap = state.snapshot()
    assert snap["totals"][check.MISS_ROWS] == 1, snap
    assert snap["misses"] == [("lost", 1, 0, 0, 1)], snap
    state.log(final=True)


def run_worker(
    rank: int,
    world: int,
    local_rank: int,
    init_method: str,
    check_hook: bool = False,
    mailbox: bool = False,
    stress_replays: int = 300,
    error_path: bool = False,
) -> None:
    """One DCP rank. Also callable from a multi-node launcher. ``check_hook``
    runs the VLLM_DSA_BPRIME_CHECK case instead of the merge's own;
    ``mailbox`` takes round 1 through the multicast mailbox
    (VLLM_DSA_BPRIME_MAILBOX) and adds its checks, including
    ``stress_replays`` graph replays."""
    import os
    from types import SimpleNamespace

    if check_hook:
        os.environ["VLLM_DSA_BPRIME_CHECK"] = "1"
    os.environ["VLLM_DSA_BPRIME_MAILBOX"] = str(int(mailbox))

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        dcp_topk_merge_max_rows,
        init_dcp_topk_merge,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        get_dcp_topk_rescue_workspace,
    )

    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            group = get_dcp_group()
            n = dcp_merge_local_topk(TOPK, world)
            gate_config = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                # The longest context whose local ids fit the packed words.
                model_config=SimpleNamespace(max_model_len=world << 17),
                kv_transfer_config=None,
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=None,
                num_speculative_tokens=0,
            )
            assert init_dcp_topk_merge(gate_config, TOPK) == n, "merge disabled"
            ws = get_dcp_topk_rescue_workspace(TOPK, n)
            assert ws is not None and ws.max_rows >= 8
            assert dcp_topk_merge_max_rows(TOPK, n) == ws.max_rows
            assert (ws.mailbox is not None) == mailbox
            if check_hook:
                _check_hook(group, n)
                return
            if error_path:
                _check_error_after_publish(group, n)
                return
            if mailbox:
                _check_exchange_ab(group, ws, n)
                _check_late_publisher(group, n)
                _check_graph_stress(group, ws, n, stress_replays)
            _check_cases(group, ws, n)
            _check_deterministic(group, ws, n)
            _check_nonfinite(group, n)
            _check_cuda_graph(group, ws, n)
            _check_late_peer(group, ws, n)
        finally:
            cleanup_dist_env_and_memory()


def _spawned(
    rank: int,
    world: int,
    init_method: str,
    check_hook: bool,
    mailbox: bool,
    error_path: bool = False,
) -> None:
    run_worker(
        rank, world, rank, init_method, check_hook, mailbox, error_path=error_path
    )


_WORLDS = [
    pytest.param(
        w,
        marks=pytest.mark.skipif(
            torch.accelerator.device_count() < w, reason=f"Need {w} GPUs."
        ),
    )
    for w in (2, 4, 8)
]


@pytest.mark.parametrize("mailbox", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
def test_dcp_topk_merge_exact(world: int, mailbox: bool):
    _run_group(world, check_hook=False, mailbox=mailbox)


@pytest.mark.parametrize("mailbox", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
def test_dcp_topk_merge_check_hook(world: int, mailbox: bool):
    _run_group(world, check_hook=True, mailbox=mailbox)


@pytest.mark.parametrize("mailbox", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
def test_dcp_topk_merge_error_after_publish(world: int, mailbox: bool):
    _run_group(world, check_hook=False, mailbox=mailbox, error_path=True)


def _run_group(
    world: int, check_hook: bool, mailbox: bool, error_path: bool = False
) -> None:
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_spawned,
            args=(r, world, init_method, check_hook, mailbox, error_path),
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    # One deadline for the group: a hung rank fails the test, never CI.
    deadline = time.monotonic() + 600
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world
