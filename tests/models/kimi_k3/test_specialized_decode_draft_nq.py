# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, the DSpark drafter under the new drafter geometry
(kimi-k3.1-pro-dspark-preview: 96 heads, 6 per rank at TP16, Markov rank 512) and
arbitrary captured drafter batches.

CPU (pytest): the policy and coverage logic, no kernel launch: open table ranges,
Markov rank / shape / shared-memory coverage, the MarkovWide launch plan over request
slices, the split-K tactics of the new projection shapes, the wide drafter rows'
DcpValue residency rule and the work-list limits.

GPU (torchrun, one process per GPU; Rubin, multicast memory): the fused Markov
drafts at Markov rank 256 and 512 against the stock sequential loop (BF16 F.linear
bias, BF16 add, torch.argmax), at batches covering MarkovGreedy (plain, gated,
staged) and MarkovWide (one launch and several launches over request slices):

    torchrun --nproc-per-node 4 tests/models/kimi_k3/test_specialized_decode_draft_nq.py
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (  # noqa: E402
    markov,
    markov_wide,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    draft,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Shape,
    rows_policy,
)

VOCAB = 163840  # (Kimi K3 / K3.1 drafter vocabulary)
SMEM_227K, SMEM_SM107 = 232448, 334848
# (SMs, opt-in shared memory per CTA, shared memory per SM)
LIMITS_227K = (148, SMEM_227K, SMEM_227K + 1024)
LIMITS_SM107 = (160, SMEM_SM107, SMEM_SM107 + 1024)


# --------------------------------------------------------------------------- tables


def test_closed_open_ranges():
    spec = "1-7:1,8-64:2,65-:3"
    assert draft.closed(spec, 384) == "1-7:1,8-64:2,65-384:3"
    assert draft.closed(spec, 50) == "1-7:1,8-64:2"  # (the open range starts above)
    assert draft.closed("1-:1", 896) == "1-896:1"
    assert draft.closed("1-8:1", 896) == "1-8:1"  # (no open range: unchanged)


@pytest.mark.parametrize("steps,max_batch", [(3, 166), (3, 192), (7, 36), (7, 128)])
def test_tp16_open_tables_cover_every_captured_batch(steps, max_batch):
    """FUSED_MARKOV, FAST_DCP, FAST_CONTEXT and FAST_AR cover every TP16 drafter
    graph (DL3 / DL7, B1 up to the deployment's largest)."""
    batches = range(1, max_batch + 1)
    markov_table = draft.batch_table(draft.closed(draft.FUSED_MARKOV[16], max_batch))
    top_rows, top_context = max_batch * steps, max_batch * (steps + 1)
    for b in batches:
        assert markov_table.get(b), b
        rows, context = b * steps, b * (steps + 1)
        for table in (draft.FAST_DCP, draft.FAST_AR):
            spec = draft.closed(table[16], top_rows)
            assert draft.covered(spec, rows, steps), (table, rows)
        spec = draft.closed(draft.FAST_CONTEXT[16], top_context)
        assert draft.covered(spec, context, steps), context
    # FAST_AR: one-shot to 7 rows, two-shot 8-64, ArNormRows above.
    spec = draft.closed(draft.FAST_AR[16], top_rows)
    assert rows_policy(spec, Shape(7, steps)) == 1
    assert rows_policy(spec, Shape(64, steps)) == 2
    if top_rows > 64:
        assert rows_policy(spec, Shape(top_rows, steps)) == 3


def test_overlay_ranges_kept():
    """The cap128 overlay's measured ranges stay covered with their values."""
    table = draft.batch_table(draft.closed(draft.FUSED_MARKOV[16], 128))
    assert [table[b] for b in (1, 2, 3, 8, 9, 64, 65, 128)] == [2, 2, 3, 3] + [1] * 4
    for rows in (576, 768):
        assert draft.covered(draft.closed(draft.FAST_CONTEXT[16], 768), rows, 3)
    assert draft.covered(draft.closed(draft.FAST_AR[16], 576), 576, 3) == 3
    assert draft.covered(draft.closed(draft.FAST_DCP[16], 576), 576, 3)


# VR200 kernel bench (nqbench kern-draft gemm/lm_head) rows where the evict-first
# LM-head GEMM was slower than the stock cuBLAS shard GEMM (rows: steps).
LM_HEAD_CUBLAS_ROWS = {
    12: 3, 14: 7, 15: 3, 35: 7, 36: 3, 42: 7, 48: 3, 49: 7, 56: 7, 60: 3, 63: 7
}  # fmt: skip


@pytest.mark.parametrize("steps", [3, 7])
def test_tp16_lm_head_policy(steps):
    """The drafter's LM-head shard GEMM per query rows (DL3 / DL7, B1-256): the BF16
    GEMV to 7 rows, the evict-first GEMM only at 17-32 rows (cuBLAS splits K there;
    bench 0.887-0.916x), the stock cuBLAS shard GEMM elsewhere, by choice."""
    for b in range(1, 257):
        rows = b * steps
        gemv = draft.covered(draft.FAST_LM_HEAD[16], rows, steps)
        wide = draft.covered(draft.WIDE_LM_HEAD[16], rows, steps)
        assert bool(gemv) == (rows <= 7), rows
        assert bool(wide) == (17 <= rows <= 32), rows


def test_tp16_lm_head_slower_rows_run_cublas():
    """Every row count the bench measured the evict-first GEMM slower than cuBLAS at
    (12-15 rows 1.016-1.021x, 35-63 rows 1.099-1.115x) runs the stock cuBLAS GEMM."""
    for rows, steps in LM_HEAD_CUBLAS_ROWS.items():
        assert not draft.covered(draft.WIDE_LM_HEAD[16], rows, steps), rows
        assert not draft.covered(draft.FAST_LM_HEAD[16], rows, steps), rows


def test_wide_lm_head_rows_shared_with_verify():
    """Target verification (runtime.verify) runs the evict-first LM-head GEMM at
    lmhead.WIDE_ROWS, the same shard GEMM the bench measured: 17-32 rows only, the
    drafter's WIDE_LM_HEAD rows, cuBLAS at the measured-slower rows."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead

    assert (lmhead.WIDE_ROWS.start, lmhead.WIDE_ROWS.stop) == (17, 33)
    for steps in (3, 7):
        table = draft.WIDE_LM_HEAD[16]
        wide = {r for r in range(1, 257 * steps) if draft.covered(table, r, steps)}
        assert wide <= set(lmhead.WIDE_ROWS), steps
    assert not set(LM_HEAD_CUBLAS_ROWS) & set(lmhead.WIDE_ROWS)


# ---------------------------------------------------------------------- projections


@pytest.mark.parametrize(
    "shape", [(1152, 1536), (7168, 768), (768, 1536), (7168, 512), (7168, 896)]
)
def test_drafter_projection_tactics(shape):
    """The 64- and 96-head drafters' q_b / o_proj shapes have structurally valid
    split-K tactics (K in whole 128-column tiles per split) at <= 8 rows and in
    the 16-row bin; the rings cover both o_proj shapes with every K tile staged."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        CTA_K,
    )

    tactic = draft.PROJECTIONS[16][shape]
    _, k = shape
    assert k % (CTA_K * tactic[2]) == 0
    bins = dict(draft.PROJECTIONS_WIDE[16])
    if shape in bins[16]:
        assert k % (CTA_K * bins[16][shape][2]) == 0
    if shape[0] == 7168:  # o_proj / down_proj: rings, every K tile staged
        ring = draft.RING_PROJECTIONS[16][shape]
        assert ring[2] == 1 and ring[3] == k // CTA_K


def test_ring_producers_cover_both_drafters():
    """prepare_producers needs o_proj's and down_proj's shapes in the ring table."""
    rings = draft.RING_PROJECTIONS[16]
    down, gate_up = (7168, 896), (1792, 7168)
    for o_proj in ((7168, 512), (7168, 768)):
        assert {o_proj, down, gate_up} <= set(rings)
    assert draft.RING_PROJECTIONS[16][down][3] == 7


def test_projection_policies_new_shapes():
    """``projection``'s per-row tactic choice finds the new shapes at <= 8 rows and
    in the 9-16 / 17-32 bins (o_proj), cuBLAS above where the tables stop."""

    def policies(shape, rows):
        found = {}
        for r in rows:
            if r <= draft.PROJECTION_ROWS:
                tactic = draft.PROJECTIONS[16].get(shape)
            else:
                bins = draft.PROJECTIONS_WIDE[16]
                table = next((t for top, t in bins if r <= top), {})
                tactic = table.get(shape)
            if tactic is not None:
                found[r] = tactic
        return found

    rows = (7, 14, 21, 28, 56, 168)
    assert set(policies((7168, 768), rows)) == {7, 14, 21, 28}
    assert set(policies((1152, 1536), rows)) == {7, 14}
    assert {(1152, 1536), (7168, 768)} <= draft.tabled_shapes(16)


# --------------------------------------------------------------------------- Markov


def test_markov_supports_rank_and_shapes():
    for rank in (256, 512):
        shape = (VOCAB, rank)
        assert markov.supports(shape, shape, 16)
        assert markov.supports(shape, shape, 16, LIMITS_227K)
    # TP8: 80 CTAs per rank at rank 256, 160 at rank 512 (one per SM).
    assert markov.supports((VOCAB, 256), (VOCAB, 256), 8, LIMITS_227K)
    assert not markov.supports((VOCAB, 512), (VOCAB, 512), 8, LIMITS_227K)
    assert not markov.supports((VOCAB, 384), (VOCAB, 384), 16)  # (no CTA_ROWS entry)
    assert not markov.supports((VOCAB, 512), (VOCAB, 256), 16)  # (w1 != w2)
    assert not markov.supports((VOCAB + 8, 512), (VOCAB + 8, 512), 16)  # (split)
    # Co-residency: rank 512 at TP16 has 80 CTAs per rank (one per SM).
    few_sms = (64, SMEM_227K, SMEM_227K)
    assert not markov.supports((VOCAB, 512), (VOCAB, 512), 16, few_sms)


def test_markov_rank512_geometry():
    """Rank 512: 128 rows per CTA (8 tile warps, 32 k-steps in order as two 256-column
    halves), the same ~133 KB of staging as rank 256, twice the CTAs."""
    g256 = markov.MarkovGreedy(8, 7, 16, VOCAB // 16)
    g512 = markov.MarkovGreedy(8, 7, 16, VOCAB // 16, rank=512)
    assert (g256.rows, g256.row_warps, g256.k_halves, g256.ctas) == (256, 16, 1, 40)
    assert (g512.rows, g512.row_warps, g512.k_halves, g512.ctas) == (128, 8, 2, 80)
    assert g512.rows * g512.pitch <= g256.rows * g256.pitch  # 133120 <= 135168
    assert markov.greedy_smem(512) < SMEM_227K
    assert g512.per_thread <= 32  # (the pending bit mask)
    w256 = markov_wide.MarkovWide(64, 7, 16, VOCAB // 16)
    w512 = markov_wide.MarkovWide(64, 7, 16, VOCAB // 16, rank=512)
    assert (w256.halves, w256.a_registers, w256.k_halves) == (1, True, 1)
    assert (w512.halves, w512.a_registers, w512.k_halves) == (2, False, 2)


def test_markov_wide_launch_width():
    vr = VOCAB // 16
    # Rank 256: the base tree's 64 (and up to 72) at 227 KiB; the overlay's 128 at
    # SM107's opt-in (~304 KB sampled).
    assert markov_wide.launch_width(16, 256, True, LIMITS_227K, vr) == 72
    assert markov_wide.launch_width(16, 256, True, LIMITS_SM107, vr) == 128
    assert 300_000 < markov_wide.wide_smem(128, 16, 256, True) < SMEM_SM107
    # Rank 512: fewer slots per launch (1040 B rows).
    assert markov_wide.launch_width(16, 512, True, LIMITS_227K, vr) == 56
    assert markov_wide.launch_width(16, 512, False, LIMITS_227K, vr) >= 56
    assert markov_wide.launch_width(16, 512, True, LIMITS_SM107, vr) == 120
    for rank in (256, 512):
        for sampled in (False, True):
            for limits in (LIMITS_227K, LIMITS_SM107):
                width = markov_wide.launch_width(16, rank, sampled, limits, vr)
                assert width % 8 == 0 and 0 < width <= markov_wide.MAX_WIDTH
                assert markov_wide.wide_smem(width, 16, rank, sampled) <= limits[1]
    # Not co-resident (too few SMs for 80 CTAs): no wide launch.
    assert markov_wide.launch_width(16, 512, True, (40, SMEM_227K, SMEM_227K), vr) == 0


def _plan_only(rank, limits, wide=True):
    """A PreparedMarkov with just what ``plan`` reads (no mailboxes)."""
    prepared = object.__new__(markov.PreparedMarkov)
    prepared.tp, prepared.markov_rank, prepared.limits = 16, rank, limits
    prepared.vocab_rank, prepared.plans = VOCAB // 16, {}
    prepared.wide_box = object() if wide else None
    return prepared


@pytest.mark.parametrize("rank", [256, 512])
@pytest.mark.parametrize("limits", [LIMITS_227K, LIMITS_SM107])
@pytest.mark.parametrize("sampled", [False, True])
def test_markov_plan_covers_any_batch(rank, limits, sampled):
    prepared = _plan_only(rank, limits)
    limit = markov_wide.launch_width(16, rank, sampled, limits, VOCAB // 16)
    for batch in list(range(1, 200)) + [256, 333, 512]:
        launches = prepared.plan(batch, sampled)
        assert launches, batch
        if batch <= markov.MAX_WIDTH:
            assert launches == [(0, batch, 0)]
            continue
        # Request slices in order, covering every request once, each at most its
        # launch's width, every width a multiple of 8 within the device's limit.
        assert launches[0][0] == 0 and launches[-1][1] == batch
        for (lo, hi, width), nxt in zip(launches, launches[1:] + [None]):
            assert 0 < hi - lo <= width <= limit and width % 8 == 0
            if nxt is not None:
                assert nxt[0] == hi
        assert len(launches) == -(-batch // limit)
    # Without the wide mailbox only MarkovGreedy batches are covered.
    narrow = _plan_only(rank, limits, wide=False)
    assert narrow.covers(8, sampled) and not narrow.covers(9, sampled)


def test_markov_chained_launch_keys():
    """MarkovWide launches after the first of a plan compile ``chained`` (ordinal read
    and re-arm after the PDL wait: back to back, nothing else orders them)."""
    prepared = _plan_only(512, LIMITS_227K)
    keys = prepared.keys(130, 1, 7, True)
    width = markov_wide.launch_width(16, 512, True, LIMITS_227K, VOCAB // 16)
    count = -(-130 // width)
    assert count > 1
    assert [k[0][2] for k in keys] == [False] + [True] * (count - 1)
    assert all(k[0][:2] == ("wide", keys[0][0][1]) for k in keys)
    assert prepared.keys(48, 1, 7, True) == [(("wide", 48, False), 1, 7, True)]
    assert prepared.keys(8, 1, 7, True) == [(8, 1, 7, True)]
    assert markov_wide.MarkovWide(64, 7, 16, VOCAB // 16, chained=True).chained
    assert not markov_wide.MarkovWide(64, 7, 16, VOCAB // 16).chained


def test_markov_plan_single_launch_matches_base_tree():
    """One launch where the base tree had one: width = the batch rounded up to 8."""
    prepared = _plan_only(256, LIMITS_227K)
    for batch in (9, 16, 33, 64):
        assert prepared.plan(batch) == [(0, batch, -(-batch // 8) * 8)]


# ------------------------------------------------------------- drafter attention


def test_draft_uncovered_rows():
    """DraftDcp's stock-path rows by cause (logged per cause by runtime.draft)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )

    rows = (3, 6, 8, 63, 96, 192, 200)
    dropped = attention.uncovered_rows(rows, 3, 64, 16, 160)
    assert dropped == {8: "not a whole draft length", 200: "not a whole draft length"}
    gb300 = attention.uncovered_rows(rows, 3, 64, 16, 148)
    assert set(gb300) == {8, 96, 192, 200}
    assert "Rubin" in gb300[96] and gb300[8] == "not a whole draft length"
    assert "without DCP" in attention.uncovered_rows(rows, 3, 64, 1, 160)[96]


def test_draft_work_list_limits(monkeypatch):
    """The work list (8-bit request and tile, 4096 slots: mla.wlist.fits) is skipped,
    not asserted, past its limits: the long split runs without the list."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import wlist

    made = []
    monkeypatch.setattr(attention, "long_split", lambda *a, **k: (1, 64, 128))
    monkeypatch.setattr(wlist, "WorkList", lambda *a: made.append(a) or a)
    monkeypatch.setattr(wlist, "_LISTS", {})
    monkeypatch.setattr(attention, "DRAFT_WLIST", {r: 1 for r in range(1, 4000)})
    dcp = SimpleNamespace(steps=3, device="cpu", size=16)
    work = attention.DraftDcp.work_list
    assert work(dcp, 3 * 192, 96, 3, 4, 160) is not None  # 192 x 3 x 4 = 2304 slots
    assert work(dcp, 3 * 257, 96, 3, 1, 160) is None  # 257 requests
    assert work(dcp, 3 * 200, 96, 3, 8, 160) is None  # 4800 slots
    assert len(made) == 1


def test_query_heads_reads_mlaquery_table():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import query

    for heads in (4, 6, 8, 12, 16):
        assert attention.query_heads(heads) == (heads in query.DRAFT_TILED_FROM)
    # Both drafters at TP16 and TP8: 64 heads (4 / 8 per rank), 96 heads (6 / 12).
    assert all(attention.query_heads(h) for h in (4, 8, 6, 12))


def test_draft_fold_splits_role(monkeypatch):
    """The drafter reads mla.dcp's split tables at the gathered head counts measured
    on a drafter (64, its own (rows, 64) keys) and at its packed twins (96 heads, q 3
    / 7: the twin verify's keys, mla.dcp PACKED_TWIN), both through
    verify_fold_splits."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )

    called = []
    monkeypatch.setattr(
        attention,
        "verify_fold_splits",
        lambda *a, **k: called.append((*a, k)) or (9, 128),
    )
    assert attention.draft_fold_splits(56, 7, 64, 1 << 16, "cpu", 16) == (9, 128)
    assert attention.draft_fold_splits(54, 3, 96, 1 << 16, "cpu", 4) == (9, 128)
    assert attention.draft_fold_splits(63, 7, 96, 1 << 16, "cpu", 16) == (9, 128)
    # (the drafter's role, its DCP size: its own rules and twins)
    assert called == [
        (56, 7, 64, 1 << 16, "cpu", {"draft": 16}),
        (54, 3, 96, 1 << 16, "cpu", {"draft": 4}),
        (63, 7, 96, 1 << 16, "cpu", {"draft": 16}),
    ]
    assert 64 in attention.DRAFT_RULE_HEADS and 96 not in attention.DRAFT_RULE_HEADS


def test_packed_twin_keys():
    """The drafter's (``draft``) 96-head packed core at q 3 / 7 (the causal drafter)
    reads its twin verify's rows (the same requests at q 4 / 8: the same 3 / 6 query
    tiles per request); every target verify (whatever its q) and other head counts
    read their own rows."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    for (heads, q), twin in dcp.PACKED_TWIN.items():
        assert -(-heads * q // 128) == -(-heads * twin // 128)
    assert dcp.split_key(54, 96, 3, 16) == ((72, 96), True)
    assert dcp.split_key(63, 96, 7, 4) == ((72, 96), True)
    assert dcp.split_key(480, 96, 3, 16) == ((640, 96), True)
    for rows, heads, q in ((72, 96, 4), (72, 96, 8), (63, 64, 7), (54, 64, 3),
                           (8, 96, 1), (54, 96, None), (55, 96, 3)):  # fmt: skip
        assert dcp.split_key(rows, heads, q, 16) == ((rows, heads), False)
    # The target's verify rows read LONG_SPLIT at their own rows at every q, the
    # drafter's q 3 / 7 included (a K3 target at 2 / 6 speculative tokens): no twin,
    # no drafter rule (e.g. q 3 B4, 12 rows, has none: its gathered verify core stays
    # a valid startup).
    for (rows, heads), rule in dict.items(dcp.LONG_SPLIT):
        for q in (3, 4, 7, 8):
            if heads == 96 and rows % q == 0:
                assert dcp.split_rule(rows, heads, q) == rule
    assert dcp.split_rule(12, 96, 3) is None
    assert dcp.long_split(12, 96, 8, 200, 3) is None
    # The drafter's seam rows (DL3 B18 / B19 / B21, DL7 B9): its cake rule at both
    # DCP sizes (the twin's short split 2, its launched split), only at that split.
    sms = 200
    for size in (4, 16):
        _, floor, over = dcp.DRAFT_CAKE[size][0]
        for rows, q in ((54, 3), (57, 3), (63, 3), (63, 7)):
            qtiles = -(-q * 96 // 128)
            launched = dcp.split_rule(rows, 96, q, size)[1]
            long = dcp.long_split(rows, 96, launched, sms, q, size)
            want = (2, 0, floor, sms // 2, qtiles, over)
            assert long == want, (rows, q, size)
            assert dcp.long_split(rows, 96, 2, sms, q, size) is None  # (short split)
            assert dcp.long_split(rows, 96, launched, 148, q, size) is None  # (GB300)
    # Odd DL3 batches below B17 (no q4 rule): the next batch's short split, twin only.
    assert dcp.split_rule(9, 96, 3, 16)[0] == 4
    assert dcp.split_rule(12, 96, 4, 16) is None
    assert dcp.TWIN_FOLD_SPLITS[12, 96] == 4


def test_draft_cake_rules():
    """The 96-head drafter's cake rules (mla.dcp DRAFT_LONG_SPLIT, per DCP size
    DRAFT_CAKE): every DL3 / DL7 fold and wide row whose twin launches a split above
    1, with the twin's short split, long tiles 0 (every request on the per-step
    quantum), the fold rows at DRAFT_CAKE's launched split and the wide rows at the
    twin's (its workspace); the target reads none of them."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    for (rows, heads, q, size), rule in dcp.DRAFT_LONG_SPLIT.items():
        key = (rows // q * dcp.PACKED_TWIN[heads, q], heads)
        twin = dcp.TWIN_LONG_SPLIT.get(key) or dcp.LONG_SPLIT[key]
        fold, wide = dcp.DRAFT_CAKE[size]
        want = (twin[1], *wide) if rows > 64 else fold
        launched = want[0]
        if rows <= 64:
            launched = dcp.DRAFT_CAKE_LAUNCHED[size].get((q, rows), launched)
        assert rule == (twin[0], launched, 0, want[1], want[2])
        assert dcp.split_rule(rows, heads, q, size) == rule
        assert dcp.split_rule(rows, heads, q) == dcp.LONG_SPLIT.get((rows, heads))
    for size in (4, 16):
        assert {r for r, _, q, s in dcp.DRAFT_LONG_SPLIT if s == size and q == 3} == (
            set(range(9, 481, 3))
        )
        assert {r for r, _, q, s in dcp.DRAFT_LONG_SPLIT if s == size and q == 7} == (
            set(range(14, 449, 7))
        )
    assert dcp.split_rule(66, 96, 3, 4)[1] == dcp.LONG_SPLIT[88, 96][1]


def test_cake_partition():
    """The cake partition (mla.dcp ``cake_quant`` + ``long_per``, host twins of the
    device word and the cores' / DcpValue's partition): a uniform batch takes the
    most splits that still fit one wave of items (not the ones just past a wave),
    one long request among short ones takes the launched splits and the short ones
    one or two; every partition covers its request's tiles exactly once
    (contiguous splits, the last one partial)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    sms, q, launched = 212, 3, 16
    long = (2, 0, 16, 106, 3, 112)  # (short, long tiles, floor, pairs, q tiles, over)

    def splits(lengths):
        quant = dcp.cake_quant(lengths, launched, long)
        out = []
        for n in lengths:
            tiles = -(-n // 128)
            per = dcp.long_per_host(tiles, launched, long, quant)
            filled = -(-tiles // per) if tiles else 0
            assert filled <= launched and (filled - 1) * per < tiles <= filled * per
            out.append(filled)
        return out

    # DL3 B6, 390 tiles each: 5 splits (90 items: one wave of 78 tiles), not 6 (108
    # items: a second wave) nor 11 (2 waves of 36)
    assert splits([390 * 128] * 6) == [5] * 6
    skewed = splits([1560 * 128] + [100 * 128] * 5)
    assert skewed[0] == launched and max(skewed[1:]) <= 2
    assert splits([16 * 128] * 18) == [1] * 18  # (the floor: one split each)
    assert splits([0, 5_000])[0] == 0
    assert dcp.cake_quant([0, 0], launched, long) == 16
    rule = dcp.long_split(54, 96, dcp.DRAFT_CAKE[4][0][0], sms, q, 4)
    assert rule[3:] == (sms // 2, 3, dcp.DRAFT_CAKE[4][0][2])


@pytest.mark.parametrize("strict, gather_q, raises", [
    (False, True, False), (False, False, True), (True, True, True),
])  # fmt: skip
def test_merge_refuses_the_stock_core(monkeypatch, strict, gather_q, raises):
    """NQ-NOFALLBACK: a drafter row with no specialized core reaches ``merge``'s stock
    tokenspeed forward_mqa only outside a strict captured graph and where MlaQuery
    gathers the q rows it would read (24 heads per rank: never); otherwise it is
    refused there."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Uncovered

    class Stock(Exception):
        pass

    def forward_mqa(*args):
        raise Stock

    monkeypatch.setattr(attention, "stream", lambda device: None)
    monkeypatch.setattr(attention, "full_capture", lambda: True)
    shared = SimpleNamespace(
        box=SimpleNamespace(q=torch.zeros((2, 64, 4, 576), dtype=torch.uint8)),
        device="cpu", wide_cores={}, strict=strict, gather_q=gather_q, steps=3,
        core_of=lambda rows: "stock tokenspeed forward_mqa + DcpValue",
    )  # fmt: skip
    layer = object.__new__(attention.DraftLayerDcp)
    layer.shared, layer.parity = shared, 0
    layer.join_context = lambda: None
    layer.fold_core = lambda *args: False
    module = SimpleNamespace(
        impl=SimpleNamespace(forward_mqa=forward_mqa),
        _attn_read_kv_cache=lambda: None,
    )
    out = torch.empty((54, 512))
    with pytest.raises(Uncovered if raises else Stock):
        layer.merge(module, SimpleNamespace(decode=None), 54, out)


@pytest.mark.parametrize("steps, top", [(3, 192), (7, 96)])
def test_packed_twin_rules_cover_drafter_rows(steps, top):
    """Every 96-head drafter fold / wide row (not rows mode) has a length-keyed rule
    through its twin: none is left to tokenspeed's shape-only split."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    for reqs in range(1, top + 1):
        rows = reqs * steps
        if attention.ROWS_MODE.get(rows):
            continue
        for size in (4, 16):
            assert dcp.split_rule(rows, 96, steps, draft=size) is not None, rows


def test_dcp_value_uv_layout():
    """DcpValue's W_UV fragments are 16 B loads of latent runs: compile_value refuses
    an output-contiguous W_UV (lane group g's load at byte 2 g mod 16: the DCP check's
    CUDA_ERROR_MISALIGNED_ADDRESS) and takes the model's layout, which the DCP check's
    layers now use."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        uv_layout_error,
    )

    bf16 = torch.bfloat16
    weight = torch.zeros((DCP_HEADS * 256, 512), dtype=bf16)  # kv_b_proj [N(P+V), L]
    model = weight.T.view(512, DCP_HEADS, 256).split([128, 128], dim=-1)[1]
    assert uv_layout_error(layout(model.transpose(0, 1))) is None
    gen = torch.Generator().manual_seed(0)
    check = _DraftAttention(gen, "cpu", None, 4, 0, 1).W_UV
    assert check.stride(1) == 1 and uv_layout_error(layout(check)) is None
    output_contiguous = torch.zeros((512, DCP_HEADS, 128), dtype=bf16).transpose(0, 1)
    assert "latent-contiguous" in uv_layout_error(layout(output_contiguous))
    assert uv_layout_error(layout(torch.zeros((DCP_HEADS, 512, 128)))) is not None


# ------------------------------------------------------------- GPU (torchrun)


def _rubin_world():
    """Devices for the torchrun checks: Rubin-class (more SMs than GB300's 152),
    at least 2."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return 0
    if torch.cuda.get_device_properties(0).multi_processor_count <= 152:
        return 0
    return torch.cuda.device_count()


@pytest.mark.parametrize(
    "part", ["markov", "dcp", "dcp96", "query96", "c512rows", "context", "arnorm"]
)
def test_gpu_torchrun(part):
    """The torchrun checks below on every local GPU (skipped without 2+ Rubin-class
    GPUs). The DCP check runs at DCP = the local GPU count (6 heads per rank: 96
    gathered heads only at 16 GPUs, ``torchrun --nnodes 4``); ``dcp96`` / ``query96``
    at 96 gathered heads (96 / GPUs per rank: 24 at 4 GPUs, C512's drafter layer;
    ``query96`` from the in-projection rows through MlaQuery; ``c512rows``: C512's
    whole DL3 row range served at 4 GPUs). ``context`` / ``arnorm``:
    the drafter's wide context row gather and wide all-reduce sites at row counts
    above one wave (the persistent kernels)."""
    import subprocess

    world = _rubin_world()
    if not world:
        pytest.skip("needs 2+ Rubin-class GPUs (multicast memory)")
    command = [sys.executable, "-m", "torch.distributed.run"]
    command += [f"--nproc-per-node={world}", __file__, part]
    subprocess.run(command, check=True, timeout=3600)


def _stock_drafts(w1, w2, base, anchors, steps):
    """The stock sequential Markov loop: bias = F.linear(markov_w1[prev], markov_w2)
    (BF16), logits = base + bias (BF16), torch.argmax."""
    prev = anchors.long()
    out = []
    for i in range(steps):
        bias = torch.nn.functional.linear(w1[prev], w2)
        logits = base[:, i] + bias
        prev = torch.argmax(logits, dim=-1)
        out.append(prev)
    return torch.stack(out, dim=1)


def _stock_sampled(w1, w2, base, anchors, steps, sampling):
    """The stock probabilistic loop (DSparkSpeculator._sample_sequential with
    ``_sample_logits``): logits = base + F.linear(markov_w1[prev], markov_w2) (BF16),
    gumbel_sample (temperature, the drafting salt, position - 1) writing every
    position's pre-temperature logits into the cache column ``i``."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    temperature, seeds, slots, positions, cache = sampling
    batch = anchors.shape[0]
    slots, positions = slots.view(batch, steps), positions.view(batch, steps)
    cols = torch.arange(steps, dtype=torch.int32, device=base.device)
    prev = anchors.long()
    out = []
    for i in range(steps):
        logits = base[:, i] + torch.nn.functional.linear(w1[prev], w2)
        prev = gumbel_sample(
            logits,
            slots[:, i].contiguous(),
            temperature,
            seeds,
            positions[:, i] - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache,
            logits_cache_col=cols[i],
        )
        out.append(prev)
    return torch.stack(out, dim=1)


def _markov_inputs(batch, steps, vocab, gen, device):
    """Identical on every rank (the same generator seed and call order): base logits
    [batch, steps, V], anchors, and the sampling state: temperature (a fifth of the
    slots at 0: argmax) / seeds per slot, each request's slot (all distinct, live)
    and its positions (consecutive predicted-token positions)."""
    base = torch.randn((batch, steps, vocab), generator=gen, device=device).bfloat16()
    anchors = torch.randint(
        0, vocab, (batch,), generator=gen, device=device, dtype=torch.int32
    )
    max_reqs = batch + 5
    temperature = torch.rand((max_reqs,), generator=gen, device=device) + 0.5
    temperature[::5] = 0.0
    seeds = torch.randint(
        0, 1 << 62, (max_reqs,), generator=gen, device=device, dtype=torch.int64
    )
    slots = torch.randperm(max_reqs, generator=gen, device=device)[:batch]
    start = torch.randint(
        1, 1 << 20, (batch,), generator=gen, device=device, dtype=torch.int64
    )
    positions = start[:, None] + torch.arange(steps, device=device)
    slots = slots.repeat_interleave(steps).contiguous()
    positions = positions.reshape(-1).contiguous()
    return base, anchors, (temperature, seeds, slots, positions)


def _markov_check(case, batch, sampled, label, inputs=None):
    """One fused drafting call of ``batch`` requests against the stock loop
    (``case``: the PreparedMarkov and its weights / shard); the failures found."""
    steps, lo, vocab_rank = case.steps, case.lo, case.vocab_rank
    base_full, anchors, state = inputs or _markov_inputs(
        batch, steps, case.vocab, case.gen, case.device
    )
    base = base_full[:, :, lo : lo + vocab_rank].reshape(batch * steps, -1)
    base = base.contiguous()
    draft_out = torch.zeros((batch, steps), dtype=torch.int64, device=case.device)
    sampling = expected_cache = None
    temperature, seeds, slots, positions = state
    if sampled:
        cache = torch.zeros(
            (temperature.shape[0], steps, case.vocab),
            dtype=torch.bfloat16,
            device=case.device,
        )
        expected_cache = cache.clone()
        stock = (temperature, seeds, slots, positions, expected_cache)
        expected = _stock_sampled(case.w1, case.w2, base_full, anchors, steps, stock)
        sampling = case.prepared.sampling_args(
            temperature, seeds, slots, positions, cache
        )
    else:
        expected = _stock_drafts(case.w1, case.w2, base_full, anchors, steps)
    case.prepared.run(base, anchors, draft_out, sampling)
    torch.accelerator.synchronize(case.device)
    bad = []
    if not torch.equal(draft_out, expected):
        bad.append(f"{(draft_out != expected).sum().item()} drafts")
    if sampling is not None:
        cache, columns = sampling[4], slice(lo, lo + vocab_rank)
        live = slots[::steps]
        if not torch.equal(cache[live, :, columns], expected_cache[live, :, columns]):
            bad.append("cached logits")
        untouched = cache.clone()
        untouched[live, :, columns] = 0
        if untouched.any():
            bad.append("cache written outside this rank's live columns")
    if bad:
        print(f"rank {case.rank}: Markov rank {case.markov_rank} {label}: {bad}")
    return len(bad)


def _markov_main(rank, world, device, group):
    """Fused Markov drafts (greedy and sampled) at Markov rank 256 and 512 against the
    stock loops: the drafts bitwise and, sampled, this rank's columns of every live
    request's draft-logits cache rows bitwise (no other column or slot written). B9+
    run MarkovWide, B130 / B200 several launches over request slices; a loop of
    multi-launch calls with fresh inputs each time (separated only by plain kernels,
    as the LM head separates them in serving) checks the chained launches' parity /
    ordinal hand-off (a stale ordinal reuses a published parity: stale drafts or a
    hang)."""
    steps, failures, loops = 7, 0, 24
    for markov_rank in (256, 512):
        rows = markov.CTA_ROWS[markov_rank]
        vocab = world * rows * 8
        gen = torch.Generator(device=device).manual_seed(1234 + markov_rank)
        shape = (vocab, markov_rank)
        w1 = (torch.randn(shape, generator=gen, device=device) * 0.05).bfloat16()
        w2 = (torch.randn(shape, generator=gen, device=device) * 0.05).bfloat16()
        prepared = markov.PreparedMarkov(
            w1, w2, world, rank, group, steps, gated=(1, 2, 3, 8), staged=(3, 8),
            wide=True,
        )  # fmt: skip
        case = SimpleNamespace(
            prepared=prepared, w1=w1, w2=w2, steps=steps, vocab=vocab, gen=gen,
            vocab_rank=vocab // world, lo=rank * (vocab // world), device=device,
            rank=rank, markov_rank=markov_rank,
        )  # fmt: skip
        for sampled in (False, True):
            mode = "sampled" if sampled else "greedy"
            for batch in (1, 2, 3, 8, 9, 64, 130, 200):
                prepared.prepare(batch, 1, steps, sampled)
                failures += _markov_check(case, batch, sampled, f"{mode} B{batch}")
            for batch in (130, 200):
                launches = len(prepared.plan(batch, sampled))
                cases = [
                    _markov_inputs(batch, steps, vocab, gen, device) for _ in range(4)
                ]
                for i in range(loops):
                    label = f"{mode} B{batch} x{launches} launches, call {i}"
                    failures += _markov_check(case, batch, sampled, label, cases[i % 4])
    return failures


# The DraftDcp check: the drafter's MLA geometry (heads per rank 6: the 96-head
# drafter), FP8 KV pages of 64 tokens, q / k scales as a calibrated checkpoint.
DCP_HEADS, PAGE, Q_SCALE, K_SCALE = 6, 64, 1.0, 0.5
SOFTMAX_SCALE = (128 + 64) ** -0.5


def _local_len(total, world, rank, interleave):
    """Tokens of a ``total``-token sequence on ``rank`` (block-interleaved DCP)."""
    chunk = interleave * world
    full = total // chunk
    rest = total - full * chunk
    return full * interleave + min(max(rest - rank * interleave, 0), interleave)


def _dcp_inputs(
    reqs, steps, world, rank, interleave, gen, pages_gen, device, long=0,
    heads_rank=DCP_HEADS,
):  # fmt: skip
    """Contexts mixing very short requests (each token weighs a few percent: a mask
    error shows), medium ones and ``long`` past 64 local KV tiles (the long splits);
    every rank's FP8 KV of its own tokens in pages shuffled by the rank's
    ``pages_gen``; the global FP8 query rows [rows, DCP x heads, 576]. ``gen`` draws
    the same on every rank (contexts, then queries)."""
    heads = world * heads_rank
    kinds = torch.randint(0, 3, (reqs,), generator=gen, device="cpu").tolist()
    totals = []
    for i, kind in enumerate(kinds):
        if i < long:
            low, high = 64 * 128 * world + 1, 64 * 128 * world + 4096
        elif kind == 0:
            low, high = steps + 1, 48
        else:
            low, high = 64, 6000
        totals.append(int(torch.randint(low, high, (1,), generator=gen)))
    rows = reqs * steps
    q = torch.randn((rows, heads, 576), generator=gen, device="cpu") * 0.5
    q = q.to(torch.float8_e4m3fn).view(torch.uint8)
    q[q == 0x80] = 0  # (-0 -> +0: no chunk starts with the validity sentinel)
    local = [_local_len(t, world, rank, interleave) for t in totals]
    pages = [max(1, -(-n // PAGE)) for n in local]
    width = max(pages)
    blocks = sum(pages) + 1
    order = torch.randperm(blocks - 1, generator=pages_gen) + 1
    table = torch.zeros((reqs, width), dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[used : used + n]
        used += n
    table = table.to(device)
    cache = torch.zeros((blocks, PAGE, 576), dtype=torch.uint8, device=device)
    for r, total in enumerate(totals):
        g = torch.Generator(device=device).manual_seed(9000 + r)
        kv = torch.randn((total, 576), generator=g, device=device) * 0.5
        kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        t = torch.arange(total, device=device)
        mine = (t // interleave) % world == rank
        t = t[mine]
        position = (t // (interleave * world)) * interleave + t % interleave
        page = table[r].long()[position // PAGE]
        cache[page, position % PAGE] = kv[mine]
    i32 = dict(dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=table,
            seq_lens=torch.tensor(local, **i32),
            dcp_tot_seq_lens=torch.tensor(totals, **i32),
        ),
        num_decodes=reqs,
        num_decode_tokens=rows,
        num_prefills=0,
        max_seq_len=max(totals),
    )
    return cache.view(torch.float8_e4m3fn), q.to(device), metadata


def _tokenspeed(q, cache, metadata, causal, world, rank, interleave):
    """Stock TokenspeedMLAImpl.forward_mqa under DCP (return_lse): the local shard
    over every gathered head, [rows, heads, 512] BF16 and LSE (log2) [rows, heads]."""
    from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode

    decode = metadata.decode
    reqs = decode.seq_lens.shape[0]
    rows, heads = q.shape[0], q.shape[1]
    q = q.view(torch.float8_e4m3fn).view(reqs, rows // reqs, heads, 576)
    size = get_num_sm(q.device) * heads * 8 * 513 * 4
    workspace = torch.empty(size, dtype=torch.int8, device=q.device)
    o, lse = tokenspeed_mla_decode(
        query=q,
        kv_cache=cache,
        workspace_buffer=workspace,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=decode.block_table,
        seq_lens=decode.seq_lens,
        max_seq_len=metadata.max_seq_len,
        softmax_scale=SOFTMAX_SCALE * Q_SCALE * K_SCALE,
        output_scale=K_SCALE,
        enable_pdl=False,
        return_lse=True,
        causal_mask=causal,
        causal_seqs=decode.dcp_tot_seq_lens,
        cp_world=world,
        cp_rank=rank,
        cp_interleave_size=interleave,
        min_split_kv=1,
    )
    return o.view(rows, heads, 512), lse.reshape(rows, heads)


def _stock_dcp(q, cache, metadata, causal, w_uv, world, rank, interleave, group):
    """The stock drafter's DCP attention output [rows, H*128]: tokenspeed, the empty
    shards masked, cp_lse_ag_out_rs (correct_attn_out on the all-gathered LSEs, then
    the BF16 reduce-scatter over heads) and W_UV (torch.bmm, as _v_up_proj)."""
    import torch.distributed as dist

    from vllm.v1.attention.ops.dcp import correct_attn_out, mask_dcp_empty_shards_

    o, lse = _tokenspeed(q, cache, metadata, causal, world, rank, interleave)
    decode = metadata.decode
    reqs = decode.seq_lens.shape[0]
    rows, heads = o.shape[0], o.shape[1]
    starts = torch.arange(reqs + 1, dtype=torch.int32, device=o.device)
    lse = lse.contiguous()
    mask_dcp_empty_shards_(lse, decode.seq_lens, starts * (rows // reqs))
    # (an empty shard's undefined rows; their weight is 0)
    o = torch.where(torch.isinf(lse)[..., None], torch.zeros_like(o), o)
    lses = torch.empty((world, rows, heads), dtype=lse.dtype, device=o.device)
    dist.all_gather_into_tensor(lses, lse, group=group)
    out, _ = correct_attn_out(o, lses, rank, None, is_lse_base_on_e=False)
    per = heads // world
    mine = torch.empty((rows, per, 512), dtype=out.dtype, device=o.device)
    chunks = out.view(rows, world, per, 512).transpose(0, 1).contiguous()
    dist.reduce_scatter_tensor(mine, chunks, group=group)
    result = torch.empty((rows, per, 128), dtype=out.dtype, device=o.device)
    torch.bmm(mine.transpose(0, 1), w_uv, out=result.transpose(0, 1))
    return result.view(rows, per * 128)


def _exact_dcp(q, cache, metadata, w_uv, world, rank, interleave, group):
    """The drafter's causal DCP attention in FP32 from the FP8 q / KV bytes, with no
    intermediate rounding: per rank the local softmax over its tokens (global
    positions, interleave) for every gathered head, the ranks' (max, sum, rows)
    merged exactly, then this rank's heads through W_UV. [rows, H*128] FP32: the
    reference both the stock and the specialized path round away from."""
    import torch.distributed as dist

    decode = metadata.decode
    reqs = decode.seq_lens.shape[0]
    rows, heads = q.shape[0], q.shape[1]
    steps = rows // reqs
    f32 = torch.float32
    qf = q.view(torch.float8_e4m3fn).to(f32)
    flat = cache.view(torch.float8_e4m3fn).view(-1, 576)
    scale = SOFTMAX_SCALE * Q_SCALE * K_SCALE
    o = torch.zeros((rows, heads, 512), dtype=f32, device=q.device)
    m = torch.full((rows, heads), float("-inf"), dtype=f32, device=q.device)
    total = torch.zeros((rows, heads), dtype=f32, device=q.device)
    lens = decode.seq_lens.tolist()
    totals = decode.dcp_tot_seq_lens.tolist()
    table = decode.block_table.long()
    for r in range(reqs):
        n = lens[r]
        if not n:
            continue
        i = torch.arange(n, device=q.device)
        g = (i // interleave) * interleave * world + rank * interleave + i % interleave
        kv = flat[table[r, i // PAGE] * PAGE + i % PAGE].to(f32)
        part = slice(r * steps, (r + 1) * steps)
        pos = torch.arange(totals[r] - steps, totals[r], device=q.device)
        logits = torch.einsum("shd,nd->shn", qf[part], kv) * scale
        logits = logits.masked_fill((g[None, :] > pos[:, None])[:, None, :], -torch.inf)
        top = logits.amax(-1)
        weights = torch.exp(logits - top.clamp_min(-1e30)[..., None])
        o[part] = torch.einsum("shn,nd->shd", weights, kv[:, :512]) * K_SCALE
        m[part], total[part] = top, weights.sum(-1)
    gathered = []
    for t in (o, m, total):
        buf = torch.empty((world, *t.shape), dtype=t.dtype, device=t.device)
        dist.all_gather_into_tensor(buf, t.contiguous(), group=group)
        gathered.append(buf)
    os, ms, sums = gathered  # (os: each rank's unnormalized sum of weighted rows)
    top = ms.amax(0)
    weights = torch.exp(ms - top.clamp_min(-1e30)[None])
    weights = torch.where(sums > 0, weights, torch.zeros_like(weights))
    merged = (os * weights[..., None]).sum(0) / (weights * sums).sum(0)[..., None]
    per = heads // world
    mine = merged[:, rank * per : (rank + 1) * per]
    result = torch.einsum("rhl,hlv->rhv", mine, w_uv.float())
    return result.reshape(rows, per * 128)


def _q_ideal(layer, qkv, staged, positions=None, rope_bound=False):
    """This rank's q rows [rows, heads, 576] as FP8 bytes, in FP64 from the
    in-projection rows: RMSNorm, Q_B, W_UK (nope), GPT-J RoPE at ``positions`` (pe);
    ``staged``: rounded to BF16 after each stage as stock's kernels store them (the
    rope rotates the BF16 Q_B output), else only at the FP8 store. ``rope_bound``:
    also the scaled FP64 rope values [rows, heads, 64] and their inputs' magnitude
    |x||cos| + |y||sin| (per output; see _rope_violations)."""
    rows, heads = qkv.shape[0], layer.num_local_heads
    f64 = torch.float64
    q_c = qkv[:, :1536].to(f64)
    variance = (q_c * q_c).mean(-1, keepdim=True)
    q_c = q_c * torch.rsqrt(variance + layer.rms_norm_eps)
    q_c = q_c * layer.q_a_layernorm.weight.to(f64)
    if staged:
        q_c = q_c.bfloat16().to(f64)
    q = q_c @ layer.q_b_proj.weight.to(f64).T
    if staged:
        q = q.bfloat16().to(f64)
    q = q.view(rows, heads, 192)
    nope = torch.einsum("rhp,hpl->rhl", q[..., :128], layer.W_UK_T.to(f64))
    if staged:
        nope = nope.bfloat16().to(f64)
    cos_sin = layer.rotary_emb.cos_sin_cache[positions].to(f64)[:, None]
    cos, sin = cos_sin[..., :32], cos_sin[..., 32:]
    x, y = q[..., 128::2], q[..., 129::2]
    pe = torch.stack((x * cos - y * sin, y * cos + x * sin), dim=-1).flatten(-2)
    out = torch.cat((nope, pe), dim=-1) * layer._q_scale_inv.to(f64)
    q_bytes = out.float().to(torch.float8_e4m3fn).view(torch.uint8)
    if not rope_bound:
        return q_bytes
    ax, ay, ac, as_ = x.abs(), y.abs(), cos.abs(), sin.abs()
    mag = torch.stack((ax * ac + ay * as_, ay * ac + ax * as_), dim=-1).flatten(-2)
    return q_bytes, out[..., 512:], mag * layer._q_scale_inv.to(f64)


def _rope_violations(q_bytes, value, mag):
    """How many FP8 rope bytes lie further from the FP64 rope ``value`` than one FP8
    spacing (the store's rounding) plus 2^-6 of their inputs' magnitude ``mag``: a
    rotation of BF16 Q_B values up to one BF16 ulp (2^-7 relative) from the staged
    ones, as a Q_B accumulated in another order rounds, moves the result by up to
    2^-7 mag, which is many FP8 spacings where x cos - y sin cancels. A wrong
    position, head, row or scale moves bytes by about mag itself."""
    got = q_bytes.view(torch.float8_e4m3fn).double()
    big = torch.maximum(got.abs(), value.abs())
    spacing = torch.exp2(torch.floor(torch.log2(big.clamp_min(2.0**-6))) - 3)
    return int(((got - value).abs() > spacing + mag * 2.0**-6).sum())


def _fp8_ulps(a, b):
    """Largest |a - b| of two E4M3 byte tensors in units of the FP8 spacing at the
    larger magnitude (subnormal spacing 2^-9 at the bottom)."""
    x, y = (t.view(torch.float8_e4m3fn).float() for t in (a, b))
    big = torch.maximum(x.abs(), y.abs())
    exponent = torch.floor(torch.log2(big.clamp_min(2.0**-6)))
    spacing = torch.exp2(exponent - 3)
    return float(((x - y).abs() / spacing).max()) if x.numel() else 0.0


class _DraftAttention:
    """What DraftDcp reads of a drafter MLA layer (no RoPE: MlaQuery is not built,
    the check fills the q mailboxes itself); ``impl.forward_mqa``: the stock
    tokenspeed decode (the gathered rows without a fold)."""

    def __init__(self, gen, device, cache, world, rank, interleave, heads=DCP_HEADS):
        self.num_local_heads = heads
        # W_UV [H, 512, 128] latent-contiguous, as the model's (split_kv_b_proj: the
        # kv_b_proj weight's .T view [L, N, V], then (L, N, V) -> (N, L, V)): DcpValue
        # loads its MMA fragments as 16 B runs of latent values (mla.dcp
        # uv_layout_error). (As many draws from ``gen`` as before.)
        w_uv = torch.randn((heads * 128, 512), generator=gen, device="cpu") * 0.05
        w_uv = w_uv.bfloat16().to(device).T.view(512, heads, 128)
        self.W_UV = w_uv.transpose(0, 1)
        self.scale = SOFTMAX_SCALE
        self._q_scale_float, self._k_scale_float = Q_SCALE, K_SCALE
        self.rotary_emb = None
        self.kv_cache = cache
        cp = world, rank, interleave
        self.impl = SimpleNamespace(
            forward_mqa=lambda q, kv, meta, layer: _tokenspeed(
                q, kv, meta, meta.causal, *cp
            )
        )

    def _attn_read_kv_cache(self):
        return self.kv_cache


def _dcp_path(shared, rows):
    if rows in shared.forks:
        return "fork"
    if rows in shared.row_modes:
        return "rows"
    if rows in shared.wide_cores:
        return "wide"
    if rows in shared.long_polled:
        return "long_polled"
    if rows in shared.polled:
        return "polled"
    if rows in shared.folds:
        return "fold"
    return "stock core"


def _dcp_case(steps, rows_list, causal, world, rank, group, device, interleave,
              label, seen, long_rows=(), heads_rank=DCP_HEADS):  # fmt: skip
    """One DraftDcp over ``rows_list`` drafter rows (q = ``steps``): per rows, two
    layers (parities 0 and 1) twice, each against the stock output."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.attention import (
        DraftDcp,
    )

    gen = torch.Generator().manual_seed(77 + steps + 2 * causal + rank * 1000)
    shared_gen = torch.Generator().manual_seed(4242 + steps + 2 * causal)
    largest = max(rows_list)
    inputs = world, rank, interleave, shared_gen, gen, device
    cache, _, _ = _dcp_inputs(1, steps, *inputs, heads_rank=heads_rank)
    layers = [
        _DraftAttention(gen, device, cache, world, rank, interleave, heads_rank)
        for _ in (0, 1)
    ]
    shared = DraftDcp(
        layers, group, world, rank, rows_list, steps, 1 << 20, interleave, causal
    )
    failures = 0
    for rows in rows_list:
        reqs = rows // steps
        long = 2 if rows >= 6 * steps or rows in long_rows else 0
        cache, q, metadata = _dcp_inputs(
            reqs, steps, world, rank, interleave, shared_gen, gen, device, long,
            heads_rank,
        )  # fmt: skip
        metadata.causal = causal
        path = _dcp_path(shared, rows)
        seen.setdefault(path, []).append((label, rows))
        if rows in shared.dropped:
            print(f"rank {rank}: {label} rows {rows}: {shared.dropped[rows]}")
            continue
        for layer in layers:
            layer.kv_cache = cache
            layer.draft_dcp.bind(cache)
        torch.accelerator.synchronize(device)
        expected = [
            _stock_dcp(q, cache, metadata, causal, layer.W_UV, world, rank,
                       interleave, group)
            for layer in layers
        ]  # fmt: skip
        # The check sees the mask: the other one differs. (Not in rows mode, whose
        # metadata carries the mask as each row's own length, so the stock call with
        # the other flag computes the same rows.)
        if rows == largest and "rows mode" not in label:
            other = _stock_dcp(q, cache, metadata, not causal, layers[0].W_UV, world,
                               rank, interleave, group)  # fmt: skip
            if _close(other, expected[0]):
                print(f"rank {rank}: {label} rows {rows}: the mask is not observable")
                failures += 1
        padded = shared.row_modes.get(rows, (rows,))[0]
        heads = world * heads_rank
        for step in range(4):
            index = step % 2
            layer = layers[index]
            dcp = layer.draft_dcp
            if rows in shared.wide_cores:
                mailbox = shared.wide_box.mailbox(True)
            else:
                mailbox = shared.box.mailbox(rows in shared.valid)
            mailbox[dcp.parity, :rows, :heads].copy_(q)
            out = torch.empty(
                (padded, heads_rank * 128), dtype=torch.bfloat16, device=device
            )
            dcp.merge(layer, metadata, rows, out, rows_mode=rows in shared.row_modes)
            torch.accelerator.synchronize(device)
            if not _close(out[:rows], expected[index]):
                error = (out[:rows].float() - expected[index].float()).abs().max()
                print(
                    f"rank {rank}: {label} rows {rows} ({path}) layer {index} call "
                    f"{step}: max error {error.item():.4g}"
                )
                failures += 1
    return failures


def _close(out, expected):
    """Per row: max |out - stock| within 2% of the row's largest stock value (+1e-3):
    the summation order and the BF16 rounding of the stock per-rank rows differ; a
    mask or partition error moves the short requests' rows by far more."""
    out, expected = out.float(), expected.float()
    error = (out - expected).abs().amax(dim=-1)
    scale = expected.abs().amax(dim=-1)
    return bool(torch.isfinite(out).all() and (error <= 0.02 * scale + 1e-3).all())


def _dcp_main(rank, world, device, group, interleave=1):
    """DraftDcp (the drafter's fast DCP attention, 6 heads per rank) against the stock
    tokenspeed + cp_lse_ag_out_rs + W_UV, non-causal (what both K3 DSpark drafters run)
    and causal, at DL3 and DL7 rows covering every core: the polled
    fork (rows mode, FORK_ROWS), rows mode on the gathered q = 1 core (FORK_ROWS off),
    the gathered fold core, the polled core (LSU / validity), the polled core of a
    gathered fold row with a long split (``long_polled``; a LONG_SPLIT rule installed
    for that row only, in this process), and the wide rows above the box (Rubin)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    heads = world * DCP_HEADS
    dl3 = [3, 6, 12, 24, 36, 45, 60, 96, 192, 288]
    dl7 = [7, 14, 28, 42, 56, 63, 112, 224, 448]
    # long_polled: a gathered fold row (FAST_CORE, not POLLED_CORE, split > 1) that
    # no other case compiles (the compiled cores are cached without the rule).
    max_seq = dcp.local_max(1 << 20, world, interleave)
    long_row = None
    for rows in (39, 51, 54, 57):
        split = attention.draft_fold_splits(rows, 3, heads, max_seq, device, world)[0]
        polled = attention.POLLED_CORE.get(rows) or attention.ROWS_MODE.get(rows)
        if split > 1 and not polled and (rows, heads) not in dcp.LONG_SPLIT:
            long_row = rows
            dcp.LONG_SPLIT[rows, heads] = (1, split, 4)  # long past 4 local tiles
            break
    seen: dict = {}
    failures = 0
    for causal in (True, False):
        mask = "causal" if causal else "non-causal"
        args = world, rank, torch.distributed.group.WORLD, device, interleave
        failures += _dcp_case(3, dl3, causal, *args, f"DL3 {mask}", seen)
        failures += _dcp_case(7, dl7, causal, *args, f"DL7 {mask}", seen)
        fork_rows = attention.FORK_ROWS
        attention.FORK_ROWS = {}  # (rows mode on the gathered q = 1 core)
        try:
            for steps, rows in ((3, [3, 6]), (7, [7])):
                label = f"DL{steps} {mask} rows mode"
                failures += _dcp_case(steps, rows, causal, *args, label, seen)
        finally:
            attention.FORK_ROWS = fork_rows
        if long_row is not None:
            label = f"DL3 {mask} long_polled"
            failures += _dcp_case(
                3, [long_row], causal, *args, label, seen, long_rows=(long_row,)
            )
    if rank == 0:
        for path, cases in sorted(seen.items()):
            print(f"DraftDcp {path}: {cases}")
    wanted = {"fork", "rows", "fold", "polled", "long_polled", "wide"}
    missing = wanted - set(seen)
    if missing:
        print(f"rank {rank}: DraftDcp paths not exercised: {sorted(missing)}")
        failures += 1
    return failures


def _dcp96_main(rank, world, device, group, interleave=1):
    """DraftDcp at the production gathered head count (96 = world x 96 / world heads
    per rank: 24 at 4 GPUs, C512's TP4 = DCP4 shape) against the stock path, causal
    (kimi-k3.1-pro-dspark-preview), with long requests past 64 local tiles: the
    packed-twin split rules (mla.dcp PACKED_TWIN) at DL3 / DL7 fold rows (the former
    stock-core seam rows 54 / 57 / 63 included) and wide rows. At 24 heads per rank
    (MlaQuery publishes only) every fold row runs the polled core and no row runs the
    stock core or a gathered one."""
    if 96 % world:
        print(f"rank {rank}: 96 heads do not split over {world} ranks")
        return 0
    heads_rank = 96 // world
    seen: dict = {}
    args = world, rank, torch.distributed.group.WORLD, device, interleave
    dl3 = [3, 6, 9, 12, 24, 45, 54, 57, 63, 96, 192, 288]
    dl7 = [7, 14, 28, 42, 56, 63, 112, 224, 448]
    failures = _dcp_case(3, dl3, True, *args, "DL3 96h causal", seen,
                         heads_rank=heads_rank)  # fmt: skip
    failures += _dcp_case(7, dl7, True, *args, "DL7 96h causal", seen,
                          heads_rank=heads_rank)  # fmt: skip
    if rank == 0:
        for path, cases in sorted(seen.items()):
            print(f"DraftDcp 96h {path}: {cases}")
    bad = {"stock core", "fold"} & set(seen) if heads_rank == 24 else set()
    bad |= {"stock core"} & set(seen)
    missing = {"fork", "wide"} - set(seen)
    if not ({"polled", "long_polled"} & set(seen)):
        missing.add("polled")
    if bad or missing:
        print(f"rank {rank}: DraftDcp 96h paths: unexpected {bad}, missing {missing}")
        failures += 1
    return failures


def _rope_cache(positions, device):
    """[positions, 64] FP32 cos | sin of a 64-wide rope (base 10000)."""
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.arange(positions, dtype=torch.float32)[:, None] * inv[None]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


class _QueryAttention(_DraftAttention):
    """A drafter MLA layer with its query weights (DraftDcp compiles MlaQuery with
    RoPE for it): the norms and q_a output are replicated, q_b / kv_b (W_UK_T, W_UV
    as the model absorbs them) are this rank's heads."""

    def __init__(self, gen, shared_gen, device, cache, world, rank, interleave, heads,
                 cos_sin):  # fmt: skip
        super().__init__(gen, device, cache, world, rank, interleave, heads)
        bf16 = torch.bfloat16
        kv_b = torch.randn((heads * 256, 512), generator=gen) / 16
        kv_b = kv_b.to(bf16).to(device)
        split = kv_b.T.view(512, heads, 256).split([128, 128], dim=-1)
        self.W_UK_T = split[0].permute(1, 2, 0)  # (L, N, P) -> (N, P, L)
        self.W_UV = split[1].transpose(0, 1)  # (L, N, V) -> (N, L, V)
        q_b = torch.randn((heads * 192, 1536), generator=gen) / 40
        self.q_b_proj = SimpleNamespace(weight=q_b.to(bf16).to(device))
        norms = [torch.rand(n, generator=shared_gen) + 0.5 for n in (1536, 512)]
        self.q_a_layernorm = SimpleNamespace(weight=norms[0].to(bf16).to(device))
        self.kv_a_layernorm = SimpleNamespace(weight=norms[1].to(bf16).to(device))
        f32 = dict(dtype=torch.float32, device=device)
        self._q_scale_inv = torch.full((1,), 1 / Q_SCALE, **f32)
        self._k_scale_inv = torch.full((1,), 1 / K_SCALE, **f32)
        self.rms_norm_eps = 1e-6
        self.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)


def _draft_slots(metadata, steps, world, rank, interleave):
    """The draft rows' KV slots on this rank (-1 where another rank owns the token)
    and their global positions: request r's rows are its last ``steps`` tokens."""
    decode = metadata.decode
    totals = decode.dcp_tot_seq_lens.tolist()
    table = decode.block_table.cpu()
    slots, positions = [], []
    for r, total in enumerate(totals):
        for t in range(total - steps, total):
            positions.append(t)
            if (t // interleave) % world != rank:
                slots.append(-1)
                continue
            local = (t // (interleave * world)) * interleave + t % interleave
            slots.append(int(table[r, local // PAGE]) * PAGE + local % PAGE)
    i64 = dict(dtype=torch.int64, device=decode.block_table.device)
    return torch.tensor(slots, **i64), torch.tensor(positions, **i64)


def _stock_query(layer, qkv, cache, slots, positions, world, group):
    """The stock drafter query (fused q / kv RMSNorm, q_b_proj, W_UK bmm, the fused
    RoPE + FP8 concat and KV insert into ``cache``), all-gathered over the ranks'
    heads: [rows, world x heads, 576] FP8 bytes."""
    import torch.distributed as dist

    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
        fused_mla_decode_q_concat_kv_cache_insert,
    )

    rows, heads = qkv.shape[0], layer.num_local_heads
    q_c, kv_c, k_pe = qkv.split([1536, 512, 64], dim=-1)
    q_c, kv_c = fused_q_kv_rmsnorm(
        q_c, kv_c, layer.q_a_layernorm.weight, layer.kv_a_layernorm.weight,
        layer.rms_norm_eps,
    )  # fmt: skip
    q = (q_c @ layer.q_b_proj.weight.T).view(rows, heads, 192)
    nope = torch.bmm(q[..., :128].transpose(0, 1), layer.W_UK_T).transpose(0, 1)
    mqa_q = fused_mla_decode_q_concat_kv_cache_insert(
        nope, q[..., 128:], kv_c, k_pe.unsqueeze(1), cache, slots,
        q_scale_inv=layer._q_scale_inv, cache_scale_inv=layer._k_scale_inv,
        positions=positions, cos_sin_cache=layer.rotary_emb.cos_sin_cache,
    )  # fmt: skip
    mine = mqa_q.view(torch.uint8).contiguous()
    every = torch.empty((world, *mine.shape), dtype=torch.uint8, device=mine.device)
    dist.all_gather_into_tensor(every, mine, group=group)
    return every.permute(1, 0, 2, 3).reshape(rows, world * heads, 576)


def _query96_main(rank, world, device, group, interleave=1):
    """The drafter's whole fast attention at 96 gathered heads (24 per rank at 4
    GPUs: C512's TP4 = DCP4 layer) from the in-projection rows, as
    ``DraftLayerDcp.attend_projected`` runs it: MlaQuery (RoPE, FP8, KV insert,
    publish-only q) -> the fork / polled / wide core -> DcpValue, against the stock
    query + KV insert -> tokenspeed -> LSE combine -> W_UV, causal, with long
    requests; and the KV rows both paths inserted."""
    from cutlass import Int32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.attention import (  # noqa: E501
        DraftDcp,
        query_heads,
    )

    heads = 96 // world if 96 % world == 0 else 0
    if not heads or not query_heads(heads):
        print(f"rank {rank}: no drafter MlaQuery tile at 96 / {world} heads")
        return 0
    failures, seen, worst = 0, {}, 0.0
    accuracy: dict = {}  # (steps, rows) -> largest (specialized, stock) error
    q_ulps = [0.0] * 4  # (specialized, stock) nope, then rope columns
    cos_sin = _rope_cache(1 << 16, device)
    for steps, rows_list in ((3, [3, 6, 9, 24, 54, 63, 96, 120, 168]),
                             (7, [7, 14, 63, 112])):  # fmt: skip
        gen = torch.Generator().manual_seed(91 + steps + rank * 1000)
        shared_gen = torch.Generator().manual_seed(5151 + steps)
        cache0, _, _ = _dcp_inputs(1, steps, world, rank, interleave, shared_gen, gen,
                                   device, heads_rank=heads)  # fmt: skip
        layers = [
            _QueryAttention(
                gen, shared_gen, device, cache0, world, rank, interleave, heads, cos_sin
            )  # fmt: skip
            for _ in (0, 1)
        ]
        shared = DraftDcp(
            layers, group, world, rank, rows_list, steps, 1 << 20, interleave, True
        )
        for rows in rows_list:
            reqs = rows // steps
            label = f"DL{steps} 96h query rows {rows}"
            if not shared.served(rows) or rows not in shared.queries:
                print(f"rank {rank}: {label}: not served ({shared.dropped.get(rows)})")
                failures += 1
                continue
            long = 2 if reqs >= 6 else 0
            cache, _, metadata = _dcp_inputs(
                reqs, steps, world, rank, interleave, shared_gen, gen, device, long,
                heads,
            )  # fmt: skip
            metadata.causal = True
            slots, positions = _draft_slots(metadata, steps, world, rank, interleave)
            qkv = torch.randn((rows, 2112), generator=shared_gen) * 2
            qkv = qkv.to(torch.bfloat16).to(device)
            seen.setdefault(_dcp_path(shared, rows), []).append(rows)
            for step in range(4):
                index = step % 2
                layer, dcp = layers[index], layers[index].draft_dcp
                fast_cache, stock_cache = cache.clone(), cache.clone()
                layer.kv_cache = stock_cache
                q = _stock_query(layer, qkv, stock_cache, slots, positions, world,
                                 group)  # fmt: skip
                expected = _stock_dcp(q, stock_cache, metadata, True, layer.W_UV,
                                      world, rank, interleave, group)  # fmt: skip
                layer.kv_cache = fast_cache
                for each in layers:
                    each.draft_dcp.bind(fast_cache)
                torch.accelerator.synchronize(device)
                torch.distributed.barrier()
                qnorm, kvnorm, qb, uk, qinv, kinv = dcp.query_weights
                p = dcp.parity
                shared.queries[rows](
                    to_cute(qkv), qnorm, kvnorm, qb, uk,
                    dynamic_view(fast_cache.view(torch.uint8)), to_cute(slots, 8),
                    qinv, kinv, dcp.eps, to_cute(dcp.scratch[:rows]),
                    *shared.mailbox(rows, p), Int32(rank), to_cute(positions, 8),
                    dcp.cos_sin, stream(device),
                )  # fmt: skip
                torch.accelerator.synchronize(device)
                torch.distributed.barrier()
                # The published q rows (every rank's heads, multicast) before the core
                # reads and re-arms them.
                fast_q = _query_mailbox(shared, rows, p)[:rows, : world * heads]
                fast_q = fast_q.clone()
                padded = shared.row_modes.get(rows, (rows,))[0]
                out = torch.empty((padded, heads * 128), dtype=torch.bfloat16,
                                  device=device)  # fmt: skip
                dcp.merge(layer, metadata, rows, out, rows_mode=True)
                torch.accelerator.synchronize(device)
                torch.distributed.barrier()
                # Stage by stage. (1) The inserted KV bytes: bitwise stock's. (2) The
                # FP8 q bytes: each within stock's own distance (+1 FP8 ulp: the BF16
                # accumulation order of Q_B / W_UK) of the FP64 query rounded to BF16
                # where stock's kernels store it; a wrong head, row, position or scale
                # moves many bytes far. (3) The attention on those q / KV bytes: no
                # further from the exact FP32 attention (no intermediate rounding)
                # than stock's tokenspeed + LSE combine + W_UV on the same bytes, per
                # row over the calls and ranks. Both round P to FP8 in the core, ~6%
                # of the row's largest value from exact where a few new draft tokens
                # dominate the softmax, so they are not within _close of each other.
                live = slots[slots >= 0]
                ours, ref = (
                    c.view(torch.uint8)[live // PAGE, live % PAGE]
                    for c in (fast_cache, stock_cache)
                )
                if not torch.equal(ours, ref):
                    print(
                        f"rank {rank}: {label} call {step}: KV bytes differ from "
                        f"stock's insert ({_fp8_ulps(ours, ref):.2f} FP8 ulps)"
                    )
                    failures += 1
                mine = slice(rank * heads, (rank + 1) * heads)
                ideal, rope, mag = _q_ideal(layer, qkv, True, positions, True)
                for part in (slice(0, 512), slice(512, 576)):  # (nope, rope)
                    spec_ulps = _fp8_ulps(fast_q[:, mine, part], ideal[..., part])
                    stock_ulps = _fp8_ulps(q[:, mine, part], ideal[..., part])
                    which = 0 if part.start == 0 else 2
                    q_ulps[which] = max(q_ulps[which], spec_ulps)
                    q_ulps[which + 1] = max(q_ulps[which + 1], stock_ulps)
                    if part.start == 0 and spec_ulps > Q_ULPS_LIMIT:
                        print(f"rank {rank}: {label} call {step}: q nope columns: "
                              f"{spec_ulps:.2f} FP8 ulps from the staged FP64 query "
                              f"(stock {stock_ulps:.2f})")  # fmt: skip
                        failures += 1
                # The rope columns: within one FP8 spacing + 2^-6 of the rotation's
                # input magnitude of the FP64 rope (stock must hold the same bound).
                rope_bad = [
                    _rope_violations(x[:, mine, 512:], rope, mag) for x in (fast_q, q)
                ]
                if any(rope_bad):
                    print(f"rank {rank}: {label} call {step}: rope bytes past the "
                          f"rotation bound: specialized {rope_bad[0]}, stock "
                          f"{rope_bad[1]}")  # fmt: skip
                    failures += 1
                flipped = (fast_q != q).float().mean().item()
                if flipped > FP8_FLIP_LIMIT:
                    print(
                        f"rank {rank}: {label} call {step}: {flipped:.3%} of the q "
                        "bytes differ from stock's"
                    )
                    failures += 1
                same = _stock_dcp(fast_q, fast_cache, metadata, True, layer.W_UV,
                                  world, rank, interleave, group)  # fmt: skip
                exact = _exact_dcp(fast_q, fast_cache, metadata, layer.W_UV, world,
                                   rank, interleave, group)  # fmt: skip
                scale = exact.abs().amax(-1, keepdim=True)
                errors = [((x[:rows].float() - exact).abs() / scale).max().item()
                          for x in (out, same)]  # fmt: skip
                row = accuracy.setdefault((steps, rows), [0.0, 0.0])
                row[0], row[1] = max(row[0], errors[0]), max(row[1], errors[1])
                if not torch.isfinite(out[:rows]).all():
                    print(f"rank {rank}: {label} call {step}: non-finite rows")
                    failures += 1
                if not _close(out[:rows], expected):
                    error = (out[:rows].float() - expected.float()).abs().max()
                    worst = max(worst, error.item())
                untouched = torch.ones(cache.shape[:2], dtype=torch.bool,
                                       device=device)  # fmt: skip
                untouched[live // PAGE, live % PAGE] = False
                if not torch.equal(fast_cache.view(torch.uint8)[untouched],
                                   cache.view(torch.uint8)[untouched]):  # fmt: skip
                    print(f"rank {rank}: {label}: KV written outside its slots")
                    failures += 1
        del layers, shared
        torch.cuda.empty_cache()
    # The nope q bytes: no further from the staged FP64 query than stock's (+1 FP8
    # ulp), over every call and rank. (The rope columns' FP8 ulps are reported only:
    # where x cos - y sin cancels, one BF16 ulp of Q_B is several FP8 ulps of the
    # result for both paths; their check is _rope_violations.)
    ulps = torch.tensor(q_ulps, device=device)
    torch.distributed.all_reduce(ulps, op=torch.distributed.ReduceOp.MAX)
    q_ulps = ulps.tolist()
    if q_ulps[0] > q_ulps[1] + 1.0:
        print(
            f"rank {rank}: q nope columns {q_ulps[0]:.2f} FP8 ulps from the staged "
            f"FP64 query, stock {q_ulps[1]:.2f}"
        )
        failures += 1
    for (steps, rows), errors in sorted(accuracy.items()):
        both = torch.tensor(errors, device=device)
        torch.distributed.all_reduce(both, op=torch.distributed.ReduceOp.MAX)
        ours, stock = both.tolist()
        if rank == 0:
            print(
                f"DL{steps} 96h query rows {rows}: vs exact FP32, largest error per "
                f"row max: specialized {ours:.4f}, stock {stock:.4f}"
            )
        if ours > ACCURACY_SLACK * stock + 2e-3:
            print(
                f"rank {rank}: DL{steps} 96h query rows {rows}: specialized "
                f"{ours:.4f} vs exact, stock {stock:.4f}"
            )
            failures += 1
    if rank == 0:
        print(f"DraftDcp 96h query paths: {sorted(seen.items())}")
    print(
        f"rank {rank}: q FP8 ulps from the staged FP64 query (nope / rope): "
        f"specialized {q_ulps[0]:.2f} / {q_ulps[2]:.2f}, stock {q_ulps[1]:.2f} / "
        f"{q_ulps[3]:.2f}; largest end-to-end error past _close vs stock: {worst:.4g}"
    )
    return failures


def _c512_rows_main(rank, world, device, group, interleave=1):
    """C512's drafter attention at its whole captured row range: TP4 = DCP4 (24
    heads per rank, MlaQuery publish-only), DL3, max-num-seqs 128 per DP rank ->
    drafter rows 3..384, max-model-len 2^20, as runtime.draft builds DraftDcp there
    (FAST_DCP[4] = every row, strict). Every row must be served (runtime.draft
    refuses the load otherwise), have its fused query (a served row without one
    reaches ``merge``'s ungathered-q refusal at capture) and run a fold / polled /
    wide / fork core (stock_core_rows is refused)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.attention import (  # noqa: E501
        DraftDcp,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    if world != 4:
        print(f"rank {rank}: C512's drafter is DCP4, not {world}")
        return 0
    steps, heads = 3, 96 // world
    spec = draft.closed(draft.FAST_DCP.get(4, ""), 384)
    rows = [r for r in range(steps, 128 * steps + 1, steps)
            if draft.covered(spec, r, steps)]  # fmt: skip
    failures = 0
    if len(rows) != 128:
        print(f"rank {rank}: FAST_DCP[4] lists {len(rows)} of C512's 128 rows")
        failures += 1
    gen = torch.Generator().manual_seed(4512 + rank)
    shared_gen = torch.Generator().manual_seed(4512)
    cache0, _, _ = _dcp_inputs(1, steps, world, rank, interleave, shared_gen, gen,
                               device, heads_rank=heads)  # fmt: skip
    layers = [
        _QueryAttention(
            gen,
            shared_gen,
            device,
            cache0,
            world,
            rank,
            interleave,
            heads,
            _rope_cache(1 << 16, device),
        )  # fmt: skip
        for _ in (0, 1)
    ]
    shared = DraftDcp(layers, group, world, rank, rows, steps, 1 << 20, interleave,
                      True)  # fmt: skip
    unserved = {r: shared.dropped.get(r, "not covered")
                for r in rows if not shared.served(r)}  # fmt: skip
    if unserved:
        print(f"rank {rank}: C512 drafter rows not served: {unserved}")
        failures += 1
    no_query = [r for r in rows if shared.served(r) and r not in shared.queries]
    if no_query:
        print(f"rank {rank}: C512 served rows without the fused query: {no_query}")
        failures += 1
    seams = shared.stock_core_rows()
    if seams:
        print(f"rank {rank}: C512 rows on the stock tokenspeed core: {seams}")
        failures += 1
    if rank == 0:
        print(f"C512 drafter coverage: {shared.coverage(rows)}")
    del layers, shared
    torch.cuda.empty_cache()
    return failures


# Any one call's nope q bytes further than this from the staged FP64 query (FP8 ulps
# at the larger magnitude) are wrong (a head, row or scale error moves them far): both
# the fused query and stock land up to 3 away (W_UK over BF16 Q_B outputs one BF16 ulp
# apart).
Q_ULPS_LIMIT = 4.0
# The specialized drafter attention's largest error against the exact FP32 attention
# (per row, over calls and ranks) may exceed stock's by this factor: the same FP8
# roundings (P in the core, the BF16 partial rows) in another order land elsewhere.
ACCURACY_SLACK = 1.25


# Fraction of the fused query's FP8 bytes allowed to differ from stock's (measured
# at most 0.11%: order-of-accumulation flips at FP8 rounding boundaries).
FP8_FLIP_LIMIT = 0.02


def _query_mailbox(shared, rows, parity):
    """The q mailbox ``rows`` of DraftDcp read at ``parity`` (torch, FP8 bytes)."""
    if rows in shared.wide_cores:
        return shared.wide_box.qv[parity]
    box = shared.box
    return (box.qv if rows in shared.valid else box.q)[parity]


# Context rows (requests x q) through and past one wave of CtxGather's 256-thread CTAs
# (VR200: 212 resident): the q8 / q4 production rows 216..512 and arbitrary ones.
CONTEXT_ROWS = (72, 128, 208, 212, 216, 224, 256, 424, 512, 768, 1024, 1032)
# Query rows (requests x draft length) of the wide ArNormRows push sites, through and
# past one wave of their 1024-thread CTAs.
AR_ROWS = (72, 128, 217, 224, 384, 427, 512, 1024)
CTX_EPS = 1e-6


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _norm_close(out, x, gamma, eps):
    """RMSNorm reference (FP32, exact rsqrt): within 1 BF16 ulp of the kernel's
    rsqrt.approx / summation order."""
    x = x.float()
    variance = (x * x).sum(-1, keepdim=True) / x.shape[-1]
    ref = (x * torch.rsqrt(variance + eps) * gamma.float()).bfloat16().float()
    error = (out.float() - ref).abs()
    return bool((error <= ref.abs() * 2.0**-7 + 1e-30).all())


def _barrier(device):
    torch.accelerator.synchronize(device)
    torch.distributed.barrier()


def _context_main(rank, world, device, group):
    """PreparedContext's wide row gather (CtxGather; the persistent one-wave gather
    above the resident rows) at CONTEXT_ROWS: bitwise the one-CTA-per-row kernel run
    over row chunks of at most 128 rows (each chunk one wave), the persistent kernel
    with forced small grids (1 and 37 CTAs: many rows per CTA) bitwise too, and the
    FP32 RMSNorm of the gathered sanitized rows within one ulp."""
    from cutlass import Int32, Int64

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import context

    hidden = context.HIDDEN
    if hidden % world or hidden // world // 8 > 256:
        print(f"rank {rank}: context gather needs TP 4, 8 or 16 (not {world})")
        return 0
    bf16 = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(77)
    gamma = (torch.rand(hidden, generator=gen, device=device) + 0.5).to(bf16)
    # A contiguous [7168, 5 x 7168] weight (514 MB): PreparedContext prepares its
    # shard's split-K projection where a tactic covers the rows (never run here; a
    # stride-0 expand would hand that preparation an out-of-bounds view).
    weight = torch.zeros((hidden, 5 * hidden), dtype=bf16, device=device)
    model = SimpleNamespace(
        context_proj=SimpleNamespace(weight=weight),
        context_norm=SimpleNamespace(weight=gamma, variance_epsilon=CTX_EPS),
    )
    prepared = context.PreparedContext(model, group, world, rank, CONTEXT_ROWS)
    shard = hidden // world
    if rank == 0:
        print(f"CtxGather persistent rows: {prepared.persistent}")
    failures = 0 if prepared.persistent else 1
    index = device.index
    per_row = context.compile_gather(None, shard, index)
    common = (Int32(rank), to_cute(gamma), prepared.eps)

    def chunked(local, out):
        rows = local.shape[0]
        for start in range(0, rows, 128):
            stop = min(rows, start + 128)
            per_row(
                dynamic_view(local[start:stop]),
                dynamic_view(prepared.buffer[start * hidden :]),
                Int64(prepared.mc + start * hidden * 2),
                *common,
                dynamic_view(out[start:stop]),
                stream(device),
            )
        _barrier(device)

    def forced(local, out, ctas):
        kernel = context.compile_gather(None, shard, index, ctas=ctas)
        kernel(
            dynamic_view(local),
            prepared.dynamic,
            Int64(prepared.mc),
            *common,
            dynamic_view(out),
            stream(device),
        )
        _barrier(device)

    for rows in CONTEXT_ROWS:
        g = torch.Generator(device=device).manual_seed(1000 * rows + rank)
        local = torch.randn((rows, shard), generator=g, device=device).to(bf16)
        local[::7, ::5] = -0.0  # sanitized to +0.0 by the publish
        out, ref = (
            torch.empty((rows, hidden), dtype=bf16, device=device) for _ in "ab"
        )
        prepared.gather(local, out)
        _barrier(device)
        chunked(local, ref)
        path = "persistent" if rows in prepared.persistent else "per row"
        label = f"context rows {rows} ({path})"
        if not _bits_equal(out, ref):
            print(f"rank {rank}: {label}: not bitwise the per-row gather")
            failures += 1
        shards = [torch.empty_like(local) for _ in range(world)]
        torch.distributed.all_gather(shards, local + 0.0)  # (-0.0 + 0.0 = +0.0)
        if not _norm_close(out, torch.cat(shards, dim=1), gamma, CTX_EPS):
            print(f"rank {rank}: {label}: RMSNorm off by more than one ulp")
            failures += 1
        if rows in (216, 1032):
            for ctas in (1, 37):
                forced_out = torch.empty_like(out)
                forced(local, forced_out, ctas)
                if not _bits_equal(forced_out, ref):
                    print(f"rank {rank}: {label}: persistent {ctas} CTAs differ")
                    failures += 1
    return failures


def _arnorm_main(rank, world, device, group):
    """The wide drafter all-reduce push sites (ArNormRows over the wide two-shot box;
    the persistent one-wave site above the resident rows, as PreparedDraftAr picks):
    the new residual bitwise the FP32 rank-order sum rounded once plus the residual,
    the normed rows bitwise the one-CTA-per-row site where that grid is one wave and
    the persistent site with forced small grids (1 and 37 CTAs), within one ulp of the
    FP32 RMSNorm everywhere; ``trigger`` (early) sites too."""
    from cutlass import Float32, Int32, Int64

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        resident_ctas,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import (
        TwoShot,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        allreduce,
    )

    hidden = allreduce.HIDDEN
    if hidden % world or (hidden // world) % 8:
        print(f"rank {rank}: two-shot all-reduce needs TP dividing 7168 / 8")
        return 0
    bf16, index = torch.bfloat16, device.index
    cap = -(-max(AR_ROWS) // 64) * 64
    box = TwoShot(group, device, world, rank, hidden, cap, 2)
    gen = torch.Generator(device=device).manual_seed(99)
    gamma = (torch.rand(hidden, generator=gen, device=device) + 0.5).to(bf16)
    owner = SimpleNamespace(device=device, tp=world, persistent={})
    parity = [0]

    def run(kernel, partial, residual):
        partial, out = partial.clone(), torch.empty_like(partial)
        recv, peers, gather, mc = box.views(parity[0])
        parity[0] ^= 1
        kernel(
            to_cute(partial), to_cute(residual), to_cute(gamma), Float32(CTX_EPS),
            to_cute(out), to_cute(recv), to_cute(peers, 8), to_cute(gather),
            Int64(mc), Int32(rank), stream(device),
        )  # fmt: skip
        _barrier(device)
        return partial, out  # (the new residual is written in place of the partial)

    def site(rows, trigger=False, ctas=None):
        return allreduce.compile_rows_site(
            rows, world, world, cap, index, trigger, True, ctas
        )

    auto = lambda rows: allreduce.PreparedDraftAr.persistent_site(  # noqa: E731
        owner, rows, cap, False
    )
    failures, persistent = 0, 0
    for rows in AR_ROWS:
        per_row = site(rows)
        resident = resident_ctas(per_row, index, allreduce.CLUSTER * 128)
        wide = rows > resident
        chosen = auto(rows) if wide else per_row  # (PreparedDraftAr's choice)
        persistent += wide
        g = torch.Generator(device=device).manual_seed(5000 * rows + rank)
        partial = torch.randn((rows, hidden), generator=g, device=device).to(bf16)
        partial[::5, ::3] = -0.0
        g = torch.Generator(device=device).manual_seed(5000 * rows)  # same on all
        residual = torch.randn((rows, hidden), generator=g, device=device).to(bf16)
        new_res, out = run(chosen, partial, residual)
        label = f"ArNormRows rows {rows} ({'persistent' if wide else 'per row'})"
        partials = [torch.empty_like(partial) for _ in range(world)]
        torch.distributed.all_gather(partials, partial)
        total = partials[0].float()
        for other in partials[1:]:
            total = total + other.float()
        expected = (total.bfloat16().float() + residual.float()).bfloat16()
        if not torch.equal(new_res.float(), expected.float()):
            print(f"rank {rank}: {label}: new residual not the FP32 rank-order sum")
            failures += 1
        if not _norm_close(out, expected, gamma, CTX_EPS):
            print(f"rank {rank}: {label}: RMSNorm off by more than one ulp")
            failures += 1
        reference = out if wide else run(per_row, partial, residual)[1]
        variants = [(c, site(rows, ctas=c)) for c in (1, 37)]
        if not wide:
            variants.append(("auto", auto(rows)))
        if rows == max(AR_ROWS):
            variants.append(("trigger", site(rows, trigger=True, ctas=37)))
        for name, kernel in variants:
            if not _bits_equal(run(kernel, partial, residual)[1], reference):
                print(f"rank {rank}: {label}: persistent {name} differs")
                failures += 1
    if not persistent:
        print(f"rank {rank}: no ArNormRows row above one wave")
        failures += 1
    return failures


def _cake_inputs(sets, steps, world, rank, interleave, gen, pages_gen, device, heads):
    """KV pages for every request at its longest total over the length ``sets`` (one
    list of global totals per set), the global FP8 query rows, and metadata at
    ``sets[0]`` (its tensors rewritten in place for the other sets: a shorter total
    reads a prefix of the same local pages)."""
    reqs, rows = len(sets[0]), len(sets[0]) * steps
    top = [max(s[r] for s in sets) for r in range(reqs)]
    q = torch.randn((rows, world * heads, 576), generator=gen, device="cpu") * 0.5
    q = q.to(torch.float8_e4m3fn).view(torch.uint8)
    q[q == 0x80] = 0  # (no chunk starts with the validity sentinel)
    local = [_local_len(t, world, rank, interleave) for t in top]
    pages = [max(1, -(-n // PAGE)) for n in local]
    blocks = sum(pages) + 1
    order = torch.randperm(blocks - 1, generator=pages_gen) + 1
    table = torch.zeros((reqs, max(pages)), dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[used : used + n]
        used += n
    table = table.to(device)
    cache = torch.zeros((blocks, PAGE, 576), dtype=torch.uint8, device=device)
    for r, total in enumerate(top):
        g = torch.Generator(device=device).manual_seed(9000 + r)
        t = torch.arange(total, device=device)
        mine = (t // interleave) % world == rank
        kv = torch.randn((int(mine.sum()), 576), generator=g, device=device) * 0.5
        kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        t = t[mine]
        position = (t // (interleave * world)) * interleave + t % interleave
        cache[table[r].long()[position // PAGE], position % PAGE] = kv
    i32 = dict(dtype=torch.int32, device=device)
    first = [_local_len(t, world, rank, interleave) for t in sets[0]]
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=table,
            seq_lens=torch.tensor(first, **i32),
            dcp_tot_seq_lens=torch.tensor(sets[0], **i32),
        ),
        num_decodes=reqs,
        num_decode_tokens=rows,
        num_prefills=0,
        max_seq_len=max(top),
        causal=True,
    )
    return cache.view(torch.float8_e4m3fn), q.to(device), metadata


def _cake_main(rank, world, device, group, interleave=1):
    """The drafter's cake rule (mla.dcp DRAFT_LONG_SPLIT: per-step quantum on the
    GPU, mla.wlist.StepQuant) at the production gathered heads (96, causal), C512's
    DCP4 rules (and the DCP16 rules mapped onto this DCP4 group, ``as16``): one CUDA
    graph per rows captured once, replayed over skewed (one 600k context), uniform
    (every request at the skewed batch's mean) and short (below the quantum floor)
    lengths rewritten in place. Each replay against the stock path, and the step
    quantum word against its host twin (mla.dcp.cake_quant) from this rank's local
    lengths: the partition follows the lengths of the replayed step."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import core_q, dcp

    if world != 4:
        print(f"rank {rank}: the cake check runs the 96-head drafter at DCP4")
        return 0
    heads, failures = 96 // world, 0
    saved = dict(dcp.DRAFT_LONG_SPLIT)
    for role in (4, 16):
        # (role 16: the DCP16 entries on this DCP4 group; the polled cores' compile
        # cache is keyed by shape, not rule: cleared per role)
        core_q._compile_polled_q.cache_clear()
        dcp.DRAFT_LONG_SPLIT.clear()
        dcp.DRAFT_LONG_SPLIT.update({
            (r, h, q, 4): rule for (r, h, q, size), rule in saved.items()
            if size == role
        })  # fmt: skip
        for steps, rows_list in ((3, [12, 54, 120]), (7, [63, 112])):
            gen = torch.Generator().manual_seed(6100 + steps + role)
            for rows in rows_list:
                failures += _cake_case(rows, steps, role, world, rank, group, device,
                                       interleave, heads, gen)  # fmt: skip
    dcp.DRAFT_LONG_SPLIT.clear()
    dcp.DRAFT_LONG_SPLIT.update(saved)
    core_q._compile_polled_q.cache_clear()
    return failures


def _cake_case(rows, steps, role, world, rank, group, device, interleave, heads, gen):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.attention import (  # noqa: E501
        DraftDcp,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    reqs = rows // steps
    rest = torch.randint(3000, 20000, (reqs - 1,), generator=gen).tolist()
    skewed = [600_000] + rest
    mean = sum(skewed) // reqs
    short = torch.randint(steps + 2, 4000, (reqs,), generator=gen).tolist()
    sets = {"skewed": skewed, "uniform": [mean] * reqs, "short": short}
    pages_gen = torch.Generator().manual_seed(77 + rank)
    inputs = world, rank, interleave, gen, pages_gen, device, heads
    cache, q, metadata = _cake_inputs(list(sets.values()), steps, *inputs)
    layers = [
        _DraftAttention(gen, device, cache, world, rank, interleave, heads)
        for _ in (0, 1)
    ]
    shared = DraftDcp(layers, group, world, rank, [rows], steps, 1 << 20, interleave,
                      True)  # fmt: skip
    label = f"cake DCP{role} rules q{steps} rows {rows}"
    quant = shared.quants.get(rows)
    if quant is None or not shared.served(rows):
        print(f"rank {rank}: {label}: no cake rule ({_dcp_path(shared, rows)})")
        return 1
    for layer in layers:
        layer.draft_dcp.bind(cache)
    total = world * heads
    outs = [torch.empty((rows, heads * 128), dtype=torch.bfloat16, device=device)
            for _ in layers]  # fmt: skip
    wide = rows in shared.wide_cores

    def step():
        for i, layer in enumerate(layers):
            dcp_ = layer.draft_dcp
            if wide:
                box = shared.wide_box.mailbox(True)
            else:
                box = shared.box.mailbox(rows in shared.valid)
            box[dcp_.parity, :rows, :total].copy_(q)
            dcp_.merge(layer, metadata, rows, outs[i])

    step()  # (warm: compiles and the first lists outside the capture)
    torch.accelerator.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.accelerator.synchronize(device)
    failures, quanta = 0, {}
    split = shared.folds.get(rows) or shared.wide_cores[rows][0]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    long = dcp.long_split(rows, total, split, sms, steps, draft=world)
    for name, totals in list(sets.items()) + [("skewed again", skewed)]:
        local = [_local_len(t, world, rank, interleave) for t in totals]
        metadata.decode.seq_lens.copy_(torch.tensor(local, dtype=torch.int32))
        metadata.decode.dcp_tot_seq_lens.copy_(torch.tensor(totals, dtype=torch.int32))
        metadata.max_seq_len = max(totals)
        graph.replay()
        torch.accelerator.synchronize(device)
        got = [o.clone() for o in outs]
        word = int(quant.buffer.item())
        want = dcp.cake_quant(local, split, long)
        quanta[name] = word
        if word != want:
            print(f"rank {rank}: {label} {name}: step quantum {word}, host {want}")
            failures += 1
        # Against the exact FP32 attention (no intermediate rounding), per row
        # relative to its largest value, as query96: no further than stock's
        # tokenspeed + LSE combine + W_UV on the same bytes (ACCURACY_SLACK). Both
        # round P to FP8 per KV split, so a different partition of a short context
        # (a few tokens dominating the softmax) lands up to a few % from stock's.
        errors = [0.0, 0.0]
        for i, layer in enumerate(layers):
            expected = _stock_dcp(q, cache, metadata, True, layer.W_UV, world, rank,
                                  interleave, group)  # fmt: skip
            exact = _exact_dcp(q, cache, metadata, layer.W_UV, world, rank,
                               interleave, group)  # fmt: skip
            scale = exact.abs().amax(-1, keepdim=True).clamp_min(1e-6)
            for j, x in enumerate((got[i], expected)):
                if not torch.isfinite(x).all():
                    errors[j] = float("inf")
                    continue
                err = ((x.float() - exact).abs() / scale).max().item()
                errors[j] = max(errors[j], err)
        both = torch.tensor(errors, device=device)
        torch.distributed.all_reduce(both, op=torch.distributed.ReduceOp.MAX)
        ours, stock = both.tolist()
        if not ours <= ACCURACY_SLACK * stock + 2e-3:
            print(f"rank {rank}: {label} {name}: vs exact FP32 specialized "
                  f"{ours:.4f}, stock {stock:.4f}")  # fmt: skip
            failures += 1
        elif rank == 0:
            print(f"{label} {name}: vs exact FP32 specialized {ours:.4f}, stock "
                  f"{stock:.4f}")  # fmt: skip
    if quanta["skewed"] == quanta["uniform"] == quanta["short"]:
        print(f"rank {rank}: {label}: the quantum never moved ({quanta})")
        failures += 1
    if rank == 0:
        print(f"{label} ({_dcp_path(shared, rows)}, rule {long}): quanta {quanta}")
    del graph
    return failures


def _gpu_main(parts):
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    failures = 0
    if "markov" in parts:
        failures += _markov_main(rank, world, device, group)
    if "dcp" in parts:
        failures += _dcp_main(rank, world, device, group)
    if "dcp96" in parts:
        failures += _dcp96_main(rank, world, device, group)
    if "query96" in parts:
        failures += _query96_main(rank, world, device, group)
    if "c512rows" in parts:
        failures += _c512_rows_main(rank, world, device, group)
    if "cake" in parts:
        failures += _cake_main(rank, world, device, group)
    if "context" in parts:
        failures += _context_main(rank, world, device, group)
    if "arnorm" in parts:
        failures += _arnorm_main(rank, world, device, group)
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    # torchrun ... test_specialized_decode_draft_nq.py [markov] [dcp] [dcp96]
    # [query96] [c512rows] [cake] [context] [arnorm] (default: all)
    parts = sys.argv[1:] or (
        "markov",
        "dcp",
        "dcp96",
        "query96",
        "c512rows",
        "cake",
        "context",
        "arnorm",
    )
    sys.exit(1 if _gpu_main(parts) else 0)
