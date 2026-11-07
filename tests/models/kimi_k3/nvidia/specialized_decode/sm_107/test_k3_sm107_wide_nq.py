# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_wide_nq.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3.1 specialized decode without a batch cap (CPU: plans, policies and grids,
no kernel compile or launch). Every captured FULL decode graph of a TP16 = DCP16
deployment, q4 (K=3) and q8 (K=7) up to 2048 token rows, runs the specialized decode:
no whole-step stock forward and no planner exception; the wide kernels' grids stay
within one wave where their CTAs poll each other (the compile-time residency checks
and the 16-rank timings need a Rubin GPU: test_k3_sm107_dsa_wide.py)."""

import importlib.util
from pathlib import Path as FilePath
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

# The installed vllm package's sm_107 sources (the tests tree may live elsewhere).
SD = (
    FilePath(importlib.util.find_spec("vllm").submodule_search_locations[0])
    / "models/kimi_k3/nvidia/specialized_decode/sm_107"
)


def load_policy():
    """cutedsl/common/policy.py on its own (no torch import)."""
    spec = importlib.util.spec_from_file_location(
        "sd_policy", SD / "cutedsl/common/policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_table_open_ranges():
    """``batch_table``'s open-ended ranges ("65-:1") cover every batch from their first
    on; closed entries after them override; iteration sees one batch past every closed
    range (so ``values`` holds every value an open range takes)."""
    batch_table = load_policy().batch_table
    closed = batch_table("1-2:96,3-8:120")
    assert dict(closed) == {1: 96, 2: 96, **dict.fromkeys(range(3, 9), 120)}
    assert closed.get(9) is None and 9 not in closed
    table = batch_table("65-:1")
    assert table.get(64) is None and 64 not in table
    assert all(table[b] == 1 and b in table for b in (65, 512, 2048, 1 << 20))
    widths = batch_table("65-80:8,81-:16")
    assert (widths[72], widths[81], widths[4096]) == (8, 16, 16)
    assert set(widths.values()) == {8, 16}
    later = batch_table("65-:16,65-80:8")
    assert (later[70], later[81], later[2048]) == (8, 16, 16)
    assert set(later.values()) == {8, 16}
    holes = batch_table("65-:1,129-256:3")
    assert (holes[100], holes[200], holes[257], holes[9999]) == (1, 3, 1, 1)
    formats = batch_table("1-16:bf16,17-:proj")
    assert (formats[16], formats[17], formats[2048]) == ("bf16", "proj", "proj")
    merged = batch_table("24:1,68-256:1")
    merged.update(batch_table("300-:2"))
    assert (merged[256], merged.get(299), merged[300], merged[5000]) == (1, None, 2, 2)
    with pytest.raises(ValueError):
        batch_table("1-64")
    # An open table has no finite set of batches: what would drop its open range
    # raises; values, bool, copies and the explicit materialized view keep it.
    drops = [dict, len, list, sorted, max, lambda t: {**t}, lambda t: t.items()]
    drops += [lambda t: t.keys(), lambda t: {} | t, lambda t: {}.update(t)]
    for drop in drops:
        with pytest.raises(TypeError, match="open-ended"):
            drop(table)
    assert table and not batch_table("") and set(table.values()) == {1}
    assert table.materialized() == {65: 1, 66: 1}
    import copy

    for twin in (copy.copy(table), copy.deepcopy(table), table.copy()):
        assert twin.open and twin[1 << 20] == 1 and twin.get(64) is None
    assert len(closed) == 8 and sorted(closed) == list(range(1, 9))


# ---- below: the specialized decode's modules (torch, cutlass; no GPU) ----

Q_ROWS = {4: range(68, 2049, 4), 8: range(72, 2049, 8)}


def captured(q, top=2048):
    """Every FULL decode graph a q-token deployment of max-num-seqs top / q captures
    (capture sizes 1..top: each rounded up to whole requests)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import captured_rows

    return captured_rows(range(1, top + 1), q, top)


def context(q, max_rows=2048, sms=212, bf16_state=False):
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import Context

    return Context(
        arch="rubin", tp=16, rank=0, dcp=16, sms=sms, q=q, max_rows=max_rows,
        max_context=1 << 20, bf16_state=bf16_state,
    )  # fmt: skip


@pytest.mark.parametrize("q", [4, 8])
def test_every_captured_row_is_wide(q):
    """``WIDE_ROWS`` is open-ended: every captured verify row above 64 (q4: 68-2048,
    q8: 72-2048) is a wide row of the specialized decode."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.decode import (
        wide_gate,
        wide_rows,
    )

    assert wide_gate(q, rubin=True, seams=True, dsa=True, tp=16, dcp=16) is None
    rows = captured(q)
    assert wide_rows(rows, q) == tuple(m for m in rows if m > 64)
    assert wide_rows(rows, q) == tuple(Q_ROWS[q])


@pytest.mark.parametrize("q", [4, 8])
def test_families_choose_every_wide_row(q):
    """Each family's ``choose`` at every wide row up to 2048 (TP16 = DCP16, Rubin,
    FP32 KDA state): DSA, o-projection and MoE take their specialized wide paths (no
    S2-DSA / S3 / S4 seam), AttnRes runs ClusterAttnRes up to 383 rows and the stock op
    from 384 (STOCK_ROWS), KDA picks a path; nothing raises."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.attn_res.prepared import (  # noqa: E501
        choose as attn_res_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.wide import (
        choose as dsa_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda.prepared import (
        choose as kda_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.prepared import (
        choose as moe_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.oproj.prepared import (  # noqa: E501
        choose as oproj_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import Path, Shape

    for bf16_state in (False, True):
        ctx = context(q, bf16_state=bf16_state)
        for m in Q_ROWS[q]:
            shape = Shape(m, q)
            dsa = dsa_choose(shape, ctx)
            assert dsa is not None and dsa.tactic == "wide", m
            assert oproj_choose(shape, ctx).path is Path.GENERIC, m
            assert moe_choose(shape, ctx).path is Path.GENERIC, m
            # (the head / dense / final AttnRes sites: stock from 384 rows)
            expect = Path.STOCK if m >= 384 else Path.GENERIC
            assert attn_res_choose(shape, ctx).path is expect, m
            assert kda_choose(shape, ctx) is not None, m


class Built(Exception):
    """Stops a constructor under test once the part the test checks has run."""


@pytest.mark.parametrize("q", [4, 8])
def test_decode_capacity_holds_every_captured_row(monkeypatch, q):
    """SpecializedDecode at TP16 = DCP16 on Rubin with graphs up to 2048 rows: every
    captured row above 64 is wide and the decode buffers hold 2048 rows."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import DecodeConfig
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (
        decode as decode_module,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.decode import (
        SpecializedDecode,
    )

    group = SimpleNamespace(world_size=16, rank_in_group=0, device_group=None)
    platform = SimpleNamespace(current_device=lambda: "cuda:0")
    device = SimpleNamespace(major=10, minor=7, multi_processor_count=212)
    monkeypatch.setattr(decode_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(decode_module, "current_platform", platform)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(decode_module, "Context", MagicMock(side_effect=Built))
    text = SimpleNamespace(dsa_config=SimpleNamespace())
    backbone = SimpleNamespace(config=text, use_sequence_parallel=False)
    rows = captured(q)
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=16, q=q,
        capture_sizes=tuple(range(1, 2049)), captured=rows,
    )  # fmt: skip
    decode: Any = object.__new__(SpecializedDecode)
    with pytest.raises(Built):
        SpecializedDecode.__init__(decode, backbone, config)
    assert decode.wide_off is None
    assert decode.wide == tuple(Q_ROWS[q])
    assert decode.capacity == 2048
    assert set(rows) <= set(decode.batches)


@pytest.mark.parametrize("q", [4, 8])
def test_coverage_has_no_stock_step(q):
    """The coverage plan of every captured row up to 2048 (K3.1: KDA + DSA layers,
    MoE) names a specialized step at every row: the DSA, o-projection and MoE families
    on their wide paths, never seam S2-DSA / S3 / S4 or a whole-step stock forward.
    AttnRes runs the stock op exactly at the rows ``attn_res.prepared.STOCK_ROWS``
    names (a measured choice: ClusterAttnRes loses there), its wide path elsewhere."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.attn_res.prepared import (  # noqa: E501
        STOCK_ROWS,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.attn_res.prepared import (  # noqa: E501
        choose as attn_res_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.policy import (  # noqa: E501
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.wide import (
        choose as dsa_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.prepared import (
        PreparedMoE,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.oproj.prepared import (  # noqa: E501
        choose as oproj_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.decode import (
        RING_HEAD,
        SpecializedDecode,
        wide_rows,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.providers import (
        BATCHES,
        projection_policies,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (
        Path,
        Shape,
        rows_policy,
    )

    rows, ctx = captured(q), context(q)
    wide = wide_rows(rows, q)
    shapes = [Shape(m, q) for m in wide]
    decode: Any = object.__new__(SpecializedDecode)
    decode.q, decode.captured, decode.context = q, rows, ctx
    decode.wide, decode.wide_off = wide, None
    decode.batches = tuple(sorted({*BATCHES, *(m for m in rows if m <= 64), *wide}))
    decode.choices = lambda b, q: {"tuned": None}
    modes = {s.rows: attn_res_choose(s, ctx) for s in shapes}
    decode.stock_attn_res = {m for m, c in modes.items() if c.path is Path.STOCK}
    decode.sharded = frozenset()
    # As runtime.providers builds them from the same hooks
    dsa_wide = frozenset(s.rows for s in shapes if dsa_choose(s, ctx))
    oproj_wide = frozenset(s.rows for s in shapes if oproj_choose(s, ctx))
    # The coverage notes' inputs (providers.inproj_note, WideDsa.stock_pieces): the
    # in-projection's split-K rows as Attention builds them (pitch 3040: K3.1 DSA)
    split_k = projection_policies(16, 3040, decode.batches, sms=212, q=q)
    inproj = dict(
        projection=SimpleNamespace(fallback=SimpleNamespace(kernels=split_k)),
        lt_rows={},
        weight=SimpleNamespace(shape=(3040, 0)),
    )
    kda = SimpleNamespace(kind="kda", oproj_wide=oproj_wide, **inproj)
    dsa = SimpleNamespace(
        kind="dsa",
        oproj_wide=oproj_wide,
        dsa_wide=dsa_wide,
        core_wide=SimpleNamespace(stock_pieces=lambda b: "stock pieces"),
        **inproj,
    )
    decode.attention = {0: kda, 1: dsa}
    # As SpecializedDecode builds it on Rubin (ring_batches)
    ring = batch_table(RING_HEAD.get(16, ""))
    decode.head_batches = frozenset(b for b in decode.batches if ring.get(b))
    decode.feed_forward = {0: object.__new__(PreparedMoE)}
    decode.moe_batches = (*BATCHES, *wide)
    coverage = decode.plan_coverage()
    assert set(coverage.plan) == set(rows)
    assert not coverage.reasons, coverage.reasons
    for m in wide:
        plan = coverage.plan[m]
        assert plan is not None, m
        for family in ("dsa", "oproj", "moe"):
            assert plan[family].path is Path.GENERIC, (m, family, plan[family])
        # AttnRes: STOCK only where STOCK_ROWS asks for it, GENERIC everywhere else
        stock = bool(rows_policy(STOCK_ROWS, Shape(m, q), default=0))
        want = Path.STOCK if stock else Path.GENERIC
        assert plan["attn_res"].path is want, (m, "attn_res", plan["attn_res"])
        assert (m in decode.stock_attn_res) == stock, m


@pytest.mark.parametrize("sms", [212, 152])
@pytest.mark.parametrize("heads", [6, 12])
def test_mla_dcp_value_grid_one_wave(sms, heads):
    """mla.dcp.DcpValue (K3 MLA, the drafter, K3.1 DSA's BF16 partials): its merging
    CTAs poll every CTA's pushes, so the grid stays within one CTA per SM at every
    verify row up to 2048, by striding the row tiles (``tile_stride``) above one wave,
    spread (Rubin, q > 1) and per-head (the DSA BF16 mailbox's GB300 schedule); every
    row tile is merged exactly once."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.dcp import DcpValue

    for q in (3, 4, 7, 8):
        for batch in range(9 * q, 2049, q):
            value = DcpValue(heads, 16, batch, sms, q)
            grid = value.grid
            assert grid[0] * grid[1] * grid[2] <= sms, (batch, grid)
            tiles = (batch + 15) // 16
            if value.tile_stride:
                assert grid[1] == value.tile_stride
                owned = sorted(
                    y for t in range(grid[1]) for y in range(t, tiles, grid[1])
                )
                assert owned == list(range(tiles))
            else:
                assert grid[1] == tiles


@pytest.mark.parametrize("q", [4, 8])
def test_dsa_value_grids_one_wave(q):
    """K3.1 WideDsa's DcpValue at every wide row up to 2048 (DCP16, 6 heads): "proj"
    (dsa.dcp_value, merging CTAs over the row tiles) on Rubin's 212 SMs and the BF16
    latent merge (mla.dcp on the GB300 schedule, 152: 6 heads x row tiles x push-only
    CTAs) both within one wave of the device's real SMs; the BF16 merge strides its
    row tiles only past that wave (from 561 rows on Rubin), not past the capped
    schedule's 152."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import dcp_value
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.dcp import (
        GB300_SMS,
        DcpValue,
    )

    for batch in Q_ROWS[q]:
        proj = dcp_value.DcpValue(6, 16, batch, 212, partials="proj").grid
        assert proj[0] * proj[1] * proj[2] <= 212, (batch, proj)
        # the GB300 schedule, its stride past Rubin's real 212 SMs (one wave)
        value = DcpValue(6, 16, batch, GB300_SMS, q, resident_sms=212)
        latent, tiles = value.grid, (batch + 15) // 16
        assert latent[0] * latent[1] * latent[2] <= 212, (batch, latent)
        assert bool(value.tile_stride) == (6 * tiles > 212), (batch, latent)
        if batch <= 560:  # (as before the stride: 6 x 35 tiles unstrided)
            # z: the push-only CTAs (``pushers``) the capped schedule adds while the
            # merge grid leaves room in its 152 SMs (e.g. 6 x 5 x 5 at 68 rows)
            assert latent[:2] == (6, tiles), (batch, latent)
            pushers = latent[2]
            assert pushers == 1 or 6 * tiles * pushers <= GB300_SMS, (batch, latent)
        # (on a 152-SM device the same schedule strides past 152)
        gb300 = DcpValue(6, 16, batch, GB300_SMS, q).grid
        assert gb300[0] * gb300[1] * gb300[2] <= GB300_SMS, (batch, gb300)


def test_dsa_partials_automatic():
    """r1.dsa_partials without a spec: FlashInfer sparse (per-tensor FP8) "1-16:bf16,
    17-:proj" (every wide row "proj"), FlashMLA sparse "bf16"; no environment read."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import r1
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.dcp_value import (
        partials_for,
    )

    spec = r1.dsa_partials(None, fp8=True)
    assert spec == r1.AUTO_PARTIALS
    table = batch_table(spec)
    assert all(table[b] == "bf16" for b in range(1, 17))
    assert all(table[b] == "proj" for b in range(17, 4097))
    assert partials_for(spec, 2048) == "proj" and partials_for(spec, 8) == "bf16"
    assert r1.dsa_partials(None, fp8=False) == "bf16"
    # by verify length: BF16 only on the rows the DSA sparse core serves at that q
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.dsa_kernels.dsa_core.decode import ROWS, VARIANT

    for q in (1, 4, 8):
        expect = f"1-{ROWS}:bf16,{ROWS + 1}-:proj" if q in VARIANT else "proj"
        assert r1.dsa_partials(None, fp8=True, q=q) == expect
    assert r1.dsa_partials(None, fp8=True, q=4) == "proj"
    assert r1.dsa_partials(None, fp8=False, q=4) == "bf16"
    assert r1.dsa_partials("proj") == "proj"
    for bad in ("fp8", "1-64:int8", "1-64"):
        with pytest.raises(ValueError):
            r1.dsa_partials(bad)
    assert not hasattr(r1, "PARTIALS_ENV")


def test_no_wide_env_switches():
    """The DSA wide-row cap and partials switches are gone: vllm.envs no longer
    defines VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS, and dsa.wide has no env-read cap."""
    pytest.importorskip("cutlass")
    import vllm.envs as envs
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import wide

    assert "VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS" not in envs.environment_variables
    assert not hasattr(wide, "wide_max_rows") and not hasattr(wide, "WIDE_LIMIT")


def test_long_split_open_rows():
    """mla.dcp.LONG_SPLIT past its measured rows: the last measured rule per gathered
    head count at every row on its step (96 heads: q4 / q8 rows past 768; 64 heads: q3
    rows past 576), and ``long_split`` returns its partition there."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.dcp import (
        LONG_SPLIT,
        long_split,
    )

    assert LONG_SPLIT[516, 96] == LONG_SPLIT[768, 96] == (1, 2, 64, 128)
    assert LONG_SPLIT[260, 96] == (1, 4, 64, 128)
    for rows in (772, 1024, 1536, 2048):
        assert (rows, 96) in LONG_SPLIT
        assert LONG_SPLIT.get((rows, 96)) == (1, 2, 64, 128)
        assert long_split(rows, 96, 2, 212) == (1, 64, 128)
        assert long_split(rows, 96, 4, 212) is None  # launched at another split
        assert long_split(rows, 96, 2, 148) is None  # GB300
    assert (770, 96) not in LONG_SPLIT and LONG_SPLIT.get((770, 96)) is None
    assert LONG_SPLIT.get((579, 64)) == (1, 4, 64, 128)
    assert LONG_SPLIT.get((2048, 32)) is None


def test_split_workspace_shared(monkeypatch):
    """mla.dcp.split_workspace_ts: one FP32 buffer per (device, heads, split), sized
    for the largest batch reserved or requested, every batch an exact-size prefix
    view (partials, then LSEs at its own offset): the open LONG_SPLIT rows cost one
    buffer, not one per captured shape. A buffer outgrown after its address was kept
    (``keep``, or handed out during a graph capture) stays alive; else it is freed."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla import dcp

    for name in ("_PARTIALS", "_SPLIT_RESERVED", "_SPLIT_VIEWS"):
        monkeypatch.setattr(dcp, name, {})
    monkeypatch.setattr(dcp, "_SPLIT_PINNED", set())
    monkeypatch.setattr(dcp, "_SPLIT_RETIRED", [])
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    cpu, heads, split = torch.device("cpu"), 384, 2
    for reqs in range(193, 513):  # (q4 rows 772-2048)
        dcp.reserve_split_workspace(cpu, heads, reqs, split)
    small = dcp.split_workspace_ts(cpu, heads, 193, split, keep=True)
    large = dcp.split_workspace_ts(cpu, heads, 512, split)
    assert len(dcp._PARTIALS) == 1 and not dcp._SPLIT_RETIRED
    buf = dcp._PARTIALS[cpu, heads, split]
    assert buf.numel() == 512 * heads * split * 513
    for (ws, part, lse), reqs in ((small, 193), (large, 512)):
        cut = reqs * heads * split * 512
        assert ws.numel() == 4 * (cut + reqs * heads * split)
        assert ws.data_ptr() == part.data_ptr() == buf.data_ptr()
        assert part.shape == (reqs, heads, split, 512)
        assert lse.shape == (reqs, heads, split)
        assert lse.data_ptr() == buf.data_ptr() + 4 * cut
    assert dcp.split_workspace_bytes() == buf.numel() * 4
    # Outgrown after a kept address: the old buffer stays alive.
    dcp.split_workspace_ts(cpu, heads, 600, split)
    assert dcp._SPLIT_RETIRED == [buf] and dcp._PARTIALS[cpu, heads, split] is not buf
    # The kept batch keeps its views (its cores and DcpValue share one buffer).
    again = dcp.split_workspace_ts(cpu, heads, 193, split)
    assert all(a is b for a, b in zip(again, small))
    # Outgrown with no kept address: freed, not retired.
    dcp.split_workspace_ts(cpu, heads, 700, split)
    assert len(dcp._SPLIT_RETIRED) == 1
    # Handed out during a capture: kept.
    capturing[0] = True
    dcp.split_workspace_ts(cpu, heads, 8, split)
    capturing[0] = False
    dcp.split_workspace_ts(cpu, heads, 800, split)
    assert len(dcp._SPLIT_RETIRED) == 2
    assert dcp.split_workspace_ts(cpu, 96, 8, 4)[1].shape == (8, 96, 4, 512)
    assert len(dcp._PARTIALS) == 2


def test_work_list_fits():
    """mla.wlist: the list's words pack the request in 8 bits and its sort keys the slot
    in 12: shapes past that run the static order (``shared_list`` None, no assert)."""
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla import wlist

    assert wlist.WLIST.get(2048) and wlist.WLIST.get(68)
    assert wlist.fits(192, 3, 4) and wlist.fits(256, 6, 2)
    assert not wlist.fits(257, 1, 1)  # q4 1028+ rows: 257+ requests
    assert not wlist.fits(256, 6, 4)  # 6144 slots
    assert wlist.shared_list(512, 3, 2, (1, 64, 128), 106, "cpu") is None


def test_wide_experts_chunks():
    """experts_runtime.wide_chunks: one launch up to WIDE_MAX rows, above it as few
    launches of at most WIDE_MAX rows as hold the batch, each from a 64-row plan block,
    covering every row once; the wide workspace is sized at WIDE_MAX at most."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.experts_runtime import (  # noqa: E501
        wide_chunks,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.experts_wide import (  # noqa: E501
        WIDE_MAX,
    )

    for rows in range(65, 2049):
        chunks = wide_chunks(rows)
        if rows <= WIDE_MAX:
            assert chunks == ((0, rows),)
            continue
        assert len(chunks) == -(-rows // WIDE_MAX)
        assert all(n <= WIDE_MAX and start % 64 == 0 for start, n in chunks)
        assert [s for s, _ in chunks] == [
            sum(n for _, n in chunks[:i]) for i in range(len(chunks))
        ]
        assert sum(n for _, n in chunks) == rows


def test_wide_experts_chunked_run():
    """PersistentExperts.run of a chunked wide batch launches each chunk on its rows of
    x, ids and the flat routing weights, in row order, and returns the whole partial."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
        wide_chunks,
    )

    rows = 1028
    experts: Any = object.__new__(PersistentExperts)
    partial = torch.zeros(rows, 4)
    experts.chunks = {rows: (wide_chunks(rows), partial)}
    calls = []
    experts.launch = lambda key, x, ids, weights: calls.append(
        (key, x[0, 0].item(), x.shape[0], ids.shape[0], weights.shape[0])
    )
    x = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, 4)
    ids = torch.zeros(rows, 16, dtype=torch.int32)
    weights = torch.zeros(rows * 16)
    assert experts.run(x, ids, weights) is partial
    assert calls == [((rows, 0), 0.0, 576, 576, 576 * 16),
                     ((rows, 576), 576.0, 452, 452, 452 * 16)]  # fmt: skip


def test_wide_experts_planned_single_launch(monkeypatch):
    """PreparedMoE prepares the wide engines (``WideExperts``) only for the wide batches
    one launch holds (``wide_experts_batch``: at most experts_wide.WIDE_MAX rows, one
    ``wide_chunks`` launch); the wider ones stay off ``wide_batches`` (``run_rows``:
    the stock routed experts). No multi-launch wide experts batch is ever planned."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe import (
        experts_runtime,
        prepared,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.experts_wide import (  # noqa: E501
        WIDE_MAX,
    )

    seen: dict = {}

    class Wide:
        def __init__(self, experts, buffers, partials, *, planned, pushed=None):
            seen.update(partials=dict(partials), pushed=dict(pushed or {}))

    monkeypatch.setattr(experts_runtime, "WideExperts", Wide)
    monkeypatch.setattr(prepared, "stock_router", lambda router: None)
    props = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    batches = (68, 256, 512, WIDE_MAX, WIDE_MAX + 4, 1024, 2048)
    state = SimpleNamespace(batches=batches, tp=16, device=0, latent_partial=None,
                            experts={})  # fmt: skip
    front = SimpleNamespace(split={}, release_early=lambda bs: list(bs))
    experts = SimpleNamespace(grids={}, grid_for=lambda b: 212)
    rows = SimpleNamespace(experts_push=batches, experts_box=None)
    module = SimpleNamespace(experts=SimpleNamespace(router=None))
    front_rows = SimpleNamespace(down_sms={})
    moe = prepared.PreparedMoE(
        front, experts, None, state, rows=rows, module=module, front_rows=front_rows
    )
    single = {68, 256, 512, WIDE_MAX}
    assert moe.wide_batches == single
    assert set(seen["partials"]) == set(seen["pushed"]) == single
    assert all(len(experts_runtime.wide_chunks(b)) == 1 for b in seen["partials"])
    for b in range(65, 4097):
        if prepared.wide_experts_batch(b):
            assert len(experts_runtime.wide_chunks(b)) == 1 and b <= WIDE_MAX, b


def test_decode_feeds_front_rows_single_launch_batches():
    """runtime.decode hands FrontRows and the wide tail's experts push
    (``PreparedTailRows(experts_push=)``) only the wide batches the wide engines serve
    in one launch (``wide_engines``: ``wide_experts_batch``), not every wide MoE
    batch. (Source check: SpecializedDecode needs the model and its ranks.)"""
    import ast

    tree = ast.parse((SD / "runtime/decode.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def named(name):
        return [c for c in calls if getattr(c.func, "id", None) == name]

    (front_rows,) = named("FrontRows")
    assert ast.unparse(front_rows.args[4]) == "wide_engines"
    (tail_rows,) = named("PreparedTailRows")
    push = {k.arg: ast.unparse(k.value) for k in tail_rows.keywords}["experts_push"]
    assert push == "wide_engines"
    (engines,) = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "wide_engines"
    ]
    assert ast.unparse(engines) == (
        "tuple((b for b in wide_moe if wide_experts_batch(b)))"
    )
    guards = [
        ast.unparse(n.test)
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and front_rows in ast.walk(n)
    ]
    assert any("wide_engines" in guard for guard in guards), guards


def test_tail_pushes_only_single_launch_rows():
    """The wide tail's experts push (EXPERTS_PUSH, open-ended) applies only to rows the
    experts serve in one launch (at most WIDE_MAX); NORM_POLLED covers every row from
    193 on."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe import tail_rows

    table = batch_table(tail_rows.EXPERTS_PUSH[16])
    assert table.get(2048) and table.get(65)
    assert tail_rows.norm_polled(16, 2048) and not tail_rows.norm_polled(16, 192)


def test_row_exchange_chunks():
    """verify.exchange.PreparedExchange.run above MAX_ROWS rows: launches of at most
    MAX_ROWS rows in row order, each into its rows of ``gathered``."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.verify import exchange

    prepared: Any = object.__new__(exchange.PreparedExchange)
    prepared.tp, prepared.words = 16, 8
    launched = []

    def launch(entries, gathered):
        launched.append((entries.shape[0], gathered.shape[0]))
        gathered.view(torch.int32).fill_(len(launched))

    prepared.launch = launch
    rows = 2 * exchange.MAX_ROWS + 3
    entries = torch.ones((rows, 8), dtype=torch.int32).view(torch.uint32)
    gathered = prepared.run(entries)
    assert gathered.shape == (rows, 16, 8)
    max_rows = exchange.MAX_ROWS
    assert launched == [(max_rows, max_rows), (max_rows, max_rows), (3, 3)]
    assert gathered[max_rows].view(torch.int32).unique().tolist() == [2]
    with pytest.raises(ValueError):
        prepared.run(entries[:0])


def test_sampled_verify_takes_every_row():
    """SampledVerify covers every logits row count (its exchanges chunk): no stock
    rejection sampler above 512 rows."""
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.sampled_verify import (
        SampledVerify,
    )

    verify: Any = object.__new__(SampledVerify)
    assert all(verify.takes(rows) for rows in (1, 512, 513, 1024, 2048, 4096))
    assert not verify.takes(0)


def test_wide_experts_single_launch_only():
    """Wide routed experts run on the wide engines only where one launch holds the
    batch (at most experts_wide.WIDE_MAX rows); above it the stock routed experts run
    behind the stock front, into the wide tail (moe.prepared.wide_experts_batch)."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe import prepared
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.experts_wide import (  # noqa: E501
        WIDE_MAX,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import Shape

    ctx = context(4)
    for rows in (68, 256, 512, WIDE_MAX):
        assert prepared.wide_experts_batch(rows), rows
        assert "wide experts" in prepared.choose(Shape(rows, 4), ctx).note, rows
    for rows in (WIDE_MAX + 4, 1024, 2048):
        assert not prepared.wide_experts_batch(rows), rows
        note = prepared.choose(Shape(rows, 4), ctx).note
        assert "stock FlashInfer experts" in note and "stock front" in note, (
            rows,
            note,
        )
        assert "wide engines lose above one launch" in note, (rows, note)
