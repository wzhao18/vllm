# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ kernels of the rank-major MLA workspace, the fused new-token KV
insert and the token-slice query quantization (one GPU; the multi-GPU step is
covered by test_dcp_shardq.py)."""

import numpy as np
import pytest
import torch

from tests.v1.attention.test_dcp_shardq import (
    PAGE,
    SCENARIOS,
    TOPK,
    Replicated,
    _build_context,
    _flat,
    _layout,
    _shard,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

pytestmark = pytest.mark.skipif(
    not (current_platform.is_device_capability_family(100) and has_cutedsl()),
    reason="CuTe DSL on SM100 is required",
)


def _rank_major_workspace(rep: Replicated, world: int, cache_dtype: str):
    """Every rank's plan, and the workspace MLA rows after each rank packed its
    region (what each rank holds once the peers' copies landed)."""
    from vllm.v1.attention.ops.dcp_shardq import rank_major

    layout = rep.layout
    plans = [_build_context(rep.lens, layout, world, r) for r in range(world)]
    pages = plans[0].pages
    ws = torch.full(
        (pages * layout.mla_page_bytes,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    for rank, plan in enumerate(plans):
        assert plan.rank_major is not None
        shard = _shard(rep, world, rank, seed=rank)
        rank_major.pack(plan.rank_major, shard.mla, shard.mla_table, 0, ws)
    torch.accelerator.synchronize()
    return plans, ws


def _token_rows(rep: Replicated, q: int, tokens: torch.Tensor) -> torch.Tensor:
    cache, src = rep.token_bytes(q, tokens, ("mla", 0, rep.layout.mla_row_bytes))
    return _flat(cache)[src.reshape(-1)].view(len(tokens), rep.layout.mla_row_bytes)


@pytest.mark.parametrize("cache_dtype", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_pack_and_remap_address_every_context_row(cache_dtype, world, scenario):
    """Each rank's pack puts its context rows where the remap's prefill-
    workspace mode (rank-major, DCP de-interleaved) reads them, and leaves the
    new tokens' rows (``mla_slot_mapping``) to the insert."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    lens = SCENARIOS[scenario]
    rep = Replicated(lens, _layout(cache_dtype))
    plans, ws = _rank_major_workspace(rep, world, cache_dtype)
    row = rep.layout.mla_row_bytes
    rows = ws.view(-1, row)
    plan = plans[0].rank_major
    # Identical host plans on every rank.
    for other in plans[1:]:
        np.testing.assert_array_equal(other.rank_major.starts, plan.starts)
        assert other.rank_major.region_rows == plan.region_rows
        assert torch.equal(other.mla_slot_mapping, plans[0].mla_slot_mapping)
    for q, (context, new) in enumerate(lens):
        tokens = torch.arange(context, device="cuda")
        at = (tokens % world) * plan.region_rows + int(plan.starts[q]) + tokens // world
        assert torch.equal(rows[at], _token_rows(rep, q, tokens)), (q, "context")
    # New tokens' rows: disjoint from every context row, inside a region.
    slots = plans[0].mla_slot_mapping.cpu().numpy()
    qsl = np.r_[0, np.cumsum([n for _, n in lens])]
    for q, (context, new) in enumerate(lens):
        t = np.arange(context, context + new)
        expected = (t % world) * plan.region_rows + plan.starts[q] + t // world
        np.testing.assert_array_equal(slots[qsl[q] : qsl[q + 1]], expected)
    assert np.unique(slots).size == slots.size
    # The remap of every rank's top-k reads exactly those rows.
    torch.manual_seed(1)
    for rank, rank_plan in enumerate(plans):
        rmp = rank_plan.rank_major
        for q, (context, new) in enumerate(lens):
            seq = context + new
            num = min(seq, TOPK)
            topk = torch.full((1, TOPK), -1, dtype=torch.int32, device="cuda")
            topk[0, :num] = torch.randperm(seq, device="cuda")[:num].int()
            req = torch.tensor([q], dtype=torch.int32, device="cuda")
            physical, counts = triton_convert_req_index_to_global_index(
                req,
                rank_plan.block_table,
                topk,
                BLOCK_SIZE=PAGE,
                BLOCK_STRIDE_ROWS=PAGE,
                NUM_TOPK_TOKENS=TOPK,
                return_valid_counts=True,
                HAS_PREFILL_WORKSPACE=True,
                prefill_workspace_request_ids=req,
                prefill_workspace_starts=rmp.starts_by_req,
                prefill_workspace_rank_stride=rmp.region_rows,
                dcp_size=world,
                dcp_rank=rank,
                cp_kv_cache_interleave_size=1,
            )
            assert int(counts[0]) == num
            got = torch.sort(physical[0, :num]).values
            t = topk[0, :num].long()
            want = (t % world) * rmp.region_rows + int(rmp.starts[q]) + t // world
            assert torch.equal(got, torch.sort(want).values.int())


def test_plan_falls_back_when_a_region_overflows():
    """A step whose rank-major rows exceed a region keeps the token-ordered
    layout; a step of several request groups has no rank-major plan."""
    from vllm.v1.attention.ops.dcp_shardq.rank_major import plan_rank_major

    lens = SCENARIOS["page_tails"]
    seq_lens = np.array([c + n for c, n in lens])
    query_lens = np.array([n for _, n in lens])
    world = 8
    rows = int((-(-seq_lens // world)).sum())

    def plan(pages):
        return plan_rank_major(
            seq_lens, query_lens, 0, pages, PAGE, world, 0,
            lambda a, d: torch.from_numpy(np.ascontiguousarray(a)).to(d),
        )

    fits = -(-rows * world // PAGE)
    assert plan(fits) is not None
    assert plan(fits - 1) is None
    pages = int((-(-seq_lens // PAGE)).sum())
    split = _build_context(lens, _layout("fp8"), world, 0, pages - 1)
    assert split.num_groups == 2 and split.rank_major is None


@pytest.mark.parametrize("scale", [1.0, 0.37])
@pytest.mark.parametrize("ws_first", [0, 5])
def test_kv_insert_matches_concat_and_cache_mla(scale, ws_first):
    """One launch writes the local slots (-1 skipped) and the workspace slots
    of the new tokens byte for byte as two concat_and_cache_mla calls."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.dcp_shardq.kv_insert import (
        shardq_kv_insert,
        shardq_kv_insert_supported,
    )

    torch.manual_seed(0)
    num_tokens, nope, rope, block = 1037, 512, 64, 128
    # kv_c / k_pe as views of the fused projection output.
    proj = torch.randn(num_tokens, 1536 + nope + rope, device="cuda") * 30
    proj = proj.bfloat16()
    proj[3, 1600] = float("inf")
    kv_c = proj[:, 1536 : 1536 + nope]
    k_pe = proj[:, 1536 + nope :].unsqueeze(1)
    scale_t = torch.tensor([scale], device="cuda")

    def cache(blocks):
        raw = torch.randint(0, 256, (blocks, block, nope + rope + 32), device="cuda")
        return raw.to(torch.uint8)[..., : nope + rope]

    local_ref, local = cache(40), None
    local = local_ref.clone()
    ws_ref = torch.randint(
        0, 256, (num_tokens // 64 + 2, 64, nope + rope), device="cuda"
    ).to(torch.uint8)
    ws = ws_ref.clone()
    perm = torch.randperm(40 * block, device="cuda")[:num_tokens]
    local_slots = torch.where(
        torch.arange(num_tokens, device="cuda") % 4 == 1, perm, -1
    ).long()
    ws_slots = torch.randperm(ws.shape[0] * 64, device="cuda")[: num_tokens - ws_first]
    ws_slots = ws_slots.long()

    dtype = "fp8"
    fp8 = current_platform.fp8_dtype()
    ops.concat_and_cache_mla(
        kv_c, k_pe.squeeze(1), local_ref.view(fp8), local_slots, dtype, scale_t,
        skip_scale=scale == 1.0,
    )
    ops.concat_and_cache_mla(
        kv_c[ws_first:], k_pe[ws_first:].squeeze(1), ws_ref.view(fp8), ws_slots,
        dtype, scale_t, skip_scale=scale == 1.0,
    )
    assert shardq_kv_insert_supported(kv_c, k_pe.squeeze(1), local.view(fp8))
    shardq_kv_insert(
        kv_c, k_pe, scale_t, local.view(fp8), local_slots, ws.view(fp8), ws_slots,
        ws_first,
    )
    assert torch.equal(local, local_ref)
    assert torch.equal(ws, ws_ref)


@pytest.mark.parametrize("padded_heads", [96, 128])
def test_query_quant_matches_cat_and_scaled_fp8_quant(padded_heads):
    """The token slice's FP8 query is the MLA layer's cat + static quant, bit
    for bit, with zero padding heads."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.dcp_shardq.query_quant import shardq_query_quant_pad

    torch.manual_seed(0)
    tokens, heads = 333, 96
    ql_nope = (torch.randn(tokens, heads, 512, device="cuda") * 200).bfloat16()
    q = (torch.randn(tokens, heads, 192, device="cuda") * 3).bfloat16()
    q_pe = q[..., 128:]
    scale = torch.tensor([0.71], device="cuda")
    out = shardq_query_quant_pad(ql_nope, q_pe, scale, padded_heads)
    ref, _ = ops.scaled_fp8_quant(
        torch.cat((ql_nope, q_pe), -1).view(tokens * heads, -1), scale
    )
    assert torch.equal(
        out[:, :heads].reshape(tokens * heads, -1).view(torch.uint8),
        ref.view(torch.uint8),
    )
    assert not out[:, heads:].view(torch.uint8).any()
