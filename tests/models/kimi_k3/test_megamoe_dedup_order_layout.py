# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MegaMoE combine-dedup order buffer: tight layout geometry (CPU only).

The order buffer holds, per (src_token, src_rank) block, one bf16 copy of the FC2 output row for every route of that
token on this rank. Its worst case is T x R x min(K, E/R) copies (every token of every rank routes all K experts to
this rank, possible whenever E/R >= K), so the block keeps K slots. Only the per-copy padding is removable: a copy is
stored at the output row size rounded to the 256 B FC2 hidden tile, the partial last chunk at its own size.
"""

import pytest

pytest.importorskip("cutlass")


def _token_comm(world_size: int, expert_count: int, topk: int, tokens: int, hidden: int):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.api import ImplDesc, ProblemDesc
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.communication.nvlink_domain.token_comm_dedup import (
        TokenCommDedup,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.quant_def import CombineFormat

    problem = ProblemDesc(
        {
            "world_size": world_size,
            "expert_count": expert_count,
            "topk": topk,
            "max_tokens_per_rank": tokens,
            "hidden_size": hidden,
            "quant_kind": "nvfp4",
            "combine_format": CombineFormat.parse("bf16"),
            "apply_topk_at_fc1": True,
        }
    )
    impl = ImplDesc(
        {
            "token_padding_block": 128,
            "sf_padding_block": 128,
            "tokens_per_fc1_ready_slot": 128,
            "fc2_done_signals_per_token_tile": 1,
            "fc1_done_signals_per_token_tile": 24,
            "promised_launchable_sm_count": 148,
            "token_in_flag_batch": 1,
            "token_back_mode": "standalone_warps",
            "token_back_schedule_mode": "static",
            "reduce_topk_in_kernel": False,
        }
    )
    return TokenCommDedup(problem, impl)


def _chunk_size(tc, chunk_idx: int) -> int:
    return min(tc.order_chunk_bytes, tc.order_row_bytes - chunk_idx * tc.order_chunk_bytes)


@pytest.mark.parametrize(
    "world_size,expert_count,hidden",
    [(16, 896, 3584), (4, 224, 3584), (8, 448, 3584), (16, 896, 2048), (4, 224, 3712)],
)
def test_order_block_is_dense_and_disjoint(world_size, expert_count, hidden):
    tc = _token_comm(world_size, expert_count, 16, 8192, hidden)
    chunk, rows = tc.order_chunk_bytes, tc.order_chunks_per_row
    assert tc.order_row_bytes % 256 == 0 and tc.order_row_bytes >= tc.output_token_bytes
    assert tc.order_row_bytes - tc.output_token_bytes < 256
    assert (rows - 1) * chunk < tc.order_row_bytes <= rows * chunk
    assert tc.order_block_bytes == tc.topk * tc.order_row_bytes
    # Every (chunk, copy) slot of a block, as the FC2 epilogue writes it and token_back loads it.
    covered = []
    for c in range(rows):
        size = _chunk_size(tc, c)
        assert size > 0 and size % 256 == 0
        for k in range(tc.topk):
            covered.append((c * tc.topk * chunk + k * size, size))
        # token_back loads two adjacent copies with one 1-D TMA: copy k+1 follows copy k at the chunk's own size.
        assert covered[-1][0] + size == c * tc.topk * chunk + tc.topk * size
    covered.sort()
    end = 0
    for start, size in covered:
        assert start == end, "order slots must tile the block with no gap or overlap"
        end = start + size
    assert end == tc.order_block_bytes


def test_production_ep16_order_bytes():
    # DEP16 prefill: 8192-token sub-call, K=16, latent hidden 3584 (bf16 row 7168 B = 3.5 chunks of 2 KiB).
    tc = _token_comm(16, 896, 16, 8192, 3584)
    order_bytes = tc.max_tokens_per_rank * tc.world_size * tc.order_block_bytes
    assert order_bytes == 14 * 2**30  # was 16 GiB with every row padded to 4 whole chunks
    # No smaller exact bound exists: E/R = 56 >= K, so one rank can receive every route of every token.
    assert tc.experts_per_rank >= tc.topk


@pytest.mark.parametrize(
    "world_size,expert_count,hidden,topk",
    [(16, 896, 3584, 16), (8, 896, 3584, 16), (8, 448, 3584, 8), (4, 224, 3712, 16), (8, 64, 256, 2)],
)
def test_copy_mailbox_sits_in_the_unused_last_slot(world_size, expert_count, hidden, topk):
    """Combine readiness: the copy mailbox ((K-1) Int32 pool rows) lives in chunk 0 of copy slot K-1, which a block
    with cnt < K never writes (FC2 stores copies 0..cnt-1) and never reads (the reduce loads copies 0..cnt-1)."""
    tc = _token_comm(world_size, expert_count, topk, 8192, hidden)
    begin = tc.copy_mailbox_offset
    end = begin + (topk - 1) * 4
    slot_k1_chunk0 = ((topk - 1) * _chunk_size(tc, 0), topk * _chunk_size(tc, 0))
    assert slot_k1_chunk0[0] == begin and end <= slot_k1_chunk0[1]
    for c in range(tc.order_chunks_per_row):
        size = _chunk_size(tc, c)
        for k in range(topk - 1):
            start = c * topk * tc.order_chunk_bytes + k * size
            assert start + size <= begin or start >= end, "mailbox overlaps a copy slot used when cnt < K"
