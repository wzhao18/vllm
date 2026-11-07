# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP top-k filter of the specialized decode under kernel blocks finer than the
DCP interleave.

The base's ``triton_filter_and_convert_dcp_index`` asserts
``BLOCK_SIZE % cp_kv_cache_interleave_size == 0``. The specialized decode reads
the cache in 64-token kernel blocks under a larger logical block and interleave
(e.g. 896): a kernel block then holds contiguous local positions of one
interleave group, so ``local_idx // BLOCK_SIZE`` stays exact and the base's
conversion kernel is correct as is. The source branch relaxed that assert to
"the two divide one another"; this wrapper runs the same kernel with the
relaxed check, and defers to the base function whenever its own assert holds.
"""

import torch

from vllm.v1.attention.backends.mla import sparse_utils


def filter_and_convert_dcp_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int = 1,
    BLOCK_SIZE: int = 64,
    BLOCK_STRIDE_ROWS: int | None = None,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``triton_filter_and_convert_dcp_index`` with interleave and kernel block
    nesting either way."""
    interleave = cp_kv_cache_interleave_size
    kwargs = dict(
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_STRIDE_ROWS=BLOCK_STRIDE_ROWS,
        NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
        BLOCK_N=BLOCK_N,
        return_valid_counts=return_valid_counts,
        compact_valid_to_front=compact_valid_to_front,
    )
    if dcp_size == 1 or BLOCK_SIZE % interleave == 0:
        return sparse_utils.triton_filter_and_convert_dcp_index(
            req_id, block_table, token_indices, **kwargs
        )
    assert interleave % BLOCK_SIZE == 0, (
        f"BLOCK_SIZE ({BLOCK_SIZE}) and cp_kv_cache_interleave_size "
        f"({interleave}) must divide one another."
    )
    assert 0 <= dcp_rank < dcp_size
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    num_tokens = req_id.shape[0]
    token_indices_c = token_indices.contiguous()
    count_valid = return_valid_counts or compact_valid_to_front
    single_tile, _, _, _ = sparse_utils._remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, count_valid
    )
    if compact_valid_to_front and not single_tile:
        out = torch.full_like(token_indices_c, -1)
    else:
        out = torch.empty_like(token_indices_c)
    valid_counts: torch.Tensor | None = None
    if count_valid:
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)

    sparse_utils._CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL(
        req_id.contiguous(),
        block_table.contiguous(),
        token_indices_c,
        out,
        valid_counts,
        None,
        None,
        None,
        workspace_rank_stride=0,
        max_num_blocks_per_req=block_table.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_STRIDE_ROWS=(
            BLOCK_STRIDE_ROWS if BLOCK_STRIDE_ROWS is not None else BLOCK_SIZE
        ),
        BLOCK_N=BLOCK_N,
        NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
        HAS_PREFILL_WORKSPACE=False,
        HAS_IDX_MAPPING=False,
        COUNT_VALID=count_valid,
        COMPACT_TO_FRONT=compact_valid_to_front,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        DCP_INTERLEAVE=interleave,
    )
    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
