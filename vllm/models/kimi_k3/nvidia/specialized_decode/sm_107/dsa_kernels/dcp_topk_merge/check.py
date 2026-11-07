# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_DSA_BPRIME_CHECK=1`` (debug only): the source branch's
``sparse_attn_indexer._check_dcp_decode_topk_merge``, for the specialized decode's
merges. The reference is this tree's stock generic merge
(``_merge_dcp_topk_global``), which takes the logits dtype where the source branch
took its score bits; the comparison itself is the vendored
``compat.dcp_topk_merge_check``."""

import torch

from vllm.utils.math_utils import cdiv


def _check_dcp_decode_topk_merge(
    layer_name: str,
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    merged: torch.Tensor,
    dcp_local_topk: int,
    topk_backend: str,
    fp32_decode_logits: bool,
    max_seq_len: int,
    max_model_len: int,
    dcp_rank: int,
    dcp_world_size: int,
    score_bits: int,
    sat_word: torch.Tensor | None = None,
) -> None:
    """The generic merge of the same decode logits next to the one-round merge's
    ``merged`` ids, compared on the device. ``sat_word``: the merge's saturation
    words per row (default: this copy's rescue workspace's). Collective over the
    DCP group."""
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _merge_dcp_topk_global,
        as_fp32_logits,
    )

    from ...compat.dcp_topk_merge_check import check_dcp_topk_merge
    from .kernels import get_dcp_topk_rescue_workspace

    if score_bits not in (16, 32):
        raise ValueError(f"score_bits must be 16 or 32, got {score_bits}")
    topk_tokens = merged.shape[1]
    reference = torch.full_like(merged, -1)
    reference_logits = (
        as_fp32_logits(logits, seq_lens.view(-1)) if fp32_decode_logits else logits
    )
    get_indexer_topk(topk_backend)(
        reference_logits, seq_lens, next_n, reference, topk_tokens, max_seq_len
    )
    _merge_dcp_topk_global(
        reference_logits,
        reference,
        topk_tokens,
        dcp_rank,
        dcp_world_size,
        1,
        # The score bits the selector keys: 16 for BF16 logits (upcast or not).
        logits_dtype=torch.bfloat16 if score_bits == 16 else torch.float32,
        num_positions=max_model_len,
    )
    if sat_word is None:
        workspace = get_dcp_topk_rescue_workspace(topk_tokens, dcp_local_topk)
        assert workspace is not None
        sat_word = workspace.sat_word
    check_dcp_topk_merge(
        layer_name,
        logits[:, : cdiv(max_model_len, dcp_world_size)],
        row_ends,
        merged,
        reference,
        sat_word,
        dcp_rank,
        dcp_world_size,
    )
