# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model


def window_context_rows(
    query_start_loc: np.ndarray,
    seq_ends: np.ndarray,
    span: int,
    hit_alignment: int | None,
) -> np.ndarray | None:
    """Context rows whose draft K/V a sliding-window drafter can ever read,
    when that is a strict subset of the step's rows; else None.

    Draft queries sit past the context and attend through a window of W
    positions, so a request continuing from this step reads only the K/V of
    its last W context positions, and a later request resuming from a cached
    prefix at position h reads only [h - W, h); such hits land on multiples of
    ``hit_alignment`` (None: no prefix reuse). Per request, keep the rows
    within ``span`` (>= W + draft tokens) of its chunk end and of every hit
    point inside its chunk. Requests of at most ``span`` rows are kept whole,
    so verify rows (whose positions ``seq_ends`` may overstate) never trim.

    Args:
        query_start_loc: [num_reqs + 1] row offsets of the requests.
        seq_ends: [num_reqs] sequence length after this step's rows.
        span: rows kept before each read point.
        hit_alignment: prefix-cache hit granularity in tokens, or None.

    Returns:
        Sorted kept row indices, or None when every row is kept.
    """
    qlen = np.diff(query_start_loc)
    if qlen.size == 0 or int(qlen.max()) <= span:
        return None
    if hit_alignment is not None and hit_alignment <= span:
        return None
    pieces = []
    for r in range(qlen.size):
        lo, hi = int(query_start_loc[r]), int(query_start_loc[r + 1])
        if hi - lo <= span:
            pieces.append(np.arange(lo, hi, dtype=np.int64))
            continue
        end_pos = int(seq_ends[r])
        start_pos = end_pos - (hi - lo)
        keep = np.zeros(hi - lo, dtype=bool)
        keep[-span:] = True
        if hit_alignment is not None:
            first = (start_pos // hit_alignment + 1) * hit_alignment
            for h in range(first, end_pos, hit_alignment):
                keep[max(0, h - span - start_pos) : h - start_pos] = True
        pieces.append(lo + np.flatnonzero(keep))
    rows = np.concatenate(pieces)
    if rows.size == int(query_start_loc[-1]) - int(query_start_loc[0]):
        return None
    return rows
