# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicted cost of a Kimi K3 (DSA) prefill chunk on a DEP prefill rank.

The DP ranks of an EP group run their steps in lockstep, so a step costs the
rank with the most work, and a token's work grows with its context (attention
and the DSA indexer read every earlier token). The scheduler sizes prompt
chunks by this cost instead of by tokens alone, so that a long-context chunk
does not set the step for every rank (see ``Scheduler._prompt_chunk_cost``).

Costs are in "short-context tokens": a token at context ``ctx`` costs
``weight(ctx) = 1 + CTX_WEIGHT * ctx``. CTX_WEIGHT is fitted on served lockstep
steps of the int62 build (Hecate 4n DEP16 prefill testbed, c512, 1219 steps):
step time vs the busiest rank's sum of n * weight(ctx) is best at 1.0e-6 per
context token (R^2 0.943, vs 0.920 for tokens alone and 0.856 for the older
PSOL whole-step table, which overstated long context on this build: w(600K)
1.6 here vs 2.95). Scheduling only: no computation changes, the cost only
decides how many prompt tokens a step takes.
"""

from typing import Any

# Cost of one prompt token per context token, relative to a short-context token.
CTX_WEIGHT = 1.0e-6


def weight(ctx: int) -> float:
    """Cost of one prompt token computed at context ``ctx`` (>= 1.0)."""
    return 1.0 + CTX_WEIGHT * max(ctx, 0)


def chunk_cost(num_tokens: int, ctx: int) -> float:
    """Cost of a prompt chunk of ``num_tokens`` starting at context ``ctx``."""
    return num_tokens * weight(ctx) if num_tokens > 0 else 0.0


def prompt_cost(num_tokens: int, ctx: int) -> float:
    """Cost of computing ``num_tokens`` prompt tokens from context ``ctx`` over
    however many chunks (each token priced at its own context: the midpoint
    weight of the span), for planning where outstanding prompt work runs."""
    return num_tokens * weight(ctx + num_tokens // 2) if num_tokens > 0 else 0.0


def max_tokens_for_cost(cost: float, ctx: int) -> int:
    """Most prompt tokens from context ``ctx`` whose chunk costs <= ``cost``."""
    if cost <= 0:
        return 0
    return int(cost / weight(ctx) + 1e-9)


def applies(vllm_config: Any) -> bool:
    """A Kimi K3 DSA model served data-parallel (DEP) by a prefill-capable
    engine: the instances whose lockstep steps this cost model describes."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    pc = vllm_config.parallel_config
    if pc.data_parallel_size <= 1:
        return False
    hf = model_config.hf_text_config
    if getattr(hf, "dsa_config", None) is None or not callable(
        getattr(hf, "is_dsa_layer", None)
    ):
        return False
    kv = vllm_config.kv_transfer_config
    # A decode-only (KV consumer) instance computes no long prompt chunks.
    return kv is None or kv.kv_role != "kv_consumer"
