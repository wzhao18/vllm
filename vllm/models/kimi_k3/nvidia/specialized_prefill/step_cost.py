# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicted cost of a Kimi K3 (DSA) prefill chunk on a DEP prefill rank.

The DP ranks of an EP group run their steps in lockstep, so a step costs the
rank with the most work, and a token's work grows with its context (attention
and the DSA indexer read every earlier token). The scheduler sizes prompt
chunks by this cost instead of by tokens alone, so that a long-context chunk
does not set the step for every rank (see ``Scheduler._prompt_chunk_cost``).

Costs are in "short-context tokens": a token at context ``ctx`` costs
``weight(ctx)``, 1.0 up to 30K. The weights are the measured DEP16 prefill
step time of a 16K-token chunk at each prefix length (PSOL, int58 build,
milliseconds): <=30K 768.7, 60K 776.3, 130K 822.8, 300K 1137.8, 600K 2270.1,
linear in between and past the last point. Scheduling only: no computation
changes, the cost only decides how many prompt tokens a step takes.
"""

from typing import Any

# (context tokens, ms per 16K-token chunk) at the chunk start.
_SOL_MS = (
    (30_000, 768.7),
    (60_000, 776.3),
    (130_000, 822.8),
    (300_000, 1137.8),
    (600_000, 2270.1),
)
_BASE_MS = _SOL_MS[0][1]


def weight(ctx: int) -> float:
    """Cost of one prompt token computed at context ``ctx`` (>= 1.0)."""
    if ctx <= _SOL_MS[0][0]:
        return 1.0
    for (c0, t0), (c1, t1) in zip(_SOL_MS, _SOL_MS[1:]):
        if ctx <= c1:
            return (t0 + (t1 - t0) * (ctx - c0) / (c1 - c0)) / _BASE_MS
    (c0, t0), (c1, t1) = _SOL_MS[-2], _SOL_MS[-1]
    return (t1 + (t1 - t0) * (ctx - c1) / (c1 - c0)) / _BASE_MS


def chunk_cost(num_tokens: int, ctx: int) -> float:
    """Cost of a prompt chunk of ``num_tokens`` starting at context ``ctx``."""
    return num_tokens * weight(ctx) if num_tokens > 0 else 0.0


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
