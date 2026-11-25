# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MegaMoE of the wide-EP (TP1 x DP, EP = DP) prefill worker on VR200.

A P3-16 prefill rank feeds up to ``max_num_batched_tokens`` (16384 at F16K) rows to
every routed-MoE layer. The Rubin MegaMoE runs them as ``ceil(T_max / C)`` launches of
a capacity-``C`` bucket (``C`` = :data:`PREFILL_MOE_SUBCALL_TOKENS`) instead of one
launch of a ``T``-capacity bucket:

* the kernel's measured per-token cost is lower at the 8192 capacity than at 16384
  (FlashInfer SM107 MegaMoE, EP16 NVFP4: 2 x 1254 us vs 2649 us; knob records exist
  up to the 8192 capacity only);
* the symmetric dispatch / combine arena is sized for ``C`` rows (the combine slots
  alone are top_k x C x latent x 2 B), which leaves room for F16K on a TP1 rank.

The sub-call count is EP-uniform (from the forward's DP batch sizes): every rank
launches the same number of times (the kernel synchronizes the EP group per launch);
a rank with fewer rows passes empty slices. Each token's output depends only on its
own routes (per-row quantization, per-expert GEMM rows, fixed top-k reduction order),
so the sub-calls are bitwise identical to the single launch.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config.kernel import KIMI_K3_MEGA_MOE
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# MegaMoE sub-call capacity (tokens per rank) of a prefill worker: the largest
# capacity with tuned VR knob records (EP16 NVFP4 i8192 phase-interleave N256), and
# the FlashInfer SM107 MegaMoE per-token optimum (EP16 NVFP4: 1254 us at 8192 vs
# 2649 us at 16384 per call).
PREFILL_MOE_SUBCALL_TOKENS = 8192


@dataclasses.dataclass(frozen=True)
class PrefillMoEPolicy:
    """Per-layer prefill MegaMoE policy (automatic, no switch).

    ``capacity``: MegaMoE bucket capacity (tokens per rank) = sub-call size.
    ``max_num_tokens``: most tokens one rank's layer sees per forward.
    """

    capacity: int
    max_num_tokens: int

    @property
    def subcalls(self) -> bool:
        return self.max_num_tokens > self.capacity


def prefill_moe_policy(
    vllm_config: VllmConfig,
    *,
    tp_size: int,
    sequence_parallel: bool,
    max_num_tokens: int,
) -> PrefillMoEPolicy | None:
    """The policy of a routed-MoE layer that runs the Rubin MegaMoE on whole DP
    batches (TP1 wide-EP, no sequence parallelism) with prefill-sized
    batches (more than one sub-call of rows), else None (stock path)."""
    if (
        vllm_config.kernel_config.moe_backend != KIMI_K3_MEGA_MOE
        or max_num_tokens <= PREFILL_MOE_SUBCALL_TOKENS
        or tp_size != 1
        or sequence_parallel
    ):
        return None
    return PrefillMoEPolicy(
        capacity=min(max_num_tokens, PREFILL_MOE_SUBCALL_TOKENS),
        max_num_tokens=max_num_tokens,
    )


def _subcall_rows(counts: list[int], capacity: int) -> tuple[int, tuple[int, ...]]:
    """``(number of sub-calls, EP-mean live rows per sub-call)`` for per-rank row
    counts ``counts`` (identical on every rank)."""
    nsub = max(1, cdiv(max(counts), capacity))
    ep_rows = tuple(
        cdiv(sum(min(capacity, max(0, c - i * capacity)) for c in counts), len(counts))
        for i in range(nsub)
    )
    return nsub, ep_rows


class PrefillSubcallExperts:
    """Runs a ``KimiK3MegaMoEExperts`` layer over ``capacity``-row sub-calls.

    Not an ``nn.Module`` (it holds the layer's experts module, which stays registered
    on the MoE block)."""

    def __init__(self, experts, capacity: int, dp_size: int):
        if experts.max_num_tokens != capacity:
            raise ValueError(
                f"{experts.prefix}: sub-call capacity {capacity} != the experts' "
                f"bucket capacity {experts.max_num_tokens}."
            )
        self.experts = experts
        self.capacity = capacity
        self.dp_size = dp_size
        self._memo: tuple[object, int, tuple[int, tuple[int, ...]]] | None = None

    def _plan(self, num_tokens: int) -> tuple[int, tuple[int, ...]]:
        if self.dp_size == 1:
            return _subcall_rows([num_tokens], self.capacity)
        dp_metadata = (
            get_forward_context().dp_metadata if is_forward_context_available() else None
        )
        if dp_metadata is None:
            raise RuntimeError(
                f"{self.experts.prefix}: prefill MegaMoE sub-calls at DP{self.dp_size} "
                "need the forward context's DP metadata (EP-uniform launch count)."
            )
        memo = self._memo
        if memo is not None and memo[0] is dp_metadata and memo[1] == num_tokens:
            return memo[2]
        counts = [int(c) for c in dp_metadata.num_tokens_across_dp_cpu.tolist()]
        plan = _subcall_rows(counts, self.capacity)
        self._memo = (dp_metadata, num_tokens, plan)
        return plan

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        experts = self.experts
        n = hidden_states.shape[0]
        nsub, ep_rows = self._plan(n)
        if nsub == 1 and n <= self.capacity:
            return experts(hidden_states, topk_weights, topk_ids)
        if n > nsub * self.capacity:
            raise AssertionError(f"{n} rows exceed {nsub} sub-calls of {self.capacity}")
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:n]
        capture_fn = experts.capture_fn
        if capture_fn is not None:
            capture_fn(topk_ids)  # once, on the whole batch
        out = hidden_states.new_empty((n, experts.hidden_size))
        experts.capture_fn = None
        try:
            for i in range(nsub):
                lo = min(n, i * self.capacity)
                hi = min(n, lo + self.capacity)
                experts(
                    hidden_states[lo:hi],
                    topk_weights[lo:hi],
                    topk_ids[lo:hi],
                    is_padding=None if is_padding is None else is_padding[lo:hi],
                    out=out[lo:hi],
                    ep_tokens=ep_rows[i],
                )
        finally:
            experts.capture_fn = capture_fn
        return out
