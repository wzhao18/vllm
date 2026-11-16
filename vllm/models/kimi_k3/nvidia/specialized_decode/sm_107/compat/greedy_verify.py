# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification kernel of the specialized decode's verify path.

Vendored unchanged (kernel and wrapper) from the source branch's
``vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`` (``greedy_verify``),
which the base's rejection sampler does not have. The runner-side hook that
calls it (``RejectionSampler.can_verify_greedy`` / ``verify_greedy``) is wiring
outside this package.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32


@triton.jit
def _greedy_verify_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs]
    num_rejected_ptr,
    # [num_logits]
    target_argmax_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    # One program per request, all its logits rows at once: gather_draft_sampled,
    # the greedy branch of _rejection_kernel (plus what _resample_kernel and
    # _insert_resampled_kernel do for a greedy request) and
    # get_num_sampled_and_rejected, with the target argmax of each row given.
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_logits = end_idx - start_idx
    num_draft_tokens = num_logits - 1
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)

    i = tl.arange(0, BLOCK)
    is_draft = i < num_draft_tokens
    # Row i verifies the draft token that is the input token of row i + 1.
    # Draft slots of a step that starts within the prefill are placeholders
    # (e.g. the padded first step after a P/D remote KV load): rejected.
    next_token_idx = tl.load(logits_indices_ptr + start_idx + i + 1, mask=is_draft)
    draft = tl.load(input_ids_ptr + next_token_idx, mask=is_draft, other=0)
    draft_pos = tl.load(positions_ptr + next_token_idx, mask=is_draft, other=0)
    draft_local_pos = tl.load(
        expanded_local_pos_ptr + start_idx + i + 1, mask=is_draft, other=0
    )
    is_placeholder = (draft_local_pos > 0) & (draft_pos - draft_local_pos < prefill_len)
    draft = tl.where(is_placeholder, -1, draft).to(tl.int64)
    is_valid_draft = draft >= 0
    draft = tl.maximum(0, draft)
    target_argmax = tl.load(
        target_argmax_ptr + start_idx + i, mask=i < num_logits, other=0
    ).to(tl.int64)
    if SYNTHETIC_MODE:
        seed = tl.load(seed_ptr + req_state_idx)
        token_idx = tl.load(logits_indices_ptr + start_idx + i, mask=is_draft)
        pos = tl.load(positions_ptr + token_idx, mask=is_draft, other=0)
        u = tl_rand32(seed, pos, includes_zero=False)
        rate = tl.load(synthetic_conditional_rates_ptr + i, mask=is_draft, other=0.0)
        accepted = u < rate
    else:
        accepted = target_argmax == draft
    accepted &= is_valid_draft
    # The accepted length is the index of the first rejected draft.
    accepted_length = tl.min(tl.where(is_draft & ~accepted, i, num_draft_tokens))
    # Accepted drafts, then the target argmax at the first rejection (the bonus
    # token when every draft was accepted).
    tl.store(
        sampled_ptr + req_idx * sampled_stride + i,
        tl.where(i < accepted_length, draft, target_argmax),
        mask=i <= accepted_length,
    )
    num_sampled = accepted_length + 1
    num_rejected = num_logits - num_sampled
    seq_len = tl.load(seq_lens_ptr + req_idx)
    is_chunked_prefilling = seq_len < prefill_len
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_sampled_ptr + req_idx, num_sampled.to(tl.int32))
    tl.store(num_rejected_ptr + req_idx, num_rejected.to(tl.int32))


def greedy_verify(
    # [num_logits]
    target_argmax: torch.Tensor,
    # [num_tokens]
    input_ids: torch.Tensor,
    # [num_tokens]
    positions: torch.Tensor,
    # [num_logits]
    logits_indices: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [max_num_reqs]
    prefill_len: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """gather_draft_sampled + rejection_sample + get_num_sampled_and_rejected
    for a batch of greedy (temperature 0) requests, in one kernel, from the
    target argmax of each logits row instead of the full-vocab logits.
    Returns (sampled, num_sampled, num_rejected)."""
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = input_ids.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    num_rejected = torch.empty_like(num_sampled)
    _greedy_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        num_rejected,
        target_argmax,
        input_ids,
        positions,
        logits_indices,
        expanded_local_pos,
        cu_num_logits,
        idx_mapping,
        seq_lens,
        prefill_len,
        seed,
        synthetic_conditional_rates,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        BLOCK=triton.next_power_of_2(num_speculative_steps + 1),
        num_warps=1,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return sampled, num_sampled, num_rejected
