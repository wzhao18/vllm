# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The deferred-verify KDA state protocol outside the specialized decode's record
kernels (``specialized_decode/cutedsl/kda/verify_defer.py``).

Under the protocol a verify stores the state after its token 0 at the request's slot
0 and its records of tokens 1 .. q - 1 (keys, decays, corrections + a format tag) at
slot 1's page; the accepted state is folded on read (``vllm.v1.worker.mamba_utils``
``record_fold_head``: the token loop's two instructions, bit for bit the stock
checkpoint). The stock Triton verify (mixed prefill + decode, partial drafts, eager,
PIECEWISE, batches without prepared providers, the specialized decode's STOCK rows)
reads and writes the same format:

* ``materialize``: per request and head with ``accepted >= 2``, the accepted state into
  slot 0: folded when slot 1 is tagged, slot ``accepted - 1``'s page otherwise (a
  stock-format slot: an untagged slot 1);
* ``fused_recurrent_kda(..., records=True)`` then reads slot 0 and writes slot 0 after
  token 0 + slot 1's records and tag.

The align copies (prefix caching's block-boundary pre-copy and save) fold the same
way (``mamba_utils._copy_record_state_block``)."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.mamba_utils import (
    record_copy_head,
    record_fold_head,
    record_tagged,
)


@triton.jit
def _materialize_kernel(
    state,
    state_indices,
    num_accepted_tokens,
    stride_state_token,
    stride_indices_seq,
    H: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    accepted = tl.load(num_accepted_tokens + i_n)
    if accepted < 2:
        return
    row = state_indices + i_n * stride_indices_seq
    slot0 = tl.load(row).to(tl.int64)
    slot1 = tl.load(row + 1).to(tl.int64)
    stock = tl.load(row + accepted - 1).to(tl.int64)
    if (slot0 <= 0) | (slot1 <= 0):
        return
    head = i_h * 128 * 128
    state0 = state + slot0 * stride_state_token + head
    records = state + slot1 * stride_state_token + head
    if record_tagged(records):
        record_fold_head(state0, records, state0, accepted - 1)
    elif stock > 0:
        record_copy_head(state + stock * stride_state_token + head, state0)


def materialize(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_requests: int,
) -> None:
    """Each request's accepted state into its slot 0 (module docstring) of the FP32
    ``state`` [slots, heads, 128, 128]; ``state_indices`` [>= num_requests, q] the
    requests' slots, ``num_accepted_tokens`` their accepted counts."""
    if num_requests == 0:
        return
    heads = state.shape[1]
    assert state.dtype == torch.float32 and tuple(state.shape[2:]) == (128, 128)
    assert state.stride()[1:] == (128 * 128, 128, 1)
    assert state_indices.dim() == 2 and state_indices.stride(1) == 1
    assert state_indices.shape[1] >= 2
    _materialize_kernel[(num_requests, heads)](
        state,
        state_indices,
        num_accepted_tokens,
        state.stride(0),
        state_indices.stride(0),
        H=heads,
    )
