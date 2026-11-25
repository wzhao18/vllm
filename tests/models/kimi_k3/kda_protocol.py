# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol-level KDA state checks for the specialized decode's verify tests: what a
verify leaves is compared by the accepted states a later step reads (the stock
checkpoint at slot a - 1, or the record format's slot 0 + slot 1 records folded on
read, ``ops/kda_records.materialize``), not by the raw slots, so a record-form kernel
and the stock verify compare at every accepted count a = 1 .. q."""

import torch

from vllm.models.kimi_k3.nvidia.ops.kda_records import materialize


def accepted_state(state, table, a):
    """[requests, heads, 128, 128]: each request's state after its accepted token
    a - 1 (``table`` [requests, q] its slots), in either format."""
    view = state.clone()
    n = table.shape[0]
    accepted = torch.full((n,), a, dtype=torch.int32, device=state.device)
    materialize(view, table.contiguous(), accepted, n)
    return view[table[:, 0].long()]


def accepted_states(state, table):
    """The accepted states for a = 1 .. q, stacked [q, requests, heads, 128, 128]."""
    return torch.stack(
        [accepted_state(state, table, a) for a in range(1, table.shape[1] + 1)]
    )


def record_input(state, table, accepted):
    """``state`` (stock format: request i's accepted state at slot accepted[i] - 1) in
    the record format, in place: that state at slot 0 and identity records at slot 1
    (k = -0, decay 1, correction 0: the fold leaves the state bit for bit) + the tag,
    so a record-form kernel and the stock verify start from the same accepted states."""
    from vllm.v1.worker.mamba_utils import (
        RECORD_DECAY_ROW,
        RECORD_TAG_BITS,
        RECORD_TAG_ROW,
        RECORD_V_ROW,
    )

    q = table.shape[1]
    rows = table.tolist()
    for i, a in enumerate(accepted.tolist()):
        slot0, slot1 = rows[i][0], rows[i][1]
        if a > 1:
            state[slot0] = state[rows[i][a - 1]].clone()
        words = state[slot1].view(torch.int32)
        words[:, : q - 1] = -(2**31)  # -0.0
        state[slot1][:, RECORD_DECAY_ROW : RECORD_DECAY_ROW + q - 1] = 1.0
        words[:, RECORD_V_ROW : RECORD_V_ROW + q - 1] = 0
        words[:, RECORD_TAG_ROW, 0] = RECORD_TAG_BITS
    return state
