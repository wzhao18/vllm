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
