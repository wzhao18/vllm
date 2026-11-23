# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB balancedness logging reads the stats back without a host sync: ``submit``
issues only a pinned non-blocking copy, and ``flush`` logs each line once its copy
has landed, in submission order, with the values the step computed."""

import pytest
import torch

from vllm.distributed.eplb.eplb_state import BalancednessLog

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_submit_does_not_sync_and_flush_logs_in_order(monkeypatch):
    import vllm.distributed.eplb.eplb_state as eplb_state

    lines = []
    monkeypatch.setattr(
        eplb_state.logger, "info", lambda fmt, *args: lines.append(fmt % args)
    )
    log = BalancednessLog()
    stats = [torch.tensor([30.0, 40.0], device="cuda"),
             torch.tensor([10.0, 0.0], device="cuda")]  # fmt: skip
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        # A long kernel ahead of the copies: nothing may wait for it.
        x = torch.randn(4096, 4096, device="cuda")
        for _ in range(8):
            x = x @ x
        log.submit(stats[0] * 1, (50, "m", 950))
        log.submit(stats[1] * 1, (100, "m", 900))
        log.flush()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    log.flush()
    assert lines == [
        "EPLB step: 50 for model m: avg_tokens=30.00, max_tokens=40, "
        "balancedness=0.7500, steps until the next rearrangement: 950",
        "EPLB step: 100 for model m: avg_tokens=10.00, max_tokens=0, "
        "balancedness=0.0000, steps until the next rearrangement: 900",
    ]
    log.flush()  # nothing left
    assert len(lines) == 2


def test_force_flush_waits_for_the_copy(monkeypatch):
    import vllm.distributed.eplb.eplb_state as eplb_state

    lines = []
    monkeypatch.setattr(
        eplb_state.logger, "info", lambda fmt, *args: lines.append(args)
    )
    log = BalancednessLog()
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(8):
        x = x @ x
    log.submit(torch.tensor([3.0, 4.0], device="cuda"), (7, "d", 1))
    log.flush(force=True)
    assert lines == [(7, "d", 3.0, 4.0, 0.75, 1)]
