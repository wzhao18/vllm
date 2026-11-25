# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``RubinMegaMoE.poll_route_errors``: the host mirror of the sticky route-id flag is
copied on a side stream (never a copy node on the forward stream), an error still
surfaces on the next poll, and no second copy is queued while one is in flight."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.megamoe_rubin.config import RouteIdError
from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fake():
    rt = SimpleNamespace(
        route_error=torch.zeros(2, dtype=torch.int64, device="cuda"),
        _route_error_host=torch.zeros(2, dtype=torch.int64, pin_memory=True),
        _route_error_event=None,
        _route_error_stream=None,
        problem=SimpleNamespace(num_experts=112),
    )
    rt._raise_route_error = RubinMegaMoE._raise_route_error.__get__(rt)
    rt.poll = RubinMegaMoE.poll_route_errors.__get__(rt)
    return rt


def test_mirror_runs_on_a_side_stream_and_clean_flag_never_raises():
    rt = _fake()
    rt.poll()
    torch.cuda.synchronize()
    assert rt._route_error_stream is not None
    assert rt._route_error_stream != torch.cuda.current_stream()
    rt.poll()
    torch.cuda.synchronize()
    rt.poll()


def test_error_written_by_the_forward_stream_surfaces_on_a_later_poll():
    rt = _fake()
    rt.poll()
    torch.cuda.synchronize()
    # The forward stream sets the flag; the next poll's mirror is ordered after it.
    rt.route_error.copy_(torch.tensor([1, 999], dtype=torch.int64))
    rt.poll()
    torch.cuda.synchronize()
    with pytest.raises(RouteIdError):
        rt.poll()


def test_no_second_copy_while_the_mirror_is_in_flight():
    rt = _fake()
    rt.poll()
    torch.cuda.synchronize()
    side = rt._route_error_stream
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)  # keep the side stream busy
    rt.poll()  # queues a mirror behind the sleep
    assert not rt._route_error_event.query()
    rt.poll()  # previous mirror not landed: returns without queuing or raising
    assert not side.query()
    torch.cuda.synchronize()
    assert rt._route_error_event.query()
