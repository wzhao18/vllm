# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm bench forward: the state-restore snapshot falls back to pinned host memory when the copies don't fit in half
the free device memory, and restores the same bytes either way (strided views included)."""

import pytest
import torch

from vllm.benchmarks.forward.kv_init import restore_state, snapshot_state


def _views(device):
    base = torch.arange(4 * 6 * 8, dtype=torch.float32, device=device).reshape(4, 6, 8)
    # two strided views like touched_kv_views' narrow() on a block dimension
    return base, [base.narrow(1, 1, 2), base.narrow(1, 4, 2)]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("free_bytes, where", [(10**9, "device"), (1, "host")])
def test_snapshot_location_and_exact_restore(device, free_bytes, where):
    base, views = _views(device)
    expected = base.clone()
    snaps, loc = snapshot_state(views, free_bytes=free_bytes)
    assert loc == where
    for _, saved in snaps:
        assert saved.device.type == ("cpu" if where == "host" else device)
    for v in views:
        v.fill_(-1.0)  # the step writes the state
    restore_state(snaps)
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(base, expected)


def test_policy_threshold_is_half_the_free_memory():
    _, views = _views("cpu")
    total = sum(v.numel() * v.element_size() for v in views)
    assert snapshot_state(views, free_bytes=2 * total)[1] == "device"
    assert snapshot_state(views, free_bytes=2 * total - 1)[1] == "host"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_host_snapshot_is_pinned_and_restores_inside_a_graph():
    base, views = _views("cuda")
    expected = base.clone()
    snaps, loc = snapshot_state(views, free_bytes=1)
    assert loc == "host" and all(s.is_pinned() for _, s in snaps)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        restore_state(snaps)
    for v in views:
        v.fill_(-1.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(base, expected)
