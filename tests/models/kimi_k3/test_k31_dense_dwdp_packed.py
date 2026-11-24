# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense DWDP moves whole weight storages, including packed projections and every
view of them, and re-points the views in place (dense_dwdp._plan_layer /
_repoint)."""

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from vllm.models.kimi_k3.nvidia import dense_dwdp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Holder(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(w, requires_grad=False))


class _Gemm:  # stands in for a fused prefill GEMM that keeps weight views
    def __init__(self, w):
        self.weight = w


_Gemm.__module__ = "vllm.models.kimi_k3.nvidia.test_stub"  # the walk follows vllm objects


class _Layer(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(seed)
        packed = torch.randn(64, 32, device="cuda", generator=g)
        self.a = nn.Linear(32, 48, bias=False, device="cuda")
        self.b = nn.Linear(32, 16, bias=False, device="cuda")
        self.own = nn.Parameter(torch.randn(8, 8, device="cuda", generator=g))
        holder = _Holder(packed)
        self.a.weight.data = holder.weight.data[:48]
        self.b.weight.data = holder.weight.data[48:].t().contiguous().t()
        self._packed = (holder, 48)  # not a submodule: only the walk finds it
        self._gemm = _Gemm(self.a.weight.data[8:])  # a detached view


def _layers(n=3):
    return [_Layer(i) for i in range(n)]


def _plans(layers):
    """Per layer after the first: its load-time plan, and the shared storages."""
    stops = dense_dwdp._layer_stops(layers)
    shared = dense_dwdp._shared_storages(layers, stops)
    plans = [
        dense_dwdp._plan_layer(layer, stop, set(), shared)
        for layer, stop in zip(layers[1:], stops[1:])
    ]
    return plans, sum(st.nbytes() for st in shared.values())


def test_plan_groups_storages_with_all_views(monkeypatch):
    monkeypatch.setattr(dense_dwdp, "MIN_BYTES", 1)
    layers = _layers()
    plans, resident = _plans(layers)
    assert len(plans) == 2 and resident == 0
    for (items, total), layer in zip(plans, layers[1:]):
        by_ptr = {st.data_ptr(): views for st, views, _ in items}
        packed = layer._packed[0].weight
        views = by_ptr[packed.untyped_storage().data_ptr()]
        assert any(v is packed for v in views)
        assert any(v is layer.a.weight for v in views)
        assert any(v is layer._gemm.weight for v in views)
        assert layer.b.weight.untyped_storage().data_ptr() in by_ptr
        assert layer.own.untyped_storage().data_ptr() in by_ptr
        assert total >= sum(st.nbytes() for st, _, _ in items)


def test_storage_shared_by_two_layers_stays_resident(monkeypatch):
    monkeypatch.setattr(dense_dwdp, "MIN_BYTES", 1)
    layers = _layers()
    layers[2].own = layers[1].own  # tied across layers
    plans, resident = _plans(layers)
    ptr = layers[1].own.untyped_storage().data_ptr()
    for items, _ in plans:
        assert all(st.data_ptr() != ptr for st, _, _ in items)
    assert resident == layers[1].own.untyped_storage().nbytes()


def test_repoint_keeps_values_and_frees_old_storage(monkeypatch):
    monkeypatch.setattr(dense_dwdp, "MIN_BYTES", 1)
    layer = _layers(2)[1]
    before = {
        "a": layer.a.weight.clone(),
        "b": layer.b.weight.clone(),
        "gemm": layer._gemm.weight.clone(),
        "packed": layer._packed[0].weight.clone(),
    }
    plans, _ = _plans([_layers(1)[0], layer])
    ((items, total),) = plans
    del plans
    blob = torch.zeros(total + 4096, dtype=torch.uint8, device="cuda")[4096:]
    refs = []
    for st, views, off in items:
        blob[off : off + st.nbytes()].copy_(dense_dwdp._storage_u8(st))
        for t in views:
            dense_dwdp._repoint(
                t, blob.untyped_storage(), (blob.storage_offset() + off)
                // t.element_size()
            )
        refs.append(StorageWeakRef(st))
    del items, st, views
    assert all(r.expired() for r in refs)
    target = blob.untyped_storage().data_ptr()
    for t in (layer.a.weight, layer.b.weight, layer._gemm.weight,
              layer._packed[0].weight):  # fmt: skip
        assert t.untyped_storage().data_ptr() == target
    torch.testing.assert_close(layer.a.weight, before["a"], rtol=0, atol=0)
    torch.testing.assert_close(layer.b.weight, before["b"], rtol=0, atol=0)
    torch.testing.assert_close(layer._gemm.weight, before["gemm"], rtol=0, atol=0)
    torch.testing.assert_close(
        layer._packed[0].weight, before["packed"], rtol=0, atol=0
    )
