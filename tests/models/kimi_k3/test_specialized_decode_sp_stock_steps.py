# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, SP topology: serving steps that ``SpStep.plan`` does not
plan (eager, PIECEWISE, mixed prefill) run the stock forward; ``SpecializedDecode.plan``
counts each per cause (NQ-NOFALLBACK, no silent fallback) and FULL captures never count.
CPU-only logic test (the SP step and the forward context are stand-ins)."""

import types

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.specialized_decode import shapes
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import decode


class _Sp:
    def __init__(self, step=None):
        self.step = step

    def plan(self, positions, intermediate_tensors):
        return self.step


def _decode(step=None):
    d = decode.SpecializedDecode.__new__(decode.SpecializedDecode)
    d.sp = _Sp(step)
    d.stock_steps = shapes.StockSteps("decode", 0)
    return d


@pytest.fixture
def serving(monkeypatch):
    monkeypatch.setattr(shapes.StockSteps, "serving", staticmethod(lambda: True))


def _context(monkeypatch, mode, prefills=0):
    md = types.SimpleNamespace(num_prefills=prefills)
    ctx = types.SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata={"l0": md})
    monkeypatch.setattr(decode, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode, "get_forward_context", lambda: ctx)


@pytest.mark.parametrize(
    "mode,prefills,cause",
    [
        (CUDAGraphMode.PIECEWISE, 0, "PIECEWISE graph (non-uniform verify shape)"),
        (CUDAGraphMode.NONE, 1, "prompt rows (mixed prefill + decode)"),
        (CUDAGraphMode.NONE, 0, "eager step (outside a FULL graph capture)"),
    ],
)
def test_unplanned_serving_steps_are_counted(
    monkeypatch, serving, mode, prefills, cause
):
    _context(monkeypatch, mode, prefills)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = _decode(step=None)
    assert d.plan(torch.zeros(197, dtype=torch.long), None) is None
    assert d.stock_steps.counts == {cause: 1}


def test_planned_steps_and_captures_do_not_count(monkeypatch, serving):
    _context(monkeypatch, CUDAGraphMode.FULL)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = _decode(step=object())
    assert d.plan(torch.zeros(200, dtype=torch.long), None) is d.sp.step
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    d2 = _decode(step=None)
    d2.plan(torch.zeros(200, dtype=torch.long), None)
    assert d.stock_steps.counts == {} and d2.stock_steps.counts == {}
