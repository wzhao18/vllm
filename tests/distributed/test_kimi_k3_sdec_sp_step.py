# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP specialized decode step (C512: TP4 = DCP4, q4; ``runtime.sp_attention``):
the no-silent-fallback contract of its planner, its startup coverage table and the
stock-drafter declaration. Host-side only (no collective, no kernel): the blocks are
stand-ins with the providers' ``choice`` interface.

NOT covered here (needs 4 / 16 VR200 GPUs and the stock model): the per-layer A/B of
``SpAttentionBlock`` vs the stock layer, the full-step logits A/B, the RecoverSSM
multi-step state and the weight reload."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode import shapes  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    sp_attention,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (  # noqa: E402
    SpecializedDecode,
)

ROWS = tuple(range(4, 513, 4))  # C512: 128 FULL graphs, q4
GENERIC = shapes.Choice(shapes.Path.GENERIC, note="stand-in")


class Provider:
    """A provider's ``choice`` over the rows it prepared (raises elsewhere)."""

    def __init__(self, family, rows):
        self.family, self.rows = family, frozenset(rows)

    def choice(self, r):
        if r not in self.rows:
            raise shapes.Unsupported(self.family, r, "not prepared")
        return GENERIC


class Block:
    def __init__(self, kind, rows=ROWS, core_rows=None):
        self.kind = kind
        self.projection = Provider(f"{kind}_in", rows)
        self.core = Provider(kind, rows if core_rows is None else core_rows)
        self.oproj = Provider("oproj_rs", rows)

    choices = sp_attention.SpAttentionBlock.choices


def stand_in_step(monkeypatch, blocks, rows=ROWS, method="synthetic"):
    step = object.__new__(sp_attention.SpStep)
    step.q, step.rows, step.layers = 4, rows, ()
    step.ctx = SimpleNamespace(ep=16)
    step.comm = Provider("ag", rows)
    step.blocks = dict(enumerate(blocks))
    step.moe, step.dense = True, False
    spec = SimpleNamespace(rejection_sample_method=method)
    config = SimpleNamespace(speculative_config=spec)
    monkeypatch.setattr(sp_attention, "get_current_vllm_config", lambda: config)
    return step


def test_tp16_path_has_no_sp_step():
    assert SpecializedDecode.sp is None


def test_drafter_declared_stock_at_tp4_only():
    tables = sp_attention.check_stock_drafter(4)
    assert "FUSED_MARKOV" in tables and "PROJECTIONS" in tables
    with pytest.raises(shapes.Unsupported, match="TP16 entries"):
        sp_attention.check_stock_drafter(16)


def test_coverage_every_captured_row(monkeypatch):
    step = stand_in_step(monkeypatch, [Block("dsa"), Block("kda")])
    table = step.plan_coverage()
    table.assert_complete(ROWS)
    families = set(table.plan[512])
    assert {"ag", "dsa_in", "dsa", "kda_in", "kda", "oproj_rs"} <= families
    assert table.plan[4]["verify"].path is shapes.Path.DECLARED
    assert table.plan[4]["drafter"].path is shapes.Path.DECLARED
    assert table.plan[4]["moe"].path is shapes.Path.DECLARED
    assert all(c.path is not shapes.Path.STOCK for c in table.plan[324].values())


def test_coverage_raises_for_an_uncovered_row(monkeypatch):
    kda = Block("kda", core_rows=[r for r in ROWS if r != 324])
    step = stand_in_step(monkeypatch, [Block("dsa"), kda])
    with pytest.raises(shapes.Unsupported, match="kda at 324"):
        step.plan_coverage()


def test_coverage_rejects_a_stock_family():
    table = shapes.Coverage(4)
    table.specialize(8, {"dsa": shapes.Choice(shapes.Path.STOCK, note="S2")})
    with pytest.raises(RuntimeError, match="dsa=stock"):
        table.assert_complete((8,))
    table = shapes.Coverage(4)
    table.specialize(8, {"dsa": GENERIC})
    table.declare(8, "oproj_rs", "stock")
    with pytest.raises(RuntimeError, match="not a declared stock"):
        table.assert_complete((8,))


def planner(monkeypatch, capturing, mode, spec_decodes=1):
    step = stand_in_step(monkeypatch, [])
    md = SimpleNamespace(num_spec_decodes=spec_decodes)
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode, attn_metadata={"x": md}, slot_mapping={}
    )
    monkeypatch.setattr(
        sp_attention.torch.cuda, "is_current_stream_capturing", lambda: capturing
    )
    monkeypatch.setattr(sp_attention, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sp_attention, "get_forward_context", lambda: context)
    return step


def positions(rows):
    return SimpleNamespace(shape=(rows,))


def test_plan_stock_outside_full_captures(monkeypatch):
    step = planner(monkeypatch, False, CUDAGraphMode.FULL)
    assert step.plan(positions(8), None) is None  # eager / warmup / profiling
    step = planner(monkeypatch, True, CUDAGraphMode.PIECEWISE)
    assert step.plan(positions(8), None) is None


def test_plan_full_capture_never_stock(monkeypatch):
    step = planner(monkeypatch, True, CUDAGraphMode.FULL)
    assert step.plan(positions(8), None).q == 4
    with pytest.raises(shapes.Unsupported, match="outside the captured rows"):
        step.plan(positions(516), None)
    step = planner(monkeypatch, True, CUDAGraphMode.FULL, spec_decodes=0)
    with pytest.raises(shapes.Unsupported, match="not a q4 verify"):
        step.plan(positions(8), None)
