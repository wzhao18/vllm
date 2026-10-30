# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode on CPU: the routing of its wide verify steps (above
the tuned rows) between the specialized providers and the stock seams."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("flashinfer")

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.dense import (  # noqa: E402
    PreparedDenseMLP,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import (  # noqa: E402
    PreparedMoE,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (  # noqa: E402
    SpecializedDecode,
    Step,
)

HIDDEN = 16  # the steps below never read hidden values: any width


def wide_step(b, *, wide_moe):
    """A three-layer wide step (a dense layer, then two MoE layers) whose AttnRes
    sites run the stock op (``stock_attn_res``), on mocked providers. ``wide_moe``:
    the MoE providers serve the rows (each tail's ring consumer runs the next
    AttnRes site), else the stock MoE layer does (seam S4)."""
    decode = object.__new__(SpecializedDecode)
    layers = [
        SimpleNamespace(layer_idx=i, self_attn=SimpleNamespace(), mlp=MagicMock())
        for i in range(3)
    ]
    for layer in layers:
        layer.mlp.side_effect = lambda rows: torch.zeros_like(rows)
    decode.layers = tuple(layers)
    decode.verify_rows = lambda step, b: None
    decode.head_batches = decode.aux_layers = ()
    decode.dense_rings = {}
    decode.hidden = tuple(torch.zeros(b, HIDDEN) for _ in range(2))
    decode.prefix = torch.zeros(b, HIDDEN)
    decode.blocks = torch.zeros(b, 2, HIDDEN)
    decode.wide = decode.stock_attn_res = frozenset({b})
    decode.moe_batches = (b,) if wide_moe else ()
    decode.before = {i: MagicMock() for i in range(3)}
    decode.before_stock = {i: MagicMock() for i in range(3)}
    decode.final, decode.final_stock = MagicMock(), MagicMock()
    for stock in (*decode.before_stock.values(), decode.final_stock):
        stock.run.side_effect = lambda delta, *args: torch.zeros_like(delta)
    decode.attention = {i: MagicMock() for i in range(3)}
    for attention in decode.attention.values():
        attention.run.side_effect = lambda rows, *args, **kwargs: torch.zeros_like(rows)
    decode.feed_forward = {0: MagicMock(spec=PreparedDenseMLP)}
    decode.feed_forward |= {i: MagicMock(spec=PreparedMoE) for i in (1, 2)}
    for moe in decode.feed_forward.values():
        moe.run.side_effect = lambda rows, target, **kwargs: target
    decode.fused_before = {1: decode.before[2], 2: decode.final}
    decode.prefetch, decode.kda_side = MagicMock(), MagicMock()
    decode.backbone = SimpleNamespace(norm=lambda rows, residual: rows)
    step = Step(q=8, metadata=[None] * 3, slot_mapping={}, positions=None)
    return decode, step


@pytest.mark.parametrize("wide_moe", [True, False], ids=["moe-tail-ring", "stock-moe"])
def test_stock_attn_res_sites_run_once(wide_moe):
    """Above 256 rows (base STOCK_ROWS) a site that the previous MoE tail's ring
    consumer already ran is skipped: the stock op on its output would apply the
    AttnRes a second time. Sites after a stock MoE layer still run the stock op."""
    b = 264
    decode, step = wide_step(b, wide_moe=wide_moe)
    out = decode.run_path(step, None, torch.zeros(b, HIDDEN))
    ran = [decode.before_stock[i].run.call_count for i in range(3)]
    assert not any(before.run.called for before in decode.before.values())
    if wide_moe:
        # Layer 2's site and the final one are the MoE tails' ring consumers.
        assert ran == [1, 1, 0] and not decode.final_stock.run.called
        tail_rows = decode.feed_forward[1].run.call_args.args[1]
        assert decode.attention[2].run.call_args.args[0] is tail_rows
        assert out is decode.feed_forward[2].run.call_args.args[1]
    else:
        assert ran == [1, 1, 1] and decode.final_stock.run.call_count == 1
