# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 replicated EP_GROUP decode KDA core on Rubin (compute capability 10.7):
``cutedsl.kda.sp.SpKdaCore`` in "none" mode (no RecoverSSM records: the plain
``PreparedKda`` verify), built under an EP_GROUP context (``Context.ep_group``, not
the SP step) at the TP4 group's 24 heads (gate mode 0) and the TP8 group's 12 heads
(``sp.PLAIN_VERIFY_GATE``: TP8's gate mode 1), FP32 state, DSpark q4 / q8, the rows
the replicated step captures (R = n q <= 64), vs the stock verify of the same layer
(``KimiK3DeltaAttention._forward``, native Triton: ``causal_conv1d_update`` ->
``fused_recurrent_kda`` -> ``o_norm``) with the thresholds of
``test_k3_sm107_kda_loop_gpu.py`` (``assert_stock``): the conv window bit for bit,
the states within FP32 reduction order, the BF16 outputs within relerr 2e-4.

Also: the core refuses a context outside the EP_GROUP mode, a BF16 state and
q outside 2..8 (no speculation: the stock plain decode is the CUDA fused_kda_decode's
fast math, which this core does not reproduce).

One GPU; a few minutes of CuTe compiles on a cold cache."""

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")  # the decode's conv layout

from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_kda_loop_gpu import (  # noqa: E402, E501
    DEVICE,
    caches,
    kda_attention,
    pitch_of,
    projected_rows,
    spec_inputs,
)
from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_kda_stock_gpu import (  # noqa: E402, E501
    assert_stock,
    clone,
    stock_run,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda import (  # noqa: E402
    sp,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (  # noqa: E402
    Context,
    Unsupported,
)
from vllm.platforms import current_platform  # noqa: E402

_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY


def context(heads, q, **overrides):
    g = 96 // heads
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    fields = dict(
        arch="rubin", tp=g, rank=0, dcp=g, sms=sms, q=q, max_rows=64,
        max_context=1 << 20, bf16_state=False, dp=2, ep=2 * g,
        heads_per_rank=heads, ep_group=True,
    )  # fmt: skip
    return Context(**(fields | overrides))


def layer(heads, q, seed=0):
    """``kda_attention``'s fields plus those ``SpKdaCore`` reads from a stock
    KimiK3DeltaAttention of ``heads`` local heads in a q-token verify deployment."""
    attention = kda_attention(seed=seed, heads=heads)
    attention.head_dim, attention.conv_size, attention.spec_query_len = 128, 4, q
    attention.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    attention.in_proj_qkvgfab = SimpleNamespace(
        weight=torch.empty(pitch_of(heads), 8, device="meta")
    )
    attention.recoverssm_records = None
    return attention


# (heads, q, requests): R = requests x q <= 64. 24 heads: KdaSplitStock at 1-8
# requests, the stock-math wide verify (FP32_VERIFY_WIDE) at q4 x 16; 12 heads:
# KdaSplitStock (gate mode 1) everywhere.
SHAPES = [
    *(pytest.param(24, 4, n, id=f"h24-q4-n{n}") for n in (1, 4, 8, 16)),
    *(pytest.param(24, 8, n, id=f"h24-q8-n{n}") for n in (1, 8)),
    *(pytest.param(12, 4, n, id=f"h12-q4-n{n}") for n in (1, 4, 8, 16)),
    *(pytest.param(12, 8, n, id=f"h12-q8-n{n}") for n in (1, 2, 8)),
]


@pytest.mark.parametrize("heads,q,requests", SHAPES)
def test_plain_core_matches_stock_verify(monkeypatch, heads, q, requests):
    rows = requests * q
    attention = layer(heads, q)
    core = sp.SpKdaCore(attention, (rows,), records="none", ctx=context(heads, q))
    gate = 1 if heads == 12 else 0
    assert sp.plain_gate_mode(heads) == gate
    assert set(core.plain.policies.values()) == {(pitch_of(heads), gate)}
    assert core.choice(rows).path.value == "generic"
    slots = rows + 1
    base = caches(slots, q, heads=heads)
    table, accepted = spec_inputs(requests, q, slots)
    projected = projected_rows(rows, pitch=pitch_of(heads))
    core.prepare_caches(list(base))
    got_caches = clone(base)
    got = torch.empty(rows, 128 * heads, dtype=torch.bfloat16, device=DEVICE)
    metadata = SimpleNamespace(
        spec_state_indices_tensor=table, num_accepted_tokens=accepted
    )
    core.run(projected, got, caches=list(got_caches), metadata=metadata)
    torch.cuda.synchronize()
    want, want_caches = stock_run(
        attention, projected, base, table, accepted, q, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.view(-1).long())


def test_plain_core_refusals():
    """Outside the EP_GROUP mode, a BF16 state, or no speculation: refused at build
    (never a stock fallback)."""
    with pytest.raises(Unsupported, match="EP_GROUP decode only"):
        ctx = context(12, 4, ep_group=False)
        sp.SpKdaCore(layer(12, 4), (16,), records="none", ctx=ctx)
    with pytest.raises(Unsupported, match="FP32 SSM state"):
        ctx = context(12, 4, bf16_state=True)
        sp.SpKdaCore(layer(12, 4), (16,), records="none", ctx=ctx)
    with pytest.raises(Unsupported, match="verify length q1"):
        sp.SpKdaCore(layer(12, 1), (4,), records="none", ctx=context(12, 1))
    with pytest.raises(Unsupported, match="not whole requests"):
        sp.SpKdaCore(layer(24, 4), (6,), records="none", ctx=context(24, 4))
