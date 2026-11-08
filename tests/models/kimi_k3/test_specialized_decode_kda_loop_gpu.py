# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode plain KDA verify with an FP32 recurrent state at TP16
(6 heads per rank), the q4 / q8 rows of ``kda.prepared.FP32_VERIFY_WIDE``, on Rubin:

* those rows run ``KdaVerifyLoopStock`` (persistent, stock math) and match the stock
  verify of the layer (``KimiK3DeltaAttention._forward``, native Triton backend:
  ``causal_conv1d_update`` -> ``fused_recurrent_kda`` -> ``o_norm``; FlashInfer's
  packed decode does not take 6 heads) to FP32 reduction order: the conv window bit
  for bit, every token's checkpoint within FP32 order, the BF16 outputs with at most
  one in 10^3 elements off by BF16 rounding;
* ``KdaVerifyPacked`` and ``KdaVerifyLoop`` (``KdaVerifyWide``'s math: KdaSplit's)
  equal ``KdaVerifyWide`` bit for bit;
* padding requests (slot -1) leave the caches untouched and output zeros, and the
  live requests' results do not depend on them.

One GPU; a few minutes of CuTe compiles on a cold cache."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")  # the decode's conv layout

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.models.kimi_k3.nvidia import kda as kda_module  # noqa: E402
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention  # noqa: E402
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    providers,
)
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    FusedRMSNormGated,
)

DEVICE = torch.device("cuda", 0)
RUBIN = (10, 7)
HEADS, DIM = 6, 768  # TP16: 6 of 96 KDA heads per rank
PITCH = 3216  # packed in-projection rows: Q | K | V | gate | F_A 128 | beta 6 | pad
PREFIX = "model.layers.0.self_attn"

Module = SimpleNamespace

# The FP32_VERIFY_WIDE rows (q, requests): one and several items per SM.
LOOP_SHAPES = [
    pytest.param(4, 12, id="q4-B12"),
    pytest.param(4, 35, id="q4-B35"),
    pytest.param(4, 41, id="q4-B41"),
    pytest.param(4, 128, id="q4-B128"),
    pytest.param(4, 512, id="q4-B512"),
    pytest.param(8, 12, id="q8-B12"),
    pytest.param(8, 65, id="q8-B65"),
    pytest.param(8, 128, id="q8-B128"),
]


def kda_attention(seed=0):
    """The fields ``PreparedKda`` and the stock verify read from a TP16 KDA layer
    (stock dtypes: FP32 conv weight, A_log, dt bias; BF16 F_B; the norm weight)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEVICE) * scale).to(dtype)

    with set_current_vllm_config(VllmConfig()):  # o_norm picks its forward here
        norm = FusedRMSNormGated(128, eps=1e-5, activation="sigmoid").to(DEVICE)
    with torch.no_grad():
        norm.weight.copy_(1 + randn(128, scale=0.1))
    return Module(
        conv1d=Module(weight=randn(3 * DIM, 1, 4, scale=0.5), bias=None),
        dt_bias=randn(DIM, scale=0.5),
        A_log=randn(HEADS, scale=0.5),
        f_b_proj=Module(weight=randn(DIM, 128, dtype=torch.bfloat16, scale=128**-0.5)),
        o_norm=norm,
        gate_lower_bound=-5.0,
    )


def caches(slots, q, seed=1):
    """The DS conv window [slot, 3 dim, q + 2] (BF16) and the FP32 state."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    conv = torch.randn(slots, 3 * DIM, q + 2, generator=g, device=DEVICE)
    state = torch.randn(slots, HEADS, 128, 128, generator=g, device=DEVICE) * 0.1
    return conv.to(torch.bfloat16), state


def spec_inputs(requests, q, slots, seed=2):
    """Distinct slots [requests, q] (> 0) and accepted counts in 1..q."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(slots - 1, generator=g)[: requests * q] + 1
    table = order.view(requests, q).to(torch.int32).to(DEVICE)
    accepted = torch.randint(1, q + 1, (requests,), generator=g)
    return table, accepted.to(torch.int32).to(DEVICE)


def projected_rows(rows, seed=4):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    values = torch.randn(rows, PITCH, generator=g, device=DEVICE)
    return values.to(torch.bfloat16)


def bits(tensor):
    return tensor.view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def assert_bitwise(name, got, want):
    assert got.shape == want.shape, name
    diff = bits(got) != bits(want)
    assert not diff.any(), f"{name}: {int(diff.sum())} of {diff.numel()} differ"


def core_of(attention, requests, q, kind, base, monkeypatch):
    """``PreparedKda`` at (requests, q) compiled with plain verify kind ``kind``."""
    index = prepared.WIDE_NAMES.index(kind)
    prepared.executable.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(prepared, "wide_kind", lambda *args, **kwargs: index)
        core = prepared.PreparedKda(attention, {(requests, q): (PITCH, 0)}, True)
        core.prepare_caches(base)
    prepared.executable.cache_clear()
    return core


def run(core, projected, base, table, accepted):
    state = tuple(t.clone() for t in base)
    out = torch.empty(projected.shape[0], DIM, dtype=torch.bfloat16, device=DEVICE)
    core.run(projected, out, caches=state, slots=table, accepted=accepted)
    torch.cuda.synchronize()
    return out, state


def stock_verify(attention, projected, base, table, accepted, q, monkeypatch):
    """Seam S1: the stock layer's verify (``_forward``, native Triton) from the
    in-projection rows, as ``providers.Attention.stock_kda`` runs it."""
    requests = table.shape[0]
    rows = requests * q
    metadata = KimiK3KDAMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=requests, num_spec_decode_tokens=rows,
        num_actual_tokens=rows,
        spec_query_start_loc=torch.arange(
            0, rows + 1, q, dtype=torch.int32, device=DEVICE
        ),
        spec_state_indices_tensor=table, num_accepted_tokens=accepted,
    )  # fmt: skip
    context = Module(attn_metadata={PREFIX: metadata})
    monkeypatch.setattr(kda_module, "get_forward_context", lambda: context)
    state = tuple(t.clone() for t in base)
    layer = Module(
        prefix=PREFIX, kv_cache=state, kda_spec_decode_backend="native",
        kda_decode_backend="triton", kda_prefill_backend="triton",
        decode_conv1d_weight=None, decode_norm_weight=None, conv1d=attention.conv1d,
        local_projection_size=DIM, local_num_heads=HEADS, head_dim=128,
        use_recoverssm=False, A_log=attention.A_log, dt_bias=attention.dt_bias,
        gate_lower_bound=attention.gate_lower_bound, o_norm=attention.o_norm,
    )  # fmt: skip
    fb = attention.f_b_proj.weight
    layer.f_b_proj = lambda t: (F.linear(t, fb), None)
    layer._forward = lambda **kw: KimiK3DeltaAttention._forward(layer, **kw)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    providers.Attention.stock_kda(layer, projected, out)
    torch.cuda.synchronize()
    return out, state


@pytest.mark.parametrize("q,requests", LOOP_SHAPES)
def test_fp32_wide_rows_run_loopstock_matching_stock(monkeypatch, q, requests):
    assert prepared.verify_kernel(requests, q, RUBIN, bf16_state=False) == "loopstock"
    attention = kda_attention()
    slots = requests * q + 1
    base = caches(slots, q)
    table, accepted = spec_inputs(requests, q, slots)
    projected = projected_rows(requests * q)
    core = prepared.PreparedKda(attention, {(requests, q): (PITCH, 0)}, True)
    core.prepare_caches(base)  # the production kernel at this shape
    out, state = run(core, projected, base, table, accepted)
    want, want_state = stock_verify(
        attention, projected, base, table, accepted, q, monkeypatch
    )
    assert_bitwise("conv window", state[0], want_state[0])
    used = table.view(-1).long()
    untouched = torch.ones(slots, dtype=torch.bool, device=DEVICE)
    untouched[used] = False
    assert_bitwise("untouched slots", state[1][untouched], base[1][untouched])
    checkpoints = (state[1][used], want_state[1][used])
    torch.testing.assert_close(*checkpoints, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-2)
    off = (bits(out) != bits(want)).sum().item()
    assert off <= out.numel() // 1000, f"{off} of {out.numel()} outputs differ"
    relerr = (out.float() - want.float()).norm() / want.float().norm()
    assert relerr < 1e-3, relerr  # KdaSplit's math: ~3.4e-3


@pytest.mark.parametrize(
    "q,requests", [(4, 16), (4, 128), (8, 12), (8, 64)], ids=lambda v: str(v)
)
@pytest.mark.parametrize("kind", ["packed", "loop"])
def test_packed_and_loop_equal_wide(monkeypatch, q, requests, kind):
    attention = kda_attention()
    slots = requests * q + 1
    base = caches(slots, q)
    table, accepted = spec_inputs(requests, q, slots)
    projected = projected_rows(requests * q)
    got = run(
        core_of(attention, requests, q, kind, base, monkeypatch),
        projected, base, table, accepted,
    )  # fmt: skip
    want = run(
        core_of(attention, requests, q, "wide", base, monkeypatch),
        projected, base, table, accepted,
    )  # fmt: skip
    assert_bitwise("output", got[0], want[0])
    assert_bitwise("recurrent state", got[1][1], want[1][1])
    assert_bitwise("conv state", got[1][0], want[1][0])


@pytest.mark.parametrize("kind", ["loopstock", "loop"])
def test_loop_padding_requests(monkeypatch, kind):
    """Every third request is padding (slots -1, accepted 0): zero outputs, no cache
    writes, and the live requests' results equal a batch of the live ones alone."""
    q, requests = 4, 96
    attention = kda_attention()
    slots = requests * q + 1
    base = caches(slots, q)
    table, accepted = spec_inputs(requests, q, slots)
    projected = projected_rows(requests * q)
    padded = torch.arange(requests, device=DEVICE) % 3 == 1
    table[padded] = -1
    accepted[padded] = 0
    live = ~padded
    got, state = run(
        core_of(attention, requests, q, kind, base, monkeypatch),
        projected, base, table, accepted,
    )  # fmt: skip
    rows = got.view(requests, q, DIM)
    assert not rows[padded].any(), "padding rows output zeros"
    n = int(live.sum())
    alone_projected = projected.view(requests, q, PITCH)[live].reshape(n * q, PITCH)
    alone, alone_state = run(
        core_of(attention, n, q, kind, base, monkeypatch),
        alone_projected.contiguous(), base, table[live].contiguous(),
        accepted[live].contiguous(),
    )  # fmt: skip
    assert_bitwise("live outputs", rows[live].reshape(n * q, DIM), alone)
    assert_bitwise("recurrent state", state[1], alone_state[1])
    assert_bitwise("conv state", state[0], alone_state[0])
