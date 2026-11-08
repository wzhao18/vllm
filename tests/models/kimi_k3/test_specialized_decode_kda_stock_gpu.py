# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode KDA kernels with an FP32 recurrent state on Rubin vs the
stock layer (``KimiK3DeltaAttention._forward``, native Triton: ``causal_conv1d_update``
-> ``fused_recurrent_kda`` (verify) / ``fused_recurrent_kda_packed_decode`` (q = 1) ->
``o_norm``) on the same in-projection rows:

* the verify heads (``KdaHeadVerify`` B1 x q8, ``KdaHeadMulti`` 2 x q8 / 2 x q4) and the
  q = 1 head (``KdaHeadWide`` B8), plain and polled output, from x through the
  projection the heads reproduce (``reference_projection``);
* the B8 fold (``KdaInProj64`` / ``KdaInProjQ`` epilogue -> ``KdaVerifyCols``), from
  the rows the fold publishes; also bit for bit the plain stock-math pair
  (``KdaSplitStock``) on those rows;
* the q = 1 decode (``KdaSplitStock``) at TP16 (6 heads, gate mode 0) and TP8 (12
  heads, gate modes 1 / 2 / 0 by batch, as ``providers.Attention``);
* the TP8 verify (12 heads, gate mode 1: ``KdaSplitStock``).

Each matches stock to FP32 reduction order, with the thresholds of
``test_specialized_decode_kda_loop_gpu.py``: the conv window bit for bit, the states
(every verify checkpoint) within relerr 2e-7, the BF16 outputs within relerr 2e-4 and
at most 1 in 4000 of them a rounding apart. (KdaSplit's math, which these kernels ran
before for an FP32 state: output relerr ~3.4e-3, state ~2.1e-3.) The heads' bitwise
equality with the in-projection -> KdaSplitStock pair is
``test_specialized_decode_kda_fp32_gpu.py``'s.

One GPU; several minutes of CuTe compiles on a cold cache."""

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from tests.models.kimi_k3.test_specialized_decode_kda_fp32_gpu import (  # noqa: E402
    FA_ROWS,
    HIDDEN,
    reference_projection,
)
from tests.models.kimi_k3.test_specialized_decode_kda_loop_gpu import (  # noqa: E402
    DEVICE,
    PREFIX,
    Module,
    bits,
    caches,
    kda_attention,
    kda_module,
    pitch_of,
    projected_rows,
    relative_error,
    spec_inputs,
)
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention  # noqa: E402
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.head import (  # noqa: E402
    HEAD_TILE,
    KdaSide,
    PreparedKdaHead,
    head_supported,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.inproj64 import (  # noqa: E402
    PREPARED_PITCH,
    PreparedKdaInProj64,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    providers,
)

PITCH = 3216
DIM = 768
FOLD_TACTIC = providers.WIDE_B64[PITCH][:4]


def stock_run(attention, projected, base, slots, accepted, q, monkeypatch):
    """The stock layer's KDA core from the in-projection rows (seam S1,
    ``providers.Attention.stock_kda``): the spec verify for q > 1 (``slots`` [B, q],
    ``accepted`` [B]), the plain decode for q = 1 (``slots`` [B])."""
    rows = projected.shape[0]
    heads = attention.A_log.numel()
    fields = dict(num_prefills=0, num_prefill_tokens=0, num_actual_tokens=rows)
    if q > 1:
        requests = rows // q
        fields.update(
            num_decodes=0, num_decode_tokens=0, num_spec_decodes=requests,
            num_spec_decode_tokens=rows,
            spec_query_start_loc=torch.arange(
                0, rows + 1, q, dtype=torch.int32, device=DEVICE
            ),
            spec_state_indices_tensor=slots, num_accepted_tokens=accepted,
        )  # fmt: skip
    else:
        fields.update(
            num_decodes=rows, num_decode_tokens=rows, num_spec_decodes=0,
            num_spec_decode_tokens=0,
            non_spec_query_start_loc=torch.arange(
                0, rows + 1, dtype=torch.int32, device=DEVICE
            ),
            non_spec_state_indices_tensor=slots,
        )  # fmt: skip
    metadata = KimiK3KDAMetadata(**fields)
    context = Module(attn_metadata={PREFIX: metadata})
    monkeypatch.setattr(kda_module, "get_forward_context", lambda: context)
    state = tuple(t.clone() for t in base)
    layer = Module(
        prefix=PREFIX, kv_cache=state, kda_spec_decode_backend="native",
        kda_decode_backend="triton", kda_prefill_backend="triton",
        decode_conv1d_weight=None, decode_norm_weight=None, conv1d=attention.conv1d,
        local_projection_size=128 * heads, local_num_heads=heads, head_dim=128,
        use_recoverssm=False, A_log=attention.A_log, dt_bias=attention.dt_bias,
        gate_lower_bound=attention.gate_lower_bound, o_norm=attention.o_norm,
    )  # fmt: skip
    fb = attention.f_b_proj.weight
    layer.f_b_proj = lambda t: (torch.nn.functional.linear(t, fb), None)
    layer._forward = lambda **kw: KimiK3DeltaAttention._forward(layer, **kw)
    out = torch.empty(rows, 128 * heads, dtype=torch.bfloat16, device=DEVICE)
    providers.Attention.stock_kda(layer, projected, out)
    torch.cuda.synchronize()
    return out, state


def assert_stock(got, got_caches, want, want_caches, base, used):
    """The thresholds of test_specialized_decode_kda_loop_gpu.py: the conv window
    bitwise, the states of ``used`` slots within relerr 2e-7 (the others untouched),
    the outputs within relerr 2e-4 and at most 1 / 4000 one rounding apart."""
    conv_diff = bits(got_caches[0]) != bits(want_caches[0])
    assert not conv_diff.any(), f"conv window: {int(conv_diff.sum())} differ"
    untouched = torch.ones(base[1].shape[0], dtype=torch.bool, device=DEVICE)
    untouched[used] = False
    stale = bits(got_caches[1][untouched]) != bits(base[1][untouched])
    assert not stale.any(), "untouched state slots written"
    state_error = relative_error(got_caches[1][used], want_caches[1][used])
    assert state_error < 2e-7, state_error  # FP32 order; KdaSplit's math: ~2.1e-3
    torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-2)
    off = (bits(got) != bits(want)).sum().item()
    assert off <= max(4, got.numel() // 4000), f"{off} of {got.numel()} differ"
    out_error = relative_error(got, want)
    assert out_error < 2e-4, out_error  # KdaSplit's math: ~3.4e-3


def clone(pair):
    return tuple(t.clone() for t in pair)


HEAD_SHAPES = [
    pytest.param(8, 1, False, id="q1-B8"),
    pytest.param(8, 1, True, id="q1-B8-polled"),
    pytest.param(8, 8, False, id="q8-M8-verify"),
    pytest.param(8, 8, True, id="q8-M8-verify-polled"),
    pytest.param(16, 8, False, id="q8-M16-multi"),
    pytest.param(8, 4, False, id="q4-M8-multi"),
    pytest.param(8, 4, True, id="q4-M8-multi-polled"),
]


@pytest.mark.parametrize("rows,q,polled", HEAD_SHAPES)
def test_fp32_heads_match_stock(monkeypatch, rows, q, polled):
    """A KdaHead kernel (FP32 state: ``KdaHead.stock``) from x vs the stock layer on
    the projection the head reproduces."""
    attention = kda_attention()
    g = torch.Generator(device=DEVICE).manual_seed(3)
    weight = torch.randn(PITCH, HIDDEN, generator=g, device=DEVICE) * HIDDEN**-0.5
    weight = weight.to(torch.bfloat16)
    x = torch.randn(rows, HIDDEN, generator=g, device=DEVICE).to(torch.bfloat16)
    requests = rows // q
    key = rows if q == 1 else (requests, q)
    core = prepared.PreparedKda(attention, {key: (PITCH, 0)}, True)
    slots = 4 * rows + 1
    base = caches(slots, q)
    core.prepare_caches(base)
    probe = dict(core=core, fa_width=FA_ROWS, cache_tokens=q, tp=16)
    assert head_supported(6, torch.float32, False, HIDDEN, rows, q, **probe)
    if q == 1:
        order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(5))
        table, accepted = (order[:rows] + 1).to(torch.int32).to(DEVICE), None
    else:
        table, accepted = spec_inputs(requests, q, slots)

    side = KdaSide(DEVICE)
    counter = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    fa_out = torch.empty(64, FA_ROWS, dtype=torch.bfloat16, device=DEVICE)
    batches = (8,) if (q == 1 or polled) else ()
    verify = () if q == 1 else ((rows, q),)
    head = PreparedKdaHead(
        weight, core, batches, side, counter, fa_out, tile=HEAD_TILE,
        polled=(8,) if polled else (), verify=verify,
    )  # fmt: skip
    got_caches = clone(base)
    head.prepare_caches(got_caches)
    rearm = None
    if polled:
        got = torch.empty(8, DIM, dtype=torch.bfloat16, device=DEVICE)
        bits(got).fill_(-1)  # OUT_SENT
        rearm = torch.zeros(8, DIM, dtype=torch.bfloat16, device=DEVICE)
        got = got[:rows]
    else:
        got = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    if q == 1:
        head.run(x, got, caches=got_caches, slots=table, rearm=rearm)
    else:
        head.run_verify(
            x, got, caches=got_caches, slots=table, accepted=accepted, rearm=rearm
        )
    side.join()
    torch.cuda.synchronize()

    projected = reference_projection(weight, x)
    want, want_caches = stock_run(
        attention, projected, base, table, accepted, q, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.view(-1).long())


@pytest.mark.parametrize("q", [8, 4], ids=["q8-M64", "q4-M32"])
def test_fp32_fold_matches_stock_and_pair(monkeypatch, q):
    """The B8 x q fold (in-projection epilogue -> prepared verify, FP32 state) vs the
    stock layer on the rows it publishes, and bit for bit vs the plain stock-math
    pair (``KdaSplitStock``) on those rows."""
    attention = kda_attention()
    g = torch.Generator(device=DEVICE).manual_seed(3)
    weight = torch.randn(PITCH, HIDDEN, generator=g, device=DEVICE) * HIDDEN**-0.5
    weight = weight.to(torch.bfloat16)
    requests, rows = 8, 8 * q
    x = torch.randn(rows, HIDDEN, generator=g, device=DEVICE).to(torch.bfloat16)
    slots = rows + 1
    base = caches(slots, q)
    table, accepted = spec_inputs(requests, q, slots)
    key = (requests, q)
    core = prepared.PreparedKda(attention, {key: (PITCH, 0)}, True, prepared=(key,))
    core.prepare_caches(base)
    kernel = prepared.make_kernel(
        (6, requests, 0, False, prepared.schedule_of(6, requests, 0, q, (10, 7), False),
         q, False), (10, 7), prepared=True,
    )  # fmt: skip
    assert kernel.stock, type(kernel)
    fold = PreparedKdaInProj64(weight, core, FOLD_TACTIC, tokens=q)
    got_caches = clone(base)
    fold.prepare_caches(got_caches)
    projected = torch.zeros(rows, PITCH, dtype=torch.bfloat16, device=DEVICE)
    projected[:, 4 * DIM : 4 * DIM + 128].view(torch.int16).fill_(-1)  # FA_SENT
    if q < 8:
        fold.x_map(x)
    operands = torch.empty(rows, PREPARED_PITCH, dtype=torch.float32, device=DEVICE)
    got = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    fold.run(x, projected, operands, caches=got_caches, slots=table, accepted=accepted)
    torch.cuda.synchronize()
    # the published rows (the prepared verify's last reader re-arms their F_A)
    published = projected.clone()
    core.run(
        projected, got, caches=got_caches, slots=table, accepted=accepted,
        prepared=operands,
    )  # fmt: skip
    torch.cuda.synchronize()

    pair_caches = clone(base)
    pair = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    core.run(published, pair, caches=pair_caches, slots=table, accepted=accepted)
    torch.cuda.synchronize()
    assert not (bits(got) != bits(pair)).any(), "fold output vs the pair"
    assert not (bits(got_caches[1]) != bits(pair_caches[1])).any(), "state vs pair"
    assert not (bits(got_caches[0]) != bits(pair_caches[0])).any(), "conv vs pair"

    want, want_caches = stock_run(
        attention, published, base, table, accepted, q, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.view(-1).long())


def tp8_decode_mode(batch):
    """``providers.Attention``'s TP8 q = 1 gate mode by batch."""
    return 1 if batch == 1 else 2 if batch <= 8 else 0


DECODE_SHAPES = [
    *(pytest.param(6, b, 0, id=f"tp16-B{b}") for b in (1, 4, 8, 16, 64)),
    *(pytest.param(12, b, tp8_decode_mode(b), id=f"tp8-B{b}") for b in (1, 4, 16, 64)),
]


@pytest.mark.parametrize("heads,batch,mode", DECODE_SHAPES)
def test_fp32_decode_q1_matches_stock(monkeypatch, heads, batch, mode):
    """The q = 1 decode of an FP32 state (``KdaSplitStock``, every gate mode) vs the
    stock plain decode."""
    options = (heads, batch, mode, False, (0, 0, 0), 1, False)
    schedule = prepared.schedule_of(heads, batch, mode, 1, (10, 7), False)
    kernel = prepared.make_kernel((*options[:4], schedule, *options[5:]), (10, 7))
    assert type(kernel) is prepared.KdaSplitStock, type(kernel)
    attention = kda_attention(heads=heads)
    slots = batch + 2
    base = caches(slots, 1, heads=heads)
    g = torch.Generator(device="cpu").manual_seed(5)
    table = (torch.randperm(slots - 1, generator=g)[:batch] + 1).to(torch.int32)
    table = table.to(DEVICE)
    pitch = pitch_of(heads)
    projected = projected_rows(batch, pitch=pitch)
    core = prepared.PreparedKda(attention, {batch: (pitch, mode)}, True)
    core.prepare_caches(base)
    got_caches = clone(base)
    got = torch.empty(batch, 128 * heads, dtype=torch.bfloat16, device=DEVICE)
    core.run(projected, got, caches=got_caches, slots=table)
    torch.cuda.synchronize()
    want, want_caches = stock_run(
        attention, projected, base, table, None, 1, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.long())


@pytest.mark.parametrize("q", [4, 8])
@pytest.mark.parametrize("requests", [1, 4, 16, 64])
def test_fp32_tp8_verify_matches_stock(monkeypatch, q, requests):
    """TP8's verify (12 heads, gate mode 1) of an FP32 state (``KdaSplitStock``) vs
    the stock verify."""
    heads, mode = 12, 1
    schedule = prepared.schedule_of(heads, requests, mode, q, (10, 7), False)
    options = (heads, requests, mode, False, schedule, q, False)
    assert type(prepared.make_kernel(options, (10, 7))) is prepared.KdaSplitStock
    attention = kda_attention(heads=heads)
    slots = requests * q + 1
    base = caches(slots, q, heads=heads)
    table, accepted = spec_inputs(requests, q, slots)
    pitch = pitch_of(heads)
    projected = projected_rows(requests * q, pitch=pitch)
    core = prepared.PreparedKda(attention, {(requests, q): (pitch, mode)}, True)
    core.prepare_caches(base)
    got_caches = clone(base)
    got = torch.empty(requests * q, 128 * heads, dtype=torch.bfloat16, device=DEVICE)
    core.run(projected, got, caches=got_caches, slots=table, accepted=accepted)
    torch.cuda.synchronize()
    want, want_caches = stock_run(
        attention, projected, base, table, accepted, q, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.view(-1).long())
