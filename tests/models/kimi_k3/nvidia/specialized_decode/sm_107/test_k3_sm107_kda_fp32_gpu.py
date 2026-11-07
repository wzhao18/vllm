# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_kda_fp32_gpu.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3 specialized decode KDA kernels with an FP32 recurrent state
(--mamba-ssm-cache-dtype float32) on Rubin (compute capability 10.7), bitwise:

* the KdaHead kernels (input projection + recurrence in one kernel: ``KdaHeadWide``
  q1 B8, ``KdaHeadVerify`` B1 x q8, ``KdaHeadMulti`` 2 x q8 and 2 x q4, plain and
  polled output) vs the in-projection -> KdaSplit pair they replace: the split-K
  projection whose sums the head reproduces (Q / K / V / gate: (128, N, 4); F_A /
  beta: ``SignalGemm``'s split 8), then the layer's ``PreparedKda`` at that shape.
  Outputs, the recurrent state (verify: every token's checkpoint slot) and the conv
  state must match bit for bit. The BF16 state (the measured serving dtype) runs the
  same comparison as a control: a mismatch in both dtypes points at the reference
  pair, one in FP32 only at the FP32 head.
* the wide verify kernel an FP32 state runs at q4 / q8 (``KdaVerifyWide``:
  ``kda.prepared.FP32_VERIFY_WIDE``; ``KdaVerifyHalves`` only at other q) vs KdaSplit
  at the same shape.

One GPU; a few minutes of CuTe compiles on a cold cache."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.projection import (  # noqa: E402
    PreparedProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda.head import (  # noqa: E402
    FA_TACTIC,
    FA_TACTIC_ROWS,
    HEAD_TILE,
    KdaSide,
    PreparedKdaHead,
    head_supported,
)
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

DEVICE = torch.device("cuda", 0)
RUBIN = (10, 7)
HEADS, DIM, HIDDEN = 6, 768, 7168  # TP16: 6 of 96 KDA heads per rank
PITCH = 3216  # packed in-projection rows: Q | K | V | gate | F_A 128 | beta 6 | pad
FA_ROWS = PITCH - 4 * DIM
DTYPES = [torch.float32, torch.bfloat16]
DTYPE_IDS = ["fp32", "bf16-control"]


Module = SimpleNamespace


def kda_attention(seed=0):
    """The fields ``PreparedKda`` reads from a TP16 KDA layer (stock dtypes: FP32
    conv weight, A_log, dt bias and norm weight; BF16 F_B projection)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, dtype=torch.float32, scale=1.0):
        values = torch.randn(*shape, generator=g, device=DEVICE) * scale
        return values.to(dtype)

    return Module(
        conv1d=Module(weight=randn(3 * DIM, 1, 4, scale=0.5), bias=None),
        dt_bias=randn(DIM, scale=0.5),
        A_log=randn(HEADS, scale=0.5),
        f_b_proj=Module(weight=randn(DIM, 128, dtype=torch.bfloat16, scale=128**-0.5)),
        o_norm=Module(weight=1 + randn(128, scale=0.1), eps=1e-5),
        gate_lower_bound=-5.0,
    )


def caches(state_dtype, slots, tokens, seed=1):
    """(conv, recurrent state) of ``slots`` slots: the BF16 conv state in the DS
    layout [slot, 3 dim, q + 2] and the state [slot, heads, value, key]."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    conv = torch.randn(slots, 3 * DIM, tokens + 2, generator=g, device=DEVICE)
    state = torch.randn(slots, HEADS, 128, 128, generator=g, device=DEVICE) * 0.1
    return conv.to(torch.bfloat16), state.to(state_dtype)


def clone(pair):
    return tuple(t.clone() for t in pair)


def bits(tensor):
    return tensor.view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def assert_bitwise(name, got, want):
    assert got.shape == want.shape, name
    assert not torch.isnan(want.float()).any(), f"{name}: NaN in the reference"
    diff = bits(got) != bits(want)
    assert not diff.any(), (
        f"{name}: {int(diff.sum())} of {diff.numel()} elements differ, max |diff| "
        f"{(got.float() - want.float()).abs().max().item():.3e}"
    )


def spec_inputs(requests, tokens, slots, seed=2):
    """Distinct slots [requests, q] (vLLM's spec slots, > 0) and accepted counts in
    1..q."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(slots - 1, generator=g)[: requests * tokens] + 1
    slot_table = order.view(requests, tokens).to(torch.int32).to(DEVICE)
    accepted = torch.randint(1, tokens + 1, (requests,), generator=g)
    return slot_table, accepted.to(torch.int32).to(DEVICE)


def reference_projection(weight, x):
    """The in-projection the head's sums reproduce: Q / K / V / gate rows with the
    split-K (128, N, 4) tactic (((p0 + p1) + p2) + p3), F_A / beta rows with
    ``SignalGemm``'s tactic (split 8)."""
    rows = x.shape[0]
    projected = torch.zeros(rows, PITCH, dtype=torch.bfloat16, device=DEVICE)
    # (ring stages do not change the sums)
    main = PreparedProjection(weight[: 4 * DIM], {rows: (128, rows, 4, 4)}).prepare()
    fa_tactic = FA_TACTIC_ROWS.get(rows, FA_TACTIC)
    fa = PreparedProjection(weight[4 * DIM :], {rows: fa_tactic}).prepare()
    main.run(x, projected[:, : 4 * DIM])
    fa.run(x, projected[:, 4 * DIM :])
    return projected


HEAD_SHAPES = [
    pytest.param(8, 1, False, id="q1-B8"),
    pytest.param(8, 1, True, id="q1-B8-polled"),
    pytest.param(8, 8, False, id="q8-M8-verify"),
    pytest.param(8, 8, True, id="q8-M8-verify-polled"),
    pytest.param(16, 8, False, id="q8-M16-multi"),
    pytest.param(8, 4, False, id="q4-M8-multi"),
    pytest.param(8, 4, True, id="q4-M8-multi-polled"),
]


@pytest.mark.parametrize("state_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("rows,q,polled", HEAD_SHAPES)
def test_kda_head_matches_the_pair(rows, q, polled, state_dtype):
    """KdaHead (plain or polled output) vs in-projection -> PreparedKda, bitwise on
    the output, the recurrent state / checkpoints and the conv state; the head is
    admitted for this shape and dtype (its compile probe passes)."""
    attention = kda_attention()
    g = torch.Generator(device=DEVICE).manual_seed(3)
    weight = torch.randn(PITCH, HIDDEN, generator=g, device=DEVICE) * HIDDEN**-0.5
    weight = weight.to(torch.bfloat16)
    x = torch.randn(rows, HIDDEN, generator=g, device=DEVICE).to(torch.bfloat16)
    requests = rows // q
    key = rows if q == 1 else (requests, q)
    core = prepared.PreparedKda(attention, {key: (PITCH, 0)}, True)
    slots = 4 * rows + 1
    base = caches(state_dtype, slots, q)
    core.prepare_caches(base)
    probe = dict(core=core, fa_width=FA_ROWS, cache_tokens=q, tp=16)
    assert head_supported(HEADS, state_dtype, False, HIDDEN, rows, q, **probe)

    if q == 1:
        order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(5))
        slot_index, accepted = (order[:rows] + 1).to(torch.int32).to(DEVICE), None
    else:
        slot_index, accepted = spec_inputs(requests, q, slots)

    # the pair
    want_caches = clone(base)
    want = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    projected = reference_projection(weight, x)
    core.run(projected, want, caches=want_caches, slots=slot_index, accepted=accepted)
    torch.cuda.synchronize()

    # the head (q1 batch 8 is the KdaHead batch the verify / M8 heads ride on)
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
    if polled:  # this layer's polled buffer (armed) and the previous layer's
        got = torch.empty(8, DIM, dtype=torch.bfloat16, device=DEVICE)
        bits(got).fill_(-1)  # OUT_SENT
        rearm = torch.zeros(8, DIM, dtype=torch.bfloat16, device=DEVICE)
        got = got[:rows]
    else:
        got = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    if q == 1:
        head.run(x, got, caches=got_caches, slots=slot_index, rearm=rearm)
    else:
        head.run_verify(
            x, got, caches=got_caches, slots=slot_index, accepted=accepted, rearm=rearm
        )
    side.join()
    torch.cuda.synchronize()

    assert_bitwise("output", got, want)
    assert_bitwise("recurrent state", got_caches[1], want_caches[1])
    assert_bitwise("conv state", got_caches[0], want_caches[0])
    if polled:
        assert (bits(rearm[:rows]) == -1).all(), "previous output re-armed"


WIDE_SHAPES = [
    pytest.param(35, 4, "wide", id="q4-B35"),
    pytest.param(41, 4, "wide", id="q4-B41"),
    pytest.param(12, 8, "wide", id="q8-B12"),
    pytest.param(64, 4, "wide", id="q4-B64"),
    pytest.param(128, 4, "wide", id="q4-B128"),
    pytest.param(512, 4, "wide", id="q4-B512"),
    pytest.param(65, 8, "wide", id="q8-B65"),
    pytest.param(128, 8, "wide", id="q8-B128"),
]


@pytest.mark.parametrize("requests,q,kind", WIDE_SHAPES)
def test_kda_fp32_wide_verify_matches_kdasplit(monkeypatch, requests, q, kind):
    """With an FP32 state the plain verify above the tuned rows runs KdaVerifyWide /
    KdaVerifyHalves (never the BF16-only chunked verify), bitwise KdaSplit's verify
    at the same shape: outputs, every token's checkpoint and the conv window."""
    state_dtype = torch.float32
    assert prepared.verify_kernel(requests, q, RUBIN, bf16_state=False) == kind
    attention = kda_attention()
    rows, slots = requests * q, requests * q + 1
    base = caches(state_dtype, slots, q)
    key = (requests, q)
    core = prepared.PreparedKda(attention, {key: (PITCH, 0)}, True)
    core.prepare_caches(base)  # the production kernel at this shape
    names = {prepared.KdaVerifyWide: "wide", prepared.KdaVerifyHalves: "halves"}
    schedule = prepared.schedule_of(HEADS, requests, 0, q, RUBIN, False)
    kernel = prepared.make_kernel(
        (HEADS, requests, 0, False, schedule, q, False), RUBIN
    )
    assert names[type(kernel)] == kind

    # KdaSplit at the same shape and schedule table: the same executable, with the
    # wide kinds switched off (its compile cache cleared on both sides)
    prepared.executable.cache_clear()
    monkeypatch.setattr(prepared, "wide_kind", lambda *args, **kwargs: 0)
    reference = prepared.PreparedKda(attention, {key: (PITCH, 0)}, True)
    reference.prepare_caches(base)
    monkeypatch.undo()
    prepared.executable.cache_clear()

    g = torch.Generator(device=DEVICE).manual_seed(4)
    projected = torch.randn(rows, PITCH, generator=g, device=DEVICE)
    projected = projected.to(torch.bfloat16)
    slot_index, accepted = spec_inputs(requests, q, slots)
    outputs = []
    for runner in (core, reference):
        state = clone(base)
        out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
        runner.run(projected, out, caches=state, slots=slot_index, accepted=accepted)
        outputs.append((out, state))
    torch.cuda.synchronize()
    (got, got_caches), (want, want_caches) = outputs
    assert_bitwise("output", got, want)
    assert_bitwise("recurrent state", got_caches[1], want_caches[1])
    assert_bitwise("conv state", got_caches[0], want_caches[0])
