# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP8 / TP16 (DFn / DA) KDA cores on an FP32 conv state are BITWISE the BF16-conv
run of the same step (DCRIT r125: the window holds BF16 in-projection inputs whatever
its dtype; the kernels convert each FP32 tap to BF16 at the load and each stored value
to FP32, window.py):

* every plain verify kind (``prepared.WIDE_NAMES``: KdaSplit / KdaSplitStock, Wide,
  Halves, Stream, Stream3, Packed, Loop, LoopStock, Feed, live and padded) at the
  state dtypes it takes, q1..q8: output, recurrent state / checkpoints bitwise, and
  the FP32 window == the BF16 window .float();
* the KDA heads (KdaHead q1, KdaHeadVerify, KdaHeadMulti, plain and polled), the
  fold (KdaInProj64 -> prepared verify), the q1 decode and the TP8 verify: every
  tensor their stock / pair tests check, recorded at BF16 and at FP32 conv;
* the invariant state == state.bfloat16().float() for every FP32 window after the
  stock prefill (causal_conv1d_fn) and after each decode / verify (the rollback of
  rejected tokens is the verify's own window rewrite from ``accepted - 1``)."""

import contextlib

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_kda_fp32_gpu as tp_fp32
from tests.models.kimi_k3 import test_specialized_decode_kda_loop_gpu as tp_loop
from tests.models.kimi_k3 import test_specialized_decode_kda_stock_gpu as tp_stock
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared

FP32, BF16 = torch.float32, torch.bfloat16


@contextlib.contextmanager
def conv_dtype(monkeypatch, dtype):
    with monkeypatch.context() as patch:
        for module in (tp_fp32, tp_loop):
            patch.setattr(module, "CONV_DTYPE", dtype)
        yield


def assert_twin(name, fp32, bf16):
    """``fp32`` (FP32-conv run) equals ``bf16`` (BF16-conv run) bit for bit; an FP32
    conv window (the only tensor whose dtype differs between the runs) equals the
    BF16 one widened and is BF16-representable."""
    if fp32.dtype == FP32 and bf16.dtype == BF16:  # the conv window
        tp_loop.assert_bitwise(f"{name} (fp32 vs bf16.float())", fp32, bf16.float())
        assert_bf16_valued(name, fp32)  # the invariant (FP32 windows only)
    elif not fp32.is_floating_point():  # indices / counts
        assert torch.equal(fp32, bf16), name
    else:
        assert fp32.dtype == bf16.dtype, (name, fp32.dtype, bf16.dtype)
        tp_loop.assert_bitwise(name, fp32, bf16)


def assert_bf16_valued(name, window):
    tp_loop.assert_bitwise(f"{name} BF16-representable", window, window.bfloat16().float())


# ---- plain verify / decode kinds (PreparedKda) ----

FP32_ANY = ("kdasplit", "wide", "halves", "packed", "loop", "loopstock")
KIND_CASES = [
    *(pytest.param(kind, q, b, FP32, False, id=f"{kind}-q{q}-B{b}-fp32")
      for kind in FP32_ANY for q in (2, 4, 8) for b in (12, 64)),
    *(pytest.param(kind, 4, 12, BF16, False, id=f"{kind}-q4-B12-bf16")
      for kind in ("kdasplit", "wide", "halves", "packed")),
    *(pytest.param(kind, q, b, BF16, False, id=f"{kind}-q{q}-B{b}-bf16")
      for kind in ("stream", "stream3") for q in (2, 4) for b in (8, 32)),
    *(pytest.param("feed", q, b, FP32, pad, id=f"feed-q{q}-B{b}-fp32{'-pad' * pad}")
      for q in (2, 4) for b in (16, 90) for pad in (False, True)),
    pytest.param("loopstock", 4, 96, FP32, True, id="loopstock-q4-B96-fp32-pad"),
    pytest.param("loopstock", 5, 40, FP32, False, id="loopstock-q5-B40-fp32"),
    pytest.param("loopstock", 3, 40, FP32, False, id="loopstock-q3-B40-fp32"),
    pytest.param("loopstock", 6, 20, FP32, False, id="loopstock-q6-B20-fp32"),
    pytest.param("loopstock", 7, 20, FP32, False, id="loopstock-q7-B20-fp32"),
]  # fmt: skip


def run_kind(monkeypatch, kind, q, requests, state_dtype, pad, dtype):
    with conv_dtype(monkeypatch, dtype):
        attention = tp_loop.kda_attention()
        slots = requests * q + 1
        conv, state = tp_loop.caches(slots, q)
        base = (conv, state.to(state_dtype))
        table, accepted = tp_loop.spec_inputs(requests, q, slots)
        if pad:  # every third request is padding
            padded = torch.arange(requests, device=tp_loop.DEVICE) % 3 == 1
            table[padded] = -1
            accepted[padded] = 0
        projected = tp_loop.projected_rows(requests * q)
        core = tp_loop.core_of(attention, requests, q, kind, base, monkeypatch)
        out, caches = tp_loop.run(core, projected, base, table, accepted)
    return out, caches


@pytest.mark.parametrize("kind,q,requests,state_dtype,pad", KIND_CASES)
def test_verify_kind_fp32_conv_bitwise(monkeypatch, kind, q, requests, state_dtype, pad):
    want = run_kind(monkeypatch, kind, q, requests, state_dtype, pad, BF16)
    got = run_kind(monkeypatch, kind, q, requests, state_dtype, pad, FP32)
    assert_twin("output", got[0], want[0])
    assert_twin("recurrent state", got[1][1], want[1][1])
    assert_twin("conv window", got[1][0], want[1][0])


@pytest.mark.parametrize("heads,batch,state_dtype", [
    (6, 8, FP32), (6, 64, FP32), (12, 16, FP32), (6, 8, BF16), (12, 8, BF16),
])  # fmt: skip
def test_decode_q1_fp32_conv_bitwise(monkeypatch, heads, batch, state_dtype):
    """The q = 1 decode (KdaSplitStock with an FP32 state, KdaSplit with a BF16 one)."""

    def once(dtype):
        with conv_dtype(monkeypatch, dtype):
            attention = tp_loop.kda_attention(heads=heads)
            slots = 2 * batch + 1
            conv, state = tp_loop.caches(slots, 1, heads=heads)
            base = (conv, state.to(state_dtype))
            order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(5))
            index = (order[:batch] + 1).to(torch.int32).to(tp_loop.DEVICE)
            pitch = tp_loop.pitch_of(heads)
            projected = tp_loop.projected_rows(batch, pitch=pitch)
            prepared.executable.cache_clear()
            core = prepared.PreparedKda(attention, {batch: (pitch, int(heads == 12))}, True)
            core.prepare_caches(base)
            caches = tuple(t.clone() for t in base)
            out = torch.empty(batch, core.dim, dtype=BF16, device=tp_loop.DEVICE)
            core.run(projected, out, caches=caches, slots=index)
            torch.cuda.synchronize()
            return out, caches

    want, got = once(BF16), once(FP32)
    assert_twin("output", got[0], want[0])
    assert_twin("recurrent state", got[1][1], want[1][1])
    assert_twin("conv window", got[1][0], want[1][0])


# ---- heads / fold / TP8 verify: record every tensor their own tests check ----


def recorded(monkeypatch, dtype, test, **params):
    """Run ``test`` at conv dtype ``dtype``; return every tensor its checks saw."""
    seen = []

    def keep(*tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                seen.append(t.detach().clone())
            elif isinstance(t, (tuple, list)):
                keep(*t)

    def wrap(original):
        def checked(*args, **kwargs):
            keep(*args[1:] if isinstance(args[0], str) else args)
            return original(*args, **kwargs)

        return checked

    with conv_dtype(monkeypatch, dtype), monkeypatch.context() as patch:
        patch.setattr(tp_fp32, "assert_bitwise", wrap(tp_fp32.assert_bitwise))
        patch.setattr(tp_stock, "assert_stock", wrap(tp_stock.assert_stock))
        test(monkeypatch=monkeypatch, **params) if "monkeypatch" in (
            test.__code__.co_varnames[: test.__code__.co_argcount]
        ) else test(**params)  # fmt: skip
    return seen


def assert_recorded_twins(monkeypatch, test, **params):
    want = recorded(monkeypatch, BF16, test, **params)
    got = recorded(monkeypatch, FP32, test, **params)
    assert want and len(got) == len(want), (len(got), len(want))
    for i, (g, w) in enumerate(zip(got, want)):
        assert_twin(f"{test.__name__} tensor {i}", g, w)


HEADS = [
    pytest.param(8, 1, False, id="q1-B8"),
    pytest.param(8, 1, True, id="q1-B8-polled"),
    pytest.param(8, 8, False, id="q8-M8-verify"),
    pytest.param(8, 8, True, id="q8-M8-verify-polled"),
    pytest.param(16, 8, False, id="q8-M16-multi"),
    pytest.param(8, 4, True, id="q4-M8-multi-polled"),
]


@pytest.mark.parametrize("rows,q,polled", HEADS)
@pytest.mark.parametrize("state_dtype", [FP32, BF16], ids=["fp32", "bf16"])
def test_heads_fp32_conv_bitwise(monkeypatch, rows, q, polled, state_dtype):
    """KdaHead / KdaHeadVerify / KdaHeadMulti and their pair (in-projection ->
    PreparedKda): output, state, conv window."""
    assert_recorded_twins(
        monkeypatch, tp_fp32.test_kda_head_matches_the_pair,
        rows=rows, q=q, polled=polled, state_dtype=state_dtype,
    )  # fmt: skip


@pytest.mark.parametrize("q", [8, 4])
def test_fold_fp32_conv_bitwise(monkeypatch, q):
    """KdaInProj64 -> the prepared verify (stock math): output and caches."""
    assert_recorded_twins(
        monkeypatch, tp_stock.test_fp32_fold_matches_stock_and_pair, q=q, seed=3
    )


@pytest.mark.parametrize("q,requests", [(4, 8), (8, 16)])
def test_tp8_verify_fp32_conv_bitwise(monkeypatch, q, requests):
    assert_recorded_twins(
        monkeypatch, tp_stock.test_fp32_tp8_verify_matches_stock, q=q, requests=requests
    )


# ---- the invariant after the stock prefill ----


def test_prefill_keeps_fp32_window_bf16_valued():
    """The stock prefill conv (causal_conv1d_fn) writes the FP32 window from BF16
    inputs only: every value BF16-representable, equal to the BF16 window widened."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    device, dim, width, seqs, length = tp_loop.DEVICE, 768, 4, 3, 37
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(seqs * length, dim, generator=g, device=device).to(BF16).T
    weight = torch.randn(dim, width, generator=g, device=device) * 0.5
    starts = torch.arange(0, seqs * length + 1, length, dtype=torch.int32, device=device)
    indices = torch.arange(1, seqs + 1, dtype=torch.int32, device=device)
    has_initial = torch.zeros(seqs, dtype=torch.bool, device=device)
    windows = {}
    for dtype in (BF16, FP32):
        # the DS layout [slot, dim, taps] (the decode's, tp_loop sets it)
        state = torch.zeros(seqs + 1, dim, width - 1, device=device, dtype=dtype)
        causal_conv1d_fn(
            x.clone(), weight, None, conv_states=state, query_start_loc=starts,
            cache_indices=indices, has_initial_state=has_initial,
        )  # fmt: skip
        torch.cuda.synchronize()
        windows[dtype] = state
    assert_twin("prefill window", windows[FP32], windows[BF16])


# ---- TP8 B13-16 verify schedule (Rubin override 1.8.2.2 vs GB300's 1.8.3.2) ----


@pytest.mark.parametrize("q,requests", [(4, 13), (4, 16), (8, 13), (8, 16)])
@pytest.mark.parametrize("dtype", [BF16, FP32], ids=["bf16-conv", "fp32-conv"])
def test_tp8_b13_16_schedule_bitwise(monkeypatch, q, requests, dtype):
    """Two CTAs per SM (1.8.2.2) instead of three (1.8.3.2) at TP8 B13-16: the same
    lanes per state row (8), so the same reduction order. Output, recurrent state
    and conv window bitwise equal to the GB300 schedule's."""
    heads, mode, rubin = 12, 1, (10, 7)
    assert prepared.schedule_of(heads, requests, mode, q, rubin, False) == (1, 8, 2, 2)

    def once(override):
        with conv_dtype(monkeypatch, dtype), monkeypatch.context() as patch:
            if override is not None:
                patch.setattr(prepared, "DEVICE_SPEC_SCHEDULE", override)
            attention = tp_loop.kda_attention(heads=heads)
            slots = requests * q + 1
            base = tp_loop.caches(slots, q, heads=heads)
            table, accepted = tp_loop.spec_inputs(requests, q, slots)
            pitch = tp_loop.pitch_of(heads)
            projected = tp_loop.projected_rows(requests * q, pitch=pitch)
            prepared.executable.cache_clear()
            core = prepared.PreparedKda(attention, {(requests, q): (pitch, mode)}, True)
            core.prepare_caches(base)
            result = tp_loop.run(core, projected, base, table, accepted)
            prepared.executable.cache_clear()
            return result

    gb300 = {rubin: {6: prepared.DEVICE_SPEC_SCHEDULE[rubin][6]}}  # no TP8 override
    want, got = once(gb300), once(None)
    tp_loop.assert_bitwise("output", got[0], want[0])
    tp_loop.assert_bitwise("recurrent state", got[1][1], want[1][1])
    tp_loop.assert_bitwise("conv window", got[1][0], want[1][0])
