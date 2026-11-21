# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-shape policy of prefill AttnRes dataflow C (attn_res_resid): a prefill
forward runs it from ``MIN_ROWS`` rows per rank only (BUG-PRES-1: below that
the reduce-store GEMM is slower than the stock GEMM + AttnRes with a delta),
and the decoder layer asks the policy before enabling any producer."""

import types

import torch


def test_use_for_threshold():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_resid as rr

    # Short prefix-hit steps (1K / 2K rows) and the 4K-6K wave step stay on
    # the stock GEMM + AttnRes with a delta (measured crossover: 8K rows).
    assert rr.MIN_ROWS == 8192
    assert not rr.use_for(0)
    assert not rr.use_for(1024)
    assert not rr.use_for(6144)
    assert not rr.use_for(rr.MIN_ROWS - 1)
    assert rr.use_for(rr.MIN_ROWS)
    assert rr.use_for(16384)


def test_decoder_layer_gates_producers_on_rows(monkeypatch):
    from vllm.models.kimi_k3.nvidia import model as km
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_resid as rr

    # A planned attn_res_lp forward whose prefix is the die-local buffer: only
    # the row count decides.
    monkeypatch.setattr(km.prefill_attn_res_lp, "active", lambda *a: True)
    monkeypatch.setattr(km.prefill_attn_res_dielocal, "is_prefix", lambda *a: True)
    producers = km.KimiDecoderLayer._resid_producers
    for caps in ((True, True), (True, False), (False, True)):
        layer = types.SimpleNamespace(_resid_caps=caps)
        for rows, want in (
            (1024, (False, False)),
            (2048, (False, False)),
            (6144, (False, False)),
            (rr.MIN_ROWS - 1, (False, False)),
            (rr.MIN_ROWS, caps),
            (16384, caps),
        ):
            prefix = torch.empty(rows, 8, dtype=torch.bfloat16)
            assert producers(layer, prefix, prefix) == want, (caps, rows)
    # No planned forward / no die-local prefix: never, at any row count.
    monkeypatch.setattr(km.prefill_attn_res_dielocal, "is_prefix", lambda *a: False)
    layer = types.SimpleNamespace(_resid_caps=(True, True))
    prefix = torch.empty(16384, 8, dtype=torch.bfloat16)
    assert producers(layer, prefix, prefix) == (False, False)
    assert producers(layer, None, prefix) == (False, False)
