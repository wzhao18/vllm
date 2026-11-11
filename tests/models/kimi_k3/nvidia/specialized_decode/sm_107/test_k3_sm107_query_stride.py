# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``compile_query``'s static layouts (no GPU: the CuTe compile is recorded, not run).

MlaQuery is compiled against fake tensors whose strides are baked into the kernel and
not checked at launch: the packed DSA in-projection rows (``projected``) must keep the
``pitch`` argument as their row stride, and the FP8 query rows their own (heads or
FlashInfer's padded heads) x 576. A regression that compiled ``projected`` at the
query stride read rows 1+ from the wrong addresses (batch 1 unaffected)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import (  # noqa: E402
    query_stage,
)

# (heads, FlashInfer's padded query heads, the packed in-projection pitch):
# DEP 96 -> 128, EP_GROUP TP8, C512 SP (TP4), TP16.
SHAPES = [(96, 128, 14560), (12, 0, 3808), (24, 0, 5344), (6, 0, 3040)]


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("heads,fi_heads,pitch", SHAPES)
@pytest.mark.parametrize("batch", [1, 4, 64])
def test_compile_query_strides(monkeypatch, batch, heads, fi_heads, pitch, fp8):
    compiled = []

    def fake(shape, stride=None, dtype=None, align=16):
        return ("fake", tuple(shape), tuple(stride) if stride else None)

    def compile_(*args):
        compiled.append(args)
        return "kernel"

    monkeypatch.setattr(query_stage, "fake", fake)
    mla_query = SimpleNamespace(carveout=None)
    monkeypatch.setattr(query_stage, "MlaQuery", lambda *a, **k: mla_query)
    monkeypatch.setattr(query_stage, "make_fake_stream", lambda: "stream")
    monkeypatch.setattr(query_stage, "prepare_shared_carveout", lambda k, d, c: k)
    monkeypatch.setattr(query_stage.cute, "compile", compile_)
    fakes = ("qnorm", "kvnorm", "qb", "uk", "kv")
    query_stage.compile_query(
        batch, heads, pitch, fakes, "cpu", fp8=fp8, fi_heads=fi_heads if fp8 else 0
    )
    (args,) = compiled
    # cute.compile(mla_query, projected, qnorm, kvnorm, qb, uk, kv, slots,
    #              q_scale_inv, k_scale, eps, query, qc, stream)
    projected, query = args[1], args[11]
    assert projected == ("fake", (batch, pitch), (pitch, 1))
    if fp8:
        qpitch = (fi_heads or heads) * 576
        assert query == ("fake", (batch, heads, 576), (qpitch, 576, 1))
    else:
        rows = query_stage.QUERY_HEADS
        assert query == ("fake", (batch, rows, 576), (rows * 576, 576, 1))


@pytest.mark.parametrize("heads", [96, 24, 12, 6])
def test_lean_query_keeps_default_carveout(heads):
    """Only the tiled MlaQuery asks for the 100% shared-memory carveout: on sm_107a
    the lean (one-CTA) kernel fails its launch (cudaErrorLaunchFailure) under it."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.query import (  # noqa: E501
        MlaQuery,
    )

    for batch in range(1, 9):
        query = MlaQuery(heads, batch, dsa=True, fp8=True)
        assert query.carveout == (100 if query.wide else None), (heads, batch)
