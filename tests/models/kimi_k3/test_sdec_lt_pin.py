# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned cuBLASLt drafter GEMMs (specialized_decode.lt_pin): every pin of the loaded
cuBLASLt version passes the algorithm check and computes F.linear's product (FP32
accumulation, one BF16 rounding; a split-K pin sums in another order)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="cuBLASLt pins need CUDA"
)


def _cases():
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    table = lt_pin.PINS.get(lt_pin.version(), {})
    return [(shape, rows) for shape, pins in sorted(table.items()) for rows in pins]


def test_pins_match_linear():
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    cases = _cases()
    if not cases:
        pytest.skip(f"no pins for cuBLASLt {lt_pin.version()}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    for (n, k), rows in cases:
        w = (torch.randn((n, k), device="cuda", generator=gen) / k**0.5).bfloat16()
        x = torch.randn((rows, k), device="cuda", generator=gen).bfloat16()
        lin = lt_pin.pinned(w, [rows])
        assert lin is not None and lin.kernels.keys() == {rows}, (n, k, rows)
        assert lin.takes(x)
        out = lin.run(x, x.new_empty((rows, n)))
        ref = x.float() @ w.float().t()
        stock = torch.nn.functional.linear(x, w)
        scale = ref.abs().max().item()
        err = (out.float() - ref).abs().max().item() / scale
        stock_err = (stock.float() - ref).abs().max().item() / scale
        assert err <= max(2 * stock_err, 4e-3), ((n, k), rows, err, stock_err)
        # A strided activation (row stride > K) reads the same rows.
        wide = torch.zeros((rows, k + 64), device="cuda", dtype=torch.bfloat16)
        wide[:, :k] = x
        again = lin.run(wide[:, :k], x.new_empty((rows, n)))
        torch.testing.assert_close(again, out, atol=0, rtol=0)


def test_unpinned_rows_and_shapes():
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    w = torch.zeros((7168, 7168), device="cuda", dtype=torch.bfloat16)
    assert lt_pin.pinned(w, [5]) is None
    w = torch.zeros((100, 7168), device="cuda", dtype=torch.bfloat16)
    assert lt_pin.pinned(w, [288]) is None
    assert lt_pin.with_pins(None, w, [288]) is None


def test_bad_descriptor_rejected():
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    w = torch.zeros((7168, 7168), device="cuda", dtype=torch.bfloat16)
    bad = lt_pin.PinnedLinear(w, {288: "f" * 128})
    assert bad.rejected == [288] and not bad.kernels


def test_bin_pins_cover_and_match_linear():
    """BIN_PINS (the TP4 drafter linears at DSpark k1-k7 rows): bins are contiguous
    and increasing per shape, and each bin's pin passes the algorithm check and
    computes F.linear's product at the bin's first, middle and last row (a row
    the pin was not searched at must still be served correctly); the strict build
    raises on a rejected pin instead of leaving F.linear."""
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Uncovered

    table = lt_pin.BIN_PINS.get(lt_pin.version(), {})
    if not table:
        pytest.skip(f"no row-bin pins for cuBLASLt {lt_pin.version()}")
    gen = torch.Generator(device="cuda").manual_seed(1)
    for (n, k), bins in sorted(table.items()):
        assert all(lo <= hi for lo, hi, _, _ in bins), (n, k)
        assert all(a[1] + 1 == b[0] for a, b in zip(bins, bins[1:])), (n, k)
        w = (torch.randn((n, k), device="cuda", generator=gen) / k**0.5).bfloat16()
        rows = sorted({r for lo, hi, _, _ in bins for r in (lo, (lo + hi) // 2, hi)})
        lin = lt_pin.pinned(w, rows, strict=True)
        assert lin is not None and set(lin.kernels) >= set(rows) - set(
            lt_pin.pins_for((n, k))
        ), (n, k)
        for r in rows:
            x = torch.randn((r, k), device="cuda", generator=gen).bfloat16()
            out = lin.run(x, x.new_empty((r, n)))
            ref = x.float() @ w.float().t()
            stock = torch.nn.functional.linear(x, w)
            scale = ref.abs().max().item()
            err = (out.float() - ref).abs().max().item() / scale
            stock_err = (stock.float() - ref).abs().max().item() / scale
            assert err <= max(2 * stock_err, 4e-3), ((n, k), r, err, stock_err)
        del w
    # A rejected pin raises under strict (a descriptor of another shape's tile).
    w = torch.zeros((7168, 1536), device="cuda", dtype=torch.bfloat16)
    lo, hi, label, _ = table[(7168, 1536)][0]
    bad = list(table[(7168, 1536)])
    bad[0] = (lo, hi, label, "ff" * 64)
    saved = table[(7168, 1536)]
    table[(7168, 1536)] = tuple(bad)
    try:
        with pytest.raises(Uncovered, match="pinned cuBLASLt"):
            lt_pin.pinned(w, [lo], strict=True)
    finally:
        table[(7168, 1536)] = saved


def test_strict_refuses_pins_of_another_cublaslt(monkeypatch):
    """Pins are cuBLASLt-version specific: a strict build on a cuBLASLt the pins were
    not searched with raises (no silent heuristic fallback)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Uncovered

    if not lt_pin.BIN_PINS.get(lt_pin.version()):
        pytest.skip(f"no row-bin pins for cuBLASLt {lt_pin.version()}")
    w = torch.zeros((7168, 1536), device="cuda", dtype=torch.bfloat16)
    real = lt_pin.version()
    monkeypatch.setattr(lt_pin, "version", lambda: real + 1)
    with pytest.raises(Uncovered, match="not portable"):
        lt_pin.pinned(w, [66], strict=True)
    assert lt_pin.pinned(w, [66]) is None  # non-strict: unpinned, logged by the caller


def test_exact_pins_match_linear():
    """EXACT_PINS (DCRIT r120: every captured drafter M, the searched best or the
    cuBLASLt heuristic's own algorithm): each pin passes the algorithm check under
    strict and computes F.linear's product at its row."""
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    table = lt_pin.EXACT_PINS.get(lt_pin.version(), {})
    if not table:
        pytest.skip(f"no exact pins for cuBLASLt {lt_pin.version()}")
    gen = torch.Generator(device="cuda").manual_seed(2)
    for (n, k), rows in sorted(table.items()):
        w = (torch.randn((n, k), device="cuda", generator=gen) / k**0.5).bfloat16()
        lin = lt_pin.pinned(w, sorted(rows), strict=True)
        assert lin is not None and set(lin.kernels) == set(rows), (n, k)
        for r in sorted(rows):
            x = torch.randn((r, k), device="cuda", generator=gen).bfloat16()
            out = lin.run(x, x.new_empty((r, n)))
            ref = x.float() @ w.float().t()
            stock = torch.nn.functional.linear(x, w)
            scale = ref.abs().max().item()
            err = (out.float() - ref).abs().max().item() / scale
            stock_err = (stock.float() - ref).abs().max().item() / scale
            assert err <= max(2 * stock_err, 4e-3), ((n, k), r, err, stock_err)
        del w
