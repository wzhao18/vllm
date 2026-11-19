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
