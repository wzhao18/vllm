# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE fused shared expert, CPU only: one resident copy of the shared
gate_up (MEMFIX W4).

``weights.stripe_mxfp8_gate_up_`` stripes the stock MXFP8 gate_up (rows + F8_128x4
scales) in place; the fused FC1 (``prepare_mxfp8_shared_weights(striped=True)``) is
then a view of that storage with exactly the bytes of the old striped copy, and a
GEMM on the striped weight de-stripes (``deinterleave_gate_up_cols_16``) to the
stock ``[gate | up]`` output bit for bit.

The backend package is loaded under an alias straight from its directory (as
test_megamoe_rubin_conversion.py), so only torch and CPU-safe modules are imported.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PKG_DIR = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "models"
    / "kimi_k3"
    / "nvidia"
    / "megamoe_rubin"
)
_ALIAS = "_megamoe_rubin_cpu_test"


def _load(module: str):
    if _ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _ALIAS, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_ALIAS] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{_ALIAS}.{module}")


W = _load("weights")
FP8, E8M0 = torch.float8_e4m3fn, torch.float8_e8m0fnu


def _shared(gen, inter, hidden):
    """Random stock-layout MXFP8 shared expert: gate_up [2I, H], down [H, I] FP8
    bytes (NaN codes avoided) + F8_128x4 swizzled E8M0 scale bytes."""

    def fp8(rows, cols):
        b = torch.randint(0, 256, (rows, cols), generator=gen, dtype=torch.uint8)
        b[(b & 0x7F) == 0x7F] = 0x3C  # no NaN encodings
        return b.view(FP8)

    def sf(rows, cols):
        plane = torch.randint(100, 140, (rows, cols), generator=gen, dtype=torch.uint8)
        return W.to_blocked_batched(plane[None]).reshape(-1).clone()

    return (
        fp8(2 * inter, hidden),
        sf(2 * inter, hidden // 32),
        fp8(hidden, inter),
        sf(hidden, inter // 32),
    )


@pytest.mark.parametrize("inter,hidden", [(128, 128), (384, 256), (256, 512)])
def test_scale_swizzle_round_trip(inter, hidden):
    gen = torch.Generator().manual_seed(1)
    _, gu_s, _, _ = _shared(gen, inter, hidden)
    plane = W.from_blocked(gu_s, 2 * inter, hidden // 32)
    again = W.to_blocked_batched(plane[None].contiguous()).reshape(-1)
    assert torch.equal(again, gu_s)


@pytest.mark.parametrize("inter,hidden", [(128, 128), (384, 256), (256, 512)])
def test_striped_view_equals_striped_copy(inter, hidden):
    gen = torch.Generator().manual_seed(2)
    gu, gu_s, dn, dn_s = _shared(gen, inter, hidden)
    ref_fc1, ref_fc2 = W.prepare_mxfp8_shared_weights(
        gu.clone(), gu_s.clone(), dn, dn_s, hidden=hidden, intermediate=inter
    )
    ptr, sptr = gu.data_ptr(), gu_s.data_ptr()
    W.stripe_mxfp8_gate_up_(gu, gu_s, hidden=hidden, intermediate=inter)
    assert (gu.data_ptr(), gu_s.data_ptr()) == (ptr, sptr)  # in place
    fc1, fc2 = W.prepare_mxfp8_shared_weights(
        gu, gu_s, dn, dn_s, hidden=hidden, intermediate=inter, striped=True
    )
    # FC1 views the (striped) stock storage: no second copy of the weight.
    assert fc1[0].data_ptr() == ptr
    assert fc1[1].data_ptr() == sptr
    for got, ref in ((fc1, ref_fc1), (fc2, ref_fc2)):
        assert got[0].shape == ref[0].shape and got[0].stride() == ref[0].stride()
        assert torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8))
        assert got[1].shape == ref[1].shape
        assert torch.equal(got[1].view(torch.uint8), ref[1].view(torch.uint8))


@pytest.mark.parametrize("inter,hidden", [(128, 128), (384, 256)])
def test_striped_gemm_destripes_to_stock_output(inter, hidden):
    """Dequantize both layouts (FP8 value x 2^(e8m0 - 127) per 32-block) and run
    the GEMM in FP32: the striped weight's output de-stripes to the stock one
    exactly (a pure column permutation, same products and sums per column)."""
    gen = torch.Generator().manual_seed(3)
    gu, gu_s, _, _ = _shared(gen, inter, hidden)
    ref_w, ref_s = gu.clone(), gu_s.clone()
    W.stripe_mxfp8_gate_up_(gu, gu_s, hidden=hidden, intermediate=inter)

    def dequant(w, s):
        plane = W.from_blocked(s, 2 * inter, hidden // 32).float()
        scale = torch.exp2(plane - 127).repeat_interleave(32, dim=1)
        return w.float() * scale

    x = torch.randn((5, hidden), generator=gen)
    ref = x @ dequant(ref_w, ref_s).t()
    out = x @ dequant(gu, gu_s).t()
    assert not torch.equal(out, ref)  # really striped
    assert torch.equal(W.deinterleave_gate_up_cols_16(out), ref)
    # bf16 GEMM outputs (the stock / sdec gate_up dtype) de-stripe the same way
    assert torch.equal(
        W.deinterleave_gate_up_cols_16(out.bfloat16()), ref.bfloat16()
    )


def test_stripe_rejects_bad_layouts():
    gen = torch.Generator().manual_seed(4)
    gu, gu_s, _, _ = _shared(gen, 128, 128)
    with pytest.raises(ValueError):
        W.stripe_mxfp8_gate_up_(gu.t(), gu_s, hidden=256, intermediate=64)
    with pytest.raises(ValueError):
        W.stripe_mxfp8_gate_up_(gu, gu_s[:-1], hidden=128, intermediate=128)
