# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill shared gate_up GEMM with the SiTU + MXFP8 epilogue (sm_107) is
bitwise equal to cuBLASLt gate_up + situ_and_mul_quantize_auto.

Exact-input operands (small-integer FP8, power-of-two scales) make every
summation order give the same FP32 accumulator, so the comparison checks the
epilogue bytes (values and scales), padding rows included.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)


def _operands(m, n, k, dev):
    g = torch.Generator(device=dev).manual_seed(m)
    x = torch.randint(-2, 3, (m, k), device=dev, generator=g).float()
    w = torch.randint(-2, 3, (n, k), device=dev, generator=g).float()
    sx = torch.randint(122, 126, (((m + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    return x.to(torch.float8_e4m3fn), sx, w.to(torch.float8_e4m3fn), sw


@pytest.mark.parametrize("m", [16384, 4100, 2048])
@pytest.mark.parametrize("betas", [(4.0, 25.0), (4.0, None)])
def test_shared_situ_gemm_bitwise(m, betas):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    beta, lbeta = betas
    dev = torch.device("cuda")
    inter, k = 1536, 7168
    x, sx, w, sw = _operands(m, 2 * inter, k, dev)
    y = torch._scaled_mm(
        x, w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu), scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    ref.data.view(torch.uint8).zero_()
    ref.scale.zero_()
    mxfp8_act.situ_and_mul_quantize_auto(y, beta, lbeta, out=ref)

    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    out = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    out.data.view(torch.uint8).fill_(0x7F)
    out.scale.zero_()
    ssg.SharedSituGemm(wi, swi, beta, lbeta)(x, sx, out)
    torch.cuda.synchronize()
    assert torch.equal(out.data.view(torch.uint8), ref.data.view(torch.uint8))
    assert torch.equal(out.scale, ref.scale)


@pytest.mark.parametrize("m", [4100])
def test_shared_situ_gemm_die_split_bitwise(m):
    """The die-split scheduler (weight halves homed per die, per-die tile
    claims) writes the same bytes at the deployed width (inter 6144)."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        die_sched,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    inter, k = 6144, 7168
    x, sx, w, sw = _operands(m, 2 * inter, k, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    tactic = ssg.die_tactic(wi)
    if tactic is None:
        pytest.skip("single-die device")
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    ssg.SharedSituGemm(wi, swi, 4.0, 25.0)(x, sx, ref)
    gemm = ssg.SharedSituGemm(die_sched.localize_rows(wi), swi, 4.0, 25.0, tactic=tactic)
    out = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    for _ in range(2):  # the second launch runs on the counters the first reset
        out.data.view(torch.uint8).fill_(0x7F)
        out.scale.zero_()
        gemm(x, sx, out)
        torch.cuda.synchronize()
        assert torch.equal(out.data.view(torch.uint8), ref.data.view(torch.uint8))
        assert torch.equal(out.scale, ref.scale)


# N tiles past column N (BUG-KDASHAPE-1). The SiTU epilogues store with raw
# addresses bounded by row only; the persistent scheduler hands out N tiles
# past column N when the cluster N does not divide n / 256 (ceil) or the "m"
# raster pads the cluster column count to a multiple of the swizzle. Those
# tiles must not store: each case checks the bytes bitwise against
# cuBLASLt + situ_and_mul_quantize_auto AND guard bytes past both buffers.
_GUARD = 1 << 16
_CANARY = 0x5A


class _GuardedAct:
    """An ``Mxfp8Buffer`` whose data and scale are followed by guard bytes."""

    def __init__(self, m, inter, dev):
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

        ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
        nd, ns = ref.data.numel(), ref.scale.numel()
        self._d = torch.full((nd + _GUARD,), _CANARY, dtype=torch.uint8, device=dev)
        self._s = torch.full((ns + _GUARD,), _CANARY, dtype=torch.uint8, device=dev)
        ref.data = self._d[:nd].view(torch.float8_e4m3fn).view(m, inter)
        ref.scale = self._s[:ns]
        self.buf, self.nd, self.ns = ref, nd, ns

    def guards_intact(self) -> bool:
        return bool((self._d[self.nd:] == _CANARY).all()) and bool(
            (self._s[self.ns:] == _CANARY).all()
        )


def _situ_ref(x, sx, w, sw, m, inter, beta, lbeta, dev):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    y = torch._scaled_mm(
        x, w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu), scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    ref.data.view(torch.uint8).fill_(_CANARY)
    ref.scale.fill_(_CANARY)
    mxfp8_act.situ_and_mul_quantize_auto(y[:, -2 * inter:].contiguous(), beta, lbeta, out=ref)
    return y, ref


def _check(out: "_GuardedAct", ref):
    torch.cuda.synchronize()
    assert out.guards_intact(), "stores past the MXFP8 buffer"
    assert torch.equal(out.buf.data.view(torch.uint8), ref.data.view(torch.uint8))
    assert torch.equal(out.buf.scale, ref.scale)


# (inter, tactic): N = 2 inter; DEFAULT/DIE = 512x256 tile, 2x4 cluster,
# swizzle 4, "m" raster (drain epilogue). Cluster columns n / 1024: 1, 3,
# 1.25 (n % 1024 != 0), 12 (deployed inter 6144). "nodrain" = the same
# cluster / swizzle / raster with the non-drain SiTU epilogue (256x256 tile).
_NODRAIN = ((256, 256), (2, 4), 256, 4, "m", 0, 0, "")


@pytest.mark.parametrize(
    "inter, tactic",
    [
        (512, "default"),
        (1536, "default"),
        (640, "default"),
        (6144, "default"),
        (6144, "die"),
        (1536, "nodrain"),
        (640, "nodrain"),
        (6144, "nodrain"),
    ],
)
def test_shared_situ_gemm_padded_n_tiles(inter, tactic):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        die_sched,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    m, k, beta, lbeta = 4100, 7168, 4.0, 25.0
    x, sx, w, sw = _operands(m, 2 * inter, k, dev)
    _, ref = _situ_ref(x, sx, w, sw, m, inter, beta, lbeta, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    if tactic == "die":
        tac = ssg.die_tactic(wi)
        if tac is None:
            pytest.skip("single-die device")
        wi = die_sched.localize_rows(wi)
    else:
        tac = _NODRAIN if tactic == "nodrain" else ssg.DEFAULT_TACTIC
    gemm = ssg.SharedSituGemm(wi, swi, beta, lbeta, tactic=tac)
    for _ in range(2):
        out = _GuardedAct(m, inter, dev)
        gemm(x, sx, out.buf)
        _check(out, ref)


@pytest.mark.parametrize(
    "n_lat, inter, tactic",
    [
        # 3 N tiles over 2x2 clusters: the 4th (padded) tile is a shared tile.
        (256, 256, ((256, 256), (2, 2), True)),
        # 11 N tiles (2 latent + 9 shared) over 2x2 clusters.
        (512, 1152, ((256, 256), (2, 2), True)),
        # The packed GEMM's default tactic (2x1 cluster: no padded N tiles).
        (3584, 6144, None),
    ],
)
def test_packed_situ_gemm_padded_n_tiles(n_lat, inter, tactic):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        packed_situ_gemm as psg,
    )

    dev = torch.device("cuda")
    m, k, beta, lbeta = 4100, 7168, 4.0, 25.0
    n = n_lat + 2 * inter
    x, sx, w, sw = _operands(m, n, k, dev)
    y, ref = _situ_ref(x, sx, w, sw, m, inter, beta, lbeta, dev)
    wi, swi = w.clone(), sw.clone()
    psg.interleave_shared_rows(wi, swi, n_lat, k)
    gemm = psg.PackedSituGemm(
        wi, swi, n_lat, beta, lbeta, tactic=tactic or psg.DEFAULT_TACTIC
    )
    for _ in range(2):
        out = _GuardedAct(m, inter, dev)
        lat = torch.empty(m, n_lat, dtype=torch.bfloat16, device=dev)
        gemm(x, sx, out.buf, lat)
        _check(out, ref)
        assert torch.equal(lat.view(torch.int16), y[:, :n_lat].view(torch.int16))


def test_situ_gemm_rejects_tile_n():
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        packed_situ_gemm as psg,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    x, sx, w, sw = _operands(256, 1024, 256, dev)
    with pytest.raises(ValueError, match="tile N"):
        ssg.SharedSituGemm(w, sw, 4.0, 25.0, tactic=((512, 128), (2, 1), 256, 1, "m", 0, 0, ""))
    with pytest.raises(ValueError, match="tile N"):
        psg.PackedSituGemm(w, sw, 256, 4.0, 25.0, tactic=((256, 128), (2, 1), False))
