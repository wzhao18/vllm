# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MegaMoE layer's latent up-projection with stock's in-place beta-add,
``residual += x @ W^T`` (``residual.addmm_``: FP32 accumulation, the BF16 residual
added in FP32, one rounding), on ``SplitKTmemGemm64``: the weight slices are staged
on chip before the PDL wait, so a launch right behind the MegaMoE top-k reduce (which
releases its dependents at entry) streams the 51 MB weight while the reduce still
polls its combine cells, and only x, the MMAs and the epilogue follow the reduce.

W [7168, 3584] in 56 tiles of 128 rows, K split over a cluster of 3 (19 / 19 / 18 K
tiles of 64: 13 in tensor memory, the rest in shared memory), owners 2 (64-row
shares). The owner sums the three FP32 partials in rank order ((p0 + p1) + p2), adds
the residual and rounds once: stock's BF16 linear up to the FP32 summation order."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import architecture
from ..common.host import stream as current_stream
from ..common.primitives import (
    asm,
    at,
    bf16x2,
    bits_float,
    load_bf16x8,
    named_barrier,
    store_words,
)
from .projection import matrix_view, weight_map
from .splitk import SplitKTactic
from .splitk_tmem64 import N_TILES, SplitKTmemGemm64, _ld_shared_v4

OWNERS = 2  # 3 K-split ranks per tile: two of them reduce and store 64-row shares
SPLIT = 3  # 56 tiles x 3 = 168 CTAs: one wave at one CTA per SM


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


class LatentUpTmem(SplitKTmemGemm64):
    """``SplitKTmemGemm64`` whose output C is the residual, updated in place."""

    def __init__(self, *, tactic: SplitKTactic, k: int):
        super().__init__(tactic=tactic, k=k, owners=OWNERS)

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Owner rows [row0, row0 + share) of tokens col0 ..: thread (row rr, tokens
        width g .. width g + width - 1) holds ``totals`` (FP32), staged through the
        FP32 ``tile`` [N tokens][share]; then 16-byte chunks (8 rows of one token):
        residual chunk (BF16) + the 8 FP32 sums, rounded once, stored in place."""
        mC = out[0]
        share = self.share
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = self.n * share // 128
        for t in cutlass.range_constexpr(width):
            _st_shared_f32(tile + ((width * g + t) * share + rr) * 4, totals[t])
        named_barrier(3, 128)
        # The residual is the MegaMoE kernel's shared output: ordered before this read
        # by the top-k reduce's completion (its producer arrivals), which the PDL wait
        # covers; already passed by the x warp, so this returns at once.
        cute.arch.griddepcontrol_wait()
        columns = mC.layout.shape[1]
        per_token: cutlass.Constexpr = share // 8
        chunks: cutlass.Constexpr = self.n * per_token
        for i in cutlass.range_constexpr((chunks + 127) // 128):
            q = epi_tid + 128 * i
            token, c = q // per_token, q % per_token
            if (q < chunks) & (col0 + token < columns):
                source = tile + (token * share + 8 * c) * 4
                low = _ld_shared_v4(source)
                high = _ld_shared_v4(source + 16)
                pointer = at(mC, row0 + 8 * c, col0 + token, l_idx)
                r = load_bf16x8(pointer, mutable=True)
                s = [bits_float(low[j]) + r[j] for j in range(4)]
                s += [bits_float(high[j]) + r[4 + j] for j in range(4)]
                words = tuple(bf16x2(s[2 * j], s[2 * j + 1]) for j in range(4))
                store_words(pointer, words)


def tactic_for(rows):
    """(128, N tile, split, x's slots) for ``rows`` tokens, or None (no tactic)."""
    if rows <= 64:
        return (128, 64, SPLIT, 4)
    if rows <= 96:
        return (128, 96, SPLIT, 4)
    return None


@cache
def compile_latent_up(tactic, k, arch):
    """Code-only cache; weights, descriptors and activation pointers stay private."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    assert tactic[1] in N_TILES
    kernel = LatentUpTmem(tactic=SplitKTactic(*tactic), k=k)
    y = fake(0)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class PreparedLatentUp:
    """``residual += x @ weight^T`` for the prepared row counts (in place, returns
    ``residual``). Rows without a tactic are refused at build (``supports``)."""

    def __init__(self, weight: torch.Tensor, rows):
        if weight.dtype != torch.bfloat16 or weight.dim() != 2:
            raise ValueError(f"latent_up weight {tuple(weight.shape)} {weight.dtype}")
        n, k = weight.shape
        pitch = weight.stride(0) * weight.element_size()
        if weight.stride(1) != 1 or pitch % 16 or weight.data_ptr() % 16 or n % 128:
            raise ValueError("latent_up weight: unit stride, 16 B pitch, 128-row tiles")
        if k % 64 or k // 64 < SPLIT * 13:
            raise ValueError(f"latent_up K {k}: 64-wide tiles, >= 13 per split rank")
        missing = [r for r in rows if tactic_for(r) is None]
        if missing:
            raise ValueError(f"latent_up rows without a tactic: {missing}")
        self.weight, self.rows = weight, tuple(rows)
        self.weight_view = matrix_view(weight)
        self.kernels, self.map = {}, None

    @staticmethod
    def supports(rows) -> bool:
        return tactic_for(rows) is not None

    def prepare(self):
        arch, k = architecture(self.weight.device), self.weight.shape[1]
        with torch.accelerator.device_index(self.weight.device.index):
            for r in self.rows:
                self.kernels[r] = compile_latent_up(tactic_for(r), k, arch)
            self.map = weight_map(self.weight, 128)
        return self

    def run(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        if rows not in self.kernels or x.shape[1] != self.weight.shape[1]:
            raise ValueError(f"latent_up input {tuple(x.shape)}: prepared {self.rows}")
        if residual.shape != (rows, self.weight.shape[0]) or residual.stride(1) != 1:
            raise ValueError(f"latent_up residual {tuple(residual.shape)}")
        y = matrix_view(residual, output=True)
        stream = current_stream(self.weight.device)
        a_map = cutlass.Int64(self.map.data_ptr())
        self.kernels[rows](self.weight_view, matrix_view(x), y, y, a_map, stream)
        return residual


__all__ = ["LatentUpTmem", "PreparedLatentUp", "compile_latent_up", "tactic_for"]
