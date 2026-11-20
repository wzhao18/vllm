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
from cutlass import Float32, Int32, Int64
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor

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
# After the wait x's K tiles are the only traffic and stream at bytes in flight /
# L2 latency: x's slots hold (nearly) the whole K slice (4 slots of 12 KB measured
# ~6.5 us after the wait, ~5 round trips). The kernel runs alone behind the
# reduce, so it may take most of Rubin's ~328 KB per SM.
MAX_SMEM = 300 * 1024


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


@asm(Int32)
def _ld_acquire(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Int32)
def _atom_add(address, value):
    code = "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (Int64(address), Int32(value))


@asm()
def _st_relaxed(address, value):
    code = "st.relaxed.gpu.global.b32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm()
def _nanosleep():
    return "nanosleep.u32 64;", "~{memory}", ()


class LatentUpTmem(SplitKTmemGemm64):
    """``SplitKTmemGemm64`` whose output C is the residual, updated in place.

    No PDL wait: on Rubin a programmatic dependent's ``griddepcontrol.wait`` returns
    only once the whole chain of primaries has completed, i.e. after the MegaMoE
    kernel's serial tail (reset, system fence, peer signals). x and the residual are
    ordered by the reduce's ``[rows done, passed]`` pair instead (the ``bias``
    argument): each reduce CTA release-adds 1 once its row slice is stored, the
    producer-done acquirer after that acquire (which orders the shared output); the
    x warp acquires ``rows x reduce_split`` arrivals, and the last of this grid's
    CTAs to pass re-arms the pair for the next call. Everything the epilogue reads
    is causally after that acquire (x warp -> x barriers -> MMA warp -> epilogue)."""

    def __init__(self, *, tactic: SplitKTactic, k: int, reduce_split: int):
        super().__init__(tactic=tactic, k=k, owners=OWNERS, max_smem=MAX_SMEM)
        self.reduce_split = reduce_split

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mS: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major (the residual,
        in place); ``mS``: the reduce's int32 ``[rows done, passed]`` pair."""
        self.sync = mS
        self.rows = mB.layout.shape[0]
        self.body(mA, mB, mS, a_map, (mC,))

    @cute.experimental.jit
    def wait_rows(self):
        """One lane: acquire the reduce's arrivals; the last CTA through re-arms."""
        counter = self.sync.iterator.toint()
        expected = Int32(self.rows) * Int32(self.reduce_split)
        if cute.arch.lane_idx() == 0:
            while _ld_acquire(counter) < expected:
                _nanosleep()
            gx, gy, gz = cute.arch.grid_dim()
            ctas = Int32(gx * gy * gz)
            if _atom_add(counter + 4, Int32(1)) == ctas - Int32(1):
                _atom_add(counter, -expected)
                _st_relaxed(counter + 4, Int32(0))
        cute.arch.sync_warp()

    @cute.experimental.jit
    def x_warp64(
        self, b_full, b_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, first: cutlass.Int32, tiles: cutlass.Int32,
    ):  # fmt: skip
        """``SplitKTmemGemm64.x_warp64`` behind the reduce's arrivals instead of the
        PDL wait."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        self.wait_rows()
        self.stamp(3, cute.arch.lane_idx() == 0)
        for k_tile in cutlass.range(tiles, unroll=1):
            slot = k_tile % self.slots
            if k_tile >= self.slots:
                cute.arch.mbarrier_wait(b_empty + k_tile - self.slots, 0)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, first + k_tile],
                s_tile[None, None, None, slot],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

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
    if rows <= 64:  # 18 x 8 KB slots (+ 5 A tiles in shared memory): 224 KB
        return (128, 64, SPLIT, 18)
    if rows <= 96:  # 16 x 12 KB slots (+ 6 A tiles): 288 KB
        return (128, 96, SPLIT, 16)
    if rows <= 128:  # 11 x 16 KB slots (+ 7 A tiles): 288 KB
        return (128, 128, SPLIT, 11)
    return None


@cache
def compile_latent_up(tactic, k, reduce_split, arch):
    """Code-only cache; weights, descriptors and activation pointers stay private."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    assert tactic[1] in N_TILES
    kernel = LatentUpTmem(tactic=SplitKTactic(*tactic), k=k, reduce_split=reduce_split)
    y = fake(0)
    sync = make_fake_tensor(cutlass.Int32, (2,), (1,), assumed_align=8)
    args = (fake(1), fake(1), y, sync, cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class PreparedLatentUp:
    """``residual += x @ weight^T`` behind the fused-norm top-k reduce (in place,
    returns ``residual``), for the prepared row counts; ``reduce_split``: the
    reduce's CTAs per row (its arrivals per row on the ``sync`` pair)."""

    def __init__(self, weight: torch.Tensor, rows, reduce_split: int):
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
        self.weight, self.rows, self.reduce_split = weight, tuple(rows), reduce_split
        self.weight_view = matrix_view(weight)
        self.kernels, self.map = {}, None

    @staticmethod
    def supports(rows) -> bool:
        return tactic_for(rows) is not None

    def prepare(self):
        arch, k = architecture(self.weight.device), self.weight.shape[1]
        with torch.accelerator.device_index(self.weight.device.index):
            for tactic in {tactic_for(r) for r in self.rows}:
                self.kernels[tactic] = compile_latent_up(tactic, k, self.reduce_split, arch)
            self.map = weight_map(self.weight, 128)
        return self

    def run(
        self, x: torch.Tensor, residual: torch.Tensor, sync: torch.Tensor
    ) -> torch.Tensor:
        rows = x.shape[0]
        code = self.kernels.get(tactic_for(rows))
        if code is None or x.shape[1] != self.weight.shape[1]:
            raise ValueError(f"latent_up input {tuple(x.shape)}: prepared {self.rows}")
        if residual.shape != (rows, self.weight.shape[0]) or residual.stride(1) != 1:
            raise ValueError(f"latent_up residual {tuple(residual.shape)}")
        if sync is None or sync.dtype != torch.int32 or sync.shape != (2,) or sync.data_ptr() % 8:
            raise ValueError("latent_up sync: the reduce's int32 [rows done, passed]")
        y = matrix_view(residual, output=True)
        stream = current_stream(self.weight.device)
        a_map = cutlass.Int64(self.map.data_ptr())
        pair = from_dlpack(sync, assumed_align=8)
        code(self.weight_view, matrix_view(x), y, pair, a_map, stream)
        return residual


__all__ = ["LatentUpTmem", "PreparedLatentUp", "compile_latent_up", "tactic_for"]
