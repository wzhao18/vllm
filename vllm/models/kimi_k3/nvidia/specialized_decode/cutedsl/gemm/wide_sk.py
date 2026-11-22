# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local, persistent, stream-K token-wide BF16 GEMM for the DSpark drafter's LM
head (and other large drafter linears) at 96-384 rows.

y[T, N] = x[T, K] @ W[N, K]^T on ``wide.WideGemm``'s MMA form (swap-AB 2-CTA, 256
weight rows on M, all T tokens on N, FP32 accumulator in TMEM), restructured for the
hardware limits of these shapes:

- Die-local weights: W's two N halves are homed on the two dies (one localized
  allocation, 2 MiB chunks); a CTA pair streams only rows homed on the die it runs on
  (``%smid`` -> die table), so the weight stream reads at the die-local rate.
- One persistent wave, stream-K: one CTA pair per SM pair; each die's (tile, K step)
  space is cut into equal contiguous ranges, one per pair of that die. Slots are
  claimed two-ended (die-0 pairs from the bottom of the slot list, die-1 pairs from
  the top), so every slot is claimed exactly once whatever the placement.
- A tile shared by two ranges is finished by the range holding its first K step (the
  head; that segment is its range's last): the range holding the rest publishes its
  FP32 partial first thing, the head adds it after its own sum (K order:
  deterministic) and rounds once to BF16, as the stock GEMM's FP32 accumulation
  (order may differ).
- Many TMA issuing threads: one issuing thread keeps about one op in flight
  (~0.5-1 us each under load), so every lane of the weight warp and of the activation
  warp is its own issuer (K-step residue); a CTA's activation rows for a K step are
  one box (both MMA chunks: a token permutation the epilogue undoes).
- ``spread``: any number of ranges per tile (the drafter's square / skinny linears,
  ~4-14 ranges per tile). Every segment publishes its FP32 partial to a die-homed
  workspace; once a tile's ranges have all published, its 2 KiB units are split over
  them, and each sums its units' partials in K order (bulk loads into the drained
  ring; deterministic) and stores BF16, or (``silu``: gate/up rows interleaved per 64)
  vLLM's silu_and_mul of the BF16 gate and up values.
"""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ..common import dies, tensormap
from ..common.primitives import asm, div_rn
from .wide import (
    _atom_add,
    _bar,
    _bulk_load,
    _fence_after,
    _fence_proxy_global,
    _ld_acquire,
    _lds_v4,
    _nanosleep,
    _now,
    _red_release_add,
    _round_bf16,
    _st_bf16,
    _st_f32,
    _st_relaxed,
    _st_u64,
    _st_v4,
    mma_columns,
)

STAMPS = 16  # u64 stamp words per CTA (debug bit 8)
K_TILE = 64  # K per ring stage (one 128-byte swizzle atom)
CHUNK = 16  # tokens per TMEM load (tcgen05.ld 32x32b.x16)
MAX_CONTRIBUTORS = 1  # partial ranges per tile (beyond the head's)


@asm(Int32)
def _smid():
    return "mov.u32 $0, %smid;", "=r,~{memory}", ()


@asm((Int32, Int32))
def _claim(address, inc):
    code = (
        "{ .reg .b64 o; atom.relaxed.gpu.global.add.u64 o, [$2], $3; "
        "mov.b64 {$0, $1}, o; }"
    )
    return code, "=r,=r,l,l,~{memory}", (Int64(address), Int64(inc))


@asm()
def _st_u64_zero(address):
    return "st.relaxed.gpu.global.u64 [$0], 0;", "l,~{memory}", (Int64(address),)


@asm()
def _sts_u32(address, value):
    return "st.shared.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Int32(value))


@asm()
def _sts_peer(address, peer, value):
    code = (
        "{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; "
        "st.shared::cluster.u32 [a], $2; }"
    )
    return code, "r,r,r,~{memory}", (Int32(address), Int32(peer), Int32(value))


@asm(Int32)
def _lds_u32(address):
    return "ld.shared.u32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def _st_u32(address, value):
    return "st.global.u32 [$0], $1;", "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int32)
def _mapa(address, rank):
    code = "mapa.shared::cluster.u32 $0, $1, $2;"
    return code, "=r,r,r", (Int32(address), Int32(rank))


@asm()
def _tma2_stream(dst, tmap, c0, c1, mbar):
    """2-D tile load into this CTA, completing on the pair leader's barrier ``mbar``
    (shared::cluster address); L2 evict-first (weights: streamed once)."""
    code = (
        "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier"
        "::complete_tx::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    )
    return code, "r,l,r,r,r,~{memory}", (
        Int32(dst), Int64(tmap), Int32(c0), Int32(c1), Int32(mbar)
    )


@asm(Int32, pure=True)
def _pack_bf16x2(lo, hi):
    """BF16x2 of two FP32 values (each rounded to nearest even, as cvt.rn.bf16.f32)."""
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (Float32(lo), Float32(hi))


@asm()
def _st_bf16_pair(lo_addr, hi_addr, packed):
    code = ("{ .reg .b16 l, h; mov.b32 {l, h}, $2; st.global.b16 [$0], l; "
            "st.global.b16 [$1], h; }")
    return code, "l,l,r,~{memory}", (Int64(lo_addr), Int64(hi_addr), Int32(packed))


@asm()
def _tmem_alloc_x(holder, cols):
    """Exclusive TMEM allocation (sm_107: up to 576 columns, one TMEM-using CTA per SM)."""
    code = "tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32 [$0], $1;"
    return code, "r,r,~{memory}", (Int32(holder), Int32(cols))


@asm()
def _tmem_dealloc_x(taddr, cols):
    code = "tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 $0, $1;"
    return code, "r,r,~{memory}", (Int32(taddr), Int32(cols))


@asm()
def _tma2(dst, tmap, c0, c1, mbar):
    code = (
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier"
        "::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];"
    )
    return code, "r,l,r,r,r,~{memory}", (
        Int32(dst), Int64(tmap), Int32(c0), Int32(c1), Int32(mbar)
    )


@asm((Float32,) * 16)
def _tmem_ld16w(taddr):
    """16 TMEM columns and the wait in ONE asm: with separate statements the compiler
    may copy / spill the destination registers before the async load wrote them."""
    regs = ",".join(f"${i}" for i in range(16))
    code = (
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16]; "
        "tcgen05.wait::ld.sync.aligned;"
    )
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),)


@asm()
def _fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def _bar_n(barrier_id, count):
    return "bar.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm()
def _sts_v4(address, a, b, c, d):
    code = "st.shared.v4.f32 [$0], {$1, $2, $3, $4};"
    return code, "r,f,f,f,f,~{memory}", (Int32(address), a, b, c, d)


@asm(Float32)
def _add_lds_if(acc, address, pred):
    code = (
        "{ .reg .pred p; .reg .f32 t; setp.ne.s32 p, $3, 0; mov.f32 $0, $1; "
        "@p ld.shared.f32 t, [$2]; @p add.rn.f32 $0, $1, t; }"
    )
    return code, "=f,f,r,r,~{memory}", (Float32(acc), Int32(address), Int32(pred))


def ranges(total, slots):
    return [s * total // slots for s in range(slots + 1)]


def max_contributors(total, slots, ks):
    """The most ranges that start strictly inside one tile of ``ks`` K steps."""
    starts = ranges(total, slots)[:-1]
    worst = 0
    for t in range(total // ks):
        worst = max(worst, sum(1 for s in starts if t * ks < s < (t + 1) * ks))
    return worst


def tile_ranges(total, slots, ks):
    """(first range, ranges) of every tile of ``ks`` K steps; range s holds the K steps
    [s total // slots, (s + 1) total // slots), so step x lies in range
    ((x + 1) slots + total - 1) // total - 1."""
    out = []
    for t in range(total // ks):
        a = ((t * ks + 1) * slots + total - 1) // total - 1
        b = ((t + 1) * ks * slots + total - 1) // total - 1
        out.append((a, b - a + 1))
    return out


OVERSIZED_BUDGET = 320 * 1024  # Rubin's oversized shared memory mode (327 KiB per CTA)
MAX_W_STAGES = 20


def smem_budget(device=None):
    """Shared memory one CTA may use: the oversized budget on sm_107, else opt-in."""
    props = torch.cuda.get_device_properties(device)
    limit = props.shared_memory_per_block_optin
    if (props.major, props.minor) == (10, 7):
        limit = max(limit, OVERSIZED_BUDGET)
    return limit


def ring_bytes(rows, x_stages=4, smem=None):
    """(weight stages, ring bytes) of the shared-memory budget ``smem``."""
    n_mma, chunks = mma_columns(rows)
    a_bytes = 128 * K_TILE * 2
    b_bytes = chunks * (n_mma // 2) * K_TILE * 2
    smem = smem or utils.get_smem_capacity_in_bytes("sm_100")
    w_stages = min(MAX_W_STAGES, (smem - 4096 - x_stages * b_bytes) // a_bytes)
    return w_stages, w_stages * a_bytes + x_stages * b_bytes


def pack_tiles(weight):
    """[R, K] -> [R / 128 * K / 64 * 128, 64]: each [128 rows, 64 K] tile contiguous
    (row tile major, then K step), so one weight stage is one 16 KiB DRAM run."""
    r, k = weight.shape
    tiles = weight.view(r // 128, 128, k // K_TILE, K_TILE).permute(0, 2, 1, 3)
    return tiles.contiguous().view(-1, K_TILE)


def spread_bytes(rows, ranges):
    """Shared memory a range's reduction of one tile needs: its unit pairs' (warps q
    and q + 2) partials from all ``ranges`` ranges, 2 KiB each."""
    units = 4 * (rows // 2 // CHUNK)
    return -(-units // ranges) * 2 * ranges * 2048


def supports(rows, n, k, pairs, spread=False, x_stages=4, smem=None):
    n_mma, chunks = mma_columns(rows)
    if not (
        rows % 32 == 0 and chunks > 0 and n_mma % 16 == 0 and n % 512 == 0
        and k % K_TILE == 0 and min(pairs) > 0
    ):  # fmt: skip
        return False
    ks = k // K_TILE
    total = n // 512 * ks
    if spread:  # every range holds at least one K step
        ring = ring_bytes(rows, x_stages, smem)[1]
        return all(total >= p for p in pairs) and all(
            spread_bytes(rows, c) <= ring for p in pairs
            for _, c in tile_ranges(total, p, ks)
        )  # fmt: skip
    return all(max_contributors(total, p, ks) <= MAX_CONTRIBUTORS for p in pairs)


class WideSK:
    """One (rows, N, K, pairs per die) specialization; see the module docstring.
    ``debug`` bits: 1 no MMA (rings and epilogues only), 2 no activation loads,
    4 no weight loads (skeletons), 8 globaltimer stamps per CTA after the flags
    (``STAMPS`` u64: entry, slot, then per segment acc_full seen / done, end; the
    one-accumulator form also stamps the TMEM release of segments 0 / 1 in 12 / 13)."""

    def __init__(self, rows, n, k, pairs, ldo=None, w_issuers=3, x_issuers=2, debug=0,
                 x_stages=4, spread=False, silu=False, smem=None, packed=False,
                 keep_w=False, nocopy=True):  # fmt: skip
        """``smem``: the shared-memory budget (default: the sm_100 capacity);
        ``packed``: the weight halves are ``pack_tiles`` layouts; ``keep_w``: load the
        weight with the default L2 policy (it is read again soon), not evict-first;
        ``nocopy`` (non-spread): two TMEM accumulators, see below."""
        assert supports(rows, n, k, pairs, spread, x_stages, smem), (rows, n, k, pairs)
        assert spread or not silu
        self.packed, self.keep_w = packed, keep_w
        self.rows, self.n, self.k = rows, n, k
        self.p0, self.p1 = pairs
        self.pairs = self.p0 + self.p1
        self.spread, self.silu = spread, silu
        self.ldo = ldo or (n // 2 if silu else n)
        self.debug = debug
        self.nch = rows // 2 // CHUNK  # 16-column units per epilogue thread
        self.tpd = n // 512  # tiles per die
        # spread: workspace [die][tile][range in tile][CTA] of one CTA's FP32 partial
        # (8 warps x nch 2 KiB units), each die's part 2 MiB aligned (homed there);
        # then 2 x tpd tile counters per die (published, done) after the stamps
        ks_ = k // K_TILE
        self.cmax = max(c for p in pairs for _, c in tile_ranges(n // 512 * ks_, p, ks_))
        self.cta_bytes = 8 * self.nch * 2048
        part = self.tpd * self.cmax * 2 * self.cta_bytes
        self.die_stride = -(-part // (2 << 20)) * (2 << 20)
        self.cnt_byte = self.pairs * 8 + 8 + 2 * self.pairs * STAMPS * 8
        self.done_byte = self.cnt_byte + 2 * self.tpd * 2 * 4
        self.n_mma, self.chunks = mma_columns(rows)
        self.mma_tiler = (256, self.n_mma, K_TILE)
        self.nw, self.nx = w_issuers, x_issuers  # issuing lanes
        assert 1 <= w_issuers <= 32 and 1 <= x_issuers <= 32
        # warps: 0-7 epilogue (two per TMEM lane quarter), 8 MMA, 9 weights, 10 x
        self.threads = 32 * 11
        self.ks = k // K_TILE
        self.total = n // 512 * self.ks  # K steps per die
        # ``nocopy``: two TMEM accumulators (2 x 288 columns take Rubin's exclusive
        # 576-column allocation), so no epilogue copies a whole accumulator into
        # registers before releasing it (the 144-value copy of the one-accumulator
        # form spilled ~78 MB of local-memory traffic per 288-row launch): a
        # contributor streams its FP32 partial out 16 columns at a time while the MMA
        # runs the next segment in the other buffer; whole / head tiles hold their
        # BF16 values packed in pairs. Same sums and roundings as the copy form.
        self.nocopy = bool(nocopy) and not spread
        self.acc_bufs = 2 if self.nocopy else 1
        assert self.acc_bufs * rows <= 576
        # Separate rings: the weights stream from HBM (deep ring: the die-local rate
        # needs >= ~128-192 KiB in flight per SM), the activations from L2 (shallow).
        a_bytes = 128 * K_TILE * 2
        b_bytes = self.chunks * (self.n_mma // 2) * K_TILE * 2
        self.a_stage, self.b_stage = a_bytes, b_bytes
        self.x_stages = x_stages
        self.w_stages, ring = ring_bytes(rows, x_stages, smem)
        assert ring >= rows * 512  # owner partial staging
        assert not spread or ring >= spread_bytes(rows, self.cmax)
        # one issuing lane per in-flight stage at most (two lanes would arm one phase)
        assert self.nw <= self.w_stages and self.nx <= x_stages
        self.tmem_cols = 32
        while self.tmem_cols < self.acc_bufs * rows:
            self.tmem_cols *= 2
        if self.tmem_cols > 512:  # exclusive allocation (one TMEM-using CTA per SM)
            self.tmem_cols = 576

    @cute.jit
    def __call__(
        self, w0: Int64, w1: Int64, x: Int64, x1: Int64, y: cute.Tensor,
        ws: cute.Tensor, flags: cute.Tensor, table: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        """w0 / w1: addresses of the die-0 / die-1 row halves [N / 2, K] and x of the
        activations [T, K] (x1: the same, read by the CTAs on die 1), BF16 K-major;
        y [T, N] row-major BF16 (row pitch ``ldo``); ws [pairs, 2, T, 128] FP32;
        flags [2 * pairs + 2] int32 (zeroed once); table [SMs] int32 smid -> die."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.TWO, self.mma_tiler[:2]
        )
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, dtype, self.w_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, dtype, self.x_stages * self.chunks
        )
        bf16 = _cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        sw128 = _cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = _cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        k, half, rows = self.k, self.n // 2, self.rows
        # weights: a CTA's 128 rows x one K step; activations: a CTA's T / 2 rows
        if cutlass.const_expr(self.packed):  # [tiles x 128, 64]: one box = one 16 KiB run
            wdims, wstride = (K_TILE, half * k // K_TILE), (K_TILE * 2,)
        else:
            wdims, wstride = (k, half), (k * 2,)
        wmap0 = tensormap.encode_map(w0, bf16, wdims, wstride, (K_TILE, 128), sw128, l2)
        wmap1 = tensormap.encode_map(w1, bf16, wdims, wstride, (K_TILE, 128), sw128, l2)
        xmap0 = tensormap.encode_map(x, bf16, (k, rows), (k * 2,), (K_TILE, rows // 2),
                                     sw128, l2)  # fmt: skip
        xmap1 = tensormap.encode_map(x1, bf16, (k, rows), (k * 2,), (K_TILE, rows // 2),
                                     sw128, l2)  # fmt: skip
        self.tx_a = self.a_stage * 2
        self.tx_b = self.b_stage * 2

        @cute.struct
        class Storage:
            w_full: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            w_empty: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            x_full: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            x_empty: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 2]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 2]
            red: cute.struct.MemRange[cutlass.Int64, 8]
            dealloc: cutlass.Int64
            xfer: cutlass.Int64
            tmem_holder: cutlass.Int32
            slot: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout.outer)], 1024
            ]

        self.storage = Storage
        self.kernel(
            tiled_mma, wmap0, wmap1, xmap0, xmap1, a_layout, b_layout, y, ws, flags,
            table,
        ).launch(  # fmt: skip
            grid=(2 * self.pairs, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=True,
            # one CTA per SM: the full register file (the default budget spilled the
            # unrolled epilogue at 72 registers per thread)
            min_blocks_per_mp=1,
            max_number_threads=[self.threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, wmap0: tensormap.GridMap,
        wmap1: tensormap.GridMap, xmap0: tensormap.GridMap, xmap1: tensormap.GridMap, a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout, mY: cute.Tensor, mWs: cute.Tensor,
        mFlags: cute.Tensor, mTable: cute.Tensor,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        t_entry = _now()
        v = bidx % 2  # rank in the CTA pair; 0 issues the MMAs
        leader = v == 0
        rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        sw, sx, chunks, nb = self.w_stages, self.x_stages, self.chunks, self.acc_bufs
        ks, total, rows = self.ks, self.total, self.rows

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        w_full, w_empty = st.w_full.data_ptr(), st.w_empty.data_ptr()
        x_full, x_empty = st.x_full.data_ptr(), st.x_empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        red = st.red.data_ptr()
        dealloc, xfer = st.dealloc.ptr, st.xfer.ptr
        tmem_holder = st.tmem_holder.ptr
        slot_addr = Int32(st.slot.ptr.toint())
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        stage_addr = Int32(st.sA.data_ptr().toint())
        flags = mFlags.iterator.toint()
        claim = flags + Int64(self.pairs * 8)

        if warp == 0:
            with cute.arch.elect_one():
                for m in (wmap0, wmap1, xmap0, xmap1):
                    tensormap.prefetch_map(tensormap.map_address(m))
                for i in cutlass.range_constexpr(sw):
                    cute.arch.mbarrier_init(w_full + i, 1)
                    cute.arch.mbarrier_init(w_empty + i, 1)
                for i in cutlass.range_constexpr(sx):
                    cute.arch.mbarrier_init(x_full + i, 1)
                    cute.arch.mbarrier_init(x_empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, 16)
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(red + i, 1)
                cute.arch.mbarrier_init(dealloc, 32)
                cute.arch.mbarrier_init(xfer, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        # The pair's slot: claimed by the leader, two-ended by die, then shared.
        if warp == 0:
            if leader:
                with cute.arch.elect_one():
                    my_die = mTable[_smid()]
                    inc = Int64(1)
                    if my_die != 0:
                        inc = Int64(1 << 32)
                    lo, hi = _claim(claim, inc)
                    mine = lo
                    if my_die != 0:
                        mine = self.pairs - 1 - hi
                    if lo + hi == self.pairs - 1:
                        _st_u64_zero(claim)
                    _sts_u32(slot_addr, mine)
                    _sts_peer(slot_addr, 1, mine)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        g = _lds_u32(slot_addr)
        stamp = flags + Int64(self.pairs * 8 + 8) + Int64(bidx) * (STAMPS * 8)
        if cutlass.const_expr(self.debug & 8):
            if warp == 0:
                if lane == 0:
                    _st_u64(stamp + 8, _now())
                    _st_u32(stamp + 14 * 8, g)
        if warp == 0:
            cute.arch.griddepcontrol_launch_dependents()
        wd = Int32(0)
        j = g
        nd = Int32(self.p0)
        if g >= self.p0:
            wd = Int32(1)
            j = g - self.p0
            nd = Int32(self.p1)
        s0 = j * total // nd
        s1 = (j + 1) * total // nd
        t0 = s0 // ks
        nseg = (s1 - 1) // ks - t0 + 1

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, chunks * nb))

        w_steps = Int32(0) if self.debug & 4 else s1 - s0
        x_steps = Int32(0) if self.debug & 2 else s1 - s0
        sa_addr = Int32(st.sA.data_ptr().toint())
        sb_addr = Int32(st.sB.data_ptr().toint())
        if warp == 9:  # weights: one issuer per lane, before the PDL wait
            wmap = tensormap.map_address(wmap0)
            if wd != 0:
                wmap = tensormap.map_address(wmap1)
            if lane < self.nw:
                for step in cutlass.range(lane, w_steps, self.nw, unroll=1):
                    s = step % sw
                    cute.arch.mbarrier_wait(w_empty + s, ((step // sw) & 1) ^ 1)
                    gk = s0 + step
                    t = gk // ks
                    bar = Int32((w_full + s).toint())
                    if leader:
                        cute.arch.mbarrier_arrive_and_expect_tx(w_full + s, self.tx_a)
                    c0 = (gk - t * ks) * K_TILE
                    c1 = t * 256 + v * 128
                    if cutlass.const_expr(self.packed):
                        c0 = Int32(0)
                        c1 = ((t * 2 + v) * ks + gk - t * ks) * 128
                    if cutlass.const_expr(self.keep_w):  # re-read soon: keep in L2
                        _tma2(sa_addr + s * self.a_stage, wmap, c0, c1, _mapa(bar, 0))
                    else:
                        _tma2_stream(sa_addr + s * self.a_stage, wmap, c0, c1,
                                     _mapa(bar, 0))  # fmt: skip
        elif warp == 10:  # activations: one issuer per lane, after the PDL wait
            xmap = tensormap.map_address(xmap0)
            if mTable[_smid()] != 0:  # the copy homed on this CTA's die
                xmap = tensormap.map_address(xmap1)
            cute.arch.griddepcontrol_wait()
            if lane < self.nx:
                for step in cutlass.range(lane, x_steps, self.nx, unroll=1):
                    q = step % sx
                    cute.arch.mbarrier_wait(x_empty + q, ((step // sx) & 1) ^ 1)
                    bar = Int32((x_full + q).toint())
                    if leader:
                        cute.arch.mbarrier_arrive_and_expect_tx(x_full + q, self.tx_b)
                    _tma2(sb_addr + q * self.b_stage, xmap, ((s0 + step) % ks) * K_TILE,
                          v * (rows // 2), _mapa(bar, 0))  # fmt: skip
        elif warp == 8:
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if leader:
                kblocks = cute.size(tCrA, mode=[2])
                for si in cutlass.range(nseg, unroll=1):
                    t = t0 + si
                    lo_k = t * ks
                    if lo_k < s0:
                        lo_k = s0
                    hi_k = (t + 1) * ks
                    if hi_k > s1:
                        hi_k = s1
                    b = si % nb
                    cute.arch.mbarrier_wait(acc_empty + b, ((si // nb) & 1) ^ 1)
                    _fence_after()  # the epilogue's TMEM reads before these MMAs
                    for kk in cutlass.range(lo_k, hi_k, unroll=1):
                        step = kk - s0
                        s = step % sw
                        q = step % sx
                        if cutlass.const_expr((self.debug & 4) == 0):
                            cute.arch.mbarrier_wait(w_full + s, (step // sw) & 1)
                        if cutlass.const_expr((self.debug & 2) == 0):
                            cute.arch.mbarrier_wait(x_full + q, (step // sx) & 1)
                        if cutlass.const_expr((self.debug & 1) == 0):
                            for c in cutlass.range_constexpr(chunks):
                                acc_c = acc[(None, None, None, b * chunks + c)]
                                for kb in cutlass.range_constexpr(kblocks):
                                    if cutlass.const_expr(kb == 0):
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, kk != lo_k)
                                    else:
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(tiled_mma, acc_c, tCrA[(None, None, kb, s)],
                                              tCrB[(None, None, kb, q * chunks + c)],
                                              acc_c)  # fmt: skip
                        with cute.arch.elect_one():
                            tcgen05.commit(w_empty + s, 3, tcgen05.CtaGroup.TWO)
                            tcgen05.commit(x_empty + q, 3, tcgen05.CtaGroup.TWO)
                    with cute.arch.elect_one():
                        tcgen05.commit(acc_full + b, 3, tcgen05.CtaGroup.TWO)
        elif warp < 8:
            if warp == 0:
                if cutlass.const_expr(self.tmem_cols > 512):
                    _tmem_alloc_x(Int32(tmem_holder.toint()), self.tmem_cols)
                else:
                    cute.arch.alloc_tmem(self.tmem_cols, tmem_holder, is_two_cta=True)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if cutlass.const_expr(self.spread):
                if warp < 4:
                    self.epilogue_spread(acc, acc_full, acc_empty, red, warp, lane, v, wd,
                                         j, nd, s0, s1, stage_addr, mY, mWs, flags, stamp,
                                         0)  # fmt: skip
                else:
                    self.epilogue_spread(acc, acc_full, acc_empty, red, warp, lane, v, wd,
                                         j, nd, s0, s1, stage_addr, mY, mWs, flags, stamp,
                                         1)  # fmt: skip
            else:
                if warp < 4:
                    self.epilogue(acc, acc_full, acc_empty, xfer, warp, lane, v, g, wd, j,
                                  nd, s0, s1, stage_addr, mY, mWs, flags, stamp, 0)  # fmt: skip
                else:
                    self.epilogue(acc, acc_full, acc_empty, xfer, warp, lane, v, g, wd, j,
                                  nd, s0, s1, stage_addr, mY, mWs, flags, stamp, 1)  # fmt: skip
            if cutlass.const_expr(self.debug & 8):
                if warp == 0:
                    if lane == 0:
                        _st_u64(stamp, t_entry)
                        _st_u64(stamp + 15 * 8, _now())
            if warp == 0:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            if warp == 0:
                cute.arch.mbarrier_arrive(dealloc, rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(dealloc, 0)
                if cutlass.const_expr(self.tmem_cols > 512):
                    _tmem_dealloc_x(Int32(tmem_ptr.toint()), self.tmem_cols)
                else:
                    cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=True)

    def token(self, col):
        """Token of accumulator column ``col`` (chunk-major): MMA chunk c's columns
        [0, h) are CTA 0's rows [c h, c h + h), [h, 2 h) CTA 1's (tokens T / 2 on)."""
        c, j = divmod(col, self.n_mma)
        h = self.n_mma // 2
        return c * h + j if j < h else self.rows // 2 + c * h + (j - h)

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, acc_empty, xfer, ew: Int32, lane: Int32,
        v: Int32, g: Int32, wd: Int32, j: Int32, nd: Int32, s0: Int32, s1: Int32,
        stage_addr: Int32, mY: cute.Tensor, mWs: cute.Tensor, flags: Int64,
        stamp: Int64, half: cutlass.Constexpr[int],
    ):  # fmt: skip
        """Warp ew (0-7) reads TMEM lane quarter ew % 4 (weight row r of the CTA's 128)
        and accumulator columns [h T / 2, (h + 1) T / 2), h = ew // 4: the whole
        accumulator moves to registers and goes back to the MMA warp before any global
        store, so the stores overlap the next segment's MMAs. Per segment of the range
        [s0, s1): a later part of a tile publishes its FP32 partial ws[g][v]
        [column][row] and flags it; a first part (a whole tile, or the head of a
        shared one) adds the next range's partial (staged in the drained ring) and
        stores BF16."""
        rows, ks, total, n_mma, chunks = self.rows, self.ks, self.total, self.n_mma, self.chunks
        quarter = ew % 4
        hcols = rows // 2
        r = quarter * 32 + lane
        lane_bits = (quarter * 32) << 16
        base = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(chunks)
        ]
        ws = mWs.iterator.toint()
        y = mY.iterator.toint()
        first_col = half * hcols  # accumulator column of vals[0] (chunk-major)
        lead = ew * 32 + lane == 0
        t0 = s0 // ks
        nseg = (s1 - 1) // ks - t0 + 1
        for si in cutlass.range(nseg, unroll=1):
            t = t0 + si
            lo_k = t * ks
            if lo_k < s0:
                lo_k = s0
            b = si % self.acc_bufs
            cute.arch.mbarrier_wait(acc_full + b, (si // self.acc_bufs) & 1)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 6:
                        _st_u64(stamp + (2 + 2 * si) * 8, _now())
            _fence_after()
            if cutlass.const_expr(self.nocopy):
                if lo_k != t * ks:  # a later part: publish the FP32 partial, 16 at a time
                    wsb = ws + (Int64(g * 2 + v) * rows + first_col) * 512 + r * 4
                    for jj in cutlass.range_constexpr(hcols // CHUNK):
                        v16 = _tmem_ld16w(self.taddr(base, b, first_col + jj * CHUNK))
                        for i in cutlass.range_constexpr(CHUNK):
                            _st_f32(wsb + (jj * CHUNK + i) * 512, v16[i])
                    _fence_before()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(acc_empty + b, Int32(0))
                    _bar(2, 256)
                    if lead:
                        _red_release_add(flags + Int64(g * 2 + v) * 4, 1)
                else:  # a whole tile or the head of a shared one
                    last = (t + 1) * ks
                    ncon = (last * nd + total - 1) // total - 1 - j
                    if ncon < 0:
                        ncon = Int32(0)
                    if ncon > 0:  # the next range's partial into the drained ring
                        if ew == 0:
                            if lane == 0:
                                flag = flags + Int64((g + 1) * 2 + v) * 4
                                sgn, sgt = spin_start()
                                while _ld_acquire(flag) == 0:
                                    _nanosleep()
                                    sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                                cute.arch.mbarrier_arrive_and_expect_tx(xfer, rows * 512)
                            cute.arch.sync_warp()
                            _fence_proxy_global()
                            per = rows * 512 // 32
                            src = ws + (Int64((g + 1) * 2 + v) * rows) * 512 + lane * per
                            _bulk_load(stage_addr + lane * per, src, per, xfer)
                        cute.arch.mbarrier_wait(xfer, 0)
                    packed = []  # the FP32 sum first, then one rounding (as the copy form)
                    for jj in cutlass.range_constexpr(hcols // CHUNK):
                        v16 = _tmem_ld16w(self.taddr(base, b, first_col + jj * CHUNK))
                        for i in cutlass.range_constexpr(0, CHUNK, 2):
                            col = first_col + jj * CHUNK + i
                            a0 = _add_lds_if(v16[i], stage_addr + (col * 128 + r) * 4, ncon)
                            a1 = _add_lds_if(v16[i + 1],
                                             stage_addr + ((col + 1) * 128 + r) * 4, ncon)
                            packed.append(_pack_bf16x2(a0, a1))
                    _fence_before()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(acc_empty + b, Int32(0))
                    yrow = y + (wd * (self.n // 2) + t * 256 + v * 128 + r) * 2
                    for i in cutlass.range_constexpr(hcols // 2):
                        col = first_col + 2 * i
                        _st_bf16_pair(yrow + Int64(self.token(col)) * (self.ldo * 2),
                                      yrow + Int64(self.token(col + 1)) * (self.ldo * 2),
                                      packed[i])  # fmt: skip
                    if ncon > 0:
                        _bar(2, 256)
                        if lead:
                            _st_relaxed(flags + Int64((g + 1) * 2 + v) * 4, 0)
            else:
                vals = []
                for jj in cutlass.range_constexpr(hcols // CHUNK):
                    vals += list(_tmem_ld16w(self.taddr(base, b, first_col + jj * CHUNK)))
                _fence_before()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty + b, Int32(0))
                if cutlass.const_expr(self.debug & 8):
                    if lead:
                        if si < 2:  # TMEM released (stamps 12 / 13: segments 0 / 1)
                            _st_u64(stamp + (12 + si) * 8, _now())
                if lo_k != t * ks:  # a later part: publish the partial
                    wsb = ws + (Int64(g * 2 + v) * rows + first_col) * 512 + r * 4
                    for i in cutlass.range_constexpr(hcols):
                        _st_f32(wsb + i * 512, vals[i])
                    _bar(2, 256)
                    if lead:
                        _red_release_add(flags + Int64(g * 2 + v) * 4, 1)
                else:  # a whole tile or the head of a shared one
                    last = (t + 1) * ks
                    ncon = (last * nd + total - 1) // total - 1 - j
                    if ncon < 0:
                        ncon = Int32(0)
                    if ncon > 0:  # the next range's partial into the drained ring
                        if ew == 0:
                            if lane == 0:
                                flag = flags + Int64((g + 1) * 2 + v) * 4
                                sgn, sgt = spin_start()
                                while _ld_acquire(flag) == 0:
                                    _nanosleep()
                                    sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                                cute.arch.mbarrier_arrive_and_expect_tx(xfer, rows * 512)
                            cute.arch.sync_warp()
                            _fence_proxy_global()
                            per = rows * 512 // 32
                            src = ws + (Int64((g + 1) * 2 + v) * rows) * 512 + lane * per
                            _bulk_load(stage_addr + lane * per, src, per, xfer)
                        cute.arch.mbarrier_wait(xfer, 0)
                    yrow = y + (wd * (self.n // 2) + t * 256 + v * 128 + r) * 2
                    for i in cutlass.range_constexpr(hcols):
                        val = _add_lds_if(vals[i], stage_addr + ((first_col + i) * 128 + r) * 4,
                                          ncon)  # fmt: skip
                        tok = self.token(first_col + i)
                        _st_bf16(yrow + Int64(tok) * (self.ldo * 2), val)
                    if ncon > 0:
                        _bar(2, 256)
                        if lead:
                            _st_relaxed(flags + Int64((g + 1) * 2 + v) * 4, 0)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 6:
                        _st_u64(stamp + (3 + 2 * si) * 8, _now())

    @cute.jit
    def epilogue_spread(
        self, acc: cute.Tensor, acc_full, acc_empty, red, ew: Int32, lane: Int32,
        v: Int32, wd: Int32, j: Int32, nd: Int32, s0: Int32, s1: Int32,
        stage_addr: Int32, mY: cute.Tensor, mWs: cute.Tensor, flags: Int64,
        stamp: Int64, half: cutlass.Constexpr[int],
    ):  # fmt: skip
        """``spread``: warp ew (0-7) drains TMEM lane quarter ew % 4 and columns
        [h T / 2, (h + 1) T / 2), h = ew // 4, per segment and publishes them as nch
        2 KiB units ([4][32 lanes][4] FP32: a warp store covers 512 contiguous bytes),
        then counts the tile. After its last segment the range takes its share of every
        tile it touched: unit pair u (warps q, q + 2 and one 16-column unit) belongs to
        the tile's range (u mod c) (c ranges, K order); it bulk-loads the unit's c
        partials into the drained ring, sums them in K order and stores BF16 (silu: the
        up warp hands its sums to the gate warp, which stores silu_and_mul)."""
        rows, ks, total, nch = self.rows, self.ks, self.total, self.nch
        unit = 2048
        cta = self.cta_bytes
        quarter = ew % 4
        hcols = rows // 2
        lane_bits = (quarter * 32) << 16
        base = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(self.chunks)
        ]
        ws = mWs.iterator.toint() + Int64(wd) * self.die_stride
        y = mY.iterator.toint()
        first_col = half * hcols
        lead = ew * 32 + lane == 0
        t0 = s0 // ks
        nseg = (s1 - 1) // ks - t0 + 1
        cnt0 = flags + Int64(self.cnt_byte + wd * self.tpd * 8)
        done0 = flags + Int64(self.done_byte + wd * self.tpd * 8)
        for si in cutlass.range(nseg, unroll=1):
            t = t0 + si
            b = si % self.acc_bufs
            cute.arch.mbarrier_wait(acc_full + b, (si // self.acc_bufs) & 1)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 4:
                        _st_u64(stamp + (2 + 2 * si) * 8, _now())
            _fence_after()
            vals = []
            for jj in cutlass.range_constexpr(nch):
                vals += list(_tmem_ld16w(self.taddr(base, b, first_col + jj * CHUNK)))
            _fence_before()
            if lane == 0:
                cute.arch.mbarrier_arrive(acc_empty + b, Int32(0))
            jf = ((t * ks + 1) * nd + total - 1) // total - 1
            dst = ws + Int64(((t * self.cmax + j - jf) * 2 + v) * cta + ew * nch * unit)
            dst = dst + Int64(lane * 16)
            for jj in cutlass.range_constexpr(nch):
                for q4 in cutlass.range_constexpr(4):
                    i0 = jj * CHUNK + q4 * 4
                    _st_v4(dst + (jj * unit + q4 * 512), vals[i0], vals[i0 + 1],
                           vals[i0 + 2], vals[i0 + 3])  # fmt: skip
            _bar(2, 256)
            if lead:
                _red_release_add(cnt0 + Int64(t * 2 + v) * 4, 1)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 4:
                        _st_u64(stamp + (3 + 2 * si) * 8, _now())
        ph = Int32(0)
        up_id = half * 2 + quarter % 2
        p = quarter // 2
        sums = cute.make_rmem_tensor(CHUNK, Float32)
        for si in cutlass.range(nseg, unroll=1):
            t = t0 + si
            jf = ((t * ks + 1) * nd + total - 1) // total - 1
            jl = ((t + 1) * ks * nd + total - 1) // total - 1
            c = jl - jf + 1
            me = j - jf
            cnt = cnt0 + Int64(t * 2 + v) * 4
            if lead:
                sgn, sgt = spin_start()
                while _ld_acquire(cnt) < c:
                    _nanosleep()
                    sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
            _bar(2, 256)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si == 0:
                        _st_u64(stamp + 10 * 8, _now())
            _fence_proxy_global()
            src = ws + Int64((t * self.cmax * 2 + v) * cta + ew * nch * unit)
            own = Int32(0)
            for jj in cutlass.range_constexpr(nch):
                if (up_id * nch + jj) % c == me:
                    own = own + 1
            if own > 0:
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(red + ew, own * c * unit)
                cute.arch.sync_warp()
                for jj in cutlass.range_constexpr(nch):
                    u = up_id * nch + jj
                    if u % c == me:
                        sdst = stage_addr + ((u // c) * 2 + p) * c * unit
                        for i in cutlass.range(lane, c, 32, unroll=1):
                            _bulk_load(sdst + i * unit, src + Int64(i * 2 * cta + jj * unit),
                                       unit, red + ew)  # fmt: skip
                cute.arch.mbarrier_wait(red + ew, ph)
                ph = ph ^ 1
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si == 0:
                        _st_u64(stamp + 11 * 8, _now())
            for jj in cutlass.range_constexpr(nch):
                u = up_id * nch + jj
                if u % c == me:
                    sb = stage_addr + ((u // c) * 2 + p) * c * unit + lane * 16
                    for e in cutlass.range_constexpr(CHUNK):
                        sums[e] = Float32(0)
                    for i in cutlass.range(c, unroll=1):
                        for q4 in cutlass.range_constexpr(4):
                            a = _lds_v4(sb + i * unit + q4 * 512)
                            for e in cutlass.range_constexpr(4):
                                sums[q4 * 4 + e] = sums[q4 * 4 + e] + a[e]
                    if cutlass.const_expr(self.silu):
                        xb = stage_addr + ((u // c) * 2 + 1) * c * unit + lane * 16
                        if p == 1:  # up rows: hand the sums to the gate warp
                            for q4 in cutlass.range_constexpr(4):
                                _sts_v4(xb + q4 * 512, sums[q4 * 4], sums[q4 * 4 + 1],
                                        sums[q4 * 4 + 2], sums[q4 * 4 + 3])  # fmt: skip
                        _bar_n(3 + up_id, 64)
                        if p == 0:
                            col = wd * (self.n // 4) + t * 128 + v * 64 + quarter * 32 + lane
                            for q4 in cutlass.range_constexpr(4):
                                ups = _lds_v4(xb + q4 * 512)
                                for e in cutlass.range_constexpr(4):
                                    gate = _round_bf16(sums[q4 * 4 + e])
                                    up = _round_bf16(ups[e])
                                    act = div_rn(gate, Float32(1) + cute.math.exp(
                                        -gate, fastmath=False))  # fmt: skip
                                    act = Float32(cutlass.BFloat16(act)) * (up + Float32(0))
                                    tok = self.token(first_col + jj * CHUNK + q4 * 4 + e)
                                    _st_bf16(y + (Int64(tok) * self.ldo + col) * 2, act)
                    else:
                        row = wd * (self.n // 2) + t * 256 + v * 128 + quarter * 32 + lane
                        for e in cutlass.range_constexpr(CHUNK):
                            tok = self.token(first_col + jj * CHUNK + e)
                            _st_bf16(y + (Int64(tok) * self.ldo + row) * 2, sums[e])
            _bar(2, 256)
            if lead:  # the tile's last range to finish resets its counters
                done = done0 + Int64(t * 2 + v) * 4
                if _atom_add(done, 1) == c - 1:
                    _st_relaxed(cnt, 0)
                    _st_relaxed(done, 0)
        if cutlass.const_expr(self.debug & 8):
            if lead:
                _st_u64(stamp + 12 * 8, _now())

    def taddr(self, base, b, col):
        """TMEM address of accumulator column ``col`` (chunk-major) of buffer b."""
        c, off = divmod(col, self.n_mma)
        return base[c] + b * self.rows + off


_COMPILED = {}


def die_pairs(device):
    """CTA pairs per die (two dies required)."""
    index = torch.device(device).index
    doms = dies.sm_domains(torch.cuda.current_device() if index is None else index)
    pairs = tuple(sum(1 for d in doms if d == die) // 2 for die in (0, 1))
    if min(pairs) == 0:
        raise ValueError(f"die-local GEMM needs two dies with SMs, got {pairs}")
    return doms, pairs


def place_halves(weight):
    """``weight`` [N, K] in one localized allocation: the 2 MiB chunks holding (most
    of) rows [0, N / 2) on die 0, the rest on die 1."""
    chunk = 2 << 20
    nbytes = weight.numel() * weight.element_size()
    half = nbytes // 2
    domains = [
        0 if i * chunk + chunk // 2 <= half else 1 for i in range(-(-nbytes // chunk))
    ]
    return dies.localized_like(weight.contiguous(), domains)


def silu_rows(weight):
    """A merged [gate; up] weight [2 I, K] with its rows interleaved per 64: rows
    [128 b, 128 b + 64) gate rows [64 b, 64 b + 64), the next 64 the same up rows."""
    n, k = weight.shape
    return weight.view(2, n // 128, 64, k).transpose(0, 1).reshape(n, k)


class PreparedWideSK:
    """``WideSK`` per prepared row count on one [N, K] BF16 weight. ``place``: copy it
    into die-homed halves (else the kernel streams it where it lies, e.g. striped).
    ``spread``: the any-ranges-per-tile reduction; ``silu``: ``weight`` is a merged
    [gate; up] weight and the output its silu_and_mul [T, N / 2].
    ``run(x, out)``: out [T, N] BF16 (row pitch ``ldo``; [T, N / 2] with ``silu``)."""

    def __init__(self, weight, rows, ldo=None, place=True, w_issuers=3, x_issuers=2,
                 debug=0, x_stages=4, spread=False, silu=False, share=None, smem=None,
                 packed=False, keep_w=False, nocopy=True):  # fmt: skip
        """``share``: a prepared instance of the same shape and options whose workspace
        and counters this one reuses (launches on one stream never overlap); ``smem``:
        the shared-memory budget (default: the 227 KiB opt-in; above it, in Rubin's
        oversized mode, the 288-row LM head measured 2x slower); ``packed``: stream the
        halves as ``pack_tiles`` copies."""
        self.share = share
        self.doms, self.pairs = die_pairs(weight.device)
        if silu:
            weight = silu_rows(weight)
        self.n, self.k = weight.shape
        if packed:
            h = self.n // 2
            weight = torch.cat([pack_tiles(weight[:h]), pack_tiles(weight[h:])])
        self.weight = place_halves(weight) if place else weight.contiguous()
        self.out_n = self.n // 2 if silu else self.n
        self.rows = tuple(rows)
        self.ldo = ldo
        self.opts = (w_issuers, x_issuers, debug, x_stages, spread, silu, smem, packed,
                     keep_w, nocopy)
        self.kernels, self.state = {}, {}

    def prepare(self):
        dev = self.weight.device
        self.w0 = self.weight.data_ptr()  # each half: N / 2 x K elements (either layout)
        self.w1 = self.w0 + self.n // 2 * self.k * self.weight.element_size()
        self.table = torch.tensor(self.doms, dtype=torch.int32, device=dev)
        self.table_c = from_dlpack(self.table, assumed_align=16)
        npairs = sum(self.pairs)
        for rows in self.rows:
            gemm = WideSK(rows, self.n, self.k, self.pairs, self.ldo, *self.opts)
            x = torch.empty(rows, self.k, dtype=torch.bfloat16, device=dev)
            y = torch.empty(rows, self.ldo or self.out_n, dtype=torch.bfloat16, device=dev)
            if self.share is not None:
                assert (self.share.n, self.share.k, self.share.opts) == (
                    self.n, self.k, self.opts)  # fmt: skip
                ws, flags = self.share.state[rows][:2]
            elif gemm.spread:  # each die's partials homed on that die
                chunks = gemm.die_stride // (2 << 20)
                ws = dies.localized_empty(2 * gemm.die_stride, [0] * chunks + [1] * chunks,
                                          dev)  # fmt: skip
            else:
                ws = torch.empty(npairs, 2, rows, 128, dtype=torch.float32, device=dev)
            if self.share is None:
                flags = torch.zeros(2 * npairs + 2 + 4 * npairs * STAMPS + 8 * gemm.tpd,
                                    dtype=torch.int32, device=dev)  # fmt: skip
            # The weight / activation addresses are launch values (``run`` passes
            # them every call): DSL Int64 values make them runtime arguments of the
            # compiled kernel, so its code (and CuTe cache key) does not depend on
            # where this process's buffers happen to live.
            args = (Int64(self.w0), Int64(self.w1), Int64(x.data_ptr()),
                    Int64(x.data_ptr()), self._y(y[:, : self.out_n]),
                    from_dlpack(ws, assumed_align=16),
                    from_dlpack(flags, assumed_align=16), self.table_c)  # fmt: skip
            key = (rows, self.n, self.k, self.pairs, self.ldo, self.opts)
            if key not in _COMPILED:
                stream = _cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
                _COMPILED[key] = cute.compile(gemm, *args, stream)
            self.kernels[rows] = _COMPILED[key]
            self.state[rows] = (ws, flags, args[5], args[6])
        return self

    @staticmethod
    def _k(t):
        return from_dlpack(t.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )

    @staticmethod
    def _y(y):
        return from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def stamps(self, rows):
        """[2 * pairs, STAMPS] int64 stamps of the last ``debug & 8`` launch."""
        flags = self.state[rows][1]
        npairs = sum(self.pairs)
        st = flags[2 * npairs + 2 : 2 * npairs + 2 + 4 * npairs * STAMPS]
        return st.view(torch.int64).view(2 * npairs, STAMPS)

    def takes(self, x):
        return (
            x.dim() == 2 and x.shape[0] in self.kernels and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.shape[1] == self.k
        )  # fmt: skip

    def run(self, x, out, x1=None):
        """x1: a copy of x homed on die 1 (x then on die 0), else the
        CTAs of both dies read x."""
        rows = x.shape[0]
        _, _, ws, flags = self.state[rows]
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        xa = x.data_ptr()
        self.kernels[rows](self.w0, self.w1, xa, xa if x1 is None else x1.data_ptr(),
                           self._y(out), ws, flags, self.table_c, stream)  # fmt: skip
        return out
