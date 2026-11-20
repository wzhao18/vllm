# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-wide swap-AB BF16 GEMM for the DSpark drafter's per-layer linears at 96-288
rows (TP4 shards: qkv, o_proj, gate_up, down).

y[T, N] = x[T, K] @ W[N, K]^T with the weight rows on the 2-CTA MMA's M (256 rows per
CTA pair, 128 per CTA) and ALL T tokens on N (one MMA of N = T up to 256 columns, else
two of T / 2), so every activation byte is read once per 256 weight rows: at these row
counts the all-SM broadcast of x out of L2 (hw37: ~24-46 B/clk/SM), not the weight
stream, is what holds a 128-row-tile library GEMM at ~1 PF. K splits over ``split``
CTA pairs per tile (one wave on the SMs); the splits exchange FP32 partials through
global memory by 16-token column chunk: split s owns chunks j with j % split == s,
writes its partials of the other chunks, and after every split of the tile has
published (one release counter per 128-row tile) sums its chunks in split order
(p0 + p1) + ... (deterministic; an FP32 accumulation reorder of the stock BF16
linear), rounds once to BF16 and stores. The weight warp issues its TMA ring before
the PDL wait; the activations follow it.

``silu``: the gate_up form. The weight rows are permuted at prepare so every 128-row
tile holds 64 gate rows then their 64 up rows; the owner applies vLLM's silu_and_mul
(gate and up rounded to BF16, silu(gate) = gate / (1 + exp(-gate)) in IEEE FP32
rounded to BF16, times up in FP32, BF16 out) and writes h [T, N / 2].
"""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ..common.primitives import asm, div_rn

CHUNK = 16  # token columns per exchange chunk (one tcgen05.ld 32x32b.x16)
K_TILE = 64  # K per ring stage (one 128-byte swizzle atom)
PT = 32  # tokens per bulk piece of the split exchange (16 KiB of FP32 partials)


@asm()
def _tmem_ld16(taddr):
    regs = ",".join(f"${i}" for i in range(16))
    code = "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16];"
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 16


@asm()
def _tmem_wait16(*vals):
    cons = ",".join(["=f"] * 16) + "," + ",".join(str(i) for i in range(16))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * 16


@asm()
def _st_f32(address, value):
    return "st.global.f32 [$0], $1;", "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Float32)
def _ld_f32_cg(address):
    return "ld.global.cg.f32 $0, [$1];", "=f,l,~{memory}", (Int64(address),)


@asm()
def _st_bf16(address, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }"
    return code, "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Float32, pure=True)
def _round_bf16(value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }"
    return code, "=f,f", (Float32(value),)


@asm()
def _red_release_add(address, value):
    code = "red.release.gpu.global.add.s32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int32)
def _atom_add(address, value):
    code = "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int32)
def _ld_acquire(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def _st_relaxed(address, value):
    code = "st.relaxed.gpu.global.b32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm()
def _st_v4(address, a, b, c, d):
    code = "st.global.v4.f32 [$0], {$1, $2, $3, $4};"
    return code, "l,f,f,f,f,~{memory}", (Int64(address), a, b, c, d)


@asm((Float32,) * 4)
def _ld_v4(address):
    code = "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];"
    return code, "=f,=f,=f,=f,l", (Int64(address),)


@asm()
def _st_v4_b32(address, a, b, c, d):
    code = "st.global.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "l,r,r,r,r,~{memory}", (Int64(address), a, b, c, d)


@asm()
def _sts_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


@asm(Float32)
def _lds_f32(address):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Int32(address),)


@asm(cutlass.Uint32, pure=True)
def _pack_bf16(lo, hi):
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (Float32(lo), Float32(hi))


@asm(Int64)
def _now():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Int64)
def _clk():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


@asm()
def _st_u64(address, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm()
def _fence_proxy_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def _fence_proxy_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm()
def _bulk_store(gaddr, saddr, nbytes):
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (Int64(gaddr), Int32(saddr), Int32(nbytes))


@asm()
def _bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm()
def _bulk_wait():
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def _bulk_load(saddr, gaddr, nbytes, mbar):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    code += "[$0], [$1], $2, [$3];"
    return code, "r,l,r,r,~{memory}", (Int32(saddr), Int64(gaddr), Int32(nbytes), mbar)


@asm((Float32,) * 4)
def _lds_v4(address):
    code = "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];"
    return code, "=f,=f,=f,=f,r,~{memory}", (Int32(address),)


@asm()
def _nanosleep():
    return "nanosleep.u32 64;", "~{memory}", ()


@asm(Int32)
def _pace_state(target, mbar):
    """2 if ``mbar`` (the PDL-wait barrier) completed phase 0, else 1 while
    %clock64 < ``target``, else 0 (a paced pre-wait issuer spins while 1)."""
    code = "{ .reg .pred p, q; .reg .u64 t; .reg .u32 a; "
    code += "mbarrier.test_wait.parity.shared::cta.b64 q, [$2], 0; "
    code += "mov.u64 t, %clock64; setp.lt.u64 p, t, $1; selp.u32 a, 1, 0, p; "
    code += "selp.u32 $0, 2, a, q; }"
    return code, "=r,l,r,~{memory}", (Int64(target), mbar)


@asm()
def _fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def _st_relaxed_f32(address, value):
    return "st.relaxed.gpu.global.f32 [$0], $1;", "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Int32)
def _ld_relaxed_b32(address):
    return "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Float32, pure=True)
def _as_f32(bits):
    return "mov.b32 $0, $1;", "=f,r", (Int32(bits),)


@asm()
def _st_bf16_f32(address, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }"
    return code, "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Int32, pure=True)
def _sel_eq(a, b, x, y):
    code = "{ .reg .pred p; setp.eq.s32 p, $1, $2; selp.b32 $0, $3, $4, p; }"
    return code, "=r,r,r,r,r", (Int32(a), Int32(b), Int32(x), Int32(y))


@asm(Int32, pure=True)
def _sel_neg(a, x, y):
    code = "{ .reg .pred p; setp.lt.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }"
    return code, "=r,r,r,r", (Int32(a), Int32(x), Int32(y))


@asm(Int32)
def _smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Int32)
def _ld_nc_s32(address):
    return "ld.global.nc.s32 $0, [$1];", "=r,l", (Int64(address),)


@asm()
def _bar(barrier_id, count):
    return f"bar.sync {barrier_id}, {count};", "~{memory}", ()


def mma_columns(rows):
    """(MMA N, chunks) covering ``rows`` tokens: the fewest equal MMAs of N <= 256
    columns with N / 2 (each CTA's B rows) a multiple of 16."""
    for chunks in range(1, 5):
        n = rows // chunks
        if n * chunks == rows and n <= 256 and n % 32 == 0:
            return n, chunks
    return rows, 0


ACT_WARPS = (1, 4, 5, 6, 7)  # warps that may issue activation stages (4-7: before the epilogue)
WEIGHT_WARPS = (0, 3, 4, 5, 6, 7)  # warps that may issue weight stages (4-7: before the epilogue)
STAMP_BYTES = 128  # %globaltimer stamps per CTA after the split flags (see ``kernel``)
SENTINEL = -1  # an unwritten exchange slot (FP32 bits no BF16 product sums to)


def _rings(rows, wstages=None, budget=None, ktile=K_TILE):
    """(weight stages, activation stages, ring bytes): separate rings, the weight
    ring as deep as the activation ring (>= min(4, weight stages)) leaves room for,
    so most of a split's weights are staged before the PDL wait. ``wstages`` caps
    the weight ring (the activation ring takes the rest, up to 16)."""
    n_mma, chunks = mma_columns(rows)
    a_bytes = 128 * ktile * 2
    b_bytes = chunks * (n_mma // 2) * ktile * 2
    default = utils.get_smem_capacity_in_bytes("sm_100") - 4096
    budget = budget or default
    if wstages is not None:
        acts = min(16, (budget - wstages * a_bytes) // b_bytes)
        if acts < 2:
            raise ValueError(f"wide GEMM: {wstages} weight stages leave no activation ring")
        return wstages, acts, wstages * a_bytes + acts * b_bytes
    for ws in range(12 if budget <= default else 16, 1, -1):
        acts = min(ws, (budget - ws * a_bytes) // b_bytes)
        if acts >= min(4, ws):
            return ws, acts, ws * a_bytes + acts * b_bytes
    raise ValueError(f"wide GEMM: no ring fits {rows} rows")


def _exchange_fits(rows, split, wstages=None):
    """Uneven token regions (``split`` not dividing ``rows``) receive the peers'
    partials beside the [token][row] staging, which must fit in the drained ring."""
    if rows % split == 0:
        return True
    return (rows + split * -(-rows // split)) * 512 <= _rings(rows, wstages)[2]


def supports(rows, n, k, split, silu=False, out_f32=False):
    n_mma, chunks = mma_columns(rows)
    return (
        rows % 32 == 0
        and n_mma % 16 == 0
        and n_mma <= 256
        and chunks > 0
        and n % (256 if silu else 128) == 0
        and not (silu and out_f32)
        and k % K_TILE == 0
        and 1 <= split <= k // K_TILE
        and split <= rows
        and _exchange_fits(rows, split)
    )


class WideGemm:
    """One (rows, N, K, split, silu) specialization; see the module docstring.

    ``act_warps``: the warps that issue the activation stages round-robin after the
    PDL wait (each TMA issuing thread completes one op at a time, ~0.4-0.5 us per op
    of any size, hw51: one issuer serializes a split's K tiles)."""

    def __init__(self, rows, n, k, split, silu=False, act_warps=(1,), out_f32=False,
                 wstages=None, weight_warps=(0, 3), direct=False, rotate=False,
                 partials=False, tpcs=0, smem=None, trace=False, xcopies=1,
                 ktile=K_TILE, prewait=None):  # fmt: skip
        assert supports(rows, n, k, split, silu, out_f32), (rows, n, k, split, silu)
        # Experiments only (co-residency vs pre-wait traffic behind a latency-bound
        # producer): weight bytes issued before the PDL wait. None: the whole ring;
        # ("none",): none; ("n", s): the first s stages; ("pace", clk): stage t not
        # before clock t0 + t * clk until the wait passes; ("l2", clk): TMA L2
        # prefetch of the split's stages at that pace, the ring filled after the wait.
        self.prewait = tuple(prewait) if prewait else ("all",)
        assert self.prewait[0] in ("all", "none", "n", "pace", "l2"), prewait
        # Benchmarks only: ``trace`` stores the MMA warp's clock64 waits on weight /
        # activation stages; ``xcopies`` reads x from x[(tile % xcopies) * rows + t]
        # (copies of the same rows: the hot-activation ablation).
        self.trace, self.xcopies = trace, xcopies
        # K per ring stage: 64 (one swizzle atom) or 128 (two TMA boxes on one barrier:
        # a stage costs its issuer ~one op time whatever its ops, hwdb TMA stage law).
        assert ktile in (64, 128) and k % ktile == 0 and split <= k // ktile, (k, ktile)
        self.ktile = ktile
        # ``tpcs`` > 0: die-static work. One CTA pair per TPC (every SM); each pair
        # reads its (tile, split) from the table after the split flags (``table_at``,
        # indexed by %smid / 2: a 2-CTA cluster sits on one TPC, smids 2j / 2j + 1),
        # so it streams weight rows homed on its own die (PreparedWide places them).
        self.tpcs = tpcs
        self.act_warps = tuple(sorted(act_warps))
        assert self.act_warps[0] == 1 and set(act_warps) <= set(ACT_WARPS), act_warps
        self.rows, self.n, self.k, self.split, self.silu = rows, n, k, split, silu
        self.out_f32 = out_f32
        # The split exchange through sentinel-armed FP32 slots (``epilogue_direct``).
        self.direct = direct and not silu
        self.rotate = rotate
        # ``partials``: no split exchange; y is [split * T, N] FP32, split s's FP32
        # partial in rows [s T, (s + 1) T) (the consumer sums them in split order).
        self.partials = partials
        assert not (partials and (silu or direct)), (partials, silu, direct)
        self.weight_warps = tuple(sorted(weight_warps))
        assert self.weight_warps[:2] == (0, 3) and set(weight_warps) <= set(WEIGHT_WARPS)
        assert not set(self.weight_warps) & set(self.act_warps), (weight_warps, act_warps)
        self.n_pad = -(-n // 256) * 256  # CTA pairs cover 256 rows (TMA zero-fills past n)
        self.n_mma, self.chunks = mma_columns(rows)
        self.mma_tiler = (256, self.n_mma, ktile)
        self.threads = 256  # warps: 0+3 weights, 1 activations, 2 MMA, 4-7 epilogue
        # smem: ring budget in KiB (Rubin's oversized mode, up to 320), None = default.
        self.budget = smem * 1024 - 4096 if smem else None
        self.wstages, self.astages, _ = _rings(rows, wstages, self.budget, ktile)
        self.maxreg = -(-rows // split)  # tokens per owner (largest region)
        self.uneven = rows % split != 0
        # Pipelined split exchange (``exchange_fast``): owner regions of whole
        # PT-token bulk pieces, BF16 out.
        self.fast_x = (split > 1 and not self.uneven and not silu and not out_f32
                       and (rows // split) % PT == 0)  # fmt: skip
        self.tmem_cols = 32
        while self.tmem_cols < rows:
            self.tmem_cols *= 2
        # int32 offsets in ``flags``: split counters, then STAMP_BYTES per CTA slot
        # (split x n_pad / 128 work slots, then one per launched CTA for idle pairs),
        # then the die-static work table.
        self.stamp_slots = split * (self.n_pad // 128) + 2 * tpcs
        self.table_at = 2 * self.n_pad // 128 + self.stamp_slots * STAMP_BYTES // 4

    @cute.jit
    def __call__(
        self, w: cute.Tensor, x: cute.Tensor, y: cute.Tensor, ws: cute.Tensor,
        flags: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        """w (N, K, 1) and x (T, K, 1) K-major; y [T, N or N / 2] row-major BF16;
        ws [split, N, T] FP32; flags [2 * N / 128] int32 (zeroed once)."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.TWO, self.mma_tiler[:2]
        )
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, dtype, self.wstages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, dtype, self.astages * self.chunks
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((2, 1), tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((2, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_a, tma_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w, a_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        tma_b, tma_x = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        a_bytes = cute.size_in_bytes(dtype, a_one)
        b_bytes = cute.size_in_bytes(dtype, b_one)
        self.tx_a, self.tx_b = a_bytes * 2, self.chunks * b_bytes * 2

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.wstages]
            empty: cute.struct.MemRange[cutlass.Int64, self.wstages]
            bfull: cute.struct.MemRange[cutlass.Int64, self.astages]
            bempty: cute.struct.MemRange[cutlass.Int64, self.astages]
            acc_full: cutlass.Int64
            dealloc: cutlass.Int64
            dep: cutlass.Int64
            tmem_holder: cutlass.Int32
            xfer: cutlass.Int64
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout.outer)], 1024
            ]

        self.storage = Storage
        self.kernel(
            tiled_mma, tma_a, tma_w, tma_b, tma_x, cluster_vmnk, a_layout, b_layout,
            y, ws, flags,
        ).launch(  # fmt: skip
            grid=(2 * self.tpcs, 1, 1) if self.tpcs else (2 * (self.n_pad // 256), self.split, 1),
            block=[self.threads, 1, 1],
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mW: cute.Tensor,
        tma_b: cute.CopyAtom, mX: cute.Tensor, cluster_vmnk: cute.Layout,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
        mY: cute.Tensor, mWs: cute.Tensor, mFlags: cute.Tensor,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, split_rank, _ = cute.arch.block_idx()
        v = bidx % 2  # rank in the CTA pair; 0 issues the MMAs
        leader = v == 0
        tiles = self.n_pad // 256
        slot = split_rank * (self.n_pad // 128) + bidx  # stamp slot
        work = Int32(0)
        if cutlass.const_expr(self.tpcs):
            # Die-static work of this TPC: tile << 8 | split, or < 0 (idle pair: no
            # K tiles, a tile index past n, so it skips every load and store).
            table = mFlags.iterator.toint() + Int64(self.table_at * 4)
            work = _ld_nc_s32(table + Int64(_smid() >> 1) * 4)
            slot = _sel_neg(work, Int32(self.split * (self.n_pad // 128)) + bidx,
                            (work & 255) * (self.n_pad // 128) + 2 * (work >> 8) + v)  # fmt: skip
            split_rank = _sel_neg(work, Int32(0), work & 255)
            bidx = 2 * _sel_neg(work, Int32(tiles), work >> 8) + v
        pair = bidx // 2
        xoff = (pair % self.xcopies) * self.chunks  # 0 unless the xcopies ablation
        rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_vmnk = cluster_vmnk.get_flat_coord(rank_in_cluster)
        wst, ast, chunks = self.wstages, self.astages, self.chunks

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        bfull, bempty = st.bfull.data_ptr(), st.bempty.data_ptr()
        acc_full, dealloc = st.acc_full.ptr, st.dealloc.ptr
        tmem_holder = st.tmem_holder.ptr
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(wst):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(ast):
                    cute.arch.mbarrier_init(bfull + i, 1)
                    cute.arch.mbarrier_init(bempty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(dealloc, 32)
                cute.arch.mbarrier_init(st.dep.ptr, 1)
                cute.arch.mbarrier_init(st.xfer.ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        # K range of this split (64-wide tiles, as even as possible).
        k_tiles = self.k // self.ktile
        k0 = split_rank * k_tiles // self.split
        k_count = (split_rank + 1) * k_tiles // self.split - k0
        if cutlass.const_expr(self.tpcs):
            k_count = _sel_neg(work, Int32(0), k_count)
        # Pairs of one split walk its K tiles from staggered starts, so the CTAs
        # sharing an activation slice do not read the same lines in lock step
        # (hw37: a hot L2 region broadcast to every SM runs at ~1/3 the spread rate).
        rot = Int32(0)
        if cutlass.const_expr(self.rotate):
            rot = pair * k_count // (self.n_pad // 256)

        gW = cute.local_tile(mW, cute.slice_(self.mma_tiler, (None, 0, None)),
                             (None, None, None))  # fmt: skip
        gX = cute.local_tile(mX, cute.slice_(self.mma_tiler, (0, None, None)),
                             (None, None, None))  # fmt: skip
        thr_mma = tiled_mma.get_slice(v)
        tCgW = thr_mma.partition_A(gW)
        tCgX = thr_mma.partition_B(gX)
        a_cta = cute.make_layout(cute.slice_(cluster_vmnk, (0, 0, None, 0)).shape)
        b_cta = cute.make_layout(cute.slice_(cluster_vmnk, (0, None, 0, 0)).shape)
        tAsA, tAgW = cpasync.tma_partition(
            tma_a, coord_vmnk[2], a_cta, cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )  # fmt: skip
        tBsB, tBgX = cpasync.tma_partition(
            tma_b, coord_vmnk[1], b_cta, cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )  # fmt: skip
        a_mask = cpasync.create_tma_multicast_mask(cluster_vmnk, coord_vmnk, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(cluster_vmnk, coord_vmnk, mcast_mode=1)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, chunks))

        cute.arch.cluster_wait()
        # Stamps (u64 %globaltimer, STAMP_BYTES per CTA after the split flags):
        # +0 epilogue start, +8 acc_full, +16 TMEM staged, +24 all splits published,
        # +32 peers' partials landed, +40 stored, +48 PDL wait done (activations),
        # +56 activations issued, +64 first full stage, +72 last full stage,
        # +80 weights issued; K tile 8: +88 MMA loop top, +96 stages full, +104 issued.
        stamp = mFlags.iterator.toint() + self.n_pad // 16 + Int64(slot) * STAMP_BYTES
        n_act, n_w = len(self.act_warps), len(self.weight_warps)

        if warp == 0:  # weights: before the PDL wait (stages 0, n_w, ...)
            self.weights(tma_a, tAgW[(None, pair, None, 0)], tAsA, full, empty, a_mask,
                         leader, k0, k_count, rot, Int32(0), st.dep.ptr, n_w)  # fmt: skip
            if lane == 0:
                _st_u64(stamp + 80, _now())
        elif warp == 3:  # weights: stages 1, 1 + n_w, ... (one op at a time per issuer, hw51)
            self.weights(tma_a, tAgW[(None, pair, None, 0)], tAsA, full, empty, a_mask,
                         leader, k0, k_count, rot, Int32(1), st.dep.ptr, n_w)  # fmt: skip
        elif warp == 1:  # activations: after it
            if work >= 0:  # an idle die-static pair exits without waiting on the producer
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.prewait[0] in ("pace", "l2")):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(st.dep.ptr)
            if lane == 0:
                _st_u64(stamp + 48, _now())
            self.acts(tma_b, tBgX, tBsB, bfull, bempty, b_mask, leader, k0, k_count,
                      rot, Int32(0), xoff, n_act)  # fmt: skip
            if lane == 0:
                _st_u64(stamp + 56, _now())
        elif warp == 2:
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 5)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if leader:
                kblocks = cute.size(tCrA, mode=[2])
                w_wait, a_wait = Int64(0), Int64(0)
                for t in cutlass.range(k_count, unroll=1):
                    s, sb = t % wst, t % ast
                    if t == 8:
                        if lane == 0:
                            _st_u64(stamp + 88, _now())
                    if cutlass.const_expr(self.trace):
                        c0 = _clk()
                    cute.arch.mbarrier_wait(full + s, (t // wst) % 2)
                    if cutlass.const_expr(self.trace):
                        c1 = _clk()
                    cute.arch.mbarrier_wait(bfull + sb, (t // ast) % 2)
                    if cutlass.const_expr(self.trace):
                        c2 = _clk()
                        w_wait = w_wait + (c1 - c0)
                        a_wait = a_wait + (c2 - c1)
                    if t == 0:
                        if lane == 0:
                            _st_u64(stamp + 64, _now())
                    if t == 8:
                        if lane == 0:
                            _st_u64(stamp + 96, _now())
                    for c in cutlass.range_constexpr(chunks):
                        acc_c = acc[(None, None, None, c)]
                        for kb in cutlass.range_constexpr(kblocks):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, (t + kb) != 0)
                            cute.gemm(tiled_mma, acc_c, tCrA[(None, None, kb, s)],
                                      tCrB[(None, None, kb, sb * chunks + c)], acc_c)  # fmt: skip
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + s, 3, tcgen05.CtaGroup.TWO)
                        tcgen05.commit(bempty + sb, 3, tcgen05.CtaGroup.TWO)
                    if t == 8:
                        if lane == 0:
                            _st_u64(stamp + 104, _now())
                if lane == 0:
                    _st_u64(stamp + 72, _now())
                    if cutlass.const_expr(self.trace):
                        # clock64 the MMA warp spent waiting on weight / activation stages
                        _st_u64(stamp + 112, w_wait)
                        _st_u64(stamp + 120, a_wait)
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full, 3, tcgen05.CtaGroup.TWO)
        elif warp >= 4:
            if warp == 4:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder, is_two_cta=True)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 5)
            # Extra weight / activation issuers (after the TMEM barrier the MMA warp
            # waits on, so they never hold it up).
            for i in cutlass.range_constexpr(2, n_w):
                if warp == self.weight_warps[i]:
                    self.weights(tma_a, tAgW[(None, pair, None, 0)], tAsA, full, empty,
                                 a_mask, leader, k0, k_count, rot, Int32(i), st.dep.ptr, n_w)  # fmt: skip
            for i in cutlass.range_constexpr(1, n_act):
                if warp == self.act_warps[i]:
                    if work >= 0:
                        cute.arch.griddepcontrol_wait()
                    self.acts(tma_b, tBgX, tBsB, bfull, bempty, b_mask, leader, k0,
                              k_count, rot, Int32(i), xoff, n_act)  # fmt: skip
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if bidx * 128 < self.n:
                if cutlass.const_expr(self.partials):
                    self.epilogue_partials(
                        acc, acc_full, warp - 4, lane, bidx, split_rank, mY, stamp
                    )
                elif cutlass.const_expr(self.direct):
                    self.epilogue_direct(
                        acc, acc_full, warp - 4, lane, bidx, split_rank, mY, mWs, stamp
                    )
                else:
                    self.epilogue(
                        acc, acc_full, st.xfer.ptr, warp - 4, lane, bidx, split_rank,
                        Int32(st.sA.data_ptr().toint()),
                        mY, mWs, mFlags, stamp,
                    )  # fmt: skip
            else:  # the zero-filled rows past n: nothing to store
                cute.arch.mbarrier_wait(acc_full, 0)
            if warp == 4:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if warp == 4:
                cute.arch.mbarrier_arrive(dealloc, rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(dealloc, 0)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=True)

    @cute.jit
    def weights(self, tma_a, tAgW_tile, tAsA, full, empty, a_mask, leader, k0, k_count, rot,
                first, dep, step: cutlass.Constexpr):  # fmt: skip
        """Weight stages ``first``, ``first + 2``, ... of this split: one TMA issuer
        completes one bulk op at a time (hw51), so two issuer warps double the weight
        rate a CTA can pull before and after the PDL wait. ``dep``: the barrier the
        activation warp arrives on after the PDL wait (``prewait`` pacing)."""
        stages = self.wstages
        mode = self.prewait[0]
        if cutlass.const_expr(mode == "none"):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(mode == "l2"):
            pace = Int64(self.prewait[1])
            t0 = _clk()
            t = first
            go = Int32(1)
            while go != 0:
                if t >= k_count:
                    go = Int32(0)
                else:
                    state = _pace_state(t0 + Int64(t) * pace, dep)
                    while state == 1:
                        _nanosleep()
                        state = _pace_state(t0 + Int64(t) * pace, dep)
                    if state == 2:
                        go = Int32(0)
                    else:
                        cute.prefetch(tma_a, tAgW_tile[(None, k0 + (t + rot) % k_count)])
                        t = t + step
            cute.arch.griddepcontrol_wait()
        t_wait = Int32(-1)
        if cutlass.const_expr(mode == "n"):
            # This warp's first stage at or past the pre-wait budget.
            t_wait = first + step * ((Int32(self.prewait[1]) - first + step - 1) // step)
        t_pace = Int64(0)
        if cutlass.const_expr(mode == "pace"):
            t_pace = _clk()
        for t in cutlass.range(first, k_count, step, unroll=1):
            s = t % stages
            phase = Int32(1) ^ ((t // stages) % 2)
            if cutlass.const_expr(mode == "n"):
                if t == t_wait:
                    cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(mode == "pace"):
                due = t_pace + Int64(t) * Int64(self.prewait[1])
                state = _pace_state(due, dep)
                while state == 1:
                    _nanosleep()
                    state = _pace_state(due, dep)
            cute.arch.mbarrier_wait(empty + s, phase)
            if leader:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_a)
            kt = (t + rot) % k_count
            cute.copy(tma_a, tAgW_tile[(None, k0 + kt)], tAsA[(None, s)],
                      tma_bar_ptr=full + s, mcast_mask=a_mask)  # fmt: skip

    @cute.jit
    def acts(self, tma_b, tBgX, tBsB, full, empty, b_mask, leader, k0, k_count, rot, first,
             xoff, step: cutlass.Constexpr):  # fmt: skip
        """Activation stages ``first``, ``first + step``, ... into their own ring (the
        caller has passed the PDL wait)."""
        stages, chunks = self.astages, self.chunks
        for t in cutlass.range(first, k_count, step, unroll=1):
            s = t % stages
            phase = Int32(1) ^ ((t // stages) % 2)
            cute.arch.mbarrier_wait(empty + s, phase)
            if leader:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_b)
            kt = (t + rot) % k_count
            for c in cutlass.range_constexpr(chunks):
                cute.copy(tma_b, tBgX[(None, c + xoff, k0 + kt, 0)], tBsB[(None, s * chunks + c)],
                          tma_bar_ptr=full + s, mcast_mask=b_mask)  # fmt: skip

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, xfer, ew: Int32, lane: Int32, tile: Int32,
        split_rank: Int32, stage_addr: Int32, mY: cute.Tensor, mWs: cute.Tensor,
        mFlags: cute.Tensor, stamp: Int64,
    ):  # fmt: skip
        """Thread (ew, lane) owns TMEM lane / weight row r = 32 ew + lane of 128-row
        tile ``tile``. The FP32 partial is staged as [token][row] in the drained
        operand ring; split o owns tokens [o T / S, (o + 1) T / S). Thread o issues
        the bulk op of region / peer o (one op at a time per issuing thread, hw51)."""
        rows, split = self.rows, self.split
        r = ew * 32 + lane
        tid = r
        lo = split_rank * rows // split
        mine_n = (split_rank + 1) * rows // split - lo  # tokens I own
        if tid == 0:
            _st_u64(stamp + 0, _now())
        cute.arch.mbarrier_wait(acc_full, 0)
        if tid == 0:
            _st_u64(stamp + 8, _now())
        cute.arch.griddepcontrol_launch_dependents()
        _fence_after()
        lane_bits = (ew * 32) << 16
        col0 = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(self.chunks)
        ]
        flag = mFlags.iterator.toint() + Int64(tile) * 8
        done = flag + 4
        if cutlass.const_expr(self.fast_x):
            self.exchange_fast(col0, r, split_rank, lo, stage_addr, xfer, mWs, mY,
                               tile, flag, stamp)  # fmt: skip
        else:
            self.exchange(col0, r, split_rank, lo, mine_n, stage_addr, xfer, mWs,
                          mY, tile, flag, stamp)  # fmt: skip
        if tid == 0:
            _st_u64(stamp + 40, _now())
        # 5. The last split to finish reading re-arms the tile's counters.
        if cutlass.const_expr(split > 1):
            _bar(3, 128)
            if tid == 0:
                if _atom_add(done, 1) == split - 1:
                    _st_relaxed(flag, 0)
                    _st_relaxed(done, 0)

    @cute.jit
    def exchange(self, col0, r: Int32, split_rank: Int32, lo: Int32, mine_n: Int32,
                 stage_addr: Int32, xfer, mWs: cute.Tensor, mY: cute.Tensor, tile: Int32,
                 flag: Int64, stamp: Int64):  # fmt: skip
        """Steps 1-4 of ``epilogue`` (any owner regions, gate_up / FP32 out)."""
        rows, split = self.rows, self.split
        tid = r
        ntiles = self.n_pad // 128
        # 1. TMEM -> shared [token][row] FP32 (all tokens).
        for j in cutlass.range_constexpr(rows // CHUNK):
            c, off = divmod(j * CHUNK, self.n_mma)
            vals = _tmem_wait16(*_tmem_ld16(col0[c] + off))
            for i in cutlass.range_constexpr(CHUNK):
                _sts_f32(stage_addr + ((j * CHUNK + i) * 128 + r) * 4, vals[i])
        if tid == 0:
            _st_u64(stamp + 16, _now())
        ws_tile = mWs.iterator.toint() + (Int64(split_rank) * ntiles + tile) * (rows * 512)
        mine = stage_addr + lo * 512
        # The peers' partials of my tokens: in place over their (published) regions
        # when the regions are equal, else in slots after the staging.
        if cutlass.const_expr(self.uneven):
            recv, rstride = stage_addr + rows * 512, self.maxreg * 512
        else:
            recv, rstride = stage_addr, (rows // split) * 512
        # 2. Publish the other owners' token ranges (bulk async stores), signal.
        if cutlass.const_expr(split > 1):
            _fence_proxy_shared()
            _bar(3, 128)
            for o in cutlass.range_constexpr(split):
                o_lo = o * rows // split
                o_bytes = ((o + 1) * rows // split - o_lo) * 512
                if tid == o:
                    if split_rank != o:
                        _bulk_store(ws_tile + o_lo * 512, stage_addr + o_lo * 512, o_bytes)
                        _bulk_commit()
                        _bulk_wait()
                        _fence_proxy_global()
            _bar(3, 128)
            if tid == 0:
                _red_release_add(flag, 1)
                while _ld_acquire(flag) < split:
                    _nanosleep()
                _st_u64(stamp + 24, _now())
                cute.arch.mbarrier_arrive_and_expect_tx(xfer, (split - 1) * mine_n * 512)
            _bar(3, 128)
            # 3. The peers' partials of my tokens.
            for s in cutlass.range_constexpr(split):
                if tid == s:
                    if split_rank != s:
                        _fence_proxy_global()
                        src = mWs.iterator.toint() + (Int64(s) * ntiles + tile) * (
                            rows * 512) + lo * 512  # fmt: skip
                        _bulk_load(recv + s * rstride, src, mine_n * 512, xfer)
            cute.arch.mbarrier_wait(xfer, 0)
        if tid == 0:
            _st_u64(stamp + 32, _now())
        # 4. Sum in split order, round, store 16 B per thread-iteration.
        y = mY.iterator.toint()
        if cutlass.const_expr(not self.silu):
            for it in cutlass.range_constexpr((self.maxreg * 16 + 127) // 128):
                e = it * 128 + tid  # unit of 8 rows: token e // 16, rows 8 (e % 16)
                if e < mine_n * 16:
                    self.store_plain(recv, rstride, mine, split_rank, lo, y, tile, e)
        else:
            for it in cutlass.range_constexpr((self.maxreg * 8 + 127) // 128):
                e = it * 128 + tid  # unit: token e // 8, gate rows 8 (e % 8), their up rows
                if e < mine_n * 8:
                    self.store_silu(recv, rstride, mine, split_rank, lo, y, tile, e)

    @cute.jit
    def exchange_fast(self, col0, r: Int32, split_rank: Int32, lo: Int32,
                      stage_addr: Int32, xfer, mWs: cute.Tensor, mY: cute.Tensor,
                      tile: Int32, flag: Int64, stamp: Int64):  # fmt: skip
        """Steps 1-4 of ``epilogue`` with equal owner regions of whole PT-token pieces:
        the peers' token chunks leave TMEM first and are published (16 KiB bulk pieces
        from several threads) while my own chunks are staged; the peers' partials of my
        tokens arrive the same way; the sum issues the shared loads of four 8-row units
        before it adds and stores them. Same FP32 sums in split order as ``exchange``."""
        rows, split = self.rows, self.split
        tid = r
        ntiles = self.n_pad // 128
        per = rows // split
        nch = rows // CHUNK
        pch = per // CHUNK
        ppo = per // PT  # pieces per owner region
        npc = (split - 1) * ppo
        ws_tile = mWs.iterator.toint() + (Int64(split_rank) * ntiles + tile) * (rows * 512)
        cstep = col0[self.chunks - 1] - col0[0] - (self.chunks - 1) * self.n_mma
        # 1. TMEM -> shared [token][row] FP32: the peers' chunks first, then mine.
        for jj in cutlass.range_constexpr(nch):
            col = ((split_rank * pch + pch + jj) % nch) * CHUNK
            taddr = col0[0] + col
            if cutlass.const_expr(self.chunks > 1):
                taddr = taddr + (col // self.n_mma) * cstep
            vals = _tmem_wait16(*_tmem_ld16(taddr))
            for i in cutlass.range_constexpr(CHUNK):
                _sts_f32(stage_addr + ((col + i) * 128 + r) * 4, vals[i])
            if cutlass.const_expr(jj == nch - pch - 1):
                # 2. Publish the peers' token ranges (bulk pieces), mine still in TMEM.
                _fence_proxy_shared()
                _bar(3, 128)
                for p in cutlass.range_constexpr(npc):
                    if tid == p:
                        o = (split_rank + 1 + p // ppo) % split
                        t0 = o * per + (p % ppo) * PT
                        _bulk_store(ws_tile + t0 * 512, stage_addr + t0 * 512, PT * 512)
                        _bulk_commit()
        if tid == 0:
            _st_u64(stamp + 16, _now())
        if tid < npc:
            _bulk_wait()
            _fence_proxy_global()
        _bar(3, 128)
        if tid == 0:
            _red_release_add(flag, 1)
            while _ld_acquire(flag) < split:
                _nanosleep()
            _st_u64(stamp + 24, _now())
            cute.arch.mbarrier_arrive_and_expect_tx(xfer, (split - 1) * per * 512)
        _bar(3, 128)
        # 3. The peers' partials of my tokens, in place over their published regions.
        for p in cutlass.range_constexpr(npc):
            if tid == p:
                s = (split_rank + 1 + p // ppo) % split
                q = p % ppo
                _fence_proxy_global()
                src = mWs.iterator.toint() + (Int64(s) * ntiles + tile) * (rows * 512) + (
                    lo + q * PT) * 512  # fmt: skip
                _bulk_load(stage_addr + s * (per * 512) + q * (PT * 512), src, PT * 512, xfer)
        cute.arch.mbarrier_wait(xfer, 0)
        if tid == 0:
            _st_u64(stamp + 32, _now())
        # 4. Sum in split order, round, store: 4 units' loads before their sums.
        y = mY.iterator.toint()
        mine = stage_addr + lo * 512
        units = per * 16 // 128
        for it0 in cutlass.range_constexpr(0, units, 4):
            nb = min(4, units - it0)
            parts = []
            for b in cutlass.range_constexpr(nb):
                e = (it0 + b) * 128 + tid
                off = (e // 16) * 512 + (e % 16) * 32
                ps = []
                for s in cutlass.range_constexpr(split):
                    a = _sel_eq(split_rank, Int32(s), mine + off, stage_addr + s * (per * 512) + off)
                    ps.append(list(_lds_v4(a)) + list(_lds_v4(a + 16)))
                parts.append(ps)
            for b in cutlass.range_constexpr(nb):
                e = (it0 + b) * 128 + tid
                total = parts[b][0]
                for s in cutlass.range_constexpr(1, split):
                    total = [u + v for u, v in zip(total, parts[b][s])]
                w = [_pack_bf16(total[2 * q], total[2 * q + 1]) for q in range(4)]
                t = lo + e // 16
                _st_v4_b32(y + (Int64(t) * self.n + tile * 128 + (e % 16) * 8) * 2, *w)

    @cute.jit
    def epilogue_partials(
        self, acc: cute.Tensor, acc_full, ew: Int32, lane: Int32, tile: Int32,
        split_rank: Int32, mY: cute.Tensor, stamp: Int64,
    ):  # fmt: skip
        """Thread r (TMEM lane / weight row r of the tile) stores its FP32 partial of
        every token straight from TMEM into rows [split T, (split + 1) T) of y (one
        coalesced 512 B row segment per token); the consumer sums the splits."""
        rows = self.rows
        r = ew * 32 + lane
        if r == 0:
            _st_u64(stamp + 0, _now())
        cute.arch.mbarrier_wait(acc_full, 0)
        if r == 0:
            _st_u64(stamp + 8, _now())
        cute.arch.griddepcontrol_launch_dependents()
        _fence_after()
        lane_bits = (ew * 32) << 16
        col0 = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(self.chunks)
        ]
        base = mY.iterator.toint() + (
            (Int64(split_rank) * rows) * self.n + tile * 128 + r) * 4  # fmt: skip
        for j in cutlass.range_constexpr(rows // CHUNK):
            c, off = divmod(j * CHUNK, self.n_mma)
            vals = _tmem_wait16(*_tmem_ld16(col0[c] + off))
            for i in cutlass.range_constexpr(CHUNK):
                _st_f32(base + (j * CHUNK + i) * self.n * 4, vals[i])
        if r == 0:
            _st_u64(stamp + 40, _now())

    @cute.jit
    def epilogue_direct(
        self, acc: cute.Tensor, acc_full, ew: Int32, lane: Int32, tile: Int32,
        split_rank: Int32, mY: cute.Tensor, mWs: cute.Tensor, stamp: Int64,
    ):  # fmt: skip
        """The split exchange with the data as its own flag: ``ws`` slots hold
        SENTINEL until written. Thread r (TMEM lane / weight row r of the tile) stores
        its FP32 partial of every token straight from TMEM (one coalesced 512 B row
        per token), then, for the tokens its split owns, polls every split's slot
        until it is written, re-arms it, sums in split order and stores. No staging,
        no fence or release/acquire hop: each value is observed on its own."""
        rows, split = self.rows, self.split
        r = ew * 32 + lane
        tid = r
        ntiles = self.n_pad // 128
        lo = split_rank * rows // split
        mine_n = (split_rank + 1) * rows // split - lo
        if tid == 0:
            _st_u64(stamp + 0, _now())
        cute.arch.mbarrier_wait(acc_full, 0)
        if tid == 0:
            _st_u64(stamp + 8, _now())
        cute.arch.griddepcontrol_launch_dependents()
        _fence_after()
        lane_bits = (ew * 32) << 16
        col0 = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(self.chunks)
        ]
        ws = mWs.iterator.toint() + Int64(r) * 4
        mine_tile = ws + (Int64(split_rank) * ntiles + tile) * (rows * 512)
        # 1. My partial of every token (the owners' slots, mine included).
        for j in cutlass.range_constexpr(rows // CHUNK):
            c, off = divmod(j * CHUNK, self.n_mma)
            vals = _tmem_wait16(*_tmem_ld16(col0[c] + off))
            for i in cutlass.range_constexpr(CHUNK):
                _st_relaxed_f32(mine_tile + (j * CHUNK + i) * 512, vals[i])
        if tid == 0:
            _st_u64(stamp + 16, _now())
        # 2. My tokens: every split's slot (all loads in flight, then poll the rest).
        y = mY.iterator.toint()
        base = ws + Int64(tile) * (rows * 512)
        sstride = ntiles * rows * 512
        full_q = rows // split  # tokens every split owns
        slots = [
            base + s * sstride + (lo + q) * 512
            for q in range(full_q)
            for s in range(split)
        ]
        bits = [_ld_relaxed_b32(a) for a in slots]
        for q in cutlass.range_constexpr(full_q):
            total = Float32(0)
            for s in cutlass.range_constexpr(split):
                b = bits[q * split + s]
                while b == SENTINEL:
                    b = _ld_relaxed_b32(slots[q * split + s])
                _st_relaxed(slots[q * split + s], Int32(SENTINEL))
                total = _as_f32(b) if s == 0 else total + _as_f32(b)
            col = Int64(lo + q) * self.n + tile * 128 + r
            if cutlass.const_expr(self.out_f32):
                _st_f32(y + col * 4, total)
            else:
                _st_bf16_f32(y + col * 2, total)
        if cutlass.const_expr(self.uneven):
            if mine_n > full_q:
                tail = [base + s * sstride + (lo + full_q) * 512 for s in range(split)]
                tbits = [_ld_relaxed_b32(a) for a in tail]
                total = Float32(0)
                for s in cutlass.range_constexpr(split):
                    b = tbits[s]
                    while b == SENTINEL:
                        b = _ld_relaxed_b32(tail[s])
                    _st_relaxed(tail[s], Int32(SENTINEL))
                    total = _as_f32(b) if s == 0 else total + _as_f32(b)
                col = Int64(lo + full_q) * self.n + tile * 128 + r
                if cutlass.const_expr(self.out_f32):
                    _st_f32(y + col * 4, total)
                else:
                    _st_bf16_f32(y + col * 2, total)
        if tid == 0:
            _st_u64(stamp + 40, _now())

    @cute.jit
    def store_plain(self, recv, rstride, mine, split_rank, lo, y, tile, e):
        tok = e // 16
        total = self.sum8(recv, rstride, mine, tok * 512 + (e % 16) * 32, split_rank)
        t = lo + tok
        if cutlass.const_expr(self.out_f32):
            dst = y + (Int64(t) * self.n + tile * 128 + (e % 16) * 8) * 4
            _st_v4(dst, *total[:4])
            _st_v4(dst + 16, *total[4:])
        else:
            w = [_pack_bf16(total[2 * q], total[2 * q + 1]) for q in range(4)]
            _st_v4_b32(y + (Int64(t) * self.n + tile * 128 + (e % 16) * 8) * 2, *w)

    @cute.jit
    def store_silu(self, recv, rstride, mine, split_rank, lo, y, tile, e):
        tok = e // 8
        off = tok * 512 + (e % 8) * 32
        gate = self.sum8(recv, rstride, mine, off, split_rank)
        up = self.sum8(recv, rstride, mine, off + 256, split_rank)
        h = []
        for q in cutlass.range_constexpr(8):
            g = _round_bf16(gate[q])
            den = Float32(1) + cute.math.exp(-g, fastmath=False)
            h.append(_round_bf16(div_rn(g, den)) * _round_bf16(up[q]))
        w = [_pack_bf16(h[2 * q], h[2 * q + 1]) for q in range(4)]
        t = lo + tok
        _st_v4_b32(y + (Int64(t) * (self.n // 2) + tile * 64 + (e % 8) * 8) * 2, *w)

    def sum8(self, recv, rstride, mine, off, split_rank):
        """8 consecutive FP32 values at ``off`` of my tokens, summed over the splits
        in split order: split s's partial sits in slot s of ``recv`` (mine in my
        own staging region)."""
        total = []
        for s in range(self.split):
            src = _sel_eq(split_rank, Int32(s), mine + off, recv + s * rstride + off)
            p = list(_lds_v4(src)) + list(_lds_v4(src + 16))
            total = p if s == 0 else [a + b for a, b in zip(total, p)]
        return total


_COMPILED = {}


def die_layout(n, device):
    """(die of each TPC, indexed by %smid // 2; weight tiles of 256 rows homed on die
    0). The two SMs of a TPC share a die; the tiles split the rows evenly (each die
    streams its own half at the die-local rate). Single-die: every tile on die 0."""
    from ..common.dies import sm_domains

    table = sm_domains(torch.device(device).index or 0)
    if len(table) % 2:
        raise ValueError(f"wide GEMM dies: odd smid range {len(table)}")
    tpc_die = []
    for t in range(len(table) // 2):
        a, b = table[2 * t], table[2 * t + 1]
        if a != b or a < 0:
            raise ValueError(f"wide GEMM dies: TPC {t} SMs on dies {a} / {b}")
        tpc_die.append(a)
    tiles = -(-n // 256)
    return tuple(tpc_die), (round(n / 512) if max(tpc_die) else tiles)


def die_place(weight, t0):
    """A copy of ``weight`` [N, K] whose rows [0, 256 t0) are homed on die 0 and the
    rest on die 1 (2 MiB chunks; the chunk across the boundary goes to the die holding
    most of it). Single-die devices keep ``weight``."""
    from ..common.dies import CHUNK, domain_count, localized_like

    if domain_count(weight.device.index or 0) == 1:
        return weight
    n, k = weight.shape
    boundary = min(256 * t0, n) * k * weight.element_size()
    chunks = -(-weight.numel() * weight.element_size() // CHUNK)
    domains = [0 if (i + 0.5) * CHUNK < boundary else 1 for i in range(chunks)]
    return localized_like(weight, domains).view(weight.dtype).view(n, k)


def work_table(tpc_die, t0, tiles, split):
    """Per TPC: ``tile << 8 | split`` of its die's next work, or -1 (idle pair). Die 0
    takes tiles [0, t0), die 1 the rest; raises when a die has more works than TPCs."""
    works = (
        [(t, s) for t in range(t0) for s in range(split)],
        [(t, s) for t in range(t0, tiles) for s in range(split)],
    )
    table, used = [], [0, 0]
    for d in tpc_die:
        i = used[d]
        used[d] += 1
        table.append(works[d][i][0] << 8 | works[d][i][1] if i < len(works[d]) else -1)
    for d in (0, 1):
        if used[d] < len(works[d]):
            raise ValueError(
                f"wide GEMM dies: {len(works[d])} works on die {d}, {used[d]} TPCs"
            )
    return table


def permute_gate_up(weight):
    """[gate; up] rows -> per 128-row tile 64 gate rows then their 64 up rows."""
    n, k = weight.shape
    half = n // 2
    g = weight[:half].view(half // 64, 64, k)
    u = weight[half:].view(half // 64, 64, k)
    return torch.cat((g, u), dim=1).reshape(n, k).contiguous()


class PreparedWide:
    """``WideGemm`` per prepared row count on one weight (``silu``: a gate_up weight
    in stock [gate; up] order, permuted here). ``run(x, out)``: out [T, N] (or
    [T, N / 2] with ``silu``)."""

    def __init__(self, weight, policies, silu=False, out_f32=False, partials=False,
                 dies=False, trace=False, xcopies=1):  # fmt: skip
        self.silu, self.out_f32, self.partials = silu, out_f32 or partials, partials
        self.weight = permute_gate_up(weight) if silu else weight.contiguous()
        self.n, self.k = self.weight.shape
        # ``dies``: die-static work (``WideGemm`` ``tpcs``) on a die-placed copy of
        # the weight (``die_place``; the caller re-points other users at
        # ``self.weight``), or "placed": ``weight`` already comes from ``die_place``.
        self.dies = dies
        self.trace, self.xcopies = trace, xcopies  # benchmarks only (``WideGemm``)
        # rows -> split, or rows -> (split[, activation issuer warps[, weight stages
        # [, weight issuer warps[, direct exchange[, rotate]]]]]) (``WideGemm`` arguments)
        self.policies = {}
        for r, v in dict(policies).items():
            v = (v,) if isinstance(v, int) else tuple(v)
            # 9th field: die-static launch for this row (None: whenever ``dies``);
            # 10th: ``WideGemm`` ``prewait`` (experiments).
            v += ((1,), None, (0, 3), False, False, None, K_TILE, None, None)[len(v) - 1:]
            die = bool(dies) if v[8] is None else bool(v[8]) and bool(dies)
            self.policies[r] = (
                v[0], tuple(v[1]), v[2], tuple(v[3]), bool(v[4]), bool(v[5]), v[6], v[7],
                die, tuple(v[9]) if v[9] else None,
            )  # fmt: skip
        self.kernels, self.state = {}, {}

    def prepare(self):
        dev = self.weight.device
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        tpc_die = t0 = None
        if self.dies:
            tpc_die, t0 = die_layout(self.n, dev)
            if self.dies != "placed":  # "placed": the caller already ran die_place
                self.weight = die_place(self.weight, t0)
        w = from_dlpack(self.weight.unsqueeze(-1), assumed_align=16)
        w = w.mark_layout_dynamic(leading_dim=1)
        for rows, policy in self.policies.items():
            split, act_warps, wst, w_warps, direct, rotate, smem, ktile, die, pw = policy
            gemm = WideGemm(
                rows, self.n, self.k, split, self.silu, act_warps, self.out_f32, wst,
                w_warps, direct, rotate, self.partials, len(tpc_die) if die else 0,
                smem, self.trace, self.xcopies, ktile, pw,
            )  # fmt: skip
            n_pad = gemm.n_pad
            if n_pad // 128 * split > sms:  # the splits spin on each other: one wave
                raise ValueError(f"wide GEMM: {n_pad // 128 * split} CTAs > {sms} SMs")
            x = torch.empty(rows * self.xcopies, self.k, dtype=torch.bfloat16, device=dev)
            cols = self.n // 2 if self.silu else self.n
            ydt = torch.float32 if self.out_f32 else torch.bfloat16
            y = torch.empty(rows * (split if self.partials else 1), cols, dtype=ydt, device=dev)
            # Sentinel-armed slots for the direct exchange (re-armed by their readers).
            ws = torch.full(
                (max(split, 1), n_pad, rows), SENTINEL, dtype=torch.int32, device=dev
            ).view(torch.float32)
            table = work_table(tpc_die, t0, n_pad // 256, split) if die else []
            flags = torch.zeros(gemm.table_at + len(table), dtype=torch.int32, device=dev)
            if die:
                flags[gemm.table_at :] = torch.tensor(table, dtype=torch.int32)
            args = (w, self._x(x), self._y(y), self._ws(ws), self._ws(flags))
            key = (rows, self.n, self.k, split, self.silu, self.out_f32, self.partials)
            key += self.policies[rows] + (gemm.tpcs, self.trace, self.xcopies)
            if key not in _COMPILED:
                stream = _cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
                _COMPILED[key] = cute.compile(gemm, *args, stream)
            self.kernels[rows] = _COMPILED[key]
            self.state[rows] = (ws, flags, self._ws(ws), self._ws(flags))
        self.w = w
        return self

    @staticmethod
    def _x(x):
        return from_dlpack(x.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )

    @staticmethod
    def _y(y):
        return from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    @staticmethod
    def _ws(t):
        return from_dlpack(t, assumed_align=16)

    def run(self, x, out):
        rows = x.shape[0] // self.xcopies
        _, _, ws, flags = self.state[rows]
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        y = out.view(-1, out.shape[-1])  # partials: [split, T, N] -> [split * T, N]
        self.kernels[rows](self.w, self._x(x), self._y(y), ws, flags, stream)
        return out
