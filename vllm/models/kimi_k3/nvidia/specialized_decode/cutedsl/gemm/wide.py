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


@asm()
def _fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


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


def supports(rows, n, k, split, silu=False):
    n_mma, chunks = mma_columns(rows)
    return (
        rows % 32 == 0
        and n_mma % 16 == 0
        and n_mma <= 256
        and chunks > 0
        and n % 256 == 0
        and k % K_TILE == 0
        and 1 <= split <= k // K_TILE
    )


class WideGemm:
    """One (rows, N, K, split, silu) specialization; see the module docstring."""

    def __init__(self, rows, n, k, split, silu=False, ldo=None):
        assert supports(rows, n, k, split, silu), (rows, n, k, split, silu)
        self.rows, self.n, self.k, self.split, self.silu = rows, n, k, split, silu
        # output row pitch (elements): a column slice of a wider output (lm_head parts)
        self.ldo = ldo or (n // 2 if silu else n)
        self.n_mma, self.chunks = mma_columns(rows)
        self.mma_tiler = (256, self.n_mma, K_TILE)
        self.threads = 256  # warps: 0 weights, 1 activations, 2 MMA, 3 idle, 4-7 epilogue
        a_bytes = 128 * K_TILE * 2
        b_bytes = self.chunks * (self.n_mma // 2) * K_TILE * 2
        self.stage_bytes = a_bytes + b_bytes
        smem = utils.get_smem_capacity_in_bytes("sm_100")
        self.stages = min(8, (smem - 16384) // self.stage_bytes)
        self.tmem_cols = 32
        while self.tmem_cols < rows:
            self.tmem_cols *= 2

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
            tiled_mma, self.mma_tiler, dtype, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, dtype, self.stages * self.chunks
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
        self.tx_bytes = (a_bytes + self.chunks * b_bytes) * 2

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cutlass.Int64
            dealloc: cutlass.Int64
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
            grid=(2 * (self.n // 256), self.split, 1),
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
        pair = bidx // 2
        rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_vmnk = cluster_vmnk.get_flat_coord(rank_in_cluster)
        stages, chunks = self.stages, self.chunks

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        acc_full, dealloc = st.acc_full.ptr, st.dealloc.ptr
        tmem_holder = st.tmem_holder.ptr
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(dealloc, 32)
                cute.arch.mbarrier_init(st.xfer.ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        # K range of this split (64-wide tiles, as even as possible).
        k_tiles = self.k // K_TILE
        k0 = split_rank * k_tiles // self.split
        k_count = (split_rank + 1) * k_tiles // self.split - k0

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

        if warp == 0:  # weights: before the PDL wait
            tAgW_tile = tAgW[(None, pair, None, 0)]
            phase = Int32(1)
            for t in cutlass.range(k_count, unroll=1):
                s = t % stages
                cute.arch.mbarrier_wait(empty + s, phase)
                if leader:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_bytes)
                cute.copy(tma_a, tAgW_tile[(None, k0 + t)], tAsA[(None, s)],
                          tma_bar_ptr=full + s, mcast_mask=a_mask)  # fmt: skip
                if s == stages - 1:
                    phase = phase ^ 1
        elif warp == 1:  # activations: after it
            cute.arch.griddepcontrol_wait()
            phase = Int32(1)
            for t in cutlass.range(k_count, unroll=1):
                s = t % stages
                cute.arch.mbarrier_wait(empty + s, phase)
                for c in cutlass.range_constexpr(chunks):
                    cute.copy(tma_b, tBgX[(None, c, k0 + t, 0)],
                              tBsB[(None, s * chunks + c)],
                              tma_bar_ptr=full + s, mcast_mask=b_mask)  # fmt: skip
                if s == stages - 1:
                    phase = phase ^ 1
        elif warp == 2:
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 5)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if leader:
                phase = Int32(0)
                kblocks = cute.size(tCrA, mode=[2])
                for t in cutlass.range(k_count, unroll=1):
                    s = t % stages
                    cute.arch.mbarrier_wait(full + s, phase)
                    for c in cutlass.range_constexpr(chunks):
                        acc_c = acc[(None, None, None, c)]
                        for kb in cutlass.range_constexpr(kblocks):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, (t + kb) != 0)
                            cute.gemm(tiled_mma, acc_c, tCrA[(None, None, kb, s)],
                                      tCrB[(None, None, kb, s * chunks + c)], acc_c)  # fmt: skip
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + s, 3, tcgen05.CtaGroup.TWO)
                    if s == stages - 1:
                        phase = phase ^ 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full, 3, tcgen05.CtaGroup.TWO)
        elif warp >= 4:
            if warp == 4:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder, is_two_cta=True)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 5)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            self.epilogue(
                acc, acc_full, st.xfer.ptr, warp - 4, lane, bidx, split_rank,
                Int32(st.sA.data_ptr().toint()),
                mY, mWs, mFlags,
            )  # fmt: skip
            if warp == 4:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if warp == 4:
                cute.arch.mbarrier_arrive(dealloc, rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(dealloc, 0)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=True)

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, xfer, ew: Int32, lane: Int32, tile: Int32,
        split_rank: Int32, stage_addr: Int32, mY: cute.Tensor, mWs: cute.Tensor,
        mFlags: cute.Tensor,
    ):  # fmt: skip
        """Thread (ew, lane) owns TMEM lane / weight row r = 32 ew + lane of 128-row
        tile ``tile``. The FP32 partial is staged as [token][row] in the drained
        operand ring; split o owns tokens [o T / S, (o + 1) T / S)."""
        rows, n, split = self.rows, self.n, self.split
        r = ew * 32 + lane
        tid = r
        ntiles = n // 128
        region = rows // split  # tokens per owner
        rbytes = region * 512
        stamp = mFlags.iterator.toint() + n // 16 + (Int64(split_rank) * (n // 128) + tile) * 64
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
        # 1. TMEM -> shared [token][row] FP32 (all tokens).
        for j in cutlass.range_constexpr(rows // CHUNK):
            c, off = divmod(j * CHUNK, self.n_mma)
            vals = _tmem_wait16(*_tmem_ld16(col0[c] + off))
            for i in cutlass.range_constexpr(CHUNK):
                _sts_f32(stage_addr + ((j * CHUNK + i) * 128 + r) * 4, vals[i])
        if tid == 0:
            _st_u64(stamp + 16, _now())
        flag = mFlags.iterator.toint() + Int64(tile) * 8
        done = flag + 4
        ws_tile = mWs.iterator.toint() + (Int64(split_rank) * ntiles + tile) * (rows * 512)
        # 2. Publish the other owners' token ranges (bulk async stores), signal.
        if cutlass.const_expr(split > 1):
            _fence_proxy_shared()
            _bar(3, 128)
            if tid == 0:
                for o in cutlass.range_constexpr(split):
                    if o != split_rank:
                        _bulk_store(ws_tile + o * rbytes, stage_addr + o * rbytes, rbytes)
                _bulk_commit()
                _bulk_wait()
                _fence_proxy_global()
                _red_release_add(flag, 1)
                while _ld_acquire(flag) < split:
                    _nanosleep()
                _st_u64(stamp + 24, _now())
                # 3. The peers' partials of my tokens into their (published) regions.
                cute.arch.mbarrier_arrive_and_expect_tx(xfer, (split - 1) * rbytes)
                for s in cutlass.range_constexpr(split):
                    if s != split_rank:
                        src = mWs.iterator.toint() + (Int64(s) * ntiles + tile) * (
                            rows * 512) + split_rank * rbytes  # fmt: skip
                        _bulk_load(stage_addr + s * rbytes, src, rbytes, xfer)
            cute.arch.mbarrier_wait(xfer, 0)
        if tid == 0:
            _st_u64(stamp + 32, _now())
        # 4. Sum in split order, round, store 16 B per thread-iteration.
        y = mY.iterator.toint()
        mine = stage_addr + split_rank * rbytes
        if cutlass.const_expr(not self.silu):
            for it in cutlass.range_constexpr((region * 16 + 127) // 128):
                e = it * 128 + tid  # unit of 8 rows: token e // 16, rows 8 (e % 16)
                if e < region * 16:
                    self.store_plain(stage_addr, rbytes, mine, split_rank, y, tile, region, e)
        else:
            for it in cutlass.range_constexpr((region * 8 + 127) // 128):
                e = it * 128 + tid  # unit: token e // 8, gate rows 8 (e % 8), their up rows
                if e < region * 8:
                    self.store_silu(stage_addr, rbytes, mine, split_rank, y, tile, region, e)
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
    def store_plain(self, stage_addr, rbytes, mine, split_rank, y, tile, region, e):
        tok = e // 16
        total = self.sum8(stage_addr, rbytes, mine, tok * 512 + (e % 16) * 32, split_rank)
        w = [_pack_bf16(total[2 * q], total[2 * q + 1]) for q in range(4)]
        t = split_rank * region + tok
        _st_v4_b32(y + (Int64(t) * self.ldo + tile * 128 + (e % 16) * 8) * 2, *w)

    @cute.jit
    def store_silu(self, stage_addr, rbytes, mine, split_rank, y, tile, region, e):
        tok = e // 8
        off = tok * 512 + (e % 8) * 32
        gate = self.sum8(stage_addr, rbytes, mine, off, split_rank)
        up = self.sum8(stage_addr, rbytes, mine, off + 256, split_rank)
        h = []
        for q in cutlass.range_constexpr(8):
            g = _round_bf16(gate[q])
            den = Float32(1) + cute.math.exp(-g, fastmath=False)
            h.append(_round_bf16(div_rn(g, den)) * _round_bf16(up[q]))
        w = [_pack_bf16(h[2 * q], h[2 * q + 1]) for q in range(4)]
        t = split_rank * region + tok
        _st_v4_b32(y + (Int64(t) * self.ldo + tile * 64 + (e % 8) * 8) * 2, *w)

    def sum8(self, stage_addr, rbytes, mine, off, split_rank):
        """8 consecutive FP32 values at ``off`` of my region, summed over the splits in
        split order: split s's partial sits in region s (mine in my own region)."""
        total = []
        for s in range(self.split):
            src = stage_addr + s * rbytes + off
            p = list(_lds_v4(src)) + list(_lds_v4(src + 16))
            total = p if s == 0 else [a + b for a, b in zip(total, p)]
        return total


_COMPILED = {}


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

    def __init__(self, weight, policies, silu=False, ldo=None):
        self.silu, self.ldo = silu, ldo
        self.weight = permute_gate_up(weight) if silu else weight.contiguous()
        self.n, self.k = self.weight.shape
        self.policies = dict(policies)  # rows -> split
        self.kernels, self.state = {}, {}

    def prepare(self):
        dev = self.weight.device
        w = from_dlpack(self.weight.unsqueeze(-1), assumed_align=16)
        w = w.mark_layout_dynamic(leading_dim=1)
        for rows, split in self.policies.items():
            gemm = WideGemm(rows, self.n, self.k, split, self.silu, self.ldo)
            x = torch.empty(rows, self.k, dtype=torch.bfloat16, device=dev)
            cols = self.n // 2 if self.silu else self.n
            y = torch.empty(rows, self.ldo or cols, dtype=torch.bfloat16, device=dev)[:, :cols]
            ws = torch.empty(max(split, 1), self.n, rows, dtype=torch.float32, device=dev)
            flags = torch.zeros(
                2 * self.n // 128 + 16 * split * self.n // 128, dtype=torch.int32, device=dev
            )
            args = (w, self._x(x), self._y(y), self._ws(ws), self._ws(flags))
            key = (rows, self.n, self.k, split, self.silu, self.ldo)
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
        rows = x.shape[0]
        _, _, ws, flags = self.state[rows]
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self.kernels[rows](self.w, self._x(x), self._y(out), ws, flags, stream)
        return out
