"""Die-local BF16 decode GEMM (VR200, two locality domains): cluster split-K.

``y = x @ W^T`` for up to ``cta_n`` (<= 256) BF16 rows ``x`` [rows, K], a BF16 weight
``W`` [N, K] (K-major) placed on the dies by halves (128-row tiles below ``split`` on
die 0, the rest on die 1), FP32 accumulation, ``y`` [rows, N] row-major (BF16, BF16
``y += x W^T`` rounded once, or FP32):

* one cluster of ``S`` CTAs per 128-row tile, all on one die (clusters never span
  dies); its leader takes a rank among the clusters of its SM's die from a ticket word
  and the cluster works on that die's tile of that rank (a die's tiles taken, one of
  the other die's), so every weight byte streams from the memory beside its SMs;
* CTA ``s`` of the cluster owns K blocks [s KB / S, (s + 1) KB / S) of the tile:
  its weights are loaded from the kernel's start, before the grid dependency wait
  (only the activations' load waits);
* after the MMAs, a cluster barrier frees every CTA's operand ring, each epilogue
  thread (one tensor-memory lane = one weight row) pushes its FP32 row to the shared
  memory of the row's owner (CTA ``row // ceil(128 / S)``), and after a second
  barrier the owners add the S partials in CTA (K) order, round once and store:
  the result does not depend on timing.

The ticket word is the call's own (the last cluster resets it), so calls on one
weight must not overlap."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from ...common import dies
from .placement import halves_domains, tile_split
from .primitives import (
    CTA_K,
    CTA_M,
    EPI_N,
    EPILOGUE_WARP,
    EVICT_FIRST,
    OUT_BF16_ADD,
    OUT_F32,
    _asm,
    _load_s32,
    _load_tmem16,
    _smid,
    atom_add_u64,
    store_u64,
)


def tmem_columns1(cta_n: int) -> int:
    need = max(32, cta_n)
    return 1 << (need - 1).bit_length()


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release; barrier.cluster.wait.acquire;",
        "~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def _mapa(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    operands = [
        Int32(addr).ir_value(loc=loc, ip=ip),
        Int32(rank).ir_value(loc=loc, ip=ip),
    ]
    return Int32(
        _asm(
            T.i32(), operands, "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", loc, ip
        )
    )


@dsl_user_op
def _st_cluster_v4(addr: Int32, a, b, c, d, *, loc=None, ip=None) -> None:
    operands = [Int32(addr).ir_value(loc=loc, ip=ip)] + [
        Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)
    ]
    _asm(
        None,
        operands,
        "st.shared::cluster.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def _ld_cluster_s32(addr: Int32, *, loc=None, ip=None) -> Int32:
    operands = [Int32(addr).ir_value(loc=loc, ip=ip)]
    return Int32(
        _asm(
            T.i32(),
            operands,
            "ld.shared::cluster.s32 $0, [$1];",
            "=r,r,~{memory}",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_shared_s32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    operands = [
        Int32(addr).ir_value(loc=loc, ip=ip),
        Int32(value).ir_value(loc=loc, ip=ip),
    ]
    _asm(None, operands, "st.shared.s32 [$0], $1;", "r,r,~{memory}", loc, ip)


@dsl_user_op
def _ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    operands = [Int32(addr).ir_value(loc=loc, ip=ip)]
    return Float32(
        _asm(T.f32(), operands, "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", loc, ip)
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    operands = [
        Float32(hi).ir_value(loc=loc, ip=ip),
        Float32(lo).ir_value(loc=loc, ip=ip),
    ]
    return Int32(
        _asm(T.i32(), operands, "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", loc, ip)
    )


@dsl_user_op
def _st_global_v4_b32(addr: Int64, a, b, c, d, *, loc=None, ip=None) -> None:
    operands = [Int64(addr).ir_value(loc=loc, ip=ip)] + [
        Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)
    ]
    _asm(
        None,
        operands,
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def _st_global_v4_f32(addr: Int64, a, b, c, d, *, loc=None, ip=None) -> None:
    operands = [Int64(addr).ir_value(loc=loc, ip=ip)] + [
        Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)
    ]
    _asm(
        None,
        operands,
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def _ld_global_bf16x8(addr: Int64, *, loc=None, ip=None):
    """Eight BF16 at ``addr`` (16 B) as FP32."""
    ty = llvm.StructType.get_literal([T.f32()] * 8)
    text = (
        "{ .reg .b32 w0, w1, w2, w3; ld.global.v4.b32 {w0, w1, w2, w3}, [$8]; "
        "shl.b32 $0, w0, 16; and.b32 $1, w0, 0xFFFF0000; shl.b32 $2, w1, 16; "
        "and.b32 $3, w1, 0xFFFF0000; shl.b32 $4, w2, 16; and.b32 $5, w2, 0xFFFF0000; "
        "shl.b32 $6, w3, 16; and.b32 $7, w3, 0xFFFF0000; }"
    )
    res = _asm(
        ty,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        text,
        ",".join(["=f"] * 8 + ["l"]),
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(8)
    ]


class DieLocalClusterGemm:
    """The cluster split-K GEMM of the module docstring: ``n`` weight rows, ``k``
    columns (a multiple of ``CTA_K``), ``cta_n`` token columns (a multiple of
    ``EPI_N``), clusters of ``split_k`` CTAs, tiles below ``split`` on die 0."""

    def __init__(self, n, k, cta_n, split_k, num_ab_stage, split, out_mode, m_sub=1):
        assert cta_n % EPI_N == 0 and EPI_N <= cta_n <= 256, cta_n
        assert k % CTA_K == 0 and 1 <= split_k <= 8 and m_sub in (1, 2)
        assert split_k >= m_sub
        assert n % 8 == 0  # 16-byte row groups of C
        self.m, self.k, self.cta_n, self.s, self.m_sub = n, k, cta_n, split_k, m_sub
        self.tile_rows = CTA_M * m_sub
        self.m_tiles = -(-n // self.tile_rows)
        self.kb = k // CTA_K
        assert self.kb >= split_k
        self.split = split
        # rows per owner CTA, a multiple of 8 (the store groups); the last owners may
        # hold fewer rows or none
        per_owner = -(-self.tile_rows // split_k)
        self.owner_rows = -(-per_owner // 8) * 8
        self.num_ab_stage = num_ab_stage
        self.out_mode = out_mode
        self.tmem_cols = tmem_columns1(m_sub * cta_n)
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        stage_bytes = (self.tile_rows + cta_n) * CTA_K * 2
        mailbox = split_k * self.owner_rows * cta_n * 4
        assert mailbox <= num_ab_stage * stage_bytes, (
            mailbox,
            num_ab_stage,
            stage_bytes,
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_addr: Int64,
        tickets: Int64,
        domains: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            (CTA_M, self.cta_n),
        )
        tile_mnk = (CTA_M, self.cta_n, CTA_K)
        a_sl = sm100_utils.make_smem_layout_a(
            tiled_mma, tile_mnk, cutlass.BFloat16, self.num_ab_stage * self.m_sub
        )
        b_sl = sm100_utils.make_smem_layout_b(
            tiled_mma, tile_mnk, cutlass.BFloat16, self.num_ab_stage
        )
        cl = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        a_s1 = cute.slice_(a_sl, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
            a,
            a_s1,
            tile_mnk,
            tiled_mma,
            cl.shape,
        )
        b_s1 = cute.slice_(b_sl, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id),
            b,
            b_s1,
            tile_mnk,
            tiled_mma,
            cl.shape,
        )
        self.tma_bytes_a = cute.size_in_bytes(cutlass.BFloat16, a_s1)
        self.tma_bytes_b = cute.size_in_bytes(cutlass.BFloat16, b_s1)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            (tickets, domains, rows, c_addr),
            a_sl,
            b_sl,
        ).launch(
            grid=(self.m_tiles * self.s, 1, 1),
            block=[256, 1, 1],
            cluster=(self.s, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        local: tuple,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        tickets, domains, rows, c_addr = local
        me = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        AB_Stage = self.num_ab_stage

        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, AB_Stage * 2]
            acc_bar: cutlass.Int64
            tmem_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            tile_word: cutlass.Int32

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=AB_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage
        acc_bar = storage.acc_bar.ptr
        tmem_bar = storage.tmem_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        tile_word_addr = storage.tile_word.ptr.toint()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(acc_bar, 1)
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)
        sA = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        mailbox = sA.iterator.toint()

        # The cluster's tile: the leader's ticket, read by every CTA after a barrier.
        if warp_idx == 3 and me == 0:  # noqa: SIM102 (traced: nested, not `and`)
            if cute.arch.lane_idx() == 0:
                _st_shared_s32(tile_word_addr, self.take_tile(tickets, domains))
        _cluster_sync()
        tile = _ld_cluster_s32(_mapa(tile_word_addr, Int32(0)))
        kb0 = (me * self.kb) // self.s
        kb1 = ((me + 1) * self.kb) // self.s

        thr_mma = tiled_mma.get_slice(Int32(0))
        if warp_idx == 0:
            gA = cute.local_tile(mA_mkl, (CTA_M, CTA_K), (None, None, None))
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            policy = Int64(EVICT_FIRST)
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            for k_tile in cutlass.range(kb0, kb1, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        ab_full_bar + state.index, self.tma_bytes_a * self.m_sub
                    )
                for sub_m in cutlass.range_constexpr(self.m_sub):
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, tile * self.m_sub + sub_m, k_tile, 0)],
                        tAsA[(None, state.index * self.m_sub + sub_m)],
                        tma_bar_ptr=ab_full_bar + state.index,
                        cache_policy=policy,
                    )
                state.advance()
            cute.arch.griddepcontrol_launch_dependents()
        elif warp_idx == 1:
            cute.arch.griddepcontrol_wait()
            gB = cute.local_tile(mB_nkl, (self.cta_n, CTA_K), (None, None, None))
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            tBgB_t = tBgB[(None, 0, None, 0)]
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            for k_tile in cutlass.range(kb0, kb1, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        ab_full_bar + state.index, self.tma_bytes_b
                    )
                cute.copy(
                    tma_atom_b,
                    tBgB_t[(None, k_tile)],
                    tBsB[(None, state.index)],
                    tma_bar_ptr=ab_full_bar + state.index,
                )
                state.advance()
        elif warp_idx == 2:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            cute.arch.alloc_tmem(self.tmem_cols, tmem_base_smem_ptr)
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, 16, tmem_base_smem_ptr
            )
            acc_shape = tiled_mma.partition_shape_C((CTA_M, self.cta_n))
            tCtAcc_all = cute.make_tensor(
                tmem_ptr,
                tiled_mma.make_fragment_C(cute.append(acc_shape, self.m_sub)).layout,
            )
            num_kblocks = cute.size(tCrA, mode=[2])
            ab_state = make_pipeline_state(PipelineUserType.Consumer, AB_Stage)
            for k_tile in cutlass.range(kb0, kb1, unroll=1):
                cute.arch.mbarrier_wait(ab_full_bar + ab_state.index, ab_state.phase)
                for sub_m in cutlass.range_constexpr(self.m_sub):
                    tCtAcc = tCtAcc_all[(None, None, None, sub_m)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != kb0)
                    for kblk in cutlass.range_constexpr(num_kblocks):
                        operand_a = (
                            None,
                            None,
                            kblk,
                            ab_state.index * self.m_sub + sub_m,
                        )
                        operand_b = (None, None, kblk, ab_state.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[operand_a], tCrB[operand_b], tCtAcc
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + ab_state.index, None, self.cta_group)
                ab_state.advance()
            with cute.arch.elect_one():
                tcgen05.commit(acc_bar, None, self.cta_group)
            cute.arch.mbarrier_wait(tmem_bar, 0)
            cute.arch.mbarrier_arrive(tmem_bar)

        epi_tid = tidx - 32 * EPILOGUE_WARP
        R = self.owner_rows
        cn = self.cta_n
        if warp_idx >= EPILOGUE_WARP:
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 0)
            cute.arch.mbarrier_wait(acc_bar, 0)
        # every CTA's MMAs are done: its operand ring is free for the mailboxes
        _cluster_sync()
        if warp_idx >= EPILOGUE_WARP:
            base = Int32(cute.make_tensor(tmem_base_smem_ptr, cute.make_layout(1))[0])
            lane_addr = base + ((epi_tid // 32) * 32 << 16)
            for sub_m in cutlass.range_constexpr(self.m_sub):
                g = sub_m * CTA_M + epi_tid
                owner = g // R
                slot = mailbox + ((me * R + (g - owner * R)) * cn) * 4
                for c in cutlass.range_constexpr(cn // EPI_N):
                    v = _load_tmem16(lane_addr + sub_m * cn + c * EPI_N)
                    for q in cutlass.range_constexpr(EPI_N // 4):
                        _st_cluster_v4(
                            _mapa(slot + (c * EPI_N + 4 * q) * 4, owner),
                            v[4 * q],
                            v[4 * q + 1],
                            v[4 * q + 2],
                            v[4 * q + 3],
                        )
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(tmem_bar)  # TMEM read: the MMA warp may free it
        _cluster_sync()
        if warp_idx == 2:
            cute.arch.mbarrier_wait(tmem_bar, 1)
            tmem_ptr2 = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, 16, tmem_base_smem_ptr
            )
            cute.arch.dealloc_tmem(tmem_ptr2, self.tmem_cols)
        if warp_idx >= EPILOGUE_WARP:
            # Every epilogue thread: (token, 8-row group) items of this CTA's rows;
            # the S partials added in CTA (K) order, 16-byte vector stores of C.
            first = me * R
            owned = self.tile_rows - first
            if owned > R:
                owned = R
            groups = owned // 8
            if groups > 0:
                m0 = tile * self.tile_rows + first
                items = rows * groups
                for i in cutlass.range(epi_tid, items, 128, unroll=1):
                    t = i // groups
                    g = i - t * groups
                    m = m0 + 8 * g
                    if m < self.m:
                        acc = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
                        acc.fill(Float32(0.0))
                        for s in cutlass.range_constexpr(self.s):
                            src = mailbox + ((s * R + 8 * g) * cn + t) * 4
                            for r in cutlass.range_constexpr(8):
                                acc[r] = acc[r] + _ld_shared_f32(src + r * cn * 4)
                        self.store8([acc[r] for r in range(8)], c_addr, m, t)

    @cute.jit
    def take_tile(self, tickets: Int64, domains: Int64) -> Int32:
        """Leader lane: the cluster's tile from the ticket word (a rank among the
        clusters of its SM's die; past the die's tile count, one of the other die's
        from the top); the last cluster resets the word."""
        clusters = self.m_tiles
        home = _load_s32(domains + Int64(_smid()) * 4)
        add = Int64(1) + (Int64(1) << (Int64(32) + Int64(16) * Int64(home)))
        old = atom_add_u64(tickets, add)
        if Int32(old & Int64(0xFFFFFFFF)) == clusters - 1:
            store_u64(tickets, Int64(0))
        ticket = Int32((old >> (Int64(32) + Int64(16) * Int64(home))) & Int64(0xFFFF))
        cap0 = self.split
        cap1 = self.m_tiles - self.split
        home_cap = cap0 + home * (cap1 - cap0)
        other_cap = cap1 + home * (cap0 - cap1)
        tile = home * cap0 + ticket
        if ticket >= home_cap:
            tile = (1 - home) * cap0 + other_cap - 1 - (ticket - home_cap)
        return tile

    @cute.jit
    def store8(self, v, c_addr: Int64, m: Int32, t: Int32):
        """C[t, m .. m + 8) = v (rounded once to BF16; plus the old values in FP32 for
        ``OUT_BF16_ADD``) or FP32."""
        if cutlass.const_expr(self.out_mode == OUT_F32):
            addr = c_addr + (Int64(t) * self.m + Int64(m)) * 4
            _st_global_v4_f32(addr, v[0], v[1], v[2], v[3])
            _st_global_v4_f32(addr + 16, v[4], v[5], v[6], v[7])
        else:
            addr = c_addr + (Int64(t) * self.m + Int64(m)) * 2
            if cutlass.const_expr(self.out_mode == OUT_BF16_ADD):
                old = _ld_global_bf16x8(addr)
                v = [v[r] + old[r] for r in range(8)]
            _st_global_v4_b32(
                addr,
                _pack_bf16x2(v[0], v[1]),
                _pack_bf16x2(v[2], v[3]),
                _pack_bf16x2(v[4], v[5]),
                _pack_bf16x2(v[6], v[7]),
            )


# ---- host side ----

_MMA_SMEM = 1 << 18
_OTHER = 1024


def _ptr(dtype, address, align):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


# Clusters of N one-CTA-per-SM CTAs (200 KB shared memory) a 212-SM VR200 runs at once,
# half on each die (cluster_probe.py, 11-23).
MAX_CLUSTERS = {1: 212, 2: 106, 3: 60, 4: 46, 5: 32, 6: 28, 7: 24, 8: 22}


def cluster_split(
    tiles_per_die: int,
    sms_per_die: int,
    kb: int,
    least: int = 1,
    tiles: int | None = None,
) -> int:
    """The largest cluster size (<= 8, <= K blocks, >= least) whose clusters of the
    busier die fit its SMs once and whose clusters all co-schedule."""
    tiles = 2 * tiles_per_die if tiles is None else tiles
    best = least
    for s in range(least, 9):
        if tiles_per_die * s <= sms_per_die and s <= kb and tiles <= MAX_CLUSTERS[s]:
            best = s
    return best


def cluster_stages(cta_n, smem_optin, m_sub=1):
    smem = min(smem_optin, _MMA_SMEM)
    return (smem - _OTHER) // ((CTA_M * m_sub + cta_n) * CTA_K * 2)


@functools.cache
def compile_cluster(n, k, cn, split_k, stages, split, out_mode, m_sub=1):
    """Code only (shared by every layer of one shape); pointers are call arguments."""
    g = DieLocalClusterGemm(n, k, cn, split_k, stages, split, out_mode, m_sub)
    return cute.compile(
        g,
        _ptr(cutlass.BFloat16, 16, 16),
        _ptr(cutlass.BFloat16, 16, 16),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(1),
        make_fake_stream(),
    )


class DieLocalClusterLinear:
    def __init__(
        self,
        weight,
        cta_ns,
        out_mode,
        relocate=True,
        split_k=None,
        domains_table=None,
        m_sub=1,
    ):
        n, k = weight.shape
        assert (
            weight.dtype == torch.bfloat16 and weight.is_contiguous() and k % CTA_K == 0
        )
        dev = weight.device
        self.n, self.k = n, k
        self.weight = (
            dies.localized_like(weight, halves_domains(n, k)) if relocate else weight
        )
        self.m_sub = m_sub
        tiles = -(-n // (CTA_M * m_sub))
        split = tile_split(n, k, CTA_M * m_sub)
        table = (
            dies.sm_domain_table(dev.index) if domains_table is None else domains_table
        )
        self.domains = table.to(torch.int32).contiguous()
        doms = [d for d in self.domains.tolist() if d >= 0]
        sms_die = min(doms.count(0), doms.count(1))
        self.split_k = split_k or cluster_split(
            max(split, tiles - split), sms_die, k // CTA_K, m_sub, tiles
        )
        self.tickets = torch.zeros(1, dtype=torch.int64, device=dev)
        props = torch.cuda.get_device_properties(dev)
        self.kernels = {}
        for cn in sorted(set(cta_ns)):
            st = cluster_stages(cn, props.shared_memory_per_block_optin, m_sub)
            self.kernels[cn] = compile_cluster(
                n, k, cn, self.split_k, st, split, out_mode, m_sub
            )

    def __call__(self, x, out):
        rows = x.shape[0]
        cn = next(c for c in sorted(self.kernels) if rows <= c)
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self.kernels[cn](
            _ptr(cutlass.BFloat16, self.weight.data_ptr(), 16),
            _ptr(cutlass.BFloat16, x.data_ptr(), 16),
            Int64(out.data_ptr()),
            Int64(self.tickets.data_ptr()),
            Int64(self.domains.data_ptr()),
            Int32(rows),
            stream,
        )
        return out
