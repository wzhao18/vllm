# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA-R24: the KDA input projection and the deferred-verify recurrence of B4 x q8
(``requests`` = 4, 32 token rows) in one kernel on six 24-CTA clusters (Rubin TP16,
one cluster per head, 144 SMs).

Projection: CTA ``t + 4 s`` computes section t (Q, K, V, gate; 128 rows) of its head
over K slice s of six, with ``SplitKTmemGemm64``'s slices (112 K tiles of 64, rank
s < 4 holds 19, the others 18). Before the PDL wait the weight warp streams the slice
through the ring and the MMA warp stages its first 15 tiles in tensor memory; the
last 3-4 stay in their ring stages, so the whole slice is on chip when x lands
(``KdaHeadMulti``'s staged form with an N32 accumulator). Hop 1: every CTA pushes
its FP32 partial of each token row to the row's owner (CTA row % 24, owned slot
row // 24), which sums the six slices in rank order, ((p0 + p1) + ...) + p5, and
rounds once to BF16: bitwise ``SplitKTmemGemm64``'s split-6 projection.

Recurrence: request r runs on CTAs 6 r .. 6 r + 5 (one request per six CTAs); CTA
6 r + b owns state rows [base_b, base_b + rows_b) with bases 0 / 24 / 48 / 72 / 96 /
112 (24, 24, 24, 24, 16, 16 rows: the V hop's 8-row chunks never straddle two
CTAs), 8 lanes per row (``KdaHeadMulti``'s B1 verify schedule) on six loop warps.
Hop 2 sends an owned token's Q / K channels to the request's six CTAs and its V rows
to the CTA owning them. KDA CTA b normalizes token b, CTAs 4 / 5 also tokens 6 / 7.
The gate warp projects F_B for the CTA's rows' features. The outputs go to the token
row's owner, which applies the gated RMS norm. Deferred-verify state protocol and
formulas as ``KdaHeadMultiDefer`` (fold before the wait, slot 0 after token 0, the
corrections / k / decay records), so given the same projected inputs every output,
state word and conv window is bit for bit ``KdaHeadMultiDefer``'s / the stock
pair's."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute import experimental as cute_ext
from cutlass.cute.arch import store_async_dsmem

from ..attn_res.kernel import BESIDE_SMEM
from ..common.primitives import (
    at,
    copy_g2s,
    fma,
    float_bits,
    load_f32x4,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    wait,
    warp_sum,
    zeros,
)
from ..gemm.splitk import SplitKGemm, SplitKTactic, _smem
from .head import (
    B_ISSUERS,
    SECTIONS,
    _acquire_load,
    _fetch_add,
    _store_zero,
    load_bf16x4,
)
from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from .head_defer import HeadRecords
from .head_multi import TOKEN_INPUTS, KdaHeadMulti, _align
from .kernel import sigmoids
from .verify_defer import DECAY_ROW, V_ROW, put_tag
from .window import conv_put, conv_tap

R24_CLUSTER = 24
SPLITS = R24_CLUSTER // SECTIONS  # 6 K slices
KC = 6  # KDA CTAs per request
RMAX = 24  # state rows of the widest KDA CTA
MAX_OWNED = 2  # token rows a CTA owns (32 rows over 24 CTAs)
LOOP_THREADS = 192  # six loop warps: 24 rows x 8 lanes


@cute.jit
def _min(a, b):
    return cutlass.select_(a < b, a, b)


class KdaHeadR24(HeadRecords, KdaHeadMulti):
    """``requests`` x 8 verify rows (requests = 4) on 24-CTA clusters; extra operand
    ``accepted`` [requests]; ``slots`` [requests, >= 8] (row 0: state, row 1: records,
    the deferred protocol). Warps 0-2: weights / x + re-arm / MMA; warp 3: gate; warps
    4-9: the projection epilogue (4-7), hop 1 / 2 and the recurrence (4-9)."""

    def __init__(
        self, heads, *, requests=4, tokens=8, stages=5, cta_k=64, tmem_tiles=15,
        early_loads=19, bf16_state=False, has_bias=False, hidden=7168,
    ):  # fmt: skip
        # KdaHeadVerify's set-up (its vkda: the B1 verify schedule, lanes 8)
        super(KdaHeadMulti, self).__init__(
            heads, tokens=8, cta_k=cta_k, tmem_tiles=tmem_tiles,
            early_loads=early_loads, bf16_state=bf16_state, has_bias=has_bias,
            hidden=hidden,
        )  # fmt: skip
        assert tokens == 8 and requests * KC == R24_CLUSTER, "B4 x q8"
        assert not bf16_state, "the FP32 state's deferred-verify records"
        assert self.vkda.lanes_per_row == 8 and self.vkda.passes == 1
        self.tokens, self.requests = tokens, requests
        n = requests * tokens
        self.rows_total = n
        assert n <= MAX_OWNED * R24_CLUSTER
        SplitKGemm.__init__(self, tactic=SplitKTactic(128, n, SECTIONS, stages))
        self.cluster_shape = (R24_CLUSTER, 1, 1)
        self.threads_per_cta = 128 + LOOP_THREADS
        self.cta_k = cta_k
        self.k_total = hidden // cta_k
        self.k_tiles = -(-self.k_total // SPLITS)  # the longest slice (19)
        self.extra = self.k_total % SPLITS  # slices below it hold k_tiles
        self.tmem_tiles = tmem_tiles
        assert n + tmem_tiles * cta_k // 2 <= 512 and tmem_tiles > 0
        assert self.k_tiles - 1 >= tmem_tiles
        assert stages > early_loads - tmem_tiles, "pre-wait A tiles beyond TMEM"
        T = tokens
        # shared memory in allocation order (the FP32 partial reuses the A ring)
        parts = (
            (stages * 128 * cta_k * 2, 1024),  # A ring
            (self.k_tiles * n * cta_k * 2, 1024),  # resident x
            (stages * 8, 8),
            (stages * 8, 8),
            (self.k_tiles * 8, 8),
            (8, 8),
            (8, 8),
            (4, 4),
            (MAX_OWNED * SPLITS * TOKEN_INPUTS * 4, 128),  # hop 1 mailboxes
            (2 * T * 128 * 2, 128),  # hop 2: raw Q / K
            (T * RMAX * 2, 16),  # raw V rows
            (MAX_OWNED * TOKEN_INPUTS * 2, 16),  # owned rows' rounded inputs
            (3 * T * 128 * 4, 16),  # q, k, decays
            (T * RMAX * 4, 16),  # v
            (MAX_OWNED * 128 * 4, 16),  # owned rows' outputs
            (T * RMAX * 4, 16),  # local outputs
            (T * 4, 16),  # slots
            (16 * 4, 16),  # norms (two tokens)
            (8 * 4, 16),  # betas
            ((5 + MAX_OWNED) * 8, 8),  # in / hop / gate / qk / outputs barriers
        )
        used = 0
        for size, alignment in parts:
            used = _align(used, alignment) + size
        assert 128 * n * 4 <= stages * 128 * cta_k * 2  # partial inside the ring
        self.smem_used = used
        self.smem_bytes = (used + 2560 + 1023) // 1024 * 1024
        assert self.smem_bytes <= BESIDE_SMEM, (self.smem_bytes, BESIDE_SMEM)

    # ----------------------------------------------------------------- geometry
    @cute.jit
    def block_rows(self, block):
        """(first state row, rows) of KDA CTA ``block`` of a request."""
        base = Int32(24) * block
        rows = Int32(24)
        if block >= 4:
            base = Int32(96) + Int32(16) * (block - 4)
            rows = Int32(16)
        return base, rows

    @cute.jit
    def row_block(self, row):
        """The KDA CTA (of a request) owning state row ``row``."""
        block = row // 24
        if row >= 96:
            block = Int32(4) + (row - 96) // 16
        return block

    @cute.jit
    def slice_tiles(self, split_rank):
        """``SplitKTmemGemm64``'s K slice of ``split_rank``: (first tile, tiles)."""
        base = self.k_total // SPLITS
        tiles = base + Int32(split_rank < self.extra)
        over = Int32(split_rank > self.extra) * (split_rank - self.extra)
        first = split_rank * (base + 1) - over
        return first, tiles

    # ------------------------------------------------------------------- launch
    @cute.experimental.jit
    def __call__(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64, stream,
    ):  # fmt: skip
        # A 24-CTA cluster is non-portable: the DSL verifies a static cluster shape
        # against the portable 16, so the shape is a runtime value (launch-time
        # cluster dims; the kernel sets non_portable_cluster_size_allowed).
        cluster_x = target - target + R24_CLUSTER
        self.kernel(
            mA, mB, a_map, conv_w, bias, conv_state, fa_beta, a_log, dt_bias, slots,
            state, output, norm_w, f_b, accepted, lower_bound, scale, eps, counter,
            target, out_rearm,
        ).launch(
            grid=(R24_CLUSTER, self.heads, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(cluster_x, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64,
    ):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        N: cutlass.Constexpr = self.rows_total
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        rank, head, _ = cute.arch.block_idx()
        section, split_rank = rank % SECTIONS, rank // SECTIONS
        row0 = section * self.dim + head * 128
        request, block = rank // KC, rank % KC  # this CTA's KDA role
        base, nrows = self.block_rows(block)
        owned = (N - rank + R24_CLUSTER - 1) // R24_CLUSTER  # token rows owned
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        a_shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.k_tiles)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*a_shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*b_shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, self.k_tiles, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        # the FP32 partial [128 rows, N tokens] in the A ring (written after this
        # CTA's last MMA completed: bar_mma_epilog)
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        partial_ptr = cute.make_ptr(
            Float32, sA.iterator.toint(), cute.AddressSpace.smem, assumed_align=1024
        )
        partial = cute.make_tensor(partial_ptr, tile_layout)
        x_in = _smem(Float32, MAX_OWNED * SPLITS * TOKEN_INPUTS, 128)
        raw_qk = _smem(BFloat16, 2 * T * 128, 128)
        raw_v = _smem(BFloat16, T * RMAX, 16)
        mine = _smem(BFloat16, MAX_OWNED * TOKEN_INPUTS, 16)
        shared_q = _smem(Float32, T * 128, 16)
        shared_k = _smem(Float32, T * 128, 16)
        shared_decay = _smem(Float32, T * 128, 16)
        shared_v = _smem(Float32, T * RMAX, 16)
        shared_o = _smem(Float32, MAX_OWNED * 128, 16)
        local_o = _smem(Float32, T * RMAX, 16)
        slot_rows = _smem(Int32, T, 16)
        reductions = _smem(Float32, 16, 16)
        shared_beta = _smem(Float32, 8, 16)
        bar_in = _smem(cutlass.Int64, 1, 8).iterator
        bar_hop = _smem(cutlass.Int64, 1, 8).iterator
        bar_gate = _smem(cutlass.Int64, 1, 8).iterator
        bar_qk = _smem(cutlass.Int64, 1, 8).iterator
        bar_spare = _smem(cutlass.Int64, 1, 8).iterator  # noqa: F841 (layout)
        bar_out = _smem(cutlass.Int64, MAX_OWNED, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(self.k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        if tid == 128:
            cute.arch.mbarrier_init(bar_in, 1)  # owned rows x 6 slices x 512
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar_in, owned * (SPLITS * TOKEN_INPUTS * 4)
            )
            cute.arch.mbarrier_init(bar_hop, 1)  # Q / K of T tokens; V: T x rows
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar_hop, 2 * T * 128 * 2 + T * nrows * 2
            )
            cute.arch.mbarrier_init(bar_gate, 1)  # 128 decays x T + T betas
            cute.arch.mbarrier_arrive_and_expect_tx(bar_gate, 128 * T * 4 + T * 4)
            cute.arch.mbarrier_init(bar_qk, 1)  # normalized q / k of T tokens
            cute.arch.mbarrier_arrive_and_expect_tx(bar_qk, T * 2 * 128 * 4)
            for s in cutlass.range_constexpr(MAX_OWNED):
                cute.arch.mbarrier_init(bar_out + s, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_out + s, 512)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic

        first, tiles = self.slice_tiles(split_rank)
        offset = accepted[request] - 1
        slot = slots[request, offset]
        if warp_idx < 3:
            cute.arch.cluster_wait()
            if warp_idx == 0:
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, first, tiles)
            elif warp_idx == 1:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.resident_x(b_full, b_tile, sB, b_map, first, tiles, 0)
            else:
                self.staged_mma_r24(
                    bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc, sA, sB,
                    tmem_base_ptr, tiles,
                )  # fmt: skip
            cute.arch.cluster_arrive_relaxed()
        elif warp_idx == 3:
            gate_args = (fa_beta, a_log, dt_bias, f_b, lower_bound, counter, target)
            coords = (head, rank, slot, request, block, base, nrows)
            self.r24_gate(gate_args, coords, shared_decay, shared_beta, bar_gate)
            cute.arch.cluster_arrive_relaxed()
        else:
            kt = tid - 128
            tensors = (conv_w, bias, conv_state, state, output, norm_w)
            coords = (head, rank, slot, offset, request, block, base, nrows)
            early = self.r24_early(kt, tensors, coords, slots, (shared_q, shared_k))
            cute.arch.cluster_wait()
            if warp_idx < 8:  # the other B_ISSUERS - 1 warps of x's tiles
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.resident_x(b_full, b_tile, sB, b_map, first, tiles, warp_idx - 3)
                self.push_rows(
                    kt, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout,
                    partial, section, split_rank, x_in, bar_in,
                )  # fmt: skip
            wait(bar_in)
            if kt < 128:
                for s in cutlass.range_constexpr(MAX_OWNED):
                    if s < owned:
                        self.round_owned_r24(kt, x_in, mine, s)
            named_barrier(1, LOOP_THREADS)
            if kt < 128:
                for s in cutlass.range_constexpr(MAX_OWNED):
                    if s < owned:
                        r = rank + R24_CLUSTER * s
                        self.hop_r24(kt, r // T, r % T, mine, s, raw_qk, raw_v, bar_hop)
            cute.arch.cluster_arrive_relaxed()  # hop 1 done; hop 2 issued
            rows = (
                shared_q, shared_k, shared_decay, shared_v, shared_o, reductions,
                local_o, slot_rows,
            )  # fmt: skip
            raws = (raw_qk, raw_v, mine, bar_hop)
            bars = (bar_gate, bar_out, shared_beta, bar_qk)
            self.r24_main(
                kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
            )
            if kt < 128:
                for s in cutlass.range_constexpr(MAX_OWNED):
                    if s < owned:
                        r = rank + R24_CLUSTER * s
                        owned_args = (mine, shared_o, bar_out, output, norm_w, accepted)
                        self.norm_owned(kt, owned_args, slots, head, r, s, eps)
        cute.arch.cluster_wait()  # no CTA leaves while a peer may write it

    # --------------------------------------------------------------- projection
    @cute.experimental.jit
    def resident_x(
        self, b_full, g_tile: cute.Tensor, s_tile: cute.Tensor, cta_v_map: cute.Layout,
        k_first: cutlass.Int32, tiles: cutlass.Int32, issuer: cutlass.Int32,
    ):  # fmt: skip
        """``KdaHeadWide.resident_b`` over this slice's ``tiles`` K tiles."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        for k_tile in cutlass.range(issuer, tiles, B_ISSUERS, unroll=1):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_first + k_tile],
                s_tile[None, None, None, k_tile],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def staged_mma_r24(
        self, bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc,
        sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr, tiles: cutlass.Int32,
    ):  # fmt: skip
        """``KdaHeadMulti.staged_mma_multi`` over ``tiles`` (18 / 19) K tiles. The
        tensor memory is freed as soon as the epilogue has read the accumulator."""
        from cutlass.cute.nvgpu import tcgen05

        from ..gemm.splitk import _fence_tcgen05
        from .head import _copy_to_tmem
        from .head_multi import _mma_ops

        N: cutlass.Constexpr = self.rows_total
        mma_smem, mma_tmem = _mma_ops(self.rows_total)
        stages, staged, k_blocks = self.num_ab_stage, self.tmem_tiles, self.cta_k // 16
        a_layout = cute.slice_(sA.layout, (None, None, None, 0))
        b_layout = cute.slice_(sB.layout, (None, None, None, 0))
        a_step = cute.size_in_bytes(sA.element_type, a_layout) // 16
        b_step = cute.size_in_bytes(sB.element_type, b_layout) // 16
        columns = self.cta_k // 2
        cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        a_desc = ((sA.iterator.toint() >> 4) & 0x3FFF) | 0x10000
        b_desc = ((sB.iterator.toint() >> 4) & 0x3FFF) | 0x10000
        for k in cutlass.range_constexpr(staged):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = a_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + N + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(staged):  # A from tensor memory
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    b = b_desc + k * b_step + 2 * kb
                    accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                    mma_tmem(base, base + N + columns * k + 8 * kb, b, accumulate)
        for k in cutlass.range(staged, tiles, unroll=1):  # A from the ring
            ring_stage, ring_phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + ring_stage, ring_phase)
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    a_ring = a_desc + ring_stage * a_step + 2 * kb
                    mma_smem(base, a_ring, b_desc + k * b_step + 2 * kb, Int32(1))
                tcgen05.commit(bar_empty + ring_stage, None, self.cta_group)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.experimental.jit
    def push_rows(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, partial: cute.Tensor, section: cutlass.Int32,
        split_rank: cutlass.Int32, x_in: cute.Tensor, bar_in,
    ):  # fmt: skip
        """``KdaHeadMulti.push_tokens`` with token row r's owner CTA r % 24 (mailbox
        slot r // 24)."""
        from ..gemm.splitk import _fence_tcgen05

        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        d_layout = utils.LayoutEnum.from_tensor(partial)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, BFloat16, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        part_epi = cute.flat_divide(partial, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, part_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute_ext.partition_and_copy(thr_t2r, rAcc, part_epi[None, None, 0, 0])
        named_barrier(3, 128)
        rows = self.cta_m
        for m in cutlass.range_constexpr(self.rows_total * 32 // 128):
            i = epi_tid + 128 * m
            token, chunk = i // 32, i % 32
            src = partial.iterator + cute.assume(token * rows + 4 * chunk, divby=4)
            words = load_words(src, count=4, shared=True)
            owner, mailbox = token % R24_CLUSTER, token // R24_CLUSTER
            column = (mailbox * SPLITS + split_rank) * TOKEN_INPUTS
            column = column + section * 128 + 4 * chunk
            dst = x_in.iterator + cute.assume(column, divby=4)
            smem = cute.AddressSpace.smem
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_in, owner)

    @cute.jit
    def round_owned_r24(self, kt, x_in, mine, s):
        """Owned row ``s``'s Q, K, V and gate (channel ``kt``): the six slices'
        partials summed in rank order, rounded once to BF16."""
        box = s * SPLITS * TOKEN_INPUTS
        for c in cutlass.range_constexpr(4):
            value = x_in[box + c * 128 + kt]
            for q in cutlass.range_constexpr(1, SPLITS):
                value = value + x_in[box + q * TOKEN_INPUTS + c * 128 + kt]
            mine[s * TOKEN_INPUTS + c * 128 + kt] = BFloat16(value)

    @cute.jit
    def hop_r24(self, kt, request, position, mine, s, raw_qk, raw_v, bar_hop):
        """Owned row ``s`` (token ``position`` of ``request``) to the request's six
        KDA CTAs, 16 bytes per store: Q / K channels to all six, V rows [8 c, 8 c + 8)
        to the CTA owning them."""
        T: cutlass.Constexpr = self.tokens
        smem = cute.AddressSpace.smem
        first = KC * request
        src_base = mine.iterator + s * TOKEN_INPUTS
        for m in cutlass.range_constexpr(2):
            i = kt + 128 * m
            dest, c = i // 32, i % 32
            if dest < KC:
                sec, sub = c // 16, c % 16
                src = src_base + 128 * sec + 8 * sub
                dst = raw_qk.iterator + (sec * T + position) * 128 + 8 * sub
                words = load_words(src, count=4, shared=True)
                dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
                store_async_dsmem(dst, words, bar_hop, first + dest)
        if kt < 16:  # V rows 8 kt .. 8 kt + 7
            row = 8 * kt
            j = self.row_block(row)
            jb, _ = self.block_rows(j)
            src = src_base + 256 + row
            dst = raw_v.iterator + position * RMAX + (row - jb)
            words = load_words(src, count=4, shared=True)
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_hop, first + j)

    # ----------------------------------------------------------- recurrence
    @cute.jit
    def r24_early(self, kt, tensors, coords, slots, staging):
        """``KdaHeadMultiDefer.multi_early`` for this CTA's rows (thread row
        base + min(kt // 8, rows - 1)); conv taps of Q / K channel kt (kt < 128) and
        V row kt (kt < rows)."""
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, _, offset, request, block, base, nrows = coords
        slot, records = slots[request, 0], slots[request, 1]
        pending = offset
        if (offset < 0) | (records <= 0):
            pending = Int32(0)
        group, group_lane = kt // 8, kt % 8
        row = base + _min(group, nrows - 1)
        kq = kt % 128
        safe = slot
        if safe < 0:
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        history = zeros((self.vkda.passes, self.vkda.chunks, 4), Float32)
        for j in cutlass.range_constexpr(self.vkda.chunks):
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            pointer = at(state_page, head, row, column)
            values = self.vkda.load_state(pointer)
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = values[e]
        if (slot > 0) & (pending > 0):  # the fold (pre-wait)
            record_page = state[Int64(records), None, None, None]
            self.fold_r24(history, record_page, head, pending, row, group_lane, kt,
                          staging)  # fmt: skip
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3 * self.tokens, Float32)
        if slot > 0:
            dim = self.dim
            qk_channel = head * 128 + kq
            value_channel = head * 128 + base + _min(kt, nrows - 1)
            conv_page = conv_state[Int64(slots[request, 0]), None, None]
            for s in cutlass.range_constexpr(3):
                channel = value_channel if s == 2 else qk_channel
                for t in cutlass.range_constexpr(3):
                    taps[s, t] = conv_tap(conv_page, s * dim + channel, offset + t)
                for t in cutlass.range_constexpr(4):
                    coeff[s, t] = conv_w[s, t, channel]
                if cutlass.const_expr(self.has_bias):
                    accum[s] = bias[s * dim + channel]
                    for t in cutlass.range_constexpr(1, self.tokens):
                        accum[3 * t + s] = accum[s]
        return history, taps, coeff, accum, norm_w[kq]

    @cute.jit
    def fold_r24(self, history, record_page, head, pending, row, group_lane, kt,
                 staging):  # fmt: skip
        """``HeadRecords.head_fold`` for state row ``row`` on the 192 loop threads."""
        vk = self.vkda
        stage_k, stage_d = staging
        for which in cutlass.range_constexpr(2):  # k rows 0 .., decay rows 8 ..
            target = (stage_k, stage_d)[which]
            for c in cutlass.range_constexpr(2):
                index = kt + LOOP_THREADS * c
                if index < 7 * 32:
                    r, column = index // 32, 4 * (index % 32)
                    if r < pending:
                        source = at(record_page, head, DECAY_ROW * which + r, column)
                        copy_g2s(target.iterator + r * 128 + column, source)
        cute.arch.cp_async_commit_group()
        vs = [record_page[head, V_ROW + i, row] for i in range(7)]
        cute.arch.cp_async_wait_group(0)
        named_barrier(4, LOOP_THREADS)  # every thread's staged chunks
        for i in cutlass.range_constexpr(7):
            if i < pending:
                for j in cutlass.range_constexpr(vk.chunks):
                    column = 4 * (group_lane + vk.lanes_per_row * j)
                    d4 = load_f32x4(stage_d.iterator + i * 128 + column, shared=True)
                    for e in cutlass.range_constexpr(4):
                        history[0, j, e] *= d4[e]
                for j in cutlass.range_constexpr(vk.chunks):
                    column = 4 * (group_lane + vk.lanes_per_row * j)
                    k4 = load_f32x4(stage_k.iterator + i * 128 + column, shared=True)
                    for e in cutlass.range_constexpr(4):
                        history[0, j, e] = fma(
                            k4[e], vs[i], history[0, j, e], ftz=vk.fast_math
                        )
        named_barrier(4, LOOP_THREADS)  # staging read before anyone reuses it

    @cute.jit
    def r24_gate(self, args, coords, shared_decay, shared_beta, barrier):
        """``KdaHeadMulti.multi_gate`` for this CTA's features [base, base + rows):
        F_B of up to 24 features (two M16 tiles), decays of the request's T tokens to
        its six KDA CTAs."""
        fa_beta, a_log, dt_bias, weight, lower_bound, counter, target = args
        head, rank, slot, request, block, base, nrows = coords
        T: cutlass.Constexpr = self.tokens
        lane = cute.arch.lane_idx()
        group, quarter = lane >> 2, lane & 3
        features = [group, group + 8, group + 16, group + 24]
        dt = [dt_bias[head * 128 + _min(base + f, 127)] for f in features]
        wr = cute.make_rmem_tensor((2, 8, 4), cutlass.Uint32)
        for tile in cutlass.range_constexpr(2):
            row = head * 128 + _min(base + 16 * tile + group, 119)
            for chunk in cutlass.range_constexpr(8):
                col = 16 * chunk + 4 * quarter
                low = load_bf16x4(at(weight, row, col))
                high = load_bf16x4(at(weight, row + 8, col))
                wr[tile, chunk, 0], wr[tile, chunk, 2] = low
                wr[tile, chunk, 1], wr[tile, chunk, 3] = high
        a = a_log[head]
        cute.arch.cluster_wait()
        token = T * request + group % T
        ar, beta = self.poll_fa(fa_beta, counter, target, token, quarter, head)
        cute.arch.griddepcontrol_launch_dependents()
        if slot > 0:
            first = KC * request
            for tile in cutlass.range_constexpr(2):
                acc = zeros(4, Float32)
                for chunk in cutlass.range_constexpr(8):
                    wa = tuple(wr[tile, chunk, e] for e in range(4))
                    mma_into(acc, wa, ar[chunk], bf16=True)
                for i in cutlass.range_constexpr(2):
                    feature = features[2 * tile + i]
                    for column in cutlass.range_constexpr(2):
                        raw_g = Float32(BFloat16(acc[2 * i + column]))
                        decay = self.head_decay(a, raw_g, dt[2 * tile + i], lower_bound)
                        decay = float_bits(decay)
                        t = 2 * quarter + column
                        pointer = shared_decay.iterator + t * 128 + base + feature
                        if (t < T) & (feature < nrows):
                            for dest in cutlass.range_constexpr(KC):
                                store_async_dsmem(pointer, decay, barrier, first + dest)
            if (quarter == 0) & (group < T):  # this CTA's betas
                beta_gate = self.activate(Float32(beta))
                push_float(shared_beta.iterator + group, beta_gate, barrier, rank)
        self.rearm_r24(counter, target, lane, fa_beta)

    @cute.jit
    def rearm_r24(self, counter, target, lane, fa_beta):
        """``KdaHead.rearm`` (fa_poll) with 24 readers per head."""
        last = Int32(0)
        if (target > 0) & (lane == 0):
            readers = Int32(R24_CLUSTER * self.heads)
            if Int32(_fetch_add(counter + 4)) == readers - 1:
                last = Int32(1)
                seen = Int32(_acquire_load(counter))
                sgn, sgt = spin_start()
                while seen < target:
                    seen = Int32(_acquire_load(counter))
                    sgn, sgt = spin_tick(sgn, sgt, observed=seen, target=0)
                _store_zero(counter)
                _store_zero(counter + 4)
        self.refill_fa(fa_beta, cute.arch.shuffle_sync(last, 0), lane)

    @cute.jit
    def r24_main(self, kt, early, raws, rows, bars, tensors, coords, slots, scale, eps):
        """``KdaHeadMultiDefer.multi_main`` for this CTA's request and rows on the 192
        loop threads; the outputs go to each token row's owner (``norm_owned``)."""
        T: cutlass.Constexpr = self.tokens
        dim = self.dim
        history, taps, coeff, accum, nw = early
        raw_qk, raw_v, mine, bar_hop = raws
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows[:6]
        local_o, slot_rows = rows[6:]
        bar_gate, bar_out, shared_beta, bar_qk = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset, request, block, base, nrows = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // 8, kt % 8
        row = base + _min(group, nrows - 1)
        kq = kt % 128
        qk_channel = head * 128 + kq
        first = KC * request
        conv_slot = slots[request, 0]
        records = slots[request, 1]
        live = records > 0
        record_page = state[Int64(cutlass.select_(live, records, Int32(0))), None, None, None]
        if slot > 0:
            conv_page = conv_state[Int64(conv_slot), None, None]
            wait(bar_hop)
            inputs = cute.make_rmem_tensor(3 * T, BFloat16)
            vrow = _min(kt, nrows - 1)
            for t in cutlass.range_constexpr(T):
                inputs[3 * t] = raw_qk[t * 128 + kq]
                inputs[3 * t + 1] = raw_qk[(T + t) * 128 + kq]
                inputs[3 * t + 2] = raw_v[t * RMAX + vrow]
            for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        cf = Float32(coeff[s, i])
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), cf, acc)
            if kt < nrows:
                for t in cutlass.range_constexpr(T):
                    shared_v[t * RMAX + kt] = accum[3 * t + 2]
            # token ``block`` (and ``block + 2`` for blocks 4 / 5: tokens 6 / 7)
            for n in cutlass.range_constexpr(2):
                tn = block + 2 * n
                if (n == 0) | (block >= 4):
                    pre_q, pre_k = accum[0], accum[1]
                    for t in cutlass.range_constexpr(1, T):
                        pre_q = Float32(cutlass.select_(tn == t, accum[3 * t], pre_q))
                        pre_k = Float32(cutlass.select_(tn == t, accum[3 * t + 1], pre_k))
                    if cutlass.const_expr(self.stock):
                        q, k = self.silu(pre_q), self.silu(pre_k)
                    else:
                        act = sigmoids([pre_q, pre_k])
                        q, k = pre_q * act[0], pre_k * act[1]
                    square_q, square_k = warp_sum(q * q), warp_sum(k * k)
                    if (lane == 0) & (kt < 128):
                        reductions[8 * n + warp] = square_q
                        reductions[8 * n + warp + 4] = square_k
                    named_barrier(1, LOOP_THREADS)
                    r = reductions
                    qn = (r[8 * n] + r[8 * n + 2]) + (r[8 * n + 1] + r[8 * n + 3])
                    kn = (r[8 * n + 4] + r[8 * n + 6]) + (r[8 * n + 5] + r[8 * n + 7])
                    fq = self.norm_factor(qn + 1.0e-6, scale)
                    fk = self.norm_factor(kn + 1.0e-6)
                    q_n, k_n = self.normalized(q, fq, scale), self.normalized(k, fk)
                    if kt < 128:
                        q_ptr = shared_q.iterator + tn * 128 + kt
                        k_ptr = shared_k.iterator + tn * 128 + kt
                        for dest in cutlass.range_constexpr(KC):
                            push_float(q_ptr, q_n, bar_qk, first + dest)
                            push_float(k_ptr, k_n, bar_qk, first + dest)
            named_barrier(1, LOOP_THREADS)  # shared_v
            pre_v = shared_v[kt]  # (row kt % 24, token kt // 24; rows past nrows unused)
            shared_v[kt] = self.silu_v(pre_v)
            if kt < T:
                slot_rows[kt] = slots[request, kt]
            named_barrier(1, LOOP_THREADS)  # shared_v, slot_rows
            wait(bar_qk)
            wait(bar_gate)
            if (block == 0) & live & (kt < 128):
                self.key_records(record_page, head, kt, shared_k, shared_decay)
            lanes = (group, group_lane)
            chunks: cutlass.Constexpr = self.vkda.chunks
            buffers = (
                cute.make_rmem_tensor((3, chunks, 4), Float32),
                cute.make_rmem_tensor((3, chunks, 4), Float32),
            )
            for j in cutlass.range_constexpr(chunks):
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, j, e] = Float32(0)
            sources = (shared_q, shared_k, shared_decay)
            self.load_operands(buffers[0], sources, 0, group_lane)
            mine_row = group < nrows
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    cur, other = buffers[half], buffers[1 - half]
                    v_row = shared_v[t * RMAX + row - base]
                    beta = shared_beta[t]
                    out_prev, v_new = self.rolled_update_defer(
                        history, cur, other, sources, t, group_lane, v_row, beta
                    )
                    if (group_lane == 0) & (t > 0):
                        local_o[(t - 1) * RMAX + group] = self.round_output(out_prev)
                    if t == 0:  # the one state store: slot 0, after token 0
                        self.store_rows(history, state, slot_rows[0], head, row,
                                        group_lane, mine_row)  # fmt: skip
                    elif (group_lane == 0) & live & mine_row:  # token t's corrections
                        record_page[head, V_ROW + t - 1, row] = v_new
            out_last = self.query_dot(history, buffers[1][0, None, None])
            if group_lane == 0:
                local_o[(T - 1) * RMAX + group] = self.round_output(out_last)
            named_barrier(1, LOOP_THREADS)
            # thread kt: row kt % 24 of token kt // 24 -> that token row's owner
            t_o, r_o = kt // RMAX, kt % RMAX
            if r_o < nrows:
                token_row = T * request + t_o
                owner, mailbox = token_row % R24_CLUSTER, token_row // R24_CLUSTER
                position = self.vkda.output_position(base + r_o)
                dst = shared_o.iterator + mailbox * 128 + position
                push_float(dst, local_o[kt], bar_out + mailbox, owner)
            if kt < nrows:
                channel = 2 * dim + head * 128 + base + kt
                window = (
                    taps[2, 1],
                    taps[2, 2],
                    *(inputs[3 * t + 2] for t in range(T)),
                )
                for i, value in enumerate(window):
                    conv_put(conv_page, (channel, i), value)
            if (block == 0) & (kt < 128):
                for s in cutlass.range_constexpr(2):
                    channel = s * dim + qk_channel
                    window = (taps[s, 1], taps[s, 2])
                    window += tuple(inputs[3 * u + s] for u in range(T))
                    for i, value in enumerate(window):
                        conv_put(conv_page, (channel, i), value)

    @cute.jit
    def store_rows(self, history, state, slot, head, row, group_lane, mine_row):
        """``KdaHeadVerify.store_checkpoint`` of state row ``row`` (if this thread's
        group owns a row of the CTA)."""
        vk = self.vkda
        page = state[Int64(slot), None, None, None]
        for j in cutlass.range_constexpr(vk.chunks):
            column = 4 * (group_lane + vk.lanes_per_row * j)
            values = [history[0, j, e] for e in range(4)]
            pointer = at(page, head, row, column)
            vk.store_state(pointer, values, (slot > 0) & mine_row)

    @cute.jit
    def norm_owned(self, kt, owned, slots, head, r, s, eps):
        """``KdaHeadMulti.owned_norm`` of owned token row ``r`` (mailbox ``s``)."""
        mine, shared_o, bar_out, output, norm_w, accepted = owned
        T: cutlass.Constexpr = self.tokens
        request = r // T
        qk_channel = head * 128 + kt
        if slots[request, accepted[request] - 1] <= 0:
            output[r, qk_channel] = BFloat16(0)
        else:
            out_gate = Float32(0)
            gate = Float32(mine[s * TOKEN_INPUTS + 384 + kt])
            if cutlass.const_expr(self.stock):
                out_gate = self.activate(gate)
            else:
                out_gate = sigmoids([gate])[0]
            nw = norm_w[kt]
            wait(bar_out + s)
            box = cute.make_tensor(shared_o.iterator + s * 128, cute.make_layout(128))
            self.norm_row(kt, box, output, r, qk_channel, nw, out_gate, eps)
