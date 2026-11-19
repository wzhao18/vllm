# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused prefill DSA indexer for Kimi K3.1 on sm_107 (VR200).

Replaces the stock prefill chain of ``sparse_attn_indexer`` (per 512 MB chunk:
DeepGEMM ``fp8_fp4_mqa_logits`` writing BF16 logits to HBM, then the trtllm_gvr2
top-k re-reading them) by two kernels that never materialize the logits:

  IndexerScoreSelect  persistent tcgen05 kernel, one CTA per SM. A work item is
                      8 query tokens of one request (B operand: 8 x 32 heads x 128
                      MXFP4, N = 256); it walks the request's keys in 128-key tiles
                      (A operand, M = 128) from the newest to the oldest with one
                      2x-K block-scaled MXF4 MMA (K = 128) per tile. The epilogue
                      (four warpgroups, two tokens each, one thread per key) forms
                      the BF16 logit at stock precision and streams it through an
                      exact top-k filter: a per-token candidate buffer in shared
                      memory holds every key at or above a running threshold
                      ``tau`` (a key value with at least ``topk`` seen keys at or
                      above it), compacted to the top ``topk`` (ties to the lowest
                      key index) whenever it fills. At the end of an item the
                      buffer is compacted to exactly the top ``topk`` and written
                      out. Exactness: a key below ``tau`` has ``topk`` better keys,
                      and a compaction drops only keys with ``topk`` better keys
                      in the buffer, so the buffer always contains the exact top-k.
  sort_rows           Triton: each row's selected key indices in ascending order,
                      -1 padded (deterministic output order).

Stock precision (DeepGEMM SM100 MQA logits, MXFP4 Q and K, BF16 weights):
logit[t, k] = sum over heads h of bf16(relu(q[t, h] . k)) * w[t, h], each head
term rounded to BF16 by cvt.rn.relu.bf16x2.f32 and accumulated by BF16x2 FMAs in
two partial sums (heads 4i, 4i + 1 and 4i + 2, 4i + 3), then
(s0.lo + s1.lo) + (s0.hi + s1.hi) in BF16. The only order difference is inside
the MMA (one K = 128 instruction instead of two K = 64 instructions).
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.experimental import primitives as nvvm

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import indexer as _sdec
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
    BINS,
    CANDIDATES,
    FINE,
    HOT,
    LOW,
    SELECT_THREADS,
    SELECT_WARPS,
    block_exclusive,
    fine_bin_of,
    half,
    load_16b,
    low_digit,
    shared_add,
    store_shared_words,
    topdown,
    vector_keys,
    vector_max,
)

HEADS = 32
DIM = 128
TILE = 128  # keys per MMA tile (M)
BQ = 8  # query tokens per work item
N = BQ * HEADS  # MMA N
KV_STAGES = 6
Q_STAGES = 2
ACC_STAGES = 2
EPI_WG = 4  # epilogue warpgroups
TOK_WG = BQ // EPI_WG  # tokens per epilogue warpgroup
PRODUCER = 4 * EPI_WG
MMA_WARP = PRODUCER + 1
THREADS = 32 * (MMA_WARP + 1)
TOK_PROC = BQ * ACC_STAGES // EPI_WG  # tokens per warpgroup and processed tile
OUT_BUFS = 3  # staging buffers per warpgroup (one per processed tile in flight)
TMEM_COLS = 576  # exclusive allocation: 2 x N accumulators + scale factors
SF_WORDS_A = TILE  # one 32-bit word (four UE8M0) per key
SF_WORDS_B = N


# --------------------------------------------------------------------------
# PTX helpers
# --------------------------------------------------------------------------
@asm(Uint32, pure=True)
def relu_bf16x2(lo: Float32, hi: Float32):
    """cvt.rn.relu.bf16x2.f32 (DeepGEMM cvt_relu_bf16x2_f32): ``lo`` -> low half."""
    return "cvt.rn.relu.bf16x2.f32 $0, $2, $1;", "=r,f,f", (lo, hi)


@asm(Uint32, pure=True)
def fma_bf16x2(a: Uint32, b: Uint32, c: Uint32):
    return "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r", (a, b, c)


@asm(Uint32, pure=True)
def add_bf16x2(a: Uint32, b: Uint32):
    return "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", (a, b)


@asm()
def bulk_g2s(dst, src, barrier, size):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    code += "[$0], [$1], $3, [$2];"
    return code, "r,l,r,r,~{memory}", (Int32(dst), Int64(src), barrier, Int32(size))


@asm()
def bulk_s2g(dst, src, size):
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (Int64(dst), Int32(src), Int32(size))


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm()
def bulk_wait_read1():
    return "cp.async.bulk.wait_group.read 1;", "~{memory}", ()


@asm()
def bulk_wait_all():
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm(Uint32)
def ld_global_u32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def st_global_u32(address, value):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def st_global_u16(address, value):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0], t; }"
    return code, "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def lds_v4(address):
    """Four shared words (16-byte aligned)."""
    code = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),), (Uint32,) * 4


@asm(Uint32)
def ld_shared_u32(address):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm(Uint32)
def ld_shared_u16(address):
    return "ld.shared.u16 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def st_shared_u32(address, value):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def st_shared_u16(address, value):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.shared.b16 [$0], t; }"
    return code, "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def st_shared_v4(address, a, b, c, d):
    code = "st.shared.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "r,r,r,r,r,~{memory}", (
        Int32(address),
        Uint32(a),
        Uint32(b),
        Uint32(c),
        Uint32(d),
    )


@asm(Uint32)
def atom_add_shared(address, value):
    code = "atom.shared.add.u32 $0, [$1], $2;"
    return code, "=r,r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def red_add_u64(address, value):
    return "red.global.add.u64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint32, pure=True)
def lanemask_lt():
    return "mov.u32 $0, %lanemask_lt;", "=r", ()


@asm(Uint32, pure=True)
def clz(value):
    return "clz.b32 $0, $1;", "=r,r", (Uint32(value),)


@asm(Uint32, pure=True)
def popc(value):
    return "popc.b32 $0, $1;", "=r,r", (Uint32(value),)


@asm()
def fence_async_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@cute.jit
def order16(bits: Uint32) -> Uint32:
    """Order-preserving 16-bit key of BF16 ``bits``; -0 is keyed as +0."""
    b = bits
    if b == Uint32(0x8000):
        b = Uint32(0)
    flip = Uint32(0x8000)
    if (b & Uint32(0x8000)) != Uint32(0):
        flip = Uint32(0xFFFF)
    return b ^ flip


@cute.jit
def wg_sync(wg):
    cute.arch.barrier(barrier_id=2 + wg, number_of_threads=128)


def _mn_layout(t):
    """A tcgen05 accumulator as ((M), (N), stages)."""
    sh, sd = t.layout.shape, t.layout.stride
    new = cute.make_layout(
        ((sh[0][0], sh[1]), (sh[0][1], sh[2]), *sh[3:]),
        stride=((sd[0][0], sd[1]), (sd[0][1], sd[2]), *sd[3:]),
    )
    return cute.make_tensor(t.iterator, new)


def sf_word(row: int) -> int:
    """32-bit word of row ``row``'s four UE8M0 scales in the block-scaled SF atom
    ((32, 4), (32, 4)):((16, 4), (0, 1)) bytes, 128 rows per 512-byte chunk."""
    return 128 * (row // 128) + 4 * (row % 32) + (row % 128) // 32


class IndexerScore:
    """Exact top-``topk`` key indices of every query row of the step's prefill
    requests (see the module docstring)."""

    def __init__(
        self, ctas: int, mode: int = 0,
    ):
        """mode (bench only): 0 full, 1 logits without selection, 2 MMA +
        pipeline only (the epilogue releases the accumulators unread)."""
        self.no_store = mode == 3
        self.prof = mode == 4
        self.mode = 0 if mode in (3, 4) else mode
        self.ctas = ctas
        self.tiler = (TILE, N, DIM)

    @cute.jit
    def __call__(
        self,
        q8: cute.Tensor,  # Uint8 [T * 32, 64]: MXFP4 query, two E2M1 per byte
        q_sf: cute.Tensor,  # Int32 [T * 32]: four UE8M0 per (token, head)
        k8: cute.Tensor,  # Uint8 [NK, 64]: MXFP4 keys of all requests
        k_sf: cute.Tensor,  # Int32 [NK]
        weights: cute.Tensor,  # Int32 [T, 16]: BF16x2 head weights
        row_len: cute.Tensor,  # Int32 [T]: valid keys of each row
        items: cute.Tensor,  # Int32 [n_items, 4]: token0, ntok, key0, ntiles
        row_base: cute.Tensor,  # Int64 [T]: row t's first element in logits
        logits: cute.Tensor,  # Int16 [*]: BF16 logits scratch
        tops: cute.Tensor,  # Int16 [*]: each 8 logits' largest order-preserving key
        s,
    ):
        fp4 = cutlass.Float4E2M1FN
        sf_t = cutlass.Float8E8M0FNU
        nq = q8.shape[0]
        nk = k8.shape[0]
        mQ = cute.make_tensor(
            cute.recast_ptr(q8.iterator, dtype=fp4),
            cute.make_layout((nq, DIM, 1), stride=(DIM, 1, nq * DIM)),
        )
        mK = cute.make_tensor(
            cute.recast_ptr(k8.iterator, dtype=fp4),
            # Rows of a 128-key tile in SF-atom order: MMA row r holds key
            # 4 * (r % 32) + r // 32, so the tile's 128 scale words copy contiguously
            # into the scale-factor atom ((32, 4)):((16 B, 4 B)) (one bulk copy).
            cute.make_layout(
                ((32, 4, nk // TILE), DIM, 1),
                stride=((4 * DIM, DIM, TILE * DIM), 1, nk * DIM),
            ),
        )
        mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            fp4,
            fp4,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            sf_t,
            32,
            tcgen05.CtaGroup.ONE,
            (TILE, N, DIM),
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, fp4, KV_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, fp4, Q_STAGES)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(
            mma, self.tiler, 32, KV_STAGES
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            mma, self.tiler, 32, Q_STAGES
        )
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        sfb_one = cute.slice_(sfb_layout, (None, None, None, 0))
        tsfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.tiler, 32, sfa_one
        )
        tsfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.tiler, 32, sfb_one
        )
        self.sfa_cols = (
            cute.cosize(cute.recast_layout(32, sf_t.width, tsfa_layout)) & 0xFFFF
        )
        self.sfb_cols = (
            cute.cosize(cute.recast_layout(32, sf_t.width, tsfb_layout)) & 0xFFFF
        )
        assert ACC_STAGES * N + self.sfa_cols + self.sfb_cols <= TMEM_COLS
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_k, tma_k_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mma.thr_id),
            mK,
            a_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            mQ,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.k_bytes = cute.size_in_bytes(fp4, a_one)
        self.q_bytes = cute.size_in_bytes(fp4, b_one)
        self.kernel(
            mma,
            tma_k,
            tma_k_tensor,
            tma_q,
            tma_q_tensor,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            tsfa_layout,
            tsfb_layout,
            q_sf,
            k_sf,
            weights,
            row_len,
            items,
            row_base,
            logits,
            tops,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=s,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        tsfa_layout: cute.Layout,
        tsfb_layout: cute.Layout,
        q_sf: cute.Tensor,
        k_sf: cute.Tensor,
        weights: cute.Tensor,
        row_len: cute.Tensor,
        items: cute.Tensor,
        row_base: cute.Tensor,
        logits: cute.Tensor,
        tops: cute.Tensor,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        n_items = items.shape[0]
        sf_t = cutlass.Float8E8M0FNU

        @cute.struct
        class Storage:
            kv_full: cute.struct.MemRange[cutlass.Int64, KV_STAGES]
            kv_empty: cute.struct.MemRange[cutlass.Int64, KV_STAGES]
            q_full: cute.struct.MemRange[cutlass.Int64, Q_STAGES]
            q_empty: cute.struct.MemRange[cutlass.Int64, Q_STAGES]
            acc_full: cute.struct.MemRange[cutlass.Int64, ACC_STAGES]
            acc_empty: cute.struct.MemRange[cutlass.Int64, ACC_STAGES]
            stage_out: cute.struct.Align[
                cute.struct.MemRange[Uint32, EPI_WG * OUT_BUFS * TOK_PROC * 72], 16
            ]
            tmem_hold: Int32
            row_end: cute.struct.MemRange[Int32, Q_STAGES * BQ]
            w: cute.struct.Align[cute.struct.MemRange[Uint32, Q_STAGES * BQ * 16], 16]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        sA = smem.allocate_tensor(
            cutlass.Float4E2M1FN, a_layout.outer, 1024, swizzle=a_layout.inner
        )
        sB = smem.allocate_tensor(
            cutlass.Float4E2M1FN, b_layout.outer, 1024, swizzle=b_layout.inner
        )
        sSFA = smem.allocate_tensor(sf_t, sfa_layout, 128)
        sSFB = smem.allocate_tensor(sf_t, sfb_layout, 128)

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        end_a = st.row_end.data_ptr().toint()
        w_a = st.w.data_ptr().toint()
        sfa_a = sSFA.iterator.toint()
        sfb_a = sSFB.iterator.toint()
        if tidx == 0:
            for i in cutlass.range_constexpr(KV_STAGES):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
            for i in cutlass.range_constexpr(Q_STAGES):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + 4 * EPI_WG)
            for i in cutlass.range_constexpr(ACC_STAGES):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 4 * EPI_WG // ACC_STAGES)
            cute.arch.mbarrier_init_fence()
        if warp == PRODUCER:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)
        if warp == MMA_WARP:
            nvvm.tcgen05_alloc(
                st.tmem_hold.ptr,
                TMEM_COLS,
                is_exclusive=True,
                group=nvvm.CTAGroup.CTA_1,
            )
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, st.tmem_hold.ptr)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))
        tAcc = cute.make_tensor(tmem_ptr, acc_fake.layout)
        cute.arch.griddepcontrol_wait()

        if warp == PRODUCER:
            thr_mma = mma.get_slice(0)
            gK = cute.local_tile(
                mK, cute.slice_(self.tiler, (None, 0, None)), (None, None, None)
            )
            tKs, tKg = cpasync.tma_partition(
                tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gK), 0, 3),
            )
            kv_it = Int32(0)
            jq = Int32(0)
            for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
                token0 = items[item, 0]
                ntok = items[item, 1]
                key0 = items[item, 2]
                ntiles = items[item, 3]
                qs = jq % Q_STAGES
                cute.arch.mbarrier_wait(q_empty + qs, ((jq // Q_STAGES) & 1) ^ 1)
                # row ends, head weights, query scales (SF atom order)
                if lane < BQ:
                    e = Int32(0)
                    if lane < ntok:
                        e = row_len[token0 + lane]
                    st_shared_u32(end_a + (qs * BQ + lane) * 4, Uint32(e))
                for i in cutlass.range_constexpr(4):
                    wi = lane * 4 + i  # BQ * 16 = 128 words
                    tok = wi // 16
                    v = Uint32(0)
                    if tok < ntok:
                        v = Uint32(weights[token0 + tok, wi % 16])
                    st_shared_u32(w_a + (qs * BQ * 16 + wi) * 4, v)
                for c in cutlass.range_constexpr(N // 128):
                    words = []
                    for i in cutlass.range_constexpr(4):
                        r = c * 128 + 32 * i + lane
                        v = Uint32(0)
                        if r // HEADS < ntok:
                            v = Uint32(q_sf[token0 * HEADS + r])
                        words.append(v)
                    st_shared_v4(
                        sfb_a + (qs * SF_WORDS_B + c * 128 + 4 * lane) * 4, *words
                    )
                fence_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full + qs, self.q_bytes)
                gQ = cute.local_tile(
                    cute.domain_offset((token0 * HEADS, 0, 0), mQ),
                    cute.slice_(self.tiler, (0, None, None)),
                    (None, None, None),
                )
                tQs, tQg = cpasync.tma_partition(
                    tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gQ), 0, 3),
                )
                cute.copy(tma_q, tQg[(None, 0, 0, 0)], tQs[(None, qs)], tma_bar_ptr=q_full + qs)
                jq += 1
                for tt in cutlass.range(ntiles, unroll=1):
                    t = ntiles - 1 - tt
                    stage = kv_it % KV_STAGES
                    cute.arch.mbarrier_wait(kv_empty + stage, ((kv_it // KV_STAGES) & 1) ^ 1)
                    k_first = key0 + t * TILE
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kv_full + stage, self.k_bytes + TILE * 4
                        )
                        bulk_g2s(
                            sfa_a + stage * SF_WORDS_A * 4,
                            k_sf.iterator.toint() + Int64(k_first) * 4,
                            kv_full + stage,
                            TILE * 4,
                        )
                    cute.copy(
                        tma_k,
                        tKg[(None, k_first // TILE, 0, 0)],
                        tKs[(None, stage)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    kv_it += 1
        elif warp == MMA_WARP:
            tCrA = mma.make_fragment_A(sA)
            tCrB = mma.make_fragment_B(sB)
            sfa_ptr = cute.recast_ptr(tmem_ptr + ACC_STAGES * N, dtype=sf_t)
            sfb_ptr = cute.recast_ptr(tmem_ptr + ACC_STAGES * N + self.sfa_cols, dtype=sf_t)
            tCtSFA = cute.make_tensor(sfa_ptr, tsfa_layout)
            tCtSFB = cute.make_tensor(sfb_ptr, tsfb_layout)
            s2t_a = self.s2t(sSFA, tCtSFA)
            s2t_b = self.s2t(sSFB, tCtSFB)
            kv_it = Int32(0)
            acc_it = Int32(0)
            jq = Int32(0)
            for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
                ntiles = items[item, 3]
                qs = jq % Q_STAGES
                cute.arch.mbarrier_wait(q_full + qs, (jq // Q_STAGES) & 1)
                cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, qs)], s2t_b[2])
                for _t in cutlass.range(ntiles, unroll=1):
                    stage = kv_it % KV_STAGES
                    a = acc_it % ACC_STAGES
                    cute.arch.mbarrier_wait(acc_empty + a, ((acc_it // ACC_STAGES) & 1) ^ 1)
                    cute.arch.mbarrier_wait(kv_full + stage, (kv_it // KV_STAGES) & 1)
                    cute.copy(s2t_a[0], s2t_a[1][(None, None, None, None, stage)], s2t_a[2])
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(
                            mma,
                            tAcc[(None, None, None, a)],
                            [tCrA[(None, None, kb, stage)], tCtSFA[(None, None, kb)]],
                            [tCrB[(None, None, kb, qs)], tCtSFB[(None, None, kb)]],
                            tAcc[(None, None, None, a)],
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + stage)
                        tcgen05.commit(acc_full + a)
                    kv_it += 1
                    acc_it += 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty + qs)
                jq += 1
        else:
            self.epilogue(
                tAcc, acc_full, acc_empty, q_full, q_empty, items, row_base, logits,
                tops, (end_a, w_a, st.stage_out.data_ptr().toint()), warp, tidx, bid,
                lane,
            )

        cute.arch.sync_threads()
        if warp == MMA_WARP:
            nvvm.tcgen05_dealloc(
                nvvm.make_tmem_ptr(Int32(ld_shared_u32(st.tmem_hold.ptr.toint())), cutlass.Int8),
                TMEM_COLS,
                is_exclusive=True,
                group=nvvm.CTAGroup.CTA_1,
            )

    def s2t(self, sSF, tSF):
        """SMEM -> TMEM scale-factor copy (tcgen05.cp 4x32dp128bit), as FlashInfer's
        dense block-scaled GEMM."""
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU)
        tiled = tcgen05.make_s2t_copy(atom, tc)
        thr = tiled.get_slice(0)

        def bcast(lay):
            mn = cute.get(lay, mode=[0, 0])
            mn = cute.append(mn, cute.make_layout((4), stride=(0)))
            out = cute.append(cute.group_modes(mn, 0), cute.get(lay, mode=[0, 1]))
            out = cute.append(cute.group_modes(out, 0), cute.get(lay, mode=[1]))
            out = cute.append(out, cute.get(lay, mode=[2]))
            out = cute.append(out, cute.get(lay, mode=[3]))
            return out

        src = thr.partition_S(cute.make_tensor(sc.iterator, bcast(sc.layout)))
        src = tcgen05.get_s2t_smem_desc_tensor(tiled, src)
        dst = thr.partition_D(tc)
        return tiled, src, dst

    @cute.jit
    def epilogue(
        self, tAcc, acc_full, acc_empty, q_full, q_empty, items, row_base, logits,
        tops, sm, warp, tidx, bid, lane,
    ):
        """Warpgroup pair p = wg // 2 owns accumulator stage p (every other tile);
        warpgroup wg takes tokens 4 * (wg % 2) .. + 4 of it, so the pairs' TMEM
        reads and math interleave. Per tile: BF16 logits at stock precision in
        MMA-row (position) order and their 8-position maxima, staged in shared
        memory, then bulk stores (row t's scratch at row_base[t])."""
        end_a, w_a, stage_a = sm
        wg = warp // 4
        pair = wg // 2
        tok0 = (wg % 2) * TOK_PROC
        lt = tidx % 128
        out_s = stage_a + wg * OUT_BUFS * TOK_PROC * 288
        gq = Int32(0)
        n_items = items.shape[0]
        ld_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32
        )
        acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
        t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
        thr = t2r.get_slice(lt)
        tAll = thr.partition_S(acc_cols)
        cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
        out_a = logits.iterator.toint()
        tops_a = tops.iterator.toint()
        acc_it = Int32(0)
        jq = Int32(0)
        pw_acc = Int64(0)
        pw_math = Int64(0)
        pw_stage = Int64(0)
        pw_bar = Int64(0)
        pw_n = Int64(0)
        for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
            token0 = items[item, 0]
            ntok = items[item, 1]
            ntiles = items[item, 3]
            qs = jq % Q_STAGES
            cute.arch.mbarrier_wait(q_full + qs, (jq // Q_STAGES) & 1)
            ends = []
            bases = []
            tbases = []
            for j in cutlass.range_constexpr(TOK_PROC):
                tok = tok0 + j
                ends.append(Int32(ld_shared_u32(end_a + (qs * BQ + tok) * 4)))
                rb = Int64(0)
                if tok < ntok:
                    rb = row_base[token0 + tok]
                bases.append(out_a + rb * 2)
                tbases.append(tops_a + (rb >> 3) * 2)
            wbase = w_a + (qs * BQ * 16 + tok0 * 16) * 4
            for tt in cutlass.range(ntiles, unroll=1):
                t = ntiles - 1 - tt
                a = acc_it % ACC_STAGES
                if a == pair:
                    c0 = clock64()
                    cute.arch.mbarrier_wait(acc_full + a, (acc_it // ACC_STAGES) & 1)
                    c1 = clock64()
                    bits = []
                    for bt in cutlass.range_constexpr(TOK_PROC // 2):
                        frags = []
                        for j in cutlass.range_constexpr(2):
                            frag = cute.make_fragment_like(cD, Float32)
                            cute.copy(
                                t2r,
                                tAll[(None, None, None, 0, tok0 + 2 * bt + j, a)],
                                frag,
                            )
                            frags.append(frag)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(bt == TOK_PROC // 2 - 1):
                            cute.arch.sync_warp()
                            if lane == 0:
                                cute.arch.mbarrier_arrive(acc_empty + a)
                        if cutlass.const_expr(self.mode < 2):
                            ss = [[Uint32(0), Uint32(0)] for _ in range(2)]
                            for h in cutlass.range_constexpr(0, HEADS, 8):
                                for j in cutlass.range_constexpr(2):
                                    fr = frags[j]
                                    wr = lds_v4(wbase + ((2 * bt + j) * 16 + h // 2) * 4)
                                    for g in cutlass.range_constexpr(2):
                                        hh = h + 4 * g
                                        ss[j][0] = fma_bf16x2(
                                            relu_bf16x2(fr[hh], fr[hh + 1]), wr[2 * g], ss[j][0]
                                        )
                                        ss[j][1] = fma_bf16x2(
                                            relu_bf16x2(fr[hh + 2], fr[hh + 3]),
                                            wr[2 * g + 1],
                                            ss[j][1],
                                        )
                            for j in cutlass.range_constexpr(2):
                                pr = add_bf16x2(ss[j][0], ss[j][1])
                                bits.append(add_bf16x2(pr, pr >> Uint32(16)) & Uint32(0xFFFF))
                    c2 = clock64()
                    if cutlass.const_expr(self.mode == 0):
                        kidx = t * TILE + 4 * (lt % 32) + lt // 32
                        ob = out_s + (gq % OUT_BUFS) * TOK_PROC * 288
                        for j in cutlass.range_constexpr(TOK_PROC):
                            v = Uint32(0xFFFF)
                            if kidx < ends[j]:
                                v = bits[j]
                            st_shared_u16(ob + j * 288 + lt * 2, v)
                        fence_async_shared()
                        c3 = clock64()
                        wg_sync(wg)
                        c4 = clock64()
                        if cutlass.const_expr(self.prof):
                            pw_acc += c1 - c0
                            pw_math += c2 - c1
                            pw_stage += c3 - c2
                            pw_bar += c4 - c3
                            pw_n += Int64(1)
                        if cutlass.const_expr(not self.no_store):
                            if lt == 0:
                                for j in cutlass.range_constexpr(TOK_PROC):
                                    if ends[j] > 0:
                                        bulk_s2g(
                                            bases[j] + Int64(t * TILE) * 2, ob + j * 288, 256
                                        )
                                bulk_commit()
                                bulk_wait_read1()
                        gq += 1
                acc_it += 1
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(q_empty + qs)
            jq += 1
        if lt == 0:
            bulk_wait_all()
        if cutlass.const_expr(self.prof):
            if lane == 0:
                pa = tops_a + Int64(tops.shape[0]) * 2 - 64
                red_add_u64(pa, pw_acc)
                red_add_u64(pa + 8, pw_math)
                red_add_u64(pa + 16, pw_stage)
                red_add_u64(pa + 24, pw_bar)
                red_add_u64(pa + 32, pw_n)


# --------------------------------------------------------------------------
# Exact top-k over the logits scratch (one CTA per row)
# --------------------------------------------------------------------------
SEL_TOPS = 16384  # vector maxima per row in shared memory (rows <= 131072 keys)
SEL_UNROLL = 8  # 16-byte logits loads in flight per thread


class PrefillSelect(_sdec.IndexerSelect):
    """sdec IndexerSelect (exact top-k, ties to the lowest key index, output in key
    order) for prefill rows: row r's BF16 logits at logits[row_base[r] + key] for
    key < row_len[r]; pass 1 reads the logits themselves (the vector maxima are
    computed here, not by the scorer). Writes the key indices, -1 padded."""

    def __init__(self, topk: int = 2048):
        assert topk % SELECT_THREADS == 0
        self.topk = topk
        self.q = 1
        self.rows = 0
        self.logical = False
        self.release = True
        self.page_shift = 6

    @cute.jit
    def __call__(self, logits, tops, row_base, row_len, out, s):
        self.kernel((logits, tops, row_base, row_len, out)).launch(
            grid=(row_len.shape[0], 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def emit(self, out, ids, table, request, row, pos, key, block_rows):
        # logits position -> key index (rows of a tile in SF-atom order)
        p = Int32(key)
        out[row, pos] = (p & Int32(-TILE)) | ((p & Int32(31)) << 2) | ((p >> 5) & Int32(3))

    @cute.jit
    def vector_top(self, tops_s, tops_address, v):
        return Uint32(tops_s[v])

    @cute.kernel
    def kernel(self, args):
        logits, tops, row_base, row_ends, out = args
        K = self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        whist = smem.allocate_tensor(Uint32, cute.make_layout((LOW, SELECT_WARPS)), 16)
        wabove = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        wpre = smem.allocate_tensor(Uint32, cute.make_layout((2, SELECT_WARPS)), 16)
        pub = smem.allocate_tensor(Uint32, cute.make_layout(LOW + 1), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        fine_hist = smem.allocate_tensor(Uint32, cute.make_layout(FINE), 16)
        cand_index = smem.allocate_tensor(Int32, cute.make_layout(CANDIDATES), 16)
        hot_list = smem.allocate_tensor(Int32, cute.make_layout(HOT), 16)
        cand_key = smem.allocate_tensor(Uint32, cute.make_layout(CANDIDATES), 16)
        tops_s = smem.allocate_tensor(cutlass.Uint16, cute.make_layout(SEL_TOPS), 16)
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        whist[lane, warp] = Uint32(0)
        if tid < FINE:
            fine_hist[tid] = Uint32(0)
        cute.arch.sync_threads()
        self.pdl()
        length = row_ends[row]
        emit_args = out, out, Int32(0), Int32(0), row, Int32(0)
        if length <= K:
            n = cutlass.max(length, Int32(0))
            for e in cutlass.range(tid, K, SELECT_THREADS, unroll=1):
                if e < n:
                    out[row, e] = e
                else:
                    out[row, e] = Int32(-1)
        else:
            # positions (MMA-row order) of whole tiles; padding keys hold 0xFFFF
            length = (length + TILE - 1) // TILE * TILE
            address = logits.iterator.toint() + row_base[row] * 2
            vectors = (length + 7) // 8
            dense = vectors < 2 * K
            run = (vectors + SELECT_THREADS - 1) // SELECT_THREADS
            run = run + 1 - run % 2
            l0 = cutlass.min(tid * run, vectors)
            l1 = cutlass.min(l0 + run, vectors)
            # Pass 1: histogram of the scorer's vector maxima, kept in shared
            # memory (dense rows: of the keys, maxima computed here).
            step = SELECT_THREADS * SEL_UNROLL
            for base in cutlass.range(0, vectors, step, unroll=1):
                batch = []
                for u in cutlass.range_constexpr(SEL_UNROLL):
                    v = base + u * SELECT_THREADS + tid
                    words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if v < vectors:
                        words = self.vector(address, v)
                    batch.append(words)
                for u in cutlass.range_constexpr(SEL_UNROLL):
                    v = base + u * SELECT_THREADS + tid
                    if v < vectors:
                        keys = vector_keys(batch[u], v, length)
                        top = vector_max(keys)
                        tops_s[v] = cutlass.Uint16(top)
                        if dense:
                            for e in cutlass.range_constexpr(8):
                                if v * 8 + e < length:
                                    shared_add(hist.iterator + (half(keys, e) >> Uint32(5)), 1)
                        else:
                            shared_add(hist.iterator + (top >> Uint32(5)), 1)
            cute.arch.sync_threads()
            topdown(hist, BINS, Uint32(K), info, 2, scan, tid, warp, lane)
            b0 = Uint32(info[2])
            low = b0 << Uint32(5)
            if not dense:
                for v in cutlass.range(l0, l1, unroll=1):
                    top = Uint32(tops_s[v])
                    if (top >> Uint32(5)) == b0:
                        shared_add(fine_hist.iterator + (top & Uint32(31)), 1)
                cute.arch.sync_threads()
                if warp == 0:
                    need = Uint32(K) - Uint32(info[3])
                    digit, _ = low_digit(fine_hist[lane], need, lane)
                    if lane == 0:
                        info[4] = Int32(low | digit)
                cute.arch.sync_threads()
                if tid < LOW:
                    fine_hist[tid] = Uint32(0)
                cute.arch.sync_threads()
                low = Uint32(info[4])
            count = Uint32(0)
            for v in cutlass.range(l0, l1, unroll=1):
                if Uint32(tops_s[v]) >= low:
                    count += Uint32(1)
            slot, hot = block_exclusive(count, scan, warp, lane)
            if hot <= Uint32(HOT):
                for v in cutlass.range(l0, l1, unroll=1):
                    if Uint32(tops_s[v]) >= low:
                        hot_list[slot] = v
                        slot += Uint32(1)
            cute.arch.sync_threads()
            listed = Uint32(CANDIDATES + 1)
            if hot <= Uint32(HOT):
                per = (hot + SELECT_THREADS - 1) // SELECT_THREADS
                h0 = cutlass.min(Uint32(tid) * per, hot)
                h1 = cutlass.min(h0 + per, hot)
                batch = []
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    v = Int32(0)
                    if h0 + u < h1:
                        v = hot_list[h0 + u]
                        words = self.vector(address, v)
                    batch.append((v, words))
                count = Uint32(0)
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    v, words = batch[u]
                    if h0 + u < h1:
                        keys = vector_keys(words, v, length)
                        count += self.count_at_least(keys, v, length, low)
                slot, listed = block_exclusive(count, scan, warp, lane)
                if listed <= Uint32(CANDIDATES):
                    for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                        v, words = batch[u]
                        if h0 + u < h1:
                            keys = vector_keys(words, v, length)
                            for e in cutlass.range_constexpr(8):
                                key = half(keys, e)
                                if key >= low and v * 8 + e < length:
                                    cand_index[slot] = v * 8 + e
                                    cand_key[slot] = key
                                    fine = fine_bin_of(key, b0)
                                    shared_add(fine_hist.iterator + fine, 1)
                                    slot += Uint32(1)
            if tid == 0:
                info[2] = Int32(FINE - 1)
                info[3] = Int32(0)
            cute.arch.sync_threads()
            topdown(fine_hist, FINE, Uint32(K), info, 2, scan, tid, warp, lane)
            fine_bin = Uint32(info[2])
            above_fine = Uint32(info[3])
            sm = hist, whist, wabove, wpre, pub, scan, info
            if listed > Uint32(CANDIDATES) or fine_bin == Uint32(FINE - 1):
                self.exact(sm, address, length, emit_args, tid, warp, lane)
            else:
                share = (listed + SELECT_WARPS - 1) // SELECT_WARPS
                first = cutlass.min(warp * share, listed)
                last = cutlass.min(first + share, listed)
                above = Uint32(0)
                for i in cutlass.range(first + lane, last, 32, unroll=1):
                    key = cand_key[i]
                    fine = fine_bin_of(key, b0)
                    if fine > fine_bin:
                        above += Uint32(1)
                    elif fine == fine_bin:
                        shared_add(whist.iterator + warp * LOW + (key & Uint32(31)), 1)
                for i in cutlass.range_constexpr(5):
                    above += cute.arch.shuffle_sync_bfly(above, 1 << i)
                if lane == 0:
                    wabove[warp] = above
                cute.arch.sync_threads()
                self.publish(whist, wabove, pub, tid)
                self.resolve(pub, info, whist, wabove, wpre, above_fine, warp, lane)
                threshold = ((b0 + fine_bin) << Uint32(5)) | Uint32(info[4])
                take = Uint32(info[5])
                carry_a = wpre[0, warp]
                carry_t = wpre[1, warp]
                for start in cutlass.range(first, last, 32, unroll=1):
                    i = start + lane
                    key = Uint32(0)
                    index = Int32(0)
                    n = Uint32(0)
                    if i < last:
                        key = cand_key[i]
                        index = cand_index[i]
                        n = Uint32(1)
                    carry_a, carry_t = self.write_selected(
                        key, index, n, threshold, take, carry_a, carry_t, lane, emit_args
                    )


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------
def plan(row_counts, row_lens_last, key_bases, device):
    """Work items and the logits scratch layout of the step's prefill requests.

    Request r has ``row_counts[r]`` query rows; its last row sees
    ``row_lens_last[r]`` keys (row i sees row_lens_last[r] - row_counts[r] + 1 + i),
    its keys start at ``key_bases[r]`` (128-aligned) of the key tensor.

    Returns (items Int32 [n, 4] longest first, row_len Int32 [T], row_base
    Int64 [T], scratch elements): a request's rows are rows of
    round_up(row_lens_last, 128) logits in the scratch.
    """
    items, lens, bases = [], [], []
    token = 0
    scratch = 0
    for rows, last, kb in zip(row_counts, row_lens_last, key_bases):
        assert kb % TILE == 0, "request key bases must be 128-aligned"
        first = last - rows + 1
        pitch = (last + TILE - 1) // TILE * TILE
        lens.extend(range(first, last + 1))
        bases.extend(scratch + i * pitch for i in range(rows))
        scratch += rows * pitch
        for o in range(0, rows, BQ):
            n = min(BQ, rows - o)
            hi = first + o + n - 1
            items.append((token + o, n, kb, (hi + TILE - 1) // TILE))
        token += rows
    items.sort(key=lambda it: -it[3])
    it = torch.tensor(items, dtype=torch.int32, device=device).view(-1, 4)
    ln = torch.tensor(lens, dtype=torch.int32, device=device)
    rb = torch.tensor(bases, dtype=torch.int64, device=device)
    return it, ln, rb, scratch


@functools.cache
def _score_kernel(ctas, mode=0):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    t, nk = cute.sym_int(), cute.sym_int()
    args = (
        fake((cute.sym_int(), 64), dtype=cutlass.Uint8),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((nk, 64), dtype=cutlass.Uint8),
        fake((nk,), dtype=cutlass.Int32),
        fake((t, 16), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        fake((cute.sym_int(), 4), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int64),
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        make_fake_stream(),
    )
    return cute.compile(IndexerScore(ctas, mode), *args, options="--enable-tvm-ffi")


@functools.cache
def _select_kernel(topk):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    t = cute.sym_int()
    args = (
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((t,), dtype=cutlass.Int64),
        fake((t,), dtype=cutlass.Int32),
        fake((t, topk), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(PrefillSelect(topk), *args, options="--enable-tvm-ffi")


def score(q, q_scale, keys, key_scale, weights_bf16, row_len, items, row_base,
          logits, tops, ctas=None, mode=0):
    """BF16 logits of every prefill row into ``logits`` (see ``plan``)."""
    dev = q.device
    t = q.shape[0]
    assert keys.shape[0] % TILE == 0
    ctas = ctas or torch.cuda.get_device_properties(dev).multi_processor_count
    _score_kernel(ctas, mode)(
        q.view(t * HEADS, DIM // 2),
        q_scale.reshape(-1),
        keys,
        key_scale.reshape(-1),
        weights_bf16.contiguous().view(torch.int32).view(t, HEADS // 2),
        row_len,
        items,
        row_base,
        logits.view(torch.int16),
        tops.view(torch.int16),
        torch.cuda.current_stream(dev).cuda_stream,
    )


def select(logits, tops, row_base, row_len, out):
    """Exact top-k key indices of every row (key order, -1 padded) into ``out``."""
    _select_kernel(out.shape[1])(
        logits.view(torch.int16),
        tops.view(torch.int16),
        row_base,
        row_len,
        out,
        torch.cuda.current_stream(out.device).cuda_stream,
    )
