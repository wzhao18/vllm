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

HEADS = 32
DIM = 128
TILE = 128  # keys per MMA tile (M)
BQ = 8  # query tokens per work item
N = BQ * HEADS  # MMA N
KV_STAGES = 6
Q_STAGES = 2
ACC_STAGES = 2
EPI_WG = 4  # epilogue warpgroups (2 tokens per thread)
TOK_WG = BQ // EPI_WG  # tokens per epilogue warpgroup
PRODUCER = 4 * EPI_WG
MMA_WARP = PRODUCER + 1
THREADS = 32 * (MMA_WARP + 1)
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
def st_global_u16_off(address, value, off: int):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0+" + str(off) + "], t; }"
    return code, "l,r,~{memory}", (Int64(address), Uint32(value))


@asm(Uint32, pure=True)
def bfe(value, pos: int, length: int):
    return f"bfe.u32 $0, $1, {pos}, {length};", "=r,r", (Uint32(value),)


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
def tmem_ld32(taddr):
    """tcgen05.ld 32x32b.x32: 32 columns of this warp's 32 TMEM lanes (one per thread)."""
    regs = ",".join(f"${i}" for i in range(32))
    code = "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];"
    return code, ",".join(["=f"] * 32) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 32


@asm()
def tmem_ld16(taddr):
    """tcgen05.ld 32x32b.x16: 16 columns of this warp's 32 TMEM lanes (one per thread)."""
    regs = ",".join(f"${i}" for i in range(16))
    code = "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16];"
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 16


@asm()
def tmem_wait_n(*vals):
    """tcgen05.wait::ld, with the loaded registers tied through it."""
    n = len(vals)
    cons = ",".join(["=f"] * n) + "," + ",".join(str(i) for i in range(n))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * n


@asm()
def tmem_wait32(*vals):
    """tcgen05.wait::ld, with the loaded registers tied through it (their uses cannot
    move above the wait)."""
    cons = ",".join(["=f"] * 32) + "," + ",".join(str(i) for i in range(32))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * 32


@asm(Uint32, pure=True)
def prmt(a, b, sel: int):
    return f"prmt.b32 $0, $1, $2, {sel:#x};", "=r,r,r", (Uint32(a), Uint32(b))


@asm()
def fence_acq_rel_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


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


PUBLISH = 6  # IndexerScore mode: keep only the keys at or above a row threshold
PUBLISH_STAGGER = 7  # PUBLISH, odd warpgroups read the accumulator stages in reverse
SLOTS = 96  # PUBLISH keys kept per (row, key lane): more and the row is recomputed
REGION = (SLOTS + 1) * TILE  # words per row: lane counts, then SLOTS words per lane
CAP = SLOTS * TILE


@asm(Uint32, pure=True)
def ge_bf16x2(r, thr):
    """bit 0: low BF16 of r >= low of thr, bit 1: high halves (NaN: false)."""
    code = (
        "{ .reg .pred p, q; .reg .u32 a, b; setp.ge.bf16x2 p|q, $1, $2; "
        "selp.u32 a, 1, 0, p; selp.u32 b, 2, 0, q; or.b32 $0, a, b; }"
    )
    return code, "=r,r,r", (Uint32(r), Uint32(thr))


def _publish_code(half: str, sel: str) -> str:
    return (
        "{ .reg .pred a, b, c; .reg .b32 v; setp.ge.bf16x2 a|b, $2, $3; "
        f"setp.lt.and.u64 c, $1, $5, {half}; prmt.b32 v, $2, $4, {sel}; "
        "mov.b64 $0, $1; @c st.global.b32 [$1], v; @c add.u64 $0, $1, 512; }"
    )


@asm(Int64)
def publish_lo(ptr, r, thr, tw, lim):
    """Low BF16 of r >= low of thr (and ptr < lim): store (tw | low) at ptr and
    advance it one 128-word slot."""
    return _publish_code("a", "0x7610"), "=l,l,r,r,r,l,~{memory}", (
        Int64(ptr), Uint32(r), Uint32(thr), Uint32(tw), Int64(lim))


@asm(Int64)
def publish_hi(ptr, r, thr, tw, lim):
    return _publish_code("b", "0x7632"), "=l,l,r,r,r,l,~{memory}", (
        Int64(ptr), Uint32(r), Uint32(thr), Uint32(tw), Int64(lim))


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
        self, ctas: int, mode: int = 0, topk: int = 2048,
    ):
        """mode (bench only): 0 full; 2 MMA + accumulator reads, no reduction;
        3 no logits stores; 5 no accumulator reads (reduction of stale registers)."""
        self.mode = mode
        self.ctas = ctas
        self.topk = topk
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
        row_base: cute.Tensor,  # Int64 [T + 1]: row t's first element in logits (T: dummy row)
        logits: cute.Tensor,  # Int16 [*]: BF16 logits scratch
        row_bin: cute.Tensor,  # Int32 [2 * T]: per row the bound rank and the count above it
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
            row_bin,
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
        row_bin: cute.Tensor,
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
            tmem_hold: Int32
            row_end: cute.struct.MemRange[Int32, Q_STAGES * BQ]
            w: cute.struct.Align[cute.struct.MemRange[Uint32, Q_STAGES * BQ * 16], 16]
            hist: cute.struct.Align[cute.struct.MemRange[Uint32, BQ * S_BINS], 16]
            wscan: cute.struct.Align[cute.struct.MemRange[Uint32, EPI_WG * 4], 16]

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
                cute.arch.mbarrier_init(acc_empty + i, 4 * EPI_WG)
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
                # an even tile count (the epilogue takes the two accumulator stages
                # per step): an odd count repeats the newest tile, fully masked
                ntp = ntiles + (ntiles & 1)
                for tt in cutlass.range(ntp, unroll=1):
                    t = cutlass.min(ntp - 1 - tt, ntiles - 1)
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
                for _t in cutlass.range(ntiles + (ntiles & 1), unroll=1):
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
                row_bin, (end_a, w_a, st.hist.data_ptr().toint(), st.wscan.data_ptr().toint()),
                warp, tidx, bid,
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
        row_bin, sm, warp, tidx, bid, lane,
    ):
        """Every epilogue warp reads every tile: warpgroup wg owns tokens 2 wg and
        2 wg + 1 of the item, its warp w % 4 the TMEM lane quarter (32 keys). The
        head weights of the two tokens stay in registers for the whole item. Tiles
        come in pairs (accumulator stages 0 and 1; items have an even tile count),
        each read with one 32-head load per token and released at once. Each thread
        forms the BF16 logits of its key for both tokens at stock precision, stores
        them straight to the row-major scratch (row t at row_base[t], MMA-row order;
        tokens past the item's end write a dummy row) and counts them in the row's
        histogram of the top 11 bits. At the end of the item the warpgroup finds,
        per row, the rank (in value order) of the bin holding the topk-th largest
        logit and the count above it (``row_bin``), for the select."""
        end_a, w_a, hist_a, wscan_a = sm
        wg = warp // 4
        lt = tidx % 128
        wq = lt // 32
        out_a = logits.iterator.toint()
        ph = Int32(0)
        jq = Int32(0)
        n_items = items.shape[0]
        n_rows = row_base.shape[0] - 1
        key_in_tile = 4 * (lt % 32) + lt // 32
        tok_a = 2 * wg
        # this warp's TMEM lane quarter, column 0 of accumulator stage 0
        tmem_q = Int32(tAcc.iterator.toint()) + ((wq * 32) << 16) + tok_a * HEADS
        hist0 = hist_a + tok_a * (S_BINS * 4)
        hist1 = hist0 + S_BINS * 4
        # zero the histograms once (re-zeroed after every item)
        for i in cutlass.range_constexpr(2 * S_BINS // 128):
            st_shared_u32(hist0 + (i * 128 + lt) * 4, Uint32(0))
        wg_sync(wg)
        for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
            token0 = items[item, 0]
            ntok = items[item, 1]
            ntiles = items[item, 3]
            qs = jq % Q_STAGES
            cute.arch.mbarrier_wait(q_full + qs, (jq // Q_STAGES) & 1)
            end0 = Int32(ld_shared_u32(end_a + (qs * BQ + tok_a) * 4))
            end1 = Int32(ld_shared_u32(end_a + (qs * BQ + tok_a + 1) * 4))
            row0 = token0 + tok_a
            row1 = token0 + tok_a + 1
            if tok_a >= ntok:
                row0 = n_rows
            if tok_a + 1 >= ntok:
                row1 = n_rows
            ntp = ntiles + (ntiles & 1)
            # running pointers: position lt of the item's newest tile pair's first tile
            first = Int64((ntp - 1) * TILE + lt) * 2
            ptr0 = out_a + row_base[row0] * 2 + first
            ptr1 = out_a + row_base[row1] * 2 + first
            if cutlass.const_expr(self.mode >= PUBLISH):
                # BF16 thresholds of both rows (row_bin[3 r], the dummy row's is NaN)
                # and their candidate lists (row_base in 16-bit units)
                thr = (Uint32(row_bin[3 * row1]) << Uint32(16)) | (
                    Uint32(row_bin[3 * row0]) & Uint32(0xFFFF)
                )
                # this thread's column of each row's candidate region: word 0
                # the count, then one word per kept key (tile << 16 | BF16)
                cand0 = out_a + row_base[row0] * 2 + lt * 4
                cand1 = out_a + row_base[row1] * 2 + lt * 4
                p0 = cand0 + TILE * 4
                p1 = cand1 + TILE * 4
                lim0 = cand0 + REGION * 4
                lim1 = cand1 + REGION * 4
            # tile pairs whose newer tile reaches past a row end (causal edge)
            full_tiles = cutlass.min(end0, end1) // TILE
            # head weights (BF16x2) of both tokens: w[j][i] = heads 2i, 2i + 1
            wbase = w_a + (qs * BQ * 16 + tok_a * 16) * 4
            w = []
            for j in cutlass.range_constexpr(2):
                wj = []
                for i in cutlass.range_constexpr(4):
                    v = lds_v4(wbase + (j * 16 + 4 * i) * 4)
                    wj.extend(v)
                w.append(wj)
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(q_empty + qs)
            for pp in cutlass.range(ntp // 2, unroll=1):
                t_hi = ntp - 1 - 2 * pp
                for st0 in cutlass.range_constexpr(ACC_STAGES):
                    st = st0
                    if cutlass.const_expr(self.mode == PUBLISH_STAGGER):
                        # odd warpgroups take the pair's stages in the other order,
                        # so half the warps read TMEM while the others reduce
                        st = Int32(st0) ^ (wg & 1)
                    t = t_hi - st
                    cute.arch.mbarrier_wait(acc_full + st, ph)
                    col = tmem_q + st * N
                    fa = tmem_ld32(col)
                    fb = tmem_ld32(col + HEADS)
                    fa = tmem_wait32(*fa)
                    fb = tmem_wait32(*fb)
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(acc_empty + st)
                    r = self.reduce2((fa, fb), w[0], w[1])
                    if cutlass.const_expr(self.mode >= PUBLISH):
                        kidx = t * TILE + key_in_tile
                        if t_hi >= full_tiles:
                            # causal edge, and the repeated tile of an odd count
                            if kidx >= end0 or t >= ntiles:
                                r = r | Uint32(0xFFFF)
                            if kidx >= end1 or t >= ntiles:
                                r = r | Uint32(0xFFFF0000)
                        tw = Uint32(t) << Uint32(16)
                        p0 = publish_lo(p0, r, thr, tw, lim0)
                        p1 = publish_hi(p1, r, thr, tw, lim1)
                    elif t_hi >= full_tiles:
                        # causal edge (and the repeated tile of an odd count)
                        kidx = t * TILE + key_in_tile
                        if kidx >= end0:
                            r = r | Uint32(0xFFFF)
                        if kidx >= end1:
                            r = r | Uint32(0xFFFF0000)
                        if t < ntiles:
                            st_global_u16(ptr0 - st * TILE * 2, r)
                            st_global_u16(ptr1 - st * TILE * 2, r >> Uint32(16))
                    else:
                        st_global_u16_off(ptr0, r, -st * TILE * 2)
                        st_global_u16_off(ptr1, r >> Uint32(16), -st * TILE * 2)
                    if cutlass.const_expr(self.mode < PUBLISH):
                        # histogram of the raw top 11 bits (masked keys land in bin
                        # 2047, the lowest rank)
                        red_shared(hist0 + (bfe(r, 5, 11) << Uint32(2)), 1)
                        red_shared(hist1 + (r >> Uint32(21) << Uint32(2)), 1)
                ptr0 = ptr0 - 2 * TILE * 2
                ptr1 = ptr1 - 2 * TILE * 2
                ph = ph ^ 1
            if cutlass.const_expr(self.mode >= PUBLISH):
                # this thread's candidate counts of both rows
                # (a full column counts as overflowed: the row is recomputed)
                st_global_u32(cand0, Uint32((p0 - cand0) // (TILE * 4) - 1))
                st_global_u32(cand1, Uint32((p1 - cand1) // (TILE * 4) - 1))
            else:
                # per row: bound bin rank and the count above it, then clear
                wg_sync(wg)
                self.row_bound(hist0, end0, row0, n_rows, row_bin, wscan_a + wg * 16, wg, lt, wq, lane)
                self.row_bound(hist1, end1, row1, n_rows, row_bin, wscan_a + wg * 16, wg, lt, wq, lane)
                for i in cutlass.range_constexpr(2 * S_BINS // 128):
                    st_shared_u32(hist0 + (i * 128 + lt) * 4, Uint32(0))
                wg_sync(wg)
            jq += 1

    @cute.jit
    def reduce2(self, buf, w0, w1):
        """BF16 logits of two tokens (packed: token 0 low) at stock precision: per
        token, s0 takes head pairs (4i, 4i + 1), s1 (4i + 2, 4i + 3), each head term
        relu-rounded to BF16 and BF16x2-FMA'd with its weight; then
        (s0.lo + s1.lo) + (s0.hi + s1.hi)."""
        ss = [[Uint32(0), Uint32(0)] for _ in range(2)]
        ws = [w0, w1]
        for h in cutlass.range_constexpr(0, HEADS, 4):
            for j in cutlass.range_constexpr(2):
                fr = buf[j]
                ss[j][0] = fma_bf16x2(relu_bf16x2(fr[h], fr[h + 1]), ws[j][h // 2], ss[j][0])
                ss[j][1] = fma_bf16x2(
                    relu_bf16x2(fr[h + 2], fr[h + 3]), ws[j][h // 2 + 1], ss[j][1]
                )
        p0 = add_bf16x2(ss[0][0], ss[0][1])
        p1 = add_bf16x2(ss[1][0], ss[1][1])
        return add_bf16x2(prmt(p0, p1, 0x5410), prmt(p0, p1, 0x7632))

    @cute.jit
    def row_bound(self, hist, length, row, n_rows, row_bin, scan_a, wg, lt, wq, lane):
        """row_bin[row] = (rank r0 of the bin holding the topk-th largest logit, the
        count in the bins of higher rank); rank = order-preserving key of the bin
        (positive bin b: b + 1024, negative bin b: b ^ 2047)."""
        per = S_BINS // 128
        r_top = S_BINS - 1 - per * lt
        counts = []
        mine = Uint32(0)
        for i in cutlass.range_constexpr(per):
            rk = r_top - i
            b = rk - S_BINS // 2
            if rk < S_BINS // 2:
                b = rk ^ (S_BINS - 1)
            c = ld_shared_u32(hist + b * 4)
            counts.append(c)
            mine += c
        inclusive = warp_inclusive(mine, lane)
        if lane == 31:
            st_shared_u32(scan_a + wq * 4, inclusive)
        wg_sync(wg)
        prior = Uint32(0)
        for i in cutlass.range_constexpr(4):
            x = ld_shared_u32(scan_a + i * 4)
            if i < wq:
                prior += x
        wg_sync(wg)
        above = prior + inclusive - mine
        need = Uint32(self.topk)
        if row < n_rows and length > self.topk:
            if above < need and above + mine >= need:
                running = above
                found = Int32(0)
                for i in cutlass.range_constexpr(per):
                    if found == 0 and running + counts[i] >= need:
                        row_bin[3 * row] = r_top - i
                        row_bin[3 * row + 1] = Int32(running)
                        row_bin[3 * row + 2] = Int32(0)
                        found = Int32(1)
                    running += counts[i]


# --------------------------------------------------------------------------
# Exact top-k over the logits scratch (row groups, persistent, one streaming read)
# --------------------------------------------------------------------------
SG = 256  # threads per row group (one row at a time)
SGW = SG // 32
GROUPS = 4  # row groups per CTA (independent rows overlap their phases)
S_UNROLL = 8  # 16-byte logits loads in flight per thread
S_BINS = 2048  # bins of the scorer's histogram: the top 11 bits
FINE_BITS = 5  # the key bits below a bin
FMASK = (1 << FINE_BITS) - 1
SEG = 131072  # positions per bitmap segment (rows longer than this take two reads)
SEG_WORDS = SEG // 32
S_CAND = 4096  # keys of the bound bin kept from the streaming read
S_HOTV = 2 * SEG_WORDS  # vectors whose maximum reaches the bound bin (u16 ids; aliases the tie bitmap)
S_DENSE = 4  # hot vectors loaded per thread and round of the dense pass
# per-group shared memory (bytes)
S_OFF_SEL = 0
S_OFF_TIE = S_OFF_SEL + 4 * SEG_WORDS
S_OFF_CAND = S_OFF_TIE + 4 * SEG_WORDS
S_OFF_FINE = S_OFF_CAND + 4 * S_CAND
S_OFF_SCAN = S_OFF_FINE + 4 * 32 * SGW
S_OFF_INFO = S_OFF_SCAN + 4 * SGW
S_GROUP_BYTES = (S_OFF_INFO + 32 + 127) // 128 * 128


@asm()
def red_shared(address, value):
    return "red.shared.add.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def red_or_shared(address, value):
    return "red.shared.or.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def ldg_v4(address):
    code = "ld.global.nc.L1::no_allocate.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (Int64(address),), (Uint32,) * 4


@asm()
def st_global_s32(address, value):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Uint32)
def ballot(flag):
    code = "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }"
    return code, "=r,r", (Uint32(flag),)


@asm(Uint32, pure=True)
def max_bf16x2(a, b):
    return "max.bf16x2 $0, $1, $2;", "=r,r,r", (Uint32(a), Uint32(b))


@asm(Uint32, pure=True)
def max_u16x2(a, b):
    return "max.u16x2 $0, $1, $2;", "=r,r,r", (Uint32(a), Uint32(b))


@asm(Float32, pure=True)
def float_of_bits(b):
    return "mov.b32 $0, $1;", "=f,r", (Uint32(b),)


@cute.jit
def keys16x2(w: Uint32) -> Uint32:
    """Order-preserving 16-bit keys of two packed BF16 (a larger value, a larger key);
    the key's top 11 bits are the rank of the scorer's histogram bin."""
    flip = Uint32(0x80008000) | (((w >> Uint32(15)) & Uint32(0x00010001)) * Uint32(0x7FFF))
    return w ^ flip


@cute.jit
def warp_inclusive(value: Uint32, lane: Int32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            value += up
    return value


@cute.jit
def gsync(g):
    cute.arch.barrier(barrier_id=1 + g, number_of_threads=SG)


@cute.jit
def g_exclusive(value: Uint32, scan_a, g, gw, lane):
    """(exclusive prefix over the group's threads in order, group total)."""
    inclusive = warp_inclusive(value, lane)
    if lane == 31:
        st_shared_u32(scan_a + gw * 4, inclusive)
    gsync(g)
    prior = Uint32(0)
    total = Uint32(0)
    for i in cutlass.range_constexpr(SGW):
        x = ld_shared_u32(scan_a + i * 4)
        if i < gw:
            prior += x
        total += x
    gsync(g)
    return prior + inclusive - value, total


@cute.jit
def low_digit(count: Uint32, need: Uint32, lane: Int32):
    """Warp-wide over 32 bins (lane b holds bin b): (the bin where the counts summed
    from bin 31 down reach ``need``, the count in the bins above it)."""
    rev = cute.arch.shuffle_sync(count, 31 - lane)
    above = warp_inclusive(rev, lane) - rev
    hit = Int32(0)
    if above < need and above + rev >= need:
        hit = lane
    for i in cutlass.range_constexpr(5):
        hit += cute.arch.shuffle_sync_bfly(hit, 1 << i)
    return Uint32(31) - Uint32(hit), cute.arch.shuffle_sync(above, hit)


@cute.jit
def key_index(p: Int32) -> Int32:
    """Logits position (rows of a tile in SF-atom order) -> key index."""
    return (p & Int32(-TILE)) | ((p & Int32(31)) << 2) | ((p >> 5) & Int32(3))


@cute.jit
def key_of(words, e: int) -> Uint32:
    return (keys16x2(words[e // 2]) >> Uint32(16 * (e % 2))) & Uint32(0xFFFF)


CAND = 4096  # bound-bin keys kept per row (more: the finish kernel's exact path)
CHUNK_V = 2048  # 16-byte vectors per mark work unit (16384 positions)
MARK_THREADS = 256
MARK_LOADS = CHUNK_V // MARK_THREADS  # loads in flight per thread (one round per unit)
MARK_BUF = 2048  # bound-bin keys of one mark chunk kept in shared memory
FIN_THREADS = 128  # four rows per finish CTA, one warp each
TIE_BUF = 512  # threshold ties ranked in shared memory per row
FIN_BATCH = 8  # global loads in flight per lane in the finish kernel


@asm(Uint32)
def atom_add_global(address, value):
    code = "atom.global.add.u32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def red_or_global(address, value):
    return "red.global.or.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def ldg_cg_v4(address):
    code = "ld.global.cg.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (Int64(address),), (Uint32,) * 4


@asm(Uint32)
def ld_global_cg(address):
    return "ld.global.cg.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@cute.jit
def vector_masks(wd, v, vectors, low, high, low_f, positive):
    """(8-bit mask of the vector's keys above the bound bin, count of its keys in
    the bound bin); a BF16-maxima prefilter skips the vectors below the bin."""
    hot = Uint32(0)
    if positive:
        m = max_bf16x2(max_bf16x2(wd[0], wd[1]), max_bf16x2(wd[2], wd[3]))
        m = max_bf16x2(m, prmt(m, m, 0x1032))
        if float_of_bits(m << Uint32(16)) >= low_f:
            hot = Uint32(1)
    else:
        hot = Uint32(1)
    sel = Uint32(0)
    nb = Uint32(0)
    if hot != Uint32(0) and v < vectors:
        for e in cutlass.range_constexpr(8):
            key = key_of(wd, e)
            if key >= high:
                sel = sel | Uint32(1 << e)
            elif key >= low:
                nb += Uint32(1)
    return sel, nb


class PrefillMark:
    """Select, part 1 (streaming, fully parallel): per (row, 8192-position chunk),
    writes the row's position bitmap words (bit set: key above the bound bin r0 from
    the scorer's ``row_bin``) and appends the bound bin's keys (position << 5 | low
    5 key bits) to the row's candidate list (``cand``, count in row_bin[3 r + 2],
    zeroed by the scorer). Rows with at most ``topk`` keys are skipped."""

    def __init__(self, ctas: int, topk: int, words: int):
        self.ctas = ctas
        self.topk = topk
        self.words = words  # bitmap words per row (stride)

    @cute.jit
    def __call__(self, logits, row_base, row_len, row_bin, bitmap, cand, s):
        self.kernel(logits, row_base, row_len, row_bin, bitmap, cand).launch(
            grid=(self.ctas, 1, 1), block=(MARK_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, logits, row_base, row_len, row_bin, bitmap, cand):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        smem = utils.SmemAllocator()
        mbuf = smem.allocate_tensor(Uint32, cute.make_layout(MARK_BUF + 4), 16)
        cnt_a = mbuf.iterator.toint()
        buf_a = cnt_a + 16
        if tid == 0:
            st_shared_u32(cnt_a, Uint32(0))
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        n_rows = row_len.shape[0]
        cpr = (self.words * 32 + CHUNK_V * 8 - 1) // (CHUNK_V * 8)
        logits_a = logits.iterator.toint()
        bm_a = bitmap.iterator.toint()
        cand_a = cand.iterator.toint()
        bin_a = row_bin.iterator.toint()
        for u in cutlass.range(bid, n_rows * cpr, self.ctas, unroll=1):
            row = u // cpr
            chunk = u - row * cpr
            length = row_len[row]
            vectors = (length + TILE - 1) // TILE * (TILE // 8)
            v0 = chunk * CHUNK_V
            if length > self.topk and v0 < vectors:
                address = logits_a + row_base[row] * 2
                low = Uint32(row_bin[3 * row]) << Uint32(5)
                high = low + Uint32(32)
                positive = low >= Uint32(0x8000)
                low_f = float_of_bits((low ^ Uint32(0x8000)) << Uint32(16))
                for i in cutlass.range_constexpr(1):
                    # the unit's vectors all in flight at once
                    batch = []
                    for k in cutlass.range_constexpr(MARK_LOADS):
                        v = v0 + k * MARK_THREADS + tid
                        words4 = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if v < vectors:
                            words4 = ldg_v4(address + Int64(v) * 16)
                        batch.append(words4)
                    for k in cutlass.range_constexpr(MARK_LOADS):
                        v = v0 + k * MARK_THREADS + tid
                        sel, nb = vector_masks(batch[k], v, vectors, low, high, low_f, positive)
                        # bitmap word of four consecutive vectors (lanes 4j .. 4j + 3)
                        wbits = sel << Uint32((lane % 4) * 8)
                        wbits = wbits | cute.arch.shuffle_sync_bfly(wbits, 1)
                        wbits = wbits | cute.arch.shuffle_sync_bfly(wbits, 2)
                        if lane % 4 == 0 and v < vectors:
                            st_global_u32(bm_a + (Int64(row) * self.words + (v >> 2)) * 4, wbits)
                        # bound-bin keys into the CTA's shared buffer (one slot claim
                        # per warp)
                        if cute.arch.vote_any_sync(nb > Uint32(0)):
                            incl = warp_inclusive(nb, lane)
                            first = Uint32(0)
                            if lane == 31:
                                first = atom_add_shared(cnt_a, incl)
                            slot = cute.arch.shuffle_sync(first, 31) + incl - nb
                            if nb > Uint32(0):
                                for e in cutlass.range_constexpr(8):
                                    key = key_of(batch[k], e)
                                    if key >= low and key < high:
                                        if slot < Uint32(MARK_BUF):
                                            st_shared_u32(
                                                buf_a + slot * 4,
                                                (Uint32(v * 8 + e) << Uint32(5)) | (key & Uint32(31)),
                                            )
                                        slot += Uint32(1)
                cute.arch.sync_threads()
                # one claim in the row's candidate list for the chunk's keys
                n = ld_shared_u32(cnt_a)
                if tid == 0:
                    first = Uint32(0)
                    if n > Uint32(0):
                        first = atom_add_global(bin_a + (Int64(row) * 3 + 2) * 4, n)
                        if n > Uint32(MARK_BUF):
                            # the chunk overflowed its buffer: force the exact path
                            atom_add_global(bin_a + (Int64(row) * 3 + 2) * 4, Uint32(CAND + 1))
                    st_shared_u32(cnt_a + 4, first)
                cute.arch.sync_threads()
                first = ld_shared_u32(cnt_a + 4)
                for c in cutlass.range(tid, Int32(cutlass.min(n, Uint32(MARK_BUF))), MARK_THREADS, unroll=1):
                    if first + Uint32(c) < Uint32(CAND):
                        st_global_u32(
                            cand_a + (Int64(row) * CAND + Int64(first) + Int64(c)) * 4,
                            ld_shared_u32(buf_a + c * 4),
                        )
                cute.arch.sync_threads()
                if tid == 0:
                    st_shared_u32(cnt_a, Uint32(0))
                cute.arch.sync_threads()


class PrefillFinish:
    """Select, part 2 (one warp per row): exact threshold's low 5 bits from the
    candidate list, the remaining keys of the bound bin (ties: the lowest
    positions) into the bitmap, then the bitmap's positions in order as key
    indices (-1 pads rows with at most ``topk`` keys). Rows whose candidates
    overflowed re-read their logits (exact, slower)."""

    def __init__(self, ctas: int, topk: int, words: int):
        self.ctas = ctas
        self.topk = topk
        self.words = words

    @cute.jit
    def __call__(self, logits, row_base, row_len, row_bin, bitmap, cand, out, s):
        self.kernel(logits, row_base, row_len, row_bin, bitmap, cand, out).launch(
            grid=(self.ctas, 1, 1), block=(FIN_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, logits, row_base, row_len, row_bin, bitmap, cand, out):
        K = self.topk
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        w = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = utils.SmemAllocator()
        fine = smem.allocate_tensor(Uint32, cute.make_layout(32 * (FIN_THREADS // 32)), 16)
        fine_a = fine.iterator.toint() + w * 128
        tbuf = smem.allocate_tensor(
            Uint32, cute.make_layout((TIE_BUF + 4) * (FIN_THREADS // 32)), 16
        )
        tcnt_a = tbuf.iterator.toint() + w * (TIE_BUF + 4) * 4
        tie_buf_a = tcnt_a + 16
        cute.arch.griddepcontrol_wait()
        n_rows = row_len.shape[0]
        out_a = out.iterator.toint()
        bm_a = bitmap.iterator.toint()
        cand_a = cand.iterator.toint()
        logits_a = logits.iterator.toint()
        rpc = FIN_THREADS // 32
        for row in cutlass.range(bid * rpc + w, n_rows, self.ctas * rpc, unroll=1):
            length = row_len[row]
            orow = out_a + Int64(row) * K * 4
            if length <= K:
                for e in cutlass.range(lane, K, 32, unroll=1):
                    v = Int32(-1)
                    if e < length:
                        v = e
                    st_global_s32(orow + e * 4, v)
            else:
                brow = bm_a + Int64(row) * self.words * 4
                crow = cand_a + Int64(row) * CAND * 4
                address = logits_a + row_base[row] * 2
                low = Uint32(row_bin[3 * row]) << Uint32(5)
                high = low + Uint32(32)
                need = Uint32(K) - Uint32(row_bin[3 * row + 1])
                n_cand = Uint32(row_bin[3 * row + 2])
                positions = (length + TILE - 1) // TILE * TILE
                vectors = positions // 8
                words = positions // 32
                exact = n_cand > Uint32(CAND)
                st_shared_u32(fine_a + lane * 4, Uint32(0))
                cute.arch.sync_warp()
                if exact:
                    for v in cutlass.range(lane, vectors, 32, unroll=1):
                        wd = ldg_v4(address + Int64(v) * 16)
                        for e in cutlass.range_constexpr(8):
                            key = key_of(wd, e)
                            if key >= low and key < high:
                                red_shared(fine_a + (key & Uint32(31)) * 4, 1)
                else:
                    for c0 in cutlass.range(lane, Int32(n_cand), 32 * FIN_BATCH, unroll=1):
                        xs = []
                        for k in cutlass.range_constexpr(FIN_BATCH):
                            x = Uint32(0)
                            if c0 + 32 * k < Int32(n_cand):
                                x = ld_global_u32(crow + (c0 + 32 * k) * 4)
                            xs.append(x)
                        for k in cutlass.range_constexpr(FIN_BATCH):
                            if c0 + 32 * k < Int32(n_cand):
                                red_shared(fine_a + (xs[k] & Uint32(31)) * 4, 1)
                cute.arch.sync_warp()
                count = ld_shared_u32(fine_a + lane * 4)
                d, above_d = low_digit(count, need, lane)
                ties = cute.arch.shuffle_sync(count, Int32(d))
                take = need - above_d
                all_ties = ties == take
                if exact:
                    # bound-bin keys above the threshold (and all ties) into the bitmap
                    for v in cutlass.range(lane, vectors, 32, unroll=1):
                        wd = ldg_v4(address + Int64(v) * 16)
                        mk = Uint32(0)
                        for e in cutlass.range_constexpr(8):
                            key = key_of(wd, e)
                            lo = key & Uint32(31)
                            if key >= low and key < high and (lo > d or (lo == d and all_ties)):
                                mk = mk | Uint32(1 << e)
                        if mk != Uint32(0):
                            red_or_global(brow + (v >> 2) * 4, mk << Uint32((v & 3) * 8))
                    if not all_ties:
                        # ties in position order: the warp walks the row's vectors in order
                        left = take
                        for v0 in cutlass.range(0, vectors, 32, unroll=1):
                            v = v0 + lane
                            mk = Uint32(0)
                            if v < vectors:
                                wd = ldg_v4(address + Int64(v) * 16)
                                for e in cutlass.range_constexpr(8):
                                    key = key_of(wd, e)
                                    if key >= low and key < high and (key & Uint32(31)) == d:
                                        mk = mk | Uint32(1 << e)
                            n = popc(mk)
                            before = warp_inclusive(n, lane) - n
                            keep = Uint32(0)
                            if left > before:
                                keep = cutlass.min(left - before, n)
                            while n > keep:
                                mk = mk ^ (Uint32(0x80000000) >> clz(mk))
                                n -= Uint32(1)
                            if mk != Uint32(0):
                                red_or_global(brow + (v >> 2) * 4, mk << Uint32((v & 3) * 8))
                            total = cute.arch.shuffle_sync(warp_inclusive(popc(mk), lane), 31)
                            left = left - cutlass.min(left, total)
                else:
                    # keys above the threshold (and all ties when all are taken) into the
                    # bitmap; otherwise the ties' positions into a warp buffer
                    st_shared_u32(tcnt_a, Uint32(0))
                    cute.arch.sync_warp()
                    for c0 in cutlass.range(lane, Int32(n_cand), 32 * FIN_BATCH, unroll=1):
                      xs = []
                      for k in cutlass.range_constexpr(FIN_BATCH):
                        x = Uint32(0xFFFFFFFF)
                        if c0 + 32 * k < Int32(n_cand):
                            x = ld_global_u32(crow + (c0 + 32 * k) * 4)
                        xs.append(x)
                      for k in cutlass.range_constexpr(FIN_BATCH):
                        x = xs[k]
                        lo = x & Uint32(31)
                        p = x >> Uint32(5)
                        if x == Uint32(0xFFFFFFFF):
                            lo = d + Uint32(64)
                        if lo == d + Uint32(64):
                            pass
                        elif lo > d or (lo == d and all_ties):
                            red_or_global(brow + (p >> Uint32(5)) * 4, Uint32(1) << (p & Uint32(31)))
                        elif lo == d:
                            slot = atom_add_shared(tcnt_a, Uint32(1))
                            if slot < Uint32(TIE_BUF):
                                st_shared_u32(tie_buf_a + slot * 4, p)
                    cute.arch.sync_warp()
                    nt = ld_shared_u32(tcnt_a)
                    if not all_ties:
                        # the ``take`` lowest tie positions: rank each tie among the ties
                        for i in cutlass.range(lane, Int32(nt), 32, unroll=1):
                            p = Uint32(0)
                            if nt <= Uint32(TIE_BUF):
                                p = ld_shared_u32(tie_buf_a + i * 4)
                            rank = Uint32(0)
                            if nt <= Uint32(TIE_BUF):
                                for j in cutlass.range(Int32(0), Int32(nt), unroll=1):
                                    if ld_shared_u32(tie_buf_a + j * 4) < p:
                                        rank += Uint32(1)
                            if rank < take and nt <= Uint32(TIE_BUF):
                                red_or_global(
                                    brow + (p >> Uint32(5)) * 4, Uint32(1) << (p & Uint32(31))
                                )
                        if nt > Uint32(TIE_BUF):
                            # many ties: rank them from the candidate list (slow, exact)
                            for c in cutlass.range(lane, Int32(n_cand), 32, unroll=1):
                                x = ld_global_u32(crow + c * 4)
                                p = x >> Uint32(5)
                                if (x & Uint32(31)) == d:
                                    rank = Uint32(0)
                                    for c2 in cutlass.range(Int32(0), Int32(n_cand), unroll=1):
                                        y = ld_global_u32(crow + c2 * 4)
                                        if (y & Uint32(31)) == d and (y >> Uint32(5)) < p:
                                            rank += Uint32(1)
                                    if rank < take:
                                        red_or_global(
                                            brow + (p >> Uint32(5)) * 4,
                                            Uint32(1) << (p & Uint32(31)),
                                        )
                fence_acq_rel_gpu()
                cute.arch.sync_warp()
                # the bitmap's positions in order (each lane a contiguous run of words)
                per = ((words + 31) // 32 + 3) // 4 * 4
                w0 = cutlass.min(lane * per, words)
                w1 = cutlass.min(w0 + per, words)
                cnt = Uint32(0)
                for i0 in cutlass.range(w0, w1, 4 * FIN_BATCH, unroll=1):
                    qs = []
                    for k in cutlass.range_constexpr(FIN_BATCH):
                        q = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if i0 + 4 * k < w1:
                            q = ldg_cg_v4(brow + (i0 + 4 * k) * 4)
                        qs.append(q)
                    for k in cutlass.range_constexpr(FIN_BATCH):
                        for j in cutlass.range_constexpr(4):
                            if i0 + 4 * k + j < w1:
                                cnt += popc(qs[k][j])
                slot = warp_inclusive(cnt, lane) - cnt
                for i0 in cutlass.range(w0, w1, 4 * FIN_BATCH, unroll=1):
                    qs = []
                    for k in cutlass.range_constexpr(FIN_BATCH):
                        q = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if i0 + 4 * k < w1:
                            q = ldg_cg_v4(brow + (i0 + 4 * k) * 4)
                        qs.append(q)
                    for k in cutlass.range_constexpr(FIN_BATCH):
                        for j in cutlass.range_constexpr(4):
                            m = qs[k][j]
                            if i0 + 4 * k + j >= w1:
                                m = Uint32(0)
                            while m != Uint32(0):
                                b = popc((m & (Uint32(0) - m)) - Uint32(1))
                                p = (i0 + 4 * k + j) * 32 + Int32(b)
                                st_global_s32(orow + Int64(slot) * 4, key_index(p))
                                slot += Uint32(1)
                                m = m & (m - Uint32(1))
            cute.arch.sync_warp()


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------
CS_THREADS = 256


@asm(Uint32)
def match_any(value):
    code = "{ .reg .b32 m; activemask.b32 m; match.any.sync.b32 $0, $1, m; }"
    return code, "=r,r", (Uint32(value),)


@asm(Uint32)
def warp_slot(flag, cnt_a):
    """Slot of each flagged active lane in a shared counter (one atomic per warp)."""
    code = (
        "{ .reg .b32 m, b, lt, base, n, t, ldr; .reg .pred p, q; "
        "activemask.b32 m; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 b, p, m; "
        "mov.u32 lt, %lanemask_lt; and.b32 n, m, lt; setp.eq.u32 q, n, 0; "
        "popc.b32 n, b; mov.u32 base, 0; @q atom.shared.add.u32 base, [$2], n; "
        "brev.b32 t, m; bfind.shiftamt.u32 ldr, t; "
        "shfl.sync.idx.b32 base, base, ldr, 0x1f, m; "
        "and.b32 n, b, lt; popc.b32 n, n; add.u32 $0, base, n; }"
    )
    return code, "=r,r,r,~{memory}", (Uint32(flag), Int32(cnt_a))


@cute.jit
def hist_add(hist_a, b):
    """hist[b] += 1, one shared atomic per distinct bin of the active lanes
    (candidate keys crowd into few bins)."""
    same = match_any(b)
    if (same & lanemask_lt()) == Uint32(0):
        red_shared(hist_a + (b << Uint32(2)), popc(same))


@cute.jit
def _top_bin(hist_a, need, lane, info_a):
    """Warp 0: the highest bin b of a 256-bin histogram with count(bins >= b) >=
    need; info = (b, count above b, count of b)."""
    s = Uint32(0)
    for j in cutlass.range_constexpr(8):
        s += ld_shared_u32(hist_a + (255 - 8 * lane - j) * 4)
    incl = warp_inclusive(s, lane)
    hit = ballot(incl >= need)
    first = Int32(popc((hit & (Uint32(0) - hit)) - Uint32(1)))
    if lane == first:
        run = incl - s
        found = Int32(0)
        for j in cutlass.range_constexpr(8):
            c = ld_shared_u32(hist_a + (255 - 8 * lane - j) * 4)
            if found == 0 and run + c >= need:
                st_shared_u32(info_a, Uint32(255 - 8 * lane - j))
                st_shared_u32(info_a + 4, run)
                st_shared_u32(info_a + 8, c)
                found = Int32(1)
            run += c


class CandSelect:
    """Exact top-k of each row from its PUBLISH candidates (one CTA per row). The
    row's region holds, per key lane j (0..127, key 4 (j % 32) + j // 32 of a
    tile), its count and then SLOTS words (tile << 16 | BF16). Two 8-bit radix
    passes over the order-preserving 16-bit keys find the k-th value t; keys
    above t are emitted, ties at t by lowest key index. Rows whose candidates are
    fewer than min(k, row length) or overflowed a lane are flagged in
    row_bin[3 r + 1] (recomputed by the caller). Output order is unspecified."""

    def __init__(self, topk: int):
        self.topk = topk

    @cute.jit
    def __call__(self, cand, row_len, row_bin, out, s):
        self.kernel(cand, row_len, row_bin, out).launch(
            grid=(row_len.shape[0], 1, 1), block=(CS_THREADS, 1, 1), stream=s,
            use_pdl=True,
        )

    @cute.jit
    def decode(self, wd, j):
        """(key index, order16 key) of a candidate word of key lane j."""
        pos = Int32(wd >> Uint32(16)) * TILE + 4 * (j % 32) + j // 32
        return pos, order16(wd & Uint32(0xFFFF))

    @cute.kernel
    def kernel(self, cand, row_len, row_bin, out):
        K = self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        w = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(256 + 8 + TILE), 16)
        hist_a = hist.iterator.toint()
        # b, above, count of b, out slot, tie slot, total, overflow
        info_a = hist_a + 256 * 4
        cnt_a = info_a + 8 * 4
        ties = smem.allocate_tensor(Int32, cute.make_layout(CAP), 16)
        ties_a = ties.iterator.toint()
        cute.arch.griddepcontrol_wait()
        n = row_len[row]
        kk = cutlass.min(n, Int32(K))
        base = Int64(row) * REGION
        orow = Int64(row) * K
        for i in cutlass.range_constexpr((256 + 8) // CS_THREADS + 1):
            if i * CS_THREADS + tid < 256 + 8:
                st_shared_u32(hist_a + (i * CS_THREADS + tid) * 4, Uint32(0))
        cute.arch.sync_threads()
        if tid < TILE:
            c = Uint32(cand[base + tid])
            if c >= Uint32(SLOTS):
                st_shared_u32(info_a + 24, Uint32(1))
                c = Uint32(SLOTS)
            st_shared_u32(cnt_a + tid * 4, c)
            red_shared(info_a + 20, c)
        cute.arch.sync_threads()
        total = Int32(ld_shared_u32(info_a + 20))
        j = tid % TILE
        half = tid // TILE
        cj = Int32(ld_shared_u32(cnt_a + j * 4))
        # this thread's candidates (lane j, slots half, half + 2, ...): one batch
        # of independent loads, every pass then works from registers
        words = []
        for i in cutlass.range_constexpr(SLOTS // 2):
            wv = Uint32(0)
            if half + 2 * i < cj:
                wv = Uint32(cand[base + (half + 2 * i + 1) * TILE + j])
            words.append(wv)
        bad = Int32(ld_shared_u32(info_a + 24))
        if total < kk:
            bad = Int32(1)
        if tid == 0:
            row_bin[3 * row + 1] = bad
        if bad == 0:
            if total <= Int32(K):
                for i in cutlass.range_constexpr(SLOTS // 2):
                  if half + 2 * i < cj:
                    pos, key = self.decode(words[i], j)
                    slot = warp_slot(Uint32(1), info_a + 12)
                    out[orow + Int32(slot)] = pos
                for i in cutlass.range(total + tid, K, CS_THREADS, unroll=1):
                    out[orow + i] = Int32(-1)
            else:
                for i in cutlass.range_constexpr(SLOTS // 2):
                  if half + 2 * i < cj:
                    pos, key = self.decode(words[i], j)
                    hist_add(hist_a, key >> Uint32(8))
                cute.arch.sync_threads()
                if w == 0:
                    _top_bin(hist_a, Uint32(K), lane, info_a)
                cute.arch.sync_threads()
                b1 = ld_shared_u32(info_a)
                above1 = ld_shared_u32(info_a + 4)
                cute.arch.sync_threads()
                st_shared_u32(hist_a + tid * 4, Uint32(0))
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(SLOTS // 2):
                  if half + 2 * i < cj:
                    pos, key = self.decode(words[i], j)
                    if (key >> Uint32(8)) == b1:
                        hist_add(hist_a, key & Uint32(0xFF))
                cute.arch.sync_threads()
                if w == 0:
                    _top_bin(hist_a, Uint32(K) - above1, lane, info_a)
                cute.arch.sync_threads()
                t = (b1 << Uint32(8)) | ld_shared_u32(info_a)
                above = above1 + ld_shared_u32(info_a + 4)
                m = ld_shared_u32(info_a + 8)
                need = Uint32(K) - above
                for i in cutlass.range_constexpr(SLOTS // 2):
                  if half + 2 * i < cj:
                    pos, key = self.decode(words[i], j)
                    keep = Uint32(0)
                    if key > t:
                        keep = Uint32(1)
                    if key == t and m == need:
                        keep = Uint32(1)
                    slot = warp_slot(keep, info_a + 12)
                    if keep != Uint32(0):
                        out[orow + Int32(slot)] = pos
                    elif key == t:
                        jt = atom_add_shared(info_a + 16, Uint32(1))
                        st_shared_u32(ties_a + jt * 4, Uint32(pos))
                cute.arch.sync_threads()
                if m != need:
                    for jt in cutlass.range(tid, Int32(m), CS_THREADS, unroll=1):
                        p = ld_shared_u32(ties_a + jt * 4)
                        rank = Uint32(0)
                        for x in cutlass.range(Int32(m), unroll=4):
                            if ld_shared_u32(ties_a + x * 4) < p:
                                rank += Uint32(1)
                        if rank < need:
                            out[orow + Int32(above + rank)] = Int32(p)


@functools.cache
def _cand_select_kernel(topk):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    t = cute.sym_int()
    args = (
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(CandSelect(topk), *args, options="--enable-tvm-ffi")


SAMPLES = 1024  # keys per request sampled for the PUBLISH thresholds
MARGIN = 2.0  # expected candidates per row = MARGIN * topk


TS_THREADS = 128


class ThrSelect:
    """PUBLISH threshold of each row (one CTA per row): the BF16 value of rank
    rank[r] (0 = largest) among the row's SAMPLES sampled logits, by two 8-bit
    radix passes; rank -1 keeps every key (-inf). Also clears the row's flag."""

    @cute.jit
    def __call__(self, s_logits, rank, row_bin, s):
        self.kernel(s_logits, rank, row_bin).launch(
            grid=(rank.shape[0], 1, 1), block=(TS_THREADS, 1, 1), stream=s,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, s_logits, rank, row_bin):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        w = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(256 + 8), 16)
        hist_a = hist.iterator.toint()
        info_a = hist_a + 256 * 4
        cute.arch.griddepcontrol_wait()
        rk = rank[row]
        if rk < 0:
            if tid == 0:
                row_bin[3 * row] = Int32(0xFF80)
                row_bin[3 * row + 1] = Int32(0)
        else:
            per = SAMPLES // TS_THREADS
            keys = []
            for i in cutlass.range_constexpr(per):
                v = Uint32(s_logits[Int64(row) * SAMPLES + i * TS_THREADS + tid])
                keys.append(order16(v & Uint32(0xFFFF)))
            for i in cutlass.range_constexpr(2):
                st_shared_u32(hist_a + (i * TS_THREADS + tid) * 4, Uint32(0))
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(per):
                red_shared(hist_a + ((keys[i] >> Uint32(8)) << Uint32(2)), 1)
            cute.arch.sync_threads()
            need = Uint32(rk + 1)
            if w == 0:
                _top_bin(hist_a, need, lane, info_a)
            cute.arch.sync_threads()
            b1 = ld_shared_u32(info_a)
            above1 = ld_shared_u32(info_a + 4)
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(2):
                st_shared_u32(hist_a + (i * TS_THREADS + tid) * 4, Uint32(0))
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(per):
                if (keys[i] >> Uint32(8)) == b1:
                    red_shared(hist_a + ((keys[i] & Uint32(0xFF)) << Uint32(2)), 1)
            cute.arch.sync_threads()
            if w == 0:
                _top_bin(hist_a, need - above1, lane, info_a)
            cute.arch.sync_threads()
            if tid == 0:
                t = (b1 << Uint32(8)) | ld_shared_u32(info_a)
                flip = Uint32(0xFFFF)
                if (t & Uint32(0x8000)) != Uint32(0):
                    flip = Uint32(0x8000)
                row_bin[3 * row] = Int32(t ^ flip)
                row_bin[3 * row + 1] = Int32(0)


@functools.cache
def _thr_select_kernel():
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    args = (
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(ThrSelect(), *args, options="--enable-tvm-ffi")


class SelectivePlan:
    """Per-step layout of the selective path (sampled thresholds -> PUBLISH ->
    CandSelect) for requests r with ``row_counts[r]`` rows whose last row sees
    ``row_lens_last[r]`` keys starting at ``key_bases[r]`` (128-aligned)."""

    def __init__(self, row_counts, row_lens_last, key_bases, device, topk=2048):
        self.topk = topk
        self.items, self.row_len, _, _ = plan(row_counts, row_lens_last, key_bases, device)
        t = sum(row_counts)
        self.t = t
        # candidate regions: row r at r * REGION words (int16 units here), dummy row t
        self.cand_base = torch.arange(t + 1, dtype=torch.int64, device=device) * (REGION * 2)
        # sampled keys of each request, all of a request's rows see all of them
        sidx, s_items, rank = [], [], []
        token = 0
        for r, (rows, last, kb) in enumerate(zip(row_counts, row_lens_last, key_bases)):
            first = last - rows + 1
            lens = torch.arange(first, last + 1)
            span = max(first, CAP + 1)
            if last > CAP:
                pos = ((torch.arange(SAMPLES) + 0.5) * span / SAMPLES).long()
                sidx.append(kb + pos)
                need = int(MARGIN * topk * SAMPLES / span + 0.999)
                rk = torch.full((rows,), min(max(need, 1), SAMPLES) - 1)
            else:
                sidx.append(torch.full((SAMPLES,), kb))
                rk = torch.zeros(rows, dtype=torch.int64)
            # rows that see at most CAP keys keep every key (rank -1: -inf)
            rank.append(torch.where(lens <= CAP, -1, rk))
            sb = r * SAMPLES
            for o in range(0, rows, BQ):
                s_items.append((token + o, min(BQ, rows - o), sb, SAMPLES // TILE))
            token += rows
        self.sample_idx = torch.cat(sidx).to(device)
        self.s_items = torch.tensor(s_items, dtype=torch.int32, device=device).view(-1, 4)
        self.s_row_len = torch.full((t,), SAMPLES, dtype=torch.int32, device=device)
        self.s_row_base = torch.arange(t + 1, dtype=torch.int64, device=device) * SAMPLES
        self.rank = torch.cat(rank).to(torch.int32).to(device)
        # per row: threshold, flag, -; the dummy row t keeps nothing (NaN)
        self.row_bin = torch.zeros(3 * (t + 1), dtype=torch.int32, device=device)
        self.row_bin[3 * t] = 0x7FC0
        self.s_bin = torch.zeros(3 * (t + 1), dtype=torch.int32, device=device)


def selective_topk(sp, q, q_scale, keys, key_scale, weights_bf16, out, ctas=None,
                   mode=PUBLISH):
    """Exact top-k of every row of ``sp`` into ``out`` (Int32 [T, topk], unordered,
    -1 padded). Returns the number of rows recomputed by the full path."""
    dev = q.device
    t = sp.t
    # 1) thresholds: score the sampled keys, the rank-th largest per row
    sk = keys[sp.sample_idx].contiguous()
    ssf = key_scale.reshape(-1)[sp.sample_idx].contiguous()
    s_logits = torch.empty((t + 1) * SAMPLES, dtype=torch.bfloat16, device=dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    score(q, q_scale, sk, ssf, weights_bf16, sp.s_row_len, sp.s_items, sp.s_row_base,
          s_logits, sp.s_bin, ctas=ctas)
    _thr_select_kernel()(s_logits.view(torch.int16), sp.rank, sp.row_bin, stream)
    # 2) scores, keeping the keys at or above the threshold
    cand = torch.empty((t + 1) * REGION, dtype=torch.int32, device=dev)
    score(q, q_scale, keys, key_scale, weights_bf16, sp.row_len, sp.items, sp.cand_base,
          cand.view(torch.bfloat16), sp.row_bin, ctas=ctas, mode=mode)
    # 3) exact top-k from the candidates
    _cand_select_kernel(out.shape[1])(
        cand, sp.row_len, sp.row_bin, out.view(-1), stream
    )
    return sp.row_bin, cand


def plan(row_counts, row_lens_last, key_bases, device):
    """Work items and the logits scratch layout of the step's prefill requests.

    Request r has ``row_counts[r]`` query rows; its last row sees
    ``row_lens_last[r]`` keys (row i sees row_lens_last[r] - row_counts[r] + 1 + i),
    its keys start at ``key_bases[r]`` (128-aligned) of the key tensor.

    Returns (items Int32 [n, 4] longest first, row_len Int32 [T], row_base
    Int64 [T + 1], scratch elements): a request's rows are rows of
    round_up(row_lens_last, 128) logits in the scratch; row_base[T] is a dummy row.
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
    # dummy row (tokens past an item's end write there)
    bases.append(scratch)
    scratch += max(((last + TILE - 1) // TILE * TILE for last in row_lens_last), default=0)
    it = torch.tensor(items, dtype=torch.int32, device=device).view(-1, 4)
    ln = torch.tensor(lens, dtype=torch.int32, device=device)
    rb = torch.tensor(bases, dtype=torch.int64, device=device)
    return it, ln, rb, scratch


@functools.cache
def _score_kernel(ctas, mode=0, topk=2048):
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
        fake((cute.sym_int(),), dtype=cutlass.Int64),
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(IndexerScore(ctas, mode, topk), *args, options="--enable-tvm-ffi")


@functools.cache
def _select_kernels(ctas, topk, words):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    t = cute.sym_int()
    common = (
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int64),
        fake((t,), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
    )
    mark = cute.compile(
        PrefillMark(ctas, topk, words), *common, make_fake_stream(), options="--enable-tvm-ffi"
    )
    fin = cute.compile(
        PrefillFinish(ctas * 8, topk, words),
        *common,
        fake((t, topk), dtype=cutlass.Int32),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    return mark, fin


def score(q, q_scale, keys, key_scale, weights_bf16, row_len, items, row_base,
          logits, row_bin, ctas=None, mode=0, topk=2048):
    """BF16 logits of every prefill row into ``logits`` and each row's top-k bound
    into ``row_bin`` (Int32 [3 T]: bound bin rank, count above, candidate count
    zeroed for the select) (see ``plan``)."""
    dev = q.device
    t = q.shape[0]
    assert keys.shape[0] % TILE == 0
    ctas = ctas or torch.cuda.get_device_properties(dev).multi_processor_count
    _score_kernel(ctas, mode, topk)(
        q.view(t * HEADS, DIM // 2),
        q_scale.reshape(-1),
        keys,
        key_scale.reshape(-1),
        weights_bf16.contiguous().view(torch.int32).view(t, HEADS // 2),
        row_len,
        items,
        row_base,
        logits.view(torch.int16),
        row_bin,
        torch.cuda.current_stream(dev).cuda_stream,
    )


def select(logits, row_base, row_len, row_bin, bitmap, cand, out, ctas=None):
    """Exact top-k key indices of every row (position order, -1 padded) into ``out``.
    ``bitmap``: Int32 [T * words] (words = bitmap words per row, the longest row's
    positions / 32), ``cand``: Int32 [T * CAND]; ``row_bin`` Int32 [3 T] from score."""
    ctas = ctas or torch.cuda.get_device_properties(out.device).multi_processor_count
    words = bitmap.numel() // row_len.shape[0]
    mark, fin = _select_kernels(ctas * 4, out.shape[1], words)
    stream = torch.cuda.current_stream(out.device).cuda_stream
    args = (logits.view(torch.int16), row_base, row_len, row_bin, bitmap, cand)
    mark(*args, stream)
    fin(*args, out, stream)
