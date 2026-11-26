# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused prefill DSA indexer for Kimi K3.1 on sm_107 (VR200).

Replaces the stock prefill chain of ``sparse_attn_indexer`` (per 512 MB chunk:
DeepGEMM ``fp8_fp4_mqa_logits`` writing BF16 logits to HBM, then the trtllm_gvr2
top-k re-reading them) by a selective path that never materializes the logits
(``selective_topk``):

  ThrSelect     per row, a threshold from a sampled score of SAMPLES keys (a
                value expected to keep about MARGIN x topk keys).
  IndexerScore  persistent tcgen05 kernel, one CTA per SM. A work item is BQ
                query tokens of one request (B operand: BQ x 32 heads x 128
                MXFP4); it walks the request's keys in 128-key tiles (A operand,
                M = 128) from the newest to the oldest with one 2x-K block-scaled
                MXF4 MMA (K = 128) per tile into ACC_STAGES TMEM accumulator
                stages. The epilogue (two tokens per warpgroup, one thread per key)
                forms the BF16 logit at stock precision and, in PUBLISH mode,
                appends every key at or above the row threshold to the row's
                candidate region (per key lane: a count, then SLOTS words).
  CandSelect2   exact top-k of each row from its candidates (one exact histogram
                pass; ties at the k-th value to the lowest key index).
  recovery      rows CandSelect2 cannot settle (too few candidates, a full lane,
                too many threshold ties) are recomputed by the full path (score
                mode 0 + PrefillMark / PrefillFinish), on device.

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

from vllm.triton_utils import tl, triton

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
def skel_bits(a: Float32, b: Float32):
    """Benchmark skeleton: a value tied to two loaded registers that never reaches
    a row threshold (two tiny BF16 values)."""
    return (
        "{ .reg .b32 x, y; mov.b32 x, $1; mov.b32 y, $2; xor.b32 x, x, y; "
        "and.b32 $0, x, 0x00010001; }"
    ), "=r,f,f", (a, b)


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
def tmem_ld16(taddr):
    """tcgen05.ld 32x32b.x16: 16 columns of this warp's 32 TMEM lanes (one per thread)."""
    regs = ",".join(f"${i}" for i in range(16))
    code = "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16];"
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 16


@asm()
def tmem_ld32(taddr):
    """tcgen05.ld 32x32b.x32: 32 columns of this warp's 32 TMEM lanes (one per thread)."""
    regs = ",".join(f"${i}" for i in range(32))
    code = "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];"
    return code, ",".join(["=f"] * 32) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 32


@asm()
def tmem_wait_n(*vals):
    """tcgen05.wait::ld, with the loaded registers tied through it."""
    n = len(vals)
    cons = ",".join(["=f"] * n) + "," + ",".join(str(i) for i in range(n))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * n


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


STORE = 1  # IndexerScore mode: the logits only (no histograms / row bounds)
PUBLISH = 6  # IndexerScore mode: keep only the keys at or above a row threshold
SLOTS = 96  # PUBLISH keys kept per (row, key lane): more and the row is recomputed
# candidate columns rotate within a warp every 2^ROT_SHIFT key tiles (publish_rot):
# a fixed key position spreads over 32 columns, the shuffle runs once per block
ROT_SHIFT = 5
# words per row: lane counts, then SLOTS words per lane and two spill words (a lane
# is clamped to its spill slot once per tile pair, so it never writes past them)
REGION = (SLOTS + 3) * TILE
CAP = SLOTS * TILE


@asm(Uint32, pure=True)
def ge_bf16x2(r, thr):
    """bit 0: low BF16 of r >= low of thr, bit 1: high halves (NaN: false)."""
    code = (
        "{ .reg .pred p, q; .reg .u32 a, b; setp.ge.bf16x2 p|q, $1, $2; "
        "selp.u32 a, 1, 0, p; selp.u32 b, 2, 0, q; or.b32 $0, a, b; }"
    )
    return code, "=r,r,r", (Uint32(r), Uint32(thr))


@asm()
def publish2(p0, p1, r, thr, tw):
    """Low BF16 of r >= low of thr: store (tw | low) at p0 and advance it one
    128-word slot; the high halves likewise at p1."""
    code = (
        "{ .reg .pred a, b; .reg .b32 v, u; setp.ge.bf16x2 a|b, $4, $5; "
        "prmt.b32 v, $4, $6, 0x7610; prmt.b32 u, $4, $6, 0x7632; "
        "@a st.global.b32 [$2], v; @a add.u64 $0, $2, 512; "
        "@b st.global.b32 [$3], u; @b add.u64 $1, $3, 512; }"
    )
    return code, "=l,=l,0,1,r,r,r,~{memory}", (
        Int64(p0), Int64(p1), Uint32(r), Uint32(thr), Uint32(tw)), (Int64, Int64)


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


def _qn_layout():
    """MMA N row n' = h' + 16 tok + 128 half -> Q row (tok, head 16 half + h') of the item,
    as a linear index of the (16 h', 2 half, BQ tok) row mode: half-major, so a token
    pair's 16-head half is 32 contiguous accumulator columns (one TMEM x32 load)."""
    return cute.make_layout((16, BQ, 2), stride=(1, HEADS, 16))


def _sigma(h: int) -> int:
    """Token 1's head paired with token 0's head h (skew within each group of 4)."""
    return h - h % 4 + (h % 4 + 1) % 4


def sf_word(row: int) -> int:
    """32-bit word of row ``row``'s four UE8M0 scales in the block-scaled SF atom
    ((32, 4), (32, 4)):((16, 4), (0, 1)) bytes, 128 rows per 512-byte chunk."""
    return 128 * (row // 128) + 4 * (row % 32) + (row % 128) // 32


class IndexerScore:
    """Exact top-``topk`` key indices of every query row of the step's prefill
    requests (see the module docstring)."""

    def __init__(
        self, ctas: int, mode: int = 0, topk: int = 2048, epi: int = 0,
        split: bool = False,
    ):
        """mode: 0 logits + per-row top-k bound (row_bin); STORE logits only;
        PUBLISH candidates at or above the row thresholds (row_bin), no logits.
        split (mode 0): items are [n, 7] key splits (see SplitPlan): fields 4, 5
        the split's tile range [lo, hi) of the item's (even-rounded) tiles, field 6
        its histogram slot (-1: the whole item, row bounds in the kernel; else the
        rows' histograms go to ``ghist`` slot rows, finished by RowBound)."""
        assert not split or mode in (0, PUBLISH)
        self.split = split
        self.mode = mode
        self.ctas = ctas
        self.topk = topk
        # epilogue math (benchmark calibration only): 0 full; 1 skeleton (TMEM loads,
        # stage release and publish of a value tied to the loads, no math);
        # 2 heads 0-15 only (half the math, all loads); 3 no TMEM loads (MMA path
        # alone); 4 skeleton epilogue with the MMAs skipped (TMEM loads alone);
        # 6 full epilogue math with the MMAs skipped (TMEM loads + math alone)
        self.epi = epi
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
        n_items: cute.Tensor,  # Int32 [1]: items processed (a prefix of ``items``)
        row_base: cute.Tensor,  # Int64 [T + 1]: row t's first element in logits (T: dummy row)
        logits: cute.Tensor,  # Int16 [*]: BF16 logits scratch
        row_bin: cute.Tensor,  # Int32 [2 * T]: per row the bound rank and the count above it
        ghist: cute.Tensor,  # Int32 [slots * BQ * S_BINS]: split items' row histograms
        s,
    ):
        fp4 = cutlass.Float4E2M1FN
        sf_t = cutlass.Float8E8M0FNU
        nq = q8.shape[0]
        nk = k8.shape[0]
        mQ = cute.make_tensor(
            cute.recast_ptr(q8.iterator, dtype=fp4),
            # rows (16 heads, 2 halves, tokens): the B tiler gathers them half-major
            cute.make_layout(
                ((16, 2, nq // HEADS), DIM, 1),
                stride=((DIM, 16 * DIM, HEADS * DIM), 1, nq * DIM),
            ),
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
            (TILE, _qn_layout(), DIM),
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
            n_items,
            row_base,
            logits,
            row_bin,
            ghist,
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
        n_items_t: cute.Tensor,
        row_base: cute.Tensor,
        logits: cute.Tensor,
        row_bin: cute.Tensor,
        ghist: cute.Tensor,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
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
        n_items = n_items_t[0]
        if warp < PRODUCER:
            self.epilogue(
                tAcc, acc_full, acc_empty, q_full, q_empty, items, n_items, row_base,
                logits, row_bin, (end_a, w_a, st.hist.data_ptr().toint(), st.wscan.data_ptr().toint()),
                warp, tidx, bid,
                lane, ghist,
            )
        else:
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
                # running K ring slot and phase (no div/mod chains per tile: this
                # warp shares its scheduler with four issue-bound epilogue warps)
                kv_stage = Int32(0)
                kv_ph = Int32(0)
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
                            r = c * 128 + 32 * i + lane  # MMA N row (half-major)
                            q_row = (r // 16) % BQ * HEADS + 16 * (r // (16 * BQ)) + r % 16
                            v = Uint32(0)
                            if q_row // HEADS < ntok:
                                v = Uint32(q_sf[token0 * HEADS + q_row])
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
                        (_qn_layout(), DIM),
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
                    t_lo = Int32(0)
                    t_hi = ntp
                    if cutlass.const_expr(self.split):
                        t_lo = items[item, 4]
                        t_hi = items[item, 5]
                    # key bases are 128-aligned (plan): tile coordinates add up
                    kt0 = key0 >> 7
                    for tt in cutlass.range(t_hi - t_lo, unroll=1):
                        t = cutlass.min(t_hi - 1 - tt, ntiles - 1)
                        cute.arch.mbarrier_wait(kv_empty + kv_stage, kv_ph ^ 1)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                kv_full + kv_stage, self.k_bytes + TILE * 4
                            )
                            bulk_g2s(
                                sfa_a + kv_stage * SF_WORDS_A * 4,
                                k_sf.iterator.toint() + Int64(key0 + t * TILE) * 4,
                                kv_full + kv_stage,
                                TILE * 4,
                            )
                        cute.copy(
                            tma_k,
                            tKg[(None, kt0 + t, 0, 0)],
                            tKs[(None, kv_stage)],
                            tma_bar_ptr=kv_full + kv_stage,
                        )
                        kv_stage += 1
                        if kv_stage == KV_STAGES:
                            kv_stage = Int32(0)
                            kv_ph = kv_ph ^ 1
            elif warp == MMA_WARP:
                tCrA = mma.make_fragment_A(sA)
                tCrB = mma.make_fragment_B(sB)
                sfa_ptr = cute.recast_ptr(tmem_ptr + ACC_STAGES * N, dtype=sf_t)
                sfb_ptr = cute.recast_ptr(tmem_ptr + ACC_STAGES * N + self.sfa_cols, dtype=sf_t)
                tCtSFA = cute.make_tensor(sfa_ptr, tsfa_layout)
                tCtSFB = cute.make_tensor(sfb_ptr, tsfb_layout)
                s2t_a = self.s2t(sSFA, tCtSFA)
                s2t_b = self.s2t(sSFB, tCtSFB)
                # running K ring slot / phase and accumulator phase: items have an
                # even tile count, so the accumulator stages go 0, 1 per tile pair
                kv_stage = Int32(0)
                kv_ph = Int32(0)
                acc_ph = Int32(0)
                jq = Int32(0)
                for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
                    ntiles = items[item, 3]
                    npairs = (ntiles + 1) // 2
                    if cutlass.const_expr(self.split):
                        npairs = (items[item, 5] - items[item, 4]) // 2
                    qs = jq % Q_STAGES
                    cute.arch.mbarrier_wait(q_full + qs, (jq // Q_STAGES) & 1)
                    cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, qs)], s2t_b[2])
                    for _p in cutlass.range(npairs, unroll=1):
                        for a in cutlass.range_constexpr(ACC_STAGES):
                            cute.arch.mbarrier_wait(acc_empty + a, acc_ph ^ 1)
                            cute.arch.mbarrier_wait(kv_full + kv_stage, kv_ph)
                            cute.copy(
                                s2t_a[0], s2t_a[1][(None, None, None, None, kv_stage)],
                                s2t_a[2],
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(
                                cute.size(tCrA, mode=[2]) if self.epi not in (4, 6) else 0
                            ):
                                cute.gemm(
                                    mma,
                                    tAcc[(None, None, None, a)],
                                    [tCrA[(None, None, kb, kv_stage)],
                                     tCtSFA[(None, None, kb)]],
                                    [tCrB[(None, None, kb, qs)], tCtSFB[(None, None, kb)]],
                                    tAcc[(None, None, None, a)],
                                )
                                mma.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(kv_empty + kv_stage)
                                tcgen05.commit(acc_full + a)
                            kv_stage += 1
                            if kv_stage == KV_STAGES:
                                kv_stage = Int32(0)
                                kv_ph = kv_ph ^ 1
                        acc_ph = acc_ph ^ 1
                    with cute.arch.elect_one():
                        tcgen05.commit(q_empty + qs)
                    jq += 1

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
        self, tAcc, acc_full, acc_empty, q_full, q_empty, items, n_items, row_base,
        logits, row_bin, sm, warp, tidx, bid, lane, ghist,
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
        wq = warp % 4  # = lt // 32, warp-uniform: TMEM addresses stay in uniform registers
        out_a = logits.iterator.toint()
        ph = Int32(0)
        jq = Int32(0)
        n_rows = row_base.shape[0] - 1
        key_in_tile = 4 * (lt % 32) + lt // 32
        tok_a = 2 * wg
        # this warp's TMEM lane quarter, column 0 of accumulator stage 0
        tmem_q = Int32(tAcc.iterator.toint()) + ((wq * 32) << 16) + tok_a * 16
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
            # the tiles [t_lo, t_hi) of this item (split items: a key range)
            t_lo = Int32(0)
            t_hi = ntp
            if cutlass.const_expr(self.split):
                t_lo = items[item, 4]
                t_hi = items[item, 5]
            npairs = (t_hi - t_lo) // 2
            # running pointers: position lt of the item's newest tile pair's first tile
            first = Int64((t_hi - 1) * TILE + lt) * 2
            ptr0 = out_a + row_base[row0] * 2 + first
            ptr1 = out_a + row_base[row1] * 2 + first
            thr = Uint32(0)
            rs = cute.make_rmem_tensor((3,), Int64)
            cand0 = ptr0
            cand1 = ptr1
            p0 = ptr0
            p1 = ptr1
            lim0 = ptr0
            lim1 = ptr1
            if cutlass.const_expr(self.mode >= PUBLISH):
                # BF16 thresholds of both rows (row_bin[3 r], the dummy row's is NaN)
                # and their candidate lists (row_base in 16-bit units)
                thr = (Uint32(row_bin[3 * row1]) << Uint32(16)) | (
                    Uint32(row_bin[3 * row0]) & Uint32(0xFFFF)
                )
                # this thread's column of each row's candidate region: word 0
                # the count, then one word per kept key (tile << 16 | BF16)
                # split PUBLISH units (key-tile ranges of one item) write their own
                # candidate region copy: field 6 = its offset in 16-bit units
                roff = Int64(0)
                if cutlass.const_expr(self.split):
                    roff = Int64(items[item, 6]) * 2
                rowb0 = out_a + row_base[row0] * 2 + roff
                rowb1 = out_a + row_base[row1] * 2 + roff
                # rotated candidate columns (publish_rot): key lane (warp q, lane l)
                # of tile t publishes into column 32 q + (l + (t >> ROT_SHIFT)) mod 32;
                # p0 / p1 point into the current column, rs = (its limits, block)
                bp = Int32(t_hi) >> ROT_SHIFT
                col = Int64(wq * 32 + ((lt % 32 + bp) & 31)) * 4
                cand0 = rowb0 + col
                cand1 = rowb1 + col
                p0 = cand0 + TILE * 4
                p1 = cand1 + TILE * 4
                rs[0] = cand0 + (SLOTS + 1) * TILE * 4
                rs[1] = cand1 + (SLOTS + 1) * TILE * 4
                rs[2] = Int64(bp)
                lim0 = rs[0]
                lim1 = rs[1]
            # tile pairs whose newer tile reaches past a row end (causal edge)
            full_tiles = cutlass.min(end0, end1) // TILE
            # head weights (BF16x2) of both tokens: wt[j][i] = heads 2i, 2i + 1
            wbase = w_a + (qs * BQ * 16 + tok_a * 16) * 4
            wt = []
            for j in cutlass.range_constexpr(2):
                wj = []
                for i in cutlass.range_constexpr(4):
                    v = lds_v4(wbase + (j * 16 + 4 * i) * 4)
                    wj.extend(v)
                wt.append(wj)
            # token-interleaved weights: w[h] = (token 0 head h, token 1 head sigma(h))
            w = [prmt(wt[0][h // 2], wt[1][_sigma(h) // 2],
                      (0x10 if h % 2 == 0 else 0x32)
                      | ((0x54 if _sigma(h) % 2 == 0 else 0x76) << 8))
                 for h in range(HEADS)]
            # release the Q stage only after its row ends and weights are in
            # registers (the PRMTs consume every loaded word): the fence completes
            # this thread's shared-memory reads before the producer may refill the
            # stage (INFRA / D-ATTN 2026-11-20 SASS release audit)
            fence_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(q_empty + qs)
            # tile pairs whose newer tile reaches past a row end (causal edge, and
            # the repeated tile of an odd count) come first (newest to oldest);
            # the rest run without masking
            n_edge = cutlass.min(cutlass.max(t_hi + 1 - full_tiles, 0) // 2, npairs)
            for pp in cutlass.range(n_edge, unroll=1):
                ptr0, ptr1, p0, p1 = self.tile_pair(
                    True, t_hi - 1 - 2 * pp, ph, tmem_q, acc_full, acc_empty, lane,
                    w, ptr0, ptr1, p0, p1, thr, rs, lim1, end0, end1, ntiles,
                    key_in_tile, hist0, hist1,
                )
                ph = ph ^ 1
            if cutlass.const_expr(self.mode >= PUBLISH and self.epi == 0):
                # interior pairs, rotated: a stage's second half is finished in the
                # next block; the carry starts as NaN sums (nothing published)
                hc = cute.make_rmem_tensor((HEADS,), Float32)
                sc = cute.make_rmem_tensor((4,), Uint32)
                for i in cutlass.range_constexpr(HEADS):
                    hc[i] = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    sc[i] = Uint32(0x7FC07FC0)
                # publish tile word of the carried stage: (tile << 16) | lt, a
                # per-thread register (publish2 reads only its high half), one
                # tile below the pair's newer tile per stage
                tw = (Uint32(t_hi - 2 * n_edge) << Uint32(16)) | Uint32(lt)
                for pp in cutlass.range(n_edge, npairs, unroll=1):
                    p0, p1, tw = self.pair_rot(
                        ph, tmem_q, acc_full, acc_empty, w, p0, p1, thr, rs, rs,
                        hc, sc, tw, key_in_tile,
                    )
                    ph = ph ^ 1
                p0 = cutlass.min(p0, rs[0])
                p1 = cutlass.min(p1, rs[1])
                p0, p1 = self.stage_tail(
                    hc, w, [[sc[0], sc[1]], [sc[2], sc[3]]], tw, p0, p1, thr, rs,
                    key_in_tile,
                )
            else:
                for pp in cutlass.range(n_edge, npairs, unroll=1):
                    ptr0, ptr1, p0, p1 = self.tile_pair(
                        False, t_hi - 1 - 2 * pp, ph, tmem_q, acc_full, acc_empty, lane,
                        w, ptr0, ptr1, p0, p1, thr, rs, lim1, end0, end1, ntiles,
                        key_in_tile, hist0, hist1,
                    )
                    ph = ph ^ 1
            if cutlass.const_expr(self.mode >= PUBLISH):
                # the counts of the column this thread published into last
                # (a full column counts as overflowed: the row is recomputed)
                cb0 = rs[0] - (SLOTS + 1) * TILE * 4
                cb1 = rs[1] - (SLOTS + 1) * TILE * 4
                st_global_u32(cb0, Uint32((p0 - cb0) // (TILE * 4) - 1))
                st_global_u32(cb1, Uint32((p1 - cb1) // (TILE * 4) - 1))
            elif cutlass.const_expr(self.mode == 0):
                # per row: bound bin rank and the count above it, then clear
                wg_sync(wg)
                slot = Int32(-1)
                if cutlass.const_expr(self.split):
                    slot = items[item, 6]
                if slot < 0:
                    self.row_bound(hist0, end0, row0, n_rows, row_bin, wscan_a + wg * 16, wg, lt, wq, lane)
                    self.row_bound(hist1, end1, row1, n_rows, row_bin, wscan_a + wg * 16, wg, lt, wq, lane)
                else:
                    # a key split: add both rows' histograms into the slot's rows
                    # (integer counts: the merged histogram is order independent)
                    gh = ghist.iterator.toint() + Int64(slot * BQ + tok_a) * (S_BINS * 4)
                    for i in cutlass.range_constexpr(2 * S_BINS // 128):
                        j = i * 128 + lt
                        c = ld_shared_u32(hist0 + j * 4)
                        if c != Uint32(0) and tok_a + (i * 128) // S_BINS < ntok:
                            red_global_add(gh + Int64(j) * 4, c)
                for i in cutlass.range_constexpr(2 * S_BINS // 128):
                    st_shared_u32(hist0 + (i * 128 + lt) * 4, Uint32(0))
                wg_sync(wg)
            jq += 1

    @cute.jit
    def tile_pair(
        self, edge: cutlass.Constexpr, t_hi, ph, tmem_q, acc_full, acc_empty, lane, w,
        ptr0, ptr1, p0, p1, thr, lim0, lim1, end0, end1, ntiles, key_in_tile, hist0,
        hist1,
    ):
        """Both accumulator stages of one tile pair (tiles t_hi and t_hi - 1): the
        BF16 logits of this thread's key for its two tokens, stored (mode < PUBLISH,
        plus the histograms) or published. ``edge``: mask keys past the row ends
        and the repeated tile of an odd count."""
        # head-half pipeline: the TMEM load of the next half (heads 16-31 of this
        # stage, then heads 0-15 of the next stage) is issued before this half's
        # math; a stage is released as soon as its last half landed
        if cutlass.const_expr(self.mode >= PUBLISH):
            p0 = cutlass.min(p0, lim0[0])
            p1 = cutlass.min(p1, lim0[1])
        if cutlass.const_expr(self.epi == 3):
            for st in cutlass.range_constexpr(ACC_STAGES):
                cute.arch.mbarrier_wait(acc_full + st, ph)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty + st)
            return ptr0 - 2 * TILE * 2, ptr1 - 2 * TILE * 2, p0, p1
        cute.arch.mbarrier_wait(acc_full, ph)
        h0 = tmem_wait_n(*tmem_ld32(tmem_q))
        h0n = h0
        for st in cutlass.range_constexpr(ACC_STAGES):
            t = t_hi - st
            col = tmem_q + st * N
            if cutlass.const_expr(st > 0):
                h0 = tmem_wait_n(*h0n)
            h1 = tmem_ld32(col + 16 * BQ)
            if cutlass.const_expr(self.epi != 1):
                ss = self.reduce_half(h0, w, None, 0)
            h1 = tmem_wait_n(*h1)
            # tcgen05.wait::ld is warp-aligned: every lane's loads have landed
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + st)
            if cutlass.const_expr(st + 1 < ACC_STAGES):
                cute.arch.mbarrier_wait(acc_full + st + 1, ph)
                h0n = tmem_ld32(col + N)
            if cutlass.const_expr(self.epi in (0, 6)):
                ss = self.reduce_half(h1, w, ss, HEADS // 2)
                r = self.combine2(ss)
            elif cutlass.const_expr(self.epi == 2):
                r = self.combine2(ss) ^ skel_bits(h1[0], h1[31])
            else:
                r = skel_bits(h0[0], h1[31])
            if cutlass.const_expr(edge):
                kidx = t * TILE + key_in_tile
                if kidx >= end0 or t >= ntiles:
                    r = r | Uint32(0xFFFF)
                if kidx >= end1 or t >= ntiles:
                    r = r | Uint32(0xFFFF0000)
            if cutlass.const_expr(self.mode >= PUBLISH):
                p0, p1 = self.publish_rot(p0, p1, r, thr, t, lim0, key_in_tile)
            elif cutlass.const_expr(edge):
                if t < ntiles:
                    st_global_u16(ptr0 - st * TILE * 2, r)
                    st_global_u16(ptr1 - st * TILE * 2, r >> Uint32(16))
            else:
                st_global_u16_off(ptr0, r, -st * TILE * 2)
                st_global_u16_off(ptr1, r >> Uint32(16), -st * TILE * 2)
            if cutlass.const_expr(self.mode == 0):
                # histogram of the raw top 11 bits (masked keys land in bin 2047,
                # the lowest rank)
                red_shared(hist0 + (bfe(r, 5, 11) << Uint32(2)), 1)
                red_shared(hist1 + (r >> Uint32(21) << Uint32(2)), 1)
        return ptr0 - 2 * TILE * 2, ptr1 - 2 * TILE * 2, p0, p1

    @cute.jit
    def publish_rot(self, p0, p1, r, thr, t, rs, kit):
        """Publish both tokens' logits of tile t (row thresholds thr) into ROTATED
        candidate columns: key lane (warp q, lane l) of tile t writes column
        32 q + (l + (t >> ROT_SHIFT)) mod 32, so keys at a fixed position mod 128
        (attention sinks every 64 tokens, say) spread over 32 columns instead of
        filling one (loop 9: real text overflowed SLOTS on every row). Within a
        block of 2^ROT_SHIFT tiles a thread keeps its column and pointers (the
        unrotated fast path); entering a new block (warp-uniform) it converts its
        pointers to counts, takes lane l - (block change)'s counts by a warp
        shuffle and re-points into its new column. rs = (row 0 / row 1 column
        limits, current block), updated in place. The select recovers the key from
        the column and the tile word."""
        b = t >> ROT_SHIFT
        bp = Int32(rs[2])
        if b != bp:
            lane = Int32(kit >> 2)
            wq = Int32(kit & 3)
            ob0 = rs[0] - (SLOTS + 1) * TILE * 4
            ob1 = rs[1] - (SLOTS + 1) * TILE * 4
            c0 = Uint32((cutlass.min(p0, rs[0]) - ob0) // (TILE * 4) - 1)
            c1 = Uint32((cutlass.min(p1, rs[1]) - ob1) // (TILE * 4) - 1)
            cc = Uint32(cute.arch.shuffle_sync(Int32((c1 << Uint32(16)) | c0),
                                               (lane - (bp - b)) & 31))
            oc = Int64(wq * 32 + ((lane + bp) & 31)) * 4
            nc = Int64(wq * 32 + ((lane + b) & 31)) * 4
            nb0 = ob0 - oc + nc
            nb1 = ob1 - oc + nc
            p0 = nb0 + Int64((cc & Uint32(0xFFFF)) + Uint32(1)) * (TILE * 4)
            p1 = nb1 + Int64((cc >> Uint32(16)) + Uint32(1)) * (TILE * 4)
            rs[0] = nb0 + (SLOTS + 1) * TILE * 4
            rs[1] = nb1 + (SLOTS + 1) * TILE * 4
            rs[2] = Int64(b)
            p0 = cutlass.min(p0, rs[0])
            p1 = cutlass.min(p1, rs[1])
        return publish2(p0, p1, r, thr, Uint32(t) << Uint32(16))

    @cute.jit
    def stage_tail(self, h1, w, ss, tw, p0, p1, thr, rs, kit):
        """Heads 16-31 of a stage (h1) into its running sums, then publish both
        tokens' logits of tile t (tw >> 16)."""
        ss = self.reduce_half(h1, w, ss, HEADS // 2)
        r = self.combine2(ss)
        return self.publish_rot(p0, p1, r, thr, Int32(tw >> Uint32(16)), rs, kit)

    @cute.jit
    def pair_rot(self, ph, tmem_q, acc_full, acc_empty, w, p0, p1, thr, lim0, lim1,
                 hc, sc, tw, kit):
        """One interior tile pair, rotated: after each stage's barrier wait and its
        first loads, finish the previous stage (the carry hc / sc / tw for stage 0, the
        pair's own stage 0 for stage 1); returns with stage 1's second half (heads
        16-31, loaded) and first-half sums in the carry."""
        p0 = cutlass.min(p0, lim0[0])
        p1 = cutlass.min(p1, lim0[1])
        hp = [hc[i] for i in range(HEADS)]
        sp = [[sc[0], sc[1]], [sc[2], sc[3]]]
        tp = tw
        for st in cutlass.range_constexpr(ACC_STAGES):
            col = tmem_q + st * N
            cute.arch.mbarrier_wait(acc_full + st, ph)
            h0 = tmem_ld32(col)
            p0, p1 = self.stage_tail(hp, w, sp, tp, p0, p1, thr, lim0, kit)
            h0 = tmem_wait_n(*h0)
            h1 = tmem_ld32(col + 16 * BQ)
            ss = self.reduce_half(h0, w, None, 0)
            h1 = tmem_wait_n(*h1)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + st)
            hp = h1
            sp = ss
            tp = tw - Uint32((st + 1) << 16)
        for i in cutlass.range_constexpr(HEADS):
            hc[i] = hp[i]
        sc[0] = sp[0][0]
        sc[1] = sp[0][1]
        sc[2] = sp[1][0]
        sc[3] = sp[1][1]
        return p0, p1, tp

    @cute.jit
    def reduce_half(self, buf, w, ss, h0: cutlass.Constexpr):
        """Heads h0 .. h0 + 15 of both tokens (buf: token 0's 16 values, then token
        1's) into the running BF16 logit sums at stock precision. Stock keeps per
        token four chains, head classes h % 4 = 0, 1, 2, 3 (s0.lo, s0.hi, s1.lo,
        s1.hi), each head term relu-rounded to BF16 and FMA'd in BF16 in head
        order. Here accumulator c holds token 0's chain c (low half) and token 1's
        chain (c + 1) % 4 (high half): one F2FP packs (t0 head h, t1 head
        sigma(h)), one HFMA2 advances both chains with the weights w[h]."""
        if cutlass.const_expr(ss is None):
            acc = [Uint32(0) for _ in range(4)]
        else:
            acc = [ss[0][0], ss[0][1], ss[1][0], ss[1][1]]
        for h in cutlass.range_constexpr(h0, h0 + HEADS // 2):
            c = h % 4
            acc[c] = fma_bf16x2(
                relu_bf16x2(buf[h - h0], buf[16 + _sigma(h) - h0]), w[h], acc[c]
            )
        return [[acc[0], acc[1]], [acc[2], acc[3]]]

    @cute.jit
    def combine2(self, ss):
        """Both tokens' BF16 logits (t0 low, t1 high): stock (s0.lo + s1.lo) +
        (s0.hi + s1.hi) per token. x = acc0 + acc2 = (t0 s0.lo + s1.lo, t1 s0.hi +
        s1.hi), y = acc1 + acc3 = (t0 s0.hi + s1.hi, t1 s1.lo + s0.lo); BF16
        addition is commutative, so x + y is stock for both tokens."""
        x = add_bf16x2(ss[0][0], ss[1][0])
        y = add_bf16x2(ss[0][1], ss[1][1])
        return add_bf16x2(x, y)

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
def key_rank(p: Uint32) -> Uint32:
    """Logits position -> key index, unsigned (ties rank by key index)."""
    return (p & Uint32(0xFFFFFF80)) | ((p & Uint32(31)) << Uint32(2)) | ((p >> Uint32(5)) & Uint32(3))


@cute.jit
def position_of(k: Int32) -> Int32:
    """Key index -> logits position (inverse of key_index)."""
    return (k & Int32(-TILE)) | ((k & Int32(3)) << 5) | ((k & Int32(TILE - 1)) >> 2)


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
def red_global_add(address, value):
    return "red.global.add.u32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


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
    zeroed by the scorer). Rows with at most ``topk`` keys are skipped. Also writes
    each chunk's bound-bin histogram by low 5 key bits (``xh``: 32 words per (row,
    chunk)) and appends the rows whose candidates overflow CAND to ``xl`` (count in
    xl[0], zeroed by the caller) for PrefillExact."""

    def __init__(self, ctas: int, topk: int):
        self.ctas = ctas
        self.topk = topk

    @cute.jit
    def __call__(self, logits, row_base, row_len, row_bin, bitmap, cand, xh, xl, gate, s):
        self.kernel(logits, row_base, row_len, row_bin, bitmap, cand, xh, xl, gate).launch(
            grid=(self.ctas, 1, 1), block=(MARK_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, logits, row_base, row_len, row_bin, bitmap, cand, xh, xl, gate):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        smem = utils.SmemAllocator()
        mbuf = smem.allocate_tensor(Uint32, cute.make_layout(MARK_BUF + 4), 16)
        cnt_a = mbuf.iterator.toint()
        buf_a = cnt_a + 16
        hbuf = smem.allocate_tensor(Uint32, cute.make_layout(32), 16)
        hist_a = hbuf.iterator.toint()
        if tid == 0:
            st_shared_u32(cnt_a, Uint32(0))
        if tid < 32:
            st_shared_u32(hist_a + tid * 4, Uint32(0))
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        n_rows = Int32(row_len.shape[0])
        if gate[0] == 0:
            n_rows = Int32(0)  # nothing to select (recovery with no flagged row)
        # bitmap words per row (row stride), a run-time value: one compile serves
        # every context length
        stride = Int32(bitmap.shape[0]) // cutlass.max(Int32(row_len.shape[0]), Int32(1))
        cpr = (stride * 32 + CHUNK_V * 8 - 1) // (CHUNK_V * 8)
        logits_a = logits.iterator.toint()
        bm_a = bitmap.iterator.toint()
        cand_a = cand.iterator.toint()
        bin_a = row_bin.iterator.toint()
        xh_a = xh.iterator.toint()
        xl_a = xl.iterator.toint()
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
                            st_global_u32(bm_a + (Int64(row) * Int64(stride) + (v >> 2)) * 4, wbits)
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
                                        red_shared(hist_a + (key & Uint32(31)) * 4, 1)
                                        if slot < Uint32(MARK_BUF):
                                            st_shared_u32(
                                                buf_a + slot * 4,
                                                (Uint32(v * 8 + e) << Uint32(5)) | (key & Uint32(31)),
                                            )
                                        slot += Uint32(1)
                cute.arch.sync_threads()
                # the chunk's bound-bin histogram (PrefillExact: overflowed rows)
                if tid < 32:
                    st_global_u32(
                        xh_a + ((Int64(row) * cpr + chunk) * 32 + tid) * 4,
                        ld_shared_u32(hist_a + tid * 4),
                    )
                    st_shared_u32(hist_a + tid * 4, Uint32(0))
                # one claim in the row's candidate list for the chunk's keys
                n = ld_shared_u32(cnt_a)
                if tid == 0:
                    first = Uint32(0)
                    if n > Uint32(0):
                        first = atom_add_global(bin_a + (Int64(row) * 3 + 2) * 4, n)
                        over = first <= Uint32(CAND) and first + n > Uint32(CAND)
                        if n > Uint32(MARK_BUF):
                            # the chunk overflowed its buffer: force the exact path
                            f2 = atom_add_global(bin_a + (Int64(row) * 3 + 2) * 4, Uint32(CAND + 1))
                            if f2 <= Uint32(CAND):
                                over = True
                        if over:
                            # the claim that took the row's count past CAND (one per
                            # row: the count only grows): the row into the exact list
                            slot = atom_add_global(xl_a, Uint32(1))
                            st_global_u32(xl_a + (Int64(slot) + 1) * 4, Uint32(row))
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


EXACT_SCAN = 8  # key-order tie scan: 256-key steps loaded at once


class PrefillExact:
    """Select, part 1b (rows whose bound-bin candidates overflowed CAND, from
    PrefillMark's ``xl``): one CTA per (row, CHUNK_V-vector chunk). The row's exact
    threshold digit d comes from the chunks' histograms ``xh``; the chunk's keys
    above it go into the bitmap, and its digit-d ties by the stock rule (the
    ``take`` lowest key indices of the row): chunks with every tie taken (by the
    ties of the chunks before) set them all, the one chunk where the count
    crosses ``take`` scans its keys in key order. Chunks are whole 128-key tiles,
    so chunk order is key order."""

    def __init__(self, ctas: int, topk: int):
        self.ctas = ctas
        self.topk = topk

    @cute.jit
    def __call__(self, logits, row_base, row_len, row_bin, bitmap, xh, xl, gate, s):
        self.kernel(logits, row_base, row_len, row_bin, bitmap, xh, xl, gate).launch(
            grid=(self.ctas, 1, 1), block=(MARK_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, logits, row_base, row_len, row_bin, bitmap, xh, xl, gate):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        w = tid // 32
        lane = tid % 32
        smem = utils.SmemAllocator()
        # [0..7]: d, take - before (rem), mode; [8..23]: warp tie counts (2 sets)
        sb = smem.allocate_tensor(Uint32, cute.make_layout(32), 16)
        sb_a = sb.iterator.toint()
        cute.arch.griddepcontrol_wait()
        n_x = Int32(xl[0])
        if gate[0] == 0:
            n_x = Int32(0)
        stride = Int32(bitmap.shape[0]) // cutlass.max(Int32(row_len.shape[0]), Int32(1))
        cpr = (stride * 32 + CHUNK_V * 8 - 1) // (CHUNK_V * 8)
        logits_a = logits.iterator.toint()
        bm_a = bitmap.iterator.toint()
        xh_a = xh.iterator.toint()
        for u in cutlass.range(bid, n_x * cpr, self.ctas, unroll=1):
            row = Int32(xl[1 + u // cpr])
            chunk = u - (u // cpr) * cpr
            length = row_len[row]
            vectors = (length + TILE - 1) // TILE * (TILE // 8)
            v0 = chunk * CHUNK_V
            if v0 < vectors:
                address = logits_a + row_base[row] * 2
                brow = bm_a + Int64(row) * Int64(stride) * 4
                low = Uint32(row_bin[3 * row]) << Uint32(5)
                high = low + Uint32(32)
                if w == 0:
                    # row histogram (lane b: digit b) and the ties before this chunk
                    n_ch = (vectors + CHUNK_V - 1) // CHUNK_V
                    hrow = xh_a + (Int64(row) * cpr * 32 + lane) * 4
                    tot = Uint32(0)
                    pre = Uint32(0)
                    mine = Uint32(0)
                    for c in cutlass.range(0, n_ch, unroll=4):
                        h = ld_global_u32(hrow + Int64(c) * 128)
                        tot += h
                        if c < chunk:
                            pre += h
                        if c == chunk:
                            mine = h
                    need = Uint32(self.topk) - Uint32(row_bin[3 * row + 1])
                    d, above_d = low_digit(tot, need, lane)
                    ties = cute.arch.shuffle_sync(tot, Int32(d))
                    before = cute.arch.shuffle_sync(pre, Int32(d))
                    here = cute.arch.shuffle_sync(mine, Int32(d))
                    take = need - above_d
                    # mode 0: no tie of the chunk taken, 1: all, 2: the first rem
                    mode = Uint32(2)
                    rem = Uint32(0)
                    if ties == take or before + here <= take:
                        mode = Uint32(1)
                    elif before >= take:
                        mode = Uint32(0)
                    else:
                        rem = take - before
                    if lane == 0:
                        st_shared_u32(sb_a, d)
                        st_shared_u32(sb_a + 4, rem)
                        st_shared_u32(sb_a + 8, mode)
                cute.arch.sync_threads()
                d = ld_shared_u32(sb_a)
                rem = ld_shared_u32(sb_a + 4)
                mode = ld_shared_u32(sb_a + 8)
                # keys above the threshold digit (and every tie with mode 1)
                batch = []
                for k in cutlass.range_constexpr(MARK_LOADS):
                    v = v0 + k * MARK_THREADS + tid
                    words4 = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if v < vectors:
                        words4 = ldg_v4(address + Int64(v) * 16)
                    batch.append(words4)
                for k in cutlass.range_constexpr(MARK_LOADS):
                    v = v0 + k * MARK_THREADS + tid
                    mk = Uint32(0)
                    for e in cutlass.range_constexpr(8):
                        key = key_of(batch[k], e)
                        lo = key & Uint32(31)
                        if key >= low and key < high:
                            if lo > d:
                                mk = mk | Uint32(1 << e)
                            elif lo == d and mode == Uint32(1):
                                if key_index(v * 8 + e) < length:
                                    mk = mk | Uint32(1 << e)
                    if mk != Uint32(0) and v < vectors:
                        red_or_global(brow + (v >> 2) * 4, mk << Uint32((v & 3) * 8))
                if mode == Uint32(2):
                    # the crossing chunk: its first ``rem`` ties in key order (256
                    # keys per step, EXACT_SCAN steps loaded at once)
                    kb = v0 * 8
                    ke = cutlass.min(kb + CHUNK_V * 8, length)
                    run = Uint32(0)
                    for s0 in cutlass.range(kb, ke, MARK_THREADS * EXACT_SCAN, unroll=1):
                        if run < rem:
                            wds = []
                            for j in cutlass.range_constexpr(EXACT_SCAN):
                                k = s0 + j * MARK_THREADS + tid
                                wd = Uint32(0)
                                if k < ke:
                                    p = Uint32(position_of(k))
                                    wd = ld_global_u32(address + Int64(p >> Uint32(1)) * 4)
                                wds.append(wd)
                            for j in cutlass.range_constexpr(EXACT_SCAN):
                                k = s0 + j * MARK_THREADS + tid
                                p = Uint32(position_of(k))
                                hit = Uint32(0)
                                if k < ke:
                                    key = (keys16x2(wds[j]) >> ((p & Uint32(1)) * Uint32(16))) & Uint32(0xFFFF)
                                    if key >= low and key < high and (key & Uint32(31)) == d:
                                        hit = Uint32(1)
                                b = ballot(hit)
                                slot_a = sb_a + (8 + (j % 2) * 8 + w) * 4
                                if lane == 0:
                                    st_shared_u32(slot_a, popc(b))
                                cute.arch.sync_threads()
                                off = Uint32(0)
                                tot_s = Uint32(0)
                                for i in cutlass.range_constexpr(MARK_THREADS // 32):
                                    c = ld_shared_u32(sb_a + (8 + (j % 2) * 8 + i) * 4)
                                    if i < w:
                                        off += c
                                    tot_s += c
                                if hit != Uint32(0) and run + off + popc(b & lanemask_lt()) < rem:
                                    red_or_global(brow + (p >> Uint32(5)) * 4, Uint32(1) << (p & Uint32(31)))
                                run += tot_s
                cute.arch.sync_threads()


class PrefillFinish:
    """Select, part 2 (one warp per row): exact threshold's low 5 bits from the
    candidate list, the remaining keys of the bound bin (ties: the lowest
    positions) into the bitmap, then the bitmap's positions in order as key
    indices (-1 pads rows with at most ``topk`` keys). Rows whose candidates
    overflowed CAND were completed by PrefillExact (bitmap only here)."""

    def __init__(self, ctas: int, topk: int):
        self.ctas = ctas
        self.topk = topk

    @cute.jit
    def __call__(self, logits, row_base, row_len, row_bin, bitmap, cand, out, gate, s):
        self.kernel(logits, row_base, row_len, row_bin, bitmap, cand, out, gate).launch(
            grid=(self.ctas, 1, 1), block=(FIN_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, logits, row_base, row_len, row_bin, bitmap, cand, out, gate):
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
        n_rows = Int32(row_len.shape[0])
        if gate[0] == 0:
            n_rows = Int32(0)  # nothing to select (recovery with no flagged row)
        stride = Int32(bitmap.shape[0]) // cutlass.max(Int32(row_len.shape[0]), Int32(1))
        out_a = out.iterator.toint()
        bm_a = bitmap.iterator.toint()
        cand_a = cand.iterator.toint()
        logits_a = logits.iterator.toint()
        rpc = FIN_THREADS // 32
        for row in cutlass.range(bid * rpc + w, n_rows, self.ctas * rpc, unroll=1):
            length = row_len[row]
            orow = out_a + Int64(row) * K * 4
            if length < 0:
                pass  # row settled elsewhere (selective path)
            elif length <= K:
                for e in cutlass.range(lane, K, 32, unroll=1):
                    v = Int32(-1)
                    if e < length:
                        v = e
                    st_global_s32(orow + e * 4, v)
            else:
                brow = bm_a + Int64(row) * Int64(stride) * 4
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
                if not exact:
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
                    pass  # PrefillExact filled the bitmap
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
                                    if key_rank(ld_shared_u32(tie_buf_a + j * 4)) < key_rank(p):
                                        rank += Uint32(1)
                            if rank < take and nt <= Uint32(TIE_BUF):
                                red_or_global(
                                    brow + (p >> Uint32(5)) * 4, Uint32(1) << (p & Uint32(31))
                                )
                        if nt > Uint32(TIE_BUF):
                            # many ties: the take-th smallest key index among them by an
                            # exact 4 x 8-bit radix select over the candidate list (the
                            # overflowed tie buffer is the 256-bin histogram), then every
                            # tie at or below it. Key indices are distinct, so exactly
                            # ``take`` ties are kept: the lowest key indices (stock rule).
                            prefix = Uint32(0)
                            pmask = Uint32(0)
                            rem = take
                            for ps in cutlass.range_constexpr(4):
                                shift = 24 - 8 * ps
                                for i in cutlass.range_constexpr(8):
                                    st_shared_u32(tie_buf_a + (lane * 8 + i) * 4, Uint32(0))
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
                                        if x != Uint32(0xFFFFFFFF) and (x & Uint32(31)) == d:
                                            kr = key_rank(x >> Uint32(5))
                                            if (kr & pmask) == prefix:
                                                red_shared(
                                                    tie_buf_a + ((kr >> Uint32(shift)) & Uint32(255)) * 4, 1
                                                )
                                cute.arch.sync_warp()
                                # lane l holds bins 8 l .. 8 l + 7: the bin where the
                                # ascending running count reaches rem
                                s = Uint32(0)
                                for i in cutlass.range_constexpr(8):
                                    s += ld_shared_u32(tie_buf_a + (lane * 8 + i) * 4)
                                incl = warp_inclusive(s, lane)
                                running = incl - s
                                bsel = Uint32(0)
                                below = Uint32(0)
                                found = Uint32(0)
                                if running < rem and incl >= rem:
                                    for i in cutlass.range_constexpr(8):
                                        h = ld_shared_u32(tie_buf_a + (lane * 8 + i) * 4)
                                        if found == Uint32(0) and running + h >= rem:
                                            bsel = Uint32(lane * 8 + i)
                                            below = running
                                            found = Uint32(1)
                                        running += h
                                for i in cutlass.range_constexpr(5):
                                    bsel += cute.arch.shuffle_sync_bfly(bsel, 1 << i)
                                    below += cute.arch.shuffle_sync_bfly(below, 1 << i)
                                rem = rem - below
                                prefix = prefix | (bsel << Uint32(shift))
                                pmask = pmask | Uint32(255 << shift)
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
                                    p = x >> Uint32(5)
                                    if x != Uint32(0xFFFFFFFF) and (x & Uint32(31)) == d:
                                        if key_rank(p) <= prefix:
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
CS2_BINS = 2048  # histogram bins: one BF16 value each above the row threshold
CS2_CTAS_PER_SM = 3  # ~105 KiB shared memory per CTA
RP_THREADS = 1024  # RecPlan threads (one CTA)


@asm()
def atom_min_shared(address, value):
    return "red.shared.min.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def atom_max_shared(address, value):
    return "red.shared.max.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


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


@asm(Uint32, pure=True)
def key16_of_word(wd):
    """order16 of the low BF16 of wd, branch-free (-0 keyed as +0)."""
    code = (
        "{ .reg .pred z; .reg .b32 c, f; and.b32 c, $1, 0xFFFF; "
        "setp.eq.u32 z, c, 0x8000; selp.b32 c, 0, c, z; shr.u32 f, c, 15; "
        "mad.lo.u32 f, f, 0x7FFF, 0x8000; xor.b32 $0, c, f; }"
    )
    return code, "=r,r", (Uint32(wd),)


TIE_RANK = 256  # threshold ties ranked directly (more: the tile histogram path)
# Tile-histogram tie path: CS2_BINS position bins of 128 keys (rows of up to
# CS2_BINS * 128 = 262,144 keys) or, for longer rows, of 256 keys, as long as the
# position bitmap of the bins below the boundary bin fits the consumed candidate
# buffer after the bins and the boundary bin's ties (up to 256 of them).
TIE_BM_OFF = CS2_BINS + 2 * TILE  # words: bins, boundary-bin ties, then the bitmap
TIE_LONG_KEYS = (CAP - TIE_BM_OFF) * 32  # longest row of the 256-key bins


@asm()
def hist4_pos(hist_a, s, c0, c1, c2, c3, w0, w1, w2, w3, thr):
    """Histogram of one lane quad of candidate slot s when the row threshold thr
    is a positive BF16 (every candidate is a positive BF16 at or above it): one
    count per valid word (s below its lane count) in bin min(bits - thr,
    CS2_BINS - 1). Branch-free."""
    code = "{ .reg .pred p; .reg .b32 v, a; "
    for k in range(4):
        code += (
            f"setp.lt.u32 p, $1, ${2 + k}; and.b32 v, ${6 + k}, 0xFFFF; "
            f"sub.u32 v, v, $10; min.u32 v, v, {CS2_BINS - 1}; "
            f"mad.lo.u32 a, v, 4, $0; @p red.shared.add.u32 [a], 1; "
        )
    return code + "}", ",".join(["r"] * 11) + ",~{memory}", (
        Int32(hist_a), Uint32(s), Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3),
        Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(thr),
    )


STAGE = CS2_BINS + TIE_RANK  # staged output row (K words) + the ranked ties


def _quad_code(ties: bool, write: bool) -> str:
    """PTX over one lane quad of candidate slot $1 (counts $2-$5, words $6-$9) of a
    positive-threshold row (t = $10, the BF16 bits of the k-th value). Takes are
    the valid words above t (ties=True) or at or above t (ties=False); with
    ties=True the valid words at t are ties. count: $0 = $11 + takes + ties << 16.
    write: the key index of each take is stored at shared $12 and $12 advances one
    word ($0 returns it), each tie likewise at $13 ($1); pb = $14. No branches."""
    cmp = "gt" if ties else "ge"
    code = "{ .reg .pred v, g, e; .reg .b32 x, q, tq, rq, bq; "
    if write:
        code += "mov.b32 $0, $13; mov.b32 $1, $14; "
    else:
        code += "mov.b32 $0, $11; "
    for k in range(4):
        code += (f"setp.lt.u32 v, $1X, ${2 + k}X; and.b32 x, ${6 + k}X, 0xFFFF; "
                 f"setp.{cmp}.and.u32 g, x, $10X, v; ")
        if ties:
            code += "setp.eq.and.u32 e, x, $10X, v; "
        if write:
            # key = tile * 128 + 4 * ((column % 32 - tile) mod 32) + column // 32
            # (rotated columns, IndexerScore.publish_rot); pb = (4 lane) % 32 |
            # (lane >> 3) << 8, column = 4 lane + k
            code += (f"shr.u32 tq, ${6 + k}X, 16; shl.b32 q, tq, 7; "
                     f"and.b32 rq, $14X, 0xFF; add.u32 rq, rq, {k}; shr.u32 bq, tq, {ROT_SHIFT}; sub.u32 rq, rq, bq; "
                     "and.b32 rq, rq, 31; shl.b32 rq, rq, 2; add.u32 q, q, rq; "
                     "shr.u32 rq, $14X, 8; add.u32 q, q, rq; "
                     "@g st.shared.b32 [$0], q; @g add.u32 $0, $0, 4; ")
            if ties:
                code += "@e st.shared.b32 [$1], q; @e add.u32 $1, $1, 4; "
        else:
            code += "@g add.u32 $0, $0, 1; "
            if ties:
                code += "@e add.u32 $0, $0, 0x10000; "
    code += "}"
    # inputs are numbered after the outputs
    shift = 2 if write else 1
    out = []
    i = 0
    while i < len(code):
        if code[i] == "$" and code[i + 1].isdigit():
            j = i + 1
            while j < len(code) and code[j].isdigit():
                j += 1
            num = int(code[i + 1:j])
            if j < len(code) and code[j] == "X":
                out.append(f"${num - 1 + shift}")
                i = j + 1
                continue
            out.append(code[i:j])
            i = j
            continue
        out.append(code[i])
        i += 1
    return "".join(out)


@asm(Uint32)
def count4(s, c0, c1, c2, c3, w0, w1, w2, w3, t, n):
    """n + this lane quad's takes + (ties << 16) (ties = ties not all taken)."""
    return _quad_code(True, False), "=r," + ",".join(["r"] * 11), (
        Uint32(s), Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3),
        Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(t), Uint32(n))


@asm(Uint32)
def count4_all(s, c0, c1, c2, c3, w0, w1, w2, w3, t, n):
    """n + this lane quad's takes when every tie is taken."""
    return _quad_code(False, False), "=r," + ",".join(["r"] * 11), (
        Uint32(s), Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3),
        Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(t), Uint32(n))


@asm()
def write4(s, c0, c1, c2, c3, w0, w1, w2, w3, t, n, tk, te, pb):
    """Stage this lane quad's takes at tk and ties at te; returns both advanced."""
    return _quad_code(True, True), "=r,=r," + ",".join(["r"] * 14) + ",~{memory}", (
        Uint32(s), Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3),
        Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(t), Uint32(n),
        Int32(tk), Int32(te), Uint32(pb)), (Int32, Int32)


@asm()
def write4_all(s, c0, c1, c2, c3, w0, w1, w2, w3, t, n, tk, te, pb):
    """Stage this lane quad's takes (every tie taken) at tk; returns tk advanced."""
    return _quad_code(False, True), "=r,=r," + ",".join(["r"] * 14) + ",~{memory}", (
        Uint32(s), Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3),
        Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(t), Uint32(n),
        Int32(tk), Int32(te), Uint32(pb)), (Int32, Int32)


@asm()
def st_global_v4(address, a, b, c, d):
    code = "st.global.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "l,r,r,r,r,~{memory}", (Int64(address), Uint32(a), Uint32(b),
                                          Uint32(c), Uint32(d))


class CandSelect2:
    """Exact top-k of each row from its PUBLISH candidates, persistent and
    double-buffered: a CTA walks rows bid, bid + G, ...; while it selects row r,
    one bulk copy brings the next row's candidate lines (the first max lane count
    128-word slots of its region) into the other shared buffer, and the counts of
    the row after are already in flight. A thread reads four key lanes of a slot
    per 16-byte load (warp w takes the slots w, w + 8, ...). Every candidate is at
    or above the row's PUBLISH threshold thr, so one pass of a 2048-bin histogram
    of order16(key) - order16(thr) gives the exact k-th value t (BF16 values: one
    bin each); a row whose k-th value is not resolved there (keep-all rows, a span
    of more than 2048 values) takes a min/max-ranged histogram and, when the range
    needs it, a second pass over the low bits. Keys above t are emitted (warp
    ballots at per-warp bases from a count pass and a block scan: no atomics, the
    output order is a fixed function of the candidates, so the order-sensitive
    sparse MLA reproduces); of the ties at t the lowest key indices are emitted
    at their rank among the ties, found by a histogram of their tiles. Rows with fewer
    candidates than min(k, row length), an overflowed lane or more than CS2_BINS
    ties are flagged in row_bin[3 r + 1] (recomputed by the caller). Output order
    is unspecified. Rows with a positive threshold (every candidate a positive
    BF16, ordered by its bits) take branch-free histogram and emission passes
    (hist4_pos, emit4_pos); up to TIE_RANK ties are ranked directly."""

    def __init__(self, topk: int, ctas: int, skeleton: bool = False):
        self.topk = topk
        self.ctas = ctas
        self.skeleton = skeleton

    @cute.jit
    def __call__(self, cand, row_len, row_bin, out, s):
        self.kernel(cand, row_len, row_bin, out).launch(
            grid=(self.ctas, 1, 1), block=(CS_THREADS, 1, 1), stream=s, use_pdl=True,
        )

    @cute.jit
    def quad(self, buf_a, cnt4, q, s):
        """Lanes 4q .. 4q + 3 of slot s: (valid, key index, order16 key) each."""
        wd = lds_v4(buf_a + (s * TILE + 4 * q) * 4)
        res = []
        for k in cutlass.range_constexpr(4):
            lane_k = 4 * q + k
            valid = Uint32(0)
            if s < Int32(cnt4[k]):
                valid = Uint32(1)
            # rotated columns (IndexerScore.publish_rot): column 32 q + c of tile t
            # holds key lane (q, (c - t) mod 32) = key 4 ((c - t) mod 32) + q
            tile_k = Int32(wd[k] >> Uint32(16))
            pos = tile_k * TILE + 4 * (((lane_k % 32) - (tile_k >> ROT_SHIFT)) & 31) + lane_k // 32
            res.append((valid, pos, key16_of_word(wd[k])))
        return res

    @cute.jit
    def zero_hist(self, hist_a, tid):
        for i in cutlass.range_constexpr(CS2_BINS // CS_THREADS):
            st_shared_u32(hist_a + (i * CS_THREADS + tid) * 4, Uint32(0))

    @cute.jit
    def hist(self, buf_a, hist_a, cnt4, q, w, mc, k0, shift, level2, bsel):
        """Histogram of min((key - k0) >> shift, BINS - 1); level2: of the low
        shift bits of the keys whose (key - k0) >> shift is bsel."""
        mask = (Uint32(1) << shift) - Uint32(1)
        for s in cutlass.range(w, mc, CS_THREADS // 32, unroll=2):
            for valid, pos, key in self.quad(buf_a, cnt4, q, s):
                d = key - k0
                if valid != Uint32(0):
                    if level2 == Uint32(0):
                        red_shared(hist_a + (cutlass.min(d >> shift, Uint32(CS2_BINS - 1)) << Uint32(2)), 1)
                    elif (d >> shift) == bsel:
                        red_shared(hist_a + ((d & mask) << Uint32(2)), 1)

    @cute.jit
    def scan(self, hist_a, scr_a, need, tid, w, lane):
        """The highest bin b with count(bins >= b) >= need: (b, count above b,
        count of b). Every thread calls it (two block barriers)."""
        hi = Int32(CS2_BINS - 1) - 8 * tid
        sm = Uint32(0)
        for k in cutlass.range_constexpr(8):
            sm += ld_shared_u32(hist_a + (hi - k) * 4)
        incl = warp_inclusive(sm, lane)
        if lane == 31:
            st_shared_u32(scr_a + w * 4, incl)
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(CS_THREADS // 32):
            if k < w:
                incl += ld_shared_u32(scr_a + k * 4)
        if incl >= need and incl - sm < need:
            run = incl - sm
            found = Int32(0)
            for k in cutlass.range_constexpr(8):
                c = ld_shared_u32(hist_a + (hi - k) * 4)
                if found == 0 and run + c >= need:
                    st_shared_u32(scr_a + 32, Uint32(hi - k))
                    st_shared_u32(scr_a + 36, run)
                    st_shared_u32(scr_a + 40, c)
                    found = Int32(1)
                run += c
        cute.arch.sync_threads()
        return (ld_shared_u32(scr_a + 32), ld_shared_u32(scr_a + 36),
                ld_shared_u32(scr_a + 40))

    @cute.jit
    def ballot_count(self, flags, n):
        """n + the flagged lanes of the warp (warp-uniform; all lanes call it)."""
        for f in flags:
            n += popc(ballot(f))
        return n

    @cute.jit
    def warp_prefix(self, n, scr_a, w, lane):
        """The flagged count of the warps before w, from each warp's (warp-uniform)
        count n. Every thread calls it (one block barrier; scr_a[0, WARPS) words)."""
        if lane == 0:
            st_shared_u32(scr_a + w * 4, n)
        cute.arch.sync_threads()
        pre = Uint32(0)
        for k in cutlass.range_constexpr(CS_THREADS // 32):
            if k < w:
                pre += ld_shared_u32(scr_a + k * 4)
        return pre

    @cute.jit
    def emit4(self, flags, poss, out, orow, lane, base):
        """Write each flagged pos at output slots base, base + 1, ... in (quad,
        lane) order; returns base advanced (warp-uniform; all lanes call it). With
        bases from warp_prefix over the same flags the emission order is a fixed
        function of the candidate layout: no atomics, so a row's output (and the
        sparse MLA's key order, which its online softmax depends on) is
        reproducible."""
        bs = [ballot(f) for f in flags]
        lt = lanemask_lt()
        for k in cutlass.range_constexpr(4):
            if flags[k] != Uint32(0):
                out[orow + Int64(base + popc(bs[k] & lt))] = poss[k]
            base += popc(bs[k])
        return base

    @cute.jit
    def stage_counts(self, c, slot, cnt_a, part_a, tid, w, lane):
        """Lane tid's count of the next row (threads < TILE): clamped count in
        shared memory, per-warp (sum, max, overflow) partials."""
        ovf = Uint32(0)
        if c >= Uint32(SLOTS):
            ovf = Uint32(1)
            c = Uint32(SLOTS)
        st_shared_u32(cnt_a + (slot * TILE + tid) * 4, c)
        sm = c
        mx = c
        for i in cutlass.range_constexpr(5):
            sm += cute.arch.shuffle_sync_bfly(sm, 1 << i)
            mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, 1 << i))
            ovf = cutlass.max(ovf, cute.arch.shuffle_sync_bfly(ovf, 1 << i))
        if lane == 0:
            p = part_a + (slot * 4 + w) * 16
            st_shared_u32(p, sm)
            st_shared_u32(p + 4, mx)
            st_shared_u32(p + 8, ovf)

    @cute.kernel
    def kernel(self, cand, row_len, row_bin, out):
        K = self.topk
        G = self.ctas
        WARPS = CS_THREADS // 32
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        w = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        q = tid % 32
        smem = utils.SmemAllocator()
        buf = smem.allocate_tensor(Uint32, cute.make_layout(2 * CAP), 128)
        buf_a = buf.iterator.toint()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(STAGE), 16)
        hist_a = hist.iterator.toint()  # the threshold ties alias it after the scan
        cnts = smem.allocate_tensor(Uint32, cute.make_layout(2 * TILE), 16)
        cnt_a = cnts.iterator.toint()
        part = smem.allocate_tensor(Uint32, cute.make_layout(2 * 4 * 4), 16)
        part_a = part.iterator.toint()
        # 0..7 warp scan totals, 8..10 scan result, 11 boundary-tile ties, 12 -,
        # 13 tie slot, 14 key min, 15 key max
        scr = smem.allocate_tensor(Uint32, cute.make_layout(16), 16)
        scr_a = scr.iterator.toint()
        bars = smem.allocate_tensor(Int64, cute.make_layout(2), 8)
        bar = bars.iterator
        cand_a = cand.iterator.toint()
        n_rows = row_len.shape[0]
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.griddepcontrol_wait()
        nxt = Uint32(0)
        if tid < TILE and bid < n_rows:
            nxt = Uint32(cand[Int64(bid) * REGION + tid])
        if tid < TILE:
            self.stage_counts(nxt, 0, cnt_a, part_a, tid, w, lane)
        cute.arch.sync_threads()
        if tid == 0 and bid < n_rows:
            mc = Uint32(0)
            for k in cutlass.range_constexpr(4):
                mc = cutlass.max(mc, ld_shared_u32(part_a + k * 16 + 4))
            cute.arch.mbarrier_arrive_and_expect_tx(bar, Int32(mc) * (TILE * 4))
            if mc != Uint32(0):
                bulk_g2s(buf_a, cand_a + (Int64(bid) * REGION + TILE) * 4, bar,
                         Int32(mc) * (TILE * 4))
        if tid < TILE and bid + G < n_rows:
            nxt = Uint32(cand[Int64(bid + G) * REGION + tid])
        # row length and PUBLISH threshold, one row ahead (their loads stay off
        # the row's critical path)
        n_nx = Int32(0)
        thr_nx = Uint32(0)
        if bid < n_rows:
            n_nx = row_len[bid]
            thr_nx = Uint32(row_bin[3 * bid])
        lt_mask = lanemask_lt()
        # this lane quad's key-lane base (rotated columns: _quad_code)
        pb0 = Uint32((4 * lane) % 32) | (Uint32(lane // 8) << Uint32(8))
        out_base = Int64(out.iterator.toint())
        n_it = (n_rows - bid + G - 1) // G
        for it in cutlass.range(n_it, unroll=1):
            row = bid + it * G
            b = it % 2
            nb = 1 - b
            n = n_nx
            thr_row = thr_nx & Uint32(0xFFFF)
            if row + G < n_rows:
                n_nx = row_len[row + G]
                thr_nx = Uint32(row_bin[3 * (row + G)])
            # next row: counts -> shared, then its bulk copy into the other buffer
            if tid < TILE:
                self.stage_counts(nxt, nb, cnt_a, part_a, tid, w, lane)
            self.zero_hist(hist_a, tid)
            if tid == 0:
                st_shared_u32(scr_a + 44, Uint32(0))
                st_shared_u32(scr_a + 52, Uint32(0))
                st_shared_u32(scr_a + 56, Uint32(0xFFFF))
                st_shared_u32(scr_a + 60, Uint32(0))
            cute.arch.sync_threads()
            if tid == 0 and row + G < n_rows:
                mcn = Uint32(0)
                for k in cutlass.range_constexpr(4):
                    mcn = cutlass.max(mcn, ld_shared_u32(part_a + (nb * 4 + k) * 16 + 4))
                fence_async_shared()
                cute.arch.mbarrier_arrive_and_expect_tx(bar + nb, Int32(mcn) * (TILE * 4))
                if mcn != Uint32(0):
                    bulk_g2s(buf_a + nb * (CAP * 4),
                             cand_a + (Int64(row + G) * REGION + TILE) * 4, bar + nb,
                             Int32(mcn) * (TILE * 4))
            if tid < TILE and row + 2 * G < n_rows:
                nxt = Uint32(cand[Int64(row + 2 * G) * REGION + tid])
            # this row
            total = Uint32(0)
            ovf = Uint32(0)
            mcu = Uint32(0)
            for k in cutlass.range_constexpr(4):
                total += ld_shared_u32(part_a + (b * 4 + k) * 16)
                mcu = cutlass.max(mcu, ld_shared_u32(part_a + (b * 4 + k) * 16 + 4))
                ovf = cutlass.max(ovf, ld_shared_u32(part_a + (b * 4 + k) * 16 + 8))
            mc = Int32(mcu)
            cnt4 = lds_v4(cnt_a + (b * TILE + 4 * q) * 4)
            rb_a = buf_a + b * (CAP * 4)
            orow = Int64(row) * K
            cute.arch.mbarrier_wait(bar + b, (it // 2) % 2)
            kk = cutlass.min(n, Int32(K))
            bad = Int32(0)
            if ovf != Uint32(0) or Int32(total) < kk:
                bad = Int32(1)
            if tid == 0:
                row_bin[3 * row + 1] = bad
            if cutlass.const_expr(self.skeleton):
                acc = Uint32(0)
                for s in cutlass.range(w, mc, WARPS, unroll=2):
                    for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                        acc += key & valid
                for i in cutlass.range(tid, K, CS_THREADS, unroll=1):
                    out[orow + Int64(i)] = Int32(acc)
            else:
                if bad == 0:
                    if Int32(total) <= Int32(K):
                        nw = Uint32(0)
                        for s in cutlass.range(w, mc, WARPS, unroll=1):
                            fl = []
                            for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                                fl.append(valid)
                            nw = self.ballot_count(fl, nw)
                        base = self.warp_prefix(nw, scr_a, w, lane)
                        for s in cutlass.range(w, mc, WARPS, unroll=1):
                            fl = []
                            ps = []
                            for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                                fl.append(valid)
                                ps.append(pos)
                            base = self.emit4(fl, ps, out, orow, lane, base)
                        for i in cutlass.range(Int32(total) + tid, K, CS_THREADS, unroll=1):
                            out[orow + Int64(i)] = Int32(-1)
                    else:
                        thr = thr_row
                        k0 = key16_of_word(thr)
                        shift = Uint32(0)
                        general = Uint32(0)
                        bb = Uint32(0)
                        above = Uint32(0)
                        m = Uint32(0)
                        # a positive finite threshold: every candidate is a positive
                        # BF16 at or above it, ordered by its raw bits
                        fast = Uint32(0)
                        if thr != Uint32(0) and thr < Uint32(0x7F80):
                            fast = Uint32(1)
                        if thr == Uint32(0xFF80):
                            general = Uint32(1)
                        else:
                            if fast != Uint32(0):
                                for s in cutlass.range(w, mc, WARPS, unroll=2):
                                    wd = lds_v4(rb_a + (s * TILE + 4 * q) * 4)
                                    hist4_pos(hist_a, s, cnt4[0], cnt4[1], cnt4[2], cnt4[3],
                                              wd[0], wd[1], wd[2], wd[3], thr)
                            else:
                                self.hist(rb_a, hist_a, cnt4, q, w, mc, k0, shift, Uint32(0), Uint32(0))
                            cute.arch.sync_threads()
                            bb, above, m = self.scan(hist_a, scr_a, Uint32(K), tid, w, lane)
                            if bb == Uint32(CS2_BINS - 1):
                                general = Uint32(1)
                        if general != Uint32(0):
                            kmin = Uint32(0xFFFF)
                            kmax = Uint32(0)
                            for s in cutlass.range(w, mc, WARPS, unroll=1):
                                for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                                    if valid != Uint32(0):
                                        kmin = cutlass.min(kmin, key)
                                        kmax = cutlass.max(kmax, key)
                            for i in cutlass.range_constexpr(5):
                                kmin = cutlass.min(kmin, cute.arch.shuffle_sync_bfly(kmin, 1 << i))
                                kmax = cutlass.max(kmax, cute.arch.shuffle_sync_bfly(kmax, 1 << i))
                            self.zero_hist(hist_a, tid)
                            if lane == 0:
                                atom_min_shared(scr_a + 56, kmin)
                                atom_max_shared(scr_a + 60, kmax)
                            cute.arch.sync_threads()
                            k0 = ld_shared_u32(scr_a + 56)
                            span = ld_shared_u32(scr_a + 60) - k0
                            if span > Uint32(CS2_BINS - 1):
                                shift = Uint32(21) - clz(span)
                            self.hist(rb_a, hist_a, cnt4, q, w, mc, k0, shift, Uint32(0), Uint32(0))
                            cute.arch.sync_threads()
                            bb, above, m = self.scan(hist_a, scr_a, Uint32(K), tid, w, lane)
                        t = k0 + (bb << shift)
                        if shift != Uint32(0):
                            self.zero_hist(hist_a, tid)
                            cute.arch.sync_threads()
                            self.hist(rb_a, hist_a, cnt4, q, w, mc, k0, shift, Uint32(1), bb)
                            cute.arch.sync_threads()
                            b2, above2, m2 = self.scan(hist_a, scr_a, Uint32(K) - above, tid, w, lane)
                            t = t + b2
                            above = above + above2
                            m = m2
                        need = Uint32(K) - above
                        all_ties = Uint32(0)
                        if m == need:
                            all_ties = Uint32(1)
                        staged = Uint32(0)
                        if fast != Uint32(0) and general == Uint32(0):
                            if all_ties != Uint32(0) or m <= Uint32(TIE_RANK):
                                staged = Uint32(1)
                        if staged != Uint32(0):
                            # t - k0 = bb: the k-th value's BF16 bits are thr + bb.
                            # Counts, a block scan, then every thread stages its
                            # takes (and ties) contiguously in shared memory; the
                            # ranked ties fill the row's tail; one coalesced copy out
                            t16 = thr + bb
                            cnt = Uint32(0)
                            if all_ties != Uint32(0):
                                for s in cutlass.range(w, mc, WARPS, unroll=1):
                                    wd = lds_v4(rb_a + (s * TILE + 4 * q) * 4)
                                    cnt = count4_all(s, cnt4[0], cnt4[1], cnt4[2], cnt4[3],
                                                     wd[0], wd[1], wd[2], wd[3], t16, cnt)
                            else:
                                for s in cutlass.range(w, mc, WARPS, unroll=1):
                                    wd = lds_v4(rb_a + (s * TILE + 4 * q) * 4)
                                    cnt = count4(s, cnt4[0], cnt4[1], cnt4[2], cnt4[3],
                                                 wd[0], wd[1], wd[2], wd[3], t16, cnt)
                            incl = warp_inclusive(cnt, lane)
                            if lane == 31:
                                st_shared_u32(scr_a + w * 4, incl)
                            cute.arch.sync_threads()
                            pre = incl - cnt
                            tot = Uint32(0)
                            for k in cutlass.range_constexpr(WARPS):
                                wt = ld_shared_u32(scr_a + k * 4)
                                tot += wt
                                if k < w:
                                    pre += wt
                            n_take = tot & Uint32(0xFFFF)
                            tie_a = hist_a + ((n_take + Uint32(3)) & Uint32(0xFFFFFFFC)) * 4
                            tk = Int32(hist_a + (pre & Uint32(0xFFFF)) * 4)
                            te = Int32(tie_a + (pre >> Uint32(16)) * 4)
                            if all_ties != Uint32(0):
                                for s in cutlass.range(w, mc, WARPS, unroll=1):
                                    wd = lds_v4(rb_a + (s * TILE + 4 * q) * 4)
                                    tk, te = write4_all(s, cnt4[0], cnt4[1], cnt4[2], cnt4[3],
                                                        wd[0], wd[1], wd[2], wd[3], t16,
                                                        Uint32(0), tk, te, pb0)
                            else:
                                for s in cutlass.range(w, mc, WARPS, unroll=1):
                                    wd = lds_v4(rb_a + (s * TILE + 4 * q) * 4)
                                    tk, te = write4(s, cnt4[0], cnt4[1], cnt4[2], cnt4[3],
                                                    wd[0], wd[1], wd[2], wd[3], t16,
                                                    Uint32(0), tk, te, pb0)
                            cute.arch.sync_threads()
                            if all_ties == Uint32(0):
                                # the need lowest key indices of the m ties: tie of
                                # rank r (ties of smaller key index) -> slot above + r
                                p = Uint32(0)
                                rank = Uint32(K)
                                if tid < Int32(m):
                                    p = ld_shared_u32(tie_a + tid * 4)
                                    rank = Uint32(0)
                                    for x in cutlass.range(0, Int32(m), 4, unroll=2):
                                        v4 = lds_v4(tie_a + x * 4)
                                        for k in cutlass.range_constexpr(4):
                                            if x + k < Int32(m) and v4[k] < p:
                                                rank += Uint32(1)
                                cute.arch.sync_threads()
                                if rank < need:
                                    st_shared_u32(hist_a + (above + rank) * 4, p)
                                cute.arch.sync_threads()
                            orow_a = out_base + orow * 4
                            for i in cutlass.range(tid * 4, K, CS_THREADS * 4, unroll=1):
                                v4 = lds_v4(hist_a + i * 4)
                                st_global_v4(orow_a + i * 4, v4[0], v4[1], v4[2], v4[3])
                        else:
                            # takes (keys above t; every tie when all are taken) at
                            # [0, above (+ m)) in a fixed order; ties at t placed by
                            # their rank among the ties (below)
                            nw = Uint32(0)
                            for s in cutlass.range(w, mc, WARPS, unroll=1):
                                fl = []
                                for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                                    take = Uint32(0)
                                    if valid != Uint32(0):
                                        if key > t:
                                            take = Uint32(1)
                                        elif key == t and all_ties != Uint32(0):
                                            take = Uint32(1)
                                    fl.append(take)
                                nw = self.ballot_count(fl, nw)
                            base = self.warp_prefix(nw, scr_a, w, lane)
                            for s in cutlass.range(w, mc, WARPS, unroll=1):
                                fl = []
                                ps = []
                                for valid, pos, key in self.quad(rb_a, cnt4, q, s):
                                    take = Uint32(0)
                                    if valid != Uint32(0):
                                        if key > t:
                                            take = Uint32(1)
                                        elif key == t:
                                            if all_ties != Uint32(0):
                                                take = Uint32(1)
                                            else:
                                                jt = atom_add_shared(scr_a + 52, Uint32(1))
                                                if jt < Uint32(CS2_BINS):
                                                    st_shared_u32(hist_a + jt * 4, Uint32(pos))
                                    fl.append(take)
                                    ps.append(pos)
                                base = self.emit4(fl, ps, out, orow, lane, base)
                        cute.arch.sync_threads()
                        n_tiles = (n + TILE - 1) // TILE
                        if staged != Uint32(0):
                            pass
                        elif all_ties == Uint32(0) and m <= Uint32(TIE_RANK):
                            # the need lowest key indices of the m ties: rank each
                            # tie by the ties of smaller key index
                            for jt in cutlass.range(tid, Int32(m), CS_THREADS, unroll=1):
                                p = ld_shared_u32(hist_a + jt * 4)
                                rank = Uint32(0)
                                for x in cutlass.range(0, Int32(m), 4, unroll=2):
                                    v4 = lds_v4(hist_a + x * 4)
                                    for k in cutlass.range_constexpr(4):
                                        if x + k < Int32(m) and v4[k] < p:
                                            rank += Uint32(1)
                                if rank < need:
                                    out[orow + Int64(above + rank)] = Int32(p)
                        elif all_ties == Uint32(0) and (
                            m > Uint32(CS2_BINS)
                            or (n_tiles > Int32(CS2_BINS) and n > Int32(TIE_LONG_KEYS))
                        ):
                            if tid == 0:
                                row_bin[3 * row + 1] = Int32(1)
                        elif all_ties == Uint32(0):
                            # the need lowest tie key indices: a histogram of the ties'
                            # tiles (in this row's consumed buffer, newest tile in the
                            # lowest bin) finds the boundary tile T; ties in older
                            # tiles are taken, those in T ranked by key index
                            for i in cutlass.range_constexpr(CS2_BINS // CS_THREADS):
                                st_shared_u32(rb_a + (i * CS_THREADS + tid) * 4, Uint32(0))
                            cute.arch.sync_threads()
                            # bin = 128 keys (one tile) up to CS2_BINS tiles, else 256 keys
                            sh = Uint32(7)
                            if n_tiles > Int32(CS2_BINS):
                                sh = Uint32(8)
                            bw = Uint32(1) << (sh - Uint32(5))  # bitmap words per bin
                            for jt in cutlass.range(tid, Int32(m), CS_THREADS, unroll=1):
                                p = ld_shared_u32(hist_a + jt * 4)
                                red_shared(rb_a + (Uint32(CS2_BINS - 1) - (p >> sh)) * 4, 1)
                            cute.arch.sync_threads()
                            tb, below, _ = self.scan(rb_a, scr_a, need, tid, w, lane)
                            tile_t = Uint32(CS2_BINS - 1) - tb
                            # every tie of an older tile is taken, at slot above +
                            # (ties of older tiles than its own: an exclusive suffix
                            # sum of the bins, in place) + (ties of its tile with a
                            # smaller key index: a 128-bit mask per tile)
                            bm_a = rb_a + TIE_BM_OFF * 4
                            for i in cutlass.range(tid, Int32(tile_t * bw), CS_THREADS, unroll=1):
                                st_shared_u32(bm_a + i * 4, Uint32(0))
                            hi = Int32(CS2_BINS - 1) - 8 * tid
                            sm = Uint32(0)
                            for k in cutlass.range_constexpr(8):
                                sm += ld_shared_u32(rb_a + (hi - k) * 4)
                            incl = warp_inclusive(sm, lane)
                            run = incl - sm + self.warp_prefix(
                                cute.arch.shuffle_sync(incl, 31), scr_a, w, lane
                            )
                            for k in cutlass.range_constexpr(8):
                                c = ld_shared_u32(rb_a + (hi - k) * 4)
                                st_shared_u32(rb_a + (hi - k) * 4, run)
                                run += c
                            for jt in cutlass.range(tid, Int32(m), CS_THREADS, unroll=1):
                                p = ld_shared_u32(hist_a + jt * 4)
                                if (p >> sh) < tile_t:
                                    red_or_shared(
                                        bm_a + ((p >> Uint32(5)) << Uint32(2)),
                                        Uint32(1) << (p & Uint32(31)),
                                    )
                                elif (p >> sh) == tile_t:
                                    o = atom_add_shared(scr_a + 44, Uint32(1))
                                    st_shared_u32(rb_a + (CS2_BINS + o) * 4, p)
                            cute.arch.sync_threads()
                            for jt in cutlass.range(tid, Int32(m), CS_THREADS, unroll=1):
                                p = ld_shared_u32(hist_a + jt * 4)
                                tl = p >> sh
                                if tl < tile_t:
                                    r = ld_shared_u32(rb_a + (Uint32(CS2_BINS - 1) - tl) * 4)
                                    wq = (p >> Uint32(5)) & (bw - Uint32(1))
                                    for k in cutlass.range_constexpr(8):
                                        if Uint32(k) < bw:
                                            word = ld_shared_u32(bm_a + ((tl * bw) + Uint32(k)) * 4)
                                            if Uint32(k) < wq:
                                                r += popc(word)
                                            elif Uint32(k) == wq:
                                                r += popc(word & ((Uint32(1) << (p & Uint32(31))) - Uint32(1)))
                                    out[orow + Int64(above + r)] = Int32(p)
                            nt = Int32(ld_shared_u32(scr_a + 44))
                            if tid < nt:
                                p = ld_shared_u32(rb_a + (CS2_BINS + tid) * 4)
                                rank = Uint32(0)
                                for x in cutlass.range(nt, unroll=4):
                                    if ld_shared_u32(rb_a + (CS2_BINS + x) * 4) < p:
                                        rank += Uint32(1)
                                if rank < need - below:
                                    out[orow + Int64(above + below + rank)] = Int32(p)
            cute.arch.sync_threads()


@functools.cache
def _cand_select2_kernel(topk, ctas, skeleton=False):
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
    return cute.compile(CandSelect2(topk, ctas, skeleton), *args, options="--enable-tvm-ffi")


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


class RecPlan:
    """Recovery plan of the selective path (one CTA): the items holding a row that
    CandSelect flagged (row_bin[3 r + 1]), compacted in order, their count, and
    the full-path select's row lengths (-1: the row is settled)."""

    @cute.jit
    def __call__(self, row_bin, items, row_len, items_rec, n_active, rec_len, s):
        self.kernel(row_bin, items, row_len, items_rec, n_active, rec_len).launch(
            grid=(1, 1, 1), block=(RP_THREADS, 1, 1), stream=s,
        )

    @cute.kernel
    def kernel(self, row_bin, items, row_len, items_rec, n_active, rec_len):
        tid, _, _ = cute.arch.thread_idx()
        w = tid // 32
        lane = tid % 32
        smem = utils.SmemAllocator()
        buf = smem.allocate_tensor(Uint32, cute.make_layout(RP_THREADS // 32 + 1), 16)
        sum_a = buf.iterator.toint()
        tot_a = sum_a + (RP_THREADS // 32) * 4
        t = row_len.shape[0]
        n = items.shape[0]
        for i in cutlass.range(tid, t, RP_THREADS, unroll=1):
            v = Int32(-1)
            if row_bin[3 * i + 1] != 0:
                v = row_len[i]
            rec_len[i] = v
        base = Uint32(0)
        for c0 in cutlass.range(0, n, RP_THREADS, unroll=1):
            item = c0 + tid
            act = Uint32(0)
            if item < n:
                tok0 = items[item, 0]
                ntok = items[item, 1]
                for k in cutlass.range_constexpr(BQ):
                    if k < ntok:
                        if row_bin[3 * (tok0 + k) + 1] != 0:
                            act = Uint32(1)
            b = ballot(act)
            if lane == 0:
                st_shared_u32(sum_a + w * 4, popc(b))
            cute.arch.sync_threads()
            if w == 0:
                v = ld_shared_u32(sum_a + lane * 4)
                inc = warp_inclusive(v, lane)
                st_shared_u32(sum_a + lane * 4, inc - v)
                if lane == 31:
                    st_shared_u32(tot_a, inc)
            cute.arch.sync_threads()
            if act != Uint32(0):
                dst = Int32(base + ld_shared_u32(sum_a + w * 4) + popc(b & lanemask_lt()))
                for k in cutlass.range_constexpr(4):
                    items_rec[dst, k] = items[item, k]
            base = base + ld_shared_u32(tot_a)
            cute.arch.sync_threads()
        if tid == 0:
            n_active[0] = Int32(base)


@functools.cache
def _rec_plan_kernel():
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    t, n = cute.sym_int(), cute.sym_int()
    args = (
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((n, 4), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        fake((n, 4), dtype=cutlass.Int32),
        fake((1,), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(RecPlan(), *args, options="--enable-tvm-ffi")


SPLIT_MIN_PAIRS = 16  # tile pairs per key split at least (4096 keys)


class SplitPlan:
    """Key splits of the full-path score (one CTA): when the ``n`` items (device
    count) do not fill the GPU (n < sms), each item's tile pairs are cut into
    splits of P = max(SPLIT_MIN_PAIRS, ceil(total pairs / (sms - n))) pairs (all
    splits fit one wave), so the call takes about one split's time instead of one
    whole item's; split items
    send their rows' histograms to slot i (the item's index; ``par``: its token0
    and token count, 0 tokens: no slot) for RowBound. Otherwise every item runs
    whole. Output items [*, 7]: token0, ntok, key0, ntiles, tile lo, tile hi,
    slot."""

    def __init__(self, sms: int):
        self.sms = sms

    @cute.jit
    def __call__(self, items, n_in, items_x, n_x, par, s):
        self.kernel(items, n_in, items_x, n_x, par).launch(
            grid=(1, 1, 1), block=(RP_THREADS, 1, 1), stream=s,
        )

    @cute.kernel
    def kernel(self, items, n_in, items_x, n_x, par):
        tid, _, _ = cute.arch.thread_idx()
        w = tid // 32
        lane = tid % 32
        smem = utils.SmemAllocator()
        buf = smem.allocate_tensor(Uint32, cute.make_layout(RP_THREADS // 32 + 1), 16)
        sum_a = buf.iterator.toint()
        tot_a = sum_a + (RP_THREADS // 32) * 4
        n = n_in[0]
        if n >= self.sms:
            for i in cutlass.range(tid, n, RP_THREADS, unroll=1):
                nt = items[i, 3]
                for k in cutlass.range_constexpr(4):
                    items_x[i, k] = items[i, k]
                items_x[i, 4] = Int32(0)
                items_x[i, 5] = nt + (nt & 1)
                items_x[i, 6] = Int32(-1)
            if tid == 0:
                n_x[0] = n
        else:
            # n < sms <= RP_THREADS: one thread per item
            pairs = Uint32(0)
            if tid < n:
                pairs = Uint32((items[tid, 3] + 1) // 2)
            # block sum of the pairs
            inc = warp_inclusive(pairs, lane)
            if lane == 31:
                st_shared_u32(sum_a + w * 4, inc)
            cute.arch.sync_threads()
            if w == 0:
                v = ld_shared_u32(sum_a + lane * 4)
                vi = warp_inclusive(v, lane)
                if lane == 31:
                    st_shared_u32(tot_a, vi)
            cute.arch.sync_threads()
            total = ld_shared_u32(tot_a)
            cute.arch.sync_threads()
            # sum over items of ceil(pairs / per) <= total / per + n <= sms: every
            # split runs in one wave, none longer than per pairs (never slower
            # than whole items, which take max pairs)
            room = Uint32(self.sms) - Uint32(n)
            per = cutlass.max(Uint32(SPLIT_MIN_PAIRS), (total + room - Uint32(1)) // room)
            ns = Uint32(0)
            if tid < n:
                ns = cutlass.max((pairs + per - Uint32(1)) // per, Uint32(1))
            # exclusive scan of the split counts
            inc = warp_inclusive(ns, lane)
            if lane == 31:
                st_shared_u32(sum_a + w * 4, inc)
            cute.arch.sync_threads()
            if w == 0:
                v = ld_shared_u32(sum_a + lane * 4)
                vi = warp_inclusive(v, lane)
                st_shared_u32(sum_a + lane * 4, vi - v)
                if lane == 31:
                    st_shared_u32(tot_a, vi)
            cute.arch.sync_threads()
            if tid < n:
                dst = Int32(ld_shared_u32(sum_a + w * 4) + inc - ns)
                nt = items[tid, 3]
                ntp = nt + (nt & 1)
                slot = Int32(-1)
                if ns > Uint32(1):
                    slot = Int32(tid)
                    par[tid, 0] = items[tid, 0]
                    par[tid, 1] = items[tid, 1]
                else:
                    par[tid, 0] = Int32(0)
                    par[tid, 1] = Int32(0)
                for k0 in cutlass.range(Int32(ns), unroll=1):
                    # split k0 from the newest tiles down: [hi - 2 per, hi)
                    hi = ntp - Int32(2 * per) * k0
                    lo = cutlass.max(hi - Int32(2 * per), Int32(0))
                    for k in cutlass.range_constexpr(4):
                        items_x[dst + k0, k] = items[tid, k]
                    items_x[dst + k0, 4] = lo
                    items_x[dst + k0, 5] = hi
                    items_x[dst + k0, 6] = slot
            if tid == 0:
                n_x[0] = Int32(ld_shared_u32(tot_a))


class RowBound:
    """Row bounds of key-split items' rows (one 128-thread CTA per row): the
    IndexerScore mode-0 row_bound over the merged histogram ``ghist`` (slot c,
    token k: row c * BQ + k), then the histogram row is cleared."""

    def __init__(self, ctas: int, topk: int):
        self.ctas = ctas
        self.topk = topk

    @cute.jit
    def __call__(self, ghist, par, n_par, row_len, row_bin, s):
        self.kernel(ghist, par, n_par, row_len, row_bin).launch(
            grid=(self.ctas, 1, 1), block=(128, 1, 1), stream=s, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, ghist, par, n_par, row_len, row_bin):
        lt, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        wq = lt // 32
        lane = lt % 32
        smem = utils.SmemAllocator()
        sb = smem.allocate_tensor(Uint32, cute.make_layout(4), 16)
        scan_a = sb.iterator.toint()
        cute.arch.griddepcontrol_wait()
        n = n_par[0]
        gh_a = ghist.iterator.toint()
        per = S_BINS // 128
        for r in cutlass.range(bid, n * BQ, self.ctas, unroll=1):
            c = r // BQ
            tok = r - c * BQ
            if tok < par[c, 1]:
                row = par[c, 0] + tok
                length = row_len[row]
                hist = gh_a + Int64(r) * (S_BINS * 4)
                r_top = S_BINS - 1 - per * lt
                counts = []
                mine = Uint32(0)
                for i in cutlass.range_constexpr(per):
                    rk = r_top - i
                    b = rk - S_BINS // 2
                    if rk < S_BINS // 2:
                        b = rk ^ (S_BINS - 1)
                    cnt = ld_global_u32(hist + Int64(b) * 4)
                    counts.append(cnt)
                    mine += cnt
                inclusive = warp_inclusive(mine, lane)
                if lane == 31:
                    st_shared_u32(scan_a + wq * 4, inclusive)
                cute.arch.sync_threads()
                prior = Uint32(0)
                for i in cutlass.range_constexpr(4):
                    x = ld_shared_u32(scan_a + i * 4)
                    if i < wq:
                        prior += x
                cute.arch.sync_threads()
                above = prior + inclusive - mine
                need = Uint32(self.topk)
                if length > self.topk:
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
                for i in cutlass.range_constexpr(per):
                    st_global_u32(hist + Int64(i * 128 + lt) * 4, Uint32(0))


@functools.cache
def _split_kernels(sms, topk):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    n, m, sl = cute.sym_int(), cute.sym_int(), cute.sym_int()
    plan_k = cute.compile(
        SplitPlan(sms),
        fake((n, 4), dtype=cutlass.Int32),
        fake((1,), dtype=cutlass.Int32),
        fake((m, 7), dtype=cutlass.Int32),
        fake((1,), dtype=cutlass.Int32),
        fake((sl, 2), dtype=cutlass.Int32),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    bound_k = cute.compile(
        RowBound(sms, topk),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((sl, 2), dtype=cutlass.Int32),
        fake((1,), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    return plan_k, bound_k


class SelectivePlan:
    """Per-step layout of the selective path (sampled thresholds -> PUBLISH ->
    CandSelect) for requests r with ``row_counts[r]`` rows whose last row sees
    ``row_lens_last[r]`` keys starting at ``key_bases[r]`` (128-aligned)."""

    def __init__(self, row_counts, row_lens_last, key_bases, device, topk=2048):
        self.topk = topk
        self.items, self.row_len, self.row_base, self.scratch = plan(
            row_counts, row_lens_last, key_bases, device
        )
        t = sum(row_counts)
        self.t = t
        # recovery (full path for flagged rows): the bitmap stride
        self.n_items = torch.full((1,), self.items.shape[0], dtype=torch.int32,
                                  device=device)
        self.words = max(
            ((last + TILE - 1) // TILE * TILE for last in row_lens_last), default=TILE
        ) // 32
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
        self.n_s_items = torch.full((1,), self.s_items.shape[0], dtype=torch.int32,
                                    device=device)
        self.s_row_len = torch.full((t,), SAMPLES, dtype=torch.int32, device=device)
        self.s_row_base = torch.arange(t + 1, dtype=torch.int64, device=device) * SAMPLES
        self.rank = torch.cat(rank).to(torch.int32).to(device)
        # per row: threshold, flag, -; the dummy row t keeps nothing (NaN)
        self.row_bin = torch.zeros(3 * (t + 1), dtype=torch.int32, device=device)
        self.row_bin[3 * t] = 0x7FC0
        self.s_bin = torch.zeros(3 * (t + 1), dtype=torch.int32, device=device)
        # PUBLISH work units: with fewer items than a whole number of waves of the
        # persistent grid, each item's key tiles split into ``pub_split`` ranges
        # (newest first), each publishing into its own candidate region copy; the
        # copies are concatenated per key lane in range order (= the unsplit
        # order) before the select. Exact: the same keys pass the same thresholds.
        self.pub_split, self.pub_items = _publish_units(
            self.items.cpu().tolist(), t, device
        )
        self.n_pub_items = torch.full((1,), self.pub_items.shape[0], dtype=torch.int32,
                                      device=device)


def selective_topk(sp, q, q_scale, keys, key_scale, weights_bf16, out, ctas=None,
                   mode=PUBLISH, recover_all=False):
    """Exact top-k of every row of ``sp`` into ``out`` (Int32 [T, topk], unordered,
    -1 padded), without host reads. Returns (row_bin, None): row r was
    recomputed by the full path iff row_bin[3 r + 1] != 0 (``recover_all``: every
    row, for tests)."""
    args = (sp, q, q_scale, keys, key_scale, weights_bf16)
    thresholds(*args, ctas=ctas)
    cand = publish(*args, ctas=ctas, mode=mode)
    cand_select(sp, cand, out)
    # the candidate regions are dead: free them before the recovery's scratch
    del cand
    recover(*args, out, ctas=ctas, recover_all=recover_all)
    return sp.row_bin, None


def thresholds(sp, q, q_scale, keys, key_scale, weights_bf16, ctas=None):
    """Each row's PUBLISH threshold (row_bin[3 r]): the rank-th largest logit of
    the request's sampled keys."""
    dev = q.device
    sk = keys[sp.sample_idx].contiguous()
    ssf = key_scale.reshape(-1)[sp.sample_idx].contiguous()
    s_logits = torch.empty((sp.t + 1) * SAMPLES, dtype=torch.bfloat16, device=dev)
    score(q, q_scale, sk, ssf, weights_bf16, sp.s_row_len, sp.s_items, sp.s_row_base,
          s_logits, sp.s_bin, ctas=ctas, mode=STORE, n_items=sp.n_s_items)
    _thr_select_kernel()(
        s_logits.view(torch.int16), sp.rank, sp.row_bin,
        torch.cuda.current_stream(dev).cuda_stream,
    )


# Key-tile splits of the PUBLISH pass (each one more candidate region copy of
# (T + 1) x REGION words). The copies are freed before the recovery allocates its
# logits scratch, so the transient peak is the larger of the two phases
# (indexer_step.group_bytes): 4 copies stay below the recovery's scratch at long
# rows.
PUB_SPLIT_MAX = 4


def _publish_units(items, t, device, ctas=None):
    """(split, units): the PUBLISH items, each cut into ``split`` key-tile ranges
    when that shortens the persistent grid's makespan (ceil(n split / ctas) / split
    item-times), as [n split, 7] Int32 (token0, ntok, key0, ntiles, t_lo, t_hi,
    region offset in 16-bit units), longest first; split 1: the items as they are."""
    ctas = ctas or torch.cuda.get_device_properties(device).multi_processor_count
    n = len(items)
    best, split = None, 1
    for s in range(1, PUB_SPLIT_MAX + 1):
        span = -(-n * s // ctas) / s
        if best is None or span < best - 1e-9:
            best, split = span, s
    if split == 1 or n == 0:
        return 1, torch.tensor(items, dtype=torch.int32, device=device).view(-1, 4)
    region = (t + 1) * REGION * 2
    units = []
    for tok0, ntok, key0, ntiles in items:
        npairs = (ntiles + (ntiles & 1)) // 2
        ntp = 2 * npairs
        for k in range(split):
            pa = npairs * k // split
            pb = npairs * (k + 1) // split
            if pb > pa:  # newest range first: tiles [ntp - 2 pb, ntp - 2 pa)
                units.append((tok0, ntok, key0, ntiles, ntp - 2 * pb, ntp - 2 * pa,
                              k * region))
    units.sort(key=lambda u: -(u[5] - u[4]))
    return split, torch.tensor(units, dtype=torch.int32, device=device).view(-1, 7)


@triton.jit
def _concat_regions_kernel(cand_ptr, region_words, row_words, n_split: tl.constexpr,
                           slots: tl.constexpr, tile: tl.constexpr, sb: tl.constexpr):
    """Program (row, k), k >= 1: copy k's slots of every key lane appended after the
    slots of copies 0 .. k - 1 in copy 0's column (copy order = the unsplit order);
    slots past ``slots`` are dropped (that column counts as overflowed below).
    Programs of one row write disjoint slots and only read count words."""
    row = tl.program_id(0)
    k = tl.program_id(1) + 1
    lane = tl.arange(0, tile)
    sl = tl.arange(0, sb)[:, None]
    base = cand_ptr + row.to(tl.int64) * row_words
    off = tl.zeros([tile], dtype=tl.int32)
    for j in tl.static_range(0, n_split - 1):
        if j < k:
            off += tl.load(base + j * region_words + lane).to(tl.int32)
    src = base + k * region_words
    c = tl.load(src + lane).to(tl.int32)
    cmax = tl.minimum(tl.max(c, axis=0), slots)
    for s0 in range(0, cmax, sb):
        s = s0 + sl
        dst = off[None, :] + s
        m = (s < c[None, :]) & (dst < slots)
        v = tl.load(src + (1 + s) * tile + lane[None, :], mask=m)
        tl.store(base + (1 + dst) * tile + lane[None, :], v, mask=m)


@triton.jit
def _concat_counts_kernel(cand_ptr, region_words, row_words, n_split: tl.constexpr,
                          slots: tl.constexpr, tile: tl.constexpr):
    """Copy 0's count words := the concatenated counts; a column whose copies reach
    ``slots`` keeps count ``slots`` (overflowed: the select flags the row and the
    exact full path recomputes it). After _concat_regions_kernel (it reads them)."""
    row = tl.program_id(0)
    lane = tl.arange(0, tile)
    base = cand_ptr + row.to(tl.int64) * row_words
    total = tl.zeros([tile], dtype=tl.int32)
    ovf = tl.zeros([tile], dtype=tl.int1)
    for j in tl.static_range(0, n_split):
        c = tl.load(base + j * region_words + lane).to(tl.int32)
        ovf = ovf | (c >= slots)
        total += c
    ovf = ovf | (total >= slots)
    tl.store(base + lane, tl.where(ovf, slots, total).to(tl.int32))


def publish(sp, q, q_scale, keys, key_scale, weights_bf16, ctas=None, mode=PUBLISH):
    """Scores of every row, keeping the keys at or above its threshold (returns the
    candidate regions)."""
    split = getattr(sp, "pub_split", 1)
    cand = torch.empty(split * (sp.t + 1) * REGION, dtype=torch.int32, device=q.device)
    if split == 1:
        score(q, q_scale, keys, key_scale, weights_bf16, sp.row_len, sp.items,
              sp.cand_base, cand.view(torch.bfloat16), sp.row_bin, ctas=ctas,
              mode=mode, n_items=sp.n_items)
        return cand
    # a range with no tiles of a row (short rows: range 0 can be empty too) never
    # writes that row's counts in its copy
    cand.view(split, sp.t + 1, REGION)[:, :, :TILE].zero_()
    score(q, q_scale, keys, key_scale, weights_bf16, sp.row_len, sp.pub_items,
          sp.cand_base, cand.view(torch.bfloat16), sp.row_bin, ctas=ctas, mode=mode,
          n_items=sp.n_pub_items)
    _concat_regions_kernel[(sp.t, split - 1)](
        cand, (sp.t + 1) * REGION, REGION, split, SLOTS, TILE, 8, num_warps=4,
    )
    _concat_counts_kernel[(sp.t,)](
        cand, (sp.t + 1) * REGION, REGION, split, SLOTS, TILE, num_warps=4,
    )
    return cand[: (sp.t + 1) * REGION]


def cand_select(sp, cand, out):
    """Exact top-k of every row from its candidates; flags the rows it cannot
    settle (row_bin[3 r + 1])."""
    ctas = CS2_CTAS_PER_SM * torch.cuda.get_device_properties(out.device).multi_processor_count
    _cand_select2_kernel(out.shape[1], ctas)(
        cand, sp.row_len, sp.row_bin, out.view(-1),
        torch.cuda.current_stream(out.device).cuda_stream,
    )


def recover(sp, q, q_scale, keys, key_scale, weights_bf16, out, ctas=None,
            recover_all=False):
    """Flagged rows (threshold missed, a lane overflowed, too many ties) through the
    full path, without a host read (graph-safe): only the items holding a flagged
    row are scored; unflagged rows keep the candidate result."""
    dev = q.device
    t = sp.t
    if recover_all:  # tests: every row
        items, n_active = sp.items, sp.n_items
        rec_len = sp.row_len
    else:
        # the items holding a flagged row (compacted, in order), their count and
        # the select's row lengths (-1: settled) in one small kernel
        items = torch.empty_like(sp.items)
        n_active = torch.empty(1, dtype=torch.int32, device=dev)
        rec_len = torch.empty_like(sp.row_len)
        _rec_plan_kernel()(
            sp.row_bin, sp.items, sp.row_len, items, n_active, rec_len,
            torch.cuda.current_stream(dev).cuda_stream,
        )
    logits = torch.empty(sp.scratch, dtype=torch.bfloat16, device=dev)
    rec_bin = torch.empty(3 * (t + 1), dtype=torch.int32, device=dev)
    # key splits when the items do not fill the GPU (one item's latency otherwise)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    plan_k, bound_k = _split_kernels(sms, out.shape[1])
    n_slots = min(items.shape[0], sms)
    items_x = torch.empty((max(items.shape[0], 2 * sms), 7), dtype=torch.int32, device=dev)
    n_x = torch.empty(1, dtype=torch.int32, device=dev)
    par = torch.zeros((n_slots, 2), dtype=torch.int32, device=dev)
    ghist = torch.zeros(n_slots * BQ * S_BINS, dtype=torch.int32, device=dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    plan_k(items, n_active, items_x, n_x, par, stream)
    score(q, q_scale, keys, key_scale, weights_bf16, sp.row_len, items_x, sp.row_base,
          logits, rec_bin, ctas=ctas, n_items=n_x, ghist=ghist)
    n_par = torch.clamp(n_active, max=n_slots)
    bound_k(ghist, par, n_par, sp.row_len, rec_bin, stream)
    bitmap = torch.empty(t * sp.words, dtype=torch.int32, device=dev)
    rec_cand = torch.empty(t * CAND, dtype=torch.int32, device=dev)
    select(logits, sp.row_base, rec_len, rec_bin, bitmap, rec_cand, out, ctas=ctas,
           gate=n_active)


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
def _score_kernel(ctas, mode=0, topk=2048, epi=0, split=False):
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
        fake((cute.sym_int(), 7 if split else 4), dtype=cutlass.Int32),
        fake((1,), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int64),
        fake((cute.sym_int(),), dtype=cutlass.Int16),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        make_fake_stream(),
    )
    return cute.compile(
        IndexerScore(ctas, mode, topk, epi, split), *args, options="--enable-tvm-ffi"
    )


@functools.cache
def _select_kernels(ctas, topk):
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
    gate = fake((1,), dtype=cutlass.Int32)
    xh = fake((cute.sym_int(),), dtype=cutlass.Int32)
    xl = fake((cute.sym_int(),), dtype=cutlass.Int32)
    mark = cute.compile(
        PrefillMark(ctas, topk), *common, xh, xl, gate, make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    exact = cute.compile(
        PrefillExact(ctas, topk), *common[:5], xh, xl, gate, make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    fin = cute.compile(
        PrefillFinish(ctas * 8, topk),
        *common,
        fake((t, topk), dtype=cutlass.Int32),
        gate,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    return mark, exact, fin


def score(q, q_scale, keys, key_scale, weights_bf16, row_len, items, row_base,
          logits, row_bin, ctas=None, mode=0, topk=2048, n_items=None, epi=0,
          ghist=None):
    """BF16 logits of every prefill row into ``logits`` and each row's top-k bound
    into ``row_bin`` (Int32 [3 T]: bound bin rank, count above, candidate count
    zeroed for the select) (see ``plan``). ``n_items`` (Int32 [1], device): only
    the first n_items items run (default all). ``items`` with 7 columns are key
    splits (SplitPlan; mode 0): their rows' histograms go to ``ghist``."""
    dev = q.device
    t = q.shape[0]
    assert keys.shape[0] % TILE == 0
    ctas = ctas or torch.cuda.get_device_properties(dev).multi_processor_count
    if n_items is None:
        n_items = torch.full((1,), items.shape[0], dtype=torch.int32, device=dev)
    split = items.shape[1] == 7
    if ghist is None:
        ghist = row_bin  # unused without splits
    _score_kernel(ctas, mode, topk, epi, split)(
        q.view(t * HEADS, DIM // 2),
        q_scale.reshape(-1),
        keys,
        key_scale.reshape(-1),
        weights_bf16.contiguous().view(torch.int32).view(t, HEADS // 2),
        row_len,
        items,
        n_items,
        row_base,
        logits.view(torch.int16),
        row_bin,
        ghist,
        torch.cuda.current_stream(dev).cuda_stream,
    )


def select(logits, row_base, row_len, row_bin, bitmap, cand, out, ctas=None, gate=None):
    """Exact top-k key indices of every row (ties to the lowest key index, -1 padded)
    into ``out``; rows with a negative length are skipped. ``bitmap``: Int32
    [T * words] (words = bitmap words per row, the longest row's positions / 32),
    ``cand``: Int32 [T * CAND]; ``row_bin`` Int32 [3 T] from score. ``gate``
    (Int32 [1], device): 0 skips every row."""
    ctas = ctas or torch.cuda.get_device_properties(out.device).multi_processor_count
    mark, exact, fin = _select_kernels(ctas * 4, out.shape[1])
    stream = torch.cuda.current_stream(out.device).cuda_stream
    if gate is None:
        gate = torch.ones(1, dtype=torch.int32, device=out.device)
    t = row_len.shape[0]
    # per (row, mark chunk) bound-bin histograms and the overflowed-row list
    chunks = (bitmap.shape[0] // max(t, 1) * 32 + CHUNK_V * 8 - 1) // (CHUNK_V * 8)
    xh = torch.empty(t * chunks * 32, dtype=torch.int32, device=out.device)
    xl = torch.zeros(t + 1, dtype=torch.int32, device=out.device)
    args = (logits.view(torch.int16), row_base, row_len, row_bin, bitmap, cand)
    mark(*args, xh, xl, gate, stream)
    exact(*args[:5], xh, xl, gate, stream)
    fin(*args, out, gate, stream)
