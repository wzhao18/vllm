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
KV_STAGES = 4
Q_STAGES = 2
ACC_STAGES = 2
EPI_WG = 4  # epilogue warpgroups
TOK_WG = BQ // EPI_WG  # tokens per epilogue warpgroup
PRODUCER = 4 * EPI_WG
MMA_WARP = PRODUCER + 1
THREADS = 32 * (MMA_WARP + 1)
CAP = 3072  # candidate buffer per token
CHECK = 4  # tiles between buffer checks
TRIGGER = CAP - CHECK * TILE  # compact when a buffer holds more
TMEM_COLS = 576  # exclusive allocation: 2 x N accumulators + scale factors
SF_WORDS_A = TILE  # one 32-bit word (four UE8M0) per key
SF_WORDS_B = N
HIST = 256
KEY_BITS = 17  # key indices < 2^17 (max_model_len 131072)


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


class IndexerScoreSelect:
    """Exact top-``topk`` key indices of every query row of the step's prefill
    requests (see the module docstring)."""

    def __init__(self, ctas: int, topk: int = 2048, debug_logits: bool = False):
        assert topk <= TRIGGER - TILE
        self.ctas = ctas
        self.topk = topk
        self.debug_logits = debug_logits
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
        out_idx: cute.Tensor,  # Int32 [T, topk]: selected key indices (unordered)
        out_cnt: cute.Tensor,  # Int32 [T]: selected count (min(topk, row_len))
        logits: cute.Tensor,  # Int16 [T, pitch] (debug_logits) else [1, 1]
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
            out_idx,
            out_cnt,
            logits,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=s,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    # Selection helpers (one epilogue warpgroup, 128 threads)
    # ------------------------------------------------------------------
    @cute.jit
    def insert(self, tok, key, kbits, valid, tau, bufs, lane):
        """Append (key, kbits) to token ``tok``'s buffer if valid and kbits >= tau."""
        buf_i, buf_k, cnt_a = bufs
        ok = cutlass.Boolean(False)
        if valid:
            if kbits >= tau:
                ok = cutlass.Boolean(True)
        m = cute.arch.vote_ballot_sync(ok)
        if m != Uint32(0):
            base = Uint32(0)
            if lane == 0:
                base = atom_add_shared(cnt_a + tok * 4, popc(m))
            base = cute.arch.shuffle_sync(base, 0)
            if ok:
                pos = base + popc(m & lanemask_lt())
                st_shared_u32(buf_i + (tok * CAP + pos) * 4, key)
                st_shared_u16(buf_k + (tok * CAP + pos) * 2, kbits)

    @cute.jit
    def topdown(self, hist_a, need, lane):
        """Warp-level: the bin b (scanning from the top) where the running count
        reaches ``need``; returns (b, count of keys in bins above b)."""
        # lane l owns bins 255 - 8l - j, j < 8 (descending)
        counts = [ld_shared_u32(hist_a + (255 - 8 * lane - j) * 4) for j in range(8)]
        total = Uint32(0)
        for j in cutlass.range_constexpr(8):
            total += counts[j]
        incl = total
        for i in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, offset=1 << i, mask_and_clamp=0)
            if lane >= Int32(1 << i):
                incl += up
        excl = incl - total
        hit = cutlass.Boolean(False)
        if excl < need:
            if incl >= need:
                hit = cutlass.Boolean(True)
        m = cute.arch.vote_ballot_sync(hit)
        src = Int32(31)
        if m != Uint32(0):
            src = Int32(31) - Int32(clz(m))
        b = Uint32(0)
        above = Uint32(0)
        if hit:
            run = excl
            found = Int32(0)
            for j in cutlass.range_constexpr(8):
                if found == 0:
                    if run + counts[j] >= need:
                        b = Uint32(255 - 8 * lane - j)
                        above = run
                        found = Int32(1)
                    else:
                        run += counts[j]
        b = cute.arch.shuffle_sync(b, src)
        above = cute.arch.shuffle_sync(above, src)
        return b, above

    @cute.jit
    def compact(self, tok, n, need, keep_ties, sm, lt, wg, lane):
        """Keep the top ``need`` of token ``tok``'s ``n`` buffered keys (ties to the
        lowest key index; with ``keep_ties`` every key equal to the threshold is
        kept when they fit). Returns (kept, threshold key)."""
        buf_i, buf_k, cnt_a, hist_a, sel_a = sm
        warp_in = lt // 32
        # pass 1: high byte histogram
        st_shared_u32(hist_a + lt * 4, Uint32(0))
        st_shared_u32(hist_a + (lt + 128) * 4, Uint32(0))
        wg_sync(wg)
        for i in cutlass.range(lt, Int32(n), 128, unroll=1):
            k = ld_shared_u16(buf_k + (tok * CAP + i) * 2)
            atom_add_shared(hist_a + (k >> Uint32(8)) * 4, Uint32(1))
        wg_sync(wg)
        if warp_in == 0:
            b, above = self.topdown(hist_a, need, lane)
            if lane == 0:
                st_shared_u32(sel_a, b)
                st_shared_u32(sel_a + 4, above)
        wg_sync(wg)
        b = ld_shared_u32(sel_a)
        above = ld_shared_u32(sel_a + 4)
        st_shared_u32(hist_a + lt * 4, Uint32(0))
        st_shared_u32(hist_a + (lt + 128) * 4, Uint32(0))
        wg_sync(wg)
        for i in cutlass.range(lt, Int32(n), 128, unroll=1):
            k = ld_shared_u16(buf_k + (tok * CAP + i) * 2)
            if (k >> Uint32(8)) == b:
                atom_add_shared(hist_a + (k & Uint32(0xFF)) * 4, Uint32(1))
        wg_sync(wg)
        if warp_in == 0:
            lo, above2 = self.topdown(hist_a, need - above, lane)
            if lane == 0:
                eq = ld_shared_u32(hist_a + lo * 4)
                st_shared_u32(sel_a + 8, (b << Uint32(8)) | lo)
                st_shared_u32(sel_a + 12, above + above2)
                st_shared_u32(sel_a + 16, eq)
        wg_sync(wg)
        thr = ld_shared_u32(sel_a + 8)
        gt = ld_shared_u32(sel_a + 12)
        eq = ld_shared_u32(sel_a + 16)
        # Ties: keep all of them if allowed and they fit, else the lowest indices.
        cut = Uint32(0xFFFFFFFF)  # keep ties with key index < cut
        cut_ties = cutlass.Boolean(True)
        if cutlass.const_expr(keep_ties):
            if gt + eq <= Uint32(TRIGGER):
                cut_ties = cutlass.Boolean(False)
        if cut_ties:
            want = need - gt
            if eq > want:
                lo_i = Uint32(0)
                hi_i = Uint32(1 << KEY_BITS)
                while hi_i - lo_i > Uint32(1):
                    mid = (lo_i + hi_i) >> Uint32(1)
                    if lt == 0:
                        st_shared_u32(sel_a + 20, Uint32(0))
                    wg_sync(wg)
                    c = Uint32(0)
                    for i in cutlass.range(lt, Int32(n), 128, unroll=1):
                        k = ld_shared_u16(buf_k + (tok * CAP + i) * 2)
                        if k == thr:
                            if ld_shared_u32(buf_i + (tok * CAP + i) * 4) < mid:
                                c += Uint32(1)
                    for o in cutlass.range_constexpr(5):
                        c += cute.arch.shuffle_sync_bfly(c, offset=1 << o)
                    if lane == 0:
                        atom_add_shared(sel_a + 20, c)
                    wg_sync(wg)
                    total = ld_shared_u32(sel_a + 20)
                    wg_sync(wg)
                    if total <= want:
                        lo_i = mid
                    else:
                        hi_i = mid
                cut = lo_i
        # compaction: strided elements into registers, scan, write back
        ks = []
        ids = []
        keep = []
        mine = Uint32(0)
        for j in cutlass.range_constexpr(CAP // 128):
            i = lt + 128 * j
            k = Uint32(0)
            idx = Uint32(0)
            if i < n:
                k = ld_shared_u16(buf_k + (tok * CAP + i) * 2)
                idx = ld_shared_u32(buf_i + (tok * CAP + i) * 4)
            kp = cutlass.Boolean(False)
            if i < n:
                if k > thr:
                    kp = cutlass.Boolean(True)
                elif k == thr:
                    if idx < cut:
                        kp = cutlass.Boolean(True)
            if kp:
                mine += Uint32(1)
            ks.append(k)
            ids.append(idx)
            keep.append(kp)
        incl = mine
        for i in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, offset=1 << i, mask_and_clamp=0)
            if lane >= Int32(1 << i):
                incl += up
        if lane == 31:
            st_shared_u32(sel_a + 32 + warp_in * 4, incl)
        wg_sync(wg)
        wsum = Uint32(0)
        total = Uint32(0)
        for w in cutlass.range_constexpr(4):
            v = ld_shared_u32(sel_a + 32 + w * 4)
            if Int32(w) < warp_in:
                wsum += v
            total += v
        pos = wsum + incl - mine
        for j in cutlass.range_constexpr(CAP // 128):
            if keep[j]:
                st_shared_u32(buf_i + (tok * CAP + pos) * 4, ids[j])
                st_shared_u16(buf_k + (tok * CAP + pos) * 2, ks[j])
                pos += Uint32(1)
        if lt == 0:
            st_shared_u32(cnt_a + tok * 4, total)
        wg_sync(wg)
        return total, thr

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
        out_idx: cute.Tensor,
        out_cnt: cute.Tensor,
        logits: cute.Tensor,
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
            cnt: cute.struct.MemRange[Uint32, BQ]
            sel: cute.struct.MemRange[Uint32, EPI_WG * 16]
            hist: cute.struct.Align[cute.struct.MemRange[Uint32, EPI_WG * HIST], 16]
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
        sBufI = smem.allocate_tensor(Uint32, cute.make_layout((BQ * CAP,)), 16)
        sBufK = smem.allocate_tensor(cutlass.Uint16, cute.make_layout((BQ * CAP,)), 16)

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        end_a = st.row_end.data_ptr().toint()
        cnt_a = st.cnt.data_ptr().toint()
        w_a = st.w.data_ptr().toint()
        sfa_a = sSFA.iterator.toint()
        sfb_a = sSFB.iterator.toint()
        buf_i = sBufI.iterator.toint()
        buf_k = sBufK.iterator.toint()
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
        if tidx < BQ:
            st_shared_u32(cnt_a + tidx * 4, Uint32(0))
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
                tAcc, acc_full, acc_empty, q_full, q_empty, items, out_idx, out_cnt,
                logits, (end_a, w_a, cnt_a, buf_i, buf_k, st), warp, tidx, bid, lane,
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
        self, tAcc, acc_full, acc_empty, q_full, q_empty, items, out_idx, out_cnt,
        logits, sm, warp, tidx, bid, lane,
    ):
        end_a, w_a, cnt_a, buf_i, buf_k, st = sm
        wg = warp // 4
        lt = tidx % 128
        hist_a = st.hist.data_ptr().toint() + wg * HIST * 4
        sel_a = st.sel.data_ptr().toint() + wg * 64
        csm = (buf_i, buf_k, cnt_a, hist_a, sel_a)
        n_items = items.shape[0]
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32)
        acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
        t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
        thr = t2r.get_slice(lt)
        tAll = thr.partition_S(acc_cols)
        cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
        acc_it = Int32(0)
        jq = Int32(0)
        topk = Uint32(self.topk)
        for item in cutlass.range(bid, n_items, self.ctas, unroll=1):
            token0 = items[item, 0]
            ntok = items[item, 1]
            ntiles = items[item, 3]
            qs = jq % Q_STAGES
            cute.arch.mbarrier_wait(q_full + qs, (jq // Q_STAGES) & 1)
            tau0 = Uint32(0)
            tau1 = Uint32(0)
            ends = [
                Int32(ld_shared_u32(end_a + (qs * BQ + wg * TOK_WG + j) * 4))
                for j in range(TOK_WG)
            ]
            wregs = [
                [
                    ld_shared_u32(w_a + (qs * BQ * 16 + (wg * TOK_WG + j) * 16 + i) * 4)
                    for i in range(16)
                ]
                for j in range(TOK_WG)
            ]
            for tt in cutlass.range(ntiles, unroll=1):
                t = ntiles - 1 - tt
                a = acc_it % ACC_STAGES
                cute.arch.mbarrier_wait(acc_full + a, (acc_it // ACC_STAGES) & 1)
                key = t * TILE + 4 * (lt % 32) + lt // 32
                bits = []
                for j in cutlass.range_constexpr(TOK_WG):
                    tok = wg * TOK_WG + j
                    frag = cute.make_fragment_like(cD, Float32)
                    cute.copy(t2r, tAll[(None, None, None, 0, tok, a)], frag)
                    s0 = Uint32(0)
                    s1 = Uint32(0)
                    wr = wregs[j]
                    for h in cutlass.range_constexpr(0, HEADS, 4):
                        s0 = fma_bf16x2(relu_bf16x2(frag[h], frag[h + 1]), wr[h // 2], s0)
                        s1 = fma_bf16x2(
                            relu_bf16x2(frag[h + 2], frag[h + 3]), wr[h // 2 + 1], s1
                        )
                    pair = add_bf16x2(s0, s1)
                    total = add_bf16x2(pair, pair >> Uint32(16)) & Uint32(0xFFFF)
                    bits.append(total)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty + a)
                for j in cutlass.range_constexpr(TOK_WG):
                    tok = wg * TOK_WG + j
                    valid = key < ends[j]
                    if cutlass.const_expr(self.debug_logits):
                        if valid:
                            row = Int64(token0 + tok) * Int64(logits.shape[1])
                            st_global_u16(
                                logits.iterator.toint() + (row + Int64(key)) * 2, bits[j]
                            )
                    self.insert(
                        Uint32(tok), Uint32(key), order16(bits[j]), valid,
                        tau0 if j == 0 else tau1,
                        (buf_i, buf_k, cnt_a), lane,
                    )
                acc_it += 1
                if tt % CHECK == CHECK - 1:
                    wg_sync(wg)
                    for j in cutlass.range_constexpr(TOK_WG):
                        tok = wg * TOK_WG + j
                        c = ld_shared_u32(cnt_a + tok * 4)
                        if c > Uint32(TRIGGER):
                            _kept, thr_k = self.compact(
                                Uint32(tok), c, topk, True, csm, lt, wg, lane
                            )
                            if cutlass.const_expr(j == 0):
                                tau0 = thr_k
                            else:
                                tau1 = thr_k
            # item end: exact top-k of each own token, written out
            wg_sync(wg)
            for j in cutlass.range_constexpr(TOK_WG):
                tok = wg * TOK_WG + j
                if tok < ntok:
                    c = ld_shared_u32(cnt_a + tok * 4)
                    kept = c
                    if c > topk:
                        kept, _thr = self.compact(Uint32(tok), c, topk, False, csm, lt, wg, lane)
                    row = token0 + tok
                    for i in cutlass.range(lt, Int32(kept), 128, unroll=1):
                        st_global_u32(
                            out_idx.iterator.toint() + (Int64(row) * Int64(self.topk) + Int64(i)) * 4,
                            ld_shared_u32(buf_i + (tok * CAP + i) * 4),
                        )
                    if lt == 0:
                        out_cnt[row] = Int32(kept)
                wg_sync(wg)
                if lt == 0:
                    st_shared_u32(cnt_a + tok * 4, Uint32(0))
            wg_sync(wg)
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(q_empty + qs)
            jq += 1


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------
def plan_items(row_counts, row_lens_last, key_bases, device):
    """Work items for requests with ``row_counts[r]`` query rows whose last row sees
    ``row_lens_last[r]`` keys (row i of request r sees row_lens_last[r] -
    row_counts[r] + 1 + i keys), keys of request r starting at ``key_bases[r]``.
    Returns (items Int32 [n, 4], row_len Int32 [T]); items are ordered by
    decreasing tile count (longest first)."""
    items = []
    lens = []
    token = 0
    for rows, last, kb in zip(row_counts, row_lens_last, key_bases):
        assert kb % TILE == 0, "request key bases must be 128-aligned"
        first = last - rows + 1
        lens.extend(range(first, last + 1))
        for o in range(0, rows, BQ):
            n = min(BQ, rows - o)
            hi = first + o + n - 1
            items.append((token + o, n, kb, (hi + TILE - 1) // TILE))
        token += rows
    items.sort(key=lambda it: -it[3])
    it = torch.tensor(items, dtype=torch.int32, device=device).view(-1, 4)
    ln = torch.tensor(lens, dtype=torch.int32, device=device)
    return it, ln


@functools.cache
def _compiled(ctas, topk, debug_logits):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    kernel = IndexerScoreSelect(ctas, topk, debug_logits)
    t, nk, ni = cute.sym_int(), cute.sym_int(), cute.sym_int()
    pitch = cute.sym_int() if debug_logits else 1
    lt = cute.sym_int() if debug_logits else 1
    args = (
        fake((cute.sym_int(), 64), dtype=cutlass.Uint8),
        fake((cute.sym_int(),), dtype=cutlass.Int32),
        fake((nk, 64), dtype=cutlass.Uint8),
        fake((nk,), dtype=cutlass.Int32),
        fake((t, 16), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        fake((ni, 4), dtype=cutlass.Int32),
        fake((t, topk), dtype=cutlass.Int32),
        fake((t,), dtype=cutlass.Int32),
        fake((lt, pitch), (pitch, 1), dtype=cutlass.Int16) if debug_logits else fake((1, 1), dtype=cutlass.Int16),
        make_fake_stream(),
    )
    return cute.compile(kernel, *args, options="--enable-tvm-ffi")


def score_select(q, q_scale, keys, key_scale, weights_bf16, row_len, items,
                 topk=2048, logits=None, ctas=None):
    """Launch IndexerScoreSelect; returns (out_idx [T, topk] unordered, out_cnt [T])."""
    dev = q.device
    t = q.shape[0]
    assert keys.shape[0] % TILE == 0
    ctas = ctas or torch.cuda.get_device_properties(dev).multi_processor_count
    out_idx = torch.empty(t, topk, dtype=torch.int32, device=dev)
    out_cnt = torch.empty(t, dtype=torch.int32, device=dev)
    dbg = logits is not None
    fn = _compiled(ctas, topk, dbg)
    if logits is None:
        logits = torch.empty(1, 1, dtype=torch.int16, device=dev)
    fn(
        q.view(t * HEADS, DIM // 2),
        q_scale.reshape(-1),
        keys,
        key_scale.reshape(-1),
        weights_bf16.contiguous().view(torch.int32).view(t, HEADS // 2),
        row_len,
        items,
        out_idx,
        out_cnt,
        logits,
        torch.cuda.current_stream(dev).cuda_stream,
    )
    return out_idx, out_cnt
