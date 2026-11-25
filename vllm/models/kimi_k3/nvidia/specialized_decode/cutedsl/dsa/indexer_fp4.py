# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 scorer of the fused decode indexer (indexer.py) under DCP: the BF16 paged
MQA logits of the step's rows over this rank's MXFP4 indexer cache, bit for bit as
DeepGEMM's SM100 ``fp8_fp4_paged_mqa_logits`` writes them, in its layout (row ``t``
at element ``t * pitch``, key index = column, keys past a row's end untouched).

Key-balanced persistent tcgen05 kernel (IndexerScore's schedule): the 128-key
tiles of all requests are cut into equal ranges, one per SM. A tile is two 64-key
cache pages (A, M = 128 keys, K-major E2M1 by TMA per page); the request's ``q``
verify rows (B, N = 8 x 32 query heads, by one TMA per request) are scored by two
K = 64 block-scaled MXF4 MMAs (kind::mxf4, UE8M0 per 32 values: the key scales
from the page tail, the query scales of DsaIndex) into one TMEM accumulator, as
DeepGEMM issues them (A = keys, B = query, K split 64 + 64). Stock's epilogue
(DeepGEMM sm100_mqa_logits): each head's FP32 value through
cvt.rn.relu.bf16x2.f32, the BF16 head weights accumulated by BF16x2 FMAs in two
sums (heads 4i, 4i + 1 and 4i + 2, 4i + 3, in head order), then
(s0 + s1).lo + (s0 + s1).hi in BF16; stored as is.

The key scales of a page are 64 contiguous UE8M0 x4 words; the MMA's scale-factor
atom wants row ``r``'s word at ``128 (r / 128) + 4 (r % 32) + (r % 128) / 32``: the
MMA warp permutes the tile's 128 words from a staging buffer (bulk copies) before
the SMEM -> TMEM scale copy.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..common.primitives import asm, load_words
from .indexer import (
    DIM,
    HEADS,
    ITEM_TILES,
    MAX_REQUESTS,
    PAGE,
    TILE,
    _first_request,
    _next_item,
    add_bf16x2,
    bulk_g2s,
    fma_bf16x2,
    relu_bf16x2,
    request_tiles,
    store_b16,
    stride_pages,
    table_entry,
    warp_inclusive,
)

BQ = 8  # query rows per MMA (N = BQ * HEADS); a request's q <= 8 rows
N = BQ * HEADS
STAGES = 4
EPI_GROUPS = 2  # epilogue warpgroups: tokens 4 g .. 4 g + 3 of the item
PRODUCER = 4 * EPI_GROUPS
MMA_WARP = PRODUCER + 1
THREADS = 32 * (MMA_WARP + 1)
HIST = 2048 + 32 + 32  # a select warp's 11-bit bins, low-digit bins, plane + count
TMEM_COLS = 512  # N accumulator columns + the scale factors
SF_A_BYTES = 512  # one stage's key scale-factor atom (128 rows x 4 UE8M0)
SF_B_BYTES = 1024  # one stage's query scale-factor atoms (256 rows x 4 UE8M0)
FP4_TILE_BYTES = TILE * DIM // 2
assert ITEM_TILES * 2 <= 32  # one page id per producer lane


@asm()
def tmem_ld32(taddr):
    """tcgen05.ld 32x32b.x32: 32 columns of this warp's 32 TMEM lanes."""
    regs = ",".join(f"${i}" for i in range(32))
    code = "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];"
    return code, ",".join(["=f"] * 32) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 32


@asm()
def tmem_wait_n(*vals):
    """tcgen05.wait::ld, with the loaded registers tied through it."""
    n = len(vals)
    cons = ",".join(["=f"] * n) + "," + ",".join(str(i) for i in range(n))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * n


@asm()
def st_shared_v4(address, a, b, c, d):
    return (
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        (Int32(address), Uint32(a), Uint32(b), Uint32(c), Uint32(d)),
    )


@asm(Uint32)
def ld_shared_u32(address):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm(Uint32)
def ld_global_u32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def fence_async_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def bar_sync(barrier: int, threads: int):
    return f"bar.sync {barrier}, {threads};", "~{memory}", ()


@asm()
def red_shared_add(address, value):
    return "red.shared.add.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def red_shared_or(address, value):
    return "red.shared.or.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm(Uint32)
def atom_shared_add(address, value):
    code = "atom.shared.add.u32 $0, [$1], $2;"
    return code, "=r,r,r,~{memory}", (Int32(address), Uint32(value))


@asm()
def st_shared_u32(address, value):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(value))


@asm(Uint32)
def atom_add_gpu(address, value):
    code = "atom.add.relaxed.gpu.global.u32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def st_global_u32(address, value):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def fence_acq_rel_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def load_v4(address):
    code = "ld.global.cg.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (Int64(address),), (Uint32,) * 4


@cute.jit
def ord16(bits: Uint32) -> Uint32:
    """Order-preserving 16-bit key of BF16 ``bits`` (the #63 pack's ordering)."""
    out = bits | Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        out = ~bits & Uint32(0xFFFF)
    return out


class IndexerScoreFp4:
    """BF16 logits of ``rows`` rows (``q`` per request) into DeepGEMM's paged MQA
    logits layout (row ``t`` at element ``t * pitch``) over the MXFP4 indexer
    cache: see the module docstring. Every request is scored."""

    def __init__(self, rows: int, q: int, ctas: int, pitch: int, topn: int = 0):
        assert rows % q == 0 and 1 <= q <= BQ and rows // q <= MAX_REQUESTS
        assert pitch > 0 and pitch % 512 == 0 and topn % 32 == 0
        self.rows, self.q, self.ctas, self.pitch = rows, q, ctas, pitch
        # topn > 0: the CTA that scores a request's last tile also takes each of
        # its rows' exact local top-``topn`` as B' rows (IndexerLocalTopk's output).
        self.topn = topn
        self.tiler = (TILE, N, DIM)

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,  # Uint8 [rows * 32, 64]: E2M1, two per byte
        q_sf: cute.Tensor,  # Int32 [rows * 32]: four UE8M0 per (row, head)
        weights: cute.Tensor,  # BF16 [rows * 32]
        keys: cute.Tensor,  # Uint8 [64, 64, blocks]: each page's E2M1 keys
        scales: cute.Tensor,  # Int32 [blocks, 64]: each page's UE8M0 x4 words
        table_address: Int64,  # Int32 [requests, pages]: index block ids
        table_pitch: Int32,
        table_width: Int32,
        row_ends: cute.Tensor,  # Int32 [rows]
        logits: cute.Tensor,  # BF16 bits as Int16 [capacity]
        out: cute.Tensor,  # Int32 [rows, topn + topn / 32]: B' rows (topn)
        valid: cute.Tensor,  # Int32 [rows]: their counts (topn)
        counter: cute.Tensor,  # Int32 [MAX_REQUESTS]: tiles scored, zero between calls
        s,
    ):
        fp4 = cutlass.Float4E2M1FN
        sf_t = cutlass.Float8E8M0FNU
        nq = query.shape[0]
        mQ = cute.make_tensor(
            cute.recast_ptr(query.iterator, dtype=fp4),
            cute.make_layout((nq, DIM, 1), stride=(DIM, 1, nq * DIM)),
        )
        mK = cute.make_tensor(
            cute.recast_ptr(keys.iterator, dtype=fp4),
            cute.make_layout(
                (PAGE, DIM, keys.shape[2]), stride=(DIM, 1, keys.stride[2] * 2)
            ),
        )
        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            fp4,
            fp4,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            sf_t,
            32,
            tcgen05.CtaGroup.ONE,
            self.tiler[:2],
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, fp4, STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, fp4, 2)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(
            mma, self.tiler, 32, STAGES
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler, 32, 2)
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
        assert N + self.sfa_cols + self.sfb_cols <= TMEM_COLS
        # 64-byte K-major rows: the A stage is SW64 atoms of 8 rows, so each page
        # (64 keys) is one half of a stage.
        atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW64, fp4
        )
        page_layout = cute.tile_to_shape(atom, (PAGE, DIM, 2 * STAGES), (0, 1, 2))
        tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.slice_(page_layout, (None, None, 0)),
            (PAGE, DIM),
        )
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            mQ,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(fp4, b_one)
        self.kernel(
            mma,
            tma_k,
            tma_k_tensor,
            tma_q,
            tma_q_tensor,
            a_layout,
            b_layout,
            page_layout,
            sfa_layout,
            sfb_layout,
            tsfa_layout,
            tsfb_layout,
            q_sf,
            scales,
            weights,
            (table_address, table_pitch, table_width),
            row_ends,
            logits,
            out,
            valid,
            counter,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=s,
            use_pdl=True,
        )

    def s2t(self, sSF, tSF):
        """SMEM -> TMEM scale-factor copy (tcgen05.cp 4x32dp128bit), as CUTLASS's
        SM100 dense block-scaled GEMM."""
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
        )
        tiled = tcgen05.make_s2t_copy(atom, tc)
        thr = tiled.get_slice(0)
        src = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(sc))
        return tiled, src, thr.partition_D(tc)

    @cute.jit
    def tile_out(self, taddr, tok0, key, qs, w_a, sEnd, row0, out_base, acc_empty, lane):
        """One tile's logits of this thread's key for tokens tok0 .. tok0 + 3:
        stock's epilogue (see the module docstring); frees the accumulator once
        every load landed."""
        Q = self.q
        pitch = Int64(self.pitch)
        for pair in cutlass.range_constexpr(2):
            ta = tmem_ld32(taddr + (tok0 + 2 * pair) * HEADS)
            tb = tmem_ld32(taddr + (tok0 + 2 * pair + 1) * HEADS)
            vals = tmem_wait_n(*ta, *tb)
            if cutlass.const_expr(pair == 1):
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty)
            for half in cutlass.range_constexpr(2):
                tok = tok0 + 2 * pair + half
                f = vals[32 * half : 32 * half + 32]
                s0 = Uint32(0)
                s1 = Uint32(0)
                wrow = w_a + (qs * N + tok * HEADS) * 2
                for h in cutlass.range_constexpr(0, HEADS, 8):
                    w = load_words(wrow + h * 2, count=4, shared=True)
                    s0 = fma_bf16x2(relu_bf16x2(f[h], f[h + 1]), w[0], s0)
                    s1 = fma_bf16x2(relu_bf16x2(f[h + 2], f[h + 3]), w[1], s1)
                    s0 = fma_bf16x2(relu_bf16x2(f[h + 4], f[h + 5]), w[2], s0)
                    s1 = fma_bf16x2(relu_bf16x2(f[h + 6], f[h + 7]), w[3], s1)
                pair_sum = add_bf16x2(s0, s1)
                total = add_bf16x2(pair_sum, pair_sum >> Uint32(16))
                if tok < Q:
                    if key < sEnd[tok, qs]:
                        element = row0 + Int64(tok) * pitch + Int64(key)
                        store_b16(out_base + element * 2, total)

    @cute.jit
    def request_done(
        self, req, ntile, tiles, end, counter, last_a, hist_a, out_base, out, valid,
        warp, lane, tidx,
    ):
        """After this CTA's logits of an item of request ``req`` (``ntile`` of its
        ``tiles``): count them; the CTA that completes the request takes each of its
        rows' local top-N (epilogue warp w: row w)."""
        bar_sync(2, 32 * PRODUCER)
        if tidx == 0:
            fence_acq_rel_gpu()
            address = counter.iterator.toint() + Int64(req) * 4
            done = atom_add_gpu(address, Uint32(ntile)) + Uint32(ntile)
            flag = Uint32(0)
            if done == Uint32(tiles):
                flag = Uint32(1)
                st_global_u32(address, Uint32(0))  # zero for the next call
                fence_acq_rel_gpu()
            st_shared_u32(last_a, flag)
        bar_sync(2, 32 * PRODUCER)
        if ld_shared_u32(last_a) != Uint32(0):
            if warp < self.q:
                self.select_row(
                    req * self.q + warp, end, out_base, out, valid,
                    hist_a + warp * (HIST * 4), lane,
                )  # fmt: skip

    @cute.jit
    def select_row(self, row, length, out_base, out, valid, hist, lane):
        """One warp: the exact top-``topn`` keys of logits row ``row`` (``length``
        keys; ties at the threshold to the lowest key indices) as B' words
        ``(ord16 << 16) | (0xFFFF - (key & 0xFFFF))`` and the plane of key bit 16
        (IndexerLocalTopk's contract; slots in emission order), and the count."""
        n = self.topn
        low = hist + 2048 * 4
        cnt = low + 32 * 4
        plane = cnt + 4
        base = out_base + Int64(row) * Int64(self.pitch) * 2
        for i in cutlass.range_constexpr(1 + n // 32 // 32 + 1):
            if lane + 32 * i < 32 + 1 + n // 32:
                st_shared_u32(low + (lane + 32 * i) * 4, Uint32(0))
        cute.arch.sync_warp()
        vecs = (length + 7) // 8
        if length <= n:
            # Every key: slot = key index.
            for v in cutlass.range(lane, vecs, 32, unroll=1):
                w = load_v4(base + Int64(v) * 16)
                for e in cutlass.range_constexpr(8):
                    k = v * 8 + e
                    if k < length:
                        bits = (w[e // 2] >> Uint32(16 * (e % 2))) & Uint32(0xFFFF)
                        word = (ord16(bits) << Uint32(16)) | (
                            Uint32(0xFFFF) - (Uint32(k) & Uint32(0xFFFF))
                        )
                        out[row, k] = Int32(word)
                        if (k >> 16) != 0:
                            red_shared_or(plane + (k // 32) * 4, Uint32(1) << Uint32(k % 32))
            for k in cutlass.range(length + lane, n, 32, unroll=1):
                out[row, k] = Int32(0)
            if lane == 0:
                valid[row] = length
        else:
            for i in cutlass.range_constexpr(2048 // 32):
                st_shared_u32(hist + (i * 32 + lane) * 4, Uint32(0))
            cute.arch.sync_warp()
            # pass 1: the 11-bit histogram
            for v in cutlass.range(lane, vecs, 32, unroll=4):
                w = load_v4(base + Int64(v) * 16)
                for e in cutlass.range_constexpr(8):
                    if v * 8 + e < length:
                        bits = (w[e // 2] >> Uint32(16 * (e % 2))) & Uint32(0xFFFF)
                        red_shared_add(hist + (ord16(bits) >> Uint32(5)) * 4, 1)
            cute.arch.sync_warp()
            # the bin b holding the n-th largest key: lane l sums bins
            # 2047 - 64 l down to 2047 - 64 l - 63
            mine = Uint32(0)
            for i in cutlass.range_constexpr(64):
                mine += ld_shared_u32(hist + (2047 - 64 * lane - i) * 4)
            above = warp_inclusive(mine, lane) - mine
            need = Uint32(n)
            hit = Int32(0)
            if above < need and above + mine >= need:
                hit = lane
            for i in cutlass.range_constexpr(5):
                hit += cute.arch.shuffle_sync_bfly(hit, 1 << i)
            b = Uint32(0)
            run = cute.arch.shuffle_sync(above, hit)
            found = Int32(0)
            top = 2047 - 64 * hit
            for i in cutlass.range_constexpr(64):
                c = ld_shared_u32(hist + (top - i) * 4)
                if found == 0:
                    if run + c >= need:
                        b = Uint32(top - i)
                        found = Int32(1)
                    else:
                        run += c
            # run: keys in bins above b. pass 2: emit them; b's low digits
            for v in cutlass.range(lane, vecs, 32, unroll=4):
                w = load_v4(base + Int64(v) * 16)
                for e in cutlass.range_constexpr(8):
                    k = v * 8 + e
                    if k < length:
                        bits = (w[e // 2] >> Uint32(16 * (e % 2))) & Uint32(0xFFFF)
                        key = ord16(bits)
                        if (key >> Uint32(5)) > b:
                            self.emit(row, out, key, k, atom_shared_add(cnt, 1), plane)
                        elif (key >> Uint32(5)) == b:
                            red_shared_add(low + (key & Uint32(31)) * 4, 1)
            cute.arch.sync_warp()
            # the low digit d of the threshold, and the ties it takes
            lc = ld_shared_u32(low + (31 - lane) * 4)
            lab = warp_inclusive(lc, lane) - lc
            rest = need - run
            dh = Int32(0)
            if lab < rest and lab + lc >= rest:
                dh = lane
            for i in cutlass.range_constexpr(5):
                dh += cute.arch.shuffle_sync_bfly(dh, 1 << i)
            d = Uint32(31) - Uint32(dh)
            ties = rest - cute.arch.shuffle_sync(lab, dh)
            tau = (b << Uint32(5)) | d
            # pass 3: bin b above tau, then the first ``ties`` keys at tau in key
            # order (lane l holds keys 8 (32 i + l) .. + 7)
            taken = Uint32(0)
            for i in cutlass.range(0, (vecs + 31) // 32, unroll=1):
                v = i * 32 + lane
                at = Uint32(0)
                # past the row (v >= vecs) every k >= length: masked below
                w = load_v4(base + Int64(cutlass.min(v, vecs - 1)) * 16)
                keys = [
                    ord16((w[e // 2] >> Uint32(16 * (e % 2))) & Uint32(0xFFFF))
                    for e in range(8)
                ]
                for e in cutlass.range_constexpr(8):
                    k = v * 8 + e
                    if k < length:
                        if keys[e] == tau:
                            at += Uint32(1)
                        elif (keys[e] >> Uint32(5)) == b and keys[e] > tau:
                            self.emit(row, out, keys[e], k, atom_shared_add(cnt, 1), plane)
                prior = warp_inclusive(at, lane) - at
                for e in cutlass.range_constexpr(8):
                    k = v * 8 + e
                    if k < length:
                        if keys[e] == tau:
                            if taken + prior < ties:
                                self.emit(row, out, tau, k, atom_shared_add(cnt, 1), plane)
                            prior += Uint32(1)
                taken += cute.arch.shuffle_sync(prior, 31)
            if lane == 0:
                valid[row] = Int32(n)
        cute.arch.sync_warp()
        if lane < n // 32:
            out[row, n + lane] = Int32(ld_shared_u32(plane + lane * 4))

    @cute.jit
    def emit(self, row, out, key, k, slot, plane):
        out[row, Int32(slot)] = Int32(
            (key << Uint32(16)) | (Uint32(0xFFFF) - (Uint32(k) & Uint32(0xFFFF)))
        )
        if (k >> 16) != 0:
            red_shared_or(plane + (Int32(slot) // 32) * 4, Uint32(1) << (slot % Uint32(32)))

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
        page_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        tsfa_layout: cute.Layout,
        tsfb_layout: cute.Layout,
        q_sf: cute.Tensor,
        scales: cute.Tensor,
        weights: cute.Tensor,
        table,
        row_ends: cute.Tensor,
        logits: cute.Tensor,
        out: cute.Tensor,
        valid: cute.Tensor,
        counter: cute.Tensor,
    ):
        Q = self.q
        sf_t = cutlass.Float8E8M0FNU
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        @cute.struct
        class Storage:
            kv_full: cute.struct.MemRange[cutlass.Int64, STAGES]
            kv_empty: cute.struct.MemRange[cutlass.Int64, STAGES]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            q_empty: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cutlass.Int64
            acc_empty: cutlass.Int64
            tmem_hold: Int32
            last: Int32
            hist: cute.struct.Align[
                cute.struct.MemRange[Uint32, BQ * HIST if self.topn else 4], 16
            ]
            row_end: cute.struct.MemRange[Int32, BQ * 2]
            start: cute.struct.MemRange[Int32, MAX_REQUESTS + 1]
            w: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 2 * N], 16]
            scale: cute.struct.Align[cute.struct.MemRange[Int32, STAGES * TILE], 16]

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
        sPage = cute.make_tensor(sA.iterator, page_layout.outer)
        sEnd = st.row_end.get_tensor(cute.make_layout((BQ, 2)))
        sStart = st.start.get_tensor(cute.make_layout((MAX_REQUESTS + 1,)))
        sSc = st.scale.get_tensor(cute.make_layout((TILE, STAGES)))

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.ptr
        acc_empty = st.acc_empty.ptr
        w_a = st.w.data_ptr().toint()
        sc_a = st.scale.data_ptr().toint()
        sfa_a = sSFA.iterator.toint()
        sfb_a = sSFB.iterator.toint()
        if tidx == 0:
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + 4 * EPI_GROUPS)
            cute.arch.mbarrier_init(acc_full, 1)
            cute.arch.mbarrier_init(acc_empty, 4 * EPI_GROUPS)
            cute.arch.mbarrier_init_fence()
        if warp == PRODUCER:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)

        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=1, num_threads=THREADS
            ),
            allocator_warp_id=MMA_WARP,
        )
        tmem.allocate(TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        tAcc = cute.make_tensor(tmem_ptr, mma.make_fragment_C(acc_shape).layout)

        # The schedule: sStart[r] = first global tile of request r (row ends are
        # step metadata, written before the graph runs).
        requests = row_ends.shape[0] // Q
        for r in cutlass.range(tidx, requests, THREADS, unroll=1):
            sStart[r + 1] = request_tiles(row_ends, r, Q, Int32(0))
        cute.arch.sync_threads()
        if warp == 0:
            carry = Int32(0)
            for base in cutlass.range(0, requests, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < requests:
                    tiles = sStart[r + 1]
                tiles = Int32(warp_inclusive(Uint32(tiles), lane))
                if r < requests:
                    sStart[r + 1] = carry + tiles
                carry += cute.arch.shuffle_sync(tiles, 31)
            if lane == 0:
                sStart[0] = Int32(0)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        total = sStart[requests]
        lo = Int32((Int64(total) * Int64(bid)) // Int64(self.ctas))
        hi = Int32((Int64(total) * Int64(bid + 1)) // Int64(self.ctas))
        r_lo = _first_request(sStart, requests, lo)

        it = Int32(0)  # tiles issued / consumed by this role
        j = Int32(0)  # items issued / consumed by this role
        g = lo
        r = r_lo
        if warp == PRODUCER:
            thr_mma = mma.get_slice(0)
            gK = cute.local_tile(mK, (PAGE, DIM), (None, None, None))
            tKsK, tKgK = cpasync.tma_partition(
                tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sPage, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            max_page = table[2] - 1
            blocks = scales.shape[0]
            n_req = Int32(0)
            n_k0 = Int32(0)
            n_nt = Int32(0)
            n_page = Int32(0)
            n_end = Int32(0)
            if g < hi:
                n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                n_page = table_entry(
                    table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                )
                n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
            has_item = n_nt > 0
            while has_item:
                req = n_req
                ntile = n_nt
                my_page = n_page
                my_end = n_end
                n_nt = Int32(0)
                if g < hi:
                    n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                    n_page = table_entry(
                        table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                    )
                    n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
                qs = j % 2
                row_q = req * (Q * HEADS)
                # The query scales in the scale-factor atom: N row
                # c * 128 + 32 i + lane (token row // 32, head row % 32) -> word
                # c * 128 + 4 lane + i.
                sfw = []
                for c in cutlass.range_constexpr(N // 128):
                    for i in cutlass.range_constexpr(4):
                        n_row = c * 128 + 32 * i + lane
                        v = Uint32(0)
                        if n_row < Q * HEADS:
                            v = ld_global_u32(
                                q_sf.iterator.toint() + Int64(row_q + n_row) * 4
                            )
                        sfw.append(v)
                cute.arch.mbarrier_wait(q_empty + qs, ((j // 2) & 1) ^ 1)
                if lane < BQ:
                    sEnd[lane, qs] = my_end
                for c in cutlass.range_constexpr(N // 128):
                    st_shared_v4(
                        sfb_a + qs * SF_B_BYTES + (c * 128 + 4 * lane) * 4,
                        *sfw[4 * c : 4 * c + 4],
                    )
                fence_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        q_full + qs, self.q_bytes + Q * HEADS * 2
                    )
                    bulk_g2s(
                        st.w.data_ptr() + qs * N,
                        weights.iterator + row_q,
                        q_full + qs,
                        Q * HEADS * 2,
                    )
                gQ = cute.local_tile(
                    cute.domain_offset((row_q, 0, 0), mQ),
                    cute.slice_(self.tiler, (0, None, None)),
                    (None, None, None),
                )
                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gQ), 0, 3),
                )
                cute.copy(
                    tma_q, tQgQ[(None, 0, 0, 0)], tQsQ[(None, qs)], tma_bar_ptr=q_full + qs
                )
                j += 1
                for t in cutlass.range(ntile, unroll=1):
                    stage = it % STAGES
                    p0 = cute.arch.shuffle_sync(my_page, 2 * t)
                    p1 = cute.arch.shuffle_sync(my_page, 2 * t + 1)
                    # A partial last tile's second page may lie past the request's
                    # blocks: keep its (unused) reads in bounds.
                    p1 = cutlass.min(cutlass.max(p1, Int32(0)), blocks - 1)
                    cute.arch.mbarrier_wait(kv_empty + stage, ((it // STAGES) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kv_full + stage, FP4_TILE_BYTES + TILE * 4
                        )
                        bulk_g2s(
                            sSc.iterator + stage * TILE,
                            scales.iterator + stride_pages(p0, scales),
                            kv_full + stage,
                            PAGE * 4,
                        )
                        bulk_g2s(
                            sSc.iterator + stage * TILE + PAGE,
                            scales.iterator + stride_pages(p1, scales),
                            kv_full + stage,
                            PAGE * 4,
                        )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p0)],
                        tKsK[(None, 2 * stage)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p1)],
                        tKsK[(None, 2 * stage + 1)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    it += 1
                has_item = n_nt > 0
        elif warp == MMA_WARP:
            tCrA = mma.make_fragment_A(sA)
            tCrB = mma.make_fragment_B(sB)
            sfa_ptr = cute.recast_ptr(tmem_ptr + N, dtype=sf_t)
            sfb_ptr = cute.recast_ptr(tmem_ptr + N + self.sfa_cols, dtype=sf_t)
            tCtSFA = cute.make_tensor(sfa_ptr, tsfa_layout)
            tCtSFB = cute.make_tensor(sfb_ptr, tsfb_layout)
            s2t_a = self.s2t(sSFA, tCtSFA)
            s2t_b = self.s2t(sSFB, tCtSFB)
            while g < hi:
                _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, qs)], s2t_b[2])
                for _t in cutlass.range(ntile, unroll=1):
                    stage = it % STAGES
                    cute.arch.mbarrier_wait(acc_empty, (it & 1) ^ 1)
                    cute.arch.mbarrier_wait(kv_full + stage, (it // STAGES) & 1)
                    # The tile's key scales into the scale-factor atom: key
                    # lane + 32 i -> word 4 lane + i.
                    src = sc_a + (stage * TILE + lane) * 4
                    st_shared_v4(
                        sfa_a + stage * SF_A_BYTES + lane * 16,
                        ld_shared_u32(src),
                        ld_shared_u32(src + 128),
                        ld_shared_u32(src + 256),
                        ld_shared_u32(src + 384),
                    )
                    fence_async_shared()
                    cute.arch.sync_warp()
                    cute.copy(
                        s2t_a[0], s2t_a[1][(None, None, None, None, stage)], s2t_a[2]
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                        cute.gemm(
                            mma,
                            tAcc,
                            tCrA[(None, None, kb, stage)],
                            tCrB[(None, None, kb, qs)],
                            tAcc,
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + stage)
                        tcgen05.commit(acc_full)
                    it += 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty + qs)
                j += 1
        else:
            # Epilogue: warp w reads TMEM lanes 32 (w % 4) .. (its keys) of the
            # accumulator, warpgroup w / 4 the tokens 4 (w / 4) .. + 3.
            wq = warp % 4
            tok0 = (warp // 4) * 4
            lt = wq * 32 + lane
            taddr = Int32(tAcc.iterator.toint()) + ((wq * 32) << 16)
            out_base = logits.iterator.toint()
            pitch = Int64(self.pitch)
            if cutlass.const_expr(self.topn):
                # Requests without keys (CUDA-graph padding): empty B' rows.
                width = self.topn + self.topn // 32
                for rq in cutlass.range(bid, requests, self.ctas, unroll=1):
                    if sStart[rq + 1] == sStart[rq]:
                        for i in cutlass.range(tidx, Q * width, 32 * PRODUCER, unroll=1):
                            out[rq * Q + i // width, i % width] = Int32(0)
                        if tidx < Q:
                            valid[rq * Q + tidx] = Int32(0)
            while g < hi:
                req, k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                row0 = Int64(req * Q) * pitch
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                for t in cutlass.range(ntile, unroll=1):
                    key = k0 + t * TILE + lt
                    cute.arch.mbarrier_wait(acc_full, it & 1)
                    self.tile_out(
                        taddr, tok0, key, qs, w_a, sEnd, row0, out_base, acc_empty, lane
                    )
                    it += 1
                if cutlass.const_expr(self.topn):
                    end = Int32(0)
                    if warp < Q:
                        end = sEnd[warp, qs]
                    tiles = sStart[req + 1] - sStart[req]
                    self.request_done(
                        req, ntile, tiles, end, counter, st.last.ptr.toint(),
                        st.hist.data_ptr().toint(), out_base, out, valid, warp, lane,
                        tidx,
                    )
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(q_empty + qs)
                j += 1

        cute.arch.sync_threads()
        if warp == MMA_WARP:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
