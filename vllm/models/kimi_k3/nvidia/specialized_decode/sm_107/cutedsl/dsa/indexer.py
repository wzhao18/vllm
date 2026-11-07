# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode indexer of a K3.1 DSA layer at stock Kimi K3 precision (FP32 head
weights, FP32 logits, exact top-k on the FP32 logits). TP (every rank scores the
whole context): the FP8 indexer query and FP32 head weights of the step's rows over
the paged FP8 indexer cache -> the exact top-k of every row as the physical KV rows
the sparse attention reads, compacted to a prefix, and their counts. Two
PDL-chained kernels replace the stock chain (DeepGEMM FP32 logits, top-k, then the
logical -> physical conversion on a side stream):

  IndexerScore   key-balanced persistent tcgen05 scorer: the 128-key tiles of
                 all requests are cut into equal ranges, one per SM, and one
                 [128 keys x q * 32] FP8 MMA per tile scores every verify row of
                 the request, so each key page is read once per step. The
                 epilogue reduces the heads in FP32 exactly as DeepGEMM's FP32
                 reduction does (below), scales by the key's UE8M0 scale, and
                 stores the FP32 logits into a request-compact scratch, and each
                 8 keys' largest order-preserving 32-bit key into a scratch 1/8
                 its size. Requests whose rows all fit the top-k are not scored.
  IndexerSelect  one CTA per row, exact over 32-bit keys: the K-th largest of the
                 row's 8-key maxima bounds its K-th largest key from below (to 22
                 bits: two 11-bit histograms), so only the keys of the about K
                 vectors at or above it are candidates; an exact radix select over
                 them (11, 11, then 10 bits) and one ordered pass write the
                 selected keys in key order. Ties at the threshold go to the
                 lowest key indices, so the result is deterministic. A row of at
                 most ``topk`` keys takes all of them (the short-context bypass).

Under DCP the scorer writes this rank's shard in DeepGEMM's logits layout (FP32
rows, key index = column), and the same select in "candidates" mode takes each
row's full local top-K as (score, global id) words for the FP32 DCP merge
(dcp_topk.py), as stock's FP32 rule does: no one-round merge.

logits[t, k] = scale[k] * 0.5 * ((s0 + s2) + (s1 + s3)), s_j = the FP32 FMA chain
over heads h = j mod 4 (in head order) of (x_h + |x_h|) * w[t, h], x_h = q[t, h] .
key[k]: DeepGEMM SM100 paged MQA logits with FP32 weights and logits
(sm100_mqa_logits.cuh, kIsFP32Reduce), on the FP8 query with unit MX scales and the
FP32 head weights that fold its scale (DsaIndex). No value is rounded to BF16.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, stream, to_cute
from ..common.primitives import (
    asm,
    bits_float,
    float_bits,
    load_words,
    store_shared_words,
)

PAGE = 64  # keys per indexer cache block
DIM = 128
HEADS = 32
TILE = 128  # keys per MMA tile (two pages)
ITEM_TILES = 16  # tiles per work item (the producer caches 32 page ids)
MAX_REQUESTS = 512  # tile prefix sum in shared memory
STAGES = 4
EPI_GROUPS = 2  # epilogue warpgroups, one TMEM accumulator each
PRODUCER = 4 * EPI_GROUPS
MMA_WARP = PRODUCER + 1
SCORE_THREADS = 32 * (MMA_WARP + 1)

SELECT_THREADS = 1024
SELECT_WARPS = SELECT_THREADS // 32
BINS = 2048  # 11-bit radix digits (the last digit of a 32-bit key: 10 bits)
UNROLL = 4  # 16-byte maxima loads in flight per thread
TOPS = 32768  # 32-bit vector maxima of a row kept in shared memory (128 KB)
CANDIDATES = 6144  # keys at or above the bound (more: the exact fallback)
HOT = 4096  # vectors holding a candidate (more: the exact fallback)
LOGITS_DTYPE = torch.float32  # stock Kimi K3's indexer logits (the op's default)
SF_DTYPE = cutlass.Float8E8M0FNU  # the MX scale factors (UE8M0)
SF_VEC = 32  # elements per scale factor
UNIT_SCALES = 0x7F7F7F7F  # four UE8M0 1.0 (the FP8 query's MX scales)


# --------------------------------------------------------------------------
# PTX helpers
# --------------------------------------------------------------------------
@asm()
def bulk_g2s(dst, src, barrier, size):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    code += "[$0], [$1], $3, [$2];"
    return code, "r,l,r,r,~{memory}", (dst, src, barrier, Int32(size))


@asm()
def store_b32(address, value):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def shared_add(pointer, value):
    return "red.shared.add.u32 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm(Int32)
def load_i32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Int32)
def load_shared_acquire(pointer):
    """A shared word read before any later memory operation of this thread, e.g.
    the mbarrier arrive that frees its buffer (ptxas issues that arrive without
    waiting for a plain ld.shared)."""
    return "ld.acquire.cta.shared::cta.b32 $0, [$1];", "=r,r,~{memory}", (pointer,)


@asm(Float32, pure=True)
def relu2(x: Float32):
    """2 relu(x) = x + |x| in FP32 (DeepGEMM's FP32 reduction; exact)."""
    return "{ .reg .f32 a; abs.f32 a, $1; add.rn.f32 $0, $1, a; }", "=f,f", (x,)


@asm(Float32, pure=True)
def ffma(a: Float32, b: Float32, c: Float32):
    """Return a * b + c, one rounding (fma.rn.f32, as __ffma2_rn per lane)."""
    return "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


@asm(Float32, pure=True)
def fadd(a: Float32, b: Float32):
    return "add.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def fmul(a: Float32, b: Float32):
    return "mul.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@asm(Uint32, pure=True)
def order32(bits):
    """Order-preserving 32-bit key of FP32 ``bits`` (a larger value, a larger key):
    the sign bit flipped, every bit of a negative value flipped."""
    code = "{ .reg .b32 m; shr.s32 m, $1, 31; or.b32 m, m, 0x80000000; "
    code += "xor.b32 $0, $1, m; }"
    return code, "=r,r", (Uint32(bits),)


@asm(Uint32, pure=True)
def umax(a, b):
    """Unsigned max of two 32-bit keys (max.u32: cutlass.max of Uint32 compares
    them signed)."""
    return "max.u32 $0, $1, $2;", "=r,r,r", (Uint32(a), Uint32(b))


@cute.jit
def stride_pages(page, tensor):
    """The element offset of page ``page`` of ``tensor`` (mode 0), in 64 bits."""
    return Int64(page) * Int64(tensor.stride[0])


@cute.jit
def table_entry(table, r: Int32, column: Int32) -> Int32:
    """``table``: (address, row stride, width) of an int32 block table."""
    address, pitch, _ = table
    return load_i32(address + (Int64(r) * Int64(pitch) + Int64(column)) * 4)


@cute.jit
def vector_words(address, v, plain: cutlass.Constexpr[bool] = False):
    """The eight FP32 logits (bits) of 8-key vector ``v`` of a row at ``address``
    (32-byte aligned rows)."""
    policy = None if plain else "cg"
    a = load_words(address + Int64(v) * 32, count=4, cache_policy=policy)
    b = load_words(address + Int64(v) * 32 + 16, count=4, cache_policy=policy)
    return (*a, *b)


@cute.jit
def vector_keys(words, v: Int32, length: Int32):
    """The order-preserving keys of a vector's eight FP32 logits; keys at or past
    ``length`` are 0 (below every logit's key)."""
    keys = [order32(words[e]) for e in range(8)]
    if (v + 1) * 8 > length:
        for e in cutlass.range_constexpr(8):
            if v * 8 + e >= length:
                keys[e] = Uint32(0)
    return keys


@cute.jit
def vector_max(keys) -> Uint32:
    m0 = umax(umax(keys[0], keys[1]), umax(keys[2], keys[3]))
    m1 = umax(umax(keys[4], keys[5]), umax(keys[6], keys[7]))
    return umax(m0, m1)


@cute.jit
def warp_inclusive(value: Uint32, lane: Int32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            value += up
    return value


@cute.jit
def topdown(
    counts,
    bins: cutlass.Constexpr[int],
    need,
    info,
    slot: cutlass.Constexpr[int],
    scan,
    tid,
    warp,
    lane,
):
    """info[slot] = the bin where the counts summed from the top bin down reach
    ``need``, info[slot + 1] = the count above it (unchanged when they never
    do)."""
    per = max(1, bins // SELECT_THREADS)
    top = bins - 1 - per * tid
    mine = Uint32(0)
    if top >= 0:
        for i in cutlass.range_constexpr(per):
            mine += counts[top - i]
    inclusive = warp_inclusive(mine, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, _ = warp_totals(scan, warp, lane)
    above = prior + inclusive - mine
    if above < need and above + mine >= need:
        running = above
        found = Int32(0)
        for i in cutlass.range_constexpr(per):
            count = counts[top - i]
            if found == 0 and running + count >= need:
                info[slot] = top - i
                info[slot + 1] = Int32(running)
                found = Int32(1)
            running += count
    cute.arch.sync_threads()


@cute.jit
def warp_totals(scan, warp, lane):
    """(sum of ``scan`` over the warps before ``warp``, sum over all): every
    warp scans the per-warp words itself."""
    x = Uint32(0)
    if lane < SELECT_WARPS:
        x = scan[lane]
    inclusive = warp_inclusive(x, lane)
    return (
        cute.arch.shuffle_sync(inclusive - x, warp),
        cute.arch.shuffle_sync(inclusive, 31),
    )


@cute.jit
def block_exclusive(value: Uint32, scan, warp, lane):
    """(exclusive prefix over the CTA's threads in order, total); ``scan``:
    one word per warp."""
    inclusive = warp_inclusive(value, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, total = warp_totals(scan, warp, lane)
    cute.arch.sync_threads()
    return prior + inclusive - value, total


@cute.jit
def warp_sum(value: Uint32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << i)
    return value


@cute.jit
def request_tiles(row_ends: cute.Tensor, r: Int32, q: cutlass.Constexpr[int], topk):
    """128-key tiles of request ``r``; 0 when all its rows fit the top-k (its
    last row sees the most keys)."""
    end = row_ends[r * q + q - 1]
    tiles = Int32(0)
    if end > topk:
        tiles = (end + Int32(TILE - 1)) // Int32(TILE)
    return tiles


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
@cute.jit
def _first_request(start: cute.Tensor, requests: Int32, g: Int32) -> Int32:
    """The request whose tiles contain global tile ``g`` (binary search)."""
    lo = Int32(0)
    hi = requests
    while lo < hi:
        mid = (lo + hi) // 2
        if start[mid + 1] > g:
            hi = mid
        else:
            lo = mid + 1
    return lo


@cute.jit
def _next_item(start: cute.Tensor, requests: Int32, g: Int32, r: Int32, hi: Int32):
    """The work item at global tile ``g`` of request ``r`` (the CTA range ends at
    ``hi``): (request, first key, tiles, next g, next r)."""
    r_end = start[r + 1]
    tiles = cutlass.min(cutlass.min(hi, r_end) - g, Int32(ITEM_TILES))
    k0 = (g - start[r]) * Int32(TILE)
    req = r
    g += tiles
    while r < requests and start[r + 1] <= g:
        r += 1
    return req, k0, tiles, g, r


def _mn_layout(t):
    """A tcgen05 accumulator as ((M), (N), stages)."""
    sh, sd = t.layout.shape, t.layout.stride
    new = cute.make_layout(
        ((sh[0][0], sh[1]), (sh[0][1], sh[2]), *sh[3:]),
        stride=((sd[0][0], sd[1]), (sd[0][1], sd[2]), *sd[3:]),
    )
    return cute.make_tensor(t.iterator, new)


class IndexerScore:
    """FP32 logits of ``rows`` rows (``q`` per request) into the request-compact
    scratch: request r's rows are consecutive ``tiles(r) * 128``-key rows starting
    at element ``q * 128 * (tiles of the requests before r)``.

    ``pitch`` > 0 (DCP): DeepGEMM's paged MQA logits layout instead, row ``t`` at
    element ``t * pitch`` (key index = column; ``pitch`` a multiple of 512, so
    1 KiB rows), no vector maxima, and every request is scored (``topk`` 0: a
    rank's share of the global top-k is unknown before the merge).

    ``fp4``: MXFP4 keys and query (E2M1, one UE8M0 scale per 32 elements), else
    FP8 E4M3 keys with their UE8M0 scales and the FP8 query at unit scales (the MX
    inputs stock passes DeepGEMM, mqa_logits_mx_inputs). Each 128-key tile is the
    block-scaled tcgen05 MMA chain DeepGEMM's SM100 paged MQA logits issue over
    K = 128 (kind::mxf8f6f4: four K32 steps; kind::mxf4: two K64 steps; the scale
    factors in TMEM), so the FP32 head scores are DeepGEMM's.

    TMEM holds one [128 x N] FP32 accumulator per epilogue warpgroup and the scale
    factors (keys: 4 columns; query: 4 per 128 rows). Where that passes its 512
    columns (N = q x 32 = 256), each tile's MMA splits into two N / 2 halves,
    warpgroup h taking half h of every tile (half of its rows), through a ring of
    three N / 2 accumulators (the MMA of the next tile's first half overlaps the
    epilogue of this one)."""

    def __init__(
        self,
        rows: int,
        q: int,
        ctas: int,
        topk: int,
        pitch: int = 0,
        fp4: bool = False,
    ):
        if rows % q or not 1 <= q <= 8 or rows // q > MAX_REQUESTS:
            raise ValueError(f"IndexerScore: rows {rows}, q {q}")
        if pitch % 512 or (pitch and topk):
            raise ValueError(f"IndexerScore: pitch {pitch}, top-k {topk}")
        self.rows, self.q, self.ctas, self.topk = rows, q, ctas, topk
        self.pitch, self.fp4 = pitch, fp4
        self.ab = cutlass.Float4E2M1FN if fp4 else cutlass.Float8E4M3FN
        self.n = q * HEADS
        sfb_cols = -(-self.n // 128) * 4
        self.halves = 1 if EPI_GROUPS * self.n + 4 + sfb_cols <= 512 else 2
        if self.halves == 2 and q % 2:
            raise ValueError(f"IndexerScore: q {q} splits into halves")
        self.nt = self.n // self.halves  # MMA N
        self.tiler = (TILE, self.nt, DIM)
        self.sfb_half = -(-self.nt // 128) * 4  # TMEM columns of a half's query SFs
        # Accumulator slots: one per warpgroup, or the halves' ring of three.
        self.slots = EPI_GROUPS if self.halves == 1 else 3
        self.sfa_col = self.slots * self.nt
        self.sfb_col = self.sfa_col + 4
        used = self.sfb_col + self.halves * self.sfb_half
        self.columns = max(32, 1 << (used - 1).bit_length())
        if self.columns > 512:
            raise ValueError(f"IndexerScore: {used} TMEM columns")
        # Bytes of one tile's keys and of one request's query rows.
        self.key_bytes = TILE * DIM * self.ab.width // 8

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,  # FP8 [rows * 32, 128, 1]; fp4: uint8 [rows * 32, 64, 1]
        qscales: cute.Tensor,  # Int32 [rows * 32] (fp4: UE8M0 x4 per row and head)
        weights: cute.Tensor,  # FP32 [rows * 32]
        keys: cute.Tensor,  # FP8 [64, 128, blocks]; fp4: uint8 [64, 64, blocks]
        scales: cute.Tensor,  # Int32 (UE8M0 x4) [blocks, 64]
        table_address: Int64,  # Int32 [requests, pages]: index block ids
        table_pitch: Int32,
        table_width: Int32,
        row_ends: cute.Tensor,  # Int32 [rows]
        logits: cute.Tensor,  # FP32 bits as Int32 [capacity]
        tops: cute.Tensor,  # Int32 [capacity / 8]: each 8 logits' largest key
        s,
    ):
        ab = self.ab
        if cutlass.const_expr(self.fp4):
            # The packed E2M1 bytes as 4-bit elements (two per byte).
            query = cute.make_tensor(
                cute.recast_ptr(query.iterator, dtype=ab),
                cute.make_layout(
                    (query.shape[0], DIM, 1),
                    stride=(DIM, 1, query.shape[0] * DIM),
                ),
            )
            keys = cute.make_tensor(
                cute.recast_ptr(keys.iterator, dtype=ab),
                cute.make_layout(
                    (PAGE, DIM, keys.shape[2]), stride=(DIM, 1, keys.stride[2] * 2)
                ),
            )
        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            ab,
            ab,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            SF_DTYPE,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.tiler[:2],
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, ab, STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, ab, 2 * self.halves)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(
            mma, self.tiler, SF_VEC, STAGES
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            mma, self.tiler, SF_VEC, 2 * self.halves
        )
        kind = tcgen05.SmemLayoutAtomKind.K_SW128
        if cutlass.const_expr(self.fp4):
            kind = tcgen05.SmemLayoutAtomKind.K_SW64  # 128 E2M1 = 64 B rows
        atom = sm100_utils.make_smem_layout_atom(kind, ab)
        page_layout = cute.tile_to_shape(atom, (PAGE, DIM, 2 * STAGES), (0, 1, 2))
        tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            keys,
            cute.slice_(page_layout, (None, None, 0)),
            (PAGE, DIM),
        )
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            query,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(ab, b_one)
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
            page_layout,
            scales,
            qscales,
            weights,
            (table_address, table_pitch, table_width),
            row_ends,
            logits,
            tops,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[SCORE_THREADS, 1, 1],
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def rows_out(
        self,
        lo: cutlass.Constexpr[int],
        hi: cutlass.Constexpr[int],
        t2r,
        tAll,
        cD,
        sW,
        sEnd,
        rows,
        bases,
    ):
        """Verify rows [lo, hi) of one tile (accumulator ``acc``'s tokens from
        ``lo`` when the tile's MMA splits in halves, else from 0): relu and the
        head weights -> FP32 logits, and each 8 keys' largest (the select's
        bound)."""
        row0, pitch, key, qs, acc, lt = rows
        out_base, tops_base = bases
        first: cutlass.Constexpr = lo if self.halves == 2 else 0
        frags = [cute.make_fragment_like(cD, Float32) for _ in range(lo, hi)]
        for tok in cutlass.range_constexpr(lo, hi):
            src = tAll[(None, None, None, 0, tok - first, acc)]
            cute.copy(t2r, src, frags[tok - lo])
        for tok in cutlass.range_constexpr(lo, hi):
            frag = frags[tok - lo]
            # Stock's precision (DeepGEMM SM100 paged MQA logits, FP32 weights and
            # logits): per head 2 relu(x) = x + |x| in FP32 on the block-scaled
            # MMA's FP32 score, four FP32 FMA chains over the FP32 head weights
            # (heads 4i, 4i + 1, 4i + 2, 4i + 3 in head order), (s0 + s2) + (s1 +
            # s3), halved. Nothing is rounded to BF16.
            s0 = Float32(0)
            s1 = Float32(0)
            s2 = Float32(0)
            s3 = Float32(0)
            for h in cutlass.range_constexpr(0, HEADS, 4):
                w = load_words(
                    sW.iterator + qs * self.n + tok * HEADS + h, count=4, shared=True
                )
                s0 = ffma(relu2(frag[h]), bits_float(w[0]), s0)
                s1 = ffma(relu2(frag[h + 1]), bits_float(w[1]), s1)
                s2 = ffma(relu2(frag[h + 2]), bits_float(w[2]), s2)
                s3 = ffma(relu2(frag[h + 3]), bits_float(w[3]), s3)
            total = fmul(fadd(fadd(s0, s2), fadd(s1, s3)), Float32(0.5))
            element = row0 + Int64(tok) * pitch + Int64(key)
            bits = Uint32(float_bits(total))
            if cutlass.const_expr(self.pitch):
                if key < sEnd[tok, qs]:
                    store_b32(out_base + element * 4, bits)
            else:
                top = Uint32(0)
                if key < sEnd[tok, qs]:
                    store_b32(out_base + element * 4, bits)
                    top = order32(bits)
                for i in cutlass.range_constexpr(3):
                    top = umax(top, cute.arch.shuffle_sync_bfly(top, 1 << i))
                if lt % 8 == 0:
                    store_b32(tops_base + (element >> 3) * 4, top)

    @cute.jit
    def transpose_sf(self, raw, dst, count, lane):
        """``count`` rows' UE8M0 words (``raw``: Int32 shared pointer, row order;
        None: unit scales) -> one 128-row block of the tcgen05 SF atom at ``dst``
        (row r's word at (r % 32) * 4 + r // 32): lane l's 16 bytes hold rows l,
        l + 32, l + 64, l + 96."""
        words = []
        for i in cutlass.range_constexpr(4):
            word = Uint32(UNIT_SCALES)
            if cutlass.const_expr(raw is not None):
                word = Uint32(0)
                if i * 32 + lane < count:
                    word = Uint32(
                        load_words(raw + i * 32 + lane, count=1, shared=True)[0]
                    )
            words.append(word)
        store_shared_words(dst + lane * 4, tuple(words))

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
        page_layout: cute.ComposedLayout,
        scales: cute.Tensor,
        qscales: cute.Tensor,
        weights: cute.Tensor,
        table,
        row_ends: cute.Tensor,
        logits: cute.Tensor,
        tops: cute.Tensor,
    ):
        Q, H = self.q, self.halves
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        qsc_words: cutlass.Constexpr = 2 * self.n if self.fp4 else 4

        @cute.struct
        class Storage:
            kv_full: cute.struct.MemRange[cutlass.Int64, STAGES]
            kv_empty: cute.struct.MemRange[cutlass.Int64, STAGES]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            q_empty: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.slots]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.slots]
            tmem_hold: Int32
            row_end: cute.struct.MemRange[Int32, 8 * 2]
            start: cute.struct.MemRange[Int32, MAX_REQUESTS + 1]
            w: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.n], 16]
            scale: cute.struct.Align[cute.struct.MemRange[Int32, STAGES * TILE], 16]
            qsc: cute.struct.Align[cute.struct.MemRange[Int32, qsc_words], 16]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        sA = smem.allocate_tensor(self.ab, a_layout.outer, 1024, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(self.ab, b_layout.outer, 1024, swizzle=b_layout.inner)
        sSFA = smem.allocate_tensor(SF_DTYPE, sfa_layout, 128)
        sSFB = smem.allocate_tensor(SF_DTYPE, sfb_layout, 128)
        sPage = cute.make_tensor(sA.iterator, page_layout.outer)
        sW = st.w.get_tensor(cute.make_layout((self.n, 2)))
        sEnd = st.row_end.get_tensor(cute.make_layout((8, 2)))
        sStart = st.start.get_tensor(cute.make_layout((MAX_REQUESTS + 1,)))
        sSc = st.scale.get_tensor(cute.make_layout((TILE, STAGES)))
        sQsc = st.qsc.get_tensor(cute.make_layout((qsc_words,)))
        # The SF atoms as 32-bit words: one 128-row block is 128 words.
        sfa_words = cute.recast_ptr(sSFA.iterator, dtype=Int32)
        sfb_words = cute.recast_ptr(sSFB.iterator, dtype=Int32)
        sfb_stage: cutlass.Constexpr = self.sfb_half * 32  # words per (qs, half)

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + 4 * EPI_GROUPS)
            for i in cutlass.range_constexpr(self.slots):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 128)
            cute.arch.mbarrier_init_fence()
        if warp == PRODUCER:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)

        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=1, num_threads=SCORE_THREADS
            ),
            allocator_warp_id=MMA_WARP,
        )
        tmem.allocate(self.columns)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, self.slots))
        tAcc = cute.make_tensor(tmem_ptr, acc_fake.layout)

        # The schedule: sStart[r] = first global tile of request r (row ends are
        # step metadata, written before the graph runs).
        requests = row_ends.shape[0] // Q
        for r in cutlass.range(tidx, requests, SCORE_THREADS, unroll=1):
            sStart[r + 1] = request_tiles(row_ends, r, Q, self.topk)
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

        gQ = cute.local_tile(
            mQ, cute.slice_(self.tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = mma.get_slice(0)
        tCgQ = thr_mma.partition_B(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        gK = cute.local_tile(mK, (PAGE, DIM), (None, None, None))
        tKsK, tKgK = cpasync.tma_partition(
            tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sPage, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        it = Int32(0)  # tiles issued / consumed by this role
        j = Int32(0)  # items issued / consumed by this role
        g = lo
        r = r_lo
        max_page = table[2] - 1
        blocks = scales.shape[0]
        if warp == PRODUCER:
            # One item of metadata ahead: the next item's dependent loads overlap
            # the current item's tiles.
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
            q_tx: cutlass.Constexpr = self.n * 4 * (2 if self.fp4 else 1)
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
                cute.arch.mbarrier_wait(q_empty + qs, ((j // 2) & 1) ^ 1)
                if lane < Q:
                    sEnd[lane, qs] = my_end
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        q_full + qs, self.q_bytes * H + q_tx
                    )
                    bulk_g2s(
                        sW.iterator + qs * self.n,
                        weights.iterator + req * Q * HEADS,
                        q_full + qs,
                        self.n * 4,
                    )
                    if cutlass.const_expr(self.fp4):
                        bulk_g2s(
                            sQsc.iterator + qs * self.n,
                            qscales.iterator + req * Q * HEADS,
                            q_full + qs,
                            self.n * 4,
                        )
                for h in cutlass.range_constexpr(H):
                    cute.copy(
                        tma_q,
                        tQgQ[(None, req * H + h, 0, 0)],
                        tQsQ[(None, qs * H + h)],
                        tma_bar_ptr=q_full + qs,
                    )
                j += 1
                for t in cutlass.range(ntile, unroll=1):
                    stage = it % STAGES
                    p0 = cute.arch.shuffle_sync(my_page, 2 * t)
                    p1 = cute.arch.shuffle_sync(my_page, 2 * t + 1)
                    # A partial last tile's second page may lie past the request's
                    # blocks: keep its (unused) scale read in bounds.
                    p1 = cutlass.min(cutlass.max(p1, Int32(0)), blocks - 1)
                    ph = ((it // STAGES) & 1) ^ 1
                    cute.arch.mbarrier_wait(kv_empty + stage, ph)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kv_full + stage, self.key_bytes + TILE * 4
                        )
                        # 64-bit page offsets: page ids index the whole KV pool
                        # view, so page x stride (2112 words) passes Int32 from
                        # page 1,016,801 (the static stride makes it a 32-bit
                        # multiply, and the bulk copy is not bounds-checked).
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
            # The scale factors in TMEM (after the accumulators): the keys' (one
            # slot, rewritten per tile) and the query's (one slot per half,
            # rewritten per item). tcgen05.cp and tcgen05.mma issue in order, so a
            # copy lands after the MMAs before it read the slot.
            sfa_tmem = blockscaled_utils.make_tmem_layout_sfa(
                mma, self.tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            )
            sfb_tmem = blockscaled_utils.make_tmem_layout_sfb(
                mma, self.tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            )
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.sfa_col, dtype=SF_DTYPE), sfa_tmem
            )
            copy_atom = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF_DTYPE
            )
            copy_a, src_a, dst_a = s2t_partition(copy_atom, sSFA, tCtSFA)
            tCtSFB = []
            copies_b = []
            for h in cutlass.range_constexpr(H):
                col = self.sfb_col + h * self.sfb_half
                tSF = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + col, dtype=SF_DTYPE), sfb_tmem
                )
                tCtSFB.append(tSF)
                copies_b.append(s2t_partition(copy_atom, sSFB, tSF))
            if cutlass.const_expr(not self.fp4):
                # The FP8 query's unit scales (UE8M0 127), in every query stage.
                for b in cutlass.range_constexpr(2 * H * self.sfb_half // 4):
                    self.transpose_sf(None, sfb_words + b * 128, 0, lane)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
            while g < hi:
                _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                if cutlass.const_expr(self.fp4):
                    # This item's query scale words -> the SF atom of each half.
                    for h in cutlass.range_constexpr(H):
                        for b in cutlass.range_constexpr(self.sfb_half // 4):
                            q_raw = sQsc.iterator + qs * self.n + h * self.nt + b * 128
                            q_dst = sfb_words + (qs * H + h) * sfb_stage + b * 128
                            count = min(self.nt - b * 128, 128)
                            self.transpose_sf(q_raw, q_dst, count, lane)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                for h in cutlass.range_constexpr(H):
                    copy_b, src_b, dst_b = copies_b[h]
                    cute.copy(
                        copy_b, src_b[(None, None, None, None, qs * H + h)], dst_b
                    )
                for _t in cutlass.range(ntile, unroll=1):
                    stage = it % STAGES
                    cute.arch.mbarrier_wait(kv_full + stage, (it // STAGES) & 1)
                    # The tile's key scale words (key order) -> the SF atom.
                    k_raw = sSc.iterator + stage * TILE
                    self.transpose_sf(k_raw, sfa_words + stage * 128, TILE, lane)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    cute.copy(copy_a, src_a[(None, None, None, None, stage)], dst_a)
                    for h in cutlass.range_constexpr(H):
                        # Accumulator slot a of this (half-)tile, free once its
                        # previous user's epilogue read it.
                        a = it % EPI_GROUPS
                        ph = ((it // EPI_GROUPS) & 1) ^ 1
                        if cutlass.const_expr(H == 2):
                            k = it * 2 + h
                            a = k % 3
                            ph = ((k // 3) & 1) ^ 1
                        cute.arch.mbarrier_wait(acc_empty + a, ph)
                        mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            mma.set(
                                tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator
                            )
                            mma.set(
                                tcgen05.Field.SFB, tCtSFB[h][(None, None, kb)].iterator
                            )
                            cute.gemm(
                                mma,
                                tAcc[(None, None, None, a)],
                                tCrA[(None, None, kb, stage)],
                                tCrB[(None, None, kb, qs * H + h)],
                                tAcc[(None, None, None, a)],
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(acc_full + a)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + stage)
                    it += 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty + qs)
                j += 1
        else:
            # Epilogue: warpgroup ``acc`` owns TMEM accumulator acc: every other
            # tile, or (halves) half ``acc`` of every tile's rows from the ring.
            acc = warp // 4
            lt = tidx % 128
            ld_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32
            )
            acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
            t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
            thr = t2r.get_slice(lt)
            tAll = thr.partition_S(acc_cols)
            cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
            bases = logits.iterator.toint(), tops.iterator.toint()
            half: cutlass.Constexpr = Q // 2
            while g < hi:
                req, k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                if cutlass.const_expr(self.pitch):
                    pitch = Int64(self.pitch)
                    row0 = Int64(req * Q) * pitch
                else:
                    tiles = sStart[req + 1] - sStart[req]
                    row0 = Int64(sStart[req]) * Int64(TILE * Q)
                    pitch = Int64(tiles) * Int64(TILE)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                for t in cutlass.range(ntile, unroll=1):
                    if cutlass.const_expr(H == 2):
                        key = k0 + t * TILE + lt
                        k = it * 2 + acc
                        slot = k % 3
                        cute.arch.mbarrier_wait(acc_full + slot, (k // 3) & 1)
                        rows = row0, pitch, key, qs, slot, lt
                        if acc == 0:
                            self.rows_out(0, half, t2r, tAll, cD, sW, sEnd, rows, bases)
                        else:
                            self.rows_out(half, Q, t2r, tAll, cD, sW, sEnd, rows, bases)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.mbarrier_arrive(acc_empty + slot)
                    else:
                        if it % EPI_GROUPS == acc:
                            key = k0 + t * TILE + lt
                            ph = (it // EPI_GROUPS) & 1
                            cute.arch.mbarrier_wait(acc_full + acc, ph)
                            rows = row0, pitch, key, qs, acc, lt
                            self.rows_out(0, Q, t2r, tAll, cD, sW, sEnd, rows, bases)
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.mbarrier_arrive(acc_empty + acc)
                    it += 1
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(q_empty + qs)
                j += 1

        cute.arch.sync_threads()
        if warp == MMA_WARP:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


def s2t_partition(copy_atom, sSF, tSF):
    """The tcgen05.cp (4x32dp128bit) of a staged SF atom in shared memory into its
    TMEM slot: (tiled copy, source descriptors (.., stage), destination)."""
    tiled = tcgen05.make_s2t_copy(copy_atom, cute.filter_zeros(tSF))
    thr = tiled.get_slice(0)
    source = tcgen05.get_s2t_smem_desc_tensor(
        tiled, thr.partition_S(cute.filter_zeros(sSF))
    )
    return tiled, source, thr.partition_D(cute.filter_zeros(tSF))


# --------------------------------------------------------------------------
# Select (exact, 32-bit keys)
# --------------------------------------------------------------------------
# The radix digits of a 32-bit key, from the top: (shift, width).
DIGITS = ((21, 11), (10, 11), (0, 10))


class IndexerSelect:
    """The exact top-``topk`` of every row (one CTA per row) of FP32 logits, ties to
    the lowest key indices, written in key order. ``mode``:

    - "rows" (TP, the scorer's request-compact scratch and its 8-key maxima):
      ``out`` [rows, topk] the physical KV rows (``block_rows`` rows per
      ``page``-token block of the MLA cache, a power of two: 64, or e.g. 2048 in
      hybrid models whose attention pages match the mamba state's) of the
      selected keys, -1 past the row's count; ``ids`` (``logical``): the key
      indices too;
    - "compact_ids" (TP, the same scratch and maxima as "rows"): ``out`` [rows,
      topk] the key indices, -1 past the count (the top-k buffer a FlashMLA
      sparse decode reads);
    - "ids" (row ``t`` at ``t * pitch`` FP32 elements, DeepGEMM's layout):
      ``out`` [rows, topk] the key indices, -1 past the count;
    - "candidates" (the same layout; DCP, this rank's local logits): ``out``
      [rows, 2 topk] int32, per selected key a 64-bit word, low half
      0xFFFFFFFF - its global id (``rank``, ``world``, ``interleave``), high half
      its order-preserving score key: the FP32 DCP merge's candidates (zero words
      past the count).

    ``valid`` [rows]: the counts.

    The K-th largest of the 8-key vector maxima bounds the K-th largest key from
    below (each of those vectors holds a key at least as large): pass 1
    histograms the top 11 bits of the maxima (a row of fewer than 2K vectors: of
    its keys) and pass 2 the next 11 bits of those in the bound's bin, so the
    bound holds 22 bits; pass 3 lists the keys at or above it (about K) in key
    order, and a radix select over them (11, 11, 10 bits) finds the threshold.
    Rows whose candidates overflow the shared list take the same select over
    every key at or above the bound."""

    def __init__(
        self,
        rows: int,
        q: int,
        topk: int,
        mode: str = "rows",
        logical: bool = False,
        release=True,
        page: int = PAGE,
        world: int = 1,
        interleave: int = 1,
    ):
        if topk % SELECT_THREADS or rows % q:
            raise ValueError(f"IndexerSelect: top-k {topk}, rows {rows}, q {q}")
        if page <= 0 or page & (page - 1):
            raise ValueError(f"IndexerSelect: page {page}")
        if mode not in ("rows", "compact_ids", "ids", "candidates"):
            raise ValueError(f"IndexerSelect: mode {mode!r}")
        self.rows, self.q, self.topk, self.mode = rows, q, topk, mode
        # The scorer's request-compact scratch and 8-key maxima (TP), else rows of
        # ``pitch`` FP32 elements (DeepGEMM's layout).
        self.compact = mode in ("rows", "compact_ids")
        self.logical = logical and mode == "rows"
        # release: launch the next kernel early (PDL). Off when it reads the rows
        # without waiting for this grid first (FlashInfer prims_ts' split-KV decode
        # launches with PDL but does not wait before it reads its indices).
        self.release = release
        self.page_shift = page.bit_length() - 1
        self.world, self.interleave = world, interleave

    @cute.jit
    def pdl(self):
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.release):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        logits: Int64,  # FP32 logits: the scratch, or row 0 of the pitched rows
        pitch: Int32,  # FP32 elements per row (pitched modes)
        tops: cute.Tensor,  # Int32 [capacity / 8] ("rows")
        row_ends: cute.Tensor,
        table_address: Int64,
        table_pitch: Int32,
        block_rows: Int32,
        rank: Int32,
        out: cute.Tensor,
        valid: cute.Tensor,
        ids: cute.Tensor,
        s,
    ):
        table = (table_address, table_pitch, Int32(0))
        args = (logits, pitch, tops, row_ends, table, block_rows, rank, out, valid, ids)
        self.kernel(args).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def emit(self, emit_args, pos, index, key):
        """Selected key ``index`` (order key ``key``) into output slot ``pos``."""
        out, ids, table, request, row, block_rows, rank = emit_args
        if cutlass.const_expr(self.mode == "rows"):
            shift: cutlass.Constexpr = self.page_shift
            page = table_entry(table, request, index >> shift)
            out[row, pos] = page * block_rows + (index & ((1 << shift) - 1))
            if cutlass.const_expr(self.logical):
                ids[row, pos] = index
        elif cutlass.const_expr(self.mode in ("ids", "compact_ids")):
            out[row, pos] = index
        else:
            span: cutlass.Constexpr = self.interleave
            stride: cutlass.Constexpr = self.world * self.interleave
            gid = (index // span) * stride + rank * span + index % span
            out[row, 2 * pos] = Int32(Uint32(0xFFFFFFFF) - Uint32(gid))
            out[row, 2 * pos + 1] = Int32(key)

    @cute.jit
    def empty(self, emit_args, pos):
        out, ids, _, _, row, _, _ = emit_args
        if cutlass.const_expr(self.mode == "candidates"):
            out[row, 2 * pos] = Int32(0)
            out[row, 2 * pos + 1] = Int32(0)
        else:
            out[row, pos] = Int32(-1)
            if cutlass.const_expr(self.logical):
                ids[row, pos] = Int32(-1)

    @cute.jit
    def vector_top(self, tops_s, tops_address, address, v, length):
        """Vector ``v``'s largest key: shared memory holds the first TOPS; past
        them the scorer's maxima ("rows") or the vector itself."""
        top = Uint32(0)
        if v < TOPS:  # noqa: SIM108 (a branch: the global load stays conditional)
            top = tops_s[v]
        else:
            if cutlass.const_expr(self.compact):
                top = Uint32(load_words(tops_address + Int64(v) * 4, count=1)[0])
            else:
                top = vector_max(vector_keys(vector_words(address, v), v, length))
        return top

    @cute.jit
    def digit(self, d: cutlass.Constexpr[int], key, prefix, hist):
        """Radix pass ``d``: histogram the digit of ``key`` when its higher digits
        are ``prefix``."""
        shift, width = DIGITS[d]
        bin_ = (key >> Uint32(shift)) & Uint32((1 << width) - 1)
        if cutlass.const_expr(d == 0):
            shared_add(hist.iterator + bin_, 1)
        else:
            if (key >> Uint32(shift + width)) == prefix:
                shared_add(hist.iterator + bin_, 1)

    @cute.jit
    def write_selected(
        self, key, index, n, threshold, take, carry_a, carry_t, lane, emit_args
    ):
        """One warp round of the ordered output: ``n`` (0/1) per lane of key
        ``key`` at ``index``; returns the carries."""
        n_a = Uint32(0)
        n_t = Uint32(0)
        if n != Uint32(0):
            if key > threshold:
                n_a = Uint32(1)
            elif key == threshold:
                n_t = Uint32(1)
        packed = (n_a << Uint32(16)) | n_t
        inclusive = warp_inclusive(packed, lane)
        exclusive = inclusive - packed
        total = cute.arch.shuffle_sync(inclusive, 31)
        pos_a = carry_a + (exclusive >> Uint32(16))
        pos_t = carry_t + (exclusive & Uint32(0xFFFF))
        if n_a != Uint32(0):
            pos = Int32(pos_a + cutlass.min(pos_t, take))
            self.emit(emit_args, pos, index, key)
        if n_t != Uint32(0) and pos_t < take:
            pos = Int32(pos_a + pos_t)
            self.emit(emit_args, pos, index, key)
        return carry_a + (total >> Uint32(16)), carry_t + (total & Uint32(0xFFFF))

    @cute.kernel
    def kernel(self, args):
        base, pitch, tops, row_ends, table, block_rows, rank, out, valid, ids = args
        Q, K = self.q, self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        request = row // Q

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        above_w = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        ties_w = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        cand_index = smem.allocate_tensor(Int32, cute.make_layout(CANDIDATES), 16)
        cand_key = smem.allocate_tensor(Uint32, cute.make_layout(CANDIDATES), 16)
        hot_list = smem.allocate_tensor(Int32, cute.make_layout(HOT), 16)
        tops_s = smem.allocate_tensor(Uint32, cute.make_layout(TOPS), 16)

        if cutlass.const_expr(self.compact):  # noqa: SIM102 (constexpr)
            # This row's logits: request ``request``'s row ``row % Q`` of the
            # scratch.
            if warp == 0:
                before = Int32(0)
                for first in cutlass.range(0, request + 1, 32, unroll=1):
                    r = first + lane
                    tiles = Int32(0)
                    if r < request:
                        tiles = request_tiles(row_ends, r, Q, K)
                    for i in cutlass.range_constexpr(5):
                        tiles += cute.arch.shuffle_sync_bfly(tiles, 1 << i)
                    before += tiles
                if lane == 0:
                    info[0] = before
                    info[1] = request_tiles(row_ends, request, Q, K)
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        cute.arch.sync_threads()
        length = row_ends[row]
        address = base + Int64(row) * Int64(pitch) * 4
        tops_address = Int64(0)
        if cutlass.const_expr(self.compact):
            element = Int64(info[0]) * Int64(TILE * Q)
            element += Int64(row % Q) * Int64(info[1]) * Int64(TILE)
            address = base + element * 4
            tops_address = tops.iterator.toint() + (element >> 3) * 4
        emit_args = out, ids, table, request, row, block_rows, rank
        if length <= K:
            # Short-context bypass: every key ("rows" / "ids": no logits read).
            self.pdl()
            n = cutlass.max(length, Int32(0))
            for e in cutlass.range(tid, K, SELECT_THREADS, unroll=1):
                if e < n:
                    key = Uint32(0)
                    if cutlass.const_expr(self.mode == "candidates"):
                        word = load_words(address + Int64(e) * 4, count=1)[0]
                        key = order32(word)
                    self.emit(emit_args, e, e, key)
                else:
                    self.empty(emit_args, e)
            if tid == 0:
                valid[row] = n
        else:
            self.select(
                address,
                tops_address,
                length,
                emit_args,
                tid,
                warp,
                lane,
                (
                    hist,
                    scan,
                    above_w,
                    ties_w,
                    info,
                    cand_index,
                    cand_key,
                    hot_list,
                    tops_s,
                ),
            )
            if tid == 0:
                valid[row] = Int32(K)

    @cute.jit
    def select(self, address, tops_address, length, emit_args, tid, warp, lane, sm):
        hist, scan, above_w, ties_w, info, cand_index, cand_key, hot_list, tops_s = sm
        K = self.topk
        vectors = (length + 7) // 8
        # Under 2K vectors the K-th largest maximum bounds loosely: histogram the
        # keys (the bound is then the threshold's own 22-bit bin).
        dense = vectors < 2 * K
        # Lane runs (in key order); an odd run spreads the lanes' shared reads
        # over the banks.
        run = (vectors + SELECT_THREADS - 1) // SELECT_THREADS
        run = run + 1 - run % 2
        l0 = cutlass.min(tid * run, vectors)
        l1 = cutlass.min(l0 + run, vectors)
        self.pdl()

        # Pass 1: the top 11 bits of the vector maxima (dense: of the keys); the
        # maxima of the first TOPS vectors into shared memory.
        if cutlass.const_expr(self.compact):
            if dense:
                self.maxima_pass(address, vectors, length, dense, hist, tops_s, tid)
            else:
                chunks = (vectors + 3) // 4
                step = SELECT_THREADS * UNROLL
                for first in cutlass.range(0, chunks, step, unroll=1):
                    batch = []
                    for u in cutlass.range_constexpr(UNROLL):
                        j = first + u * SELECT_THREADS + tid
                        words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if j < chunks:
                            words = load_words(tops_address + Int64(j) * 16, count=4)
                        batch.append(words)
                    for u in cutlass.range_constexpr(UNROLL):
                        j = first + u * SELECT_THREADS + tid
                        if j < chunks:
                            if j * 4 < TOPS:
                                store_shared_words(tops_s.iterator + j * 4, batch[u])
                            for e in cutlass.range_constexpr(4):
                                if j * 4 + e < vectors:
                                    top = Uint32(batch[u][e])
                                    shared_add(hist.iterator + (top >> Uint32(21)), 1)
        else:
            self.maxima_pass(address, vectors, length, dense, hist, tops_s, tid)
        cute.arch.sync_threads()
        topdown(hist, BINS, Uint32(K), info, 2, scan, tid, warp, lane)
        b0 = Uint32(info[2])
        need = Uint32(K) - Uint32(info[3])
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        cute.arch.sync_threads()
        # Pass 2: the next 11 bits of those in bin b0 -> the 22-bit bound.
        if dense:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                if (tops_s[v] >> Uint32(21)) >= b0:
                    keys = vector_keys(vector_words(address, v), v, length)
                    for e in cutlass.range_constexpr(8):
                        if (keys[e] >> Uint32(21)) == b0 and v * 8 + e < length:
                            bin_ = (keys[e] >> Uint32(10)) & Uint32(0x7FF)
                            shared_add(hist.iterator + bin_, 1)
        else:
            for v in cutlass.range(l0, l1, unroll=1):
                top = self.vector_top(tops_s, tops_address, address, v, length)
                if (top >> Uint32(21)) == b0:
                    shared_add(hist.iterator + ((top >> Uint32(10)) & Uint32(0x7FF)), 1)
        cute.arch.sync_threads()
        topdown(hist, BINS, need, info, 4, scan, tid, warp, lane)
        low = (b0 << Uint32(21)) | (Uint32(info[4]) << Uint32(10))

        # Pass 3: the candidates (keys >= low) in key order. The hot vectors
        # (largest key >= low; about K of them) are listed from the maxima, then
        # every thread loads a run of the list, counts its keys >= low and writes
        # them at its offset.
        count = Uint32(0)
        for v in cutlass.range(l0, l1, unroll=1):
            if self.vector_top(tops_s, tops_address, address, v, length) >= low:
                count += Uint32(1)
        slot, hot = block_exclusive(count, scan, warp, lane)
        if hot <= Uint32(HOT):
            for v in cutlass.range(l0, l1, unroll=1):
                if self.vector_top(tops_s, tops_address, address, v, length) >= low:
                    hot_list[slot] = v
                    slot += Uint32(1)
        cute.arch.sync_threads()
        listed = Uint32(CANDIDATES + 1)
        if hot <= Uint32(HOT):
            per = (hot + SELECT_THREADS - 1) // SELECT_THREADS
            h0 = cutlass.min(Uint32(tid) * per, hot)
            h1 = cutlass.min(h0 + per, hot)
            count = Uint32(0)
            for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                if h0 + u < h1:
                    v = hot_list[h0 + u]
                    keys = vector_keys(vector_words(address, v, True), v, length)
                    for e in cutlass.range_constexpr(8):
                        if keys[e] >= low:
                            count += Uint32(1)
            slot, listed = block_exclusive(count, scan, warp, lane)
            if listed <= Uint32(CANDIDATES):
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    if h0 + u < h1:
                        v = hot_list[h0 + u]
                        keys = vector_keys(vector_words(address, v, True), v, length)
                        for e in cutlass.range_constexpr(8):
                            if keys[e] >= low:
                                cand_index[slot] = v * 8 + e
                                cand_key[slot] = keys[e]
                                slot += Uint32(1)
        cute.arch.sync_threads()
        listed_ok = listed <= Uint32(CANDIDATES)
        # Each warp owns one contiguous share of the list (in key order), or of
        # the row's vectors when the list overflowed (e.g. rows of many equal
        # logits: every key >= low is then read from the logits per pass).
        share = (listed + SELECT_WARPS - 1) // SELECT_WARPS
        first = cutlass.min(warp * share, listed)
        last = cutlass.min(first + share, listed)
        per_warp = (vectors + SELECT_WARPS - 1) // SELECT_WARPS
        w0 = cutlass.min(warp * per_warp, vectors)
        w1 = cutlass.min(w0 + per_warp, vectors)

        # The threshold: three radix digits over the candidates.
        need = Uint32(K)
        prefix = Uint32(0)
        for d in cutlass.range_constexpr(3):
            shift, width = DIGITS[d]
            if cutlass.const_expr(d > 0):
                for b in cutlass.range(tid, 1 << width, SELECT_THREADS, unroll=1):
                    hist[b] = Uint32(0)
                cute.arch.sync_threads()
            else:
                for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
                    hist[b] = Uint32(0)
                cute.arch.sync_threads()
            if listed_ok:
                for i in cutlass.range(first + lane, last, 32, unroll=1):
                    self.digit(d, cand_key[i], prefix, hist)
            else:
                for v in cutlass.range(w0 + lane, w1, 32, unroll=1):
                    keys = vector_keys(vector_words(address, v), v, length)
                    for e in cutlass.range_constexpr(8):
                        if keys[e] >= low:
                            self.digit(d, keys[e], prefix, hist)
            cute.arch.sync_threads()
            topdown(hist, 1 << width, need, info, 6 + 2 * d, scan, tid, warp, lane)
            prefix = (prefix << Uint32(width)) | Uint32(info[6 + 2 * d])
            need = need - Uint32(info[7 + 2 * d])
        threshold = prefix
        take = need  # the ties at the threshold to take (the lowest indices)

        # Each warp's keys above the threshold and ties at it, then its carries.
        n_a = Uint32(0)
        n_t = Uint32(0)
        if listed_ok:
            for i in cutlass.range(first + lane, last, 32, unroll=1):
                key = cand_key[i]
                if key > threshold:
                    n_a += Uint32(1)
                elif key == threshold:
                    n_t += Uint32(1)
        else:
            for v in cutlass.range(w0 + lane, w1, 32, unroll=1):
                keys = vector_keys(vector_words(address, v), v, length)
                for e in cutlass.range_constexpr(8):
                    if keys[e] > threshold:
                        n_a += Uint32(1)
                    elif keys[e] == threshold:
                        n_t += Uint32(1)
        n_a = warp_sum(n_a)
        n_t = warp_sum(n_t)
        if lane == 0:
            above_w[warp] = n_a
            ties_w[warp] = n_t
        cute.arch.sync_threads()
        carry_a, _ = warp_totals(above_w, warp, lane)
        carry_t, _ = warp_totals(ties_w, warp, lane)
        if listed_ok:
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
        else:
            for start in cutlass.range(w0, w1, 32, unroll=1):
                v = start + lane
                keys = [Uint32(0)] * 8
                if v < w1:
                    keys = vector_keys(vector_words(address, v), v, length)
                a = Uint32(0)
                t = Uint32(0)
                for e in cutlass.range_constexpr(8):
                    if keys[e] > threshold:
                        a += Uint32(1)
                    elif keys[e] == threshold:
                        t += Uint32(1)
                packed = (a << Uint32(16)) | t
                inclusive = warp_inclusive(packed, lane)
                exclusive = inclusive - packed
                total = cute.arch.shuffle_sync(inclusive, 31)
                pos_a = carry_a + (exclusive >> Uint32(16))
                pos_t = carry_t + (exclusive & Uint32(0xFFFF))
                if packed != Uint32(0):
                    for e in cutlass.range_constexpr(8):
                        key = keys[e]
                        if key > threshold:
                            pos = Int32(pos_a + cutlass.min(pos_t, take))
                            self.emit(emit_args, pos, v * 8 + e, key)
                            pos_a += Uint32(1)
                        elif key == threshold:
                            if pos_t < take:
                                self.emit(
                                    emit_args, Int32(pos_a + pos_t), v * 8 + e, key
                                )
                            pos_t += Uint32(1)
                carry_a += total >> Uint32(16)
                carry_t += total & Uint32(0xFFFF)

    @cute.jit
    def maxima_pass(self, address, vectors, length, dense, hist, tops_s, tid):
        """Pass 1 from the logits: each vector's largest key into shared memory
        (the first TOPS), and the top 11 bits of its keys (``dense``) or of its
        largest key into ``hist``."""
        for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
            keys = vector_keys(vector_words(address, v), v, length)
            top = vector_max(keys)
            if v < TOPS:
                tops_s[v] = top
            if dense:
                for e in cutlass.range_constexpr(8):
                    if v * 8 + e < length:
                        shared_add(hist.iterator + (keys[e] >> Uint32(21)), 1)
            else:
                shared_add(hist.iterator + (top >> Uint32(21)), 1)


# --------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------
_SCRATCH: dict = {}


def kv_cache_blocks():
    """The 64-token blocks the KV cache holds, if known before it is allocated (a
    num_gpu_blocks override, or a cache already sized): a request's context fits in
    it, so it bounds the fused indexer's logits with the longest context. None if
    unknown (the KV cache is sized after the specialized decode is built)."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None:
        return None
    cache = config.cache_config
    blocks = cache.num_gpu_blocks or cache.num_gpu_blocks_override
    if not blocks or not cache.block_size:
        return None
    return -(-blocks * cache.block_size // PAGE)


def logits_scratch(device, elements: int, name: str = "logits") -> torch.Tensor:
    """One int32 scratch (FP32 logits bits, or 32-bit maxima) per device shared by
    every layer (every layer prepares before any graph capture, so the last,
    largest allocation serves them all)."""
    t = _SCRATCH.get((device, name))
    if t is None or t.numel() < elements:
        t = torch.empty(elements, dtype=torch.int32, device=device)
        _SCRATCH[device, name] = t
    return t


@functools.cache
def select_outputs(device, rows, topk, fill=-1):
    """The physical rows and counts, shared by every layer (each layer's consumer
    reads them before the next layer's indexer runs), and the per-row ends of
    steps with fewer requests than rows."""
    out = torch.full((rows, topk), fill, dtype=torch.int32, device=device)
    valid = torch.zeros(rows, dtype=torch.int32, device=device)
    return out, valid, torch.zeros(rows, dtype=torch.int32, device=device)


@functools.cache
def unit_scales(device, rows):
    """The FP8 query's scale words [rows * 32] int32 the scorer takes (unread: its
    MX scales are the unit ones), shared by every layer."""
    return torch.zeros(rows * HEADS, dtype=torch.int32, device=device)


def dcp_interleave(indexer) -> int:
    """The indexer's DCP interleave once final (after the block size setup, ahead of
    the KV cache and of any capture), read from its parallel config: the
    KimiK3DSAIndexer property needs a forward context."""
    config = getattr(indexer, "_parallel_config", None)
    if config is not None:
        return config.cp_kv_cache_interleave_size
    return indexer.cp_kv_cache_interleave_size


def local_context(indexer) -> int:
    """The most keys of one context on a DCP rank (its interleave groups)."""
    world, span = indexer.dcp_world_size, dcp_interleave(indexer)
    groups = -(-indexer.max_model_len // span)
    return -(-groups // world) * span


def reserve(indexer, device, max_rows, dcp=False, blocks=None):
    """The fused indexer's buffers of ``max_rows`` rows, shared by every layer:
    (logits, tops, rows, counts, ends, pitch). ``dcp``: this rank's keys of every
    row in DeepGEMM's layout (FP32 rows of ``pitch`` elements). ``blocks``: the
    indexer cache's (None: no KV cache yet, so the longest context bounds the TP
    logits). PreparedDsa reserves them when the specialized decode is built,
    before vLLM profiles its memory for the KV cache; FusedIndexer
    (prepare_cache, after the KV cache is allocated) then takes the same buffers
    and allocates nothing."""
    if dcp:
        pitch = -(-local_context(indexer) // 512) * 512  # 2 KiB rows
        capacity = max_rows * pitch
    else:
        # Request-compact rows: at most every row a whole context (requests
        # may share their prefix blocks, so the cache size does not bound
        # the sum).
        pitch, context = 0, indexer.max_model_len
        if blocks is not None:
            context = min(context, blocks * PAGE)
        capacity = max_rows * (-(-context // TILE) * TILE + TILE)
    logits = logits_scratch(device, capacity)
    tops = logits_scratch(device, capacity // 8, "tops")
    out, valid, ends = select_outputs(device, max_rows, indexer.topk_tokens)
    unit_scales(device, max_rows)
    return logits, tops, out, valid, ends, pitch


def select_unsupported(topk) -> str | None:
    """Why IndexerSelect cannot take a top-``topk`` (None: it can)."""
    if topk % SELECT_THREADS or topk > 4096:
        return f"top-k {topk}"
    return None


def static_unsupported(indexer, rows, q) -> str | None:
    """Why the fused indexer cannot run this layer's (rows, q), from what is known
    before the KV cache exists (None: it can, if its cache layout does)."""
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    if not mqa_logits_use_mx_scales():
        return "needs the SM100 MX-scaled indexer cache"
    reason = select_unsupported(indexer.topk_tokens)
    if reason is not None:
        return reason
    if not 1 <= q <= 8 or rows % q or rows // q > MAX_REQUESTS:
        return f"rows {rows} q {q}"
    return None


def supports(indexer, rows, q) -> str | None:
    """Why the fused indexer cannot run this layer (None: it can): FP8 [blocks, 64,
    132] or MXFP4 [blocks, 64, 68] pages (keys, then their UE8M0 x4 words)."""
    reason = static_unsupported(indexer, rows, q)
    cache = indexer.k_cache.kv_cache
    width = DIM // 2 + 4 if indexer.use_fp4_cache else DIM + 4
    if reason is None and (
        cache.dim() != 3 or cache.shape[1] != PAGE or cache.shape[2] != width
    ):
        reason = f"indexer cache {tuple(cache.shape)} (needs [blocks, 64, {width}])"
    return reason


@functools.cache
def compile_select(
    rows,
    q,
    topk,
    mode,
    device,
    tops,
    logical=False,
    release=True,
    page=PAGE,
    world=1,
    interleave=1,
):
    """IndexerSelect of one (rows, q) (see its ``mode``); ``tops``: the maxima
    scratch's elements ("rows")."""
    select = IndexerSelect(
        rows,
        q,
        topk,
        mode,
        logical=logical,
        release=release,
        page=page,
        world=world,
        interleave=interleave,
    )
    width = 2 * topk if mode == "candidates" else topk
    with torch.accelerator.device_index(device):
        tops = fake((tops,), (1,), cutlass.Int32)
        ends = fake((rows,), (1,), torch.int32, 4)
        out = fake((rows, width), (width, 1), torch.int32, 4)
        valid = fake((rows,), (1,), torch.int32, 4)
        s = make_fake_stream()
        args = Int64(0), Int32(0), tops, ends, Int64(0), Int32(0), Int32(0), Int32(0)
        return cute.compile(select, *args, out, valid, out, s)


@functools.cache
def compile_indexer(
    rows,
    q,
    topk,
    ctas,
    keys,
    scales,
    logits,
    tops,
    device,
    ids,
    pitch=0,
    release=True,
    page=PAGE,
    fp4=False,
    mode="rows",
):
    """The score and (TP) select kernels of one (rows, q) for an indexer cache
    layout (every DSA layer's is the same). DCP (``pitch``): the scorer only, into
    DeepGEMM-layout logits. ``page``: the MLA cache's tokens per block (TP
    select); ``mode``: the select's ("rows": physical rows, "compact_ids": the
    key indices); ``fp4``: MXFP4 keys and query."""
    score = IndexerScore(rows, q, ctas, 0 if pitch else topk, pitch, fp4)
    width = DIM // 2 if fp4 else DIM
    element = torch.uint8 if fp4 else cutlass.Float8E4M3FN
    with torch.accelerator.device_index(device):
        query = fake(
            (rows * HEADS, width, 1), (width, 1, rows * HEADS * width), element
        )
        qscales = fake((rows * HEADS,), (1,), torch.int32)
        weights = fake((rows * HEADS,), (1,), torch.float32)
        key_view = fake(*keys, element)
        scale_view = fake(*scales, torch.int32)
        table = Int64(0), Int32(0), Int32(0)
        ends = fake((rows,), (1,), torch.int32, 4)
        scratch = fake(logits, (1,), cutlass.Int32)
        maxima = fake(tops, (1,), cutlass.Int32)
        s = make_fake_stream()
        args = query, qscales, weights, key_view, scale_view, *table, ends
        score_kernel = cute.compile(score, *args, scratch, maxima, s)
    if pitch:
        return score_kernel, None
    args = rows, q, topk, mode, device, tops[0], ids, release, page
    return score_kernel, compile_select(*args)


class FusedIndexer:
    """The fused decode indexer of one DSA layer, compiled per (rows, q) for the
    layer's indexer cache (FP8 or MXFP4 keys). TP: ``run`` -> (physical rows [B,
    topk] of the MLA cache, valid counts [B]), or (``ids``) ``run_ids`` -> the
    top-k key indices in a buffer. DCP (``dcp``): ``run_dcp`` scores this rank's
    keys into DeepGEMM's FP32 logits layout (the DCP top-k, dcp_topk.py,
    selects)."""

    def __init__(
        self,
        indexer,
        shapes,
        device,
        max_rows,
        dcp=False,
        hold=(),
        page=PAGE,
        ids=False,
    ):
        from .index import fp4_query_views, is_fp4_cache

        cache = indexer.k_cache.kv_cache
        for shape in shapes:
            reason = supports(indexer, *shape)
            if reason is not None:
                raise ValueError(
                    f"Kimi K3.1 specialized DSA: the indexer scorer cannot run rows x "
                    f"q {shape[0]}x{shape[1]} ({reason})"
                )
        # The (rows, q) whose select must not launch the next kernel early (TP: it
        # is a decode that reads the rows before its PDL wait; IndexerSelect).
        self.hold = frozenset(hold)
        self.indexer, self.device, self.page = indexer, device, page
        self.topk = indexer.topk_tokens
        self.fp4 = is_fp4_cache(cache)
        self._fp4_query = fp4_query_views
        self.mode = "compact_ids" if ids else "rows"
        if ids and dcp:
            raise ValueError("FusedIndexer: ids is a TP mode")
        blocks, unit = cache.shape[0], cache.stride(0)
        width = DIM // 2 if self.fp4 else DIM  # key bytes
        keys = torch.as_strided(cache, (PAGE, width, blocks), (width, 1, unit))
        self.keys = keys if self.fp4 else keys.view(torch.float8_e4m3fn)
        words = cache.view(torch.int32)
        self.scales = torch.as_strided(
            words,
            (blocks, PAGE),
            (unit // 4, 1),
            storage_offset=words.storage_offset() + PAGE * width // 4,
        )
        buffers = reserve(indexer, device, max_rows, dcp, blocks)
        self.logits, self.tops, self.out, self.valid, self.ends, self.pitch = buffers
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.unit_scales = unit_scales(device, max_rows)
        self.ids: torch.Tensor | None = None
        self.kernels = {shape: self.compile(*shape) for shape in shapes}

    def compile(self, rows, q):
        logits = tuple(self.logits.shape)
        keys = tuple(self.keys.shape), tuple(self.keys.stride())
        scales = tuple(self.scales.shape), tuple(self.scales.stride())
        tops = tuple(self.tops.shape)
        args = rows, q, self.topk, self.ctas, keys, scales, logits, tops
        args += (self.device.index, self.ids is not None, self.pitch)
        release = (rows, q) not in self.hold
        return compile_indexer(
            *args, release=release, page=self.page, fp4=self.fp4, mode=self.mode
        )

    def write_ids(self, ids: torch.Tensor):
        """Also write the selected key indices (the logical top-k, -1 past the
        count) into ``ids`` [>= rows, topk] (tests; TP "rows" only)."""
        if self.pitch or self.mode != "rows":
            raise ValueError("FusedIndexer.write_ids: TP physical rows only")
        self.ids = ids
        self.kernels = {shape: self.compile(*shape) for shape in self.kernels}

    def _ids(self, out, rows):
        return out if self.ids is None else self.ids[:rows]

    @staticmethod
    def _table(t):
        """(address, row pitch, width) of an int32 block table view."""
        if t.dtype != torch.int32 or t.stride(1) != 1:
            raise ValueError(f"FusedIndexer: int32 block table rows, {t.dtype}")
        return Int64(t.data_ptr()), Int32(t.stride(0)), Int32(t.shape[1])

    def score(self, q_fp8, head_w, index_decode, rows, q):
        """The scorer over the step's rows -> the per-row ends (cute)."""
        score, _ = self.kernels[rows, q]
        if head_w.dtype != torch.float32:
            raise TypeError(f"FusedIndexer: FP32 head weights, got {head_w.dtype}")
        ends = index_decode.seq_lens.reshape(-1)
        index_table = index_decode.block_table
        if q > 1 and index_table.shape[0] == ends.shape[0]:
            index_table = index_table[::q]  # expanded per row
        if ends.shape[0] < rows:  # rows past the step's decodes: empty
            self.ends[ends.shape[0] : rows].zero_()
            self.ends[: ends.shape[0]].copy_(ends)
            ends = self.ends
        ends = ends[:rows]
        if self.fp4:  # DsaIndex's E2M1 rows and UE8M0 x4 words of the step
            packed, qscales = self._fp4_query(q_fp8, rows, rows)
            width = DIM // 2
            query = torch.as_strided(
                packed, (rows * HEADS, width, 1), (width, 1, rows * HEADS * width)
            )
            qscales = qscales.reshape(-1)
        else:
            query = torch.as_strided(
                q_fp8[:rows].view(torch.float8_e4m3fn),
                (rows * HEADS, DIM, 1),
                (DIM, 1, rows * HEADS * DIM),
            )
            qscales = self.unit_scales[: rows * HEADS]
        score(
            to_cute(query),
            to_cute(qscales),
            to_cute(head_w[:rows].reshape(-1)),
            to_cute(self.keys),
            to_cute(self.scales),
            *self._table(index_table),
            to_cute(ends, 4),
            to_cute(self.logits),
            to_cute(self.tops),
            stream(self.device),
        )
        return ends

    def run(self, q_fp8, head_w, index_decode, block_table, block_rows, rows, q):
        """TP. ``index_decode``: the indexer's decode metadata (per-row ends and
        index block table); ``block_table`` / ``block_rows``: the MLA cache's."""
        if self.pitch or self.mode != "rows":
            raise RuntimeError("FusedIndexer.run: a TP physical-rows indexer")
        _, select = self.kernels[rows, q]
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        out, valid = self.out[:rows], self.valid[:rows]
        table = self._table(block_table[: rows // q])[:2]
        select(
            Int64(self.logits.data_ptr()),
            Int32(0),
            to_cute(self.tops),
            to_cute(ends, 4),
            *table,
            Int32(block_rows),
            Int32(0),
            to_cute(out, 4),
            to_cute(valid, 4),
            to_cute(self._ids(out, rows), 4),
            stream(self.device),
        )
        return out, valid

    def run_ids(self, q_fp8, head_w, index_decode, rows, q, buffer):
        """TP: the exact top-k key indices of the step's ``rows`` rows into
        ``buffer`` [>= rows, topk] int32 (-1 past each row's count)."""
        if self.pitch or self.mode != "compact_ids":
            raise RuntimeError("FusedIndexer.run_ids: a TP ids indexer")
        if buffer.dtype != torch.int32 or buffer.stride(0) != self.topk:
            raise ValueError(f"FusedIndexer.run_ids: int32 [rows, {self.topk}] buffer")
        _, select = self.kernels[rows, q]
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        out = buffer[:rows]
        select(
            Int64(self.logits.data_ptr()),
            Int32(0),
            to_cute(self.tops),
            to_cute(ends, 4),
            Int64(0),
            Int32(0),
            Int32(0),
            Int32(0),
            to_cute(out, 4),
            to_cute(self.valid[:rows], 4),
            to_cute(out, 4),
            stream(self.device),
        )
        return buffer

    def run_dcp(self, q_fp8, head_w, index_decode, rows, q):
        """DCP -> (the FP32 logits [rows, pitch] of this rank's keys in DeepGEMM's
        layout, the row ends [rows])."""
        if not self.pitch:
            raise RuntimeError("FusedIndexer.run_dcp: a DCP indexer")
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        logits = self.logits[: rows * self.pitch].view(torch.float32)
        return logits.view(rows, self.pitch), ends
