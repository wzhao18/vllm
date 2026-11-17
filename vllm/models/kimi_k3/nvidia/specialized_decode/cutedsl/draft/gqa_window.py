# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's sliding-window GQA attention in one kernel per layer
(Inferact/Kimi-K3.1-Pro-DSpark: Qwen3DSparkForCausalLM, DFlashQwen3Attention).

Stock, per drafter layer (qwen3_dflash.DFlashQwen3Attention.forward on the qkv_proj
rows): q_norm / k_norm (per-head RMSNorm, eps from the config), neox RoPE from the
layer's BF16 ``cos_sin_cache``, V times the layer's value scale (when the checkpoint
sets one), the K/V insert of the query rows at their slots, then causal sliding-window
attention (FLASH_ATTN / FLASHINFER, window W = 2048: query row i of a request with
``seq_len`` L sits at key L - q + i and reads keys [L - q + i - W + 1, L - q + i]) over
the BF16 paged cache [blocks, kv heads, page, K | V] (128 values per token: K then V).

``GqaWindow`` runs all of it: grid (splits, kv heads, requests), one cluster of
``splits`` CTAs per (request, kv head) (no cluster for one split). The CTAs split the
request's window union [max(0, L - q - W + 1), L) into 32-key stages; inside a CTA
``kgroups`` key groups take contiguous runs of them, each through its own ring of
``ring`` TMA slots (box {64, 32, 1, 1} of K and of V, 128 B swizzle). A key group is
one warp per 16-row tile of the request's ``group * q`` query rows (query row i, head g:
tile row i * group + g). Per stage: S = Q K^T with mma.sync m16n8k16 BF16 (FP32
accumulate), online softmax (exp2 of the stock scale x log2 e; the O rescale runs only
when some row max moved: alpha is exactly 1 otherwise), P rounded to BF16, O += P V
(FP32). S of stage k + 1 is issued before stage k's softmax (two score buffers). The
last warp of a group done with a slot refills it, ``ahead`` stages past the ring kept
requested into L2. The key groups' partials merge in shared memory (one CTA barrier),
the cluster's by a DSMEM push to the CTA owning each tile row.

Latency path: before its PDL wait a CTA reads the metadata as it stands (lengths, block
table, query positions) only to choose addresses: it prefetches its first stages into
L2 and loads the constant inputs (norm weights, the positions' RoPE rows). Right after
the wait it issues its first TMA loads at those addresses and reads the metadata again;
a CTA whose addresses moved drains and reissues them (never in a decode step: the
metadata is written before the graph) and a moved position reloads its RoPE row. The Q
tile (norm + RoPE, 8 lanes per head) and the query rows' K / V are computed by every
thread of the CTA between the two.

The query rows' K / V: the CTA whose stages hold a query row's position computes its
k_norm + RoPE (and V), stores both at the row's slot (when the slot is >= 0, as stock)
and uses them in place of the cache rows it loaded; a row whose slot is -1 keeps the
cache row (stock reads it there). Positions outside [window start, L) are zeroed in the
tiles (their cache rows may hold any bytes); a request with ``seq_len`` 0 (a padded
one) writes zeros.

Rounding points (FP32 arithmetic inside, sums in another order). The norm: (x * rsqrt(
sum(x^2) / 64 + eps)) * w (rsqrt.approx). With ``round_norm`` it is rounded to BF16
before RoPE (the stock C kernels rms_norm + rotary_embedding: eager drafters and the
context step); without it RoPE takes the FP32 values (inductor fuses the stock native
norm and RoPE into one kernel and elides the BF16 cast between them: the compiled
drafter, the default). RoPE with the BF16 cache values: bf16(fma(x1, cos, -(x2 *
sin))), bf16(fma(x1, sin, x2 * cos)) (as nvcc and Triton contract it). V: as stored, or
bf16(v * value scale). Attention: S in FP32, P in BF16, O in FP32, the output rounded
once.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, fake
from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    float_bits,
    fma,
    launch,
    ldmatrix,
    load_words,
    mma,
    shared_address,
    store_words,
    wait,
)

HEAD = 64  # head_dim of the release drafter (q / k / v per head)
TB = 32  # keys per pipeline stage (one TMA box of K and one of V)
STAGE = TB * 128  # bytes of one stage's K (or V) tile: 32 rows x 128 B
PITCH = 68  # FP32 words per partial row (64 values + max, sum, 2 spare)
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")


@asm(Float32, pure=True)
def rsqrt_approx(value):
    """rsqrtf (MUFU.RSQ), as vLLM's rms_norm kernel."""
    return "rsqrt.approx.f32 $0,$1;", "=f,f", (Float32(value),)


@asm(Float32, pure=True)
def div_rn(a, b):
    return "div.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def ex2(value):
    """exp2f under fast math (ex2.approx.ftz), as the stock attention kernels."""
    return "ex2.approx.ftz.f32 $0,$1;", "=f,f", (Float32(value),)


@asm()
def tma_load4(dst, tensor_map, barrier, c0, c1, c2, c3):
    """One box of a 4-D map at (c0, c1, c2, c3) to shared address ``dst``, completing
    on ``barrier`` (shared address)."""
    code = "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes [$0], [$1, {$2, $3, $4, $5}], [$6];"
    operands = (Uint32(dst), Int64(tensor_map), Int32(c0), Int32(c1), Int32(c2))
    operands += (Int32(c3), Uint32(barrier))
    return code, "r,l,r,r,r,r,r,~{memory}", operands


@asm()
def prefetch_l2(address, size):
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r", (Int64(address), size)


@asm()
def prefetch_tensormap(address):
    return "prefetch.tensormap [$0];", "l", (Int64(address),)


@asm()
def arm(barrier, size):
    """mbarrier arrive + expect_tx ``size`` bytes (CTA scope)."""
    code = "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;"
    return code, "r,r,~{memory}", (Uint32(barrier), Int32(size))


@asm()
def arrive(barrier):
    code = "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];"
    return code, "r,~{memory}", (Uint32(barrier),)


@asm()
def init_barrier(barrier, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", (
        Uint32(barrier),
        Int32(count),
    )


@asm(Int32)
def load_i32(pointer):
    """An L2 (``.cg``) load: never an L1 line read before the PDL wait."""
    return "ld.global.cg.b32 $0, [$1];", "=r,l,~{memory}", (pointer,)


@asm(Int64)
def load_i64(pointer):
    return "ld.global.cg.b64 $0, [$1];", "=l,l,~{memory}", (pointer,)


@asm()
def push4(local, words, barrier, peer):
    """st.async of four words to ``peer``'s copy of the shared address ``local``,
    completing on ``peer``'s copy of ``barrier``."""
    code = "{ .reg .b32 dst, bar; mapa.shared::cluster.u32 dst, $0, $6; "
    code += "mapa.shared::cluster.u32 bar, $5, $6; "
    code += "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
    code += "[dst], {$1, $2, $3, $4}, [bar]; }"
    operands = (Uint32(local), *(Uint32(w) for w in words), Uint32(barrier))
    return code, "r,r,r,r,r,r,r,~{memory}", operands + (Int32(peer),)


@asm(Int32)
def release_count(counter):
    """One consumer warp done with a ring slot: the slot's count before it."""
    code = "atom.acq_rel.cta.shared::cta.add.u32 $0, [$1], 1;"
    return code, "=r,r,~{memory}", (Uint32(counter),)


@asm()
def store_shared_u32(address, value):
    return "st.shared.u32 [$0], $1;", "r,r,~{memory}", (Uint32(address), Int32(value))


@asm(Int64)
def global_timer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm()
def store_i64(address, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm()
def push2(local, words, barrier, peer):
    """``push4`` of two words."""
    code = "{ .reg .b32 dst, bar; mapa.shared::cluster.u32 dst, $0, $4; "
    code += "mapa.shared::cluster.u32 bar, $3, $4; "
    code += "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 "
    code += "[dst], {$1, $2}, [bar]; }"
    operands = (Uint32(local), *(Uint32(w) for w in words), Uint32(barrier))
    return code, "r,r,r,r,r,~{memory}", operands + (Int32(peer),)


@asm(Int32)
def atomic_or_shared(address, value):
    return "atom.shared.or.b32 $0, [$1], $2;", "=r,r,r,~{memory}", (
        Uint32(address),
        Int32(value),
    )


@asm()
def fence_async_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm(Int32)
def vote_any(flag):
    code = "{ .reg .pred p, q; setp.ne.b32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; "
    code += "selp.b32 $0, 1, 0, q; }"
    return code, "=r,r", (Int32(flag),)


def tree(op, values):
    """``op``-reduce ``values`` as a balanced tree (trace-time Python)."""
    values = list(values)
    while len(values) > 1:
        values = [op(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
                  for i in range(0, len(values), 2)]  # fmt: skip
    return values[0]


def float_word(value):
    return Uint32(float_bits(Float32(value)))


def swz(row, chunk):
    """Byte offset of 16 B chunk ``chunk`` of 128 B row ``row`` in a 128B-swizzled
    tile (1024 B aligned): the chunk index XOR the row's position in its 8-row atom."""
    return row * 128 + ((chunk ^ (row & 7)) << 4)


@cute.jit
def norm_rope8(x, weights, cos, sin, eps, upper, round_norm: cutlass.Constexpr):
    """RMSNorm + neox RoPE of one 64-value head held by an aligned group of 8 lanes,
    8 values each (lane j of the group: dims 8j..8j+7; ``upper``: j >= 4). ``x``: the
    8 FP32 values, ``weights``: the norm weight words (4 x BF16x2), ``cos`` / ``sin``
    of dims 8(j % 4).. (4 words each); ``round_norm``: the norm output rounded to BF16
    before RoPE (module notes). Returns 4 BF16x2 words."""
    total = Float32(0)
    for e in cutlass.range_constexpr(8):
        total = fma(x[e], x[e], total)
    for offset in (1, 2, 4):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    scale = rsqrt_approx(div_rn(total, Float32(HEAD)) + eps)
    normed = [x[e] * scale * bf16(weights[e // 2], e % 2 == 1) for e in range(8)]
    if cutlass.const_expr(round_norm):
        words = [bf16x2(normed[2 * k], normed[2 * k + 1]) for k in range(4)]
        normed = [bf16(words[e // 2], e % 2 == 1) for e in range(8)]
    partner = [cute.arch.shuffle_sync_bfly(v, 4) for v in normed]
    out = []
    for k in cutlass.range_constexpr(4):
        pair = []
        for half in (False, True):
            mine = normed[2 * k + int(half)]
            other = partner[2 * k + int(half)]
            c, s = bf16(cos[k], half), bf16(sin[k], half)
            # first half (x1 = mine, x2 = other): x1 cos - x2 sin
            first = fma(mine, c, -(other * s))
            # second half (x1 = other, x2 = mine): x2 cos + x1 sin
            second = fma(other, s, mine * c)
            value = first
            if upper:
                value = second
            pair.append(value)
        out.append(bf16x2(pair[0], pair[1]))
    return out


class GqaWindow:
    """One drafter layer's fused attention (module notes): ``group`` query heads per
    KV head, ``kv_heads`` per rank, ``q`` query rows per request, the BF16 cache's
    ``page`` tokens per kernel block, ``splits`` CTAs per (request, kv head) (a
    cluster), ``kgroups`` key groups per CTA, ``ring`` stages per key group,
    ``window`` keys. ``debug`` (development only, never set by the install): 1 / 2 /
    4 / 8 skip the math / the TMA loads / the prep / the combine, 16 returns after
    the PDL wait (latency ablations), 32 writes globaltimer phase stamps (an extra
    pointer argument), 64 takes the redo path (tests)."""

    def __init__(
        self, group, kv_heads, q, page, splits, kgroups, ring, window, v_scale=None,
        ahead=0, round_norm=False, debug=0,
    ):  # fmt: skip
        assert page % TB == 0 and window >= 1 and 1 <= q <= 8
        # ``ahead``: stages past the ring each key group keeps requested into L2
        self.debug, self.ahead = debug, ahead
        # the norm output rounded to BF16 before RoPE (the stock C kernels) or kept in
        # FP32 (inductor's fused norm + RoPE): module notes
        self.round_norm = round_norm
        self.group, self.kv_heads, self.q, self.page = group, kv_heads, q, page
        self.splits, self.kgroups, self.ring = splits, kgroups, ring
        self.window, self.v_scale = window, v_scale
        self.rows = group * q  # query rows of one (request, kv head)
        self.mt = -(-self.rows // 16)
        self.qrows = 16 * self.mt
        self.owned = -(-self.rows // splits)  # tile rows each cluster CTA finalizes
        self.consumers = self.mt * kgroups
        self.threads = 32 * self.consumers
        # table entries one CTA's stages can span (its run of the window union)
        self.pages = -(-(window + q - 1 + 2 * TB) // page) + 1
        assert self.pages <= self.threads
        assert ring >= 2  # (the pipelined loop waits on stage k + 1 before k's release)
        part = kgroups * self.qrows * PITCH * 4  # (the key groups' partials, after)
        assert part <= kgroups * ring * 2 * STAGE, (part, kgroups, ring)

    @cute.jit
    def __call__(self, *args):
        out = args[8]
        requests = out.shape[0] // self.q
        cluster = None
        if cutlass.const_expr(self.splits > 1):
            cluster = (self.splits, 1, 1)
        launch(
            self.kernel, args, (self.splits, self.kv_heads, requests), self.threads,
            cluster=cluster, max_number_threads=[self.threads, 1, 1],
        )  # fmt: skip

    # ------------------------------------------------------------------ helpers
    @cute.jit
    def stamp(self, args, k, tid):
        """Debug (``debug & 32``): globaltimer at phase ``k`` of this CTA (thread 0)."""
        if cutlass.const_expr(self.debug & 32):
            if tid == 0:
                split, head, request = cute.arch.block_idx()
                cta = (request * self.kv_heads + head) * self.splits + split
                store_i64(args[15] + (cta * 8 + k) * 8, global_timer())

    @cute.jit
    def stage_split(self, length, split):
        """(window start, the CTA's first stage, the CTA's stage count)."""
        lo = max(length - self.q - (self.window - 1), Int32(0))
        first = Int32(0)
        count = Int32(0)
        if length > 0:
            first = lo // TB
            count = (length - 1) // TB - first + 1
        begin = first + count * split // self.splits
        end = first + count * (split + 1) // self.splits
        return lo, begin, end - begin

    @cute.jit
    def group_split(self, begin, count, g):
        """(first stage, stage count) of key group ``g`` in the CTA's run."""
        first = begin + count * g // self.kgroups
        return first, begin + count * (g + 1) // self.kgroups - first

    @cute.jit
    def stage_coords(self, table, request, j, args):
        """(block, offset) of stage ``j``'s 32 tokens; the block table entry is read
        with ``load_i32`` and clamped into the cache."""
        position = j * TB
        page = min(position // self.page, Int32(table.shape[1] - 1))
        block = load_i32(table.iterator + request * table.layout.stride[0] + page)
        block = min(max(block, Int32(0)), args[13] - 1)
        return block, position % self.page

    @cute.jit
    def issue(self, kv_map, bar, k_tile, head, block, offset):
        if cutlass.const_expr(self.debug & 2):
            arm(bar, 0)
        else:
            arm(bar, 2 * STAGE)
            tma_load4(k_tile, kv_map, bar, Int32(0), offset, head, block)
            tma_load4(k_tile + STAGE, kv_map, bar, Int32(HEAD), offset, head, block)

    @cute.jit
    def rope_rows(self, cos_sin, position, group8):
        # (a padded row's position may be anything: clamped into the cache)
        row = min(max(position, Int32(0)), Int32(cos_sin.shape[0] - 1))
        base = cos_sin.iterator + row * cos_sin.layout.stride[0]
        cos = load_words(base + 8 * (group8 % 4), count=4)
        sin = load_words(base + 32 + 8 * (group8 % 4), count=4)
        return cos, sin

    @cute.jit
    def scores(self, tile, qa, lane, sc):
        """S = Q K^T of this warp's 16 rows x the stage's 32 keys, into ``sc``."""
        for n in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                sc[n, e] = Float32(0)
        for ks in cutlass.range_constexpr(4):
            for qd in cutlass.range_constexpr(2):
                key_row = 16 * qd + (lane & 7) + 8 * (lane >> 4)
                chunk = 2 * ks + ((lane >> 3) & 1)
                kb = ldmatrix(tile + swz(key_row, chunk))
                for nn in cutlass.range_constexpr(2):
                    n = 2 * qd + nn
                    res = mma(qa[ks], (kb[2 * nn], kb[2 * nn + 1]),
                              tuple(sc[n, e] for e in range(4)), bf16=True)  # fmt: skip
                    for e in cutlass.range_constexpr(4):
                        sc[n, e] = res[e]

    @cute.jit
    def mask(self, sc, p0, g, t4, rt, length, lo):
        """-inf for keys outside each row's window (the edge stages only)."""
        G: cutlass.Constexpr = self.group
        q_first = length - self.q
        for half in cutlass.range_constexpr(2):
            r = 16 * rt + 8 * half + g
            hi_r = length - 1
            lo_r = lo
            if r < self.rows:
                hi_r = q_first + r // G
                lo_r = max(hi_r - (self.window - 1), Int32(0))
            for n in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(2):
                    key = p0 + 8 * n + 2 * t4 + e
                    if (key < lo_r) | (key > hi_r):
                        sc[n, 2 * half + e] = Float32(NEG_INF)

    @cute.jit
    def softmax(self, sc, m_run, l_run, acc, c):
        """Online softmax of the stage's scores into probabilities (in ``sc``)."""
        m_new = [Float32(0), Float32(0)]
        m_use = [Float32(0), Float32(0)]
        moved = Int32(0)
        for half in cutlass.range_constexpr(2):
            mx = tree(cute.arch.fmax,
                      [sc[n, 2 * half + e] for n in range(4) for e in range(2)])
            mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, 1))
            mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, 2))
            top = cute.arch.fmax(m_run[half], mx)
            if top > m_run[half]:
                moved = Int32(1)
            use = top
            if top == Float32(NEG_INF):
                use = Float32(0)
            m_new[half] = top
            m_use[half] = use
        # alpha = exp2((m_old - m_new) c) is exactly 1 for a row whose max did not
        # move: the rescale runs only when some row of the warp moved.
        if vote_any(moved) != 0:
            for half in cutlass.range_constexpr(2):
                alpha = ex2((m_run[half] - m_use[half]) * c)
                l_run[half] = l_run[half] * alpha
                for n in cutlass.range_constexpr(8):
                    for e in cutlass.range_constexpr(2):
                        acc[n, 2 * half + e] = acc[n, 2 * half + e] * alpha
        for half in cutlass.range_constexpr(2):
            m_run[half] = m_new[half]
            bias = -(m_use[half] * c)
            probs = []
            for n in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(2):
                    p = ex2(fma(sc[n, 2 * half + e], c, bias))
                    sc[n, 2 * half + e] = p
                    probs.append(p)
            l_run[half] = l_run[half] + tree(lambda a, b: a + b, probs)

    @cute.jit
    def accumulate(self, tile, sc, acc, lane):
        """O += P V over the stage's 32 keys."""
        for kk in cutlass.range_constexpr(2):
            pa = (
                bf16x2(sc[2 * kk, 0], sc[2 * kk, 1]),
                bf16x2(sc[2 * kk, 2], sc[2 * kk, 3]),
                bf16x2(sc[2 * kk + 1, 0], sc[2 * kk + 1, 1]),
                bf16x2(sc[2 * kk + 1, 2], sc[2 * kk + 1, 3]),
            )
            for dp in cutlass.range_constexpr(4):
                key_row = 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)
                chunk = 2 * dp + (lane >> 4)
                vb = ldmatrix(tile + STAGE + swz(key_row, chunk), trans=True)
                for nn in cutlass.range_constexpr(2):
                    n = 2 * dp + nn
                    res = mma(pa, (vb[2 * nn], vb[2 * nn + 1]),
                              tuple(acc[n, e] for e in range(4)), bf16=True)  # fmt: skip
                    for e in cutlass.range_constexpr(4):
                        acc[n, e] = res[e]

    @cute.jit
    def prepare(self, k, sc, fixbits, frame, qa):
        """Stage ``k`` of the warp's key group: its TMA landed, its edge rows fixed
        (by the group's first row-tile warp, the others wait on the slot's ``fixed``
        barrier: no named barrier, so CTAs still co-reside), S into ``sc``. Returns
        the slots' fix-phase bits."""
        first_g, ring_g, full_g, fixed_g, rt, lane, lo, length, kv_new, newvalid = frame[
            0:10
        ]
        offset_bits = frame[10]
        slot_i = k % self.ring
        tile = ring_g + slot_i * (2 * STAGE)
        p0 = (first_g + k) * TB
        phase = ((k // self.ring) + ((offset_bits >> slot_i) & 1)) & 1
        wait(full_g + slot_i * 8, phase, scope="cta")
        bits = fixbits
        if cutlass.const_expr(not (self.debug & 1)):
            if (p0 < lo) | (p0 + TB > length - self.q):
                if rt == 0:
                    self.fixup(tile, p0, lane, lo, length, kv_new, newvalid)
                    fence_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        arrive(fixed_g + slot_i * 8)
                else:
                    wait(fixed_g + slot_i * 8, (fixbits >> slot_i) & 1, scope="cta")
                bits = fixbits ^ (Int32(1) << slot_i)
            self.scores(tile, qa, lane, sc)
        return bits

    @cute.jit
    def finish(self, k, sc, frame, state, args, c):
        """Stage ``k``: mask (edge stages), softmax, O += P V, then the slot's
        release (the last of the group's warps refills it with stage k + ring)."""
        first_g, ring_g, full_g, count_g, count_slots, rt, g, t4, lane = frame[0:9]
        lo, length, page_lo, blocks, kv_map, head = frame[9:15]
        m_run, l_run, acc = state
        R: cutlass.Constexpr = self.ring
        slot_i = k % R
        tile = ring_g + slot_i * (2 * STAGE)
        p0 = (first_g + k) * TB
        if cutlass.const_expr(not (self.debug & 1)):
            lo_max = max(length - self.window, Int32(0))
            if (p0 < lo_max) | (p0 + TB > length - self.q):
                self.mask(sc, p0, g, t4, rt, length, lo)
            self.softmax(sc, m_run, l_run, acc, c)
            self.accumulate(tile, sc, acc, lane)
        if lane == 0:
            if release_count(count_slots + 4 * slot_i) == self.mt - 1:
                store_shared_u32(count_slots + 4 * slot_i, 0)
                if k + R < count_g:
                    j_next = first_g + k + R
                    r_block = blocks[j_next * TB // self.page - page_lo]
                    fence_async_shared()
                    self.issue(kv_map, full_g + slot_i * 8, tile, head, r_block,
                               j_next * TB % self.page)  # fmt: skip
                if k + R + self.ahead < count_g:
                    j_far = first_g + k + R + self.ahead
                    f_block = blocks[j_far * TB // self.page - page_lo]
                    f_elem = Int64(f_block) * args[11] + Int64(head) * args[12]
                    f_elem = f_elem + Int64(j_far * TB % self.page) * 128
                    prefetch_l2(args[10] + f_elem * 2, Int32(2 * STAGE))

    @cute.jit
    def fixup(self, tile, p0, lane, lo, length, kv_new, newvalid):
        """Lane r: tile row r (position p0 + r): zero outside [lo, length); the
        computed K / V of a query row whose slot was valid."""
        position = p0 + lane
        q_first = length - self.q
        zero = (Uint32(0),) * 4
        if (position < lo) | (position >= length):
            for ch in cutlass.range_constexpr(8):
                store_words(tile + swz(lane, ch), zero, shared=True)
                store_words(tile + STAGE + swz(lane, ch), zero, shared=True)
        elif position >= q_first:
            i = position - q_first
            if newvalid[i] != 0:
                for ch in cutlass.range_constexpr(8):
                    kq = load_words(kv_new + i * 256 + 16 * ch, count=4, shared=True)
                    vq = load_words(kv_new + i * 256 + 128 + 16 * ch, count=4,
                                    shared=True)  # fmt: skip
                    store_words(tile + swz(lane, ch), kq, shared=True)
                    store_words(tile + STAGE + swz(lane, ch), vq, shared=True)

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(self, args):
        qkv, positions, slots, seq_lens, table, cos_sin, qw, kw, out = args[0:9]
        kv_map, kv_base, row_stride, head_stride = args[9:13]
        eps = args[14]
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        MT: cutlass.Constexpr = self.mt
        KG: cutlass.Constexpr = self.kgroups
        R: cutlass.Constexpr = self.ring
        S: cutlass.Constexpr = self.splits
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        ROWS: cutlass.Constexpr = self.rows
        QROWS: cutlass.Constexpr = self.qrows
        OWN: cutlass.Constexpr = self.owned
        PARTS: cutlass.Constexpr = self.splits  # cluster partials per owned row
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        HKV: cutlass.Constexpr = self.kv_heads
        threads: cutlass.Constexpr = self.threads
        c = Float32(LOG2E / (HEAD**0.5))  # softmax scale x log2(e)

        split, head, request = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        g, t4 = lane >> 2, lane & 3
        group8 = lane & 7
        upper = group8 >= 4
        rt = warp % MT  # the warp's 16-row tile
        kg = warp // MT  # the warp's key group
        table_row = table.iterator + request * table.layout.stride[0]
        table_last = Int32(table.shape[1] - 1)
        blocks_max = args[13] - 1

        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, cutlass.Uint8, KG * R * 2 * STAGE, align=1024)
        qtile = alloc(smem, cutlass.Uint8, QROWS * 128, align=1024)
        newkv = alloc(smem, cutlass.Uint8, Q * 256, align=16)
        newvalid = alloc(smem, Int32, Q, align=16)
        bars = alloc(smem, cutlass.Int64, 2 * KG * R + 1, align=8)
        counters = alloc(smem, Int32, KG * R + 1, align=16)  # [KG * R]: the redo flag
        blocks = alloc(smem, Int32, NP, align=16)
        recv = alloc(smem, Float32, max(PARTS * OWN * PITCH, 4), align=16)
        ring_base = Int32(shared_address(ring.iterator))
        q_base = Int32(shared_address(qtile.iterator))
        kv_new = Int32(shared_address(newkv.iterator))
        full_base = Int32(shared_address(bars.iterator))
        count_base = Int32(shared_address(counters.iterator))
        redo_flag = count_base + 4 * (KG * R)
        fixed_base = full_base + 8 * (KG * R)
        recv_bar = full_base + 16 * (KG * R)
        recv_base = Int32(shared_address(recv.iterator))
        ring_g = ring_base + kg * (R * 2 * STAGE)
        full_g = full_base + kg * (R * 8)
        count_g_base = count_base + kg * (R * 4)

        # tile rows this CTA finalizes; the partial rows it receives (one per CTA of
        # the cluster: 64 values, max, sum, 2 spare: PITCH words)
        my_rows = min(max(ROWS - split * OWN, 0), OWN)
        self.stamp(args, 0, tid)
        if tid == 0:
            for i in cutlass.range_constexpr(KG * R):
                init_barrier(full_base + 8 * i, 1)
                init_barrier(fixed_base + 8 * i, 1)
                store_shared_u32(count_base + 4 * i, 0)
            store_shared_u32(redo_flag, 0)
            if cutlass.const_expr(S > 1):
                init_barrier(recv_bar, 1)
                arm(recv_bar, S * my_rows * (PITCH * 4))
        if tid < (QROWS - ROWS) * 8:  # the tile's padding rows
            store_words(q_base + swz(ROWS + tid // 8, group8), (Uint32(0),) * 4,
                        shared=True)  # fmt: skip
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if cutlass.const_expr(S > 1):
            cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM push

        # ---- before the PDL wait: the metadata as it stands. It decides only
        # addresses: the L2 prefetch, and the first TMA loads after the wait, which
        # the post-wait metadata confirms (else they are drained and reissued).
        length0 = load_i32(seq_lens.iterator + request)
        if tid == 0:
            prefetch_tensormap(kv_map)
        _, begin0, count0 = self.stage_split(length0, split)
        page_lo0 = begin0 * TB // N
        first_g0, count_g0 = self.group_split(begin0, count0, kg)
        for j0 in cutlass.range(rt * 32 + lane, min(count_g0, Int32(R + self.ahead)),
                                MT * 32, unroll=1):  # fmt: skip
            p_page = min((first_g0 + j0) * TB // N, table_last)
            p_block = min(max(load_i32(table_row + p_page), Int32(0)), blocks_max)
            p_elem = Int64(p_block) * row_stride + Int64(head) * head_stride
            p_elem = p_elem + Int64((first_g0 + j0) * TB % N) * 128
            prefetch_l2(kv_base + p_elem * 2, Int32(2 * STAGE))
        issue_slot = lane * MT + rt  # (slot s of the key group: warp s % MT)
        spec_n = min(count_g0, Int32(R))  # slots issued from the pre-wait metadata
        j_spec = first_g0 + issue_slot
        page_spec = min(j_spec * TB // N, table_last)
        e_spec0 = Int32(0)
        if issue_slot < spec_n:
            e_spec0 = load_i32(table_row + page_spec)
        # Prep tasks, 8 lanes per 64-value head: t < ROWS*8 a Q tile row, then the
        # request's query rows' K / V (every task runs the norm + RoPE so the 8-lane
        # shuffles converge; only the stores differ).
        TASKS: cutlass.Constexpr = ROWS * 8 + Q * 8
        tasks = []
        for it in cutlass.range_constexpr(-(-TASKS // threads)):
            t = it * threads + tid
            is_kv = t >= ROWS * 8
            i = min(t // 8, ROWS - 1) // G
            if is_kv:
                i = min((t - ROWS * 8) // 8, Q - 1)
            row = request * Q + i
            pos0 = load_i64(positions.iterator + row)
            cos, sin = self.rope_rows(cos_sin, Int32(pos0), group8)
            tasks.append((t, is_kv, i, row, pos0, cos, sin))
        q_w = load_words(qw.iterator + 8 * group8, count=4)
        k_w = load_words(kw.iterator + 8 * group8, count=4)
        cute.arch.griddepcontrol_wait()
        self.stamp(args, 1, tid)
        if cutlass.const_expr(self.debug & 16):
            cute.arch.griddepcontrol_launch_dependents()
            return

        # ---- after the wait: the first stages' TMA loads at the pre-wait addresses
        # at once, then one round of loads (lengths, table entries, qkv rows,
        # positions, slots) that confirms them and feeds the prep math
        if issue_slot < spec_n:
            self.issue(
                kv_map, full_g + 8 * issue_slot, ring_g + issue_slot * (2 * STAGE),
                head, min(max(e_spec0, Int32(0)), blocks_max), j_spec * TB % N,
            )  # fmt: skip
        length = load_i32(seq_lens.iterator + request)
        e_spec = Int32(0)
        if issue_slot < spec_n:
            e_spec = load_i32(table_row + page_spec)
        entry_spec = Int32(0)
        if tid < NP:
            entry_spec = load_i32(table_row + min(page_lo0 + tid, table_last))
        stride0 = qkv.layout.stride[0]
        loaded = []
        for t, is_kv, i, row, pos0, cos0, sin0 in tasks:
            gq = min(t // 8, ROWS - 1) % G
            col = (head * G + gq) * HEAD + 8 * group8
            if is_kv:
                col = HQ * HEAD + head * HEAD + 8 * group8
            words = load_words(qkv.iterator + row * stride0 + col, count=4,
                               mutable=True)  # fmt: skip
            vwords = load_words(qkv.iterator + row * stride0 + col + HKV * HEAD,
                                count=4, mutable=True)  # fmt: skip
            pos = load_i64(positions.iterator + row)
            slot = load_i64(slots.iterator + row)
            loaded.append((words, vwords, pos, slot))
        lo, begin, count = self.stage_split(length, split)
        q_first = length - Q
        page_lo = begin * TB // N
        first_g, count_g = self.group_split(begin, count, kg)
        # (a moved length moves every address: the CTA redoes its first loads)
        moved = (length != length0) | (e_spec != e_spec0)
        if cutlass.const_expr(self.debug & 64):
            moved = tid == 0  # (tests: the redo path with unmoved metadata)
        if moved:
            atomic_or_shared(redo_flag, 1)
        if tid < NP:
            entry = entry_spec
            if page_lo != page_lo0:
                entry = load_i32(table_row + min(page_lo + tid, table_last))
            blocks[tid] = min(max(entry, Int32(0)), blocks_max)
        self.stamp(args, 2, tid)

        # ---- the Q tile and the held query rows' K / V
        held_lo = max(begin * TB, q_first)
        held_hi = min((begin + count) * TB, length)
        for n_task in cutlass.range_constexpr(len(tasks) if not (self.debug & 4) else 0):
            t, is_kv, i, row, pos0, cos0, sin0 = tasks[n_task]
            words, vwords, pos, slot = loaded[n_task]
            # (word arrays live in register tensors: a dynamic branch may write them)
            wts = cute.make_rmem_tensor(4, Uint32)
            rope = cute.make_rmem_tensor(8, Uint32)
            for k in cutlass.range_constexpr(4):
                wts[k] = q_w[k]
                rope[k] = cos0[k]
                rope[4 + k] = sin0[k]
            if is_kv:
                for k in cutlass.range_constexpr(4):
                    wts[k] = k_w[k]
            if pos != pos0:  # (the position moved since the prefetch)
                cos1, sin1 = self.rope_rows(cos_sin, Int32(pos), group8)
                for k in cutlass.range_constexpr(4):
                    rope[k] = cos1[k]
                    rope[4 + k] = sin1[k]
            weights = tuple(wts[k] for k in range(4))
            cos = tuple(rope[k] for k in range(4))
            sin = tuple(rope[4 + k] for k in range(4))
            x = [bf16(words[e // 2], e % 2 == 1) for e in range(8)]
            roped = norm_rope8(x, weights, cos, sin, eps, upper, self.round_norm)
            if t < ROWS * 8:
                store_words(q_base + swz(t // 8, group8), tuple(roped), shared=True)
            elif t < TASKS:
                position_i = q_first + i
                if (position_i >= held_lo) & (position_i < held_hi):
                    value = list(vwords)
                    if cutlass.const_expr(self.v_scale is not None):
                        scaled = []
                        for k in cutlass.range_constexpr(4):
                            lo_v = bf16(vwords[k], False) * Float32(self.v_scale)
                            hi_v = bf16(vwords[k], True) * Float32(self.v_scale)
                            scaled.append(bf16x2(lo_v, hi_v))
                        value = scaled
                    store_words(kv_new + i * 256 + 16 * group8, tuple(roped),
                                shared=True)  # fmt: skip
                    store_words(kv_new + i * 256 + 128 + 16 * group8, tuple(value),
                                shared=True)  # fmt: skip
                    if group8 == 0:
                        newvalid[i] = Int32(slot >= 0)
                    if slot >= 0:
                        s_block = Int64(slot // N)
                        s_offset = Int64(slot % N)
                        s_elem = s_block * row_stride + Int64(head) * head_stride
                        s_elem = s_elem + s_offset * 128 + 8 * group8
                        s_addr = kv_base + s_elem * 2
                        store_words(s_addr, tuple(roped))
                        store_words(s_addr + 128, tuple(value))
        cute.arch.sync_threads()
        self.stamp(args, 3, tid)

        # ---- the speculated first loads were wrong (the metadata moved between
        # the two reads; never in a decode step): drain them, reissue from the
        # confirmed metadata. Those slots' phases then start at 1 (``offset``).
        offset_bits = Int32(0)
        if load_words(redo_flag, count=1, shared=True)[0] != 0:
            if issue_slot < spec_n:
                wait(full_g + 8 * issue_slot, 0, scope="cta")
            cute.arch.sync_threads()
            if issue_slot < min(count_g, Int32(R)):
                j_re = first_g + issue_slot
                re_block = load_i32(table_row + min(j_re * TB // N, table_last))
                self.issue(
                    kv_map, full_g + 8 * issue_slot,
                    ring_g + issue_slot * (2 * STAGE), head,
                    min(max(re_block, Int32(0)), blocks_max), j_re * TB % N,
                )  # fmt: skip
            offset_bits = (Int32(1) << spec_n) - 1

        # ---- the key group's stages
        m_run = cute.make_rmem_tensor(2, Float32)
        l_run = cute.make_rmem_tensor(2, Float32)
        acc = cute.make_rmem_tensor((8, 4), Float32)
        for half in cutlass.range_constexpr(2):
            m_run[half] = Float32(NEG_INF)
            l_run[half] = Float32(0)
        for n in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                acc[n, e] = Float32(0)
        qa = []
        for ks in cutlass.range_constexpr(4):
            q_r = 16 * rt + (lane & 7) + 8 * ((lane >> 3) & 1)
            qa.append(ldmatrix(q_base + swz(q_r, 2 * ks + (lane >> 4))))
        # Software-pipelined: S of stage k + 1 is issued before stage k's softmax
        # (two score buffers, so the loop runs in pairs).
        fixed_g = fixed_base + kg * (R * 8)
        pre = (first_g, ring_g, full_g, fixed_g, rt, lane, lo, length, kv_new, newvalid)
        pre += (offset_bits,)
        fin = (first_g, ring_g, full_g, count_g, count_g_base, rt, g, t4, lane, lo)
        fin += (length, page_lo, blocks, kv_map, head)
        state = (m_run, l_run, acc)
        sc_a = cute.make_rmem_tensor((4, 4), Float32)
        sc_b = cute.make_rmem_tensor((4, 4), Float32)
        fixbits = Int32(0)
        if count_g > 0:
            fixbits = self.prepare(Int32(0), sc_a, fixbits, pre, qa)
        for k2 in cutlass.range(0, count_g, 2, unroll=1):
            if k2 + 1 < count_g:
                fixbits = self.prepare(k2 + 1, sc_b, fixbits, pre, qa)
            self.finish(k2, sc_a, fin, state, args, c)
            if k2 + 1 < count_g:
                if k2 + 2 < count_g:
                    fixbits = self.prepare(k2 + 2, sc_a, fixbits, pre, qa)
                self.finish(k2 + 1, sc_b, fin, state, args, c)
        for half in cutlass.range_constexpr(2):
            total = l_run[half]
            total = total + cute.arch.shuffle_sync_bfly(total, 1)
            total = total + cute.arch.shuffle_sync_bfly(total, 2)
            l_run[half] = total
        cute.arch.griddepcontrol_launch_dependents()
        self.stamp(args, 4, tid)
        if cutlass.const_expr(self.debug & 8):
            return

        # ---- key groups -> the CTA's rows: every warp's partial into the freed ring
        # (one CTA barrier), then per (row, 4 columns) the key groups' merge, each
        # task folding its row's maxima itself (no second barrier)
        cute.arch.sync_threads()
        part_g = ring_base + kg * (QROWS * PITCH * 4)
        for half in cutlass.range_constexpr(2):
            r = 16 * rt + 8 * half + g
            row_base = part_g + r * (PITCH * 4)
            if t4 == 0:
                stat2 = (float_word(m_run[half]), float_word(l_run[half]))
                store_words(row_base + 64 * 4, stat2, shared=True)
            for n in cutlass.range_constexpr(8):
                pair_words = (float_word(acc[n, 2 * half]),
                              float_word(acc[n, 2 * half + 1]))  # fmt: skip
                store_words(row_base + (8 * n + 2 * t4) * 4, pair_words,
                            shared=True)  # fmt: skip
        if cutlass.const_expr(S > 1):
            cute.arch.cluster_wait()  # every peer's receive barrier is armed
        cute.arch.sync_threads()
        self.stamp(args, 5, tid)
        for it in cutlass.range_constexpr(-(-(ROWS * 16) // threads)):
            task = it * threads + tid
            if task < ROWS * 16:
                r, d4 = task // 16, task % 16
                stats = [
                    load_words(ring_base + (w * QROWS + r) * (PITCH * 4) + 256, count=2,
                               shared=True)  # fmt: skip
                    for w in range(KG)
                ]
                top = tree(cute.arch.fmax, [bits_float(st[0]) for st in stats])
                use = top
                if top == Float32(NEG_INF):
                    use = Float32(0)
                total = Float32(0)
                values = [Float32(0)] * 4
                for w in cutlass.range_constexpr(KG):
                    weight = ex2((bits_float(stats[w][0]) - use) * c)
                    total = fma(bits_float(stats[w][1]), weight, total)
                    got4 = load_words(
                        ring_base + (w * QROWS + r) * (PITCH * 4) + 16 * d4, count=4,
                        shared=True,
                    )  # fmt: skip
                    for e in cutlass.range_constexpr(4):
                        values[e] = fma(bits_float(got4[e]), weight, values[e])
                if cutlass.const_expr(S == 1):
                    inv = Float32(0)
                    if total > Float32(0):
                        inv = Float32(1) / total
                    out_words = (
                        bf16x2(values[0] * inv, values[1] * inv),
                        bf16x2(values[2] * inv, values[3] * inv),
                    )
                    o_row = request * Q + r // G
                    o_col = (head * G + r % G) * HEAD + 4 * d4
                    store_words(out.iterator + o_row * out.layout.stride[0] + o_col,
                                out_words)  # fmt: skip
                else:
                    owner = r // OWN
                    slot_r = (split * OWN + r % OWN) * PITCH
                    push4(recv_base + (slot_r + 4 * d4) * 4,
                          tuple(float_word(v) for v in values), recv_bar, owner)
                    if d4 == 0:
                        stat4 = (float_word(top), float_word(total), Uint32(0),
                                 Uint32(0))  # fmt: skip
                        push4(recv_base + (slot_r + 64) * 4, stat4, recv_bar, owner)
        if cutlass.const_expr(S > 1):
            # ---- the owned rows: every CTA's partial landed; each (row, 4 columns)
            # task folds the row's maxima itself
            wait(recv_bar, 0, scope="cta")
            self.stamp(args, 6, tid)
            for it in cutlass.range_constexpr(-(-(OWN * 16) // threads)):
                task = it * threads + tid
                rr, d4 = task // 16, task % 16
                if (task < OWN * 16) & (rr < my_rows):
                    tops = [recv[(sp * OWN + rr) * PITCH + 64] for sp in range(S)]
                    top = tree(cute.arch.fmax, tops)
                    use = top
                    if top == Float32(NEG_INF):
                        use = Float32(0)
                    total = Float32(0)
                    values = [Float32(0)] * 4
                    for sp in cutlass.range_constexpr(S):
                        base_s = (sp * OWN + rr) * PITCH
                        weight = ex2((tops[sp] - use) * c)
                        total = fma(recv[base_s + 65], weight, total)
                        got4 = load_words(recv_base + (base_s + 4 * d4) * 4, count=4,
                                          shared=True)  # fmt: skip
                        for e in cutlass.range_constexpr(4):
                            values[e] = fma(bits_float(got4[e]), weight, values[e])
                    inv = Float32(0)
                    if total > Float32(0):
                        inv = Float32(1) / total
                    own_words = (
                        bf16x2(values[0] * inv, values[1] * inv),
                        bf16x2(values[2] * inv, values[3] * inv),
                    )
                    r_own = split * OWN + rr
                    o_row = request * Q + r_own // G
                    o_col = (head * G + r_own % G) * HEAD + 4 * d4
                    store_words(out.iterator + o_row * out.layout.stride[0] + o_col,
                                own_words)  # fmt: skip
        self.stamp(args, 7, tid)


@cache
def compile_window(
    group, kv_heads, q, page, splits, kgroups, ring, window, v_scale, max_position,
    device, ahead=0, round_norm=False, debug=0,
):  # fmt: skip
    """The fused attention for any request count (row-dynamic views)."""
    kernel = GqaWindow(
        group, kv_heads, q, page, splits, kgroups, ring, window, v_scale, ahead,
        round_norm, debug,
    )
    i64, i32 = torch.int64, torch.int32
    count = cute.sym_int()
    args = (
        dynamic_fake(2),  # qkv [rows, width]
        fake((count,), None, i64, 8),  # positions
        fake((count,), None, i64, 8),  # slots
        fake((cute.sym_int(),), None, i32, 4),  # seq_lens
        # block table: any row stride (max blocks per request), no divisibility
        fake((cute.sym_int(), cute.sym_int()), (cute.sym_int(), 1), i32, 4),
        fake((max_position, 2 * 32)),  # cos_sin (BF16)
        fake((HEAD,)),  # q_norm weight
        fake((HEAD,)),  # k_norm weight
        dynamic_fake(2),  # out [rows, heads x 64]
        Int64(0),  # the layer cache's tensor map address
        Int64(0),  # the layer cache's base address
        Int64(0),  # row (block) stride, elements
        Int64(0),  # head stride, elements
        Int32(1),  # blocks
        Float32(1e-5),  # eps
    )
    if debug & 32:
        args += (Int64(0),)  # phase stamps [ctas, 8] int64
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
