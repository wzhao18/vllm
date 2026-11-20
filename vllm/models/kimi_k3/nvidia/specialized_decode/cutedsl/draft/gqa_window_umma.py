# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's sliding-window GQA attention on tcgen05
(``GqaWindowUmma``): ``GqaWindow``'s computation (module notes of ``gqa_window``:
q / k norm, RoPE, the query rows' K / V insert, the causal 2048-key window over the
BF16 paged cache), ``splits`` CTAs (a cluster) per (request, KV head), for batches
where ``GqaWindow``'s mma.sync loop is issue-latency bound (B96 TP4 q3: 384 CTAs x 4
warps, 0.6 waves, warps stalled on dependent mma.sync / softmax chains, DRAM at 37%).

CTA = four softmax warps + one producer warp; 64-key stages through a ring of ``ring``
slots, each one 16 KB TMA box of a 5-D view of the cache (``kv_map_5d``: K image then
V image, 128 B swizzle):
- Q tile: the ``group * q`` query rows (zero padded to 32) replicated into the four
  32-row TMEM lane quarters, so S = Q K^T (one M128 N64 K64 BF16 MMA block into TMEM)
  gives every warp all rows; warp w owns the stage's keys 16 w .. 16 w + 15 of its
  quarter's rows: a row-per-thread online softmax with no shuffles;
- P (BF16) goes to a TMEM P buffer (the warp's 8 packed columns; the other warps'
  columns of its lanes stay zero, written once), and O += P V runs with P from TMEM
  (A) and V MN-major (B), so quarter w of O accumulates warp w's keys only; the four
  quarters merge (LSE) at the end, then the cluster's CTAs (DSMEM pushes to the CTA
  owning each row, as ``GqaWindow``'s);
- S and P in two TMEM buffers each: the producer issues S of stage t + 1 before it
  waits for stage t's P; S of stage t + 2 reuses stage t's columns once the warps
  loaded them, P of stage t + 2 once stage t's PV completed;
- before the PDL wait: the metadata as it stands (length, block table, positions,
  RoPE rows) decides the first stages' addresses; after the wait the first loads go
  out at once and the metadata is read again: loads from moved metadata are drained
  and reissued (never in a decode step).

Rounding: S in FP32, P = exp2(s * scale log2 e - m) rounded to BF16, O in FP32, the
row sums of the FP32 p, the output rounded once (as ``GqaWindow``). The running max m
of a row moves only when a stage's max exceeds it by more than 2^THRESH (or from
-inf): P then stays <= 2^THRESH, exact in BF16 relative terms; only the order of the
FP32 rescales differs from a strict running max (stock FlashAttention's max is per
its own key blocks as well).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, fake
from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    float_bits,
    commit_asm,
    fence_tmem,
    fma,
    free_tmem,
    launch,
    load_words,
    mma_bf16_block,
    named_barrier,
    shared_address,
    store_words,
    wait,
)
from ..common.resident import land, wait_landed
from .gqa_window import (
    HEAD,
    LOG2E,
    NEG_INF,
    arm,
    arrive,
    ex2,
    fence_async_shared,
    atomic_or_shared,
    float_word,
    global_timer,
    push4,
    init_barrier,
    load_i32,
    load_i64,
    norm_rope8,
    store_shared_u32,
    swz,
    tree,
    vote_any,
)

KB = 64  # keys per stage
KTILE = KB * 128  # bytes of one stage's K (or V) image
SLOT = 2 * KTILE  # K then V
THREADS = 160  # warps 0-3 softmax (TMEM lane quarters), warp 4 producer / MMA
PRODUCER = 4
SWARP = 5  # (swarp) the S issuer: S = Q K^T as stages land, apart from PV / refill
TMEM_COLS = 256
NS = 2  # S buffers (64 FP32 columns each) and P buffers (32 packed BF16x2 columns)
P_COL = 64 * NS
O_COL = P_COL + 32 * NS
THRESH = 8.0  # log2 headroom of the lazy running max (module notes)
PITCH = 68  # FP32 words per partial O row in shared memory
# FP32 D, BF16 A and B, K-major both, N = 64, M = 128 (S = Q K^T)
S_INSTR = 1 << 4 | 1 << 7 | 1 << 10 | (KB >> 3) << 17 | (128 >> 4) << 24
# FP32 D, BF16 A (TMEM, K-major) and B (MN-major V), N = 64 (head dim), M = 128
PV_INSTR = 1 << 4 | 1 << 7 | 1 << 10 | 1 << 16 | (HEAD >> 3) << 17 | (128 >> 4) << 24


@asm()
def mma_pv_block(d, a, b, instruction, accumulate, done):
    """Warp-uniform issue of a stage's PV (P from TMEM, K16 slices at +8 columns;
    V MN-major, +2048 B per slice): the elected lane runs the four BF16 MMAs (the
    first accumulates iff ``accumulate``) and commits ``done``."""
    mma = "tcgen05.mma.cta_group::1.kind::f16"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt; .reg .b32 a1,a2,a3; .reg .b64 b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,$4,0; "
    code += "add.u32 a1,$1,8; add.u32 a2,$1,16; add.u32 a3,$1,24; "
    code += "add.s64 b1,$2,128; add.s64 b2,$2,256; add.s64 b3,$2,384; "
    code += f"@e {mma} [$0],[$1],$2,$3,p0; @e {mma} [$0],[a1],b1,$3,pt; "
    code += f"@e {mma} [$0],[a2],b2,$3,pt; @e {mma} [$0],[a3],b3,$3,pt; "
    code += f"@e {commit} [$5]; }}"
    operands: tuple = (Uint32(d), Uint32(a), Uint64(b), Uint32(instruction))
    return code, "r,r,l,r,r,r", operands + (Int32(accumulate), Uint32(done))


@asm()
def tma_load5(dst, tensor_map, barrier, token, head, block):
    """One {64 values, 64 tokens, K|V} box of the 5-D cache map."""
    code = "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes [$0], [$1, {$2, $3, $2, $4, $5}], [$6];"
    operands = (Uint32(dst), Int64(tensor_map), Int32(0), Int32(token), Int32(head))
    operands += (Int32(block), Uint32(barrier))
    return code, "r,l,r,r,r,r,r,~{memory}", operands


def kv_map_5d(cache, kv_heads, head_stride):
    """The TMA map spec (``common.host.tensor_maps``) of a [blocks, heads, page,
    K | V] BF16 cache view as (64 values, page tokens, K | V, heads, blocks): one
    {64, 64, 2} box lands a stage's K image then its V image (128 B swizzle)."""
    import cuda.bindings.driver as cuda

    page, blocks = cache.shape[2], cache.shape[0]
    return (
        cache.data_ptr(),
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        (HEAD, page, 2, kv_heads, blocks),
        (4 * HEAD, 2 * HEAD, head_stride * 2, cache.stride(0) * 2),
        (HEAD, KB, 2, 1, 1),
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    )


@asm()
def ld16(address):
    """16 FP32 TMEM columns of this lane, waited."""
    names = ",".join(f"${i}" for i in range(16))
    code = f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{names}}},[$16]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * 16 + ["r", "~{memory}"]), (Uint32(address),), (
        (Float32,) * 16
    )


@asm()
def ld8(address):
    """8 FP32 TMEM columns of this lane, waited."""
    names = ",".join(f"${i}" for i in range(8))
    code = f"tcgen05.ld.sync.aligned.32x32b.x8.b32 {{{names}}},[$8]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * 8 + ["r", "~{memory}"]), (Uint32(address),), (
        (Float32,) * 8
    )


@asm()
def ld64(address):
    names = ",".join(f"${i}" for i in range(64))
    code = f"tcgen05.ld.sync.aligned.32x32b.x64.b32 {{{names}}},[$64]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * 64 + ["r", "~{memory}"]), (Uint32(address),), (
        (Float32,) * 64
    )


@asm()
def st64(address, values):
    names = ",".join(f"${i + 1}" for i in range(64))
    code = f"tcgen05.st.sync.aligned.32x32b.x64.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",f" * 64 + ",~{memory}", (Uint32(address), *values)


@asm()
def st8w(address, words):
    names = ",".join(f"${i + 1}" for i in range(8))
    code = f"tcgen05.st.sync.aligned.32x32b.x8.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",r" * 8 + ",~{memory}", (Uint32(address), *words)


@asm()
def st4w(address, words):
    names = ",".join(f"${i + 1}" for i in range(4))
    code = f"tcgen05.st.sync.aligned.32x32b.x4.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",r" * 4 + ",~{memory}", (Uint32(address), *words)


@asm()
def pair_barrier(barrier_id):
    """bar.sync of 64 threads (a warp pair) on a register barrier id."""
    return "bar.sync $0, 64;", "r,~{memory}", (Int32(barrier_id),)


@asm()
def st32w(address, words):
    names = ",".join(f"${i + 1}" for i in range(32))
    code = f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",r" * 32 + ",~{memory}", (Uint32(address), *words)


@asm()
def alloc_tmem(slot, columns):
    code = f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0],{columns}; "
    code += "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    return code, "r,~{memory}", (Uint32(slot),)


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


@asm()
def store_u64(address, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm(Float32, pure=True)
def fmax(a, b):
    return "max.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@cute.jit
def desc_sw128(address, lbo: cutlass.Constexpr[int]):
    """128B-swizzled shared memory descriptor of a 1024 B aligned image (SBO 1024 B:
    8-row atoms); ``lbo``: the MN-major atom stride (K-major: unused)."""
    value = Uint64((Uint32(address) & 262143) >> 4) | Uint64(1) << 46
    value |= Uint64(lbo >> 4) << 16 | Uint64(64) << 32 | Uint64(2) << 61
    return value


class GqaWindowUmma:
    """One drafter layer's fused attention on tcgen05 (module notes): ``group`` query
    heads per KV head (``group * q`` <= 32 rows), ``kv_heads`` per rank, the cache's
    ``page`` tokens per block (a multiple of 64), ``ring`` stages in flight,
    ``window`` keys. Same arguments as ``GqaWindow``."""

    def __init__(self, group, kv_heads, q, page, ring, window, v_scale=None,
                 round_norm=False, debug=0, redo=False, splits=1, fill=4, trace=0,
                 trace_req=0, swarp=False, die_sms=None, wide=False):  # fmt: skip
        # swarp: a sixth warp issues S = Q K^T of each stage as soon as it landed
        # and its S buffer is free; the producer warp then only issues PV and
        # refills (the S issue no longer queues behind the P-ready wait)
        self.swarp = swarp
        # wide (needs swarp): warps 6-9 are four more softmax warps; warp w and
        # warp w + 4 (w + 8 for quarters 0-1) split their lane quarter's 16 keys of
        # a stage in two halves of 8, agreeing on the row max of the stage
        # through shared memory, so each warp's dependent chain is half as long
        assert not wide or swarp
        self.wide = wide
        self.threads = THREADS + (32 if swarp else 0) + (128 if wide else 0)
        # die_sms (die split): a resident grid of two CTAs per SM
        # (common.resident); the CTAs of die d take the (request, KV head) units
        # whose KV head the cache homes on die d (mask argument, bit h = die of
        # KV head h), one unit per CTA
        self.die_sms = die_sms
        assert die_sms is None or splits == 1
        # ``trace`` (development only): a global address of int64 [48, 16] per-stage
        # clock64 events of the CTA (request ``trace_req``, KV head 1) (``trc``)
        self.trace, self.trace_req = trace, trace_req
        # ``fill`` (development only, never set by the install): where the first ring
        # stages are issued: lanes [0, a) before griddepcontrol.wait, [a, b) right
        # after it, the rest after the Q prep. Default 4: two stages after the wait,
        # the rest after the Q prep, whose qkv loads otherwise queue behind the
        # whole first fill (B96 setup 3.1 -> 2.3 us, layer -1.2 us)
        self.fill = ((0, ring), (0, 0), (ring, ring), (2, 2), (0, 2))[fill]
        # ``splits`` CTAs per (request, KV head), a cluster: an even share of the
        # stages each; their partial rows LSE-merge by DSMEM pushes to the CTA owning
        # each row (as GqaWindow's)
        self.splits = splits
        self.owned = -(-group * q // splits)
        self.debug_redo = redo  # (tests: the redo path with unmoved metadata)
        # ``debug`` (development only, never set by the install): a global address of
        # int64 [CTAs, 16] phase stamps / wait cycles (``rec``)
        self.debug = debug
        assert page % KB == 0 and window >= 1 and 1 <= q <= 8
        assert group * q <= 32, (group, q)
        assert ring >= 3
        self.group, self.kv_heads, self.q, self.page = group, kv_heads, q, page
        self.ring, self.window, self.v_scale = ring, window, v_scale
        self.round_norm = round_norm
        self.rows = group * q
        self.pages = -(-(window + q - 1 + 2 * KB) // page) + 1
        assert self.pages <= THREADS
        assert 4 * 32 * PITCH * 4 <= ring * SLOT  # (partials in the freed ring)
        if splits == 1:  # (+ the own rows' partials, recv)
            assert (4 * 32 + self.owned) * PITCH * 4 <= ring * SLOT

    @cute.jit
    def __call__(self, *args):
        out = args[8]
        requests = out.shape[0] // self.q
        cluster = None
        if cutlass.const_expr(self.splits > 1):
            cluster = (self.splits, 1, 1)
        grid = (self.splits, self.kv_heads, requests)
        if cutlass.const_expr(self.die_sms is not None):
            grid = (2 * sum(self.die_sms), 1, 1)
        launch(
            self.kernel, args, grid, self.threads,
            cluster=cluster, max_number_threads=[self.threads, 1, 1],
        )  # fmt: skip

    @cute.jit
    def issue(self, kv_map, bar, tile, head, block, offset):
        """Stage at (block, offset): one box of the 5-D map (``kv_map_5d``), K rows
        0..63 then V rows 0..63."""
        arm(bar, SLOT)
        tma_load5(tile, kv_map, bar, offset, head, block)

    @cute.jit
    def fixup(self, tile, p0, lane, lo, length, kv_new, newvalid):
        """Rows lane and lane + 32 of a stage: zero outside [lo, length); the computed
        K / V of a query row whose slot was valid."""
        q_first = length - self.q
        zero = (Uint32(0),) * 4
        for h in cutlass.range_constexpr(2):
            row = lane + 32 * h
            position = p0 + row
            if (position < lo) | (position >= length):
                for ch in cutlass.range_constexpr(8):
                    store_words(tile + swz(row, ch), zero, shared=True)
                    store_words(tile + KTILE + swz(row, ch), zero, shared=True)
            elif position >= q_first:
                i = position - q_first
                if newvalid[i] != 0:
                    for ch in cutlass.range_constexpr(8):
                        kq = load_words(kv_new + i * 256 + 16 * ch, count=4,
                                        shared=True)  # fmt: skip
                        vq = load_words(kv_new + i * 256 + 128 + 16 * ch, count=4,
                                        shared=True)  # fmt: skip
                        store_words(tile + swz(row, ch), kq, shared=True)
                        store_words(tile + KTILE + swz(row, ch), vq, shared=True)

    @cute.jit
    def stages(self, length, split):
        """(window start, this CTA's first 64-key stage, its stage count)."""
        lo = max(length - self.q - (self.window - 1), Int32(0))
        first = Int32(0)
        count = Int32(0)
        if length > 0:
            first = lo // KB
            count = (length - 1) // KB - first + 1
        begin = first + count * split // self.splits
        end = first + count * (split + 1) // self.splits
        return lo, begin, end - begin

    @cute.jit
    def rope_rows(self, cos_sin, position, group8):
        row = min(max(position, Int32(0)), Int32(cos_sin.shape[0] - 1))
        base = cos_sin.iterator + row * cos_sin.layout.stride[0]
        cos = load_words(base + 8 * (group8 % 4), count=4)
        sin = load_words(base + 32 + 8 * (group8 % 4), count=4)
        return cos, sin

    @cute.kernel
    def kernel(self, args):
        qkv, positions, slots, seq_lens, table, cos_sin, qw, kw, out = args[0:9]
        kv_map, kv_base, row_stride, head_stride = args[9:13]
        eps = args[14]
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        R: cutlass.Constexpr = self.ring
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        HKV: cutlass.Constexpr = self.kv_heads
        c = Float32(LOG2E / (HEAD**0.5))

        split, head, request = cute.arch.block_idx()
        TH: cutlass.Constexpr = self.threads
        S: cutlass.Constexpr = self.splits
        OWN: cutlass.Constexpr = self.owned
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        group8 = lane & 7
        upper = group8 >= 4
        table_last = Int32(table.shape[1] - 1)
        blocks_max = args[13] - 1

        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, cutlass.Uint8, R * SLOT, align=1024)
        qtile = alloc(smem, cutlass.Uint8, 128 * 128, align=1024)
        newkv = alloc(smem, cutlass.Uint8, Q * 256, align=16)
        newvalid = alloc(smem, Int32, Q, align=16)
        bars = alloc(smem, cutlass.Int64, R + 4 * NS + 1, align=8)
        # S = 1: the own-row partials land in the freed ring behind the quarters'
        # partials, and the merge stats in the Q tile (its last MMA completed before
        # the merge reads PV's result): 2 CTAs per SM keep a 6-slot ring
        if cutlass.const_expr(S == 1):
            recv = cute.make_tensor(
                cute.recast_ptr(ring.iterator + 4 * 32 * PITCH * 4, dtype=Float32),
                cute.make_layout(OWN * PITCH),
            )  # fmt: skip
        else:
            recv = alloc(smem, Float32, S * OWN * PITCH, align=16)
        recv_base = Int32(shared_address(recv.iterator))
        blocks = alloc(smem, Int32, NP, align=16)
        stats = cute.make_tensor(
            cute.recast_ptr(qtile.iterator, dtype=Float32), cute.make_layout(4 * 32 * 2)
        )  # fmt: skip
        tslot = alloc(smem, Int32, 2, align=16)  # [1]: the redo flag
        # wide: [3][2 halves][4 quarters][32 lanes] stage maxima (stage parity
        # 0 / 1) and the halves' row sums (2) the pair exchanges
        WIDE: cutlass.Constexpr = self.wide
        SW: cutlass.Constexpr = 8 if self.wide else 4  # softmax warps
        xch = alloc(smem, Float32, 768 if self.wide else 4, align=16)
        # ---- die split: this CTA's (request, KV head) from the resident grid's
        # landing: die d's CTAs (slot-major over its SMs) take die d's units
        live = Int32(1)
        launch_i = Int32(0)
        if cutlass.const_expr(self.die_sms is not None):
            lslot = alloc(smem, Int32, 2, align=8)
            mask = args[18]
            if tid == 0:
                d0, rank, slot, nth = land(args[15], args[16], args[17], 2)
                n_d = Int32(self.die_sms[0])
                if d0 != 0:
                    n_d = Int32(self.die_sms[1])
                lslot[0] = (d0 << 16) | (slot * n_d + rank)
                lslot[1] = nth
            cute.arch.sync_threads()
            die = lslot[0] >> 16
            cu = lslot[0] & 0xFFFF
            launch_i = lslot[1]
            split = Int32(0)
            heads_d = Int32(0)
            for h in cutlass.range_constexpr(HKV):
                if ((mask >> h) & 1) == die:
                    heads_d = heads_d + 1
            units_d = (out.shape[0] // Q) * heads_d
            if cu >= units_d:
                live = Int32(0)
            cu = max(min(cu, units_d - 1), Int32(0))
            request = cu // max(heads_d, Int32(1))
            k_h = cu - request * max(heads_d, Int32(1))
            seen = Int32(0)
            head = Int32(0)
            for h in cutlass.range_constexpr(HKV):
                if ((mask >> h) & 1) == die:
                    if seen == k_h:
                        head = Int32(h)
                    seen = seen + 1
        tr_cta = (request == self.trace_req) & (head == 1) & (split == 0)
        table_row = table.iterator + request * table.layout.stride[0]
        ring_base = Int32(shared_address(ring.iterator))
        q_base = Int32(shared_address(qtile.iterator))
        kv_new = Int32(shared_address(newkv.iterator))
        redo_flag = Int32(shared_address(tslot.iterator)) + 4
        full = Int32(shared_address(bars.iterator))
        sready = full + 8 * R  # [NS]
        pready = sready + 8 * NS  # [NS]
        pvdone = pready + 8 * NS  # [NS]
        sfree = pvdone + 8 * NS  # [NS]: the warps loaded S
        recv_bar = sfree + 8 * NS  # the cluster's partial rows landed
        my_rows = min(max(ROWS - split * OWN, 0), OWN)

        if tid == 0:
            for i in cutlass.range_constexpr(R):
                init_barrier(full + 8 * i, 1)
            for b in cutlass.range_constexpr(NS):
                init_barrier(sready + 8 * b, 1)
                init_barrier(pready + 8 * b, SW)
                init_barrier(pvdone + 8 * b, 1)
                init_barrier(sfree + 8 * b, SW)
            if cutlass.const_expr(S > 1):
                init_barrier(recv_bar, 1)
                arm(recv_bar, S * my_rows * (PITCH * 4))
        # the Q tile's padding rows (32 - ROWS per lane quarter)
        PAD: cutlass.Constexpr = (32 - ROWS) * 4 * 8
        for it in cutlass.range_constexpr(-(-PAD // TH)):
            t = it * TH + tid
            if t < PAD:
                quarter, rest = t // ((32 - ROWS) * 8), t % ((32 - ROWS) * 8)
                prow = 32 * quarter + ROWS + rest // 8
                store_words(q_base + swz(prow, rest % 8), (Uint32(0),) * 4,
                            shared=True)  # fmt: skip
        q_w = load_words(qw.iterator + 8 * group8, count=4)
        k_w = load_words(kw.iterator + 8 * group8, count=4)
        # ---- before the PDL wait: the metadata as it stands decides only addresses
        # (the first stages' TMA loads right after the wait, the RoPE rows); after
        # the wait it is read again, and loads from moved metadata are redone (never
        # in a decode step: the metadata is written before the graph)
        length0 = load_i32(seq_lens.iterator + request) * live
        _, first0, count0 = self.stages(length0, split)
        page_lo0 = first0 * KB // N
        if tid < NP:
            entry0 = load_i32(table_row + min(page_lo0 + tid, table_last))
            blocks[tid] = min(max(entry0, Int32(0)), blocks_max)
        TASKS: cutlass.Constexpr = ROWS * 8 + Q * 8
        tasks = []
        for it in cutlass.range_constexpr(-(-TASKS // TH)):
            t = it * TH + tid
            is_kv = t >= ROWS * 8
            i = min(t // 8, ROWS - 1) // G
            gq = min(t // 8, ROWS - 1) % G
            col = (head * G + gq) * HEAD + 8 * group8
            if is_kv:
                i = min((t - ROWS * 8) // 8, Q - 1)
                col = HQ * HEAD + head * HEAD + 8 * group8
            row = request * Q + i
            pos0 = load_i64(positions.iterator + row)
            cos0, sin0 = self.rope_rows(cos_sin, Int32(pos0), group8)
            tasks.append((t, is_kv, i, col, row, pos0, cos0, sin0))
        if tid == 0:
            store_shared_u32(redo_flag, 0)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if cutlass.const_expr(S > 1):
            cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM push
        spec_n = min(count0, Int32(R))
        FA: cutlass.Constexpr = self.fill[0]
        FB: cutlass.Constexpr = self.fill[1]
        if cutlass.const_expr(FA > 0):
            if warp == PRODUCER:
                if lane < min(spec_n, Int32(FA)):
                    j = first0 + lane
                    self.issue(kv_map, full + 8 * lane, ring_base + lane * SLOT, head,
                               blocks[j * KB // N - page_lo0], j * KB % N)  # fmt: skip
        self.rec(request, head, 0, global_timer(), tid == 0)
        cute.arch.griddepcontrol_wait()
        # (TMEM after the wait: allocated before it, while the previous kernel's CTAs
        # held theirs, some CTAs measured ~20 us late out of the wait in-step)
        if warp == PRODUCER:
            alloc_tmem(Int32(shared_address(tslot.iterator)), TMEM_COLS)
        self.rec(request, head, 1, global_timer(), tid == 0)

        # ---- after the wait: the first stages' loads at the pre-wait addresses, then
        # the metadata again (confirming them) and the prep math
        if warp == PRODUCER:
            if (lane >= FA) & (lane < min(spec_n, Int32(FB))):
                j = first0 + lane
                self.issue(kv_map, full + 8 * lane, ring_base + lane * SLOT, head,
                           blocks[j * KB // N - page_lo0], j * KB % N)  # fmt: skip
        length = load_i32(seq_lens.iterator + request) * live
        lo, first, count = self.stages(length, split)
        page_lo = first * KB // N
        moved = length != length0
        if tid < NP:
            entry = load_i32(table_row + min(page_lo0 + tid, table_last))
            moved = moved | (min(max(entry, Int32(0)), blocks_max) != blocks[tid])
        if cutlass.const_expr(self.debug_redo):
            moved = tid == 0
        if moved:
            atomic_or_shared(redo_flag, 1)

        q_first = length - Q
        stride0 = qkv.layout.stride[0]
        for t, is_kv, i, col, row, pos0, cos0, sin0 in tasks:
            # (every thread runs a task: norm_rope8's 8-lane shuffles)
            pos = load_i64(positions.iterator + row)
            words = load_words(qkv.iterator + row * stride0 + col, count=4,
                               mutable=True)  # fmt: skip
            wts = cute.make_rmem_tensor(4, Uint32)
            rope = cute.make_rmem_tensor(8, Uint32)
            for k in cutlass.range_constexpr(4):
                wts[k] = q_w[k]
                rope[k] = cos0[k]
                rope[4 + k] = sin0[k]
            if is_kv:
                for k in cutlass.range_constexpr(4):
                    wts[k] = k_w[k]
            if pos != pos0:  # (the position moved since the pre-wait read)
                cos1, sin1 = self.rope_rows(cos_sin, Int32(pos), group8)
                for k in cutlass.range_constexpr(4):
                    rope[k] = cos1[k]
                    rope[4 + k] = sin1[k]
            x = [bf16(words[e // 2], e % 2 == 1) for e in range(8)]
            roped = norm_rope8(
                x, tuple(wts[k] for k in range(4)), tuple(rope[k] for k in range(4)),
                tuple(rope[4 + k] for k in range(4)), eps, upper, self.round_norm,
            )  # fmt: skip
            if t < ROWS * 8:
                for quarter in cutlass.range_constexpr(4):
                    store_words(q_base + swz(32 * quarter + t // 8, group8),
                                tuple(roped), shared=True)  # fmt: skip
            elif (t < TASKS) & (length > 0) & (q_first + i >= first * KB) & (
                q_first + i < (first + count) * KB
            ):  # (the CTA whose stages hold the query row's position)
                vwords = load_words(qkv.iterator + row * stride0 + col + HKV * HEAD,
                                    count=4, mutable=True)  # fmt: skip
                slot = load_i64(slots.iterator + row)
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
                    s_elem = Int64(slot // N) * row_stride
                    s_elem = s_elem + Int64(head) * head_stride
                    s_elem = s_elem + Int64(slot % N) * 128 + 8 * group8
                    s_addr = kv_base + s_elem * 2
                    store_words(s_addr, tuple(roped))
                    store_words(s_addr + 128, tuple(value))
        if cutlass.const_expr(FB < R):
            if warp == PRODUCER:
                if (lane >= FB) & (lane < spec_n):
                    j = first0 + lane
                    self.issue(kv_map, full + 8 * lane, ring_base + lane * SLOT, head,
                               blocks[j * KB // N - page_lo0], j * KB % N)  # fmt: skip
        fence_async_shared()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Uint32(tslot[0])

        # ---- moved metadata: drain the speculated loads, reissue from the confirmed
        # metadata; those slots' phases then start at 1 (``offset_bits``)
        offset_bits = Int32(0)
        if load_words(redo_flag, count=1, shared=True)[0] != 0:
            if warp == PRODUCER:
                if lane < spec_n:
                    wait(full + 8 * lane, 0, scope="cta")
            if tid < NP:
                entry = load_i32(table_row + min(page_lo + tid, table_last))
                blocks[tid] = min(max(entry, Int32(0)), blocks_max)
            cute.arch.sync_threads()
            if warp == PRODUCER:
                if lane < min(count, Int32(R)):
                    j = first + lane
                    self.issue(kv_map, full + 8 * lane, ring_base + lane * SLOT, head,
                               blocks[j * KB // N - page_lo], j * KB % N)  # fmt: skip
            offset_bits = (Int32(1) << spec_n) - 1
        lo_max = max(length - W, Int32(0))
        self.rec(request, head, 2, global_timer(), tid == 0)
        self.rec(request, head, 12, Int64(count), tid == 0)

        if warp == PRODUCER:
            # ---- producer: S of stage t + 1 issued before stage t's PV waits on its
            # P; ring refills behind the PVs
            q_desc = desc_sw128(q_base, 16)
            a_full = Int64(0)
            a_pready = Int64(0)
            loop0 = self.tick()
            trp = tr_cta & (lane == 0)
            if cutlass.const_expr(self.swarp):
                # PV of each stage once its P is ready; the stage's slot refills as
                # soon as its PV completed (S issued by the S warp)
                for t in cutlass.range(count, unroll=1):
                    b = t % NS
                    self.trc(trp, t, 4)
                    wait(pready + 8 * b, (t // NS) & 1, scope="cta")
                    self.trc(trp, t, 5)
                    fence_tmem()
                    v_desc = desc_sw128(ring_base + (t % R) * SLOT + KTILE, KTILE)
                    mma_pv_block(
                        tmem + O_COL, tmem + P_COL + Uint32(32) * Uint32(b), v_desc,
                        Uint32(PV_INSTR), Int32(t > 0), pvdone + 8 * b,
                    )  # fmt: skip
                    self.trc(trp, t, 6)
                    nxt = t + R
                    if nxt < count:
                        wait(pvdone + 8 * b, (t // NS) & 1, scope="cta")
                        self.trc(trp, t, 7)
                        if lane == 0:
                            j = first + nxt
                            blk = blocks[j * KB // N - page_lo]
                            r_i = t % R
                            self.issue(kv_map, full + 8 * r_i, ring_base + r_i * SLOT,
                                       head, blk, j * KB % N)  # fmt: skip
            else:
                if count > 0:
                    a_full = a_full + self.issue_s(
                        Int32(0), first, lo, lo_max, q_first, length, ring_base, full,
                        sready, sfree, tmem, q_desc, lane, kv_new, newvalid, offset_bits,
                        trp,
                    )  # fmt: skip
                for t in cutlass.range(count, unroll=1):
                    if t + 1 < count:
                        a_full = a_full + self.issue_s(
                            t + 1, first, lo, lo_max, q_first, length, ring_base, full,
                            sready, sfree, tmem, q_desc, lane, kv_new, newvalid,
                            offset_bits, trp,
                        )  # fmt: skip
                    c0 = self.tick()
                    b = t % NS
                    self.trc(trp, t, 4)
                    wait(pready + 8 * b, (t // NS) & 1, scope="cta")
                    self.trc(trp, t, 5)
                    a_pready = a_pready + self.tick() - c0
                    fence_tmem()
                    v_desc = desc_sw128(ring_base + (t % R) * SLOT + KTILE, KTILE)
                    mma_pv_block(
                        tmem + O_COL, tmem + P_COL + Uint32(32) * Uint32(b), v_desc,
                        Uint32(PV_INSTR), Int32(t > 0), pvdone + 8 * b,
                    )  # fmt: skip
                    self.trc(trp, t, 6)
                    if t >= 1:
                        nxt = t - 1 + R
                        if nxt < count:  # stage t - 1's PV (issued last iteration) done
                            wait(pvdone + 8 * ((t - 1) % NS), ((t - 1) // NS) & 1,
                                 scope="cta")  # fmt: skip
                            self.trc(trp, t - 1, 7)
                            if lane == 0:
                                j = first + nxt
                                blk = blocks[j * KB // N - page_lo]
                                r_i = (t - 1) % R
                                self.issue(kv_map, full + 8 * r_i, ring_base + r_i * SLOT,
                                           head, blk, j * KB % N)  # fmt: skip
            first_lane = lane == 0
            self.rec(request, head, 3, global_timer(), first_lane)
            self.rec(request, head, 4, a_full, first_lane)
            self.rec(request, head, 6, a_pready, first_lane)
            self.rec(request, head, 15, self.tick() - loop0, first_lane)
            if cutlass.const_expr(S > 1):
                cute.arch.cluster_wait()
        elif warp == SWARP:
            if cutlass.const_expr(self.swarp):
                # ---- S issuer: S of stage t as soon as it landed and S buffer t % NS
                # is free (the warps loaded stage t - NS's S)
                q_desc_s = desc_sw128(q_base, 16)
                trq = tr_cta & (lane == 0)
                for t in cutlass.range(count, unroll=1):
                    self.issue_s(
                        t, first, lo, lo_max, q_first, length, ring_base, full,
                        sready, sfree, tmem, q_desc_s, lane, kv_new, newvalid,
                        offset_bits, trq,
                    )  # fmt: skip
        else:
            # ---- softmax: warp w = TMEM lane quarter w, the stages' keys 16w..16w+15
            # (wide: quarter w % 4, half hf of those keys, KW = 8 keys each)
            KW: cutlass.Constexpr = 8 if self.wide else 16
            qtr = warp % 4
            hf = Int32(0)
            if warp > SWARP:
                hf = Int32(1)
            kofs = 16 * qtr + KW * hf  # the warp's first key column of a stage
            r = lane
            lanes = Uint32(32 * qtr) << 16
            hi_r = length - 1
            lo_r = lo
            if r < ROWS:
                hi_r = q_first + r // G
                lo_r = max(hi_r - (W - 1), Int32(0))
            m_use = Float32(NEG_INF)
            l_run = Float32(0)
            a_sr = Int64(0)
            a_res = Int64(0)
            p_lanes = tmem + lanes + P_COL
            if cutlass.const_expr(self.wide):
                # the other quarters' 8-column chunks of every P buffer: zero (the
                # quarter's own chunk is written whole every stage by its pair)
                for k in cutlass.range_constexpr(NS):
                    for ch in cutlass.range_constexpr(4):
                        if ch != qtr:
                            st8w(p_lanes + Uint32(32 * k + 8 * ch), (Uint32(0),) * 8)
            else:
                for k in cutlass.range_constexpr(NS):  # every P buffer's columns: zero
                    st32w(p_lanes + Uint32(32 * k), (Uint32(0),) * 32)
            xch_base = Int32(shared_address(xch.iterator))
            trs = tr_cta & (lane == 0) & (warp == 0)
            for t in cutlass.range(count, unroll=1):
                b = t % NS
                s_addr = tmem + lanes + Uint32(64) * Uint32(b)
                c0 = self.tick()
                self.trc(trs, t, 8)
                wait(sready + 8 * b, (t // NS) & 1, scope="cta")
                self.trc(trs, t, 9)
                a_sr = a_sr + self.tick() - c0
                fence_tmem()
                if cutlass.const_expr(self.wide):
                    raw = ld8(s_addr + Uint32(kofs))
                else:
                    raw = ld16(s_addr + Uint32(16) * Uint32(warp))
                fence_tmem(before=True)
                cute.arch.sync_warp()
                self.trc(trs, t, 10)
                if lane == 0:
                    arrive(sfree + 8 * b)
                ps0 = (first + t) * KB
                xs = cute.make_rmem_tensor(KW, Float32)
                for e in cutlass.range_constexpr(KW):
                    xs[e] = raw[e]
                if (ps0 < lo_max) | (ps0 + KB > q_first):
                    for e in cutlass.range_constexpr(KW):
                        key = ps0 + kofs + e
                        if (key < lo_r) | (key > hi_r):
                            xs[e] = Float32(NEG_INF)
                mx = tree(fmax, [xs[e] for e in range(KW)]) * c
                if cutlass.const_expr(self.wide):
                    # the pair's row max of the stage (both halves then take the
                    # same rescale decisions and P scale)
                    xslot = ((t % 2) * 8 + hf * 4 + qtr) * 32 + lane
                    store_shared_u32(xch_base + 4 * xslot, float_bits(mx))
                    pair_barrier(8 + qtr)
                    xother = load_words(
                        xch_base + 4 * (xslot + 128 - 256 * hf), count=1, shared=True,
                        result_type=Float32, word_type="f32",
                    )[0]  # fmt: skip
                    mx = fmax(mx, xother)
                need = mx > m_use + Float32(THRESH)
                rescale = Int32(0)
                if need & (m_use > Float32(NEG_INF)):
                    rescale = Int32(1)
                if vote_any(rescale) != 0:
                    alpha = Float32(1)
                    if rescale != 0:
                        alpha = ex2(m_use - mx)
                    l_run = l_run * alpha
                    a_res = a_res + 1
                    if hf == 0:  # (wide: one warp of the pair rescales the quarter's O)
                        wait(pvdone + 8 * ((t - 1) % NS), ((t - 1) // NS) & 1,
                             scope="cta")  # fmt: skip
                        fence_tmem()
                        o_addr = tmem + lanes + O_COL
                        o = ld64(o_addr)
                        st64(o_addr, tuple(o[e] * alpha for e in range(64)))
                if need:
                    m_use = mx
                bias = Float32(0)
                if m_use > Float32(NEG_INF):
                    bias = -m_use
                ps = [ex2(fma(xs[e], c, bias)) for e in range(KW)]
                l_run = l_run + tree(lambda u, v: u + v, ps)
                packed = tuple(bf16x2(ps[2 * k], ps[2 * k + 1]) for k in range(KW // 2))
                self.trc(trs, t, 11)
                if t >= NS:  # stage t - NS's PV read this P buffer
                    wait(pvdone + 8 * b, ((t - NS) // NS) & 1, scope="cta")
                    fence_tmem()
                self.trc(trs, t, 12)
                if cutlass.const_expr(self.wide):
                    st4w(p_lanes + Uint32(32) * Uint32(b) + Uint32(kofs // 2), packed)
                else:
                    p_own = p_lanes + Uint32(32) * Uint32(b) + Uint32(8) * Uint32(warp)
                    st8w(p_own, packed)
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(pready + 8 * b)
                self.trc(trs, t, 13)
                self.trc(tr_cta & (lane == 0) & (warp == 3), t, 14)
            cute.arch.griddepcontrol_launch_dependents()
            self.rec(request, head, 8, a_sr, tid == 0)
            self.rec(request, head, 10, global_timer(), tid == 0)
            if cutlass.const_expr(self.wide):
                # the pair's row sums: half 1 hands its l to half 0, which merges
                if hf != 0:
                    store_shared_u32(xch_base + 4 * (512 + qtr * 32 + lane),
                                     float_bits(l_run))  # fmt: skip
                pair_barrier(8 + qtr)
                if hf == 0:
                    l_run = l_run + load_words(
                        xch_base + 4 * (512 + qtr * 32 + lane), count=1, shared=True,
                        result_type=Float32, word_type="f32",
                    )[0]  # fmt: skip
            if hf == 0:
                # ---- merge the four quarters' partials (S > 1: then the cluster's)
                if cutlass.const_expr(S > 1):
                    cute.arch.cluster_wait()  # every peer's receive barrier is armed
                if count > 0:
                    last = count - 1
                    wait(pvdone + 8 * (last % NS), (last // NS) & 1, scope="cta")
                    fence_tmem()
                ov = cute.make_rmem_tensor(64, Float32)
                for e in cutlass.range_constexpr(64):
                    ov[e] = Float32(0)
                if count > 0:
                    o = ld64(tmem + lanes + O_COL)
                    for e in cutlass.range_constexpr(64):
                        ov[e] = o[e]
                named_barrier(1, 128)  # every warp past its PV wait: the ring is free
                part = ring_base + (warp * 32 + r) * (PITCH * 4)
                for k in cutlass.range_constexpr(16):
                    vals = (ov[4 * k], ov[4 * k + 1], ov[4 * k + 2], ov[4 * k + 3])
                    store_words(part + 16 * k, tuple(float_word(v) for v in vals),
                                shared=True)  # fmt: skip
                stats[(warp * 32 + r) * 2] = m_use
                stats[(warp * 32 + r) * 2 + 1] = l_run
                named_barrier(1, 128)
                for it in cutlass.range_constexpr(-(-(ROWS * 8) // 128)):
                    task = it * 128 + tid
                    if task < ROWS * 8:
                        rr, d8 = task // 8, task % 8
                        ms = [stats[(w * 32 + rr) * 2] for w in range(4)]
                        top = fmax(fmax(ms[0], ms[1]), fmax(ms[2], ms[3]))
                        use = top
                        if top == Float32(NEG_INF):
                            use = Float32(0)
                        total = Float32(0)
                        values = [Float32(0)] * 8
                        for w in cutlass.range_constexpr(4):
                            weight = ex2(ms[w] - use)
                            total = fma(stats[(w * 32 + rr) * 2 + 1], weight, total)
                            src = ring_base + (w * 32 + rr) * (PITCH * 4) + 32 * d8
                            lo4 = load_words(src, count=4, shared=True)
                            hi4 = load_words(src + 16, count=4, shared=True)
                            for e in cutlass.range_constexpr(4):
                                values[e] = fma(bits_float(lo4[e]), weight, values[e])
                                hi = fma(bits_float(hi4[e]), weight, values[4 + e])
                                values[4 + e] = hi
                        if cutlass.const_expr(S == 1):
                            if live != 0:
                                self.store_row(out, request, head, rr, d8, values, total)
                        else:
                            slot_r = (split * OWN + rr % OWN) * PITCH
                            dst = recv_base + (slot_r + 8 * d8) * 4
                            lo_w = tuple(float_word(values[e]) for e in range(4))
                            hi_w = tuple(float_word(values[4 + e]) for e in range(4))
                            push4(dst, lo_w, recv_bar, rr // OWN)
                            push4(dst + 16, hi_w, recv_bar, rr // OWN)
                            if d8 == 0:
                                stat4 = (float_word(top), float_word(total), Uint32(0),
                                         Uint32(0))  # fmt: skip
                                push4(recv_base + (slot_r + 64) * 4, stat4, recv_bar,
                                      rr // OWN)  # fmt: skip
                if cutlass.const_expr(S > 1):
                    # ---- the owned rows: every CTA's partial landed
                    wait(recv_bar, 0, scope="cta")
                    for it in cutlass.range_constexpr(-(-(OWN * 8) // 128)):
                        task = it * 128 + tid
                        rr, d8 = task // 8, task % 8
                        if (task < OWN * 8) & (rr < my_rows):
                            tops = [recv[(sp * OWN + rr) * PITCH + 64] for sp in range(S)]
                            top = tree(fmax, tops)
                            use = top
                            if top == Float32(NEG_INF):
                                use = Float32(0)
                            total = Float32(0)
                            values = [Float32(0)] * 8
                            for sp in cutlass.range_constexpr(S):
                                base_s = (sp * OWN + rr) * PITCH
                                weight = ex2(tops[sp] - use)
                                total = fma(recv[base_s + 65], weight, total)
                                src = recv_base + (base_s + 8 * d8) * 4
                                lo4 = load_words(src, count=4, shared=True)
                                hi4 = load_words(src + 16, count=4, shared=True)
                                for e in cutlass.range_constexpr(4):
                                    values[e] = fma(bits_float(lo4[e]), weight, values[e])
                                    hi = fma(bits_float(hi4[e]), weight, values[4 + e])
                                    values[4 + e] = hi
                            self.store_row(out, request, head, split * OWN + rr, d8,
                                           values, total)  # fmt: skip
        self.rec(request, head, 11, global_timer(), tid == 0)
        if cutlass.const_expr(self.die_sms is not None):
            if tid == 0:  # (no CTA leaves before every CTA of the launch landed)
                wait_landed(args[17], launch_i, 2 * sum(self.die_sms))
        # ---- teardown: the producer frees the TMEM after every tcgen05 operation
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if warp == PRODUCER:
            fence_tmem()
            free_tmem(tmem, TMEM_COLS)

    @cute.jit
    def store_row(self, out, request, head, row, d8, values, total):
        """Output dims 8 d8.. of tile row ``row``: the merged sums over the total."""
        G: cutlass.Constexpr = self.group
        inv = Float32(0)
        if total > Float32(0):
            inv = Float32(1) / total
        words = tuple(
            bf16x2(values[2 * k] * inv, values[2 * k + 1] * inv) for k in range(4)
        )
        o_row = request * self.q + row // G
        o_col = (head * G + row % G) * HEAD + 8 * d8
        store_words(out.iterator + o_row * out.layout.stride[0] + o_col, words)

    @cute.jit
    def issue_s(
        self, t, first, lo, lo_max, q_first, length, ring_base, full, sready, sfree,
        tmem, q_desc, lane, kv_new, newvalid, offset_bits, tr=False,
    ):  # fmt: skip
        """S = Q K^T of stage ``t`` into S buffer t % NS once its K / V landed (edge
        rows fixed first) and the warps loaded stage t - NS's S. Returns the cycles
        waited on the stage's TMA (debug)."""
        s_i = t % self.ring
        tile = ring_base + s_i * SLOT
        p0 = (first + t) * KB
        c0 = self.tick()
        self.trc(tr, t, 0)
        wait(full + 8 * s_i, ((t // self.ring) + ((offset_bits >> s_i) & 1)) & 1,
             scope="cta")  # fmt: skip
        self.trc(tr, t, 1)
        waited = self.tick() - c0
        if (p0 < lo_max) | (p0 + KB > q_first):
            self.fixup(tile, p0, lane, lo, length, kv_new, newvalid)
            fence_async_shared()
            cute.arch.sync_warp()
        b = t % NS
        if t >= NS:  # the warps loaded stage t - NS's S from these columns
            wait(sfree + 8 * b, ((t - NS) // NS) & 1, scope="cta")
        self.trc(tr, t, 2)
        fence_tmem()
        mma_bf16_block(
            tmem + Uint32(64) * Uint32(b), q_desc, desc_sw128(tile, 16),
            Uint32(S_INSTR), Int32(0), sready + 8 * b, Int32(1),
        )  # fmt: skip
        self.trc(tr, t, 3)
        return waited

    @cute.jit
    def tick(self):
        value = Int64(0)
        if cutlass.const_expr(self.debug != 0):
            value = clock64()
        return value

    @cute.jit
    def trc(self, on, t, e: cutlass.Constexpr[int]):
        if cutlass.const_expr(self.trace != 0):
            if on & (t < 48):
                store_u64(Int64(self.trace) + (t * 16 + e) * 8, clock64())

    @cute.jit
    def rec(self, request, head, k: cutlass.Constexpr[int], value, cond):
        if cutlass.const_expr(self.debug != 0):
            if cond:
                cta = request * self.kv_heads + head
                store_u64(Int64(self.debug) + (cta * 16 + k) * 8, value)


def umma_smem(ring, q, page, window, die_split=False):
    """Dynamic shared memory bytes of a GqaWindowUmma CTA (the kernel's
    allocation order and alignments)."""
    off = 0
    pages = -(-(window + q - 1 + 2 * KB) // page) + 1
    sizes = [(ring * SLOT, 1024), (128 * 128, 1024), (q * 256, 16), (4 * q, 16),
             (8 * (ring + 4 * NS + 1), 8), (4 * pages, 16), (8, 16)]  # fmt: skip
    if die_split:
        sizes.append((8, 8))
    for n, align in sizes:
        off = -(-off // align) * align + n
    return off


@cache
def compile_window_umma(
    group, kv_heads, q, page, ring, window, v_scale, max_position, device,
    round_norm=False, debug=0, redo=False, splits=1, fill=4, trace=0, trace_req=0,
    swarp=False, die_sms=None, wide=False,
):  # fmt: skip
    """``GqaWindowUmma`` for any request count (row-dynamic views); ``GqaWindow``'s
    arguments."""
    kernel = GqaWindowUmma(
        group, kv_heads, q, page, ring, window, v_scale, round_norm, debug, redo,
        splits, fill, trace, trace_req, swarp, die_sms, wide,
    )
    i64, i32 = torch.int64, torch.int32
    count = cute.sym_int()
    args = (
        dynamic_fake(2),  # qkv [rows, width]
        fake((count,), None, i64, 8),  # positions
        fake((count,), None, i64, 8),  # slots
        fake((cute.sym_int(),), None, i32, 4),  # seq_lens
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
    if die_sms is not None:  # (the resident grid: SM table, tickets, landings; mask)
        args += (Int64(0), Int64(0), Int64(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
