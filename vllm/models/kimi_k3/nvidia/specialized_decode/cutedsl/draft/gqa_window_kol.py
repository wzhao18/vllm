# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's sliding-window GQA attention on tcgen05 with the keys
on the TMEM lanes (``GqaWindowKol``): ``GqaWindow``'s computation (module notes of
``gqa_window``: q / k norm, RoPE, the query rows' K / V insert, the causal 2048-key
window over the BF16 paged cache) for one (request, KV head) per CTA.

Why keys on lanes: ``GqaWindowUmma`` puts the 24 query rows on the TMEM lanes, so its
S = Q K^T replicates Q into four lane quarters (one M128 N64 SS MMA per 64 keys of
which each quarter keeps 16 keys) and each softmax thread walks a dependent chain of
16 keys per 64-key stage (~490 cycles, the loop's critical path at B32 / B64). Here
S^T = K Q^T (M128 = 128 keys, N32 = the query rows zero padded, K64): every lane of
the four softmax warps holds one key's scores of all rows, so a 128-key stage is one
pass of ROWS exp2 per thread with no replication.

CTA = two softmax warpgroups (even / odd stages, each its own running max, row sums
and O^T accumulator, merged at the end), a PV warp, an S issuer warp and a TMA warp;
128-key stages through a ring of ``ring`` slots of 32 KB (64-key K box 0, K box 1,
V box 0, V box 1; one barrier stage, 8 KB TMA boxes of a 5-D view of the cache,
``kv_map_kol``):
- S^T (FP32, TMEM, NSB buffers of 32 columns) by the S warp as stages land;
- softmax: thread = key; a warpgroup's running max m of each row moves only when a
  stage's scaled score exceeds it by more than 2^THRESH (one bar.red.or vote per
  stage; then the stage's exact row max over the warpgroup, its O^T and row sums
  rescaled); P = exp2(s * scale * log2 e - m) rounded to BF16 goes to shared memory
  as P^T (MN-major, 64B swizzle: 64 B per key);
- O^T (FP32, M64 = head dim x N32 rows; TMEM lanes 0-15 of each lane quarter) +=
  V^T P^T: eight M64 N32 K16 MMAs per stage (V^T MN-major from the ring, P^T
  MN-major) by the PV warp; the TMA warp refills a slot once its stage's PV
  completed;
- the row sums are per-thread FP32 partials of the FP32 p, summed over the keys at the
  end; the output is O^T / l rounded once to BF16.

Rounding: S in FP32, P = exp2(s * scale log2 e - m) rounded to BF16, O in FP32, the
row sums of the FP32 p, the output rounded once (as ``GqaWindow`` / ``GqaWindowUmma``);
only the order of the FP32 sums and rescales differs.
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
    fence_tmem,
    fma,
    free_tmem,
    launch,
    load_words,
    named_barrier,
    shared_address,
    store_words,
    wait,
)
from .gqa_window import (
    HEAD,
    LOG2E,
    NEG_INF,
    arm,
    arrive,
    atomic_or_shared,
    ex2,
    fence_async_shared,
    float_word,
    global_timer,
    init_barrier,
    load_i32,
    load_i64,
    norm_rope8,
    store_shared_u32,
    swz,
    tree,
)
from .gqa_window_umma import alloc_tmem, clock64, desc_sw128, fmax, store_u64

KB = 64  # keys per TMA box
BOX = KB * 128  # bytes of one box's K (or V) image
KS = 2 * KB  # keys per stage
KT = 2 * BOX  # one stage's K (or V) tile: boxes 0 and 1, 16 KB
PTB = KS * 64  # one P^T buffer: 128 keys x 32 rows BF16
NSB = 4  # S^T TMEM buffers (32 FP32 columns each)
NPB = 4  # P^T shared buffers
# warps 0-3 / 4-7: softmax warpgroups 0 / 1 (even / odd stages; TMEM lane quarter =
# warp % 4), 8: PV issuer, 9: S^T issuer, 10 / 11: the K / V TMA warps
THREADS = 384
PVW = 8
SWARP = 9
KTMA = 10  # K ring: refills a K tile once its stage's S^T completed
VTMA = 11  # V ring: refills a V tile once its stage's PV completed
PRODUCER = KTMA  # (TMEM allocation)
O_COL = 32 * NSB  # O^T of warpgroup g at O_COL + 32 g
KRING = NSB  # K tiles in flight (<= NSB: a K tile frees with its S^T)
TMEM_COLS = 256
THRESH = 8.0  # log2 headroom of the lazy running max
POS_INF = float("inf")
BAR_GROW = 1  # named barrier ids of softmax warpgroup g: + 2 g
BAR_SM = 2
BAR_ALL = 5  # both softmax warpgroups
# TMEM lanes of the M64 O^T accumulator (GqaWindowKol.o_layout)
O_LAYOUT = 0
# FP32 D, BF16 A (K tile, K-major) and B (Q^T, K-major), N = 32, M = 128 (S^T = K Q^T)
S_INSTR = 1 << 4 | 1 << 7 | 1 << 10 | (32 >> 3) << 17 | (128 >> 4) << 24
# FP32 D, BF16 A (V^T, MN-major) and B (P^T, MN-major), N = 32, M = 64 (O^T += V^T P^T)
PV_INSTR = (
    1 << 4 | 1 << 7 | 1 << 10 | 1 << 15 | 1 << 16 | (32 >> 3) << 17 | (64 >> 4) << 24
)


@asm()
def mma_s4(d, a, b, instruction, done, free):
    """Warp-uniform issue of a stage's S^T = K Q^T: four K16 BF16 MMAs (K-major 128B
    swizzle both, +32 B per slice); commit ``done`` and ``free``."""
    mma = "tcgen05.mma.cta_group::1.kind::f16"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt; .reg .b64 a1,a2,a3,b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,0,0; "
    code += "add.s64 a1,$1,2; add.s64 a2,$1,4; add.s64 a3,$1,6; "
    code += "add.s64 b1,$2,2; add.s64 b2,$2,4; add.s64 b3,$2,6; "
    code += f"@e {mma} [$0],$1,$2,$3,p0; @e {mma} [$0],a1,b1,$3,pt; "
    code += f"@e {mma} [$0],a2,b2,$3,pt; @e {mma} [$0],a3,b3,$3,pt; "
    code += f"@e {commit} [$4]; @e {commit} [$5]; }}"
    operands: tuple = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    return code, "r,l,l,r,r,r", operands + (Uint32(done), Uint32(free))


@asm()
def mma_pv8(d, a, b, instruction, accumulate, done, free):
    """Warp-uniform issue of a stage's PV: eight K16 BF16 MMAs (V^T MN-major 128B
    swizzle, +2048 B per slice; P^T MN-major 64B swizzle, +1024 B per slice), the
    first accumulating iff ``accumulate``; commit ``done`` and ``free``."""
    mma = "tcgen05.mma.cta_group::1.kind::f16"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt; .reg .b64 a<8>, b<8>; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,$4,0; "
    for j in range(1, 8):
        code += f"add.s64 a{j},$1,{128 * j}; add.s64 b{j},$2,{64 * j}; "
    code += f"@e {mma} [$0],$1,$2,$3,p0; "
    for j in range(1, 8):
        code += f"@e {mma} [$0],a{j},b{j},$3,pt; "
    code += f"@e {commit} [$5]; @e {commit} [$6]; }}"
    operands: tuple = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    operands += (Int32(accumulate), Uint32(done), Uint32(free))
    return code, "r,l,l,r,r,r,r", operands


@asm()
def tma_box(dst, tensor_map, barrier, token, kv, head, block):
    """One {64 values, 64 tokens} K (kv 0) or V (kv 1) box of the 5-D cache map."""
    code = "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes [$0], [$1, {$2, $3, $4, $5, $6}], [$7];"
    operands = (Uint32(dst), Int64(tensor_map), Int32(0), Int32(token), Int32(kv))
    operands += (Int32(head), Int32(block), Uint32(barrier))
    return code, "r,l,r,r,r,r,r,r,~{memory}", operands


@asm()
def prefetch_box(tensor_map, token, kv, head, block):
    """L2 prefetch of one {64 values, 64 tokens} K (kv 0) or V (kv 1) box."""
    code = "cp.async.bulk.prefetch.tensor.5d.L2.global.tile [$0, {$1, $2, $3, $4, $5}];"
    operands = (Int64(tensor_map), Int32(0), Int32(token), Int32(kv), Int32(head))
    return code, "l,r,r,r,r,r,~{memory}", operands + (Int32(block),)


@asm()
def ld32(address):
    """32 FP32 TMEM columns of this lane, waited."""
    names = ",".join(f"${i}" for i in range(32))
    code = f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{names}}},[$32]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * 32 + ["r", "~{memory}"]), (Uint32(address),), (
        (Float32,) * 32
    )


@asm()
def st32(address, values):
    names = ",".join(f"${i + 1}" for i in range(32))
    code = f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",f" * 32 + ",~{memory}", (Uint32(address), *values)


@asm(Int32)
def bar_red_or(barrier_id, count, flag):
    """bar.red.or over the threads of a named barrier: 1 iff any passed flag != 0."""
    code = "{ .reg .pred p, q; setp.ne.u32 p, $3, 0; bar.red.or.pred q, $1, $2, p; "
    code += "selp.u32 $0, 1, 0, q; }"
    return code, "=r,r,r,r", (Uint32(barrier_id), Uint32(count), Uint32(flag))


@asm(Float32)
def shfl_xor(value, offset):
    code = "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;"
    return code, "=f,f,r", (Float32(value), Int32(offset))


@cute.jit
def desc_sw64(address):
    """64B-swizzled MN-major shared memory descriptor of a 512 B aligned image of
    64 B rows (SBO 512 B: 8-row atoms; one atom along MN)."""
    value = Uint64((Uint32(address) & 262143) >> 4) | Uint64(1) << 46
    value |= Uint64(512 >> 4) << 16 | Uint64(512 >> 4) << 32 | Uint64(4) << 61
    return value


def kv_map_kol(cache, kv_heads, head_stride):
    """The TMA map spec (``common.host.tensor_maps``) of a [blocks, heads, page,
    K | V] BF16 cache view as (64 values, page tokens, K | V, heads, blocks) with
    {64, 64, 1} boxes: one 64-key K or V image (128 B swizzle)."""
    import cuda.bindings.driver as cuda

    page, blocks = cache.shape[2], cache.shape[0]
    return (
        cache.data_ptr(),
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        (HEAD, page, 2, kv_heads, blocks),
        (4 * HEAD, 2 * HEAD, head_stride * 2, cache.stride(0) * 2),
        (HEAD, KB, 1, 1, 1),
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    )


class GqaWindowKol:
    """One drafter layer's fused attention, keys on the TMEM lanes (module notes):
    ``group`` query heads per KV head (``group * q`` <= 32 rows), ``kv_heads`` per
    rank, the cache's ``page`` tokens per block (a multiple of 64), ``ring`` 128-key
    stages in flight, ``window`` keys. Same arguments as ``GqaWindow`` (+ the
    {64, 64, 1}-box map ``kv_map_kol`` as the tensor map)."""

    def __init__(self, group, kv_heads, q, page, ring, window, v_scale=None,
                 round_norm=False, debug=0, redo=False, fill=1, o_layout=0,
                 skeleton=False, trace=0, trace_req=0, l2ahead=0):  # fmt: skip
        assert page % KB == 0 and window >= 1 and 1 <= q <= 8
        assert group * q <= 32, (group, q)
        assert ring >= 2
        # (ring: the V ring's 16 KB tiles; the K ring: ``KRING``)
        self.group, self.kv_heads, self.q, self.page = group, kv_heads, q, page
        self.ring, self.window, self.v_scale = ring, window, v_scale
        self.kring = KRING
        self.round_norm = round_norm
        self.rows = group * q
        self.chunks = -(-self.rows // 8)  # 16 B P^T chunks per key
        # block-table entries of a CTA's stages (+ one 64-key box of slack)
        self.pages = -(-(window + q - 1 + 2 * KS) // page) + 1
        assert self.pages <= THREADS
        # (development only) ``fill``: stages issued right after the PDL wait; the rest
        # after the Q prep (the qkv loads otherwise queue behind the first fill)
        self.fill = min(fill, ring, KRING)
        self.debug_redo = redo  # (tests: the redo path with unmoved metadata)
        # ``debug`` (development only): int64 [CTAs, 16] phase stamps (``rec``)
        self.debug = debug
        # O^T accumulator (M64) lane of head dim d: 0 = lanes 0-15 of each TMEM lane
        # quarter (dim 16 w + lane), 1 = lanes 0-63 (dim 32 w + lane, warps 0-1)
        self.o_layout = o_layout
        # ``skeleton`` (development only): the producer streams the window through the
        # ring with no MMA / softmax (the data-movement floor of this form)
        self.skeleton = skeleton
        self.threads = THREADS
        # ``l2ahead``: each TMA warp prefetches its tiles this many stages beyond its
        # ring into L2 (the ring's refills then wait on L2, not HBM)
        self.l2ahead = l2ahead
        # trace (development only): int64 [48, 16] per-stage clock64 events of
        # the CTA (request trace_req, KV head 1) (trc)
        self.trace, self.trace_req = trace, trace_req

    @cute.jit
    def __call__(self, *args):
        out = args[8]
        requests = out.shape[0] // self.q
        grid = (1, self.kv_heads, requests)
        TH: cutlass.Constexpr = self.threads
        launch(self.kernel, args, grid, TH, max_number_threads=[TH, 1, 1])

    @cute.jit
    def issue(self, kv_map, bar, tile, kv: cutlass.Constexpr[int], head, blocks,
              page_lo, box0, last_box):  # fmt: skip
        """K (kv 0) or V (kv 1) tile of the stage at 64-key box ``box0``: boxes box0
        and box0 + 1 (if <= ``last_box``) under one barrier."""
        N: cutlass.Constexpr = self.page
        two = box0 + 1 <= last_box
        nbytes = Int32(BOX)
        if two:
            nbytes = Int32(2 * BOX)
        arm(bar, nbytes)
        token = box0 * KB
        tma_box(tile, kv_map, bar, token % N, kv, head, blocks[token // N - page_lo])
        if two:
            token1 = token + KB
            blk1 = blocks[token1 // N - page_lo]
            tma_box(tile + BOX, kv_map, bar, token1 % N, kv, head, blk1)

    @cute.jit
    def prefetch(self, kv_map, kv: cutlass.Constexpr[int], head, blocks, page_lo, box0,
                 last_box):  # fmt: skip
        """L2 prefetch of the K (kv 0) or V (kv 1) tile at 64-key box ``box0``."""
        N: cutlass.Constexpr = self.page
        token = box0 * KB
        prefetch_box(kv_map, token % N, kv, head, blocks[token // N - page_lo])
        if box0 + 1 <= last_box:
            token1 = token + KB
            prefetch_box(kv_map, token1 % N, kv, head, blocks[token1 // N - page_lo])

    @cute.jit
    def fixup(self, tile, kv: cutlass.Constexpr[int], p0, lane, lo, length, kv_new,
              newvalid):  # fmt: skip
        """Rows lane + 32 j of a stage's K (kv 0) or V (kv 1) tile: zero outside [lo,
        length); the computed K / V of a query row whose slot was valid."""
        q_first = length - self.q
        zero = (Uint32(0),) * 4
        for j in cutlass.range_constexpr(4):
            row = lane + 32 * j
            img = tile + (row // KB) * BOX
            rr = row % KB
            position = p0 + row
            if (position < lo) | (position >= length):
                for ch in cutlass.range_constexpr(8):
                    store_words(img + swz(rr, ch), zero, shared=True)
            elif position >= q_first:
                i = position - q_first
                if newvalid[i] != 0:
                    for ch in cutlass.range_constexpr(8):
                        src = kv_new + i * 256 + 128 * kv + 16 * ch
                        store_words(img + swz(rr, ch),
                                    load_words(src, count=4, shared=True),
                                    shared=True)  # fmt: skip

    @cute.jit
    def stages(self, length):
        """(window start, first 64-key box, last box, 128-key stage count)."""
        lo = max(length - self.q - (self.window - 1), Int32(0))
        first = Int32(0)
        last = Int32(-1)
        if length > 0:
            first = lo // KB
            last = (length - 1) // KB
        count = (last - first + 2) // 2
        return lo, first, last, count

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
        RK: cutlass.Constexpr = self.kring
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        CH: cutlass.Constexpr = self.chunks
        TH: cutlass.Constexpr = self.threads
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        HKV: cutlass.Constexpr = self.kv_heads
        c = Float32(LOG2E / (HEAD**0.5))

        _, head, request = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        group8 = lane & 7
        upper = group8 >= 4
        table_last = Int32(table.shape[1] - 1)
        blocks_max = args[13] - 1

        smem = cutlass.utils.SmemAllocator()
        kring = alloc(smem, cutlass.Uint8, RK * KT, align=1024)
        vring = alloc(smem, cutlass.Uint8, R * KT, align=1024)
        qtile = alloc(smem, cutlass.Uint8, 32 * 128, align=1024)
        ptile = alloc(smem, cutlass.Uint8, NPB * PTB, align=1024)
        newkv = alloc(smem, cutlass.Uint8, Q * 256, align=16)
        newvalid = alloc(smem, Int32, Q, align=16)
        bars = alloc(smem, cutlass.Int64, 2 * RK + 2 * R + 2 * NSB + 2 * NPB, align=8)
        blocks = alloc(smem, Int32, NP, align=16)
        stats = alloc(smem, Float32, 9 * 32, align=16)
        tslot = alloc(smem, Int32, 2, align=16)  # [1]: the redo flag
        table_row = table.iterator + request * table.layout.stride[0]
        tr_cta = (request == self.trace_req) & (head == 1)
        k_base = Int32(shared_address(kring.iterator))
        v_base = Int32(shared_address(vring.iterator))
        q_base = Int32(shared_address(qtile.iterator))
        pt_base = Int32(shared_address(ptile.iterator))
        kv_new = Int32(shared_address(newkv.iterator))
        redo_flag = Int32(shared_address(tslot.iterator)) + 4
        kfull = Int32(shared_address(bars.iterator))  # [RK]: K tile landed
        kfree = kfull + 8 * RK  # [RK]: its S^T completed
        vfull = kfree + 8 * RK  # [R]: V tile landed
        vfree = vfull + 8 * R  # [R]: its PV completed
        sready = vfree + 8 * R  # [NSB]: S^T landed
        sfree = sready + 8 * NSB  # [NSB]: the softmax warps loaded S^T
        pready = sfree + 8 * NSB  # [NPB]: P^T written
        pvdone = pready + 8 * NPB  # [NPB]: PV completed (P^T buffer + ring slot free)

        if tid == 0:
            for i in cutlass.range_constexpr(RK):
                init_barrier(kfull + 8 * i, 1)
                init_barrier(kfree + 8 * i, 1)
            for i in cutlass.range_constexpr(R):
                init_barrier(vfull + 8 * i, 1)
                init_barrier(vfree + 8 * i, 1)
            for b in cutlass.range_constexpr(NSB):
                init_barrier(sready + 8 * b, 1)
                init_barrier(sfree + 8 * b, 4)
            for b in cutlass.range_constexpr(NPB):
                init_barrier(pready + 8 * b, 4)
                init_barrier(pvdone + 8 * b, 1)
        # the Q^T tile's padding rows ROWS..31, the P^T buffers' padding chunks
        PADQ: cutlass.Constexpr = (32 - ROWS) * 8
        for it in cutlass.range_constexpr(-(-PADQ // TH)):
            t = it * TH + tid
            if t < PADQ:
                store_words(q_base + swz(ROWS + t // 8, t % 8), (Uint32(0),) * 4,
                            shared=True)  # fmt: skip
        if cutlass.const_expr(CH < 4):
            if tid < 128:
                key_z = tid
                for b in cutlass.range_constexpr(NPB):
                    for ch in cutlass.range_constexpr(CH, 4):
                        dst = pt_base + b * PTB + key_z * 64
                        dst = dst + ((ch ^ ((key_z >> 1) & 3)) << 4)
                        store_words(dst, (Uint32(0),) * 4, shared=True)
        q_w = load_words(qw.iterator + 8 * group8, count=4)
        k_w = load_words(kw.iterator + 8 * group8, count=4)
        # ---- before the PDL wait: the metadata as it stands decides only addresses
        length0 = load_i32(seq_lens.iterator + request)
        _, first0, last0, count0 = self.stages(length0)
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
        spec_k = min(count0, Int32(RK))
        spec_v = min(count0, Int32(R))
        FB: cutlass.Constexpr = self.fill
        self.rec(request, head, 0, global_timer(), tid == 0)
        cute.arch.griddepcontrol_wait()
        if warp == PRODUCER:
            alloc_tmem(Int32(shared_address(tslot.iterator)), TMEM_COLS)
        self.rec(request, head, 1, global_timer(), tid == 0)

        # ---- after the wait: the first stages' loads at the pre-wait addresses, then
        # the metadata again (confirming them) and the prep math
        if warp == KTMA:
            if lane < min(spec_k, Int32(FB)):
                self.issue(kv_map, kfull + 8 * lane, k_base + lane * KT, 0, head,
                           blocks, page_lo0, first0 + 2 * lane, last0)  # fmt: skip
        if warp == VTMA:
            if lane < min(spec_v, Int32(FB)):
                self.issue(kv_map, vfull + 8 * lane, v_base + lane * KT, 1, head,
                           blocks, page_lo0, first0 + 2 * lane, last0)  # fmt: skip
        length = load_i32(seq_lens.iterator + request)
        lo, first, last, count = self.stages(length)
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
                store_words(q_base + swz(t // 8, group8), tuple(roped), shared=True)
            elif (t < TASKS) & (length > 0):
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
        if warp == KTMA:
            if (lane >= FB) & (lane < spec_k):
                self.issue(kv_map, kfull + 8 * lane, k_base + lane * KT, 0, head,
                           blocks, page_lo0, first0 + 2 * lane, last0)  # fmt: skip
        if warp == VTMA:
            if (lane >= FB) & (lane < spec_v):
                self.issue(kv_map, vfull + 8 * lane, v_base + lane * KT, 1, head,
                           blocks, page_lo0, first0 + 2 * lane, last0)  # fmt: skip
        D: cutlass.Constexpr = self.l2ahead
        if cutlass.const_expr(D > 0):
            if warp == KTMA:
                if (lane < D) & (RK + lane < count0):
                    self.prefetch(kv_map, 0, head, blocks, page_lo0,
                                  first0 + 2 * (RK + lane), last0)  # fmt: skip
            if warp == VTMA:
                if (lane < D) & (R + lane < count0):
                    self.prefetch(kv_map, 1, head, blocks, page_lo0,
                                  first0 + 2 * (R + lane), last0)  # fmt: skip
        fence_async_shared()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Uint32(tslot[0])

        # ---- moved metadata: drain the speculated loads, reissue from the confirmed
        # metadata; those tiles' phases then start at 1 (``koff`` / ``voff``)
        koff = Int32(0)
        voff = Int32(0)
        if load_words(redo_flag, count=1, shared=True)[0] != 0:
            if warp == KTMA:
                if lane < spec_k:
                    wait(kfull + 8 * lane, 0, scope="cta")
            if warp == VTMA:
                if lane < spec_v:
                    wait(vfull + 8 * lane, 0, scope="cta")
            if tid < NP:
                entry = load_i32(table_row + min(page_lo + tid, table_last))
                blocks[tid] = min(max(entry, Int32(0)), blocks_max)
            cute.arch.sync_threads()
            if warp == KTMA:
                if lane < min(count, Int32(RK)):
                    self.issue(kv_map, kfull + 8 * lane, k_base + lane * KT, 0, head,
                               blocks, page_lo, first + 2 * lane, last)  # fmt: skip
            if warp == VTMA:
                if lane < min(count, Int32(R)):
                    self.issue(kv_map, vfull + 8 * lane, v_base + lane * KT, 1, head,
                               blocks, page_lo, first + 2 * lane, last)  # fmt: skip
            koff = (Int32(1) << spec_k) - 1
            voff = (Int32(1) << spec_v) - 1
        lo_max = max(length - W, Int32(0))
        self.rec(request, head, 2, global_timer(), tid == 0)
        self.rec(request, head, 12, Int64(count), tid == 0)

        if cutlass.const_expr(self.skeleton):
            # ---- data-movement skeleton: each TMA warp consumes and refills its ring
            if (warp == KTMA) | (warp == VTMA):
                RR = Int32(RK)
                bars_kv = kfull
                base_kv = k_base
                off_kv = koff
                if warp == VTMA:
                    RR = Int32(R)
                    bars_kv = vfull
                    base_kv = v_base
                    off_kv = voff
                for t in cutlass.range(count, unroll=1):
                    s_i = t % RR
                    ph = ((t // RR) + ((off_kv >> s_i) & 1)) & 1
                    wait(bars_kv + 8 * s_i, ph, scope="cta")
                    if (t + RR < count) & (lane == 0):
                        if warp == KTMA:
                            self.issue(kv_map, bars_kv + 8 * s_i, base_kv + s_i * KT, 0,
                                       head, blocks, page_lo, first + 2 * (t + RR),
                                       last)  # fmt: skip
                        else:
                            self.issue(kv_map, bars_kv + 8 * s_i, base_kv + s_i * KT, 1,
                                       head, blocks, page_lo, first + 2 * (t + RR),
                                       last)  # fmt: skip
                self.rec(request, head, 3, global_timer(), (lane == 0) & (warp == VTMA))
        else:
            if warp == PVW:
                # ---- PV of each stage into its warpgroup's O^T once its P^T is ready
                # and its V tile landed (edge rows fixed first)
                loop0 = self.tick()
                a_pready = Int64(0)
                trw = tr_cta & (lane == 0)
                for t in cutlass.range(count, unroll=1):
                    b = t % NPB
                    c0 = self.tick()
                    wait(pready + 8 * b, (t // NPB) & 1, scope="cta")
                    a_pready = a_pready + self.tick() - c0
                    self.trc(trw, t, 8)
                    v_i = t % R
                    tile_v = v_base + v_i * KT
                    wait(vfull + 8 * v_i, ((t // R) + ((voff >> v_i) & 1)) & 1,
                         scope="cta")  # fmt: skip
                    p0 = (first + 2 * t) * KB
                    if (p0 < lo_max) | (p0 + KS > q_first) | (p0 < lo) | (
                        p0 + KS > length
                    ):
                        self.fixup(tile_v, 1, p0, lane, lo, length, kv_new, newvalid)
                        fence_async_shared()
                        cute.arch.sync_warp()
                    fence_tmem()
                    mma_pv8(
                        tmem + Uint32(O_COL) + Uint32(32) * Uint32(t % 2),
                        desc_sw128(tile_v, BOX), desc_sw64(pt_base + b * PTB),
                        Uint32(PV_INSTR), Int32(t >= 2), pvdone + 8 * b,
                        vfree + 8 * v_i,
                    )  # fmt: skip
                    self.trc(trw, t, 9)
                self.rec(request, head, 3, global_timer(), lane == 0)
                self.rec(request, head, 6, a_pready, lane == 0)
                self.rec(request, head, 15, self.tick() - loop0, lane == 0)
            elif warp == KTMA:
                # ---- K ring: tile t % RK takes stage t + RK once S^T(t) completed
                trf = tr_cta & (lane == 0)
                for t in cutlass.range(count, unroll=1):
                    if t + RK < count:
                        wait(kfree + 8 * (t % RK), (t // RK) & 1, scope="cta")
                        if lane == 0:
                            self.issue(kv_map, kfull + 8 * (t % RK),
                                       k_base + (t % RK) * KT, 0, head, blocks,
                                       page_lo, first + 2 * (t + RK), last)  # fmt: skip
                            if cutlass.const_expr(D > 0):
                                if t + RK + D < count:
                                    self.prefetch(kv_map, 0, head, blocks, page_lo,
                                                  first + 2 * (t + RK + D), last)
                        self.trc(trf, t, 10)
            elif warp == VTMA:
                # ---- V ring: tile t % R takes stage t + R once PV(t) completed
                for t in cutlass.range(count, unroll=1):
                    if t + R < count:
                        wait(vfree + 8 * (t % R), (t // R) & 1, scope="cta")
                        if lane == 0:
                            self.issue(kv_map, vfull + 8 * (t % R),
                                       v_base + (t % R) * KT, 1, head, blocks,
                                       page_lo, first + 2 * (t + R), last)  # fmt: skip
                            if cutlass.const_expr(D > 0):
                                if t + R + D < count:
                                    self.prefetch(kv_map, 1, head, blocks, page_lo,
                                                  first + 2 * (t + R + D), last)
            elif warp == SWARP:
                # ---- S^T = K Q^T of stage t once its K tile landed (edge rows fixed
                # first) and S^T buffer t % NSB is free
                q_desc = desc_sw128(q_base, 16)
                a_full = Int64(0)
                trs = tr_cta & (lane == 0)
                for t in cutlass.range(count, unroll=1):
                    k_i = t % RK
                    tile_k = k_base + k_i * KT
                    p0 = (first + 2 * t) * KB
                    c0 = self.tick()
                    wait(kfull + 8 * k_i, ((t // RK) + ((koff >> k_i) & 1)) & 1,
                         scope="cta")  # fmt: skip
                    a_full = a_full + self.tick() - c0
                    self.trc(trs, t, 6)
                    if (p0 < lo_max) | (p0 + KS > q_first) | (p0 < lo) | (
                        p0 + KS > length
                    ):
                        self.fixup(tile_k, 0, p0, lane, lo, length, kv_new, newvalid)
                        fence_async_shared()
                        cute.arch.sync_warp()
                    b = t % NSB
                    if t >= NSB:
                        wait(sfree + 8 * b, ((t - NSB) // NSB) & 1, scope="cta")
                    fence_tmem()
                    mma_s4(
                        tmem + Uint32(32) * Uint32(b), desc_sw128(tile_k, 16), q_desc,
                        Uint32(S_INSTR), sready + 8 * b, kfree + 8 * k_i,
                    )  # fmt: skip
                    self.trc(trs, t, 7)
                self.rec(request, head, 4, a_full, lane == 0)
            elif warp < 4:
                self.softmax(
                    0, tmem, warp, lane, count, first, q_first, lo_max, c, sready,
                    sfree, pready, pvdone, pt_base, stats, request, head, out,
                    tr_cta & (warp == 0) & (lane == 0),
                )  # fmt: skip
            elif warp < 8:
                self.softmax(
                    1, tmem, warp, lane, count, first, q_first, lo_max, c, sready,
                    sfree, pready, pvdone, pt_base, stats, request, head, out,
                    tr_cta & (warp == 4) & (lane == 0),
                )  # fmt: skip

        # ---- epilogue (softmax warps): row sums, O^T / l, BF16 rows out
        self.rec(request, head, 11, global_timer(), tid == 0)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if warp == PRODUCER:
            fence_tmem()
            free_tmem(tmem, TMEM_COLS)

    @cute.jit
    def softmax(
        self, g: cutlass.Constexpr[int], tmem, warp, lane, count, first, q_first, lo_max,
        c, sready, sfree, pready, pvdone, pt_base, stats, request, head, out, trs,
    ):  # fmt: skip
        """Softmax warpgroup ``g``: the stages t = g, g + 2, ... with its own running
        max m, row sums and O^T accumulator (O_COL + 32 g); then warpgroup 0 merges
        both and stores the rows."""
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        CH: cutlass.Constexpr = self.chunks
        BG: cutlass.Constexpr = BAR_GROW + 2 * g
        BS: cutlass.Constexpr = BAR_SM + 2 * g
        SB: cutlass.Constexpr = 128 * g  # this warpgroup's stats
        q4 = warp % 4
        lanes = Uint32(32 * q4) << 16
        o_mine = tmem + lanes + Uint32(O_COL + 32 * g)
        key_l = 32 * q4 + lane  # this thread's key within a stage
        sw = (key_l >> 1) & 3
        bias = cute.make_rmem_tensor(ROWS, Float32)  # -m (+inf: no reference yet)
        lsum = cute.make_rmem_tensor(ROWS, Float32)
        for r in cutlass.range_constexpr(ROWS):
            bias[r] = Float32(POS_INF)
            lsum[r] = Float32(0)
        a_sr = Int64(0)
        a_res = Int64(0)
        n_g = (count - g + 1) // 2
        for i in cutlass.range(n_g, unroll=1):
            t = 2 * i + g
            b = t % NSB
            c0 = self.tick()
            self.trc(trs, t, 0)
            wait(sready + 8 * b, (t // NSB) & 1, scope="cta")
            a_sr = a_sr + self.tick() - c0
            self.trc(trs, t, 1)
            fence_tmem()
            raw = ld32(tmem + lanes + Uint32(32) * Uint32(b))
            fence_tmem(before=True)
            cute.arch.sync_warp()
            if lane == 0:
                arrive(sfree + 8 * b)
            self.trc(trs, t, 2)
            p0 = (first + 2 * t) * KB
            edge = (p0 < lo_max) | (p0 + KS > q_first)
            ok = cute.make_rmem_tensor(Q, Int32)
            for k in cutlass.range_constexpr(Q):
                ok[k] = Int32(1)
            y = cute.make_rmem_tensor(ROWS, Float32)
            for r in cutlass.range_constexpr(ROWS):
                y[r] = fma(raw[r], c, bias[r])
            if edge:
                key = p0 + key_l
                for k in cutlass.range_constexpr(Q):
                    hi = q_first + k
                    lo_k = max(hi - (W - 1), Int32(0))
                    ok[k] = Int32((key >= lo_k) & (key <= hi))
                for r in cutlass.range_constexpr(ROWS):
                    if ok[r // G] == 0:
                        y[r] = Float32(NEG_INF)
            top = tree(fmax, [y[r] for r in range(ROWS)])
            grow = Int32(top > Float32(THRESH))
            self.trc(trs, t, 3)
            anyg = bar_red_or(Uint32(BG), Uint32(128), grow)
            self.trc(trs, t, 4)
            if anyg != 0:
                # ---- the reference moves: every row's exact stage max over the
                # warpgroup; rescale the row sums and (after PV(t - 2)) O^T
                a_res = a_res + 1
                for r in cutlass.range_constexpr(ROWS):
                    mv = raw[r] * c
                    if ok[r // G] == 0:
                        mv = Float32(NEG_INF)
                    for k in cutlass.range_constexpr(5):
                        mv = fmax(mv, shfl_xor(mv, Int32(16 >> k)))
                    if lane == 0:
                        stats[SB + q4 * 32 + r] = mv
                named_barrier(BS, 128)
                alpha = cute.make_rmem_tensor(ROWS, Float32)
                for r in cutlass.range_constexpr(ROWS):
                    mx = fmax(
                        fmax(stats[SB + r], stats[SB + 32 + r]),
                        fmax(stats[SB + 64 + r], stats[SB + 96 + r]),
                    )  # fmt: skip
                    m_old = -bias[r]
                    m_new = fmax(m_old, mx)
                    alpha[r] = Float32(1)
                    if m_new > Float32(NEG_INF):
                        alpha[r] = ex2(m_old - m_new)
                        bias[r] = -m_new
                    lsum[r] = lsum[r] * alpha[r]
                    y[r] = fma(raw[r], c, bias[r])
                    if ok[r // G] == 0:
                        y[r] = Float32(NEG_INF)
                if t >= 2:
                    wait(pvdone + 8 * ((t - 2) % NPB), ((t - 2) // NPB) & 1,
                         scope="cta")  # fmt: skip
                    fence_tmem()
                    ox = ld32(o_mine)
                    scaled = [ox[r] * alpha[r] if r < ROWS else ox[r] for r in range(32)]
                    st32(o_mine, tuple(scaled))
                named_barrier(BS, 128)  # (stats reused by the next move)
            ps = [ex2(y[r]) for r in range(ROWS)]
            for r in cutlass.range_constexpr(ROWS):
                lsum[r] = lsum[r] + ps[r]
            pwords = [
                bf16x2(ps[2 * k], ps[2 * k + 1] if 2 * k + 1 < ROWS else Float32(0))
                for k in range(-(-ROWS // 2))
            ]  # fmt: skip
            pwords = pwords + [Uint32(0)] * (4 * CH - (ROWS + 1) // 2)
            bp = t % NPB
            self.trc(trs, t, 5)
            if t >= NPB:  # stage t - NPB's PV read this P^T buffer
                wait(pvdone + 8 * bp, ((t - NPB) // NPB) & 1, scope="cta")
            self.trc(trs, t, 11)
            dst = pt_base + bp * PTB + key_l * 64
            for ch in cutlass.range_constexpr(CH):
                store_words(dst + ((ch ^ sw) << 4), tuple(pwords[4 * ch : 4 * ch + 4]),
                            shared=True)  # fmt: skip
            self.trc(trs, t, 12)
            fence_async_shared()
            fence_tmem(before=True)
            cute.arch.sync_warp()
            if lane == 0:
                arrive(pready + 8 * bp)
            self.trc(trs, t, 13)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(g == 0):
            self.rec(request, head, 8, a_sr, (q4 == 0) & (lane == 0))
            self.rec(request, head, 9, a_res, (q4 == 0) & (lane == 0))
            self.rec(request, head, 10, global_timer(), (q4 == 0) & (lane == 0))
        # ---- the row sums over the warp's keys, then per warp into stats; m of
        # warpgroup 1 (uniform over its threads) behind them
        for r in cutlass.range_constexpr(ROWS):
            lv = lsum[r]
            for k in cutlass.range_constexpr(5):
                lv = lv + shfl_xor(lv, Int32(16 >> k))
            lsum[r] = lv
        named_barrier(BS, 128)
        if lane == 0:
            for r in cutlass.range_constexpr(ROWS):
                stats[SB + q4 * 32 + r] = lsum[r]
                if cutlass.const_expr(g == 1):
                    if q4 == 0:
                        stats[256 + r] = -bias[r]
        named_barrier(BAR_ALL, 256)
        if cutlass.const_expr(g == 0):
            o0 = cute.make_rmem_tensor(32, Float32)
            o1 = cute.make_rmem_tensor(32, Float32)
            for e in cutlass.range_constexpr(32):
                o0[e] = Float32(0)
                o1[e] = Float32(0)
            if count > 0:
                last = count - 1
                wait(pvdone + 8 * (last % NPB), (last // NPB) & 1, scope="cta")
                fence_tmem()
                l0 = ld32(tmem + lanes + Uint32(O_COL))
                for e in cutlass.range_constexpr(32):
                    o0[e] = l0[e]
            if count > 1:
                l1 = ld32(tmem + lanes + Uint32(O_COL + 32))
                for e in cutlass.range_constexpr(32):
                    o1[e] = l1[e]
            # O rows (FP32 [ROWS, 64]) in the P^T buffers (every PV completed)
            obuf = pt_base
            if lane < 16:
                dim = Int32(16) * q4 + lane
                for r in cutlass.range_constexpr(ROWS):
                    m0 = -bias[r]
                    m1 = stats[256 + r]
                    mtop = fmax(m0, m1)
                    w0 = Float32(0)
                    w1 = Float32(0)
                    if m0 > Float32(NEG_INF):
                        w0 = ex2(m0 - mtop)
                    if m1 > Float32(NEG_INF):
                        w1 = ex2(m1 - mtop)
                    s0 = stats[r] + stats[32 + r] + stats[64 + r] + stats[96 + r]
                    s1 = stats[128 + r] + stats[160 + r] + stats[192 + r] + stats[224 + r]
                    total = fma(w0, s0, w1 * s1)
                    inv = Float32(0)
                    if total > Float32(0):
                        inv = Float32(1) / total
                    value = fma(w0, o0[r], w1 * o1[r]) * inv
                    store_words(obuf + (r * 64 + dim) * 4, (float_word(value),),
                                shared=True)  # fmt: skip
            named_barrier(BS, 128)
            tid = q4 * 32 + lane
            for it in cutlass.range_constexpr(-(-(ROWS * 8) // 128)):
                task = it * 128 + tid
                if task < ROWS * 8:
                    rr, d8 = task // 8, task % 8
                    lo4 = load_words(obuf + (rr * 64 + 8 * d8) * 4, count=4,
                                     shared=True)  # fmt: skip
                    hi4 = load_words(obuf + (rr * 64 + 8 * d8 + 4) * 4, count=4,
                                     shared=True)  # fmt: skip
                    vals = [bits_float(lo4[e]) for e in range(4)]
                    vals += [bits_float(hi4[e]) for e in range(4)]
                    owords = tuple(
                        bf16x2(vals[2 * k], vals[2 * k + 1]) for k in range(4)
                    )  # fmt: skip
                    o_row = request * Q + rr // G
                    o_col = (head * G + rr % G) * HEAD + 8 * d8
                    store_words(out.iterator + o_row * out.layout.stride[0] + o_col,
                                owords)  # fmt: skip

    @cute.jit
    def trc(self, on, t, e: cutlass.Constexpr[int]):
        if cutlass.const_expr(self.trace != 0):
            if on & (t < 48):
                store_u64(Int64(self.trace) + (t * 16 + e) * 8, clock64())

    @cute.jit
    def tick(self):
        value = Int64(0)
        if cutlass.const_expr(self.debug != 0):
            value = clock64()
        return value

    @cute.jit
    def rec(self, request, head, k: cutlass.Constexpr[int], value, cond):
        if cutlass.const_expr(self.debug != 0):
            if cond:
                cta = request * self.kv_heads + head
                store_u64(Int64(self.debug) + (cta * 16 + k) * 8, value)


def kol_smem(ring, q, page, window):
    """Dynamic shared memory bytes of a GqaWindowKol CTA (the kernel's allocation
    order and alignments)."""
    off = 0
    pages = -(-(window + q - 1 + 2 * KS) // page) + 1
    sizes = [(KRING * KT, 1024), (ring * KT, 1024), (32 * 128, 1024), (NPB * PTB, 1024),
             (q * 256, 16), (4 * q, 16),
             (8 * (2 * KRING + 2 * ring + 2 * NSB + 2 * NPB), 8), (4 * pages, 16),
             (4 * 9 * 32, 16), (8, 16)]  # fmt: skip
    for n, align in sizes:
        off = -(-off // align) * align + n
    return off


@cache
def compile_window_kol(
    group, kv_heads, q, page, ring, window, v_scale, max_position, device,
    round_norm=False, debug=0, redo=False, fill=1, o_layout=0, skeleton=False,
    trace=0, trace_req=0, l2ahead=0,
):  # fmt: skip
    """``GqaWindowKol`` for any request count (row-dynamic views); ``GqaWindow``'s
    arguments (the tensor map: ``kv_map_kol``'s)."""
    kernel = GqaWindowKol(
        group, kv_heads, q, page, ring, window, v_scale, round_norm, debug, redo,
        fill, o_layout, skeleton, trace, trace_req, l2ahead,
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
        Int64(0),  # the layer cache's tensor map address (kv_map_kol)
        Int64(0),  # the layer cache's base address
        Int64(0),  # row (block) stride, elements
        Int64(0),  # head stride, elements
        Int32(1),  # blocks
        Float32(1e-5),  # eps
    )
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
