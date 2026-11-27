# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent KDA record core with the deferred RecoverSSM fold: one state pass per
item, two independent item streams per SM, all per-item global traffic as
asynchronous bulk copies (KdaFlow).

One CTA per SM (256 threads) walks a contiguous, head-major range of (request, head)
items (at most two heads, at most MAX_ITEMS items) in two groups of 4 warps; group g
runs the CTA's items g, g + 2, ... start to finish.

Prologue (once per CTA): the items' slots, pending counts, record rows and betas are
tabled; the per-head channel weights are staged; the F_B gate projection of every
item (mma.sync; the F_B rows of the CTA's heads bulk-copied before the PDL wait)
leaves the raw gates in shared memory.

Per item (group g):

1. **prep** (thread c = channel c), shared memory only: convolution + SiLU, q / k
   normalization, decays, the fold's per-key operands. Its inputs (projection rows,
   convolution window, the pending commit's records) were bulk-copied one item
   ahead; its outputs (the new key / raw-gate records, the convolution window) leave
   by bulk stores.
2. **scan** (thread (b, l), b = c // 8, l = c % 8: rows 8 b + [0, 8), keys
   4 (l + 8 j) + [0, 4), j < 4): the item's FP32 state page goes from its slot of
   the CTA's page ring to registers, the pending commit is folded in (the stock
   Triton commit's math), the committed checkpoint leaves straight from the
   registers (streaming 16 B stores), the q tokens are verified and the outputs
   RMS-normed and gated; corrections and outputs leave by bulk stores. Each loaded
   4-key vector chunk feeds 8 rows: the shared-memory data path (128 B per clock,
   whatever the broadcast) carries ~3K cycles per item, not the ~7K of a 2-row
   layout. A row's dot product: KdaWS's per-thread partials and 8-lane transposed
   butterfly (lane l ends with row 8 b + l); its correction goes to the store
   buffer, from which the lane group reads its 8 rows' corrections back.

State stream: a ring of three page slots per CTA (two above q4, ``ring_slots``); item
j uses slot j % slots, and each warp refills its own rows of the slot with item
j + slots's as soon as its scan has read them.

Verify lengths q2..q8 (DSpark k1..k7): odd q pad the 2q scan vectors to whole float4
chunks (``vectors``); above q4 the block sums run in rounds of 32 (``sums_count``) and
the scan walks its 8 rows in two halves of 4 (``scan_halves``: the q8 partials fit the
register file), with a lane -> row permutation the epilogue follows.

Prologue order: the head constants (synchronous loads) and the small copies the
prologue waits on (F_B, F_A with the betas' 16 B row chunks, the first items'
inputs) come before the first-page burst; issued after it, they wait behind the
chip-wide burst for thousands of cycles.

Synchronous global accesses wait behind the saturated state stream (L2 round trips
of thousands of cycles), so the item loops issue none: one thread per group issues
every bulk copy, and the groups wait only on mbarriers of data loaded an item ahead.

Fold (prep, thread c = key c): the recorded keys normalized as the commit's
``rsqrt.approx.ftz(sum k^2 + 1e-6) * k`` (warp butterfly, then the four warps'
partials as (w0 + w2) + (w1 + w3), KdaWide's order), the suffix decays P_t (the
commit's final_decay before token t) and D_n; per state element (scan)
fc = fma(u_t * kn_t, P_t, fc) over t = q - 1 .. 0, then S = fma(S_0, D_n, fc), the
commit's association. Tokens t >= n (n pending) enter with u = kn = P = 0, which
leaves fc at its start value bit for bit: +0 (the commit's) when n > 0; -0 when
n = 0, with D_n = 1, so S is unchanged bit for bit. Branch-free.

Per item the math is the stock Triton verify's (KdaWS's formulas and instructions):
BF16 roundings of the convolution outputs, raw gates and recurrence outputs, FP32
state; only the FP32 reduction order of the recurrence's dot products differs.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from ..common import mx_rows

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_bf16x8,
    mma_into,
    scatter_reduce,
    store_words,
    try_wait,
    wait,
    warp_sum,
    zeros,
)
from .recover import (
    mul_rn,
    reciprocal,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
)
from .specialized import (
    COPY_BYTES,
    PAGE_BYTES,
    KdaWS,
    fence_async_shared,
    load_shared_f32x4,
    load_shared_u32x2,
    sync_threads,
    wait_bulk_reads,
)
from .wide import rsqrt_ftz

K = 128
GROUPS = 2
GROUP_THREADS = 128
THREADS = GROUPS * GROUP_THREADS
ROWS = 8  # state rows per scan thread: 8 (c // 8) + [0, 8)
LANES = 8  # scan threads per row
CHUNKS = 4  # 4-key chunks per thread and row: keys 4 (c % 8 + 8 j)
SLOTS = 3  # the CTA's state page ring (q <= 4)
# The page ring above q4 (q5..q8): two slots. The groups take alternate items, so slot
# j % 2 is always the same group's: its next page loads under its own prep and solve
# (more work per item than at q4). Two slots leave room for the larger per-group
# blocks (2q scan vectors, q^2 couplings) in the 320 KiB budget.
SLOTS_WIDE = 2


def ring_slots(tokens):
    """Page slots of the ring at verify length ``tokens``."""
    return SLOTS if tokens <= 4 else SLOTS_WIDE
MAX_ITEMS = 16  # items per CTA (tables, raw gates) of the default build
# Item-capacity builds (a KdaFlow specialization parameter): 16 serves up to 141
# requests per DP group at 24 heads on Rubin's 212 SMs (24 n / 212 <= 16); 20 serves
# only the captured sizes above that, up to 176 (C1152 / C1280); 24 serves 177 to 212
# (decode 192 / 208 per rank). The 16 and 20 builds are unchanged; ``sp.make_core``
# picks the smallest build that fits per captured shape.
ITEM_BUILDS = (16, 20, 24)
# Rubin's oversized shared memory mode (327 KiB per CTA; budget 320 KiB). Elsewhere
# the opt-in limit.
OVERSIZED_BUDGET = 320 * 1024
# Diagnostics (benchmarks only): per-phase cycle breakdown printed by a few CTAs.
TRACE = False
# Diagnostics (benchmarks only): parts of the scan to leave out ("fold", "store",
# "verify", "prep", "reduce", "epi"); results are wrong with any of them set.
SKIP = ()
# Diagnostics (benchmarks only): bounded mbarrier waits that report a stuck wait.
HANG = False
# FP32 pair ops as packed f32x2 instructions (False: scalar; bit-identical).
PACKED = True
# Tuning: a warp sync after every FOLD_FENCE scan chunks (bounds how far ptxas
# hoists the independent chunks' loads into spills).
FOLD_FENCE = 1
# Code size (G1-F1): in step KdaFlow runs after foreign code (in-proj GEMM, MegaMoE), and
# its hot loop is refetched each item pass (instruction-supply bound: 2 warps/scheduler,
# no_instruction ~48 % of stalls), so its bytes cost time per pass. ROLL bits: 1 the
# scan's 4 key chunks as a runtime loop, 2 scan_halves' two row halves as a runtime loop
# (smaller, but slower warm: off), 4 prep's coupling-norm select chain decoded per
# thread. Same operations in the same order: outputs, records and state bit-identical
# (q4, q8). Default 5: q8 hot loop 71.3 -> 47.1 KB; DB32 24L slice k7 B25 -3.3 %, k3 B59
# -2.1 % (G1F1, Hecate 2026-11-27).
ROLL = 5
# Block sums per prep: 2q norms, q^2 token couplings, q recorded-key norms, q previous
# output squares (q^2 + 4q: 12, 21, 32 at q2..q4; one 32-value reduction). Above q4
# (45 / 60 / 77 / 96 at q5..q8) ``block_sums_rounds`` reduces them 32 at a time.
SUMS = 32


def sums_count(tokens):
    """Block sums one prep reduces (SUMS up to q4, else whole 32-value rounds)."""
    need = tokens * tokens + 4 * tokens
    return SUMS if need <= SUMS else 32 * -(-need // 32)


# Verify lengths KdaFlow serves (DSpark k1..k7), each checked against the stock Triton
# verify / commit (test_specialized_decode_sp_kda_anyq_gpu). q <= 4 is the C512 build
# (q3 padded vectors); q5..8 add the block-sum rounds (``sums_count``), the two-half
# scan (``scan_halves``) and the two-slot page ring (``ring_slots``).
FLOW_TOKENS = (2, 3, 4, 5, 6, 7, 8)


def vectors(tokens):
    """Scan vectors stored per key (float4 chunks): the 2q vectors G_t k_t, G_t q_t,
    padded to a multiple of 4 with zeros (odd q: 2q = 6, 10, 14). Without the pad the
    chunk loops (2q // 4 chunks) dropped the last two output vectors at odd q (q3:
    the outputs of tokens 1, 2 were wrong, corrections and checkpoints right)."""
    return 4 * -(-2 * tokens // 4)


def padded_vectors_bytes(tokens):
    """Shared bytes of the separate padded vector block (odd q) per group; 0 where
    the 2q vectors fill whole float4 chunks (their block overlays qkvd as before)."""
    if vectors(tokens) == 2 * tokens:
        return 0
    return vectors(tokens) * K * 4


# Item table columns.
SLOT, HEAD, REQUEST, CONV_SLOT, CONV_OFFSET, PENDING, RECORD = range(7)
META = 8
# Per-head channel constants: conv weights [3, 4], biases [3], exp(A_log), dt.
CONSTS = 17
A_EXP, DT = 15, 16


def group_bytes(tokens, window, conv_size, conv_bufs=2):
    """One group's operand block, staging and store buffers (bytes); ``conv_bufs``
    convolution window buffers (see ``conv_buffers``)."""
    block = 6 * tokens * K * 4 + K * 4 + tokens * K * 2 + sums_count(tokens) * 16  # x v w u, D
    staging = (4 * tokens * K * 2 + conv_bufs * 3 * K * window * conv_size
               + tokens * K * 8)  # fmt: skip
    stores = tokens * 2 * K * 2  # new key / gate records
    return block + staging + stores + 16 * 4 * 4


def fa_spill_bytes(tokens, items):
    """Bytes of the prologue F_A scratch (``items`` x tokens x K BF16) beyond the region
    it overlays (the q / k / v / decay blocks and the old corrections u_t, which are
    unused until the first prep). 0 for the 16- and 20-item builds at every token
    count; the 24-item build reserves the rest so the scratch never runs into the
    buffers after it (D_n, couplings, staging), which are live in the prologue."""
    covered = GROUPS * 4 * tokens * K * 4 + GROUPS * tokens * K * 4
    return max(0, items * tokens * K * 2 - covered)


def shared_bytes(tokens, conv_dim_first, window, conv_size, stages=1, items=MAX_ITEMS,
                 conv_bufs=2):  # fmt: skip
    pages = ring_slots(tokens) * PAGE_BYTES
    heads = 2 * CONSTS * K * 4
    gates = items * tokens * K * 2
    tables = items * (META + 8) * 4
    groups = GROUPS * (group_bytes(tokens, window, conv_size, conv_bufs)
                       + padded_vectors_bytes(tokens))  # fmt: skip
    spill = fa_spill_bytes(tokens, items)
    return pages + heads + gates + tables + groups + spill + 64 * 8 + 4096  # (+ alignment)


def smem_budget(device):
    """Shared memory one KdaFlow CTA may use on ``device``."""
    props = torch.cuda.get_device_properties(device)
    limit = props.shared_memory_per_block_optin
    if (props.major, props.minor) == (10, 7):
        limit = max(limit, OVERSIZED_BUDGET)
    return limit


def max_stages(tokens, conv_dim_first, window, conv_size, limit, items=MAX_ITEMS):
    """1 when KdaFlow (an ``items``-item build) runs this layout (dim-first
    convolution window) within ``limit`` bytes of shared memory (with one window
    buffer per group at least), else 0."""
    if not conv_dim_first:
        return 0
    size = shared_bytes(tokens, conv_dim_first, window, conv_size, items=items,
                        conv_bufs=1)  # fmt: skip
    return int(size <= limit)


def conv_buffers(tokens, window, conv_size, limit, items=MAX_ITEMS):
    """Convolution window buffers per group: 2 (the next item's window streams in
    while this item's leaves) where they fit ``limit``, else 1. One buffer is
    refilled only after this item's window bulk store has read it (FP32 conv
    state: q4 20- / 24-item builds, q7 / q8). The window bytes and the math are the
    same either way; only when the next window copy is issued differs."""
    two = shared_bytes(tokens, True, window, conv_size, items=items, conv_bufs=2)
    return 2 if two <= limit else 1


def page_barrier(state_full, j, slots=SLOTS):
    """The page barrier of item j: ring slot j % SLOTS, one of two barriers by the
    slot's generation j // SLOTS. The two groups share the ring, so the consumer of
    item j can reach its wait before item j - SLOTS's page has landed (the other
    group's item, refilled by this group). On a single barrier per slot that wait
    tests the parity of the phase two back, which is complete: it passes before the
    page is loaded, and the refills then arrive in the wrong phase (a fifth arrival:
    'Warp Barrier Arrival Mismatch'). With a barrier per generation parity, the
    phase two back on item j's barrier is item j - 2 SLOTS's, which this group
    consumed itself, so the parity wait is exact."""
    return state_full + j % slots + slots * ((j // slots) % 2)


def page_parity(j, slots=SLOTS):
    """The mbarrier phase parity of item j's page on page_barrier(j)."""
    return (j // (2 * slots)) % 2


def item_build(items, requests, sms):
    """(persistent CTAs, items per CTA, the smallest ITEM_BUILDS capacity serving
    them or None) for ``items`` (heads x requests) on ``sms`` SMs."""
    ctas = min(items, sms)
    per_cta = -(-items // ctas)
    build = next(
        (c for c in ITEM_BUILDS if per_cta <= max_items_per_cta(requests, c)), None
    )
    return ctas, per_cta, build


def max_items_per_cta(requests, items=MAX_ITEMS):
    """KdaFlow tables the weights of two heads (a CTA's head-major item range may span
    at most two heads) and ``items`` items (the build's capacity)."""
    return min(requests + 1, items)


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l", ()


@asm()
def store_shared_f32x4(pointer, values):
    return "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f,~{memory}", (
        pointer,
        *values,
    )


@asm((Uint32,) * 4)
def load_global_u32x4(pointer):
    return "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", (pointer,)


@asm()
def store_shared_u32x4(pointer, values):
    return "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", (
        pointer,
        *values,
    )


@asm()
def store_global_f32x4(pointer, values):
    """16 B register -> global store, streaming (evict-first) cache policy."""
    return "st.global.cs.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f,~{memory}", (
        pointer,
        *values,
    )


@asm()
def store_bulk(dst, src, size):
    """Shared -> global bulk copy (in the open bulk group)."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def store_bulk_evict(dst, src, size):
    """Shared -> global bulk copy, L2 evict-first (in the open bulk group)."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
    code += "[$0], [$1], $2, pol; }"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def arrive_threads(barrier_id, count):
    """bar.arrive: count this warp in and go on (the waiters use bar.sync)."""
    return "bar.arrive $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


class KdaFlow(KdaWS):
    """KdaWS's operands and semantics for RecoverSSM record steps (``tokens`` q > 1,
    records always written, dim-first convolution window); with the deferred
    operands (``record_rows``, ``pending`` after the records) every item folds its
    pending commit first. ``stages``: 1 (the page ring has ring_slots(tokens) slots); ``ctas``
    persistent CTAs."""

    def __init__(
        self,
        heads,
        tokens,
        gate_mode,
        has_bias,
        conv_dim_first,
        window,
        ctas,
        stages=1,
        mx=False,
        items=MAX_ITEMS,
        conv_bufs=2,
    ):
        assert tokens in FLOW_TOKENS and stages == 1 and conv_dim_first
        assert conv_bufs in (1, 2), conv_bufs
        assert items in ITEM_BUILDS and items % 2 == 0, items
        super().__init__(
            heads, tokens, True, gate_mode, has_bias, conv_dim_first, window, ctas
        )
        self.stages = stages
        self.slots = ring_slots(tokens)  # page ring depth (3 up to q4, 2 above)
        self.conv_bufs = conv_bufs  # convolution window buffers per group
        self.items = items  # items per CTA this build tables (ITEM_BUILDS)
        # MXFP8 output (``quantize_outputs``): two more operands after the others,
        # the E4M3 bytes [R, D] and the F8_128x4 scales of the online-MXFP8
        # o_proj's input.
        self.mx = mx
        self.trace, self.hang, self.packed = TRACE, HANG, PACKED
        self.fold_fence = FOLD_FENCE
        self.roll = ROLL
        assert not self.roll & 1 or self.fold_fence == 1, "ROLL syncs after every chunk"
        self.skip = tuple(SKIP)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    def folds(self, args):
        """Whether the deferred operands (record rows, pending counts) follow."""
        return len(args) - (2 if self.mx else 0) > 20

    @cute.jit
    def quantize_outputs(self, args, tid, count, table):
        """``mx``: after both groups stored their items' BF16 outputs (the caller's
        CTA barrier makes them visible), the CTA quantizes its items' output rows
        into the MXFP8 o_proj input, ``ops.mxfp8_act.quantize`` bit for bit (4
        lanes per 32-wide block, 8 columns each: its quant8_store). Off the item
        chains: the scan loop is unchanged."""
        output, data, scale = args[10], args[len(args) - 2], args[len(args) - 1]
        T: cutlass.Constexpr = self.tokens
        width: cutlass.Constexpr = self.heads * K
        units: cutlass.Constexpr = T * (K // 8)  # 8-column lane units per item
        rounds: cutlass.Constexpr = -(-self.items * units // THREADS)
        lanes = count * units
        # Every round's loads first (one L2 round trip), then the quantization.
        held = cute.make_rmem_tensor((rounds, 8), Float32)
        rows = cute.make_rmem_tensor(rounds, Int32)
        cols = cute.make_rmem_tensor(rounds, Int32)
        for it in cutlass.range_constexpr(rounds):
            unit = it * THREADS + tid
            j = unit // units
            if unit >= lanes:
                j = Int32(0)
            within = unit % units
            rows[it] = table[j, REQUEST] * T + within // (K // 8)
            cols[it] = table[j, HEAD] * K + (within % (K // 8)) * 8
            for i in cutlass.range_constexpr(8):
                held[it, i] = Float32(0)
            if unit < lanes:
                loaded = load_bf16x8(at(output, rows[it], cols[it]), mutable=True)
                for i in cutlass.range_constexpr(8):
                    held[it, i] = loaded[i]
        for it in cutlass.range_constexpr(rounds):
            values = [held[it, i] for i in range(8)]
            w0, w1, e = mx_rows.quantize8_plain(values)
            if it * THREADS + tid < lanes:
                row, col = rows[it], cols[it]
                store_words(at(data, row, col), (w0, w1))
                if tid % 4 == 0:
                    where = mx_rows.scale_offset(row, col // mx_rows.BLOCK,
                                                 width // mx_rows.BLOCK)  # fmt: skip
                    scale[where] = Uint8(e)

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def bwait(self, barrier, phase, tag: cutlass.Constexpr):
        """mbarrier parity wait (CTA scope); HANG: report (and leave) a wait stuck
        for more than 2^31 cycles."""
        if cutlass.const_expr(self.hang):
            start = clock64()
            done = try_wait(barrier, phase, scope="cta")
            stuck = Int32(0)
            while done == 0:
                done = try_wait(barrier, phase, scope="cta")
                if stuck == 0:
                    if clock64() - start > Int64(1 << 31):
                        stuck = Int32(1)
                        done = Int32(1)
                        tid, _, _ = cute.arch.thread_idx()
                        cta, _, _ = cute.arch.block_idx()
                        if tid % 32 == 0:
                            if cta < 3:
                                cute.printf(f"HANG {tag} cta {{}} tid {{}} phase {{}}", cta, tid, phase)  # noqa: E501
        else:
            wait(barrier, phase, scope="cta")

    @cute.jit
    def group_sync(self, g):
        cute.arch.sync_warp()  # converged before the aligned barrier
        sync_threads(1 + g, GROUP_THREADS)

    @cute.jit
    def item_slots(self, args, item):
        """(head, request, state slot, conv slot, window offset) of an item; items
        are head-major."""
        slots = args[8]
        requests: cutlass.Constexpr = args[10].shape[0] // self.tokens
        head, request = item // requests, item % requests
        offset = args[17][request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        return head, request, Int32(slot), Int32(conv_slot), Int32(offset)

    @cute.jit
    def item_info(self, args, item):
        """(head, request, state slot, conv slot, window offset, pending n, record
        slot) of an item; the window offset includes the roll of the pending
        commit (the commit's compaction)."""
        head, request, slot, conv_slot, offset = self.item_slots(args, item)
        pending = Int32(0)
        rec = Int32(request)
        if cutlass.const_expr(self.folds(args)):
            rec = Int32(args[20][request])
            pending = Int32(0)
            if rec >= 0:
                pending = Int32(args[21][rec])
            if pending > self.tokens:
                pending = Int32(self.tokens)
            if pending < 0:
                pending = Int32(0)
        conv_offset = offset
        if pending > 0:
            conv_offset = offset + pending - 1
        return head, request, slot, conv_slot, conv_offset, pending, rec

    @cute.jit
    def load_quarter(self, args, pages, s, table, j, state_full, w):
        """Bulk-copy rows 32 w + [0, 32) (warp w's) of table item j's state page into
        page slot ``s``: one of the slot barrier's four arrivals (a padded row's
        arrival carries no bytes)."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = page_barrier(state_full, j, self.slots)
        if slot > 0:
            rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, COPY_BYTES)
            source = args[9][Int64(slot), None, None, None]
            copy_g2s(at(pages, s, w * rows * K), at(source, head, w * rows, 0),
                     COPY_BYTES, barrier)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def load_page(self, args, pages, s, table, j, state_full):
        """Table item j's whole state page into page slot ``s`` (one thread)."""
        for w in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
            self.load_quarter(args, pages, s, table, j, state_full, w)

    @cute.jit
    def stage_item(self, args, table, j, buffers, barrier, parity, c):
        """Bulk-copy table item j's inputs (group thread c; lane 0 of the group's
        warps 0-3 issue one share each): q / k / v and output gate rows, the
        convolution window (buffer ``parity``) and, with a pending commit, its
        records."""
        x, conv_state, gate = args[0], args[3], args[11]
        convs_all, xs, gs, kgs, uss = buffers
        convs = convs_all[parity, None, None]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        part_bytes: cutlass.Constexpr = K * self.window * size
        slot, head, pending = table[j, SLOT], table[j, HEAD], table[j, PENDING]
        if cutlass.const_expr("stagein" in self.skip):
            slot = Int32(0)
        if slot > 0:
            rows: cutlass.Constexpr = 4 * T * K * 2 + 3 * part_bytes
            records: cutlass.Constexpr = T * 2 * K * 2 + T * K * 4
            total = Int32(rows)
            if pending > 0:
                total = total + records
            token = table[j, REQUEST] * T
            if c == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, total)
                for t in cutlass.range_constexpr(T):
                    copy_g2s(at(gs, t, 0), at(gate, token + t, head * K), K * 2, barrier)
            for part in cutlass.range_constexpr(3):
                if c == 32 * (part + 1):
                    for t in cutlass.range_constexpr(T):
                        copy_g2s(at(xs, t, part, 0),
                                 at(x, token + t, part * dim + head * K), K * 2,
                                 barrier)  # fmt: skip
            # The window: thread 32 with two buffers; with one, thread 0 once its
            # bulk store of the current window (same buffer) has read it.
            conv_thread: cutlass.Constexpr = 32 if self.conv_bufs == 2 else 0
            if c == conv_thread:
                if cutlass.const_expr(self.conv_bufs == 1):
                    wait_bulk_reads()
                page = conv_state[Int64(table[j, CONV_SLOT]), None, None]
                for part in cutlass.range_constexpr(3):
                    copy_g2s(at(convs, part, 0), at(page, part * dim + head * K, 0),
                             part_bytes, barrier)  # fmt: skip
            if pending > 0:
                rec = table[j, RECORD]
                if c == 64:
                    copy_g2s(at(kgs, 0, 0), at(args[19], rec, head, 0, 0),
                             T * 2 * K * 2, barrier)  # fmt: skip
                if c == 96:
                    copy_g2s(at(uss, 0, 0), at(args[18], rec, head, 0, 0), T * K * 4,
                             barrier)  # fmt: skip
        else:
            if c == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        if cutlass.const_expr("noop" not in self.skip):
            self.body(args)

    @cute.jit
    def body(self, args):
        k0 = Int64(0)
        if cutlass.const_expr(self.trace):
            k0 = clock64()
        conv_state, output = args[3], args[10]
        f_a, beta, f_b = args[4], args[5], args[16]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first
        conv_type: cutlass.Constexpr = conv_state.element_type

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (self.slots, K * K), align=1024)
        hcs = alloc(smem, Float32, (2, CONSTS, K))  # per head slot, channel
        raw = alloc(smem, BFloat16, (self.items, T, K), align=16)  # raw gates
        table = alloc(smem, Int32, (self.items, META))
        betas = alloc(smem, Float32, (self.items, 8))  # sigmoid(beta) per token
        # Per group: the 2q scan vectors (2 blocks), v (then the outputs), the fold
        # weights w_t.
        qkvd = alloc(smem, Float32, (GROUPS, 4, T, K), align=128)
        xv = cute.make_tensor(
            at(qkvd, 0, 0, 0, 0),
            cute.make_layout((GROUPS, 2 * T * K), stride=(4 * T * K, 1)),
        )
        v_s, fw_s = qkvd[None, 2, None, None], qkvd[None, 3, None, None]
        if cutlass.const_expr(vectors(T) != 2 * T):  # odd q: a padded vector block
            xv = alloc(smem, Float32, (GROUPS, vectors(T) * K), align=128)
        fu_s = alloc(smem, Float32, (GROUPS, T, K))  # the old corrections u_t
        # The prologue F_A scratch (``fas`` below) overlays qkvd and fu_s; a build with
        # more items reserves the bytes past them here, directly after fu_s, so the
        # scratch stays off the buffers that follow (fa_spill_bytes).
        FA_SPILL: cutlass.Constexpr = fa_spill_bytes(T, self.items)
        if cutlass.const_expr(FA_SPILL > 0):
            alloc(smem, Uint8, (FA_SPILL,))
        fd_s = alloc(smem, Float32, (GROUPS, K))  # D_n
        ce_s = alloc(smem, Float32, (GROUPS, T * T))  # the scan's token couplings
        g_s = alloc(smem, BFloat16, (GROUPS, T, K))
        convs = alloc(smem, conv_type, (GROUPS, self.conv_bufs, 3, K * self.window),
                      align=128)  # fmt: skip
        xs = alloc(smem, BFloat16, (GROUPS, T, 3, K), align=128)
        gs = alloc(smem, BFloat16, (GROUPS, T, K), align=128)
        kgs = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)
        uss = alloc(smem, Float32, (GROUPS, T, K), align=128)
        kgo = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)  # new key/gate
        sums = alloc(smem, Float32, (GROUPS, sums_count(T), 4))
        # Two page barriers per ring slot, alternating by the slot's generation
        # (item j // self.slots): see page_barrier.
        bars = smem.allocate(cutlass.Uint64, 2 * self.slots + GROUPS + 2)
        state_full, stage_full = bars, bars + 2 * self.slots
        fb_full = bars + 2 * self.slots + GROUPS
        fa_full = bars + 2 * self.slots + GROUPS + 1
        # Prologue scratch: the F_B rows of the CTA's heads in the last page slot (its
        # page is loaded after the gate projection), the F_A rows of its items over
        # the q / k / v / decay blocks (unused until the first prep).
        fbs = cute.make_tensor(
            cute.recast_ptr(at(pages, self.slots - 1, 0), dtype=BFloat16),
            cute.make_layout((2, K, K), stride=(K * K, K, 1)),
        )
        fas = cute.make_tensor(
            cute.recast_ptr(at(qkvd, 0, 0, 0, 0), dtype=BFloat16),
            cute.make_layout((self.items, T, K), stride=(T * K, K, 1)),
        )

        if tid == 0:
            for s in cutlass.range_constexpr(2 * self.slots):  # one arrival per quarter
                cute.arch.mbarrier_init(state_full + s, PAGE_BYTES // COPY_BYTES)
            for g in cutlass.range_constexpr(GROUPS):
                cute.arch.mbarrier_init(stage_full + g, 1)
            cute.arch.mbarrier_init(fb_full, 1)
            cute.arch.mbarrier_init(fa_full, 1)
            cute.arch.mbarrier_init_fence()
        # The item table: host-staged step metadata (slot table, accepted counts,
        # record rows, pending counts), no predecessor output.
        if tid < count:
            head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
                args, first + tid
            )
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, CONV_SLOT] = conv_slot
            table[tid, CONV_OFFSET] = conv_offset
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        ps = zeros(8, Int64)  # TRACE: prologue stamps
        if cutlass.const_expr(self.trace):
            ps[1] = clock64()
        h0 = first // requests
        h1 = (first + count - 1) // requests
        heads = h1 - h0 + 1
        # Before the PDL wait: the channel weights of the CTA's heads (synchronous
        # loads, issued before the state stream they would queue behind), then the
        # first pages (this layer's state) and the F_B rows.
        hslot = tid // K
        BULK_HC: cutlass.Constexpr = self.bulk_head_constants(args)
        if cutlass.const_expr(BULK_HC):  # conv weight and dt rows: 13 copies a head
            lane, hw = tid % 32, tid // 32
            if hw < heads:
                rows = hcs[hw, None, None]
                if lane < 12:
                    copy_g2s(at(rows, lane, 0),
                             at(args[1], lane // 4, lane % 4, (h0 + hw) * K), K * 4,
                             fb_full)  # fmt: skip
                if lane == 12:
                    copy_g2s(at(rows, DT, 0), at(args[7], (h0 + hw) * K), K * 4, fb_full)
        elif hslot < heads:
            self.head_constants(hcs[hslot, None, None], args, h0 + hslot, tid % K)
        if tid == 0:  # F_B first (L2-resident): the gate projection waits on it
            hc_bytes = 0
            if cutlass.const_expr(BULK_HC):
                hc_bytes = heads * 13 * K * 4
            cute.arch.mbarrier_arrive_and_expect_tx(fb_full,
                                                    heads * K * K * 2 + hc_bytes)  # fmt: skip
            for h in cutlass.range_constexpr(2):
                if h < heads:
                    copy_g2s(at(fbs, h, 0, 0), at(f_b, (h0 + h) * K, 0), K * K * 2,
                             fb_full)  # fmt: skip
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trace):
            ps[2] = clock64()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.trace):
            ps[3] = clock64()
        # The F_A rows (16 B per thread, all in flight at once) and the betas:
        # synchronous loads, before the page burst they would queue behind (dozens
        # of small bulk copies cost thousands of cycles of issue).
        FA_CHUNKS: cutlass.Constexpr = K * 2 // 16
        FA_ROUNDS: cutlass.Constexpr = -(-self.items * T * FA_CHUNKS // THREADS)
        fav = cute.make_rmem_tensor((FA_ROUNDS, 4), Uint32)
        for k in cutlass.range_constexpr(FA_ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * FA_CHUNKS)
            if item < count:
                t, part = (idx // FA_CHUNKS) % T, idx % FA_CHUNKS
                row = table[item, REQUEST] * T + t
                v = load_global_u32x4(at(f_a, row, part * 8))
                for e in cutlass.range_constexpr(4):
                    fav[k, e] = v[e]
        if tid < count * T:  # read by the groups after the gate projection's barrier
            item, t = tid // T, tid % T
            row = table[item, REQUEST] * T + t
            betas[item, t] = triton_sigmoid(Float32(beta[row, table[item, HEAD]]))
        if cutlass.const_expr(BULK_HC):  # noqa: SIM102
            if hslot < heads:  # read by the groups after the gate projection's barrier
                hcs[hslot, A_EXP, tid % K] = triton_exp(Float32(args[6][h0 + hslot]))
        for k in cutlass.range_constexpr(FA_ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * FA_CHUNKS)
            if item < count:
                t, part = (idx // FA_CHUNKS) % T, idx % FA_CHUNKS
                store_shared_u32x4(at(fas, item, t, part * 8),
                                   [fav[k, e] for e in range(4)])  # fmt: skip
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        buffers = (
            convs[g, None, None, None], xs[g, None, None, None], gs[g, None, None],
            kgs[g, None, None], uss[g, None, None],
        )  # fmt: skip
        if g < count:  # the group's first item's inputs
            self.stage_item(args, table, g, buffers, stage_full + g, 0, c)
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trace):
            ps[0] = clock64()
        self.bwait(fb_full, 0, "fb_full")
        if cutlass.const_expr(self.trace):
            ps[7] = clock64()
        # The state stream starts once the small copies the prologue waits on have
        # landed (issued earlier, they queue behind the chip-wide page burst); its
        # first pages fill during the gate projection.
        if tid == 0:
            for s in cutlass.range_constexpr(self.slots - 1):
                if s < count:
                    self.load_page(args, pages, s, table, s, state_full)
        cute.arch.sync_warp()
        cute.arch.barrier()  # F_A
        if cutlass.const_expr(self.trace):
            ps[4] = clock64()
        if cutlass.const_expr("gate" not in self.skip):
            self.gate_projection(tid, count, h0, table, fbs, fas, raw)
        cute.arch.sync_warp()
        cute.arch.barrier()  # raw gates; the last page slot and the blocks are free
        if cutlass.const_expr(self.trace):
            ps[5] = clock64()
            if tid == 0:
                self.bwait(state_full, 0, "trace page 0")
                ps[6] = clock64()
                cta_t, _, _ = cute.arch.block_idx()
                if cta_t % 53 == 0:
                    cute.printf("PTRACE cta {} count {} table {} hc+issue {} pdl {} fa+stage-issue {} fbwait {} fawait+burst {} gate {} page0 {}", cta_t, count, ps[1] - k0, ps[2] - ps[1], ps[3] - ps[2], ps[0] - ps[3], ps[7] - ps[0], ps[4] - ps[7], ps[5] - ps[4], ps[6] - k0)  # noqa: E501
        if tid == 0:
            if self.slots - 1 < count:
                last = self.slots - 1
                self.load_page(args, pages, last, table, last, state_full)
        cute.arch.sync_warp()
        ops = (xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s)
        outs = (kgo[g, None, None],)
        sums_g = (sums[g, None, None],)
        if cutlass.const_expr("loop" not in self.skip):
            self.group_items(args, g, c, count, h0, pages, table, betas, raw, hcs,
                             buffers, ops, outs, sums_g, state_full,
                             stage_full + g, k0)  # fmt: skip
        if cutlass.const_expr(self.mx):
            cute.arch.barrier()  # both groups' outputs stored
            self.quantize_outputs(args, tid, count, table)

    @cute.jit
    def gate_projection(self, tid, count, h0, table, fbs, fas, raw):
        """raw[j, t, f] = BF16(F_B[head_j] @ F_A[j, t]) for the CTA's items (warp w:
        features 32 (w % 4) + [0, 32), items w // 4, w // 4 + 2, ...; KdaWS's MMA
        chain per item). Results are held until every warp has read F_A (which
        overlays the operand blocks)."""
        T: cutlass.Constexpr = self.tokens
        lane, warp = tid % 32, tid // 32
        group_id, quarter = lane >> 2, lane & 3
        wq, half = warp % 4, warp // 4
        PER: cutlass.Constexpr = self.items // 2
        acc = cute.make_rmem_tensor((PER, 2, 4), Float32)
        wr = cute.make_rmem_tensor((2, 8, 4), Uint32)
        cached = Int32(-1)
        for i in cutlass.range_constexpr(PER):
            j = half + 2 * i
            if j < count:
                head = table[j, HEAD]
                if head != cached:
                    fb = fbs[head - h0, None, None]
                    for m in cutlass.range_constexpr(2):
                        for chunk in cutlass.range_constexpr(8):
                            feature = 32 * wq + 16 * m + group_id
                            col = 16 * chunk + 4 * quarter
                            low = load_shared_u32x2(at(fb, feature, col))
                            high = load_shared_u32x2(at(fb, feature + 8, col))
                            wr[m, chunk, 0], wr[m, chunk, 2] = low
                            wr[m, chunk, 1], wr[m, chunk, 3] = high
                    cached = head
                cute.arch.sync_warp()  # converged for mma.sync
                for m in cutlass.range_constexpr(2):
                    chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                    part = zeros((chains, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        chain: cutlass.Constexpr = (
                            chunk // 2 if self.gate_mode == 1 else 0
                        )
                        wa = tuple(wr[m, chunk, e] for e in range(4))
                        ab = load_shared_u32x2(
                            at(fas, j, group_id % T, 16 * chunk + 4 * quarter)
                        )
                        mma_into(part[chain, None], wa, ab, bf16=True)
                    for e in cutlass.range_constexpr(4):
                        value = part[0, e]
                        if cutlass.const_expr(self.gate_mode == 1):
                            value = Float32(0) + (
                                ((value + part[1, e]) + part[2, e]) + part[3, e]
                            )
                        acc[i, m, e] = value
        cute.arch.sync_warp()
        cute.arch.barrier()  # every warp has read F_A
        for i in cutlass.range_constexpr(PER):
            j = half + 2 * i
            if j < count:
                for m in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        t = 2 * quarter + e % 2
                        feature = 32 * wq + 16 * m + group_id + 8 * (e // 2)
                        if t < T:
                            raw[j, t, feature] = BFloat16(acc[i, m, e])

    # ------------------------------------------------------------- group loop
    @cute.jit
    def group_items(self, args, g, c, count, h0, pages, table, betas, raw, hcs,
                    buffers, ops, outs, sums_g, state_full, stage_full, k0):  # fmt: skip
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        (kgo,) = outs
        state, output = args[9], args[10]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        lane, warp = c % 32, c // 32
        lane_in, base = c % LANES, (c // LANES) * ROWS  # rows base + [0, 8)
        own = base + lane_in
        if cutlass.const_expr(T > 4):  # scan_halves' row order (see there)
            own = base + 4 * (lane_in & 1) + 2 * ((lane_in >> 2) & 1) + ((lane_in >> 1) & 1)
        # The output channel / norm weight / gate of this thread's row: c up to q4.
        col = c
        if cutlass.const_expr(T > 4):
            col = own
        nw = args[12][col]
        mine = (count - g + 1) // 2  # this group's items: g, g + 2, ...
        pt = zeros(6, Int64)  # TRACE: prep sub-phases
        # The group's previous item (outputs before the gated norm, output gates,
        # valid / token / head): its norm rides on the next prep's block reduction.
        po = cute.make_rmem_tensor(T, Float32)
        pg = cute.make_rmem_tensor(T, Float32)
        pm = cute.make_rmem_tensor(3, Int32)
        for t in cutlass.range_constexpr(T):
            po[t] = Float32(0)
        pm[0] = Int32(0)
        tr = zeros(9, Int64)  # TRACE: total, prep, state wait, scan, solve, epi, waits, compute, issue
        t0 = Int64(0)
        ta, tb, tc, td, te = (Int64(0) for _ in range(5))
        if cutlass.const_expr(self.trace):
            t0 = clock64()
        for jj in cutlass.range(mine, unroll=1):
            j = g + GROUPS * jj
            slot, head = table[j, SLOT], table[j, HEAD]
            pending, rec = table[j, PENDING], table[j, RECORD]
            token = table[j, REQUEST] * T
            if cutlass.const_expr(self.trace):
                ta = clock64()
            self.bwait(stage_full, jj % 2, "stage_full")
            if c == 0:  # the previous item's bulk stores have read their buffers
                wait_bulk_reads()
            cute.arch.sync_warp()
            if cutlass.const_expr(self.trace):
                tw = clock64()
                tr[6] = tr[6] + (tw - ta)
            item_buffers = (buffers[0][jj % self.conv_bufs, None, None], *buffers[1:])
            if cutlass.const_expr("prep" not in self.skip):
                self.prep_item(args, g, c, j, slot, pending, table[j, CONV_OFFSET],
                               raw, hcs[head - h0, None, None], item_buffers, ops,
                               sums_g, kgo, pt, po)  # fmt: skip
            if cutlass.const_expr(self.trace):
                tp = clock64()
                tr[7] = tr[7] + (tp - tw)
            if pm[0] != 0:  # the previous item's outputs
                if slot > 0:
                    self.finish(args, col, sums_g[0], NV + T * T + T, po, pg, pm, nw)
                else:
                    self.finish_alone(args, g, c, sums_g[0], po, pg, pm, nw, col)
                pm[0] = Int32(0)
            for t in cutlass.range_constexpr(T):  # this item's output gates
                pg[t] = Float32(g_s[g, t, col])
            self.group_sync(g)  # the block is complete; the staging buffers are read
            if c == 0:
                recst = slot > 0
                if cutlass.const_expr("recst" in self.skip):
                    recst = slot < 0
                if recst:  # the new key / gate records and convolution window
                    convs = item_buffers[0]
                    part_bytes: cutlass.Constexpr = (
                        K * self.window * convs.element_type.width // 8
                    )
                    store_bulk(at(args[19], rec, head, 0, 0), at(kgo, 0, 0),
                               T * 2 * K * 2)  # fmt: skip
                    page = args[3][Int64(table[j, CONV_SLOT]), None, None]
                    for part in cutlass.range_constexpr(3):
                        store_bulk(at(page, part * dim + head * K, 0),
                                   at(convs, part, 0), part_bytes)  # fmt: skip
                    bulk_commit()  # read by the next item's start (one window buffer: its stage)
            if jj + 1 < mine:  # the group's next item's inputs
                self.stage_item(args, table, j + GROUPS, buffers, stage_full,
                                (jj + 1) % self.conv_bufs, c)  # fmt: skip
            cute.arch.sync_warp()
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[1] = tr[1] + (tb - ta)
            s = j % self.slots
            self.bwait(page_barrier(state_full, j, self.slots),
                       page_parity(j, self.slots), "state_full")  # fmt: skip
            if cutlass.const_expr(self.trace):
                tc = clock64()
                tr[2] = tr[2] + (tc - tb)
            if slot > 0:
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):
                    stored = pending < 0
                destination = state[Int64(slot), head, None, None]
                if cutlass.const_expr(T <= 4):
                    acc = cute.make_rmem_tensor((ROWS, NV), Float32)
                    self.scan(acc, g, s, pages, pending, lane_in, base, xv, fw_s, fu_s,
                              fd_s, destination, stored)  # fmt: skip
                else:
                    values = self.scan_halves(g, s, pages, pending, lane_in, base, xv,
                                              fw_s, fu_s, fd_s, destination, stored)  # fmt: skip
                # The checkpoint left from registers: the warp's quarter of the slot
                # (its own rows, read) is free for the next page at once.
                fence_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    if j + self.slots < count:
                        self.load_quarter(args, pages, s, table, j + self.slots,
                                          state_full, warp)  # fmt: skip
                cute.arch.sync_warp()
                if cutlass.const_expr(self.trace):
                    td = clock64()
                    tr[3] = tr[3] + (td - tc)
                if cutlass.const_expr("solve" not in self.skip):
                    if cutlass.const_expr(T <= 4):
                        values = self.rows_reduce(acc, lane_in)
                    self.solve(values, g, betas, j, own, v_s, ce_s,
                               args[18][rec, head, None, None], po)  # fmt: skip
                    pm[0], pm[1], pm[2] = Int32(1), token, head
                # Each warp refills its own quarter of the slot once that quarter's
                # checkpoint store has read it (no group barrier: thread c owns row
                # c's outputs, which its epilogue thread reads).
                if cutlass.const_expr(self.trace):
                    te = clock64()
                    tr[4] = tr[4] + (te - td)
                if cutlass.const_expr(self.trace):
                    tr[5] = tr[5] + (clock64() - te)
            else:
                self.release_slot(args, c, j, count, pages, table, state_full, False)
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
        if pm[0] != 0:  # the group's last item
            self.finish_alone(args, g, c, sums_g[0], po, pg, pm, nw, col)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("TRACE cta {} g {} items {} prologue {} total {} prep {} (waits {} compute {}) wstate {} scan {} solve {} epi {} | conv {} decay {} bsum {} sync {} vec {} fold {}", cta, g, mine, t0 - k0, clock64() - t0, tr[1], tr[6], tr[7], tr[2], tr[3], tr[4], tr[5], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])  # noqa: E501

    @cute.jit
    def release_slot(self, args, c, j, count, pages, table, state_full, stored):
        """Item j's slot is free once its checkpoint store has read it: thread 0
        refills it with item j + self.slots's page."""
        if c == 0:
            if stored:
                wait_bulk_reads()
            if j + self.slots < count:
                self.load_page(args, pages, j % self.slots, table, j + self.slots,
                               state_full)  # fmt: skip
        cute.arch.sync_warp()

    # ------------------------------------------------------------------- scan
    def pfma(self, a, b, c):
        """Lanewise a * b + c of FP32 pairs (packed, or two FFMAs: the same bits)."""
        if self.packed:
            return fma2(a, b, c)
        return (fma(a[0], b[0], c[0]), fma(a[1], b[1], c[1]))

    @cute.jit
    def scan(self, acc, b, s, pages, pending, lane_in, base, xv, fw_s, fu_s, fd_s,
             destination, stored):
        """One pass over this thread's state elements (rows base + [0, 8), keys
        4 (lane_in + 8 j) + [0, 4)): the pending commit is folded in,
        S = fma(S_0, D_n, fc), fc = sum_t fma(u_t, w_t, fc) over t = q - 1 .. 0
        (w_t = kn_t P_t; tokens t >= n are zeros), the committed element goes back
        into the page slot, and acc[r, m] += S[r, key] x_m[key] over the 2q scan
        vectors (x = [G_t k_t; G_t q_t])."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        NVP: cutlass.Constexpr = vectors(T)
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        us = cute.make_rmem_tensor((ROWS, T), Float32)
        for t in cutlass.range_constexpr(T):
            for h in cutlass.range_constexpr(2):
                u4 = load_shared_f32x4(at(fu_s, b, t, base + 4 * h))
                for e in cutlass.range_constexpr(4):
                    us[4 * h + e, t] = u4[e]
        for rr in cutlass.range_constexpr(ROWS):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        if cutlass.const_expr(self.roll & 1):
            for ch in cutlass.range(CHUNKS, unroll=1):
                self.scan_chunk(acc, ch, b, s, pages, start, us, lane_in, base, xv,
                                fw_s, fd_s, destination, stored)  # fmt: skip
                cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)
        else:
            for ch in cutlass.range_constexpr(CHUNKS):
                self.scan_chunk(acc, ch, b, s, pages, start, us, lane_in, base, xv,
                                fw_s, fd_s, destination, stored)  # fmt: skip
                if cutlass.const_expr(ch % self.fold_fence == self.fold_fence - 1):
                    cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)

    @cute.jit
    def scan_chunk(self, acc, ch, b, s, pages, start, us, lane_in, base, xv, fw_s, fd_s,
                   destination, stored):
        """``scan``'s key chunk ch (keys 4 (lane_in + 8 ch) + [0, 4)) over the 8 rows."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        NVP: cutlass.Constexpr = vectors(T)
        key = 4 * (lane_in + LANES * ch)
        d4 = load_shared_f32x4(at(fd_s, b, key))
        w4 = [load_shared_f32x4(at(fw_s, b, t, key)) for t in range(T)]
        xs = cute.make_rmem_tensor((4, NVP), Float32)
        for e in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(NVP // 4):
                # swizzled: [chunk][e][h][lane][4] (lanes read 128 B in a row)
                pos = (((ch * 4 + e) * (NVP // 4) + h) * LANES + lane_in) * 4
                x4 = load_shared_f32x4(at(xv, b, pos))
                for i in cutlass.range_constexpr(4):
                    xs[e, 4 * h + i] = x4[i]
        if cutlass.const_expr("scan" in self.skip):
            for e in cutlass.range_constexpr(4):
                for m in cutlass.range_constexpr(NV):
                    acc[0, m] = acc[0, m] + xs[e, m] + w4[0][e] + d4[e]
        for rr in cutlass.range_constexpr(ROWS):
            address = at(pages, s, (base + rr) * K + key)
            sv = load_shared_f32x4(address)
            new = [Float32(0) for _ in range(4)]
            if cutlass.const_expr("scan" in self.skip):
                new = [sv[e] for e in range(4)]
            else:
                for p in cutlass.range_constexpr(2):
                    fc = (start, start)
                    for i in cutlass.range_constexpr(T):
                        t = T - 1 - i
                        fc = self.pfma((us[rr, t], us[rr, t]),
                                       (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # fmt: skip
                    pair = self.pfma((sv[2 * p], sv[2 * p + 1]),
                                     (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                    new[2 * p], new[2 * p + 1] = pair
                for e in cutlass.range_constexpr(4):
                    for p in cutlass.range_constexpr(NV // 2):
                        pair = (acc[rr, 2 * p], acc[rr, 2 * p + 1])
                        pair = self.pfma((new[e], new[e]),
                                         (xs[e, 2 * p], xs[e, 2 * p + 1]), pair)  # fmt: skip
                        acc[rr, 2 * p], acc[rr, 2 * p + 1] = pair
            if cutlass.const_expr("store" not in self.skip):  # noqa: SIM102
                if stored:
                    store_global_f32x4(at(destination, base + rr, key), new)

    @cute.jit
    def scan_halves(self, b, s, pages, pending, lane_in, base, xv, fw_s, fu_s, fd_s,
                    destination, stored):
        """``scan`` + ``rows_reduce`` above q4 (2q = 10..16 vectors): the thread's 8 rows
        in two halves of 4 (rows base + 4 h + [0, 4)), so the live partials are
        acc[4][2q] and us[4][q] (~200 registers at q8, not ~290). Each half walks every
        key chunk (the chunk's vectors and fold weights are read once per half) and is
        reduced over the 8 lanes by the transposed butterfly's ^4, ^2 stages; a last
        transposed ^1 across the two halves leaves lane l the full sums of row
        base + 4 (l & 1) + 2 ((l >> 2) & 1) + ((l >> 1) & 1). Per element the fold and
        the dot-product partials are ``scan``'s operations in ``scan``'s order (chunks,
        then keys), and every row sum is the same xor-4, 2, 1 tree: only which lane ends
        up with which row differs."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        NVP: cutlass.Constexpr = vectors(T)
        HALF: cutlass.Constexpr = ROWS // 2
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        red = cute.make_rmem_tensor((2, NV), Float32)
        if cutlass.const_expr(self.roll & 2):  # the halves as a runtime loop
            for hf in cutlass.range(2, unroll=1):
                self.scan_half(red, hf, b, s, pages, start, lane_in, base, xv, fw_s, fu_s,
                               fd_s, destination, stored)  # fmt: skip
        else:
            for hf in cutlass.range_constexpr(2):
                self.scan_half(red, hf, b, s, pages, start, lane_in, base, xv, fw_s, fu_s,
                               fd_s, destination, stored)  # fmt: skip
        reduced = [[red[h, m] for m in range(NV)] for h in range(2)]
        # Across the halves: transposed ^1 (lane bit 0 keeps the second half's row).
        upper = (lane_in & 1) != 0
        values = []
        for m in cutlass.range_constexpr(NV):
            send = reduced[0][m] if upper else reduced[1][m]
            keep = reduced[1][m] if upper else reduced[0][m]
            values.append(keep + cute.arch.shuffle_sync_bfly(send, 1))
        return values

    @cute.jit
    def scan_half(self, red, hf, b, s, pages, start, lane_in, base, xv, fw_s, fu_s, fd_s,
                  destination, stored):
        """``scan_halves``' half hf (rows base + 4 hf + [0, 4)): every key chunk, then the
        ^4, ^2 transposed butterfly; this lane's partial row sums into red[hf]."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        HALF: cutlass.Constexpr = ROWS // 2
        us = cute.make_rmem_tensor((HALF, T), Float32)
        for t in cutlass.range_constexpr(T):
            u4 = load_shared_f32x4(at(fu_s, b, t, base + HALF * hf))
            for e in cutlass.range_constexpr(4):
                us[e, t] = u4[e]
        acc = cute.make_rmem_tensor((HALF, NV), Float32)
        for rr in cutlass.range_constexpr(HALF):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        if cutlass.const_expr(self.roll & 1):
            for ch in cutlass.range(CHUNKS, unroll=1):
                self.halves_chunk(acc, ch, b, s, pages, start, us, hf, lane_in, base, xv,
                                  fw_s, fd_s, destination, stored)  # fmt: skip
                cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)
        else:
            for ch in cutlass.range_constexpr(CHUNKS):
                self.halves_chunk(acc, ch, b, s, pages, start, us, hf, lane_in, base, xv,
                                  fw_s, fd_s, destination, stored)  # fmt: skip
                if cutlass.const_expr(ch % self.fold_fence == self.fold_fence - 1):
                    cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)
        # This half's 4 rows over the 8 lanes: transposed ^4, ^2.
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(HALF)]
        for stage in cutlass.range_constexpr(2):
            offset: cutlass.Constexpr = 4 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            upper = (lane_in & offset) != 0
            merged = []
            for i in cutlass.range_constexpr(half):
                row_v = []
                for m in cutlass.range_constexpr(NV):
                    send = rows[i][m] if upper else rows[i + half][m]
                    keep = rows[i + half][m] if upper else rows[i][m]
                    row_v.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                merged.append(row_v)
            rows = merged
        if cutlass.const_expr(isinstance(hf, int)):
            for m in cutlass.range_constexpr(NV):
                red[hf, m] = rows[0][m]
        else:
            if hf == 0:
                for m in cutlass.range_constexpr(NV):
                    red[0, m] = rows[0][m]
            else:
                for m in cutlass.range_constexpr(NV):
                    red[1, m] = rows[0][m]

    @cute.jit
    def halves_chunk(self, acc, ch, b, s, pages, start, us, hf, lane_in, base, xv, fw_s,
                     fd_s, destination, stored):
        """``scan_halves``' key chunk ch of half hf (rows base + 4 hf + [0, 4))."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        NVP: cutlass.Constexpr = vectors(T)
        HALF: cutlass.Constexpr = ROWS // 2
        key = 4 * (lane_in + LANES * ch)
        d4 = load_shared_f32x4(at(fd_s, b, key))
        w4 = [load_shared_f32x4(at(fw_s, b, t, key)) for t in range(T)]
        xs = cute.make_rmem_tensor((4, NVP), Float32)
        for e in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(NVP // 4):
                pos = (((ch * 4 + e) * (NVP // 4) + h) * LANES + lane_in) * 4
                x4 = load_shared_f32x4(at(xv, b, pos))
                for i in cutlass.range_constexpr(4):
                    xs[e, 4 * h + i] = x4[i]
        for r4 in cutlass.range_constexpr(HALF):
            row = base + HALF * hf + r4
            sv = load_shared_f32x4(at(pages, s, row * K + key))
            new = [Float32(0) for _ in range(4)]
            for p in cutlass.range_constexpr(2):
                fc = (start, start)
                for i in cutlass.range_constexpr(T):
                    t = T - 1 - i
                    fc = self.pfma((us[r4, t], us[r4, t]),
                                   (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # fmt: skip
                pair = self.pfma((sv[2 * p], sv[2 * p + 1]),
                                 (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                new[2 * p], new[2 * p + 1] = pair
            for e in cutlass.range_constexpr(4):
                for p in cutlass.range_constexpr(NV // 2):
                    pair = (acc[r4, 2 * p], acc[r4, 2 * p + 1])
                    pair = self.pfma((new[e], new[e]),
                                     (xs[e, 2 * p], xs[e, 2 * p + 1]), pair)  # fmt: skip
                    acc[r4, 2 * p], acc[r4, 2 * p + 1] = pair
            if cutlass.const_expr("store" not in self.skip):  # noqa: SIM102
                if stored:
                    store_global_f32x4(at(destination, row, key), new)

    @cute.jit
    def rows_reduce(self, acc, lane_in):
        """The 8 lanes' partials of rows base + [0, 8) -> lane l holds the 2q full
        sums of row base + l (transposed butterfly: lanes ^4, ^2, ^1)."""
        NV: cutlass.Constexpr = 2 * self.tokens
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(ROWS)]
        for stage in cutlass.range_constexpr(3):
            offset: cutlass.Constexpr = 4 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            upper = (lane_in & offset) != 0
            merged = []
            for i in cutlass.range_constexpr(half):
                row = []
                for m in cutlass.range_constexpr(NV):
                    send = rows[i][m] if upper else rows[i + half][m]
                    keep = rows[i + half][m] if upper else rows[i][m]
                    row.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                merged.append(row)
            rows = merged
        return rows[0]

    @cute.jit
    def solve(self, values, b, betas, j, own, v_s, ce_s, u_dst, po):
        """Row ``own``'s tokens from its scan sums A_t = S_0 (G_t k_t),
        B_t = S_0 (G_t q_t): dot_t = A_t + sum_s<t c_ts u_s, u_t = (v_t - dot_t) beta_t,
        out_t = B_t + sum_s<=t e_ts u_s (BF16-rounded, into ``po``); the corrections
        leave from registers into ``u_dst``."""
        T: cutlass.Constexpr = self.tokens
        index = couplings(T)
        u = [Float32(0) for _ in range(T)]
        vs = [v_s[b, t, own] for t in range(T)]
        for t in cutlass.range_constexpr(T):
            dot = values[t]
            for s in cutlass.range_constexpr(t):
                dot = fma(ce_s[b, index[("c", t, s)]], u[s], dot)
            u[t] = (vs[t] - dot) * betas[j, t]
            u_dst[t, own] = u[t]
        for t in cutlass.range_constexpr(T):
            o = values[T + t]
            for s in cutlass.range_constexpr(t + 1):
                o = fma(ce_s[b, index[("e", t, s)]], u[s], o)
            po[t] = Float32(BFloat16(o))

    @cute.jit
    def finish(self, args, c, sums, at_sum, po, pg, pm, nw):
        """Gated RMSNorm of the group's previous item (``c``: this thread's channel,
        its scan row: c itself up to q4, scan_halves' permutation above) from its squares' block sums sums[at_sum + t] (synced by the caller);
        the outputs leave from registers."""
        eps, output = args[15], args[10]
        T: cutlass.Constexpr = self.tokens
        token, head = pm[1], pm[2]
        for t in cutlass.range_constexpr(T):
            sumsq = self.quad_sum(sums[at_sum + t, None])
            variance = sumsq * (1.0 / 128) + eps
            rstd = reciprocal(sqrt_approx(variance))
            sigmoid_gate = triton_sigmoid(pg[t])
            result = po[t] * rstd * nw * sigmoid_gate
            output[token + t, head * K + c] = BFloat16(result)

    @cute.jit
    def finish_alone(self, args, g, c, sums, po, pg, pm, nw, col):
        """``finish`` with its own block reduction (no prep to ride on)."""
        T: cutlass.Constexpr = self.tokens
        count: cutlass.Constexpr = 2 if T <= 2 else (4 if T <= 4 else 8)
        terms = [Float32(0) for _ in range(count)]
        for t in cutlass.range_constexpr(T):
            terms[t] = po[t] * po[t]
        self.block_sums(terms, sums, c, count)
        self.group_sync(g)
        self.finish(args, col, sums, 0, po, pg, pm, nw)
        self.group_sync(g)  # sums free again

    @cute.jit
    def block_sums_rounds(self, values, sums, p, count: cutlass.Constexpr):
        """``block_sums`` for more than 32 values: rounds of 32 (sums[32 r + i, warp]).
        Every value's reduction tree is the 32-value one (lanes xor 16, 8, 4, 2, 1),
        whatever its index, so each total is the q <= 4 build's bit for bit."""
        lane, warp = p % 32, p // 32
        for r in cutlass.range_constexpr(count // 32):
            chunk = [values[32 * r + i] for i in range(32)]
            total, selected = scatter_reduce(chunk, lane, 32)
            sums[32 * r + selected, warp] = total

    # ------------------------------------------------------------------- prep
    def bulk_head_constants(self, args):
        """The conv weight and dt rows bulk-copy straight into the head table (FP32,
        no conv bias)."""
        return (
            not self.has_bias
            and args[1].element_type == Float32
            and args[7].element_type == Float32
        )

    @cute.jit
    def head_constants(self, hc, args, head, c):
        """Per-head operands of channel c into hc[:, c]: conv weights [3, 4], biases
        [3], exp(A_log), dt."""
        weights, bias, a_log, dt_bias = args[1], args[2], args[6], args[7]
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        consts = hc[None, c]
        for part in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(4):
                consts[4 * part + i] = Float32(weights[part, i, head * K + c])
            if cutlass.const_expr(self.has_bias):
                consts[12 + part] = Float32(bias[part * dim + head * K + c])
            else:
                consts[12 + part] = Float32(0)
        consts[A_EXP] = triton_exp(Float32(a_log[head]))
        consts[DT] = Float32(dt_bias[head * K + c])

    @cute.jit
    def prep_item(self, args, g, c, j, slot, pending, offset, raw, hc, buffers, ops,
                  sums_g, kgo, pt, po):  # fmt: skip
        """Item j's scan operands into the group's block (thread c = channel / key c):
        the 2q scan vectors G_t k_t, G_t q_t (G_t: the decays of tokens 0 .. t), the
        token couplings c_ts = k_s R_ts k_t, e_ts = k_s R_ts q_t (R_ts: the decays of
        tokens s + 1 .. t), v, the fold weights w_t = kn_t P_t and D_n; its new key /
        raw-gate records into ``kgo`` and its window into the staged one. One block
        reduction for the q / k norms, the couplings and the recorded-key norms.
        Shared memory only."""
        lower_bound, scale = args[13], args[14]
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        convs, xs, gs, kgs, uss = buffers
        (sums,) = sums_g
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        FOLD: cutlass.Constexpr = self.folds(args)
        blk = g
        conv_type: cutlass.Constexpr = convs.element_type
        consts = hc[None, c]
        index = couplings(T)
        q0 = Int64(0)
        if cutlass.const_expr(self.trace):
            q0 = clock64()
        if slot > 0:
            # The pending commit's records (staged before this verify writes the new
            # ones); tokens t >= n fold as zeros (+0, or -0 corrections when nothing
            # is pending: the fold's identity).
            masked = Float32(0)
            if pending == 0:
                masked = Float32(-0.0)
            old_keys = zeros(T, Float32)
            old_gates = zeros(T, Float32)
            for t in cutlass.range_constexpr(T):
                fu_s[blk, t, c] = masked
                if cutlass.const_expr(FOLD):  # noqa: SIM102
                    if t < pending:
                        old_keys[t] = Float32(kgs[t, c])
                        old_gates[t] = Float32(kgs[t, K + c])
                        fu_s[blk, t, c] = uss[t, c]
            a = consts[A_EXP]
            dt = consts[DT]

            # Convolution + SiLU of q, k, v (thread c: channel c of each part).
            values = cute.make_rmem_tensor((3, T), Float32)
            for part in cutlass.range_constexpr(3):
                taps = cute.make_rmem_tensor(3, conv_type)
                for t in cutlass.range_constexpr(3):
                    taps[t] = convs[part, c * self.window + offset + t]
                inputs = [xs[t, part, c] for t in range(T)]
                for t in cutlass.range_constexpr(T):
                    accum = Float32(0)
                    if cutlass.const_expr(self.has_bias):
                        accum = consts[12 + part]
                    for i in cutlass.range_constexpr(4):
                        tap = t + i
                        term = taps[tap] if tap < 3 else inputs[tap - 3]
                        accum = fma(Float32(term), consts[4 * part + i], accum)
                    values[part, t] = self.silu(accum)
                row = c * self.window  # the window [tap 1, tap 2, x_0, ..., x_q-1]
                convs[part, row] = conv_type(taps[1])
                convs[part, row + 1] = conv_type(taps[2])
                for t in cutlass.range_constexpr(T):
                    convs[part, row + 2 + t] = conv_type(inputs[t])
            if cutlass.const_expr(self.trace):
                q1 = clock64()
            decays = zeros(T, Float32)
            for t in cutlass.range_constexpr(T):
                v_s[blk, t, c] = values[2, t]
                g_s[blk, t, c] = gs[t, c]
                kgo[t, c] = BFloat16(values[1, t])
                raw_g = raw[j, t, c]
                kgo[t, K + c] = raw_g
                decays[t] = self.decay(a, Float32(raw_g), dt, lower_bound)
            # The decay products of this key: G_t (tokens 0 .. t), R_ts (s + 1 .. t).
            cumulative = [decays[0]]
            for t in cutlass.range_constexpr(1, T):
                cumulative.append(mul_rn(cumulative[t - 1], decays[t]))
            ratio = {}
            for t in cutlass.range_constexpr(1, T):
                ratio[(t, t - 1)] = decays[t]
                for s in cutlass.range_constexpr(t - 1):
                    ratio[(t, s)] = mul_rn(ratio[(t - 1, s)], decays[t])

            # The pending commit's recorded-key squares and decays.
            old_decays = zeros(T, Float32)
            if cutlass.const_expr(FOLD):  # noqa: SIM102
                if pending > 0:
                    for t in cutlass.range_constexpr(T):
                        old_decays[t] = self.decay(a, old_gates[t], dt, lower_bound)
            if cutlass.const_expr(self.trace):
                q2 = clock64()

            # One block reduction: q / k squares, the couplings (raw q / k), the
            # recorded-key squares.
            NS: cutlass.Constexpr = sums_count(T)
            terms = [Float32(0) for _ in range(NS)]
            for t in cutlass.range_constexpr(T):
                terms[2 * t] = values[0, t] * values[0, t]
                terms[2 * t + 1] = values[1, t] * values[1, t]
                terms[NV + T * T + t] = old_keys[t] * old_keys[t]
                terms[NV + T * T + T + t] = po[t] * po[t]  # the previous outputs
            for i in cutlass.range_constexpr(len(index)):
                key = list(index)[i]
                kind, t, s = key
                weighted = values[1, s]
                if cutlass.const_expr(s < t):
                    weighted = mul_rn(weighted, ratio[(t, s)])
                other = values[1, t] if kind == "c" else values[0, t]
                terms[NV + i] = mul_rn(weighted, other)
            if cutlass.const_expr(NS == SUMS):
                self.block_sums(terms, sums, c, SUMS)
            else:
                self.block_sums_rounds(terms, sums, c, NS)
            if cutlass.const_expr(self.trace):
                q3 = clock64()
            fence_async_shared()  # the window and records leave by bulk stores
            self.group_sync(g)  # the partial sums
            if cutlass.const_expr(self.trace):
                q4 = clock64()
            fq = [Float32(0) for _ in range(T)]
            fk = [Float32(0) for _ in range(T)]
            for t in cutlass.range_constexpr(T):
                qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                fq[t] = reciprocal(sqrt_approx(qn))
                fk[t] = reciprocal(sqrt_approx(kn))
            # The scan vectors (swizzled: [chunk][e][h][lane][4], key c =
            # 4 (lane + 8 chunk) + e).
            NVP: cutlass.Constexpr = vectors(T)
            vec = [Float32(0) for _ in range(NVP)]  # the pad stays +0
            for t in cutlass.range_constexpr(T):
                q_t = mul_rn(mul_rn(values[0, t], fq[t]), scale)
                k_t = mul_rn(values[1, t], fk[t])
                vec[t] = mul_rn(cumulative[t], k_t)
                vec[T + t] = mul_rn(cumulative[t], q_t)
            chunk, lane_in, e = c // (4 * LANES), (c // 4) % LANES, c % 4
            for h in cutlass.range_constexpr(NVP // 4):
                pos = (((chunk * 4 + e) * (NVP // 4) + h) * LANES + lane_in) * 4
                store_shared_f32x4(at(xv, blk, pos), [vec[4 * h + i] for i in range(4)])
            if cutlass.const_expr(self.trace):
                q5 = clock64()
            # The couplings, normalized (thread i: coupling i; one select chain, no
            # per-lane branches).
            norm = Float32(0)
            if cutlass.const_expr(self.roll & 4):
                # Coupling c's (kind, t, s) decoded (token t's block starts at t^2: its
                # t key-key then t + 1 key-query couplings), its operands selected by
                # token: the same products as below, ~4x fewer instructions.
                tc = Int32(0)
                for u in cutlass.range_constexpr(1, T):
                    tc = tc + Int32(cutlass.select_(c >= u * u, Int32(1), Int32(0)))
                rc = c - tc * tc
                sc = rc
                if rc >= tc:  # a key-query coupling
                    sc = rc - tc
                fk_s, fk_t, fq_t = Float32(0), Float32(0), Float32(0)
                for t in cutlass.range_constexpr(T):
                    fk_s = Float32(cutlass.select_(sc == t, fk[t], fk_s))
                    fk_t = Float32(cutlass.select_(tc == t, fk[t], fk_t))
                    fq_t = Float32(cutlass.select_(tc == t, fq[t], fq_t))
                norm = mul_rn(fk_s, fk_t)
                if rc >= tc:
                    norm = mul_rn(fk_s, mul_rn(fq_t, scale))
            else:
                for i in cutlass.range_constexpr(len(index)):
                    key = list(index)[i]
                    kind, t, s = key
                    norm_i = mul_rn(fk[s], fk[t])
                    if cutlass.const_expr(kind == "e"):
                        norm_i = mul_rn(fk[s], mul_rn(fq[t], scale))
                    norm = Float32(cutlass.select_(c == i, norm_i, norm))
            if c < len(index):
                ce_s[blk, c] = mul_rn(self.quad_sum(sums[NV + c, None]), norm)
            # The commit's reverse token loop over this key: w_t = kn_t P_t (P_t:
            # the final_decay token t's update is scaled by) and D_n; tokens t >= n
            # are zeros.
            final_decay = Float32(1)
            for i in cutlass.range_constexpr(T):
                t = T - 1 - i
                w_t = Float32(0)
                if cutlass.const_expr(FOLD):  # noqa: SIM102
                    if t < pending:
                        total = self.quad_sum(sums[NV + T * T + t, None]) + 1.0e-6
                        kn_t = mul_rn(rsqrt_ftz(total), old_keys[t])
                        w_t = mul_rn(kn_t, final_decay)
                        final_decay = mul_rn(final_decay, old_decays[t])
                fw_s[blk, t, c] = w_t
            fd_s[blk, c] = final_decay
            if cutlass.const_expr(self.trace):
                q6 = clock64()
                pt[0] = pt[0] + (q1 - q0)
                pt[1] = pt[1] + (q2 - q1)
                pt[2] = pt[2] + (q3 - q2)
                pt[3] = pt[3] + (q4 - q3)
                pt[4] = pt[4] + (q5 - q4)
                pt[5] = pt[5] + (q6 - q5)


def couplings(tokens):
    """Index of each token coupling: ("c", t, s) for s < t (key-key), ("e", t, s)
    for s <= t (key-query)."""
    index = {}
    for t in range(tokens):
        for s in range(t):
            index[("c", t, s)] = len(index)
        for s in range(t + 1):
            index[("e", t, s)] = len(index)
    return index
