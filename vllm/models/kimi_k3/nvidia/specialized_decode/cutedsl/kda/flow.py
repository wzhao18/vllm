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
   Triton commit's math), the committed checkpoint goes back into the slot and out
   by one bulk store (L2 evict-first), the q tokens are verified and the outputs
   RMS-normed and gated; corrections and outputs leave by bulk stores. Each loaded
   4-key vector chunk feeds 8 rows: the shared-memory data path (128 B per clock,
   whatever the broadcast) carries ~3K cycles per item, not the ~7K of a 2-row
   layout. A row's dot product: KdaWS's per-thread partials and 8-lane transposed
   butterfly (lane l ends with row 8 b + l); its correction goes to the store
   buffer, from which the lane group reads its 8 rows' corrections back.

State stream: a ring of three page slots per CTA; item j uses slot j % 3, and once
item j's checkpoint store has read its slot, the slot is refilled with item j + 3.

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
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    mma_into,
    mul2,
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
    pair_total,
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
SLOTS = 3  # the CTA's state page ring
MAX_ITEMS = 16  # items per CTA (tables, raw gates)
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
# Tuning: a warp sync after every FOLD_FENCE fold chunks (bounds how far ptxas
# hoists the independent chunks' loads into spills); verify passes load the next
# chunk's vectors before using the current ones (PREFETCH).
FOLD_FENCE = 1
PREFETCH = False
# Verify passes: token t's update pass fused with token t + 1's decay pass (one pass
# over the state per token, two independent accumulator sets; same per-element
# operations and order).
FUSED = True
# Item table columns.
SLOT, HEAD, REQUEST, CONV_SLOT, CONV_OFFSET, PENDING, RECORD = range(7)
META = 8
# Per-head channel constants: conv weights [3, 4], biases [3], exp(A_log), dt.
CONSTS = 17
A_EXP, DT = 15, 16


def group_bytes(tokens, window, conv_size):
    """One group's operand block, staging and store buffers (bytes)."""
    block = 7 * tokens * K * 4 + K * 4 + tokens * K * 2  # q k v d kn P u, D, gate
    staging = 4 * tokens * K * 2 + 3 * K * window * conv_size + tokens * K * 8
    stores = tokens * 2 * K * 2 + tokens * K * 4 + tokens * K * 2
    return block + staging + stores + 16 * 4 * 4 + 2 * 8 * 4 * 4


def shared_bytes(tokens, conv_dim_first, window, conv_size, stages=1):
    pages = SLOTS * PAGE_BYTES
    heads = 2 * CONSTS * K * 4
    gates = MAX_ITEMS * tokens * K * 2
    tables = MAX_ITEMS * (META + 8) * 4
    groups = GROUPS * group_bytes(tokens, window, conv_size)
    return pages + heads + gates + tables + groups + 64 * 8 + 4096  # (+ alignment)


def smem_budget(device):
    """Shared memory one KdaFlow CTA may use on ``device``."""
    props = torch.cuda.get_device_properties(device)
    limit = props.shared_memory_per_block_optin
    if (props.major, props.minor) == (10, 7):
        limit = max(limit, OVERSIZED_BUDGET)
    return limit


def max_stages(tokens, conv_dim_first, window, conv_size, limit):
    """1 when KdaFlow runs this layout (dim-first convolution window) within
    ``limit`` bytes of shared memory, else 0."""
    if not conv_dim_first:
        return 0
    return int(shared_bytes(tokens, conv_dim_first, window, conv_size) <= limit)


def max_items_per_cta(requests):
    """KdaFlow tables the weights of two heads (a CTA's head-major item range may span
    at most two heads) and MAX_ITEMS items."""
    return min(requests + 1, MAX_ITEMS)


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l", ()


@asm()
def store_shared_f32x4(pointer, values):
    return "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f,~{memory}", (
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
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


class KdaFlow(KdaWS):
    """KdaWS's operands and semantics for RecoverSSM record steps (``tokens`` q > 1,
    records always written, dim-first convolution window); with the deferred
    operands (``record_rows``, ``pending`` after the records) every item folds its
    pending commit first. ``stages``: 1 (the page ring has SLOTS slots); ``ctas``
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
    ):
        assert 2 <= tokens <= 8 and stages == 1 and conv_dim_first
        super().__init__(
            heads, tokens, True, gate_mode, has_bias, conv_dim_first, window, ctas
        )
        self.stages = stages
        self.trace, self.hang, self.packed = TRACE, HANG, PACKED
        self.fold_fence, self.prefetch, self.fused = FOLD_FENCE, PREFETCH, FUSED
        self.skip = tuple(SKIP)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

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
        if cutlass.const_expr(len(args) > 20):
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
    def load_page(self, args, pages, s, table, j, state_full):
        """Bulk-copy table item j's state page into page slot ``s`` (or complete the
        phase of a padded row)."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = state_full + s
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, PAGE_BYTES)
            source = args[9][Int64(slot), None, None, None]
            for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
                rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
                copy_g2s(
                    at(pages, s, i * rows * K),
                    at(source, head, i * rows, 0),
                    COPY_BYTES,
                    barrier,
                )
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def stage_item(self, args, table, j, buffers, barrier):
        """Bulk-copy table item j's inputs (one thread): q / k / v and output gate
        rows, the convolution window and, with a pending commit, its records."""
        x, conv_state, gate = args[0], args[3], args[11]
        convs, xs, gs, kgs, uss = buffers
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        part_bytes: cutlass.Constexpr = K * self.window * size
        slot, head, pending = table[j, SLOT], table[j, HEAD], table[j, PENDING]
        if slot > 0:
            rows: cutlass.Constexpr = 4 * T * K * 2 + 3 * part_bytes
            records: cutlass.Constexpr = T * 2 * K * 2 + T * K * 4
            total = Int32(rows)
            if pending > 0:
                total = total + records
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, total)
            token = table[j, REQUEST] * T
            for t in cutlass.range_constexpr(T):
                for part in cutlass.range_constexpr(3):
                    copy_g2s(at(xs, t, part, 0), at(x, token + t, part * dim + head * K),
                             K * 2, barrier)  # fmt: skip
                copy_g2s(at(gs, t, 0), at(gate, token + t, head * K), K * 2, barrier)
            page = conv_state[Int64(table[j, CONV_SLOT]), None, None]
            for part in cutlass.range_constexpr(3):
                copy_g2s(at(convs, part, 0), at(page, part * dim + head * K, 0),
                         part_bytes, barrier)  # fmt: skip
            if pending > 0:
                rec = table[j, RECORD]
                copy_g2s(at(kgs, 0, 0), at(args[19], rec, head, 0, 0), T * 2 * K * 2,
                         barrier)  # fmt: skip
                copy_g2s(at(uss, 0, 0), at(args[18], rec, head, 0, 0), T * K * 4,
                         barrier)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
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
        pages = alloc(smem, Float32, (SLOTS, K * K), align=1024)
        hcs = alloc(smem, Float32, (2, CONSTS, K))  # per head slot, channel
        raw = alloc(smem, BFloat16, (MAX_ITEMS, T, K), align=16)  # raw gates
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        betas = alloc(smem, Float32, (MAX_ITEMS, 8))  # sigmoid(beta) per token
        qkvd = alloc(smem, Float32, (4, GROUPS, T, K), align=128)  # q, k, v, decay
        q_s, k_s = qkvd[0, None, None, None], qkvd[1, None, None, None]
        v_s, d_s = qkvd[2, None, None, None], qkvd[3, None, None, None]
        fk_s = alloc(smem, Float32, (GROUPS, T, K))  # kn_t of the pending commit
        fp_s = alloc(smem, Float32, (GROUPS, T, K))  # P_t
        fu_s = alloc(smem, Float32, (GROUPS, T, K))  # the old corrections u_t
        fd_s = alloc(smem, Float32, (GROUPS, K))  # D_n
        g_s = alloc(smem, BFloat16, (GROUPS, T, K))
        convs = alloc(smem, conv_type, (GROUPS, 3, K * self.window), align=128)
        xs = alloc(smem, BFloat16, (GROUPS, T, 3, K), align=128)
        gs = alloc(smem, BFloat16, (GROUPS, T, K), align=128)
        kgs = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)
        uss = alloc(smem, Float32, (GROUPS, T, K), align=128)
        kgo = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)  # new key/gate
        uo = alloc(smem, Float32, (GROUPS, T, K), align=128)  # new corrections
        oo = alloc(smem, BFloat16, (GROUPS, T, K), align=128)  # outputs
        sums = alloc(smem, Float32, (GROUPS, 16, 4))
        fsums = alloc(smem, Float32, (GROUPS, 8, 4))  # recorded k squares per warp
        out_sums = alloc(smem, Float32, (GROUPS, 8, 4))
        bars = smem.allocate(cutlass.Uint64, SLOTS + GROUPS + 2)
        state_full, stage_full = bars, bars + SLOTS
        fb_full, fa_full = bars + SLOTS + GROUPS, bars + SLOTS + GROUPS + 1
        # Prologue scratch: the F_B rows of the CTA's heads in the last page slot (its
        # page is loaded after the gate projection), the F_A rows of its items over
        # the q / k / v / decay blocks (unused until the first prep).
        fbs = cute.make_tensor(
            cute.recast_ptr(at(pages, SLOTS - 1, 0), dtype=BFloat16),
            cute.make_layout((2, K, K), stride=(K * K, K, 1)),
        )
        fas = cute.make_tensor(
            cute.recast_ptr(at(qkvd, 0, 0, 0, 0), dtype=BFloat16),
            cute.make_layout((MAX_ITEMS, T, K), stride=(T * K, K, 1)),
        )

        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):
                cute.arch.mbarrier_init(state_full + s, 1)
            for g in cutlass.range_constexpr(GROUPS):
                cute.arch.mbarrier_init(stage_full + g, 1)
            cute.arch.mbarrier_init(fb_full, 1)
            cute.arch.mbarrier_init(fa_full, 1)
            cute.arch.mbarrier_init_fence()
        # The page sources (the host's slot table: no predecessor output).
        if tid < count:
            head, _, slot, _, _ = self.item_slots(args, first + tid)
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
        cute.arch.sync_warp()
        cute.arch.barrier()
        h0 = first // requests
        h1 = (first + count - 1) // requests
        heads = h1 - h0 + 1
        # Before the PDL wait: the first pages (this layer's state), the F_B rows
        # and channel weights of the CTA's heads.
        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS - 1):
                if s < count:
                    self.load_page(args, pages, s, table, s, state_full)
            cute.arch.mbarrier_arrive_and_expect_tx(fb_full, heads * K * K * 2)
            for h in cutlass.range_constexpr(2):
                if h < heads:
                    copy_g2s(at(fbs, h, 0, 0), at(f_b, (h0 + h) * K, 0), K * K * 2,
                             fb_full)  # fmt: skip
        cute.arch.sync_warp()
        hslot = tid // K
        if hslot < heads:
            self.head_constants(hcs[hslot, None, None], args, h0 + hslot, tid % K)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid < count:
            _, request, _, conv_slot, conv_offset, pending, rec = self.item_info(
                args, first + tid
            )
            table[tid, REQUEST] = request
            table[tid, CONV_SLOT] = conv_slot
            table[tid, CONV_OFFSET] = conv_offset
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
            head_t = table[tid, HEAD]
            for t in cutlass.range_constexpr(T):
                betas[tid, t] = triton_sigmoid(Float32(beta[request * T + t, head_t]))
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(fa_full, count * T * K * 2)
        cute.arch.sync_warp()
        cute.arch.barrier()  # the table; fa_full's byte count
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        buffers = (
            convs[g, None, None], xs[g, None, None, None], gs[g, None, None],
            kgs[g, None, None], uss[g, None, None],
        )  # fmt: skip
        if c == 0:  # the group's first item's inputs
            if g < count:
                self.stage_item(args, table, g, buffers, stage_full + g)
        cute.arch.sync_warp()
        if tid < count * T:
            item, t = tid // T, tid % T
            row = table[item, REQUEST] * T + t
            copy_g2s(at(fas, item, t, 0), at(f_a, row, 0), K * 2, fa_full)
        cute.arch.sync_warp()
        self.bwait(fb_full, 0, "fb_full")
        self.bwait(fa_full, 0, "fa_full")
        self.gate_projection(tid, count, h0, table, fbs, fas, raw)
        cute.arch.sync_warp()
        cute.arch.barrier()  # raw gates; the last page slot and the blocks are free
        if tid == 0:
            if SLOTS - 1 < count:
                self.load_page(args, pages, SLOTS - 1, table, SLOTS - 1, state_full)
        cute.arch.sync_warp()
        ops = (q_s, k_s, v_s, d_s, g_s, fk_s, fp_s, fu_s, fd_s)
        outs = (kgo[g, None, None], uo[g, None, None], oo[g, None, None])
        sums_g = (sums[g, None, None], fsums[g, None, None], out_sums[g, None, None])
        self.group_items(args, g, c, count, h0, pages, table, betas, raw, hcs,
                         buffers, ops, outs, sums_g, state_full,
                         stage_full + g)  # fmt: skip

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
        PER: cutlass.Constexpr = MAX_ITEMS // 2
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
                    buffers, ops, outs, sums_g, state_full, stage_full):  # fmt: skip
        q_s, k_s, v_s, d_s, g_s, fk_s, fp_s, fu_s, fd_s = ops
        kgo, uo, oo = outs
        state, output = args[9], args[10]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        lane_in, base = c % LANES, (c // LANES) * ROWS  # rows base + [0, 8)
        nw = args[12][c]
        mine = (count - g + 1) // 2  # this group's items: g, g + 2, ...
        tr = zeros(6, Int64)  # TRACE: total, prep, state wait, fold, verify, epi
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
            if cutlass.const_expr("prep" not in self.skip):
                self.prep_item(args, g, c, j, slot, pending, table[j, CONV_OFFSET],
                               raw, hcs[head - h0, None, None], buffers, ops,
                               sums_g, kgo)  # fmt: skip
            self.group_sync(g)  # the block is complete; the staging buffers are read
            if c == 0:
                if slot > 0:  # the new key / gate records and convolution window
                    convs = buffers[0]
                    part_bytes: cutlass.Constexpr = (
                        K * self.window * convs.element_type.width // 8
                    )
                    store_bulk(at(args[19], rec, head, 0, 0), at(kgo, 0, 0),
                               T * 2 * K * 2)  # fmt: skip
                    page = args[3][Int64(table[j, CONV_SLOT]), None, None]
                    for part in cutlass.range_constexpr(3):
                        store_bulk(at(page, part * dim + head * K, 0),
                                   at(convs, part, 0), part_bytes)  # fmt: skip
                    bulk_commit()
                    wait_bulk_reads()  # the staging buffers are free
                if jj + 1 < mine:  # the group's next item's inputs
                    self.stage_item(args, table, j + GROUPS, buffers, stage_full)
            cute.arch.sync_warp()
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[1] = tr[1] + (tb - ta)
            s = j % SLOTS
            self.bwait(state_full + s, (j // SLOTS) % 2, "state_full")
            if cutlass.const_expr(self.trace):
                tc = clock64()
                tr[2] = tr[2] + (tc - tb)
            if slot > 0:
                hist = cute.make_rmem_tensor((ROWS, CHUNKS, 4), Float32)
                for rr in cutlass.range_constexpr(ROWS):
                    for ch in cutlass.range_constexpr(CHUNKS):
                        values = load_shared_f32x4(
                            at(pages, s, (base + rr) * K + 4 * (lane_in + LANES * ch))
                        )
                        for e in cutlass.range_constexpr(4):
                            hist[rr, ch, e] = values[e]
                if cutlass.const_expr("fold" not in self.skip):
                    self.fold(hist, g, pending, lane_in, base, fk_s, fp_s, fu_s, fd_s)
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):
                    stored = pending < 0
                if stored:  # the committed checkpoint back into the slot
                    for rr in cutlass.range_constexpr(ROWS):
                        for ch in cutlass.range_constexpr(CHUNKS):
                            store_shared_f32x4(
                                at(pages, s,
                                   (base + rr) * K + 4 * (lane_in + LANES * ch)),
                                [hist[rr, ch, e] for e in range(4)],
                            )  # fmt: skip
                    fence_async_shared()
                self.group_sync(g)  # the slot is read (and rewritten) by every thread
                if c == 0:
                    if stored:  # one bulk store of the committed checkpoint
                        destination = state[Int64(slot), None, None, None]
                        for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
                            n_rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
                            store_bulk_evict(
                                at(destination, head, i * n_rows, 0),
                                at(pages, s, i * n_rows * K),
                                COPY_BYTES,
                            )
                        bulk_commit()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.trace):
                    td = clock64()
                    tr[3] = tr[3] + (td - tc)
                if cutlass.const_expr(self.fused):
                    release = (args, c, j, count, pages, table, state_full, stored)
                    if cutlass.const_expr("verify" not in self.skip):
                        self.verify_fused(hist, g, betas, j, lane_in, base, q_s, k_s,
                                          v_s, d_s, uo, release)  # fmt: skip
                    else:
                        self.release_slot(*release)
                else:
                    for t in cutlass.range_constexpr(T):
                        if cutlass.const_expr("verify" not in self.skip):
                            self.verify_token(hist, t, g, betas[j, t], lane_in, base,
                                              q_s, k_s, v_s, d_s, uo)  # fmt: skip
                        if cutlass.const_expr(t == min(1, T - 1)):
                            self.release_slot(args, c, j, count, pages, table,
                                              state_full, stored)  # fmt: skip
                if cutlass.const_expr(self.trace):
                    te = clock64()
                    tr[4] = tr[4] + (te - td)
                if cutlass.const_expr("epi" not in self.skip):
                    self.epilogue(args, g, c, v_s, g_s, sums_g[2], nw, oo)
                fence_async_shared()  # corrections and outputs leave by bulk stores
                self.group_sync(g)
                if c == 0:
                    store_bulk(at(args[18], rec, head, 0, 0), at(uo, 0, 0), T * K * 4)
                    for t in cutlass.range_constexpr(T):
                        store_bulk(at(output, token + t, head * K), at(oo, t, 0), K * 2)
                    bulk_commit()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.trace):
                    tr[5] = tr[5] + (clock64() - te)
            else:
                self.release_slot(args, c, j, count, pages, table, state_full, False)
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("TRACE cta {} g {} items {} total {} prep {} wstate {} fold {} verify {} epi {}", cta, g, mine, clock64() - t0, tr[1], tr[2], tr[3], tr[4], tr[5])  # noqa: E501

    @cute.jit
    def release_slot(self, args, c, j, count, pages, table, state_full, stored):
        """Item j's slot is free once its checkpoint store has read it: thread 0
        refills it with item j + SLOTS's page."""
        if c == 0:
            if stored:
                wait_bulk_reads()
            if j + SLOTS < count:
                self.load_page(args, pages, j % SLOTS, table, j + SLOTS, state_full)
        cute.arch.sync_warp()

    # ------------------------------------------------------------------- scan
    def pmul(self, a, b):
        """Lanewise a * b of FP32 pairs (packed, or two FMULs: the same bits)."""
        if self.packed:
            return mul2(a, b)
        return (mul_rn(a[0], b[0]), mul_rn(a[1], b[1]))

    def pfma(self, a, b, c):
        """Lanewise a * b + c of FP32 pairs (packed, or two FFMAs: the same bits)."""
        if self.packed:
            return fma2(a, b, c)
        return (fma(a[0], b[0], c[0]), fma(a[1], b[1], c[1]))

    @cute.jit
    def fold(self, hist, b, pending, lane_in, base, keys, suffix, old, decay):
        """The stock commit of the pending tokens into this thread's state elements:
        fc = fma(u_t * kn_t, P_t, fc) over t = q - 1 .. 0 (tokens t >= n are zeros),
        then S = fma(S_0, D_n, fc); four rows at a time (registers)."""
        T: cutlass.Constexpr = self.tokens
        HALF: cutlass.Constexpr = ROWS // 2
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        for half in cutlass.range_constexpr(2):
            us = cute.make_rmem_tensor((HALF, T), Float32)
            for rr in cutlass.range_constexpr(HALF):
                for t in cutlass.range_constexpr(T):
                    us[rr, t] = old[b, t, base + HALF * half + rr]
            for ch in cutlass.range_constexpr(CHUNKS):
                key = 4 * (lane_in + LANES * ch)
                d4 = load_shared_f32x4(at(decay, b, key))
                fc = cute.make_rmem_tensor((HALF, 4), Float32)
                for rr in cutlass.range_constexpr(HALF):
                    for e in cutlass.range_constexpr(4):
                        fc[rr, e] = start
                for i in cutlass.range_constexpr(T):
                    t = T - 1 - i
                    k4 = load_shared_f32x4(at(keys, b, t, key))
                    p4 = load_shared_f32x4(at(suffix, b, t, key))
                    for rr in cutlass.range_constexpr(HALF):
                        u = us[rr, t]
                        for p in cutlass.range_constexpr(2):
                            update = self.pmul((u, u), (k4[2 * p], k4[2 * p + 1]))
                            acc = (fc[rr, 2 * p], fc[rr, 2 * p + 1])
                            acc = self.pfma(update, (p4[2 * p], p4[2 * p + 1]), acc)
                            fc[rr, 2 * p], fc[rr, 2 * p + 1] = acc
                for rr in cutlass.range_constexpr(HALF):
                    r = HALF * half + rr
                    for p in cutlass.range_constexpr(2):
                        pair = (hist[r, ch, 2 * p], hist[r, ch, 2 * p + 1])
                        pair = self.pfma(pair, (d4[2 * p], d4[2 * p + 1]),
                                         (fc[rr, 2 * p], fc[rr, 2 * p + 1]))  # fmt: skip
                        hist[r, ch, 2 * p], hist[r, ch, 2 * p + 1] = pair
                if cutlass.const_expr(ch % self.fold_fence == self.fold_fence - 1):
                    cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)

    @cute.jit
    def row_sum8(self, partials, lane_in):
        """The eight rows' partials of 8 lanes -> the full sum of row ``lane_in``
        (KdaWS.row_sum's tree: lanes ^4, then ^2, then ^1)."""
        upper = (lane_in & 4) != 0
        a = [Float32(0) for _ in range(4)]
        for i in cutlass.range_constexpr(4):
            send = partials[i] if upper else partials[i + 4]
            keep = partials[i + 4] if upper else partials[i]
            a[i] = keep + cute.arch.shuffle_sync_bfly(send, 4)
        upper = (lane_in & 2) != 0
        bb = [Float32(0) for _ in range(2)]
        for i in cutlass.range_constexpr(2):
            send = a[i] if upper else a[i + 2]
            keep = a[i + 2] if upper else a[i]
            bb[i] = keep + cute.arch.shuffle_sync_bfly(send, 2)
        upper = (lane_in & 1) != 0
        send = bb[0] if upper else bb[1]
        keep = bb[1] if upper else bb[0]
        return keep + cute.arch.shuffle_sync_bfly(send, 1)

    @cute.jit
    def verify_token(self, hist, t, b, beta_t, lane_in, base, q_s, k_s, v_s, d_s,
                     uo):  # fmt: skip
        """Verify token t from the state in registers: the decay pass (S *= d_t,
        dot = S k_t), lane l's correction of row base + l (into the store buffer,
        whence the lane group reads its rows' corrections), the update pass
        (S += k_t u, out = S q_t), lane l's BF16 out_t (into v_s)."""
        o_s = v_s
        own = base + lane_in

        def chunk_of(buffer, ch):
            return load_shared_f32x4(at(buffer, b, t, 4 * (lane_in + LANES * ch)))

        dots = zeros((ROWS, 2, 2), Float32)
        for ch in cutlass.range_constexpr(CHUNKS):
            d4, k4 = chunk_of(d_s, ch), chunk_of(k_s, ch)
            for rr in cutlass.range_constexpr(ROWS):
                for pp in cutlass.range_constexpr(2):
                    pair = (hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1])
                    pair = self.pmul(pair, (d4[2 * pp], d4[2 * pp + 1]))
                    hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1] = pair
                    acc = (dots[rr, pp, 0], dots[rr, pp, 1])
                    acc = self.pfma(pair, (k4[2 * pp], k4[2 * pp + 1]), acc)
                    dots[rr, pp, 0], dots[rr, pp, 1] = acc
        partials = [pair_total(dots[rr, None, None]) for rr in range(ROWS)]
        if cutlass.const_expr("reduce" in self.skip):  # keeps every row live
            dot = partials[0]
            for rr in cutlass.range_constexpr(1, ROWS):
                dot = dot + partials[rr]
        else:
            dot = self.row_sum8(partials, lane_in)
        uo[t, own] = (v_s[b, t, own] - dot) * beta_t
        cute.arch.sync_warp()  # the lane group's corrections are in uo
        u_lo = load_shared_f32x4(at(uo, t, base))
        u_hi = load_shared_f32x4(at(uo, t, base + 4))
        u = [u_lo[0], u_lo[1], u_lo[2], u_lo[3], u_hi[0], u_hi[1], u_hi[2], u_hi[3]]
        outs = zeros((ROWS, 2, 2), Float32)
        for ch in cutlass.range_constexpr(CHUNKS):
            k4, q4 = chunk_of(k_s, ch), chunk_of(q_s, ch)
            for rr in cutlass.range_constexpr(ROWS):
                for pp in cutlass.range_constexpr(2):
                    pair = (hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1])
                    kp = (k4[2 * pp], k4[2 * pp + 1])
                    pair = self.pfma(kp, (u[rr], u[rr]), pair)
                    hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1] = pair
                    acc = (outs[rr, pp, 0], outs[rr, pp, 1])
                    acc = self.pfma(pair, (q4[2 * pp], q4[2 * pp + 1]), acc)
                    outs[rr, pp, 0], outs[rr, pp, 1] = acc
        partials = [pair_total(outs[rr, None, None]) for rr in range(ROWS)]
        if cutlass.const_expr("reduce" in self.skip):
            o = partials[0]
            for rr in cutlass.range_constexpr(1, ROWS):
                o = o + partials[rr]
        else:
            o = self.row_sum8(partials, lane_in)
        o_s[b, t, own] = Float32(BFloat16(o))

    @cute.jit
    def correction(self, dots, t, b, beta_t, lane_in, base, v_s, uo):
        """Token t's corrections from the decay pass's partials: lane l's row
        base + l into the store buffer, then the lane group's 8 rows back."""
        own = base + lane_in
        partials = [pair_total(dots[rr, None, None]) for rr in range(ROWS)]
        dot = self.row_sum8(partials, lane_in)
        uo[t, own] = (v_s[b, t, own] - dot) * beta_t
        cute.arch.sync_warp()  # the lane group's corrections are in uo
        u_lo = load_shared_f32x4(at(uo, t, base))
        u_hi = load_shared_f32x4(at(uo, t, base + 4))
        return [u_lo[0], u_lo[1], u_lo[2], u_lo[3], u_hi[0], u_hi[1], u_hi[2],
                u_hi[3]]  # fmt: skip

    @cute.jit
    def verify_fused(self, hist, b, betas, j, lane_in, base, q_s, k_s, v_s, d_s, uo,
                     release):  # fmt: skip
        """The q verify tokens with one pass over the state per token: token t's
        update pass (S += k_t u_t, out_t = S q_t) fused with token t + 1's decay pass
        (S *= d_t+1, dot_t+1 = S k_t+1); each element sees KdaWS's operations in
        KdaWS's order."""
        T: cutlass.Constexpr = self.tokens
        o_s = v_s
        own = base + lane_in

        def chunk_of(buffer, t, ch):
            return load_shared_f32x4(at(buffer, b, t, 4 * (lane_in + LANES * ch)))

        dots = zeros((ROWS, 2, 2), Float32)
        for ch in cutlass.range_constexpr(CHUNKS):
            d4, k4 = chunk_of(d_s, 0, ch), chunk_of(k_s, 0, ch)
            for rr in cutlass.range_constexpr(ROWS):
                for pp in cutlass.range_constexpr(2):
                    pair = (hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1])
                    pair = self.pmul(pair, (d4[2 * pp], d4[2 * pp + 1]))
                    hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1] = pair
                    acc = (dots[rr, pp, 0], dots[rr, pp, 1])
                    acc = self.pfma(pair, (k4[2 * pp], k4[2 * pp + 1]), acc)
                    dots[rr, pp, 0], dots[rr, pp, 1] = acc
        u = self.correction(dots, 0, b, betas[j, 0], lane_in, base, v_s, uo)
        for t in cutlass.range_constexpr(T):
            NEXT: cutlass.Constexpr = t + 1 < T
            outs = zeros((ROWS, 2, 2), Float32)
            dots = zeros((ROWS, 2, 2), Float32)
            for ch in cutlass.range_constexpr(CHUNKS):
                k4, q4 = chunk_of(k_s, t, ch), chunk_of(q_s, t, ch)
                if cutlass.const_expr(NEXT):
                    d4n, k4n = chunk_of(d_s, t + 1, ch), chunk_of(k_s, t + 1, ch)
                for rr in cutlass.range_constexpr(ROWS):
                    for pp in cutlass.range_constexpr(2):
                        pair = (hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1])
                        kp = (k4[2 * pp], k4[2 * pp + 1])
                        pair = self.pfma(kp, (u[rr], u[rr]), pair)
                        acc = (outs[rr, pp, 0], outs[rr, pp, 1])
                        acc = self.pfma(pair, (q4[2 * pp], q4[2 * pp + 1]), acc)
                        outs[rr, pp, 0], outs[rr, pp, 1] = acc
                        if cutlass.const_expr(NEXT):
                            pair = self.pmul(pair, (d4n[2 * pp], d4n[2 * pp + 1]))
                            acc = (dots[rr, pp, 0], dots[rr, pp, 1])
                            acc = self.pfma(pair, (k4n[2 * pp], k4n[2 * pp + 1]), acc)
                            dots[rr, pp, 0], dots[rr, pp, 1] = acc
                        hist[rr, ch, 2 * pp], hist[rr, ch, 2 * pp + 1] = pair
            if cutlass.const_expr(NEXT):
                u = self.correction(dots, t + 1, b, betas[j, t + 1], lane_in, base, v_s,
                                    uo)  # fmt: skip
            partials = [pair_total(outs[rr, None, None]) for rr in range(ROWS)]
            o = self.row_sum8(partials, lane_in)
            o_s[b, t, own] = Float32(BFloat16(o))
            if cutlass.const_expr(t == min(1, T - 1)):
                self.release_slot(*release)

    @cute.jit
    def epilogue(self, args, g, c, o_s, g_s, out_sums, nw, oo):
        """Gated RMSNorm of the scanned item into ``oo`` (group thread c: channel c of
        every token; KdaWS.epilogue's sums)."""
        eps = args[15]
        T: cutlass.Constexpr = self.tokens
        lane, warp = c % 32, c // 32
        self.group_sync(g)  # every row's outputs are in o_s
        for t in cutlass.range_constexpr(T):
            o = o_s[g, t, c]
            total = warp_sum(o * o)
            if lane == 0:
                out_sums[t, warp] = total
        self.group_sync(g)
        for t in cutlass.range_constexpr(T):
            sumsq = self.quad_sum(out_sums[t, None])
            variance = sumsq * (1.0 / 128) + eps
            rstd = reciprocal(sqrt_approx(variance))
            sigmoid_gate = triton_sigmoid(Float32(g_s[g, t, c]))
            result = o_s[g, t, c] * rstd * nw * sigmoid_gate
            oo[t, c] = BFloat16(result)

    # ------------------------------------------------------------------- prep
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
                  sums_g, kgo):  # fmt: skip
        """Item j's operands into the group's block, its new key / raw-gate records
        into ``kgo`` and its window into the staged one (thread c = channel c);
        shared memory only."""
        lower_bound, scale = args[13], args[14]
        q_s, k_s, v_s, d_s, g_s, fk_s, fp_s, fu_s, fd_s = ops
        convs, xs, gs, kgs, uss = buffers
        sums, fsums, _ = sums_g
        T: cutlass.Constexpr = self.tokens
        P2: cutlass.Constexpr = 1 << (2 * T - 1).bit_length()  # q and k norms
        FOLD: cutlass.Constexpr = len(args) > 20
        lane, warp = c % 32, c // 32
        blk = g
        conv_type: cutlass.Constexpr = convs.element_type
        consts = hc[None, c]
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
            for t in cutlass.range_constexpr(T):
                v_s[blk, t, c] = values[2, t]
                g_s[blk, t, c] = gs[t, c]
                kgo[t, c] = BFloat16(values[1, t])
                raw_g = raw[j, t, c]
                kgo[t, K + c] = raw_g
                d_s[blk, t, c] = self.decay(a, Float32(raw_g), dt, lower_bound)

            # q and k norms over the 128 channels (one reduction for all 2T).
            squares = [Float32(0) for _ in range(P2)]
            for t in cutlass.range_constexpr(T):
                squares[2 * t] = values[0, t] * values[0, t]
                squares[2 * t + 1] = values[1, t] * values[1, t]
            self.block_sums(squares, sums, c, P2)

            # The pending commit's per-key operands (KdaWide's order).
            old_decays = zeros(T, Float32)
            if cutlass.const_expr(FOLD):  # noqa: SIM102
                if pending > 0:
                    for t in cutlass.range_constexpr(T):
                        total = warp_sum(old_keys[t] * old_keys[t])
                        if lane == 0:
                            fsums[t, warp] = total
                        old_decays[t] = self.decay(a, old_gates[t], dt, lower_bound)
            fence_async_shared()  # the window and records leave by bulk stores
            self.group_sync(g)  # the norms' partial sums
            for t in cutlass.range_constexpr(T):
                qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                fq = reciprocal(sqrt_approx(qn))
                fk = reciprocal(sqrt_approx(kn))
                q_s[blk, t, c] = mul_rn(mul_rn(values[0, t], fq), scale)
                k_s[blk, t, c] = mul_rn(values[1, t], fk)
            # The commit's reverse token loop over this key: kn_t, the final_decay
            # P_t token t's update is scaled by, and D_n (tokens t >= n: zeros).
            final_decay = Float32(1)
            for i in cutlass.range_constexpr(T):
                t = T - 1 - i
                kn_t = Float32(0)
                p_t = Float32(0)
                if cutlass.const_expr(FOLD):  # noqa: SIM102
                    if t < pending:
                        total = self.quad_sum(fsums[t, None]) + 1.0e-6
                        kn_t = mul_rn(rsqrt_ftz(total), old_keys[t])
                        p_t = final_decay
                        final_decay = mul_rn(final_decay, old_decays[t])
                fk_s[blk, t, c] = kn_t
                fp_s[blk, t, c] = p_t
            fd_s[blk, c] = final_decay
