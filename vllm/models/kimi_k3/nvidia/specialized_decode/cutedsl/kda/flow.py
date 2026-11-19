# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent KDA record core with the deferred RecoverSSM fold: one state pass per
item, two independent item streams per SM (KdaFlow).

One CTA per SM (256 threads) walks a contiguous, head-major range of (request, head)
items in two groups of 4 warps; group g runs the CTA's items g, g + 2, ... start to
finish:

1. **prep** (thread c = channel c): convolution + SiLU, q / k normalization, the F_B
   gate projection (mma.sync, F_B fragments from L2) and decays, beta and the fold's
   per-key operands into the group's operand block. The projection rows, the
   convolution window and the pending commit's records are cp.async-staged one item
   ahead; the CTA's items' slots / pending counts and the per-head weights are
   tabled once in shared memory.
2. **scan** (thread c: rows c // 2 and c // 2 + 64, keys of half h = c % 2; register
   chunk j holds the half's 4-key chunk j ^ (c % 8), so every shared-memory access
   of a warp hits 8 distinct 16-byte bank groups): the item's FP32 state page goes
   from its slot of the CTA's page ring to registers, the pending commit is folded
   in (the stock Triton commit's math), the committed checkpoint goes back into the
   slot and out by one asynchronous bulk store (L2 evict-first), the q tokens are
   verified (each row's dot products: two per-thread partials summed over the lane
   pair, one shuffle) and the outputs RMS-normed and gated.

State stream: a ring of three page slots per CTA; item j uses slot j % 3, and once
item j's bulk store has read its slot, the slot is refilled with item j + 3, so the
page of a group's next item is loaded while both groups compute.

The two groups are independent: while one is in its latency-bound prep or
reductions, the other issues its FP32 passes. Every thread may use 255 registers.

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
    store_bulk,
    sync_threads,
    wait_bulk_reads,
)
from .wide import load_weight_bf16x4, rsqrt_ftz

K = 128
GROUPS = 2
GROUP_THREADS = 128
THREADS = GROUPS * GROUP_THREADS
ROWS = 2  # state rows per scan thread: c // 2 and c // 2 + 64
CHUNKS = 16  # 4-key chunks per thread: half c % 2 of the keys
SLOTS = 3  # the CTA's state page ring
# Rubin's oversized shared memory mode (327 KiB per CTA; budget 320 KiB). Elsewhere
# the opt-in limit.
OVERSIZED_BUDGET = 320 * 1024
# Diagnostics (benchmarks only): per-phase cycle breakdown printed by a few CTAs.
TRACE = False
# Diagnostics (benchmarks only): parts of the scan to leave out ("fold", "store",
# "verify", "prep"); results are wrong with any of them set.
SKIP = ()
# Diagnostics (benchmarks only): bounded mbarrier waits that report a stuck wait.
HANG = False
MAX_ITEMS = 64  # items per CTA (the item table)
# FP32 pair ops as packed f32x2 instructions (False: scalar; bit-identical).
PACKED = True
# Item table columns.
SLOT, HEAD, REQUEST, CONV_SLOT, CONV_OFFSET, PENDING, RECORD = range(7)
META = 8
CONSTS = 22  # per-head operands of a channel (KdaFlow.head_constants)


def operand_bytes(tokens):
    """One operand block: q, k, v (= out), decay, fold keys, suffix decays, old
    corrections (FP32), D_n, the output gate (BF16), beta."""
    return 7 * tokens * K * 4 + K * 4 + tokens * K * 2 + 8 * 4


def staging_bytes(tokens, conv_dim_first, window, conv_size):
    """One staging buffer: x (3 parts), gate and F_A rows, the convolution window,
    the pending commit's records (keys | raw gates, corrections)."""
    rows = 5 * tokens * K * 2
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    return rows + conv + tokens * 2 * K * 2 + tokens * K * 4


def shared_bytes(tokens, conv_dim_first, window, conv_size, stages):
    staging = staging_bytes(tokens, conv_dim_first, window, conv_size)
    sums = GROUPS * (16 * 4 * 4 + 8 * 4 * 4 + 8 * 4 * 4)
    blocks = GROUPS * operand_bytes(tokens)
    bars = SLOTS * 8
    pages = SLOTS * PAGE_BYTES
    table = MAX_ITEMS * META * 4 + 2 * CONSTS * GROUP_THREADS * 4
    return pages + blocks + GROUPS * staging + sums + bars + table + 2048


def smem_budget(device):
    """Shared memory one KdaFlow CTA may use on ``device``."""
    props = torch.cuda.get_device_properties(device)
    limit = props.shared_memory_per_block_optin
    if (props.major, props.minor) == (10, 7):
        limit = max(limit, OVERSIZED_BUDGET)
    return limit


def max_stages(tokens, conv_dim_first, window, conv_size, limit):
    """1 when KdaFlow's shared memory (a ring of SLOTS state pages) fits ``limit``,
    else 0."""
    return int(shared_bytes(tokens, conv_dim_first, window, conv_size, 1) <= limit)


def max_items_per_cta(requests):
    """KdaFlow tables the weights of two heads (a CTA's head-major item range may span
    at most two heads) and its items."""
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
def store_bulk_evict(dst, src, size):
    """Shared -> global bulk copy, L2 evict-first (in the open bulk group)."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], $2, pol; }"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm(Int32)
def smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l", ()


class KdaFlow(KdaWS):
    """KdaWS's operands and semantics for RecoverSSM record steps (``tokens`` q > 1,
    records always written); with the deferred operands (``record_rows``,
    ``pending`` after the records) every item folds its pending commit first.
    ``stages``: 1 (the page ring has SLOTS slots); ``ctas`` persistent CTAs."""

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
        assert 2 <= tokens <= 8 and stages == 1
        super().__init__(
            heads, tokens, True, gate_mode, has_bias, conv_dim_first, window, ctas
        )
        self.stages = stages
        self.trace, self.hang = TRACE, HANG
        self.packed = PACKED
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
        are head-major (a CTA's items share at most two heads)."""
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
    def stage_item(self, args, table, j, convs, xs, gs, fas, kgs, uss, p):
        """cp.async table item j's projection rows, window and pending records."""
        slot, head = table[j, SLOT], table[j, HEAD]
        if slot > 0:
            request = table[j, REQUEST]
            self.stage_conv(args, convs, p, head, table[j, CONV_SLOT],
                            table[j, CONV_OFFSET])  # fmt: skip
            self.stage_rows(args, xs, gs, fas, p, head, request)
            if cutlass.const_expr(len(args) > 20):
                self.stage_records(args, kgs, uss, p, head, table[j, RECORD],
                                   table[j, PENDING])  # fmt: skip

    @cute.jit
    def stage_records(self, args, kgs, uss, p, head, rec, pending):
        """cp.async the pending commit's records (keys | raw gates, corrections)."""
        T: cutlass.Constexpr = self.tokens
        if pending > 0:
            for i in cutlass.range_constexpr(-(-(T * 32) // GROUP_THREADS)):
                g = p + GROUP_THREADS * i
                if g < T * 32:
                    t, k = g // 32, g % 32
                    copy_g2s(at(kgs, t, 8 * k), at(args[19], rec, head, t, 8 * k))
                    copy_g2s(at(uss, t, 4 * k), at(args[18], rec, head, t, 4 * k))

    @cute.jit
    def load_stage(self, args, pages, s, table, j, state_full):
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

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state, output = args[3], args[10]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        entry = Int64(0)
        if cutlass.const_expr(self.trace):
            entry = globaltimer()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (SLOTS, K * K), align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        q_s = alloc(smem, Float32, (GROUPS, T, K))
        k_s = alloc(smem, Float32, (GROUPS, T, K))
        v_s = alloc(smem, Float32, (GROUPS, T, K))  # a row's out_t replaces v_t
        d_s = alloc(smem, Float32, (GROUPS, T, K))
        fk_s = alloc(smem, Float32, (GROUPS, T, K))  # kn_t of the pending commit
        fp_s = alloc(smem, Float32, (GROUPS, T, K))  # P_t
        fu_s = alloc(smem, Float32, (GROUPS, T, K))  # the old corrections u_t
        fd_s = alloc(smem, Float32, (GROUPS, K))  # D_n
        g_s = alloc(smem, BFloat16, (GROUPS, T, K))
        beta_s = alloc(smem, Float32, (GROUPS, 8))
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (GROUPS, 3, K * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (GROUPS, 3, 3 * K), align=16)
        xs = alloc(smem, BFloat16, (GROUPS, T, 3, K), align=16)
        gs = alloc(smem, BFloat16, (GROUPS, T, K), align=16)
        fas = alloc(smem, BFloat16, (GROUPS, T, K), align=16)
        kgs = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=16)
        uss = alloc(smem, Float32, (GROUPS, T, K), align=16)
        sums = alloc(smem, Float32, (GROUPS, 16, 4))
        fsums = alloc(smem, Float32, (GROUPS, 8, 4))  # recorded k squares per warp
        out_sums = alloc(smem, Float32, (GROUPS, 8, 4))
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        hcs = alloc(smem, Float32, (2, CONSTS, GROUP_THREADS))  # per head slot
        state_full = smem.allocate(cutlass.Uint64, SLOTS)

        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):
                cute.arch.mbarrier_init(state_full + s, 1)
            cute.arch.mbarrier_init_fence()
        # The page sources (the host's slot table: no predecessor output).
        if tid < count:
            head, _, slot, _, _ = self.item_slots(args, first + tid)
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
        cute.arch.sync_warp()
        cute.arch.barrier()
        # Before the PDL wait: the first pages (this layer's state) and the per-head
        # weights.
        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):
                if s < count:
                    self.load_stage(args, pages, s, table, s, state_full)
        cute.arch.sync_warp()
        h0 = first // requests
        h1 = (first + count - 1) // requests
        # Per-head operands of channel tid % 128 for the CTA's (at most two) heads.
        hslot = tid // GROUP_THREADS
        if hslot <= h1 - h0:
            self.head_constants(hcs[hslot, None, None], args, h0 + hslot,
                                tid % GROUP_THREADS)  # fmt: skip
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
        cute.arch.sync_warp()
        cute.arch.barrier()  # the table
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        staging = (
            convs[g, None, None], xs[g, None, None, None], gs[g, None, None],
            fas[g, None, None], kgs[g, None, None], uss[g, None, None],
            sums[g, None, None], fsums[g, None, None],
        )  # fmt: skip
        if g < count:
            self.stage_item(args, table, g, *staging[:6], c)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        cute.arch.barrier()  # the head weights and the first staging
        ops = (q_s, k_s, v_s, d_s, g_s, beta_s, fk_s, fp_s, fu_s, fd_s)
        looped = Int64(0)
        cycles = Int64(0)
        if cutlass.const_expr(self.trace):
            looped = globaltimer()
            cycles = clock64()
        self.group_items(args, g, c, first, count, h0, pages, table, staging, hcs,
                         ops, out_sums[g, None, None], state_full)  # fmt: skip
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if tid == 0:
                stamp = globaltimer()
                spent = clock64() - cycles
                if cta % 53 == 0:
                    cute.printf("TIMER cta {} smid {} entry {} loop {} end {} loop_ns {} loop_cycles {}", cta, smid(), entry, looped, stamp, stamp - looped, spent)  # noqa: E501
                if cta >= self.ctas - 3:
                    cute.printf("TIMER cta {} smid {} entry {} loop {} end {} loop_ns {} loop_cycles {}", cta, smid(), entry, looped, stamp, stamp - looped, spent)  # noqa: E501

    # ------------------------------------------------------------- group loop
    @cute.jit
    def group_items(self, args, g, c, first, count, h0, pages, table, staging, hcs,
                    ops, out_sums, state_full):  # fmt: skip
        q_s, k_s, v_s, d_s, g_s, beta_s, fk_s, fp_s, fu_s, fd_s = ops
        beta, state = args[5], args[9]
        T: cutlass.Constexpr = self.tokens
        half, x = c % 2, c % 8  # key half; register chunk j holds chunk j ^ x
        rows = (c // 2, c // 2 + 64)
        nw = args[12][c]
        mine = (count - g + 1) // 2  # this group's items: g, g + 2, ...
        beta_next = Float32(0)
        if mine > 0:
            if c < T:
                beta_next = Float32(beta[table[g, REQUEST] * T + c, table[g, HEAD]])
        tr = zeros(6, Int64)  # TRACE: total, prep, state wait, fold, verify, epi
        pt = zeros(8, Int64)  # TRACE: prep phases
        t0 = Int64(0)
        ta, tb, tc, td, te = (Int64(0) for _ in range(5))
        if cutlass.const_expr(self.trace):
            t0 = clock64()
        for jj in cutlass.range(mine, unroll=1):
            j = g + GROUPS * jj
            slot, head = table[j, SLOT], table[j, HEAD]
            request, pending, rec = table[j, REQUEST], table[j, PENDING], table[j, RECORD]
            conv_slot, offset = table[j, CONV_SLOT], table[j, CONV_OFFSET]
            token = request * T
            if cutlass.const_expr(self.trace):
                ta = clock64()
            if jj > 0:  # the item's staging (issued during the previous item)
                cute.arch.cp_async_wait_group(0)
                self.group_sync(g)
            beta_value = beta_next
            if jj + 1 < mine:
                if c < T:
                    j1 = j + GROUPS
                    beta_next = Float32(beta[table[j1, REQUEST] * T + c, table[j1, HEAD]])
            if cutlass.const_expr("prep" not in self.skip):
                self.prep_item(args, g, c, slot, head, request, conv_slot, offset,
                               pending, rec, beta_value, hcs[head - h0, None, None],
                               staging, ops, pt)  # fmt: skip
            # The staging buffer is free: the group's next item's rows.
            if jj + 1 < mine:
                self.stage_item(args, table, j + GROUPS, *staging[:6], c)
            cute.arch.cp_async_commit_group()
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
                            at(pages, s, rows[rr] * K + 64 * half + 4 * (x ^ ch))
                        )
                        for e in cutlass.range_constexpr(4):
                            hist[rr, ch, e] = values[e]
                if cutlass.const_expr("fold" not in self.skip):
                    self.fold(hist, g, pending, half, x, rows, fk_s, fp_s, fu_s, fd_s)
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):
                    stored = pending < 0
                if stored:  # the committed checkpoint back into the slot
                    for rr in cutlass.range_constexpr(ROWS):
                        for ch in cutlass.range_constexpr(CHUNKS):
                            store_shared_f32x4(
                                at(pages, s, rows[rr] * K + 64 * half + 4 * (x ^ ch)),
                                [hist[rr, ch, e] for e in range(4)],
                            )
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
                for t in cutlass.range_constexpr(T):
                    if cutlass.const_expr("verify" not in self.skip):
                        self.verify_token(args, hist, t, g, head, rec, c, half, x,
                                          rows, q_s, k_s, v_s, d_s, beta_s)  # fmt: skip
                    if cutlass.const_expr(t == min(1, T - 1)):
                        self.release_slot(args, c, j, count, pages, table, state_full,
                                          stored)  # fmt: skip
                if cutlass.const_expr(self.trace):
                    te = clock64()
                    tr[4] = tr[4] + (te - td)
                if cutlass.const_expr("epi" not in self.skip):
                    self.epilogue(args, g, c, head, token, v_s, g_s, out_sums, nw)
                if cutlass.const_expr(self.trace):
                    tr[5] = tr[5] + (clock64() - te)
            else:
                self.release_slot(args, c, j, count, pages, table, state_full, False)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("TRACE cta {} g {} items {} total {} prep {} wstate {} fold {} verify {} epi {}", cta, g, mine, clock64() - t0, tr[1], tr[2], tr[3], tr[4], tr[5])  # noqa: E501
                    cute.printf("PREP cta {} g {} fb {} conv {} foldsums {} gate {} sync1 {} tail {} sync2 {}", cta, g, pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7])  # noqa: E501

    @cute.jit
    def release_slot(self, args, c, j, count, pages, table, state_full, stored):
        """Item j's slot is free once its bulk store has read it: thread 0 refills it
        with item j + SLOTS's page."""
        if c == 0:
            if stored:
                wait_bulk_reads()
            if j + SLOTS < count:
                self.load_stage(args, pages, j % SLOTS, table, j + SLOTS, state_full)
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
    def fold(self, hist, b, pending, half, x, rows, keys, suffix, old, decay):
        """The stock commit of the pending tokens into this thread's state elements:
        fc = fma(u_t * kn_t, P_t, fc) over t = q - 1 .. 0 (tokens t >= n are zeros),
        then S = fma(S_0, D_n, fc)."""
        T: cutlass.Constexpr = self.tokens
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        us = cute.make_rmem_tensor((ROWS, T), Float32)
        for rr in cutlass.range_constexpr(ROWS):
            for t in cutlass.range_constexpr(T):
                us[rr, t] = old[b, t, rows[rr]]
        for ch in cutlass.range_constexpr(CHUNKS):
            key = 64 * half + 4 * (x ^ ch)
            d4 = load_shared_f32x4(at(decay, b, key))
            fc = cute.make_rmem_tensor((ROWS, 4), Float32)
            for rr in cutlass.range_constexpr(ROWS):
                for e in cutlass.range_constexpr(4):
                    fc[rr, e] = start
            for i in cutlass.range_constexpr(T):
                t = T - 1 - i
                k4 = load_shared_f32x4(at(keys, b, t, key))
                p4 = load_shared_f32x4(at(suffix, b, t, key))
                for rr in cutlass.range_constexpr(ROWS):
                    u = us[rr, t]
                    for p in cutlass.range_constexpr(2):
                        update = self.pmul((u, u), (k4[2 * p], k4[2 * p + 1]))
                        acc = (fc[rr, 2 * p], fc[rr, 2 * p + 1])
                        acc = self.pfma(update, (p4[2 * p], p4[2 * p + 1]), acc)
                        fc[rr, 2 * p], fc[rr, 2 * p + 1] = acc
            for rr in cutlass.range_constexpr(ROWS):
                for p in cutlass.range_constexpr(2):
                    pair = (hist[rr, ch, 2 * p], hist[rr, ch, 2 * p + 1])
                    pair = self.pfma(pair, (d4[2 * p], d4[2 * p + 1]),
                                     (fc[rr, 2 * p], fc[rr, 2 * p + 1]))  # fmt: skip
                    hist[rr, ch, 2 * p], hist[rr, ch, 2 * p + 1] = pair
            # Bounds how far ptxas hoists the independent chunks' loads (spills).
            cute.arch.sync_warp()

    @cute.jit
    def verify_token(self, args, hist, t, b, head, rec, c, half, x, rows, q_s, k_s,
                     v_s, d_s, beta_s):  # fmt: skip
        """Verify token t from the state in registers: the decay pass (S *= d_t,
        dot = S k_t), the corrections of both rows, the update pass (S += k_t u,
        out = S q_t); records u_t and the BF16 out_t (into v_s) of row c // 2 + 64 h."""
        o_s = v_s

        def chunk_of(buffer, ch):
            return load_shared_f32x4(at(buffer, b, t, 64 * half + 4 * (x ^ ch)))

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
        u = [Float32(0) for _ in range(ROWS)]
        for rr in cutlass.range_constexpr(ROWS):
            part = pair_total(dots[rr, None, None])
            if cutlass.const_expr("reduce" not in self.skip):
                part = part + cute.arch.shuffle_sync_bfly(part, 1)
            u[rr] = (v_s[b, t, rows[rr]] - part) * beta_s[b, t]
        own_row = rows[0] + 64 * half
        mine = u[1] if half != 0 else u[0]
        args[18][rec, head, t, own_row] = mine
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
        p0 = pair_total(outs[0, None, None])
        p1 = pair_total(outs[1, None, None])
        send = p0 if half != 0 else p1
        keep = p1 if half != 0 else p0
        o = keep
        if cutlass.const_expr("reduce" not in self.skip):
            o = keep + cute.arch.shuffle_sync_bfly(send, 1)
        o_s[b, t, own_row] = Float32(BFloat16(o))

    @cute.jit
    def epilogue(self, args, g, c, head, token, o_s, g_s, out_sums, nw):
        """Gated RMSNorm and output of the scanned item (group thread c: channel c
        of every token; KdaWS.epilogue's sums)."""
        output, eps = args[10], args[15]
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
            output[token + t, head * K + c] = BFloat16(result)

    # ------------------------------------------------------------------- prep
    @cute.jit
    def head_constants(self, hc, args, head, c):
        """Per-head operands of prep thread c into hc[:, c]: conv weights [3, 4],
        biases [3], exp(A_log), dt of channel c, dt of this thread's four gate
        features."""
        weights, bias, a_log, dt_bias = args[1], args[2], args[6], args[7]
        lane, warp = c % 32, c // 32
        group_id = lane >> 2
        consts = hc[None, c]
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        for part in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(4):
                consts[4 * part + i] = Float32(weights[part, i, head * K + c])
            if cutlass.const_expr(self.has_bias):
                consts[12 + part] = Float32(bias[part * dim + head * K + c])
            else:
                consts[12 + part] = Float32(0)
        consts[15] = triton_exp(Float32(a_log[head]))
        consts[16] = Float32(dt_bias[head * K + c])
        for m in cutlass.range_constexpr(2):
            for h in cutlass.range_constexpr(2):
                feature = 32 * warp + 16 * m + group_id + 8 * h
                consts[17 + 2 * m + h] = Float32(dt_bias[head * K + feature])
        consts[21] = Float32(0)

    @cute.jit
    def prep_item(self, args, g, c, slot, head, request, conv_slot, offset, pending,
                  rec, beta_value, hc, staging, ops, pt):  # fmt: skip
        """The item's operands into the group's block (thread c = channel c); ends
        with a group barrier (the block is visible, the staging buffer free)."""
        conv_state, output = args[3], args[10]
        lower_bound, scale = args[13], args[14]
        q_s, k_s, v_s, d_s, g_s, beta_s, fk_s, fp_s, fu_s, fd_s = ops
        convs, xs, gs, fas, kgs, uss, sums, fsums = staging
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        P2: cutlass.Constexpr = 1 << (2 * T - 1).bit_length()  # q and k norms
        FOLD: cutlass.Constexpr = len(args) > 20
        lane, warp = c % 32, c // 32
        group_id, quarter = lane >> 2, lane & 3
        p = c
        blk = g
        token = request * T
        conv_type: cutlass.Constexpr = conv_state.element_type
        consts = hc[None, c]
        q0 = Int64(0)
        q1, q2, q3, q4, q5, q6 = (Int64(0) for _ in range(6))
        if cutlass.const_expr(self.trace):
            q0 = clock64()
        if slot > 0:
            # F_B fragments (L2) of this warp's 32 features, consumed by the gate
            # projection after the convolution.
            f_b = args[16]
            wr = cute.make_rmem_tensor((2, 8, 4), Uint32)
            for m in cutlass.range_constexpr(2):
                for chunk in cutlass.range_constexpr(8):
                    feature = head * K + 32 * warp + 16 * m + group_id
                    col = 16 * chunk + 4 * quarter
                    low = load_weight_bf16x4(at(f_b, feature, col))
                    high = load_weight_bf16x4(at(f_b, feature + 8, col))
                    wr[m, chunk, 0], wr[m, chunk, 2] = low
                    wr[m, chunk, 1], wr[m, chunk, 3] = high
            if cutlass.const_expr(self.trace):
                q1 = clock64()
                pt[1] = pt[1] + (q1 - q0)
            # The pending commit's records, read before this verify writes the new
            # ones into the same record slot; tokens t >= n fold as zeros (+0, or
            # -0 corrections when nothing is pending: the fold's identity).
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
            a = consts[15]

            # Convolution + SiLU of q, k, v (thread c: channel c of each part).
            values = cute.make_rmem_tensor((3, T), Float32)
            conv_page = conv_state[Int64(conv_slot), None, None]
            for part in cutlass.range_constexpr(3):
                channel = part * dim + head * K + c
                taps = cute.make_rmem_tensor(3, conv_type)
                for t in cutlass.range_constexpr(3):
                    if cutlass.const_expr(self.conv_dim_first):
                        taps[t] = convs[part, c * self.window + offset + t]
                    else:
                        taps[t] = convs[part, t * K + c]
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
                if cutlass.const_expr(self.conv_dim_first):
                    row = c * self.window
                    convs[part, row] = conv_type(taps[1])
                    convs[part, row + 1] = conv_type(taps[2])
                    for t in cutlass.range_constexpr(T):
                        convs[part, row + 2 + t] = conv_type(inputs[t])
                else:
                    conv_page[channel, 0] = conv_type(taps[1])
                    conv_page[channel, 1] = conv_type(taps[2])
                    for t in cutlass.range_constexpr(T):
                        conv_page[channel, 2 + t] = conv_type(inputs[t])
            if cutlass.const_expr(self.conv_dim_first):
                fence_async_shared()
            for t in cutlass.range_constexpr(T):
                v_s[blk, t, c] = values[2, t]
                g_s[blk, t, c] = gs[t, c]
                args[19][rec, head, t, c] = BFloat16(values[1, t])

            if cutlass.const_expr(self.trace):
                q2 = clock64()
                pt[2] = pt[2] + (q2 - q1)
            # q and k norms over the 128 channels (one reduction for all 2T).
            squares = [Float32(0) for _ in range(P2)]
            for t in cutlass.range_constexpr(T):
                squares[2 * t] = values[0, t] * values[0, t]
                squares[2 * t + 1] = values[1, t] * values[1, t]
            self.block_sums(squares, sums, p, P2)

            # The pending commit's per-key operands (KdaWide's order).
            old_decays = zeros(T, Float32)
            if cutlass.const_expr(FOLD):  # noqa: SIM102
                if pending > 0:
                    for t in cutlass.range_constexpr(T):
                        total = warp_sum(old_keys[t] * old_keys[t])
                        if lane == 0:
                            fsums[t, warp] = total
                        old_decays[t] = self.decay(
                            a, old_gates[t], consts[16], lower_bound
                        )

            if cutlass.const_expr(self.trace):
                q3 = clock64()
                pt[3] = pt[3] + (q3 - q2)
            # Gate projection: D[feature, token], tile m of this warp's 32 features;
            # the decays and raw gates.
            cute.arch.sync_warp()  # converged for mma.sync
            for m in cutlass.range_constexpr(2):
                chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                acc = zeros((chains, 4), Float32)
                for chunk in cutlass.range_constexpr(8):
                    chain: cutlass.Constexpr = chunk // 2 if self.gate_mode == 1 else 0
                    wa = tuple(wr[m, chunk, e] for e in range(4))
                    ab = load_shared_u32x2(
                        at(fas, group_id % T, 16 * chunk + 4 * quarter)
                    )
                    mma_into(acc[chain, None], wa, ab, bf16=True)
                for e in cutlass.range_constexpr(4):
                    t = 2 * quarter + e % 2
                    feature = 32 * warp + 16 * m + group_id + 8 * (e // 2)
                    value = acc[0, e]
                    if cutlass.const_expr(self.gate_mode == 1):
                        value = Float32(0) + (
                            ((value + acc[1, e]) + acc[2, e]) + acc[3, e]
                        )
                    raw_g = Float32(BFloat16(value))
                    if t < T:
                        dt = consts[17 + 2 * m + e // 2]
                        d_s[blk, t, feature] = self.decay(a, raw_g, dt, lower_bound)
                        args[19][rec, head, t, K + feature] = BFloat16(raw_g)
            if p < T:
                beta_s[blk, p] = triton_sigmoid(beta_value)
            if cutlass.const_expr(self.trace):
                q4 = clock64()
                pt[4] = pt[4] + (q4 - q3)
            self.group_sync(g)  # the norms' partial sums
            if cutlass.const_expr(self.trace):
                q5 = clock64()
                pt[5] = pt[5] + (q5 - q4)
            if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                if p == 0:
                    size: cutlass.Constexpr = K * self.window * conv_type.width // 8
                    for part in cutlass.range_constexpr(3):
                        channel0 = part * dim + head * K
                        store_bulk(
                            at(conv_page, channel0, 0), at(convs, part, 0), size
                        )

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
        else:
            for t in cutlass.range_constexpr(T):
                output[token + t, head * K + c] = BFloat16(0)
        if cutlass.const_expr(self.trace):
            q6 = clock64()
            if slot > 0:
                pt[6] = pt[6] + (q6 - q5)
        if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
            if p == 0:
                wait_bulk_reads()
        self.group_sync(g)
        if cutlass.const_expr(self.trace):
            pt[7] = pt[7] + (clock64() - q6)
