# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent warp-specialized KDA record core with the deferred RecoverSSM fold and a
multi-stage state ring (KdaFlow).

KdaWS's structure (one CTA per SM walking a contiguous range of (request, head) items;
an 8-warp scan group holding one item's [128, 128] FP32 state in registers, two prep
warpgroups producing the per-item operands), plus:

- **the deferred fold** (KdaWide's): before its verify, an item folds the request's
  pending commit (the n accepted tokens of its last verify, from the records in its
  record slot) into the checkpoint with the stock Triton commit's math
  (``_commit_kda_state_kernel``) bit for bit, writes the committed checkpoint back
  once and verifies from it; so a step reads and writes each state page once and
  needs no separate flush pass (KdaWS + flush: three state passes);
- **a ``stages``-deep state ring**: the bulk copies of the next ``stages`` items' pages
  are in flight while the scan runs an item, so each SM keeps more state bytes in
  flight (KdaWS: one page, issued only once the scan holds the previous item).

Fold operands (prep, thread c = key c): the recorded keys normalized as the commit's
``rsqrt.approx.ftz(sum k^2 + 1e-6) * k`` (warp butterfly, then the four warps'
partials as (w0 + w2) + (w1 + w3), KdaWide's order), the suffix decays P_t (the
commit's final_decay before token t) and D_n; the scan folds per state element
fc = fma(u_t * kn_t, P_t, fc) over t = n - 1 .. 0 from 0, then S = fma(S_0, D_n, fc).

Per item the verify math is KdaWide's / KdaWS's: the stock Triton formulas and
instructions, BF16 roundings of the convolution outputs, raw gates and recurrence
outputs, FP32 state; only FP32 reduction order differs.

Synchronization (mbarriers, phase = use count & 1 of each barrier):
  state_full[s]  bulk copy bytes of stage s
  state_empty[s] scan -> scan thread 0 (stage s is in registers: it copies the page
                 of the item ``stages`` on into it)
  ops_full[blk]  prep -> scan (operand block blk = 2 w + (j // 2) % 2 ready)
  done[blk]      scan -> prep (item scanned: the operand block is free)
"""

import cutlass
import cutlass.cute as cute
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
    store_f32x4,
    wait,
    warp_sum,
    zeros,
)
from .recover import (
    mul_rn,
    reciprocal,
    record_row,
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
    set_max_registers,
    store_bulk,
    sync_threads,
    wait_bulk_reads,
)
from .wide import rsqrt_ftz

K = 128
GROUP_THREADS = 256
PREP_THREADS = 128
THREADS = GROUP_THREADS + 2 * PREP_THREADS
ROWS = 4  # state rows per scan thread: g + 32 r
LANES = 8  # scan threads per row
CHUNKS = 4  # 4-key chunks per thread and row
SCAN_REGISTERS, PREP_REGISTERS = 168, 88
SMEM_LIMIT = 227 * 1024
# Diagnostics (benchmarks only): per-role cycle breakdown printed by a few CTAs.
TRACE = False
# Diagnostics (benchmarks only): parts of the scan to leave out ("fold", "store",
# "verify"); results are wrong with any of them set.
SKIP = ()
# Per operand block: live, request, head, token, pending n, checkpoint slot.
META = 8
BLOCKS = 4


def operand_bytes(tokens):
    """One operand block: q, k, v, decay, out, fold keys, suffix decays, old
    corrections (FP32), D_n, the output gate (BF16), beta, meta."""
    return 7 * tokens * K * 4 + K * 4 + tokens * K * 2 + 8 * 4 + META * 4


def staging_bytes(tokens, conv_dim_first, window, conv_size):
    rows = 5 * tokens * K * 2  # x (3 parts), gate, F_A
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    return rows + conv


def shared_bytes(tokens, conv_dim_first, window, conv_size, stages):
    staging = staging_bytes(tokens, conv_dim_first, window, conv_size)
    sums = 2 * 16 * 4 * 4 + 2 * 8 * 4 * 4 + 2 * 8 * 4 * 4
    blocks = BLOCKS * operand_bytes(tokens)
    bars = (2 * stages + 2 * BLOCKS) * 8
    head = K * K * 2 + tokens * (2 * K * 2 + K * 4)  # F_B rows, old records
    return stages * PAGE_BYTES + blocks + 2 * (staging + head) + sums + bars + 256


def max_stages(tokens, conv_dim_first, window, conv_size, limit):
    """The deepest state ring (1 or 2 pages) that fits ``limit`` bytes, or 0."""
    for stages in (2, 1):
        size = shared_bytes(tokens, conv_dim_first, window, conv_size, stages)
        if size <= min(SMEM_LIMIT, limit):
            return stages
    return 0


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l", ()


class KdaFlow(KdaWS):
    """KdaWS's operands and semantics for RecoverSSM record steps (``tokens`` q > 1,
    records always written); with the deferred operands (``record_rows``,
    ``pending`` after the records) every item folds its pending commit first.
    ``stages``: state pages in flight per CTA; ``ctas`` persistent CTAs."""

    def __init__(
        self,
        heads,
        tokens,
        gate_mode,
        has_bias,
        conv_dim_first,
        window,
        ctas,
        stages=2,
    ):
        assert 2 <= tokens <= 8 and stages in (1, 2)
        super().__init__(
            heads, tokens, True, gate_mode, has_bias, conv_dim_first, window, ctas
        )
        self.stages = stages
        self.trace = TRACE
        self.skip = tuple(SKIP)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def item_info(self, args, item):
        """(head, request, state slot, conv slot, window offset, pending n, record
        slot) of an item; the window offset includes the roll of the pending
        commit (the commit's compaction)."""
        head, request, slot, conv_slot, offset = self.item_slots(args, item)
        pending = Int32(0)
        rec = Int32(-1)
        if cutlass.const_expr(len(args) > 20):
            rec = Int32(args[20][request])
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
    def item_slots(self, args, item):
        """(head, request, state slot, conv slot, window offset) of an item; items
        are head-major (a CTA's items mostly share a head: F_B rows stay in L2)."""
        slots = args[8]
        requests: cutlass.Constexpr = args[10].shape[0] // self.tokens
        head, request = item // requests, item % requests
        offset = args[17][request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        return head, request, Int32(slot), Int32(conv_slot), Int32(offset)

    @staticmethod
    def fb_at(fbs, feature, chunk, quarter):
        """F_B element (feature, 16 chunk + 4 quarter) in the swizzled rows."""
        granule = (2 * chunk + quarter // 2) ^ (feature % 8)
        return at(fbs, feature, 8 * granule + 4 * (quarter % 2))

    @cute.jit
    def stage_head(self, args, fbs, kgs, uss, p, head, rec, pending):
        """cp.async the item's F_B rows and, with a pending commit, its old
        records (prep thread p)."""
        f_b = args[16]
        T: cutlass.Constexpr = self.tokens
        for i in cutlass.range_constexpr(K * K // 8 // PREP_THREADS):
            g = p + PREP_THREADS * i
            f, k = g // 16, g % 16
            copy_g2s(at(fbs, f, 8 * (k ^ (f % 8))), at(f_b, head * K + f, 8 * k))
        if cutlass.const_expr(len(args) > 20):  # noqa: SIM102
            if pending > 0:
                for i in cutlass.range_constexpr(-(-(T * 32) // PREP_THREADS)):
                    g = p + PREP_THREADS * i
                    if g < T * 32:
                        t, k = g // 32, g % 32
                        copy_g2s(at(kgs, t, 8 * k), at(args[19], rec, head, t, 8 * k))
                        copy_g2s(at(uss, t, 4 * k), at(args[18], rec, head, t, 4 * k))

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state, output = args[3], args[10]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        S: cutlass.Constexpr = self.stages
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (S, K * K), align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        q_s = alloc(smem, Float32, (BLOCKS, T, K))
        k_s = alloc(smem, Float32, (BLOCKS, T, K))
        v_s = alloc(smem, Float32, (BLOCKS, T, K))
        d_s = alloc(smem, Float32, (BLOCKS, T, K))
        o_s = v_s  # a row's out of token t replaces its v_t once v_t is read
        fk_s = alloc(smem, Float32, (BLOCKS, T, K))  # kn_t of the pending commit
        fp_s = alloc(smem, Float32, (BLOCKS, T, K))  # P_t
        fu_s = alloc(smem, Float32, (BLOCKS, T, K))  # the old corrections u_t
        fd_s = alloc(smem, Float32, (BLOCKS, K))  # D_n
        g_s = alloc(smem, BFloat16, (BLOCKS, T, K))
        beta_s = alloc(smem, Float32, (BLOCKS, 8))
        meta = alloc(smem, Int32, (BLOCKS, META))
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (2, 3, K * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (2, 3, 3 * K), align=16)
        xs = alloc(smem, BFloat16, (2, T, 3, K), align=16)
        gs = alloc(smem, BFloat16, (2, T, K), align=16)
        fas = alloc(smem, BFloat16, (2, T, K), align=16)
        sums = alloc(smem, Float32, (2, 16, 4))
        fsums = alloc(smem, Float32, (2, 8, 4))  # recorded k squares per warp
        # Per prep warpgroup: the item's F_B rows (16-byte granules XOR-swizzled
        # by row) and the pending commit's records (keys | raw gates, u).
        fbs = alloc(smem, BFloat16, (2, K, K), align=128)
        kgs = alloc(smem, BFloat16, (2, T, 2 * K), align=16)
        uss = alloc(smem, Float32, (2, T, K), align=16)
        out_sums = alloc(smem, Float32, (2, 8, 4))  # the scan's, by item parity
        bars = smem.allocate(cutlass.Uint64, 2 * S + 2 * BLOCKS)
        state_full, state_empty = bars, bars + S
        ops_full, done = bars + 2 * S, bars + 2 * S + BLOCKS

        if tid == 0:
            for s in cutlass.range_constexpr(S):
                cute.arch.mbarrier_init(state_full + s, 1)
                cute.arch.mbarrier_init(state_empty + s, GROUP_THREADS)
            for blk in cutlass.range_constexpr(BLOCKS):
                cute.arch.mbarrier_init(ops_full + blk, PREP_THREADS)
                cute.arch.mbarrier_init(done + blk, GROUP_THREADS)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        # The first pages do not depend on the predecessor (this layer's state).
        if tid == 0:
            for s in cutlass.range_constexpr(S):
                if s < count:
                    self.load_stage(args, pages, s, first + s, state_full)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        ops = (q_s, k_s, v_s, d_s, o_s, g_s, beta_s, meta, fk_s, fp_s, fu_s, fd_s)

        if tid >= GROUP_THREADS:
            set_max_registers(PREP_REGISTERS, increase=False)
            w = (tid - GROUP_THREADS) // PREP_THREADS
            p = (tid - GROUP_THREADS) % PREP_THREADS
            staging = (
                convs[w, None, None], xs[w, None, None, None], gs[w, None, None],
                fas[w, None, None], sums[w, None, None], fsums[w, None, None],
                fbs[w, None, None], kgs[w, None, None], uss[w, None, None],
            )  # fmt: skip
            self.prep_items(args, w, p, first, count, staging, ops, ops_full, done)
        else:
            set_max_registers(SCAN_REGISTERS, increase=True)
            self.scan_items(args, tid, first, count, pages, ops, out_sums,
                            state_full, state_empty, ops_full, done)  # fmt: skip

    @cute.jit
    def load_stage(self, args, pages, s, item, state_full):
        """Bulk-copy an item's state page into stage ``s`` (or complete the phase of
        a padded row)."""
        head, _, slot, _, _ = self.item_slots(args, item)
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

    # -------------------------------------------------------------- scan role
    @cute.jit
    def scan_items(self, args, t_g, first, count, pages, ops, out_sums,
                   state_full, state_empty, ops_full, done):  # fmt: skip
        q_s, k_s, v_s, d_s, o_s, g_s, beta_s, meta, fk_s, fp_s, fu_s, fd_s = ops
        state = args[9]
        T: cutlass.Constexpr = self.tokens
        S: cutlass.Constexpr = self.stages
        lane = t_g % 32
        lane_in, base_row = t_g % LANES, t_g // LANES
        own_row = base_row + 32 * (lane_in // 2)  # the row this lane reduces
        nw = args[12][t_g % K]
        tr = zeros(6, Int64)  # TRACE: total, state wait, ops wait, fold, verify, epi
        t0 = clock64()
        for j in cutlass.range(count, unroll=1):
            b = 2 * (j % 2) + (j // 2) % 2  # operand block
            s = j % S
            ta = clock64()
            wait(state_full + s, (j // S) % 2, scope="cta")
            tb = clock64()
            tr[1] = tr[1] + (tb - ta)
            history = cute.make_rmem_tensor((ROWS, CHUNKS, 4), Float32)
            for r in cutlass.range_constexpr(ROWS):
                for c in cutlass.range_constexpr(CHUNKS):
                    row = base_row + 32 * r
                    values = load_shared_f32x4(
                        at(pages, s, row * K + 4 * (lane_in + LANES * c))
                    )
                    for e in cutlass.range_constexpr(4):
                        history[r, c, e] = values[e]
            cute.arch.mbarrier_arrive(state_empty + s)
            if t_g == 0:  # once every thread holds item j: item j + S's page
                wait(state_empty + s, (j // S) % 2, scope="cta")
                if j + S < count:
                    self.load_stage(args, pages, s, first + j + S, state_full)
            ta = clock64()
            wait(ops_full + b, (j // 4) % 2, scope="cta")
            tb = clock64()
            tr[2] = tr[2] + (tb - ta)
            if meta[b, 0] != 0:
                request, head = meta[b, 1], meta[b, 2]
                pending = meta[b, 4]
                if cutlass.const_expr("fold" in self.skip):
                    pending = Int32(0)
                if pending > 0:  # the committed checkpoint, written back once
                    self.fold(history, b, pending, lane_in, base_row,
                              fk_s, fp_s, fu_s, fd_s)  # fmt: skip
                    destination = state[Int64(meta[b, 5]), None, None, None]
                    for r in cutlass.range_constexpr(ROWS if "store" not in self.skip else 0):
                        for c in cutlass.range_constexpr(CHUNKS):
                            row_values = [history[r, c, e] for e in range(4)]
                            key = 4 * (lane_in + LANES * c)
                            row = base_row + 32 * r
                            store_f32x4(at(destination, head, row, key), row_values)
                tc = clock64()
                tr[3] = tr[3] + (tc - tb)
                def chunk_of(buffer, t, c):
                    return load_shared_f32x4(
                        at(buffer, b, t, 4 * (lane_in + LANES * c))
                    )

                # Token 0's decay and k dot; then per token t one pass over the
                # state: the k v update and q dot of t fused with the decay and
                # k dot of t + 1 (each element's operations in the stock order;
                # the reductions of t's out and t + 1's dot overlap).
                if cutlass.const_expr("verify" not in self.skip):
                    dots = zeros((ROWS, 2, 2), Float32)
                    for c in cutlass.range_constexpr(CHUNKS):
                        d4, k4 = chunk_of(d_s, 0, c), chunk_of(k_s, 0, c)
                        for r in cutlass.range_constexpr(ROWS):
                            for pp in cutlass.range_constexpr(2):
                                pair = (history[r, c, 2 * pp], history[r, c, 2 * pp + 1])
                                pair = mul2(pair, (d4[2 * pp], d4[2 * pp + 1]))
                                history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                                acc = (dots[r, pp, 0], dots[r, pp, 1])
                                acc = fma2(pair, (k4[2 * pp], k4[2 * pp + 1]), acc)
                                dots[r, pp, 0], dots[r, pp, 1] = acc
                    outs = zeros((ROWS, 2, 2), Float32)
                    for t in cutlass.range_constexpr(T):
                        partials = [pair_total(dots[r, None, None]) for r in range(ROWS)]
                        dot = self.row_sum(partials, lane_in)
                        if cutlass.const_expr(t > 0):
                            partials = [
                                pair_total(outs[r, None, None]) for r in range(ROWS)
                            ]
                            o = self.row_sum(partials, lane_in)
                            if lane_in % 2 == 0:
                                o_s[b, t - 1, own_row] = Float32(BFloat16(o))
                        v = (v_s[b, t, own_row] - dot) * beta_s[b, t]
                        if lane_in % 2 == 0:
                            args[18][record_row(args, request), head, t, own_row] = v
                        v_new = [
                            cute.arch.shuffle_sync(v, (lane & ~7) | (2 * r))
                            for r in range(ROWS)
                        ]
                        NEXT: cutlass.Constexpr = t + 1 < T
                        outs = zeros((ROWS, 2, 2), Float32)
                        dots = zeros((ROWS, 2, 2), Float32)
                        for c in cutlass.range_constexpr(CHUNKS):
                            k4, q4 = chunk_of(k_s, t, c), chunk_of(q_s, t, c)
                            if cutlass.const_expr(NEXT):
                                d4, k4n = chunk_of(d_s, t + 1, c), chunk_of(k_s, t + 1, c)
                            for r in cutlass.range_constexpr(ROWS):
                                for pp in cutlass.range_constexpr(2):
                                    pair = (
                                        history[r, c, 2 * pp],
                                        history[r, c, 2 * pp + 1],
                                    )
                                    kp = (k4[2 * pp], k4[2 * pp + 1])
                                    pair = fma2(kp, (v_new[r], v_new[r]), pair)
                                    acc = (outs[r, pp, 0], outs[r, pp, 1])
                                    acc = fma2(pair, (q4[2 * pp], q4[2 * pp + 1]), acc)
                                    outs[r, pp, 0], outs[r, pp, 1] = acc
                                    if cutlass.const_expr(NEXT):
                                        pair = mul2(pair, (d4[2 * pp], d4[2 * pp + 1]))
                                        acc = (dots[r, pp, 0], dots[r, pp, 1])
                                        acc = fma2(
                                            pair, (k4n[2 * pp], k4n[2 * pp + 1]), acc
                                        )
                                        dots[r, pp, 0], dots[r, pp, 1] = acc
                                    history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                    partials = [pair_total(outs[r, None, None]) for r in range(ROWS)]
                    o = self.row_sum(partials, lane_in)
                    if lane_in % 2 == 0:
                        o_s[b, T - 1, own_row] = Float32(BFloat16(o))
                td = clock64()
                tr[4] = tr[4] + (td - tc)
                self.epilogue(args, t_g, j, b, head, meta[b, 3], o_s, g_s,
                              out_sums[j % 2, None, None], nw)  # fmt: skip
                tr[5] = tr[5] + (clock64() - td)
            cute.arch.mbarrier_arrive(done + b)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if t_g == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("TRACE scan cta {} items {} total {} wstate {} wops {} fold {} verify {} epi {}", cta, count, clock64() - t0, tr[1], tr[2], tr[3], tr[4], tr[5])

    @cute.jit
    def fold(self, history, b, n, lane_in, base_row, keys, suffix, old, decay):
        """The stock commit of the n pending tokens into this thread's state
        elements (KdaWide.fold's per-element math): fc = fma(u_t * kn_t, P_t, fc)
        over t = n - 1 .. 0 from 0, then S = fma(S_0, D_n, fc)."""
        T: cutlass.Constexpr = self.tokens
        us = cute.make_rmem_tensor((ROWS, T), Float32)  # u_t of this thread's rows
        for r in cutlass.range_constexpr(ROWS):
            for t in cutlass.range_constexpr(T):
                us[r, t] = old[b, t, base_row + 32 * r]
        for c in cutlass.range_constexpr(CHUNKS):
            key = 4 * (lane_in + LANES * c)
            d4 = load_shared_f32x4(at(decay, b, key))
            fc = zeros((ROWS, 4), Float32)
            for i in cutlass.range_constexpr(T):
                t = T - 1 - i
                if t < n:
                    k4 = load_shared_f32x4(at(keys, b, t, key))
                    p4 = load_shared_f32x4(at(suffix, b, t, key))
                    for r in cutlass.range_constexpr(ROWS):
                        u = us[r, t]
                        for p in cutlass.range_constexpr(2):
                            update = mul2((u, u), (k4[2 * p], k4[2 * p + 1]))
                            acc = (fc[r, 2 * p], fc[r, 2 * p + 1])
                            acc = fma2(update, (p4[2 * p], p4[2 * p + 1]), acc)
                            fc[r, 2 * p], fc[r, 2 * p + 1] = acc
            for r in cutlass.range_constexpr(ROWS):
                for p in cutlass.range_constexpr(2):
                    pair = (history[r, c, 2 * p], history[r, c, 2 * p + 1])
                    pair = fma2(pair, (d4[2 * p], d4[2 * p + 1]),
                                (fc[r, 2 * p], fc[r, 2 * p + 1]))  # fmt: skip
                    history[r, c, 2 * p], history[r, c, 2 * p + 1] = pair

    # -------------------------------------------------------------- prep role
    @cute.jit
    def prep_items(self, args, w, p, first, count, staging, ops, ops_full, done):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        output, gate, lower_bound, scale = args[10], args[11], args[13], args[14]
        q_s, k_s, v_s, d_s, _, g_s, beta_s, meta, fk_s, fp_s, fu_s, fd_s = ops
        convs, xs, gs, fas, sums, fsums, fbs, kgs, uss = staging
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        R: cutlass.Constexpr = args[10].shape[0] // self.tokens
        dim: cutlass.Constexpr = H * K
        P2: cutlass.Constexpr = 1 << (2 * T - 1).bit_length()  # q and k norms
        FOLD: cutlass.Constexpr = len(args) > 20
        lane, warp = p % 32, p // 32
        group_id, quarter = lane >> 2, lane & 3
        c = p  # this thread's channel
        bar = 1 + w  # this warpgroup's named barrier
        conv_type: cutlass.Constexpr = conv_state.element_type

        # Prologue: the first item's window and rows.
        start = first + w
        if w >= count:  # read a valid item's slots
            start = first
        head, request, slot, conv_slot, offset, pending, rec = self.item_info(
            args, start
        )
        if w < count:  # noqa: SIM102
            if slot > 0:
                self.stage_conv(args, convs, p, head, conv_slot, offset)
                self.stage_rows(args, xs, gs, fas, p, head, request)
                self.stage_head(args, fbs, kgs, uss, p, head, rec, pending)
        cute.arch.cp_async_commit_group()

        pr = zeros(4, Int64)  # TRACE: total, staging wait, done wait, body
        p0 = clock64()
        for j in cutlass.range(w, count, 2, unroll=1):
            item = first + j
            head, request = item // R, item % R
            token = request * T
            blk = 2 * w + (j // 2) % 2
            pa = clock64()
            cute.arch.cp_async_wait_group(0)
            sync_threads(bar, PREP_THREADS)
            pb = clock64()
            pr[1] = pr[1] + (pb - pa)
            if j >= 4:  # the block's previous item (j - 4) is scanned
                wait(done + blk, ((j - 4) // 4) % 2, scope="cta")
            pc = clock64()
            pr[2] = pr[2] + (pc - pb)
            if slot > 0:
                # The pending commit's records, read before this verify writes the
                # new ones into the same record slot (the barrier below).
                old_keys = zeros(T, Float32)
                old_gates = zeros(T, Float32)
                if cutlass.const_expr(FOLD):  # noqa: SIM102
                    if pending > 0:
                        for t in cutlass.range_constexpr(T):
                            old_keys[t] = Float32(kgs[t, c])
                            old_gates[t] = Float32(kgs[t, K + c])
                            fu_s[blk, t, c] = uss[t, c]
                # Early: the F_B rows of tile 0 (L2) and beta.
                wr = cute.make_rmem_tensor((8, 4), Uint32)
                for chunk in cutlass.range_constexpr(8):
                    feature = 32 * warp + group_id
                    low = load_shared_u32x2(self.fb_at(fbs, feature, chunk, quarter))
                    high = load_shared_u32x2(
                        self.fb_at(fbs, feature + 8, chunk, quarter)
                    )
                    wr[chunk, 0], wr[chunk, 2] = low
                    wr[chunk, 1], wr[chunk, 3] = high
                beta_value = Float32(0)
                if p < T:
                    beta_value = Float32(beta[token + p, head])
                a = triton_exp(Float32(a_log[head]))

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
                    coeff = [Float32(weights[part, i, head * K + c]) for i in range(4)]
                    inputs = [xs[t, part, c] for t in range(T)]
                    for t in cutlass.range_constexpr(T):
                        accum = Float32(0)
                        if cutlass.const_expr(self.has_bias):
                            accum = Float32(bias[channel])
                        for i in cutlass.range_constexpr(4):
                            tap = t + i
                            term = taps[tap] if tap < 3 else inputs[tap - 3]
                            accum = fma(Float32(term), coeff[i], accum)
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
                    args[19][record_row(args, request), head, t, c] = BFloat16(
                        values[1, t]
                    )

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
                        dt_channel = Float32(dt_bias[head * K + c])
                        for t in cutlass.range_constexpr(T):
                            total = warp_sum(old_keys[t] * old_keys[t])
                            if lane == 0:
                                fsums[t, warp] = total
                            old_decays[t] = self.decay(
                                a, old_gates[t], dt_channel, lower_bound
                            )

                # Gate projection: D[feature, token], tile m of this warp's 32
                # features; the decays and raw gates.
                for m in cutlass.range_constexpr(2):
                    if cutlass.const_expr(m == 1):
                        for chunk in cutlass.range_constexpr(8):
                            feature = 32 * warp + 16 + group_id
                            low = load_shared_u32x2(
                                self.fb_at(fbs, feature, chunk, quarter)
                            )
                            high = load_shared_u32x2(
                                self.fb_at(fbs, feature + 8, chunk, quarter)
                            )
                            wr[chunk, 0], wr[chunk, 2] = low
                            wr[chunk, 1], wr[chunk, 3] = high
                    chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                    acc = zeros((chains, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        chain: cutlass.Constexpr = (
                            chunk // 2 if self.gate_mode == 1 else 0
                        )
                        wa = tuple(wr[chunk, e] for e in range(4))
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
                            dt = dt_bias[head * K + feature]
                            d_s[blk, t, feature] = self.decay(a, raw_g, dt, lower_bound)
                            args[19][record_row(args, request), head, t, K + feature] = (
                                BFloat16(raw_g)
                            )
                if p < T:
                    beta_s[blk, p] = triton_sigmoid(beta_value)
                sync_threads(bar, PREP_THREADS)  # the norms' partial sums
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
                if cutlass.const_expr(FOLD):  # noqa: SIM102
                    if pending > 0:
                        # The commit's reverse token loop over this key: kn_t, the
                        # final_decay P_t token t's update is scaled by, and D_n.
                        final_decay = Float32(1)
                        for i in cutlass.range_constexpr(T):
                            t = T - 1 - i
                            if t < pending:
                                total = self.quad_sum(fsums[t, None]) + 1.0e-6
                                fk_s[blk, t, c] = mul_rn(rsqrt_ftz(total), old_keys[t])
                                fp_s[blk, t, c] = final_decay
                                final_decay = mul_rn(final_decay, old_decays[t])
                        fd_s[blk, c] = final_decay
                if p == 0:
                    meta[blk, 0] = 1
                    meta[blk, 1] = request
                    meta[blk, 2] = head
                    meta[blk, 3] = token
                    meta[blk, 4] = pending
                    meta[blk, 5] = slot
            else:
                if p == 0:
                    meta[blk, 0] = 0
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
            pr[3] = pr[3] + (clock64() - pc)
            cute.arch.mbarrier_arrive(ops_full + blk)

            # This block's next item's window and rows (the staging is free).
            if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                if p == 0:
                    wait_bulk_reads()
            sync_threads(bar, PREP_THREADS)
            if j + 2 < count:
                head, request, slot, conv_slot, offset, pending, rec = (
                    self.item_info(args, item + 2)
                )
                if slot > 0:
                    self.stage_conv(args, convs, p, head, conv_slot, offset)
                    self.stage_rows(args, xs, gs, fas, p, head, request)
                self.stage_head(args, fbs, kgs, uss, p, head, rec, pending)
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if p == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("TRACE prep cta {} w {} total {} wstage {} wdone {} body {}", cta, w, clock64() - p0, pr[1], pr[2], pr[3])
