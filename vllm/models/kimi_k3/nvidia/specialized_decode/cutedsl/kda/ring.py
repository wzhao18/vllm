# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaRing: KdaFlow's deferred-commit record step as two kernels.

1. ``prep_kernel`` (one 128-thread CTA per item, thread c = channel c): KdaFlow's
   per-item prep (convolution + SiLU, raw gates by the F_B MMA chain, decays,
   q / k norms, token couplings, fold weights, betas), the new key / gate records
   and the rolled convolution window; the item's scan operands go to a global
   operand block (L2-resident, read once by the ring).
2. ``kernel`` (persistent, 512 threads, one CTA per SM): a TMA ring of SLOTS state
   pages, each with its operand block on the page's barrier. Every item is scanned
   by all 16 warps: quarter q (threads 128 q + [0, 128)) owns state rows
   32 q + [0, 32), two rows per thread with KdaFlow's lane / key layout, and refills
   its rows of the page as soon as it has read them. Row owners solve the tokens,
   and quarter t computes token t's gated RMSNorm.

The math is KdaFlow's operation for operation (same per-thread accumulation order,
same butterfly pairing, same block-reduction trees), so the outputs, records and
state are bitwise KdaFlow's.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    launch,
    load_words,
    mma_into,
    scatter_reduce,
    zeros,
)
from .flow import (
    A_EXP,
    CHUNKS,
    CONSTS,
    DT,
    HEAD,
    LANES,
    MAX_ITEMS,
    META,
    PENDING,
    RECORD,
    REQUEST,
    SC_MINE,
    SLOT,
    SUMS,
    K,
    KdaFlow,
    couplings,
    store_global_f32x4,
)
from .recover import mul_rn, reciprocal, sqrt_approx, triton_sigmoid
from .specialized import (
    COPY_BYTES,
    PAGE_BYTES,
    fence_async_shared,
    load_shared_f32x4,
    sync_threads,
)
from .wide import rsqrt_ftz

QUARTERS = 4
QUARTER_THREADS = 128
THREADS = QUARTERS * QUARTER_THREADS
ROWS = 2  # state rows per scan thread: 32 q + 2 (tq // 8) + [0, 2)
SLOTS = 3  # the CTA's page ring (page + operand block per slot)
PREP_THREADS = 128
ALL_BARRIER = 1 + QUARTERS  # named barrier of the 512 threads (quarters: 1 + q)


def ops_layout(tokens):
    """Word offsets of the operand block: (xv, v, w, u, d, ce, betas, gates (BF16),
    words). xv: KdaFlow's swizzled 2q scan vectors."""
    tk = tokens * K
    xv, v, w, u, d = 0, 2 * tk, 3 * tk, 4 * tk, 5 * tk
    ce = d + K
    betas = ce + 16
    gates = betas + 8
    words = gates + tk // 2
    words = -(-words // 256) * 256  # 1 KiB multiple (bulk copy size)
    return xv, v, w, u, d, ce, betas, gates, words


def shared_bytes(tokens):
    words = ops_layout(tokens)[-1]
    return (
        SLOTS * (PAGE_BYTES + words * 4)
        + MAX_ITEMS * META * 4
        + 2 * tokens * K * 4
        + QUARTERS * 4 * 4
        + 8 * (SLOTS + 1)
        + 2048  # alignment
    )


def scratch_shape(items, tokens):
    return (items, ops_layout(tokens)[-1])


@asm()
def store_global_v4(pointer, values):
    """16 B register -> global store, default cache policy (read back from L2)."""
    return (
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        (
            pointer,
            *values,
        ),
    )


class KdaRing(KdaFlow):
    """KdaFlow's operands plus the operand-block scratch (last operand)."""

    def __init__(self, heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                 ctas, items, dies=False, mx=False):  # fmt: skip
        super().__init__(heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                         ctas, 1, dies, mx)  # fmt: skip
        self.items = items
        self.ops_at = self.mx_at + 2 * int(mx)
        self.layout = ops_layout(tokens)

    @cute.jit
    def __call__(self, *args):
        launch(self.prep_kernel, args, self.items, PREP_THREADS)
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    # ------------------------------------------------------------------- prep
    @cute.kernel
    def prep_kernel(self, args):
        T: cutlass.Constexpr = self.tokens
        c, _, _ = cute.arch.thread_idx()
        item, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        hc = alloc(smem, Float32, (CONSTS, K))
        raw = alloc(smem, BFloat16, (T, K), align=16)
        sums = alloc(smem, Float32, (SUMS, 4))
        head, request, slot, conv_slot, offset, pending, rec = self.item_info(
            args, item
        )
        if slot > 0:
            self.head_constants(hc, args, head, c)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if slot > 0:
            self.item_gates(args, c, head, request, raw)
            cute.arch.barrier()  # the raw gates; the head constants
            self.prep_ops(args, c, item, head, request, conv_slot, offset, pending,
                          rec, hc, raw, sums)  # fmt: skip

    @cute.jit
    def item_gates(self, args, c, head, request, raw):
        """raw[t, f] = BF16(F_B[head] @ F_A[request, t]): KdaFlow's gate projection
        for one item (warp w: features 32 w + [0, 32)), operands from global."""
        T: cutlass.Constexpr = self.tokens
        f_a, f_b = args[4], args[16]
        lane, warp = c % 32, c // 32
        group_id, quarter = lane >> 2, lane & 3
        wr = cute.make_rmem_tensor((2, 8, 4), cutlass.Uint32)
        for m in cutlass.range_constexpr(2):
            for chunk in cutlass.range_constexpr(8):
                feature = 32 * warp + 16 * m + group_id
                col = 16 * chunk + 4 * quarter
                low = load_words(at(f_b, head * K + feature, col), count=2)
                high = load_words(at(f_b, head * K + feature + 8, col), count=2)
                wr[m, chunk, 0], wr[m, chunk, 2] = low
                wr[m, chunk, 1], wr[m, chunk, 3] = high
        acc = cute.make_rmem_tensor((2, 4), Float32)
        for m in cutlass.range_constexpr(2):
            chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
            part = zeros((chains, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                chain: cutlass.Constexpr = chunk // 2 if self.gate_mode == 1 else 0
                wa = tuple(wr[m, chunk, e] for e in range(4))
                ab = load_words(
                    at(f_a, request * T + group_id % T, 16 * chunk + 4 * quarter),
                    count=2,
                    mutable=True,
                )
                mma_into(part[chain, None], wa, ab, bf16=True)
            for e in cutlass.range_constexpr(4):
                value = part[0, e]
                if cutlass.const_expr(self.gate_mode == 1):
                    value = Float32(0) + (
                        ((value + part[1, e]) + part[2, e]) + part[3, e]
                    )
                acc[m, e] = value
        for m in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(4):
                t = 2 * quarter + e % 2
                feature = 32 * warp + 16 * m + group_id + 8 * (e // 2)
                if t < T:
                    raw[t, feature] = BFloat16(acc[m, e])

    @cute.jit
    def prep_ops(self, args, c, item, head, request, conv_slot, offset, pending, rec,
                 hc, raw, sums):  # fmt: skip
        """KdaFlow.prep_item for one item from global inputs (thread c = channel /
        key c): the operand block into the scratch row, the records and window."""
        x, conv_state, gate, lower_bound, scale = (
            args[0],
            args[3],
            args[11],
            args[13],
            args[14],
        )
        uu, kg, ops = args[18], args[19], args[self.ops_at]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        XV, V, W, U, D, CE, BETAS, GATES, _ = self.layout
        conv_type: cutlass.Constexpr = conv_state.element_type
        consts = hc[None, c]
        index = couplings(T)
        token = request * T
        masked = Float32(0)
        if pending == 0:
            masked = Float32(-0.0)
        old_keys = zeros(T, Float32)
        old_gates = zeros(T, Float32)
        for t in cutlass.range_constexpr(T):
            u_old = masked
            if t < pending:
                old_keys[t] = Float32(kg[rec, head, t, c])
                old_gates[t] = Float32(kg[rec, head, t, K + c])
                u_old = uu[rec, head, t, c]
            ops[item, U + t * K + c] = u_old
        a = consts[A_EXP]
        dt = consts[DT]

        # Convolution + SiLU of q, k, v; the rolled window back in place.
        page = conv_state[Int64(conv_slot), None, None]
        values = cute.make_rmem_tensor((3, T), Float32)
        for part in cutlass.range_constexpr(3):
            row = part * dim + head * K + c
            taps = cute.make_rmem_tensor(3, conv_type)
            for t in cutlass.range_constexpr(3):
                taps[t] = page[row, offset + t]
            inputs = [x[token + t, part * dim + head * K + c] for t in range(T)]
            for t in cutlass.range_constexpr(T):
                accum = Float32(0)
                if cutlass.const_expr(self.has_bias):
                    accum = consts[12 + part]
                for i in cutlass.range_constexpr(4):
                    tap = t + i
                    term = taps[tap] if tap < 3 else inputs[tap - 3]
                    accum = fma(Float32(term), consts[4 * part + i], accum)
                values[part, t] = self.silu(accum)
            page[row, 0] = conv_type(taps[1])
            page[row, 1] = conv_type(taps[2])
            for t in cutlass.range_constexpr(T):
                page[row, 2 + t] = conv_type(inputs[t])
        decays = zeros(T, Float32)
        gview = cute.make_tensor(
            cute.recast_ptr(at(ops, item, GATES), dtype=BFloat16),
            cute.make_layout(T * K),
        )
        for t in cutlass.range_constexpr(T):
            ops[item, V + t * K + c] = values[2, t]
            gview[t * K + c] = gate[token + t, head * K + c]
            kg[rec, head, t, c] = BFloat16(values[1, t])
            raw_g = raw[t, c]
            kg[rec, head, t, K + c] = raw_g
            decays[t] = self.decay(a, Float32(raw_g), dt, lower_bound)
        cumulative = [decays[0]]
        for t in cutlass.range_constexpr(1, T):
            cumulative.append(mul_rn(cumulative[t - 1], decays[t]))
        ratio = {}
        for t in cutlass.range_constexpr(1, T):
            ratio[(t, t - 1)] = decays[t]
            for s in cutlass.range_constexpr(t - 1):
                ratio[(t, s)] = mul_rn(ratio[(t - 1, s)], decays[t])
        old_decays = zeros(T, Float32)
        if pending > 0:
            for t in cutlass.range_constexpr(T):
                old_decays[t] = self.decay(a, old_gates[t], dt, lower_bound)

        # KdaFlow's block reduction (the previous outputs' squares are zeros here).
        terms = [Float32(0) for _ in range(SUMS)]
        for t in cutlass.range_constexpr(T):
            terms[2 * t] = values[0, t] * values[0, t]
            terms[2 * t + 1] = values[1, t] * values[1, t]
            terms[NV + T * T + t] = old_keys[t] * old_keys[t]
        for i in cutlass.range_constexpr(len(index)):
            key = list(index)[i]
            kind, t, s = key
            weighted = values[1, s]
            if cutlass.const_expr(s < t):
                weighted = mul_rn(weighted, ratio[(t, s)])
            other = values[1, t] if kind == "c" else values[0, t]
            terms[NV + i] = mul_rn(weighted, other)
        self.block_sums(terms, sums, c, SUMS)
        cute.arch.barrier()
        fq = [Float32(0) for _ in range(T)]
        fk = [Float32(0) for _ in range(T)]
        for t in cutlass.range_constexpr(T):
            qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
            kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
            fq[t] = reciprocal(sqrt_approx(qn))
            fk[t] = reciprocal(sqrt_approx(kn))
        vec = [Float32(0) for _ in range(NV)]
        for t in cutlass.range_constexpr(T):
            q_t = mul_rn(mul_rn(values[0, t], fq[t]), scale)
            k_t = mul_rn(values[1, t], fk[t])
            vec[t] = mul_rn(cumulative[t], k_t)
            vec[T + t] = mul_rn(cumulative[t], q_t)
        chunk, lane_in, e = c // (4 * LANES), (c // 4) % LANES, c % 4
        for h in cutlass.range_constexpr(NV // 4):
            pos = (((chunk * 4 + e) * (NV // 4) + h) * LANES + lane_in) * 4
            store_global_v4(at(ops, item, XV + pos), [vec[4 * h + i] for i in range(4)])
        norm = Float32(0)
        for i in cutlass.range_constexpr(len(index)):
            key = list(index)[i]
            kind, t, s = key
            norm_i = mul_rn(fk[s], fk[t])
            if cutlass.const_expr(kind == "e"):
                norm_i = mul_rn(fk[s], mul_rn(fq[t], scale))
            norm = Float32(cutlass.select_(c == i, norm_i, norm))
        if c < len(index):
            ops[item, CE + c] = mul_rn(self.quad_sum(sums[NV + c, None]), norm)
        if c < T:
            beta = args[5]
            ops[item, BETAS + c] = triton_sigmoid(Float32(beta[token + c, head]))
        final_decay = Float32(1)
        for i in cutlass.range_constexpr(T):
            t = T - 1 - i
            w_t = Float32(0)
            if t < pending:
                total = self.quad_sum(sums[NV + T * T + t, None]) + 1.0e-6
                kn_t = mul_rn(rsqrt_ftz(total), old_keys[t])
                w_t = mul_rn(kn_t, final_decay)
                final_decay = mul_rn(final_decay, old_decays[t])
            ops[item, W + t * K + c] = w_t
        ops[item, D + c] = final_decay

    # ------------------------------------------------------------------- ring
    @cute.kernel
    def kernel(self, args):
        self.ring_body(args)

    @cute.jit
    def load_ops(self, args, opsb, s, table, j, state_full):
        """Table item j's operand block into slot ``s`` (the slot barrier's fifth
        arrival; a padded item's carries no bytes)."""
        requests: cutlass.Constexpr = args[10].shape[0] // self.tokens
        words: cutlass.Constexpr = self.layout[-1]
        barrier = state_full + s
        if table[j, SLOT] > 0:
            item = table[j, HEAD] * requests + table[j, REQUEST]
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, words * 4)
            copy_g2s(at(opsb, s, 0), at(args[self.ops_at], item, 0), words * 4, barrier)
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def ring_body(self, args):
        output, state = args[10], args[9]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        NV: cutlass.Constexpr = 2 * T
        XV, V, W, U, D, CE, BETAS, GATES, WORDS = self.layout
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (SLOTS, K * K), align=1024)
        opsb = alloc(smem, Float32, (SLOTS, WORDS), align=128)
        scratch = cute.make_tensor(
            cute.recast_ptr(at(pages, 0, 0), dtype=Int32), cute.make_layout(1024)
        )
        if cutlass.const_expr(self.dies):
            count = self.die_claim(args, tid, cta, scratch)
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        po_s = alloc(smem, Float32, (2, T, K), align=16)
        sums = alloc(smem, Float32, (QUARTERS, 4))
        bars = smem.allocate(cutlass.Uint64, SLOTS)
        state_full = bars
        gview = cute.make_tensor(
            cute.recast_ptr(at(opsb, 0, 0), dtype=BFloat16),
            cute.make_layout((SLOTS, 2 * WORDS), stride=(2 * WORDS, 1)),
        )
        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):  # 4 row quarters + the block
                cute.arch.mbarrier_init(state_full + s, PAGE_BYTES // COPY_BYTES + 1)
            cute.arch.mbarrier_init_fence()
        if tid < count:
            item = first + tid
            if cutlass.const_expr(self.dies):
                item = scratch[SC_MINE + tid]
            head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
                args, item
            )
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        # The state pages do not depend on the predecessor: in flight before the
        # PDL wait; the operand blocks after it.
        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):
                if s < count:
                    self.load_page(args, pages, s, table, s, state_full)
        q, tq = tid // QUARTER_THREADS, tid % QUARTER_THREADS
        lane = tq % 32
        lane_in = tq % LANES
        base = 32 * q + ROWS * (tq // LANES)
        nw = args[12][tq]
        cute.arch.sync_warp()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.mx):
            cute.arch.griddepcontrol_launch_dependents()
        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):
                if s < count:
                    self.load_ops(args, opsb, s, table, s, state_full)
        cute.arch.sync_warp()
        for j in cutlass.range(count, unroll=1):
            s = j % SLOTS
            slot, head = table[j, SLOT], table[j, HEAD]
            pending, rec = table[j, PENDING], table[j, RECORD]
            token = table[j, REQUEST] * T
            self.bwait(state_full + s, (j // SLOTS) % 2, "state_full")
            if slot > 0:
                pg = Float32(0)
                if q < T:
                    pg = Float32(gview[s, 2 * GATES + q * K + tq])
                acc = cute.make_rmem_tensor((ROWS, NV), Float32)
                destination = state[Int64(slot), head, None, None]
                self.ring_scan(acc, opsb, s, pages, pending, lane_in, base,
                               destination, pending > 0)  # fmt: skip
                fence_async_shared()
                sync_threads(1 + q, QUARTER_THREADS)  # the quarter's rows are read
                if tq == 0:  # noqa: SIM102
                    if j + SLOTS < count:
                        self.load_quarter(
                            args, pages, s, table, j + SLOTS, state_full, q
                        )
                cute.arch.sync_warp()
                values = self.ring_reduce(acc, lane_in)
                if (lane_in & 3) == 0:
                    own = base + (lane_in >> 2)
                    self.ring_solve(values, opsb, s, own, j % 2, po_s,
                                    args[18][rec, head, None, None])  # fmt: skip
                fence_async_shared()
                sync_threads(ALL_BARRIER, THREADS)  # the block is read; po complete
                if tid == 0:  # noqa: SIM102
                    if j + SLOTS < count:
                        self.load_ops(args, opsb, s, table, j + SLOTS, state_full)
                cute.arch.sync_warp()
                if q < T:
                    self.ring_finish(args, q, tq, lane, sums, po_s[j % 2, q, tq], pg,
                                     nw, token, head)  # fmt: skip
            else:
                if tid == 0:  # noqa: SIM102
                    if j + SLOTS < count:
                        self.load_page(args, pages, s, table, j + SLOTS, state_full)
                        self.load_ops(args, opsb, s, table, j + SLOTS, state_full)
                cute.arch.sync_warp()
                if q < T:
                    self.store_output(args, token + q, head, tq, BFloat16(0))

    @cute.jit
    def ring_scan(self, acc, opsb, s, pages, pending, lane_in, base, destination,
                  stored):  # fmt: skip
        """KdaFlow.scan over rows base + [0, 2) (keys 4 (lane_in + 8 j) + [0, 4)),
        operands from the slot's block."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        XV, V, W, U, D, CE, BETAS, GATES, WORDS = self.layout
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)
        us = cute.make_rmem_tensor((ROWS, T), Float32)
        for t in cutlass.range_constexpr(T):
            for rr in cutlass.range_constexpr(ROWS):
                us[rr, t] = opsb[s, U + t * K + base + rr]
        for rr in cutlass.range_constexpr(ROWS):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        for ch in cutlass.range_constexpr(CHUNKS):
            key = 4 * (lane_in + LANES * ch)
            d4 = load_shared_f32x4(at(opsb, s, D + key))
            w4 = [load_shared_f32x4(at(opsb, s, W + t * K + key)) for t in range(T)]
            xs = cute.make_rmem_tensor((4, NV), Float32)
            for e in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(NV // 4):
                    pos = (((ch * 4 + e) * (NV // 4) + h) * LANES + lane_in) * 4
                    x4 = load_shared_f32x4(at(opsb, s, XV + pos))
                    for i in cutlass.range_constexpr(4):
                        xs[e, 4 * h + i] = x4[i]
            for rr in cutlass.range_constexpr(ROWS):
                sv = load_shared_f32x4(at(pages, s, (base + rr) * K + key))
                new = [Float32(0) for _ in range(4)]
                for p in cutlass.range_constexpr(2):
                    fc = (start, start)
                    for i in cutlass.range_constexpr(T):
                        t = T - 1 - i
                        fc = self.pfma((us[rr, t], us[rr, t]),
                                       (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # noqa: E501
                    pair = self.pfma((sv[2 * p], sv[2 * p + 1]),
                                     (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                    new[2 * p], new[2 * p + 1] = pair
                for e in cutlass.range_constexpr(4):
                    for p in cutlass.range_constexpr(NV // 2):
                        pair = (acc[rr, 2 * p], acc[rr, 2 * p + 1])
                        pair = self.pfma((new[e], new[e]),
                                         (xs[e, 2 * p], xs[e, 2 * p + 1]), pair)  # noqa: E501
                        acc[rr, 2 * p], acc[rr, 2 * p + 1] = pair
                if stored:
                    store_global_f32x4(at(destination, base + rr, key), new)
            if cutlass.const_expr(ch % self.fold_fence == self.fold_fence - 1):
                cute.arch.sync_warp()

    @cute.jit
    def ring_reduce(self, acc, lane_in):
        """The 8 lanes' partials of rows base + [0, 2) -> lanes 4 r + [0, 4) hold the
        2q full sums of row base + r (KdaFlow's butterfly pairing: ^4, ^2, ^1)."""
        NV: cutlass.Constexpr = 2 * self.tokens
        upper = (lane_in & 4) != 0
        row = []
        for m in cutlass.range_constexpr(NV):
            send = acc[0, m] if upper else acc[1, m]
            keep = acc[1, m] if upper else acc[0, m]
            row.append(keep + cute.arch.shuffle_sync_bfly(send, 4))
        for stage in cutlass.range_constexpr(2):
            offset: cutlass.Constexpr = 2 >> stage
            row = [v + cute.arch.shuffle_sync_bfly(v, offset) for v in row]
        return row

    @cute.jit
    def ring_solve(self, values, opsb, s, own, parity, po_s, u_dst):
        """KdaFlow.solve for row ``own``; the outputs into po_s[parity]."""
        T: cutlass.Constexpr = self.tokens
        XV, V, W, U, D, CE, BETAS, GATES, WORDS = self.layout
        index = couplings(T)
        u = [Float32(0) for _ in range(T)]
        vs = [opsb[s, V + t * K + own] for t in range(T)]
        for t in cutlass.range_constexpr(T):
            dot = values[t]
            for k in cutlass.range_constexpr(t):
                dot = fma(opsb[s, CE + index[("c", t, k)]], u[k], dot)
            u[t] = (vs[t] - dot) * opsb[s, BETAS + t]
            u_dst[t, own] = u[t]
        for t in cutlass.range_constexpr(T):
            o = values[T + t]
            for k in cutlass.range_constexpr(t + 1):
                o = fma(opsb[s, CE + index[("e", t, k)]], u[k], o)
            po_s[parity, t, own] = Float32(BFloat16(o))

    @cute.jit
    def ring_finish(self, args, q, c, lane, sums, po, pg, nw, token, head):
        """KdaFlow.finish of token q (quarter q, thread c = channel c)."""
        eps = args[15]
        terms = [po * po]
        total, _ = scatter_reduce(terms, lane, 1)
        if lane == 0:
            sums[q, c // 32] = total
        sync_threads(1 + q, QUARTER_THREADS)
        sumsq = self.quad_sum(sums[q, None])
        variance = sumsq * (1.0 / 128) + eps
        rstd = reciprocal(sqrt_approx(variance))
        sigmoid_gate = triton_sigmoid(pg)
        result = po * rstd * nw * sigmoid_gate
        self.store_output(args, token + q, head, c, BFloat16(result))
