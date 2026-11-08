# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyPacked`` as a persistent kernel: one CTA of 16 warps per SM walks a
contiguous range of (request, head) items and bulk-copies the next item's FP32 state
page into shared memory while it runs the current one.

Why: ``KdaVerifyWide`` / ``KdaVerifyPacked`` run one CTA per item and one CTA per SM
(512 threads x 128 registers fill the register file), so every CTA serializes its
dependent loads (accepted count -> base slot -> 64 KB state page), its recurrence and
the drain of its 4-8 checkpoint stores, wave after wave; at q4 B128 (768 items, 3.6
waves) the kernel moves its bytes at ~8 TB/s of the measured ~12.5-13 TB/s copy
rate, its warps mostly stalled on those loads (ncu: long-scoreboard stalls on the
slot / page loads and the exit barrier lead). Here the item slots are read once at
the start (accepted counts and slot rows in parallel, one round trip), each live
item's state page arrives by ``cp.async.bulk`` into the shared page while the
previous item runs (issued once that item holds its page in registers), and the
checkpoint stores of one item drain under the next. (Measured on VR200 and dropped: a
second page in flight and L2 prefetches of the next item's conv taps, both within
noise or slower.)

Math: per item exactly ``KdaVerifyPacked``'s (Wide's prologue, gates, conv window,
packed token loop and gated norm; the state values pass through shared memory
unchanged), so outputs, checkpoints and the conv window are bit for bit
``KdaVerifyWide``'s / ``KdaSplit``'s verify."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_f32x4,
    mma_into,
    mul2,
    named_barrier,
    sigmoid,
    wait,
    warp_sum,
    zeros,
)
from .kernel import load_activation_bf16x4, load_bf16x4, sigmoids
from .recover import (
    div_full,
    mul_rn,
    reciprocal,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
)
from .verify_packed import CHUNKS, KdaVerifyPacked
from .verify_wide import THREADS

K = 128
PAGE_BYTES = K * K * 4  # one (slot, head) FP32 state page
COPY_BYTES = 16384  # per bulk copy of the page


def persistent_ctas(items, device=None):
    """One CTA per SM (never more than the items)."""
    if device is None:
        device = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(items, sms))


class KdaVerifyLoop(KdaVerifyPacked):
    """Persistent ``KdaVerifyPacked``: grid (ctas,), 512 threads; operands and launch
    arguments are ``KdaSplit``'s (FP32 state only: the page is staged as FP32)."""

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state):
        super().__init__(
            heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        assert not bf16_state, "KdaVerifyLoop stages an FP32 state page"
        self.items = heads * batch
        self.ctas = self.most = None  # set at compile time (the device's SMs)

    @cute.jit
    def __call__(self, *args):
        assert args[0].shape[0] // self.tokens * self.heads == self.items
        self.ctas = persistent_ctas(self.items)
        self.most = -(-self.items // self.ctas)  # items per CTA at most
        threads = [THREADS, 1, 1]
        launch(self.kernel, args, self.ctas, THREADS, max_number_threads=threads)

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        M: cutlass.Constexpr = self.most
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = thread % 32, thread // 32
        group, j = lane // 8, lane % 8
        first = Int32(self.items * cta // self.ctas)
        count = Int32(self.items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        page = alloc(smem, Float32, K * K, align=1024)
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, 128 * T) for _ in range(5)
        )
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        # per item of the CTA: [base slot, conv slot, offset, live index, rows[T]]
        meta = alloc(smem, Int32, (M, 4 + T))
        live_items = alloc(smem, Int32, M + 1)  # [n]: the n-th live item; [M]: count
        pairs: cutlass.Constexpr = (T + 1) // 2
        bars = alloc(smem, Int64, pairs + 1, 8).iterator
        ready, state_full = bars, bars + pairs

        # The CTA's item slots: thread i < count reads item i's accepted count and slot
        # row together (one round trip); thread 0 then numbers the live items.
        if thread < count:
            request = (first + thread) // H
            taken = accepted[request]
            for t in cutlass.range_constexpr(T):
                meta[thread, 4 + t] = slots[request, t]
            offset = taken - 1
            if taken < 1:  # a padding row's count may be 0 (its slots are <= 0)
                offset = Int32(0)
            meta[thread, 2] = offset
        if thread == 0:
            for p in cutlass.range_constexpr(pairs):
                cute.arch.mbarrier_init(ready + p, 128 + 32 * 8)
            cute.arch.mbarrier_init(state_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if thread == 0:
            live = Int32(0)
            for i in cutlass.range(count, unroll=1):
                base = meta[i, 4 + meta[i, 2]]
                meta[i, 0] = base
                meta[i, 1] = meta[i, 4]
                meta[i, 3] = live
                if base > 0:
                    live_items[live] = i
                    live += 1
            live_items[M] = live
            self.stage(state, page, state_full, meta, live_items, first, 0)
        cute.arch.sync_threads()
        nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
        row0 = 8 * warp + group

        for item_index in cutlass.range(count, unroll=1):
            item = first + item_index
            head, request = item % H, item // H
            token = request * T
            slot, conv_slot = meta[item_index, 0], meta[item_index, 1]
            offset, live = meta[item_index, 2], meta[item_index, 3]
            phase = live % 2
            if slot <= 0:  # a padding request
                cute.arch.griddepcontrol_wait()
                if thread < 128:
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + thread] = BFloat16(0)
            else:
                wait(state_full, phase, scope="cta")
                history = zeros((2, CHUNKS, 4), Float32)
                for r in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(CHUNKS):
                        pointer = page.iterator + (row0 + 4 * r) * K + 4 * (j + 8 * c)
                        values = load_f32x4(pointer, shared=True)
                        for e in cutlass.range_constexpr(4):
                            history[r, c, e] = values[e]
                cute.arch.sync_threads()  # every thread holds the page
                if thread == 0:  # the next live item's page into the freed page
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.stage(state, page, state_full, meta, live_items, first,
                               live + 1)  # fmt: skip
                own_gates = zeros(2, gate.element_type)
                if thread < 128:
                    self.loop_prologue(args, thread, token, head, offset, conv_slot,
                                       shared_q, shared_k, shared_v, reductions,
                                       shared_beta, own_gates, ready)  # fmt: skip
                elif thread < 128 + 32 * 8:
                    self.loop_gate(args, thread // 32 - 4, lane, head, token,
                                   shared_decay, own_gates)  # fmt: skip
                    for p in cutlass.range_constexpr(pairs):
                        cute.arch.mbarrier_arrive(ready + p)
                else:
                    cute.arch.griddepcontrol_wait()
                    self.loop_own_gates(args, token, head, thread // 64, thread % 64,
                                        own_gates)  # fmt: skip
                cute.arch.griddepcontrol_launch_dependents()

                for t in cutlass.range_constexpr(T):
                    if cutlass.const_expr(t % 2 == 0):  # token pair t // 2's operands
                        cute.arch.mbarrier_wait(ready + t // 2, phase)
                    self.loop_token(history, t, group, j, row0, shared_q, shared_k,
                                    shared_decay, shared_v, shared_beta, shared_o,
                                    state, meta, item_index, head)  # fmt: skip
                q_last = self.column_chunks(shared_q, T - 1, j)
                sums = self.row_dots(history, q_last)
                upper = (j & 4) != 0
                send = sums[0] if upper else sums[1]
                keep = sums[1] if upper else sums[0]
                owned = keep + cute.arch.shuffle_sync_bfly(send, 4)
                owned += cute.arch.shuffle_sync_bfly(owned, 2)
                owned += cute.arch.shuffle_sync_bfly(owned, 1)
                if (j & 3) == 0:
                    row = row0 + 4 * (j // 4)
                    position = (T - 1) * 128 + self.output_position(row)
                    shared_o[position] = self.round_out(owned)
                cute.arch.sync_threads()
                t, c, glane = thread // 64, thread % 64, thread % 32
                if t < T:
                    self.gated_norm(t, c, glane, shared_o, own_gates, nw, eps, output,
                                    token, head)  # fmt: skip
                # the item's shared operands and outputs are free for the next one
                cute.arch.sync_threads()
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def stage(self, state, page, state_full, meta, live_items, first, n):
        """Thread 0: bulk-copy the ``n``-th live item's state page (if there is one)
        into ``page`` on ``state_full``."""
        H: cutlass.Constexpr = self.heads
        M: cutlass.Constexpr = self.most
        if n < live_items[M]:
            index = live_items[n]
            slot, head = meta[index, 0], (first + index) % H
            source = state[Int64(slot), None, None, None]
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, PAGE_BYTES)
            rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
            for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
                copy_g2s(
                    page.iterator + i * rows * K,
                    at(source, head, i * rows, 0),
                    COPY_BYTES,
                    state_full,
                )

    def round_out(self, value):
        """A recurrence output as the gated norm reads it (KdaSplit: FP32)."""
        return value

    @cute.jit
    def loop_own_gates(self, args, token, head, t, c, own_gates):
        """``load_own_gates`` of item head ``head``."""
        gate = args[11]
        if t < self.tokens:
            for h in cutlass.range_constexpr(2):
                own_gates[h] = gate[token + t, head * 128 + c + 64 * h]

    @cute.jit
    def loop_prologue(self, args, thread, token, head, offset, conv_slot, shared_q,
                      shared_k, shared_v, reductions, shared_beta, own_gates,
                      ready):  # fmt: skip
        """``KdaVerifyWide.prologue`` of item head ``head`` (same operations)."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        scale = args[14]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        lane, warp = thread % 32, thread // 32
        channel = head * 128 + thread
        conv_page = conv_state[Int64(conv_slot), None, None]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), weights.element_type)
        accum = zeros(3 * T, Float32)
        for s in cutlass.range_constexpr(3):
            for t in cutlass.range_constexpr(3):
                taps[s, t] = conv_page[s * dim + channel, offset + t]
            for t in cutlass.range_constexpr(4):
                coeff[s, t] = weights[s, t, channel]
            if cutlass.const_expr(self.has_bias):
                accum[s] = bias[s * dim + channel]
                for t in cutlass.range_constexpr(1, T):
                    accum[3 * t + s] = accum[s]
        cute.arch.griddepcontrol_wait()
        inputs = cute.make_rmem_tensor(3 * T, BFloat16)
        for t in cutlass.range_constexpr(T):
            for s in cutlass.range_constexpr(3):
                inputs[3 * t + s] = x[token + t, s * dim + channel]
        beta_in = beta[token + thread % T, head]
        self.loop_own_gates(args, token, head, thread // 64, thread % 64, own_gates)
        for p in cutlass.range_constexpr((T + 1) // 2):
            tokens = [t for t in (2 * p, 2 * p + 1) if t < T]
            for t in tokens:  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), Float32(coeff[s, i]), acc)
            raw = [accum[3 * t + s] for t in tokens for s in range(3)]
            act = sigmoids(raw + [beta_in] if p == 0 else raw)
            if cutlass.const_expr(p == 0):  # noqa: SIM102
                if thread < T:
                    shared_beta[thread] = act[len(raw)]
            qks = []
            for jj, t in enumerate(tokens):
                q = accum[3 * t] * act[3 * jj]
                k = accum[3 * t + 1] * act[3 * jj + 1]
                shared_v[128 * t + thread] = accum[3 * t + 2] * act[3 * jj + 2]
                qks += [q, k]
                square_q, square_k = warp_sum(q * q), warp_sum(k * k)
                if lane == 0:
                    reductions[8 * t + warp] = square_q
                    reductions[8 * t + warp + 4] = square_k
            named_barrier(1, 128)
            for jj, t in enumerate(tokens):
                r = [reductions[8 * t + i] for i in range(8)]
                qn = (r[0] + r[2]) + (r[1] + r[3])
                kn = (r[4] + r[6]) + (r[5] + r[7])
                fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
                fk = cute.math.rsqrt(kn + 1.0e-6, fastmath=True)
                shared_q[128 * t + thread] = qks[2 * jj] * fq
                shared_k[128 * t + thread] = qks[2 * jj + 1] * fk
            cute.arch.mbarrier_arrive(ready + p)
        for s in cutlass.range_constexpr(3):  # the conv window
            window = (taps[s, 1], taps[s, 2], *(inputs[3 * t + s] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[s * dim + channel, i] = value

    @cute.jit
    def loop_gate(self, args, g, lane, head, token, shared_decay, own_gates):
        """``KdaVerifyWide.gate`` of item head ``head`` (same operations)."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        T: cutlass.Constexpr = self.tokens
        group, quarter = lane >> 2, lane & 3
        base = 16 * g
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
        row = head * 128 + base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(weight, row, col))
            high = load_bf16x4(at(weight, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = a_log[head]
        cute.arch.griddepcontrol_wait()
        thread = 128 + 32 * g + lane
        self.loop_own_gates(args, token, head, thread // 64, thread % 64, own_gates)
        fa_row = token + group % T  # the lane's token column
        ar = [
            load_activation_bf16x4(at(f_a, fa_row, 16 * c + 4 * quarter))
            for c in range(8)
        ]
        acc = zeros(4, Float32)
        for chunk in cutlass.range_constexpr(8):
            wa = tuple(wr[chunk, e] for e in range(4))
            mma_into(acc, wa, ar[chunk], bf16=True)
        for i in cutlass.range_constexpr(2):
            for column in cutlass.range_constexpr(2):
                raw_g = Float32(BFloat16(acc[2 * i + column]))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gate = sigmoid(gate, False)
                decay = cute.math.exp(lower_bound * gate, fastmath=True)
                t = 2 * quarter + column
                if t < T:
                    shared_decay[128 * t + base + features[i]] = decay

    @cute.jit
    def loop_token(self, history, t, group, j, row0, shared_q, shared_k,
                   shared_decay, shared_v, shared_beta, shared_o, state, meta,
                   item_index, head):  # fmt: skip
        """``KdaVerifyPacked.packed_token`` with the checkpoint slot from ``meta``."""
        k = self.column_chunks(shared_k, t, j)
        decay = self.column_chunks(shared_decay, t, j)
        values = [Float32(0), Float32(0)]
        if cutlass.const_expr(t > 0):
            values = self.row_dots(history, self.column_chunks(shared_q, t - 1, j))
        for c in cutlass.range_constexpr(CHUNKS):
            for e in cutlass.range_constexpr(4):
                x = decay[c, e]
                pair = mul2((history[0, c, e], history[1, c, e]), (x, x))
                history[0, c, e], history[1, c, e] = pair
        values += self.row_dots(history, k)
        upper = (j & 4) != 0
        for i in cutlass.range_constexpr(2):
            send = values[i] if upper else values[i + 2]
            keep = values[i + 2] if upper else values[i]
            values[i] = keep + cute.arch.shuffle_sync_bfly(send, 4)
        upper = (j & 2) != 0
        send = values[0] if upper else values[1]
        keep = values[1] if upper else values[0]
        owned = keep + cute.arch.shuffle_sync_bfly(send, 2)
        owned += cute.arch.shuffle_sync_bfly(owned, 1)
        base = 8 * group
        dot_k = [cute.arch.shuffle_sync(owned, base + 4 + 2 * r) for r in range(2)]
        beta = shared_beta[t]
        v_new = [
            (shared_v[128 * t + row0 + 4 * r] - dot_k[r]) * beta for r in range(2)
        ]
        for c in cutlass.range_constexpr(CHUNKS):
            for e in cutlass.range_constexpr(4):
                x = k[c, e]
                pair = (history[0, c, e], history[1, c, e])
                pair = fma2((x, x), (v_new[0], v_new[1]), pair)
                history[0, c, e], history[1, c, e] = pair
        if cutlass.const_expr(t > 0):  # noqa: SIM102
            if (j & 5) == 0:
                row = row0 + 4 * (j // 2)
                shared_o[(t - 1) * 128 + self.output_position(row)] = self.round_out(
                    owned
                )
        slot = meta[item_index, 4 + t]
        page = state[Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(CHUNKS):
                values4 = [history[r, c, e] for e in range(4)]
                pointer = at(page, head, row0 + 4 * r, 4 * (j + 8 * c))
                self.store_state(pointer, values4, slot > 0)


class KdaVerifyLoopStock(KdaVerifyLoop):
    """``KdaVerifyLoop`` with the stock composed verify's math (the Triton formulas
    and instructions ``recover.KdaSplit`` / ``wide.KdaWide`` reproduce): BF16-rounded
    convolution + SiLU outputs (``div.full`` SiLU), q / k L2 norms by ``sqrt.approx`` /
    ``rcp.approx`` and round-to-nearest products, BF16 raw gates and the Triton
    exp / sigmoid decays and betas, BF16-rounded recurrence outputs, and the gated
    RMS norm by ``sqrt.approx`` / ``rcp.approx`` with the Triton sigmoid of the output
    gate. FP32 state and recurrence as stock; only FP32 reduction order differs
    (the recurrence dots, q / k / output squares)."""

    @cute.jit
    def round_out(self, value):
        return Float32(BFloat16(value))

    @cute.jit
    def stock_silu(self, value):
        """The convolution's SiLU: x / (1 + exp(-x)), stored BF16 (stock)."""
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def loop_prologue(self, args, thread, token, head, offset, conv_slot, shared_q,
                      shared_k, shared_v, reductions, shared_beta, own_gates,
                      ready):  # fmt: skip
        """Channel ``thread``: the stock convolution (tap order), SiLU (BF16) and
        q / k L2 norms per token pair p (then ``ready[p]``); thread t < q token t's
        beta (Triton sigmoid); the conv window; its raw output gates."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        scale = args[14]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        lane, warp = thread % 32, thread // 32
        channel = head * 128 + thread
        conv_page = conv_state[Int64(conv_slot), None, None]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), Float32)
        accum = zeros(3 * T, Float32)
        for s in cutlass.range_constexpr(3):
            for t in cutlass.range_constexpr(3):
                taps[s, t] = conv_page[s * dim + channel, offset + t]
            for t in cutlass.range_constexpr(4):
                coeff[s, t] = Float32(weights[s, t, channel])
            if cutlass.const_expr(self.has_bias):
                for t in cutlass.range_constexpr(T):
                    accum[3 * t + s] = Float32(bias[s * dim + channel])
        cute.arch.griddepcontrol_wait()
        inputs = cute.make_rmem_tensor(3 * T, BFloat16)
        for t in cutlass.range_constexpr(T):
            for s in cutlass.range_constexpr(3):
                inputs[3 * t + s] = x[token + t, s * dim + channel]
        if thread < T:
            shared_beta[thread] = triton_sigmoid(Float32(beta[token + thread, head]))
        self.loop_own_gates(args, token, head, thread // 64, thread % 64, own_gates)
        for p in cutlass.range_constexpr((T + 1) // 2):
            tokens = [t for t in (2 * p, 2 * p + 1) if t < T]
            qks = []
            for t in tokens:  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), coeff[s, i], acc)
                q = self.stock_silu(accum[3 * t])
                k = self.stock_silu(accum[3 * t + 1])
                shared_v[128 * t + thread] = self.stock_silu(accum[3 * t + 2])
                qks += [q, k]
                square_q, square_k = warp_sum(q * q), warp_sum(k * k)
                if lane == 0:
                    reductions[8 * t + warp] = square_q
                    reductions[8 * t + warp + 4] = square_k
            named_barrier(1, 128)
            for jj, t in enumerate(tokens):
                r = [reductions[8 * t + i] for i in range(8)]
                qn = (r[0] + r[2]) + (r[1] + r[3]) + 1.0e-6
                kn = (r[4] + r[6]) + (r[5] + r[7]) + 1.0e-6
                fq = reciprocal(sqrt_approx(qn))
                fk = reciprocal(sqrt_approx(kn))
                shared_q[128 * t + thread] = mul_rn(mul_rn(qks[2 * jj], fq), scale)
                shared_k[128 * t + thread] = mul_rn(qks[2 * jj + 1], fk)
            cute.arch.mbarrier_arrive(ready + p)
        for s in cutlass.range_constexpr(3):  # the conv window
            window = (taps[s, 1], taps[s, 2], *(inputs[3 * t + s] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[s * dim + channel, i] = value

    @cute.jit
    def loop_gate(self, args, g, lane, head, token, shared_decay, own_gates):
        """Warp ``g`` of 8: the F_B tile of features [16 g, 16 g + 16) (one chain of
        eight MMAs, as stock's gate mode 0 order), BF16 raw gates and the Triton
        decays exp(lb * sigmoid(exp(A_log) * (g + dt)))."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        T: cutlass.Constexpr = self.tokens
        group, quarter = lane >> 2, lane & 3
        base = 16 * g
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
        row = head * 128 + base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(weight, row, col))
            high = load_bf16x4(at(weight, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = triton_exp(Float32(a_log[head]))
        cute.arch.griddepcontrol_wait()
        thread = 128 + 32 * g + lane
        self.loop_own_gates(args, token, head, thread // 64, thread % 64, own_gates)
        fa_row = token + group % T  # the lane's token column
        ar = [
            load_activation_bf16x4(at(f_a, fa_row, 16 * c + 4 * quarter))
            for c in range(8)
        ]
        acc = zeros(4, Float32)
        for chunk in cutlass.range_constexpr(8):
            wa = tuple(wr[chunk, e] for e in range(4))
            mma_into(acc, wa, ar[chunk], bf16=True)
        for i in cutlass.range_constexpr(2):
            for column in cutlass.range_constexpr(2):
                raw_g = Float32(BFloat16(acc[2 * i + column]))
                gate = lower_bound * triton_sigmoid(a * (raw_g + dt[i]))
                t = 2 * quarter + column
                if t < T:
                    shared_decay[128 * t + base + features[i]] = triton_exp(gate)

    @cute.jit
    def gated_norm(self, t, c, glane, shared_o, own_gates, nw, eps, output, token,
                   head):  # fmt: skip
        """Token t's gated RMS norm of channels c and c + 64 (BF16 recurrence
        outputs): each 8-lane group sums the 128 squares; rstd by ``sqrt.approx`` /
        ``rcp.approx``, the Triton sigmoid of the output gate (stock's)."""
        row_o = shared_o.iterator + 128 * t
        raw_o = [shared_o[128 * t + self.output_position(c + 64 * h)] for h in range(2)]
        sums = cute.make_rmem_tensor(4, Float32)
        for g4 in cutlass.range_constexpr(4):
            o4 = load_f32x4(row_o + glane % 8 * 16 + g4 * 4, shared=True)
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g4] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = reciprocal(sqrt_approx(sumsq * (1.0 / 128) + eps))
        for h in cutlass.range_constexpr(2):
            activated = triton_sigmoid(Float32(own_gates[h]))
            result = raw_o[h] * rstd * nw[h] * activated
            output[token + t, head * 128 + c + 64 * h] = BFloat16(result)
