# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyLoopStock`` with its per-item prologue and gated norm moved off the
recurrence warps onto feed warps that work from shared memory: one persistent CTA
per SM of 8 recurrence warps and 4 feed warps, so that the SM streams checkpoints
without the per-item gaps.

Why: the plain FP32 verify writes q state checkpoints per (request, head) (stock's
``spec_state_indices``: one 64 KiB FP32 page per token) and reads one page, so at q4
80% of its bytes are checkpoint writes. LoopStock (one 16-warp CTA per SM) runs
every item as prologue (dependent operand loads, convolution, norms, F_B MMAs),
token loop and tail on the same warps, and its CTAs run the same item sequence from
the same start: during the token loops all SMs store at once (at the HBM write
ceiling: 212 x 256 KiB in ~4 us), during the prologues and tails none does. VR200
globaltimer stamps, q4 B90: prologue 2.25 us, token loop 3.75 us, tail 0.75 us per
item. Measured alternatives: operands staged one item ahead on the same warps
(parity: the token loops stay at the write ceiling, the gaps remain), two 8-warp
CTAs per SM (they stay in lockstep), feed warps loading from global memory (the
prologue's chain of loads takes 5-6 us beside the checkpoint stores: the feed must
not wait on global memory).

Here, per SM (items head-major: a CTA's contiguous range spans at most two heads):

* feed warps 8-11 stage the CTA's head constants once (the F_B rows at a padded
  pitch, conv weights and bias, dt; exp(A_log)), prefetch each live item's operand
  rows two items ahead with 16-byte cp.async (q x 3 x rows, F_A, gate rows, the beta
  chunk, the conv taps; arrivals on an mbarrier), and per item run LoopStock's
  convolution prologue (thread = channel), the eight F_B gate tiles (two per warp)
  and the output gate rows from shared memory into one of two operand buffers, then
  the gated norm of the item the recurrence warps finished;
* recurrence warps 0-7 hold the state rows of LoopStock's threads (warp w, lane l)
  and (warp w + 8, lane l): rows 8 w + g, + 4, + 64, + 68 (64 FP32 values a thread);
  per live item they wait for its page (staged one item ahead, as LoopStock) and
  operands, run the token loop (``pair_token``: per row exactly LoopStock's
  operations; each column chunk of k / decay / q loaded once for both row sets),
  the last token's query dots, and hand the raw outputs to the feed warps.

VR200 (1 GPU, 69 layers, own cold FP32 slots, CUDA graph, us per layer, q4),
LoopStock -> Feed: B41 15.52 -> 14.22, B64 16.19 -> 14.86, B90 23.15 -> 20.64,
B128 31.31 -> 27.73, B140 32.70 -> 28.44, B256 62.87 -> 55.94, B512 117.63 ->
106.04; one item per CTA (B12-35) +0.4..+1.1 (``prepared.FP32_FEED``: from B41).

Math: LoopStock's, operation for operation (every value meets the same instructions
in the same order), so outputs, every checkpoint and the conv window are bit for
bit ``KdaVerifyLoopStock``'s. 384 threads (up to 168 registers a thread), one CTA
per SM; shared memory ~190 KB at q4 (q <= 4: at q8 the buffers exceed it)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_f32x4,
    load_words,
    mma_into,
    mul2,
    named_barrier,
    wait,
    warp_sum,
    zeros,
)
from .recover import mul_rn, reciprocal, sqrt_approx, triton_exp, triton_sigmoid
from .verify_loop import COPY_BYTES, PAGE_BYTES, K, KdaVerifyLoopStock, persistent_ctas
from .verify_packed import CHUNKS
from .window import conv_put, conv_tap

MATH_THREADS = 256  # recurrence warps 0-7
FEED_THREADS = 128  # feed warps 8-11
ALL_THREADS = MATH_THREADS + FEED_THREADS
MATH_BARRIER, FEED_BARRIER = 2, 3  # named barriers of the two warp groups
FB_PITCH = 136  # F_B row pitch in shared memory (BF16): 272 B, 2-way fragment loads
BETA_CHUNK = 8  # BF16 betas per staged 16-byte chunk
MAX_TOKENS = 4  # shared memory: the operand buffers of q > 4 do not fit
SETS = 2  # state row sets per recurrence thread: rows r and r + 64


@asm()
def copies_arrive(barrier):
    """Arrive on ``barrier`` once every earlier cp.async of the thread has landed
    (no pending-count increment: the barrier counts the issuing threads)."""
    code = "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];"
    return code, "r,~{memory}", (barrier,)


def staged_words(pointer):
    """Two 32-bit words (four BF16) from shared memory."""
    return load_words(pointer, count=2, shared=True, word_type="u32")


def part(tensor, b, size):
    """Buffer ``b`` of a [2 x size] shared tensor as a [size] tensor."""
    return cute.make_tensor(tensor.iterator + b * size, cute.make_layout(size))


class KdaVerifyFeed(KdaVerifyLoopStock):
    """``KdaVerifyLoopStock`` on 8 recurrence + 4 feed warps per SM (module
    docstring): grid (ctas,), 384 threads; operands and launch arguments are
    ``KdaSplit``'s (FP32 state, FP32 conv weights / bias / dt; the x / F_A / beta /
    gate views share one row pitch that is a multiple of 16 bytes, the conv state's
    rows are its q + 2 contiguous taps)."""

    shared_carveout = None

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state):
        super().__init__(
            heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        assert tokens <= MAX_TOKENS, f"KdaVerifyFeed: q {tokens} > {MAX_TOKENS}"
        self.requests = batch

    @cute.jit
    def __call__(self, *args):
        assert args[0].shape[0] // self.tokens * self.heads == self.items
        self.ctas = persistent_ctas(self.items)
        self.most = -(-self.items // self.ctas)  # items per CTA at most
        assert self.most <= MATH_THREADS, f"{self.items} items on {self.ctas} CTAs"
        # head-major ranges of at most `most` <= B items: at most two heads per CTA
        assert self.most <= self.requests
        threads = [ALL_THREADS, 1, 1]
        launch(self.kernel, args, self.ctas, ALL_THREADS, max_number_threads=threads)

    def head_of(self, item):
        return item // self.requests

    def request_of(self, item):
        return item % self.requests

    @cute.kernel
    def kernel(self, args):
        weights, bias, a_log, dt_bias = args[1], args[2], args[6], args[7]
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        M: cutlass.Constexpr = self.most
        W: cutlass.Constexpr = self.tokens + 2  # conv taps per channel row
        TK: cutlass.Constexpr = 128 * T
        # 16-byte chunks of four FP32 constants
        assert weights.element_type.width == 32 and dt_bias.element_type.width == 32
        assert bias.element_type.width == 32
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = thread % 32, thread // 32
        group, j = lane // 8, lane % 8
        first = Int32(self.items * cta // self.ctas)
        count = Int32(self.items * (cta + 1) // self.ctas) - first
        h0, h1 = self.head_of(first), self.head_of(first + count - 1)

        smem = cutlass.utils.SmemAllocator()
        page = alloc(smem, Float32, K * K, align=1024)
        # the CTA's head constants (two head slots)
        fb = alloc(smem, BFloat16, (2, 128, FB_PITCH), align=128)
        conv_w = alloc(smem, weights.element_type, (2, 3, 4, 128), align=128)
        conv_b = alloc(smem, bias.element_type, (2, 3, 128), align=128)
        dts = alloc(smem, dt_bias.element_type, (2, 128), align=128)
        exp_a = alloc(smem, Float32, 2)
        # raw operand rows (two items in flight)
        raw = (
            alloc(smem, BFloat16, (2, T, 3, 128), align=128),  # x rows
            alloc(smem, BFloat16, (2, T, 128), align=128),  # F_A rows
            alloc(smem, BFloat16, (2, T, 128), align=128),  # gate rows
            alloc(smem, BFloat16, (2, T, BETA_CHUNK), align=128),  # beta chunks
            alloc(smem, BFloat16, (2, 3, 128 * W), align=128),  # conv taps
        )
        # prepared operands (two buffers): q, k, decay, v, raw outputs, betas, gates
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, (2, TK)) for _ in range(5)
        )
        shared_beta = alloc(smem, Float32, (2, T))
        gates = alloc(smem, gate.element_type, (2, TK), align=128)
        reductions = alloc(smem, Float32, 8 * T)
        meta = alloc(smem, Int32, (M, 4 + T))
        live_items = alloc(smem, Int32, M + 1)
        bars = alloc(smem, cutlass.Int64, 9, 8).iterator
        state_full, raw_full = bars, bars + 1
        ready, done, free = bars + 3, bars + 5, bars + 7

        if thread < count:
            request = self.request_of(first + thread)
            taken = accepted[request]
            for t in cutlass.range_constexpr(T):
                meta[thread, 4 + t] = slots[request, t]
            offset = taken - 1
            if taken < 1:  # a padding row's count may be 0 (its slots are <= 0)
                offset = Int32(0)
            meta[thread, 2] = offset
        if thread == 0:
            cute.arch.mbarrier_init(state_full, 1)
            for b in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(raw_full + b, FEED_THREADS)  # rows landed
                cute.arch.mbarrier_init(ready + b, FEED_THREADS)  # operands of b
                cute.arch.mbarrier_init(done + b, MATH_THREADS)  # raw outputs of b
                cute.arch.mbarrier_init(free + b, FEED_THREADS)  # outputs of b normed
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
        lives = live_items[M]

        if thread < MATH_THREADS:  # the recurrence warps
            row0 = 8 * warp + group  # and row0 + 64 (LoopStock's warp + 8)
            cute.arch.griddepcontrol_wait()  # (before the checkpoint stores)
            for n in cutlass.range(lives, unroll=1):
                index = live_items[n]
                head = self.head_of(first + index)
                b, use = n % 2, n // 2
                wait(state_full, n % 2, scope="cta")
                histories = [zeros((2, CHUNKS, 4), Float32) for _ in range(SETS)]
                for s in cutlass.range_constexpr(SETS):
                    for r in cutlass.range_constexpr(2):
                        for c in cutlass.range_constexpr(CHUNKS):
                            row = row0 + 64 * s + 4 * r
                            pointer = page.iterator + row * K + 4 * (j + 8 * c)
                            values = load_f32x4(pointer, shared=True)
                            for e in cutlass.range_constexpr(4):
                                histories[s][r, c, e] = values[e]
                named_barrier(MATH_BARRIER, MATH_THREADS)  # every thread holds the page
                if thread == 0:  # the next live item's page into the freed page
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.stage(state, page, state_full, meta, live_items, first,
                               n + 1)  # fmt: skip
                wait(ready + b, use % 2, scope="cta")  # the item's operands
                if n >= 2:  # buffer b's outputs of item n - 2 are normed
                    wait(free + b, (use - 1) % 2, scope="cta")
                q_b, o_b = part(shared_q, b, TK), part(shared_o, b, TK)
                for t in cutlass.range_constexpr(T):
                    self.pair_token(histories, t, group, j, row0, q_b,
                                    part(shared_k, b, TK), part(shared_decay, b, TK),
                                    part(shared_v, b, TK), part(shared_beta, b, T),
                                    o_b, state, meta, index, head)  # fmt: skip
                q_last = self.column_chunks(q_b, T - 1, j)
                for s in cutlass.range_constexpr(SETS):
                    sums = self.row_dots(histories[s], q_last)
                    upper = (j & 4) != 0
                    send = sums[0] if upper else sums[1]
                    keep = sums[1] if upper else sums[0]
                    owned = keep + cute.arch.shuffle_sync_bfly(send, 4)
                    owned += cute.arch.shuffle_sync_bfly(owned, 2)
                    owned += cute.arch.shuffle_sync_bfly(owned, 1)
                    if (j & 3) == 0:
                        row = row0 + 64 * s + 4 * (j // 4)
                        position = (T - 1) * 128 + self.output_position(row)
                        o_b[position] = self.round_out(owned)
                cute.arch.mbarrier_arrive(done + b)
        else:  # the feed warps
            p = thread - MATH_THREADS  # channel / feed thread
            self.stage_constants(args, p, h0, h1, fb, conv_w, conv_b, dts)
            if p == 0:
                exp_a[0] = triton_exp(Float32(a_log[h0]))
                exp_a[1] = triton_exp(Float32(a_log[h1]))
            cute.arch.griddepcontrol_wait()
            for m in cutlass.range_constexpr(2):  # the first two items' rows
                self.stage_rows(args, p, m, lives, first, meta, live_items, raw,
                                raw_full)  # fmt: skip
            nw = [norm_weight[p % 64 + 64 * h] for h in range(2)]
            for i in cutlass.range(count, unroll=1):  # padding requests: zeros
                if meta[i, 0] <= 0:
                    item = first + i
                    token = self.request_of(item) * T
                    head = self.head_of(item)
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + p] = BFloat16(0)
            cute.arch.cp_async_wait_group(0)  # (this thread's constants)
            named_barrier(FEED_BARRIER, FEED_THREADS)
            if lives > 0:
                self.feed(args, p, 0, lives, first, h0, meta, live_items, raw,
                          raw_full, fb, conv_w, conv_b, dts, exp_a, shared_q,
                          shared_k, shared_decay, shared_v, shared_beta, gates,
                          reductions, ready)  # fmt: skip
            cute.arch.griddepcontrol_launch_dependents()
            for n in cutlass.range(lives, unroll=1):
                b, use = n % 2, n // 2
                if n + 1 < lives:  # its buffers: item n - 1 normed, its reads done
                    self.feed(args, p, n + 1, lives, first, h0, meta, live_items, raw,
                              raw_full, fb, conv_w, conv_b, dts, exp_a, shared_q,
                              shared_k, shared_decay, shared_v, shared_beta, gates,
                              reductions, ready)  # fmt: skip
                wait(done + b, use % 2, scope="cta")  # item n's raw outputs
                item = first + live_items[n]
                token, head = self.request_of(item) * T, self.head_of(item)
                for pas in cutlass.range_constexpr((T + 1) // 2):
                    t = p // 64 + 2 * pas
                    if t < T:
                        self.feed_norm(t, p % 64, lane, part(shared_o, b, TK),
                                       part(gates, b, TK), nw, eps, output, token,
                                       head)  # fmt: skip
                cute.arch.mbarrier_arrive(free + b)

    @cute.jit
    def stage(self, state, page, state_full, meta, live_items, first, n):
        """``KdaVerifyLoop.stage`` with head-major items."""
        M: cutlass.Constexpr = self.most
        if n < live_items[M]:
            index = live_items[n]
            slot, head = meta[index, 0], self.head_of(first + index)
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

    @cute.jit
    def stage_constants(self, args, p, h0, h1, fb, conv_w, conv_b, dts):
        """Feed thread ``p``: 16-byte cp.async of the CTA's head constants (per head
        16 F_B row chunks and 4 conv weight / bias / dt chunks a thread), one
        commit group."""
        weights, bias, dt_bias, fb_weight = args[1], args[2], args[7], args[16]
        dim: cutlass.Constexpr = self.heads * 128
        for hs in cutlass.range(h1 - h0 + 1, unroll=1):
            head = h0 + hs
            for i in cutlass.range_constexpr(16):
                c = p + FEED_THREADS * i
                r, o = c // 16, c % 16
                copy_g2s(at(fb, hs, r, 8 * o), at(fb_weight, head * 128 + r, 8 * o))
            for i in cutlass.range_constexpr(4):
                segment, o = (p + FEED_THREADS * i) // 32, p % 32
                if segment < 12:
                    s, t = segment // 4, segment % 4
                    source = at(weights, s, t, head * 128 + 4 * o)
                    copy_g2s(at(conv_w, hs, s, t, 4 * o), source)
                elif segment < 15:
                    if cutlass.const_expr(self.has_bias):
                        s = segment - 12
                        source = at(bias, s * dim + head * 128 + 4 * o)
                        copy_g2s(at(conv_b, hs, s, 4 * o), source)
                else:
                    copy_g2s(at(dts, hs, 4 * o), at(dt_bias, head * 128 + 4 * o))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def stage_rows(self, args, p, m, lives, first, meta, live_items, raw, raw_full):
        """Feed thread ``p``: 16-byte cp.async of the ``m``-th live item's operand
        rows (if there is one) into raw buffer m % 2, then the thread's arrival on
        ``raw_full[m % 2]`` once its copies landed. Chunks: x rows (3 q x 16), F_A
        rows (q x 16), gate rows (q x 16), the 16-byte beta chunks (q), the conv tap
        blocks (3 x 16 (q + 2))."""
        x, conv_state, f_a, beta, gate = (args[i] for i in (0, 3, 4, 5, 11))
        xs, fas, gts, betas, taps = raw
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = self.tokens + 2
        dim: cutlass.Constexpr = self.heads * 128
        n_x: cutlass.Constexpr = 48 * T
        n_fa: cutlass.Constexpr = n_x + 16 * T
        n_g: cutlass.Constexpr = n_fa + 16 * T
        n_b: cutlass.Constexpr = n_g + T
        # the conv tap blocks are staged from a BF16 window only; an FP32 window's
        # taps are loaded (converted to BF16) per thread in ``feed_prologue``
        staged: cutlass.Constexpr = conv_state.element_type == BFloat16
        total: cutlass.Constexpr = n_b + 48 * W if staged else n_b
        if m < lives:
            index = live_items[m]
            item = first + index
            head, request = self.head_of(item), self.request_of(item)
            token, conv_slot, b = request * T, meta[index, 1], m % 2
            chunk = head // BETA_CHUNK * BETA_CHUNK
            conv_page = conv_state[Int64(conv_slot), None, None]
            rounds: cutlass.Constexpr = (total + FEED_THREADS - 1) // FEED_THREADS
            for i in cutlass.range_constexpr(rounds):
                c = p + FEED_THREADS * i
                if c < n_x:
                    seg, o = c // 16, c % 16
                    t, s = seg // 3, seg % 3
                    source = at(x, token + t, s * dim + head * 128 + 8 * o)
                    copy_g2s(at(xs, b, t, s, 8 * o), source)
                elif c < n_fa:
                    t, o = (c - n_x) // 16, (c - n_x) % 16
                    copy_g2s(at(fas, b, t, 8 * o), at(f_a, token + t, 8 * o))
                elif c < n_g:
                    t, o = (c - n_fa) // 16, (c - n_fa) % 16
                    source = at(gate, token + t, head * 128 + 8 * o)
                    copy_g2s(at(gts, b, t, 8 * o), source)
                elif c < n_b:
                    t = c - n_g
                    copy_g2s(at(betas, b, t, 0), at(beta, token + t, chunk))
                elif c < total:
                    s, o = (c - n_b) // (16 * W), (c - n_b) % (16 * W)
                    source = at(conv_page, s * dim + head * 128, 0) + 8 * o
                    copy_g2s(at(taps, b, s, 8 * o), source)
            copies_arrive(raw_full + b)

    @cute.jit
    def feed(self, args, p, n, lives, first, h0, meta, live_items, raw, raw_full, fb,
             conv_w, conv_b, dts, exp_a, shared_q, shared_k, shared_decay, shared_v,
             shared_beta, gates, reductions, ready):  # fmt: skip
        """Feed thread ``p``: the ``n``-th live item's operands from its raw rows
        into operand buffer n % 2 (LoopStock's convolution prologue for channel p,
        LoopStock's gate warps p // 32 and p // 32 + 4, the output gate rows), the
        arrival on ``ready[n % 2]``, then the rows of live item n + 2 into the freed
        raw buffer. (Operand buffer n % 2 is free: item n - 2's recurrence finished
        before its outputs were normed, which this thread did.)"""
        T: cutlass.Constexpr = self.tokens
        TK: cutlass.Constexpr = 128 * T
        xs, fas, gts, betas, taps = raw
        b = n % 2
        index = live_items[n]
        item = first + index
        head = self.head_of(item)
        hs, offset, conv_slot = head - h0, meta[index, 2], meta[index, 1]
        wait(raw_full + b, (n // 2) % 2, scope="cta")
        self.feed_prologue(args, p, head, hs, b, offset, conv_slot, conv_w, conv_b,
                           xs, betas, taps, part(shared_q, b, TK),
                           part(shared_k, b, TK), part(shared_v, b, TK), reductions,
                           part(shared_beta, b, T))  # fmt: skip
        decay = part(shared_decay, b, TK)
        for half in cutlass.range_constexpr(2):  # LoopStock's gate warps g, g + 4
            self.feed_gate(args, p // 32 + 4 * half, p % 32, hs, b, fb, dts, exp_a,
                           fas, decay)  # fmt: skip
        gates_b = part(gates, b, TK)
        for t in cutlass.range_constexpr(T):
            gates_b[128 * t + p] = gts[b, t, p]
        cute.arch.mbarrier_arrive(ready + b)
        # every feed thread is past its reads of raw buffer b
        named_barrier(FEED_BARRIER, FEED_THREADS)
        self.stage_rows(args, p, n + 2, lives, first, meta, live_items, raw, raw_full)

    @cute.jit
    def feed_prologue(self, args, thread, head, hs, b, offset, conv_slot, conv_w,
                      conv_b, xs, betas, taps_s, shared_q, shared_k, shared_v,
                      reductions, shared_beta):  # fmt: skip
        """``KdaVerifyLoopStock.loop_prologue`` (same operations on the same
        values) from the staged rows, on the feed threads, without its ready
        arrivals and own-gate loads; the norms' partial sums meet at FEED_BARRIER."""
        conv_state = args[3]
        scale = args[14]
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = self.tokens + 2
        dim: cutlass.Constexpr = self.heads * 128
        lane, warp = thread % 32, thread // 32
        channel = head * 128 + thread
        conv_page = conv_state[Int64(conv_slot), None, None]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), Float32)
        accum = zeros(3 * T, Float32)
        for s in cutlass.range_constexpr(3):
            for t in cutlass.range_constexpr(3):
                if cutlass.const_expr(conv_state.element_type == BFloat16):
                    taps[s, t] = taps_s[b, s, thread * W + offset + t]
                else:  # FP32 window: not staged (``stage_rows``), lossless to BF16
                    tap = conv_tap(conv_page, s * dim + channel, offset + t)
                    taps[s, t] = tap
            for t in cutlass.range_constexpr(4):
                coeff[s, t] = Float32(conv_w[hs, s, t, thread])
            if cutlass.const_expr(self.has_bias):
                for t in cutlass.range_constexpr(T):
                    accum[3 * t + s] = Float32(conv_b[hs, s, thread])
        inputs = cute.make_rmem_tensor(3 * T, BFloat16)
        for t in cutlass.range_constexpr(T):
            for s in cutlass.range_constexpr(3):
                inputs[3 * t + s] = xs[b, t, s, thread]
        if thread < T:
            value = Float32(betas[b, thread, head % BETA_CHUNK])
            shared_beta[thread] = triton_sigmoid(value)
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
            named_barrier(FEED_BARRIER, FEED_THREADS)
            for jj, t in enumerate(tokens):
                r = [reductions[8 * t + i] for i in range(8)]
                qn = (r[0] + r[2]) + (r[1] + r[3]) + 1.0e-6
                kn = (r[4] + r[6]) + (r[5] + r[7]) + 1.0e-6
                fq = reciprocal(sqrt_approx(qn))
                fk = reciprocal(sqrt_approx(kn))
                shared_q[128 * t + thread] = mul_rn(mul_rn(qks[2 * jj], fq), scale)
                shared_k[128 * t + thread] = mul_rn(qks[2 * jj + 1], fk)
            # (the next pair's partial sums overwrite `reductions`)
            named_barrier(FEED_BARRIER, FEED_THREADS)
        for s in cutlass.range_constexpr(3):  # the conv window
            window = (taps[s, 1], taps[s, 2], *(inputs[3 * t + s] for t in range(T)))
            for i, value in enumerate(window):
                conv_put(conv_page, (s * dim + channel, i), value)

    @cute.jit
    def feed_gate(self, args, g, lane, hs, b, fb, dts, exp_a, fas, shared_decay):
        """``KdaVerifyLoopStock.loop_gate`` (same operations on the same values)
        from the staged rows: LoopStock's gate warp ``g``, the F_B tile of features
        [16 g, 16 g + 16), one chain of eight MMAs, BF16 raw gates, the Triton
        decays."""
        lower_bound = args[13]
        T: cutlass.Constexpr = self.tokens
        group, quarter = lane >> 2, lane & 3
        base = 16 * g
        features = [group, group + 8]
        dt = [Float32(dts[hs, base + f]) for f in features]
        wr = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
        row = base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = staged_words(at(fb, hs, row, col))
            high = staged_words(at(fb, hs, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = exp_a[hs]
        fa_row = group % T  # the lane's token column
        ar = [staged_words(at(fas, b, fa_row, 16 * c + 4 * quarter)) for c in range(8)]
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
    def feed_norm(self, t, c, glane, shared_o, gates, nw, eps, output, token, head):
        """``KdaVerifyLoopStock.gated_norm`` (same operations) with the output gates
        from the feed's gate rows."""
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
            activated = triton_sigmoid(Float32(gates[128 * t + c + 64 * h]))
            result = raw_o[h] * rstd * nw[h] * activated
            output[token + t, head * 128 + c + 64 * h] = BFloat16(result)

    @cute.jit
    def chain(self, history, c, x4):
        """``row_dots``' chunk ``c`` FMA chain of the row pair (packed)."""
        acc = mul2((history[0, c, 0], history[1, c, 0]), (x4[0], x4[0]))
        for e in cutlass.range_constexpr(1, 4):
            pair = (history[0, c, e], history[1, c, e])
            acc = fma2(pair, (x4[e], x4[e]), acc)
        return acc

    @cute.jit
    def pair_token(self, histories, t, group, j, row0, shared_q, shared_k,
                   shared_decay, shared_v, shared_beta, shared_o, state, meta,
                   item_index, head):  # fmt: skip
        """``KdaVerifyLoop.loop_token`` for both row sets: the previous token's
        query dots (before the decay), the decay, this token's key dots, the
        xor 4 / 2 / 1 reduction, the delta-rule update, the previous token's
        outputs and this token's checkpoint, per set exactly LoopStock's; the column
        chunks stream in the row-dot tree's order (c0 + c2) + (c1 + c3)."""
        # [set][kind q / k][half (c0 + c2) / (c1 + c3)][row]
        partial = [[[None, None] for _ in range(2)] for _ in range(SETS)]
        for c in (0, 2, 1, 3):
            half = c % 2
            k4 = load_f32x4(shared_k.iterator + 128 * t + 4 * (j + 8 * c), shared=True)
            d4 = load_f32x4(
                shared_decay.iterator + 128 * t + 4 * (j + 8 * c), shared=True
            )
            q4 = None
            if cutlass.const_expr(t > 0):
                q4 = load_f32x4(
                    shared_q.iterator + 128 * (t - 1) + 4 * (j + 8 * c), shared=True
                )
            for s in cutlass.range_constexpr(SETS):
                history = histories[s]
                chains = [None, None]
                if cutlass.const_expr(t > 0):
                    chains[0] = self.chain(history, c, q4)
                for e in cutlass.range_constexpr(4):
                    pair = mul2((history[0, c, e], history[1, c, e]), (d4[e], d4[e]))
                    history[0, c, e], history[1, c, e] = pair
                chains[1] = self.chain(history, c, k4)
                for kind in cutlass.range_constexpr(2):
                    if cutlass.const_expr(chains[kind] is not None):
                        low = partial[s][kind][half]
                        if cutlass.const_expr(low is None):  # c0 / c1
                            partial[s][kind][half] = list(chains[kind])
                        else:  # c0 + c2, c1 + c3
                            partial[s][kind][half] = [
                                low[r] + chains[kind][r] for r in range(2)
                            ]
        beta = shared_beta[t]
        v_new = []
        for s in cutlass.range_constexpr(SETS):
            values = [Float32(0), Float32(0)]
            if cutlass.const_expr(t > 0):
                q = partial[s][0]
                values = [q[0][r] + q[1][r] for r in range(2)]
            k = partial[s][1]
            values += [k[0][r] + k[1][r] for r in range(2)]
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
            rows = [row0 + 64 * s + 4 * r for r in range(2)]
            v_new.append(
                [(shared_v[128 * t + rows[r]] - dot_k[r]) * beta for r in range(2)]
            )
            if cutlass.const_expr(t > 0):  # noqa: SIM102
                if (j & 5) == 0:
                    row = row0 + 64 * s + 4 * (j // 2)
                    position = (t - 1) * 128 + self.output_position(row)
                    shared_o[position] = self.round_out(owned)
        for c in cutlass.range_constexpr(CHUNKS):
            k4 = load_f32x4(shared_k.iterator + 128 * t + 4 * (j + 8 * c), shared=True)
            for s in cutlass.range_constexpr(SETS):
                history = histories[s]
                for e in cutlass.range_constexpr(4):
                    pair = (history[0, c, e], history[1, c, e])
                    pair = fma2((k4[e], k4[e]), (v_new[s][0], v_new[s][1]), pair)
                    history[0, c, e], history[1, c, e] = pair
        slot = meta[item_index, 4 + t]
        page = state[Int64(slot), None, None, None]
        for s in cutlass.range_constexpr(SETS):
            for r in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(CHUNKS):
                    values4 = [histories[s][r, c, e] for e in range(4)]
                    pointer = at(page, head, row0 + 64 * s + 4 * r, 4 * (j + 8 * c))
                    self.store_state(pointer, values4, slot > 0)
