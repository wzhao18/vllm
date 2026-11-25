# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyLoopStock``'s item on two CTAs: each (request, head) item's 128 state
rows are split into two halves of 64 rows (value rows evolve independently), one
8-warp CTA per half, grid 2 x items.

Why: at one item per CTA (q8 B4-B17 at TP16: 24-102 CTAs on 212 SMs) LoopStock's
time per layer is flat in the item count (VR200 1-GPU verify alone, 69 layers, own
cold FP32 slots, us per layer: B4 11.89, B8 12.77, B16 13.20): each item is bound by
its SM. Per-CTA phase stamps (B16): PDL wait -> pair 0 ready 1.7 us, the 16 warps'
token loops ~7.3 us (0.95 us a token: instruction issue of 16 warps and the SM's
checkpoint store stream, 512 KiB per item at the ~71.5 GB/s per-SM store ceiling),
gated norm + exit 0.7 us; warps 0-3 run all four operand pairs' prologue before their
own loop (+2.7 us). Two CTAs per item halve each SM's loop instructions and
checkpoint bytes, and the two warp groups split the prologue's pairs: group 0 (warps
0-3) the convolution / SiLU / norms of tokens 0-3, group 1 (warps 4-7) the F_B gate
tiles (two per warp) and then tokens 4-7.

The gated RMS norm needs all 128 outputs of a token: a ninth warp that issues no
checkpoint stores (so its fence waits for its own 4 KiB, not for the SM's checkpoint
drain) writes this half's outputs to the item's scratch row, arrives on the item's
counter (release) and waits for the other half (acquire), then merges the other
half's rows; each half then runs LoopStock's gated norm for half of the tokens. The
halves wait for each other, so the grid must be resident at once: at most one CTA
per SM (``HALF_SMEM``) and 2 x items <= SMs (asserted; the policy rows fit).

Math: per state row exactly LoopStock's (same thread layout: lanes j of row group g,
two rows per thread, the same packed token loop, prologue, gates, betas and gated
norm instructions in the same order), so outputs, every checkpoint and the conv
window are bit for bit ``KdaVerifyLoopStock``'s."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    launch,
    load_f32x4,
    named_barrier,
    wait,
    warp_sum,
    zeros,
)
from .recover import mul_rn, reciprocal, sqrt_approx, triton_sigmoid
from .verify_loop import K, KdaVerifyLoopStock, persistent_ctas
from .verify_packed import CHUNKS
from .window import conv_put, conv_tap

HALF_THREADS = 256  # 8 warps: rows 64 h + 8 w + g (+ 4), as LoopStock's warps w
MESSENGER = 8  # warp 8: the output exchange (it issues no checkpoint stores)
ALL_THREADS = HALF_THREADS + 32
HALF_ROWS = 64
HALF_PAGE = HALF_ROWS * K * 4  # 32 KiB of the item's FP32 page
COPY_BYTES = 16384
GROUP = 128  # threads of a prologue warp group (thread = channel)
# Dynamic shared memory requested per CTA (the kernel uses ~53 KB at q8): more than
# half of an SM's, so no SM holds two of its CTAs. Launched early (PDL) while the
# in-projection drains, two halves per freed SM would share it (1-GPU chain at q8
# B16: 192 CTAs on 97 SMs, +1.2-1.5 us per layer vs LoopStock).
HALF_SMEM = 120 * 1024


@asm(Uint32)
def _arrive_release(address):
    """release atomic add 1 on the item's arrival counter; the old value."""
    code = "atom.release.gpu.global.add.u32 $0,[$1],1;"
    return code, "=r,l,~{memory}", (Int64(address),)


@asm(Uint32)
def _acquire_u32(address):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Float32)
def _load_f32_gpu(address):
    return "ld.relaxed.gpu.global.f32 $0,[$1];", "=f,l,~{memory}", (Int64(address),)


@asm()
def _fence():
    return "fence.acq_rel.gpu;", "~{memory}", ()


class KdaVerifyHalfStock(KdaVerifyLoopStock):
    """``KdaVerifyLoopStock`` with each item on two CTAs (grid 2 x items, 256
    threads); extra operands after ``accepted``: the layer's FP32 scratch
    [>= items, q x 128] and Int32 counters [>= items] (zeros; re-armed in kernel)."""

    @cute.jit
    def __call__(self, *args):
        assert args[0].shape[0] // self.tokens * self.heads == self.items
        ctas = 2 * self.items
        # the halves wait for each other: every CTA must be resident (one per SM)
        assert persistent_ctas(ctas) == ctas, f"KdaVerifyHalfStock: {ctas} CTAs"
        # <= 128 registers a thread (another kernel's CTA can share the SM), and
        # HALF_SMEM: at most one of these CTAs per SM
        launch(
            self.kernel, args, ctas, ALL_THREADS, smem=HALF_SMEM,
            max_number_threads=[ALL_THREADS, 1, 1], min_blocks_per_mp=1,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted, scratch, counters = args[15], args[17], args[18], args[19]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        item, half = cta // 2, cta % 2
        head, request = item % H, item // H
        token = request * T
        lane, warp = thread % 32, thread // 32
        group, j = lane // 8, lane % 8

        smem = cutlass.utils.SmemAllocator()
        page = alloc(smem, Float32, HALF_ROWS * K, align=1024)
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, 128 * T) for _ in range(5)
        )
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        meta = alloc(smem, Int32, (1, 4 + T))  # LoopStock's per-item metadata
        pairs: cutlass.Constexpr = (T + 1) // 2
        bars = alloc(smem, cutlass.Int64, pairs + 1, 8).iterator
        ready, state_full = bars, bars + pairs

        taken = accepted[request]
        offset = taken - 1
        if taken < 1:  # a padding row's count may be 0 (its slots are <= 0)
            offset = Int32(0)
        slot, conv_slot = slots[request, offset], slots[request, 0]
        if thread < T:
            meta[0, 4 + thread] = slots[request, thread]
        if thread == 0:
            for p in cutlass.range_constexpr(pairs):
                # group 0's or group 1's 128 prologue threads + the 4 gate warps
                cute.arch.mbarrier_init(ready + p, GROUP + 128)
            cute.arch.mbarrier_init(state_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if (thread == 0) & (slot > 0):  # this half's 64 rows of the accepted page
            source = state[Int64(slot), None, None, None]
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, HALF_PAGE)
            rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
            for i in cutlass.range_constexpr(HALF_PAGE // COPY_BYTES):
                copy_g2s(
                    page.iterator + i * rows * K,
                    at(source, head, HALF_ROWS * half + i * rows, 0),
                    COPY_BYTES,
                    state_full,
                )

        if slot <= 0:  # a padding request (half 0 writes its zero outputs)
            cute.arch.griddepcontrol_wait()
            if (half == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
        else:
            # group 0's conv window values (taps 1-2, then the inputs), written only
            # after the exchange: both halves read the window's taps in their prologues
            window = (zeros((3, 3), BFloat16), zeros(3 * T, BFloat16))
            if thread < HALF_THREADS:
                self.half_math(
                    args, (thread, lane, warp, group, j), (item, half, head, token),
                    (offset, conv_slot, slot), (page, state_full, ready, meta),
                    (shared_q, shared_k, shared_decay, shared_v, shared_o, reductions,
                     shared_beta),
                    window,
                )  # fmt: skip
            cute.arch.sync_threads()  # this half's outputs of every token in shared_o
            if warp == MESSENGER:
                self.exchange(scratch, counters, item, half, lane, shared_o)
            cute.arch.sync_threads()  # the other half's rows merged; both past the taps
            if (thread < GROUP) & (half == 0):  # the conv window (one writer)
                conv_state = args[3]
                conv_page = conv_state[Int64(conv_slot), None, None]
                dim: cutlass.Constexpr = H * 128
                taps, inputs = window
                for s in cutlass.range_constexpr(3):
                    values = (taps[s, 1], taps[s, 2], *(inputs[3 * t + s] for t in range(T)))
                    for i, value in enumerate(values):
                        conv_put(conv_page, (s * dim + head * 128 + thread, i), value)
            if thread < HALF_THREADS:  # this half's tokens: the gated RMS norm
                first: cutlass.Constexpr = (T + 1) // 2
                nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
                c, glane = thread % 64, thread % 32
                t = first * half + thread // 64
                if (thread // 64 < first) & (t < T):
                    gates = zeros(2, gate.element_type)
                    for h in cutlass.range_constexpr(2):
                        gates[h] = gate[token + t, head * 128 + c + 64 * h]
                    self.gated_norm(t, c, glane, shared_o, gates, nw, eps, output,
                                    token, head)  # fmt: skip
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def half_math(self, args, ids, where, rows, staging, shared, window):
        """Warps 0-7: the half's state rows, the prologue roles, the token loop and the
        last token's query dots (this half's rows of ``shared_o``)."""
        thread, lane, warp, group, j = ids
        item, half, head, token = where
        offset, conv_slot, slot = rows
        page, state_full, ready, meta = staging
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = shared[:6]
        shared_beta = shared[6]
        state, gate = args[9], args[11]
        T: cutlass.Constexpr = self.tokens
        pairs: cutlass.Constexpr = (T + 1) // 2
        local0 = 8 * warp + group  # this thread's rows in the staged half page
        row0 = HALF_ROWS * half + local0
        wait(state_full, 0, scope="cta")
        history = zeros((2, CHUNKS, 4), Float32)
        for r in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(CHUNKS):
                pointer = page.iterator + (local0 + 4 * r) * K + 4 * (j + 8 * c)
                values = load_f32x4(pointer, shared=True)
                for e in cutlass.range_constexpr(4):
                    history[r, c, e] = values[e]
        own_gates = zeros(2, gate.element_type)
        shared = (shared_q, shared_k, shared_v, reductions, shared_beta)
        if thread < GROUP:  # group 0: tokens 0 .. 2 (pairs // 2) - 1 (+ window)
            self.half_prologue(
                args, thread, token, head, offset, conv_slot, shared, ready,
                (0, pairs // 2), 1, window,
            )  # fmt: skip
        else:  # group 1: the F_B gate tiles (two a warp), then the other pairs
            for i in cutlass.range_constexpr(2):
                self.loop_gate(args, 2 * (warp - 4) + i, lane, head, token,
                               shared_decay, own_gates)  # fmt: skip
            for p in cutlass.range_constexpr(pairs):
                cute.arch.mbarrier_arrive(ready + p)
            self.half_prologue(
                args, thread - GROUP, token, head, offset, conv_slot, shared,
                ready, (pairs // 2, pairs), 2, None,
            )  # fmt: skip
        cute.arch.griddepcontrol_launch_dependents()

        for t in cutlass.range_constexpr(T):
            if cutlass.const_expr(t % 2 == 0):  # token pair t // 2's operands
                cute.arch.mbarrier_wait(ready + t // 2, 0)
            self.loop_token(history, t, group, j, row0, shared_q, shared_k,
                            shared_decay, shared_v, shared_beta, shared_o,
                            state, meta, 0, head)  # fmt: skip
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

    @cute.jit
    def exchange(self, scratch, counters, item, half, lane, shared_o):
        """Warp MESSENGER: this half's ``shared_o`` to its scratch region, arrive on
        the item's counter (release), wait for the other half's arrival (acquire),
        and merge the other half's rows into ``shared_o``. The counter only grows (two
        arrivals per launch): this launch's pair completes at the next even value.
        The warp stored nothing before, so its fence waits for its 4 KiB only, not
        for the math warps' checkpoint stores."""
        T: cutlass.Constexpr = self.tokens
        size: cutlass.Constexpr = T * 128
        for i in cutlass.range_constexpr(size // 32):
            index = lane + 32 * i
            scratch[item, half * size + index] = shared_o[index]
        _fence()
        cute.arch.sync_warp()
        counter = (counters.iterator + item).toint()
        if lane == 0:
            old = _arrive_release(counter)
            target = (old & Uint32(0xFFFFFFFE)) + Uint32(2)
            seen = _acquire_u32(counter)
            while Int32(seen - target) < 0:
                seen = _acquire_u32(counter)
        cute.arch.sync_warp()
        _fence()
        other = scratch.iterator + Int64(item) * (2 * size) + (1 - half) * size
        for i in cutlass.range_constexpr(-(-T * HALF_ROWS // 32)):
            index = lane + 32 * i
            if index < T * HALF_ROWS:
                t, row = index // HALF_ROWS, HALF_ROWS * (1 - half) + index % HALF_ROWS
                position = t * 128 + self.output_position(row)
                shared_o[position] = _load_f32_gpu((other + position).toint())

    @cute.jit
    def half_prologue(self, args, thread, token, head, offset, conv_slot, shared,
                      ready, span, barrier_id, window):  # fmt: skip
        """``KdaVerifyLoopStock.loop_prologue`` for channel ``thread`` (0..127) and
        token pairs [span[0], span[1]) on named barrier ``barrier_id`` (each warp
        group its own); the betas with the first pair; ``window``: (taps, inputs)
        register tensors that receive the conv window's taps and inputs (the kernel
        writes the window after the halves' exchange)."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        scale = args[14]
        shared_q, shared_k, shared_v, reductions, shared_beta = shared
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
                taps[s, t] = conv_tap(conv_page, s * dim + channel, offset + t)
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
        if cutlass.const_expr(span[0] == 0):
            if thread < T:
                shared_beta[thread] = triton_sigmoid(Float32(beta[token + thread, head]))
        for p in cutlass.range_constexpr(span[0], span[1]):
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
            named_barrier(barrier_id, GROUP)
            for jj, t in enumerate(tokens):
                r = [reductions[8 * t + i] for i in range(8)]
                qn = (r[0] + r[2]) + (r[1] + r[3]) + 1.0e-6
                kn = (r[4] + r[6]) + (r[5] + r[7]) + 1.0e-6
                fq = reciprocal(sqrt_approx(qn))
                fk = reciprocal(sqrt_approx(kn))
                shared_q[128 * t + thread] = mul_rn(mul_rn(qks[2 * jj], fq), scale)
                shared_k[128 * t + thread] = mul_rn(qks[2 * jj + 1], fk)
            cute.arch.mbarrier_arrive(ready + p)
        if cutlass.const_expr(window is not None):
            keep_taps, keep_inputs = window
            for s in cutlass.range_constexpr(3):
                for t in cutlass.range_constexpr(3):
                    keep_taps[s, t] = taps[s, t]
            for i in cutlass.range_constexpr(3 * T):
                keep_inputs[i] = inputs[i]
