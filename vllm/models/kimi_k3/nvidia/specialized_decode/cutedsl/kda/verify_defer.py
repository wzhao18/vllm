# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyHalfStock`` under the deferred-verify state protocol (no per-token
checkpoint stores).

Protocol (per request, per layer; slots = the request's q state slots, accepted =
the previous verify's accepted count a, 1 <= a <= q):

- slot 0 holds the state after token 0 of the previous verify (token 0 is always
  accepted; after a prefill, the prefill's state with a = 1, as today);
- slot 1's page holds the previous verify's records of tokens 1 .. q - 1, per head:
  rows ``t - 1`` the key k_t (FP32, as the recurrence used it), rows
  ``DECAY_ROW + t - 1`` the decay exp(g_t) (FP32), rows ``V_ROW + t - 1`` the
  correction v_new_t = (v_t - k_t . S) beta_t (FP32, per state row);
- before its PDL wait the verify folds the a - 1 accepted tokens 1 .. a - 1 into the
  slot-0 state: S = S * decay_t, then S = fma(k_t, v_new_t, S) per element, the
  token loop's own two instructions on the same FP32 values, so the folded state is
  bit for bit the checkpoint a - 1 the stock protocol would have stored (and read);
- the token loop then runs as today, stores the state after token 0 to slot 0 (one
  page instead of q) and the records of tokens 1 .. q - 1 to slot 1 (its rows'
  corrections in the loop; k and decay, shared by both halves, after the halves'
  exchange, once both have folded).

Slots 2 .. q - 1 are no longer written. Outputs and the conv window are bit for bit
``KdaVerifyHalfStock``'s. The records and accepted counts of a layer are written /
host-ordered before the launch (like the conv taps and slots ``KdaSplit`` loads
early), so the fold needs no PDL wait."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma2,
    load_f32x4,
    mul2,
    wait,
    zeros,
)
from .verify_half import (
    COPY_BYTES,
    GROUP,
    HALF_PAGE,
    HALF_ROWS,
    HALF_THREADS,
    MESSENGER,
    KdaVerifyHalfStock,
)
from .verify_loop import K
from .verify_packed import CHUNKS
from .window import conv_put

DECAY_ROW = 8  # record page rows: k 0 .. q - 2, decay 8 .., corrections 16 ..
V_ROW = 16
RECORD_ROWS = V_ROW + 7  # rows staged with the state page (q <= 8)
RECORD_BYTES = RECORD_ROWS * 128 * 4


class KdaVerifyHalfDefer(KdaVerifyHalfStock):
    """``KdaVerifyHalfStock`` under the deferred-verify protocol (module docstring);
    same operands (q >= 2)."""

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
        rec = alloc(smem, Float32, RECORD_ROWS * K, align=128)  # the records' rows
        meta = alloc(smem, Int32, (1, 4 + T))  # LoopStock's per-item metadata
        pairs: cutlass.Constexpr = (T + 1) // 2
        bars = alloc(smem, cutlass.Int64, pairs + 1, 8).iterator
        ready, state_full = bars, bars + pairs

        taken = accepted[request]
        offset = taken - 1
        if taken < 1:  # a padding row's count may be 0 (its slots are <= 0)
            offset = Int32(0)
        # deferred protocol: the state at slot 0 (the previous verify's token 0),
        # ``offset`` pending tokens (1 .. offset) in the records at slot 1
        slot, conv_slot = slots[request, 0], slots[request, 0]
        records = slots[request, 1]
        if thread < T:
            meta[0, 4 + thread] = slots[request, thread]
        if thread == 0:
            for p in cutlass.range_constexpr(pairs):
                # group 0's or group 1's 128 prologue threads + the 4 gate warps
                cute.arch.mbarrier_init(ready + p, GROUP + 128)
            cute.arch.mbarrier_init(state_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if (thread == 0) & (slot > 0):  # (+ the records' rows)
            source = state[Int64(slot), None, None, None]
            staged = HALF_PAGE + RECORD_BYTES * Int32(records > 0)
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, staged)
            if records > 0:
                record_source = state[Int64(records), None, None, None]
                copy_g2s(rec.iterator, at(record_source, head, 0, 0), RECORD_BYTES,
                         state_full)  # fmt: skip
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
                    (offset, conv_slot, slot, records),
                    (page, state_full, ready, meta, rec),
                    (shared_q, shared_k, shared_decay, shared_v, shared_o, reductions,
                     shared_beta),
                    window,
                )  # fmt: skip
            cute.arch.sync_threads()  # this half's outputs of every token in shared_o
            if warp == MESSENGER:
                self.exchange(scratch, counters, item, half, lane, shared_o)
            cute.arch.sync_threads()  # the other half's rows merged; both past the taps
            if (thread < GROUP) & (half == 0) & (records > 0):  # k / decay records
                record_page = state[Int64(records), None, None, None]
                for t in cutlass.range_constexpr(1, T):
                    record_page[head, t - 1, thread] = shared_k[128 * t + thread]
                    record_page[head, DECAY_ROW + t - 1, thread] = shared_decay[
                        128 * t + thread
                    ]
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
        offset, conv_slot, slot, records = rows
        page, state_full, ready, meta, rec = staging
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
        # (no records slot: nothing to fold; never at q >= 2)
        self.fold(history, rec, row0, j, offset * Int32(records > 0))
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
    def fold(self, history, rec, row0, j, pending):
        """The ``pending`` accepted tokens 1 .. pending of the previous verify into the
        thread's rows from the staged records ``rec`` (shared memory), with
        ``loop_token``'s decay and update instructions in the same order."""
        for i in cutlass.range(pending, unroll=1):
            k = self.column_chunks(rec, i, j)
            decay = self.column_chunks(rec, DECAY_ROW + i, j)
            v_new = [rec[(V_ROW + i) * K + row0 + 4 * r] for r in range(2)]
            for c in cutlass.range_constexpr(CHUNKS):
                for e in cutlass.range_constexpr(4):
                    x = decay[c, e]
                    pair = mul2((history[0, c, e], history[1, c, e]), (x, x))
                    history[0, c, e], history[1, c, e] = pair
            for c in cutlass.range_constexpr(CHUNKS):
                for e in cutlass.range_constexpr(4):
                    x = k[c, e]
                    pair = (history[0, c, e], history[1, c, e])
                    pair = fma2((x, x), (v_new[0], v_new[1]), pair)
                    history[0, c, e], history[1, c, e] = pair

    @cute.jit
    def loop_token(self, history, t, group, j, row0, shared_q, shared_k,
                   shared_decay, shared_v, shared_beta, shared_o, state, meta,
                   item_index, head):  # fmt: skip
        """``KdaVerifyLoop.loop_token`` (same math, same order) storing the state after
        token 0 only and, for tokens 1 .., the record of the rows' corrections."""
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
        if cutlass.const_expr(t == 0):  # the one state store: slot 0, after token 0
            slot = meta[item_index, 4]
            page = state[Int64(slot), None, None, None]
            for r in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(CHUNKS):
                    values4 = [history[r, c, e] for e in range(4)]
                    pointer = at(page, head, row0 + 4 * r, 4 * (j + 8 * c))
                    self.store_state(pointer, values4, slot > 0)
        else:  # token t's record: the rows' corrections (k / decay after the exchange)
            records = meta[item_index, 5]
            if (j == 0) & (records > 0):
                record_page = state[Int64(records), None, None, None]
                for r in cutlass.range_constexpr(2):
                    record_page[head, V_ROW + t - 1, row0 + 4 * r] = v_new[r]
