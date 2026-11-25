# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused verify heads (``KdaHeadVerify`` B1, ``KdaHeadMulti`` B2) under the
deferred-verify state protocol (``verify_defer``'s module docstring).

Each KDA CTA reads its rows of the state at slot 0 and folds the previous verify's
pending accepted tokens 1 .. accepted - 1 into them before the cluster sync / PDL
wait, from the records at slot 1 (k and decay per column, the corrections per row,
FP32) with ``rolled_update``'s decay and update instructions (so the state is bit for
bit the checkpoint the stock protocol stores). The token loop stores the state after
token 0 to slot 0 (one store instead of q) and token t's corrections to slot 1's
record rows; the CTA owning token 0 of the request writes the k / decay rows once its
token's outputs arrived from every KDA CTA of the request (each pushes after its
fold), so no CTA still reads them. Everything else is the stock heads' code.

``set_head_protocol(defer)`` selects the kernels ``compile_head_verify`` /
``compile_head_multi`` build (and clears their caches)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    at,
    fma,
    load_f32x4,
    named_barrier,
    push_float,
    wait,
    warp_sum,
    zeros,
)
from . import head_multi, head_verify
from .head_multi import KdaHeadMulti
from .head import store_bf16_canonical
from .head_verify import KDA_CTAS, ROWS, KdaHeadVerify
from .kernel import sigmoids
from .verify_defer import DECAY_ROW, V_ROW
from .window import conv_put, conv_tap


class HeadRecords:
    """The record forms' shared pieces (``KdaHeadVerify``'s geometry: ``vkda``)."""

    @cute.jit
    def rolled_update_defer(self, history, ops, other, sources, t, group_lane, v_row, beta):
        """``rolled_update`` also returning v_new. ``KdaSplit.update`` of pass 0 for token t from register operands (``ops``: q,
        k, decays), software-pipelined: the previous token's query dot (``other``'s q
        against the state before this token's decay, i.e. after the previous token's
        update) is reduced across the row's lanes together with this token's key dot;
        ``other`` then receives token t + 1's operands. Returns the previous token's
        output; ``query_dot`` gives the last token's."""
        vk = self.vkda
        kr = ops[1, None, None]
        dot_q = vk.row_dot(history, other[0, None, None], 0)
        if t + 1 < self.tokens:
            self.load_operands(other, sources, t + 1, group_lane)
        for j in cutlass.range_constexpr(vk.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] *= ops[2, j, e]
        dot_k = vk.row_dot(history, kr, 0)
        for shift in reversed(range(vk.lanes_per_row.bit_length() - 1)):
            dot_k += cute.arch.shuffle_sync_bfly(dot_k, 1 << shift)
            dot_q += cute.arch.shuffle_sync_bfly(dot_q, 1 << shift)
        v_new = (v_row - dot_k) * beta
        for j in cutlass.range_constexpr(vk.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = fma(
                    kr[j, e], v_new, history[0, j, e], ftz=vk.fast_math
                )
        return dot_q, v_new


    @cute.jit
    def key_records(self, record_page, head, kt, shared_k, shared_decay):
        """Column ``kt``'s k / decay records of tokens 1 .. q - 1."""
        for u in cutlass.range_constexpr(1, self.tokens):
            record_page[head, u - 1, kt] = shared_k[128 * u + kt]
            record_page[head, DECAY_ROW + u - 1, kt] = shared_decay[128 * u + kt]

    @cute.jit
    def head_fold(self, history, record_page, head, pending, value_base, group,
                  group_lane):  # fmt: skip
        """The ``pending`` accepted tokens 1 .. pending into the thread's row (pass 0)
        from the records in global memory (before the PDL wait), ``rolled_update``'s
        decay and update instructions; the next token's records load while this one
        folds."""
        vk = self.vkda
        chunks: cutlass.Constexpr = vk.chunks
        ops = cute.make_rmem_tensor((2, 2, chunks, 4), Float32)  # [buffer, k | decay]
        vs = cute.make_rmem_tensor(2, Float32)
        row = value_base + group % ROWS
        for i in cutlass.range_constexpr(2):  # tokens 1 and 2 (clamped: valid rows)
            self.fold_load(ops, vs, i, i, record_page, head, row, group_lane)
        for pair in cutlass.range((pending + 1) // 2, unroll=1):
            for b in cutlass.range_constexpr(2):  # (constant buffer indices)
                i = 2 * pair + b
                if i < pending:
                    for j in cutlass.range_constexpr(chunks):
                        for e in cutlass.range_constexpr(4):
                            history[0, j, e] *= ops[b, 1, j, e]
                    v_new = vs[b]
                    for j in cutlass.range_constexpr(chunks):
                        for e in cutlass.range_constexpr(4):
                            history[0, j, e] = fma(
                                ops[b, 0, j, e], v_new, history[0, j, e],
                                ftz=vk.fast_math,
                            )  # fmt: skip
                    following = cutlass.select_(i + 2 < 6, i + 2, Int32(6))
                    self.fold_load(ops, vs, b, following, record_page, head, row,
                                   group_lane)  # fmt: skip

    @cute.jit
    def fold_load(self, ops, vs, b, i, record_page, head, row, group_lane):
        vk = self.vkda
        for j in cutlass.range_constexpr(vk.chunks):
            column = 4 * (group_lane + vk.lanes_per_row * j)
            k4 = load_f32x4(at(record_page, head, i, column))
            d4 = load_f32x4(at(record_page, head, DECAY_ROW + i, column))
            for e in cutlass.range_constexpr(4):
                ops[b, 0, j, e] = k4[e]
                ops[b, 1, j, e] = d4[e]
        vs[b] = record_page[head, V_ROW + i, row]


class KdaHeadVerifyDefer(HeadRecords, KdaHeadVerify):
    """``KdaHeadVerify`` under the deferred protocol (module docstring)."""

    @cute.jit
    def verify_early(self, kt, tensors, coords, slots):
        """Loads before the cluster sync (KDA CTAs): state rows from the accepted slot,
        conv taps and weights of channel ``kt`` (and V row 16 r + kt), the norm
        weight."""
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, _, offset = coords
        slot, records = slots[0, 0], slots[0, 1]  # deferred: slot 0 + records
        pending = offset
        if (offset < 0) | (records <= 0):
            pending = Int32(0)
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base = rank * ROWS
        safe = slot
        if (safe < 0) | (rank >= KDA_CTAS):
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        history = zeros((self.vkda.passes, self.vkda.chunks, 4), Float32)
        for j in cutlass.range_constexpr(self.vkda.chunks):  # (non-KDA CTAs: slot 0)
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            pointer = at(state_page, head, value_base + group, column)
            values = self.vkda.load_state(pointer)
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = values[e]
        if (rank < KDA_CTAS) & (slot > 0) & (pending > 0):  # the fold (pre-wait)
            record_page = state[Int64(records), None, None, None]
            self.head_fold(history, record_page, head, pending, value_base, group,
                           group_lane)  # fmt: skip
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3 * self.tokens, Float32)
        if (rank < KDA_CTAS) & (slot > 0):
            # The conv window (written by this layer's previous step) and weights of
            # channel kt (and V row 16 r + kt), loaded before the cluster sync so the
            # convolution does not wait for them after hop 2.
            dim = self.dim
            qk_channel = head * 128 + kt
            value_channel = head * 128 + value_base + kt
            conv_page = conv_state[Int64(slots[0, 0]), None, None]
            for s in cutlass.range_constexpr(3):
                channel = value_channel if s == 2 else qk_channel
                if (s != 2) | (kt < ROWS):
                    for t in cutlass.range_constexpr(3):
                        taps[s, t] = conv_tap(conv_page, s * dim + channel, offset + t)
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = conv_w[s, t, channel]
                    if cutlass.const_expr(self.has_bias):
                        accum[s] = bias[s * dim + channel]
                        for t in cutlass.range_constexpr(1, self.tokens):
                            accum[3 * t + s] = accum[s]
        return history, taps, coeff, accum, norm_w[kt]


    @cute.jit
    def verify_main(
        self, kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
    ):
        """``KdaSplit``'s verify main threads (B1 schedule) on the hop-2 inputs."""
        T: cutlass.Constexpr = self.tokens
        dim = self.dim
        history, taps, coeff, accum, nw = early
        raw_qk, raw_v, raw_g, bar_hop = raws
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows[:6]
        local_o, slot_rows = rows[6:]
        bar_gate, bar_out, shared_beta, bar_qk = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base, qk_channel = rank * ROWS, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        partition = rank
        conv_slot = slots[0, 0]
        records = slots[0, 1]
        live = records > 0
        record_page = state[Int64(cutlass.select_(records > 0, records, Int32(0))), None, None, None]
        if slot <= 0:
            output[partition, qk_channel] = BFloat16(0)
        else:
            conv_page = conv_state[Int64(conv_slot), None, None]
            wait(bar_hop)
            inputs = cute.make_rmem_tensor(3 * T, BFloat16)
            for t in cutlass.range_constexpr(T):
                inputs[3 * t] = raw_qk[t * 128 + kt]
                inputs[3 * t + 1] = raw_qk[(T + t) * 128 + kt]
                inputs[3 * t + 2] = BFloat16(0)
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    inputs[3 * t + 2] = raw_v[t * ROWS + kt]
            gate_owned = raw_g[kt]
            for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        cf = Float32(coeff[s, i])
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), cf, acc)
            # KDA CTA r activates and normalizes only token r's Q / K channels (the
            # same operations as for every token before) and all-gathers them to the
            # eight KDA CTAs; its V rows stay local (threads kt < 16, every token).
            pre_q, pre_k = accum[0], accum[1]
            for t in cutlass.range_constexpr(1, T):
                pre_q = Float32(cutlass.select_(rank == t, accum[3 * t], pre_q))
                pre_k = Float32(cutlass.select_(rank == t, accum[3 * t + 1], pre_k))
            if kt < ROWS:  # V pre-activations; thread kt activates (row kt % 16, token
                for t in cutlass.range_constexpr(T):  # kt // 16) after the barrier
                    shared_v[t * ROWS + kt] = accum[3 * t + 2]
            if cutlass.const_expr(self.stock):
                q, k = self.silu(pre_q), self.silu(pre_k)
                out_gate = self.activate(Float32(gate_owned))
            else:  # (the three sigmoids' fast paths together)
                act = sigmoids([pre_q, pre_k, Float32(gate_owned)])
                q, k, out_gate = pre_q * act[0], pre_k * act[1], act[2]
            square_q, square_k = warp_sum(q * q), warp_sum(k * k)
            if lane == 0:
                reductions[warp] = square_q
                reductions[warp + 4] = square_k
            named_barrier(1, 128)
            pre_v = shared_v[kt]
            shared_v[kt] = self.silu_v(pre_v)
            qn = (reductions[0] + reductions[2]) + (reductions[1] + reductions[3])
            kn = (reductions[4] + reductions[6]) + (reductions[5] + reductions[7])
            fq = self.norm_factor(qn + 1.0e-6, scale)
            fk = self.norm_factor(kn + 1.0e-6)
            q_n, k_n = self.normalized(q, fq, scale), self.normalized(k, fk)
            q_ptr = shared_q.iterator + rank * 128 + kt
            k_ptr = shared_k.iterator + rank * 128 + kt
            for dest in cutlass.range_constexpr(KDA_CTAS):
                push_float(q_ptr, q_n, bar_qk, dest)
                push_float(k_ptr, k_n, bar_qk, dest)
            if kt < T:  # the tokens' checkpoint slots, for the rolled loop
                slot_rows[kt] = slots[0, kt]
            named_barrier(1, 128)  # shared_v, slot_rows
            wait(bar_qk)
            wait(bar_gate)
            # The token loop is a rolled loop (the next token's operands double-buffered
            # in registers): one ~130-instruction body instead of ~1000 instructions of
            # straight-line code fetched once per layer, which the serving neighbours'
            # code evicts from the instruction caches (KDA layer proxy: 807 cycles per
            # token unrolled vs 505 in the bare chain). KdaSplit's operations in its
            # order, so the same states and outputs.
            lanes = (group, group_lane)
            chunks: cutlass.Constexpr = self.vkda.chunks
            # Two operand buffers (q, k, decays of a token) alternate roles over a
            # body of two tokens: token t uses one and the other still holds token
            # t - 1's q for the pipelined query dot, then receives token t + 1's.
            buffers = (
                cute.make_rmem_tensor((3, chunks, 4), Float32),
                cute.make_rmem_tensor((3, chunks, 4), Float32),
            )
            for j in cutlass.range_constexpr(chunks):
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, j, e] = Float32(0)  # (token -1's q: output unused)
            sources = (shared_q, shared_k, shared_decay)
            self.load_operands(buffers[0], sources, 0, group_lane)
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    cur, other = buffers[half], buffers[1 - half]
                    v_row = shared_v[t * ROWS + group % ROWS]
                    beta = shared_beta[t]
                    out_prev, v_new = self.rolled_update_defer(
                        history, cur, other, sources, t, group_lane, v_row, beta
                    )
                    if (group_lane == 0) & (t > 0):  # (pushed together after the loop)
                        local_o[(t - 1) * ROWS + group] = self.round_output(out_prev)
                    if t == 0:  # the one state store: slot 0, after token 0
                        self.store_checkpoint(
                            history, state, slot_rows[0], head, value_base, lanes
                        )
                    elif (group_lane == 0) & live:  # token t's record: corrections
                        record_page[head, V_ROW + t - 1, value_base + group % ROWS] = (
                            v_new
                        )
            out_last = self.query_dot(history, buffers[1][0, None, None])
            if group_lane == 0:
                local_o[(T - 1) * ROWS + group] = self.round_output(out_last)
            # The outputs of every token: thread kt pushes row kt % 16 of token kt // 16
            # to that token's owner (one remote store per thread instead of eight
            # dependent ones per row lane inside the loop).
            named_barrier(1, 128)
            position = self.vkda.output_position(value_base + kt % ROWS)
            push_float(shared_o.iterator + position, local_o[kt], bar_out, kt // ROWS)
            if kt < ROWS:
                channel = 2 * dim + value_channel
                window = (
                    taps[2, 1],
                    taps[2, 2],
                    *(inputs[3 * t + 2] for t in range(T)),
                )
                for i, value in enumerate(window):
                    conv_put(conv_page, (channel, i), value)
            if partition < T:  # token ``partition``: the gated RMS norm
                wait(bar_out)
                if (partition == 0) & live:  # every KDA CTA folded: k / decay records
                    self.key_records(record_page, head, kt, shared_k, shared_decay)
                raw_o = shared_o[self.vkda.output_position(kt)]
                sums = cute.make_rmem_tensor(4, Float32)
                for g in cutlass.range_constexpr(4):
                    base = lane % 8 * 16 + g * 4
                    o4 = load_f32x4(shared_o.iterator + base, shared=True)
                    s0, s1, s2, s3 = (o * o for o in o4)
                    sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                rstd = self.inverse_sqrt(sumsq * (1.0 / 128) + eps)
                result = raw_o * rstd * nw * out_gate
                if cutlass.const_expr(self.polled_out):
                    store_bf16_canonical(at(output, partition, qk_channel), result)
                else:
                    output[partition, qk_channel] = BFloat16(result)
                if partition == 0:
                    for s in cutlass.range_constexpr(2):
                        channel = s * dim + qk_channel
                        window = (taps[s, 1], taps[s, 2])
                        window += tuple(inputs[3 * u + s] for u in range(T))
                        for i, value in enumerate(window):
                            conv_put(conv_page, (channel, i), value)

class KdaHeadMultiDefer(HeadRecords, KdaHeadMulti):
    """``KdaHeadMulti`` (q = 8) under the deferred protocol (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokens == KDA_CTAS, "KdaHeadMultiDefer: q8 (one token per KDA CTA)"

    @cute.jit
    def multi_early(self, kt, tensors, coords, slots):
        """``KdaHeadVerify.verify_early`` for this CTA's request and row block."""
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, _, offset, request, block = coords
        slot, records = slots[request, 0], slots[request, 1]  # deferred protocol
        pending = offset
        if (offset < 0) | (records <= 0):
            pending = Int32(0)
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base = block * ROWS
        safe = slot
        if safe < 0:
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        history = zeros((self.vkda.passes, self.vkda.chunks, 4), Float32)
        for j in cutlass.range_constexpr(self.vkda.chunks):
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            pointer = at(state_page, head, value_base + group, column)
            values = self.vkda.load_state(pointer)
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = values[e]
        if (slot > 0) & (pending > 0):  # the fold (pre-wait)
            record_page = state[Int64(records), None, None, None]
            self.head_fold(history, record_page, head, pending, value_base, group,
                           group_lane)  # fmt: skip
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3 * self.tokens, Float32)
        if slot > 0:
            dim = self.dim
            qk_channel = head * 128 + kt
            value_channel = head * 128 + value_base + kt
            conv_page = conv_state[Int64(slots[request, 0]), None, None]
            for s in cutlass.range_constexpr(3):
                channel = value_channel if s == 2 else qk_channel
                if (s != 2) | (kt < ROWS):
                    for t in cutlass.range_constexpr(3):
                        taps[s, t] = conv_tap(conv_page, s * dim + channel, offset + t)
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = conv_w[s, t, channel]
                    if cutlass.const_expr(self.has_bias):
                        accum[s] = bias[s * dim + channel]
                        for t in cutlass.range_constexpr(1, self.tokens):
                            accum[3 * t + s] = accum[s]
        return history, taps, coeff, accum, norm_w[kt]


    @cute.jit
    def multi_main(
        self, kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
    ):
        """``KdaHeadVerify.verify_main`` for this CTA's request and row block; at q = 8
        also the gated RMS norm of token row ``rank`` (the same request's position
        ``block``; at q < 8 ``owned_norm`` runs it)."""
        T: cutlass.Constexpr = self.tokens
        dim = self.dim
        history, taps, coeff, accum, nw = early
        raw_qk, raw_v, mine, bar_hop = raws
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows[:6]
        local_o, slot_rows = rows[6:]
        bar_gate, bar_out, shared_beta, bar_qk = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset, request, block = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base, qk_channel = block * ROWS, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        first = KDA_CTAS * request
        conv_slot = slots[request, 0]
        records = slots[request, 1]
        live = records > 0
        record_page = state[Int64(cutlass.select_(records > 0, records, Int32(0))), None, None, None]
        if slot <= 0:
            if cutlass.const_expr(T == KDA_CTAS):
                output[rank, qk_channel] = BFloat16(0)
        else:
            conv_page = conv_state[Int64(conv_slot), None, None]
            wait(bar_hop)
            inputs = cute.make_rmem_tensor(3 * T, BFloat16)
            for t in cutlass.range_constexpr(T):
                inputs[3 * t] = raw_qk[t * 128 + kt]
                inputs[3 * t + 1] = raw_qk[(T + t) * 128 + kt]
                inputs[3 * t + 2] = BFloat16(0)
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    inputs[3 * t + 2] = raw_v[t * ROWS + kt]
            if cutlass.const_expr(T == KDA_CTAS):  # the owned row's output gate
                gate_owned = mine[384 + kt]
            for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        cf = Float32(coeff[s, i])
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), cf, acc)
            # KDA CTA j of the request activates and normalizes only token j's Q / K
            # channels and all-gathers them to the request's eight KDA CTAs.
            pre_q, pre_k = accum[0], accum[1]
            for t in cutlass.range_constexpr(1, T):
                pre_q = Float32(cutlass.select_(block == t, accum[3 * t], pre_q))
                pre_k = Float32(cutlass.select_(block == t, accum[3 * t + 1], pre_k))
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    shared_v[t * ROWS + kt] = accum[3 * t + 2]
            if cutlass.const_expr(self.stock):
                q, k = self.silu(pre_q), self.silu(pre_k)
                if cutlass.const_expr(T == KDA_CTAS):
                    out_gate = self.activate(Float32(gate_owned))
            else:
                if cutlass.const_expr(T == KDA_CTAS):
                    act = sigmoids([pre_q, pre_k, Float32(gate_owned)])
                    q, k, out_gate = pre_q * act[0], pre_k * act[1], act[2]
                else:  # (owners gate their rows in ``owned_norm``)
                    act = sigmoids([pre_q, pre_k])
                    q, k = pre_q * act[0], pre_k * act[1]
            square_q, square_k = warp_sum(q * q), warp_sum(k * k)
            if lane == 0:
                reductions[warp] = square_q
                reductions[warp + 4] = square_k
            named_barrier(1, 128)
            if cutlass.const_expr(T * ROWS == 128):  # noqa: SIM114
                pre_v = shared_v[kt]
                shared_v[kt] = self.silu_v(pre_v)
            elif kt < T * ROWS:  # (row kt % 16, token kt // 16)
                pre_v = shared_v[kt]
                shared_v[kt] = self.silu_v(pre_v)
            qn = (reductions[0] + reductions[2]) + (reductions[1] + reductions[3])
            kn = (reductions[4] + reductions[6]) + (reductions[5] + reductions[7])
            fq = self.norm_factor(qn + 1.0e-6, scale)
            fk = self.norm_factor(kn + 1.0e-6)
            q_n, k_n = self.normalized(q, fq, scale), self.normalized(k, fk)
            q_ptr = shared_q.iterator + block * 128 + kt
            k_ptr = shared_k.iterator + block * 128 + kt
            if cutlass.const_expr(T == KDA_CTAS):  # noqa: SIM114
                for dest in cutlass.range_constexpr(KDA_CTAS):
                    push_float(q_ptr, q_n, bar_qk, first + dest)
                    push_float(k_ptr, k_n, bar_qk, first + dest)
            elif block < T:  # KDA CTAs past the request's tokens normalize none
                for dest in cutlass.range_constexpr(KDA_CTAS):
                    push_float(q_ptr, q_n, bar_qk, first + dest)
                    push_float(k_ptr, k_n, bar_qk, first + dest)
            if kt < T:
                slot_rows[kt] = slots[request, kt]
            named_barrier(1, 128)  # shared_v, slot_rows
            wait(bar_qk)
            wait(bar_gate)
            lanes = (group, group_lane)
            chunks: cutlass.Constexpr = self.vkda.chunks
            buffers = (
                cute.make_rmem_tensor((3, chunks, 4), Float32),
                cute.make_rmem_tensor((3, chunks, 4), Float32),
            )
            for j in cutlass.range_constexpr(chunks):
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, j, e] = Float32(0)
            sources = (shared_q, shared_k, shared_decay)
            self.load_operands(buffers[0], sources, 0, group_lane)
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    cur, other = buffers[half], buffers[1 - half]
                    v_row = shared_v[t * ROWS + group % ROWS]
                    beta = shared_beta[t]
                    out_prev, v_new = self.rolled_update_defer(
                        history, cur, other, sources, t, group_lane, v_row, beta
                    )
                    if (group_lane == 0) & (t > 0):
                        local_o[(t - 1) * ROWS + group] = self.round_output(out_prev)
                    if t == 0:  # the one state store: slot 0, after token 0
                        self.store_checkpoint(
                            history, state, slot_rows[0], head, value_base, lanes
                        )
                    elif (group_lane == 0) & live:  # token t's record: corrections
                        record_page[head, V_ROW + t - 1, value_base + group % ROWS] = (
                            v_new
                        )
            out_last = self.query_dot(history, buffers[1][0, None, None])
            if group_lane == 0:
                local_o[(T - 1) * ROWS + group] = self.round_output(out_last)
            # thread kt pushes row kt % 16 of token kt // 16 to that token row's owner
            named_barrier(1, 128)
            position = self.vkda.output_position(value_base + kt % ROWS)
            if cutlass.const_expr(T == KDA_CTAS):
                push_float(
                    shared_o.iterator + position,
                    local_o[kt],
                    bar_out,
                    first + kt // ROWS,
                )
            elif kt < T * ROWS:  # token kt // 16's owner: token row T request + t
                owner = T * request + kt // ROWS
                push_float(shared_o.iterator + position, local_o[kt], bar_out, owner)
            if kt < ROWS:
                channel = 2 * dim + value_channel
                window = (
                    taps[2, 1],
                    taps[2, 2],
                    *(inputs[3 * t + 2] for t in range(T)),
                )
                for i, value in enumerate(window):
                    conv_put(conv_page, (channel, i), value)
            if cutlass.const_expr(T == KDA_CTAS):
                # token row ``rank`` (position ``block`` of this CTA's request)
                wait(bar_out)
                if (block == 0) & live:  # the request's KDA CTAs folded: k / decay
                    self.key_records(record_page, head, kt, shared_k, shared_decay)
                self.norm_row(kt, shared_o, output, rank, qk_channel, nw, out_gate, eps)
            if block == 0:
                for s in cutlass.range_constexpr(2):
                    channel = s * dim + qk_channel
                    window = (taps[s, 1], taps[s, 2])
                    window += tuple(inputs[3 * u + s] for u in range(T))
                    for i, value in enumerate(window):
                        conv_put(conv_page, (channel, i), value)



def set_head_protocol(defer: bool) -> None:
    """The fused heads ``compile_head_verify`` / ``compile_head_multi`` build: the
    record forms (``defer``) or the stock checkpoint ones."""
    head_verify.HEAD_VERIFY_KIND = KdaHeadVerifyDefer if defer else KdaHeadVerify
    head_multi.HEAD_MULTI_KIND = KdaHeadMultiDefer if defer else KdaHeadMulti
    head_verify.compile_head_verify.cache_clear()
    head_multi.compile_head_multi.cache_clear()
