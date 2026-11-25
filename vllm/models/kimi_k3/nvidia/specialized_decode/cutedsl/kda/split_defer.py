# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaSplitStock``'s verify under the deferred-verify state protocol
(``verify_defer``'s module docstring): the state at slot 0 plus the previous
verify's records at slot 1, folded before the PDL wait; one state store (slot 0,
after token 0) instead of q; the records of tokens 1 .. q - 1 to slot 1.

Each CTA bulk-copies the head's record rows (``RECORD_BYTES``: k, decay, corrections
of tokens 1 .. 7) into shared memory before the PDL wait when the request has
pending tokens, and folds them into its registers with ``update``'s two instructions
(``history *= decay``, then ``fma(k, v_new, history)``), so the folded state is bit
for bit the checkpoint the stock protocol stores. A CTA writes its rows'
corrections (as the RecoverSSM record mode does); partition 0 writes the head's k /
decay rows once every CTA of the cluster has folded (after the token-0 output
barrier, which every CTA's token-0 outputs arrive on, i.e. after its fold)."""

import cutlass
import cutlass.cute as cute
from ..common import diagstamp as ds
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma,
    load_f32x4,
    push_float,
    wait,
    warp_sum,
    zeros,
)
from .recover import KdaSplit as RecoverKdaSplit
from .recover import advance, mul_rn, reciprocal, sqrt_approx
from .verify_defer import DECAY_ROW, RECORD_BYTES, RECORD_ROWS, V_ROW, put_tag
from .window import conv_tap


class KdaSplitDefer(RecoverKdaSplit):
    """``prepared.KdaSplitStock`` (verify, q >= 2) under the deferred protocol."""

    bf16_state = False
    # The fold after the PDL wait, under the input loads (its state loads then
    # overlap them as the stock verify's do) instead of before the wait.
    fold_late = False
    # The fold unrolled over the record slots with selects (``fold_selected``).
    fold_unrolled = True

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens):
        assert tokens >= 2, "KdaSplitDefer: verify only"
        schedule = tuple(schedule)
        if schedule[0] < 2:  # not staged
            schedule = schedule[:4]
        elif schedule[0] == 3:
            schedule = (2, *schedule[1:])
        super().__init__(heads, batch, 0, has_bias, schedule, tokens)
        self.narrow_fb = True
        assert not self.staged

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, raw_gate, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b_weight = args[
            9:17
        ]
        T: cutlass.Constexpr = self.tokens
        thread, _, _ = cute.arch.thread_idx()
        partition, head, token = cute.arch.block_idx()
        # DIAG (STAMP arm, KDA U2): thread 0 phase stamps, CTA key (token, head, part)
        nparts, nheads, _ = cute.arch.grid_dim()
        dkey = (token * nheads + head) * nparts + partition
        dbase, dctr = ds.buffer("kdasplit", x.shape[0])
        saddr = ds.begin(dbase, dctr, dkey, thread == 0)
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_wait()
        request, token = token, token * T
        taken = args[17][request]
        offset = taken - 1
        pending = offset
        if taken < 1:  # a padding row's count may be 0 (its slots are <= 0)
            pending = Int32(0)
        # deferred protocol: state at slot 0, the pending tokens' records at slot 1
        slot, conv_slot = slots[request, 0], slots[request, 0]
        records = slots[request, 1]
        if records <= 0:
            pending = Int32(0)
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            if (partition == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[advance(token, t), head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            lane, warp = thread % 32, thread // 32
            group = thread // self.lanes_per_row
            group_lane = thread % self.lanes_per_row
            dim: cutlass.Constexpr = self.heads * 128
            value_base, qk_channel = partition * self.rows, head * 128 + thread
            value_channel = head * 128 + value_base + thread

            smem = cutlass.utils.SmemAllocator()
            shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = (
                alloc(smem, Float32, size * T)
                for size in (128, 128, 128, self.rows, 128, 8)
            )
            barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
            owned: cutlass.Constexpr = -(-T // self.cluster)  # tokens a CTA normalizes
            if cutlass.const_expr(owned > 1):  # one output barrier per owned token
                barrier = smem.allocate(8 * owned, 8)
                barrier = cute.recast_ptr(barrier, dtype=cutlass.Uint64)
            rec = alloc(smem, Float32, RECORD_ROWS * 128, align=128)
            rec_barrier = smem.allocate(cutlass.Uint64, 8)
            if cutlass.const_expr(self.trigger_at_entry):
                cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(not self.early):
                self.setup(barrier, gate_barrier, partition, thread)
                cute.arch.cluster_wait()
            ds.stamp(saddr, 1)  # setup / cluster barrier passed
            if thread >= 128:
                if cutlass.const_expr(self.early):
                    cute.arch.cluster_arrive_relaxed()
                coords = (head, token, value_base, lane)
                self.gate_warp(args, shared_decay, gate_barrier, coords, None)

            if thread < 128:
                if cutlass.const_expr(self.early):  # thread 0 arms; barrier 1 orders it
                    self.setup(barrier, gate_barrier, partition, thread)
                page = lambda s: state[Int64(s), None, None, None]
                state_page = page(slot)
                conv_page = conv_state[Int64(conv_slot), None, None]
                state_row = lambda r, c, s=state_page: at(s, head, value_base + r, c)
                if (thread == 0) & (pending > 0):  # the head's records, with the rows
                    cute.arch.mbarrier_init(rec_barrier, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(rec_barrier, RECORD_BYTES)
                    copy_g2s(rec.iterator, at(page(records), head, 0, 0),
                             RECORD_BYTES, rec_barrier)  # fmt: skip
                history = zeros((self.passes, self.chunks, 4), Float32)
                for p in cutlass.range_constexpr(self.passes):
                    row = p * self.groups + group
                    if row < self.rows:
                        for j in cutlass.range_constexpr(self.chunks):
                            column = 4 * (group_lane + self.lanes_per_row * j)
                            values = load_f32x4(state_row(row, column))
                            for e in cutlass.range_constexpr(4):
                                history[p, j, e] = values[e]
                conv_type: cutlass.Constexpr = conv_state.element_type
                taps = zeros((3, 3), BFloat16)
                coeff = zeros((3, 4), weights.element_type if self.early else Float32)
                accum = zeros(3 * T, Float32)
                for s in cutlass.range_constexpr(3):
                    channel = value_channel if s == 2 else qk_channel
                    if (s != 2) | (thread < self.rows):
                        for t in cutlass.range_constexpr(3):
                            tap_row = s * dim + channel
                            taps[s, t] = conv_tap(conv_page, tap_row, offset + t)
                        for t in cutlass.range_constexpr(4):
                            coeff[s, t] = weights[s, t, channel]
                        if cutlass.const_expr(self.has_bias):
                            accum[s] = bias[s * dim + channel]
                            for t in cutlass.range_constexpr(1, T):
                                accum[3 * t + s] = accum[s]
                nw = norm_weight[thread]
                # rec_barrier armed before any thread waits on it
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if cutlass.const_expr(not self.fold_late):  # the fold before the wait
                    if pending > 0:
                        wait(rec_barrier, 0, scope="cta")
                        self.fold(history, rec, pending, value_base, group, group_lane)
                ds.stamp(saddr, 2)  # state rows + taps loaded (+ early fold)

                cute.arch.griddepcontrol_wait()
                ds.stamp(saddr, 3)  # PDL wait returned (in-proj rows ready)
                inputs = cute.make_rmem_tensor(3 * T, BFloat16)
                betas, gates = [], []
                for t in cutlass.range_constexpr(T):
                    row = advance(token, t)
                    inputs[3 * t], inputs[3 * t + 1] = (
                        x[row, qk_channel],
                        x[row, dim + qk_channel],
                    )
                    inputs[3 * t + 2] = BFloat16(0)
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102 (all tokens)
                        if thread < self.rows:
                            for u in cutlass.range_constexpr(T):
                                channel = 2 * dim + value_channel
                                inputs[3 * u + 2] = x[advance(token, u), channel]
                    if cutlass.const_expr(self.early):  # converted at first use
                        betas.append(beta[row, head])
                        gates.append(gate[row, qk_channel])
                    else:
                        betas.append(Float32(beta[row, head]))
                        gates.append(Float32(gate[row, qk_channel]))
                cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.fold_late):  # the fold under the input loads
                    if pending > 0:
                        wait(rec_barrier, 0, scope="cta")
                        self.fold(history, rec, pending, value_base, group, group_lane)

                vs, squares, qks = [], [], []
                for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                    for s in cutlass.range_constexpr(3):
                        for i in cutlass.range_constexpr(4):
                            w = t + i
                            term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                            term = Float32(term)
                            cf = Float32(coeff[s, i]) if self.early else coeff[s, i]
                            acc = accum[3 * t + s]
                            accum[3 * t + s] = fma(term, cf, acc)
                for t in cutlass.range_constexpr(T):
                    q = self.silu(accum[3 * t])
                    k = self.silu(accum[3 * t + 1])
                    v = self.silu(accum[3 * t + 2])
                    betas[t] = self.sigmoid(Float32(betas[t]))
                    gates[t] = self.sigmoid(Float32(gates[t]))
                    shared_q[advance(thread, 128 * t)] = q
                    shared_k[advance(thread, 128 * t)] = k
                    vs.append(v)
                    qks += [q, k]
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102
                        if thread < self.rows:
                            for u in cutlass.range_constexpr(T):
                                shared_v[advance(thread, self.rows * u)] = vs[u]
                    squares += [warp_sum(q * q), warp_sum(k * k)]
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102
                        if lane == 0:
                            for u in cutlass.range_constexpr(T):
                                reductions[advance(warp, 8 * u)] = squares[2 * u]
                                reductions[advance(warp + 4, 8 * u)] = squares[
                                    2 * u + 1
                                ]
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                def factors(t):
                    r = lambda i: reductions[8 * t + i]
                    qn = (r(0) + r(2)) + (r(1) + r(3)) + 1.0e-6
                    kn = (r(4) + r(6)) + (r(5) + r(7)) + 1.0e-6
                    return reciprocal(sqrt_approx(qn)), reciprocal(sqrt_approx(kn))

                def normalize(value, factor, query):  # q / |q| * scale
                    value = mul_rn(value, factor)
                    return mul_rn(value, scale) if query else value

                for t in cutlass.range_constexpr(T):  # normalize each token's q, k once
                    fq, fk = factors(t)
                    q_n = normalize(qks[2 * t], fq, True)
                    k_n = normalize(qks[2 * t + 1], fk, False)
                    shared_q[128 * t + thread] = q_n
                    shared_k[128 * t + thread] = k_n
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                def operands(t):
                    qr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    kr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        q4 = load_f32x4(
                            advance(shared_q.iterator, 128 * t) + column, shared=True
                        )
                        k4 = load_f32x4(
                            advance(shared_k.iterator, 128 * t) + column, shared=True
                        )
                        for e in cutlass.range_constexpr(4):
                            qr[j, e] = q4[e]
                            kr[j, e] = k4[e]
                    if cutlass.const_expr(t == 0):
                        wait(gate_barrier)
                    decays = []
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        decay_ptr = advance(shared_decay.iterator, 128 * t) + column
                        decays.append(load_f32x4(decay_ptr, shared=True))
                    pointer = shared_v.iterator + self.rows * t
                    values = cute.make_tensor(pointer, cute.make_layout(self.rows))
                    return kr, qr, decays, values, betas[t]

                ds.stamp(saddr, 4)  # conv + q/k normalized (late fold done)
                record_page = page(records)
                ready = operands(0)
                for t in cutlass.range_constexpr(T):
                    current, ready = ready, operands(t + 1) if t + 1 < T else None
                    dots = cute.make_rmem_tensor(self.passes, Float32)
                    owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                    for start in cutlass.range_constexpr(0, self.passes, self.wave):
                        stop = min(start + self.wave, self.passes)
                        lanes = (group, group_lane)
                        corrections = None
                        if cutlass.const_expr(t > 0):  # token t's record: corrections
                            corrections = record_page[head, V_ROW + t - 1, None]
                        self.update_defer(
                            history, dots, current, lanes, (start, stop),
                            (corrections, value_base, records > 0),
                        )  # fmt: skip
                        if cutlass.const_expr(self.early and start == 0 and t == 0):
                            cute.arch.cluster_wait()
                        for p in cutlass.range_constexpr(start, stop):
                            row = p * self.groups + group
                            send = (row < self.rows) & (group_lane == 0)
                            o = dots[p]
                            o = Float32(BFloat16(o))  # stored BF16
                            position = self.output_position(value_base + row)
                            dst = advance(shared_o.iterator, 128 * t) + position
                            push_float(dst, o, bar, owner, when=send)
                        if cutlass.const_expr(t == 0):  # the one state store: slot 0
                            span = (start, stop)
                            self.store_rows(history, state_row, span, lanes, slot > 0)
                ds.stamp(saddr, 5)  # recurrence over T tokens done, state stored
                if thread < self.rows:
                    channel = 2 * dim + value_channel
                    window = (
                        taps[2, 1],
                        taps[2, 2],
                        *(inputs[3 * t + 2] for t in range(T)),
                    )
                    for i, value in enumerate(window):
                        conv_page[channel, i] = conv_type(value)

                for t in cutlass.range_constexpr(T):
                    if partition == t % self.cluster:
                        wait(advance(barrier, t // self.cluster))
                        raw_o = shared_o[advance(self.output_position(thread), 128 * t)]
                        sums = cute.make_rmem_tensor(4, Float32)
                        for g in cutlass.range_constexpr(4):
                            base = lane % 8 * 16 + g * 4
                            o_row = advance(shared_o.iterator, 128 * t)
                            o4 = load_f32x4(o_row + base, shared=True)
                            s0, s1, s2, s3 = (o * o for o in o4)
                            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                        variance = sumsq * (1.0 / 128) + eps
                        rstd = reciprocal(sqrt_approx(variance))
                        result = raw_o * rstd * nw * gates[t]
                        output[advance(token, t), qk_channel] = BFloat16(result)
                        if cutlass.const_expr(t == 0):
                            # every CTA's token-0 outputs arrived: all have folded,
                            # so the head's k / decay records may be overwritten
                            if records > 0:
                                self.key_records(record_page, head, thread, shared_k,
                                                 shared_decay)  # fmt: skip
                            for section in cutlass.range_constexpr(2):
                                channel = section * dim + qk_channel
                                window = (taps[section, 1], taps[section, 2])
                                window += tuple(
                                    inputs[3 * u + section] for u in range(T)
                                )
                                for i, value in enumerate(window):
                                    conv_page[channel, i] = conv_type(value)
                ds.stamp(saddr, 7)  # outputs normalized + written (exit)

    @cute.jit
    def key_records(self, record_page, head, thread, shared_k, shared_decay):
        """Column ``thread``'s k / decay records of tokens 1 .. q - 1."""
        for u in cutlass.range_constexpr(1, self.tokens):
            record_page[head, u - 1, thread] = shared_k[128 * u + thread]
            record_page[head, DECAY_ROW + u - 1, thread] = shared_decay[128 * u + thread]
        if thread == 0:
            put_tag(record_page, head)

    @cute.jit
    def fold(self, history, rec, pending, value_base, group, group_lane):
        """The ``pending`` accepted tokens 1 .. pending of the previous verify into
        the thread's rows from the staged records, ``update``'s instructions."""
        if cutlass.const_expr(self.fold_unrolled):
            self.fold_selected(history, rec, pending, value_base, group, group_lane)
        else:
            self.fold_loop(history, rec, pending, value_base, group, group_lane)

    @cute.jit
    def fold_selected(self, history, rec, pending, value_base, group, group_lane):
        """``fold_loop`` unrolled over the 7 record slots: every slot's operands load
        unconditionally (staged rows; past ``pending`` they are stale) and a token
        past ``pending`` keeps the state (a select of the unchanged value), so the
        loads of later tokens issue ahead of the earlier tokens' arithmetic."""
        for i in cutlass.range_constexpr(self.tokens - 1):
            live = i < pending
            for p in cutlass.range_constexpr(self.passes):
                safe_row = (p * self.groups + group) % self.rows
                v_new = rec[(V_ROW + i) * 128 + value_base + safe_row]
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    k4 = load_f32x4(rec.iterator + i * 128 + column, shared=True)
                    d4 = load_f32x4(
                        rec.iterator + (DECAY_ROW + i) * 128 + column, shared=True
                    )
                    for e in cutlass.range_constexpr(4):
                        h = history[p, j, e]
                        folded = fma(k4[e], v_new, h * d4[e])
                        history[p, j, e] = cutlass.select_(live, folded, h)

    @cute.jit
    def fold_loop(self, history, rec, pending, value_base, group, group_lane):
        for i in cutlass.range(pending, unroll=1):
            kr = cute.make_rmem_tensor((self.chunks, 4), Float32)
            decays = []
            for j in cutlass.range_constexpr(self.chunks):
                column = 4 * (group_lane + self.lanes_per_row * j)
                k4 = load_f32x4(rec.iterator + i * 128 + column, shared=True)
                for e in cutlass.range_constexpr(4):
                    kr[j, e] = k4[e]
                pointer = rec.iterator + (DECAY_ROW + i) * 128 + column
                decays.append(load_f32x4(pointer, shared=True))
            for p in cutlass.range_constexpr(self.passes):
                safe_row = (p * self.groups + group) % self.rows
                v_new = rec[(V_ROW + i) * 128 + value_base + safe_row]
                for j in cutlass.range_constexpr(self.chunks):
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] *= decays[j][e]
                for j in cutlass.range_constexpr(self.chunks):
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] = fma(kr[j, e], v_new, history[p, j, e])

    @cute.jit
    def update_defer(self, history, dots, operands, lanes, span, record):
        """``update`` (unstaged) writing the token's corrections when ``live``."""
        kr, qr, decays, shared_v, beta = operands
        (group, lane), (start, stop) = lanes, span
        corrections, value_base, live = record
        for p in cutlass.range_constexpr(start, stop):
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[p, j, e] *= decays[j][e]
            dots[p] = self.row_dot(history, kr, p)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for p in cutlass.range_constexpr(start, stop):
                dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)
        for p in cutlass.range_constexpr(start, stop):
            safe_row = (p * self.groups + group) % self.rows
            v_new = (shared_v[safe_row] - dots[p]) * beta
            if cutlass.const_expr(corrections is not None):  # noqa: SIM102
                if (lane == 0) & (p * self.groups + group < self.rows) & live:
                    corrections[value_base + safe_row] = v_new
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[p, j, e] = fma(kr[j, e], v_new, history[p, j, e])
            dots[p] = self.row_dot(history, qr, p)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for p in cutlass.range_constexpr(start, stop):
                dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)


