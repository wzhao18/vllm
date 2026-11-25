# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyLoopStock`` as a deferred RecoverSSM record core (``sp.SpKdaCore`` kind
"loop"): the persistent 16-warp stock-math verify, with one checkpoint slot per
request instead of q.

Why: the plain verify writes every token's FP32 state (q = 8 pages of 64 KiB per
(request, head)) so that the next step can start from the accepted one. At TP16 (6
heads) those stores cost 5.5 / 6.1 / 12.0 us per layer at B16 / B32 / B64 (VR200,
checkpoint-store ablation of LoopStock), 40-45 % of the core. Under ``--use-replayssm``
the next step folds the accepted tokens itself, so a verify writes one state page
(the folded checkpoint, only when there is something to fold) and ~8 KiB of records.

Per (request, head) item, on top of ``KdaVerifyLoopStock`` (same thread layout, token
loop, prologue, gates and gated norm, so the outputs and the recurrence are LoopStock's
bit for bit given the same starting state):

1. the item's record slot ``record_rows[request]`` and its pending count (the accepted
   tokens of its last verify, clipped to [0, q]) are read with its slot;
2. before any record of this step is written, every thread reads the old corrections
   of its two state rows (the old keys and raw gates feed step 3);
3. as soon as the records land (bulk-copied ahead of the state page), all 16 warps
   build the stock commit's per-key operands exactly as ``wide.KdaWide`` does (warp
   butterfly + cross-warp sum of the recorded key squares, rsqrt.approx.ftz
   normalized keys, the Triton gate decays, the reverse product P_t and D_n), and
   every thread folds its 32 state elements with the commit's instructions
   (fc = fma(u_t kn_t, P_t, fc) over t = n - 1 .. 0, then S = fma(S_0, D_n, fc)):
   bit for bit the stock Triton commit; the folded checkpoint is written back once;
4. the conv window is read at the offset n - 1 the commit's compaction would have rolled
   it by;
5. the verify writes the step's records into the record slot: the raw keys (BF16
   SiLU outputs) and raw gates (BF16) per token, and the corrections u_t (FP32) per
   state row; no state page per token.

Precision: the stock verify's math (the Triton formulas, BF16 roundings at the stock
points, FP32 state and recurrence), the stock commit bit for bit; only the FP32
reduction order of the recurrence dots differs from the Triton verify (as every record
core: ``sp`` docstring)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma,
    fma2,
    load_f32x4,
    mma_into,
    mul2,
    named_barrier,
    store_f32x4,
    wait,
    warp_sum,
    zeros,
)
from .kernel import load_activation_bf16x4, load_bf16x4
from .recover import (
    mul_rn,
    reciprocal,
    sqrt_approx,
    triton_decay,
    triton_exp,
    triton_sigmoid,
)
from .verify_loop import COPY_BYTES, PAGE_BYTES, K, KdaVerifyLoopStock
from .verify_packed import CHUNKS
from .wide import rsqrt_ftz
from .window import conv_put, conv_tap


class KdaVerifyLoopRecords(KdaVerifyLoopStock):
    """``KdaVerifyLoopStock`` writing deferred RecoverSSM records (module docstring).
    Operands: ``sp.SpKdaCore``'s deferred launch (slots [B, 1], accepted [B], then the
    corrections [R, H, q, 128] FP32, keys / raw gates [R, H, q, 256] BF16, record rows
    [B] and pending counts [R])."""

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        correction, record_rows, pending = (args[i] for i in (18, 20, 21))
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        M: cutlass.Constexpr = self.most
        RQ: cutlass.Constexpr = correction.shape[2]
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
        fold_keys = alloc(smem, Float32, 128 * RQ)  # kn_t
        fold_suffix = alloc(smem, Float32, 128 * RQ)  # P_t
        fold_decay = alloc(smem, Float32, 128)  # D_n
        fold_sums = alloc(smem, Float32, 4 * RQ)  # recorded key squares per warp
        fold_dec = alloc(smem, Float32, 128 * RQ)  # the recorded tokens' decays
        # the item's old records, bulk-copied with its state page
        staged_u = alloc(smem, Float32, RQ * 128, align=128)
        staged_kg = alloc(smem, BFloat16, RQ * 256, align=128)
        # per item of the CTA: [slot, conv offset, live index, record row, pending]
        meta = alloc(smem, Int32, (M, 5))
        live_items = alloc(smem, Int32, M + 1)  # [n]: the n-th live item; [M]: count
        pairs: cutlass.Constexpr = (T + 1) // 2
        bars = alloc(smem, Int64, pairs + 2, 8).iterator
        ready, state_full, records_full = bars, bars + pairs, bars + pairs + 1

        # The CTA's items: thread i < count reads item i's slot, accepted count,
        # record row and pending count (two dependent round trips).
        if thread < count:
            request = (first + thread) // H
            taken = accepted[request]
            offset = taken - 1
            if taken < 1:  # a padding row's count may be 0 (its slot is <= 0)
                offset = Int32(0)
            meta[thread, 0] = slots[request, 0]
            rec = record_rows[request]
            n = Int32(0)
            if rec >= 0:
                n = pending[rec]
            if n > RQ:
                n = Int32(RQ)
            if n < 0:
                n = Int32(0)
            if n > 0:  # the window the commit's compaction would have rolled
                offset = offset + n - 1
            meta[thread, 1] = offset
            meta[thread, 3] = rec
            meta[thread, 4] = n
        if thread == 0:
            for p in cutlass.range_constexpr(pairs):
                cute.arch.mbarrier_init(ready + p, 128 + 32 * 8)
            cute.arch.mbarrier_init(state_full, 1)
            cute.arch.mbarrier_init(records_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if thread == 0:
            live = Int32(0)
            for i in cutlass.range(count, unroll=1):
                meta[i, 2] = live
                if meta[i, 0] > 0:
                    live_items[live] = i
                    live += 1
            live_items[M] = live
            self.stage_records(args, page, staged_u, staged_kg, state_full,
                               records_full, meta, live_items, first, 0)  # fmt: skip
        cute.arch.sync_threads()
        nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
        row0 = 8 * warp + group

        for item_index in cutlass.range(count, unroll=1):
            item = first + item_index
            head, request = item % H, item // H
            token = request * T
            slot, offset = meta[item_index, 0], meta[item_index, 1]
            live, rec, n = meta[item_index, 2], meta[item_index, 3], meta[item_index, 4]
            phase = live % 2
            if slot <= 0:  # a padding request
                cute.arch.griddepcontrol_wait()
                if thread < 128:
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + thread] = BFloat16(0)
            else:
                # The pending commit's per-key operands as soon as the old records
                # land (staged ahead of the page), on every warp.
                wait(records_full, phase, scope="cta")
                if n > 0:
                    self.fold_operands(args, thread, head, n, staged_kg, fold_sums,
                                       fold_dec, fold_keys, fold_suffix,
                                       fold_decay)  # fmt: skip
                wait(state_full, phase, scope="cta")
                history = zeros((2, CHUNKS, 4), Float32)
                for r in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(CHUNKS):
                        pointer = page.iterator + (row0 + 4 * r) * K + 4 * (j + 8 * c)
                        values = load_f32x4(pointer, shared=True)
                        for e in cutlass.range_constexpr(4):
                            history[r, c, e] = values[e]
                # The old corrections of the thread's rows (read before this step
                # writes any record).
                u_old = zeros((2, RQ), Float32)
                if n > 0:
                    for r in cutlass.range_constexpr(2):
                        for t in cutlass.range_constexpr(RQ):
                            u_old[r, t] = staged_u[128 * t + row0 + 4 * r]
                cute.arch.sync_threads()  # every thread holds the page and old records
                if thread == 0:  # the next live item's page into the freed page
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.stage_records(args, page, staged_u, staged_kg, state_full,
                                       records_full, meta, live_items, first,
                                       live + 1)  # fmt: skip

                # The pending commit before the PDL wait (it reads only the last
                # step's records and this layer's checkpoint), the folded checkpoint
                # written back once.
                if n > 0:
                    self.fold_rows(history, u_old, n, RQ, j,
                                   (fold_keys, fold_suffix, fold_decay))  # fmt: skip
                    destination = state[Int64(slot), None, None, None]
                    for r in cutlass.range_constexpr(2):
                        for c in cutlass.range_constexpr(CHUNKS):
                            values4 = [history[r, c, e] for e in range(4)]
                            target = at(
                                destination, head, row0 + 4 * r, 4 * (j + 8 * c)
                            )
                            store_f32x4(target, values4)

                own_gates = zeros(2, gate.element_type)
                if thread < 128:
                    self.record_prologue(args, thread, token, head, offset, slot,
                                         shared_q, shared_k, shared_v, reductions,
                                         shared_beta, own_gates, ready, rec)  # fmt: skip
                elif thread < 128 + 32 * 8:
                    self.record_gate(args, thread // 32 - 4, lane, head, token,
                                     shared_decay, own_gates, rec)  # fmt: skip
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
                    self.record_token(history, t, group, j, row0, shared_q, shared_k,
                                      shared_decay, shared_v, shared_beta, shared_o,
                                      correction, rec, head)  # fmt: skip
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
    def stage_records(self, args, page, staged_u, staged_kg, state_full,
                      records_full, meta, live_items, first, n):  # fmt: skip
        """Thread 0: for the ``n``-th live item (if there is one), first its old
        records (corrections, keys / raw gates of its record slot and head; a plain
        arrival without a pending commit) on ``records_full``, then its state page on
        ``state_full``."""
        state, correction, key_gate = args[9], args[18], args[19]
        H: cutlass.Constexpr = self.heads
        M: cutlass.Constexpr = self.most
        RQ: cutlass.Constexpr = correction.shape[2]
        u_bytes: cutlass.Constexpr = RQ * 128 * 4
        kg_bytes: cutlass.Constexpr = RQ * 256 * 2
        if n < live_items[M]:
            index = live_items[n]
            slot, head = meta[index, 0], (first + index) % H
            rec, pending = meta[index, 3], meta[index, 4]
            source = state[Int64(slot), None, None, None]
            if pending > 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    records_full, u_bytes + kg_bytes
                )
                copy_g2s(staged_kg.iterator, at(key_gate, rec, head, 0, 0), kg_bytes,
                         records_full)  # fmt: skip
                copy_g2s(staged_u.iterator, at(correction, rec, head, 0, 0), u_bytes,
                         records_full)  # fmt: skip
            else:
                cute.arch.mbarrier_arrive(records_full)
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
    def fold_operands(self, args, thread, head, n, staged_kg, fold_sums, fold_dec,
                      fold_keys, fold_suffix, fold_decay):  # fmt: skip
        """The stock commit's per-key operands of the n pending tokens (``wide.KdaWide``
        bit for bit), on all 16 warps: the recorded key squares (warp butterfly over
        32 keys, then the cross-block sum), the Triton gate decays, then per key the
        normalized keys kn_t, the reverse products P_t and D_n."""
        a_log, dt_bias, lower_bound = args[6], args[7], args[13]
        RQ: cutlass.Constexpr = args[18].shape[2]
        lane, warp = thread % 32, thread // 32
        # squares: task (t, b) = keys [32 b, 32 b + 32) of token t; warp w: w, w + 16
        for i in cutlass.range_constexpr(-(-4 * RQ // 16)):
            task = warp + 16 * i
            t, b = task // 4, task % 4
            if (task < 4 * RQ) & (t < n):
                x = Float32(staged_kg[256 * t + 32 * b + lane])
                squares = warp_sum(x * x)
                if lane == 0:
                    fold_sums[4 * t + b] = squares
        # decays: key = thread % 128 of tokens thread // 128 + 4 i
        key = thread % 128
        a = Float32(a_log[head])
        dtc = Float32(dt_bias[head * 128 + key])
        for i in cutlass.range_constexpr(-(-RQ // 4)):
            t = thread // 128 + 4 * i
            if t < n:
                raw = Float32(staged_kg[256 * t + 128 + key])
                fold_dec[128 * t + key] = triton_decay(a, raw, dtc, lower_bound)
        cute.arch.sync_threads()
        if thread < 128:
            final_decay = Float32(1)
            for i in cutlass.range_constexpr(RQ):
                t = RQ - 1 - i
                if t < n:
                    r4 = [fold_sums[4 * t + w] for w in range(4)]
                    total = (r4[0] + r4[2]) + (r4[1] + r4[3]) + 1.0e-6
                    raw_key = Float32(staged_kg[256 * t + thread])
                    fold_keys[128 * t + thread] = mul_rn(rsqrt_ftz(total), raw_key)
                    fold_suffix[128 * t + thread] = final_decay
                    final_decay = mul_rn(final_decay, fold_dec[128 * t + thread])
            fold_decay[thread] = final_decay
        cute.arch.sync_threads()

    @cute.jit
    def fold_rows(self, history, u_old, n, RQ: cutlass.Constexpr, j, operands):
        """``wide.KdaWide.fold`` on this thread's rows row0 + 4 r and keys
        4 (j + 8 c) + [0, 4): the stock commit's instructions per element."""
        keys, suffix, decay = operands
        for c in cutlass.range_constexpr(CHUNKS):
            key = 4 * (j + 8 * c)
            d4 = load_f32x4(decay.iterator + key, shared=True)
            fc = zeros((2, 4), Float32)
            for i in cutlass.range_constexpr(RQ):
                t = RQ - 1 - i
                if t < n:
                    k4 = load_f32x4(keys.iterator + 128 * t + key, shared=True)
                    p4 = load_f32x4(suffix.iterator + 128 * t + key, shared=True)
                    for r in cutlass.range_constexpr(2):
                        u = u_old[r, t]
                        for p in cutlass.range_constexpr(2):
                            update = mul2((u, u), (k4[2 * p], k4[2 * p + 1]))
                            acc = (fc[r, 2 * p], fc[r, 2 * p + 1])
                            acc = fma2(update, (p4[2 * p], p4[2 * p + 1]), acc)
                            fc[r, 2 * p], fc[r, 2 * p + 1] = acc
            for r in cutlass.range_constexpr(2):
                for p in cutlass.range_constexpr(2):
                    pair = (history[r, c, 2 * p], history[r, c, 2 * p + 1])
                    pair = fma2(pair, (d4[2 * p], d4[2 * p + 1]),
                                (fc[r, 2 * p], fc[r, 2 * p + 1]))  # fmt: skip
                    history[r, c, 2 * p], history[r, c, 2 * p + 1] = pair

    @cute.jit
    def record_prologue(self, args, thread, token, head, offset, conv_slot, shared_q,
                        shared_k, shared_v, reductions, shared_beta, own_gates, ready,
                        rec):  # fmt: skip
        """``KdaVerifyLoopStock.loop_prologue`` (same operations) that also records
        each token's raw key (the BF16 SiLU output of channel ``thread``)."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        scale, key_gate = args[14], args[19]
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
                key_gate[rec, head, t, thread] = BFloat16(k)
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
                conv_put(conv_page, (s * dim + channel, i), value)

    @cute.jit
    def record_gate(self, args, g, lane, head, token, shared_decay, own_gates, rec):
        """``KdaVerifyLoopStock.loop_gate`` (same operations) that also records each
        token's BF16 raw gate of the warp's features."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        key_gate = args[19]
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
                    key_gate[rec, head, t, 128 + base + features[i]] = BFloat16(raw_g)

    @cute.jit
    def record_token(self, history, t, group, j, row0, shared_q, shared_k,
                     shared_decay, shared_v, shared_beta, shared_o, correction, rec,
                     head):  # fmt: skip
        """``KdaVerifyLoop.loop_token`` (same operations) recording the token's
        corrections u_t of the thread's two rows instead of storing its state."""
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
        v_new = [(shared_v[128 * t + row0 + 4 * r] - dot_k[r]) * beta for r in range(2)]
        if j == 0:
            for r in cutlass.range_constexpr(2):
                correction[rec, head, t, row0 + 4 * r] = v_new[r]
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
