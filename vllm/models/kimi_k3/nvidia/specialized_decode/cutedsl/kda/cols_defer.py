# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyCols`` under the deferred-verify state protocol (``verify_defer``'s
module docstring): the state at slot 0 and the previous verify's records at slot 1,
bulk-copied into shared memory and folded into the registers before the PDL wait
with ``cols_update``'s two instructions (so bit for bit the checkpoint the stock
protocol stores); the state after token 0 to slot 0 (one store instead of q), the
warp's rows' corrections of tokens 1 .. q - 1 to slot 1 (lane 0), and the head's k /
decay rows by partition 0 once its token's outputs arrived from every CTA of the
cluster (each pushes after its fold)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    dot,
    fma,
    load_f32x4,
    named_barrier,
    push_float,
    scatter_reduce,
    wait,
    zeros,
)
from .verify import BETA, GATE, V, KdaVerifyCols
from .verify_defer import DECAY_ROW, RECORD_BYTES, RECORD_ROWS, V_ROW, put_tag


class KdaVerifyColsDefer(KdaVerifyCols):
    """``KdaVerifyCols`` (q8 rows) under the deferred protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokens == 8, "KdaVerifyColsDefer: q8 rows (the prepared copies)"

    @cute.kernel
    def kernel(self, args):
        prepared, slots, state, output, norm_weight = (
            args[i] for i in (0, 8, 9, 10, 12)
        )
        T: cutlass.Constexpr = self.tokens
        ROWS: cutlass.Constexpr = self.rows  # x T == 128
        dim: cutlass.Constexpr = self.heads * 128
        thread, _, _ = cute.arch.thread_idx()
        partition, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        value_base = partition * ROWS
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay = (
            alloc(smem, Float32, 128 * T) for _ in range(3)
        )
        shared_v, local_o, shared_o = (
            alloc(smem, Float32, size) for size in (128, 128, 128)
        )
        shared_beta = alloc(smem, Float32, T)
        rec = alloc(smem, Float32, RECORD_ROWS * 128, align=128)
        barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
        rec_barrier = smem.allocate(cutlass.Uint64, 8)
        if (thread >= 128) & (thread < 160):  # the gate warp (KdaVerifyRows)
            coords = (head, token, value_base, lane)
            self.gate_rows(args, shared_decay, gate_barrier, coords, request)
        else:
            taken = args[17][request]
            pending = taken - 1
            # deferred protocol: the state at slot 0, the records at slot 1
            slot, records = slots[request, 0], slots[request, 1]
            if (taken < 1) | (records <= 0):
                pending = Int32(0)
            if slot <= 0:
                cute.arch.griddepcontrol_wait()
                if (partition == 0) & (thread < 128):
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + thread] = BFloat16(0)
                cute.arch.griddepcontrol_launch_dependents()
            elif thread >= 160:  # spare warps: the cluster barrier only
                cute.arch.cluster_arrive_relaxed()
            else:
                self.setup(barrier, gate_barrier, partition, thread)
                page = state[Int64(slot), None, None, None]
                record_page = state[Int64(records), None, None, None]
                if (thread == 0) & (pending > 0):  # the head's records
                    cute.arch.mbarrier_init(rec_barrier, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(rec_barrier, RECORD_BYTES)
                    copy_g2s(rec.iterator, at(record_page, head, 0, 0),
                             RECORD_BYTES, rec_barrier)  # fmt: skip
                history = zeros((4, 4), Float32)  # [row 4 warp + r, column 4 lane + e]
                for r in cutlass.range_constexpr(4):
                    row = value_base + 4 * warp + r
                    values = self.load_state(at(page, head, row, 4 * lane))
                    for e in cutlass.range_constexpr(4):
                        history[r, e] = values[e]
                nw = norm_weight[thread]
                named_barrier(2, 128)  # rec_barrier armed
                if pending > 0:  # the fold, before the PDL wait
                    wait(rec_barrier, 0, scope="cta")
                    self.cols_fold(history, rec, pending, value_base, warp, lane)

                cute.arch.griddepcontrol_wait()
                for c in cutlass.range_constexpr(4):
                    index = thread + 128 * (c % 2)
                    t, column = index // 32, 4 * (index % 32)
                    target = (shared_q, shared_k)[c // 2].iterator + 128 * t + column
                    section = (c // 2) * dim + head * 128 + column
                    copy_g2s(target, at(prepared, token + t, section))
                cute.arch.cp_async_commit_group()
                v_token, v_row = thread // ROWS, thread % ROWS
                v_channel = V * dim + head * 128 + value_base + v_row
                gate_channel = GATE * dim + head * 128 + thread
                v_value = prepared[token + v_token, v_channel]
                out_gate = prepared[token + partition, gate_channel]
                beta = Float32(0)
                if thread < T:
                    beta = prepared[token + thread, BETA * dim + head]
                cute.arch.griddepcontrol_launch_dependents()
                shared_v[thread] = v_value
                if thread < T:
                    shared_beta[thread] = beta
                cute.arch.cp_async_wait_group(0)
                named_barrier(1, 128)  # q, k, v, betas
                wait(gate_barrier)

                buffers = (
                    cute.make_rmem_tensor((3, 4), Float32),
                    cute.make_rmem_tensor((3, 4), Float32),
                )
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, e] = Float32(0)  # (token -1's q: unused)
                sources = (shared_q, shared_k, shared_decay)
                self.cols_operands(buffers[0], sources, 0, lane)
                live = records > 0
                for pair in cutlass.range(T // 2, unroll=1):
                    for half in cutlass.range_constexpr(2):
                        t = 2 * pair + half
                        cur, other = buffers[half], buffers[1 - half]
                        v_t = [shared_v[t * ROWS + 4 * warp + r] for r in range(4)]
                        beta_t = shared_beta[t]
                        out_prev, v_news = self.cols_update_defer(
                            history, cur, other, sources, t, lane, v_t, beta_t
                        )
                        if (lane < 16) & (lane % 4 == 0) & (t > 0):
                            local_o[(t - 1) * ROWS + 4 * warp + lane // 4] = out_prev
                        if t == 0:  # the one state store: slot 0, after token 0
                            self.cols_checkpoint(
                                history, state, slot, head, value_base, lane
                            )
                        elif (lane == 0) & live:  # token t's record: corrections
                            for r in cutlass.range_constexpr(4):
                                record_page[
                                    head, V_ROW + t - 1, value_base + 4 * warp + r
                                ] = v_news[r]
                q_last = [buffers[(T + 1) % 2][0, e] for e in range(4)]
                dots = [
                    dot([history[r, e] for e in range(4)], q_last) for r in range(4)
                ]
                out_last, _ = scatter_reduce(dots, lane, 4)
                out_last = self.round_output(out_last)
                if lane % 8 == 0:  # lane 8 r: row r
                    local_o[(T - 1) * ROWS + 4 * warp + lane // 8] = out_last
                named_barrier(1, 128)
                cute.arch.cluster_wait()
                position = self.output_position(value_base + thread % ROWS)
                owner = thread // ROWS
                push_float(shared_o.iterator + position, local_o[thread], barrier, owner)
                operands = (thread, lane, nw, out_gate)
                self.cols_norm(shared_o, barrier, operands, args)
                # every CTA's outputs of this CTA's token arrived (each pushed after
                # its fold): the head's k / decay records may be overwritten
                if (partition == 0) & live:
                    for u in cutlass.range_constexpr(1, T):
                        record_page[head, u - 1, thread] = shared_k[128 * u + thread]
                        record_page[head, DECAY_ROW + u - 1, thread] = shared_decay[
                            128 * u + thread
                        ]
                    if thread == 0:
                        put_tag(record_page, head)

    @cute.jit
    def cols_fold(self, history, rec, pending, value_base, warp, lane):
        """The ``pending`` accepted tokens 1 .. pending into the warp's 4 rows (lane
        columns 4 lane + e) from the staged records, ``cols_update``'s instructions."""
        for i in cutlass.range(pending, unroll=1):
            k4 = load_f32x4(rec.iterator + i * 128 + 4 * lane, shared=True)
            d4 = load_f32x4(rec.iterator + (DECAY_ROW + i) * 128 + 4 * lane, shared=True)
            v = [rec[(V_ROW + i) * 128 + value_base + 4 * warp + r] for r in range(4)]
            for r in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(4):
                    history[r, e] *= d4[e]
            for r in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(4):
                    history[r, e] = fma(k4[e], v[r], history[r, e], ftz=self.fast_math)

    @cute.jit
    def cols_update_defer(self, history, ops, other, sources, t, lane, v_rows, beta):
        """``cols_update``, also returning the rows' corrections v_new."""
        kr = [ops[1, e] for e in range(4)]
        qr = [other[0, e] for e in range(4)]
        dots = [dot([history[r, e] for e in range(4)], qr) for r in range(4)]
        if t + 1 < self.tokens:
            self.cols_operands(other, sources, t + 1, lane)
        for r in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                history[r, e] *= ops[2, e]
        for r in cutlass.range_constexpr(4):
            dots.append(dot([history[r, e] for e in range(4)], kr))
        owned, _ = scatter_reduce(dots, lane, 8)  # lane 16 + 4 r: row r's key dot
        v_news = []
        for r in cutlass.range_constexpr(4):
            dot_k = cute.arch.shuffle_sync(owned, 16 + 4 * r)
            v_new = (v_rows[r] - dot_k) * beta
            v_news.append(v_new)
            for e in cutlass.range_constexpr(4):
                history[r, e] = fma(kr[e], v_new, history[r, e], ftz=self.fast_math)
        return self.round_output(owned), v_news
