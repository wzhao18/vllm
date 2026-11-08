# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyWide`` with the token loop's state layout regrouped so that half of
its warp reduction runs in registers and the FP32 state arithmetic runs on packed
``f32x2`` instructions: the same operations in the same orders, so the outputs, every
state checkpoint and the conv window are bit for bit ``KdaVerifyWide``'s (and so
``KdaSplit``'s verify).

Wide's thread (warp w, lane l) holds state rows 8 w .. 8 w + 7, columns 4 l .. 4 l + 3:
each row's query / key dot is the lane's 4-term FMA chain, then a 16-value
``scatter_reduce`` over lanes xor 16, 8, 4, 2, 1 and 8 key-dot broadcasts (24
shuffles per token), and every FP32 operation is a scalar instruction. Here lane l =
8 g + j of warp w holds rows 8 w + g and 8 w + g + 4 and, of each, the four column
chunks of Wide's lanes j, j + 8, j + 16 and j + 24 (columns 4 (j + 8 c) .. + 3, c =
0..3): the same 32 state values per thread, the same per-chunk FMA chains, and the
first two levels of Wide's lane tree, (c0 + c2) + (c1 + c3), are in-thread adds
(xor 16 pairs chunks c and c + 2, xor 8 then the two sums); xor 4, 2, 1 stay
shuffles over the 8 lanes of the row pair. Per token: 4 values (the previous
token's two query dots, this token's two key dots) reduced in 4 shuffles, 2 key-dot
broadcasts; the decay, the dots and the delta-rule update on ``mul.rn.f32x2`` /
``fma.rn.f32x2`` over the row pair (each lane of a packed instruction is the scalar
IEEE operation Wide issues). Prologue, gates, conv window and the gated norm are
Wide's."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    at,
    fma2,
    load_f32x4,
    mul2,
    zeros,
)
from .verify_wide import KdaVerifyWide

CHUNKS = 4  # column chunks per row per thread (Wide's lanes j, j + 8, j + 16, j + 24)


class KdaVerifyPacked(KdaVerifyWide):
    """``KdaVerifyWide`` (grid (1, heads, B), 512 threads) with the register-tree token
    loop; operands and launch arguments are ``KdaSplit``'s."""

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        thread, _, _ = cute.arch.thread_idx()
        _, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        group, j = lane // 8, lane % 8
        offset = accepted[request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, 128 * T) for _ in range(5)
        )
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        slot_rows = alloc(smem, Int32, T)
        pairs: cutlass.Constexpr = (T + 1) // 2
        ready = alloc(smem, cutlass.Int64, pairs, 8).iterator
        if slot <= 0:  # a padding request
            cute.arch.griddepcontrol_wait()
            if thread < 128:
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            # The thread's two state rows, four column chunks each (predecessor-
            # independent: this layer's previous step), before the PDL wait.
            page = state[Int64(slot), None, None, None]
            row0 = 8 * warp + group
            history = zeros((2, CHUNKS, 4), Float32)  # [row pair r, chunk c, e]
            for r in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(CHUNKS):
                    pointer = at(page, head, row0 + 4 * r, 4 * (j + 8 * c))
                    values = self.load_state(pointer)
                    for e in cutlass.range_constexpr(4):
                        history[r, c, e] = values[e]
            if thread < T:
                slot_rows[thread] = slots[request, thread]
            if thread == 0:
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_init(ready + p, 128 + 32 * 8)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()  # (before the PDL wait)
            nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
            own_gates = zeros(2, gate.element_type)
            if thread < 128:
                self.prologue(args, thread, token, offset, conv_slot, shared_q,
                              shared_k, shared_v, reductions, shared_beta,
                              own_gates, ready)  # fmt: skip
            elif thread < 128 + 32 * 8:
                self.gate(args, thread // 32 - 4, lane, head, token, shared_decay,
                          own_gates)  # fmt: skip
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_arrive(ready + p)
            else:
                cute.arch.griddepcontrol_wait()
                self.load_own_gates(args, token, thread // 64, thread % 64, own_gates)
            cute.arch.griddepcontrol_launch_dependents()

            for t in cutlass.range_constexpr(T):
                if cutlass.const_expr(t % 2 == 0):  # token pair t // 2's operands
                    cute.arch.mbarrier_wait(ready + t // 2, 0)
                self.packed_token(history, t, lane, group, j, row0, shared_q,
                                  shared_k, shared_decay, shared_v, shared_beta,
                                  shared_o, state, slot_rows, head)  # fmt: skip
            # The last token's query dots (the state after its update).
            q_last = self.column_chunks(shared_q, T - 1, j)
            sums = self.row_dots(history, q_last)
            # xor 4: lanes j < 4 keep row 0's, j >= 4 row 1's; then xor 2, 1
            upper = (j & 4) != 0
            send = sums[0] if upper else sums[1]
            keep = sums[1] if upper else sums[0]
            owned = keep + cute.arch.shuffle_sync_bfly(send, 4)
            owned += cute.arch.shuffle_sync_bfly(owned, 2)
            owned += cute.arch.shuffle_sync_bfly(owned, 1)
            if (j & 3) == 0:
                row = row0 + 4 * (j // 4)
                shared_o[(T - 1) * 128 + self.output_position(row)] = owned
            cute.arch.sync_threads()

            t, c, glane = thread // 64, thread % 64, thread % 32
            if t < T:
                self.gated_norm(t, c, glane, shared_o, own_gates, nw, eps, output,
                                token, head)  # fmt: skip

    @cute.jit
    def column_chunks(self, source, t, j):
        """Token t's [128] shared operand at the thread's 4 column chunks:
        [c][e] = source[128 t + 4 (j + 8 c) + e]."""
        result = cute.make_rmem_tensor((CHUNKS, 4), Float32)
        for c in cutlass.range_constexpr(CHUNKS):
            v4 = load_f32x4(source.iterator + 128 * t + 4 * (j + 8 * c), shared=True)
            for e in cutlass.range_constexpr(4):
                result[c, e] = v4[e]
        return result

    @cute.jit
    def row_dots(self, history, operand):
        """The two rows' dots with ``operand`` [c][e]: per chunk Wide's lane chain
        h0 q0, then fma(h e, q e, acc) for e = 1..3 (both rows in one packed
        instruction), then Wide's lane-tree levels xor 16 and xor 8 in registers:
        (c0 + c2) + (c1 + c3)."""
        chains = []
        for c in cutlass.range_constexpr(CHUNKS):
            x = operand[c, 0]
            acc = mul2((history[0, c, 0], history[1, c, 0]), (x, x))
            for e in cutlass.range_constexpr(1, 4):
                x = operand[c, e]
                pair = (history[0, c, e], history[1, c, e])
                acc = fma2(pair, (x, x), acc)
            chains.append(acc)
        return [
            (chains[0][r] + chains[2][r]) + (chains[1][r] + chains[3][r])
            for r in range(2)
        ]

    @cute.jit
    def packed_token(self, history, t, lane, group, j, row0, shared_q, shared_k,
                     shared_decay, shared_v, shared_beta, shared_o, state, slot_rows,
                     head):  # fmt: skip
        """Token t of the thread's row pair: the previous token's query dots (the
        state after its update), the decay, this token's key dots, the xor 4 / 2 / 1
        reduction of the four, the delta-rule update, the previous token's outputs
        and this token's checkpoint."""
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
        values += self.row_dots(history, k)  # [q row 0, q row 1, k row 0, k row 1]
        # Wide's scatter_reduce levels xor 4, 2, 1 over [q0, q1, k0, k1]: lane j
        # ends with value 2 (j >> 2 & 1) + (j >> 1 & 1)
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
            if (j & 5) == 0:  # j = 0: query row 0, j = 2: query row 1
                row = row0 + 4 * (j // 2)
                shared_o[(t - 1) * 128 + self.output_position(row)] = owned
        slot = slot_rows[t]
        page = state[Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(CHUNKS):
                values4 = [history[r, c, e] for e in range(4)]
                pointer = at(page, head, row0 + 4 * r, 4 * (j + 8 * c))
                self.store_state(pointer, values4, slot > 0)
