# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA RecoverSSM state commit: one CTA per (head, request, layer).

Folds the n accepted tokens of a verify window into the checkpoint:

    S_n = S_0 * e^{L_n} + sum_{t<n} u_t (k_t * e^{L_n - L_t})

(columns scale per key channel; ``u_t`` are the verify's corrections, ``k_t`` the
normalized recorded keys and ``L_t`` the log decay of tokens 0..t inclusive).
Thread c rebuilds key channel c's decays and scaled keys in shared memory; then
16 lanes per state row (8 keys each, one 16-byte load and store) stream the rows.
Layer tensors are addressed through the per-layer base-address tables of
``KDARecoverSSMCommitContext``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    bf16,
    bf16x2,
    launch,
    load_words,
    sigmoid,
    store_words,
    warp_sum,
)

K = 128
THREADS = 128
LANES_PER_ROW = 16
KEYS_PER_LANE = K // LANES_PER_ROW  # 8
ROWS_PER_PASS = THREADS // LANES_PER_ROW  # 8
PASSES = K // ROWS_PER_PASS  # 16


class KdaCommit:
    """Args: state / correction / key-gate base addresses [L], their slot strides
    [L] (elements), a_log [L, H], dt_bias [L, H, K], source slots [B], commit
    lengths [B], destination slots [B], lower bound, null slot, stream. Per slot the
    layouts are state [H, V, K] BF16, corrections [H, Q, V] FP32 and key / gate
    records [H, Q, 2K] BF16, all contiguous."""

    def __init__(self, heads: int, window: int):
        assert 1 <= window <= 8
        self.heads, self.window = heads, window

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads, args[8].shape[0], args[0].shape[0])
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        state_addrs, corr_addrs, kg_addrs = args[:3]
        state_strides, corr_strides, kg_strides, a_log, dt_bias = args[3:8]
        sources, commit_lens, destinations, lower_bound, null_slot = args[8:13]
        Q: cutlass.Constexpr = self.window
        tid, _, _ = cute.arch.thread_idx()
        head, request, layer = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        group, sub = tid // LANES_PER_ROW, tid % LANES_PER_ROW
        chunk = sub * KEYS_PER_LANE
        smem = cutlass.utils.SmemAllocator()
        scaled = alloc(smem, Float32, (Q + 1, K))  # k_t e^{L_n - L_t}; row Q: e^{L_n}
        norms = alloc(smem, Float32, (4, Q))
        corrections = alloc(smem, Float32, (Q, K))  # u_t, staged once
        kv = cute.make_rmem_tensor((Q,), Float32)
        logs = cute.make_rmem_tensor((Q,), Float32)
        coef = cute.make_rmem_tensor((Q + 1, KEYS_PER_LANE), Float32)
        u = cute.make_rmem_tensor((Q,), Float32)
        rows = cute.make_rmem_tensor((PASSES, 4), Uint32)  # the state, prefetched

        cute.arch.griddepcontrol_wait()
        n = commit_lens[request]
        source = Int64(sources[request])
        destination = Int64(destinations[request])
        live = (n > 0) & (source > Int64(null_slot)) & (destination > Int64(null_slot))
        if live:
            V: cutlass.Constexpr = K
            page = head * V * K
            state_ptr = cute.make_ptr(
                cutlass.BFloat16,
                state_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            corr_ptr = cute.make_ptr(
                Float32, corr_addrs[layer], cute.AddressSpace.gmem, assumed_align=16
            )
            kg_ptr = cute.make_ptr(
                cutlass.BFloat16,
                kg_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            stride = Int64(state_strides[layer])
            src = cute.make_tensor(
                state_ptr + source * stride + page,
                cute.make_layout((V, K), stride=(K, 1)),
            )
            dst = cute.make_tensor(
                state_ptr + destination * stride + page,
                cute.make_layout((V, K), stride=(K, 1)),
            )
            corr = cute.make_tensor(
                corr_ptr + source * Int64(corr_strides[layer]) + head * Q * V,
                cute.make_layout((Q, V), stride=(V, 1)),
            )
            kg = cute.make_tensor(
                kg_ptr + source * Int64(kg_strides[layer]) + head * Q * 2 * K,
                cute.make_layout((Q, 2 * K), stride=(2 * K, 1)),
            )

            # The state streams in while the coefficients are built.
            for it in cutlass.range_constexpr(PASSES):
                words = load_words(
                    src.iterator + (it * ROWS_PER_PASS + group) * K + chunk,
                    count=4,
                    word_type="b32",
                )
                for w in cutlass.range_constexpr(4):
                    rows[it, w] = words[w]

            # The corrections, staged in shared memory: read in the row loop, they
            # would wait behind the previous pass's stores.
            for t in cutlass.range_constexpr(Q):
                value = Float32(0)
                if t < n:
                    value = corr[t, tid]
                corrections[t, tid] = value

            # ---- Thread c: key channel c's decays and scaled keys. ----
            bias = dt_bias[layer, head, tid].to(Float32)
            a = cute.math.exp(a_log[layer, head].to(Float32))
            cum = Float32(0)
            for t in cutlass.range_constexpr(Q):
                kv[t] = Float32(0)
                if t < n:
                    kv[t] = kg[t, tid].to(Float32)
                    cum = cum + lower_bound * sigmoid(
                        a * (kg[t, K + tid].to(Float32) + bias), True
                    )
                logs[t] = cum
            for t in cutlass.range_constexpr(Q):
                s = warp_sum(kv[t] * kv[t])
                if lane == 0:
                    norms[warp, t] = s
            cute.arch.barrier()
            for t in cutlass.range_constexpr(Q):
                total = norms[0, t] + norms[1, t] + norms[2, t] + norms[3, t]
                k_n = kv[t] * cute.math.rsqrt(total + Float32(1e-6))
                # Token t's update decays by the tokens after it: e^{L_n - L_t}.
                scaled[t, tid] = k_n * cute.math.exp(cum - logs[t])
            scaled[Q, tid] = cute.math.exp(cum)
            cute.arch.barrier()

            # ---- Rows: 16 lanes per row, 8 keys per lane. ----
            for t in cutlass.range_constexpr(Q + 1):
                for j in cutlass.range_constexpr(KEYS_PER_LANE):
                    coef[t, j] = scaled[t, chunk + j]
            for it in cutlass.range_constexpr(PASSES):
                r = it * ROWS_PER_PASS + group
                for t in cutlass.range_constexpr(Q):
                    u[t] = corrections[t, r]
                out = []
                for w in cutlass.range_constexpr(4):
                    pair = []
                    for h in cutlass.range_constexpr(2):
                        j = 2 * w + h
                        value = bf16(rows[it, w], h == 1) * coef[Q, j]
                        for t in cutlass.range_constexpr(Q):
                            value = value + u[t] * coef[t, j]
                        pair.append(value)
                    out.append(bf16x2(pair[0], pair[1]))
                store_words(dst.iterator + r * K + chunk, tuple(out))
        cute.arch.griddepcontrol_launch_dependents()
