# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked KDA RecoverSSM verify: one CTA (4 warps) per (head, request).

With per-channel log decays ``L_t = g_1 + ... + g_t`` and normalized ``k_t``, ``q_t``
(``q_t`` carries the 1/sqrt(K) scale), the verify window satisfies

    u_t = beta_t * (v_t - S_0 p_t - sum_{i<t} c_it u_i),   p_t = e^{L_t} k_t
    o_t = S_0 r_t + sum_{i<=t} d_it u_i,                    r_t = e^{L_t} q_t

with c_it = <w_i, p_t>, d_it = <w_i, r_t> and w_i = e^{-L_i} k_i.

1. Thread c owns key channel c: it builds p_t, r_t (as BF16 high + low halves, an
   FP32-accurate operand for two BF16 MMAs) and w_i in shared memory.
2. The coupling scalars: two threads per (probe, i) pair, half the keys each.
3. Warp w projects state rows [32 w, 32 w + 32) on the 16 (padded) probes with
   m16n8k16 BF16 MMAs: the BF16 state rows are exact A fragments read straight
   from global memory (issued before the PDL wait: the state is written by the
   previous step's commit, not the preceding kernel), FP32 accumulators. The MMA
   k order is a relabeling of the key channels, the same for both operands: thread
   ``l`` of a fragment group holds channels 32 j + 8 l + [0, 8) for j < 4, so each
   16-byte load of a fragment group reads one contiguous 64-byte run (state rows
   from global memory, probe columns from padded, bank-conflict-free rows).
4. Thread r solves row r's triangular system and stores row r of every output and
   correction, so those stores are coalesced across the CTA.

Outputs and records (``u_t``, raw key and gate) match ``kda_recoverssm_verify``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    at,
    fma,
    launch,
    load_words,
    mma,
    sigmoid,
    warp_sum,
)

K = 128
THREADS = 128  # one per key channel, one per state row
PROBES = 16  # 2 * Q <= 16 probe columns, zero padded
PROBE_PITCH = K + 32  # BF16 per probe row: adjacent columns 64 bytes apart in banks
F32_PITCH = K + 4
ROWS_PER_WARP = 32


def load_x4(pointer):
    return load_words(pointer, count=4, word_type="b32")


def smem_x4(pointer):
    return load_words(pointer, count=4, shared=True, mutable=True, word_type="b32")


def smem_f32x4(pointer):
    return load_words(
        pointer,
        count=4,
        shared=True,
        mutable=True,
        result_type=Float32,
        word_type="f32",
    )


@cute.jit
def bf16_split(value: Float32):
    """BF16 (high, low) with high + low ~= value to ~2^-17."""
    high = value.to(cutlass.BFloat16)
    low = (value - high.to(Float32)).to(cutlass.BFloat16)
    return high, low


class KdaChunkedVerify:
    """Args: qkv [T, 3HK], raw_g [T, HK], raw_beta [T, H], a_log [H], dt_bias [HK],
    state [slots, H, V, K], corr [slots, H, Q, V], kg [slots, H, Q, 2K], out [T, HV],
    query_start [B + 1], request slots [B], lower_bound, null slot, stream."""

    def __init__(self, heads: int, window: int):
        assert 1 <= window <= 8
        self.heads, self.window = heads, window

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (self.heads, args[10].shape[0], 1), THREADS)

    @cute.kernel
    def kernel(self, args):
        qkv, raw_g, raw_beta, a_log, dt_bias, state, corr, kg, out = args[:9]
        query_start, slots, lower_bound, null_slot = args[9:13]
        Q: cutlass.Constexpr = self.window
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        head, request, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        quad, quad_lane = lane // 4, lane % 4  # MMA fragment group / thread in group
        smem = cutlass.utils.SmemAllocator()
        # Probe columns as BF16 [column, key] high and low halves; w_i in FP32.
        probe_hi = alloc(smem, cutlass.BFloat16, (PROBES, PROBE_PITCH))
        probe_lo = alloc(smem, cutlass.BFloat16, (PROBES, PROBE_PITCH))
        probe_f32 = alloc(smem, Float32, (2 * Q, F32_PITCH))
        w_f32 = alloc(smem, Float32, (Q, F32_PITCH))
        proj = alloc(smem, Float32, (K, PROBES + 1))  # +1: conflict-free rows
        coef = alloc(smem, Float32, (2 * Q * Q,))  # c[t * Q + i] then d[...]
        betas = alloc(smem, Float32, (8,))
        norms = alloc(smem, Float32, (4, 2 * Q))
        qv = cute.make_rmem_tensor((Q,), Float32)
        kv = cute.make_rmem_tensor((Q,), Float32)
        logs = cute.make_rmem_tensor((Q,), Float32)
        g_raws = cute.make_rmem_tensor((Q,), Float32)
        u = cute.make_rmem_tensor((Q,), Float32)
        # This lane's channels 32 j + 8 l + [0, 8) of rows 32 w + 16 m + quad (+ 8):
        # word 2 k + h is the k-tile k, half h BF16 pair of the permuted k order.
        a_words = cute.make_rmem_tensor((2, 2, 16), Uint32)  # [m tile, +8 row, word]
        b_words = cute.make_rmem_tensor((2, 16), Uint32)  # [high/low, word]

        slot = Int64(slots[request])
        dead = slot <= Int64(null_slot)
        base = quad_lane * 8
        if not dead:
            for m in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(2):
                    row = warp * ROWS_PER_WARP + m * 16 + e * 8 + quad
                    for j in cutlass.range_constexpr(4):
                        words = load_x4(at(state, slot, head, row, 32 * j + base))
                        for w in cutlass.range_constexpr(4):
                            a_words[m, e, 4 * j + w] = words[w]

        cute.arch.griddepcontrol_wait()
        bos = query_start[request]
        qlen = query_start[request + 1] - bos
        if dead:
            for t in cutlass.range_constexpr(Q):
                if t < qlen:
                    out[bos + t, head * K + tid] = out.element_type(0)
        else:
            # ---- 1. Probes: thread c owns key channel c. ----
            for t in cutlass.range_constexpr(Q):
                row = bos + cutlass.min(t, qlen - 1)
                raw_q = qkv[row, head * K + tid]
                raw_k = qkv[row, H * K + head * K + tid]
                g_raw = raw_g[row, head * K + tid]
                qv[t], kv[t] = raw_q.to(Float32), raw_k.to(Float32)
                g_raws[t] = g_raw.to(Float32)
                if t < qlen:
                    kg[slot, head, t, tid] = raw_k
                    kg[slot, head, t, K + tid] = g_raw
            bias = dt_bias[head * K + tid].to(Float32)
            a = cute.math.exp(a_log[head].to(Float32))
            cum = Float32(0)
            for t in cutlass.range_constexpr(Q):
                if t < qlen:
                    cum = cum + lower_bound * sigmoid(a * (g_raws[t] + bias), True)
                else:
                    qv[t], kv[t] = Float32(0), Float32(0)
                logs[t] = cum
            for t in cutlass.range_constexpr(Q):
                sq = warp_sum(qv[t] * qv[t])
                sk = warp_sum(kv[t] * kv[t])
                if lane == 0:
                    norms[warp, 2 * t] = sq
                    norms[warp, 2 * t + 1] = sk
            if tid < Q:
                beta = Float32(0)
                if tid < qlen:
                    beta = sigmoid(raw_beta[bos + tid, head].to(Float32), True)
                betas[tid] = beta
            cute.arch.barrier()
            for t in cutlass.range_constexpr(Q):
                sq = (
                    norms[0, 2 * t]
                    + norms[1, 2 * t]
                    + norms[2, 2 * t]
                    + norms[3, 2 * t]
                )
                sk = (
                    norms[0, 2 * t + 1]
                    + norms[1, 2 * t + 1]
                    + norms[2, 2 * t + 1]
                    + norms[3, 2 * t + 1]
                )
                q_n = qv[t] * cute.math.rsqrt(sq + Float32(1e-6)) * Float32(K**-0.5)
                k_n = kv[t] * cute.math.rsqrt(sk + Float32(1e-6))
                decay = cute.math.exp(logs[t])
                p_t, r_t = decay * k_n, decay * q_n
                probe_f32[t, tid] = p_t
                probe_f32[Q + t, tid] = r_t
                w_f32[t, tid] = cute.math.exp(-logs[t]) * k_n
                probe_hi[t, tid], probe_lo[t, tid] = bf16_split(p_t)
                probe_hi[Q + t, tid], probe_lo[Q + t, tid] = bf16_split(r_t)
            for p in cutlass.range_constexpr(2 * Q, PROBES):
                probe_hi[p, tid] = cutlass.BFloat16(0)
                probe_lo[p, tid] = cutlass.BFloat16(0)
            cute.arch.barrier()

            # ---- 2. Coupling scalars: two threads per (probe, i) pair. ----
            pair, half = tid // 2, tid % 2
            p, i = cutlass.min(pair // Q, 2 * Q - 1), pair % Q
            acc = Float32(0)
            for c in cutlass.range_constexpr(0, 64, 4):
                wv = smem_f32x4(at(w_f32, i, half * 64 + c))
                pv = smem_f32x4(at(probe_f32, p, half * 64 + c))
                for j in cutlass.range_constexpr(4):
                    acc = fma(wv[j], pv[j], acc)
            acc += cute.arch.shuffle_sync_bfly(acc, 1)
            t = p % Q
            # c_it needs i < t, d_it needs i <= t (p = t, or Q + t).
            if (half == 0) & (pair < 2 * Q * Q):
                keep = (i < t) | ((p >= Q) & (i == t))
                value = Float32(0)
                if keep:
                    value = acc
                coef[(p // Q) * Q * Q + t * Q + i] = value
            cute.arch.barrier()

            # ---- 3. Projections on the tensor cores. ----
            for n in cutlass.range_constexpr(2):
                for h, probe in enumerate((probe_hi, probe_lo)):
                    for j in cutlass.range_constexpr(4):
                        words = smem_x4(at(probe, n * 8 + quad, 32 * j + base))
                        for w in cutlass.range_constexpr(4):
                            b_words[h, 4 * j + w] = words[w]
                for m in cutlass.range_constexpr(2):
                    acc4 = (Float32(0), Float32(0), Float32(0), Float32(0))
                    for kt in cutlass.range_constexpr(8):
                        a4 = (
                            a_words[m, 0, 2 * kt],
                            a_words[m, 1, 2 * kt],
                            a_words[m, 0, 2 * kt + 1],
                            a_words[m, 1, 2 * kt + 1],
                        )
                        for h in cutlass.range_constexpr(2):
                            b2 = (b_words[h, 2 * kt], b_words[h, 2 * kt + 1])
                            acc4 = mma(a4, b2, acc4, bf16=True)
                    row = warp * ROWS_PER_WARP + m * 16 + quad
                    col = n * 8 + quad_lane * 2
                    proj[row, col], proj[row, col + 1] = acc4[0], acc4[1]
                    proj[row + 8, col], proj[row + 8, col + 1] = acc4[2], acc4[3]
            cute.arch.barrier()

            # ---- 4. Thread r solves row r. ----
            r = tid
            for t in cutlass.range_constexpr(Q):
                row = bos + cutlass.min(t, qlen - 1)
                acc = qkv[row, 2 * H * K + head * K + r].to(Float32) - proj[r, t]
                for i in cutlass.range_constexpr(t):
                    acc = acc - coef[t * Q + i] * u[i]
                u[t] = betas[t] * acc
                o = proj[r, Q + t]
                for i in cutlass.range_constexpr(t + 1):
                    o = fma(coef[Q * Q + t * Q + i], u[i], o)
                if t < qlen:
                    out[bos + t, head * K + r] = o.to(out.element_type)
                    corr[slot, head, t, r] = u[t]
        cute.arch.griddepcontrol_launch_dependents()
