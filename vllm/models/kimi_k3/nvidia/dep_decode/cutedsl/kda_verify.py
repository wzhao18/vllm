# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked KDA RecoverSSM verify: one warp per (head, request), four heads per CTA.

With per-channel log decays ``L_t = g_1 + ... + g_t`` and normalized ``k_t``, ``q_t``
(``q_t`` carries the 1/sqrt(K) scale), the verify window satisfies

    u_t = beta_t * (v_t - S_0 p_t - sum_{i<t} c_it u_i),   p_t = e^{L_t} k_t
    o_t = S_0 r_t + sum_{i<=t} d_it u_i,                    r_t = e^{L_t} q_t

with c_it = <w_i, p_t>, d_it = <w_i, r_t> and w_i = e^{-L_i} k_i.

The probe columns are p_0..p_{Q-1}, r_0..r_{Q-1} (zero padded to 8 or 16). Lane
``l`` of a warp is thread ``l % 4`` of fragment group ``l // 4`` of the m16n8k16
MMAs, and owns key channels ``32 j + 8 (l % 4) + [0, 8)`` (j < 4): the k order of
the MMAs is a relabeling of the key channels, the same for both operands.

1. The state streams through a per-lane ``cp.async`` ring, 16 rows per stage
   (issued before the PDL wait: the state is written by the previous step's
   commit, not the preceding kernel). Each lane copies exactly the 16-byte pieces
   it later reads as A fragments, so the ring needs no barrier.
2. Fragment group ``g`` computes probe columns ``g`` and ``8 + g`` on its lanes'
   channels (norms are group reductions), which are exactly its lanes' B
   fragments (BF16 high + low halves: an FP32-accurate operand for two MMAs), and
   their coupling scalars ``c``, ``d`` (dots of the probe with ``w_i``).
3. Per 16-row tile: the MMAs, then lanes 0-15 each solve one state row's
   triangular system and store its outputs and corrections.

Outputs and records (``u_t``, raw key and gate) match ``kda_recoverssm_verify``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    copy_g2s,
    fp8_bits,
    fma,
    launch,
    load_words,
    mma,
    mx_exponent,
    mx_rescale,
    sigmoid,
    store_words,
    warp_sum,
)

K = 128
DEFAULT_WARPS = 2  # heads per CTA (one warp each)
TILES = K // 16  # 16-row state tiles
CHUNKS = 8  # 16-byte A pieces per lane per tile: 2 rows x 4 key runs
DEFAULT_STAGES = 2


@asm()
def cp_async_commit():
    return "cp.async.commit_group;", "~{memory}", ()


@asm()
def cp_async_wait(pending):
    return f"cp.async.wait_group {pending};", "~{memory}", ()


@cute.jit
def store_scale(scales, row, block, exponent, heads: cutlass.Constexpr):
    """UE8M0 ``exponent`` of MX block ``block`` of activation row ``row``, in the
    F8_128x4 swizzled layout of a [T, heads * 128] MXFP8 GEMM input."""
    k_tiles = heads * K // 128
    offset = (
        (((row // 128) * k_tiles + block // 4) * 32 + row % 32) * 4 + (row // 32) % 4
    ) * 4 + block % 4
    scales[offset] = Uint8(exponent)


class KdaChunkedVerify:
    """Args: qkv [T, 3HK], raw_g [T, HK], raw_beta [T, H], a_log [H], dt_bias [HK],
    state [slots, H, V, K], corr [slots, H, Q, V], kg [slots, H, Q, 2K], out [T, HV],
    query_start [B + 1], request slots [B], lower_bound, null slot, stream."""

    def __init__(
        self,
        heads: int,
        window: int,
        stages: int = DEFAULT_STAGES,
        conv: bool = False,
        norm: bool = False,
        warps: int = DEFAULT_WARPS,
        conv_dim_first: bool = False,
    ):
        assert 1 <= window <= 8 and heads % warps == 0
        self.warps = warps
        self.heads, self.window, self.stages = heads, window, stages
        # Fused short convolution: qkv is the raw projection, and three more
        # tensors follow the request slots: conv state [slots, Q + 2, 3HK] (token
        # major, or [slots, 3HK, Q + 2] with ``conv_dim_first``), conv weight
        # [3HK, 4] and accepted tokens [B].
        self.conv = conv
        self.conv_dim_first = conv_dim_first
        # Fused output epilogue: KDA's gated RMSNorm (sigmoid gate) and MXFP8
        # quantization of the result, written instead of ``out``. Four more tensors
        # follow (after the conv tensors): g_proj [T, HK], norm weight [K], E4M3
        # values [T, HK] (as bytes) and swizzled scales; the norm eps is the last
        # scalar.
        self.norm = norm

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads // self.warps, args[10].shape[0], 1)
        launch(self.kernel, args, grid, 32 * self.warps)

    @cute.jit
    def issue(self, state, slot, head, tile, ring, warp, lane):
        """``cp.async`` this lane's A pieces of state rows [16 tile, 16 tile + 16)."""
        g, t4 = lane // 4, lane % 4
        stage = tile % self.stages
        for e in cutlass.range_constexpr(2):
            row = tile * 16 + e * 8 + g
            for j in cutlass.range_constexpr(4):
                copy_g2s(
                    at(ring, warp, stage, e * 4 + j, lane, 0),
                    at(state, slot, head, row, 32 * j + 8 * t4),
                )

    @cute.kernel
    def kernel(self, args):
        qkv, raw_g, raw_beta, a_log, dt_bias, state, corr, kg, out = args[:9]
        query_start, slots = args[9:11]
        if cutlass.const_expr(self.norm):
            lower_bound, null_slot, norm_eps = args[-3], args[-2], args[-1]
            extra = 14 if self.conv else 11
            g_proj, norm_w, q_out, s_out = args[extra : extra + 4]
        else:
            lower_bound, null_slot = args[-2], args[-1]
        Q: cutlass.Constexpr = self.window
        H: cutlass.Constexpr = self.heads
        S: cutlass.Constexpr = self.stages
        NT: cutlass.Constexpr = 2 if 2 * self.window > 8 else 1  # probe n tiles
        tid, _, _ = cute.arch.thread_idx()
        head_group, request, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        WARPS: cutlass.Constexpr = self.warps
        head = head_group * WARPS + warp
        g, t4 = lane // 4, lane % 4
        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, Uint32, (WARPS, S, CHUNKS, 32, 4))
        # c[t * Q + i], then d[...], then beta[t].
        coef = alloc(smem, Float32, (WARPS, 2 * Q * Q + 8))
        vbuf = alloc(smem, cutlass.BFloat16, (WARPS, Q, K))
        if cutlass.const_expr(self.norm):
            gbuf = alloc(smem, cutlass.BFloat16, (WARPS, Q, K))  # g_proj
        # Probe columns p_0.., r_0.. in FP32 (rows skewed by 4 banks); once the B
        # fragments are loaded, the same space holds every row's projections.
        PITCH: cutlass.Constexpr = 2 * Q + 1  # odd: conflict-free rows
        span = max(2 * Q * (K + 4), K * PITCH)
        scratch = alloc(smem, Float32, (WARPS, span))
        probes = cute.make_tensor(
            scratch.iterator,
            cute.make_layout((WARPS, 2 * Q, K + 4), stride=(span, K + 4, 1)),
        )
        proj = cute.make_tensor(
            scratch.iterator,
            cute.make_layout((WARPS, K, PITCH), stride=(span, PITCH, 1)),
        )

        slot = Int64(slots[request])
        dead = slot <= Int64(null_slot)
        if not dead:
            for tile in cutlass.range_constexpr(S):
                self.issue(state, slot, head, tile, ring, warp, lane)
                cp_async_commit()

        cute.arch.griddepcontrol_wait()
        bos = query_start[request]
        qlen = query_start[request + 1] - bos
        if dead:
            for t in cutlass.range_constexpr(Q):
                if t < qlen:
                    for i in cutlass.range_constexpr(4):
                        if cutlass.const_expr(self.norm):
                            q_out[bos + t, head * K + lane * 4 + i] = Uint8(0)
                        else:
                            out[bos + t, head * K + lane * 4 + i] = out.element_type(0)
                    if cutlass.const_expr(self.norm):
                        if lane < 4:
                            store_scale(s_out, bos + t, head * 4 + lane, Int32(0), H)
        elif qlen <= 0:
            cp_async_wait(0)
        else:
            # ---- Activations: every load of the window is issued before anything
            # consumes one (a dependent store would serialize their latencies). ----
            c0 = lane * 4  # lane l owns channels 4 l + [0, 4) of q, k, v and g
            act = cute.make_rmem_tensor((Q, 4, 2), Uint32)  # q, k, v, g words
            for t in cutlass.range_constexpr(Q):
                row = bos + cutlass.min(t, qlen - 1)
                for x in cutlass.range_constexpr(4):
                    if cutlass.const_expr(x < 3):
                        w2 = load_words(
                            at(qkv, row, x * H * K + head * K + c0),
                            count=2,
                            mutable=True,
                        )
                    else:
                        w2 = load_words(
                            at(raw_g, row, head * K + c0), count=2, mutable=True
                        )
                    act[t, x, 0], act[t, x, 1] = w2[0], w2[1]
            if cutlass.const_expr(self.conv):
                conv_state, conv_w, accepted = args[11:14]
                hist = cute.make_rmem_tensor((3, 3, 2), Uint32)  # [q/k/v, row, word]
                taps = cute.make_rmem_tensor((3, 4, 2), Uint32)  # [q/k/v, chan, word]
                first = accepted[request] - 1
                for x in cutlass.range_constexpr(3):
                    channel = x * H * K + head * K + c0
                    for i in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.conv_dim_first):
                            # Channel-major state: 4 scalar loads, repacked.
                            v4 = [
                                conv_state[slot, channel + e, first + i].to(Float32)
                                for e in range(4)
                            ]
                            hist[x, i, 0] = bf16x2(v4[0], v4[1])
                            hist[x, i, 1] = bf16x2(v4[2], v4[3])
                        else:
                            w2 = load_words(
                                at(conv_state, slot, first + i, channel),
                                count=2,
                                mutable=True,
                            )
                            hist[x, i, 0], hist[x, i, 1] = w2[0], w2[1]
                    for e in cutlass.range_constexpr(4):
                        w2 = load_words(at(conv_w, channel + e, 0), count=2)
                        taps[x, e, 0], taps[x, e, 1] = w2[0], w2[1]
            if cutlass.const_expr(self.norm):
                gw = cute.make_rmem_tensor((Q, 2), Uint32)
                for t in cutlass.range_constexpr(Q):
                    row = bos + cutlass.min(t, qlen - 1)
                    w2 = load_words(
                        at(g_proj, row, head * K + c0), count=2, mutable=True
                    )
                    gw[t, 0], gw[t, 1] = w2[0], w2[1]
                nw = cute.make_rmem_tensor((4,), Float32)
                for m in cutlass.range_constexpr(4):
                    nw[m] = norm_w[lane + 32 * m].to(Float32)
            beta = Float32(0)
            if lane < qlen:
                beta = sigmoid(raw_beta[bos + lane, head].to(Float32), True)
            a = cute.math.exp(a_log[head].to(Float32))
            bias = cute.make_rmem_tensor((4,), Float32)
            for e in cutlass.range_constexpr(4):
                bias[e] = dt_bias[head * K + c0 + e].to(Float32)
            if cutlass.const_expr(self.conv):
                # The rolled conv state: the last 2 history rows, then the raw
                # tokens (causal_conv1d_update's speculative layout).
                for x in cutlass.range_constexpr(3):
                    channel = x * H * K + head * K + c0
                    if cutlass.const_expr(self.conv_dim_first):
                        for e in cutlass.range_constexpr(4):
                            for i in cutlass.range_constexpr(2):
                                conv_state[slot, channel + e, i] = bf16(
                                    hist[x, i + 1, e // 2], e % 2
                                ).to(cutlass.BFloat16)
                            for t in cutlass.range_constexpr(Q):
                                if t < qlen:
                                    conv_state[slot, channel + e, 2 + t] = bf16(
                                        act[t, x, e // 2], e % 2
                                    ).to(cutlass.BFloat16)
                    else:
                        for i in cutlass.range_constexpr(2):
                            store_words(
                                at(conv_state, slot, i, channel),
                                (hist[x, i + 1, 0], hist[x, i + 1, 1]),
                            )
                        for t in cutlass.range_constexpr(Q):
                            if t < qlen:
                                store_words(
                                    at(conv_state, slot, 2 + t, channel),
                                    (act[t, x, 0], act[t, x, 1]),
                                )
                # Width-4 convolution and SiLU; products round to BF16 and the
                # output to BF16, as the stock kernel's.
                for x in cutlass.range_constexpr(3):
                    for pair in cutlass.range_constexpr(2):
                        res = cute.make_rmem_tensor((Q, 2), Float32)
                        for h in cutlass.range_constexpr(2):
                            e = 2 * pair + h
                            w = [bf16(taps[x, e, j // 2], j % 2) for j in range(4)]
                            win = [bf16(hist[x, i, pair], h) for i in range(3)]
                            for t in cutlass.range_constexpr(Q):
                                win.append(bf16(act[t, x, pair], h))
                                acc = Float32(0)
                                for j in cutlass.range_constexpr(4):
                                    prod = (win[t + j] * w[j]).to(cutlass.BFloat16)
                                    acc = acc + prod.to(Float32)
                                res[t, h] = acc / (Float32(1) + cute.math.exp(-acc))
                        for t in cutlass.range_constexpr(Q):
                            act[t, x, pair] = bf16x2(res[t, 0], res[t, 1])
            for t in cutlass.range_constexpr(Q):
                store_words(
                    at(vbuf, warp, t, c0), (act[t, 2, 0], act[t, 2, 1]), shared=True
                )
            if cutlass.const_expr(self.norm):
                for t in cutlass.range_constexpr(Q):
                    store_words(
                        at(gbuf, warp, t, c0), (gw[t, 0], gw[t, 1]), shared=True
                    )
            if lane < Q:
                coef[warp, 2 * Q * Q + lane] = beta

            # ---- Probes, phase 1. ----
            qx = cute.make_rmem_tensor((Q, 4), Float32)
            kx = cute.make_rmem_tensor((Q, 4), Float32)
            logs = cute.make_rmem_tensor((Q, 4), Float32)
            sums = cute.make_rmem_tensor((2 * Q,), Float32)  # |q_t|^2, |k_t|^2
            cum = cute.make_rmem_tensor((4,), Float32)
            cum.fill(0)
            for t in cutlass.range_constexpr(Q):
                qw = (act[t, 0, 0], act[t, 0, 1])
                kw = (act[t, 1, 0], act[t, 1, 1])
                gw = (act[t, 3, 0], act[t, 3, 1])
                if t < qlen:
                    store_words(at(kg, slot, head, t, c0), kw)
                    store_words(at(kg, slot, head, t, K + c0), gw)
                sq, sk = Float32(0), Float32(0)
                for e in cutlass.range_constexpr(4):
                    qv, kv = Float32(0), Float32(0)
                    if t < qlen:
                        qv, kv = bf16(qw[e // 2], e % 2), bf16(kw[e // 2], e % 2)
                        gate = sigmoid(a * (bf16(gw[e // 2], e % 2) + bias[e]), True)
                        cum[e] = cum[e] + lower_bound * gate
                    qx[t, e], kx[t, e], logs[t, e] = qv, kv, cum[e]
                    sq, sk = fma(qv, qv, sq), fma(kv, kv, sk)
                sums[2 * t], sums[2 * t + 1] = sq, sk
            for v in cutlass.range_constexpr(2 * Q):
                sums[v] = warp_sum(sums[v])
            for t in cutlass.range_constexpr(Q):
                q_scale = cute.math.rsqrt(sums[2 * t] + Float32(1e-6)) * Float32(
                    K**-0.5
                )
                k_scale = cute.math.rsqrt(sums[2 * t + 1] + Float32(1e-6))
                for e in cutlass.range_constexpr(4):
                    decay = cute.math.exp(logs[t, e])
                    k_n = kx[t, e] * k_scale
                    p_t, r_t = decay * k_n, decay * (qx[t, e] * q_scale)
                    probes[warp, t, c0 + e] = p_t
                    probes[warp, Q + t, c0 + e] = r_t
                    qx[t, e] = r_t  # r_t from here on
                    kx[t, e] = cute.math.exp(-logs[t, e]) * k_n  # w_t
                    logs[t, e] = p_t
            # Coupling scalars: c_it = <w_i, p_t> (i < t), d_it = <w_i, r_t> (i <= t).
            for t in cutlass.range_constexpr(Q):
                for i in cutlass.range_constexpr(t + 1):
                    for kind in cutlass.range_constexpr(2):
                        if cutlass.const_expr(kind == 1 or i < t):
                            dot = Float32(0)
                            for e in cutlass.range_constexpr(4):
                                probe = logs[t, e] if kind == 0 else qx[t, e]
                                dot = fma(kx[i, e], probe, dot)
                            dot = warp_sum(dot)
                            if lane == 0:
                                coef[warp, kind * Q * Q + t * Q + i] = dot
            cute.arch.sync_warp()

            # ---- Probes, phase 2: group g's B fragments, columns g and 8 + g. ----
            b_words = cute.make_rmem_tensor((NT, 2, 16), Uint32)
            for n in cutlass.range_constexpr(NT):
                col = n * 8 + g
                for j in cutlass.range_constexpr(4):
                    vals = cute.make_rmem_tensor((8,), Float32)
                    vals.fill(0)
                    if col < 2 * Q:
                        for h in cutlass.range_constexpr(2):
                            f4 = load_words(
                                at(probes, warp, col, 32 * j + 8 * t4 + 4 * h),
                                count=4,
                                shared=True,
                                mutable=True,
                                result_type=Float32,
                                word_type="f32",
                            )
                            for e in cutlass.range_constexpr(4):
                                vals[4 * h + e] = f4[e]
                    for w in cutlass.range_constexpr(4):
                        his, los = [], []
                        for h in cutlass.range_constexpr(2):
                            value = vals[2 * w + h]
                            high = value.to(cutlass.BFloat16).to(Float32)
                            his.append(high)
                            los.append(value - high)
                        b_words[n, 0, 4 * j + w] = bf16x2(his[0], his[1])
                        b_words[n, 1, 4 * j + w] = bf16x2(los[0], los[1])

            cute.arch.sync_warp()  # every lane's B fragments loaded: probes are free

            # ---- Per 16-row tile: MMAs on the streamed state. ----
            a_words = cute.make_rmem_tensor((2, 16), Uint32)  # [+8 row, word]
            # A runtime loop: unrolled, the kernel outgrows the instruction cache.
            for tile in cutlass.range(TILES, unroll=1):
                stage = tile % S
                cp_async_wait(S - 1)
                for e in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(4):
                        words4 = load_words(
                            at(ring, warp, stage, e * 4 + j, lane, 0),
                            count=4,
                            shared=True,
                            mutable=True,
                        )
                        for w in cutlass.range_constexpr(4):
                            a_words[e, 4 * j + w] = words4[w]
                if tile + S < TILES:
                    self.issue(state, slot, head, tile + S, ring, warp, lane)
                cp_async_commit()
                for n in cutlass.range_constexpr(NT):
                    # Four independent accumulator chains (k parity x high / low).
                    zero = (Float32(0), Float32(0), Float32(0), Float32(0))
                    parts = [zero, zero, zero, zero]
                    for kt in cutlass.range_constexpr(8):
                        a4 = (
                            a_words[0, 2 * kt],
                            a_words[1, 2 * kt],
                            a_words[0, 2 * kt + 1],
                            a_words[1, 2 * kt + 1],
                        )
                        for h in cutlass.range_constexpr(2):
                            b2 = (b_words[n, h, 2 * kt], b_words[n, h, 2 * kt + 1])
                            which = 2 * (kt % 2) + h
                            parts[which] = mma(a4, b2, parts[which], bf16=True)
                    acc4 = tuple(
                        (parts[0][v] + parts[1][v]) + (parts[2][v] + parts[3][v])
                        for v in range(4)
                    )
                    for e in cutlass.range_constexpr(2):
                        row = tile * 16 + e * 8 + g
                        for h in cutlass.range_constexpr(2):
                            col = n * 8 + t4 * 2 + h
                            if col < 2 * Q:
                                proj[warp, row, col] = acc4[2 * e + h]
            cute.arch.sync_warp()

            # ---- Solve: lane l, rows l + 32 m (interleaved for ILP). ----
            u = cute.make_rmem_tensor((4, Q), Float32)
            for t in cutlass.range_constexpr(Q):
                for m in cutlass.range_constexpr(4):
                    r = lane + 32 * m
                    acc = vbuf[warp, t, r].to(Float32) - proj[warp, r, t]
                    for i in cutlass.range_constexpr(t):
                        acc = acc - coef[warp, t * Q + i] * u[m, i]
                    u[m, t] = coef[warp, 2 * Q * Q + t] * acc
                    o = proj[warp, r, Q + t]
                    for i in cutlass.range_constexpr(t + 1):
                        o = fma(coef[warp, Q * Q + t * Q + i], u[m, i], o)
                    if cutlass.const_expr(self.norm):
                        # v_t is consumed: the slot holds the BF16 output.
                        vbuf[warp, t, r] = o.to(cutlass.BFloat16)
                    if t < qlen:
                        if cutlass.const_expr(not self.norm):
                            out[bos + t, head * K + r] = o.to(out.element_type)
                        corr[slot, head, t, r] = u[m, t]
            if cutlass.const_expr(self.norm):
                # ---- Gated RMSNorm over the head (the stock kernel's order and
                # BF16 rounding), then one MX block per (token, 32 rows): lane l
                # holds row l of each block. ----
                cute.arch.sync_warp()
                for t in cutlass.range(qlen, unroll=1):
                    x = cute.make_rmem_tensor((4,), Float32)
                    ss = Float32(0)
                    for m in cutlass.range_constexpr(4):
                        x[m] = vbuf[warp, t, lane + 32 * m].to(Float32)
                        ss = fma(x[m], x[m], ss)
                    ss = warp_sum(ss)
                    rstd = Float32(1) / cute.math.sqrt(ss / Float32(K) + norm_eps)
                    for m in cutlass.range_constexpr(4):
                        g_raw = gbuf[warp, t, lane + 32 * m].to(Float32)
                        gate = Float32(1) / (Float32(1) + cute.math.exp(-g_raw))
                        y = ((x[m] * rstd) * nw[m]) * gate
                        y = y.to(cutlass.BFloat16).to(Float32)
                        amax = abs(y)
                        for shift in cutlass.range_constexpr(5):
                            amax = cute.arch.fmax(
                                amax, cute.arch.shuffle_sync_bfly(amax, 1 << shift)
                            )
                        exponent = mx_exponent(amax)
                        value = y * mx_rescale(exponent)
                        q_out[bos + t, head * K + lane + 32 * m] = fp8_bits(value)
                        if lane == 0:
                            store_scale(s_out, bos + t, head * 4 + m, exponent, H)
        cute.arch.griddepcontrol_launch_dependents()
