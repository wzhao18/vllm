# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-batched cluster AttnRes for wide decode shapes (``prepared.ROWS_TOKENS``).

``ClusterAttnRes`` runs one 7-CTA cluster per token; about 181 such clusters are
resident on Rubin (~6 CTAs of 80 registers per SM), so above that many rows a second
wave pays the loads and two cluster exchanges again after the first wave's tokens
waited for their delta. Here each cluster handles ``tokens`` consecutive rows (comm3's
parked token-batched kernel, revived for 193-256 rows): the saved blocks stream one
vector at a time (only the tokens' mixtures and rows stay in registers), and every
exchange round carries all the tokens' partial sums in one DSMEM push. Per token the
arithmetic is ``ClusterAttnRes``'s: the same dot8 chains, warp butterflies (a value's
lane tree does not depend on how many values share the warp), warp and peer sum
orders and roundings, so the outputs are bitwise identical.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from ..common import diagstamp as ds
from ..common.mailbox import poll, poll_warp, rearm
from ..common.primitives import (
    add_rn,
    alloc,
    at,
    bf16x2,
    fma,
    group_barrier,
    launch,
    load_words,
    push_float,
    scatter_reduce,
    store_words,
    wait,
    zeros,
)
from .kernel import (
    ClusterAttnRes,
    arm_exchanges,
    block_ptr,
    dot8,
    round_pair,
    rstd,
    unpack,
)

# Values per warp-reduce chunk (one per lane after the warp butterflies).
CHUNK = 32
# Values per DSMEM round: up to two warp-reduce chunks (e.g. 4 tokens x 16 pre-phase
# values at 8 saved blocks).
ROUND = 2 * CHUNK


def rows_tokens(nb, final=False):
    """Tokens per cluster: as many as one exchange round holds (at most 8)."""
    npre, npost = 2 * nb, (2 if final else 4) if nb else 1
    return max(1, min(8, ROUND // max(npre, npost, 1)))


class ClusterAttnResRows(ClusterAttnRes):
    """``ClusterAttnRes`` over ``tokens`` rows per cluster (grid (7, ceil(rows /
    tokens))); rows past the batch are skipped. Same arguments, same outputs."""

    def __init__(self, nb, *, tokens=None, **kwargs):
        super().__init__(nb, **kwargs)
        assert not self.out_mc, "row-sharded outputs: ClusterAttnRes only"
        self.tokens = tokens or rows_tokens(nb, self.final)
        T = self.tokens
        assert T * max(self.npre, self.npost) <= ROUND, (nb, T)
        # Shared-memory row length of the exchanges: one chunk when every round fits
        # (the variants up to 2 tokens at 8 blocks keep their layout), else two.
        self.round = CHUNK if T * max(self.npre, self.npost) <= CHUNK else ROUND
        pad = lambda n: 1 << max(n - 1, 0).bit_length()  # noqa: E731
        self.kpre_t, self.kpost_t, self.kfin_t = (
            pad(T * self.npre),
            pad(T * self.npost),
            pad(T),
        )

    @cute.jit
    def __call__(self, *args):
        rows = (args[1].shape[0] + self.tokens - 1) // self.tokens
        if cutlass.const_expr(self.min_blocks > 0):
            launch(
                self.kernel,
                args,
                (7, rows, 1),
                128,
                7,
                min_blocks_per_mp=self.min_blocks,
            )
        else:
            launch(self.kernel, args, (7, rows, 1), 128, 7)

    @cute.jit
    def setup(self, smem):
        R: cutlass.Constexpr = self.round
        warp_buf = alloc(smem, Float32, 4 * R)
        peer_buf = alloc(smem, Float32, (3, 7 * R))
        total = alloc(smem, Float32, (3, R))
        bars = alloc(smem, cutlass.Uint64, 3, 8)
        T = self.tokens
        arm_exchanges(bars, (T * self.npre, T * self.npost, T * int(self.final)))
        return warp_buf, peer_buf, total, bars

    @cute.jit
    def exchange(
        self, values, count: cutlass.Constexpr, kpad: cutlass.Constexpr, bufs, r, ids
    ):
        """``ClusterAttnRes.exchange`` with ``count`` <= ``self.round`` values (warp
        rows and peer rows of ``self.round``; every (value, peer) push from its own
        thread). Above 32 values the warp reduce runs per chunk of 32: a value's
        butterfly tree is the same whatever shares the warp, so the sums are
        bitwise those of one chunk."""
        wb, peer_buf, total, bars = bufs
        peer_buf, total, bar = peer_buf[r, None], total[r, None], bars.iterator + r
        rank, _, tid = ids
        lane, warp = tid % 32, tid // 32
        R: cutlass.Constexpr = self.round
        if cutlass.const_expr(kpad <= CHUNK):
            reduced, index = scatter_reduce(values, lane, kpad)
            if (lane & (32 // kpad - 1)) == 0 and index < count:
                wb[warp * R + index] = reduced
        else:
            for c in cutlass.range_constexpr(kpad // CHUNK):
                part = cute.make_rmem_tensor(CHUNK, Float32)
                for i in cutlass.range_constexpr(CHUNK):
                    part[i] = values[c * CHUNK + i]
                reduced, index = scatter_reduce(part, lane, CHUNK)
                if c * CHUNK + index < count:
                    wb[warp * R + c * CHUNK + index] = reduced
        group_barrier(0)
        for j in cutlass.range(tid, 7 * count, 128, unroll=1):
            i, peer = j % count, j // count
            value = (wb[i] + wb[R + i]) + (wb[2 * R + i] + wb[3 * R + i])
            push_float(peer_buf.iterator + rank * R + i, value, bar, peer)
        wait(bar)
        if tid < count:
            sums = [Float32(0), Float32(0)]
            for peer in cutlass.range_constexpr(7):
                sums[peer % 2] += peer_buf[peer * R + tid]
            total[tid] = sums[0] + sums[1]
        group_barrier(0)

    @cute.kernel
    def kernel(self, args):
        delta, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        rank, group, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        bufs = self.setup(cutlass.utils.SmemAllocator())
        sbase, sctr = ds.buffer("arrows", prefix.shape[0])
        saddr = ds.begin(sbase, sctr, group * 7 + rank, tid == 0)  # DIAG (TPC-2)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        T: cutlass.Constexpr = self.tokens
        mixture = cute.make_rmem_tensor((T, 8), Float32)
        scal = cute.make_rmem_tensor((T, 2), Float32)
        rows, ids = (prefix, blocks, norm, query, out_norm), (rank, group, tid)
        first = group * T
        self.pre_rows(mixture, scal, bufs, rows, eps, ids, first)
        ds.stamp(saddr, 1)  # DIAG: pre-wait half (block statistics exchanged, mixture)
        args9 = args[9:]
        self.post_rows(
            mixture, scal, bufs, delta, rows, output, eps, out_eps, ids, first, args9,
            saddr,
        )
        ds.stamp(saddr, 7)
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def pre_rows(self, mixture, scal, bufs, rows, eps: Float32, ids, first):
        """The tokens' saved-block statistics in one exchange, then each token's block
        softmax and mixture (the blocks reloaded one vector at a time)."""
        prefix, blocks, norm, query, _ = rows
        rank, _, tid = ids
        _, _, total, _ = bufs
        T: cutlass.Constexpr = self.tokens
        n = prefix.shape[0]
        col = (rank * 128 + tid) * 8
        W = self.nb
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        self.weights(v, norm, query, col)
        values = zeros(self.kpre_t, Float32)
        if cutlass.const_expr(self.nb > 0):
            for t in cutlass.range_constexpr(T):
                if first + t < n:
                    for source in cutlass.range_constexpr(self.nb):
                        ptr = block_ptr(blocks, first + t, source, col)
                        unpack(load_words(ptr, count=4, mutable=True), v, 0)
                        values[t * self.npre + source] = dot8(v, 0, v, 0)
                        values[t * self.npre + self.nb + source] = dot8(v, 0, v, W)
        cute.arch.cluster_wait()
        mixture.fill(0)
        scal.fill(0)
        if cutlass.const_expr(self.nb > 0):
            self.exchange(values, T * self.npre, self.kpre_t, bufs, 0, ids)
            for t in cutlass.range_constexpr(T):
                if first + t < n:
                    base = t * self.npre
                    logits = cute.make_rmem_tensor(self.nb, Float32)
                    for b in cutlass.range_constexpr(self.nb):
                        logits[b] = total[0, base + self.nb + b] * rstd(
                            total[0, base + b], eps
                        )
                    maximum = logits[0]
                    for source in cutlass.range_constexpr(1, self.nb):
                        maximum = cute.arch.fmax(maximum, logits[source])
                    exp_sum = Float32(0)
                    for source in cutlass.range_constexpr(self.nb):
                        e = cute.math.exp(logits[source] - maximum, fastmath=True)
                        exp_sum += e
                        ptr = block_ptr(blocks, first + t, source, col)
                        unpack(load_words(ptr, count=4, mutable=True), v, 0)
                        for i in cutlass.range_constexpr(8):
                            mixture[t, i] = fma(e, v[0, i], mixture[t, i])
                    scal[t, 0], scal[t, 1] = maximum, exp_sum

    @cute.jit
    def post_rows(
        self, mixture, scal, bufs, delta, rows, output, eps: Float32, out_eps: Float32,
        ids, first, aux, saddr=None,
    ):  # fmt: skip
        """Per token: add the delta, update prefix (and its ``aux`` copies) and saved
        block; the tokens' post statistics in one exchange (the final boundary: a
        second one for the rounded mixture's squares); the normalized outputs."""
        prefix, blocks, norm, query, out_norm = rows
        rank, _, tid = ids
        _, _, total, _ = bufs
        T: cutlass.Constexpr = self.tokens
        n = prefix.shape[0]
        col = (rank * 128 + tid) * 8
        W, G, OLD, NEW = self.nb, self.nb + 1, self.nb + 2, self.nb + 3
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        self.weights(v, norm, query, col)
        unpack(load_words(out_norm.iterator + col, count=4), v, G)
        news = cute.make_rmem_tensor((T, 4), Uint32)
        news.fill(0)
        values = zeros(self.kpost_t, Float32)
        if cutlass.const_expr(self.slots == 0):
            cute.arch.griddepcontrol_wait()
        for t in cutlass.range_constexpr(T):
            token = first + t
            if token < n:
                if cutlass.const_expr(self.has_delta):
                    old = load_words(at(prefix, token, col), count=4, mutable=True)
                    unpack(old, v, OLD)
                cur = cute.make_rmem_tensor((1, 8), Float32)
                if cutlass.const_expr(self.slots == 0):
                    unpack(
                        load_words(at(delta, token, col), count=4, mutable=True), cur, 0
                    )
                else:
                    cur.fill(0)
                    base = delta.iterator + token * 7168 + col
                    if cutlass.const_expr(self.poll_warp):
                        got = poll_warp(base, delta.shape[1] * 7168, self.slots)
                    else:
                        got = poll(base, delta.shape[1] * 7168, self.slots)
                    for slot in cutlass.range_constexpr(self.slots):
                        rearm(base + slot * delta.shape[1] * 7168)
                        part = cute.make_rmem_tensor((1, 8), Float32)
                        unpack(tuple(got[slot, k] for k in range(4)), part, 0)
                        for i in cutlass.range_constexpr(8):
                            cur[0, i] = cur[0, i] + part[0, i]
                    if cutlass.const_expr(self.slots > 1):
                        for j in cutlass.range_constexpr(4):
                            round_pair(cur, 0, j)
                    if cutlass.const_expr(self.slots > 1 and self.sanitize):
                        for i in cutlass.range_constexpr(8):
                            cur[0, i] = add_rn(cur[0, i], Float32(0))
                for j in cutlass.range_constexpr(4):
                    lo, hi = cur[0, 2 * j], cur[0, 2 * j + 1]
                    if cutlass.const_expr(self.has_delta):
                        lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1] + hi
                    news[t, j] = bf16x2(lo, hi)
                unpack(tuple(news[t, j] for j in range(4)), v, NEW)
                k = t * self.npost
                values[k] = dot8(v, NEW, v, NEW)
                if cutlass.const_expr(self.nb > 0):
                    values[k + 1] = dot8(v, NEW, v, W)
                    if cutlass.const_expr(not self.final):
                        values[k + 2] = dot8(mixture, t, v, NEW)
                        values[k + 3] = dot8(mixture, t, mixture, t)
                row = tuple(news[t, j] for j in range(4))
                self.store_rows(row, prefix, blocks, aux, token, col)
        if cutlass.const_expr(saddr is not None):
            ds.stamp(saddr, 2)  # DIAG: every token's delta landed, prefix/blocks stored
        self.exchange(values, T * self.npost, self.kpost_t, bufs, 1, ids)
        if cutlass.const_expr(saddr is not None):
            ds.stamp(saddr, 3)  # DIAG: post statistics exchanged
        fins = zeros(self.kfin_t, Float32)
        results = cute.make_rmem_tensor((T, 8), Float32)
        for t in cutlass.range_constexpr(T):
            token = first + t
            if token < n:
                unpack(tuple(news[t, j] for j in range(4)), v, NEW)
                k = t * self.npost
                if cutlass.const_expr(self.nb == 0):
                    scale = rstd(total[1, k], out_eps)
                    for i in cutlass.range_constexpr(8):
                        results[t, i] = v[NEW, i] * scale * v[G, i]
                else:
                    ss, dot = total[1, k], total[1, k + 1]
                    logit = dot * rstd(ss, eps)
                    maximum, exp_sum = scal[t, 0], scal[t, 1]
                    top = cute.arch.fmax(maximum, logit)
                    alpha = cute.math.exp(maximum - top, fastmath=True)
                    ep = cute.math.exp(logit - top, fastmath=True)
                    denominator = fma(alpha, exp_sum, ep)
                    if cutlass.const_expr(self.final):
                        inverse = Float32(1) / denominator
                        for j in cutlass.range_constexpr(4):
                            for i in (2 * j, 2 * j + 1):
                                results[t, i] = fma(
                                    alpha * inverse,
                                    mixture[t, i],
                                    ep * inverse * v[NEW, i],
                                )
                            round_pair(results, t, j)
                        fins[t] = dot8(results, t, results, t)
                    else:
                        a2, t3 = alpha * alpha, total[1, k + 3]
                        ae2 = Float32(2) * alpha * ep
                        quad = fma(a2, t3, fma(ae2, total[1, k + 2], ep * ep * ss))
                        scale = rstd(quad, out_eps * denominator * denominator)
                        ca, cp = alpha * scale, ep * scale
                        for i in cutlass.range_constexpr(8):
                            results[t, i] = (
                                fma(ca, mixture[t, i], cp * v[NEW, i]) * v[G, i]
                            )
        if cutlass.const_expr(self.final and self.nb > 0):
            self.exchange(fins, T, self.kfin_t, bufs, 2, ids)
            for t in cutlass.range_constexpr(T):
                if first + t < n:
                    scale = rstd(total[2, t], out_eps)
                    for i in cutlass.range_constexpr(8):
                        results[t, i] = results[t, i] * scale * v[G, i]
        for t in cutlass.range_constexpr(T):
            token = first + t
            if token < n:
                words = tuple(
                    bf16x2(results[t, 2 * j], results[t, 2 * j + 1]) for j in range(4)
                )
                store_words(at(output, token, col), words)
        if cutlass.const_expr(saddr is not None):
            ds.stamp(saddr, 6)  # DIAG: outputs stored
