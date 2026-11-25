# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA input projection at 64 rows (spec verify B8 x q8, Rubin TP16) with
``KdaSplit``'s input prologue in its epilogue.

``SplitKTmemGemm64`` (128-row tiles split over K by 6, weights staged in tensor memory
before the PDL wait, one-hop DSMEM reduce-scatter to 4 owners of 32 rows each) stores
the projection as before (BF16 rows, the published ``x``). Each owner thread then holds
one channel (lane = channel within the owner's 32 rows) of 16 tokens (warp g: tokens
16 g .. 16 g + 15, two requests), which is ``KdaSplit``'s verify mapping (thread =
channel, warp w = channels 32 w .. 32 w + 31), so it runs that prologue with the same
operations in the same order:

- Q / K / V channels: the 4-tap causal convolution over the request's conv-state taps
  (slot ``slots[r, 0]``, columns ``accepted[r] - 1 ..``) and its 8 BF16 inputs (an FP32
  ``fma`` chain from the bias), then ``v * sigmoid(v)`` (``sigmoids``); the conv state's
  new window [tap 1, tap 2, x_0 .. x_7] is written here (``KdaSplit`` no longer does).
- Q / K: per token, ``warp_sum`` of the squares over the owner's 32 channels, the 4
  owners' partials exchanged over DSMEM and combined as ((r0 + r2) + (r1 + r3)), then
  the ``rsqrt(. + 1e-6)`` factor (times ``scale`` for q): ``qn`` / ``kn`` FP32.
- Output gate channels: ``sigmoid`` of the BF16 projection, FP32.
- beta rows: ``sigmoid`` of the BF16 projection.
With ``tokens`` q < 8 (``KdaInProjQ``, B8 x q 4-6), x arrives through a 3-D tensor map
(K, q, 8 requests; ``SplitKTmemGemm64.x_warp3d``): request r takes token columns
8 r .. 8 r + 7 of the 64-column tile (columns 8 r + t, t >= q, are zeros), so the
mapping above is unchanged; the published rows, the conv windows and ``prepared`` are
written for tokens t < q only, at the compact row r q + t.
The FP32 operands go to one token-major buffer ``prepared`` [64, PREPARED_PITCH]: q at
column 0, k at dim, v at 2 dim, the output gate at 3 dim, beta (one per head) at
4 dim (F_A stays BF16 in the published rows, stored canonical: a NaN as 0x7FFF, never
the sentinel ``KdaVerifyPrepared`` polls them for). Requests whose slot
``slots[r, accepted[r] - 1]`` is <= 0 (padding) neither read nor write conv state;
their prepared values are unspecified."""

from functools import cache
from typing import Any

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import (
    _checked,
    architecture,
    fake,
    layout,
    resolve_kernel,
    stream,
    tensor_maps,
    to_cute,
)
from ..common.primitives import (
    asm,
    at,
    fma,
    named_barrier,
    push_float,
    store_words,
    wait,
    warp_sum,
)
from ..gemm.projection import matrix_view, weight_map
from ..gemm.splitk import SplitKTactic, _smem
from ..gemm.splitk_tmem64 import SplitKTmemGemm64, _ld_shared_v4, _st_shared_bf16
from .kernel import sigmoids
from .recover import mul_rn, triton_rstd, triton_sigmoid, triton_silu
from .window import check_conv_cache, conv_put, conv_tap

__all__ = [
    "PREPARED_PITCH",
    "KdaInProj64",
    "KdaInProj80",
    "KdaInProjQ",
    "PreparedKdaInProj64",
    "PreparedKdaInProj80",
]

OWNERS, SHARE, TOKENS = 4, 32, 8  # owner CTAs per tile, rows per owner, verify length
# Registers per thread at most. The CTAs start early only where they fit beside the
# preceding AttnRes CTAs (128 threads each) in the SM's 64K registers. At 98 (104
# allocated) one AttnRes CTA fewer fits beside each CTA than at 86 for AttnRes kernels
# of 57-64 and 80 registers, and those layers lose the early start. The launch bounds
# below (320 threads, 2 CTAs per SM) make ptxas target 96, without spills.
REGISTERS = 96
PREPARED_PITCH = 4 * 768 + 8  # FP32 words per token: q, k, v, gate, 6 betas, 2 pad


@asm(Float32)
def _ld_shared_bf16(address):
    """The BF16 at shared address ``address``, as FP32 (exact)."""
    code = "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.bf16 $0, h; }"
    return code, "=f,r,~{memory}", (Int32(address),)


@asm(Float32, pure=True)
def _canonical(value):
    """``value``; a NaN becomes 0f7FFF0000 (BF16 0x7FFF, never the sentinel 0xFFFF)."""
    code = "{ .reg .pred p; setp.nan.f32 p,$1,$1; selp.f32 $0,0f7FFF0000,$1,p; }"
    return code, "=f,f", (value,)


class KdaInProj64(SplitKTmemGemm64):
    """``tactic`` (128, 64, split, x's slots) over the KDA in-projection rows
    [Q | K | V | gate | F_A, beta] of ``heads`` heads; ``has_bias``: the conv bias;
    ``stock`` (an FP32 recurrent state): the prologue computes the stock verify's math
    (``recover.KdaSplit``'s Triton formulas: the SiLU outputs rounded to BF16,
    x / (1 + exp(-x)) and Triton's sigmoid, q / k scaled by 1 / sqrt.approx(n + eps)
    and then q by the query scale) for ``KdaVerifyPrepared(bf16_state=False)``."""

    def __init__(
        self, *, tactic: SplitKTactic, heads=6, has_bias=False, k=7168, stock=False
    ):
        super().__init__(tactic=tactic, k=k, owners=OWNERS)
        assert self.share == SHARE
        self.heads, self.dim, self.has_bias = heads, heads * 128, has_bias
        self.tokens = TOKENS  # verify length q (KdaInProjQ: q < 8)
        self.stock = stock

    def kept(self, t):
        """Whether the thread's token t (of 16: requests 2 g + t // 8) is a real one."""
        return t % TOKENS < self.tokens

    def token_row(self, col0, g, t):
        """The compact row of the thread's token t: request (col0 + 16 g + t) // 8."""
        if self.tokens == TOKENS:
            return col0 + 16 * g + t
        return ((col0 + 16 * g) // TOKENS + t // TOKENS) * self.tokens + t % TOKENS

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, prepared: cute.Tensor,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        slots: cute.Tensor, accepted: cute.Tensor, scale: Float32,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        args = (a, b, c, prepared, conv_w, bias, conv_state, slots, accepted)
        self.kernel(*args, scale, a_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
            max_number_threads=[320, 1, 1],  # with 2 CTAs per SM: REGISTERS (ptxas)
            min_blocks_per_mp=2,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, prepared: cute.Tensor,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        slots: cute.Tensor, accepted: cute.Tensor, scale: Float32,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        # The norm exchange: [owner][token] FP32 partial sums of squares and its
        # barrier, initialized before body's cluster sync (ahead of any DSMEM push).
        self.norm_parts = _smem(Float32, OWNERS * 64, 16)
        self.norm_bar = _smem(cutlass.Int64, 1, 8).iterator
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            cute.arch.mbarrier_init(self.norm_bar, 1)
        cute.arch.mbarrier_init_fence()
        self.prefetch(tid, conv_w, bias, conv_state, slots, accepted)
        out = (mC, prepared, conv_w, bias, conv_state, slots, accepted, scale)
        self.body(mA, mB, mC, a_map, out)

    @cute.jit
    def prefetch(self, tid, conv_w, bias, conv_state, slots, accepted):
        """Epilogue threads of owner CTAs, Q / K / V tiles: their channel's conv taps,
        weights and bias and the requests' slots, loaded at entry (weights, host
        metadata and the conv state of this layer's previous step: ordered before the
        launch, like ``KdaSplit``'s early loads), so the epilogue does not wait for
        them."""
        dim: cutlass.Constexpr = self.dim
        tap = cute.make_rmem_tensor((2, 3), BFloat16)
        coeff = cute.make_rmem_tensor(5, Float32)  # 4 taps' weights, bias
        meta = cute.make_rmem_tensor((2, 3), Int32)  # offset, live, conv slot
        for rq in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(3):
                tap[rq, j] = BFloat16(0)
                meta[rq, j] = Int32(0)
        for i in cutlass.range_constexpr(5):
            coeff[i] = Float32(0)
        bidx, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        row0 = bidx * 128 + rank * SHARE
        section = row0 // dim
        if (tid >= 128) & (rank < OWNERS) & (section < 3):
            lane, g = (tid - 128) % SHARE, (tid - 128) // SHARE
            channel = row0 % dim + lane
            row = section * dim + channel
            for i in cutlass.range_constexpr(4):
                coeff[i] = Float32(conv_w[section, i, channel])
            if cutlass.const_expr(self.has_bias):
                coeff[4] = Float32(bias[row])
            for rq in cutlass.range_constexpr(2):
                request = 2 * g + rq
                offset = accepted[request] - 1
                live = Int32(slots[request, offset] > 0)
                meta[rq, 0] = offset
                meta[rq, 1] = live
                meta[rq, 2] = slots[request, 0] * live
                if live != 0:
                    page = conv_state[Int64(meta[rq, 2]), None, None]
                    for j in cutlass.range_constexpr(3):
                        tap[rq, j] = conv_tap(page, row, offset + j)
        self.pf_tap, self.pf_coeff, self.pf_meta = tap, coeff, meta

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """The published BF16 rows (``SplitKTmemGemm64.finish``), then the prologue
        of the owner's 32 channels x 64 tokens (see the module docstring)."""
        mC, prepared, conv_w, bias, conv_state, slots, accepted, scale = out
        dim: cutlass.Constexpr = self.dim
        self.stamp(22, epi_tid == 0)
        if (row0 // dim == 4) & (row0 % dim < 128):  # F_A rows: polled (sentinel)
            for t in cutlass.range_constexpr(16):
                totals[t] = _canonical(totals[t])
        if cutlass.const_expr(self.tokens == TOKENS):
            SplitKTmemGemm64.finish(
                self, epi_tid, totals, tile, (mC,), row0, col0, l_idx
            )
        else:
            self.publish(epi_tid, totals, tile, mC, row0, col0, l_idx)
        self.stamp(16, epi_tid == 0)
        lane, g = epi_tid % SHARE, epi_tid // SHARE
        section = row0 // dim  # 0 Q, 1 K, 2 V, 3 output gate, 4 F_A / beta
        channel = row0 % dim + lane  # within the section: head * 128 + 32 owner + lane
        owner = row0 % 128 // SHARE
        xb = cute.make_rmem_tensor(16, BFloat16)
        for t in cutlass.range_constexpr(16):
            xb[t] = BFloat16(totals[t])
        if section < 3:
            y = cute.make_rmem_tensor(16, Float32)
            coeff = [self.pf_coeff[i] for i in range(4)]
            start = self.pf_coeff[4]  # the conv bias (0 without one)
            row = section * dim + channel
            for rq in cutlass.range_constexpr(2):
                live = self.pf_meta[rq, 1]
                page = conv_state[Int64(self.pf_meta[rq, 2]), None, None]
                taps = [self.pf_tap[rq, j] for j in range(3)]
                for t in cutlass.range_constexpr(self.tokens):  # real tokens only
                    acc = start
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[w] if w < 3 else xb[8 * rq + w - 3]
                        acc = fma(Float32(term), coeff[i], acc)
                    y[8 * rq + t] = acc
                if live != 0:  # the conv state's new window
                    tokens = range(self.tokens)
                    window = (taps[1], taps[2], *(xb[8 * rq + t] for t in tokens))
                    for i, value in enumerate(window):
                        conv_put(page, (row, i), value)
            self.stamp(17, epi_tid == 0)
            kept = [t for t in range(16) if self.kept(t)]
            if cutlass.const_expr(self.stock):
                for t in kept:
                    y[t] = triton_silu(y[t])
            else:
                act = sigmoids([y[t] for t in kept])
                for i, t in enumerate(kept):
                    y[t] = y[t] * act[i]
            self.stamp(18, epi_tid == 0)
            if section == 2:
                for t in cutlass.range_constexpr(16):
                    if cutlass.const_expr(self.kept(t)):
                        prepared[self.token_row(col0, g, t), 2 * dim + channel] = y[t]
            else:
                self.normalize(
                    y, section, channel, owner, lane, g, col0, prepared, scale
                )
        elif section == 3:
            kept = [t for t in range(16) if self.kept(t)]
            act = self.sigmoids([Float32(xb[t]) for t in kept])
            for i, t in enumerate(kept):
                prepared[self.token_row(col0, g, t), 3 * dim + channel] = act[i]
        elif (channel >= 128) & (channel - 128 < self.heads):  # beta rows (4 dim + 128)
            kept = [t for t in range(16) if self.kept(t)]
            act = self.sigmoids([Float32(xb[t]) for t in kept])
            for i, t in enumerate(kept):
                row = self.token_row(col0, g, t)
                prepared[row, 4 * dim + channel - 128] = act[i]

    @cute.jit
    def normalize(self, y, section, channel, owner, lane, g, col0, prepared, scale):
        """Q / K channels: per token the head's L2 norm over its 4 owners' 32-channel
        ``warp_sum`` partials, ((r0 + r2) + (r1 + r3)); qn = q rsqrt(n + 1e-6) scale,
        kn = k rsqrt(n + 1e-6)."""
        parts = cute.make_rmem_tensor(16, Float32)
        for t in cutlass.range_constexpr(16):  # padding tokens (t % 8 >= q): unused
            parts[t] = warp_sum(y[t] * y[t]) if self.kept(t) else Float32(0)
        mine = parts[0]
        for t in cutlass.range_constexpr(1, 16):  # lane t sends token 16 g + t
            if lane == t:
                mine = parts[t]
        if (lane == 0) & (g == 0):
            cute.arch.mbarrier_arrive_and_expect_tx(self.norm_bar, OWNERS * 64 * 4)
        if lane < 16:
            slot = self.norm_parts.iterator + owner * 64 + 16 * g + lane
            for peer in cutlass.range_constexpr(OWNERS):
                push_float(slot, mine, self.norm_bar, peer)
        self.stamp(20, (lane == 0) & (g == 0))
        wait(self.norm_bar)
        self.stamp(21, (lane == 0) & (g == 0))
        for t in cutlass.range_constexpr(16):
            token = 16 * g + t
            if cutlass.const_expr(self.kept(t)):
                r = [self.norm_parts[o * 64 + token] for o in range(OWNERS)]
                n = (r[0] + r[2]) + (r[1] + r[3])
                value = Float32(0)
                if cutlass.const_expr(self.stock):  # q / |q|, then x scale
                    value = mul_rn(y[t], triton_rstd(n + 1.0e-6))
                    if section == 0:
                        value = mul_rn(value, scale)
                else:
                    factor = cute.math.rsqrt(n + 1.0e-6, fastmath=True)
                    if section == 0:
                        factor = factor * scale
                    value = y[t] * factor
                row = col0 + token  # (q = 8; else the compact row)
                if cutlass.const_expr(self.tokens != TOKENS):
                    row = self.token_row(col0, g, t)
                prepared[row, section * self.dim + channel] = value

    def sigmoids(self, values):
        """The sigmoid of each value (output gates, betas): Triton's (stock) or the
        correctly rounded ``kernel.sigmoids``; traced inline."""
        if self.stock:
            return [triton_sigmoid(v) for v in values]
        return sigmoids(values)


class KdaInProjQ(KdaInProj64):
    """``KdaInProj64`` for ``tokens`` q < 8 tokens per request (8 requests): x through
    the 3-D tensor map ``x_map`` (K, q, 8) of 8 token columns per request, and the
    published rows (C [rows, 8 q]), conv windows and ``prepared`` [8 q, PREPARED_PITCH]
    of the q real tokens at the compact row r q + t (see the module docstring)."""

    def __init__(
        self, *, tactic: SplitKTactic, heads=6, has_bias=False, k=7168, tokens,
        stock=False,
    ):  # fmt: skip
        super().__init__(
            tactic=tactic, heads=heads, has_bias=has_bias, k=k, stock=stock
        )
        assert 1 < tokens < TOKENS
        self.tokens = tokens

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, prepared: cute.Tensor,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        slots: cute.Tensor, accepted: cute.Tensor, scale: Float32,
        a_map: cutlass.Int64, x_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        args = (a, b, c, prepared, conv_w, bias, conv_state, slots, accepted)
        self.kernel(*args, scale, a_map, x_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                self.split_k,  # one 64-column tile: 8 requests x 8 token columns
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
            max_number_threads=[320, 1, 1],  # with 2 CTAs per SM: REGISTERS (ptxas)
            min_blocks_per_mp=2,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, prepared: cute.Tensor,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        slots: cute.Tensor, accepted: cute.Tensor, scale: Float32,
        a_map: cutlass.Int64, x_map: cutlass.Int64,
    ):  # fmt: skip
        self.norm_parts = _smem(Float32, OWNERS * 64, 16)
        self.norm_bar = _smem(cutlass.Int64, 1, 8).iterator
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            cute.arch.mbarrier_init(self.norm_bar, 1)
        cute.arch.mbarrier_init_fence()
        self.prefetch(tid, conv_w, bias, conv_state, slots, accepted)
        out = (mC, prepared, conv_w, bias, conv_state, slots, accepted, scale)
        self.body(mA, mB, mC, a_map, out, x_map=x_map)

    @cute.experimental.jit
    def publish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, mC: cute.Tensor,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """``SplitKTmemGemm64.finish`` storing token column p as C's compact column
        (p / 8) q + p % 8, real tokens (p % 8 < q) only."""
        rr, g = epi_tid % SHARE, epi_tid // SHARE
        for t in cutlass.range_constexpr(16):
            _st_shared_bf16(tile + ((16 * g + t) * SHARE + rr) * 2, totals[t])
        named_barrier(3, 128)
        rows = mC.layout.shape[0]
        token, half = epi_tid // 2, epi_tid % 2
        row = row0 + 16 * half
        p = col0 + token
        column = p // TOKENS * self.tokens + p % TOKENS
        if (row < rows) & (p % TOKENS < self.tokens):
            for c in cutlass.range_constexpr(2):
                words = _ld_shared_v4(tile + (token * SHARE + 16 * half + 8 * c) * 2)
                store_words(at(mC, row + 8 * c, column, l_idx), words)


@cache
def compile_inproj64(tactic, heads, has_bias, specs, arch, stock=False):
    """``KdaInProj64`` for ``tactic`` (128, 64, split, x's slots) and the layouts
    ``specs`` of (prepared, conv weight, bias, conv state view, slots, accepted);
    ``stock``: the stock verify's math (an FP32 recurrent state)."""

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = KdaInProj64(
        tactic=SplitKTactic(*tactic[:4]), heads=heads, has_bias=has_bias, stock=stock
    )
    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    args = (matrix(1), matrix(1), matrix(0), *tensors, Float32(1.0), cutlass.Int64(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


@cache
def compile_inproj_q(tactic, heads, has_bias, specs, arch, tokens, stock=False):
    """``KdaInProjQ`` (q = ``tokens`` < 8) for ``tactic`` and ``specs`` as
    ``compile_inproj64``; x comes through a tensor map argument."""

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = KdaInProjQ(
        tactic=SplitKTactic(*tactic[:4]), heads=heads, has_bias=has_bias,
        tokens=tokens, stock=stock,
    )  # fmt: skip
    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    args = (matrix(1), matrix(1), matrix(0), *tensors, Float32(1.0))
    args += (cutlass.Int64(0), cutlass.Int64(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


class PreparedKdaInProj64:
    """A KDA layer's in-projection at 64 verify rows with ``KdaSplit``'s input
    prologue (``KdaInProj64``): ``core`` is the layer's ``PreparedKda`` (conv weight,
    bias, conv-state layout, q scale); ``tactic`` the B64 split-K tactic."""

    def __init__(self, weight, core, tactic, tokens=TOKENS):
        """``tokens``: the verify length q (< 8: ``KdaInProjQ``, x through 3-D maps)."""
        self.weight, self.core, self.tactic = weight, core, tuple(tactic)
        self.view, self.map = matrix_view(weight), weight_map(weight, 128)
        self.device, self.tokens = weight.device, tokens
        self.code: Any = None  # compiled by prepare_caches
        self.x_maps: dict = {}  # (q < 8) x data pointer -> its 3-D tensor map

    def _tensors(self, prepared, caches, slots, accepted):
        conv = caches[0]
        conv = conv if self.core.conv_dim_first else conv.transpose(-1, -2)
        conv_w, bias = self.core.weights[:2]
        return (prepared, conv_w, bias, conv, slots, accepted)

    def prepare_caches(self, caches, requests=8):
        # BF16 or FP32 conv window (window.py), checked before any compile
        core = self.core
        check_conv_cache(caches[0], core.dim, core.conv_dim_first, "KdaInProj64")
        meta, tokens = dict(device="meta"), self.tokens
        rows = requests * tokens
        prepared = torch.empty(rows, PREPARED_PITCH, dtype=torch.float32, **meta)
        slots = torch.empty((requests, tokens), dtype=torch.int32, **meta)
        accepted = torch.empty((requests,), dtype=torch.int32, **meta)
        tensors = self._tensors(prepared, caches, slots, accepted)
        specs, arch = tuple(map(layout, tensors)), architecture(self.device)
        heads, bias = self.core.heads, self.core.has_bias
        # an FP32 recurrent state: the stock verify's math (KdaVerifyPrepared's too)
        stock = caches[1].dtype != torch.bfloat16
        with torch.accelerator.device_index(self.device.index):
            if tokens == TOKENS:
                self.code = compile_inproj64(
                    self.tactic, heads, bias, specs, arch, stock
                )
            else:
                self.code = compile_inproj_q(
                    self.tactic, heads, bias, specs, arch, tokens, stock
                )
        # The early start beside the AttnRes needs the register budget (REGISTERS).
        _, function = resolve_kernel(self.code, self.device.index)
        count = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS
        (registers,) = _checked(_cuda.cuFuncGetAttribute(count, function))
        if registers > REGISTERS:
            raise RuntimeError(
                f"KdaInProj64 uses {registers} registers per thread, over {REGISTERS}"
            )
        return self

    def x_map(self, x):
        """The 3-D tensor map (K, q, 8) of x [8 q, K] (q < 8), built once per buffer
        (eagerly: a graph capture needs it to exist already)."""
        key = (x.data_ptr(), x.shape[1])
        if key not in self.x_maps:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "KdaInProjQ: x's tensor map must exist before capture"
                )
            k, q = x.shape[1], self.tokens
            bf16 = _cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            swizzle = _cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
            l2 = _cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            # x as [8 requests, q tokens, K]: box (64 K, 8 tokens, 8 requests)
            spec = (x.data_ptr(), bf16, (k, q, 8), (k * 2, q * k * 2), (64, 8, 8))
            spec += (swizzle, l2)
            self.x_maps[key] = tensor_maps([spec], self.device)
        return self.x_maps[key]

    def run(self, x, projected, prepared, *, caches, slots, accepted):
        """``projected`` [8 q, rows] BF16 (as the plain projection's), ``prepared``
        [8 q, PREPARED_PITCH] FP32; ``slots`` [8, q], ``accepted`` [8]."""
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        tensors = self._tensors(prepared, caches, slots[:8], accepted[:8])
        views = [to_cute(t, 16) for t in tensors]
        y = matrix_view(projected, output=True)
        scale = self.core.scalars[1]
        maps = (cutlass.Int64(self.map.data_ptr()),)
        if self.tokens != TOKENS:
            maps += (cutlass.Int64(self.x_map(x).data_ptr()),)
        self.code(
            self.view, matrix_view(x), y, *views, scale, *maps, stream(self.device)
        )  # fmt: skip
        return projected


WIDTH80 = 20  # tokens per owner thread at N80 (80 token columns x 32 rows / 128)


def _x_at(xb, tile, base, lane, j):
    """Token base + j of the thread's channel as BF16 (trace-time ``j``): its own
    register for 0 <= j < 20, else the owner's BF16 tile [tokens][SHARE]."""
    if 0 <= j < WIDTH80:
        return xb[j]
    address = tile + ((base + j) * SHARE + lane) * 2
    return BFloat16(_ld_shared_bf16(address))


class KdaInProj80(KdaInProj64):
    """``KdaInProj64`` at 72-80 verify rows (B9-B10 x q8): ``SplitKTmemGemm64`` with
    one N tile of 80 token columns (``tactic`` (128, 80, split, x's slots); 13 A tiles
    in tensor memory), then the same prologue (same operations, same order) per
    channel. Each owner thread holds one channel of 20 tokens (warp g: tokens 20 g ..
    20 g + 19), so a request (8 tokens) may straddle two warps: warp g's 8-aligned
    base is 20 g - phase (phase 0 for even g, 4 for odd g); the requests whose first
    token lies in [20 g, 20 g + 20) are the thread's own (prefetched taps, conv state
    window written here: 3 for even g, 2 for odd g). Conv inputs of the other warp's
    tokens (odd g: the 3 before 20 g; even g: a window's tokens past 20 g + 19) are
    read back as BF16 from the owner's output tile (the published rows' transpose,
    written before ``finish``'s barrier): the same BF16 values. Rows past the batch
    (B9: tokens 72-79, x zero-filled by TMA) are computed but neither published nor
    stored; requests past it read and write no conv state. q = 8 only."""

    def __init__(self, *, tactic: SplitKTactic, heads=6, has_bias=False, k=7168,
                 stock=False):  # fmt: skip
        super().__init__(
            tactic=tactic, heads=heads, has_bias=has_bias, k=k, stock=stock
        )
        assert self.n == 80 and self.n * SHARE // 128 == WIDTH80

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, prepared: cute.Tensor,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        slots: cute.Tensor, accepted: cute.Tensor, scale: Float32,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        self.norm_parts = _smem(Float32, OWNERS * 80, 16)
        self.norm_bar = _smem(cutlass.Int64, 1, 8).iterator
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            cute.arch.mbarrier_init(self.norm_bar, 1)
        cute.arch.mbarrier_init_fence()
        self.prefetch(tid, conv_w, bias, conv_state, slots, accepted)
        out = (mC, prepared, conv_w, bias, conv_state, slots, accepted, scale)
        self.body(mA, mB, mC, a_map, out)

    @cute.jit
    def prefetch(self, tid, conv_w, bias, conv_state, slots, accepted):
        """As ``KdaInProj64.prefetch`` for the thread's own requests: k = 0, 1(, 2)
        from ceil(20 g / 8) (3 for even g, 2 for odd g), those below the batch."""
        dim: cutlass.Constexpr = self.dim
        tap = cute.make_rmem_tensor((3, 3), BFloat16)
        coeff = cute.make_rmem_tensor(5, Float32)
        meta = cute.make_rmem_tensor((3, 3), Int32)  # offset, live, conv slot
        for rq in cutlass.range_constexpr(3):
            for j in cutlass.range_constexpr(3):
                tap[rq, j] = BFloat16(0)
                meta[rq, j] = Int32(0)
        for i in cutlass.range_constexpr(5):
            coeff[i] = Float32(0)
        bidx, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        row0 = bidx * 128 + rank * SHARE
        section = row0 // dim
        if (tid >= 128) & (rank < OWNERS) & (section < 3):
            lane, g = (tid - 128) % SHARE, (tid - 128) // SHARE
            channel = row0 % dim + lane
            row = section * dim + channel
            for i in cutlass.range_constexpr(4):
                coeff[i] = Float32(conv_w[section, i, channel])
            if cutlass.const_expr(self.has_bias):
                coeff[4] = Float32(bias[row])
            first = (WIDTH80 * g + 7) // 8
            count = 3 - g % 2
            requests = accepted.layout.shape[0]
            for rq in cutlass.range_constexpr(3):
                request = first + rq
                if (rq < count) & (request < requests):
                    offset = accepted[request] - 1
                    live = Int32(slots[request, offset] > 0)
                    meta[rq, 0] = offset
                    meta[rq, 1] = live
                    meta[rq, 2] = slots[request, 0] * live
                    if live != 0:
                        page = conv_state[Int64(meta[rq, 2]), None, None]
                        for j in cutlass.range_constexpr(3):
                            tap[rq, j] = conv_tap(page, row, offset + j)
        self.pf_tap, self.pf_coeff, self.pf_meta = tap, coeff, meta

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """The published BF16 rows (``SplitKTmemGemm64.finish``: the owner's tile
        [80 tokens][32 rows] BF16 is complete after its barrier), then the prologue of
        the thread's channel x 20 tokens (see the class docstring)."""
        mC, prepared, conv_w, bias, conv_state, slots, accepted, scale = out
        dim: cutlass.Constexpr = self.dim
        width: cutlass.Constexpr = WIDTH80
        if (row0 // dim == 4) & (row0 % dim < 128):  # F_A rows: polled (sentinel)
            for t in cutlass.range_constexpr(width):
                totals[t] = _canonical(totals[t])
        self.stamp(22, epi_tid == 0)
        SplitKTmemGemm64.finish(self, epi_tid, totals, tile, (mC,), row0, col0, l_idx)
        self.stamp(16, epi_tid == 0)
        lane, g = epi_tid % SHARE, epi_tid // SHARE
        section = row0 // dim  # 0 Q, 1 K, 2 V, 3 output gate, 4 F_A / beta
        channel = row0 % dim + lane
        owner = row0 % 128 // SHARE
        rows = prepared.layout.shape[0]
        base = width * g  # the thread's first token
        xb = cute.make_rmem_tensor(width, BFloat16)
        for t in cutlass.range_constexpr(width):
            xb[t] = BFloat16(totals[t])
        if section < 3:
            y = cute.make_rmem_tensor(width, Float32)
            row = section * dim + channel
            if g % 2 == 0:
                self.conv80(y, xb, tile, lane, base, row, conv_state, 0)
            else:
                self.conv80(y, xb, tile, lane, base, row, conv_state, 4)
            self.stamp(17, epi_tid == 0)
            if cutlass.const_expr(self.stock):
                for t in cutlass.range_constexpr(width):
                    y[t] = triton_silu(y[t])
            else:
                act = sigmoids([y[t] for t in range(width)])
                for t in cutlass.range_constexpr(width):
                    y[t] = y[t] * act[t]
            self.stamp(18, epi_tid == 0)
            if section == 2:
                for t in cutlass.range_constexpr(width):
                    if base + t < rows:
                        prepared[base + t, 2 * dim + channel] = y[t]
            else:
                self.normalize80(y, section, channel, owner, lane, g, prepared, scale)
        elif section == 3:
            act = self.sigmoids([Float32(xb[t]) for t in range(width)])
            for t in cutlass.range_constexpr(width):
                if base + t < rows:
                    prepared[base + t, 3 * dim + channel] = act[t]
        elif (channel >= 128) & (channel - 128 < self.heads):  # beta rows
            act = self.sigmoids([Float32(xb[t]) for t in range(width)])
            for t in cutlass.range_constexpr(width):
                if base + t < rows:
                    prepared[base + t, 4 * dim + channel - 128] = act[t]

    @cute.jit
    def conv80(self, y, xb, tile, lane, base, row, conv_state, phase):
        """The 4-tap causal conv of the thread's 20 tokens (``KdaInProj64``'s FP32
        fma chain from the bias) and its own requests' new conv windows; ``phase``
        (trace-time): base - phase is 8-aligned. Local position p = t + phase: request
        p // 8 (own requests from 1 when phase is 4), token p % 8."""
        width: cutlass.Constexpr = WIDTH80
        coeff = [self.pf_coeff[i] for i in range(4)]
        start = self.pf_coeff[4]

        own = 0 if phase == 0 else 1  # local request index of prefetch slot 0
        for j in cutlass.range_constexpr(width):
            p = j + phase
            rq, t = p // 8, p % 8
            acc = start
            for i in cutlass.range_constexpr(4):
                w = t + i
                # (w < 3 only where t < 3: an own request)
                term = self.pf_tap[rq - own, w] if w < 3 else _x_at(xb, tile, base, lane, j + i - 3)
                acc = fma(Float32(term), coeff[i], acc)
            y[j] = acc
        for rq in cutlass.range_constexpr(own, 3):
            slot = rq - own
            live = self.pf_meta[slot, 1]
            if live != 0:  # the conv state's new window
                page = conv_state[Int64(self.pf_meta[slot, 2]), None, None]
                first = 8 * rq - phase
                window = (self.pf_tap[slot, 1], self.pf_tap[slot, 2])
                window += tuple(_x_at(xb, tile, base, lane, first + t) for t in range(TOKENS))
                for i, value in enumerate(window):
                    conv_put(page, (row, i), value)

    @cute.jit
    def normalize80(self, y, section, channel, owner, lane, g, prepared, scale):
        """``KdaInProj64.normalize`` over 80 tokens (20 per warp): per token the
        head's L2 norm over its 4 owners' partials, ((r0 + r2) + (r1 + r3))."""
        width: cutlass.Constexpr = WIDTH80
        parts = cute.make_rmem_tensor(width, Float32)
        for t in cutlass.range_constexpr(width):
            parts[t] = warp_sum(y[t] * y[t])
        mine = parts[0]
        for t in cutlass.range_constexpr(1, width):  # lane t sends token 20 g + t
            if lane == t:
                mine = parts[t]
        if (lane == 0) & (g == 0):
            cute.arch.mbarrier_arrive_and_expect_tx(self.norm_bar, OWNERS * 80 * 4)
        if lane < width:
            slot = self.norm_parts.iterator + owner * 80 + width * g + lane
            for peer in cutlass.range_constexpr(OWNERS):
                push_float(slot, mine, self.norm_bar, peer)
        self.stamp(20, (lane == 0) & (g == 0))
        wait(self.norm_bar)
        self.stamp(21, (lane == 0) & (g == 0))
        rows = prepared.layout.shape[0]
        for t in cutlass.range_constexpr(width):
            token = width * g + t
            r = [self.norm_parts[o * 80 + token] for o in range(OWNERS)]
            n = (r[0] + r[2]) + (r[1] + r[3])
            value = Float32(0)
            if cutlass.const_expr(self.stock):
                value = mul_rn(y[t], triton_rstd(n + 1.0e-6))
                if section == 0:
                    value = mul_rn(value, scale)
            else:
                factor = cute.math.rsqrt(n + 1.0e-6, fastmath=True)
                if section == 0:
                    factor = factor * scale
                value = y[t] * factor
            if token < rows:
                prepared[token, section * self.dim + channel] = value


@cache
def compile_inproj80(tactic, heads, has_bias, specs, arch, stock=False):
    """``KdaInProj80`` for ``tactic`` (128, 80, split, x's slots) and ``specs`` as
    ``compile_inproj64``."""

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = KdaInProj80(
        tactic=SplitKTactic(*tactic[:4]), heads=heads, has_bias=has_bias, stock=stock
    )
    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    args = (matrix(1), matrix(1), matrix(0), *tensors, Float32(1.0), cutlass.Int64(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


class PreparedKdaInProj80(PreparedKdaInProj64):
    """``PreparedKdaInProj64`` at 9-10 requests x q8 (``KdaInProj80``, one compile
    per request count): ``tactic`` the (128, 80, split, slots[, 13]) N80 tactic."""

    def __init__(self, weight, core, tactic, requests=(9, 10)):
        super().__init__(weight, core, tactic, tokens=TOKENS)
        self.requests = tuple(requests)
        self.codes: dict = {}

    def prepare_caches(self, caches, requests=None):
        core = self.core
        check_conv_cache(caches[0], core.dim, core.conv_dim_first, "KdaInProj80")
        meta = dict(device="meta")
        stock = caches[1].dtype != torch.bfloat16
        arch = architecture(self.device)
        heads, bias = core.heads, core.has_bias
        for count in self.requests:
            rows = count * TOKENS
            prepared = torch.empty(rows, PREPARED_PITCH, dtype=torch.float32, **meta)
            slots = torch.empty((count, TOKENS), dtype=torch.int32, **meta)
            accepted = torch.empty((count,), dtype=torch.int32, **meta)
            tensors = self._tensors(prepared, caches, slots, accepted)
            specs = tuple(map(layout, tensors))
            with torch.accelerator.device_index(self.device.index):
                code = compile_inproj80(self.tactic, heads, bias, specs, arch, stock)
            _, function = resolve_kernel(code, self.device.index)
            attr = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS
            (registers,) = _checked(_cuda.cuFuncGetAttribute(attr, function))
            if registers > REGISTERS:
                raise RuntimeError(
                    f"KdaInProj80 uses {registers} registers per thread, over {REGISTERS}"
                )
            self.codes[count] = code
        return self

    def run(self, x, projected, prepared, *, caches, slots, accepted):
        """x / ``projected`` [8 B, *], ``prepared`` [8 B, PREPARED_PITCH] FP32;
        ``slots`` [>= B, 8], ``accepted`` [>= B]; B in ``requests``."""
        count = x.shape[0] // TOKENS
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        tensors = self._tensors(prepared, caches, slots[:count], accepted[:count])
        views = [to_cute(t, 16) for t in tensors]
        y = matrix_view(projected, output=True)
        scale = self.core.scalars[1]
        maps = (cutlass.Int64(self.map.data_ptr()),)
        self.codes[count](
            self.view, matrix_view(x), y, *views, scale, *maps, stream(self.device)
        )  # fmt: skip
        return projected
