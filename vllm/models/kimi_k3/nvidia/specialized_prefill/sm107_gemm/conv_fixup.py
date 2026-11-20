# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rows the KDA conv epilogue leaves open, and the conv state update.

Programs [0, Mb): rows 0-2 of 128-row block b that are not sequence starts
(taps from the saved block-boundary rows). Programs [Mb, Mb + S): rows 0-2 of
sequence s (taps from the old conv state) and its new conv state (the last
three inputs, shifted in after the old state for sequences shorter than 3).
Arithmetic is the stock Triton conv's (FP32 FMA chain from +0, oldest tap
first; ``acc / (1 + exp(-acc))``), so outputs and state stay bit for bit.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _row(
    out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, x0, x1, x2, x3, part, c0, t,
    M, P, WITH_NORMS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc += x0 * w0
    acc += x1 * w1
    acc += x2 * w2
    acc += x3 * w3
    y = (acc / (1 + tl.exp(-acc))).to(tl.bfloat16)
    rowbase = (part * M + t).to(tl.int64)
    tl.store(out_ptr + rowbase * P + (cols - part * P), y, mask=cmask)
    if WITH_NORMS:
        if part < 2:
            yf = y.to(tl.float32)
            sq = tl.reshape(yf * yf, [BLOCK // 128, 128])
            inv = tl.math.rsqrt(tl.sum(sq, axis=1) + 1e-6)
            hd = rowbase * (P // 128) + (c0 - part * P) // 128
            tl.store(norm_ptr + hd + tl.arange(0, BLOCK // 128), inv)


@triton.jit
def _conv_fixup_kernel(
    out_ptr, side_ptr, norm_ptr, w_ptr, st_ptr, qsl_ptr, ci_ptr, init_ptr,
    M, N, P, NB, S, st_s0, st_s1, st_s2, ci_s0, null_id,
    HAS_NULL: tl.constexpr, WITH_NORMS: tl.constexpr, BLOCK: tl.constexpr,
    SP2: tl.constexpr, SEQ_ONLY: tl.constexpr = False,
):  # fmt: skip
    pid = tl.program_id(0)
    if SEQ_ONLY:
        pid += NB
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < N
    c0 = tl.program_id(1) * BLOCK
    part = c0 // P
    sidx = tl.arange(0, SP2)
    q = tl.load(qsl_ptr + sidx, mask=sidx <= S, other=2147483647)
    total = tl.load(qsl_ptr + S)
    w0 = tl.load(w_ptr + cols * 4 + 0, mask=cmask, other=0.0)
    w1 = tl.load(w_ptr + cols * 4 + 1, mask=cmask, other=0.0)
    w2 = tl.load(w_ptr + cols * 4 + 2, mask=cmask, other=0.0)
    w3 = tl.load(w_ptr + cols * 4 + 3, mask=cmask, other=0.0)
    if pid < NB:
        for j in tl.static_range(3):
            t = pid * 128 + j
            s = tl.sum((q <= t).to(tl.int32)) - 1
            s = tl.minimum(s, S - 1)
            pos = t - tl.load(qsl_ptr + s)
            if (t < total) & (t < M) & (pos >= 3):
                b0 = pid * 128
                ua = t - 3
                ub = t - 2
                uc = t - 1
                ra = tl.where(ua >= b0, pid * 6 + ua - b0, (pid - 1) * 6 + 3 + ua - (b0 - 3))
                rb = tl.where(ub >= b0, pid * 6 + ub - b0, (pid - 1) * 6 + 3 + ub - (b0 - 3))
                rc = tl.where(uc >= b0, pid * 6 + uc - b0, (pid - 1) * 6 + 3 + uc - (b0 - 3))
                rd = pid * 6 + j
                xa = tl.load(side_ptr + ra.to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
                xb = tl.load(side_ptr + rb.to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
                xc = tl.load(side_ptr + rc.to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
                xd = tl.load(side_ptr + rd.to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, xa, xb, xc, xd,
                     part, c0, t, M, P, WITH_NORMS, BLOCK)  # fmt: skip
    else:
        s = pid - NB
        start = tl.load(qsl_ptr + s)
        ln = tl.load(qsl_ptr + s + 1) - start
        line = tl.load(ci_ptr + s * ci_s0)
        ok = ln > 0
        if HAS_NULL:
            ok = ok & (line != null_id)
        if ok:
            init = tl.load(init_ptr + s).to(tl.int1)
            sbase = st_ptr + line.to(tl.int64) * st_s0 + cols.to(tl.int64) * st_s1
            h0 = tl.load(sbase, mask=cmask & init, other=0.0)
            h1 = tl.load(sbase + st_s2, mask=cmask & init, other=0.0)
            h2 = tl.load(sbase + 2 * st_s2, mask=cmask & init, other=0.0)
            r0 = NB * 6 + s * 6
            x0 = tl.load(side_ptr + r0.to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
            x1 = tl.load(side_ptr + (r0 + 1).to(tl.int64) * N + cols, mask=cmask & (ln > 1), other=0.0).to(tl.float32)
            x2 = tl.load(side_ptr + (r0 + 2).to(tl.int64) * N + cols, mask=cmask & (ln > 2), other=0.0).to(tl.float32)
            _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, h0, h1, h2, x0, part, c0, start,
                 M, P, WITH_NORMS, BLOCK)  # fmt: skip
            if ln > 1:
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, h1, h2, x0, x1, part,
                     c0, start + 1, M, P, WITH_NORMS, BLOCK)  # fmt: skip
            if ln > 2:
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, h2, x0, x1, x2, part,
                     c0, start + 2, M, P, WITH_NORMS, BLOCK)  # fmt: skip
            t0 = tl.load(side_ptr + (r0 + 3).to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
            t1 = tl.load(side_ptr + (r0 + 4).to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
            t2 = tl.load(side_ptr + (r0 + 5).to(tl.int64) * N + cols, mask=cmask, other=0.0).to(tl.float32)
            # New state j: input ln - 3 + j, or old state j + ln when that is < 0.
            n0 = tl.where(ln >= 3, t0, tl.where(ln == 2, h2, h1))
            n1 = tl.where(ln >= 2, t1, h2)
            n2 = t2
            tl.store(sbase, n0, mask=cmask)
            tl.store(sbase + st_s2, n1, mask=cmask)
            tl.store(sbase + 2 * st_s2, n2, mask=cmask)


@triton.jit
def _conv_fixup_blocks_kernel(
    out_ptr, side_ptr, norm_ptr, w_ptr, qsl_ptr, M, N, P, NB, S,
    WITH_NORMS: tl.constexpr, BLOCK: tl.constexpr, BG: tl.constexpr,
    SP2: tl.constexpr,
):  # fmt: skip
    """Rows 0-2 of 128-row blocks [BG * pid0, BG * pid0 + BG) that are not
    sequence starts, from the saved rows 125-127 of the block before and
    rows 0-2 of the block (same arithmetic as ``_conv_fixup_kernel``)."""
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < N
    c0 = tl.program_id(1) * BLOCK
    part = c0 // P
    sidx = tl.arange(0, SP2)
    q = tl.load(qsl_ptr + sidx, mask=sidx <= S, other=2147483647)
    total = tl.load(qsl_ptr + S)
    w0 = tl.load(w_ptr + cols * 4 + 0, mask=cmask, other=0.0)
    w1 = tl.load(w_ptr + cols * 4 + 1, mask=cmask, other=0.0)
    w2 = tl.load(w_ptr + cols * 4 + 2, mask=cmask, other=0.0)
    w3 = tl.load(w_ptr + cols * 4 + 3, mask=cmask, other=0.0)
    for bi in tl.static_range(BG):
        b = tl.program_id(0) * BG + bi
        if b < NB:
            b0 = b * 128
            pm = cmask & (b > 0)
            hp = ((b - 1) * 6 + 3).to(tl.int64) * N + cols
            hc = (b * 6).to(tl.int64) * N + cols
            p0 = tl.load(side_ptr + hp, mask=pm, other=0.0).to(tl.float32)
            p1 = tl.load(side_ptr + hp + N, mask=pm, other=0.0).to(tl.float32)
            p2 = tl.load(side_ptr + hp + 2 * N, mask=pm, other=0.0).to(tl.float32)
            x0 = tl.load(side_ptr + hc, mask=cmask, other=0.0).to(tl.float32)
            x1 = tl.load(side_ptr + hc + N, mask=cmask, other=0.0).to(tl.float32)
            x2 = tl.load(side_ptr + hc + 2 * N, mask=cmask, other=0.0).to(tl.float32)
            s0 = tl.minimum(tl.sum((q <= b0).to(tl.int32)) - 1, S - 1)
            s1 = tl.minimum(tl.sum((q <= b0 + 1).to(tl.int32)) - 1, S - 1)
            s2 = tl.minimum(tl.sum((q <= b0 + 2).to(tl.int32)) - 1, S - 1)
            ok0 = (b0 < total) & (b0 < M) & (b0 - tl.load(qsl_ptr + s0) >= 3)
            ok1 = (b0 + 1 < total) & (b0 + 1 < M) & (b0 + 1 - tl.load(qsl_ptr + s1) >= 3)
            ok2 = (b0 + 2 < total) & (b0 + 2 < M) & (b0 + 2 - tl.load(qsl_ptr + s2) >= 3)
            if ok0:
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, p0, p1, p2, x0,
                     part, c0, b0, M, P, WITH_NORMS, BLOCK)  # fmt: skip
            if ok1:
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, p1, p2, x0, x1,
                     part, c0, b0 + 1, M, P, WITH_NORMS, BLOCK)  # fmt: skip
            if ok2:
                _row(out_ptr, norm_ptr, w0, w1, w2, w3, cols, cmask, p2, x0, x1, x2,
                     part, c0, b0 + 2, M, P, WITH_NORMS, BLOCK)  # fmt: skip


def conv_fixup(
    out: torch.Tensor,
    side: torch.Tensor,
    norms: torch.Tensor | None,
    conv_w: torch.Tensor,
    conv_state: torch.Tensor,
    qsl: torch.Tensor,
    lines: torch.Tensor,
    has_init: torch.Tensor,
    part_dim: int,
    null_block_id: int | None,
) -> None:
    _, m, p = out.shape
    n = 3 * p
    nb = (m + 127) // 128
    nseq = qsl.numel() - 1
    block = 1024 if p % 1024 == 0 else 256
    assert p % block == 0 and conv_state.dtype == torch.float32
    sp2 = triton.next_power_of_2(nseq + 1)
    bg = 8
    _conv_fixup_blocks_kernel[((nb + bg - 1) // bg, n // block)](
        out, side, norms if norms is not None else out, conv_w, qsl, m, n, p, nb, nseq,
        WITH_NORMS=norms is not None, BLOCK=block, BG=bg, SP2=sp2, num_warps=4,
    )  # fmt: skip
    _conv_fixup_kernel[(nseq, n // block)](
        out, side, norms if norms is not None else out, conv_w, conv_state, qsl,
        lines, has_init, m, n, p, nb, nseq,
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        lines.stride(0), -1 if null_block_id is None else null_block_id,
        HAS_NULL=null_block_id is not None, WITH_NORMS=norms is not None,
        BLOCK=block, SP2=sp2, SEQ_ONLY=True, num_warps=4,
    )  # fmt: skip
