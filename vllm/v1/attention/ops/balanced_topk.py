# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-balanced exact top-k over ragged BF16 indexer logits.

The DSA decode top-k kernels (cooperative / persistent / per_row) give each
row a fixed amount of hardware (one CTA or one cluster), so a step's top-k
time is set by its longest row. Under data parallelism one rank holds every
request of a batch, and agentic contexts (a few 500K-800K requests among
many 50K ones) make that one row 10-20x the mean.

This top-k does the long-row work in a pass whose work list is the flattened
(row, chunk) list over a persistent grid, so its time follows the total
number of logits, not the longest row. It reads the BF16 logits that DeepGEMM
emits on SM10x directly (no FP32 upcast copy). Three launches:

1. ``_sample_threshold`` (one program per row): a candidate threshold
   ``t0`` on the order-preserving 16-bit key, from a strided sample of the
   row, chosen so that about ``TARGET`` keys of the row are expected at or
   above it (exact for rows of at most ``SAMPLES`` keys).
2. ``_filter`` (balanced; rows longer than ``CAP``): every key ``>= t0`` is
   appended to the row's candidate list as ``(min(key - t0, 4095) << 20) |
   column``; one atomic per block reserves the slots.
3. ``_select`` (one program per row): one 256-bin histogram of the
   candidates' ``key - t0`` gives the k-th largest key ``T`` exactly
   (whenever it lies within 255 of ``t0``, the common case); keys above
   ``T`` and the first ``k - count(> T)`` keys equal to ``T`` are written.
   Any other case (the sample misjudged ``t0``: fewer than k candidates,
   more than ``CAP``, or ``T`` far above ``t0``) runs an exact two-level
   (high byte, low byte) histogram selection over the whole row instead:
   still exact, only slower.

Output contract (same as the other decode top-k backends): row ``r`` holds
``min(k, len_r)`` column indices of ``logits[r, :len_r]`` in unspecified
order, then ``-1``. The selected multiset of values is exactly the top-k;
which of several values equal to the threshold is taken is unspecified (as
with cooperative_topk, whose tie scatter is also atomic). Rows no longer
than ``k`` take every column, in order.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

# Candidate capacity per row; rows of at most CAP logits skip the filter.
CAP = 16384
# Expected candidates per filtered row (sample quantile target).
TARGET = 8192
# Strided sample size per row.
SAMPLES = 4096
# Logits per (row, chunk) work item of the filter.
CHUNK = 8192
# Packed candidate: column in the low COL_BITS, key - t0 (clamped) above.
COL_BITS = 20
MAX_DELTA = (1 << (31 - COL_BITS)) - 1


@triton.jit
def _row_len(seq_lens_ptr, r, next_n, width, PER_ROW_LENS: tl.constexpr):
    # Effective length of logits row r (int32), clamped to [0, width].
    if PER_ROW_LENS:
        n = tl.load(seq_lens_ptr + r)
    else:
        q = r // next_n
        n = tl.load(seq_lens_ptr + q) - next_n + 1 + (r - q * next_n)
    return tl.minimum(tl.maximum(n, 0), width)


@triton.jit
def _keys(x):
    # Order-preserving 16-bit key of BF16 bits: a larger value gives a
    # larger key (negative values flip all bits, positive set the sign bit).
    u = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return tl.where(u >= 0x8000, u ^ 0xFFFF, u | 0x8000)


@triton.jit
def _load_keys(row_ptr, j, n, SUB: tl.constexpr):
    # Columns [j, j + SUB) of a row and their keys (j a multiple of SUB).
    cols = tl.max_contiguous(tl.multiple_of(j + tl.arange(0, SUB), SUB), SUB)
    valid = cols < n
    if j + SUB <= n:
        keys = _keys(tl.load(row_ptr + cols))
    else:
        keys = _keys(tl.load(row_ptr + cols, mask=valid, other=0.0))
    return cols, keys, valid


@triton.jit
def _kth_bin(h, total, need):
    # Largest bin b with count(bins >= b) >= need (1 <= need <= total), and
    # count(bins > b).
    bins = tl.arange(0, 256)
    ge = total - tl.cumsum(h, 0) + h
    b = tl.max(tl.where(ge >= need, bins, -1), 0)
    return b, tl.sum(tl.where(bins > b, h, 0), 0)


@triton.jit
def _hist(vals, valid, N: tl.constexpr):
    # 256-bin histogram of vals (in [0, 256)) over the valid lanes.
    bins = tl.arange(0, 256)
    h = tl.histogram(tl.where(valid, vals, 0), 256)
    return h - tl.where(bins == 0, N - tl.sum(valid.to(tl.int32), 0), 0)


@triton.jit
def _sample_threshold(
    logits_ptr,
    stride,
    seq_lens_ptr,
    thresh_ptr,
    counters_ptr,
    next_n,
    width,
    topk,
    TARGET: tl.constexpr,
    SAMPLES: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    r = tl.program_id(0)
    n = _row_len(seq_lens_ptr, r, next_n, width, PER_ROW_LENS)
    tl.store(counters_ptr + r, 0)
    if n > topk:
        i = tl.arange(0, SAMPLES)
        row_ptr = logits_ptr + r.to(tl.int64) * stride
        if n <= SAMPLES:
            # The whole row: t0 is the exact k-th largest key.
            valid = i < n
            keys = _keys(tl.load(row_ptr + i, mask=valid, other=0.0))
            need = topk
        else:
            # Strided positions, the middle of each of SAMPLES segments.
            valid = i < SAMPLES
            pos = ((2 * i + 1).to(tl.int64) * n // (2 * SAMPLES)).to(tl.int32)
            keys = _keys(tl.load(row_ptr + pos))
            need = tl.minimum(tl.maximum(tl.cdiv(TARGET * SAMPLES, n), 1), SAMPLES)
        num = tl.minimum(n, SAMPLES)
        h1 = _hist(keys >> 8, valid, SAMPLES)
        b1, above1 = _kth_bin(h1, num, need)
        in_b1 = valid & ((keys >> 8) == b1)
        h2 = _hist(keys & 0xFF, in_b1, SAMPLES)
        b2, _ = _kth_bin(h2, tl.sum(in_b1.to(tl.int32), 0), need - above1)
        tl.store(thresh_ptr + r, b1 * 256 + b2)


@triton.jit
def _filter(
    logits_ptr,
    stride,
    seq_lens_ptr,
    thresh_ptr,
    counters_ptr,
    cand_ptr,
    num_rows,
    next_n,
    width,
    R_MAX: tl.constexpr,
    CAP: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    COL_BITS: tl.constexpr,
    MAX_DELTA: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    # Flattened (row, chunk) work list over the rows longer than CAP.
    rows = tl.arange(0, R_MAX)
    valid = rows < num_rows
    if PER_ROW_LENS:
        n = tl.load(seq_lens_ptr + rows, mask=valid, other=0)
    else:
        q = rows // next_n
        n = tl.load(seq_lens_ptr + q, mask=valid, other=0) - next_n + 1 + (
            rows - q * next_n
        )
    n = tl.minimum(tl.maximum(n, 0), width)
    chunks = tl.where(valid & (n > CAP), tl.cdiv(n, CHUNK), 0)
    cum = tl.cumsum(chunks, 0)
    total = tl.sum(chunks, 0)
    for w in range(tl.program_id(0), total, tl.num_programs(0)):
        r = tl.sum((cum <= w).to(tl.int32), 0)
        c = w - tl.sum(tl.where(rows == r, cum - chunks, 0), 0)
        row_n = tl.sum(tl.where(rows == r, n, 0), 0)
        t0 = tl.load(thresh_ptr + r)
        row_ptr = logits_ptr + r.to(tl.int64) * stride
        for s in range(c * CHUNK, tl.minimum((c + 1) * CHUNK, row_n), BLOCK):
            cols, keys, m = _load_keys(row_ptr, s, row_n, BLOCK)
            sel = m & (keys >= t0)
            cnt = tl.sum(sel.to(tl.int32), 0)
            if cnt > 0:
                base = tl.atomic_add(counters_ptr + r, cnt, sem="relaxed")
                pos = base + tl.cumsum(sel.to(tl.int32), 0) - 1
                packed = (tl.minimum(keys - t0, MAX_DELTA) << COL_BITS) | cols
                tl.store(cand_ptr + r * CAP + pos, packed, mask=sel & (pos < CAP))


@triton.jit
def _write(out_row, cols, gt, eq, n_gt, n_eq, num_gt, ties):
    # Keys above the threshold go to [0, num_gt) in order of arrival, the
    # first `ties` keys equal to it to [num_gt, topk). One packed scan.
    cs = tl.cumsum(gt.to(tl.int32) + (eq.to(tl.int32) << 16), 0)
    pos = n_gt + (cs & 0xFFFF) - 1
    rank = n_eq + (cs >> 16) - 1
    tl.store(out_row + pos, cols, mask=gt)
    tl.store(out_row + num_gt + rank, cols, mask=eq & (rank < ties))
    last = tl.max(cs, 0)
    return n_gt + (last & 0xFFFF), n_eq + (last >> 16)


@triton.jit
def _select_exact(row_ptr, out_row, n, topk, SUB: tl.constexpr):
    # Exact top-k over a whole row by its high-byte, then low-byte histogram.
    h1 = tl.zeros((256,), dtype=tl.int32)
    for j in range(0, n, SUB):
        cols, keys, valid = _load_keys(row_ptr, j, n, SUB)
        h1 += _hist(keys >> 8, valid, SUB)
    b1, above1 = _kth_bin(h1, n, topk)
    h2 = tl.zeros((256,), dtype=tl.int32)
    for j in range(0, n, SUB):
        cols, keys, valid = _load_keys(row_ptr, j, n, SUB)
        h2 += _hist(keys & 0xFF, valid & ((keys >> 8) == b1), SUB)
    in_b1 = tl.sum(h2, 0)
    b2, above2 = _kth_bin(h2, in_b1, topk - above1)
    t = b1 * 256 + b2
    num_gt = above1 + above2
    n_gt = 0
    n_eq = 0
    for j in range(0, n, SUB):
        cols, keys, valid = _load_keys(row_ptr, j, n, SUB)
        n_gt, n_eq = _write(
            out_row, cols, valid & (keys > t), valid & (keys == t), n_gt, n_eq,
            num_gt, topk - num_gt,
        )


@triton.jit
def _cand(row_ptr, cand_row, j, num, t0, DIRECT: tl.constexpr, SUB: tl.constexpr,
          COL_BITS: tl.constexpr, MAX_DELTA: tl.constexpr):
    # Candidates [j, j + SUB): columns, key - t0 (clamped), validity. DIRECT:
    # the candidates are the row's own first `num` columns at or above t0.
    if DIRECT:
        cols, keys, valid = _load_keys(row_ptr, j, num, SUB)
        valid = valid & (keys >= t0)
        delta = tl.minimum(keys - t0, MAX_DELTA)
    else:
        i = tl.max_contiguous(tl.multiple_of(j + tl.arange(0, SUB), SUB), SUB)
        valid = i < num
        packed = tl.load(cand_row + i, mask=valid, other=0)
        cols = packed & ((1 << COL_BITS) - 1)
        delta = packed >> COL_BITS
    return cols, delta, valid


@triton.jit
def _select_cands(row_ptr, cand_row, out_row, num, t0, topk,
                  DIRECT: tl.constexpr, SUB: tl.constexpr,
                  COL_BITS: tl.constexpr, MAX_DELTA: tl.constexpr):
    # Returns False when the fast path does not apply (the caller falls back
    # to _select_exact): fewer than topk candidates, or the k-th largest more
    # than 254 above t0.
    h = tl.zeros((256,), dtype=tl.int32)
    total = 0
    for j in range(0, num, SUB):
        cols, delta, valid = _cand(row_ptr, cand_row, j, num, t0, DIRECT, SUB,
                                   COL_BITS, MAX_DELTA)
        h += _hist(tl.minimum(delta, 255), valid, SUB)
    total = tl.sum(h, 0)
    ok = total >= topk
    if ok:
        b, num_gt = _kth_bin(h, total, topk)
        ok = b < 255
        if ok:
            n_gt = 0
            n_eq = 0
            for j in range(0, num, SUB):
                cols, delta, valid = _cand(row_ptr, cand_row, j, num, t0, DIRECT,
                                           SUB, COL_BITS, MAX_DELTA)
                n_gt, n_eq = _write(
                    out_row, cols, valid & (delta > b), valid & (delta == b),
                    n_gt, n_eq, num_gt, topk - num_gt,
                )
    return ok


@triton.jit
def _select(
    logits_ptr,
    stride,
    seq_lens_ptr,
    thresh_ptr,
    counters_ptr,
    cand_ptr,
    out_ptr,
    out_stride,
    next_n,
    width,
    topk,
    CAP: tl.constexpr,
    SUB: tl.constexpr,
    TOPK_P2: tl.constexpr,
    COL_BITS: tl.constexpr,
    MAX_DELTA: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    r = tl.program_id(0)
    n = _row_len(seq_lens_ptr, r, next_n, width, PER_ROW_LENS)
    out_row = out_ptr + r.to(tl.int64) * out_stride
    row_ptr = logits_ptr + r.to(tl.int64) * stride
    if n <= topk:
        # Every column, in order, then the -1 tail.
        k_offs = tl.arange(0, TOPK_P2)
        tl.store(out_row + k_offs, tl.where(k_offs < n, k_offs, -1), mask=k_offs < topk)
    else:
        t0 = tl.load(thresh_ptr + r)
        if n <= CAP:
            ok = _select_cands(row_ptr, cand_ptr, out_row, n, t0, topk, True, SUB,
                               COL_BITS, MAX_DELTA)
        else:
            num = tl.load(counters_ptr + r)
            ok = num <= CAP
            if ok:
                ok = _select_cands(row_ptr, cand_ptr + r * CAP, out_row, num, t0,
                                   topk, False, SUB, COL_BITS, MAX_DELTA)
        if ok == 0:
            # The sample misjudged t0: exact selection over the whole row.
            _select_exact(row_ptr, out_row, n, topk, SUB)


def balanced_topk_supported(logits: torch.Tensor, topk: int) -> bool:
    """Whether ``balanced_topk`` accepts these logits."""
    return (
        current_platform.is_cuda()
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.stride(1) == 1
        and logits.shape[1] <= 1 << COL_BITS
        and 0 < topk <= TARGET // 2
    )


def balanced_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk_indices: torch.Tensor,
    topk: int,
    programs_per_sm: int = 8,
    block: int = 2048,
    filter_warps: int = 4,
    sub: int = 4096,
    select_warps: int = 8,
) -> None:
    """Exact per-row top-k of ragged BF16 logits into ``topk_indices``.

    Args:
        logits: (num_rows, width) BF16, stride(1) == 1; row ``r`` is valid in
            ``[0, len_r)``.
        seq_lens: int32 lengths. Either one per row (``numel == num_rows``),
            or one per request, (B, 1) or (B,), with ``num_rows == B *
            next_n`` and row ``r`` of request ``q = r // next_n`` valid up to
            ``seq_lens[q] - next_n + 1 + r % next_n`` (the causal spec-decode
            rows, as in the other decode top-k kernels).
        next_n: rows per request (1 + speculative tokens).
        topk_indices: (num_rows, >= topk) int32 output; columns ``[0, topk)``
            are written.
        topk: tokens selected per row.
        programs_per_sm, block, filter_warps, sub, select_warps: launch
            tuning (filter grid, filter block, selection block).
    """
    assert balanced_topk_supported(logits, topk), (logits.dtype, logits.shape)
    num_rows, width = logits.shape
    if num_rows == 0:
        return
    assert topk_indices.dtype == torch.int32 and topk_indices.stride(1) == 1
    assert seq_lens.dtype == torch.int32
    per_row = seq_lens.numel() == num_rows
    if not per_row:
        assert seq_lens.numel() * next_n == num_rows, (seq_lens.shape, num_rows)
    seq_lens = seq_lens.contiguous()
    dev = logits.device
    # Every scratch word is written before it is read (counters and
    # thresholds by _sample_threshold), so nothing needs a memset.
    scratch = torch.empty(num_rows * (CAP + 2), dtype=torch.int32, device=dev)
    cand = scratch[: num_rows * CAP]
    thresh = scratch[num_rows * CAP : num_rows * (CAP + 1)]
    counters = scratch[num_rows * (CAP + 1) :]
    _sample_threshold[(num_rows,)](
        logits,
        logits.stride(0),
        seq_lens,
        thresh,
        counters,
        next_n,
        width,
        topk,
        TARGET=TARGET,
        SAMPLES=SAMPLES,
        PER_ROW_LENS=per_row,
        num_warps=4,
    )
    _filter[(programs_per_sm * num_compute_units(dev.index),)](
        logits,
        logits.stride(0),
        seq_lens,
        thresh,
        counters,
        cand,
        num_rows,
        next_n,
        width,
        R_MAX=triton.next_power_of_2(num_rows),
        CAP=CAP,
        CHUNK=CHUNK,
        BLOCK=block,
        COL_BITS=COL_BITS,
        MAX_DELTA=MAX_DELTA,
        PER_ROW_LENS=per_row,
        num_warps=filter_warps,
    )
    _select[(num_rows,)](
        logits,
        logits.stride(0),
        seq_lens,
        thresh,
        counters,
        cand,
        topk_indices,
        topk_indices.stride(0),
        next_n,
        width,
        topk,
        CAP=CAP,
        SUB=sub,
        TOPK_P2=triton.next_power_of_2(topk),
        COL_BITS=COL_BITS,
        MAX_DELTA=MAX_DELTA,
        PER_ROW_LENS=per_row,
        num_warps=select_warps,
    )
