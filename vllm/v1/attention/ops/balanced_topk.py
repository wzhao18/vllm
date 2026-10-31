# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-balanced exact top-k over ragged BF16 indexer logits.

The DSA decode top-k kernels (cooperative / persistent / per_row) give each
row a fixed amount of hardware (one CTA or one cluster), so a step's top-k
time is set by its longest row. Under data parallelism one rank holds every
request of a batch, and agentic contexts (a few 500K-800K requests among
many 50K ones) make that one row 10-20x the mean.

This top-k does the long-row work in a pass whose work list is the flattened
(row, block) list over a persistent grid, so its time follows the total
number of logits, not the longest row. It reads the BF16 logits that DeepGEMM
emits on SM10x directly (no FP32 upcast copy). Three launches:

1. ``_sample_threshold`` (one program per row): a candidate threshold
   ``t0`` on the order-preserving 16-bit key, from a strided sample of the
   row, chosen so that about ``TARGET`` keys of the row are expected above
   it (exact for rows of at most ``SAMPLES`` keys).
2. ``_filter`` (balanced; rows longer than ``CAP``): every key above ``t0``
   is appended to the row's candidate list as ``(key << 32) | column``, and
   the columns of keys equal to ``t0`` (the threshold may sit on a large
   tie) to a separate list of at most ``topk``; atomics reserve the slots.
3. ``_select`` (one program per row): the exact k-th largest key ``T`` of
   the candidates, by one 256-bin histogram of ``key - t0 - 1`` (two more,
   high byte then low byte, when ``T`` is more than 255 above ``t0``); keys
   above ``T`` and the first ``k - count(> T)`` keys equal to ``T`` are
   written. Fewer than ``k`` candidates above ``t0`` take ties at ``t0``
   from the tie list. A row the sample misjudged (more than ``CAP``
   candidates, or fewer than ``k`` at or above ``t0``) runs the exact
   selection over the whole row instead: still exact, only slower.

``balanced_topk_cuda.py`` implements the same algorithm in CUDA with
warp-level compaction (about 2x faster on VR); the indexer uses it and falls
back to this Triton version only when the extension cannot be built.

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
    hist_ptr,
    next_n,
    width,
    topk,
    TARGET: tl.constexpr,
    SAMPLES: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    r = tl.program_id(0)
    n = _row_len(seq_lens_ptr, r, next_n, width, PER_ROW_LENS)
    tl.store(counters_ptr + 2 * r, 0)
    tl.store(counters_ptr + 2 * r + 1, 0)
    tl.store(hist_ptr + r * 256 + tl.arange(0, 256), tl.zeros((256,), tl.int32))
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
def _work_table(seq_lens_ptr, num_rows, next_n, width, R_MAX: tl.constexpr,
                CAP: tl.constexpr, BLOCK: tl.constexpr,
                PER_ROW_LENS: tl.constexpr):
    # Blocks per row (rows longer than CAP only) and their inclusive prefix.
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
    blocks = tl.where(valid & (n > CAP), tl.cdiv(n, BLOCK), 0)
    return rows, n, blocks, tl.cumsum(blocks, 0)


@triton.jit
def _locate(w, rows, n, blocks, cum):
    # Row, row length and first work item of the row holding work item w.
    r = tl.sum((cum <= w).to(tl.int32), 0)
    first = tl.sum(tl.where(rows == r, cum - blocks, 0), 0)
    return r, tl.sum(tl.where(rows == r, n, 0), 0), first


@triton.jit
def _filter(
    logits_ptr,
    stride,
    seq_lens_ptr,
    thresh_ptr,
    counters_ptr,
    hist_ptr,
    cand_ptr,
    ties_ptr,
    num_rows,
    next_n,
    width,
    R_MAX: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    TIE_CAP: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    # (row, block) work list over the rows longer than CAP. Keys above t0
    # are appended to the row's candidates as (key << 32) | column; the
    # columns of keys equal to t0 (the sampled threshold may sit on a large
    # tie) to a separate list of at most TIE_CAP (>= topk). One atomic per
    # block reserves each list's slots.
    rows, n, blocks, cum = _work_table(
        seq_lens_ptr, num_rows, next_n, width, R_MAX, CAP, BLOCK, PER_ROW_LENS
    )
    total = tl.sum(blocks, 0)
    for w in range(tl.program_id(0), total, tl.num_programs(0)):
        r, row_n, first = _locate(w, rows, n, blocks, cum)
        t0 = tl.load(thresh_ptr + r)
        row_ptr = logits_ptr + r.to(tl.int64) * stride
        cols, keys, m = _load_keys(row_ptr, (w - first) * BLOCK, row_n, BLOCK)
        gt = m & (keys > t0)
        eq = m & (keys == t0)
        n_gt = tl.sum(gt.to(tl.int32), 0)
        n_eq = tl.sum(eq.to(tl.int32), 0)
        if (n_gt > 0) | (n_eq > 0):
            # Reserve the slots first: the in-block scan below does not need
            # the bases, so it overlaps the atomics' round trips.
            base_gt = tl.atomic_add(counters_ptr + 2 * r, n_gt, sem="relaxed")
            base_eq = tl.atomic_add(counters_ptr + 2 * r + 1, n_eq, sem="relaxed")
            cs = tl.cumsum(gt.to(tl.int32) + (eq.to(tl.int32) << 16), 0)
            pos = base_gt + (cs & 0xFFFF) - 1
            packed = (keys.to(tl.int64) << 32) | cols.to(tl.int64)
            tl.store(cand_ptr + r.to(tl.int64) * CAP + pos, packed,
                     mask=gt & (pos < CAP))
            pos = base_eq + (cs >> 16) - 1
            tl.store(ties_ptr + r * TIE_CAP + pos, cols, mask=eq & (pos < TIE_CAP))


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
    b2, above2 = _kth_bin(h2, tl.sum(h2, 0), topk - above1)
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
def _cand(row_ptr, cand_row, j, num, t0, DIRECT: tl.constexpr, SUB: tl.constexpr):
    # Candidates [j, j + SUB): columns, key - t0 - 1 (>= 0 for keys above
    # t0), and validity (key above t0). DIRECT: the candidates are the row's
    # own first `num` columns.
    if DIRECT:
        cols, keys, valid = _load_keys(row_ptr, j, num, SUB)
        valid = valid & (keys > t0)
    else:
        i = tl.max_contiguous(tl.multiple_of(j + tl.arange(0, SUB), SUB), SUB)
        valid = i < num
        packed = tl.load(cand_row + i, mask=valid, other=0)
        cols = (packed & 0xFFFFFFFF).to(tl.int32)
        keys = (packed >> 32).to(tl.int32)
    return cols, keys - t0 - 1, valid


@triton.jit
def _select_above(row_ptr, cand_row, out_row, num, t0, topk, h,
                  DIRECT: tl.constexpr, SUB: tl.constexpr):
    # Exact top-k among the candidates above t0 (at least topk of them),
    # given h, their histogram of min(key - t0 - 1, 255): when the k-th key
    # is within 255 of t0 it is found from h; otherwise from the delta's
    # high, then low byte.
    total = tl.sum(h, 0)
    b, num_gt = _kth_bin(h, total, topk)
    if b == 255:
        h1 = tl.zeros((256,), dtype=tl.int32)
        for j in range(0, num, SUB):
            cols, d, valid = _cand(row_ptr, cand_row, j, num, t0, DIRECT, SUB)
            h1 += _hist(d >> 8, valid, SUB)
        b1, above1 = _kth_bin(h1, total, topk)
        h2 = tl.zeros((256,), dtype=tl.int32)
        for j in range(0, num, SUB):
            cols, d, valid = _cand(row_ptr, cand_row, j, num, t0, DIRECT, SUB)
            h2 += _hist(d & 0xFF, valid & ((d >> 8) == b1), SUB)
        b2, above2 = _kth_bin(h2, tl.sum(h2, 0), topk - above1)
        b = b1 * 256 + b2
        num_gt = above1 + above2
    n_gt = 0
    n_eq = 0
    for j in range(0, num, SUB):
        cols, d, valid = _cand(row_ptr, cand_row, j, num, t0, DIRECT, SUB)
        n_gt, n_eq = _write(
            out_row, cols, valid & (d > b), valid & (d == b), n_gt, n_eq,
            num_gt, topk - num_gt,
        )


@triton.jit
def _select(
    logits_ptr,
    stride,
    seq_lens_ptr,
    thresh_ptr,
    counters_ptr,
    hist_ptr,
    cand_ptr,
    ties_ptr,
    out_ptr,
    out_stride,
    next_n,
    width,
    topk,
    CAP: tl.constexpr,
    SUB: tl.constexpr,
    TOPK_P2: tl.constexpr,
    TIE_CAP: tl.constexpr,
    PER_ROW_LENS: tl.constexpr,
):
    r = tl.program_id(0)
    n = _row_len(seq_lens_ptr, r, next_n, width, PER_ROW_LENS)
    out_row = out_ptr + r.to(tl.int64) * out_stride
    row_ptr = logits_ptr + r.to(tl.int64) * stride
    k_offs = tl.arange(0, TOPK_P2)
    if n <= topk:
        # Every column, in order, then the -1 tail.
        tl.store(out_row + k_offs, tl.where(k_offs < n, k_offs, -1), mask=k_offs < topk)
    else:
        t0 = tl.load(thresh_ptr + r)
        if n <= CAP:
            # Short row: its own columns are the candidates.
            num_gt = 0
            num_eq = 0
            h = tl.zeros((256,), dtype=tl.int32)
            for j in range(0, n, SUB):
                cols, keys, valid = _load_keys(row_ptr, j, n, SUB)
                gt = valid & (keys > t0)
                num_gt += tl.sum(gt.to(tl.int32), 0)
                num_eq += tl.sum((valid & (keys == t0)).to(tl.int32), 0)
                h += _hist(tl.minimum(keys - t0 - 1, 255), gt, SUB)
            if num_gt >= topk:
                _select_above(row_ptr, cand_ptr, out_row, n, t0, topk, h, True, SUB)
            elif num_gt + num_eq >= topk:
                # Everything above t0, then ties at t0 in column order.
                n_gt = 0
                n_eq = 0
                for j in range(0, n, SUB):
                    cols, keys, valid = _load_keys(row_ptr, j, n, SUB)
                    n_gt, n_eq = _write(
                        out_row, cols, valid & (keys > t0), valid & (keys == t0),
                        n_gt, n_eq, num_gt, topk - num_gt,
                    )
            else:
                _select_exact(row_ptr, out_row, n, topk, SUB)
        else:
            num_gt = tl.load(counters_ptr + 2 * r)
            num_eq = tl.load(counters_ptr + 2 * r + 1)
            cand_row = cand_ptr + r.to(tl.int64) * CAP
            if (num_gt >= topk) & (num_gt <= CAP):
                h = tl.zeros((256,), dtype=tl.int32)
                for j in range(0, num_gt, SUB):
                    cols, d, valid = _cand(row_ptr, cand_row, j, num_gt, t0, False, SUB)
                    h += _hist(tl.minimum(d, 255), valid, SUB)
                _select_above(row_ptr, cand_row, out_row, num_gt, t0, topk, h,
                              False, SUB)
            elif (num_gt < topk) & (num_gt + num_eq >= topk):
                # Every candidate above t0, then topk - num_gt ties at t0.
                for j in range(0, num_gt, SUB):
                    i = j + tl.arange(0, SUB)
                    valid = i < num_gt
                    packed = tl.load(cand_row + i, mask=valid, other=0)
                    tl.store(out_row + i, (packed & 0xFFFFFFFF).to(tl.int32),
                             mask=valid)
                ties = tl.load(ties_ptr + r * TIE_CAP + k_offs,
                               mask=k_offs < topk - num_gt)
                tl.store(out_row + num_gt + k_offs, ties, mask=k_offs < topk - num_gt)
            else:
                # The sample misjudged t0: exact selection over the whole row.
                _select_exact(row_ptr, out_row, n, topk, SUB)


def balanced_topk_supported(logits: torch.Tensor, topk: int) -> bool:
    """Whether ``balanced_topk`` accepts these logits."""
    return (
        current_platform.is_cuda()
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.stride(1) == 1
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
    # Every scratch word is written before it is read (the counters are
    # zeroed by _sample_threshold), so nothing needs a memset.
    tie_cap = triton.next_power_of_2(topk)
    cand = torch.empty(num_rows * CAP, dtype=torch.int64, device=dev)
    scratch = torch.empty(num_rows * (tie_cap + 3 + 256), dtype=torch.int32, device=dev)
    ties, thresh, counters, hist = scratch.split(
        [num_rows * tie_cap, num_rows, 2 * num_rows, 256 * num_rows]
    )
    common = dict(PER_ROW_LENS=per_row)
    _sample_threshold[(num_rows,)](
        logits, logits.stride(0), seq_lens, thresh, counters, hist, next_n, width, topk,
        TARGET=TARGET, SAMPLES=SAMPLES, num_warps=4, **common,
    )
    _filter[(programs_per_sm * num_compute_units(dev.index),)](
        logits, logits.stride(0), seq_lens, thresh, counters, hist, cand, ties,
        num_rows, next_n, width, R_MAX=triton.next_power_of_2(num_rows), CAP=CAP,
        BLOCK=block, TIE_CAP=tie_cap, num_warps=filter_warps, **common,
    )
    _select[(num_rows,)](
        logits, logits.stride(0), seq_lens, thresh, counters, hist, cand, ties,
        topk_indices, topk_indices.stride(0), next_n, width, topk, CAP=CAP,
        SUB=sub, TOPK_P2=tie_cap, TIE_CAP=tie_cap, num_warps=select_warps,
        **common,
    )
