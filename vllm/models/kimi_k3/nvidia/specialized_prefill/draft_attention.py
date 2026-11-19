# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark GQA drafter's sliding-window attention on a prefill instance: a
handful of query rows per request (the ``k + 1`` draft block) against up to a
window of paged BF16 K/V, split over the keys.

FlashAttention 2 (the drafter's backend on sm_107) runs these calls on one CTA per
(request, query head) that walks the whole window serially (its split heuristic
picks one split once requests x heads fill the SMs), ~44 us per layer for ~2K keys.
Here one program per (request, KV head, key split) holds every query row of its
GQA group (rows x group <= 64) and the splits are merged by their log-sum-exp: the
FlashAttention math (FP32 scores, exp2 online softmax, BF16 probabilities into the
PV product, FP32 accumulation; the summation order differs).
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_N = 64
_SPLITS = 16
_MAX_ROWS = 64


@triton.jit
def _split_kernel(
    Q, K, V, BT, SEQK, CUQ, ACC, ML,
    stride_qt, stride_qh, stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh, stride_bt,
    scale_log2, wl, wr,
    PAGE: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr,
    D: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    heads = tl.num_programs(1)
    q0 = tl.load(CUQ + r)
    sq = tl.load(CUQ + r + 1) - q0
    sk = tl.load(SEQK + r)
    m_idx = tl.arange(0, ROWS)
    tok = m_idx // GROUP
    qh = h * GROUP + m_idx % GROUP
    row_ok = tok < sq
    d = tl.arange(0, D)
    q = tl.load(
        Q + (q0 + tok)[:, None] * stride_qt + qh[:, None] * stride_qh + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    shift = sk - sq
    lo = tl.maximum(shift - wl, 0)
    hi = sk
    chunk = tl.cdiv(tl.cdiv(hi - lo, SPLITS), BLOCK_N) * BLOCK_N
    start = lo + s * chunk
    end = tl.minimum(hi, start + chunk)
    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        j = n0 + n
        j_ok = j < end
        page = tl.load(BT + r * stride_bt + j // PAGE, mask=j_ok, other=0)
        off = j % PAGE
        k = tl.load(
            K + page[:, None].to(tl.int64) * stride_kb + off[:, None] * stride_ks
            + h * stride_kh + d[None, :],
            mask=j_ok[:, None],
            other=0.0,
        )  # fmt: skip
        v = tl.load(
            V + page[:, None].to(tl.int64) * stride_vb + off[:, None] * stride_vs
            + h * stride_vh + d[None, :],
            mask=j_ok[:, None],
            other=0.0,
        )  # fmt: skip
        qk = tl.dot(q, tl.trans(k))
        ok = (
            j_ok[None, :]
            & row_ok[:, None]
            & (j[None, :] >= tok[:, None] + shift - wl)
            & (j[None, :] <= tok[:, None] + shift + wr)
        )
        qk = tl.where(ok, qk * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    base = (r * heads + h) * SPLITS + s
    tl.store(ACC + (base * ROWS + m_idx)[:, None] * D + d[None, :], acc)
    tl.store(ML + (base * 2) * ROWS + m_idx, m_i)
    tl.store(ML + (base * 2 + 1) * ROWS + m_idx, l_i)


@triton.jit
def _merge_kernel(
    OUT, ACC, ML, CUQ, stride_ot, stride_oh,
    GROUP: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0)
    h = tl.program_id(1)
    heads = tl.num_programs(1)
    q0 = tl.load(CUQ + r)
    sq = tl.load(CUQ + r + 1) - q0
    m_idx = tl.arange(0, ROWS)
    tok = m_idx // GROUP
    qh = h * GROUP + m_idx % GROUP
    d = tl.arange(0, D)
    sp = tl.arange(0, SPLITS)
    base = (r * heads + h) * SPLITS + sp
    m = tl.load(ML + (base * 2)[:, None] * ROWS + m_idx[None, :])  # [S, ROWS]
    l_ = tl.load(ML + (base * 2 + 1)[:, None] * ROWS + m_idx[None, :])
    m_max = tl.max(m, 0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.exp2(m - m_max[None, :])
    den = tl.sum(w * l_, 0)
    out = tl.zeros([ROWS, D], tl.float32)
    for i in tl.static_range(SPLITS):
        a = tl.load(ACC + ((r * heads + h) * SPLITS + i) * ROWS * D
                    + m_idx[:, None] * D + d[None, :])  # fmt: skip
        wi = tl.sum(tl.where(sp[:, None] == i, w, 0.0), 0)
        out += a * wi[:, None]
    out = out / den[:, None]
    tl.store(
        OUT + (q0 + tok)[:, None] * stride_ot + qh[:, None] * stride_oh + d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=(tok < sq)[:, None],
    )


def covers(query_lens_max: int, num_heads: int, num_kv_heads: int, head_dim: int,
           key_cache: torch.Tensor, value_cache: torch.Tensor) -> bool:  # fmt: skip
    """Whether :func:`attention` takes a call of this shape: every query row of a
    request's GQA group in one program (<= 64 rows), head_dim 64 / 128, BF16
    paged caches [blocks, page, kv_heads, head_dim] dense in the head."""
    group = num_heads // num_kv_heads
    return (
        num_heads % num_kv_heads == 0
        and query_lens_max * group <= _MAX_ROWS
        and head_dim in (64, 128)
        and key_cache.dtype == value_cache.dtype == torch.bfloat16
        and key_cache.dim() == 4
        and value_cache.shape == key_cache.shape
        and tuple(key_cache.shape[2:]) == (num_kv_heads, head_dim)
        and key_cache.stride(3) == value_cache.stride(3) == 1
    )


def attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    scale: float,
    window: tuple[int, int],
    out: torch.Tensor,
) -> torch.Tensor:
    """Non-causal (FlashAttention local-window convention, bottom-right aligned)
    attention of ``q`` [tokens, heads, head_dim] over the paged caches into ``out``
    [tokens, heads, head_dim]; callers check :func:`covers`."""
    num_reqs = seq_lens.shape[0]
    heads, head_dim = q.shape[1], q.shape[2]
    kv_heads = key_cache.shape[2]
    group = heads // kv_heads
    rows = max(16, triton.next_power_of_2(max_query_len * group))
    acc = torch.empty(
        (num_reqs, kv_heads, _SPLITS, rows, head_dim), dtype=torch.float32,
        device=q.device,
    )  # fmt: skip
    ml = torch.empty(
        (num_reqs, kv_heads, _SPLITS, 2, rows), dtype=torch.float32, device=q.device
    )
    _split_kernel[(num_reqs, kv_heads, _SPLITS)](
        q, key_cache, value_cache, block_table, seq_lens, query_start_loc, acc, ml,
        q.stride(0), q.stride(1), key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),
        value_cache.stride(2), block_table.stride(0),
        scale * 1.4426950408889634, *(w if w >= 0 else 1 << 30 for w in window),
        PAGE=key_cache.shape[1], GROUP=group, ROWS=rows, D=head_dim,
        BLOCK_N=_BLOCK_N, SPLITS=_SPLITS, num_warps=4,
    )  # fmt: skip
    _merge_kernel[(num_reqs, kv_heads)](
        out, acc, ml, query_start_loc, out.stride(0), out.stride(1),
        GROUP=group, ROWS=rows, D=head_dim, SPLITS=_SPLITS, num_warps=4,
    )  # fmt: skip
    return out
