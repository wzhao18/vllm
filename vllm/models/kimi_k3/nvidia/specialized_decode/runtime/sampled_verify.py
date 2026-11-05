# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-decode verification of sampled (temperature 1) and greedy requests over the K3
target's vocabulary-sharded LM head, with the specialized decode on Rubin: the stock
``rejection_sample`` (standard, synthetic and block verification; probabilistic or
one-hot drafts) without gathering the full-vocabulary logits.

Each rank keeps its LM-head shard's logits rows and its columns of the draft-logits
cache (``K3DSparkForCausalLM.sampled_markov_draft`` writes only those). Two exchange
rounds of small per-row entries (cutedsl.verify.exchange, Lamport mailboxes) replace the
all-gather:

1. ``_stats_kernel``: per row, this shard's target max / sum of exp, the draft logits'
   (divided by the temperature) max / sum of exp, the target and draft logits of the
   draft token the row verifies when this rank owns it, and the (value, lowest index)
   argmax key. Exchanged.
2. ``_candidate_kernel``: every row's CTA combines its request's rows (log-sum-exps over
   ranks, the token log-probabilities, the argmax) and makes the stock decisions that
   need no more data (greedy and standard / synthetic acceptance, the block
   verification's cumulative log ratios); then, where a draw can be needed, the
   stock residual of the row (``_resample_kernel``: the target logits for a bonus row
   or a placeholder draft, else ``max(p - q, 0)``, ``p`` scaled by the prefix ratio for
   block verification) with the stock Gumbel draw (``gumbel_noised_argmax``, unsalted
   row position) over the shard: a key; and block verification's residual mass
   partial (``_compute_local_residual_mass_kernel``). Exchanged.
3. ``_finalize_kernel``: per request the winning keys and summed masses, block
   verification's acceptance (``_rejection_kernel``), the sampled tokens, and
   ``get_num_sampled_and_rejected``.

Every elementwise expression is the stock one (same Triton functions); the uniform draws
are the stock ``tl_rand32`` and the Gumbel noise is bitwise the stock noise. The sums
over the vocabulary (log-sum-exps, residual masses) run over each rank's shard and then
over ranks in rank order instead of over 8192-wide blocks: a stated FP32 reorder, so a
draw can differ from the stock sampler's only where two scores tie to the last bits.
Every rank computes identical decisions (same entries, same order). Every kernel is a
programmatic dependent launch (Rubin): it waits for its predecessor before any read
and lets its dependent launch early (the exchanges wait before theirs).
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32

STATS_WORDS = 8  # round 1 entry: t_max t_sum q_max q_sum t_tok q_tok key_lo key_hi
CAND_WORDS = 4  # round 2 entry: key_lo key_hi mass one
CHUNK = 1024  # vocabulary chunk per loop iteration


@triton.jit
def _word(x):
    """An FP32 value as a nonzero u32 word (+0 is sent as -0, which compares equal)."""
    bits = x.to(tl.uint32, bitcast=True)
    return tl.where(bits == 0, 0x80000000, bits)


@triton.jit
def _value(word):
    return word.to(tl.float32, bitcast=True)


@triton.jit
def _key(value, index):
    """(FP32 value, lowest index wins) as a u64 with no zero word (see
    cutedsl.common.gumbel.sampled_key)."""
    bits = value.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 0x80000000, 0, bits)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    ordered = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0xFFFFFFFF, ordered)
    low = (0x7FFFFFFF - index).to(tl.uint64)
    return (ordered.to(tl.uint64) << 31) | low


@triton.jit
def _key_token(key):
    return (0x7FFFFFFF - (key & 0x7FFFFFFF)).to(tl.int64)


@triton.jit
def _pick(vector, offsets, at):
    """``vector[at]`` of a small 1-D block."""
    return tl.sum(tl.where(offsets == at, vector, tl.zeros_like(vector)), axis=0)


@triton.jit
def _stats_kernel(
    entries_ptr,
    draft_tok_ptr,
    row_pos_ptr,
    logits_ptr,
    logits_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col0,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_local_pos_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
    temp_ptr,
    seed_ptr,
    rates_ptr,
    vocab_rank,
    seg_len,
    K: tl.constexpr,
    QP: tl.constexpr,
    SEGS: tl.constexpr,
    SYNTHETIC: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        # Everything below reads the step's products (logits, inputs, sampling state).
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    lo = seg * seg_len
    hi = tl.minimum(lo + seg_len, vocab_rank)
    req = row // (K + 1)
    i = row % (K + 1)
    slot = tl.load(idx_mapping_ptr + req).to(tl.int64)
    temp = tl.load(temp_ptr + slot).to(tl.float32)
    prefill_len = tl.load(prefill_len_ptr + slot)
    token_idx = tl.load(logits_indices_ptr + row)
    if seg == 0:
        tl.store(row_pos_ptr + row, tl.load(positions_ptr + token_idx).to(tl.int64))
    # The draft token this row verifies: the next row's input (gather_draft_sampled:
    # draft slots of a step that starts within the prefill are -1 placeholders).
    draft = tl.full((), -1, tl.int64)
    if i < K:
        next_idx = tl.load(logits_indices_ptr + row + 1)
        token = tl.load(input_ids_ptr + next_idx).to(tl.int64)
        next_pos = tl.load(positions_ptr + next_idx)
        next_local = tl.load(expanded_local_pos_ptr + row + 1)
        placeholder = (next_local > 0) & (next_pos - next_local < prefill_len)
        draft = tl.where(placeholder, -1, token)
    if seg == 0:
        tl.store(draft_tok_ptr + row, draft)

    offsets = tl.arange(0, BLOCK)
    source = logits_ptr + row * logits_stride
    sampled = temp != 0.0
    if SYNTHETIC:
        # Synthetic acceptance reads no logits: of a sampled request only the row of
        # the first rejection (or the bonus row) draws, so only that row needs its
        # statistics (the same decision as _candidate_kernel's).
        rows = tl.arange(0, QP)
        is_draft = rows < K
        base_row = req * (K + 1)
        next_idx = tl.load(logits_indices_ptr + base_row + rows + 1, mask=is_draft)
        next_pos = tl.load(positions_ptr + next_idx, mask=is_draft, other=0)
        next_local = tl.load(
            expanded_local_pos_ptr + base_row + rows + 1, mask=is_draft, other=0
        )
        valid = ~((next_local > 0) & (next_pos - next_local < prefill_len))
        row_idx = tl.load(logits_indices_ptr + base_row + rows, mask=rows < K + 1)
        row_pos = tl.load(positions_ptr + row_idx, mask=rows < K + 1, other=0)
        u = tl_rand32(tl.load(seed_ptr + slot), row_pos, includes_zero=False)
        rate = tl.load(rates_ptr + rows, mask=is_draft, other=0.0)
        ok = (u < rate) & valid
        first_rejection = tl.min(tl.where(is_draft & ~ok, rows, K), axis=0)
        sampled = sampled & (i == first_rejection)
    with_draft = sampled & (i < K)
    draft_row = cache_ptr + slot * cache_stride_0 + i * cache_stride_1 + col0
    # One pass over this segment of the shard: the target max and its lowest index,
    # and for sampled rows the summed exponentials of the target and of the draft
    # logits divided by the temperature (stock _compute_max_and_sumexp, here rescaled
    # online chunk by chunk: a stated FP32 reorder).
    t_max = tl.full((), float("-inf"), tl.float32)
    t_arg = tl.full((), 0, tl.int64)
    t_sum = tl.zeros((), tl.float32)
    q_max = tl.full((), float("-inf"), tl.float32)
    q_sum = tl.zeros((), tl.float32)
    if (temp == 0.0) | sampled:
        for start in range(lo, hi, BLOCK):
            mask = start + offsets < hi
            x = tl.load(source + start + offsets, mask=mask, other=float("-inf"))
            x = x.to(tl.float32)
            value, index = tl.max(x, axis=0, return_indices=True)
            better = value > t_max
            t_arg = tl.where(better, start + index.to(tl.int64), t_arg)
            new_max = tl.maximum(t_max, value)
            if sampled:
                scale = tl.where(t_max == float("-inf"), 0.0, tl.exp(t_max - new_max))
                t_sum = t_sum * scale + tl.sum(tl.exp(x - new_max), axis=0)
            t_max = new_max
            if HAS_DRAFT_LOGITS:  # noqa: SIM102 (a constexpr and a run-time condition)
                if with_draft:
                    # draft_logits is stored pre-temperature (stock: divide on load).
                    q = tl.load(
                        draft_row + start + offsets, mask=mask, other=float("-inf")
                    )
                    q = q.to(tl.float32) / temp
                    q_new = tl.maximum(q_max, tl.max(q, axis=0))
                    scale = tl.where(q_max == float("-inf"), 0.0, tl.exp(q_max - q_new))
                    q_sum = q_sum * scale + tl.sum(tl.exp(q - q_new), axis=0)
                    q_max = q_new
    key = _key(t_max, col0 + t_arg)
    q_max = tl.where(with_draft, q_max, 1.0)
    q_sum = tl.where(with_draft, q_sum, 1.0)
    t_sum = tl.where(sampled, t_sum, 1.0)
    t_tok = tl.full((), float("-inf"), tl.float32)
    q_tok = tl.full((), float("-inf"), tl.float32)
    # The bonus row (i == K) verifies no draft token: draft is -1, never owned.
    owned = sampled & (draft >= col0 + lo) & (draft < col0 + hi)
    local = tl.where(owned, draft - col0, 0)
    if owned:
        t_tok = tl.load(source + local).to(tl.float32)
        if HAS_DRAFT_LOGITS:
            q_tok = tl.load(draft_row + local).to(tl.float32) / temp
    out = entries_ptr + (row * SEGS + seg) * 8
    tl.store(out + 0, _word(t_max))
    tl.store(out + 1, _word(t_sum))
    tl.store(out + 2, _word(q_max))
    tl.store(out + 3, _word(q_sum))
    tl.store(out + 4, _word(t_tok))
    tl.store(out + 5, _word(q_tok))
    tl.store(out + 6, (key & 0xFFFFFFFF).to(tl.uint32))
    tl.store(out + 7, (key >> 32).to(tl.uint32))


@triton.jit
def _row_stats(gathered_ptr, base_row, rows, row_mask, PARTS: tl.constexpr):
    """Every rank's and segment's round-1 entries of a request's rows combined: the
    target and draft log-sum-exps (stock _compute_global_logsumexp over the parts), the
    draft token's logits and the argmax key per row."""
    parts = tl.arange(0, PARTS)
    entry = gathered_ptr + ((base_row + rows)[:, None] * PARTS + parts[None, :]) * 8
    mask = row_mask[:, None]
    t_max = _value(tl.load(entry + 0, mask=mask, other=0))
    t_sum = _value(tl.load(entry + 1, mask=mask, other=0))
    q_max = _value(tl.load(entry + 2, mask=mask, other=0))
    q_sum = _value(tl.load(entry + 3, mask=mask, other=0))
    t_tok = _value(tl.load(entry + 4, mask=mask, other=0))
    q_tok = _value(tl.load(entry + 5, mask=mask, other=0))
    lo = tl.load(entry + 6, mask=mask, other=0).to(tl.uint64)
    hi = tl.load(entry + 7, mask=mask, other=0).to(tl.uint64)
    t_global = tl.max(t_max, axis=1)
    t_lse = t_global + tl.log(tl.sum(t_sum * tl.exp(t_max - t_global[:, None]), axis=1))
    q_global = tl.max(q_max, axis=1)
    q_lse = q_global + tl.log(tl.sum(q_sum * tl.exp(q_max - q_global[:, None]), axis=1))
    argmax = _key_token(tl.max((hi << 32) | lo, axis=1))
    return t_lse, q_lse, tl.max(t_tok, axis=1), tl.max(q_tok, axis=1), argmax


@triton.jit
def _candidate_kernel(
    entries_ptr,
    argmax_ptr,
    accepted_ptr,
    cum_ptr,
    gathered_ptr,
    draft_tok_ptr,
    row_pos_ptr,
    logits_ptr,
    logits_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col0,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    rates_ptr,
    vocab_rank,
    seg_len,
    TP: tl.constexpr,
    SEGS: tl.constexpr,
    SEGS2: tl.constexpr,
    K: tl.constexpr,
    QP: tl.constexpr,
    SYNTHETIC: tl.constexpr,
    BLOCK_VERIFY: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    lo = seg * seg_len
    hi = tl.minimum(lo + seg_len, vocab_rank)
    req = row // (K + 1)
    i = row % (K + 1)
    base_row = req * (K + 1)
    slot = tl.load(idx_mapping_ptr + req).to(tl.int64)
    temp = tl.load(temp_ptr + slot).to(tl.float32)
    seed = tl.load(seed_ptr + slot)
    rows = tl.arange(0, QP)
    row_mask = rows < K + 1
    t_lse, q_lse, t_tok, q_tok, argmax = _row_stats(
        gathered_ptr, base_row, rows, row_mask, TP * SEGS
    )
    drafts = tl.load(draft_tok_ptr + base_row + rows, mask=row_mask, other=-1)
    positions = tl.load(row_pos_ptr + base_row + rows, mask=row_mask, other=0)
    is_draft = rows < K
    valid = drafts >= 0
    target_logprob = t_tok - t_lse
    # One-hot drafts: q(token) = 1, log_q = 0.
    draft_logprob = q_tok - q_lse if HAS_DRAFT_LOGITS else tl.zeros_like(t_tok)
    u = tl_rand32(seed, positions, includes_zero=False)
    if SYNTHETIC:
        rate = tl.load(rates_ptr + rows, mask=is_draft, other=0.0)
    is_greedy = temp == 0.0
    if is_greedy:
        ok = u < rate if SYNTHETIC else argmax == drafts
    elif SYNTHETIC:
        ok = u < rate
    else:
        # Probability ratio test: log p(x) > log(u) + log q(x).
        ok = target_logprob > tl.log(u) + draft_logprob
    ok = ok & valid
    accepted = tl.min(tl.where(is_draft & ~ok, rows, K), axis=0)
    # Block verification's running log ratios (stock _compute_cumulative_log_p_kernel).
    cum = tl.zeros((QP,), tl.float32)
    log_p = tl.zeros((), tl.float32)
    for step in tl.static_range(K):
        if _pick(drafts, rows, step) >= 0:
            ratio = _pick(target_logprob, rows, step) - _pick(draft_logprob, rows, step)
            log_p = tl.minimum(log_p + ratio, 0.0)
        cum = tl.where(rows == step, log_p, cum)
    if seg == 0:
        tl.store(argmax_ptr + row, _pick(argmax, rows, i))
        if i == 0:
            tl.store(accepted_ptr + req, accepted.to(tl.int32))
            tl.store(cum_ptr + req * QP + rows, cum)

    # The row's draw (stock _resample_kernel) and residual mass partial.
    key = _key(tl.full((), float("-inf"), tl.float32), col0)
    mass = tl.zeros((), tl.float32)
    need = temp != 0.0
    if not BLOCK_VERIFY:
        need = need & (i == accepted)
    if need:
        rejected = _pick(drafts, rows, i)
        lse_t = _pick(t_lse, rows, i)
        lse_q = _pick(q_lse, rows, i)
        pos = _pick(positions, rows, i)
        log_p_tau = tl.zeros((), tl.float32)
        if BLOCK_VERIFY:  # noqa: SIM102 (a constexpr and a run-time condition)
            if i > 0:
                log_p_tau = _pick(cum, rows, i - 1)
        raw = (i == K) | (rejected < 0)
        with_mass = (i > 0) & (i < K) & (rejected >= 0)
        prefix = tl.exp(log_p_tau)
        if BLOCK_VERIFY and not HAS_DRAFT_LOGITS:  # noqa: SIM102 (constexpr, run time)
            # One-hot drafts: the closed form p * (1 - M_b(draft token)) (stock
            # _compute_global_residual_mass), sent by rank 0's first segment only
            # (masses are summed over the parts).
            if with_mass & (col0 == 0) & (seg == 0):
                mass = prefix * (1.0 - tl.exp(_pick(target_logprob, rows, i)))
        offsets = tl.arange(0, BLOCK)
        source = logits_ptr + row * logits_stride
        draft_row = cache_ptr + slot * cache_stride_0 + i * cache_stride_1 + col0
        best = tl.full((), float("-inf"), tl.float32)
        best_index = tl.full((), 0, tl.int64)
        for start in range(lo, hi, BLOCK):
            mask = start + offsets < hi
            columns = col0 + start + offsets
            target_logits = tl.load(
                source + start + offsets, mask=mask, other=float("-inf")
            ).to(tl.float32)
            if raw:
                residual_logits = target_logits
            elif HAS_DRAFT_LOGITS:
                draft_logits = (
                    tl.load(
                        draft_row + start + offsets,
                        mask=mask & (i < K),
                        other=float("-inf"),
                    ).to(tl.float32)
                    / temp
                )
                target_log_probs = target_logits - lse_t
                draft_log_probs = draft_logits - lse_q
                if BLOCK_VERIFY and with_mass:
                    m_b = tl.exp(target_log_probs)
                    m_s = tl.exp(draft_log_probs)
                    mass += tl.sum(
                        tl.where(mask, tl.maximum(prefix * m_b - m_s, 0.0), 0.0),
                        axis=0,
                    )
                target_log_probs += log_p_tau
                ratio = tl.exp(draft_log_probs - target_log_probs)
                residual_logits = tl.where(
                    ratio < 1.0,
                    target_log_probs + tldevice.log1p(-ratio),
                    float("-inf"),
                ).to(tl.float32)
            else:
                residual_logits = tl.where(
                    columns != rejected, target_logits, float("-inf")
                ).to(tl.float32)
            value, index = gumbel_noised_argmax(
                residual_logits,
                columns,
                mask,
                seed,
                pos,
                temp,
                IS_DRAFTING=False,
                USE_FP64=False,
                APPLY_TEMPERATURE=False,
            )
            better = value > best
            best_index = tl.where(better, start + index.to(tl.int64), best_index)
            best = tl.where(better, value, best)
        key = _key(best, col0 + best_index)
    out = entries_ptr + (row * SEGS2 + seg) * 4
    tl.store(out + 0, (key & 0xFFFFFFFF).to(tl.uint32))
    tl.store(out + 1, (key >> 32).to(tl.uint32))
    tl.store(out + 2, _word(mass))
    tl.store(out + 3, _word(tl.full((), 1.0, tl.float32)))


@triton.jit
def _finalize_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    gathered_ptr,
    argmax_ptr,
    accepted_ptr,
    cum_ptr,
    draft_tok_ptr,
    row_pos_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    PARTS: tl.constexpr,
    K: tl.constexpr,
    QP: tl.constexpr,
    BLOCK_VERIFY: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    req = tl.program_id(0).to(tl.int64)
    base_row = req * (K + 1)
    slot = tl.load(idx_mapping_ptr + req).to(tl.int64)
    temp = tl.load(temp_ptr + slot).to(tl.float32)
    rows = tl.arange(0, QP)
    row_mask = rows < K + 1
    drafts = tl.load(draft_tok_ptr + base_row + rows, mask=row_mask, other=-1)
    accepted = tl.load(accepted_ptr + req).to(tl.int64)
    parts = tl.arange(0, PARTS)
    entry = gathered_ptr + ((base_row + rows)[:, None] * PARTS + parts[None, :]) * 4
    lo = tl.load(entry + 0, mask=row_mask[:, None], other=0).to(tl.uint64)
    hi = tl.load(entry + 1, mask=row_mask[:, None], other=0).to(tl.uint64)
    winners = _key_token(tl.max((hi << 32) | lo, axis=1))
    if temp == 0.0:
        token = tl.load(argmax_ptr + base_row + accepted)
    else:
        if BLOCK_VERIFY:
            # Block verification (stock _rejection_kernel, Sun et al. 2024).
            masses = tl.sum(
                _value(tl.load(entry + 2, mask=row_mask[:, None], other=0)), axis=1
            )
            cum = tl.load(cum_ptr + req * QP + rows)
            positions = tl.load(row_pos_ptr + base_row + rows, mask=row_mask, other=0)
            seed = tl.load(seed_ptr + slot)
            u = tl_rand32(seed, positions, includes_zero=False)
            accepted = tl.zeros((), tl.int64)
            verifying = tl.full((), 1, tl.int1)
            for i in tl.static_range(K):
                verifying &= _pick(drafts, rows, i) >= 0
                if verifying:
                    prefix_joint_ratio = tl.exp(_pick(cum, rows, i))
                    next_draft = -1
                    if i < K - 1:
                        next_draft = _pick(drafts, rows, i + 1)
                    h = prefix_joint_ratio
                    if next_draft >= 0:
                        residual_mass = _pick(masses, rows, i + 1)
                        denom = residual_mass + 1.0 - prefix_joint_ratio
                        h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                    accepted = tl.where(_pick(u, rows, i) <= h, i + 1, accepted)
        token = _pick(winners, rows, accepted)
    tl.store(
        sampled_ptr + req * sampled_stride + rows,
        tl.where(rows < accepted, drafts, token),
        mask=rows <= accepted,
    )
    num_sampled = (accepted + 1).to(tl.int32)
    num_rejected = (K + 1 - num_sampled).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + req)
    is_chunked_prefilling = seq_len < tl.load(prefill_len_ptr + slot)
    tl.store(num_sampled_ptr + req, tl.where(is_chunked_prefilling, 0, num_sampled))
    tl.store(num_rejected_ptr + req, tl.where(is_chunked_prefilling, 0, num_rejected))


# Vocabulary segments per (row, rank) by logits rows: up to 16 rows (B1, B2) 8, 32
# rows 4, 64 rows 2, else 1 (one program per row and segment in both rounds; more
# programs for few rows, since each scans its share of the shard and draws Gumbel
# noise).
SEGMENTS = ((16, 8), (32, 4), (64, 2))


def segments(rows):
    return next((segs for top, segs in SEGMENTS if rows <= top), 1)


class SampledVerify:
    """Per-rank prepared sampled verification (two row exchanges per segment count);
    built with the specialized decode after the weights load, before CUDA graph
    capture."""

    @classmethod
    def build(cls, vllm_config, model):
        """None unless the draws are FP32 (no fp64 Gumbel). The caller checked
        ``VerifyArgmax.build``'s conditions (Rubin, TP group, plain sharded LM head)."""
        if vllm_config.model_config.use_fp64_gumbel:
            return None
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        shard = model.lm_head.shard_indices
        vocab_rank = shard.org_vocab_end_index - shard.org_vocab_start_index
        device = torch.device(torch.cuda.current_device())
        return cls(
            group.world_size,
            group.rank_in_group,
            group.device_group,
            vocab_rank,
            device,
        )

    def __init__(self, tp, rank, group, vocab_rank, device):
        from ..cutedsl.verify.exchange import PreparedExchange

        self.tp, self.rank, self.vocab_rank = tp, rank, vocab_rank
        self.col0 = rank * vocab_rank
        self.stats, self.candidates = {}, {}
        for segs in sorted({segs for _, segs in SEGMENTS} | {1}):
            words = (STATS_WORDS * segs, CAND_WORDS * segs)
            self.stats[segs] = PreparedExchange(tp, rank, group, words[0], device)
            self.candidates[segs] = PreparedExchange(tp, rank, group, words[1], device)
            self.stats[segs].prepare()
            self.candidates[segs].prepare()

    def takes(self, rows):
        """Whether this verification covers ``rows`` logits rows: any (above the
        exchange's ``MAX_ROWS`` its rounds run in launches of at most MAX_ROWS rows,
        ``PreparedExchange.run``; the kernels around them take any rows). NQ-TUNE: the
        sharded verification vs the stock sampler's full-vocabulary all-gather at 516-
        2048 logits rows (16 ranks, us per step)."""
        return rows > 0

    def run(
        self,
        logits,
        steps,
        input_ids,
        positions,
        logits_indices,
        expanded_local_pos,
        idx_mapping,
        seq_lens,
        prefill_len,
        temperature,
        seeds,
        draft_logits,
        synthetic_rates,
        block_verification,
    ):
        """``logits`` [rows, vocab_rank] this rank's shard of the target logits of
        ``num_reqs`` requests with ``steps`` drafts each (rows = num_reqs x (steps +
        1), uniform); ``draft_logits`` the [max_reqs, steps, V] cache (this rank's
        columns are read) or None for one-hot drafts. Returns (sampled [num_reqs,
        steps + 1] int64, num_sampled, num_rejected [num_reqs] int32)."""
        rows = logits.shape[0]
        num_reqs = idx_mapping.shape[0]
        q = steps + 1
        assert rows == num_reqs * q and logits.stride(-1) == 1
        device = logits.device
        segs = segments(rows)
        qp = triton.next_power_of_2(q)
        synthetic = synthetic_rates is not None
        seg_len = -(-self.vocab_rank // segs)
        seg_len = -(-seg_len // 16) * 16
        has_draft = draft_logits is not None
        cache = draft_logits if has_draft else logits
        stride_0 = draft_logits.stride(0) if has_draft else 0
        stride_1 = draft_logits.stride(1) if has_draft else 0
        entries = torch.empty(
            (rows, STATS_WORDS * segs), dtype=torch.uint32, device=device
        )
        draft_tok = torch.empty(rows, dtype=torch.int64, device=device)
        row_pos = torch.empty(rows, dtype=torch.int64, device=device)
        _stats_kernel[(rows, segs)](
            entries,
            draft_tok,
            row_pos,
            logits,
            logits.stride(0),
            cache,
            stride_0,
            stride_1,
            self.col0,
            input_ids,
            positions,
            logits_indices,
            expanded_local_pos,
            idx_mapping,
            prefill_len,
            temperature,
            seeds,
            synthetic_rates if synthetic else temperature,
            self.vocab_rank,
            seg_len,
            K=steps,
            QP=qp,
            SEGS=segs,
            SYNTHETIC=synthetic,
            HAS_DRAFT_LOGITS=has_draft,
            BLOCK=CHUNK,
            num_warps=4,
            launch_pdl=True,
        )
        gathered = self.stats[segs].run(entries)
        # Round 2: standard / synthetic verification draws one row per request, block
        # verification may need every row's draw.
        segs2 = segments(rows if block_verification else num_reqs)
        seg_len2 = -(-self.vocab_rank // segs2)
        seg_len2 = -(-seg_len2 // 16) * 16
        candidates = torch.empty(
            (rows, CAND_WORDS * segs2), dtype=torch.uint32, device=device
        )
        argmax = torch.empty(rows, dtype=torch.int64, device=device)
        accepted = torch.empty(num_reqs, dtype=torch.int32, device=device)
        cum = torch.empty((num_reqs, qp), dtype=torch.float32, device=device)
        _candidate_kernel[(rows, segs2)](
            candidates,
            argmax,
            accepted,
            cum,
            gathered,
            draft_tok,
            row_pos,
            logits,
            logits.stride(0),
            cache,
            stride_0,
            stride_1,
            self.col0,
            idx_mapping,
            temperature,
            seeds,
            synthetic_rates if synthetic else temperature,
            self.vocab_rank,
            seg_len2,
            TP=self.tp,
            SEGS=segs,
            SEGS2=segs2,
            K=steps,
            QP=qp,
            SYNTHETIC=synthetic,
            BLOCK_VERIFY=block_verification,
            HAS_DRAFT_LOGITS=has_draft,
            BLOCK=CHUNK,
            num_warps=4,
            launch_pdl=True,
        )
        gathered_candidates = self.candidates[segs2].run(candidates)
        sampled = torch.empty((num_reqs, q), dtype=torch.int64, device=device)
        num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _finalize_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            gathered_candidates,
            argmax,
            accepted,
            cum,
            draft_tok,
            row_pos,
            idx_mapping,
            temperature,
            seeds,
            seq_lens,
            prefill_len,
            PARTS=self.tp * segs2,
            K=steps,
            QP=qp,
            BLOCK_VERIFY=block_verification,
            num_warps=1,
            launch_pdl=True,
        )
        return sampled, num_sampled, num_rejected
