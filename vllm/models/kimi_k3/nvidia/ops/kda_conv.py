# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill short convolution over the packed Q|K|V projection in one launch.

On SM10x the conv runs ``qkv_conv`` (CUDA, ops/kda_qkv_conv): one persistent,
balanced launch per layer that streams the packed Q|K|V columns through a shared
memory ring filled by bulk async copies, with the stock arithmetic (bitwise equal
to ``causal_conv1d_fn`` for a bf16 or FP32 conv state). It replaces the three
per-plane Triton calls below (VR200, C512 n17k: 129 -> 61 us per layer isolated;
the Triton kernels are instruction-issue bound at 68% SM throughput). The Triton
kernels remain for shapes it does not cover (channels not a multiple of 256,
unaligned rows, other GPUs).

The stock prefill runs ``causal_conv1d_fn`` once per Q, K and V. Each call first
casts the row-strided bf16 slice of the in_proj output to a dense FP32 copy (the
conv state dtype), convolves it into an FP32 output and casts that back to bf16:
per layer and rank at a 16K chunk that is 3 x (76 + 52 + 19) us, of which only
the convolution reads and writes the data the layer needs.

This kernel reads the bf16 Q|K|V columns of the in_proj output in place, does the
stock arithmetic in FP32 (bf16 -> FP32 is exact; same product order, FP32 weights,
FP32 conv state, SiLU), and stores bf16 Q, K and V as three dense ``(tokens, dim)``
planes of one buffer, the layout the stock casts produce. FP32 -> bf16 rounds to
nearest even, as the stock ``.to(bfloat16)`` does, so the outputs and the updated
conv state are bit-identical to the stock path.

Programs map to (sequence, token chunk) on the device by a radix-32 search over
``query_start_loc``, so no host-side chunk table is needed and the launch is CUDA
graph safe. A program streams its tokens over one channel vector with the 3
history rows in registers and the next 8 input rows prefetched in a register
queue. As in the stock kernel, only a sequence's first chunk reads
its conv state, and that same program then writes the updated state.

The per-plane kernel below (``supported`` / ``kda_prefill_conv``):

Kimi-K3 KDA prefill short convolution (width 4, SiLU), bitwise equal to
``causal_conv1d_fn`` (vllm/model_executor/layers/mamba/ops/causal_conv1d.py).

The generic kernel walks each program's 8 tokens one after another, one
dependent [BLOCK_N] global load per iteration, with 256 channels per program:
at the 16k prefill shape it runs 37.5 us per q / k / v call (1 GPU; 42 us
in-model) against a ~12-14 us strided-copy floor. Here the program keeps the
generic kernel's mapping (the metadata's BLOCK_M = 8 (sequence, chunk) lists),
1D [BLOCK_N] layout, taps in the same order (acc = 0; acc += x[t-3] * w0; ...;
acc += x[t] * w3, bf16 products into an fp32 accumulator), SiLU expression and
conv-state update (by each sequence's chunk-0 program, after it has read the
old state). What changes: the 8 row loads are issued up front and the 8 token
bodies are unrolled, and a program covers 512 channels (4 warps): 18.6 us.
"""


import hashlib
import os
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

logger = init_logger(__name__)


def _config(part_dim: int) -> tuple[int, int, int, int]:
    """(BLOCK_M unrolled tokens, NUM_BLOCKS blocks per program, BLOCK_N channels,
    warps). BLOCK_N divides part_dim so that a program writes one plane."""
    # Measured on GB300 at 2K-16K tokens x 3 x 1536 channels: one warp per
    # 128 channels and 64-token programs stream at ~4.8 TB/s of read + write.
    return 8, 8, 128 if part_dim % 128 == 0 else 64, 1


@triton.jit
def _load_row(x_row, r, end, stride_x_token):
    return tl.load(x_row + r * stride_x_token, mask=r < end, other=0.0)


@triton.jit
def _packed_causal_conv1d_fwd_kernel(
    x_ptr,  # (tokens, dim) channel-contiguous, any row stride
    w_ptr,  # (dim, 4), FP32
    bias_ptr,  # (dim,) or None
    state_ptr,  # (cache lines, dim, >= 3), FP32
    cache_indices_ptr,  # (seqs,)
    has_initial_state_ptr,  # (seqs,)
    query_start_loc_ptr,  # (seqs + 1,)
    out_ptr,  # (dim // part_dim, tokens, part_dim)
    num_seqs,
    search_steps,
    search_stride,
    num_col_blocks,
    part_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    stride_cache_indices,
    stride_out_part,
    stride_out_token,
    null_block_id,
    HAS_BIAS: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Width-4 taps: 3 history rows. Every tensor is one [BLOCK_N] vector in one
    # layout, so each thread owns its channels for the whole program.
    STATE_LEN: tl.constexpr = 3
    CHUNK: tl.constexpr = BLOCK_M * NUM_BLOCKS
    # The channel block varies fastest: CTAs running together cover whole rows.
    pid = tl.program_id(0) // num_col_blocks
    col_block = tl.program_id(0) % num_col_blocks

    # This program's (sequence, chunk). Sequence s owns the programs [g(s), g(s+1))
    # with g(s) = (query_start_loc[s] - query_start_loc[0]) // CHUNK + s: g strictly
    # increases, a sequence of n tokens gets at least cdiv(n, CHUNK) programs, and
    # g(num_seqs) <= cdiv(tokens, CHUNK) + num_seqs = the grid's rows. A radix-32
    # search finds the last s with g(s) <= pid: one 32-wide parallel load per level,
    # one level up to 32 sequences and two up to 1024 (a linear scan costs O(seqs)
    # and was slower than stock at ~500+ sequences; a binary search's dependent
    # loads cost ~4 us per layer at 4-32 sequences).
    qsl0 = tl.load(query_start_loc_ptr)
    lo = pid * 0
    stride = search_stride
    for _ in range(0, search_steps):
        idx = lo + tl.arange(0, 32) * stride
        in_range = idx < num_seqs
        g = (
            tl.load(query_start_loc_ptr + idx, mask=in_range, other=0) - qsl0
        ) // CHUNK + idx
        lo += (tl.sum((in_range & (g <= pid)).to(tl.int32)) - 1) * stride
        stride = stride // 32
    seq = lo
    seq_start_i = tl.load(query_start_loc_ptr + seq)
    chunk = pid - ((seq_start_i - qsl0) // CHUNK + seq)
    seqlen = tl.load(query_start_loc_ptr + seq + 1) - seq_start_i
    if chunk * CHUNK >= seqlen:  # spare program (also: empty sequences)
        return

    seq_start = seq_start_i.to(tl.int64)
    slot = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa: SIM102 (constexpr guard of a runtime branch)
        if slot == null_block_id:
            return
    load_init = tl.load(has_initial_state_ptr + seq).to(tl.int1)

    col0 = col_block * BLOCK_N
    feats = col0 + tl.arange(0, BLOCK_N)
    x_row = x_ptr + seq_start * stride_x_token + feats
    state_base = state_ptr + slot * stride_state_seq + feats * stride_state_dim
    row0 = chunk * CHUNK
    end = tl.minimum(seqlen, row0 + CHUNK)

    # Rows are prefetched 8 ahead in a register queue: a row's load is issued 8
    # rows before its use. (In program order a load follows the previous row's
    # store, and the compiler cannot hoist it there since x and out may alias;
    # loading each row just before its use leaves one row in flight per warp.)
    q0 = _load_row(x_row, row0 + 0, end, stride_x_token)
    q1 = _load_row(x_row, row0 + 1, end, stride_x_token)
    q2 = _load_row(x_row, row0 + 2, end, stride_x_token)
    q3 = _load_row(x_row, row0 + 3, end, stride_x_token)
    q4 = _load_row(x_row, row0 + 4, end, stride_x_token)
    q5 = _load_row(x_row, row0 + 5, end, stride_x_token)
    q6 = _load_row(x_row, row0 + 6, end, stride_x_token)
    q7 = _load_row(x_row, row0 + 7, end, stride_x_token)

    # History rows row0 - 3 .. row0 - 1: input rows, or before the sequence the
    # conv state (zero without an initial state).
    h0 = _history_row(
        x_row,
        state_base,
        row0 - 3,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )
    h1 = _history_row(
        x_row,
        state_base,
        row0 - 2,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )
    h2 = _history_row(
        x_row,
        state_base,
        row0 - 1,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )

    w0 = tl.load(w_ptr + feats * stride_w_dim + 0 * stride_w_width)
    w1 = tl.load(w_ptr + feats * stride_w_dim + 1 * stride_w_width)
    w2 = tl.load(w_ptr + feats * stride_w_dim + 2 * stride_w_width)
    w3 = tl.load(w_ptr + feats * stride_w_dim + 3 * stride_w_width)
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + feats).to(tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    part = col0 // part_dim
    out_row = (
        out_ptr
        + part.to(tl.int64) * stride_out_part
        + seq_start * stride_out_token
        + (col0 - part * part_dim)
        + tl.arange(0, BLOCK_N)
    )
    num_blocks = tl.cdiv(end - row0, BLOCK_M)
    for b in range(0, num_blocks):
        for i in tl.static_range(BLOCK_M):
            r = row0 + b * BLOCK_M + i
            cur = q0.to(tl.float32)
            q0 = q1
            q1 = q2
            q2 = q3
            q3 = q4
            q4 = q5
            q5 = q6
            q6 = q7
            q7 = _load_row(x_row, r + 8, end, stride_x_token)
            # The stock chain, oldest tap first: fma(x_j, w_j, acc). Explicit
            # FMAs so the contraction does not depend on the compiler.
            acc = tl.fma(h0, w0, acc0)
            acc = tl.fma(h1, w1, acc)
            acc = tl.fma(h2, w2, acc)
            acc = tl.fma(cur, w3, acc)
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            tl.store(
                out_row + r * stride_out_token,
                acc.to(out_ptr.dtype.element_ty),
                mask=r < end,
            )
            h0 = h1
            h1 = h2
            h2 = cur

    if chunk == 0:
        # New state = the last 3 inputs, shifted in after the old state when the
        # sequence is shorter. All old-state reads precede the stores, and the
        # same thread owns a channel throughout.
        n0 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            0,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        n1 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            1,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        n2 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            2,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        tl.store(state_base + 0 * stride_state_token, n0)
        tl.store(state_base + 1 * stride_state_token, n1)
        tl.store(state_base + 2 * stride_state_token, n2)


@triton.jit
def _history_row(
    x_row,
    state_base,
    pos,
    load_init,
    stride_x_token,
    stride_state_token,
    STATE_LEN: tl.constexpr,
):
    return tl.where(
        pos >= 0,
        tl.load(x_row + pos * stride_x_token, mask=pos >= 0, other=0.0).to(tl.float32),
        tl.load(
            state_base + (STATE_LEN + pos) * stride_state_token,
            mask=(pos < 0) & load_init,
            other=0.0,
        ),
    )


@triton.jit
def _new_state_row(
    x_row,
    state_base,
    seqlen,
    load_init,
    t,
    stride_x_token,
    stride_state_token,
    STATE_LEN: tl.constexpr,
):
    src = seqlen - STATE_LEN + t
    return tl.where(
        src >= 0,
        tl.load(x_row + src * stride_x_token, mask=src >= 0, other=0.0).to(tl.float32),
        tl.load(
            state_base + (t + seqlen) * stride_state_token,
            mask=(src < 0) & load_init,
            other=0.0,
        ),
    )


def packed_causal_conv1d_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    part_dim: int,
) -> bool:
    """Whether ``packed_causal_conv1d_fn`` reproduces the stock conv bit for bit.

    The stock kernel computes in the conv state dtype; with an FP32 state that is
    FP32 for every input dtype, which this kernel does."""
    return (
        x.is_cuda
        and x.dim() == 2
        and x.stride(1) == 1
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and conv_states.dtype == torch.float32
        and weight.dtype == torch.float32
        and weight.dim() == 2
        and weight.size(0) == x.size(1)
        and weight.size(1) == 4
        and conv_states.dim() == 3
        and conv_states.size(1) == x.size(1)
        and conv_states.size(2) >= 3
        and part_dim % 64 == 0
        and x.size(1) % part_dim == 0
    )


def packed_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    part_dim: int,
    activation: str | None = "silu",
    null_block_id: int | None = NULL_BLOCK_ID,
) -> torch.Tensor:
    """Varlen causal depthwise conv of ``x`` (tokens, dim), dim = parts x part_dim.

    Same semantics as ``causal_conv1d_fn(x.T, weight, bias, conv_states, ...).T``
    with ``conv_states`` (cache lines, dim, 3) updated in place; width-4 taps. Returns
    ``(dim // part_dim, tokens, part_dim)`` in ``x.dtype``: plane i is the dense
    output of channels [i * part_dim, (i + 1) * part_dim).
    """
    num_tokens, dim = x.shape
    num_seqs = query_start_loc.size(0) - 1
    out = torch.empty(
        (dim // part_dim, num_tokens, part_dim), dtype=x.dtype, device=x.device
    )
    if num_tokens == 0 or num_seqs == 0:
        return out
    block_m, num_blocks, block_n, num_warps = _config(part_dim)
    # Radix-32 levels of the kernel's program search: 32 ** steps >= num_seqs.
    search_steps = 0
    while 32**search_steps < num_seqs:
        search_steps += 1
    # Every sequence adds at most one partial chunk (see the kernel's program map).
    num_col_blocks = triton.cdiv(dim, block_n)
    grid = (
        (triton.cdiv(num_tokens, block_m * num_blocks) + num_seqs) * num_col_blocks,
    )
    _packed_causal_conv1d_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        num_seqs,
        search_steps,
        32 ** max(search_steps - 1, 0),
        num_col_blocks,
        part_dim,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        out.stride(0),
        out.stride(1),
        null_block_id if null_block_id is not None else -1,
        HAS_BIAS=bias is not None,
        HAS_NULL_BLOCK=null_block_id is not None,
        SILU_ACTIVATION=activation in ("silu", "swish", True),
        BLOCK_M=block_m,
        NUM_BLOCKS=num_blocks,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


# Per-plane kernel (``supported`` / ``kda_prefill_conv``); see the module docstring.
_BLOCK_M = 8  # must match the causal-conv1d metadata (compute_causal_conv1d_metadata)
_BLOCK_N = 512
_WARPS = 4


@triton.jit
def _tap4(c0, c1, c2, xt, w0, w1, w2, w3, BLOCK_N: tl.constexpr):
    # The generic kernel's loop body for one token: bf16 products, fp32
    # accumulator, taps oldest first, then SiLU.
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc += c0 * w0
    acc += c1 * w1
    acc += c2 * w2
    acc += xt * w3
    return acc / (1 + tl.exp(-acc))


@triton.jit(do_not_specialize=["num_cache_lines"])
def _kda_conv_fwd_kernel(
    x_ptr,  # (dim, cu_seqlen) view, channel stride 1
    w_ptr,  # (dim, 4)
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,
    dim: tl.constexpr,
    num_cache_lines,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    state_len = 3  # KERNEL_WIDTH - 1
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    conv_states_input_coord = tl.load(
        cache_indices_ptr + idx_seq * stride_cache_indices
    ).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            col2 = tl.load(prior_tokens, mask_w, 0.0)
            col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        tl.debug_barrier()

        # Conv-state update: the generic kernel's STEP 2, unchanged.
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        col1 = tl.load(
            prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
        )
        col0 = tl.load(
            prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
        )

    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # The generic kernel's token loop (BLOCK_M = 8 tokens), with the 8 row
    # loads issued up front instead of one per iteration.
    xb = x_base + token_offset * stride_x_token
    ob = (
        o_ptr
        + (sequence_start_index + token_offset) * stride_o_token
        + (idx_feats * stride_o_dim)
    )
    x0 = tl.load(xb + 0 * stride_x_token, mask=mask_w & (segment_len > 0))
    x1 = tl.load(xb + 1 * stride_x_token, mask=mask_w & (segment_len > 1))
    x2 = tl.load(xb + 2 * stride_x_token, mask=mask_w & (segment_len > 2))
    x3 = tl.load(xb + 3 * stride_x_token, mask=mask_w & (segment_len > 3))
    x4 = tl.load(xb + 4 * stride_x_token, mask=mask_w & (segment_len > 4))
    x5 = tl.load(xb + 5 * stride_x_token, mask=mask_w & (segment_len > 5))
    x6 = tl.load(xb + 6 * stride_x_token, mask=mask_w & (segment_len > 6))
    x7 = tl.load(xb + 7 * stride_x_token, mask=mask_w & (segment_len > 7))
    o = _tap4(col0, col1, col2, x0, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 0 * stride_o_token, o, mask=mask_w & (segment_len > 0))
    o = _tap4(col1, col2, x0, x1, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 1 * stride_o_token, o, mask=mask_w & (segment_len > 1))
    o = _tap4(col2, x0, x1, x2, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 2 * stride_o_token, o, mask=mask_w & (segment_len > 2))
    o = _tap4(x0, x1, x2, x3, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 3 * stride_o_token, o, mask=mask_w & (segment_len > 3))
    o = _tap4(x1, x2, x3, x4, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 4 * stride_o_token, o, mask=mask_w & (segment_len > 4))
    o = _tap4(x2, x3, x4, x5, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 5 * stride_o_token, o, mask=mask_w & (segment_len > 5))
    o = _tap4(x3, x4, x5, x6, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 6 * stride_o_token, o, mask=mask_w & (segment_len > 6))
    o = _tap4(x4, x5, x6, x7, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 7 * stride_o_token, o, mask=mask_w & (segment_len > 7))


def supported(x: torch.Tensor, weight: torch.Tensor, conv_states, metadata) -> bool:
    return (
        weight.shape[1] == 4
        and x.dim() == 2
        and x.stride(0) == 1
        and conv_states is not None
        and conv_states.dtype == x.dtype
        and metadata is not None
        and metadata.nums_dict is not None
        and _BLOCK_M in metadata.nums_dict
        and metadata.nums_dict[_BLOCK_M]["batch_ptr"] is not None
    )


def kda_prefill_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    metadata,
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """causal_conv1d_fn(x, weight, None, activation="silu", conv_states=...,
    has_initial_state=..., cache_indices=..., query_start_loc=...,
    metadata=metadata) for width 4, channel-last x (dim, tokens)."""
    out = torch.empty_like(x)
    dim, _ = x.shape
    nums = metadata.nums_dict[_BLOCK_M]
    tot = nums["tot"]
    if tot == 0:
        return out
    grid = (tot, triton.cdiv(dim, _BLOCK_N))
    _kda_conv_fwd_kernel[grid](
        x,
        weight,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        nums["batch_ptr"],
        nums["token_chunk_offset_ptr"],
        out,
        dim,
        conv_states.size(0),
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        out.stride(0),
        out.stride(1),
        pad_slot_id,
        null_block_id,
        HAS_NULL_BLOCK=null_block_id is not None,
        NP2_STATELEN=4,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_WARPS,
        num_stages=2,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out


# --- CUDA fused Q|K|V conv (ops/kda_qkv_conv) ----------------------------------

_QKV_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kda_qkv_conv", "kda_qkv_conv.cu"
)
# Shipped as package data (setup.py) and JIT-built on first use; a missing
# source is an install error and raises in _load_qkv_conv, never a fallback.
_qkv_lock = threading.Lock()
_qkv_loaded = False


def _load_qkv_conv() -> None:
    """Build (first use per source hash) and load torch.ops.k3_kda_conv."""
    global _qkv_loaded
    if _qkv_loaded:
        return
    with _qkv_lock:
        if _qkv_loaded:
            return
        from torch.utils.cpp_extension import load

        import vllm.envs as envs

        h = hashlib.sha256()
        with open(_QKV_SRC, "rb") as fh:
            h.update(fh.read())
        h.update(f"{torch.__version__} {torch.version.cuda}".encode())
        build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_kda_conv", h.hexdigest()[:16])
        os.makedirs(build, exist_ok=True)
        load(
            name="k3_kda_conv",
            sources=[_QKV_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
            build_directory=build,
            is_python_module=False,
            verbose=False,
        )
        _qkv_loaded = True


def qkv_conv_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    part_dim: int,
) -> bool:
    """Whether ``qkv_conv`` runs this call (bit for bit vs stock).

    Capability (False, the caller keeps the Triton path): a non-SM10x device (the
    build targets sm_100f), or a dtype / width combination the kernel does not
    implement: bf16 x; FP32 weights (the KDA layer's, stock FP32 FMA chain) with a
    bf16 or FP32 state, or bf16 weights with a bf16 state (stock bf16-product
    chain); Q|K|V planes of a multiple of 256 channels; conv width 4. On SM10x a
    coverage miss is logged once, so a deployed config never drops to the slow
    path unnoticed.

    Layout (raises ValueError on SM10x): within that coverage the KDA layer always
    passes 16 B aligned unit-stride rows, contiguous weights and a (lines, 3 *
    part_dim, >= 3) conv state; anything else is a caller bug, not a fallback."""
    if not (x.is_cuda and current_platform.is_device_capability_family(100)):
        return False
    gap = None
    if x.dtype != torch.bfloat16:
        gap = f"x dtype {x.dtype} (bf16 only)"
    elif not (
        weight.dtype == torch.float32
        or (weight.dtype == torch.bfloat16 and conv_states.dtype == torch.bfloat16)
    ) or conv_states.dtype not in (torch.bfloat16, torch.float32):
        gap = f"weight {weight.dtype} with conv state {conv_states.dtype}"
    elif part_dim % 256 != 0:
        gap = f"{part_dim} channels per plane (multiple of 256 only)"
    elif weight.dim() != 2 or weight.size(-1) != 4:
        gap = f"conv weight shape {tuple(weight.shape)} (width 4 only)"
    if gap is not None:
        logger.warning_once(
            "Kimi K3 fused Q|K|V conv does not cover %s; the KDA prefill conv "
            "runs the slower Triton kernels.",
            gap,
        )
        return False
    bad = []
    if x.dim() != 2 or x.size(1) != 3 * part_dim:
        bad.append(f"x shape {tuple(x.shape)} != (tokens, {3 * part_dim})")
    elif x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        bad.append(f"x rows not 16 B aligned unit-stride (strides {x.stride()})")
    if tuple(weight.shape) != (3 * part_dim, 4):
        bad.append(f"weight shape {tuple(weight.shape)} != ({3 * part_dim}, 4)")
    if not weight.is_contiguous() or weight.data_ptr() % 16 != 0:
        bad.append("weight not contiguous / 16 B aligned")
    if (
        conv_states.dim() != 3
        or conv_states.size(1) != 3 * part_dim
        or conv_states.size(2) < 3
    ):
        bad.append(
            f"conv state shape {tuple(conv_states.shape)} != "
            f"(lines, {3 * part_dim}, >= 3)"
        )
    if bad:
        raise ValueError("Kimi K3 fused Q|K|V conv: " + "; ".join(bad))
    return True


def _qkv_conv_warps(dim: int, num_tokens: int) -> int:
    """Consumer warps per CTA (256 channels each): the CTA width must divide the
    Q|K|V channels. Measured on VR200 (one persistent wave, 9216 channels): 4 warps
    at 12K-24K tokens (17.4K: 60.8 vs 63.0 us), 3 warps otherwise (8K: 32.5 vs 33.4,
    32.8K: 110.9 vs 118.6; 4608 channels: 3 only), 2 if 768 does not divide."""
    if dim % 1024 == 0 and 12288 <= num_tokens <= 24576:
        return 4
    if dim % 768 == 0:
        return 3
    return 2


def qkv_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    part_dim: int,
    null_block_id: int | None = NULL_BLOCK_ID,
) -> torch.Tensor:
    """``causal_conv1d_fn`` (width 4, SiLU, no bias) of the packed Q|K|V columns
    ``x`` (tokens, 3 * part_dim), conv state (cache lines, 3 * part_dim, >= 3)
    updated in place. Returns (3, tokens, part_dim): plane i is the dense output
    of channels [i * part_dim, (i + 1) * part_dim). Rows of padding sequences
    (cache line ``null_block_id``) and rows outside query_start_loc are not
    written. CUDA graph safe: the row split is read on the device."""
    _load_qkv_conv()
    num_tokens, dim = x.shape
    out = torch.empty((3, num_tokens, part_dim), dtype=x.dtype, device=x.device)
    torch.ops.k3_kda_conv.qkv_conv(
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        out,
        null_block_id if null_block_id is not None else -1,
        _qkv_conv_warps(dim, num_tokens),
    )
    return out
