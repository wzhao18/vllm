# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.op import exp, log
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2


@triton.heuristics(
    {
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit
def _kda_gate_beta_fwd_kernel(
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    gate,
    beta_out,
    lower_bound,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    T,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D

    p_g = raw_g + o_t[:, None] * stride_g_token + i_h * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(
            dt_bias + i_h * D + o_d,
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        b_g += b_bias[None, :]

    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_scaled = b_g * softplus_beta
        b_softplus = tl.where(
            b_scaled > softplus_threshold,
            b_g,
            log(1.0 + tl.exp(b_scaled)) / softplus_beta,
        )
        b_gate = -b_a * b_softplus

    p_gate = gate + (o_t[:, None] * H + i_h) * D + o_d[None, :]
    tl.store(
        p_gate,
        b_gate,
        mask=m_t[:, None] & m_d[None, :],
    )

    b_beta = tl.load(
        raw_beta + o_t * stride_beta_token + i_h,
        mask=m_t,
        other=0.0,
    ).to(tl.float32)
    tl.store(beta_out + o_t * H + i_h, tl.sigmoid(b_beta), mask=m_t)


def _fused_kda_gate_beta(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = raw_g.shape
    assert B == 1
    assert raw_beta.shape == (B, T, H)
    assert raw_g.stride()[2:] == (D, 1)
    assert raw_beta.stride(2) == 1
    gate = torch.empty((B, T, H, D), dtype=torch.float32, device=raw_g.device)
    beta = torch.empty((B, T, H), dtype=torch.float32, device=raw_beta.device)

    BT = 16
    _kda_gate_beta_fwd_kernel[(cdiv(T, BT), H)](
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        gate=gate,
        beta_out=beta,
        lower_bound=lower_bound,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        T=T,
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        H=H,
        D=D,
        BT=BT,
        BD=next_power_of_2(D),
        num_warps=4,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return gate, beta


@triton.jit
def _kda_spec_decode_preprocess_fwd_kernel(
    q,
    k,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    decay,
    packed_params,
    lower_bound,
    T,
    stride_qkv_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_decay_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    Q_SCALE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    mask = (o_t[:, None] < T) & (o_d[None, :] < D)

    token_offsets = o_t[:, None] * stride_qkv_token + i_h * D
    b_q = tl.load(q + token_offsets + o_d[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    b_k = tl.load(k + token_offsets + o_d[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    b_q_inv_norm = tl.rsqrt(tl.sum(b_q * b_q, axis=1) + 1e-6) * Q_SCALE
    b_k_inv_norm = tl.rsqrt(tl.sum(b_k * b_k, axis=1) + 1e-6)
    param_offsets = (o_t * H + i_h) * 3
    tl.store(packed_params + param_offsets, b_q_inv_norm, mask=o_t < T)
    tl.store(packed_params + param_offsets + 1, b_k_inv_norm, mask=o_t < T)

    gate_offsets = o_t[:, None] * stride_g_token + i_h * D
    b_gate = tl.load(
        raw_g + gate_offsets + o_d[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    b_gate += tl.load(dt_bias + i_h * D + o_d, mask=o_d < D, other=0.0)[
        None, :
    ]
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_gate = lower_bound * tl.sigmoid(b_a * b_gate)
    decay_offsets = o_t[:, None] * stride_decay_token + i_h * D
    tl.store(decay + decay_offsets + o_d[None, :], exp(b_gate), mask=mask)

    b_beta = tl.load(
        raw_beta + o_t * stride_beta_token + i_h,
        mask=o_t < T,
        other=0.0,
    ).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    tl.store(packed_params + param_offsets + 2, b_beta, mask=o_t < T)


def _kda_spec_decode_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = q.shape
    assert B == 1 and k.shape == q.shape and raw_g.shape == q.shape
    assert raw_beta.shape == (B, T, H)
    assert q.stride()[2:] == k.stride()[2:] == raw_g.stride()[2:] == (D, 1)
    assert q.stride(1) == k.stride(1)
    assert raw_beta.stride(2) == 1
    assert A_log.is_contiguous() and dt_bias.is_contiguous()
    decay = torch.empty_like(raw_g, dtype=torch.float32)
    packed_params = torch.empty((T, H, 3), dtype=torch.float32, device=q.device)
    block_t = 2
    _kda_spec_decode_preprocess_fwd_kernel[(cdiv(T, block_t), H)](
        q=q,
        k=k,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        decay=decay,
        packed_params=packed_params,
        lower_bound=lower_bound,
        T=T,
        stride_qkv_token=q.stride(1),
        stride_g_token=raw_g.stride(1),
        stride_decay_token=decay.stride(1),
        stride_beta_token=raw_beta.stride(1),
        H=H,
        D=D,
        BT=block_t,
        BD=next_power_of_2(D),
        Q_SCALE=D**-0.5,
        num_warps=1,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return decay, packed_params


@triton.heuristics(
    {
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "USE_PREPROCESSED": lambda args: args["packed_params"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T", "stride_beta_token"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    packed_params,
    A_log,
    dt_bias,
    out,
    state,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token,
    stride_out_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_PREPROCESSED: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    num_stages: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    i_v = pid % tl.cdiv(V, BV)
    i_nh = pid // tl.cdiv(V, BV)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    sequence_length = eos - bos
    if sequence_length == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_state = m_v[:, None] & m_k[None, :]

    if IS_SPEC_DECODING:
        initial_token = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    else:
        initial_token = 0
    state_index = tl.load(state_indices + i_n * stride_indices_seq + initial_token).to(
        tl.int64
    )
    p_out = out + bos * stride_out_token + i_h * V + o_v
    if state_index <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=m_v)
        return

    p_state = (
        state
        + state_index * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_state = tl.load(p_state, mask=m_state, other=0.0).to(tl.float32)

    p_q = q + bos * stride_qkv_token + i_h * K + o_k
    p_k = k + bos * stride_qkv_token + i_h * K + o_k
    p_v = v + bos * stride_qkv_token + i_h * V + o_v
    p_g = g + bos * stride_g_token + i_h * K + o_k
    if USE_PREPROCESSED:
        p_packed_params = packed_params + (bos * H + i_h) * 3
    else:
        p_beta = beta + bos * stride_beta_token + i_h
    for i_t in tl.range(
        0,
        sequence_length,
        num_stages=num_stages,
        disable_licm=USE_PREPROCESSED,
    ):
        if USE_PREPROCESSED:
            b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
            b_q *= tl.load(p_packed_params)
            b_k *= tl.load(p_packed_params + 1)
        else:
            b_q = tl.load(
                p_q, mask=m_k, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
            b_k = tl.load(
                p_k, mask=m_k, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
            b_v = tl.load(
                p_v, mask=m_v, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
        if not USE_PREPROCESSED:
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q *= scale

        if USE_PREPROCESSED:
            b_gate = tl.load(p_g, mask=m_k, other=0.0).to(tl.float32)
        else:
            b_gate = tl.load(
                p_g, mask=m_k, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                b_bias = tl.load(
                    dt_bias + i_h * K + o_k,
                    mask=m_k,
                    other=0.0,
                ).to(tl.float32)
                b_gate += b_bias
            b_a = exp(tl.load(A_log + i_h).to(tl.float32))
            if USE_LOWER_BOUND:
                b_gate = lower_bound * tl.sigmoid(b_a * b_gate)
            else:
                b_softplus = tl.where(
                    b_gate > 20.0,
                    b_gate,
                    log(1.0 + tl.exp(b_gate)),
                )
                b_gate = -b_a * b_softplus

        if not USE_PREPROCESSED:
            b_gate = exp(b_gate)
        b_state *= b_gate[None, :]
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        if USE_PREPROCESSED:
            b_beta = tl.load(p_packed_params + 2).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]
        b_out = tl.sum(b_state * b_q[None, :], axis=1)
        tl.store(
            p_out,
            b_out.to(p_out.dtype.element_ty),
            mask=m_v,
            eviction_policy="evict_first",
        )

        final_state_index = tl.load(state_indices + i_n * stride_indices_seq + i_t).to(
            tl.int64
        )
        if final_state_index > 0:
            p_final_state = (
                state
                + final_state_index * stride_state_token
                + i_h * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            if USE_PREPROCESSED:
                tl.store(
                    p_final_state,
                    b_state.to(p_final_state.dtype.element_ty),
                    mask=m_state,
                    cache_modifier=".cs",
                )
            else:
                tl.store(
                    p_final_state,
                    b_state.to(p_final_state.dtype.element_ty),
                    mask=m_state,
                )

        p_q += stride_qkv_token
        p_k += stride_qkv_token
        p_v += stride_qkv_token
        p_g += stride_g_token
        if USE_PREPROCESSED:
            p_packed_params += H * 3
        else:
            p_beta += stride_beta_token
        p_out += stride_out_token

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


# Consumed by kimi_k3_triton_warmup.py during kernel_warmup().
def get_fused_recurrent_kda_fwd_warmup_profiles(
    num_heads: int,
    tokens_per_sequence: int = 5,
) -> tuple[int, ...]:
    """Return representative sequence counts for gated launch variants."""
    if num_heads == 12:
        if tokens_per_sequence >= 4:
            preprocess_profile = 25
        elif tokens_per_sequence == 3:
            preprocess_profile = 48
        elif tokens_per_sequence == 2:
            preprocess_profile = 96
        else:
            preprocess_profile = 256
        return (1, 2, 3, 9, 19, preprocess_profile)
    return (
        1,
        48 // num_heads + 1,
        96 // num_heads + 1,
    )


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
    packed_params: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch recurrent KDA with dense inner dimensions and row strides."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert B == 1 and k.shape == q.shape
    assert v.shape == (B, T, H, V) and g.shape == (B, T, H, K)
    assert beta.shape == (B, T, H)
    assert initial_state is not None
    assert cu_seqlens is not None
    assert ssm_state_indices is not None
    assert inplace_final_state
    if out is None:
        out = torch.empty_like(v)
    assert out.shape == v.shape
    assert initial_state.shape[1:] == (H, V, K)
    assert ssm_state_indices.ndim in (1, 2)

    assert q.stride()[2:] == k.stride()[2:] == (K, 1)
    assert v.stride()[2:] == out.stride()[2:] == (V, 1)
    assert g.stride()[2:] == (K, 1)
    assert beta.stride(2) == 1
    assert q.stride(1) == k.stride(1) == v.stride(1)
    assert initial_state.stride()[1:] == (V * K, K, 1)
    N = cu_seqlens.numel() - 1
    if ssm_state_indices.ndim == 1:
        assert T == N
        assert num_accepted_tokens is None
    else:
        assert ssm_state_indices.stride(1) == 1
    assert cu_seqlens.is_contiguous()
    if use_gate_in_kernel:
        assert A_log is not None and A_log.is_contiguous()
        assert dt_bias is None or dt_bias.is_contiguous()
    if packed_params is not None:
        assert packed_params.shape == (T, H, 3)
        assert packed_params.dtype == torch.float32
        assert not use_qk_l2norm_in_kernel
        assert not use_gate_in_kernel
        assert not use_beta_sigmoid_in_kernel

    if scale is None:
        scale = K**-0.5

    num_warps = 1
    maxnreg = None
    if packed_params is not None:
        BV, num_stages, maxnreg = 16, 3, 128
    elif (
        use_gate_in_kernel
        and num_accepted_tokens is not None
        and initial_state.dtype == torch.bfloat16
        and H == 12
        and K == 128
        and V == 128
    ):
        if N == 1:
            BV, num_stages, maxnreg = 2, 3, 128
        elif N == 2:
            BV, num_stages, maxnreg = 4, 3, 128
        elif N <= 8:
            BV, num_stages, maxnreg = 8, 3, 128
        elif N <= 18:
            BV, num_stages = 16, 3
        else:
            BV, num_stages, maxnreg = 16, 3, 128
    elif use_gate_in_kernel:
        head_sequences = H * N
        if head_sequences <= 48:
            BV, num_stages = 4, 4
        elif head_sequences <= 96:
            BV, num_stages = 8, 3
        elif head_sequences <= 192:
            BV, num_stages = 16, 3
        else:
            BV, num_stages = 8, 3
    else:
        BV, num_warps, num_stages = 8, 1, 2
    grid = (cdiv(V, BV) * N * H,)
    fused_recurrent_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        packed_params=packed_params,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
        N=N,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=next_power_of_2(K),
        BV=BV,
        stride_qkv_token=q.stride(1),
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        stride_out_token=out.stride(1),
        stride_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, initial_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA from raw gate and beta inputs.

    This vLLM wrapper applies the gate activation and beta sigmoid, selecting
    whether to materialize them before launching the recurrent kernel.
    """
    if fuse_gate is None:
        fuse_gate = True

    _, _, H, D = q.shape
    num_seqs = cu_seqlens.numel() - 1
    # Tuned for BF16 Kimi-K3 speculative decode on B300.
    if num_seqs >= 256:
        min_tokens_per_seq = 1
    elif num_seqs >= 96:
        min_tokens_per_seq = 2
    elif num_seqs >= 48:
        min_tokens_per_seq = 3
    else:
        min_tokens_per_seq = 4
    use_preprocessed = (
        fuse_gate
        and num_seqs >= 25
        and q.shape[1] >= min_tokens_per_seq * num_seqs
        and num_accepted_tokens is not None
        and initial_state.dtype == torch.bfloat16
        and q.dtype
        == k.dtype
        == v.dtype
        == raw_g.dtype
        == raw_beta.dtype
        == torch.bfloat16
        and H == 12
        and D == v.shape[-1] == 128
        and dt_bias is not None
        and lower_bound is not None
        and A_log.dtype == dt_bias.dtype == torch.float32
    )
    packed_params = None
    if use_preprocessed:
        gate, packed_params = _kda_spec_decode_preprocess(
            q,
            k,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound,
        )
        beta = raw_beta
    elif fuse_gate:
        gate = raw_g
        beta = raw_beta
    else:
        gate, beta = _fused_kda_gate_beta(
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound,
        )
    return fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=1.0 if use_preprocessed else D**-0.5,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=not use_preprocessed,
        packed_params=packed_params,
        A_log=A_log if fuse_gate and not use_preprocessed else None,
        dt_bias=dt_bias if fuse_gate and not use_preprocessed else None,
        lower_bound=lower_bound if fuse_gate and not use_preprocessed else None,
        use_gate_in_kernel=fuse_gate and not use_preprocessed,
        use_beta_sigmoid_in_kernel=fuse_gate and not use_preprocessed,
        out=out,
    )


@triton.jit(
    do_not_specialize=["stride_beta_token", "stride_state_indices"]
)
def fused_recurrent_kda_packed_decode_kernel(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token,
    stride_state_token: tl.constexpr,
    stride_state_indices,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_state = state + state_idx * stride_state_token
    p_state += i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    # Q, K, and V occupy consecutive channel ranges, while the token stride
    # may also include the output-gate projection that follows packed QKV.
    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = raw_g + i_n * stride_g_token + i_h * K + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_g += b_bias
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_softplus = tl.where(
            b_g > SOFTPLUS_THRESHOLD,
            b_g,
            log(1.0 + tl.exp(b_g)),
        )
        b_gate = -b_a * b_softplus

    b_state *= exp(b_gate[None, :])
    b_v -= tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )
    b_v *= b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-token KDA decode directly from packed post-conv QKV."""
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be 2D and contiguous in its last dim.")
    if raw_g.ndim != 4 or raw_g.shape[0] != 1:
        raise ValueError("`raw_g` must have shape [1, B, H, K].")
    if raw_beta.ndim != 3 or raw_beta.shape[0] != 1:
        raise ValueError("`raw_beta` must have shape [1, B, H].")
    if initial_state.ndim != 4:
        raise ValueError("`initial_state` must have shape [cache, H, V, K].")
    _, H, V, K = initial_state.shape
    if raw_g.stride()[2:] != (K, 1):
        raise ValueError("`raw_g` must be contiguous within each token.")
    if raw_beta.stride(2) != 1:
        raise ValueError("`raw_beta` heads must be contiguous.")
    if initial_state.stride()[1:] != (V * K, K, 1):
        raise ValueError("`initial_state` must be contiguous within each cache slot.")
    if state_indices.ndim != 1:
        raise ValueError("`state_indices` must be one-dimensional.")
    if A_log.ndim != 1 or not A_log.is_contiguous():
        raise ValueError("`A_log` must be contiguous and one-dimensional.")
    if not dt_bias.is_contiguous():
        raise ValueError("`dt_bias` must be contiguous.")

    device = mixed_qkv.device
    if any(
        x.device != device
        for x in (raw_g, raw_beta, A_log, dt_bias, initial_state, state_indices)
    ):
        raise ValueError("All packed KDA inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if raw_g.shape != (1, B, H, K):
        raise ValueError(f"Unexpected raw gate shape {tuple(raw_g.shape)}.")
    if raw_beta.shape != (1, B, H):
        raise ValueError(f"Unexpected raw beta shape {tuple(raw_beta.shape)}.")
    if mixed_qkv.shape[1] != 2 * H * K + H * V:
        raise ValueError(f"Unexpected packed QKV shape {tuple(mixed_qkv.shape)}.")
    if A_log.numel() != H or dt_bias.numel() != H * K:
        raise ValueError("`A_log` or `dt_bias` has an incompatible shape.")
    if state_indices.shape[0] != B:
        raise ValueError("`state_indices` must contain one entry per token.")

    BK = next_power_of_2(K)
    BV = min(next_power_of_2(V), 32)
    if scale is None:
        scale = K**-0.5

    out = torch.empty((1, B, H, V), dtype=mixed_qkv.dtype, device=device)
    grid = (cdiv(V, BV), B * H)
    fused_recurrent_kda_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        state=initial_state,
        state_indices=state_indices,
        lower_bound=lower_bound or 0.0,
        scale=scale,
        stride_mixed_token=mixed_qkv.stride(0),
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        stride_state_token=initial_state.stride(0),
        stride_state_indices=state_indices.stride(0),
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, initial_state
