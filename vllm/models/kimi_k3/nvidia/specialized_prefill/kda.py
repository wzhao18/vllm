# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill: the cuDNN FROST chunked core and the gated o_norm + MXFP8
o_proj input at the HBM rate.

Core: cuDNN frontend >= 1.33 ``kda_frost`` (CuTe DSL, SM100/SM103/SM107; FP32
state carried in TMEM, BT = 16 chunks, in-kernel q/k L2 norm, safe gate and
beta sigmoid, the same gate and state layout as FlashKDA). At the DEP16 F16K
shape (3 x 5.4K tokens, 96 heads, FP32 state) it runs 466 us per layer on
VR200 vs 694 us for FlashKDA (prepare + recurrence); output vs FlashKDA
rel-rms 6e-3, final state 4e-3 (BF16 operand rounding order).

o_norm:

Stock (ops/kda_norm.py ``rmsnorm_gated_mxfp8``, Triton) runs the 96-head DEP16
prefill at ~5 TB/s (204 us per layer at 16K tokens on VR200). This kernel
(ops/flashkda_split/csrc/smxx/o_norm_mx.cu) gives each (token, head) row 8
lanes x 16 columns, keeps two 4-row steps of loads in flight and does 32-bit
row arithmetic. Per element it is the Triton kernel's expressions and
instructions (approximate sqrt, full-range division, ex2 sigmoid, BF16 round,
UE8M0 per 32, RN-satfinite E4M3); the row sum of squares runs as a lane
butterfly, another order than Triton's tl.sum (stock tolerance).
"""

from __future__ import annotations

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_conv, kda_segscan

_FROST_MIN_VERSION = (1, 33)
_frost_fn = None


def frost_supported(device: torch.device) -> bool:
    """cuDNN FROST KDA is importable and the GPU is SM100-class."""
    global _frost_fn
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    if _frost_fn is None:
        try:
            import cudnn
            from cudnn.linear_attention.ops import kimi_delta_attention
        except ImportError:
            _frost_fn = False
            return False
        version = tuple(int(x) for x in cudnn.__version__.split(".")[:2])
        _frost_fn = kimi_delta_attention if version >= _FROST_MIN_VERSION else False
    return _frost_fn is not False


def frost_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA chunked prefill (FlashKDA contract) on cuDNN FROST.

    q/k/v/g: (1, T, H, 128) BF16 (g raw gate logits), beta: (1, T, H) raw
    logits, initial_state: (N, H, 128, 128) FP32 (FlashKDA layout, V-major).
    Returns the output (1, T, H, 128) and the final state (N, H, 128, 128).
    """
    assert _frost_fn
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    o, final_state = _frost_fn(
        q.reshape(T, H, D).contiguous(),
        k.reshape(T, H, D).contiguous(),
        v.reshape(T, H, D).contiguous(),
        g.reshape(T, H, D).contiguous(),
        beta.reshape(T, H),  # row-strided view: FROST reads it in place
        cu_seqlens,
        scale=D**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=lower_bound,
        a_log=A_log,
        dt_bias=dt_bias.view(H, D),
        plan_name="kda_frost",
    )[:2]
    return o.view(1, T, H, D), final_state


def frost_conv_plan(
    tokens: int, num_seqs: int, heads: int, lower_bound: float, device
):
    """The FROST plan with the short conv fused into its q/k/v loads
    (ops/frost_kda) at this batch shape, or None where cuDNN plans a launch the
    fused kernels do not cover (piece chains); the caller then runs qkv_conv and
    frost_prefill."""
    from vllm.models.kimi_k3.nvidia.ops.frost_kda.launcher import (
        frost_conv_plan as _plan,
    )

    plan = _plan(tokens, num_seqs, heads, 128, lower_bound, device)
    return plan if plan.supported else None


def frost_conv_prefill(
    plan,
    qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short conv + KDA chunked prefill of the raw in_proj Q|K|V columns qkv
    (T, 3 H 128): causal_conv1d_fn + frost_prefill in one FROST launch
    (bit for bit the unfused result), then the conv state update."""
    T, H = g.shape[1], g.shape[2]
    slot = torch.where(has_initial_state, state_indices, -1).to(torch.int32)
    o, final_state = plan(
        qkv,
        g.reshape(T, H, 128).contiguous(),
        beta.reshape(T, H),
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        conv_weight,
        conv_state,
        slot,
        state_indices,
        has_initial_state,
    )
    kda_conv.qkv_conv_state(
        qkv, conv_state, cu_seqlens, state_indices, has_initial_state
    )
    return o.view(1, T, H, 128), final_state


def o_norm_mxfp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out,
    head0: int = 0,
) -> None:
    """Gated RMSNorm (sigmoid) of x (..., H, 128) with gate rows g (T, >= H 128)
    into ``out`` (an ``mxfp8_act.Mxfp8Buffer`` of T rows). ``head0`` places x at
    heads [head0, head0 + H) of g and out."""
    kda_segscan.load_extension()
    torch.ops.k3_flashkda_split.o_norm_mxfp8(
        x, g, weight.float().contiguous(), eps, out.data, out.scale, head0
    )
