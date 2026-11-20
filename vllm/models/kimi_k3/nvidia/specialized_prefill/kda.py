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

from vllm.models.kimi_k3.nvidia.ops import kda_segscan

_FROST_MIN_VERSION = (1, 33)
_frost_fn = None


def frost_supported(device: torch.device) -> bool:
    """Whether the KDA prefill core runs cuDNN FROST: every SM100-class GPU.

    Capability only: other GPUs return False. On SM10x a missing or older cuDNN
    frontend is an install error and raises (FROST is the prefill policy there; a
    silent FlashKDA fallback would hide it)."""
    global _frost_fn
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    if _frost_fn is None:
        try:
            import cudnn
            from cudnn.linear_attention.ops import kimi_delta_attention
        except ImportError as e:
            raise RuntimeError(
                "Kimi K3 KDA prefill on SM10x runs cuDNN FROST: install "
                "nvidia-cudnn-frontend >= 1.33 (with nvidia-cutlass-dsl)"
            ) from e
        version = tuple(int(x) for x in cudnn.__version__.split(".")[:2])
        if version < _FROST_MIN_VERSION:
            raise RuntimeError(
                f"Kimi K3 KDA prefill on SM10x runs cuDNN FROST: cuDNN frontend "
                f"{cudnn.__version__} at {cudnn.__file__} is older than 1.33"
            )
        _frost_fn = kimi_delta_attention
    return True


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


_persistent_plans: dict = {}


def persistent_supported(device: torch.device) -> bool:
    """The persistent CuTe DSL KDA core can run (SM100-class GPU, CuTe DSL)."""
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def persistent_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    qk_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA chunked prefill on the persistent kernel (same contract as
    ``frost_prefill``; ``cu_seqlens_cpu`` is the host copy used to plan the
    (sequence, head) chain pieces once per batch).

    With ``state_cache`` [slots, H, D, D] the kernel loads each sequence's
    initial state from ``state_cache[state_indices[i]]`` (zero where
    ``has_initial_state[i]`` is false) and stores its final state back to that
    row in place; the returned final state is then None.

    qk_normalized: the producer already l2-normalized q and k per (token,
    head) row (stock l2norm_fwd before the chunk kernels); the kernel then
    skips its own norm."""
    from .kda_persistent import forward as kp

    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    offsets = tuple(cu_seqlens_cpu.tolist())
    key = (offsets, H, q.device.index)
    plan = _persistent_plans.get(key)
    if plan is None:
        if len(_persistent_plans) >= 64:
            _persistent_plans.clear()
        plan = kp.make_plan(list(offsets), H, q.device)
        _persistent_plans[key] = plan
    out = torch.empty(1, T, H, D, dtype=q.dtype, device=q.device)
    srow = None
    if state_cache is not None:
        assert state_indices is not None and has_initial_state is not None
        nseq = len(offsets) - 1
        rows, srow = kp.state_cache_rows(
            state_cache, state_indices[:nseq], has_initial_state, H
        )
        initial_state = final_state = rows
    else:
        final_state = torch.empty_like(initial_state)
    kp.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.reshape(1, T, H).contiguous(),
        D**-0.5,
        out,
        A_log,
        dt_bias.view(H, D),
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        plan=plan,
        srow=srow,
        qk_normalized=qk_normalized,
    )
    return out, (None if srow is not None else final_state)


def o_norm_mxfp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out,
) -> None:
    """Gated RMSNorm (sigmoid) of x (..., H, 128) with gate rows g (T, >= H 128)
    into ``out`` (an ``mxfp8_act.Mxfp8Buffer`` of T rows)."""
    kda_segscan.load_extension()
    torch.ops.k3_flashkda_split.o_norm_mxfp8(
        x, g, weight.float().contiguous(), eps, out.data, out.scale
    )
