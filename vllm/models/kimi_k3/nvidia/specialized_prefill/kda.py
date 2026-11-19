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
(ops/flashkda_split/csrc/smxx/o_norm_mx.cu) stages each token's H x 128 output
and gate rows in shared memory with bulk async copies (4-deep ring, one producer
lane) and normalizes them with 12 consumer warps (two 4-row steps each), 8 lanes x 16 columns per row:
101 us per layer at 16K tokens (139 us with direct loads). Per element it is the
Triton kernel's expressions and instructions (approximate sqrt, full-range
division, ex2 sigmoid, BF16 round, UE8M0 per 32, RN-satfinite E4M3; the sigmoid's
reciprocal is rcp.approx where that equals the division); the row sum of squares
runs as a lane butterfly, another order than Triton's tl.sum (stock tolerance).
"""

from __future__ import annotations

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.triton_utils import tl, triton

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


@triton.jit
def _zero_new_state_rows(state_ptr, idx_ptr, has_init_ptr, row_stride, ROW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    if tl.load(has_init_ptr + i):
        return
    row = tl.load(idx_ptr + i).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(state_ptr + row * row_stride + offs, tl.zeros([BLOCK], state_ptr.dtype.element_ty), mask=offs < ROW)


def frost_prefill_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """``frost_prefill`` seeded from and writing back to the recurrent state pool:
    sequence i starts from ``state_pool[state_indices[i]]`` (zero without an
    initial state) and its final state replaces that row, so no gather / scatter
    copies run. Returns the output (1, T, H, 128)."""
    assert _frost_fn
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    # FROST takes buffers as dense kernel arguments: the scheduler's index
    # tensor can be a strided view.
    state_indices = state_indices.contiguous()
    row = state_pool[0].numel()
    block = 4096
    _zero_new_state_rows[(state_indices.numel(), triton.cdiv(row, block))](
        state_pool, state_indices, has_initial_state, state_pool.stride(0), ROW=row, BLOCK=block
    )
    o = _frost_fn(
        q.reshape(T, H, D).contiguous(),
        k.reshape(T, H, D).contiguous(),
        v.reshape(T, H, D).contiguous(),
        g.reshape(T, H, D).contiguous(),
        beta.reshape(T, H),
        cu_seqlens,
        scale=D**-0.5,
        initial_state=state_pool,
        state_indices=state_indices,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=lower_bound,
        a_log=A_log,
        dt_bias=dt_bias.view(H, D),
        plan_name="kda_frost",
    )[0]
    return o.view(1, T, H, D)


def frost_head_split(num_seqs: int, heads: int, device: torch.device) -> int:
    """Head split for FROST: 0 when the (sequence, head) items
    fit one wave of SMs, else the largest head count whose items fit one wave."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_seqs * heads <= sms:
        return 0
    split = sms // num_seqs
    return split if 0 < split < heads else 0


def frost_prefill_in_place_o_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    g2: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out,
    split: int,
) -> None:
    """``frost_prefill_in_place`` + ``o_norm_mxfp8`` as two head groups: heads
    [0, split) fill one wave of SMs with whole (sequence, head) items, then FROST
    plans the remaining heads on their own (its d_v-half tail over fewer items),
    each group normalized into its head window of ``out``."""
    assert _frost_fn and 0 < split < q.shape[2]
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    state_indices = state_indices.contiguous()
    row = state_pool[0].numel()
    block = 4096
    _zero_new_state_rows[(state_indices.numel(), triton.cdiv(row, block))](
        state_pool, state_indices, has_initial_state, state_pool.stride(0), ROW=row, BLOCK=block
    )
    q3, k3, v3, g3 = (x.reshape(T, H, D) for x in (q, k, v, g))
    b2 = beta.reshape(T, H)
    dt = dt_bias.view(H, D)
    w = norm_weight.float().contiguous()
    kda_segscan.load_extension()

    def core(lo: int, hi: int) -> torch.Tensor:
        return _frost_fn(
            q3[:, lo:hi],
            k3[:, lo:hi],
            v3[:, lo:hi],
            g3[:, lo:hi],
            b2[:, lo:hi],
            cu_seqlens,
            scale=D**-0.5,
            initial_state=state_pool[:, lo:hi],
            state_indices=state_indices,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            gate_lower_bound=lower_bound,
            a_log=A_log[lo:hi],
            dt_bias=dt[lo:hi],
            plan_name="kda_frost",
        )[0]

    for lo, hi in ((0, split), (split, H)):
        torch.ops.k3_flashkda_split.o_norm_mxfp8(core(lo, hi), g2, w, eps, out.data, out.scale, lo)


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
