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

# Per-step memo of the host-side KDA prefill inputs (plan, state-row table),
# keyed by the identity of the step's per-cache-group KDA metadata object. The
# metadata object is built once per step and shared by every KDA layer of its
# cache group, so an entry is valid exactly for that step; a new step brings a
# new object. Strong references keep ids unique while an entry lives.
_step_memos: dict = {}


def step_memo_for(metadata) -> dict:
    """The memo dict of this step's KDA metadata object (see _step_memos)."""
    entry = _step_memos.get(id(metadata))
    if entry is not None and entry[0] is metadata:
        return entry[1]
    if len(_step_memos) >= 16:
        _step_memos.clear()
    memo: dict = {}
    _step_memos[id(metadata)] = (metadata, memo)
    return memo


def persistent_supported(device: torch.device) -> bool:
    """The persistent CuTe DSL KDA core can run (SM100-class GPU, CuTe DSL)."""
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def warm_persistent(H: int, device: torch.device, lower_bound: float) -> None:
    """Compile every persistent-core variant a prefill step can launch for H
    heads at load, so no serving step compiles (once per process)."""
    key = (H, device.index, lower_bound)
    if key in _persistent_warmed:
        return
    from .kda_persistent import forward as kp

    kp.warm_compile(H, device, lower_bound)
    _persistent_warmed.add(key)


_persistent_warmed: set = set()


def _beta_rows(beta: torch.Tensor, T: int, H: int) -> torch.Tensor:
    """beta for the persistent kernel: its per-element loads (H % 8 != 0, e.g.
    the TP8 shard's 12 heads) read any row stride, so a row-strided view (the
    KDA in_proj output's beta columns) goes in uncopied; the beta TMA path
    (H % 8 == 0) takes dense rows."""
    beta = beta.reshape(1, T, H)
    if H % 8 == 0 or beta.stride(-1) != 1:
        beta = beta.contiguous()
    return beta


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
    out: torch.Tensor | None = None,
    step_memo: dict | None = None,
    checkpoint_offsets: list[int] | None = None,
    checkpoint_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA chunked prefill on the persistent kernel (same contract as
    ``frost_prefill``; ``cu_seqlens_cpu`` is the host copy used to plan the
    (sequence, head) chain pieces once per batch).

    ``out`` (1, T, H, 128), contiguous, receives the output in place (the KDA
    layer passes its ``core_attn_out`` rows); a new tensor otherwise.

    With ``state_cache`` [slots, H, D, D] the kernel loads each sequence's
    initial state from ``state_cache[state_indices[i]]`` (zero where
    ``has_initial_state[i]`` is false) and stores its final state back to that
    row in place; the returned final state is then None.

    ``checkpoint_offsets`` (host, one per sequence, 0 = none) requests prefill
    checkpoints: ``checkpoint_out[s]`` (an FP32 [N >= nseq, H, D, D] buffer)
    receives the state after exactly ``checkpoint_offsets[s]`` tokens of
    sequence ``s``. The checkpoint is a sequence boundary of the persistent
    schedule (kda_persistent.forward.make_cut_plan): the prefix computes as a
    run of [0, offset) alone and the rest as a run seeded with its state. Needs
    the in-place ``state_cache``."""
    from .kda_persistent import forward as kp

    if checkpoint_offsets is not None and any(checkpoint_offsets):
        return _persistent_prefill_cut(
            q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens_cpu,
            state_cache, state_indices, has_initial_state, out, step_memo,
            checkpoint_offsets, checkpoint_out,
        )
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    # Every KDA layer of a step passes the same host offsets and state rows:
    # the host plan and the state-row table are built by the step's first layer
    # and reused by the other layers (step_memo is the step's per-cache-group
    # dict, see step_memo_for()).
    memo = step_memo
    plan = memo.get(("plan", H, q.device.index)) if memo is not None else None
    if plan is None:
        offsets = tuple(cu_seqlens_cpu.tolist())
        key = (offsets, H, q.device.index)
        plan = _persistent_plans.get(key)
        if plan is None:
            if len(_persistent_plans) >= 64:
                _persistent_plans.clear()
            plan = kp.make_plan(list(offsets), H, q.device)
            _persistent_plans[key] = plan
        if memo is not None:
            memo[("plan", H, q.device.index)] = plan
            memo[("nseq", H, q.device.index)] = len(offsets) - 1
        nseq = len(offsets) - 1
    else:
        nseq = memo[("nseq", H, q.device.index)]
    if out is None:
        out = torch.empty(1, T, H, D, dtype=q.dtype, device=q.device)
    elif not (
        out.shape == (1, T, H, D) and out.dtype == q.dtype and out.is_contiguous()
    ):
        raise ValueError(
            f"persistent KDA prefill out must be a contiguous (1, {T}, {H}, {D}) "
            f"{q.dtype} tensor, got {tuple(out.shape)} {out.dtype} "
            f"stride {out.stride()}"
        )
    srow = None
    if state_cache is not None:
        assert state_indices is not None and has_initial_state is not None
        # The row table depends only on the slot indices, the zero-state flags
        # and the slot stride: one per step and stride for all layers.
        skey = (
            "srow",
            H,
            state_cache.stride(0),
            state_cache.device.index,
            state_indices.data_ptr(),
            has_initial_state.data_ptr(),
            nseq,
        )
        srow = memo.get(skey) if memo is not None else None
        if srow is None:
            rows, srow = kp.state_cache_rows(
                state_cache, state_indices[:nseq], has_initial_state, H
            )
            if memo is not None:
                memo[skey] = srow
        else:
            rows = kp.state_cache_view(state_cache, H)
        initial_state = final_state = rows
    else:
        final_state = torch.empty_like(initial_state)
    kp.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        _beta_rows(beta, T, H),
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
    )
    return out, (None if srow is not None else final_state)


def _persistent_prefill_cut(
    q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens_cpu,
    state_cache, state_indices, has_initial_state, out, step_memo,
    checkpoint_offsets, checkpoint_out,
):
    """persistent_prefill with checkpoint cuts (see its docstring)."""
    from .kda_persistent import forward as kp

    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    dev = q.device
    if state_cache is None or state_indices is None or has_initial_state is None:
        raise ValueError("persistent KDA checkpoints need the in-place state cache")
    offsets = tuple(cu_seqlens_cpu.tolist())
    nseq = len(offsets) - 1
    cuts = tuple(int(x) for x in checkpoint_offsets)
    if len(cuts) != nseq:
        raise ValueError(f"{len(cuts)} checkpoint offsets for {nseq} sequences")
    if (
        checkpoint_out is None
        or checkpoint_out.dtype != torch.float32
        or not checkpoint_out.is_contiguous()
        or checkpoint_out.shape[1:] != (H, D, D)
        or checkpoint_out.shape[0] < nseq
    ):
        raise ValueError(
            f"checkpoint_out must be a contiguous FP32 [>= {nseq}, {H}, {D}, {D}] "
            f"tensor, got {None if checkpoint_out is None else tuple(checkpoint_out.shape)}"
        )
    memo = step_memo
    mkey = ("cplan", H, dev.index, cuts)
    entry = memo.get(mkey) if memo is not None else None
    if entry is None:
        key = (offsets, cuts, H, dev.index)
        entry = _persistent_plans.get(key)
        if entry is None:
            if len(_persistent_plans) >= 64:
                _persistent_plans.clear()
            entry = kp.make_cut_plan(list(offsets), list(cuts), H, dev)
            _persistent_plans[key] = entry
        if memo is not None:
            memo[mkey] = entry
    plan, meta = entry
    # Pseudo-sequence state rows: a prefix loads its sequence's initial state
    # (its final state goes to the checkpoint buffer, not the cache); a suffix
    # seeds from that buffer and stores the sequence's final state.
    skey = (
        "csrow", H, state_cache.stride(0), state_cache.device.index,
        state_indices.data_ptr(), has_initial_state.data_ptr(), cuts,
    )
    srow = memo.get(skey) if memo is not None else None
    rows = kp.state_cache_view(state_cache, H)
    if srow is None:
        _, base = kp.state_cache_rows(
            state_cache, state_indices[:nseq], has_initial_state, H
        )
        load, store = base[: nseq * H], base[nseq * H :]
        src_chain, is_suffix = [], []
        for s, (pre, suf) in enumerate(meta["pseudo_of"]):
            for p in (pre, suf):
                if p < 0:
                    continue
                for h in range(H):
                    src_chain.append(s * H + h)
                    is_suffix.append(p == suf)
        idx = torch.tensor(src_chain, dtype=torch.int64, device=dev)
        suffix = torch.tensor(is_suffix, dtype=torch.bool, device=dev)
        srow = torch.cat(
            (torch.where(suffix, -1, load.index_select(0, idx)), store.index_select(0, idx))
        )
        if memo is not None:
            memo[skey] = srow
    flat = checkpoint_out.view(-1, D, D)
    nbuf = meta["nbuf"]
    mid = flat[:nbuf] if flat.shape[0] >= nbuf else torch.empty(
        nbuf, D, D, dtype=torch.float32, device=dev
    )
    if out is None:
        out = torch.empty(1, T, H, D, dtype=q.dtype, device=dev)
    elif not (out.shape == (1, T, H, D) and out.dtype == q.dtype and out.is_contiguous()):
        raise ValueError(
            f"persistent KDA prefill out must be a contiguous (1, {T}, {H}, {D}) "
            f"{q.dtype} tensor, got {tuple(out.shape)} {out.dtype} stride {out.stride()}"
        )
    plan = plan[:8] + (mid,) + plan[9:]
    kp.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        _beta_rows(beta, T, H),
        D**-0.5,
        out,
        A_log,
        dt_bias.view(H, D),
        lower_bound,
        initial_state=rows,
        final_state=rows,
        cu_seqlens=plan[0],
        plan=plan,
        srow=srow,
    )
    if mid.data_ptr() != flat.data_ptr():
        flat[: nseq * H].copy_(mid[: nseq * H])
    return out, None


_fp32_weights: dict = {}


def _fp32_weight(weight: torch.Tensor) -> torch.Tensor:
    """FP32 contiguous copy of a norm weight, made once per weight version
    (a cast kernel per layer and step otherwise)."""
    if weight.dtype == torch.float32 and weight.is_contiguous():
        return weight
    key = (weight.data_ptr(), weight.device, weight.dtype, weight._version)
    w32 = _fp32_weights.get(key)
    if w32 is None:
        if len(_fp32_weights) >= 1024:
            _fp32_weights.clear()
        w32 = weight.float().contiguous()
        _fp32_weights[key] = w32
    return w32


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
        x, g, _fp32_weight(weight), eps, out.data, out.scale
    )
