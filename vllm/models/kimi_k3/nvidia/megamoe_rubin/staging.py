# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 hidden states + top-k routes -> the session's staged kernel inputs, one
Triton launch on the current stream (CUDA graph safe: the grid depends only on the
static row count ``n`` and the bucket capacity, no host sync).

Grid: ``n`` live-row programs (x ``hidden / CH`` chunks) plus
``ceil((capacity - n) / DEAD_ROWS)`` dead-row programs, each of which writes
``topk_idx = -1`` for ``DEAD_ROWS`` rows of ``[n, capacity)`` (only chunk 0 works).

Rows ``< n``:

- MXFP8 (``mxfp4_mxfp8``): per 32 elements ``e = ceil_log2(amax * (1/448))``
  clamped to [-126, 127] (bit decomposition, as FlashInfer ``_ceil_log2_exponent``),
  ``x = e4m3_rne(v / 2^e)``, SF byte ``clamp(e + 127, 1, 254)``.
- NVFP4: per 16 elements ``s = e4m3_rne(min(amax * (n_in/6), 448))``,
  ``q = e2m1_rne_satfinite(v * (n_in / s))`` (0 where ``s == 0``), even element in
  the low nibble (FlashInfer ``quantize_nvfp4_block16``).
- SF columns beyond ``hidden / vec`` (kernel row padding) are zeroed.
- ``topk_idx = id`` for ``0 <= id < E``, -1 for ``id == -1`` (masked route) or for
  every route of a row with ``is_padding[row]`` set (when given: the caller's
  ``torch.where(is_padding[:, None], -1, topk_ids)`` folded into this launch);
  ``topk_weights = w`` (fp32) for valid ids, else 0.
- Any other id (``< -1`` or ``>= E``) is an error, as in FlashInfer's
  ``validate_routing_values``: the route is masked so the kernel never indexes out of
  range, and the kernel sets the sticky device flag ``route_error`` (int64 ``[2]``:
  ``[1, offending id]``) for the caller to raise on (``RubinMegaMoE`` checks it at
  warmup, on demand and asynchronously after eager forwards; never a silent drop).

``E`` (``num_experts``) is the PHYSICAL expert count of the EP group and placement must
be linear: global id ``g`` lives on rank ``g // E_local``. Logical-to-physical mapping
(EPLB) happens before this call.

Rows ``[n, capacity)``: ``topk_idx = -1``; activations / weights are left as they are
(masked routes are never read). ``reference.stage_inputs_torch`` is the pure-torch
twin.

Preconditions not checked per call (no host sync): no duplicate expert id within a
row, finite routing weights.

``n_tokens`` and the input strides are not specialized (``do_not_specialize``), so a
token count or row stride first seen during CUDA graph capture does not recompile;
dtype combinations are compiled by ``RubinMegaMoE.warmup``.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1_code(m):
    # |y| (<= 6, non-NaN) -> E2M1 magnitude code, round to nearest, ties to even code.
    code = tl.where(m <= 0.25, 0, 1)
    code = tl.where(m >= 0.75, 2, code)
    code = tl.where(m > 1.25, 3, code)
    code = tl.where(m >= 1.75, 4, code)
    code = tl.where(m > 2.5, 5, code)
    code = tl.where(m >= 3.5, 6, code)
    code = tl.where(m > 5.0, 7, code)
    return code


# Dead rows [n, capacity) written per dead-row program (topk_idx = -1 only).
DEAD_ROWS = 64


@triton.jit(
    do_not_specialize=[
        "n_tokens",
        "n_dead_rows",
        "hid_stride",
        "ids_stride",
        "wts_stride",
    ]
)
def _stage_kernel(
    hid_ptr,
    hid_stride,
    ids_ptr,
    ids_stride,
    wts_ptr,
    wts_stride,
    x_ptr,
    x_stride,
    sf_ptr,
    sf_stride,
    idx_ptr,
    sw_ptr,
    err_ptr,
    pad_ptr,
    n_tokens,
    n_dead_rows,
    num_experts,
    scale_mult,
    norm_const,
    H: tl.constexpr,
    CH: tl.constexpr,
    VEC: tl.constexpr,
    NVFP4: tl.constexpr,
    SF_COLS: tl.constexpr,
    SF_PAD_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    HAS_PAD: tl.constexpr,
    DEAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    live = row < n_tokens
    if chunk == 0:
        k = tl.arange(0, TOPK_P2)
        km = k < TOPK
        if live:
            ids = tl.load(ids_ptr + row * ids_stride + k, mask=km, other=-1).to(
                tl.int64
            )
            if HAS_PAD:
                padded = tl.load(pad_ptr + row) != 0
                ids = tl.where(padded, -1, ids)
            w = tl.load(wts_ptr + row * wts_stride + k, mask=km, other=0.0).to(
                tl.float32
            )
            valid = (ids >= 0) & (ids < num_experts)
            bad = km & (ids != -1) & (~valid)
            # Sticky error flag: [1, offending id] (racing writers store equal flags).
            tl.store(err_ptr + k * 0, tl.full([TOPK_P2], 1, tl.int64), mask=bad)
            tl.store(err_ptr + 1 + k * 0, ids, mask=bad)
            tl.store(
                idx_ptr + row * TOPK + k, tl.where(valid, ids, -1).to(tl.int32), mask=km
            )
            tl.store(sw_ptr + row * TOPK + k, tl.where(valid, w, 0.0), mask=km)
            if SF_COLS > H // VEC:
                p = tl.arange(0, SF_PAD_P2)
                tl.store(
                    sf_ptr + row * sf_stride + H // VEC + p,
                    tl.zeros([SF_PAD_P2], dtype=tl.uint8),
                    mask=p < SF_COLS - H // VEC,
                )
        else:
            # Dead-row program: -1 routes for DEAD rows starting at ``base``.
            base = n_tokens + (row - n_tokens) * DEAD
            r = tl.arange(0, DEAD)[:, None]
            m = (base + r < n_tokens + n_dead_rows) & km[None, :]
            tl.store(
                idx_ptr + (base + r) * TOPK + k[None, :],
                tl.full([DEAD, TOPK_P2], -1, tl.int32),
                mask=m,
            )
    if live:
        offs = chunk * CH + tl.arange(0, CH)
        v = tl.load(hid_ptr + row * hid_stride + offs).to(tl.float32)
        vb = tl.reshape(v, (CH // VEC, VEC))
        amax = tl.max(tl.abs(vb), axis=1)
        sf_offs = chunk * (CH // VEC) + tl.arange(0, CH // VEC)
        if NVFP4:
            s = tl.minimum(amax * scale_mult, 448.0)
            s8 = s.to(tl.float8e4nv)
            srt = s8.to(tl.float32)
            # IEEE division (Triton's "/" is approximate): norm_const / scale.
            acc = tl.where(
                srt > 0.0,
                tl.math.div_rn(
                    tl.zeros_like(srt) + norm_const,
                    tl.maximum(srt, 1.1754943508222875e-38),
                ),
                0.0,
            )
            y = vb * acc[:, None]
            m = tl.abs(y)
            m = tl.where(m == m, tl.minimum(m, 6.0), 0.0)
            code = _e2m1_code(m)
            neg = (y.to(tl.int32, bitcast=True) < 0) & (code != 0)
            nib = (code | tl.where(neg, 8, 0)).to(tl.uint8)
            lo, hi = tl.split(tl.reshape(nib, (CH // 2, 2)))
            packed = lo | (hi << 4)
            tl.store(
                x_ptr + row * x_stride + chunk * (CH // 2) + tl.arange(0, CH // 2),
                packed,
            )
            tl.store(sf_ptr + row * sf_stride + sf_offs, s8.to(tl.uint8, bitcast=True))
        else:
            t = amax * (1.0 / 448.0)
            t = tl.where((t == t) & (t < float("inf")), t, 0.0)
            t = tl.maximum(t, 1.1754943508222875e-38)
            bits = t.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            e = tl.minimum(tl.maximum(e, -126), 127)
            scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
            q = tl.math.div_rn(
                vb, tl.broadcast_to(scale[:, None], (CH // VEC, VEC))
            ).to(tl.float8e4nv)
            tl.store(
                x_ptr + row * x_stride + offs,
                tl.reshape(q, (CH,)).to(tl.uint8, bitcast=True),
            )
            sfb = tl.minimum(tl.maximum(e + 127, 1), 254).to(tl.uint8)
            tl.store(sf_ptr + row * sf_stride + sf_offs, sfb)


def _chunk(hidden: int, vec: int) -> int:
    for ch in (1024, 512, 256, 128, 64):
        if ch >= 2 * vec and hidden % ch == 0:
            return ch
    raise ValueError(f"hidden {hidden} has no power-of-two staging chunk >= {2 * vec}.")


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    staged_weights: torch.Tensor,
    *,
    route_error: torch.Tensor,
    quant_kind: str,
    num_experts: int,
    input_norm_const: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Stage ``hidden_states[:n]`` and its routes into a session's buffers (n may be
    0). ``route_error``: the caller's sticky int64 ``[2]`` device flag (see module
    docstring); ``num_experts``: the physical expert count (linear placement);
    ``is_padding``: optional bool ``(n,)`` device mask, a set row's routes are all
    masked to -1 (activations are still staged, as for any row)."""
    if (
        route_error.dtype != torch.int64
        or route_error.numel() != 2
        or route_error.device != topk_idx.device
    ):
        raise ValueError("route_error must be an int64 [2] tensor on the device.")
    n, hidden = hidden_states.shape
    capacity, topk = topk_idx.shape
    nvfp4 = quant_kind == "nvfp4"
    vec = 16 if nvfp4 else 32
    if n > capacity:
        raise ValueError(f"{n} tokens exceed the bucket capacity {capacity}.")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.stride(-1) != 1:
        raise ValueError("hidden_states must be BF16 with unit inner stride.")
    if topk_ids.shape != (n, topk) or topk_weights.shape != (n, topk):
        raise ValueError(
            f"topk_ids / topk_weights must be ({n}, {topk}), got "
            f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}."
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64.")
    if topk_weights.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("topk_weights must be fp32, bf16 or fp16.")
    if topk_ids.stride(-1) != 1 or topk_weights.stride(-1) != 1:
        raise ValueError("topk_ids / topk_weights need unit inner stride.")
    sf_cols = x_sf.shape[1]
    if sf_cols < hidden // vec:
        raise ValueError(f"x_sf has {sf_cols} columns < hidden/{vec}.")
    if topk_idx.dtype != torch.int32 or staged_weights.dtype != torch.float32:
        raise ValueError("session route buffers must be int32 ids and fp32 weights.")
    if is_padding is not None and (
        is_padding.dtype != torch.bool
        or is_padding.shape != (n,)
        or is_padding.stride(0) != 1
        or is_padding.device != topk_idx.device
    ):
        raise ValueError(
            f"is_padding must be a contiguous bool ({n},) tensor on the device, got "
            f"{is_padding.dtype} {tuple(is_padding.shape)} on {is_padding.device}."
        )
    ch = _chunk(hidden, vec)
    x_u8 = x.view(torch.uint8)
    sf_u8 = x_sf.view(torch.uint8)
    pad = sf_cols - hidden // vec
    n_dead = capacity - n
    grid = (n + triton.cdiv(n_dead, DEAD_ROWS), hidden // ch)
    _stage_kernel[grid](
        hidden_states,
        hidden_states.stride(0),
        topk_ids,
        topk_ids.stride(0),
        topk_weights,
        topk_weights.stride(0),
        x_u8,
        x_u8.stride(0),
        sf_u8,
        sf_u8.stride(0),
        topk_idx,
        staged_weights,
        route_error,
        is_padding if is_padding is not None else route_error,
        n,
        n_dead,
        num_experts,
        float(input_norm_const) / 6.0,
        float(input_norm_const),
        H=hidden,
        CH=ch,
        VEC=vec,
        NVFP4=nvfp4,
        SF_COLS=sf_cols,
        SF_PAD_P2=max(1, triton.next_power_of_2(max(pad, 1))),
        TOPK=topk,
        TOPK_P2=triton.next_power_of_2(topk),
        HAS_PAD=is_padding is not None,
        DEAD=DEAD_ROWS,
        num_warps=4,
    )
