# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch references for the Rubin MegaMoE backend (bench / tests only).

- :func:`stage_inputs_torch`: the staging semantics of ``staging.stage_inputs``
  (FlashInfer's torch ``stage_mega_moe_inputs`` plus out-of-range id masking), for GPU
  A/B.
- :func:`megamoe_oracle`: FlashInfer ``compute_megamoe_reference_sm107_block_scaled``
  (shim/kernel_helpers.py, hash in VENDOR.md) for BF16 combine, SiTU, routing weights at
  FC1: the quantization-faithful oracle from staged bytes and kernel-layout weights.
- :func:`bf16_partial`: full-precision MoE of BF16 tokens with the dequantized MXFP4
  weights (exact) and SiTU, the model's math.
- :func:`sampled_reference`: FlashInfer ``shim/correctness.py`` collective pattern; each
  rank evaluates its local experts for every source rank's sampled rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from .config import RouteIdError
from .weights import (
    GATE_UP_INTERLEAVE,
    MX_BLOCK,
    NV_BLOCK,
    dequant_mxfp4,
    e4m3_to_f32,
    e8m0_to_f32,
    from_blocked,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
    unpack_fp4_to_f32,
)


def stage_inputs_torch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    quant_kind: str,
    capacity: int,
    sf_cols: int,
    num_experts: int,
    input_norm_const: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x uint8, x_sf uint8, topk_idx int32, topk_weights fp32)`` at full capacity.

    Rows past ``n`` hold zeros / -1 (the Triton stager leaves stale activations there;
    compare live rows only). Ids other than -1 outside ``[0, num_experts)`` raise
    :class:`RouteIdError` (the Triton stager sets its device error flag instead)."""
    n, hidden = hidden_states.shape
    if n:
        ids64 = topk_ids.to(torch.int64)
        bad = (ids64 != -1) & ((ids64 < 0) | (ids64 >= num_experts))
        if bool(bad.any()):
            raise RouteIdError(int(ids64[bad][0]), num_experts)
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    nvfp4 = quant_kind == "nvfp4"
    vec = NV_BLOCK if nvfp4 else MX_BLOCK
    x = torch.zeros(
        (capacity, hidden // 2 if nvfp4 else hidden), dtype=torch.uint8, device=dev
    )
    x_sf = torch.zeros((capacity, sf_cols), dtype=torch.uint8, device=dev)
    idx = torch.full((capacity, topk), -1, dtype=torch.int32, device=dev)
    wts = torch.zeros((capacity, topk), dtype=torch.float32, device=dev)
    if n:
        if nvfp4:
            q, sf = quantize_nvfp4_block16(
                hidden_states.float(), norm_const=input_norm_const
            )
        else:
            q, sf = quantize_mxfp8_block32(hidden_states.float())
        x[:n] = q.view(torch.uint8)
        x_sf[:n, : hidden // vec] = sf.view(torch.uint8)
        ids = topk_ids.to(torch.int64)
        valid = (ids >= 0) & (ids < num_experts)
        idx[:n] = torch.where(valid, ids, -1).to(torch.int32)
        wts[:n] = torch.where(valid, topk_weights.float(), 0.0)
    return x, x_sf, idx, wts


def _situ(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float):
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _dequant_activation(quant_kind, x_data, x_sf, hidden):
    if quant_kind == "nvfp4":
        data = unpack_fp4_to_f32(x_data)[:, :hidden]
        sf = e4m3_to_f32(x_sf[:, : hidden // NV_BLOCK])
        return data * sf.repeat_interleave(NV_BLOCK, dim=1)
    data = x_data.view(torch.float8_e4m3fn)[:, :hidden].float()
    sf = e8m0_to_f32(x_sf[:, : hidden // MX_BLOCK])
    return data * sf.repeat_interleave(MX_BLOCK, dim=1)


def _dequant_weight(quant_kind, weight_k_major, sf_raw):
    vec = NV_BLOCK if quant_kind == "nvfp4" else MX_BLOCK
    scale = e4m3_to_f32(sf_raw) if quant_kind == "nvfp4" else e8m0_to_f32(sf_raw)
    return unpack_fp4_to_f32(weight_k_major) * scale.repeat_interleave(vec, dim=1)


def megamoe_oracle(
    x_data: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights,
    *,
    quant_kind: str,
    local_expert_offset: int,
    hidden: int,
    intermediate: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """fp32 ``(rows, hidden)`` partial of this rank's local experts (kernel-layout
    ``weights``: a ``RubinMoeWeights``), from staged bytes."""
    x = _dequant_activation(quant_kind, x_data, x_sf, hidden)
    idx = topk_idx.to(torch.int64)
    out = torch.zeros((x_data.shape[0], hidden), dtype=torch.float32, device=x.device)
    fc1_k_major = weights.fc1[0].permute(0, 2, 1)  # (E, 2I, H/2) physical
    fc2_k_major = weights.fc2[0].permute(0, 2, 1)  # (E, H, I/2)
    vec = NV_BLOCK if quant_kind == "nvfp4" else MX_BLOCK
    for local_e in range(fc1_k_major.shape[0]):
        routed = (idx == local_expert_offset + local_e).nonzero(as_tuple=False)
        if routed.shape[0] == 0:
            continue
        src_t, src_k = routed[:, 0], routed[:, 1]
        sf1 = from_blocked(weights.fc1[1][local_e], 2 * intermediate, hidden // vec)
        sf2 = from_blocked(weights.fc2[1][local_e], hidden, intermediate // vec)
        w1 = _dequant_weight(quant_kind, fc1_k_major[local_e], sf1)
        fc1 = x[src_t] @ w1.t()  # (v, 2I) in (pair, {gate, up}, 16) order
        if weights.fc1_alpha is not None:
            fc1 = fc1 * weights.fc1_alpha[local_e]
        pairs = fc1.view(-1, intermediate // GATE_UP_INTERLEAVE, 2, GATE_UP_INTERLEAVE)
        act = _situ(pairs[:, :, 0, :], pairs[:, :, 1, :], situ_beta, situ_linear_beta)
        act = act.reshape(-1, intermediate)
        act = act * topk_weights[src_t, src_k].float().unsqueeze(-1)
        if quant_kind == "nvfp4":
            norm = (
                1.0
                if weights.fc1_norm_const is None
                else weights.fc1_norm_const[local_e]
            )
            q, sf = quantize_nvfp4_block16(act, norm)
            act = unpack_fp4_to_f32(q) * e4m3_to_f32(sf).repeat_interleave(
                NV_BLOCK, dim=1
            )
        else:
            q, sf = quantize_mxfp8_block32(act)
            act = q.float() * e8m0_to_f32(sf).repeat_interleave(MX_BLOCK, dim=1)
        w2 = _dequant_weight(quant_kind, fc2_k_major[local_e], sf2)
        term = act @ w2.t()
        if weights.fc2_alpha is not None:
            term = term * weights.fc2_alpha[local_e]
        term = term.to(torch.bfloat16).float()  # BF16 combine wire
        out.index_add_(0, src_t, term)
    return out


def bf16_partial(
    hidden_rows: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    canonical: Sequence[torch.Tensor],
    *,
    local_expert_offset: int,
    intermediate: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """fp32 partial of this rank's local experts: BF16 tokens, exact MXFP4 weights.

    ``canonical`` is ``(w13, w13_scale, w2, w2_scale)`` of the local experts (canonical
    gate||up row order)."""
    w13, w13_scale, w2, w2_scale = canonical
    x = hidden_rows.float()
    idx = topk_ids.to(torch.int64)
    out = torch.zeros_like(x)
    for local_e in range(w13.shape[0]):
        routed = (idx == local_expert_offset + local_e).nonzero(as_tuple=False)
        if routed.shape[0] == 0:
            continue
        src_t, src_k = routed[:, 0], routed[:, 1]
        w1 = dequant_mxfp4(w13[local_e], w13_scale[local_e])
        h = x[src_t] @ w1.t()
        act = _situ(
            h[:, :intermediate], h[:, intermediate:], situ_beta, situ_linear_beta
        )
        act = act * topk_weights[src_t, src_k].float().unsqueeze(-1)
        term = act @ dequant_mxfp4(w2[local_e], w2_scale[local_e]).t()
        out.index_add_(0, src_t, term)
    return out


def sampled_reference(
    rows: Sequence[torch.Tensor],
    num_tokens: int,
    partial_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    *,
    group: dist.ProcessGroup | None = None,
    sample_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(row indices, fp32 expected rows)`` for this rank's tokens.

    ``rows`` are per-token tensors (same leading dim) whose sampled rows every rank
    needs; ``partial_fn(payload)`` returns the fp32 contribution of the local experts.
    Collective over ``group``: one broadcast + reduce per source rank."""
    count = min(sample_size, num_tokens)
    device = rows[0].device
    indices = torch.linspace(
        0, max(num_tokens - 1, 0), max(count, 1), device=device
    ).long()
    local = []
    for t in rows:
        buf = torch.zeros((sample_size, *t.shape[1:]), dtype=t.dtype, device=device)
        if t.dtype in (torch.int32, torch.int64):
            buf.fill_(-1)
        if count:
            buf[:count] = t[indices[:count]]
        local.append(buf)
    world = (
        1 if group is None or not dist.is_initialized() else dist.get_world_size(group)
    )
    rank = 0 if world == 1 else dist.get_rank(group)
    expected = None
    for source in range(world):
        payload = [t.clone() for t in local]
        if world > 1:
            src = dist.get_global_rank(group or dist.group.WORLD, source)
            for t in payload:
                dist.broadcast(t, src=src, group=group)
        partial = partial_fn(payload)
        if world > 1:
            dist.reduce(partial, dst=src, op=dist.ReduceOp.SUM, group=group)
        if source == rank:
            expected = partial[:count].clone()
    return indices[:count], expected


def output_error(
    output: torch.Tensor, indices: torch.Tensor, expected: torch.Tensor
) -> float:
    """Relative L2 on the sampled rows; any non-finite output fails (inf)."""
    actual = output[indices].float()
    if not bool(torch.isfinite(actual).all()):
        return float("inf")
    denominator = expected.float().norm().clamp_min(1e-12)
    return float((actual - expected.float()).norm() / denominator)
