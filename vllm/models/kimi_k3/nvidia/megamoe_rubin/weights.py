# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 routed-expert weights -> the Rubin MegaMoE kernel layout (pure torch,
CPU-safe).

Input contract (canonical, one layer, this rank's local experts ``[r*E_l, (r+1)*E_l)``):

- ``w13``       ``(E_l, 2I, H/2)``  packed E2M1 (element ``2j`` in the low nibble); rows
  ``[0, I)`` are gate (w1), rows ``[I, 2I)`` are up (w3)
- ``w13_scale`` ``(E_l, 2I, H/32)`` UE8M0 bytes (``2^(b-127)``, 255 = NaN)
- ``w2``        ``(E_l, H, I/2)``,  ``w2_scale`` ``(E_l, H, I/32)``

Kernel quant kinds:

- ``mxfp4_mxfp8`` and ``mxfp4`` (MXFP8 or MXFP4 activations): the MXFP4 weights as
  they are (port of FlashInfer ``preprocess_prequantized_block_scaled_weights``, one
  layout for both): 16-row gate/up interleave of data and scales, 32x4x4 scale-atom
  swizzle, K-major data handed over as the logical ``(E, K/2, N)`` view.
- ``nvfp4``: an EXACT MXFP4 -> NVFP4 conversion. The E2M1 codes are unchanged; every
  per-32 UE8M0 scale ``2^k`` becomes two per-16 E4M3 scales ``2^(k-c)`` against one FP32
  global scale ``2^c`` per (expert, GEMM), ``c = max(kmax - 8, -126)``. The dequantized
  weight is bit-identical to the MXFP4 one iff every live block of that (expert, GEMM)
  has ``k - c >= -9`` (E4M3 represents ``2^-9 .. 2^8`` exactly; ``2^-9 .. 2^-7`` as
  subnormals), i.e. ``kmax - kmin <= 17``. All-zero blocks take scale 0. Anything else
  (wider spread, a NaN scale byte, a byte-0 scale on non-zero codes) raises
  :class:`NonRepresentableError` with a per-(expert, GEMM) report; nothing is
  re-quantized.

The per-expert NVFP4 epilogue scalars are ``fc1_alpha = 2^c1 / input_norm_const``,
``fc2_alpha = 2^c2 / fc1_norm_const`` and ``fc1_norm_const``; with power-of-two norm
constants (enforced) they are exact in FP32.

This module imports only torch and the standard library (no cutlass / triton / vLLM), so
the CPU unit tests can load it on its own.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from typing import NamedTuple

import torch

MX_BLOCK = 32
NV_BLOCK = 16
GATE_UP_INTERLEAVE = 16  # fixed by the swap-AB gated-activation epilogue
SF_ATOM_ROWS = 128
SF_ATOM_COLS = 4
E4M3_MAX = 448.0
E2M1_MAX = 6.0
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

QUANT_KINDS = ("mxfp4_mxfp8", "nvfp4", "mxfp4")

# E4M3 window for exact power-of-two scales: 2^-9 (subnormal 0x01) .. 2^8 (0x78).
_M_MIN = -9
_M_MAX = 8
_M_MIN_NORMAL = -6


def fp4_storage_dtype() -> torch.dtype:
    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def is_power_of_two(value: float) -> bool:
    """Positive, finite and an exact power of two (``2^n``, any integer ``n``)."""
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
        and math.frexp(float(value))[0] == 0.5
    )


def _u8(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {t.dtype}.")
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


# --------------------------------------------------------------------------------------
# FP4 / scale helpers (ported from FlashInfer shim/kernel_helpers.py, see VENDOR.md)
# --------------------------------------------------------------------------------------


@functools.cache
def _e2m1_values(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)


def ceil_log2_exponent(x: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(x))`` clamped to FP32 normal exponents, by bit decomposition."""
    finite = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    bits = finite.clamp_min(2.0**-126).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    return (exponent + has_mantissa.to(torch.int32)).clamp(-126, 127)


def pow2_f32(exponent: torch.Tensor) -> torch.Tensor:
    """``2.0 ** exponent`` for int32 exponents in [-126, 127], via fp32 bits."""
    return ((exponent.to(torch.int32) + 127) << 23).view(torch.float32)


def pack_f32_to_fp4(fp32: torch.Tensor) -> torch.Tensor:
    """Round fp32 to E2M1 (nearest, ties to even code, satfinite) and nibble-pack
    (uint8)."""
    if fp32.shape[-1] % 2:
        raise ValueError(f"FP4 pack needs an even trailing dim, got {fp32.shape[-1]}.")
    values = _e2m1_values(fp32.device)
    magnitude = torch.nan_to_num(fp32, nan=0.0).abs().clamp(max=E2M1_MAX)
    distance = (magnitude.unsqueeze(-1) - values).abs()
    parity = torch.arange(len(_E2M1_VALUES), device=fp32.device) % 2
    tie_rank = torch.where(
        distance == distance.amin(dim=-1, keepdim=True),
        parity.to(distance.dtype).expand_as(distance),
        torch.full_like(distance, len(_E2M1_VALUES)),
    )
    code = tie_rank.argmin(dim=-1).to(torch.uint8)
    sign = (torch.signbit(fp32) & (code != 0)).to(torch.uint8) << 3
    nibble = code | sign
    pairs = nibble.reshape(*nibble.shape[:-1], -1, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def unpack_fp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Nibble-packed E2M1 -> fp32, trailing dim doubled."""
    raw = _u8(packed)
    values = _e2m1_values(raw.device)
    nibbles = torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(*raw.shape[:-1], -1)
    magnitude = values[(nibbles & 0x7).to(torch.int64)]
    return torch.where(nibbles & 0x8 != 0, -magnitude, magnitude)


def e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes -> fp32 (byte 0 = 2^-127, 255 = NaN)."""
    codes = _u8(scale).to(torch.int32)
    bits = codes << 23
    bits = torch.where(codes == 0, 0x00400000, bits)
    bits = torch.where(codes == 255, 0x7FC00000, bits)
    return bits.view(torch.float32)


def e4m3_to_f32(scale: torch.Tensor) -> torch.Tensor:
    return _u8(scale).view(torch.float8_e4m3fn).to(torch.float32)


def quantize_mxfp4_block32(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E2M1 codes (uint8 packed) + per-32 UE8M0 bytes, scales rounded up to powers of
    two."""
    if tensor.shape[-1] % MX_BLOCK:
        raise ValueError("MXFP4 trailing dim must be a multiple of 32.")
    fp32 = tensor.to(torch.float32)
    blocked = fp32.reshape(*fp32.shape[:-1], -1, MX_BLOCK)
    exponent = ceil_log2_exponent(blocked.abs().amax(dim=-1) * (1.0 / E2M1_MAX))
    data = pack_f32_to_fp4(
        (blocked / pow2_f32(exponent).unsqueeze(-1)).reshape(fp32.shape)
    )
    return data, (exponent + 127).to(torch.uint8)


def quantize_mxfp8_block32(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 E4M3 data + per-32 UE8M0 bytes (the activation staging recipe)."""
    if tensor.shape[-1] % MX_BLOCK:
        raise ValueError("MXFP8 trailing dim must be a multiple of 32.")
    fp32 = tensor.to(torch.float32)
    blocked = fp32.reshape(*fp32.shape[:-1], -1, MX_BLOCK)
    exponent = ceil_log2_exponent(blocked.abs().amax(dim=-1) * (1.0 / E4M3_MAX))
    data = (
        (blocked / pow2_f32(exponent).unsqueeze(-1))
        .reshape(fp32.shape)
        .to(torch.float8_e4m3fn)
    )
    return data, (exponent + 127).clamp(1, 254).to(torch.uint8)


def quantize_nvfp4_block16(
    tensor: torch.Tensor, norm_const: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 (packed E2M1 uint8, E4M3 scale bytes) of ``norm_const * tensor``.

    ``scale = e4m3(absmax / 6 * norm_const)``, data scaled by ``norm_const / scale``
    (zero where the scale is zero): the kernel epilogue's ``nvfp4_quant_impl``.
    """
    if tensor.shape[-1] % NV_BLOCK:
        raise ValueError("NVFP4 trailing dim must be a multiple of 16.")
    fp32 = tensor.to(torch.float32)
    blocked = fp32.reshape(*fp32.shape[:-1], -1, NV_BLOCK)
    absmax = blocked.abs().amax(dim=-1)
    scale_f32 = (absmax * (norm_const / E2M1_MAX)).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    scale_rt = scale_fp8.to(torch.float32)
    acc_scale = torch.where(
        scale_rt > 0, norm_const / scale_rt.clamp_min(2.0**-126), 0.0
    )
    scaled = (blocked * acc_scale.unsqueeze(-1)).reshape(fp32.shape)
    return pack_f32_to_fp4(scaled), scale_fp8.view(torch.uint8)


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``(..., R, K/2)`` codes + ``(..., R, K/32)`` UE8M0 -> fp32 ``(..., R, K)``."""
    return unpack_fp4_to_f32(codes) * e8m0_to_f32(scales).repeat_interleave(
        MX_BLOCK, dim=-1
    )


def dequant_nvfp4(
    codes: torch.Tensor, scales16: torch.Tensor, global_exponent: torch.Tensor
) -> torch.Tensor:
    """``(E, R, K/2)`` codes + ``(E, R, K/16)`` E4M3 bytes + per-expert ``2^c`` -> fp32.
    """
    global_scale = pow2_f32(global_exponent).view(-1, *([1] * (codes.dim() - 1)))
    return (
        unpack_fp4_to_f32(codes)
        * e4m3_to_f32(scales16).repeat_interleave(NV_BLOCK, dim=-1)
        * global_scale
    )


def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """One raw 2D scale plane -> padded, 32x4x4 atom-swizzled, flat (FlashInfer
    verbatim)."""
    if scale_2d.dim() != 2:
        raise ValueError(f"expected a 2D scale tensor, got {scale_2d.dim()}D.")
    rows, cols = scale_2d.shape
    if rows == 0 or cols == 0:
        return scale_2d.new_empty((0,))
    row_blocks = ceil_div(rows, SF_ATOM_ROWS)
    col_blocks = ceil_div(cols, SF_ATOM_COLS)
    padded_rows = row_blocks * SF_ATOM_ROWS
    padded_cols = col_blocks * SF_ATOM_COLS
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(row_blocks, SF_ATOM_ROWS, col_blocks, SF_ATOM_COLS).permute(
        0, 2, 1, 3
    )
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def to_blocked_batched(scales: torch.Tensor) -> torch.Tensor:
    """``(E, rows, cols)`` uint8 -> ``(E, swizzled_flat_sf_size(rows, cols))``.

    Same bytes as ``torch.stack([to_blocked(s) for s in scales])`` in one pass.
    """
    if scales.dim() != 3:
        raise ValueError(f"expected (E, rows, cols) scales, got {tuple(scales.shape)}.")
    experts, rows, cols = scales.shape
    row_blocks = ceil_div(rows, SF_ATOM_ROWS)
    col_blocks = ceil_div(cols, SF_ATOM_COLS)
    padded_rows = row_blocks * SF_ATOM_ROWS
    padded_cols = col_blocks * SF_ATOM_COLS
    padded = scales
    if (rows, cols) != (padded_rows, padded_cols):
        padded = scales.new_zeros((experts, padded_rows, padded_cols))
        padded[:, :rows, :cols] = scales
    blocks = padded.reshape(
        experts, row_blocks, SF_ATOM_ROWS, col_blocks, SF_ATOM_COLS
    ).permute(0, 1, 3, 2, 4)
    return blocks.reshape(experts, -1, 4, 32, 4).transpose(2, 3).reshape(experts, -1)


def swizzled_flat_sf_size(rows: int, cols: int) -> int:
    return round_up(rows, SF_ATOM_ROWS) * round_up(cols, SF_ATOM_COLS)


def from_blocked(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """One expert's swizzled bytes -> the logical ``(rows, cols)`` plane (uint8)."""
    rb, cb = ceil_div(rows, SF_ATOM_ROWS), ceil_div(cols, SF_ATOM_COLS)
    raw = _u8(scale).reshape(rb, cb, 32, 4, 4)
    raw = raw.permute(0, 1, 3, 2, 4).reshape(rb, cb, SF_ATOM_ROWS, SF_ATOM_COLS)
    raw = raw.permute(0, 2, 1, 3).reshape(rb * SF_ATOM_ROWS, cb * SF_ATOM_COLS)
    return raw[:rows, :cols].contiguous()


def interleave_gate_up_16(w13: torch.Tensor, *, intermediate_size: int) -> torch.Tensor:
    """Canonical gate||up rows -> the kernel's ``(I/16, {gate, up}, 16)`` row stripes.
    """
    fc1_out = 2 * intermediate_size
    if w13.shape[-2] != fc1_out:
        raise ValueError(f"w13 N axis {w13.shape[-2]} != 2*intermediate {fc1_out}.")
    if intermediate_size % GATE_UP_INTERLEAVE:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be a multiple of "
            f"{GATE_UP_INTERLEAVE}."
        )
    pairs = intermediate_size // GATE_UP_INTERLEAVE
    gate, up = w13.unflatten(-2, (2, intermediate_size)).unbind(dim=-3)
    gate = gate.unflatten(-2, (pairs, GATE_UP_INTERLEAVE))
    up = up.unflatten(-2, (pairs, GATE_UP_INTERLEAVE))
    return torch.stack((gate, up), dim=-3).flatten(-4, -2)


def deinterleave_gate_up_16(
    w13: torch.Tensor, *, intermediate_size: int
) -> torch.Tensor:
    """Inverse of :func:`interleave_gate_up_16`."""
    pairs = intermediate_size // GATE_UP_INTERLEAVE
    stripes = w13.unflatten(-2, (pairs, 2, GATE_UP_INTERLEAVE))
    gate, up = stripes.unbind(dim=-3)
    return torch.cat((gate.flatten(-3, -2), up.flatten(-3, -2)), dim=-2)


# ---------------------------------------------------------------------------------
# Checkpoint expert keys
# ---------------------------------------------------------------------------------

# Kimi K3.1 MXFP4 checkpoints (compressed-tensors) name the expert codes
# ``weight_packed`` (U8, (rows, cols/2)) and the UE8M0 scales ``weight_scale`` (U8,
# (rows, cols/32)), e.g.
# ``language_model.model.layers.5.block_sparse_moe.experts.7.w1.weight_packed``;
# ``weight`` is accepted for unpacked exports. ``shared_experts`` never match.
_EXPERT_KEY = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:.+\.)?experts\.(\d+)\.(w[123])\."
    r"(weight_packed|weight|weight_scale)$"
)


def parse_expert_key(key: str) -> tuple[int, int, str, str] | None:
    """``(layer, expert, "w1"|"w2"|"w3", "weight"|"weight_scale")`` of a routed
    expert tensor key (``weight_packed`` maps to ``"weight"``), else None."""
    m = _EXPERT_KEY.search(key)
    if m is None:
        return None
    part = "weight" if m.group(4) == "weight_packed" else m.group(4)
    return int(m.group(1)), int(m.group(2)), m.group(3), part


# ---------------------------------------------------------------------------------
# Canonical-input validation
# ---------------------------------------------------------------------------------


def check_canonical_mxfp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
) -> int:
    """Validate the canonical MXFP4 layer weights; returns the local expert count."""
    if any(t.dim() != 3 for t in (w13, w13_scale, w2, w2_scale)):
        raise ValueError("canonical w13 / w13_scale / w2 / w2_scale must all be 3D.")
    experts = w13.shape[0]
    expected = {
        "w13": (w13, (experts, 2 * intermediate, hidden // 2)),
        "w13_scale": (w13_scale, (experts, 2 * intermediate, hidden // MX_BLOCK)),
        "w2": (w2, (experts, hidden, intermediate // 2)),
        "w2_scale": (w2_scale, (experts, hidden, intermediate // MX_BLOCK)),
    }
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}.")
        if tensor.element_size() != 1:
            raise TypeError(f"{name} must be a 1-byte dtype, got {tensor.dtype}.")
        if tensor.device != w13.device:
            raise ValueError("all canonical weights and scales must share a device.")
    for name in ("w13_scale", "w2_scale"):
        if expected[name][0].dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError(f"{name} must be UE8M0 (uint8 or float8_e8m0fnu).")
    if experts <= 0 or hidden % (4 * MX_BLOCK) or intermediate % (2 * MX_BLOCK):
        raise ValueError(
            "need experts > 0, hidden % 128 == 0 and intermediate % 64 == 0 "
            f"(got E={experts}, H={hidden}, I={intermediate})."
        )
    return experts


# --------------------------------------------------------------------------------------
# Exact MXFP4 -> NVFP4 conversion
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExpertLegStats:
    """Scale statistics of one (expert, GEMM) for the MXFP4 -> NVFP4 conversion.

    ``leg`` is ``"fc1"`` (w13, gate and up together: one alpha per expert) or ``"fc2"``
    (w2). ``kmin`` / ``kmax`` are the UE8M0 exponents over live blocks (None when the
    expert has no live block); ``c`` the chosen global exponent. ``subnormal`` counts
    live blocks whose E4M3 scale is subnormal (``2^-9 .. 2^-7``); ``nonrep`` counts
    blocks that cannot be converted exactly (spread > 17, or a NaN scale byte).
    """

    expert: int
    leg: str
    kmin: int | None
    kmax: int | None
    c: int
    live: int
    zero: int
    nan: int
    subnormal: int
    nonrep: int

    @property
    def spread(self) -> int | None:
        return None if self.kmax is None else self.kmax - self.kmin


@dataclasses.dataclass(frozen=True)
class ConversionReport:
    layer_id: int | None
    stats: tuple[ExpertLegStats, ...]

    @property
    def offending(self) -> tuple[ExpertLegStats, ...]:
        return tuple(s for s in self.stats if s.nonrep)

    @property
    def max_spread(self) -> int | None:
        spreads = [s.spread for s in self.stats if s.spread is not None]
        return max(spreads) if spreads else None

    @property
    def subnormal_blocks(self) -> int:
        return sum(s.subnormal for s in self.stats)

    def summary(self) -> str:
        head = (
            f"layer {self.layer_id}: {len(self.stats)} (expert, leg) entries, max "
            f"spread {self.max_spread},subnormal-scale blocks {self.subnormal_blocks}, "
            f"non-representable entries {len(self.offending)}"
        )
        lines = [head]
        for s in self.offending[:64]:
            lines.append(
                f"  layer {self.layer_id} leg {s.leg} expert {s.expert}: kmin {s.kmin} "
                f"kmax {s.kmax} spread {s.spread} live {s.live} nan {s.nan} "
                f"non-representable blocks {s.nonrep}"
            )
        if len(self.offending) > 64:
            lines.append(f"  ... {len(self.offending) - 64} more")
        return "\n".join(lines)


class NonRepresentableError(ValueError):
    """The MXFP4 weights of some (expert, GEMM) have no exact NVFP4 encoding."""

    def __init__(self, report: ConversionReport):
        self.report = report
        super().__init__(
            "MXFP4 -> NVFP4 conversion is not exact for these (expert, GEMM) entries "
            "(the backend never re-quantizes; use quant kind 'mxfp4_mxfp8'):\n"
            + report.summary()
        )


def _e4m3_pow2_bytes(m: torch.Tensor) -> torch.Tensor:
    """E4M3 byte of ``2^m`` for int32 ``m`` in [-9, 8] (normal ``(m+7)<<3``; subnormal
    ``1 << (m+9)``)."""
    m = m.clamp(_M_MIN, _M_MAX)
    normal = (m + 7) << 3
    subnormal = torch.ones_like(m) << (m - _M_MIN).clamp(0, 2)
    return torch.where(m >= _M_MIN_NORMAL, normal, subnormal).to(torch.uint8)


def convert_mxfp4_leg_to_nvfp4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    leg: str,
    expert_offset: int = 0,
    chunk_experts: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[ExpertLegStats]]:
    """One GEMM's MXFP4 scales -> ``(E4M3 bytes (E, R, K/16), c (E,) int32, stats)``.

    Codes are not touched. Blocks that are not representable get byte 0 in the output
    and are counted in the stats; the caller raises (:func:`convert_mxfp4_to_nvfp4`).
    """
    c8 = _u8(codes)
    s8 = _u8(scales)
    experts, rows, half_k = c8.shape
    blocks = s8.shape[-1]
    if s8.shape != (experts, rows, blocks) or half_k != blocks * (MX_BLOCK // 2):
        raise ValueError(
            f"{leg}: codes {tuple(c8.shape)} and scales {tuple(s8.shape)} do not "
            "describe the same per-32 blocks."
        )
    out = torch.empty((experts, rows, blocks * 2), dtype=torch.uint8, device=c8.device)
    exps = torch.empty((experts,), dtype=torch.int32, device=c8.device)
    stats: list[ExpertLegStats] = []
    big = 1 << 20
    for e0 in range(0, experts, chunk_experts):
        e1 = min(e0 + chunk_experts, experts)
        n = e1 - e0
        live = ((c8[e0:e1].reshape(n, rows, blocks, MX_BLOCK // 2) & 0x77) != 0).any(
            dim=-1
        )
        b = s8[e0:e1].to(torch.int32)
        nan = b == 255
        k = b - 127
        ok = live & ~nan
        kmax = torch.where(ok, k, -big).amax(dim=(1, 2))
        kmin = torch.where(ok, k, big).amin(dim=(1, 2))
        has_live = ok.any(dim=(1, 2))
        c = torch.where(has_live, (kmax - _M_MAX).clamp_min(-126), 0).to(torch.int32)
        m = k - c.view(n, 1, 1)
        nonrep = (ok & (m < _M_MIN)) | nan
        subnormal = ok & (m < _M_MIN_NORMAL) & (m >= _M_MIN)
        enc = torch.where(ok & ~nonrep, _e4m3_pow2_bytes(m), 0).to(torch.uint8)
        out[e0:e1] = enc.repeat_interleave(2, dim=-1)
        exps[e0:e1] = c
        cols = torch.stack(
            (
                kmin,
                kmax,
                c,
                has_live.to(torch.int32),
                ok.sum(dim=(1, 2), dtype=torch.int64).to(torch.int32),
                (~live & ~nan).sum(dim=(1, 2), dtype=torch.int64).to(torch.int32),
                nan.sum(dim=(1, 2), dtype=torch.int64).to(torch.int32),
                subnormal.sum(dim=(1, 2), dtype=torch.int64).to(torch.int32),
                nonrep.sum(dim=(1, 2), dtype=torch.int64).to(torch.int32),
            ),
            dim=1,
        ).tolist()
        for i, (lo, hi, ce, any_live, nl, nz, nn, ns, nr) in enumerate(cols):
            stats.append(
                ExpertLegStats(
                    expert=expert_offset + e0 + i,
                    leg=leg,
                    kmin=lo if any_live else None,
                    kmax=hi if any_live else None,
                    c=ce,
                    live=nl,
                    zero=nz,
                    nan=nn,
                    subnormal=ns,
                    nonrep=nr,
                )
            )
    return out, exps, stats


class Nvfp4Conversion(NamedTuple):
    """Exact NVFP4 view of one layer's canonical MXFP4 weights (canonical row order)."""

    w13_codes: torch.Tensor  # (E, 2I, H/2) uint8, == the MXFP4 codes
    w13_scale16: torch.Tensor  # (E, 2I, H/16) E4M3 bytes
    c1: torch.Tensor  # (E,) int32 global exponent of fc1
    w2_codes: torch.Tensor  # (E, H, I/2) uint8
    w2_scale16: torch.Tensor  # (E, H, I/16) E4M3 bytes
    c2: torch.Tensor  # (E,) int32 global exponent of fc2
    report: ConversionReport


def scan_mxfp4_to_nvfp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    layer_id: int | None = None,
    expert_offset: int = 0,
) -> Nvfp4Conversion:
    """Convert without raising: the report says whether the result is exact."""
    check_canonical_mxfp4(
        w13, w13_scale, w2, w2_scale, hidden=hidden, intermediate=intermediate
    )
    s13, c1, st1 = convert_mxfp4_leg_to_nvfp4(
        w13, w13_scale, leg="fc1", expert_offset=expert_offset
    )
    s2, c2, st2 = convert_mxfp4_leg_to_nvfp4(
        w2, w2_scale, leg="fc2", expert_offset=expert_offset
    )
    report = ConversionReport(layer_id=layer_id, stats=tuple(st1 + st2))
    return Nvfp4Conversion(_u8(w13), s13, c1, _u8(w2), s2, c2, report)


def convert_mxfp4_to_nvfp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    layer_id: int | None = None,
    expert_offset: int = 0,
) -> Nvfp4Conversion:
    """Exact conversion; raises :class:`NonRepresentableError` listing every offender.
    """
    conv = scan_mxfp4_to_nvfp4(
        w13,
        w13_scale,
        w2,
        w2_scale,
        hidden=hidden,
        intermediate=intermediate,
        layer_id=layer_id,
        expert_offset=expert_offset,
    )
    if conv.report.offending:
        raise NonRepresentableError(conv.report)
    return conv


def nvfp4_alphas(
    c1: torch.Tensor,
    c2: torch.Tensor,
    *,
    input_norm_const: float,
    fc1_norm_const: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert ``(fc1_alpha, fc2_alpha, fc1_norm_const)`` fp32, exact or raise."""
    for name, value in (
        ("input_norm_const", input_norm_const),
        ("fc1_norm_const", fc1_norm_const),
    ):
        if not is_power_of_two(value):
            raise ValueError(
                f"{name} must be a positive power of two (exact alpha folding), got "
                f"{value!r}."
            )
    out = []
    for c, norm in ((c1, input_norm_const), (c2, fc1_norm_const)):
        exact = torch.ldexp(
            torch.ones_like(c, dtype=torch.float64), c.to(torch.int64)
        ) / float(norm)
        alpha = exact.to(torch.float32)
        if not torch.equal(alpha.to(torch.float64), exact):
            raise ValueError(
                "a per-expert NVFP4 alpha 2^c / norm_const is not exact in FP32 "
                f"(c range [{int(c.min())}, {int(c.max())}], norm {norm})."
            )
        out.append(alpha.contiguous())
    norm1 = torch.full_like(out[0], float(fc1_norm_const))
    return out[0], out[1], norm1


# --------------------------------------------------------------------------------------
# Kernel-layout weights
# --------------------------------------------------------------------------------------


class RubinMoeWeights(NamedTuple):
    """One layer's local-expert weights in the kernel layout (device tensors).

    ``fc1`` / ``fc2`` are ``(weight, scale)``: the weight is the logical ``(E, K/2, N)``
    view (``float4_e2m1fn_x2``) of K-major physical ``(E, N, K/2)`` storage, the scale
    the per-expert flattened 32x4x4-swizzled plane (``float8_e4m3fn`` for NVFP4,
    ``float8_e8m0fnu`` for MXFP4). The NVFP4 scalars are ``(E,)`` fp32, None for MX.
    """

    quant_kind: str
    fc1: tuple[torch.Tensor, torch.Tensor]
    fc2: tuple[torch.Tensor, torch.Tensor]
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None
    report: ConversionReport | None = None


def _aligned_k_major_view(data_u8: torch.Tensor) -> torch.Tensor:
    data = data_u8.contiguous()
    if data.data_ptr() % 16:
        data = data.clone()
    return data.view(fp4_storage_dtype()).permute(0, 2, 1)


def prepare_mxfp4_mxfp8_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    quant_kind: str = "mxfp4_mxfp8",
) -> RubinMoeWeights:
    """MXFP4 weights as they are (FlashInfer
    ``preprocess_prequantized_block_scaled_weights`` for ``quant_kind``
    ``"mxfp4_mxfp8"`` or ``"mxfp4"``, byte-identical)."""
    if quant_kind not in ("mxfp4_mxfp8", "mxfp4"):
        raise ValueError(f"not an MXFP4-weight quant kind: {quant_kind!r}.")
    check_canonical_mxfp4(
        w13, w13_scale, w2, w2_scale, hidden=hidden, intermediate=intermediate
    )
    if intermediate % (4 * MX_BLOCK):
        raise ValueError(f"{quant_kind} needs intermediate % 128 == 0.")
    d13 = interleave_gate_up_16(_u8(w13), intermediate_size=intermediate)
    s13 = interleave_gate_up_16(_u8(w13_scale), intermediate_size=intermediate)
    fc1 = (
        _aligned_k_major_view(d13),
        to_blocked_batched(s13.contiguous()).view(torch.float8_e8m0fnu),
    )
    fc2 = (
        _aligned_k_major_view(_u8(w2)),
        to_blocked_batched(_u8(w2_scale).contiguous()).view(torch.float8_e8m0fnu),
    )
    return RubinMoeWeights(quant_kind, fc1, fc2)


def deinterleave_gate_up_cols_16(y: torch.Tensor) -> torch.Tensor:
    """``[..., 2I]`` GEMM output of gate/up weight rows in the kernel's 16-row
    stripes (:func:`stripe_mxfp8_gate_up_`) -> ``[gate | up]`` (a new tensor)."""
    n = y.shape[-1]
    if n % (2 * GATE_UP_INTERLEAVE):
        raise ValueError(f"gate_up width {n} is not a multiple of 32.")
    return (
        y.unflatten(-1, (n // (2 * GATE_UP_INTERLEAVE), 2, GATE_UP_INTERLEAVE))
        .transpose(-3, -2)
        .reshape(y.shape)
    )


def stripe_mxfp8_gate_up_(
    gate_up: torch.Tensor, gate_up_scale: torch.Tensor, *, hidden: int, intermediate: int
) -> None:
    """In place: an MXFP8 E4M3 gate_up (row-major ``[2I, H]``, gate rows first, F8_128x4
    swizzled E8M0 block-32 scales: the stock linear's own storage) -> the same bytes
    in the GenPhase kernel's 16-row gate/up stripes (rows and their scales move
    together). A GEMM on the striped weight returns its ``[gate | up]`` output columns
    in stripes too (:func:`deinterleave_gate_up_cols_16`); the fused shared expert's
    FC1 then views the weight (``prepare_mxfp8_shared_weights(striped=True)``)
    instead of holding a striped copy of it."""
    w, s = _u8(gate_up), _u8(gate_up_scale)
    if (
        tuple(w.shape) != (2 * intermediate, hidden)
        or not w.is_contiguous()
        or hidden % 128
        or intermediate % 128
    ):
        raise ValueError(
            f"gate_up must be a row-major [{2 * intermediate}, {hidden}] FP8 weight "
            "(128-aligned)."
        )
    if s.numel() != swizzled_flat_sf_size(2 * intermediate, hidden // MX_BLOCK):
        raise ValueError(f"gate_up scale has {s.numel()} bytes, expected F8_128x4.")
    w.copy_(interleave_gate_up_16(w, intermediate_size=intermediate))
    plane = from_blocked(s.reshape(-1), 2 * intermediate, hidden // MX_BLOCK)
    plane = interleave_gate_up_16(plane, intermediate_size=intermediate)
    s.view(-1).copy_(to_blocked_batched(plane[None].contiguous()).reshape(-1))


def prepare_mxfp8_shared_weights(
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    striped: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """An MXFP8 E4M3 shared expert (the stock linears' storage: row-major FP8
    ``gate_up [2I, H]`` (gate rows first) / ``down [H, I]`` with F8_128x4 swizzled
    E8M0 block-32 scales) -> the GenPhase kernel's fused-shared (FC1, FC2) pairs
    (FlashInfer ``preprocess_prequantized_block_scaled_weights`` layout: 16-row
    gate/up stripes, K-major ``[1, K, N]`` views, re-swizzled scales). FC2 views the
    down weight in place. FC1 is a striped copy, or with ``striped`` (gate_up and its
    scales already striped in place by :func:`stripe_mxfp8_gate_up_`) a view of the
    weight itself. Same bytes, no requantization."""
    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    if (
        gate_up.dtype != fp8
        or down.dtype != fp8
        or tuple(gate_up.shape) != (2 * intermediate, hidden)
        or tuple(down.shape) != (hidden, intermediate)
        or not gate_up.is_contiguous()
        or not down.is_contiguous()
        or hidden % 128
        or intermediate % 128
    ):
        raise ValueError(
            f"shared expert must be row-major FP8 E4M3 gate_up {(2 * intermediate, hidden)}"
            f" and down {(hidden, intermediate)} (128-aligned)."
        )
    s2 = from_blocked(_u8(down_scale).reshape(-1), hidden, intermediate // MX_BLOCK)
    if striped:
        # The stock storage is already in stripes: FC1 is a view of it, and the
        # re-swizzled striped scale plane is the stock F8_128x4 bytes themselves.
        d13 = _u8(gate_up)[None]
        if d13.data_ptr() % 16:
            raise ValueError("shared gate_up weight must be 16-byte aligned.")
        sf13 = _u8(gate_up_scale).reshape(1, -1)
        if sf13.shape[1] != swizzled_flat_sf_size(2 * intermediate, hidden // MX_BLOCK):
            raise ValueError("shared gate_up scale is not an F8_128x4 plane.")
    else:
        s13 = from_blocked(
            _u8(gate_up_scale).reshape(-1), 2 * intermediate, hidden // MX_BLOCK
        )
        d13 = interleave_gate_up_16(_u8(gate_up)[None], intermediate_size=intermediate)
        s13 = interleave_gate_up_16(s13[None], intermediate_size=intermediate)
        d13 = d13.contiguous()
        sf13 = to_blocked_batched(s13.contiguous())
    d2 = _u8(down)[None]
    if d2.data_ptr() % 16:
        raise ValueError("shared down weight must be 16-byte aligned.")
    fc1 = (d13.view(fp8).permute(0, 2, 1), sf13.view(e8m0))
    fc2 = (d2.view(fp8).permute(0, 2, 1), to_blocked_batched(s2[None].contiguous()).view(e8m0))
    return fc1, fc2


def _nvfp4_kernel_weights(
    w13_codes: torch.Tensor,
    w13_scale16: torch.Tensor,
    w2_codes: torch.Tensor,
    w2_scale16: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    fc1_norm_const: torch.Tensor,
    *,
    intermediate: int,
    report: ConversionReport | None = None,
) -> RubinMoeWeights:
    """Canonical-order NVFP4 codes + E4M3 block-16 scale bytes -> kernel layout."""
    d13 = interleave_gate_up_16(_u8(w13_codes), intermediate_size=intermediate)
    s13 = interleave_gate_up_16(_u8(w13_scale16), intermediate_size=intermediate)
    fc1 = (
        _aligned_k_major_view(d13),
        to_blocked_batched(s13.contiguous()).view(torch.float8_e4m3fn),
    )
    fc2 = (
        _aligned_k_major_view(_u8(w2_codes)),
        to_blocked_batched(_u8(w2_scale16).contiguous()).view(torch.float8_e4m3fn),
    )
    return RubinMoeWeights(
        "nvfp4", fc1, fc2, fc1_alpha, fc2_alpha, fc1_norm_const, report=report
    )


def prepare_nvfp4_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    input_norm_const: float = 1.0,
    fc1_norm_const: float = 1.0,
    layer_id: int | None = None,
    expert_offset: int = 0,
) -> RubinMoeWeights:
    """Exact MXFP4 -> NVFP4 (raises :class:`NonRepresentableError`), kernel layout."""
    conv = convert_mxfp4_to_nvfp4(
        w13,
        w13_scale,
        w2,
        w2_scale,
        hidden=hidden,
        intermediate=intermediate,
        layer_id=layer_id,
        expert_offset=expert_offset,
    )
    fc1_alpha, fc2_alpha, norm1 = nvfp4_alphas(
        conv.c1,
        conv.c2,
        input_norm_const=input_norm_const,
        fc1_norm_const=fc1_norm_const,
    )
    return _nvfp4_kernel_weights(
        conv.w13_codes,
        conv.w13_scale16,
        conv.w2_codes,
        conv.w2_scale16,
        fc1_alpha,
        fc2_alpha,
        norm1,
        intermediate=intermediate,
        report=conv.report,
    )


# --------------------------------------------------------------------------------------
# Native NVFP4 checkpoints (ModelOpt export)
# --------------------------------------------------------------------------------------


def check_canonical_nvfp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_global: torch.Tensor,
    w2_global: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
) -> int:
    """Validate one layer's native NVFP4 experts; returns the local expert count.

    ``w13`` ``(E, 2I, H/2)`` / ``w2`` ``(E, H, I/2)`` packed E2M1; ``*_scale`` the
    E4M3 block-16 scales ``(E, 2I, H/16)`` / ``(E, H, I/16)``; ``*_global`` the
    per-expert FP32 ``weight_scale_2`` ``(E,)``, finite and positive.
    """
    if any(t.dim() != 3 for t in (w13, w13_scale, w2, w2_scale)):
        raise ValueError("NVFP4 w13 / w13_scale / w2 / w2_scale must all be 3D.")
    experts = w13.shape[0]
    expected = {
        "w13": (w13, (experts, 2 * intermediate, hidden // 2)),
        "w13_scale": (w13_scale, (experts, 2 * intermediate, hidden // NV_BLOCK)),
        "w2": (w2, (experts, hidden, intermediate // 2)),
        "w2_scale": (w2_scale, (experts, hidden, intermediate // NV_BLOCK)),
    }
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}.")
        if tensor.element_size() != 1:
            raise TypeError(f"{name} must be a 1-byte dtype, got {tensor.dtype}.")
    for name in ("w13_scale", "w2_scale"):
        if expected[name][0].dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise TypeError(f"{name} must be E4M3 (uint8 or float8_e4m3fn).")
    for name, g in (("w13 weight_scale_2", w13_global), ("w2 weight_scale_2", w2_global)):
        if g.shape != (experts,) or g.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32 ({experts},), got {g.dtype} {g.shape}.")
        if not bool(torch.isfinite(g).all()) or not bool((g > 0).all()):
            raise ValueError(f"{name} must be finite and positive.")
    if experts <= 0 or hidden % (4 * MX_BLOCK) or intermediate % (2 * MX_BLOCK):
        raise ValueError(
            "need experts > 0, hidden % 128 == 0 and intermediate % 64 == 0 "
            f"(got E={experts}, H={hidden}, I={intermediate})."
        )
    return experts


def prepare_nvfp4_checkpoint_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_global: torch.Tensor,
    w2_global: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
) -> RubinMoeWeights:
    """A native NVFP4 checkpoint's experts, as stored, in the kernel layout.

    Codes and E4M3 block scales are only re-laid out. The per-expert epilogue
    alphas are the checkpoint's ``weight_scale_2`` (activation global scale 1.0:
    ``input_norm_const = fc1_norm_const = 1``), the stock NVFP4 MoE alphas
    ``input_scale * weight_scale_2`` for the unit input scales the caller asserts.
    """
    check_canonical_nvfp4(
        w13,
        w13_scale,
        w2,
        w2_scale,
        w13_global,
        w2_global,
        hidden=hidden,
        intermediate=intermediate,
    )
    fc1_alpha = w13_global.to(torch.float32).contiguous()
    fc2_alpha = w2_global.to(torch.float32).contiguous()
    return _nvfp4_kernel_weights(
        w13,
        w13_scale,
        w2,
        w2_scale,
        fc1_alpha,
        fc2_alpha,
        torch.ones_like(fc1_alpha),
        intermediate=intermediate,
    )


def prepare_weights(
    quant_kind: str,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    input_norm_const: float = 1.0,
    fc1_norm_const: float = 1.0,
    layer_id: int | None = None,
    expert_offset: int = 0,
) -> RubinMoeWeights:
    if quant_kind in ("mxfp4_mxfp8", "mxfp4"):
        return prepare_mxfp4_mxfp8_weights(
            w13,
            w13_scale,
            w2,
            w2_scale,
            hidden=hidden,
            intermediate=intermediate,
            quant_kind=quant_kind,
        )
    if quant_kind == "nvfp4":
        return prepare_nvfp4_weights(
            w13,
            w13_scale,
            w2,
            w2_scale,
            hidden=hidden,
            intermediate=intermediate,
            input_norm_const=input_norm_const,
            fc1_norm_const=fc1_norm_const,
            layer_id=layer_id,
            expert_offset=expert_offset,
        )
    raise ValueError(f"unsupported quant kind {quant_kind!r} (expected {QUANT_KINDS}).")


def kernel_weight_shapes(
    quant_kind: str, experts: int, hidden: int, intermediate: int
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    """(fc1 logical shape, fc2 logical shape, fc1 SF numel/expert, fc2 SF numel/expert).
    """
    vec = NV_BLOCK if quant_kind == "nvfp4" else MX_BLOCK
    return (
        (experts, hidden // 2, 2 * intermediate),
        (experts, intermediate // 2, hidden),
        swizzled_flat_sf_size(2 * intermediate, hidden // vec),
        swizzled_flat_sf_size(hidden, intermediate // vec),
    )


def validate_kernel_weights(
    weights: RubinMoeWeights,
    *,
    quant_kind: str,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device | None = None,
) -> None:
    """Port of FlashInfer ``_validate_weight_leg`` (+ the NVFP4 scalar checks)."""
    if weights.quant_kind != quant_kind:
        raise ValueError(
            f"weights were prepared for {weights.quant_kind!r}, the backend runs "
            f"{quant_kind!r}."
        )
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = kernel_weight_shapes(
        quant_kind, experts, hidden, intermediate
    )
    sf_dtype = torch.float8_e4m3fn if quant_kind == "nvfp4" else torch.float8_e8m0fnu
    for name, (weight, scale), wshape, sf_numel in (
        ("fc1", weights.fc1, fc1_shape, fc1_sf),
        ("fc2", weights.fc2, fc2_shape, fc2_sf),
    ):
        if tuple(weight.shape) != wshape:
            raise ValueError(f"{name} weight shape {tuple(weight.shape)} != {wshape}.")
        if tuple(scale.shape) != (wshape[0], sf_numel):
            raise ValueError(
                f"{name} scale shape {tuple(scale.shape)} != {(wshape[0], sf_numel)}."
            )
        if weight.dtype != fp4_storage_dtype() or scale.dtype != sf_dtype:
            raise ValueError(
                f"{name} needs {fp4_storage_dtype()} weights and {sf_dtype} scales, "
                f"got {weight.dtype} / {scale.dtype}."
            )
        if not weight.permute(0, 2, 1).is_contiguous() or not scale.is_contiguous():
            raise ValueError(f"{name} needs K-major weights and contiguous scales.")
        for t in (weight, scale):
            if t.data_ptr() % 16 or (device is not None and t.device != device):
                raise ValueError(f"{name} tensors must be 16-byte aligned on {device}.")
    scalars = (weights.fc1_alpha, weights.fc2_alpha, weights.fc1_norm_const)
    if quant_kind == "nvfp4":
        for t in scalars:
            if (
                t is None
                or t.dtype != torch.float32
                or tuple(t.shape) != (experts,)
                or not t.is_contiguous()
                or (device is not None and t.device != device)
            ):
                raise ValueError(
                    "NVFP4 needs contiguous fp32 (E_local,) fc1_alpha / fc2_alpha / "
                    "fc1_norm_const on the backend device."
                )
    elif any(t is not None for t in scalars):
        raise ValueError(
            f"{quant_kind} takes no fc1_alpha / fc2_alpha / fc1_norm_const."
        )
