# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine API of the die-local GEMM for the model's call sites (``_die_local`` in
model.py, ``DecodeProjection``): ``serves(x, out)``, ``__call__(x, out)``, ``n``,
``out_dtype``. ``place_layer_on_dies`` places a decoder layer's weights at
construction; ``attach_*`` hang engines on placed linears. Compiled code is shared
by every layer of one shape."""

import torch

from ...common import dies
from .placement import relocate_parameter_
from .primitives import CTA_K, EPI_N, OUT_BF16, OUT_F32

MAX_ROWS = 256


def two_dies(device_index: int) -> bool:
    return dies.domain_count(device_index) == 2


def cta_ns_for(max_rows: int) -> list[int]:
    return list(range(EPI_N, -(-max_rows // EPI_N) * EPI_N + 1, EPI_N))


class DieLocalEngine:
    """``y = x @ W^T`` (or ``y += x @ W^T`` / FP32 ``y``, ``out_mode``) for 1 ..
    ``max_rows`` rows on a die-local GEMM over ``weight``, which must already be placed
    by ``placement.relocate_halves``."""

    def __init__(
        self,
        weight: torch.Tensor,
        max_rows: int,
        out_mode: int = OUT_BF16,
        m_sub: int = 1,
    ):
        from .cluster import DieLocalClusterLinear

        assert 0 < max_rows <= MAX_ROWS
        self.n, self.k = weight.shape
        self.out_mode, self.max_rows = out_mode, max_rows
        self.out_dtype = torch.float32 if out_mode == OUT_F32 else torch.bfloat16
        # 128-row tiles: 256-row tiles (m_sub 2) lose at every decode shape measured
        self.impl = DieLocalClusterLinear(
            weight, cta_ns_for(max_rows), out_mode, relocate=False, m_sub=m_sub
        )

    def serves(self, x: torch.Tensor, out: torch.Tensor | None = None) -> bool:
        if (
            x.dim() != 2
            or x.dtype != torch.bfloat16
            or not 0 < x.shape[0] <= self.max_rows
            or x.shape[1] != self.k
            or x.stride() != (self.k, 1)
            or x.data_ptr() % 16
        ):
            return False
        return out is None or (
            out.dtype == self.out_dtype
            and out.shape == (x.shape[0], self.n)
            and out.stride() == (self.n, 1)
        )

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.impl(x, out)

    run = __call__


def attach_die_local(linear, max_rows: int, out_mode: int = OUT_BF16) -> bool:
    """Place ``linear``'s BF16 weight on the dies and give it a ``die_local`` engine
    (two-die devices, packed BF16 [N, K] weights with K % 64 == 0); False otherwise."""
    weight = getattr(linear, "weight", None)
    if getattr(linear, "die_local", None) is not None:
        return True
    if (
        weight is None
        or not getattr(weight, "on_dies", False)
        or not _plain_bf16(linear)
    ):
        return False  # placed at construction (place_layer_on_dies) or not at all
    linear.die_local = DieLocalEngine(weight.data, max_rows, out_mode)
    return True


def die_local_problem(weight: torch.Tensor, rows: int) -> str | None:
    """Why the die-local GEMM cannot serve ``rows`` of ``weight`` (None: it can)."""
    if not weight.is_cuda or not two_dies(weight.device.index):
        return "die_local needs a device with two locality domains"
    n, k = weight.shape
    if weight.dtype != torch.bfloat16 or weight.stride() != (k, 1) or k % CTA_K:
        return f"die_local needs a packed BF16 [N, K] weight, K % {CTA_K} == 0"
    if weight.data_ptr() % (2 << 20):
        return "die_local needs a weight placed on the dies (relocate_halves)"
    if not 0 < rows <= MAX_ROWS:
        return f"die_local serves 1..{MAX_ROWS} rows, not {rows}"
    return None


def run_die_local(linear, x: torch.Tensor, out: torch.Tensor | None = None):
    """``linear``'s output from its die-local engine (``out``: accumulate into it,
    y += x W^T), or None where it has none or it does not serve ``x``."""
    engine = getattr(linear, "die_local", None)
    if engine is None or not engine.serves(x, out):
        return None
    if out is None:
        out = torch.empty(x.shape[0], engine.n, dtype=engine.out_dtype, device=x.device)
    return engine(x, out)


def attach_mlp_die_local(mlp, max_rows: int) -> int:
    """A stock K3.1 MegaMoE MLP on the SP shard: die-local engines for its
    replicated shared expert's gate_up / down (``_mlp_targets``). Returns the engines
    attached."""
    return sum(attach_die_local(lin, max_rows, mode) for lin, mode in _mlp_targets(mlp))


def _mlp_targets(mlp):
    """The BF16 linears of a K3.1 MegaMoE MLP on the SP shard that the die-local GEMM
    serves, with their output modes: the replicated shared expert only (the router
    gate and the latent down / up projections lose in the step: the gate's and the
    latent down's clusters do not co-schedule side by side, and the latent up has no
    early-triggering predecessor to hide the cluster's start)."""
    if getattr(mlp, "experts", None) is None or not getattr(mlp, "use_mega_moe", False):
        return []
    shared = getattr(mlp, "shared_experts", None)
    if shared is None or getattr(shared, "shard_sequence_parallel", False):
        return []
    targets = [(shared.gate_up_proj, OUT_BF16), (shared.down_proj, OUT_BF16)]
    return [(lin, mode) for lin, mode in targets if _plain_bf16(lin)]


def _plain_bf16(linear) -> bool:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    method = getattr(linear, "quant_method", None)
    weight = getattr(linear, "weight", None)
    return (
        (method is None or isinstance(method, UnquantizedLinearMethod))
        and weight is not None
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.shape[1] % CTA_K == 0
    )


def place_layer_on_dies(decoder) -> int:
    """At construction, right after ``decoder``'s modules are built (before the weights
    load): move the BF16 weights the SP decode's die-local GEMMs read (the KDA in-
    projection, the MegaMoE MLP linears and, under DSA-DP, the all-head DSA o_proj /
    g_proj) to die-halves memory in place. The
    just-built parameters' memory goes back to the allocator's pool and the next
    layers' parameters reuse it, so the placement adds no footprint (a copy after
    loading would: the cumem pool keeps freed blocks). Bytes moved."""
    attention = getattr(decoder, "self_attn", None)
    linears = []
    if attention is not None:
        names = ["in_proj_qkvgfab"]  # the KDA in-projection
        if getattr(attention, "local_full_heads", False):  # DSA-DP: all-head DSA
            names += ["o_proj", "g_proj"]
        for name in names:
            lin = getattr(attention, name, None)
            if lin is not None and _plain_bf16(lin):
                linears.append(lin)
    linears += [lin for lin, _ in _mlp_targets(getattr(decoder, "mlp", None))]
    moved = 0
    with torch.no_grad():
        for lin in linears:
            if not getattr(lin.weight, "on_dies", False):
                moved += lin.weight.numel() * 2
                relocate_parameter_(lin.weight)
    return moved
