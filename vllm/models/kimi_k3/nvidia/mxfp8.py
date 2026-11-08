# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-MXFP8 projections of Kimi K3 / K3.1 on SM10x.

Kernel choice on SM107 (Rubin): FlashInfer's CuTe DSL MXFP8 GEMM (vLLM's first
choice on SM10x) is the fastest at <= 16 rows but loses to cuBLASLt's MXFP8
``_scaled_mm`` above that for most K3.1 projection shapes (measured over
M = 1..16384 at TP1 / TP8 / TP16, k31vr W4); both read the same weight
layout ([N, K] FP8 rows, F8_128x4-swizzled E8M0 scales), so
:class:`KimiK3Mxfp8Kernel` dispatches per call without a second copy.

Row packing of online-MXFP8 linears that read the same input:

MXFP8 weights are quantized per row in blocks of 32 along K, so stacking the
rows of several MXFP8 linears (and their E8M0 scale rows) gives exactly the
MXFP8 weight of the concatenated linear: one GEMM of the packed weight equals
the per-linear GEMMs bit for bit (same activation quantization, same rows).
Kimi K3.1's fused DSA forward packs ``[q_a | kv_a | wk | weights_proj]`` and
``[q_b | wq_index]`` this way when they are online MXFP8.
"""


import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

# Above this many rows the cuBLASLt MXFP8 GEMM beats the CuTe DSL one on SM107
# for most shapes (the default for shapes not in the table below).
CUTE_DSL_MAX_ROWS = 16

# Measured winner (C = CuTe DSL, L = cuBLASLt) per K3.1 TP1 (wide-EP DEP)
# projection [N, K] at rows <= 16 / 32 / 48 / 64 / 96 / 128 / 160 / 224 / 256
# (VR SM107, pre-quantized input, weight streamed from HBM; wep-proj-gemm-p2
# bench_proj_gemm.py). A call takes the winner of the first grid point >= its
# rows; above 256 rows, or for shapes not listed, CUTE_DSL_MAX_ROWS applies.
_ROW_GRID = (16, 32, 48, 64, 96, 128, 160, 224, 256)
_TP1_WINNER = {
    (12288, 7168): "CCCCCCCCC",  # DSA g_proj, shared-expert gate_up
    (7168, 6144): "CCLCLLCCC",  # shared-expert down
    (3584, 7168): "CCCCLLLLL",  # latent-MoE down
    (7168, 3584): "CLLLLLCCC",  # latent-MoE up
    (49376, 7168): "CLLLCLLLL",  # KDA in_proj
    (7168, 12288): "CCCCCCLLL",  # KDA / DSA o_proj
    (2272, 7168): "CCCCLLLLL",  # DSA packed [q_a | kv_a | wk | weights_proj]
    (22528, 1536): "CLLCLLLLL",  # DSA packed [q_b | wq_index]
    (67584, 7168): "CCCCCCCLL",  # dense MLP gate_up
    (7168, 33792): "CCCCCCLLL",  # dense MLP down
}


def use_cute_dsl(n: int, k: int, rows: int) -> bool:
    """CuTe DSL (True) or cuBLASLt (False) for an [N, K] MXFP8 GEMM."""
    winners = _TP1_WINNER.get((n, k))
    if winners is None or rows > _ROW_GRID[-1]:
        return rows <= CUTE_DSL_MAX_ROWS
    for grid_rows, winner in zip(_ROW_GRID, winners):
        if rows <= grid_rows:
            return winner == "C"
    return False


class KimiK3Mxfp8Kernel(FlashInferCutedslMxfp8LinearKernel):
    """CuTe DSL MXFP8 GEMM at <= ``CUTE_DSL_MAX_ROWS`` rows, cuBLASLt above;
    the weight stays in the CuTe DSL layout ([K, N] column-major view)."""

    def apply_weights(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qa = as_quantized_activation(x, self.input_quant_key())
        rows = qa.data.shape[0] if qa is not None else x.numel() // x.shape[-1]
        weight = layer.weight  # [K, N], column-major
        k, n = weight.shape
        if rows == 0:
            # No rows (a ShardQ indexer step that scores none on this rank):
            # FlashInfer's MXFP8 quantize cannot launch an empty grid.
            shape = qa.orig_shape if qa is not None else x.shape
            dtype = qa.orig_dtype if qa is not None else layer.params_dtype
            return weight.new_empty((*shape[:-1], n), dtype=dtype)
        if use_cute_dsl(n, k, rows):
            return super().apply_weights(layer, x, bias)
        if qa is not None:
            x_fp8, x_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            out_dtype, input_shape = layer.params_dtype, x.shape
            x_fp8, x_scale = mxfp8_e4m3_quantize(
                x.reshape(-1, k), is_sf_swizzled_layout=True
            )
        out = torch._scaled_mm(
            x_fp8,
            weight,
            scale_a=x_scale.view(torch.float8_e8m0fnu),
            scale_b=layer.weight_scale.view(torch.float8_e8m0fnu),
            bias=bias,
            out_dtype=out_dtype,
        )
        return out.view(*input_shape[:-1], n)


def _use_dispatch_kernel() -> bool:
    return current_platform.is_device_capability((10, 7))


def enable_kimi_k3_mxfp8_dispatch(model: nn.Module) -> int:
    """On SM107, switch every online-MXFP8 linear on the CuTe DSL kernel to
    :class:`KimiK3Mxfp8Kernel` (same weights). Returns the number switched.

    Called from the post-load hook of both the K3 target and the K3 DSpark
    drafter (an online-MXFP8 drafter has its own linears)."""
    if not _use_dispatch_kernel():
        return 0
    switched = 0
    for module in model.modules():
        if not is_online_mxfp8(module):
            continue
        method = module.quant_method
        if type(method.kernel) is FlashInferCutedslMxfp8LinearKernel:
            dispatch = object.__new__(KimiK3Mxfp8Kernel)
            dispatch.__dict__.update(method.kernel.__dict__)
            method.kernel = dispatch
            switched += 1
    return switched


def unswizzle_mxfp8_scale(sf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of ``swizzle_mxfp8_scale``: F8_128x4 layout -> row-major
    ``[n, k // 32]`` uint8."""
    m_tiles, k_tiles = -(-n // 128), -(-k // (MXFP8_BLOCK_SIZE * 4))
    tiles = sf.view(torch.uint8).reshape(m_tiles, k_tiles, 32, 4, 4)
    rows = tiles.transpose(1, 3).reshape(m_tiles * 128, k_tiles * 4)
    return rows[:n, : k // MXFP8_BLOCK_SIZE].contiguous()


def _kernel_layout(linear: nn.Module) -> str | None:
    """How the linear's MXFP8 kernel stores its weight: "nk" ([N, K] row-major),
    "kn" (its transpose view), or None (a repacked layout this module can't
    unpack)."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.pytorch import (
        TorchMxFp8LinearKernel,
    )

    kernel = getattr(linear.quant_method, "kernel", None)
    if type(kernel) in (FlashInferCutedslMxfp8LinearKernel, KimiK3Mxfp8Kernel):
        return "kn"
    if type(kernel) is FlashInferCutlassMxfp8LinearKernel or (
        type(kernel) is TorchMxFp8LinearKernel and current_platform.is_cuda()
    ):
        return "nk"
    return None


def is_online_mxfp8(linear: nn.Module) -> bool:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    return isinstance(getattr(linear, "quant_method", None), Mxfp8OnlineLinearMethod)


def can_pack_mxfp8(linears, along_k: bool = False) -> bool:
    """The linears can share one MXFP8 GEMM: same K (stacked along N), or with
    ``along_k`` the same N and every K a multiple of 128 (stacked along K)."""
    if not all(is_online_mxfp8(linear) for linear in linears):
        return False
    if any(_kernel_layout(linear) is None for linear in linears):
        return False

    def nk(linear):
        w = linear.weight
        return (w.shape[1], w.shape[0]) if _kernel_layout(linear) == "kn" else w.shape

    if along_k:
        return len({nk(lin)[0] for lin in linears}) == 1 and all(
            nk(lin)[1] % (4 * MXFP8_BLOCK_SIZE) == 0 for lin in linears
        )
    return len({nk(lin)[1] for lin in linears}) == 1


def _rows(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The linear's MXFP8 weight as row-major ``[N, K]`` FP8 plus ``[N, K/32]``
    uint8 scales."""
    weight = linear.weight.data
    if _kernel_layout(linear) == "kn":
        weight = weight.t()
    n, k = weight.shape
    return weight, unswizzle_mxfp8_scale(linear.weight_scale.data, n, k)


class _PackedHolder(nn.Module):
    pass


class PackedMxfp8Linear:
    """One MXFP8 GEMM over the stacked rows of ``linears`` (all online MXFP8 on
    the same input). The linears' weights become views of the packed weight;
    their own (small) scale tensors stay so they keep working on their own.

    ``along_k=True`` stacks the linears along K instead (all with the same N):
    the GEMM of the K-concatenated input ``[x_0 | x_1 | ...]`` is
    ``sum_i linear_i(x_i)`` (FP32 accumulation, one BF16 rounding). Each
    ``x_i`` starts at a 128-aligned column, so its 32-wide MXFP8 blocks and
    F8_128x4 scale groups are exactly those of its own quantization (see
    ``ops/mxfp8_act.Mxfp8Buffer``). The linears' weights become strided views
    of the packed one: they are only used through the packed GEMM."""

    def __init__(self, linears, along_k: bool = False) -> None:
        from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

        assert can_pack_mxfp8(linears, along_k)
        parts = [_rows(linear) for linear in linears]
        dim = 1 if along_k else 0
        weight = torch.cat([w for w, _ in parts], dim=dim)
        scale = torch.cat([s for _, s in parts], dim=dim)
        n, k = weight.shape
        self.kernel = init_mxfp8_linear_kernel(weight_shape=(n, k))
        if _use_dispatch_kernel() and type(self.kernel) is (
            FlashInferCutedslMxfp8LinearKernel
        ):
            self.kernel = KimiK3Mxfp8Kernel(self.kernel.config)
        holder = _PackedHolder()
        holder.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        holder.register_parameter(
            "weight_scale", nn.Parameter(scale, requires_grad=False)
        )
        holder.input_scale = None
        holder.params_dtype = linears[0].params_dtype
        self.kernel.process_weights_after_loading(holder)
        self.holder = holder
        self.rows = [w.shape[dim] for w, _ in parts]
        # Share the weight storage when the packed and the per-linear layouts
        # agree (same kernel class, row-major or transposed rows).
        packed_rows = holder.weight.data
        layout = (
            "kn"
            if packed_rows.shape == (k, n) and packed_rows.stride() == (1, k)
            else "nk"
            if packed_rows.shape == (n, k) and packed_rows.is_contiguous()
            else None
        )
        if layout == "kn":
            packed_rows = packed_rows.t()
        if layout is not None:
            row = 0
            for linear, rows in zip(linears, self.rows):
                if _kernel_layout(linear) == layout:
                    view = (
                        packed_rows[:, row : row + rows]
                        if along_k
                        else packed_rows[row : row + rows]
                    )
                    linear.weight.data = view.t() if layout == "kn" else view
                row += rows

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self.holder, x)




def takes_mxfp8_activation(*linears: nn.Module | None) -> bool:
    """True when every linear is online MXFP8 on a kernel that consumes a
    pre-quantized MXFP8 activation (FlashInfer F8_128x4 scales), so the
    producer of their shared input may hand them a QuantizedActivation from
    ``ops/mxfp8_act.py`` instead of BF16."""
    if not current_platform.is_device_capability_family(100):
        return False
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    for linear in linears:
        if linear is None or not is_online_mxfp8(linear):
            return False
        kernel = getattr(linear.quant_method, "kernel", None)
        if kernel is None or kernel.input_quant_key() != kMxfp8Dynamic:
            return False
        if getattr(linear, "requires_unquantized_input", False):
            return False
        # The producers write whole 128-column groups of scales (every K3.1
        # projection input: 768 / 1536 / 3072 / 3584 / 6144 / 7168 / 12288).
        k = getattr(linear, "input_size_per_partition", None)
        if k is None or k % (4 * MXFP8_BLOCK_SIZE) != 0:
            return False
    return True
