# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill shared expert: gate_up GEMM with the SiTU + MXFP8 epilogue.

On a prefill worker (``specialized_prefill.moe`` policy active) with an SM107
GPU, an online-MXFP8 shared gate_up and down_proj and a SiTU activation, prefill
batches of at least ``mxfp8_act.SITU_FAST_MIN_ROWS`` rows run one sm_107 GEMM
(``sm107_gemm.shared_situ_gemm``) that writes the down_proj's MXFP8 input
directly: the bytes of ``situ_and_mul_quantize_auto`` (mode 4 at these row
counts) of the BF16 gate_up output, without the BF16 gate_up tensor and the
separate activation pass. Smaller batches keep the cuBLASLt GEMM + activation
kernel; the gate/up weight rows are interleaved in 128-row chunks for the fused
GEMM (at load, ``prepare``), so their GEMM output is de-interleaved first.
"""

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

CHUNK = 128


def supported(mlp) -> bool:
    """SM107 + SiTU + online-MXFP8 gate_up / down_proj with the MXFP8 act."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability((10, 7)):
        return False
    act = mlp.act_fn
    if not hasattr(act, "beta") or not hasattr(act, "linear_beta"):
        return False
    if not mlp._act_mxfp8_ok():
        return False
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout

    if _kernel_layout(mlp.gate_up_proj) is None:
        return False
    k = mlp.gate_up_proj.input_size_per_partition
    n = mlp.gate_up_proj.output_size_per_partition
    return n % (2 * CHUNK) == 0 and k % 128 == 0


class PrefillSharedSitu:
    """Owns the in-place gate/up row interleave of ``mlp.gate_up_proj`` and the
    fused GEMM. ``mlp`` must satisfy :func:`supported`."""

    def __init__(self, mlp) -> None:
        self.mlp = mlp
        self.beta = mlp.act_fn.beta
        self.linear_beta = mlp.act_fn.linear_beta
        self.inter = mlp.gate_up_proj.output_size_per_partition // 2
        self._gemm = None

    def _weight_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout

        lin = self.mlp.gate_up_proj
        w = lin.weight.data
        if _kernel_layout(lin) == "kn":
            w = w.t()
        if not w.is_contiguous():
            raise RuntimeError("shared SiTU GEMM needs the [N, K] K-major weight")
        return w, lin.weight_scale.data

    def prepare(self) -> None:
        """After weight loading (``process_weights_after_loading``, before any
        forward or CUDA graph capture): interleave the gate/up weight rows in
        place and compile + run the fused GEMM once."""
        if self._gemm is not None:
            return
        from .sm107_gemm import shared_situ_gemm as ssg

        w, s = self._weight_rows()
        ssg.interleave_gate_up_rows(w, s)
        self._gemm = ssg.SharedSituGemm(w, s, self.beta, self.linear_beta)
        rows = mxfp8_act.SITU_FAST_MIN_ROWS
        x = torch.zeros((rows, w.shape[1]), dtype=torch.float8_e4m3fn, device=w.device)
        sx = torch.full(
            (rows * w.shape[1] // 32,), 127, dtype=torch.uint8, device=w.device
        )
        self._gemm(x, sx, mxfp8_act.Mxfp8Buffer(rows, self.inter, w.device))
        # DIAGNOSTIC (measurement branch only, never landed): the step without
        # the shared gate_up + SiTU family. The shared down_proj reads a fixed
        # realistic MXFP8 activation (randn-scaled, no NaN codes) so its own
        # energy and time stay as in the real step.
        g = torch.Generator(device=w.device).manual_seed(0)
        y = torch.randn((16384, self.inter), generator=g, device=w.device,
                        dtype=torch.bfloat16) * 0.5  # fmt: skip
        self._fake = mxfp8_e4m3_quantize(y, is_sf_swizzled_layout=True)
        del y

    def fused(self, x):
        """down_proj's MXFP8 input for a prefill-sized batch, else None."""
        gemm = self._gemm
        assert gemm is not None, "PrefillSharedSitu.prepare() not run"
        key = self.mlp.gate_up_proj.quant_method.kernel.input_quant_key()
        qa = as_quantized_activation(x, key)
        if qa is not None:
            x_fp8, x_scale = qa.data, qa.scale
        else:
            x_fp8, x_scale = mxfp8_e4m3_quantize(
                x.reshape(-1, x.shape[-1]), is_sf_swizzled_layout=True
            )
        rows = x_fp8.shape[0]
        if rows < mxfp8_act.SITU_FAST_MIN_ROWS:
            return None
        # DIAGNOSTIC: no gate_up GEMM / SiTU; the first `rows` rows of the
        # fixed activation (128x4-swizzled scales: row blocks are contiguous).
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )

        fq, fs = self._fake
        assert rows <= fq.shape[0], rows
        nsc = (rows + 127) // 128 * 128 * (self.inter // 32)
        return QuantizedActivation(
            data=fq[:rows],
            scale=fs.view(torch.uint8).reshape(-1)[:nsc],
            orig_dtype=torch.bfloat16,
            orig_shape=torch.Size((rows, self.inter)),
            quant_key=kMxfp8Dynamic,
        )

    def deinterleave(self, gate_up: torch.Tensor) -> torch.Tensor:
        """``[gate | up]`` of the cuBLASLt GEMM output on the interleaved rows."""
        if self._gemm is None:
            return gate_up
        m, n = gate_up.shape
        return (
            gate_up.unflatten(1, (n // (2 * CHUNK), 2, CHUNK))
            .transpose(1, 2)
            .reshape(m, n)
        )
