# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill shared-expert gate_up GEMM with the SiTU + MXFP8 epilogue (sm_107).

One persistent block-scaled MXFP8 GEMM computes ``x @ W_gate_up.T`` and writes
``situ_and_mul_quantize_fast`` of its BF16-rounded output straight into the
``Mxfp8Buffer`` that the shared down_proj reads, which removes the separate
activation pass over the M x 2I BF16 gate_up tensor and its HBM round trip.

The gate/up weight rows are interleaved in 128-row chunks
(``packed_situ_gemm.interleave_shared_rows`` with ``n_lat = 0``) so a 256-wide N
tile holds the gate and up columns of the same 128 outputs. The 512x256
B-reuse tile fills TMEM (one accumulator stage), so the epilogue drains the
tile into registers as BF16 and releases TMEM before the math
(``situ_epilogue.epilogue_situ_mxfp8_drain``, 8 epilogue warps).
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_ptr

from .kda_conv_gemm import _Kernel as _BaseKernel
from .packed_situ_gemm import interleave_shared_rows, situ_constants

# (tile, cluster, inst_m, swizzle, raster, split producer, swap, flags)
DEFAULT_TACTIC = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "drain")
_COMPILED: dict = {}


class _Kernel(_BaseKernel):
    def __init__(self, tactic, a_pow: int, clamp_up: bool):
        super().__init__(tactic)
        self.situ_n_lat_tiles = 0
        self.situ_a_pow = a_pow
        self.situ_clamp_up = clamp_up


def _ptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def interleave_gate_up_rows(weight_nk: torch.Tensor, scale: torch.Tensor) -> None:
    """In place: ``[gate | up]`` rows -> 128-row ``[gate t | up t]`` pairs (FP8
    weight ``[2I, K]`` and its 128x4-swizzled UE8M0 scales)."""
    interleave_shared_rows(weight_nk, scale, 0, weight_nk.shape[1])


class SharedSituGemm:
    """``(x, x_scale, act)``: SiTU(x @ W.T) as MXFP8 into ``act`` (an
    ``Mxfp8Buffer`` with ``I`` columns). ``weight_nk`` holds interleaved rows."""

    def __init__(
        self,
        weight_nk: torch.Tensor,
        scale: torch.Tensor,
        beta: float,
        linear_beta: float | None,
        tactic=DEFAULT_TACTIC,
    ) -> None:
        n, k = weight_nk.shape
        if n % 256 or k % 128:
            raise ValueError(f"shared SiTU GEMM: unsupported N/K ({n}, {k})")
        self.weight, self.scale, self.tactic = weight_nk, scale, tactic
        self.n, self.k = n, k
        k_host, self.a_pow, self.clamp_up = situ_constants(beta, linear_beta)
        self.situ_k = k_host.to(weight_nk.device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = None

    def __call__(self, x: torch.Tensor, x_scale: torch.Tensor, act) -> None:
        m = x.shape[0]
        assert x.shape[1] == self.k and x.stride(1) == 1
        assert act.data.shape[0] == m and act.data.is_contiguous()
        assert act.data.shape[1] == self.n // 2
        f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            _ptr(f8, x),
            _ptr(f8, self.weight),
            _ptr(e8, x_scale),
            _ptr(e8, self.scale),
            _ptr(cutlass.BFloat16, act.data),  # C: never stored
            from_dlpack(self.alpha),
        )
        problem = (cutlass.Int32(m), cutlass.Int32(self.n), cutlass.Int32(self.k), 1)
        tail = (
            cutlass.Int64(self.n),  # c_ld
            cutlass.Int64(0),  # sfc_row_pitch
            cutlass.Int64(0),  # sfc_batch_offset
            cutlass.Int64(x.stride(0)),  # a_ld
            cutlass.Int64(0),  # a_batch_stride
            cutlass.Int64(0),  # c_batch_stride
        )
        situ = (
            _ptr(cutlass.Float8E4M3FN, act.data),
            make_ptr(
                cutlass.Uint8, act.scale.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            cutlass.Int64(act.data.stride(0)),
            cutlass.Int64(act.k),
            from_dlpack(self.situ_k),
        )  # fmt: skip
        if self._compiled is None:
            key = (self.tactic, self.a_pow, self.clamp_up, x.device.index, self.n, self.k)
            self._compiled = _COMPILED.get(key)
            if self._compiled is None:
                gemm = _Kernel(self.tactic, self.a_pow, self.clamp_up)
                cl = self.tactic[1]
                mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
                layouts = (
                    OperandMajorMode.K,
                    OperandMajorMode.K,
                    cutlass_utils.LayoutEnum.ROW_MAJOR,
                )
                self._compiled = cute.compile(
                    gemm, *args, layouts, problem, mac, stream, lambda v: v,
                    tail[0], None, *tail[1:], True, *situ,
                )  # fmt: skip
                _COMPILED[key] = self._compiled
        self._compiled(*args, problem, stream, *tail, *situ)
