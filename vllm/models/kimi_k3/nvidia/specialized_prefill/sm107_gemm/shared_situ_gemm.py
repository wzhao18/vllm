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
# (2, 4) clusters multicast each A tile to the 4 N tiles of the cluster: a
# quarter of the L2 -> SM operand bytes for A, which is what the kernel's time
# follows at the prefill power cap (sustained randn: 0.91-0.93x the (2, 1)
# tactic). The m raster with swizzle 4 beat swizzle 1/2/8 and the n raster.
DEFAULT_TACTIC = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "drain+roll+st8")
# Die-split form (die_sched): each die's clusters walk 3 of its N cluster
# tiles per M sweep with its weight half homed on it; MXFP8 stores evict-first
# so they do not push the die's weight half out of its L2.
DIE_TACTIC = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "drain+roll+st8+die3+sef")
_COMPILED: dict = {}


def die_tactic(weight_nk: torch.Tensor):
    """``DIE_TACTIC`` when ``weight_nk`` (the contiguous [N, K] gate/up rows)
    can be split by die: a two-die device, an even count of N cluster tiles
    that the group divides, and 2 MiB-aligned row halves; else None."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    n, k = weight_nk.shape
    tile_n, cluster_n = DIE_TACTIC[0][1], DIE_TACTIC[1][1]
    n_ct = n // (tile_n * cluster_n)
    if (
        not weight_nk.is_contiguous()
        or n % (tile_n * cluster_n)
        or n_ct % 2
        or (n_ct // 2) % 3
        or (n // 2) * k * weight_nk.element_size() % (2 << 20)
        or dies.domain_count(weight_nk.device.index or 0) != 2
    ):
        return None
    return DIE_TACTIC


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
        if tactic[0][1] != 256:
            # The SiTU epilogue pairs the 128 gate and 128 up columns of one
            # 256-wide N tile (interleave_gate_up_rows); N tiles past column N
            # (padded clusters of the swizzled "m" raster, or a cluster N that
            # does not divide n / 256) are dropped in the kernel, so any n that
            # 256 divides is safe with any cluster / swizzle.
            raise ValueError(f"shared SiTU GEMM: tile N {tactic[0][1]} (256 only)")
        self.weight, self.scale, self.tactic = weight_nk, scale, tactic
        self.n, self.k = n, k
        k_host, self.a_pow, self.clamp_up = situ_constants(beta, linear_beta)
        self.situ_k = k_host.to(weight_nk.device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = None
        self.die = any(f.startswith("die") for f in str(tactic[7]).split("+"))
        self._sched = ()
        if self.die:
            # die_sched: weight rows [0, n/2) homed on die 0, [n/2, n) on die 1
            # (die_sched.localize_rows) and an even count of N cluster tiles.
            cols = tactic[0][1] * tactic[1][1]
            group = next(
                (int(f[3:]) for f in str(tactic[7]).split("+")
                 if f.startswith("die") and f[3:].isdigit()), 0,
            )  # fmt: skip
            if n % (2 * cols) or (group and (n // (2 * cols)) % group):
                raise ValueError(
                    f"die-split SiTU GEMM: N {n} is not 2 x {group or 1} x k x {cols}"
                )
            from . import die_sched

            self._ctr = torch.zeros(4, dtype=torch.int32, device=weight_nk.device)
            self._die_map = die_sched.die_map(weight_nk.device)
            conv = (cutlass.Int64(0),) * 5 + (cutlass.Int32(0),) * 3
            self._sched = conv + (
                cutlass.Int64(self._ctr.data_ptr()),
                cutlass.Int64(self._die_map.data_ptr()),
            )

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
                    tail[0], None, *tail[1:], True, *situ, *self._sched,
                )  # fmt: skip
                _COMPILED[key] = self._compiled
        self._compiled(*args, problem, stream, *tail, *situ, *self._sched)
