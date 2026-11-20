# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed prefill MoE front GEMM with the shared-expert SiTU + MXFP8 epilogue.

One sm_107 persistent block-scaled MXFP8 GEMM computes the packed
``[latent_down | shared gate_up]`` projection of the layer input. Latent tiles
store BF16 (the routed experts' input); shared tiles store
``situ_and_mul_quantize_fast`` of their BF16-rounded accumulators straight into
the ``Mxfp8Buffer`` of the packed down/up GEMM (situ_epilogue.py), which removes
the separate activation pass over the 16384 x 12288 BF16 gate_up output.

The shared gate_up rows of the packed weight are interleaved in 128-row chunks
(``[gate c..c+127 | up c..c+127]``, ``interleave_shared_rows``) so a 256-wide N
tile holds the gate and up columns of the same 128 outputs.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_ptr

from .dense_blockscaled_gemm_sm107 import Sm107BlockScaledPersistentDenseGemmKernel

#: Output columns per gate/up chunk of the interleaved shared rows (= half the
#: 256-wide MMA N tile).
CHUNK = 128
#: (mma tile M, mma tile N), cluster, MMA instruction M: 2-CTA B-reuse tile.
TILE = (512, 256)
CLUSTER = (2, 1)
INST_M = 256
SWIZZLE = 8
RASTER = "n"
SF_VEC = 32
#: One compile per (latent tiles, SiTU form, device), shared by all layers.
_COMPILED: dict = {}


def situ_constants(beta: float, linear_beta: float | None):
    """``Situ4Consts`` of mxfp8_act.py (FP32 host math as the CUDA host code):
    (k_b, k_a, k_c, bb) as an FP32 tensor, a_pow, clamp_up."""
    f = torch.float32
    beta_t = torch.tensor(beta, dtype=f)
    log2e = torch.tensor(1.4426950408889634, dtype=f)
    clamp_up = linear_beta is not None and linear_beta > 0.0
    lb = torch.tensor(linear_beta if clamp_up else 0.0, dtype=f)
    kb = (torch.tensor(-2.0, dtype=f) * log2e) / beta_t
    kc = (torch.tensor(-2.0, dtype=f) * log2e) / lb if clamp_up else torch.zeros((), dtype=f)
    bb = beta_t * lb if clamp_up else beta_t
    half = float(torch.tensor(0.5, dtype=f) * beta_t)
    a_pow = int(half) if half == int(half) and 1.0 <= half <= 4.0 else 0
    k = torch.stack([kb, -log2e, kc, bb]).to(f)
    return k, a_pow, clamp_up


def interleave_shared_rows(
    weight_nk: torch.Tensor, scale: torch.Tensor, n_lat: int, k: int
) -> None:
    """In place: rows ``[n_lat, N)`` = ``[gate | up]`` become 128-row
    ``[gate chunk t | up chunk t]`` pairs, in the FP8 weight ``[N, K]`` and
    its F8_128x4-swizzled UE8M0 scales (128-row groups are contiguous)."""
    n = weight_nk.shape[0]
    n_sh = n - n_lat
    t = n_sh // (2 * CHUNK)
    assert n_lat % CHUNK == 0 and n_sh % (2 * CHUNK) == 0, (n_lat, n_sh)
    w = weight_nk[n_lat:]
    w.copy_(w.view(2, t, CHUNK, k).transpose(0, 1).reshape(n_sh, k))
    group = CHUNK * (-(-(k // SF_VEC) // 4) * 4)
    s = scale.view(-1)[(n_lat // CHUNK) * group : (n // CHUNK) * group]
    s.copy_(s.view(2, t, group).transpose(0, 1).reshape(-1))


def deinterleave_shared_cols(y_sh: torch.Tensor) -> torch.Tensor:
    """``[gate | up]`` of a GEMM output whose shared columns come from the
    interleaved rows (the unfused small-M path)."""
    m, n_sh = y_sh.shape
    return (
        y_sh.unflatten(1, (n_sh // (2 * CHUNK), 2, CHUNK))
        .transpose(1, 2)
        .reshape(m, n_sh)
    )


DEFAULT_TACTIC = (TILE, CLUSTER, False)


class _Kernel(Sm107BlockScaledPersistentDenseGemmKernel):
    def __init__(
        self, n_lat_tiles: int, a_pow: int, clamp_up: bool, tactic=DEFAULT_TACTIC
    ):
        tile, cluster, split_producer = tactic
        super().__init__(
            SF_VEC,
            (INST_M, tile[1], 64),
            (tile[0], tile[1], 128),
            cluster,
            prefetch_dist=0,
            swizzle_size=SWIZZLE,
            raster_order=RASTER,
        )
        if split_producer:
            self.tma_b_warp_id = self.tma_warp_id + 1
            self.threads_per_cta += 32
        self.situ_n_lat_tiles = n_lat_tiles
        self.situ_a_pow = a_pow
        self.situ_clamp_up = clamp_up


def _ptr(dtype, t: torch.Tensor):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class PackedSituGemm:
    """``(x, act, latent_out)``: latent_out = x @ W[:n_lat].T (BF16) and the
    shared SiTU of x @ W[n_lat:].T written as MXFP8 into columns [0, n_sh/2)
    of ``act`` (an ``Mxfp8Buffer``)."""

    def __init__(
        self,
        weight_nk: torch.Tensor,
        scale: torch.Tensor,
        n_lat: int,
        beta: float,
        linear_beta: float | None,
        tactic=DEFAULT_TACTIC,
    ) -> None:
        self.tactic = tactic
        n, k = weight_nk.shape
        if n_lat % TILE[1] or (n - n_lat) % TILE[1] or k % 128:
            raise ValueError(f"packed SiTU GEMM: unsupported N/K ({n_lat}, {n}, {k})")
        self.weight, self.scale = weight_nk, scale
        self.n, self.k, self.n_lat = n, k, n_lat
        k_host, self.a_pow, self.clamp_up = situ_constants(beta, linear_beta)
        self.situ_k = k_host.to(weight_nk.device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = None

    def _args(self, x, x_scale, act, lat):
        f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        return (
            _ptr(f8, x),
            _ptr(f8, self.weight),
            _ptr(e8, x_scale),
            _ptr(e8, self.scale),
            _ptr(cutlass.BFloat16, lat),
            from_dlpack(self.alpha),
        )

    def __call__(self, x, x_scale, act, lat: torch.Tensor) -> None:
        m = x.shape[0]
        assert x.shape[1] == self.k and x.stride(1) == 1 and lat.is_contiguous()
        assert act.data.shape[0] == m and act.data.is_contiguous()
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = self._args(x, x_scale, act, lat)
        problem = (cutlass.Int32(m), cutlass.Int32(self.n), cutlass.Int32(self.k), 1)
        tail = (
            cutlass.Int64(self.n_lat),  # c_ld
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
        key = (
            self.n_lat // TILE[1], self.a_pow, self.clamp_up,
            x.device.index, self.n, self.k, self.tactic,
        )  # fmt: skip
        if self._compiled is None:
            self._compiled = _COMPILED.get(key)
        if self._compiled is None:
            gemm = _Kernel(*key[:3], self.tactic)
            layouts = (
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass_utils.LayoutEnum.ROW_MAJOR,
            )
            cl = self.tactic[1]
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
            self._compiled = cute.compile(
                gemm,
                *args,
                layouts,
                problem,
                mac,
                stream,
                lambda v: v,
                tail[0],
                None,
                *tail[1:],
                True,
                *situ,
            )
            _COMPILED[key] = self._compiled
        self._compiled(*args, problem, stream, *tail, *situ)
