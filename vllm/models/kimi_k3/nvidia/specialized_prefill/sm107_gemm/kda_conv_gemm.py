# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill in_proj Q|K|V GEMM with the short conv in its epilogue.

``KdaConvGemm`` computes ``conv(x @ W_qkv.T)`` for the non-spec prefill rows:
the sm_107 block-scaled GEMM rounds each accumulator to BF16 (the in_proj
output dtype), runs the width-4 causal conv in FP32 (stock FMA order, oldest
tap first), SiLU as the stock Triton kernel lowers it, and stores BF16 Q|K|V
planes ``(3, M, P)``; optionally the FP32 inverse L2 norm of every Q and K
head row. Rows whose taps cross a 128-row block or a sequence start (and the
conv state update) are finished by ``conv_fixup``, from the pre-conv rows the
epilogue saves. Bit for bit equal to in_proj + ``kda_conv.qkv_conv``.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_ptr

from .dense_blockscaled_gemm_sm107 import Sm107BlockScaledPersistentDenseGemmKernel

SF_VEC = 32
#: (mma tile M, N), cluster (M, N), MMA instruction M, swizzle, raster.
DEFAULT_TACTIC = ((512, 256), (4, 1), 256, 8, "n")
L2_NORM_EPS = 1e-6
_COMPILED: dict = {}


class _Kernel(Sm107BlockScaledPersistentDenseGemmKernel):
    def __init__(
        self, tactic, conv: bool = False, norms: bool = False, drain: bool = False
    ):
        (tm, tn), cluster, inst_m, swz, raster = tactic[:5]
        split_producer = len(tactic) > 5 and bool(tactic[5])
        # tactic[7]: epilogue variant ("" store after full drain, "early" TMEM
        # released after the last load, "noepi" skeleton with no output).
        flags = set((tactic[7] if len(tactic) > 7 else "").split("+"))
        self.epi_early_release = "early" in flags
        self.skel_noepi = "noepi" in flags
        self.epi_staged = "staged" in flags
        self.skel_hot = "hot" in flags
        self.sf_tmem_slots = 2 if "sf2" in flags else 1
        self.epi_drain = "drain" in flags
        # die_sched: clusters claim N tiles of their die's weight half
        # (die-split weights, extra scheduler warp 10).
        self.die_sched = "die" in flags
        # Drain skeletons (microbench only): 1 = drain + release, 2 = + math.
        self.epi_x2 = "x2" in flags
        self.epi_tanh = "tanh" in flags
        self.epi_roll = 2 if "roll2" in flags else 1 if "roll" in flags else 0
        self.epi_store = (
            3 if "nost" in flags else 2 if "nosf" in flags else 1 if "st8" in flags else 0
        )
        self.epi_yield = 1 if "y1" in flags else 2 if "y2" in flags else 0
        self.ab_stage_cap = next((s for s in (3, 4, 5) if f"ab{s}" in flags), 0)
        self.epi_drain_skel = next((i for i in (11, 10, 9, 1, 2, 3, 4, 5, 7) if f"dskel{i}" in flags), 0)
        super().__init__(
            SF_VEC,
            (inst_m, tn, 64),
            (tm, tn, 256 if "k256" in flags else 128),
            cluster,
            prefetch_dist=0,
            swizzle_size=swz,
            raster_order=raster,
        )
        self.conv_norms = norms
        self.conv_drain = conv and drain
        if self.conv_drain or self.epi_drain:
            # Two warp groups drain the two M blocks of the B-reuse tile.
            self.epilog_warp_id = tuple(range(8))
            self.epilogue_warp_id = self.epilog_warp_id
            self.mma_warp_id = 8
            self.tma_warp_id = 9
            self.threads_per_cta = 32 * 10
            if self.die_sched:
                self.sched_warp_id = 10
                self.threads_per_cta = 32 * 11
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id, num_threads=256
            )
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_ptr_sync_bar_id, num_threads=32 * 9
            )
        if split_producer and tactic[5] == 2:
            # Second producer warp issues the odd k-blocks (whole stages).
            self.tma_kil_warp_id = self.tma_warp_id + 1
            self.threads_per_cta += 32
        elif split_producer:
            self.tma_b_warp_id = self.tma_warp_id + 1
            self.threads_per_cta += 32


def _ptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _common_args(x, x_scale, weight, scale, out, n):
    f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    m = x.shape[0]
    args = (
        _ptr(f8, x),
        _ptr(f8, weight),
        _ptr(e8, x_scale),
        _ptr(e8, scale),
        _ptr(cutlass.BFloat16, out),
    )
    problem = (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(x.shape[1]), 1)
    tail = (
        cutlass.Int64(n),  # c_ld
        cutlass.Int64(0),  # sfc_row_pitch
        cutlass.Int64(0),  # sfc_batch_offset
        cutlass.Int64(x.stride(0)),  # a_ld
        cutlass.Int64(0),  # a_batch_stride
        cutlass.Int64(0),  # c_batch_stride
    )
    return args, problem, tail


class PlainGemm:
    """BF16 ``x @ W.T`` with the vendored kernel (tactic benchmarking)."""

    def __init__(self, weight_nk, scale, tactic=DEFAULT_TACTIC):
        self.weight, self.scale, self.tactic = weight_nk, scale, tactic
        self.n, self.k = weight_nk.shape
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = None

    def __call__(self, x, x_scale, out):
        from cutlass.cute.runtime import from_dlpack

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args, problem, tail = _common_args(
            x, x_scale, self.weight, self.scale, out, self.n
        )
        swap = len(self.tactic) > 6 and bool(self.tactic[6])
        if swap:
            # C^T = W @ x^T: tokens on the MMA N side, C column-major (the same
            # row-major [m, n] bytes).
            a, b, sa, sb, c = args
            args = (b, a, sb, sa, c)
            m = x.shape[0]
            problem = (cutlass.Int32(self.n), cutlass.Int32(m), cutlass.Int32(self.k), 1)
            tail = (tail[0], tail[1], tail[2], cutlass.Int64(self.weight.stride(0)), *tail[4:])
        args = (*args, from_dlpack(self.alpha))
        if self._compiled is None:
            key = ("plain", self.tactic, x.device.index, self.n, self.k)
            self._compiled = _COMPILED.get(key)
            if self._compiled is None:
                gemm = _Kernel(self.tactic)
                cl = self.tactic[1]
                mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
                layouts = (
                    OperandMajorMode.K,
                    OperandMajorMode.K,
                    cutlass_utils.LayoutEnum.COL_MAJOR
                    if swap
                    else cutlass_utils.LayoutEnum.ROW_MAJOR,
                )
                self._compiled = cute.compile(
                    gemm, *args, layouts, problem, mac, stream, lambda v: v,
                    tail[0], None, *tail[1:], True,
                )  # fmt: skip
                _COMPILED[key] = self._compiled
        self._compiled(*args, problem, stream, *tail)


def side_rows(m: int, nseq: int) -> int:
    return 6 * ((m + 127) // 128) + 6 * nseq


class KdaConvGemm:
    """``(x, x_scale, conv_state, qsl, lines, has_init) -> (q|k|v, inv_norms)``.

    ``weight_nk`` / ``scale``: the MXFP8 Q|K|V rows of in_proj (cuBLAS 128x4
    swizzled scales); ``conv_w``: (3P, 4) FP32. Returns BF16 ``(3, M, P)`` and,
    with ``with_norms``, FP32 ``(2, M, P / 128)`` inverse L2 norms of the Q and
    K head rows (``rsqrt(sum(y^2) + 1e-6)``, y = the BF16 conv output).
    """

    def __init__(self, weight_nk, scale, conv_w, part_dim, tactic=None, drain=True):
        self.weight, self.scale, self.conv_w = weight_nk, scale, conv_w
        self.n, self.k = weight_nk.shape
        self.p = part_dim
        if self.n != 3 * part_dim or part_dim % 256 or self.k % 128:
            raise ValueError(f"KDA conv GEMM: unsupported N/K {self.n}/{self.k}")
        if tuple(conv_w.shape) != (self.n, 4) or conv_w.dtype != torch.float32:
            raise ValueError(f"KDA conv GEMM: conv weight {tuple(conv_w.shape)}")
        self.tactic = tactic or DEFAULT_TACTIC
        self.drain = drain and self.tactic[0][0] == 2 * self.tactic[2]
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = {}

    def _kernel(self, norms, args, problem, tail, conv, stream):
        key = ("conv", self.tactic, norms, self.drain, self.weight.device.index, self.n, self.k)
        fn = self._compiled.get(key) or _COMPILED.get(key)
        if fn is None:
            gemm = _Kernel(self.tactic, conv=True, norms=norms, drain=self.drain)
            gemm.conv_mode = True
            cl = self.tactic[1]
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
            layouts = (
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass_utils.LayoutEnum.ROW_MAJOR,
            )
            fn = cute.compile(
                gemm, *args, layouts, problem, mac, stream, lambda v: v,
                tail[0], None, *tail[1:], True,
                None, None, cutlass.Int64(0), cutlass.Int64(0), None, *conv,
            )  # fmt: skip
            _COMPILED[key] = fn
        self._compiled[key] = fn
        return fn

    def __call__(
        self, x, x_scale, conv_state, qsl, lines, has_init, with_norms=False,
        null_block_id: int | None = None,
    ):  # fmt: skip
        from cutlass.cute.runtime import from_dlpack

        from .conv_fixup import conv_fixup

        m = x.shape[0]
        nseq = qsl.numel() - 1
        dev = x.device
        out = torch.empty(3, m, self.p, dtype=torch.bfloat16, device=dev)
        side = torch.empty(side_rows(m, nseq), self.n, dtype=torch.bfloat16, device=dev)
        norms = (
            torch.empty(2, m, self.p // 128, dtype=torch.float32, device=dev)
            if with_norms
            else None
        )
        top = 1
        while top * 2 < nseq:
            top *= 2
        nsteps = top.bit_length() if nseq > 1 else 0
        conv = (
            cutlass.Int64(out.data_ptr()),
            cutlass.Int64(self.conv_w.data_ptr()),
            cutlass.Int64(side.data_ptr()),
            cutlass.Int64(qsl.data_ptr()),
            cutlass.Int64(norms.data_ptr() if norms is not None else 0),
            cutlass.Int32(nseq),
            cutlass.Int32(top),
            cutlass.Int32(nsteps),
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args, problem, tail = _common_args(x, x_scale, self.weight, self.scale, out, self.n)
        args = (*args, from_dlpack(self.alpha))
        fn = self._kernel(with_norms, args, problem, tail, conv, stream)
        fn(*args, problem, stream, *tail, cutlass.Int64(0), cutlass.Int64(0), *conv)
        conv_fixup(
            out, side, norms, self.conv_w, conv_state, qsl, lines, has_init,
            self.p, null_block_id,
        )  # fmt: skip
        return out, norms


def reference_inv_norms(y: torch.Tensor, part_dim: int) -> torch.Tensor:
    """FP32 ``rsqrt(sum(y^2) + eps)`` per Q / K head row of ``y`` (3, M, P)."""
    qk = y[:2].float().unflatten(-1, (part_dim // 128, 128))
    return torch.rsqrt(qk.square().sum(-1) + L2_NORM_EPS)


class MixedGemm:
    """BF16 ``x @ W.T``, preferred / fallback cluster launch (full SM coverage)."""

    def __init__(self, weight_nk, scale, tile, preferred, fallback):
        from .dense_blockscaled_gemm_sm107 import (
            Sm107BlockScaledPersistentDenseGemmMixedClustersKernel as K,
        )

        self.weight, self.scale = weight_nk, scale
        self.n, self.k = weight_nk.shape
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self.gemm = K(SF_VEC, (256, tile[1], 64), (tile[0], tile[1], 128), preferred, fallback)
        hw = cutlass_utils.HardwareInfo()
        self.pmac = hw.get_max_active_clusters(preferred[0] * preferred[1])
        self.fmac = hw.get_max_active_clusters(fallback[0] * fallback[1])
        self._compiled = None

    def __call__(self, x, x_scale, out):
        from cutlass.cute.runtime import from_dlpack

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args, problem, tail = _common_args(x, x_scale, self.weight, self.scale, out, self.n)
        alpha = from_dlpack(self.alpha)
        layouts = (OperandMajorMode.K, OperandMajorMode.K, cutlass_utils.LayoutEnum.ROW_MAJOR)
        if self._compiled is None:
            self._compiled = cute.compile(
                self.gemm, *args, layouts, problem, self.pmac, self.fmac, stream,
                alpha, lambda v: v, tail[0],
            )  # fmt: skip
        self._compiled(*args, problem, stream, alpha, tail[0])
