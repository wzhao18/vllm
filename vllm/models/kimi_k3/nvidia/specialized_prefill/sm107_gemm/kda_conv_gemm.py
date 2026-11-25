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
        # "ag": the A operand is the SP all-gather landing during this GEMM
        # (ag_wait: rotated cluster M order + per-cluster-tile counter waits).
        self.ag_wait = "ag" in flags
        # "red": the TMA store reduce-adds the BF16 tile into C (residual add).
        self.c_store_reduce = "red" in flags
        self.l2_store_evict_first = "sef" in flags
        # die_sched: clusters claim N tiles of their die's weight half
        # (die-split weights, extra scheduler warp 10).
        self.die_sched = any(f.startswith("die") for f in flags)
        self.die_group = next((int(f[3:]) for f in flags if f.startswith("die") and f[3:].isdigit()), 0)
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
        # Column-wise conv epilogue skeletons (microbench only, see
        # conv_epilogue.epilogue_kda_conv_t).
        self.conv_skel = next((i for i in (1, 2, 3, 4) if f"cskel{i}" in flags), 0)
        # Branch-free column pass (epilogue_kda_conv_t2) unless "ct1".
        self.conv_t2 = "ct1" not in flags and self.conv_skel == 0
        self.conv_scalar_silu = "sc" in flags
        self.conv_math_smsp23 = "c23" in flags
        self.conv_probe = 1 if "nolds" in flags else 2 if "nost" in flags else 0
        self.conv_loop_unroll = next((u for u in (1, 2, 4, 8) if f"lp{u}" in flags), 0)
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
        if self.conv_drain and self.conv_t2 and not split_producer and "nrs" not in flags:
            # Twelve warps: warpgroup 2 = MMA, TMA, two idle warps; it drops to
            # 40 registers so the epilogue warpgroups hold the drained tile
            # (128 registers of BF16 pairs) plus the column pass without spills
            # (sum 232 + 232 + 40 = 504, the SM107 limit).
            self.threads_per_cta = 32 * 12
            self.reg_split = (232, 40)
        if split_producer and tactic[5] == 2:
            # Second producer warp issues the odd k-blocks (whole stages).
            self.tma_kil_warp_id = self.tma_warp_id + 1
            self.threads_per_cta += 32
        elif split_producer:
            self.tma_b_warp_id = self.tma_warp_id + 1
            self.threads_per_cta += 32


def _ag_args(ag):
    """(ag_ctr, ag_target, ag_rot, ag_lo, ag_hi) kernel arguments (zeros: no
    overlapped all-gather)."""
    if ag is None:
        return (cutlass.Int64(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))
    return (
        cutlass.Int64(ag.ctr_ptr),
        cutlass.Int32(ag.target),
        cutlass.Int32(ag.rot),
        cutlass.Int32(ag.lo),
        cutlass.Int32(ag.hi),
    )


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


def ag_tactic(tactic):
    """``tactic`` for the overlapped all-gather form: flag "ag", cluster
    swizzle 1 (the wave front then walks the M clusters in order, so it needs
    the shards in the order the pushers deliver them)."""
    flags = [f for f in (tactic[7] if len(tactic) > 7 else "").split("+") if f]
    t = list(tactic) + [0] * (8 - len(tactic))
    t[3] = 1
    t[7] = "+".join(flags + ["ag"])
    if len(tactic) <= 7:
        t[5], t[6] = (tactic[5] if len(tactic) > 5 else 0), (tactic[6] if len(tactic) > 6 else 0)
    return tuple(t)


def cluster_rows(tactic) -> int:
    """Rows of one cluster M tile (the all-gather's counted-put unit)."""
    (tm, _), (cm, _), inst_m = tactic[0], tactic[1], tactic[2]
    cta_m = tm // 2 if inst_m == 256 else tm
    return cta_m * cm


def shape_tactic(n: int, tactic):
    """``tactic`` fitted to an ``n``-column problem, or ValueError.

    The conv epilogue stores with raw global addresses (no column bound), so
    every N tile the persistent scheduler hands out must lie inside ``n``:
    whole cluster columns (``n`` a multiple of tile N x cluster N), and with the
    "m" raster a swizzle that divides the number of cluster columns (the
    scheduler pads that count up to a multiple of the swizzle; the padded
    clusters would run tiles past column ``n`` and write past the Q|K|V planes
    and side rows). The swizzle drops to its largest divisor of the cluster
    column count; it only orders tiles, the result is bit for bit the same.
    The "n" raster pads M instead, whose rows the epilogue bounds."""
    (_, tn), (_, cn), _, swz, raster = tactic[:5]
    if n % (tn * cn):
        raise ValueError(
            f"KDA conv GEMM: N {n} is not a multiple of tile N x cluster N {tn}x{cn}"
        )
    if raster == "m":
        ncl = n // (tn * cn)
        fit = max(d for d in range(1, swz + 1) if ncl % d == 0)
        if fit != swz:
            tactic = (*tactic[:3], fit, *tactic[4:])
    return tactic


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
        self.tactic = shape_tactic(self.n, tactic or DEFAULT_TACTIC)
        self.drain = drain and self.tactic[0][0] == 2 * self.tactic[2]
        # Column-pair weight layout for the branch-free column pass:
        # [col pair][tap][col, col + 1] (aligned f32x2 operands).
        self.conv_w_pairs = conv_w.view(self.n // 2, 2, 4).transpose(1, 2).contiguous()
        flags = set((self.tactic[7] if len(self.tactic) > 7 else "").split("+"))
        self._t2 = self.drain and "ct1" not in flags and not any(
            f.startswith("cskel") for f in flags
        )
        self.alpha = torch.ones(1, dtype=torch.float32, device=weight_nk.device)
        self._compiled = {}

    def _kernel(self, norms, args, problem, tail, conv, stream, ag=None):
        tactic = ag_tactic(self.tactic) if ag is not None else self.tactic
        key = ("conv", tactic, norms, self.drain, self.weight.device.index, self.n, self.k)
        fn = self._compiled.get(key) or _COMPILED.get(key)
        if fn is None:
            gemm = _Kernel(tactic, conv=True, norms=norms, drain=self.drain)
            gemm.conv_mode = True
            gemm.conv_p_dim = self.p
            cl = tactic[1]
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
            layouts = (
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass_utils.LayoutEnum.ROW_MAJOR,
            )
            fn = cute.compile(
                gemm, *args, layouts, problem, mac, stream, lambda v: v,
                tail[0], None, *tail[1:], True,
                None, None, cutlass.Int64(0), cutlass.Int64(0), None, *conv, *_ag_args(ag),
            )  # fmt: skip
            _COMPILED[key] = fn
        self._compiled[key] = fn
        return fn

    @property
    def supports_checkpoint_halo(self) -> bool:
        """Whether ``__call__`` takes ``checkpoint_offsets`` (the branch-free
        column-pass epilogue, no norms)."""
        return self._t2

    def __call__(
        self, x, x_scale, conv_state, qsl, lines, has_init, with_norms=False,
        null_block_id: int | None = None, checkpoint_offsets=None, halo=None,
        ag=None,
    ):  # fmt: skip
        """``checkpoint_offsets`` (int32, one per sequence, 0 = none) with
        ``halo`` (BF16 ``(nseq, 3, N)``): the epilogue also writes the three
        pre-conv (in_proj output) rows before each checkpoint offset >= 3 to
        ``halo[s]``; other rows of ``halo`` are left untouched."""
        from cutlass.cute.runtime import from_dlpack

        from .conv_fixup import conv_fixup

        m = x.shape[0]
        nseq = qsl.numel() - 1
        dev = x.device
        ckpt_ptr = halo_ptr = 0
        if checkpoint_offsets is not None:
            if not self._t2 or with_norms:
                raise ValueError("KDA conv GEMM: checkpoint halo needs the t2 epilogue")
            if (
                checkpoint_offsets.dtype != torch.int32
                or checkpoint_offsets.numel() != nseq
                or not checkpoint_offsets.is_contiguous()
                or halo is None
                or halo.dtype != torch.bfloat16
                or tuple(halo.shape) != (nseq, 3, self.n)
                or not halo.is_contiguous()
            ):
                raise ValueError("KDA conv GEMM: checkpoint offsets / halo layout")
            ckpt_ptr, halo_ptr = checkpoint_offsets.data_ptr(), halo.data_ptr()
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
            cutlass.Int64(
                (self.conv_w_pairs if self._t2 and not with_norms else self.conv_w).data_ptr()
            ),
            cutlass.Int64(side.data_ptr()),
            cutlass.Int64(qsl.data_ptr()),
            cutlass.Int64(norms.data_ptr() if norms is not None else 0),
            cutlass.Int32(nseq),
            cutlass.Int32(top),
            cutlass.Int32(nsteps),
            # sched_ctr, sched_die_map (unused), then the checkpoint halo.
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(ckpt_ptr),
            cutlass.Int64(halo_ptr),
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args, problem, tail = _common_args(x, x_scale, self.weight, self.scale, out, self.n)
        args = (*args, from_dlpack(self.alpha))
        fn = self._kernel(with_norms, args, problem, tail, conv, stream, ag)
        fn(*args, problem, stream, *tail, cutlass.Int64(0), cutlass.Int64(0), *conv, *_ag_args(ag))
        if ag is not None:
            # The pushers: PDL secondary of the GEMM, on the SMs its grid leaves
            # idle; they wait for the GEMM's completion at their end, so every
            # later kernel sees both.
            ag.launch_pushers()
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
