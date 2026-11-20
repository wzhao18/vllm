# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode MXFP8 GEMM with a pre-wait weight prologue (GEMM-EX1 prototype).

``out = x @ W^T`` (MXFP8 W8A8, F8_128x4 swizzled E8M0 scales, FP32 accumulation, BF16
out) on the vendored FlashInfer sm_107 persistent block-scaled kernel
(``dense_blockscaled_gemm_sm107``), launched with PDL: the TMA warp issues the weight
operand (and its scales) of the first ``prewait`` k-blocks of its first tile before
``griddepcontrol.wait``, so weight bytes stream while the predecessor drains.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_ptr

from .dense_blockscaled_gemm_sm107 import Sm107BlockScaledPersistentDenseGemmKernel

SF_VEC = 32
INST_K = 64
TILER_K = 128


def _arch(device):
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


@cache
def _max_active_clusters(cluster_size: int) -> int:
    hw = cutlass.utils.HardwareInfo()
    return hw.get_max_active_clusters(cluster_size)


@cache
def _alpha_one(device):
    return torch.ones(1, device=device, dtype=torch.float32)


@cache
def _compile(tactic, use_pdl, prewait, device_index, quiet=False):
    tile, cluster, swap_ab, inst_m = tactic
    gemm = Sm107BlockScaledPersistentDenseGemmKernel(
        SF_VEC,
        (inst_m, tile[1], INST_K),
        (tile[0], tile[1], TILER_K),
        cluster,
        use_pdl=use_pdl,
        prewait_stages=prewait,
        weight_is_a=swap_ab,
        quiet_wait=quiet,
    )
    sym_m, sym_k, sym_n = cute.sym_int(), cute.sym_int(), cute.sym_int()
    f8 = cutlass.Float8E4M3FN
    a = cute.runtime.make_fake_compact_tensor(f8, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16)
    b = cute.runtime.make_fake_compact_tensor(f8, (sym_n, sym_k), stride_order=(1, 0), assumed_align=16)
    if swap_ab:
        c = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (sym_n, sym_m), stride_order=(0, 1), assumed_align=16
        )
    else:
        c = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16
        )
    sfa = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    sfb = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    mac = _max_active_clusters(cluster[0] * cluster[1])
    with torch.accelerator.device_index(device_index):
        arch = _arch(device_index)
        compiled = cute.compile(
            gemm.wrapper, a, b, c, cutlass.Int64(1), cutlass.Int64(1), cutlass.Int64(1), 1,
            sfa, sfb, alpha, mac, stream, swap_ab,
            options=f"--opt-level 2 --gpu-arch={arch} --enable-tvm-ffi",
        )  # fmt: skip
    return compiled, gemm


class Mxfp8PreGemm:
    """One (N, K, R) MXFP8 GEMM on a fixed tactic ``(tile_mn, cluster_mn, swap_ab,
    inst_m)`` (FlashInfer's sm_107 tactic tuple). ``out_layout``: "rn" = out [R, N]
    row-major; "cn" = the swap_ab store (out viewed with strides (1, R))."""

    def __init__(self, n, k, rows, tactic, *, use_pdl=True, prewait=0, quiet=False, device=None):
        device = torch.device("cuda") if device is None else device
        self.n, self.k, self.rows = n, k, rows
        self.swap_ab = bool(tactic[2])
        if self.swap_ab:  # the swap_ab store layout is not wired (out would be wrong)
            raise ValueError("Mxfp8PreGemm: swap_ab tactics are not supported")
        tactic = (tuple(tactic[0]), tuple(tactic[1]), self.swap_ab, int(tactic[3]))
        if prewait and prewait > k // TILER_K:
            raise ValueError(f"prewait {prewait} > {k // TILER_K} k-blocks")
        self.compiled, self.gemm = _compile(
            tactic, bool(use_pdl), int(prewait), device.index or 0, bool(quiet)
        )
        self.stages = getattr(self.gemm, "num_ab_stage", None)
        if prewait and self.stages is not None and prewait > self.stages:
            raise ValueError(f"prewait {prewait} > {self.stages} smem stages")
        self.alpha = _alpha_one(device)
        self.out_layout = "cn" if self.swap_ab else "rn"
        # (sf_m, sf_n, sf_k) of the kernel's M (weights when swap_ab) and N
        km, kn = (n, rows) if self.swap_ab else (rows, n)
        self.sf = ((km + 127) // 128, (kn + 127) // 128, (k // SF_VEC + 3) // 4)

    def __call__(self, xq, xs, w, ws, out):
        if self.swap_ab:
            a, b, asf, bsf = w, xq, ws, xs
            c = out.as_strided(out.shape, (1, out.shape[0]))
        else:
            a, b, asf, bsf = xq, w, xs, ws
            c = out
        self.compiled(a, b, c, *self.sf, asf.data_ptr(), bsf.data_ptr(), self.alpha)
        return out


class _SpinKernel:
    """Producer stand-in: every CTA triggers launch_dependents at entry, then spins
    ``iters`` dependent FMAs per thread (poll-bound host emulation), writes one word."""

    def __init__(self, iters, smem):
        self.iters, self.smem = iters, smem

    @cute.jit
    def __call__(self, sink: cute.Tensor, ctas: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(sink).launch(
            grid=(ctas, 1, 1), block=(128, 1, 1), smem=self.smem, stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, sink: cute.Tensor):
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        v = cutlass.Float32(tid) * cutlass.Float32(1e-7)
        for _ in cutlass.range(self.iters, unroll=4):
            v = v * cutlass.Float32(0.999999) + cutlass.Float32(1e-9)
        if v == cutlass.Float32(-1.0):
            sink[bid] = v


class Spin:
    def __init__(self, iters, ctas, smem=0, device=None):
        device = torch.device("cuda") if device is None else device
        self.ctas = ctas
        self.sink = torch.zeros(ctas, device=device, dtype=torch.float32)
        k = _SpinKernel(iters, smem)
        sink = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
        with torch.accelerator.device_index(device.index or 0):
            self.compiled = cute.compile(
                k, sink, cutlass.Int32(ctas), make_fake_stream(use_tvm_ffi_env_stream=True),
                options=f"--gpu-arch={_arch(device.index or 0)} --enable-tvm-ffi",
            )  # fmt: skip

    def __call__(self):
        self.compiled(self.sink, self.ctas)
