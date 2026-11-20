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
def _compile(tactic, use_pdl, prewait, device_index, quiet=False, dbg=0, producers=1,
             prefetch=0, tile_k=TILER_K, die_claim=False, max_stages=0, push=False):
    tile, cluster, swap_ab, inst_m = tactic
    gemm = Sm107BlockScaledPersistentDenseGemmKernel(
        SF_VEC,
        (inst_m, tile[1], INST_K),
        (tile[0], tile[1], tile_k),
        cluster,
        prefetch_dist=prefetch,
        use_pdl=use_pdl,
        prewait_stages=prewait,
        weight_is_a=swap_ab,
        quiet_wait=quiet,
        producers=producers,
        die_claim=die_claim,
        push=push,
    )
    gemm.dbg_skip_mma = bool(dbg & 1)
    gemm.dbg_stamps = bool(dbg & 2)
    gemm.max_ab_stages = max_stages
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
    alpha = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int() if die_claim or push or dbg & 2 else 1,),
        assumed_align=8,
    )
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


_STAMPED = []


@cache
def _stamp_dir():
    """D-GEMM debug probe (WIP branch only): a marker file in the source tree turns on
    per-CTA stamps for every Mxfp8PreGemm and names the directory of the exit dump."""
    import atexit
    import os

    import vllm

    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    marker = os.path.join(root, "benchmarks", "kimi_k3", "dg_stamps.on")
    if not os.path.exists(marker):
        return None
    out = open(marker).read().strip()
    if not out:
        return None
    os.makedirs(out, exist_ok=True)

    def dump():
        import json

        rows = []
        for g in list(_STAMPED):
            torch.cuda.set_device(g.alpha.device)
            s = torch.cuda.Stream(device=g.alpha.device)
            with torch.cuda.stream(s):
                h = g.alpha.view(torch.int32)[512:512 + 4096].to("cpu", non_blocking=True)
            s.synchronize()
            v = h.view(-1, 8).tolist()
            rows.append(dict(n=g.n, k=g.k, rows=g.rows, tactic=repr(g.tactic), stages=g.stages,
                             regs=getattr(g, "dbg_regs", None), threads=g.gemm.threads_per_cta,
                             prewait=g.gemm.prewait_stages, ctas=[r for r in v if r[0] != 0]))
        try:  # the SP gather / reduce kernels' residency (CTAs per SM, SMs, regs, smem)
            from ...attn_res import sp_fused

            rows.append(dict(residency={repr(k): v for k, v in sp_fused.RESIDENCY.items()}))
        except Exception as error:  # debug probe
            rows.append(dict(residency=repr(error)))
        path = os.path.join(out, f"stamps-{os.getpid()}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(rows, f)
        os.replace(path + ".tmp", path)

    import threading

    lock = threading.Lock()  # no dump while a CUDA graph is being captured (and vice versa)
    cls = torch.cuda.CUDAGraph
    begin, end = cls.capture_begin, cls.capture_end

    def capture_begin(self, *a, **kw):
        lock.acquire()
        return begin(self, *a, **kw)

    def capture_end(self, *a, **kw):
        try:
            return end(self, *a, **kw)
        finally:
            lock.release()

    cls.capture_begin, cls.capture_end = capture_begin, capture_end

    def loop():  # worker processes are killed, not exited: dump every 3 s
        import time

        while True:
            time.sleep(3.0)
            try:
                if _STAMPED:
                    with lock:
                        dump()
            except Exception:  # debug probe: keep trying
                pass

    threading.Thread(target=loop, daemon=True).start()
    atexit.register(dump)
    return out


class Mxfp8PreGemm:
    """One (N, K, R) MXFP8 GEMM on a fixed tactic ``(tile_mn, cluster_mn, swap_ab,
    inst_m)`` (FlashInfer's sm_107 tactic tuple). ``out_layout``: "rn" = out [R, N]
    row-major; swap_ab stores the same [R, N] row-major out (W is the MMA's M operand)."""

    def __init__(self, n, k, rows, tactic, *, use_pdl=True, prewait=0, quiet=False, dbg=0,
                 producers=1, prefetch=0, tile_k=TILER_K, die_split=None, max_stages=0,
                 push=False, device=None):
        device = torch.device("cuda") if device is None else device
        self.n, self.k, self.rows = n, k, rows
        self.swap_ab = bool(tactic[2])
        if self.swap_ab and tactic[0][1] % 64:  # (measured: N32 / N96 swap tiles store wrong rows)
            raise ValueError(f"Mxfp8PreGemm: swap_ab needs a token tile multiple of 64 ({tactic[0]})")
        tactic = (tuple(tactic[0]), tuple(tactic[1]), self.swap_ab, int(tactic[3]))
        self.tactic = tactic
        # (push GEMMs keep their argument buffer: no stamps there)
        dbg = int(dbg) | (2 if _stamp_dir() and not push else 0)
        if prewait and prewait > k // tile_k:
            raise ValueError(f"prewait {prewait} > {k // TILER_K} k-blocks")
        if push:
            # the push epilogue stages its tile in the drained A ring: one tile per CTA
            # (cluster) and 128 rows per CTA
            tile, cluster = tactic[0], tactic[1]
            tiles = -(-rows // tile[0]) * -(-n // tile[1])
            if tiles > _max_active_clusters(cluster[0] * cluster[1]):
                raise ValueError(f"push: {tiles} tiles > one per CTA (R {rows})")
            if tile[0] != 128 * cluster[0] or cluster[1] != 1 or die_split is not None:
                raise ValueError(f"push: tactic {tactic} (128 rows per CTA, no die split)")
        # push: 1 relaxed.sys 16 B stores, 2 weak stores, 3 one bulk copy per row
        self.push = int(push)
        self.compiled, self.gemm = _compile(
            tactic, bool(use_pdl), int(prewait), device.index or 0, bool(quiet), int(dbg),
            int(producers), int(prefetch), int(tile_k), die_split is not None,
            int(max_stages), self.push,
        )
        self.stages = getattr(self.gemm, "num_ab_stage", None)
        if prewait and self.stages is not None and prewait > self.stages:
            raise ValueError(f"prewait {prewait} > {self.stages} smem stages")
        self.alpha = _alpha_one(device)
        self.aux, self.calls = None, 0
        if dbg & 2:  # D-GEMM debug probe: per-CTA stamps into a large alpha buffer
            self.alpha = torch.zeros(4608, dtype=torch.float32, device=device)
            self.alpha[0] = 1.0
            try:
                from ...common.host import kernel_registers

                self.dbg_regs = kernel_registers(self.compiled, device.index or 0)
            except Exception as error:  # debug probe
                self.dbg_regs = repr(error)[:200]
            _STAMPED.append(self)
        if die_split is not None:
            self.aux = self._die_aux(tactic, rows, n, die_split, device)
        self.out_layout = "rn"
        # (sf_m, sf_n, sf_k) of the kernel's M (weights when swap_ab) and N
        km, kn = (n, rows) if self.swap_ab else (rows, n)
        self.sf = ((km + 127) // 128, (kn + 127) // 128, (k // SF_VEC + 3) // 4)

    def __call__(self, xq, xs, w, ws, out, push_args=None):
        """``push_args`` (push GEMMs only): the float32 [push_epilogue.ARGS] buffer of
        ``push_args``; ``out`` is then an unused [R, N] BF16 view (the C descriptor)."""
        if self.swap_ab:
            # The kernel stores C [N, R] column-major (only C's pointer is used, leading
            # dim N) = out [R, N] row-major; the strided view only matches the compiled
            # argument's stride order.
            a, b, asf, bsf = w, xq, ws, xs
            c = out.as_strided(out.shape, (1, out.shape[0]))
        else:
            a, b, asf, bsf = xq, w, xs, ws
            c = out
        alpha = self.alpha
        if bool(self.push) != (push_args is not None):
            raise ValueError(f"Mxfp8PreGemm(push={self.push}) push_args {push_args!r}")
        if push_args is not None:
            alpha = push_args
        if self.aux is not None:  # two counter sets, alternated per call
            alpha = self.aux[self.calls % 2]
            self.calls += 1
        self.compiled(a, b, c, *self.sf, asf.data_ptr(), bsf.data_ptr(), alpha)
        return out

    @staticmethod
    def push_args(peers: torch.Tensor, rank: int, n: int, cap: int, rows: int):
        """The push epilogue's argument buffer (``push_epilogue`` docstring) for owner
        receive bases ``peers`` (int64 [TP], device), on ``peers``' device."""
        from .push_epilogue import ARGS

        tp = peers.numel()
        if tp > ARGS // 2 - 4 or rows > tp * n or n > cap:
            raise ValueError(f"push args: TP {tp} rows {rows} n {n} cap {cap}")
        args = torch.zeros(ARGS, dtype=torch.float32, device=peers.device)
        args[0] = 1.0
        ints = args.view(torch.int32)
        ints[1], ints[2], ints[3], ints[4] = rank, n, cap, rows
        args.view(torch.int64)[4 : 4 + tp].copy_(peers.to(torch.int64))
        return args

    @staticmethod
    def _die_aux(tactic, rows, n, die_split, device):
        """alpha + die-claim state: int32 [4..6] counters (die 0, die 1, done), [8..11]
        (first tile, tiles) per die, [12] M tiles, [16 + smid] die of the SM."""
        from ...common.dies import sm_domains

        tile, cluster = tactic[0], tactic[1]
        if tuple(cluster) != (1, 1):
            raise ValueError("die_claim needs a 1x1 cluster tactic")
        m_tiles, n_tiles = -(-rows // tile[0]), -(-n // tile[1])
        total = m_tiles * n_tiles
        if total > _max_active_clusters(1):
            raise ValueError(f"die_claim needs one tile per CTA ({total} tiles)")
        table = sm_domains(device.index or 0)
        n0 = min(die_split, n_tiles) * m_tiles
        aux = torch.zeros(2, 272, dtype=torch.float32, device=device)
        ints = aux.view(torch.int32)
        aux[:, 0] = 1.0
        ints[:, 8], ints[:, 9], ints[:, 10], ints[:, 11] = 0, n0, n0, total - n0
        ints[:, 12] = m_tiles
        ints[:, 16 : 16 + len(table)] = torch.tensor(table, dtype=torch.int32, device=device)
        return aux


def die_weight(w: torch.Tensor, tile_n: int, device=None):
    """(a copy of W [N, K] in die-placed memory, n_split): W rows [0, n_split * tile_n)
    homed on die 0, the rest on die 1 (2 MiB chunks; the chunk across the split goes to
    the die holding most of it), n_split balanced to the SMs per die."""
    from ...common.dies import CHUNK, localized_like, sm_domains

    n, k = w.shape
    table = sm_domains(w.device.index or 0)
    share = sum(1 for d in table if d == 0) / len(table)
    n_tiles = -(-n // tile_n)
    n_split = round(n_tiles * share)
    split = n_split * tile_n * k * w.element_size()
    chunks = -(-w.numel() * w.element_size() // CHUNK)
    domains = [0 if (i + 0.5) * CHUNK < split else 1 for i in range(chunks)]
    return localized_like(w, domains).view(w.dtype).view(n, k), n_split


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
