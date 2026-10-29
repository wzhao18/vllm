# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel FlashMLA sparse-decode scheduler metadata (CuTe DSL).

FlashMLA computes the split-KV tile schedule with a single-warp serial greedy
loop over requests (``get_decoding_sched_meta.cu``), ~50 ns per request: ~210
us for a 4096-row DCP prefill chunk run as b=4096, s_q=1 with per-row
``topk_length``. That metadata has to be rebuilt for every call because it
bakes the lengths into the split boundaries.

This kernel builds an equivalent, valid schedule in one CTA with block scans
and writes it in FlashMLA's own layout, so a ``FlashMLASchedMeta`` can be
pre-filled and FlashMLA skips its scheduler (no FlashMLA rebuild):

  tile_scheduler_metadata [num_sm_parts, 8] int32, one DecodingSchedMeta
    {begin_req, end_req, begin_block, end_block(excl), begin_split,
     is_first_req_splitted, is_last_req_splitted, pad}   (csrc/params.h)
  num_splits [b + 1] int32, exclusive prefix of per-request split counts.

Partition: request i costs n_i + F units (n_i = max(1, ceil(len_i / 64))
blocks, F = fixed_overhead_num_blocks = 5) on one cost line; part p takes
the blocks whose position S_i + F + k falls in [p*P, (p+1)*P), P =
ceil(total / parts). Each block belongs to exactly one part, parts are
contiguous in (request, block) order, and a request is split iff its blocks
span more than one part, which is what the kernel and the combine expect
(kernel.cuh: start/end block from the meta for the first/last request of a
part, runtime topk_length for the requests strictly inside it). The part
count and F are those of FlashMLA's SM100 d_qk=576 decode impls
(fp8_ds_mla / nvfp4_ds_mla), so it is SM100 only.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta

TB = 1024
WARP = 32
NWARPS = TB // WARP
SCHED_INTS = 8  # sizeof(DecodingSchedMeta) / 4
FIXED_OVERHEAD_BLOCKS = 5  # FlashMLA's fixed_overhead_num_blocks
BLOCK_TOPK = 64  # FlashMLA's block_size_topk


@cute.jit
def _warp_scan(val: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(5):
        partial = cute.arch.shuffle_sync_up(val, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            val += partial
    return val


@cute.jit
def _block_scan(val: Int32, lane: Int32, warp_id: Int32, scratch: cute.Tensor) -> Int32:
    """Inclusive scan over the CTA; ends synced. scratch: smem [NWARPS + 1]."""
    prefix = _warp_scan(val, lane)
    if lane == Int32(WARP - 1):
        scratch[warp_id] = prefix
    cute.arch.sync_threads()
    if warp_id == Int32(0):
        wt = scratch[lane]
        wp = _warp_scan(wt, lane)
        scratch[lane] = wp - wt
        if lane == Int32(WARP - 1):
            scratch[NWARPS] = wp
    cute.arch.sync_threads()
    res = prefix + scratch[warp_id]
    cute.arch.sync_threads()
    return res


def _build(parts: int):
    F, BT = FIXED_OVERHEAD_BLOCKS, BLOCK_TOPK

    @cute.struct
    class Smem:
        scratch: cute.struct.MemRange[Int32, NWARPS + 1]

    @cute.jit
    def nblocks(length: Int32) -> Int32:
        x = length
        if x < Int32(1):
            x = Int32(1)
        return (x - Int32(1)) // Int32(BT) + Int32(1)

    @cute.kernel
    def kern(lengths: cute.Tensor, meta: cute.Tensor, num_splits: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem, 8)
        scratch = st.scratch.get_tensor(cute.make_layout((NWARPS + 1,)))
        b = Int32(lengths.shape[0])
        per = (b + Int32(TB - 1)) // Int32(TB)
        lo = tid * per
        hi = lo + per
        if hi > b:
            hi = b
        # pass 1: cost prefix
        csum = Int32(0)
        for i in range(lo, hi):
            csum += nblocks(lengths[i]) + Int32(F)
        cincl = _block_scan(csum, lane, warp_id, scratch)
        total = scratch[NWARPS]
        cute.arch.sync_threads()
        P = (total + Int32(parts - 1)) // Int32(parts)
        if Int32(1) > P:
            P = Int32(1)
        # pass 2: split counts
        s = cincl - csum
        ssum = Int32(0)
        for i in range(lo, hi):
            n = nblocks(lengths[i])
            pf = (s + Int32(F)) // P
            pl = (s + Int32(F) + n - Int32(1)) // P
            ssum += pl - pf + Int32(1)
            s += n + Int32(F)
        sincl = _block_scan(ssum, lane, warp_id, scratch)
        stotal = scratch[NWARPS]
        # empty parts are skipped by the kernel (begin_req >= b)
        for p in range(tid, parts, TB):
            meta[p, 0] = b
            meta[p, 1] = b - Int32(1)
            meta[p, 2] = Int32(0)
            meta[p, 3] = Int32(0)
            meta[p, 4] = Int32(0)
            meta[p, 5] = Int32(0)
            meta[p, 6] = Int32(0)
            meta[p, 7] = Int32(0)
        cute.arch.sync_threads()
        # pass 3: scatter
        s = cincl - csum
        sp = sincl - ssum
        for i in range(lo, hi):
            n = nblocks(lengths[i])
            first = s + Int32(F)
            pf = first // P
            pl = (first + n - Int32(1)) // P
            ns = pl - pf + Int32(1)
            split = Int32(0)
            if ns > Int32(1):
                split = Int32(1)
            num_splits[i] = sp
            prev_pl = Int32(-1)
            if i > Int32(0):
                prev_pl = (s - Int32(1)) // P
            next_pf = Int32(parts)
            if i + Int32(1) < b:
                next_pf = (first + n + Int32(F)) // P
            for p in range(pf, pl + Int32(1)):
                # begin of part p
                if p > pf:
                    meta[p, 0] = i
                    meta[p, 2] = p * P - first
                    meta[p, 4] = p - pf
                    meta[p, 5] = split
                elif prev_pl < p:
                    meta[p, 0] = i
                    meta[p, 2] = Int32(0)
                    meta[p, 4] = Int32(0)
                    meta[p, 5] = split
                # end of part p
                if p < pl:
                    meta[p, 1] = i
                    meta[p, 3] = (p + Int32(1)) * P - first
                    meta[p, 6] = split
                elif next_pf > p:
                    meta[p, 1] = i
                    meta[p, 3] = n
                    meta[p, 6] = split
            sp += ns
            s += n + Int32(F)
        if tid == Int32(0):
            num_splits[b] = stotal

    @cute.jit
    def host(lengths: cute.Tensor, meta: cute.Tensor, num_splits: cute.Tensor, stream):
        kern(lengths, meta, num_splits).launch(
            grid=(1, 1, 1), block=(TB, 1, 1), stream=stream
        )

    return host


class FlashMLASparseSchedKernel(
    VllmCuTeDSLJitKernel["FlashMLASparseSchedKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        parts: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        return _build(compile_key.parts)

    def get_warmup_keys(self) -> list[CompileKey]:
        # The DCP decode calls run s_q = 1.
        parts = flashmla_num_sm_parts(1, torch.cuda.current_device())
        return [self.CompileKey(parts)]

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        def vector() -> Any:
            return cute.runtime.make_fake_tensor(
                Int32, (cute.sym_int(),), stride=(1,), assumed_align=4
            )

        meta = cute.runtime.make_fake_tensor(
            Int32,
            (compile_key.parts, SCHED_INTS),
            stride=(SCHED_INTS, 1),
            assumed_align=4,
        )
        return vector(), meta, vector()

    @kernel_launcher
    def __call__(
        self, lengths: torch.Tensor, meta: torch.Tensor, num_splits: torch.Tensor
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        return self.CompileKey(meta.shape[0]), (lengths, meta, num_splits)


FLASHMLA_SPARSE_SCHED_KERNEL = FlashMLASparseSchedKernel()


def flashmla_num_sm_parts(s_q: int, device_id: int = 0) -> int:
    """FlashMLA's num_sm_parts for the SM100 sparse decode impls
    (max(num_sms / s_q, 1))."""
    return max(num_compute_units(device_id) // s_q, 1)


def build_sparse_sched_meta(
    topk_length: torch.Tensor,
    q_shape: tuple[int, int, int],
    topk: int,
) -> FlashMLASchedMeta:
    """A pre-filled FlashMLASchedMeta for one flash_mla_with_kvcache call.

    topk_length [b] int32 (device), q_shape (b, s_q, h_q) of the padded q,
    over a cache of 64-token pages.
    """
    b, s_q, h_q = q_shape
    parts = flashmla_num_sm_parts(s_q, topk_length.device.index)
    meta = torch.empty(
        (parts, SCHED_INTS), dtype=torch.int32, device=topk_length.device
    )
    num_splits = torch.empty((b + 1,), dtype=torch.int32, device=topk_length.device)
    FLASHMLA_SPARSE_SCHED_KERNEL(topk_length, meta, num_splits)
    sm = FlashMLASchedMeta()
    sm.have_initialized = True
    sm.config = FlashMLASchedMeta.Config(
        b, s_q, h_q, 64, 1, False, True, topk, None, None, False
    )
    sm.tile_scheduler_metadata = meta
    sm.num_splits = num_splits
    return sm
