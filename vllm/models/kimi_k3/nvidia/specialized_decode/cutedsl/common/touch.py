# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight touch: TMA bulk loads of persistent weight ranges into discarded
shared-memory slots, so that the next GEMM over those weights reads them from L2.

``cp.async.bulk.prefetch.L2`` (``prefetch.L2Prefetch``) does not leave a later GEMM
L2-warm on Rubin; a real bulk load into shared memory does (64 MB: the GEMM after it
runs ~40% faster). The touch runs on a side stream beside a predecessor that leaves
HBM idle (the SP reduce-scatter), never beside an HBM-bound kernel.

Every warp's lane 0 is one issuer: a TMA issuer completes one bulk op at a time, so
the grid's ``4 x grid`` issuers each own a 16 KiB slot and walk the chunks of the
ranges table ``[count, 2]`` (address, bytes) strided by the issuer count. The data
is never read; the touch has no effect on any result."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint8, Uint64
from cutlass.cute.runtime import make_fake_stream

from .host import fake
from .primitives import alloc, asm, init_barriers, launch, thread, wait

CHUNK = 16384
WARPS = 4


@asm()
def _bulk_load(dst, src, barrier, size):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    code += " [$0],[$1],$3,[$2];"
    return code, "r,l,r,r,~{memory}", (dst, src, barrier, Int32(size))


class L2Touch:
    def __init__(self, grid, chunk=CHUNK, warps=WARPS):
        self.grid, self.chunk, self.warps = grid, chunk, warps
        self.shared_bytes = warps * chunk + 8 * warps

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 32 * self.warps, pdl=False)

    @cute.kernel
    def kernel(self, args):
        (ranges,) = args
        cute.arch.griddepcontrol_launch_dependents()
        tid, warp, lane, block = thread()
        storage = alloc(cutlass.utils.SmemAllocator(), Uint8, self.shared_bytes, 128)
        slot = storage.iterator + warp * self.chunk
        bars = cute.recast_ptr(storage.iterator + self.warps * self.chunk, dtype=Uint64)
        if tid == 0:
            init_barriers(bars, self.warps, async_proxy=True)
        cute.arch.sync_threads()
        if lane == 0:
            bar = bars + warp
            issuer = Int64(warp * self.grid + block)  # consecutive chunks: other SMs
            issuers = Int64(self.grid * self.warps)
            phase, done, row = Int32(0), Int64(0), Int32(0)
            while row < ranges.shape[0]:
                base, nbytes = ranges[row, 0], ranges[row, 1]
                chunks = (nbytes + self.chunk - 1) // self.chunk
                index = issuer - done % issuers
                if index < 0:
                    index += issuers
                while index < chunks:
                    offset = index * self.chunk
                    size = nbytes - offset
                    if size > self.chunk:
                        size = Int64(self.chunk)
                    size = size & Int64(-16)
                    if size > 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, size.to(Int32))
                        _bulk_load(slot, base + offset, bar, size.to(Int32))
                        wait(bar, phase, scope="cta")
                        phase ^= 1
                    index += issuers
                done += chunks
                row += 1


@functools.cache
def compile_touch(count, grid, device_index, chunk=CHUNK, warps=WARPS):
    ranges = fake((count, 2), None, torch.int64)
    with torch.accelerator.device_index(device_index):
        return cute.compile(L2Touch(grid, chunk, warps), ranges, make_fake_stream())


def touch_table(tensors, device):
    """The persistent ``[count, 2]`` (address, bytes) table of ``tensors`` (each
    contiguous, 16-byte aligned, alive for as long as the table is used)."""
    rows = []
    for t in tensors:
        if not t.is_contiguous() or t.data_ptr() % 16:
            raise ValueError(f"touch range {tuple(t.shape)} not contiguous / aligned")
        rows.append((t.data_ptr(), t.numel() * t.element_size()))
    return torch.tensor(rows, dtype=torch.int64, device=device)


def sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count
