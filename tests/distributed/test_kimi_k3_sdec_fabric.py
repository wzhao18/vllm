# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: counted fabric windows (``specialized_decode.fabric``
+ ``cutedsl.common.fabric``) across processes, on 2 or 4 ranks.

Every rank puts one 4 KiB block (tagged words) to every peer with counted unicast
puts and to every rank (itself included) with one counted multimem put, then waits
on its own two counters. Checked: every block lands bit for bit in both windows,
the counters equal the running targets (``FabricWindow.expected``) after eager calls
and after CUDA-graph replays, and no fabric error is reported. Prints the per-
exchange time of the replayed graph (all ranks exchanging).
"""

import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

BLOCK = 4096
TOTALS = 64 * 1024 + 4 * 4096  # window byte of the 4 u32 reduce-add totals (after the blocks)
RED_COUNTER = 512  # counter 2
THREADS = 128
EAGER, REPLAYS, NODES = 3, 5, 64


def _probe_class():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import fabric
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (  # noqa: E501
        alloc,
        asm,
        launch,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.fabric import DATA, MAX_CTAS

    @asm()
    def count_error(address):
        return "red.global.add.u32 [$0],1;", "l,~{memory}", (Int64(address),)

    class FabricProbe:
        def __init__(self, world):
            self.world = world

        @cute.jit
        def __call__(self, *args):
            launch(self.kernel, args, 1, THREADS, pdl=False)

        @cute.kernel
        def kernel(self, args):
            local, replica, expected, status, base, mc, rank, tag = args
            tid, _, _ = cute.arch.thread_idx()
            smem = cutlass.utils.SmemAllocator()
            words = alloc(smem, Uint32, BLOCK // 4, align=128)
            bar = alloc(smem, Int64, 1, 8).iterator
            addend = alloc(smem, Uint32, 4, align=16)
            for i in cutlass.range(tid, BLOCK // 4, THREADS):
                words[i] = (Uint32(tag) << 20) | (Uint32(rank) << 16) | Uint32(i)
            if tid < 4:
                addend[tid] = Uint32(rank + 1)
            fabric.source_fence()
            if tid == 0:
                fabric.barrier_init(bar)
            cute.arch.sync_threads()
            if tid == 0:
                source = words.iterator
                offset = Int64(DATA) + Int64(rank) * BLOCK
                for r in cutlass.range_constexpr(self.world):
                    if Int32(r) != rank:
                        fabric.put(bar, base + r, offset, 0, source, BLOCK)
                        fabric.red_add(bar, base + r, TOTALS, RED_COUNTER, addend.iterator, 16)
                fabric.put_multimem(bar, mc, offset, 0, source, BLOCK)
                fabric.submit_expect(bar, BLOCK * self.world + 16 * (self.world - 1))
                errors = fabric.wait(bar, 0)
                target = fabric.advance(expected, (self.world - 1) * BLOCK)
                fabric.wait_counter(local, target)
                target = fabric.advance(expected + 8 * MAX_CTAS, self.world * BLOCK)
                fabric.wait_counter(replica, target)
                target = fabric.advance(expected + 16 * MAX_CTAS, 16 * (self.world - 1))
                fabric.wait_counter(local + RED_COUNTER, target)
                if errors != Uint32(0):
                    count_error(status)

    return FabricProbe


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        import cutlass.cute as cute
        from cutlass import Int32, Int64
        from cutlass.cute.runtime import make_fake_stream

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            stream,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.fabric import (
            DATA,
            FabricWindow,
        )

        rank, world = dist.get_rank(), dist.get_world_size()
        window = FabricWindow(dist.group.WORLD, world * BLOCK + 4096, multicast=True)
        status = torch.zeros(1, dtype=torch.int32, device=device)
        probe = _probe_class()(world)
        status_ptr = Int64(status.data_ptr())
        code = cute.compile(
            probe, Int64(0), Int64(0), Int64(0), status_ptr, Int32(0), Int32(0),
            Int32(0), Int32(0), make_fake_stream(),
        )  # fmt: skip

        def call(tag):
            code(
                Int64(window.local.data_ptr()), Int64(window.mc_local.data_ptr()),
                Int64(window.expected.data_ptr()), status_ptr, Int32(window.base),
                Int32(window.multicast), Int32(rank), Int32(tag), stream(device),
            )  # fmt: skip

        def check_counters(calls):
            counters = [
                int(t[:8].view(torch.int64).item())
                for t in (window.local, window.mc_local)
            ]
            assert counters == [calls * (world - 1) * BLOCK, calls * world * BLOCK]
            assert window.expected[0, 0].item() == counters[0]
            assert window.expected[1, 0].item() == counters[1]
            reds = int(window.local[RED_COUNTER : RED_COUNTER + 8].view(torch.int64).item())
            assert reds == calls * (world - 1) * 16 == window.expected[2, 0].item()
            totals = window.local[TOTALS : TOTALS + 16].view(torch.int32).tolist()
            peers = sum(r + 1 for r in range(world) if r != rank)
            assert totals == [calls * peers] * 4, totals

        for tag in range(1, EAGER + 1):
            call(tag)
            torch.cuda.synchronize()
            dist.barrier()
            r_ = torch.arange(BLOCK // 4, dtype=torch.int64, device=device)
            for r in range(world):
                want = (tag << 20) | (r << 16) | r_
                got = window.mc_local[DATA + r * BLOCK : DATA + (r + 1) * BLOCK]
                assert torch.equal(got.view(torch.int32).to(torch.int64), want)
                if r != rank:
                    got = window.local[DATA + r * BLOCK : DATA + (r + 1) * BLOCK]
                    assert torch.equal(got.view(torch.int32).to(torch.int64), want)
            check_counters(tag)
            dist.barrier()  # nobody overwrites a block before every rank checked it
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(NODES):
                call(EAGER + 1)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(REPLAYS):
            dist.barrier()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        dist.barrier()
        check_counters(EAGER + REPLAYS * NODES)
        assert status.item() == 0, "fabric error reports"
        if rank == 0:
            print(f"fabric probe: {world} ranks, {BLOCK} B unicast to every peer + "
                  f"multimem: {best / NODES * 1e6:.2f} us per exchange (host-timed graph)")
        dist.barrier()
        window.close()
    finally:
        dist.destroy_process_group()


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = ctx.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()
    deadline = time.monotonic() + 180
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert [process.exitcode for process in processes] == [0] * world_size


def _fabric_supported() -> bool:
    if torch.accelerator.device_count() < 2:
        return False
    try:
        from vllm.models.kimi_k3.nvidia.specialized_decode.fabric import supported

        return supported(0)
    except Exception:
        return False


@pytest.mark.skipif(
    not _fabric_supported(),
    reason="Need 2 GPUs with counted fabric logical endpoints (CUDA 13.5 driver).",
)
def test_fabric_window_counted_puts():
    pytest.importorskip("cutlass")
    world = 4 if torch.accelerator.device_count() >= 4 else 2
    _distributed_run(_worker, world_size=world)
