# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange with peer-copy swaps on a side stream, on the 4 GPUs of
one host.

1. The side-stream exchange (packs on the current stream, swaps and unpacks on
   the side stream) rewrites the cache bitwise like the NCCL exchange inline,
   for one chunk, several chunks and every stage of the staged exchanger.
2. No deadlock: every rank but 0 queues a kernel that fills all SMs and spins
   until rank 0 releases it, right after its exchange, as a forward's MegaMoE
   or DCP collective spins on a peer. Rank 0 blocks its host on its side
   stream first, as a profiler stop or a drain sync does. The partner's swap
   needs only the spinning rank's pack, which runs before the spin on the
   current stream, so rank 0's exchange finishes. With the pack on the side
   stream (the parked version), it could queue behind the spin and rank 0
   would wait forever.
"""

import importlib.util
import os
import time

import pytest
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

NUM_RANKS = 4
TIMEOUT_S = 60.0

pytestmark = pytest.mark.skipif(
    not HAS_TRITON
    or not torch.cuda.is_available()
    or torch.accelerator.device_count() < NUM_RANKS,
    reason=f"needs {NUM_RANKS} GPUs and Triton",
)


if HAS_TRITON:

    @triton.jit
    def _spin_until_set(flag_ptr):
        while tl.load(flag_ptr, volatile=True) == 0:
            pass


def _bench_module():
    import vllm

    root = os.path.dirname(os.path.dirname(vllm.__file__))
    path = os.path.join(root, "benchmarks/kernels/benchmark_nixl_dcp_pair_exchange.py")
    spec = importlib.util.spec_from_file_location("pair_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_side_stream_peer_copy_is_exact_and_cannot_deadlock(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'init'}",),
        nprocs=NUM_RANKS,
        join=True,
    )


def _worker(rank: int, init_method: str) -> None:
    import torch.distributed as dist

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        dcp_pair_exchange as dpe,
    )

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=NUM_RANKS
    )
    try:
        comm = PyNcclCommunicator(dist.group.WORLD, device=dev)
        bench = _bench_module()
        block_size, chunk = 4096, 4
        # 24 units: 6 chunks, past the ring's PAIR_RING_SLOTS slots (a wrap).
        counts = (1, 3, 7, 18, 24)
        buf, table = bench.k31_cache(24, 0, block_size, max(counts) + 2)
        gen = torch.Generator(device=dev).manual_seed(1234 + rank)
        orig = torch.randint(
            0, 256, buf.shape, dtype=torch.uint8, device=dev, generator=gen
        )

        def staged(ring_slots=1):
            strides = dpe.all_exchange_strides(NUM_RANKS)
            first = dpe.DcpPairExchanger(
                table,
                dpe.pair_layout(rank, NUM_RANKS, strides[0], block_size, 1),
                dev,
                chunk_units=chunk,
                communicator=comm,
                ring_slots=ring_slots,
            )
            stages = {strides[0]: first}
            for s in strides[1:]:
                stages[s] = first.for_layout(
                    dpe.pair_layout(rank, NUM_RANKS, s, block_size, 1)
                )
            return dpe.DcpStagedExchanger(stages, NUM_RANKS)

        def units_of(n):
            units = [dpe.PairUnit(0, b + 1, True, NUM_RANKS, 1) for b in range(n)]
            if n > 1:
                # The last unit lacks one source (P blocks run out).
                units[-1] = dpe.PairUnit(0, n, False, NUM_RANKS - 1, 1)
            return units

        nccl = staged()
        # The served exchanger: chunk slots on the side stream.
        peer = staged(dpe.PAIR_RING_SLOTS)
        assert dpe.build_peer_copy_swap(peer, dist.group.WORLD) is not None
        side = torch.cuda.Stream(device=dev)

        # 1. Bitwise: inline NCCL vs side-stream peer copy through the ring,
        # each count twice in a row (the second starts mid-ring).
        for n in [c for c in counts for _ in range(2)]:
            buf.copy_(orig)
            nccl.exchange(units_of(n))
            torch.accelerator.synchronize()
            want = buf.clone()
            buf.copy_(orig)
            peer.exchange(units_of(n), side=side)
            torch.cuda.current_stream().wait_stream(side)
            torch.accelerator.synchronize()
            assert not torch.equal(want, orig), f"units {n}: nothing exchanged"
            assert torch.equal(buf, want), f"rank {rank} units {n}: not bitwise"
        dist.barrier()

        # 2. A spinning peer cannot starve the exchange.
        ptrs = CustomAllreduce.create_shared_buffer(8, dist.group.WORLD)
        flags = [
            torch.as_tensor(dpe._DeviceBuffer(p, (8,)), device=dev) for p in ptrs
        ]
        own_flag = flags[rank]
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        for it in range(10):
            own_flag.zero_()
            torch.accelerator.synchronize()
            dist.barrier()
            buf.copy_(orig)
            # Delay the pack's eligibility so the spin is queued right behind
            # it, as a forward follows the exchange.
            torch.cuda._sleep(20_000_000)
            peer.exchange(units_of(7), side=side)
            done = torch.cuda.Event()
            done.record(side)
            if rank != 0:
                _spin_until_set[(4 * sms,)](own_flag, num_warps=16)
                torch.cuda.current_stream().wait_stream(side)
                torch.accelerator.synchronize()
                ok = True
            else:
                start = time.monotonic()
                while not done.query() and time.monotonic() - start < TIMEOUT_S:
                    time.sleep(0.001)
                ok = done.query()
                for peer_rank in range(1, NUM_RANKS):
                    flags[peer_rank].fill_(1)
                torch.accelerator.synchronize()
            verdict = torch.tensor([int(ok)])
            dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
            assert verdict.item() == 1, (
                f"iteration {it}: rank 0's side-stream exchange did not finish "
                f"within {TIMEOUT_S:.0f} s while its peers spun"
            )
        dist.barrier()

        # 3. The DSTEP B hang shape: rank 0 traces kernels (CUPTI, which
        # serializes its GPU), queues a spin after its exchange that waits for
        # its peers' exchanges, and stops the profiler (a device sync). Its
        # pack precedes the spin on the current stream, so the peers' swaps
        # get it and release the spin.
        from torch.profiler import ProfilerActivity, profile

        for it in range(5):
            own_flag.zero_()
            torch.accelerator.synchronize()
            dist.barrier()
            buf.copy_(orig)
            if rank == 0:
                with profile(activities=[ProfilerActivity.CUDA]):
                    torch.cuda._sleep(20_000_000)
                    peer.exchange(units_of(7), side=side)
                    _spin_until_set[(4 * sms,)](own_flag, num_warps=16)
                    torch.cuda.current_stream().wait_stream(side)
                ok = True
            else:
                peer.exchange(units_of(7), side=side)
                done = torch.cuda.Event()
                done.record(side)
                start = time.monotonic()
                while not done.query() and time.monotonic() - start < TIMEOUT_S:
                    time.sleep(0.001)
                ok = done.query()
                flags[0].fill_(1)
                torch.accelerator.synchronize()
            verdict = torch.tensor([int(ok)])
            dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
            assert verdict.item() == 1, (
                f"profiled iteration {it}: a peer's side-stream exchange did not "
                f"finish within {TIMEOUT_S:.0f} s while rank 0 spun under CUPTI"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
