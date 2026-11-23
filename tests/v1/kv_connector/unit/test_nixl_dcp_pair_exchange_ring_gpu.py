# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange ring (PAIR_RING_SLOTS chunk slots, swap stream, side
stream) on the 4 GPUs of one host: randomised sequences of consecutive
exchanges whose slot state carries across calls and stages.

Each round queues one exchange of random distinct blocks: ratio 2 (P DCP 2),
ratio 4 (P DCP 1, staged, with random source counts so the stages' chunk
counts differ between ranks) or both in one call, with random chunk counts
around the ring size. Rounds follow each other as serving steps do: the next
exchange is ordered after the previous one's side stream on the device
(``finalize_loaded_kv``), with no host sync. After every round the cache must
equal, bitwise, the same rounds applied by the inline NCCL exchanger (the
one-slot reference the other suites check against the torch reference).
"""

import importlib.util
import os
import random
import time

import pytest
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

NUM_RANKS = 4

pytestmark = pytest.mark.skipif(
    not HAS_TRITON
    or not torch.cuda.is_available()
    or torch.accelerator.device_count() < NUM_RANKS,
    reason=f"needs {NUM_RANKS} GPUs and Triton",
)


TIMEOUT_S = 60.0

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


@pytest.mark.parametrize("launch_blocking", [False, True], ids=["async", "blocking"])
def test_ring_cannot_deadlock_behind_a_spinning_forward(tmp_path, launch_blocking):
    """Every rank but 0 queues an exchange and right after its packs a kernel
    that fills every SM and spins until rank 0 releases it (a forward's
    MegaMoE or DCP collective spinning on a peer). Rank 0 waits on the host
    for its own exchange, which needs the spinning ranks' packs (queued
    before the spin) and their copies of its send (copy engines on the swap
    stream, never behind their SM-bound unpacks). Cases: one ring slot set
    (4 chunks), an in-call wrap (7 chunks), and a staged ratio-4 exchange
    whose stages take different chunk counts on different ranks; with
    asynchronous launches and with CUDA_LAUNCH_BLOCKING=1."""
    import torch.multiprocessing as mp

    mp.spawn(
        _spin_worker,
        args=(f"file://{tmp_path / 'init'}", launch_blocking),
        nprocs=NUM_RANKS,
        join=True,
    )


def _spin_worker(rank: int, init_method: str, launch_blocking: bool) -> None:
    if launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
        block_size, chunk, num_blocks = 4096, 4, 32
        buf, table = bench.k31_cache(24, 0, block_size, num_blocks)
        strides = dpe.all_exchange_strides(NUM_RANKS)
        first = dpe.DcpPairExchanger(
            table,
            dpe.pair_layout(rank, NUM_RANKS, strides[0], block_size, 1),
            dev,
            chunk_units=chunk,
            communicator=comm,
            ring_slots=dpe.PAIR_RING_SLOTS,
        )
        stages = {strides[0]: first}
        for s in strides[1:]:
            stages[s] = first.for_layout(dpe.pair_layout(rank, NUM_RANKS, s, block_size, 1))
        ring = dpe.DcpStagedExchanger(stages, NUM_RANKS)
        assert dpe.build_peer_copy_swap(ring, dist.group.WORLD) is not None
        side = torch.cuda.Stream(device=dev)
        ptrs = CustomAllreduce.create_shared_buffer(8, dist.group.WORLD)
        flags = [torch.as_tensor(dpe._DeviceBuffer(p, (8,)), device=dev) for p in ptrs]
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        rng = random.Random(7)
        cases = {
            "one_ring": [dpe.PairUnit(0, b, True) for b in range(1, 17)],
            "in_call_wrap": [dpe.PairUnit(0, b, True) for b in range(1, 29)],
            "staged_unequal": [
                dpe.PairUnit(0, b, s > 1, s, 1)
                for b, s in zip(range(1, 25), [rng.randint(1, 4) for _ in range(24)])
            ],
        }
        for name, units in cases.items():
            for it in range(3):
                flags[rank].zero_()
                torch.accelerator.synchronize()
                dist.barrier()
                ring.exchange(units, side=side)
                done = torch.cuda.Event()
                done.record(side)
                if rank != 0:
                    _spin_until_set[(4 * sms,)](flags[rank], num_warps=16)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.accelerator.synchronize()
                    ok = True
                else:
                    start = time.monotonic()
                    while not done.query() and time.monotonic() - start < TIMEOUT_S:
                        time.sleep(0.001)
                    ok = done.query()
                    for peer in range(1, NUM_RANKS):
                        flags[peer].fill_(1)
                    torch.accelerator.synchronize()
                verdict = torch.tensor([int(ok)])
                dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
                assert verdict.item() == 1, (
                    f"{name} iteration {it}: rank 0's exchange did not finish "
                    f"within {TIMEOUT_S:.0f} s while its peers spun"
                )
            # Adversarial: EVERY rank holds all its SMs right after its packs
            # until every rank's swaps (the only exchange work a peer waits
            # for) are done. The swaps use no SM, so a correct design finishes;
            # any SM dependency on a peer-visible path hangs deterministically.
            swaps = first._ring.swap_stream(dev)
            release = torch.cuda.Stream(device=dev)
            one = torch.ones(8, dtype=torch.uint8).pin_memory()
            for it in range(3):
                flags[rank].zero_()
                torch.accelerator.synchronize()
                dist.barrier()
                ring.exchange(units, side=side)
                swapped = torch.cuda.Event()
                swapped.record(swaps)
                _spin_until_set[(4 * sms,)](flags[rank], num_warps=16)
                start = time.monotonic()
                while not swapped.query() and time.monotonic() - start < TIMEOUT_S:
                    time.sleep(0.001)
                verdict = torch.tensor([int(swapped.query())])
                dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
                # Release through a copy engine on another stream: the spin
                # holds every SM, so a fill kernel could not run.
                with torch.cuda.stream(release):
                    flags[rank].copy_(one, non_blocking=True)
                torch.cuda.current_stream().wait_stream(side)
                torch.accelerator.synchronize()
                assert verdict.item() == 1, (
                    f"{name} adversarial iteration {it}: swaps did not finish within "
                    f"{TIMEOUT_S:.0f} s while every rank held all its SMs"
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seed", range(8))
def test_random_consecutive_ring_exchanges_are_exact(tmp_path, seed: int):
    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'init'}", seed),
        nprocs=NUM_RANKS,
        join=True,
    )


def _worker(rank: int, init_method: str, seed: int) -> None:
    import torch.distributed as dist

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
        block_size, chunk, num_blocks = 4096, 4, 40
        buf, table = bench.k31_cache(24, 0, block_size, num_blocks)
        ref_buf, ref_table = bench.k31_cache(24, 0, block_size, num_blocks)
        gen = torch.Generator(device=dev).manual_seed(99 + rank)
        orig = torch.randint(
            0, 256, buf.shape, dtype=torch.uint8, device=dev, generator=gen
        )
        buf.copy_(orig)
        ref_buf.copy_(orig)

        def staged(tbl, ring_slots):
            strides = dpe.all_exchange_strides(NUM_RANKS)
            first = dpe.DcpPairExchanger(
                tbl,
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

        reference = staged(ref_table, 1)
        ring = staged(table, dpe.PAIR_RING_SLOTS)
        assert dpe.build_peer_copy_swap(ring, dist.group.WORLD) is not None
        side = torch.cuda.Stream(device=dev)
        # The same draws on every rank: every rank passes the same units.
        rng = random.Random(seed)
        max_units = dpe.PAIR_RING_SLOTS * chunk + 5  # past one ring of chunks
        # Odd seeds: rounds are the scheduler's step budget over random bursts
        # of ready plans (BaseScheduler._take_pair_budget: oldest first, at most
        # PAIR_RING_SLOTS chunks, a larger plan alone), so >16-unit bursts are
        # deferred across rounds; even seeds: free random rounds incl. wraps.
        from types import SimpleNamespace

        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
            NixlBaseConnectorScheduler,
        )

        budget = SimpleNamespace(
            _pair_ready={}, _pair_dcp_size=NUM_RANKS, _pair_chunk_units=chunk
        )

        def plan_units(n, kind):
            blocks = rng.sample(range(1, num_blocks), n)
            out = []
            for i, b in enumerate(blocks):
                if kind == "ratio2" or (kind == "mixed" and i % 2 == 0):
                    out.append(dpe.PairUnit(0, b, rng.random() < 0.8))
                else:
                    sources = rng.randint(1, NUM_RANKS)
                    out.append(dpe.PairUnit(0, b, sources > 1, sources, 1))
            return out

        def rounds():
            if seed % 2 == 0:
                for _ in range(12):
                    kind = rng.choice(("ratio2", "ratio4", "mixed"))
                    yield kind, plan_units(rng.randint(1, max_units), kind)
                return
            req = 0
            for _ in range(16):
                # A burst: plans become ready together (their blocks distinct
                # within a step: every step's units come from one sample).
                for _ in range(rng.choice((0, 1, 3, 5))):
                    req += 1
                    budget._pair_ready[f"r{req}"] = rng.randint(1, 12)
                taken = NixlBaseConnectorScheduler._take_pair_budget(
                    SimpleNamespace(
                        _pair_ready={
                            r: [dpe.PairUnit(0, 0, True)] * n
                            for r, n in budget._pair_ready.items()
                        },
                        _pair_dcp_size=NUM_RANKS // 2 * 2,
                        _pair_chunk_units=chunk,
                    )
                )
                for r in taken:
                    budget._pair_ready.pop(r)
                n = min(sum(len(v) for v in taken.values()), num_blocks - 1)
                if n:
                    kind = rng.choice(("ratio2", "mixed"))
                    yield kind, plan_units(n, kind)

        for round_, (kind, units) in enumerate(rounds()):
            n = len(units)
            reference.exchange(units)
            # As serving orders a step after the previous steps' side work.
            torch.cuda.current_stream().wait_stream(side)
            ring.exchange(units, side=side)
            check = torch.cuda.current_stream()
            check.wait_stream(side)
            torch.accelerator.synchronize()
            assert torch.equal(buf, ref_buf), (
                f"rank {rank} seed {seed} round {round_} ({kind}, {n} units): "
                "ring exchange differs from the inline reference"
            )
            assert not torch.equal(buf, orig)
        dist.barrier()
    finally:
        dist.destroy_process_group()
