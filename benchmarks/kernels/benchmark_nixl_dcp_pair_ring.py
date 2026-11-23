# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main-stream occupancy of the DCP pair exchange with peer-copy swaps on a
side stream, one chunk slot (each pack waits for the previous chunk's swap and
unpack) vs ``PAIR_RING_SLOTS`` slots, on the 4 GPUs of one host
(K3.1 cache geometry, 4096-row blocks, DCP4 from P DCP2).

Occupancy = time from an event on the current stream before ``exchange`` to an
event after it, i.e. how long the exchange holds the stream a step's forward
queues behind. Also checks that both versions rewrite the cache bitwise alike.

    python benchmarks/kernels/benchmark_nixl_dcp_pair_ring.py --units 4 8 16 20 36
"""

import importlib.util
import os
import statistics

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser

NUM_RANKS = 4


def _bench_module():
    path = os.path.join(os.path.dirname(__file__), "benchmark_nixl_dcp_pair_exchange.py")
    spec = importlib.util.spec_from_file_location("pair_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worker(rank: int, init_method: str, args) -> None:
    import torch.distributed as dist

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        dcp_pair_exchange as dpe,
    )

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=init_method, rank=rank,
                            world_size=NUM_RANKS)  # fmt: skip
    try:
        comm = PyNcclCommunicator(dist.group.WORLD, device=dev)
        bench = _bench_module()
        block_size, chunk = 4096, args.chunk_units
        num_blocks = max(args.units) + 2
        buf, table = bench.k31_cache(args.num_dsa, args.num_draft, block_size, num_blocks)
        gen = torch.Generator(device=dev).manual_seed(4321 + rank)
        orig = torch.randint(0, 256, buf.shape, dtype=torch.uint8, device=dev,
                             generator=gen)  # fmt: skip

        def build(slots):
            strides = dpe.all_exchange_strides(NUM_RANKS)
            first = dpe.DcpPairExchanger(
                table, dpe.pair_layout(rank, NUM_RANKS, strides[0], block_size, 1),
                dev, chunk_units=chunk, communicator=comm, ring_slots=slots,
            )  # fmt: skip
            stages = {strides[0]: first}
            for s in strides[1:]:
                stages[s] = first.for_layout(
                    dpe.pair_layout(rank, NUM_RANKS, s, block_size, 1)
                )
            ex = dpe.DcpStagedExchanger(stages, NUM_RANKS)
            assert dpe.build_peer_copy_swap(ex, dist.group.WORLD) is not None
            return ex

        side = torch.cuda.Stream(device=dev)
        arms = {1: build(1), dpe.PAIR_RING_SLOTS: build(dpe.PAIR_RING_SLOTS)}
        rows = []
        for n in args.units:
            # Ratio 2 (P DCP 2 at D DCP 4): one stage.
            units = [dpe.PairUnit(0, b + 1, True) for b in range(n)]
            want = None
            res = {}
            for slots, ex in arms.items():
                buf.copy_(orig)
                ex.exchange(units, side=side)
                torch.cuda.current_stream().wait_stream(side)
                torch.accelerator.synchronize()
                got = buf.clone()
                if want is None:
                    want = got
                else:
                    assert torch.equal(got, want), f"rank {rank} units {n}: not bitwise"
                times = []
                for _ in range(args.iters):
                    buf.copy_(orig)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(2_000_000)  # queue the exchange first
                    start.record()
                    ex.exchange(units, side=side)
                    end.record()
                    torch.cuda.current_stream().wait_stream(side)
                    torch.accelerator.synchronize()
                    times.append(start.elapsed_time(end))
                res[slots] = statistics.median(times)
            rows.append((n, res[1], res[dpe.PAIR_RING_SLOTS]))
        if rank == 0:
            print(f"units  occupancy 1 slot (ms)  occupancy {dpe.PAIR_RING_SLOTS} slots (ms)  bitwise")
            for n, a, b in rows:
                print(f"{n:5d}  {a:22.3f}  {b:22.3f}  equal")
    finally:
        dist.destroy_process_group()


def main(args) -> None:
    import tempfile

    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/init", args), nprocs=NUM_RANKS, join=True)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, nargs="+", default=[4, 8, 16, 20, 36])
    parser.add_argument("--chunk-units", type=int, default=4)
    parser.add_argument("--num-dsa", type=int, default=24)
    parser.add_argument("--num-draft", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    main(parser.parse_args())
