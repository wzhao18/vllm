# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, TP8 = DCP8 (DFn) table extensions over 8 ranks on two
4-GPU nodes of one NVLink domain.

- ``test_tp8_ring_embedding``: ``runtime.decode.RING_HEAD[8]``: the vocabulary-sharded
  embedding gather publishing each token row from its owner rank into the one-slot
  [cap, 7168] mailbox the first AttnRes polls (``RingEmbeddingGather``), at 4 / 64 /
  256 rows with out-of-vocabulary ids: every rank's mailbox rows bitwise equal to the
  stock lookup (``F.embedding`` of the whole table: one owner per row, the stock
  all-reduce adds zeros), and its CUDA-graph replay bitwise the same.

Launch as tests/distributed/test_kimi_k3_sdec_tp8_wide.py: one pytest per node over
two nodes (``srun -N2 --ntasks-per-node=1``), MASTER_ADDR / MASTER_PORT set; skipped
elsewhere."""

import gc
import os
import time

import pytest
import torch
import torch.distributed as dist

LOCAL, NODES = 4, 2
WORLD = LOCAL * NODES
VOCAB, H, CAP = 8 * 4096, 7168, 256
TIMEOUT = 600


def _launch():
    if os.environ.get("SLURM_NNODES") != str(NODES):
        return None
    if "SLURM_NODEID" not in os.environ or "MASTER_ADDR" not in os.environ:
        return None
    port = int(os.environ.get("MASTER_PORT", "29571"))
    return int(os.environ["SLURM_NODEID"]), os.environ["MASTER_ADDR"], port


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < LOCAL:
        return False
    if {torch.cuda.get_device_capability(i) for i in range(LOCAL)} != {(10, 7)}:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        _launch() is None or not _rubin_node(),
        reason="needs 2 nodes x 4 Rubin (SM107) GPUs of one NVLink domain, "
        "launched with srun -N2 and MASTER_ADDR",
    ),
]


def _ring_embedding(group, dev, rank):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.embedding import (  # noqa: E501
        compile_ring_gather,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        architecture,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.prepared import (
        OprojMailbox,
    )

    gen = torch.Generator(device=dev).manual_seed(5)  # the whole table, every rank
    table = torch.randn(VOCAB, H, generator=gen, device=dev).to(torch.bfloat16)
    table[3] = -0.0  # a row of negative zeros (the mailbox sentinel's halves)
    shard = VOCAB // WORLD
    weight = table[rank * shard : (rank + 1) * shard].contiguous()
    box = OprojMailbox(group, dev, WORLD, rank, (1, 1, CAP, H))
    ring = box.buffer[0, 0]
    gather = compile_ring_gather(torch.int64, shard, architecture(dev))
    for rows in (4, 64, 256):
        ids = torch.randint(0, VOCAB, (rows,), generator=gen, device=dev)
        ids[0], ids[-1] = 3, VOCAB + 5  # the -0 row; an id no shard holds
        ref = torch.nn.functional.embedding(ids.clamp(max=VOCAB - 1), table)
        ref[-1] = 0
        ref = torch.where(ref == 0, torch.zeros_like(ref), ref)  # stock: -0 + 0 = +0
        ring[:rows].view(torch.int16).fill_(-1)
        torch.accelerator.synchronize()
        dist.barrier()
        gather(ids, weight, rank * shard, VOCAB, box.mc)
        torch.accelerator.synchronize()
        dist.barrier()
        assert torch.equal(ring[:rows], ref), rows
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        gc.collect()
        gc.disable()
        try:
            with torch.cuda.stream(side), torch.cuda.graph(
                graph, stream=side, capture_error_mode="thread_local"
            ):
                gather(ids, weight, rank * shard, VOCAB, box.mc)
        finally:
            gc.enable()
        for _ in range(2):
            ring[:rows].view(torch.int16).fill_(-1)
            torch.accelerator.synchronize()
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            dist.barrier()
            assert torch.equal(ring[:rows], ref), rows
        del graph


CASES = {"ring_embedding": _ring_embedding}


def _worker(local, node, addr, port, case, result):
    rank = node * LOCAL + local
    torch.accelerator.set_device_index(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{addr}:{port}",
        rank=rank,
        world_size=WORLD,
        device_id=dev,
    )
    group = dist.group.WORLD
    try:  # (older torch: symmetric memory is enabled per group)
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    CASES[case](group, dev, rank)
    dist.barrier()
    dist.destroy_process_group()
    result.put((rank, "ok"))


def _spawn(case):
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    node, addr, port = _launch()
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(i, node, addr, port, case, result))
        for i in range(LOCAL)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT
    done = []
    while len(done) < LOCAL:
        if time.monotonic() > deadline:
            for p in procs:
                p.kill()
            pytest.fail(f"{case}: no result in {TIMEOUT} s (a hang?)")
        if any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            pytest.fail(f"{case}: a rank failed (see its traceback)")
        try:
            done.append(result.get(timeout=5))
        except Exception:
            continue
    for p in procs:
        p.join(timeout=60)
    assert sorted(r for r, _ in done) == [node * LOCAL + i for i in range(LOCAL)]


def test_tp8_ring_embedding():
    """RING_HEAD[8]: the ring embedding gather at TP8 (module docstring)."""
    _spawn("ring_embedding")
