# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the sampled verification's row exchange
(``verify.exchange.PreparedExchange``) above its mailbox's MAX_ROWS row slots. Rows
past MAX_ROWS run in launches of at most MAX_ROWS rows; each launch advances its row
slots' parities, so steps of varying row counts (more than 512 rows, exactly 512, a
few rows) leave different rows on different parities, and a graph-captured step
replays across them. Every rank must gather every rank's entries, bit for bit, at
every row of every step."""

import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

WORDS = 8
# Row counts per step: chunked steps (> MAX_ROWS) between single launches, so the
# per-row parities of rows 0-511 and of the chunks' rows alternate unevenly.
STEPS = (1100, 513, 512, 1100, 7, 2048, 1025, 3, 1536)
GRAPH_ROWS, REPLAYS = 1100, 4


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _entries(step, rank, rows, device):
    """Rank ``rank``'s entries [rows, WORDS] u32 at ``step``: every word nonzero and
    distinct per (step, row, rank, word)."""
    row = torch.arange(rows, device=device, dtype=torch.int64).view(-1, 1)
    word = torch.arange(WORDS, device=device, dtype=torch.int64).view(1, -1)
    value = (((step * 4096 + row) * 32 + rank) * WORDS + word) % (1 << 31) + 1
    return value.to(torch.int32).view(torch.uint32).contiguous()


def _expected(step, world, rows, device):
    """Every rank's entries in rank order [rows, world, WORDS]."""
    return torch.stack([_entries(step, r, rows, device) for r in range(world)], 1)


def _check(gathered, step, world, rows):
    expected = _expected(step, world, rows, gathered.device)
    assert gathered.shape == (rows, world, WORDS), gathered.shape
    same = gathered.view(torch.int32) == expected.view(torch.int32)
    bad = (~same).any(-1).any(-1).nonzero().flatten()
    assert bad.numel() == 0, (step, rows, bad[:8].tolist())


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import (
            exchange,
        )

        assert max(STEPS) > exchange.MAX_ROWS == 512
        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.group.WORLD
        try:  # (older torch: symmetric memory is enabled per group)
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        prepared = exchange.PreparedExchange(world, rank, group, WORDS, device)
        prepared.prepare()
        for step, rows in enumerate(STEPS):
            gathered = prepared.run(_entries(step, rank, rows, device))
            torch.accelerator.synchronize(device)
            _check(gathered, step, world, rows)
        # One captured chunked step, replayed with new entries between eager steps of
        # other row counts (the graph's launches keep reading the per-row ordinals).
        entries = _entries(0, rank, GRAPH_ROWS, device)
        gathered = torch.empty(
            (GRAPH_ROWS, world, WORDS), dtype=torch.uint32, device=device
        )
        prepared.run(entries, gathered)  # (warm: every rank, the same launches)
        torch.accelerator.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepared.run(entries, gathered)
        base = len(STEPS) + 1
        for replay in range(REPLAYS):
            step = base + 2 * replay
            entries.copy_(_entries(step, rank, GRAPH_ROWS, device))
            graph.replay()
            torch.accelerator.synchronize(device)
            _check(gathered, step, world, GRAPH_ROWS)
            other = 513 if replay % 2 else 9
            eager = prepared.run(_entries(step + 1, rank, other, device))
            torch.accelerator.synchronize(device)
            _check(eager, step + 1, world, other)
        dist.barrier()
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
    # One deadline for the group: a hung rank fails the test, never CI.
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert [process.exitcode for process in processes] == [0] * world_size


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or not _has_multicast_support(),
    reason="Need 2 GPUs with symmetric-memory multicast.",
)
def test_row_exchange_chunks_two_ranks():
    """PreparedExchange.run on 2 ranks above MAX_ROWS rows: every rank gathers both
    ranks' entries bit for bit at every row of steps of 1100, 513, 512, 7, 2048,
    1025, 3 and 1536 rows (chunked launches between single ones: alternating and
    uneven row parities), and a captured 1100-row step replayed between eager
    steps of 9 and 513 rows."""
    pytest.importorskip("cutlass")
    _distributed_run(_worker, world_size=2)
