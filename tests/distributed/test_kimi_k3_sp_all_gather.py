# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 decoder-layer SP all-gather over symmetric memory
(``vllm/models/kimi_k3/nvidia/ops/sp_all_gather.py``). Without a buffer (a P/D
decode instance allocates none): the stock all-gather for every size, bit for
bit NCCL. With the buffer: bit for bit the NCCL all-gather for every shard size
up to the buffer's (row counts under and over the stock custom all-gather's
size, odd row counts, the buffer's largest shard), with the two slots
alternating over a forward's gathers and a gathered tensor intact until the
gather two later; forwards of an odd number of gathers back to back (the extra
barrier); a shard above the buffer, more gathers than a forward announced or
another width raise (on every rank, before any collective); a captured
two-gather forward replays between eager one-gather forwards."""

import queue
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

HIDDEN = 7168
MAX_TOKENS = 4096
# Rows per rank, odd counts among them. The custom all-gather's limit depends on
# the world and on MNNVL (2 MiB = 146 rows; MNNVL TP2 8 MiB = 585, TP4 4 MiB =
# 292), so the worker adds the first rows above it (``push_min``) and the
# largest shard of MAX_TOKENS.
ROWS = (1, 64, 146, 147, 333, 1023, 512, 1024)


def _shard(step, rank, rows, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000 * step + rank)
    return torch.randn(rows, HIDDEN, generator=gen).bfloat16().to(device)


def _worker(local_rank: int, world_size: int, port: int, q: mp.Queue):
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

        max_rows = MAX_TOKENS // world_size
        group = get_tp_group().device_group

        def reference(x):
            out = x.new_empty((world_size * x.shape[0], HIDDEN))
            dist.all_gather_into_tensor(out, x, group=group)
            return out

        # P/D decode instance: no buffer, the stock all-gather at every size
        # (above the push buffer's as well), outputs independent tensors.
        decode = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_consumer=True, is_kv_producer=False
            )
        )
        assert not sp_ag.init_sp_all_gather(decode, HIDDEN, torch.bfloat16)
        assert sp_ag._sp_all_gather is None
        sp_ag.begin_forward(2)
        outs = []
        for i, rows in enumerate((*ROWS, max_rows, 2 * max_rows + 1)):
            x = _shard(500 + i, local_rank, rows, device)
            out = sp_ag.kimi_sp_all_gather(x)
            ref = reference(x)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            outs.append((out, ref))
        for out, ref in outs:
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
        del outs

        if not sp_ag.init_sp_all_gather_buffer(max_rows, HIDDEN, torch.bfloat16):
            if local_rank == 0:
                q.put("no symmetric memory over the TP group")
            return

        ca_comm = get_tp_group().device_communicator.ca_comm

        def custom(rows, hidden=HIDDEN):
            """The stock custom all-gather takes this shard."""
            return ca_comm is not None and ca_comm.should_custom_all_gather(
                torch.empty((rows, hidden), dtype=torch.bfloat16, device="meta")
            )

        # The smallest shard on the push path.
        push_min = next(r for r in range(1, max_rows + 1) if not custom(r))
        assert push_min + 8 <= max_rows // 2, (push_min, max_rows)
        buffer_ptr = sp_ag._sp_all_gather.storage.untyped_storage().data_ptr()

        def check_forward(step0, sizes, announce=None):
            """One forward of gathers of ``sizes`` rows, each on its path (the
            push's output is a view of the buffer); gather g's output must
            survive gather g + 1."""
            sp_ag.begin_forward(len(sizes) if announce is None else announce)
            prev = None
            for i, rows in enumerate(sizes):
                x = _shard(step0 + i, local_rank, rows, device)
                out = sp_ag.kimi_sp_all_gather(x)
                on_buffer = out.untyped_storage().data_ptr() == buffer_ptr
                assert on_buffer == (not custom(rows)), (rows, on_buffer)
                if prev is not None:
                    torch.testing.assert_close(prev[0], prev[1], rtol=0, atol=0)
                ref = reference(x)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
                prev = (out, ref.clone())

        # Eager forwards: every size (custom and push paths), then odd-length
        # push forwards back to back (slot 0 ends one and starts the next).
        edge = (push_min - 1, push_min, push_min + 1)
        check_forward(0, (*ROWS, *(r for r in edge if r > 0), max_rows))
        for f in range(3):
            check_forward(100 + 10 * f, (push_min + 1, 1023 - f, max_rows))
        check_forward(200, (max_rows, push_min + 2))

        # Above the buffer (sized for max_num_batched_tokens, so a bug):
        # raises, without using a slot; the forward's next gather is intact.
        sp_ag.begin_forward(1)
        with pytest.raises(ValueError, match="exceeds the buffer"):
            sp_ag.kimi_sp_all_gather(_shard(300, local_rank, max_rows + 1, device))
        check_forward(310, (max_rows,), announce=1)
        # More push gathers than announced, or another width, raise (on every
        # rank before any collective).
        sp_ag.begin_forward(1)
        sp_ag.kimi_sp_all_gather(_shard(301, local_rank, push_min, device))
        with pytest.raises(AssertionError, match="announced 1"):
            sp_ag.kimi_sp_all_gather(_shard(302, local_rank, push_min, device))
        sp_ag.begin_forward(1)
        assert not custom(2 * push_min, HIDDEN // 2)
        with pytest.raises(ValueError, match="wide"):
            sp_ag.kimi_sp_all_gather(
                torch.zeros(
                    2 * push_min, HIDDEN // 2, device=device, dtype=torch.bfloat16
                )
            )

        # Captured two-gather forward, replayed between eager one-gather
        # forwards (odd: each starts with the extra barrier on slot 0).
        rows = push_min + 3
        static = _shard(99, local_rank, rows, device)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            sp_ag.begin_forward(2)
            g0 = sp_ag.kimi_sp_all_gather(static)
            g1 = sp_ag.kimi_sp_all_gather(static * 2)
            sums = g0.float().sum(1) + g1.float().sum(1)
        for rep in range(3):
            static.copy_(_shard(100 + rep, local_rank, rows, device))
            graph.replay()
            ref = reference(static).float().sum(1)
            ref += reference(static * 2).float().sum(1)
            torch.testing.assert_close(sums, ref, rtol=0, atol=0)
            check_forward(400 + rep, (push_min + 4 + rep,))
        torch.accelerator.synchronize()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("world_size", [2, 4])
def test_kimi_k3_sp_all_gather(world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(world_size, get_open_port(), q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
    if val is not None:
        pytest.skip(val)
