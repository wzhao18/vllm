# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The generic DCP top-k merge compiles nothing after JIT warmup, on any rank.

One process per GPU (DCP = TP = world size). Each rank runs the production
warmup registration (``register_dcp_candidate_merge_warmup``) through its own
``JitWarmupRegistry``, then the real ``_merge_dcp_topk_global`` (Triton pack,
DCP all-gather, CuTe DSL stable top-k) for the decode and prefill layouts and
both logits dtypes, under the JIT recorder.
Every rank's recorded compiles are all-gathered: the warmup is complete only
if it is complete on every rank (2026-11-12: TP16=DCP16 decode ranks 5..15
compiled the pack kernel while serving because its warmup key held the rank).
"""

import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.network_utils import get_open_port

TOPK = 2048
MAX_MODEL_LEN = 1 << 20

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="The DCP candidate merge needs CUDA and CuTe DSL",
)


def _check_no_compile_after_warmup(world: int, rank: int) -> None:
    from types import SimpleNamespace

    from vllm.distributed import get_dcp_group
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        register_dcp_candidate_merge_warmup,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _merge_dcp_topk_global,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.utils import jit_monitor

    group = get_dcp_group()
    assert group.rank_in_group == rank
    registry = JitWarmupRegistry(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=TOPK),
                max_model_len=MAX_MODEL_LEN,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=world, cp_kv_cache_interleave_size=1
            ),
        )
    )
    with registry.activate():
        register_dcp_candidate_merge_warmup()
    registry.warmup()

    dtypes = [torch.float32]
    if current_platform.is_device_capability_family(100):
        dtypes.append(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    width = 4096
    with jit_monitor.record_jit_events() as events:
        for rows in (1, 5):
            for dtype in dtypes:
                for use_row_starts in (False, True):
                    logits = torch.randn(
                        (rows, width + 8), generator=gen, device="cuda"
                    ).to(dtype)
                    topk = torch.stack(
                        [
                            torch.randperm(width, generator=gen, device="cuda")[:TOPK]
                            for _ in range(rows)
                        ]
                    ).to(torch.int32)
                    row_starts = (
                        torch.arange(rows, dtype=torch.int32, device="cuda")
                        if use_row_starts
                        else None
                    )
                    _merge_dcp_topk_global(
                        logits,
                        topk,
                        TOPK,
                        rank,
                        world,
                        1,
                        row_starts=row_starts,
                        score_bits=16 if dtype == torch.bfloat16 else 32,
                        num_positions=MAX_MODEL_LEN,
                    )
                    # Merged ids are global positions, identical on every rank.
                    every = group.all_gather(topk.unsqueeze(0), dim=0)
                    for r in range(world):
                        assert torch.equal(every[r], every[0]), f"rank {r} differs"
        torch.accelerator.synchronize()
    counts = group.all_gather(
        torch.tensor([len(events)], dtype=torch.int32, device="cuda"), dim=0
    )
    assert events == [], f"rank {rank} compiled after warmup: {events}"
    assert counts.tolist() == [0] * world, f"per-rank compiles: {counts.tolist()}"


def _worker(rank: int, world: int, init_method: str) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.accelerator.set_device_index(rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            _check_no_compile_after_warmup(world, rank)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(
            w,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < w, reason=f"Need {w} GPUs."
            ),
        )
        for w in (2, 4)
    ],
)
def test_dcp_candidate_merge_no_compile_after_warmup(world: int):
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, world, init_method)) for r in range(world)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 600
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world
