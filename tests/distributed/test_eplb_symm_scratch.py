# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU: sync EPLB through the symm_mem communicator with a borrowed scratch
span (no receive-buffer allocation), including spans smaller than one layer of
local experts (the rearrangement is staged in passes)."""

import pytest
import torch
import torch.distributed

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_communicator import (
    SymmetricScratch,
    create_eplb_communicator,
)
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
)
from vllm.platforms import current_platform

from .eplb_utils import distributed_run, set_env_vars_and_device
from .test_eplb_execute import (
    assert_verification_synced,
    create_expert_indices_with_redundancy,
    create_expert_weights,
    verify_expert_weights_after_shuffle,
)

GUARD = 4096  # sentinel bytes on each side of the lent span


def _test_symm_scratch(
    env, world_size, num_layers, num_local_experts, scratch_rows
) -> None:
    import torch.distributed._symmetric_memory as symm_mem

    set_env_vars_and_device(env)
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )
        group = get_tp_group()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(0)  # same placements on every rank
        physical = world_size * num_local_experts
        old = create_expert_indices_with_redundancy(
            num_layers, physical, physical, [1] * physical
        )
        new = create_expert_indices_with_redundancy(
            num_layers, physical, physical, [1] * physical
        )
        hidden_sizes = [64, 128]  # fp32 rows of 256 / 512 B: no alignment slack
        weights = create_expert_weights(
            num_layers, num_local_experts, hidden_sizes, rank, device, old
        )
        span = scratch_rows * sum(w[0].nbytes for w in weights[0])
        # A larger symmetric allocation lends [GUARD, GUARD + span) to EPLB; the
        # guards must stay untouched.
        arena = symm_mem.empty(span + 2 * GUARD, dtype=torch.uint8, device=device)
        arena.fill_(0xA5)
        handle = symm_mem.rendezvous(arena, group.device_group.group_name)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group.cpu_group)
        peers = [
            handle.get_buffer(r, arena.shape, torch.uint8)[GUARD : GUARD + span]
            for r in range(world_size)
        ]
        scratch = SymmetricScratch(
            local=arena[GUARD : GUARD + span],
            peers=peers,
            ranks=list(group.ranks),
            owner="test arena",
        )
        communicator = create_eplb_communicator(
            group, "symm_mem", weights, None, scratch=scratch
        )
        assert communicator.buffer_rows == min(scratch_rows, num_local_experts)
        rearrange_expert_weights_inplace(
            old,
            new,
            weights,
            communicator.owned_expert_buffer,
            group.device_group,
            communicator,
        )
        torch.cuda.synchronize()
        guards_ok = bool(
            (arena[:GUARD] == 0xA5).all() and (arena[GUARD + span :] == 0xA5).all()
        )
    ok = guards_ok and verify_expert_weights_after_shuffle(
        weights, new, hidden_sizes, rank, num_local_experts
    )
    assert_verification_synced(
        ok,
        "symm_mem scratch EPLB: weights wrong or bytes outside the span written.",
        cpu_group=group.cpu_group,
    )


@pytest.mark.parametrize(
    "world_size,num_layers,num_local_experts,scratch_rows",
    [
        (2, 2, 8, 2),  # passes of <= 2 staged rows per rank
        (4, 3, 8, 3),
        (4, 2, 4, 8),  # span holds a whole layer: one pass per layer
    ],
)
def test_symm_scratch_rearrangement(
    world_size, num_layers, num_local_experts, scratch_rows
):
    if not current_platform.is_cuda():
        pytest.skip("symm_mem EPLB requires CUDA")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(
        _test_symm_scratch, world_size, num_layers, num_local_experts, scratch_rows
    )
