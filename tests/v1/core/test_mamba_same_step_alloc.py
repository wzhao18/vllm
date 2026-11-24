# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""allocate_slots must never take more blocks than its free-block check
counted. A same-step connector load allocates the external hit and the step's
new tokens in one call; in Mamba "align" mode the hit's state block is taken by
allocate_external_computed_blocks apart from the blocks allocate_new_blocks
caps, so a chunk that leaves the hit's block needs one more block than a
first-allocation count that ignored it (real P/D: "Cannot get 1 free blocks
from the pool" with the pool nearly full)."""

import pytest

from tests.v1.core.test_mamba_resume_checkpoint import MAMBA_BLOCK, _manager
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = pytest.mark.cpu_test

HASH = 16


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _counted_and_taken(num_external: int, prompt_len: int) -> tuple[int, int]:
    manager = _manager(0)
    counted: list[int] = []
    orig = manager.coordinator.get_num_blocks_to_allocate

    def spy(*a, **kw):
        n = orig(*a, **kw)
        counted.append(n)
        return n

    manager.coordinator.get_num_blocks_to_allocate = spy
    request = make_request("r", list(range(prompt_len)), HASH, sha256)
    free = manager.block_pool.get_num_free_blocks()
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=prompt_len - num_external,
        num_external_computed_tokens=num_external,
        delay_cache_blocks=True,
    )
    assert blocks is not None
    return counted[-1], free - manager.block_pool.get_num_free_blocks()


@pytest.mark.parametrize(
    "num_external,prompt_len",
    [
        (192, 300),  # hit mid-block 1, the step ends in block 2
        (128, 300),  # hit on a block boundary, the step ends in block 2
        (192, 250),  # the step stays in the hit's block
        (64, 200),
    ],
)
def test_same_step_load_allocation_is_counted(num_external, prompt_len):
    counted, taken = _counted_and_taken(num_external, prompt_len)
    assert taken <= counted, (MAMBA_BLOCK, num_external, prompt_len, counted, taken)


@pytest.mark.parametrize("num_external,prompt_len", [(192, 300), (128, 300), (64, 200)])
def test_async_load_then_compute_allocations_are_counted(num_external, prompt_len):
    """The async-load path (hit first, compute in a later call) as a control."""
    manager = _manager(0)
    counted: list[int] = []
    orig = manager.coordinator.get_num_blocks_to_allocate

    def spy(*a, **kw):
        n = orig(*a, **kw)
        counted.append(n)
        return n

    manager.coordinator.get_num_blocks_to_allocate = spy
    request = make_request("r", list(range(prompt_len)), HASH, sha256)
    free = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(
        request, 0, num_external_computed_tokens=num_external, delay_cache_blocks=True
    )
    taken = free - manager.block_pool.get_num_free_blocks()
    assert taken <= counted[-1]
    request.num_computed_tokens = num_external
    manager.cache_blocks(request, num_external)
    free = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(request, prompt_len - num_external)
    assert free - manager.block_pool.get_num_free_blocks() <= counted[-1]
