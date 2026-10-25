# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefills with planned checkpoints.

With sparse retention and a backend that exports internal checkpoints, a prefill
chunk runs to the token budget and every retained Mamba state (the prompt's
replay boundary, a shared-prefix junction, multiples of the retention interval)
is exported as the chunk's internal checkpoint or kept as its end state. A chunk
ends early only when it holds two retained positions (or one it has no column
for).

These tests drive the real scheduler split and ``KVCacheManager``, play the
worker (copy-on-write copies, then each chunk's end state and its planned
checkpoint), and check that every cached Mamba block holds exactly the state
its key names, that every retained position is reusable, and that no prefill
takes more chunks than retaining its positions requires.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_group_id, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test

H = 16  # hash block (prefix-match unit)
B = 64  # Mamba block
MAMBA = 1  # Mamba KV cache group id


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _manager(retention_interval: int | None = 0, num_speculative_blocks: int = 0):
    config = KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=H, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=B,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_speculative_blocks,
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=H,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        config,
        max_model_len=8192,
        enable_caching=True,
        scheduler_block_size=B,
        hash_block_size=H,
        retention_interval=retention_interval,
    )
    assert manager.mamba_planned_checkpoints == (retention_interval is not None)
    return manager


def _split(manager, request, budget, num_local=0, blocks=None, first=False):
    """The real ``Scheduler._mamba_checkpoint_split`` on a stub scheduler with a
    token budget of ``budget``."""
    stub = SimpleNamespace(
        kv_cache_manager=manager,
        hash_block_size=H,
        cache_config=SimpleNamespace(block_size=B),
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    num_new_tokens = min(
        budget, request.num_tokens - request.num_computed_tokens - num_local
    )
    stub._mamba_grid_chunk_end = lambda *args: Scheduler._mamba_grid_chunk_end(
        stub, *args
    )
    return Scheduler._mamba_checkpoint_split(
        stub, request, num_new_tokens, num_local, 0, blocks, first
    )


def _min_chunks(start: int, end: int, positions: list[int], budget: int) -> int:
    """Fewest chunks covering ``[start, end)`` within the budget, each keeping at
    most two retained positions: one checkpoint inside it and its end state."""
    count = 0
    while start < end:
        stop = min(start + budget, end)
        inside = [p for p in positions if start < p < stop]
        start = inside[1] if len(inside) > 1 else stop
        count += 1
    return count


class _Worker:
    """Tracks which token position's Mamba state each block holds."""

    def __init__(self, manager):
        self.manager = manager
        self.mamba = manager.coordinator.single_type_managers[MAMBA]
        self.state: dict[int, int] = {}

    def load(self, request, num_loaded: int) -> None:
        """An async external hit: allocate the blocks the connector fills,
        the last of which then holds the state after ``num_loaded`` tokens."""
        manager = self.manager
        manager.get_computed_blocks(request)
        assert manager.allocate_slots(
            request, 0, num_external_computed_tokens=num_loaded, delay_cache_blocks=True
        )
        blocks = self.mamba.req_to_blocks[request.request_id]
        self.state[blocks[cdiv(num_loaded, B) - 1].block_id] = num_loaded
        manager.cache_blocks(request, num_loaded)
        request.num_computed_tokens = num_loaded
        manager.new_step_starts()

    def prefill(self, request, budget: int) -> list[tuple[int, int, object]]:
        """Run ``request``'s prefill to its last prompt token, one scheduler
        step per chunk; return the chunks (start, end, checkpoint). A request
        without computed tokens looks up the prefix cache first."""
        manager = self.manager
        hit_blocks, num_hit = None, 0
        if request.num_computed_tokens == 0:
            hit_blocks, num_hit, request.shared_prefix_boundary = (
                manager.get_computed_blocks(request)
            )
        first, chunks = True, []
        while request.num_computed_tokens + (num_hit if first else 0) < (
            request.num_tokens
        ):
            local = num_hit if first else 0
            start = request.num_computed_tokens + local
            blocks = hit_blocks if first else None
            n = _split(manager, request, budget, local, blocks, first)
            assert n > 0
            assert (
                manager.allocate_slots(
                    request,
                    n,
                    num_new_computed_tokens=local,
                    new_computed_blocks=blocks,
                    first_compute=first,
                )
                is not None
            )
            copies, retained = manager.take_kv_cache_block_copies()
            for copy in copies:
                self.state[copy.dst_block_id] = self.state.get(copy.src_block_id)
            manager.block_pool.free_blocks(retained)
            end = start + n
            mamba_blocks = self.mamba.req_to_blocks[request.request_id]
            self.state[mamba_blocks[cdiv(end, B) - 1].block_id] = end
            checkpoint = manager.mamba_checkpoint_plan(request.request_id)
            if checkpoint is not None:
                position, col = checkpoint
                assert start < position < end
                self.state[mamba_blocks[col].block_id] = position
            chunks.append((start, end, checkpoint))
            request.num_computed_tokens = end
            manager.new_step_starts()
            first = False
        return chunks

    def check_cache(self):
        """Every cached Mamba block holds the state its key names."""
        cache = self.manager.block_pool.cached_block_hash_to_block._cache
        for key, blocks in cache.items():
            if get_group_id(key) != MAMBA:
                continue
            for block in [blocks] if not isinstance(blocks, dict) else blocks.values():
                assert self.state.get(block.block_id) == block.block_hash_num_tokens

    def reusable(self, request, position: int) -> bool:
        key = request.block_hashes[position // H - 1]
        block = self.manager.block_pool.get_cached_block(key, [MAMBA])
        return block is not None and self.state.get(block[0].block_id) == position


def _run(worker, request, budget):
    hit = request.num_computed_tokens or worker.manager.get_computed_blocks(request)[1]
    chunks = worker.prefill(request, budget)
    positions = worker.manager.mamba_checkpoint_positions(request)
    worker.check_cache()
    for position in positions:
        if position > hit:
            assert worker.reusable(request, position), (position, chunks)
    assert len(chunks) == _min_chunks(hit, request.num_tokens, positions, budget), (
        chunks
    )
    return chunks


@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
@pytest.mark.parametrize("hit_offset", [H, 2 * H, B - H])
@pytest.mark.parametrize("continuation", [H + 3, B // 2, B - 5, B + 7, 2 * B + 9])
def test_short_continuation_of_local_hit_is_one_chunk(
    hit_offset, continuation, num_speculative_blocks
):
    """A mid-block local hit (copy-on-write) continued by less than, about, or
    more than a block: one chunk, its prompt checkpoint kept."""
    manager = _manager(num_speculative_blocks=num_speculative_blocks)
    worker = _Worker(manager)
    prefix = list(range(4 * B + hit_offset + 1))
    owner = make_request("owner", prefix, H, sha256)
    _run(worker, owner, budget=1024)
    request = make_request("r", prefix + [7] * continuation, H, sha256)
    chunks = _run(worker, request, budget=1024)
    assert len(chunks) == 1


@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
@pytest.mark.parametrize("loaded_offset", [H, B // 2, B - H])
@pytest.mark.parametrize("continuation", [H + 3, B // 2, B + 7, 2 * B + 9])
def test_short_continuation_of_external_load_is_one_chunk(
    loaded_offset, continuation, num_speculative_blocks
):
    """A mid-block async external load (its blocks private and unhashed)
    continued by a short prompt: one chunk."""
    manager = _manager(num_speculative_blocks=num_speculative_blocks)
    worker = _Worker(manager)
    loaded = 4 * B + loaded_offset
    request = make_request("r", list(range(loaded + continuation)), H, sha256)
    worker.load(request, loaded)
    assert len(_run(worker, request, budget=1024)) == 1


def test_block_aligned_hit_continued_within_a_block_is_one_chunk():
    """A hit on a block boundary (its block shared) continued within one block:
    the checkpoint takes a null column below."""
    manager = _manager()
    worker = _Worker(manager)
    prefix = list(range(4 * B + 1))
    _run(worker, make_request("owner", prefix, H, sha256), budget=1024)
    request = make_request("r", prefix + [7] * (B // 2), H, sha256)
    assert len(_run(worker, request, budget=1024)) == 1


@pytest.mark.parametrize("prompt_len", [3 * B + 5, 7 * B + 40, 20 * B + 1])
@pytest.mark.parametrize("budget", [B + H, 2 * B, 5 * B - H, 1024])
@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
def test_long_prefill_splits_only_at_the_budget(
    prompt_len, budget, num_speculative_blocks
):
    request = make_request("r", list(range(prompt_len)), H, sha256)
    _run(
        _Worker(_manager(num_speculative_blocks=num_speculative_blocks)),
        request,
        budget,
    )


def test_junction_and_prompt_boundary_in_one_chunk_split_once():
    """A sibling diverging from a cached prompt keeps the shared-prefix junction
    (checkpoint) and its own prompt boundary (end state): two chunks."""
    manager = _manager()
    worker = _Worker(manager)
    shared = list(range(6 * B + 3 * H))
    _run(worker, make_request("a", shared + [1] * 100, H, sha256), budget=1024)
    sibling = make_request("b", shared + [2] * 100, H, sha256)
    chunks = _run(worker, sibling, budget=1024)
    assert sibling.shared_prefix_boundary and len(chunks) == 2


@pytest.mark.parametrize("retention_interval", [2 * B, 4 * B])
@pytest.mark.parametrize("budget", [3 * B, 5 * B + H, 1024])
def test_retention_interval_states_are_all_kept(retention_interval, budget):
    """Every multiple of the interval is kept, as a checkpoint or an end state,
    with the fewest chunks that allows."""
    manager = _manager(retention_interval)
    request = make_request("r", list(range(17 * B + 21)), H, sha256)
    _run(_Worker(manager), request, budget)


def test_dense_retention_keeps_block_aligned_splits():
    assert not _manager(None).mamba_planned_checkpoints


@pytest.mark.parametrize("budget", [B + H, 2 * B + H, 5 * B - H, 16 * B])
@pytest.mark.parametrize("with_hit", [False, True])
def test_default_retention_never_splits_more_than_needed(budget, with_hit):
    """Default retention: across prompt lengths and hit offsets, every prefill
    takes exactly the fewest chunks that keep its retained positions."""
    for prompt_len in range(130, 1400, 41):
        worker = _Worker(_manager())
        tokens = list(range(prompt_len))
        if with_hit:
            owner = tokens[: prompt_len // 2 + 7]
            worker.prefill(make_request("owner", owner, H, sha256), budget)
            tokens = owner + [9] * (prompt_len - len(owner))
        _run(worker, make_request("r", tokens, H, sha256), budget)
