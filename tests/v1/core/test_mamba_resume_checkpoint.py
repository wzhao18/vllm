# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefills resuming mid-block with an internal checkpoint.

A request that resumes from a partial (sub-block) prefix hit and reaches the
next block boundary used to run its final chunk in up to three steps:
start -> next block boundary -> prompt checkpoint -> prompt end. The checkpoint
now reuses the column of the mid-block state the request resumes from, so the
final chunk runs in one step.

These tests drive the real scheduler split, KV cache manager and worker
checkpoint planner, and track which state every Mamba block holds the way the
worker writes it: CoW copies, then the align pre-copy into the running slot,
then the forward's running state and checkpoint export. Every block a hash
claims, and every boundary state handed to a connector, must hold the state
at exactly the claimed token count.
"""

from math import lcm
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_group_id, init_none_hash
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    is_mamba_prefill_checkpoint_valid,
)

pytestmark = pytest.mark.cpu_test

HASH = 16
MAMBA_BLOCK = 128
ALIGNMENT = 16
MAMBA_GID = 1


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _manager(
    num_speculative_blocks: int,
    retention_interval: int | None = None,
    shared_prefix_checkpoint: bool = False,
):
    kv_cache_config = KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=HASH, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
                # Shared-prefix checkpoints need an EAGLE drafter's group; with
                # lookahead hashes it takes no last-block drop.
                is_eagle_group=shared_prefix_checkpoint,
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=MAMBA_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_speculative_blocks,
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=ALIGNMENT,
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        scheduler_block_size=lcm(HASH, MAMBA_BLOCK),
        hash_block_size=HASH,
        retention_interval=retention_interval,
        enable_mamba_shared_prefix_checkpoint=shared_prefix_checkpoint,
        num_hash_lookahead_tokens=int(shared_prefix_checkpoint),
        num_prefill_lookahead=int(shared_prefix_checkpoint),
    )


def _split(
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    shared_prefix_checkpoint: bool = False,
):
    stub = SimpleNamespace(
        block_size=MAMBA_BLOCK,
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=int(shared_prefix_checkpoint),
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=shared_prefix_checkpoint,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ALIGNMENT,
    )
    return Scheduler._mamba_block_aligned_split(
        stub, request, num_new_tokens, num_new_local_computed_tokens
    )


class _Worker:
    """Which state each Mamba block holds, written the way the worker does."""

    def __init__(self, manager, num_speculative_blocks: int):
        self.manager = manager
        self.mamba = manager.coordinator.single_type_managers[MAMBA_GID]
        self.num_spec = num_speculative_blocks
        self.state_at: dict[int, int] = {}
        self.tables: dict[str, list[int]] = {}
        self.state_col: dict[str, int] = {}

    def load(self, request, num_tokens: int) -> None:
        """A connector load writes the state at ``num_tokens`` into its slot."""
        blocks = self.mamba.req_to_blocks[request.request_id]
        self.state_at[blocks[cdiv(num_tokens, MAMBA_BLOCK) - 1].block_id] = num_tokens

    def step(self, request, start: int, end: int, new_blocks) -> None:
        req_id = request.request_id
        if req_id not in self.tables:
            # A newly scheduled request ships its whole block table.
            self.tables[req_id] = [
                b.block_id for b in self.mamba.req_to_blocks[req_id]
            ]
        else:
            # A running request's worker table is append-only.
            self.tables[req_id].extend(new_blocks.get_block_ids()[MAMBA_GID])
        table = self.tables[req_id]
        # Partial-hit CoW copies run first (`_update_states`).
        copies, retained = self.manager.take_kv_cache_block_copies()
        for copy in copies:
            if copy.src_block_id in self.state_at:
                self.state_at[copy.dst_block_id] = self.state_at[copy.src_block_id]
        self.manager.block_pool.free_blocks(retained)
        # Align pre-copy of the previous state into the running slot.
        prev_col = self.state_col.get(req_id, (start - 1) // MAMBA_BLOCK)
        running_col = cdiv(end, MAMBA_BLOCK) - 1
        self.state_col[req_id] = running_col
        if prev_col >= 0 and prev_col != running_col:
            self.state_at[table[running_col]] = self.state_at.get(table[prev_col], -1)
        # Forward: running state and the internal checkpoint export.
        self.state_at[table[running_col]] = end
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end],
            [end - start],
            hash_block_size=HASH,
            mamba_block_size=MAMBA_BLOCK,
            checkpoint_alignment=ALIGNMENT,
            drop_eagle_block=False,
            cache_hit_alignment_tokens=self.mamba.cache_hit_alignment_tokens,
            num_hash_lookahead_tokens=self.manager.block_pool.num_hash_lookahead_tokens,
        )
        if offsets[0]:
            target = table[cols[0]]
            assert target != self.manager.block_pool.null_block.block_id or (
                self.mamba._checkpoints.get(req_id) is None
            )
            # The export never targets the running slot.
            assert cols[0] != running_col
            self.state_at[target] = start + offsets[0]

    def check(self) -> int:
        """Every hashed Mamba block and offered boundary holds its state."""
        checked = 0
        for block in self.manager.block_pool.blocks:
            if block.is_null or block.block_hash is None:
                continue
            if get_group_id(block.block_hash) != MAMBA_GID:
                continue
            claimed = block.block_hash_num_tokens
            assert self.state_at.get(block.block_id) == claimed, (
                f"block {block.block_id} is hashed as state@{claimed} but holds "
                f"state@{self.state_at.get(block.block_id)}"
            )
            checked += 1
        offloads = self.mamba.take_pending_boundary_state_offloads()
        for _, group_id, block, tokens in offloads:
            assert group_id == MAMBA_GID
            assert self.state_at.get(block.block_id) == tokens, (
                f"boundary offload of block {block.block_id} as state@{tokens} "
                f"holds state@{self.state_at.get(block.block_id)}"
            )
        return checked


def _prefill(
    manager,
    worker,
    request,
    num_local,
    local_blocks,
    cut_at: int = 0,
    reclamp: bool = True,
    shared_prefix_checkpoint: bool = False,
):
    """Schedule the rest of ``request``'s prompt; returns the chunk ends.

    ``cut_at`` cuts the first chunk there after the split, as an encoder input
    that does not fit the encoder budget does. ``reclamp`` applies the
    scheduler's re-clamp that follows those caps.
    """
    ends = []
    while request.num_computed_tokens + num_local < request.num_tokens:
        start = request.num_computed_tokens + num_local
        num_new = _split(
            request,
            request.num_tokens - start,
            num_local,
            shared_prefix_checkpoint=shared_prefix_checkpoint,
        )
        if start < cut_at < start + num_new:
            num_new = cut_at - start
            cut_at = 0
            if reclamp:
                num_new = scheduler_module._stop_mamba_chunk_at_block_end(
                    request, start, num_new, MAMBA_BLOCK
                )
        assert num_new > 0
        new_blocks = manager.allocate_slots(
            request, num_new, num_local, local_blocks
        )
        assert new_blocks is not None
        worker.step(request, start, start + num_new, new_blocks)
        request.num_computed_tokens = start + num_new
        num_local, local_blocks = 0, None
        worker.check()
        manager.new_step_starts()
        ends.append(request.num_computed_tokens)
    return ends


def _produce(manager, worker, num_tokens: int):
    producer = make_request("producer", list(range(num_tokens)), HASH, sha256)
    _prefill(manager, worker, producer, 0, None)
    manager.free(producer)
    manager.new_step_starts()
    return producer


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize(
    "prompt_len",
    [
        # Resumes at 192 (block 1, mid-block). Ends in block 2: was 3 steps.
        # 260: the checkpoint is the next block boundary itself, 256, so the
        # full-block key is already the right one.
        260,
        300,
        330,
        384,
        # Ends two or more blocks later: was one step, now reuses the column.
        450,
        700,
    ],
)
def test_partial_hit_final_chunk_runs_in_one_step(
    num_speculative_blocks, prompt_len, retention_interval
):
    manager = _manager(num_speculative_blocks, retention_interval)
    worker = _Worker(manager, num_speculative_blocks)
    _produce(manager, worker, 200)

    consumer = make_request(
        "consumer", list(range(200)) + list(range(10000, 10000 + prompt_len - 200)),
        HASH, sha256,
    )
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    assert num_local == 192 and num_local % MAMBA_BLOCK != 0
    ends = _prefill(manager, worker, consumer, num_local, local_blocks)
    assert ends == [prompt_len]

    # The next turn resumes at the prompt checkpoint.
    manager.free(consumer)
    manager.new_step_starts()
    nxt = make_request(
        "next",
        list(consumer.all_token_ids) + list(range(20000, 20100)),
        HASH,
        sha256,
    )
    _, num_hit, _ = manager.get_computed_blocks(nxt)
    # A block-aligned prompt end also publishes its full block, unless sparse
    # retention keeps only the prompt checkpoint.
    assert num_hit == (
        prompt_len
        if prompt_len % MAMBA_BLOCK == 0 and retention_interval is None
        else (prompt_len - 1) // HASH * HASH
    )


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize("prompt_len", [300, 450])
def test_connector_hit_final_chunk_runs_in_one_step(num_speculative_blocks, prompt_len):
    manager = _manager(num_speculative_blocks)
    worker = _Worker(manager, num_speculative_blocks)
    request = make_request("ext", list(range(prompt_len)), HASH, sha256)
    loaded = manager.allocate_slots(
        request, num_new_tokens=0, num_external_computed_tokens=192,
        delay_cache_blocks=True,
    )
    assert loaded is not None
    worker.load(request, 192)
    request.num_computed_tokens = 192
    manager.cache_blocks(request, 192)
    ends = _prefill(manager, worker, request, 0, None)
    assert ends == [prompt_len]
    assert worker.check() > 0


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
def test_running_mid_block_resume_reuses_its_slot(num_speculative_blocks):
    """A running request stopped mid-block (here forced by a 192 budget)."""
    manager = _manager(num_speculative_blocks)
    worker = _Worker(manager, num_speculative_blocks)
    request = make_request("run", list(range(300)), HASH, sha256)
    first = manager.allocate_slots(request, 192)
    assert first is not None
    worker.step(request, 0, 192, first)
    request.num_computed_tokens = 192
    worker.check()
    manager.new_step_starts()
    assert _prefill(manager, worker, request, 0, None) == [300]


@pytest.mark.parametrize(
    ("query_start", "query_end", "checkpoint", "expected"),
    [
        # Mid-block start, ends in the next block: reuses the initial column.
        (192, 300, 288, True),
        # Mid-block start, ends in the same block: the running slot is the
        # initial column, so there is no column to reuse.
        (192, 250, 240, False),
        # Block-aligned start: the initial column is a cacheable boundary.
        (256, 300, 288, False),
        (256, 400, 384, True),
        # Unchanged conditions.
        (200, 450, 448, False),
        (192, 210, 208, False),
    ],
)
def test_mamba_prefill_checkpoint_valid_mid_block(
    query_start, query_end, checkpoint, expected
):
    assert (
        is_mamba_prefill_checkpoint_valid(
            query_start=query_start,
            query_end=query_end,
            checkpoint_position=checkpoint,
            hash_block_size=HASH,
            mamba_block_size=MAMBA_BLOCK,
            checkpoint_alignment=ALIGNMENT,
        )
        is expected
    )


def _consumer(prompt_len: int, request_id: str = "consumer"):
    return make_request(
        request_id,
        list(range(200)) + list(range(10000, 10000 + prompt_len - 200)),
        HASH,
        sha256,
    )


def _checkpoint_block(manager, request):
    """The resumed state's column, where a mid-block start exports."""
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    return mamba.req_to_blocks[request.request_id][(192 - 1) // MAMBA_BLOCK]


def test_c_at_next_block_boundary_keeps_the_full_block_key():
    """c == (K + 1) * B: the full-block key cache_full_blocks gives column K is
    already the checkpoint's, so the re-key returns early."""
    manager = _manager(0)
    worker = _Worker(manager, 0)
    _produce(manager, worker, 200)
    consumer = _consumer(260)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    assert _prefill(manager, worker, consumer, num_local, local_blocks) == [260]
    block = _checkpoint_block(manager, consumer)
    assert block.block_hash_num_tokens == 2 * MAMBA_BLOCK
    assert worker.state_at[block.block_id] == 2 * MAMBA_BLOCK


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
@pytest.mark.parametrize("prompt_len", [300, 450, 700])
def test_encoder_cut_final_chunk_stops_at_block_end(prompt_len, retention_interval):
    """An encoder input past the next block boundary cuts the final chunk after
    the split; the re-clamp stops it at the block's end again."""
    manager = _manager(0, retention_interval)
    worker = _Worker(manager, 0)
    _produce(manager, worker, 200)
    consumer = _consumer(prompt_len)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    ends = _prefill(
        manager, worker, consumer, num_local, local_blocks, cut_at=prompt_len - 10
    )
    assert ends[0] == 2 * MAMBA_BLOCK and ends[-1] == prompt_len


@pytest.mark.parametrize("prompt_len", [300, 450, 700])
def test_unclamped_encoder_cut_publishes_no_stale_state(prompt_len):
    """Without the re-clamp, under dense retention: the cut chunk runs past its
    block, and column K must not be published as the boundary state. At 78e9
    the two-blocks-later class (450, 700) keyed column K, holding the hit's
    state@192, as state@256."""
    manager = _manager(0)
    worker = _Worker(manager, 0)
    _produce(manager, worker, 200)
    consumer = _consumer(prompt_len)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    _prefill(
        manager,
        worker,
        consumer,
        num_local,
        local_blocks,
        cut_at=prompt_len - 10,
        reclamp=False,
    )


@pytest.mark.parametrize("prompt_len", [300, 450])
def test_unclamped_encoder_cut_fails_loudly_under_sparse_retention(prompt_len):
    """Without the re-clamp, under sparse retention, the manager records no
    checkpoint for the cut chunk while the worker would export into column K:
    the allocation must refuse it."""
    manager = _manager(0, retention_interval=0)
    worker = _Worker(manager, 0)
    _produce(manager, worker, 200)
    consumer = _consumer(prompt_len)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    with pytest.raises(AssertionError, match="leaves its block without a checkpoint"):
        _prefill(
            manager,
            worker,
            consumer,
            num_local,
            local_blocks,
            cut_at=prompt_len - 10,
            reclamp=False,
        )


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
def test_resumed_request_replaying_outputs(retention_interval):
    """A resumed request replays its output tokens: the checkpoint follows the
    chunk end, not the prompt checkpoint, so column K's keys are dropped (along
    with the (K + 1) * B key cache_full_blocks gave it)."""
    manager = _manager(0, retention_interval)
    worker = _Worker(manager, 0)
    _produce(manager, worker, 200)
    request = _consumer(250, "resumed")
    request.append_output_token_ids(list(range(30000, 30050)))
    assert request.num_tokens == 300
    local_blocks, num_local, _ = manager.get_computed_blocks(request)
    assert num_local == 192
    assert _prefill(manager, worker, request, num_local, local_blocks) == [300]
    block = _checkpoint_block(manager, request)
    assert block.block_hash is None
    assert worker.state_at[block.block_id] == 288


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
def test_junction_producer_resumes_into_next_block(num_speculative_blocks):
    """A producer stops at its shared-prefix junction (192, mid-block) and
    registers it as a partial tail. Its final chunk ends in the next block: the
    CoW moves the key to a copy first, and the export overwrites the source."""
    manager = _manager(
        num_speculative_blocks, retention_interval=0, shared_prefix_checkpoint=True
    )
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    assert mamba.shared_prefix_checkpoint
    worker = _Worker(manager, num_speculative_blocks)
    request = make_request(
        "junction", list(range(300)), HASH, sha256, num_hash_lookahead_tokens=1
    )
    request.shared_prefix_boundary = 192
    ends = _prefill(manager, worker, request, 0, None, shared_prefix_checkpoint=True)
    assert ends == [192, 300]
    manager.free(request)
    manager.new_step_starts()
    sibling = make_request(
        "sibling",
        list(range(200)) + list(range(40000, 40100)),
        HASH,
        sha256,
        num_hash_lookahead_tokens=1,
    )
    _, num_hit, _ = manager.get_computed_blocks(sibling)
    assert num_hit == 192


def test_connector_load_ending_at_prompt_checkpoint():
    """A Mooncake load that ends exactly at the prompt checkpoint registers the
    loaded block as a partial tail. The rest of the prompt stays in that block,
    so nothing is exported and the next allocation moves the key to a copy."""
    manager = _manager(0)
    worker = _Worker(manager, 0)
    request = make_request("ext_c", list(range(300)), HASH, sha256)
    loaded = manager.allocate_slots(
        request,
        num_new_tokens=0,
        num_external_computed_tokens=288,
        delay_cache_blocks=True,
    )
    assert loaded is not None
    worker.load(request, 288)
    request.num_computed_tokens = 288
    manager.cache_blocks(request, 288)
    assert _prefill(manager, worker, request, 0, None) == [300]
    worker.check()
    manager.free(request)
    manager.new_step_starts()
    nxt = make_request("ext_next", list(range(300)) + [7] * 40, HASH, sha256)
    _, num_hit, _ = manager.get_computed_blocks(nxt)
    assert num_hit == 288


def test_cached_checkpoint_column_fails_loudly():
    """The export overwrites column K, so a cache entry still on it at
    allocation (here planted by hand) must be refused."""
    manager = _manager(0)
    worker = _Worker(manager, 0)
    request = make_request("stale", list(range(300)), HASH, sha256)
    first = manager.allocate_slots(request, 192)
    assert first is not None
    worker.step(request, 0, 192, first)
    request.num_computed_tokens = 192
    manager.new_step_starts()
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    manager.block_pool.cache_partial_block(
        request=request,
        block=_checkpoint_block(manager, request),
        num_tokens=192,
        kv_cache_group_id=MAMBA_GID,
        block_size=MAMBA_BLOCK,
    )
    assert mamba.req_to_blocks[request.request_id][1].block_hash is not None
    with pytest.raises(AssertionError, match="cached as state@192"):
        manager.allocate_slots(request, _split(request, 300 - 192))
