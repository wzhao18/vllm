# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefills resuming mid-block with an internal checkpoint.

The final state goes in the running column and the prompt checkpoint in the
column before it. A chunk starting mid-block may export it into the column it
resumes from: that block holds the request's private copy of the resumed
state, read before the forward. A final chunk resuming from a partial
(sub-block) prefix hit and ending in a later block then runs in one step.

These tests drive the real scheduler split, KV cache manager and worker
checkpoint planner, and track which state every Mamba block holds the way the
worker writes it: CoW copies, then the align pre-copy into the running slot,
then the forward's running state and the planned checkpoint export. Every
block a hash claims, and every boundary state handed to a connector, must hold
the state at exactly the claimed token count.
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
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
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
    alignment: int = ALIGNMENT,
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
                    prefill_checkpoint_alignment=alignment,
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
    alignment: int = ALIGNMENT,
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
        mamba_prefill_checkpoint_alignment=alignment,
    )
    return Scheduler._mamba_block_aligned_split(
        stub, request, num_new_tokens, num_new_local_computed_tokens
    )


class _Worker:
    """Which state each Mamba block holds, written the way the worker does."""

    def __init__(self, manager):
        self.manager = manager
        self.mamba = manager.coordinator.single_type_managers[MAMBA_GID]
        self.state_at: dict[int, int] = {}
        self.tables: dict[str, list[int]] = {}
        self.state_col: dict[str, int] = {}
        # Every boundary state offered to a connector: (block id, tokens).
        self.offloaded: list[tuple[int, int]] = []

    def load(self, request, num_tokens: int) -> None:
        """A connector load writes the state at `num_tokens` into its slot."""
        blocks = self.mamba.req_to_blocks[request.request_id]
        self.state_at[blocks[cdiv(num_tokens, MAMBA_BLOCK) - 1].block_id] = num_tokens

    def step(self, request, start: int, end: int, new_blocks) -> None:
        req_id = request.request_id
        if req_id not in self.tables:
            # A newly scheduled request ships its whole block table.
            self.tables[req_id] = [b.block_id for b in self.mamba.req_to_blocks[req_id]]
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
            # The column it resumes from still holds the state at `start`: an
            # export into it only follows this copy.
            assert self.state_at.get(table[prev_col]) == start
        # Forward: running state and the internal checkpoint export.
        self.state_at[table[running_col]] = end
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end],
            [end - start],
            hash_block_size=HASH,
            mamba_block_size=MAMBA_BLOCK,
            checkpoint_alignment=self.mamba.kv_cache_spec.prefill_checkpoint_alignment,
            drop_eagle_block=False,
            cache_hit_alignment_tokens=self.mamba.cache_hit_alignment_tokens,
            num_hash_lookahead_tokens=self.manager.block_pool.num_hash_lookahead_tokens,
        )
        # The worker exports exactly the manager's plan, which the rule
        # cross-checks.
        planned = self.manager.take_mamba_prefill_checkpoints().get(req_id, 0)
        if planned:
            assert start + offsets[0] == planned
            # The export never targets the running slot.
            assert cols[0] != running_col
            self.state_at[table[cols[0]]] = planned

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
        # Connectors take one sub-block boundary per request and pass.
        sub_block: dict[str, set[int]] = {}
        for req_id, _, _, tokens in offloads:
            if tokens % MAMBA_BLOCK:
                sub_block.setdefault(req_id, set()).add(tokens)
        assert all(len(b) == 1 for b in sub_block.values()), sub_block
        self.offloaded.extend(
            (block.block_id, tokens) for *_, block, tokens in offloads
        )
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
    shared_prefix_checkpoint: bool = False,
):
    """Schedule the rest of `request`'s prompt; returns the chunk ends.

    `cut_at` cuts the first chunk there after the split, as an encoder input
    that does not fit the encoder budget does.
    """
    ends = []
    while request.num_computed_tokens + num_local < request.num_tokens:
        start = request.num_computed_tokens + num_local
        num_new = _split(
            request,
            request.num_tokens - start,
            num_local,
            shared_prefix_checkpoint=shared_prefix_checkpoint,
            alignment=worker.mamba.kv_cache_spec.prefill_checkpoint_alignment,
        )
        if start < cut_at < start + num_new:
            num_new = cut_at - start
            cut_at = 0
        assert num_new > 0
        new_blocks = manager.allocate_slots(request, num_new, num_local, local_blocks)
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


def _consumer(prompt_len: int, request_id: str = "consumer"):
    return make_request(
        request_id,
        list(range(200)) + list(range(10000, 10000 + prompt_len - 200)),
        HASH,
        sha256,
    )


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize(
    "prompt_len",
    [
        # Resumes at 192 (block 1, mid-block) and ends in block 2: the
        # checkpoint goes in column 1, the resumed state's private copy.
        # 260: that checkpoint is block 1's end state, 256.
        260,
        300,
        330,
        384,
        # Ends two or more blocks later: block 1 keeps the hit's state@192.
        450,
        700,
    ],
)
def test_partial_hit_final_chunk_runs_in_one_step(
    num_speculative_blocks, prompt_len, retention_interval
):
    manager = _manager(num_speculative_blocks, retention_interval)
    worker = _Worker(manager)
    _produce(manager, worker, 200)

    consumer = _consumer(prompt_len)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    assert num_local == 192 and num_local % MAMBA_BLOCK != 0
    ends = _prefill(manager, worker, consumer, num_local, local_blocks)
    assert ends == [prompt_len]
    # The prompt checkpoint reaches connectors, from the resumed block too.
    checkpoint = (prompt_len - 1) // HASH * HASH
    assert any(tokens == checkpoint for _, tokens in worker.offloaded)

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


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    ("prompt_len", "cut_at", "alignment"),
    [
        (300, 290, ALIGNMENT),
        (450, 440, ALIGNMENT),
        (700, 690, ALIGNMENT),
        # The split approves 330 (checkpoint 320), but at the cut no checkpoint
        # is exported: 288 is off the 64-token grid from 192.
        (330, 300, 64),
    ],
)
def test_encoder_cut_chunk_publishes_no_stale_state(
    prompt_len, cut_at, alignment, retention_interval
):
    """An encoder input cuts the final chunk after the split, so it runs past
    the block it resumed in without being final. That block holds the resumed
    state or the checkpoint, not state@256 (the prompt's last block boundary,
    kept by sparse retention too); `_prefill` checks every key against the
    state its block holds."""
    manager = _manager(0, retention_interval, alignment=alignment)
    worker = _Worker(manager)
    _produce(manager, worker, 200)
    consumer = _consumer(prompt_len)
    local_blocks, num_local, _ = manager.get_computed_blocks(consumer)
    ends = _prefill(manager, worker, consumer, num_local, local_blocks, cut_at=cut_at)
    assert ends[0] == cut_at and ends[-1] == prompt_len


@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    ("num_outputs", "col", "checkpoint"),
    [
        # The checkpoint at 288 goes in column 1, the resumed block.
        (50, 1, 288),
        # Ends two blocks later: the checkpoint (416) gets column 2.
        (170, 2, 416),
    ],
)
def test_resumed_request_replaying_outputs(
    retention_interval, num_outputs, col, checkpoint
):
    """A resumed request replays its output tokens as prefill: the worker
    exports a checkpoint past the prompt, a transient state, so its column
    keeps no key (`_prefill` checks every key against its block)."""
    manager = _manager(0, retention_interval)
    worker = _Worker(manager)
    _produce(manager, worker, 200)
    request = _consumer(250, "resumed")
    request.append_output_token_ids(list(range(30000, 30000 + num_outputs)))
    local_blocks, num_local, _ = manager.get_computed_blocks(request)
    assert num_local == 192
    assert _prefill(manager, worker, request, num_local, local_blocks) == [
        request.num_tokens
    ]
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    block = mamba.req_to_blocks[request.request_id][col]
    assert block.block_hash is None
    assert worker.state_at[block.block_id] == checkpoint


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize("prompt_len", [300, 330])
def test_junction_producer_resumes_into_next_block(num_speculative_blocks, prompt_len):
    """A producer stops at its shared-prefix junction (192, mid-block) and
    registers it as a partial tail. Its final chunk ends in a later block: the
    CoW moves the key to a copy first, and offers it to connectors, which take
    one sub-block boundary per request and pass (`_Worker.check`)."""
    manager = _manager(
        num_speculative_blocks, retention_interval=0, shared_prefix_checkpoint=True
    )
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    assert mamba.shared_prefix_checkpoint
    worker = _Worker(manager)
    request = make_request(
        "junction", list(range(prompt_len)), HASH, sha256, num_hash_lookahead_tokens=1
    )
    request.shared_prefix_boundary = 192
    ends = _prefill(manager, worker, request, 0, None, shared_prefix_checkpoint=True)
    assert ends == [192, prompt_len]
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


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize(("prompt_len", "ends"), [(300, [300]), (250, [240, 250])])
def test_running_mid_block_resume(num_speculative_blocks, prompt_len, ends):
    """A running request whose previous chunk stopped mid-block (as an encoder
    cut or a sub-block chunk does) resumes in one step into the next block.
    Ending in the same block, it has no column for the checkpoint and stops
    there."""
    manager = _manager(num_speculative_blocks)
    worker = _Worker(manager)
    request = make_request("run", list(range(prompt_len)), HASH, sha256)
    first = manager.allocate_slots(request, 192)
    assert first is not None
    worker.step(request, 0, 192, first)
    request.num_computed_tokens = 192
    worker.check()
    manager.new_step_starts()
    assert _prefill(manager, worker, request, 0, None) == ends


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize(
    ("prompt_len", "ends"), [(300, [300]), (450, [450]), (250, [240, 250])]
)
def test_connector_hit_final_chunk(num_speculative_blocks, prompt_len, ends):
    """A connector-loaded mid-block state resumes like a local hit."""
    manager = _manager(num_speculative_blocks)
    worker = _Worker(manager)
    request = make_request("ext", list(range(prompt_len)), HASH, sha256)
    loaded = manager.allocate_slots(
        request,
        num_new_tokens=0,
        num_external_computed_tokens=192,
        delay_cache_blocks=True,
    )
    assert loaded is not None
    worker.load(request, 192)
    request.num_computed_tokens = 192
    manager.cache_blocks(request, 192)
    assert _prefill(manager, worker, request, 0, None) == ends
    assert worker.check() > 0


@pytest.mark.parametrize(("num_new", "num_accepted"), [(17, 15), (40, 40)])
def test_decode_step_plans_no_checkpoint(num_new, num_accepted):
    """With at least a hash block of drafts, a decode step resuming mid-block
    passes the checkpoint rule, but decode rows export none. Its column only
    gets the block-end state the worker saves once 256 tokens are committed."""
    manager = _manager(0)
    worker = _Worker(manager)
    request = make_request("decode", list(range(240)), HASH, sha256)
    assert _prefill(manager, worker, request, 0, None) == [240]
    request.append_output_token_ids([1])
    assert manager.allocate_slots(request, num_new) is not None
    mamba = manager.coordinator.single_type_managers[MAMBA_GID]
    assert request.request_id not in mamba._checkpoints
    request.append_output_token_ids(list(range(num_accepted)))
    request.num_computed_tokens = 240 + num_accepted
    manager.cache_blocks(request, request.num_computed_tokens)
    block = mamba.req_to_blocks[request.request_id][1]
    expected = 256 if request.num_computed_tokens >= 256 else None
    assert block.block_hash_num_tokens == expected
