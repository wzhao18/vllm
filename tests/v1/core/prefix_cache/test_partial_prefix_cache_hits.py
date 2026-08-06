# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fine-grained prefix-cache checkpoints for hybrid full-attention/Mamba
models: chunk splitting, partial-block registration, CoW, and deferral."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    can_use_mamba_partial_cache_hits,
    get_block_hash,
    get_group_id,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _hybrid_mamba_config(block_size: int) -> KVCacheConfig:
    return _make_hybrid_kv_cache_config(
        block_size,
        num_blocks=32,
        spec_types=["full", "mamba_align"],
    )


def _hybrid_mamba_manager(
    block_size: int,
    hash_block_size: int,
    *,
    use_eagle: bool = False,
):
    return make_kv_cache_manager(
        kv_cache_config=_hybrid_mamba_config(block_size),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
    )


def _cache_in_chunks(manager, request, chunks: list[int]) -> None:
    computed_blocks, num_computed, _ = manager.get_computed_blocks(request)
    for index, chunk in enumerate(chunks):
        result = (
            manager.allocate_slots(request, chunk, num_computed, computed_blocks)
            if index == 0
            else manager.allocate_slots(request, chunk)
        )
        assert result is not None
        request.num_computed_tokens += chunk
        manager.new_step_starts()


def test_partial_replay_grid_requires_every_group_to_support_it():
    config = _hybrid_mamba_config(8)
    assert can_use_mamba_partial_cache_hits(config.kv_cache_groups, 2)
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["swa"],
            SlidingWindowSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=8,
            ),
        )
    )

    assert not can_use_mamba_partial_cache_hits(config.kv_cache_groups, 2)


def _splitter_mock(
    block_size: int,
    hash_block_size: int,
    *,
    max_num_scheduled_tokens: int,
    use_eagle: bool = False,
    partial_hits: bool = False,
    retention_interval: int | None = None,
    long_prefill_token_threshold: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=long_prefill_token_threshold
        ),
        eagle_rewind_tokens=(
            hash_block_size if use_eagle and partial_hits else block_size
        )
        if use_eagle
        else 0,
        hash_block_size=hash_block_size,
        replay_alignment_tokens=(hash_block_size if partial_hits else block_size),
        enable_mamba_partial_hash_hits=partial_hits,
        prefix_cache_retention_interval=retention_interval,
    )


def _split_prompt(mock: SimpleNamespace, request) -> list[int]:
    chunks = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        chunk = Scheduler._mamba_block_aligned_split(
            self=mock,
            request=request,
            num_new_tokens=request.num_prompt_tokens - request.num_computed_tokens,
        )
        assert chunk > 0
        chunks.append(chunk)
        request.num_computed_tokens += chunk
    return chunks


def test_mamba_align_split_partial_tail_schedule():
    """Chunk ends with partial hits on: block-aligned chunks, one extra stop
    at the prompt's last hash boundary (registering the partial tail), then
    the remaining tokens. block=512, hash=32, prompt=10000, budget=8192:
    0 -> 8192 -> 9728 -> 9984 -> 10000."""
    block_size = 512
    hash_block_size = 32
    mock = _splitter_mock(
        block_size,
        hash_block_size,
        max_num_scheduled_tokens=8192,
        partial_hits=True,
    )
    split = Scheduler._mamba_block_aligned_split

    req = make_request("0", [0] * 10000, hash_block_size, sha256)
    req.num_computed_tokens = 0
    assert split(self=mock, request=req, num_new_tokens=8192) == 8192
    req.num_computed_tokens = 8192
    # Stop at the last block boundary (9728).
    assert split(self=mock, request=req, num_new_tokens=1808) == 1536
    req.num_computed_tokens = 9728
    # Extra stop at the prompt's last hash boundary (9984).
    assert split(self=mock, request=req, num_new_tokens=272) == 256
    req.num_computed_tokens = 9984
    # Final 16 tokens run unchanged (no mid-block-resume stop: the next
    # block boundary is past the last block boundary).
    assert split(self=mock, request=req, num_new_tokens=16) == 16

    # Partial hits off: no extra stop, the tail runs in one chunk.
    mock.enable_mamba_partial_hash_hits = False
    req.num_computed_tokens = 9728
    assert split(self=mock, request=req, num_new_tokens=272) == 272
    mock.enable_mamba_partial_hash_hits = True

    # A request resumed mid-block (partial hash hit at 9984): the first chunk
    # stops at the next block boundary (10240), later chunk ends re-align.
    req2 = make_request("1", [0] * 12000, hash_block_size, sha256)
    req2.num_computed_tokens = 9984
    assert split(self=mock, request=req2, num_new_tokens=2016) == 256
    req2.num_computed_tokens = 10240
    assert split(self=mock, request=req2, num_new_tokens=1000) == 512


@pytest.mark.parametrize(
    (
        "hash_block_size",
        "prompt_tokens",
        "use_eagle",
        "partial_hits",
        "retention_interval",
        "expected_chunks",
    ),
    [
        pytest.param(2, 15, True, True, None, [8, 4, 3], id="pmu-eagle"),
        pytest.param(2, 9, True, True, 0, [6, 3], id="pmu-before-block"),
        pytest.param(2, 80, True, True, 32, [30, 32, 16, 2], id="pmu-periodic"),
        pytest.param(8, 80, True, False, 32, [24, 32, 16, 8], id="coarse-eagle"),
        pytest.param(8, 80, False, False, 0, [72, 8], id="coarse-prompt"),
    ],
)
def test_mamba_align_split_materializes_replay_checkpoints(
    hash_block_size: int,
    prompt_tokens: int,
    use_eagle: bool,
    partial_hits: bool,
    retention_interval: int | None,
    expected_chunks: list[int],
):
    mock = _splitter_mock(
        8,
        hash_block_size,
        max_num_scheduled_tokens=128,
        use_eagle=use_eagle,
        partial_hits=partial_hits,
        retention_interval=retention_interval,
    )
    req = make_request("0", list(range(prompt_tokens)), hash_block_size, sha256)

    assert _split_prompt(mock, req) == expected_chunks


def test_mamba_align_split_uses_global_shared_prefix_checkpoint():
    """A PMU resume must not shift the shared-prefix block grid."""
    block_size = 512
    hash_block_size = 32
    mock = _splitter_mock(
        block_size,
        hash_block_size,
        max_num_scheduled_tokens=8192,
        partial_hits=True,
        retention_interval=0,
    )
    req = make_request("0", [0] * 12000, hash_block_size, sha256)
    req.num_computed_tokens = 9984
    req.shared_prefix_boundary = 11000

    # floor(11000 / 512) * 512 = 10752, independent of the PMU-aligned start.
    assert (
        Scheduler._mamba_block_aligned_split(
            self=mock,
            request=req,
            num_new_tokens=2016,
        )
        == 768
    )


def test_mamba_align_split_when_block_exceeds_scheduling_budget():
    """Sub-block chunks make progress only when no step can fit a full block."""
    block_size = 11392
    token_budget = 8192
    prompt_length = 30000
    mock = _splitter_mock(
        block_size,
        32,
        max_num_scheduled_tokens=token_budget,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    mock.max_num_scheduled_tokens = block_size
    assert split(self=mock, request=req, num_new_tokens=token_budget) == 0
    mock.max_num_scheduled_tokens = token_budget

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(token_budget, prompt_length - req.num_computed_tokens)
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= token_budget
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [8192, 3200, 8192, 3200, 7216]


def test_mamba_align_split_when_block_exceeds_long_prefill_threshold():
    """A long-prefill cap below the block size permits sub-block progress."""
    block_size = 512
    token_budget = 8192
    long_prefill_threshold = 384
    prompt_length = 1300
    mock = _splitter_mock(
        block_size,
        32,
        max_num_scheduled_tokens=token_budget,
        long_prefill_token_threshold=long_prefill_threshold,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(
            long_prefill_threshold,
            prompt_length - req.num_computed_tokens,
        )
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= long_prefill_threshold
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [384, 128, 384, 128, 276]


def test_hybrid_mamba_align_partial_hash_hit():
    hash_block_size = 2
    mamba_block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    blocks = manager.allocate_slots(req0, 6, num_computed, computed_blocks)
    assert blocks is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    assert partial_mamba_block[0].block_hash_num_tokens == 6

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [3, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block[0].block_id
    assert manager.get_blocks("1").get_block_ids()[1][1] == mamba_new_block_ids[0]
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_mamba_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block[0].block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )
    assert manager.get_blocks("1").blocks[1][1].block_hash_num_tokens == 8


def test_hybrid_mamba_partial_tail_owner_uses_cow_on_continue():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    new_blocks = manager.allocate_slots(req0, 1)
    assert new_blocks is not None

    # Reversed CoW for the owning request: it keeps its own block (the
    # worker's block table is append-only), and no new mamba block is handed
    # to the worker. The prefix-cache entry is moved to a private copy that
    # the queued block copy fills before the next forward.
    assert new_blocks.get_block_ids()[1] == []
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == partial_mamba_block_id)
    assert cow_copy.dst_block_id != partial_mamba_block_id
    # The source block gave up the hash; the copy target now owns the entry.
    assert partial_mamba_block[0].block_hash is None
    moved = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id
    assert get_block_hash(moved[0].block_hash) == partial_mamba_hash
    assert get_group_id(moved[0].block_hash) == 1
    assert moved[0].block_hash_num_tokens == 6


def test_take_mamba_checkpoint_offloads_returns_cow_target():
    """The connector offload hand-off exposes the mamba CoW *target* block Y
    (the durable boundary state), not the overwritten source X, and only at
    the CoW step."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # Step A registered the partial tail but has not CoW'd yet: no offload.
    assert manager.take_mamba_checkpoint_offloads() == {}

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    source_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert source_block is not None
    source_block_id = source_block[0].block_id

    # Step B: the producer continues, triggering the CoW X->Y.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    offloads = manager.take_mamba_checkpoint_offloads()
    assert list(offloads.keys()) == ["0"]
    assert len(offloads["0"]) == 1
    group_id, block_id, boundary_tokens = offloads["0"][0]
    assert group_id == 1  # the mamba group
    assert boundary_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == source_block_id)
    # The offload points at the durable CoW target Y, not the overwritten X.
    assert block_id == cow_copy.dst_block_id
    assert block_id != source_block_id
    # Draining clears it.
    assert manager.take_mamba_checkpoint_offloads() == {}

    # The hand-off pinned Y (its CoW retention is released after this step,
    # and Y is off the request block table); freeing the request unpins it.
    cow_block = manager.block_pool.blocks[block_id]
    pinned_ref = cow_block.ref_cnt
    assert pinned_ref >= 1
    manager.free(req0)
    assert cow_block.ref_cnt == pinned_ref - 1


def test_partial_tail_pin_survives_released_cow_retention():
    """If the CoW retention is released before the hand-off is drained
    (immediate-free mode), the drain must rescue the cow block from the free
    queue: a raw ref increment would leave a ref>0 block allocatable, and the
    next allocation would pop it and assert."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # Retention released before the drain (defer_block_free=False ordering).
    _copies, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)

    offloads = manager.take_mamba_checkpoint_offloads()
    ((_group_id, block_id, boundary_tokens),) = offloads["0"]
    assert boundary_tokens == 6
    cow_block = manager.block_pool.blocks[block_id]
    assert cow_block.ref_cnt == 1

    # The pinned block is out of the free queue: draining every free block
    # neither trips the allocator's ref_cnt assert nor hands it out.
    new_blocks = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    assert block_id not in {b.block_id for b in new_blocks}


def test_partial_tail_offload_dropped_when_request_freed_before_drain():
    """A hand-off recorded in the same scheduling pass as the request's death
    must not be drained: its release hook has already run, so draining would
    leak a pinned block."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # The request dies (preempt/abort) before the scheduler drains.
    manager.block_pool.free_blocks(manager.pop_blocks_for_free(req0))
    assert manager.take_mamba_checkpoint_offloads() == {}


def test_block_aligned_mamba_checkpoint_is_handed_to_connector():
    """A block-aligned prompt checkpoint gets the same lifetime protection."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    # 4-token prompt ends exactly on the mamba block boundary (block_size=4).
    req0 = make_request("0", [0, 0, 1, 1], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    offloads = manager.take_mamba_checkpoint_offloads()
    assert offloads.keys() == {"0"}
    group_id, block_id, boundary_tokens = offloads["0"][0]
    assert group_id == 1
    assert boundary_tokens == 4
    checkpoint_block = manager.block_pool.blocks[block_id]
    assert checkpoint_block.ref_cnt >= 2

    req0.num_computed_tokens = 4
    req0.append_output_token_ids([2])
    assert manager.allocate_slots(req0, 1) is not None
    assert manager.take_mamba_checkpoint_offloads() == {}


def test_coarse_mamba_checkpoint_stays_pinned_after_retirement(monkeypatch):
    """An async connector must not observe a reused Mamba checkpoint ID."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    block_size = 8
    manager = _hybrid_mamba_manager(block_size, block_size)
    request = make_request("owner", list(range(24)), block_size, sha256)

    computed_blocks, num_computed, _ = manager.get_computed_blocks(request)
    assert (
        manager.allocate_slots(request, 16, num_computed, computed_blocks) is not None
    )
    offloads = manager.take_mamba_checkpoint_offloads()
    assert offloads.keys() == {"owner"}
    group_id, block_id, boundary_tokens = offloads["owner"][0]
    assert group_id == 1
    assert boundary_tokens == 16
    checkpoint_block = manager.block_pool.blocks[block_id]
    assert checkpoint_block.ref_cnt == 2

    request.num_computed_tokens = 16
    manager.new_step_starts()
    assert manager.allocate_slots(request, 8) is not None
    request.num_computed_tokens = 24
    manager.new_step_starts()
    request.append_output_token_ids([99])
    assert manager.allocate_slots(request, 1) is not None

    mamba_manager = manager.coordinator.single_type_managers[1]
    assert checkpoint_block not in mamba_manager.req_to_blocks[request.request_id]
    assert checkpoint_block.ref_cnt == 1
    assert checkpoint_block.block_hash is not None

    manager.free(request)
    assert checkpoint_block.ref_cnt == 0


def test_truncate_computed_blocks_preserves_sparse_prefix_positions():
    """truncate_computed_blocks slices each group by its own block size,
    keeps null placeholders in the retained prefix, and leaves the original
    lookup result untouched (pure view, no refcount changes)."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=2 * hash_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    producer = make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 6, num_computed, blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request(
        "consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256
    )
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in blocks.blocks] == [3, 2]

    truncated = manager.truncate_computed_blocks(blocks, 4)

    assert [len(group) for group in truncated.blocks] == [2, 1]
    assert truncated.blocks[1][0].is_null
    assert [len(group) for group in blocks.blocks] == [3, 2]


def test_connector_truncation_pads_lagging_mamba_positions():
    block_size = 8
    manager = _hybrid_mamba_manager(block_size, hash_block_size=2)
    blocks = manager.create_kv_cache_blocks((manager.block_pool.get_new_blocks(2), []))

    truncated = manager.truncate_computed_blocks(blocks, block_size)

    assert [len(group) for group in truncated.blocks] == [1, 1]
    assert truncated.blocks[1][0].is_null
    assert [len(group) for group in blocks.blocks] == [2, 0]


def test_hybrid_mamba_partial_tail_owner_continue_preserves_later_hit():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    # The owner moved the prefix-cache entry to a private copy; capture its id.
    owner_copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in owner_copies if c.src_block_id == partial_mamba_block_id)
    moved_block_id = cow_copy.dst_block_id
    manager.new_step_starts()

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    # The later request hits the moved (private-copy) entry, not the source.
    assert computed_blocks.get_block_ids()[1][1] == moved_block_id

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != moved_block_id
    # The hitting request CoWs from the moved entry into its own private block.
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=moved_block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )


def test_hybrid_mamba_moved_partial_entry_defers_same_step_hit():
    """The owner's move re-arms the same-step guard: the moved entry is
    filled by this step's copy, and chained same-step copies read stale
    sources, so a request hitting it in the move step must be deferred."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.new_step_starts()

    # The owning request continues decoding: the partial entry moves to a
    # private copy in this step.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # A request hitting the moved entry in the SAME step must be deferred.
    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is None

    # Next step the moved entry is consumable.
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is not None


def test_hybrid_full_attention_partial_hash_hit_uses_cow():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_full_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_full_hash, kv_cache_group_ids=[0]
    )
    assert partial_full_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_ids = new_blocks.get_block_ids()[0]
    assert len(full_new_block_ids) == 1
    assert full_new_block_ids[0] != partial_full_block[0].block_id
    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_full_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    copies, retained = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_full_block[0].block_id,
            dst_block_id=full_new_block_ids[0],
        )
        in copies
    )
    assert partial_full_block[0].ref_cnt == 1
    manager.block_pool.free_blocks(retained)
    assert partial_full_block[0].ref_cnt == 0


def test_hybrid_partial_hit_cow_target_starts_uncached():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[0]
    )
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[1]
    )
    assert partial_full_block is not None
    assert partial_mamba_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6

    new_blocks = manager.allocate_slots(
        req1,
        2,
        num_computed,
        computed_blocks,
        delay_cache_blocks=True,
    )
    assert new_blocks is not None

    full_cow_block = manager.get_blocks("1").blocks[0][1]
    mamba_cow_block = manager.get_blocks("1").blocks[1][1]
    assert full_cow_block.block_id != partial_full_block[0].block_id
    assert mamba_cow_block.block_id != partial_mamba_block[0].block_id
    assert full_cow_block.block_hash is None
    assert full_cow_block.block_hash_num_tokens is None
    assert mamba_cow_block.block_hash is None
    assert mamba_cow_block.block_hash_num_tokens is None

    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6


def test_hybrid_partial_hash_truncates_full_attention_hit_length():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    pool = manager.block_pool
    req = make_request(
        "0",
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        hash_block_size,
        sha256,
    )

    full_blocks = pool.get_new_blocks(3)
    pool.cache_full_blocks(
        request=req,
        blocks=full_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    pool.cache_partial_block(
        request=req,
        block=full_blocks[2],
        num_tokens=10,
        kv_cache_group_id=0,
        block_size=block_size,
    )

    mamba_block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(
        request=req,
        block=mamba_block,
        num_tokens=6,
        kv_cache_group_id=1,
        block_size=block_size,
    )

    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]


def test_cow_retained_blocks_returned_for_release():
    """new_step_starts returns the CoW copy retentions instead of freeing
    them; the scheduler owns releasing them once the copy has run."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # The owner's move queues a copy and retains both endpoints.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    (cow_copy,), retained = manager.take_kv_cache_block_copies()
    assert {b.block_id for b in retained} == {
        cow_copy.src_block_id,
        cow_copy.dst_block_id,
    }
    # Not freed yet: the retention refs are still held.
    assert all(b.ref_cnt > 0 for b in retained)
    manager.block_pool.free_blocks(retained)


def test_free_cow_retained_blocks_defers_until_copy_step_processed():
    """Scheduler releases CoW retentions immediately when the copy's step has
    been processed (or deferral is off), and defers them otherwise."""
    from collections import deque

    freed: list = []
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=9)]
    mock = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(free_blocks=freed.extend)
        ),
        deferred_frees=deque(),
        defer_block_free=True,
        processed_step_seq=2,
    )
    free = Scheduler._free_cow_retained_blocks

    # Copy step still in flight: deferred with its fence.
    free(mock, list(blocks), fence_seq=3)
    assert not freed
    assert mock.deferred_frees == deque([(3, blocks[::-1])])

    # Copy step processed: freed immediately.
    mock.processed_step_seq = 3
    free(mock, list(blocks), fence_seq=3)
    assert freed == blocks

    # Deferral disabled: freed immediately regardless of the fence.
    freed.clear()
    mock.deferred_frees.clear()
    mock.defer_block_free = False
    mock.processed_step_seq = 0
    free(mock, list(blocks), fence_seq=3)
    assert freed == blocks


def test_hybrid_partial_hit_with_eagle_stays_within_group_blocks():
    """Every group must stay within the common EAGLE replay cap."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )

    # The owner prefills in scheduler-split style: stop at the block boundary
    # (4), then at the prompt's last hash boundary (6, partial entries).
    req0 = make_request("0", [7] * 6, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 4
    manager.new_step_starts()
    assert manager.allocate_slots(req0, 2) is not None
    req0.num_computed_tokens = 6
    manager.new_step_starts()

    # A longer request resumes at the checkpoint covered by every group.
    req1 = make_request("1", [7] * 6 + [9] * 2, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 4
    assert all(
        len(group) * block_size >= num_computed for group in computed_blocks.blocks
    )
    assert manager.allocate_slots(req1, 4, num_computed, computed_blocks) is not None


@pytest.mark.parametrize(
    ("hash_block_size", "use_eagle", "prompt_tokens", "chunks", "expected_hit"),
    [
        pytest.param(2, False, 14, [14], 14, id="pmu"),
        pytest.param(2, True, 18, [16, 2], 16, id="pmu-eagle-block"),
    ],
)
def test_retention_zero_keeps_prompt_replay_checkpoint(
    monkeypatch,
    hash_block_size: int,
    use_eagle: bool,
    prompt_tokens: int,
    chunks: list[int],
    expected_hit: int,
):
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    manager = _hybrid_mamba_manager(8, hash_block_size, use_eagle=use_eagle)
    owner = make_request("owner", list(range(prompt_tokens)), hash_block_size, sha256)
    _cache_in_chunks(manager, owner, chunks)

    replay = make_request(
        "replay", list(range(prompt_tokens)) + [99], hash_block_size, sha256
    )
    _, num_computed, _ = manager.get_computed_blocks(replay)
    assert num_computed == expected_hit


def test_dspark_keeps_only_usable_pmu_snapshot(monkeypatch):
    """DSpark retains its replay state, not the adjacent lookahead state."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    hash_block_size = 2
    block_size = 8
    manager = _hybrid_mamba_manager(block_size, hash_block_size, use_eagle=True)

    owner = make_request("owner", list(range(14)), hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(owner)
    assert manager.allocate_slots(owner, 12, num_computed, computed_blocks) is not None
    owner.num_computed_tokens = 12
    manager.new_step_starts()
    assert manager.allocate_slots(owner, 2) is not None
    replay_offloads = manager.take_mamba_checkpoint_offloads()
    assert [boundary for _, _, boundary in replay_offloads["owner"]] == [12]
    owner.num_computed_tokens = 14
    manager.new_step_starts()

    pool = manager.block_pool

    def has_mamba_state(boundary: int) -> bool:
        block_hash = owner.block_hashes[boundary // hash_block_size - 1]
        return pool.get_cached_block(block_hash, kv_cache_group_ids=[1]) is not None

    assert not has_mamba_state(8)
    assert not has_mamba_state(10)
    assert has_mamba_state(12)
    assert not has_mamba_state(14)

    replay = make_request("replay", list(range(14)) + [99, 100], 2, sha256)
    _, num_computed, _ = manager.get_computed_blocks(replay)
    assert num_computed == 12

    # The latest PMU boundary is an attention lookahead only. Continuing the
    # request must not create a second Mamba snapshot or Mooncake handoff.
    owner.append_output_token_ids([101])
    assert manager.allocate_slots(owner, 1) is not None
    assert manager.take_mamba_checkpoint_offloads() == {}


def test_dspark_periodic_retention_keeps_backed_off_pmu_states(monkeypatch):
    """Each block-aligned retention interval keeps its preceding PMU state."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "32")
    hash_block_size = 2
    block_size = 8
    manager = _hybrid_mamba_manager(block_size, hash_block_size, use_eagle=True)

    owner = make_request("owner", list(range(80)), hash_block_size, sha256)
    _cache_in_chunks(manager, owner, [30, 32, 16, 2])

    pool = manager.block_pool

    def has_mamba_state(boundary: int) -> bool:
        block_hash = owner.block_hashes[boundary // hash_block_size - 1]
        return pool.get_cached_block(block_hash, kv_cache_group_ids=[1]) is not None

    assert has_mamba_state(30)
    assert has_mamba_state(62)
    assert has_mamba_state(78)
    assert not has_mamba_state(32)
    assert not has_mamba_state(64)

    replay = make_request("replay", list(range(66)), hash_block_size, sha256)
    _, num_computed, _ = manager.get_computed_blocks(replay)
    assert num_computed == 62
