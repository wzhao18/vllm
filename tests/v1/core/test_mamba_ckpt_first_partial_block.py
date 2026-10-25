# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefills that start mid-block.

A prefix hit at hash-block granularity usually ends inside a Mamba block. Such
a prefill may export its prompt checkpoint into the block holding its initial
state when that block is private to the request (a fresh local hit's
copy-on-write block, or the unhashed block an external load filled), so a short
continuation runs as one chunk instead of stopping at the next block boundary.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    make_full_mamba_manager,
)
from tests.v1.core.test_mamba_mid_block_start_cache import (
    ALIGN,
    BLOCK,
    HASH,
    _compute_step,
    _external_load,
    _full_block_hash_cached,
    _manager,
    _request,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.mark.parametrize(
    ("start", "end", "position", "allow", "expected"),
    [
        # Mid-block start, checkpoint column == initial-state column.
        (4096, 4756, 4736, False, False),
        (4096, 4756, 4736, True, True),
        # Block-aligned start: the relaxed rule never applies (column below).
        (3584, 4400, 4352, True, False),
        # Checkpoint too close to the start (< one hash block).
        (4096, 4200, 4096, True, False),
        # Misaligned checkpoint offset.
        (4096, 4756, 4736 - 8, True, False),
        # Longer query: the strict rule already holds.
        (4096, 6000, 5888, False, True),
    ],
)
def test_relaxed_checkpoint_rule(start, end, position, allow, expected):
    assert (
        is_mamba_prefill_checkpoint_valid(
            query_start=start,
            query_end=end,
            checkpoint_position=position,
            hash_block_size=HASH,
            mamba_block_size=BLOCK,
            checkpoint_alignment=ALIGN,
            allow_first_partial_block=allow,
        )
        is expected
    )


@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 4])
def test_external_load_short_continuation_one_chunk(num_speculative_blocks):
    manager, _ = _manager(num_speculative_blocks)
    loaded = 4 * BLOCK + 512  # mid-block, hash-aligned (as Mooncake partial tails)
    prompt = loaded + 660
    request = _request(prompt)
    _external_load(manager, request, loaded)

    initial_col = (loaded - 1) // BLOCK
    loaded_block = manager.req_to_blocks["r"][initial_col]
    # The loaded mid-block state block is exclusively owned and not cached.
    assert not loaded_block.is_null
    assert loaded_block.ref_cnt == 1
    assert loaded_block.block_hash is None
    assert manager.initial_state_block_writable("r", loaded)

    manager.new_step_starts()
    manager.get_num_blocks_to_allocate("r", prompt, [], loaded, loaded, prompt)
    position, idx = manager._checkpoints["r"]
    assert idx == initial_col
    assert position == get_mamba_prefill_checkpoint_position(
        prompt, manager.cache_hit_alignment_tokens, manager.drop_eagle_checkpoint_block
    )
    assert loaded < position < prompt
    request.num_computed_tokens = loaded
    _compute_step(manager, request, prompt)
    # The checkpoint went into the loaded block itself, keyed at its position
    # and offered to connectors there, never at the block's end boundary.
    assert manager.req_to_blocks["r"][idx] is loaded_block
    assert loaded_block.block_hash_num_tokens == position
    assert _full_block_hash_cached(manager, request, (idx + 1) * BLOCK) is None
    offers = list(manager._pending_boundary_state_offloads)
    assert (0, loaded_block, position) in [o[1:] for o in offers]


def test_shared_initial_state_block_declines():
    """A block someone else also references is never overwritten."""
    manager, _ = _manager()
    loaded = 4 * BLOCK + 512
    prompt = loaded + 660
    request = _request(prompt)
    _external_load(manager, request, loaded)
    block = manager.req_to_blocks["r"][(loaded - 1) // BLOCK]
    block.ref_cnt += 1  # shared
    assert not manager.initial_state_block_writable("r", loaded)
    manager.new_step_starts()
    manager.get_num_blocks_to_allocate("r", prompt, [], loaded, loaded, prompt)
    assert "r" not in manager._checkpoints
    block.ref_cnt -= 1


def test_local_partial_hit_checkpoints_into_cow_block():
    """A fresh local hit ending mid-block runs a short continuation as one
    chunk: the checkpoint goes into the copy-on-write block, and the shared
    source block is left alone."""
    hash_block_size, mamba_block_size = 16, 64
    manager = make_full_mamba_manager(
        dcp_world_size=1,
        hash_block_size=hash_block_size,
        full_block_size=hash_block_size,
        mamba_block_size=mamba_block_size,
        num_blocks=128,
        num_prefill_checkpoint_blocks=1,
    )
    mamba = manager.coordinator.single_type_managers[1]
    # The owner's prompt checkpoint lands at 288, mid-block.
    owner = make_request("owner", list(range(300)), hash_block_size, sha256)
    assert manager.allocate_slots(owner, owner.num_tokens) is not None
    owner.num_computed_tokens = owner.num_tokens
    manager.free(owner)
    manager.new_step_starts()

    request = make_request("r", list(range(288)) + [7] * 40, hash_block_size, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(request)
    assert num_computed == 288
    col = (num_computed - 1) // mamba_block_size
    source = computed.blocks[1][col]
    source_hash = source.block_hash
    assert manager.mamba_initial_state_blocks_writable("r", num_computed)

    new_blocks = manager.allocate_slots(
        request, request.num_tokens - num_computed, num_computed, computed
    )
    assert new_blocks is not None
    position, idx = mamba._checkpoints["r"]
    assert (position, idx) == (320, col)
    cow = mamba.req_to_blocks["r"][col]
    assert cow is not source
    copies, retained = manager.take_kv_cache_block_copies()
    assert (source.block_id, cow.block_id) in [
        (c.src_block_id, c.dst_block_id) for c in copies
    ]
    manager.block_pool.free_blocks(retained)
    assert source.block_hash == source_hash
    assert cow.block_hash_num_tokens == position


def _split(request, num_new_tokens, *, allow, junction=0, start=None):
    """The real `Scheduler._mamba_block_aligned_split` on a stub self."""
    if start is not None:
        request.num_computed_tokens = start
    request.shared_prefix_boundary = junction
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK),
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=True,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ALIGN,
    )
    return Scheduler._mamba_block_aligned_split(
        stub, request, num_new_tokens, allow_first_partial_block=allow
    )


def test_split_runs_short_continuation_in_one_chunk_only_when_allowed():
    loaded = 4 * BLOCK + 512
    request = _request(loaded + 660)
    rest = request.num_tokens - loaded
    assert _split(request, rest, allow=True, start=loaded) == rest
    assert _split(request, rest, allow=False, start=loaded) == 5 * BLOCK - loaded


def test_split_stops_shortened_mid_block_chunk_at_block_boundary():
    """A junction stop shortens an allowed chunk so it no longer finishes the
    prefill; it must then end at the next block boundary, not run past it."""
    loaded = 4 * BLOCK + 512
    request = _request(loaded + 660)
    rest = request.num_tokens - loaded
    junction = 5 * BLOCK + 128
    assert (
        _split(request, rest, allow=True, junction=junction, start=loaded)
        == 5 * BLOCK - loaded
    )


@pytest.mark.parametrize(
    ("resumable", "fresh", "outputs", "encoder", "lookahead", "expected"),
    [
        (False, True, 0, False, 1, True),  # fresh local hit
        (False, False, 0, False, 1, True),  # after an async external load
        (True, True, 0, False, 1, True),  # streaming session, first admission
        (True, False, 0, False, 1, False),  # streaming session resuming
        (False, True, 2, False, 1, False),  # resumed request with outputs
        (False, True, 0, True, 1, False),  # encoder inputs may trim the chunk
        (False, True, 0, False, 3, False),  # MTP lookahead may trim the chunk
    ],
)
def test_initial_state_block_private_conditions(
    resumable, fresh, outputs, encoder, lookahead, expected
):
    request = SimpleNamespace(
        request_id="r",
        resumable=resumable,
        num_output_tokens=outputs,
        has_encoder_inputs=encoder,
    )
    stub = SimpleNamespace(
        num_prefill_lookahead=lookahead,
        kv_cache_manager=SimpleNamespace(
            mamba_initial_state_blocks_writable=lambda rid, n: True
        ),
    )
    assert (
        Scheduler._mamba_initial_state_block_private(
            stub, request, 4096, fresh_admission=fresh
        )
        is expected
    )


@pytest.mark.parametrize(
    ("mamba_block_size", "hash_block_size"), [(BLOCK, HASH), (64, 16), (512, 128)]
)
def test_scheduler_manager_worker_lockstep(mamba_block_size, hash_block_size):
    """For every mid-block external-load start and continuation length, the
    split, the manager's reservation and the worker's export agree: the worker
    exports a checkpoint exactly where the manager reserved one, and a chunk
    that starts mid-block and runs past the block's end never publishes that
    block as the boundary's state."""
    spec = MambaSpec(
        block_size=mamba_block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=ALIGN,
    )
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=mamba_block_size),
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=1 << 20,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=hash_block_size,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ALIGN,
    )
    hasher = get_request_block_hasher(hash_block_size, sha256)
    checked = 0
    for loaded_blocks in range(1, 4 * mamba_block_size // hash_block_size):
        loaded = loaded_blocks * hash_block_size
        if loaded % mamba_block_size == 0:
            continue
        for continuation in range(1, 3 * mamba_block_size, hash_block_size // 2 + 1):
            pool = BlockPool(
                num_gpu_blocks=256, enable_caching=True, hash_block_size=hash_block_size
            )
            manager = MambaManager(
                spec,
                block_pool=pool,
                enable_caching=True,
                kv_cache_group_id=0,
                scheduler_block_size=mamba_block_size,
            )
            manager.cache_hit_alignment_tokens = hash_block_size
            request = Request(
                request_id="r",
                prompt_token_ids=list(range(loaded + continuation)),
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_hasher=hasher,
            )
            _external_load(manager, request, loaded)
            allow = manager.initial_state_block_writable("r", loaded)
            n = Scheduler._mamba_block_aligned_split(
                stub, request, continuation, allow_first_partial_block=allow
            )
            assert n > 0
            end = loaded + n
            col = (loaded - 1) // mamba_block_size
            start_block = manager.req_to_blocks["r"][col]
            _compute_step(manager, request, end)
            offsets, cols = compute_mamba_prefill_checkpoints(
                [end],
                [n],
                hash_block_size=hash_block_size,
                mamba_block_size=mamba_block_size,
                checkpoint_alignment=ALIGN,
                drop_eagle_block=False,
                cache_hit_alignment_tokens=manager.cache_hit_alignment_tokens,
            )
            reserved = manager._checkpoints.get("r")
            if offsets[0]:
                assert reserved == (loaded + offsets[0], cols[0]), (loaded, n)
            else:
                assert reserved is None, (loaded, n)
            if end > (col + 1) * mamba_block_size:
                # Ran past the start block: it is either the checkpoint block,
                # keyed at the checkpoint, or not cached at all.
                if reserved is not None and reserved[1] == col:
                    assert start_block.block_hash_num_tokens == reserved[0]
                else:
                    assert start_block.block_hash is None, (loaded, n)
            checked += 1
    assert checked > 0
