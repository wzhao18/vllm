# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunks that start mid-block.

A prefix hit at hash-block granularity (a local partial hit, or an external
load) usually ends inside a Mamba block. A prefill chunk writes its state only
at its end (and at its checkpoint), so when such a chunk runs past that block's
end the block still holds the chunk's initial state and must never be cached
or offered to connectors as the state at that block's end.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK = 896  # Kimi K3 TEP prefill: attention block 896 = Mamba block
HASH = 128
ALIGN = 16


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _manager(num_speculative_blocks: int = 0) -> tuple[MambaManager, BlockPool]:
    spec = MambaSpec(
        block_size=BLOCK,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=ALIGN,
    )
    pool = BlockPool(num_gpu_blocks=256, enable_caching=True, hash_block_size=HASH)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
    )
    # Hash-granular hits (prefix_match_unit < block), as the coordinator sets.
    manager.cache_hit_alignment_tokens = HASH
    return manager, pool


def _request(prompt_tokens: int, request_id: str = "r") -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH, sha256),
    )


def _external_load(manager: MambaManager, request: Request, loaded: int) -> None:
    """What the scheduler does for an async external hit of ``loaded`` tokens:
    allocate the slots the connector loads into, then (in
    ``_update_waiting_for_remote_kv``) cache the loaded blocks."""
    rid = request.request_id
    manager.get_num_blocks_to_allocate(rid, loaded, [], 0, 0, loaded)
    manager.allocate_new_blocks(rid, loaded, loaded)
    manager.cache_blocks(request, loaded, retention_interval=None, replay_boundaries=[])
    request.num_computed_tokens = loaded


def _compute_step(manager: MambaManager, request: Request, end: int) -> int:
    """One prefill step from ``request.num_computed_tokens`` to ``end``; returns
    the block estimate (which must match what is taken from the pool)."""
    rid = request.request_id
    start = request.num_computed_tokens
    manager.new_step_starts()
    estimate = manager.get_num_blocks_to_allocate(rid, end, [], start, start, end)
    free_before = manager.block_pool.get_num_free_blocks()
    manager.allocate_new_blocks(rid, end, end)
    assert free_before - manager.block_pool.get_num_free_blocks() == estimate
    manager.cache_blocks(request, end, retention_interval=None, replay_boundaries=[])
    request.num_computed_tokens = end
    return estimate


def _full_block_hash_cached(manager: MambaManager, request: Request, tokens: int):
    block_hash = request.block_hashes[tokens // HASH - 1]
    return manager.block_pool.get_cached_block(block_hash, [0])


@pytest.mark.parametrize("continuation", [1500, 3000])
def test_mid_block_start_never_publishes_initial_state(continuation):
    """Regression: a long continuation from a mid-block external load exports
    its checkpoint in a later column and runs past the start block's end. The
    start column still holds the loaded state, so it must not be cached (or
    offered) as the state at that block's end."""
    manager, _ = _manager()
    loaded = 4 * BLOCK + 512
    request = _request(loaded + continuation)
    _external_load(manager, request, loaded)
    col = (loaded - 1) // BLOCK
    start_block = manager.req_to_blocks["r"][col]

    _compute_step(manager, request, request.num_tokens)
    assert manager._checkpoints["r"][1] > col
    assert manager.req_to_blocks["r"][col] is start_block
    assert start_block.block_hash is None
    assert _full_block_hash_cached(manager, request, (col + 1) * BLOCK) is None
    offers = list(manager._pending_boundary_state_offloads)
    assert all(o[2] is not start_block for o in offers)


def _decode_crossing_hashes(unpublish: bool) -> list:
    manager, _ = _manager(num_speculative_blocks=3)
    if not unpublish:
        manager._unpublish_mid_block_start = lambda *args: None
    prompt = 4 * BLOCK + 380
    request = _request(prompt)
    _compute_step(manager, request, prompt)
    # Decode up to just below the next block boundary, then one speculative
    # step (1 + 3 tokens) that crosses it.
    while request.num_computed_tokens < 5 * BLOCK - 2:
        request.append_output_token_ids([1])
        manager.new_step_starts()
        start = request.num_computed_tokens
        manager.get_num_blocks_to_allocate("r", start + 1, [], start, start, start + 1)
        manager.allocate_new_blocks("r", start + 1, start + 1)
        manager.cache_blocks(
            request, start + 1, retention_interval=None, replay_boundaries=[]
        )
        request.num_computed_tokens = start + 1
    # A decode step: the last sampled token plus three draft tokens, which are
    # not part of ``num_tokens``.
    request.append_output_token_ids([1])
    start = request.num_computed_tokens
    assert start == request.num_tokens - 1
    assert start % BLOCK != 0 and start + 4 > 5 * BLOCK
    manager.new_step_starts()
    manager.get_num_blocks_to_allocate("r", start + 4, [], start, start, start + 4)
    manager.allocate_new_blocks("r", start + 4, start + 4)
    manager.cache_blocks(
        request,
        min(start + 4, request.num_tokens),
        retention_interval=None,
        replay_boundaries=[],
    )
    return [(b.block_id, b.block_hash_num_tokens) for b in manager.req_to_blocks["r"]]


def test_decode_step_crossing_a_block_keeps_upstream_caching():
    """Only prefill chunks are unpublished: a decode step that starts mid-block
    and crosses the boundary caches exactly what it did before."""
    assert _decode_crossing_hashes(True) == _decode_crossing_hashes(False)
