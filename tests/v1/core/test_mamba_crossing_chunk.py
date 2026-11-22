# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" crossing chunks (`get_mamba_crossing_checkpoint_position`).

An intermediate prefill chunk that resumes mid-block p may run past the block
boundary B = (p + 1) * block_size into block p + 1, exporting the state at B into
column p from inside the step (the backend's prefill snapshot support), instead
of stopping at B. Block p must then hold exactly the state its full-block hash
claims, the same boundaries must be published as with block-ended chunks, and
the scheduler, `MambaManager` and the worker must agree on every chunk.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched import scheduler as sched_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_crossing_checkpoint_position,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

HASH = 16  # hash block (= attention block) size
BLK = 1600  # mamba block size
ALIGN = 16  # backend checkpoint alignment
MAMBA_GROUP_ID = 1


def _crossing(start: int, end: int, prefill_end: int, align: int | None = ALIGN):
    return get_mamba_crossing_checkpoint_position(
        query_start=start,
        query_end=end,
        prefill_end=prefill_end,
        mamba_block_size=BLK,
        hash_block_size=HASH,
        checkpoint_alignment=align,
    )


@pytest.mark.parametrize(
    ("start", "end", "prefill_end", "expected"),
    [
        # Mid-block start crossing one boundary, not the prefill's last chunk.
        (800, 2000, 9000, 1600),
        (800, 3200, 9000, 1600),  # ends exactly on the next boundary
        (2400, 4000, 9000, 3200),
        # Block-aligned start: a chunk <= one block never crosses.
        (1600, 3000, 9000, 0),
        # Ends on / before its own boundary: today's block-ended chunk.
        (800, 1600, 9000, 0),
        (800, 1200, 9000, 0),
        # Two boundaries.
        (800, 3216, 9000, 0),
        # Finishes the prefill: the prompt-tail rules apply, not this one.
        (800, 2000, 2000, 0),
        (800, 2000, 1999, 0),
        # Hash-misaligned start.
        (808, 2000, 9000, 0),
    ],
)
def test_crossing_checkpoint_position(start, end, prefill_end, expected) -> None:
    assert _crossing(start, end, prefill_end) == expected
    assert _crossing(start, end, prefill_end, align=None) == 0


def _stub(snapshot: bool, budget: int = 4096) -> SimpleNamespace:
    return SimpleNamespace(
        block_size=BLK,
        mamba_block_size=BLK,
        cache_config=SimpleNamespace(
            block_size=BLK, mamba_extra_tail_checkpoint_min_span=0
        ),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ALIGN,
        mamba_internal_snapshot=snapshot,
        mamba_crossing_chunks=snapshot,
    )


def _split(request: Request, num_new: int, snapshot: bool) -> int:
    return Scheduler._mamba_block_aligned_split(_stub(snapshot), request, num_new)


@pytest.mark.parametrize("resume_at", [400, 800, 1584, 2416, 4000])
@pytest.mark.parametrize("budget", [700, 1500, 2400, 5000])
def test_split_crosses_one_boundary_only_with_snapshot(resume_at, budget) -> None:
    prompt_len = 10 * BLK + 7
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=HASH)
    request.num_computed_tokens = resume_at
    base = _split(request, budget, snapshot=False)
    cross = _split(request, budget, snapshot=True)
    boundary = (resume_at // BLK + 1) * BLK
    # Without the snapshot a mid-block chunk stops at its boundary.
    assert resume_at + base <= boundary
    end = resume_at + cross
    if resume_at + budget > boundary:
        # It crosses exactly one boundary, ends on the hash grid, and the
        # worker / manager rule agrees.
        assert boundary < end <= boundary + BLK
        assert end % HASH == 0
        assert _crossing(resume_at, end, prompt_len) == boundary
    else:
        assert cross == base


def test_stop_at_block_end_keeps_a_crossing_chunk() -> None:
    (request,) = create_requests(1, num_tokens=10 * BLK, block_size=HASH)
    stub = _stub(snapshot=True)
    # Crossing chunk [800, 2400): kept, capped one block past the boundary.
    assert sched_mod._stop_at_block_end(stub, request, 800, 1600) == 1600
    assert sched_mod._stop_at_block_end(stub, request, 800, 3000) == 800
    stub.mamba_crossing_chunks = False
    assert sched_mod._stop_at_block_end(stub, request, 800, 1600) == 800


def _manager() -> KVCacheManager:
    config = KVCacheConfig(
        # Production (sparse) retention: crossing chunks need it.
        prefix_cache_retention_interval=0,
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=HASH, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=BLK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=ALIGN,
                    prefill_snapshot=True,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=BLK,
        hash_block_size=HASH,
        enable_caching=True,
        use_eagle=False,
    )


def _prefill(snapshot: bool, prompt_len: int, budgets: list[int], resume_at: int):
    """Chunked prefill through the real split and MambaManager. Simulates the
    kernel: the running column ends at the chunk end; a crossing chunk also
    writes the state at its boundary into the column it resumed from (the
    worker's checkpoint rule, checked against the manager's here)."""
    init_none_hash(sha256)
    manager = _manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=HASH)
    # Resume from a private mid-block state (an earlier turn's checkpoint).
    if resume_at:
        assert manager.allocate_slots(request, resume_at) is not None
        request.num_computed_tokens = resume_at
    state_at: dict[int, int] = {}
    crossings = 0
    for step in range(len(budgets) + 64):
        start = request.num_computed_tokens
        if start >= prompt_len:
            break
        budget = budgets[step] if step < len(budgets) else 2400
        num_new = _split(request, min(prompt_len - start, budget), snapshot)
        if num_new == 0:
            continue
        end = start + num_new
        assert manager.allocate_slots(request, num_new) is not None
        blocks = mamba.req_to_blocks[request.request_id]
        # Worker rule (compute_mamba_prefill_checkpoints) for this row.
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end],
            [num_new],
            hash_block_size=HASH,
            mamba_block_size=BLK,
            checkpoint_alignment=ALIGN,
            drop_eagle_block=False,
            cache_hit_alignment_tokens=HASH,
            first_chunk=[False],
            prefill_ends=[prompt_len],
        )
        boundary = _crossing(start, end, prompt_len) if snapshot else 0
        if boundary:
            crossings += 1
            assert offsets == [boundary - start]
            assert cols == [(start - 1) // BLK]
            state_at[blocks[cols[0]].block_id] = boundary
        elif offsets[0]:
            state_at[blocks[cols[0]].block_id] = start + offsets[0]
        request.num_computed_tokens = end
        running = cdiv(end, BLK) - 1
        state_at[blocks[running].block_id] = end
        manager.new_step_starts()
    assert request.num_computed_tokens == prompt_len
    cached = {}
    for block in mamba.req_to_blocks[request.request_id]:
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"block {block.block_id} hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        cached[claimed] = block.block_hash
    return cached, crossings


@pytest.mark.parametrize("resume_at", [0, 800, 2416])
@pytest.mark.parametrize(
    "budgets", [[2400, 2400, 2400], [700, 2000, 1500, 3000], [1000] * 12]
)
def test_crossing_prefill_publishes_the_same_exact_boundaries(resume_at, budgets):
    """Every cached Mamba block holds the state its hash claims, and crossing
    chunks publish exactly the boundary hashes block-ended chunks do."""
    prompt_len = 6 * BLK + 123
    aligned, n0 = _prefill(False, prompt_len, budgets, resume_at)
    crossed, n1 = _prefill(True, prompt_len, budgets, resume_at)
    assert n0 == 0
    assert set(crossed) == set(aligned)
    assert all(crossed[k] == aligned[k] for k in aligned)


def test_crossing_actually_happens() -> None:
    _, n = _prefill(True, 6 * BLK + 123, [2400, 2400, 2400], 800)
    assert n > 0


def test_worker_rule_without_prefill_ends_is_unchanged() -> None:
    args = dict(
        hash_block_size=HASH,
        mamba_block_size=BLK,
        checkpoint_alignment=ALIGN,
        drop_eagle_block=False,
        cache_hit_alignment_tokens=HASH,
        first_chunk=[False, False],
    )
    rows = ([2000, 5000], [1200, 1000])
    assert compute_mamba_prefill_checkpoints(*rows, **args) == (
        compute_mamba_prefill_checkpoints(*rows, prefill_ends=None, **args)
    )
    # A prompt-final row keeps the prompt-tail rule even with prefill_ends.
    assert compute_mamba_prefill_checkpoints(
        [2000], [1200], prefill_ends=[2000], **{**args, "first_chunk": [False]}
    ) == compute_mamba_prefill_checkpoints([2000], [1200], **{**args, "first_chunk": [False]})


def test_dp_pull_chunk_keeps_the_crossing() -> None:
    """The DP pull-up path applies the same split and the same stop rule."""
    (request,) = create_requests(1, num_tokens=10 * BLK, block_size=HASH)
    request.num_computed_tokens = 800
    stub = _stub(snapshot=True, budget=4096)
    stub.max_model_len = 262144
    stub.num_sampled_tokens_per_step = 1
    stub.need_mamba_block_aligned_split = True
    stub.num_prefill_lookahead = 1
    stub._mamba_block_aligned_split = (
        lambda r, n: Scheduler._mamba_block_aligned_split(stub, r, n)
    )
    stub._reserve_prefill_lookahead = (
        lambda r, c, n: Scheduler._reserve_prefill_lookahead(stub, r, c, n)
    )
    n = Scheduler._dp_pull_chunk(stub, request, 2000, 2000)
    assert 1600 < 800 + n <= 3200
    stub.mamba_crossing_chunks = False
    assert Scheduler._dp_pull_chunk(stub, request, 2000, 2000) == 800
