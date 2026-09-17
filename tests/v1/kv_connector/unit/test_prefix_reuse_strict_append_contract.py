"""Strict-append cache-reuse contracts for Kimi-K3 DCP8 geometry.

This is intentionally isolated from the frozen serving candidate.  It checks
the worker save/lookup contract using the two semantically distinct boundaries:
the last hash-aligned source prefix and its one-PMU EAGLE-safe rewind.
"""

import ast
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    _make_kimi_dcp8_retention_manager,
    drain_boundary_state_offloads,
)
from tests.v1.core.test_prefix_caching import make_request
from tests.v1.kv_connector.unit.test_mooncake_store_hma_e2e import (
    _run_dcp8_aligned_tail_replay,
)
from tests.v1.kv_connector.unit.test_mooncake_store_scheduler import (
    _make_bare_scheduler,
    _make_worker_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreSendingThread,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core import single_type_kv_cache_manager


PMU = 128
MAMBA = 896
FA = 7168


def test_retained_mamba_checkpoint_survives_real_branch_lookup() -> None:
    """A production-selected 64K checkpoint is usable by Mooncake lookup."""
    init_none_hash(sha256)
    prompt_len = 191_147
    branch_lcp = 100_000
    manager = _make_kimi_dcp8_retention_manager(retention_interval=64_512)
    producer = make_request(
        "producer", list(range(prompt_len)), PMU, sha256
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        use_eagle_block_drop=True,
        hash_block_size=PMU,
        mamba_partial_cache_hit=True,
        mamba_fine_grained_prefix_cache=True,
        mamba_has_prefill_checkpoint_blocks=False,
    )

    computed_blocks, num_computed, _ = manager.get_computed_blocks(producer)
    offloads: list[tuple[int, int, int]] = []
    first = True
    while producer.num_computed_tokens < producer.num_tokens:
        remaining = producer.num_tokens - producer.num_computed_tokens
        scheduled = Scheduler._mamba_block_aligned_split(
            scheduler, producer, min(8192, remaining)
        )
        manager.new_step_starts()
        allocated = manager.allocate_slots(
            producer,
            scheduled,
            num_computed if first else 0,
            computed_blocks if first else None,
        )
        first = False
        assert allocated is not None
        producer.num_computed_tokens += scheduled
        offloads.extend(
            drain_boundary_state_offloads(manager).get(producer.request_id, [])
        )

    boundaries = {boundary for _, _, boundary in offloads}
    assert boundaries == {64_512, 129_024, 190_976}
    assert len(offloads) == 3 * len(boundaries)

    manager.free(producer)
    manager.new_step_starts()
    branch = make_request(
        "branch",
        list(range(branch_lcp)) + list(range(1_000_000, 1_008_192)),
        PMU,
        sha256,
    )
    _, local_hit, _ = manager.get_computed_blocks(branch)
    assert local_hit == 64_512

    source_hash_boundary = prompt_len // PMU * PMU
    hit, _, _ = _run_dcp8_aligned_tail_replay(
        prompt_tail=source_hash_boundary,
        replay_boundary=64_512,
        computed_end_tokens=prompt_len,
        append_tokens=8192,
        boundary_state_offloads=offloads,
        branch_lcp=branch_lcp,
    )
    assert hit == 64_512


@pytest.mark.parametrize(
    "use_original_method",
    [True, False],
)
def test_forensic_original_core_method_matches_flashkda_checkpoint_path(
    use_original_method: bool,
) -> None:
    """FlashKDA's reserved checkpoint bypasses the changed fallback branch."""
    init_none_hash(sha256)
    prompt_len = 7_169
    context = nullcontext()
    if use_original_method:
        source = subprocess.check_output(
            [
                "git",
                "show",
                "97049c476437c399fd2d313126677ee3ca1043c0:"
                "vllm/v1/core/single_type_kv_cache_manager.py",
            ],
            text=True,
        )
        tree = ast.parse(source)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MambaManager"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_cache_partial_tail_block"
        )
        namespace = vars(single_type_kv_cache_manager).copy()
        module = ast.fix_missing_locations(ast.Module([method], []))
        exec(compile(module, "<pre-fix>", "exec"), namespace)
        context = patch.object(
            single_type_kv_cache_manager.MambaManager,
            "_cache_partial_tail_block",
            namespace["_cache_partial_tail_block"],
        )

    with context:
        _assert_flashkda_checkpoint_replay(prompt_len)


@pytest.mark.parametrize("prompt_len", [7_169, 7_297])
def test_flashkda_checkpoint_uses_event_provable_eagle_boundary(
    prompt_len: int,
) -> None:
    """The production checkpoint and worker preserve reusable FA proof."""
    _assert_flashkda_checkpoint_replay(prompt_len)


def test_flashkda_checkpoint_replays_observed_long_prompt_shape() -> None:
    """A multi-chunk 206K producer preserves its final reusable checkpoint."""
    _assert_flashkda_checkpoint_replay(206_256, chunk_size=8192)


def _run_observed_warm_store_sequence() -> int:
    manager = _make_kimi_dcp8_retention_manager(
        retention_interval=0,
        num_prefill_checkpoint_blocks=1,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        use_eagle_block_drop=True,
        hash_block_size=PMU,
        mamba_partial_cache_hit=True,
        mamba_fine_grained_prefix_cache=True,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=16,
    )

    def run_turn(request_id: str, prompt_len: int):
        producer = make_request(
            request_id, list(range(prompt_len)), PMU, sha256
        )
        blocks, local_hit, _ = manager.get_computed_blocks(producer)
        first = True
        offloads = []
        while producer.num_computed_tokens < producer.num_tokens:
            uncached = local_hit if first else 0
            remaining = producer.num_tokens - producer.num_computed_tokens - uncached
            scheduled = Scheduler._mamba_block_aligned_split(
                scheduler,
                producer,
                min(8192, remaining),
                num_new_local_computed_tokens=uncached,
            )
            manager.new_step_starts()
            was_first = first
            if first:
                allocated = manager.allocate_slots(
                    producer, scheduled, local_hit, blocks
                )
                first = False
            else:
                allocated = manager.allocate_slots(producer, scheduled)
            assert allocated is not None
            producer.num_computed_tokens += scheduled + (
                local_hit if was_first else 0
            )
            offloads.extend(
                drain_boundary_state_offloads(manager).get(request_id, [])
            )
        manager.free(producer)
        manager.new_step_starts()
        return local_hit, offloads

    seed_hit, seed_offloads = run_turn("seed", 34_305)
    warm_hit, warm_offloads = run_turn("warm", 206_256)
    assert seed_hit == 0
    assert {boundary for _, _, boundary in seed_offloads} == {34_176}
    assert warm_hit == 34_176
    assert {boundary for _, _, boundary in warm_offloads} == {206_080}

    hit, _, _ = _run_dcp8_aligned_tail_replay(
        prompt_tail=206_208,
        replay_boundary=206_080,
        computed_end_tokens=206_256,
        append_tokens=3_473,
        boundary_state_offloads=warm_offloads,
        prior_boundary_state_offloads=[(34_305, seed_offloads)],
    )
    return hit


def test_warm_store_sequence_reuses_latest_flashkda_checkpoint() -> None:
    """After both PUTs finish, the current worker reuses the latest state."""
    init_none_hash(sha256)
    assert _run_observed_warm_store_sequence() == 206_080


def _assert_flashkda_checkpoint_replay(
    prompt_len: int,
    expected_hit: int | None = None,
    chunk_size: int | None = None,
) -> None:
    init_none_hash(sha256)
    manager = _make_kimi_dcp8_retention_manager(
        retention_interval=0,
        num_prefill_checkpoint_blocks=1,
    )
    assert all(
        mamba_manager.drop_eagle_checkpoint_block
        for mamba_manager in manager.coordinator.single_type_managers[:3]
    )
    producer = make_request(
        "producer", list(range(prompt_len)), PMU, sha256
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        use_eagle_block_drop=True,
        hash_block_size=PMU,
        mamba_partial_cache_hit=True,
        mamba_fine_grained_prefix_cache=True,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=16,
    )
    blocks, local_hit, _ = manager.get_computed_blocks(producer)
    first = True
    offloads: list[tuple[int, int, int]] = []
    computed_end = 0
    while producer.num_computed_tokens < producer.num_tokens:
        remaining = producer.num_tokens - producer.num_computed_tokens
        scheduled = Scheduler._mamba_block_aligned_split(
            scheduler,
            producer,
            min(chunk_size or prompt_len, remaining),
        )
        manager.new_step_starts()
        if first:
            allocated = manager.allocate_slots(
                producer, scheduled, local_hit, blocks
            )
            first = False
        else:
            allocated = manager.allocate_slots(producer, scheduled)
        assert allocated is not None
        producer.num_computed_tokens += scheduled
        step_offloads = drain_boundary_state_offloads(manager).get(
            producer.request_id, []
        )
        if step_offloads:
            offloads.extend(step_offloads)
            computed_end = producer.num_computed_tokens
    source_boundary = prompt_len // PMU * PMU
    checkpoint_boundary = (prompt_len - 1) // PMU * PMU - PMU
    boundaries = {num_tokens for _, _, num_tokens in offloads}
    expected_boundaries = (
        {193_536, checkpoint_boundary}
        if prompt_len == 206_256
        else {checkpoint_boundary}
    )
    assert boundaries == expected_boundaries
    assert len(offloads) == 3 * len(expected_boundaries)

    hit, _, _ = _run_dcp8_aligned_tail_replay(
        prompt_tail=source_boundary,
        replay_boundary=checkpoint_boundary,
        computed_end_tokens=computed_end,
        append_tokens=8192,
        boundary_state_offloads=offloads,
    )
    assert hit == (checkpoint_boundary if expected_hit is None else expected_hit)


@pytest.mark.parametrize(
    ("prompt_len", "chunk_size", "expected_hit"),
    [(7_297, None, 0), (206_256, 8192, 193_536)],
)
def test_forensic_base_worker_misses_flashkda_checkpoint_without_attention_proof(
    prompt_len: int,
    chunk_size: int | None,
    expected_hit: int,
) -> None:
    """The base worker stores Mamba E but not its required FA proof at T."""
    old_method = _base_worker_boundary_method()

    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        old_method,
    ):
        _assert_flashkda_checkpoint_replay(
            prompt_len, expected_hit=expected_hit, chunk_size=chunk_size
        )


def test_forensic_base_worker_warm_store_falls_back_to_seed() -> None:
    """Completed old-worker PUTs still leave the latest proof unavailable."""
    init_none_hash(sha256)
    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        _base_worker_boundary_method(),
    ):
        assert _run_observed_warm_store_sequence() == 34_176


def _base_worker_boundary_method():
    source = subprocess.check_output(
        [
            "git",
            "show",
            "8880433fb76c3911910b25f62aad7d8f292c1791:"
            "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KVCacheStoreSendingThread"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_maybe_offload_boundary_states"
    )
    namespace = vars(mooncake_store_worker).copy()
    module = ast.fix_missing_locations(ast.Module([method], []))
    exec(compile(module, "<888>", "exec"), namespace)
    return namespace["_maybe_offload_boundary_states"]


@pytest.mark.parametrize(
    "source_hash_boundary",
    sorted(
        {
            2 * PMU,
            MAMBA - PMU,
            MAMBA,
            MAMBA + PMU,
            FA - PMU,
            FA,
            FA + PMU,
            2 * FA - PMU,
            2 * FA,
            2 * FA + PMU,
            *(18 * FA + offset * PMU for offset in range(-4, 5)),
        }
    ),
)
@pytest.mark.parametrize("append_tokens", [PMU, 8192])
def test_dcp8_strict_append_worker_reuses_eagle_safe_prefix(
    source_hash_boundary: int,
    append_tokens: int,
) -> None:
    """No avoidable loss remains after last-token and EAGLE exclusions."""
    eagle_safe = source_hash_boundary - PMU
    original = KVCacheStoreSendingThread._maybe_offload_boundary_states

    def save_production_boundary(self, req_meta):
        req_meta.boundary_state_offloads = [
            entry
            for entry in req_meta.boundary_state_offloads or ()
            if entry[2] == eagle_safe
        ]
        return original(self, req_meta)

    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        save_production_boundary,
    ):
        hit, eagle_keys, source_keys = _run_dcp8_aligned_tail_replay(
            prompt_tail=source_hash_boundary,
            replay_boundary=eagle_safe,
            computed_end_tokens=source_hash_boundary + 1,
            append_tokens=append_tokens,
        )

    # Keep one recurrent boundary only.  The worker derives the EAGLE peek
    # proof from already-computed attention bytes, without inventing Mamba
    # state or a future hash.
    assert source_keys == {0: 0, 1: 0, 2: 0, 3: 1}
    assert eagle_keys == {
        0: 1,
        1: 1,
        2: 1,
        3: int(eagle_safe % FA == 0),
    }
    assert hit == eagle_safe

    # Raw source prefix is source_hash_boundary + the token excluded for
    # sampling.  Keep the mandatory losses explicit instead of hiding them in
    # a redefined prefix: 1 last token + exactly one PMU EAGLE rewind.
    raw_source_prefix = source_hash_boundary + 1
    assert raw_source_prefix - hit == 1 + PMU
    assert eagle_safe - hit == 0


def test_dcp8_eagle_proof_never_uses_uncomputed_future_bytes() -> None:
    """A full future hash list does not authorize reading future FA bytes."""
    source_hash_boundary = MAMBA + 2 * PMU
    eagle_safe = source_hash_boundary - PMU
    original = KVCacheStoreSendingThread._maybe_offload_boundary_states

    def save_production_boundary(self, req_meta):
        req_meta.boundary_state_offloads = [
            entry
            for entry in req_meta.boundary_state_offloads or ()
            if entry[2] == eagle_safe
        ]
        return original(self, req_meta)

    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        save_production_boundary,
    ):
        hit, eagle_keys, source_keys = _run_dcp8_aligned_tail_replay(
            prompt_tail=source_hash_boundary,
            replay_boundary=eagle_safe,
            # The request metadata contains the T hash, but the event covers
            # only E. Reading T would race the forward that computes it.
            computed_end_tokens=eagle_safe,
            append_tokens=PMU,
        )

    assert source_keys == {0: 0, 1: 0, 2: 0, 3: 0}
    assert eagle_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert hit == 0


def test_dcp8_eagle_proof_requires_hash_coverage() -> None:
    """Computed bytes alone cannot name a proof beyond the hash list."""
    eagle_safe = MAMBA + PMU
    proof_boundary = eagle_safe + PMU
    hit, eagle_keys, proof_keys = _run_dcp8_aligned_tail_replay(
        prompt_tail=proof_boundary,
        replay_boundary=eagle_safe,
        computed_end_tokens=proof_boundary,
        hash_coverage_tokens=eagle_safe,
        append_tokens=PMU,
        boundary_state_offloads=[
            (group_id, 20 + group_id, eagle_safe) for group_id in range(3)
        ],
    )

    # Without the proof hash, retain FA@E instead of dropping it in favor of
    # an impossible FA@T key.
    assert eagle_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert proof_keys == {0: 0, 1: 0, 2: 0, 3: 0}
    assert hit == 0


def test_base_candidate_misses_without_event_covered_eagle_proof() -> None:
    """Negative control: candidate 888 stores E but no FA proof at T."""
    source = subprocess.check_output(
        [
            "git",
            "show",
            "8880433fb76c3911910b25f62aad7d8f292c1791:"
            "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KVCacheStoreSendingThread"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_maybe_offload_boundary_states"
    )
    namespace = vars(mooncake_store_worker).copy()
    module = ast.fix_missing_locations(ast.Module([method], []))
    exec(compile(module, "<888>", "exec"), namespace)
    old_method = namespace["_maybe_offload_boundary_states"]
    source_hash_boundary = 1152
    eagle_safe = 1024

    def save_production_boundary(self, req_meta):
        req_meta.boundary_state_offloads = [
            entry
            for entry in req_meta.boundary_state_offloads or ()
            if entry[2] == eagle_safe
        ]
        return old_method(self, req_meta)

    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        save_production_boundary,
    ):
        hit, _, source_keys = _run_dcp8_aligned_tail_replay(
            prompt_tail=source_hash_boundary,
            replay_boundary=eagle_safe,
            computed_end_tokens=1153,
            append_tokens=8192,
        )

    assert source_keys == {0: 0, 1: 0, 2: 0, 3: 0}
    assert hit == 0


def test_offload_only_metadata_carries_event_covered_compute_extent() -> None:
    """The E handoff job proves T only after the overwriting forward."""
    scheduler = _make_bare_scheduler(
        hash_block_size=PMU,
        enable_partial_hash_hits=True,
    )
    scheduler._store_group_ids = (0, 1, 2, 3)
    scheduler._store_group_id_by_kv_cache_group_id = {
        group_id: group_id for group_id in range(4)
    }
    scheduler._boundary_state_group_ids = frozenset({0, 1, 2})
    scheduler._request_trackers["producer"] = RequestTracker(
        req_id="producer",
        token_len=1153,
        allocated_block_ids=([7], [8], [9], [10]),
        prefill_end_tokens=1153,
    )
    scheduler._unfinished_requests["producer"] = (
        SimpleNamespace(block_hashes=[b"h"] * 10),
        ([7], [8], [9], [10]),
    )
    meta = MooncakeStoreConnectorMetadata(set(), set())

    scheduler._handle_boundary_state_offloads(
        {"producer": [(0, 7, 1024), (1, 8, 1024), (2, 9, 1024)]},
        meta,
    )

    [req_meta] = meta.requests
    assert req_meta.token_len_chunk == 0
    assert req_meta.computed_end_tokens == 1153
    assert req_meta.boundary_state_offloads == [
        (0, 7, 1024),
        (1, 8, 1024),
        (2, 9, 1024),
    ]


def test_eagle_proof_is_not_visible_until_put_completes() -> None:
    """A successor misses while PUT is blocked, then hits after publication."""
    original = KVCacheStoreSendingThread._maybe_offload_boundary_states

    def save_production_boundary(self, req_meta):
        req_meta.boundary_state_offloads = [
            entry
            for entry in req_meta.boundary_state_offloads or ()
            if entry[2] == 1024
        ]
        return original(self, req_meta)

    with patch.object(
        KVCacheStoreSendingThread,
        "_maybe_offload_boundary_states",
        save_production_boundary,
    ):
        hit, eagle_keys, source_keys = _run_dcp8_aligned_tail_replay(
            prompt_tail=1152,
            replay_boundary=1024,
            computed_end_tokens=1153,
            append_tokens=8192,
            block_publication=True,
        )
    assert hit == 1024
    assert eagle_keys == {0: 1, 1: 1, 2: 1, 3: 0}
    assert source_keys == {0: 0, 1: 0, 2: 0, 3: 1}


def test_finished_tail_pins_every_attention_source_until_all_workers_finish() -> None:
    """The exact state and every positional FA source outlive request free."""
    scheduler = _make_bare_scheduler(
        hash_block_size=PMU,
        enable_partial_hash_hits=True,
    )
    scheduler._store_group_ids = (0, 1, 2, 3)
    scheduler._store_group_id_by_kv_cache_group_id = {
        group_id: group_id for group_id in range(4)
    }
    scheduler._boundary_state_group_ids = frozenset({0, 1, 2})
    scheduler._num_workers = 8

    full_attention_ids = [20, 21, 22]
    mamba_ids = [7, 8, 9]
    block_ids = ([1], [2], [3], full_attention_ids)
    request = SimpleNamespace(request_id="req-0", block_hashes=[b"h"] * 64)
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=FA + PMU,
        allocated_block_ids=block_ids,
        token_ids=list(range(FA + PMU)),
        prefill_end_tokens=FA + PMU,
    )

    assert not scheduler.register_finished_partial_tail(
        request,
        block_ids,
        [
            (group_id, block_id, FA)
            for group_id, block_id in enumerate(mamba_ids)
        ],
    )
    assert (
        scheduler._finished_partial_tail_metas["req-0"].computed_end_tokens
        == FA + PMU
    )
    [store_job_id] = scheduler._pinned_saves
    expected = set(mamba_ids + full_attention_ids)
    assert set(scheduler._pinned_saves[store_job_id][0]) == expected
    assert all(scheduler._gpu_block_pool.blocks[i].ref_cnt == 1 for i in expected)

    # Seven rank completions are insufficient; every source remains pinned.
    scheduler.update_connector_output(_make_worker_output({store_job_id: 7}))
    assert store_job_id in scheduler._pinned_saves
    assert all(scheduler._gpu_block_pool.blocks[i].ref_cnt == 1 for i in expected)

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))
    assert store_job_id not in scheduler._pinned_saves
    assert all(scheduler._gpu_block_pool.blocks[i].ref_cnt == 0 for i in expected)
