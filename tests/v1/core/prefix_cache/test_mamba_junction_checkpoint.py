# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure modes of the Mamba shared-prefix junction stop.

`Scheduler._mamba_block_aligned_split` splits a prefill chunk at
`request.shared_prefix_boundary` (the junction), but
`MambaManager._cache_partial_tail_block` only registers a checkpoint at the
eagle-adjusted prompt tail or at `k*B - H`. When the scheduler stops somewhere
the manager refuses, the split costs a forward pass and buys nothing -- and
because the junction is the *earliest* mandatory stop it can REPLACE a
block-boundary stop, leaving less cached than block-flooring would have.

Every test here drives the real `_mamba_block_aligned_split` and the real
`KVCacheManager`; no chunk boundary is hard-coded.
"""

from types import SimpleNamespace

import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

MAMBA_GROUP_ID = 1
PREFIX = list(range(1, 200_001))


def _manager(block_size: int, hash_block_size: int, num_blocks: int = 8192):
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=num_blocks,
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
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )


def _stub(manager, block_size: int, hash_block_size: int):
    """`self` for the real `Scheduler._mamba_block_aligned_split`."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=1 << 20,
        use_eagle=True,
        hash_block_size=hash_block_size,
        mamba_has_prefill_checkpoint_blocks=False,  # forced False under eagle
        mamba_partial_cache_hit=(
            hash_block_size < block_size
            and manager.coordinator.enable_partial_hash_hits
        ),
    )


def _prefill(manager, stub, request, external: int = 0) -> list[int]:
    """Schedule `request` to completion the way `Scheduler.schedule()` does."""
    blocks, local, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    ends: list[int] = []
    first = True
    while request.num_computed_tokens < request.num_tokens:
        new_local = local if first else 0
        ext = external if first else 0
        start = request.num_computed_tokens + new_local + ext
        if start >= request.num_tokens:
            break
        num_new = Scheduler._mamba_block_aligned_split(
            stub, request, request.num_tokens - start, new_local, ext
        )
        if num_new == 0:
            break
        assert (
            manager.allocate_slots(
                request,
                num_new,
                num_new_computed_tokens=new_local,
                new_computed_blocks=blocks if first else None,
                num_external_computed_tokens=ext,
                has_scheduled_reqs=False,
            )
            is not None
        )
        request.num_computed_tokens = start + num_new
        ends.append(request.num_computed_tokens)
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()
        first = False
    return ends


def _orphaned_full_attention_tail(manager, stub, prompt_len: int):
    """Produce the state a KV connector leaves behind.

    NIXL returns `num_prompt_tokens - 1` external computed tokens for Mamba
    models (`nixl/base_scheduler.py:_get_remote_prefill_token_count`), so the
    producer starts past its own tail stop: FullAttentionManager still registers
    its partial tail at T, the Mamba group registers nothing at T - H.
    """
    producer = make_request(
        "producer", PREFIX[:prompt_len], stub.hash_block_size, sha256
    )
    _prefill(manager, stub, producer, external=prompt_len - 1)
    return producer


def test_junction_stop_registers_a_checkpoint_a_sibling_can_resume_from():
    """The scheduler splits at the junction; the manager must cache there."""
    block_size, hash_block_size = 8, 2
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    assert stub.mamba_partial_cache_hit, "fine-grained hits must be armed"

    _orphaned_full_attention_tail(manager, stub, 13)

    consumer = make_request("consumer", PREFIX[:12] + [-1] * 9, hash_block_size, sha256)
    _, _, junction = manager.get_computed_blocks(consumer)
    assert junction, "expected a junction against the orphaned full-attn tail"
    consumer_ends = _prefill(manager, stub, consumer)
    assert junction in consumer_ends, (
        f"scheduler did not stop at the junction {junction}: {consumer_ends}"
    )

    sibling = make_request("sibling", PREFIX[:12] + [-2] * 9, hash_block_size, sha256)
    _, sibling_hit, _ = manager.get_computed_blocks(sibling)
    assert sibling_hit == junction, (
        f"the chunk stopped at {junction} but nothing was cached there: "
        f"sibling resumes at {sibling_hit}"
    )


def test_junction_stop_never_caches_less_than_block_flooring_would():
    """The junction is the earliest mandatory stop, so it REPLACES the
    block-boundary stop. If the junction is then refused, the request caches
    less than flooring the junction to the block grid would have.
    """
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)

    _orphaned_full_attention_tail(manager, stub, 2020)

    consumer = make_request(
        "consumer", PREFIX[:2016] + [-1] * 584, hash_block_size, sha256
    )
    _, _, junction = manager.get_computed_blocks(consumer)
    assert junction, "expected a junction"
    _prefill(manager, stub, consumer)

    sibling = make_request(
        "sibling", PREFIX[:2016] + [-2] * 584, hash_block_size, sha256
    )
    _, sibling_hit, _ = manager.get_computed_blocks(sibling)
    block_floored = junction // block_size * block_size
    assert sibling_hit >= block_floored, (
        f"junction stop at {junction} cost the block-grid checkpoint: sibling "
        f"resumes at {sibling_hit}, block-flooring would have given "
        f"{block_floored}"
    )


def test_junction_clause_is_load_bearing():
    """Guard against a tautological junction test.

    `test_partial_prefix_cache_hits.py`'s junction test passes unchanged with
    `shared_prefix_boundary = 0`, because the position it checks is also the
    request's own eagle-adjusted prompt tail. This pins the difference: the
    checkpoint must appear ONLY because of the junction.
    """
    block_size, hash_block_size = 512, 32

    def sibling_hit_with(junction_seen: bool) -> int:
        manager = _manager(block_size, hash_block_size)
        stub = _stub(manager, block_size, hash_block_size)
        _orphaned_full_attention_tail(manager, stub, 2020)
        consumer = make_request(
            "consumer", PREFIX[:2016] + [-1] * 584, hash_block_size, sha256
        )
        blocks, local, junction = manager.get_computed_blocks(consumer)
        assert junction, "expected a junction"
        consumer.shared_prefix_boundary = junction if junction_seen else 0
        first = True
        while consumer.num_computed_tokens < consumer.num_tokens:
            new_local = local if first else 0
            start = consumer.num_computed_tokens + new_local
            if start >= consumer.num_tokens:
                break
            num_new = Scheduler._mamba_block_aligned_split(
                stub, consumer, consumer.num_tokens - start, new_local, 0
            )
            if num_new == 0:
                break
            assert (
                manager.allocate_slots(
                    consumer,
                    num_new,
                    num_new_computed_tokens=new_local,
                    new_computed_blocks=blocks if first else None,
                    has_scheduled_reqs=False,
                )
                is not None
            )
            consumer.num_computed_tokens = start + num_new
            _, retained = manager.take_kv_cache_block_copies()
            if retained:
                manager.block_pool.free_blocks(retained)
            manager.new_step_starts()
            first = False
        sibling = make_request(
            "sibling", PREFIX[:2016] + [-2] * 584, hash_block_size, sha256
        )
        return manager.get_computed_blocks(sibling)[1]

    with_junction = sibling_hit_with(True)
    without_junction = sibling_hit_with(False)
    assert with_junction > without_junction, (
        "the junction clause buys nothing: sibling resumes at "
        f"{with_junction} with it and {without_junction} without it, so this "
        "scenario does not exercise the junction at all"
    )


def test_cached_mamba_slot_holds_the_state_its_hash_claims():
    """Accuracy guard (#43559): a slot must never be published under a hash
    claiming a token count it does not hold, or a consumer restores the wrong
    recurrent state and emits wrong tokens.

    Includes a positive control: the same check on a hand-forced bad chunk
    sequence MUST report a violation, otherwise the check is vacuous.
    """
    block_size, hash_block_size = 16, 4

    def violations(chunk_ends: list[int] | None) -> list[str]:
        manager = _manager(block_size, hash_block_size, num_blocks=512)
        stub = _stub(manager, block_size, hash_block_size)
        mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        request = make_request("r", PREFIX[:48], hash_block_size, sha256)
        manager.get_computed_blocks(request)
        state_at: dict[int, int] = {}
        found: list[str] = []
        done = 0

        def step(num_new: int) -> None:
            nonlocal done
            assert (
                manager.allocate_slots(request, num_new, has_scheduled_reqs=False)
                is not None
            )
            done += num_new
            request.num_computed_tokens = done
            # The CoW copy runs with this step and captures source_block as of
            # the END OF THE PREVIOUS step, so propagate before this forward.
            copies, retained = manager.take_kv_cache_block_copies()
            for copy in copies:
                if copy.src_block_id in state_at:
                    state_at[copy.dst_block_id] = state_at[copy.src_block_id]
            blocks = mamba.req_to_blocks[request.request_id]
            running = cdiv(done, block_size) - 1
            if 0 <= running < len(blocks) and not blocks[running].is_null:
                state_at[blocks[running].block_id] = done
            if retained:
                manager.block_pool.free_blocks(retained)
            manager.new_step_starts()
            for pos, block in enumerate(blocks):
                if block.is_null or block.block_hash is None:
                    continue
                claimed = block.block_hash_num_tokens
                if state_at.get(block.block_id) != claimed:
                    found.append(
                        f"slot {pos} (block {block.block_id}) claims state@"
                        f"{claimed} but holds state@{state_at.get(block.block_id)}"
                    )

        if chunk_ends is None:
            while done < request.num_tokens:
                num_new = Scheduler._mamba_block_aligned_split(
                    stub, request, request.num_tokens - done
                )
                if num_new == 0:
                    break
                step(num_new)
        else:
            for end in chunk_ends:
                step(end - done)
        return found

    # Positive control: 20 leaves slot 1 mid-block, 40 moves on to slot 2, so
    # slot 1 is published as state@32 while holding state@20. The real split
    # never emits this -- that is exactly what it exists to prevent.
    assert violations([20, 40, 48]), "detector is vacuous: it missed a known poisoning"

    assert not violations(None), "real split poisoned the mamba prefix cache"


def _junction_configs():
    """(block_size, hash_block_size) pairs the junction path can be armed on.

    `prefix_match_unit` must divide every group's block size
    (`kv_cache_utils.py:705-710`) and must be strictly smaller, or
    `enable_partial_hash_hits` is False and the whole path is inert.
    """
    for block_size in (8, 16, 64, 128, 512, 1536):
        for hash_block_size in (2, 4, 8, 16, 32, 64, 128):
            if hash_block_size < block_size and block_size % hash_block_size == 0:
                yield block_size, hash_block_size


def test_scheduler_never_stops_where_the_manager_refuses():
    """The scheduler's junction stop is always a position the manager caches.

    The narrow claim -- "the scheduler can schedule a number of tokens it thinks
    can be cached, but the cache manager won't agree" -- is a property of the
    pair, not of one configuration, so assert it over the whole armed parameter
    space rather than at a single (block_size, hash_block_size).

    The manager accepts `num_tokens == request.shared_prefix_boundary`, but the
    scheduler stops at `junction // hash_block_size * hash_block_size`. Those
    agree only while the junction is a multiple of the hash unit, so pin that
    too: it is what makes the PR's floor a value-level no-op.
    """
    disagreements = []
    misaligned = []
    armed = 0

    for block_size, hash_block_size in _junction_configs():
        for prompt_len in (
            block_size + hash_block_size,
            2 * block_size - hash_block_size,
            2 * block_size + 1,
            3 * block_size + hash_block_size + 1,
            4 * block_size - 1,
        ):
            manager = _manager(block_size, hash_block_size)
            stub = _stub(manager, block_size, hash_block_size)
            if not stub.mamba_partial_cache_hit:
                continue

            _orphaned_full_attention_tail(manager, stub, prompt_len)

            shared = prompt_len // hash_block_size * hash_block_size
            suffix = block_size + 3
            consumer = make_request(
                "consumer", PREFIX[:shared] + [-1] * suffix, hash_block_size, sha256
            )
            _, _, junction = manager.get_computed_blocks(consumer)
            if not junction:
                continue
            armed += 1

            if junction % hash_block_size:
                # The scheduler would floor to a position the manager's
                # equality check can never match.
                misaligned.append((block_size, hash_block_size, prompt_len, junction))

            consumer_ends = _prefill(manager, stub, consumer)
            if junction not in consumer_ends:
                continue

            sibling = make_request(
                "sibling", PREFIX[:shared] + [-2] * suffix, hash_block_size, sha256
            )
            _, sibling_hit, _ = manager.get_computed_blocks(sibling)
            if sibling_hit < junction:
                disagreements.append(
                    (block_size, hash_block_size, prompt_len, junction, sibling_hit)
                )

    assert armed > 100, f"sweep did not arm the junction path: {armed} cases"
    assert not misaligned, (
        f"{len(misaligned)} junctions are not a multiple of the hash unit, so "
        f"the scheduler's floor moves the stop off the manager's checkpoint: "
        f"{misaligned[:5]}"
    )
    assert not disagreements, (
        f"{len(disagreements)} of {armed} configs split the chunk at a junction "
        f"the manager then refused to cache (block_size, hash_block_size, "
        f"prompt_len, junction, sibling_hit): {disagreements[:5]}"
    )


def _consumer_prefill_violations(block_size, hash_block_size, prompt_len):
    """#43559 invariant over a consumer prefill that HAS a junction.

    ``test_cached_mamba_slot_holds_the_state_its_hash_claims`` runs one request
    on a cold cache, so its junction is 0 and it never reaches the position the
    junction clause adds. This drives a real producer/consumer pair instead.

    Blocks inherited from the producer are skipped: their state was established
    by the producer's own prefill, which the single-request guard covers. What
    is checked here is everything the consumer publishes.
    """
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    if not stub.mamba_partial_cache_hit:
        return None, 0

    _orphaned_full_attention_tail(manager, stub, prompt_len)
    shared = prompt_len // hash_block_size * hash_block_size
    consumer = make_request(
        "consumer", PREFIX[:shared] + [-1] * (block_size + 3), hash_block_size, sha256
    )
    precomputed, local, junction = manager.get_computed_blocks(consumer)
    consumer.shared_prefix_boundary = junction
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]

    state_at: dict[int, int] = {}
    found: list[str] = []
    done = 0
    first = True

    while done < consumer.num_tokens:
        new_local = local if first else 0
        start = done + new_local
        if start >= consumer.num_tokens:
            break
        num_new = Scheduler._mamba_block_aligned_split(
            stub, consumer, consumer.num_tokens - start, new_local, 0
        )
        if num_new == 0:
            break
        if (
            manager.allocate_slots(
                consumer,
                num_new,
                num_new_computed_tokens=new_local,
                new_computed_blocks=precomputed if new_local else None,
                num_external_computed_tokens=0,
                has_scheduled_reqs=False,
            )
            is None
        ):
            break
        done = start + num_new
        consumer.num_computed_tokens = done

        copies, retained = manager.take_kv_cache_block_copies()
        for copy in copies:
            if copy.src_block_id in state_at:
                state_at[copy.dst_block_id] = state_at[copy.src_block_id]
        blocks = mamba.req_to_blocks[consumer.request_id]
        running = cdiv(done, block_size) - 1
        if 0 <= running < len(blocks) and not blocks[running].is_null:
            state_at[blocks[running].block_id] = done
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()

        for pos, block in enumerate(blocks):
            if block.is_null or block.block_hash is None:
                continue
            held = state_at.get(block.block_id)
            if held is None:  # inherited from the producer
                continue
            if held != block.block_hash_num_tokens:
                found.append(
                    f"block_size={block_size} hash={hash_block_size} "
                    f"prompt_len={prompt_len} junction={junction}: slot {pos} "
                    f"claims state@{block.block_hash_num_tokens} holds state@{held}"
                )
        first = False

    return found, junction


def test_junction_registration_never_publishes_a_wrong_state():
    """Accuracy guard for the position the junction clause adds.

    The fix makes the manager cache somewhere it previously refused, which is
    exactly the shape of a cache-poisoning bug (#43559): a slot published under
    a hash claiming N tokens while holding state after M != N.
    """
    violations: list[str] = []
    armed = 0

    for block_size, hash_block_size in _junction_configs():
        for prompt_len in (
            block_size + hash_block_size,
            2 * block_size - hash_block_size,
            2 * block_size + 1,
            3 * block_size + hash_block_size + 1,
            4 * block_size - 1,
        ):
            found, junction = _consumer_prefill_violations(
                block_size, hash_block_size, prompt_len
            )
            if found is None:
                continue
            if junction:
                armed += 1
            violations.extend(found)

    assert armed > 100, f"sweep never produced a junction: {armed}"
    assert not violations, (
        f"{len(violations)} cached slots hold a state their hash does not "
        f"claim: {violations[:5]}"
    )


def test_the_junction_clause_only_ever_registers_at_the_chunk_end():
    """Why the above is safe, pinned directly.

    The tail gate runs at a chunk END, where the forward for those tokens has
    just run, so the running block genuinely holds ``state@num_tokens``. If a
    junction claim away from the chunk end were honoured, the clause could
    publish a slot whose state is older than its hash claims.
    """
    accepted: list[str] = []
    tried = 0

    for block_size, hash_block_size, prompt_len in (
        (16, 4, 33),
        (16, 4, 63),
        (64, 16, 129),
        (64, 8, 255),
        (128, 32, 257),
        (512, 32, 2020),
    ):
        manager = _manager(block_size, hash_block_size)
        stub = _stub(manager, block_size, hash_block_size)
        if not stub.mamba_partial_cache_hit:
            continue
        _orphaned_full_attention_tail(manager, stub, prompt_len)
        shared = prompt_len // hash_block_size * hash_block_size
        consumer = make_request(
            "consumer",
            PREFIX[:shared] + [-1] * (block_size + 3),
            hash_block_size,
            sha256,
        )
        ends = _prefill(manager, stub, consumer)
        if not ends:
            continue
        chunk_end = ends[-1]
        mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        blocks = mamba.req_to_blocks[consumer.request_id]

        for claim in range(hash_block_size, 4 * block_size, hash_block_size):
            if claim % block_size == 0 or claim == chunk_end:
                continue
            tried += 1
            consumer.shared_prefix_boundary = claim
            before = [b.block_hash_num_tokens for b in blocks if not b.is_null]
            mamba._cache_partial_tail_block(consumer, claim)
            after = [b.block_hash_num_tokens for b in blocks if not b.is_null]
            if before != after:
                accepted.append(
                    f"block_size={block_size}: chunk ended at {chunk_end} but a "
                    f"junction claim at {claim} was registered"
                )

    assert tried > 100, f"control did not exercise enough claims: {tried}"
    assert not accepted, (
        f"{len(accepted)} junction claims away from the chunk end were "
        f"registered: {accepted[:5]}"
    )
