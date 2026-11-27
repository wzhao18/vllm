# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A same-step connector load allocates the external hit and the admission
step's new tokens in one allocate_slots call. Every such first allocation,
including the first chunk of a chunked prompt (load_same_step_chunked), must
take no more blocks than the free-block check counted, or a nearly full pool
raises "Cannot get 1 free blocks from the pool" (MCSTEP r4, c576 DEP4 prefill).

The sweep uses the DEP prefill's shapes: 16384-token Mamba (KDA) and packed
attention blocks, 128-token prefix hits, the scheduler's real chunk split
(budget, adaptive long-prefill threshold, extra tail checkpoint 1024), internal
snapshots on and off, speculative state blocks, and local prefix hits in front
of the external hit."""

import contextlib
from types import SimpleNamespace

import pytest

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    make_full_mamba_manager,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

HASH = 128  # --prefix-match-unit
BLOCK = 16384  # DEP prefill Mamba / packed attention block


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _split(manager, request, num_new, local, external, budget, snapshot):
    stub = SimpleNamespace(
        block_size=BLOCK,
        mamba_block_size=BLOCK,
        cache_config=SimpleNamespace(
            block_size=BLOCK, mamba_extra_tail_checkpoint_min_span=0
        ),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=budget),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=16,
        mamba_internal_snapshot=snapshot,
        mamba_extra_tail_checkpoint_tokens=1024,
        kv_cache_manager=manager,
    )
    return Scheduler._mamba_block_aligned_split(
        stub, request, num_new, local, external
    )


def _request(rid, num_tokens):
    return make_request(
        rid,
        list(range(num_tokens)),
        HASH,
        sha256,
        sampling_params=SamplingParams(max_tokens=1),
    )


def _prefill_and_free(manager, request, snapshot):
    """Producer: prefill the whole prompt (caching it), then finish."""
    first = True
    while request.num_computed_tokens < request.num_tokens:
        blocks, local = None, 0
        if first:
            blocks, local = manager.get_computed_blocks(request)[:2]
        start = request.num_computed_tokens + local
        num_new = _split(
            manager, request, request.num_tokens - start, local, 0, BLOCK, snapshot
        )
        assert num_new > 0
        assert manager.allocate_slots(request, num_new, local, blocks) is not None
        request.num_computed_tokens = start + num_new
        request.status = RequestStatus.RUNNING
        manager.new_step_starts()
        first = False
    manager.free(request)
    manager.new_step_starts()


def _same_step_first_allocation(
    *, snapshot, num_spec, full_block, retention, local_len, hit, prompt, budget
):
    """(counted, taken, chunk) for the same-step admission allocation."""
    manager = make_full_mamba_manager(
        dcp_world_size=1,
        hash_block_size=HASH,
        full_block_size=full_block,
        mamba_block_size=BLOCK,
        num_blocks=2048,
        num_speculative_blocks=num_spec,
        num_prefill_checkpoint_blocks=1,
        retention_interval=retention,
        prefill_snapshot=snapshot,
        max_model_len=262144,
    )
    if local_len:
        _prefill_and_free(manager, _request("producer", local_len), snapshot)
    request = _request("r", prompt)
    blocks, local = manager.get_computed_blocks(request)[:2]
    external = hit - local
    if external <= 0:
        return None
    num_new = _split(
        manager, request, min(prompt - hit, budget), local, external, budget, snapshot
    )
    if num_new <= 0:
        return None
    counted: list[int] = []
    orig = manager.coordinator.get_num_blocks_to_allocate

    def spy(*a, **kw):
        n = orig(*a, **kw)
        counted.append(n)
        return n

    manager.coordinator.get_num_blocks_to_allocate = spy
    free = manager.block_pool.get_num_free_blocks()
    new = manager.allocate_slots(
        request,
        num_new,
        num_new_computed_tokens=local,
        new_computed_blocks=blocks,
        num_external_computed_tokens=external,
        delay_cache_blocks=True,
    )
    assert new is not None
    return counted[-1], free - manager.block_pool.get_num_free_blocks(), num_new


def _cases():
    for local_len in (0, 5000, BLOCK, BLOCK + 3000):
        for hit_extra in (128, 1280, BLOCK - 128, BLOCK, BLOCK + 640):
            hit = (local_len + hit_extra) // HASH * HASH
            for tail in (100, 1100, 9000, BLOCK, BLOCK + 5000, 3 * BLOCK):
                yield local_len, hit, hit + tail


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("num_spec", [0, 1])
@pytest.mark.parametrize("full_block", [HASH, BLOCK])
@pytest.mark.parametrize("retention", [None, 0])
@pytest.mark.parametrize("budget", [16384, 8192, 3000])
def test_same_step_first_allocation_is_counted(
    snapshot, num_spec, full_block, retention, budget
):
    bad = []
    n = 0
    for local_len, hit, prompt in _cases():
        r = _same_step_first_allocation(
            snapshot=snapshot,
            num_spec=num_spec,
            full_block=full_block,
            retention=retention,
            local_len=local_len,
            hit=hit,
            prompt=prompt,
            budget=budget,
        )
        if r is None:
            continue
        n += 1
        counted, taken, chunk = r
        if taken > counted:
            bad.append(
                dict(
                    local=local_len,
                    hit=hit,
                    prompt=prompt,
                    chunk=chunk,
                    chunked=hit + chunk < prompt,
                    counted=counted,
                    taken=taken,
                )
            )
    assert n > 0
    assert not bad, bad[:8]


# K3.1 scaled down (tests/v1/core/test_slot_block_pool.py): DSA rows and KDA
# "align" states of 64 tokens, prefix-hash unit 16, the release DSpark draft's
# sliding window (128 tokens) packed as 4 slots per pool block
# (VLLM_PACKED_KV_STATE_GROUPS): the c576 DEP prefill's groups at DCP 1.
S_HASH, S_BLOCK, S_SLOTS, S_WINDOW = 16, 64, 4, 128
S_DRAFT = 2


def _packed_manager(num_rows: int):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    import torch

    config = KVCacheConfig(
        num_blocks=num_rows,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["dsa"],
                MLAAttentionSpec(
                    block_size=S_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["kda"],
                MambaSpec(
                    block_size=S_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=S_HASH,
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=S_HASH,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=S_WINDOW,
                    dcp_sharded=False,
                ),
                blocks_per_pool_block=S_SLOTS,
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=S_HASH,
        scheduler_block_size=S_BLOCK,
    )


def _tokens(offset: int, n: int) -> list[int]:
    return list(range(offset, offset + n))


@pytest.mark.parametrize("settled", [True, False])
@pytest.mark.parametrize("cached_free_slots", [False, True])
@pytest.mark.parametrize(
    "hit,prompt", [(48, 100), (32, 100), (16, 80), (128, 200)]
)
def test_packed_draft_slots_do_not_take_uncounted_pool_blocks(
    monkeypatch, settled, cached_free_slots, hit, prompt
):
    """A same-step admission (external hit + compute, one allocate_slots
    inside BLOCK_FREE_CLOCK.settled()) with the pool holding exactly the
    counted free blocks. The packed draft group has free slots in a held row
    that the step in flight just freed (recent), or that are cached: its
    external slots must not open a new row the count did not include before
    the KDA group takes its new state block (MCSTEP r4: "Cannot get 1 free
    blocks from the pool" in MambaManager.allocate_new_blocks)."""
    from vllm.v1.core import block_pool as block_pool_module
    from vllm.v1.core.block_pool import BlockFreeClock

    clock = BlockFreeClock()
    monkeypatch.setattr(block_pool_module, "BLOCK_FREE_CLOCK", clock)
    manager = _packed_manager(num_rows=64)
    parent = manager.block_pool
    draft_pool = manager.coordinator.single_type_managers[S_DRAFT].block_pool

    # Each a_k keeps a draft row held; the b_k slots in those rows are freed
    # by the step in flight (after its schedule() returned), so they are
    # recent: free slots enough for the new request, none of them settled.
    held = []
    for k in range(3):
        a = make_request(f"a{k}", _tokens(10_000 + 100 * k, 16), S_HASH, sha256)
        b = make_request(f"b{k}", _tokens(20_000 + 100 * k, 48), S_HASH, sha256)
        assert manager.allocate_slots(a, 16, delay_cache_blocks=not cached_free_slots)
        assert manager.allocate_slots(b, 48, delay_cache_blocks=not cached_free_slots)
        a.num_computed_tokens, b.num_computed_tokens = 16, 48
        held.append(b)
    assert len(draft_pool._held) == 3
    clock.step_built()
    clock.tick()
    for b in held:
        manager.free(b)
    assert draft_pool.get_num_free_blocks() == 9 and len(draft_pool._held) == 3

    request = make_request(
        "r",
        _tokens(30_000, prompt),
        S_HASH,
        sha256,
        sampling_params=SamplingParams(max_tokens=1),
    )
    num_new = prompt - hit
    counted: list[int] = []
    orig = manager.coordinator.get_num_blocks_to_allocate

    def spy(*args, **kwargs):
        n = orig(*args, **kwargs)
        counted.append(n)
        return n

    monkeypatch.setattr(manager.coordinator, "get_num_blocks_to_allocate", spy)
    # Leave exactly the counted blocks free.
    need = orig(
        request_id="r",
        num_tokens=hit + num_new,
        new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=hit,
        num_local_computed_tokens=0,
        num_tokens_main_model=hit + num_new,
    )
    filler = parent.get_new_blocks(parent.get_num_free_blocks() - need)
    free = parent.get_num_free_blocks()
    with clock.settled() if settled else contextlib.nullcontext():
        new = manager.allocate_slots(
            request,
            num_new,
            num_external_computed_tokens=hit,
            delay_cache_blocks=True,
        )
    assert new is not None
    assert free - parent.get_num_free_blocks() <= counted[-1]
    parent.free_blocks(filler)
