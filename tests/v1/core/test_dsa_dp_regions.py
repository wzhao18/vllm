# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA-DP per-card DSA KV regions (``v1/core/dsa_dp_regions.py``): the DSA
groups of a request allocate, cache and evict only in its owner card's region,
and a full region blocks only that owner's requests. With
``dsa_dp.PREFIX_OWNER`` the owner is the card holding the longest
prefix, over unsalted hashes."""

import functools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import dsa_dp
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import dsa_dp_regions
from vllm.v1.core.dsa_dp_regions import DSARegionPools, region_sizes
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlockCopy,
    get_kv_cache_capacity,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .test_prefix_caching import make_kv_cache_manager, make_request
from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test

BS = 16
S = 41  # shared ids [0, 41), id 0 is the null block
D = 8  # per-owner region
TP = 2
DSA_GROUP = 0
DRAFT_GROUP = 1


def _mla(is_draft: bool = False) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        is_draft=is_draft,
    )


def _config(regions: bool = True) -> KVCacheConfig:
    cfg = KVCacheConfig(
        num_blocks=S if regions else S + D,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["dsa.attn"], _mla()),
            KVCacheGroupSpec(["draft.attn"], _mla(is_draft=True)),
        ],
    )
    if regions:
        cfg = replace(
            cfg,
            dsa_dp_group_ids=(DSA_GROUP,),
            dsa_dp_shared_blocks=S,
            dsa_dp_region_blocks=D,
            dsa_dp_num_regions=TP,
        )
    return cfg


@pytest.fixture(autouse=True)
def _hash():
    init_none_hash(sha256)


def _manager(regions: bool = True):
    return make_kv_cache_manager(
        _config(regions), max_model_len=1024, hash_block_size=BS
    )


def _req(rid: str, owner: int, num_tokens: int, salt: str | None = None):
    req = make_request(rid, list(range(num_tokens)), BS, sha256, cache_salt=salt)
    req.dsa_owner = owner
    return req


def _region(block_id: int) -> int:
    return -1 if block_id < S else (block_id - S) // D


def _alloc(manager, req, num_tokens: int):
    blocks, hit = manager.get_computed_blocks(req)[:2]
    out = manager.allocate_slots(req, num_tokens - hit, hit, blocks)
    return out, hit


def test_region_allocation_and_free():
    manager = _manager()
    regions = manager.coordinator.dsa_regions
    assert isinstance(regions, DSARegionPools)
    assert manager.block_pool.num_gpu_blocks == S
    a, b = _req("a", 0, 3 * BS), _req("b", 1, 2 * BS)
    assert _alloc(manager, a, 3 * BS)[0] is not None
    assert _alloc(manager, b, 2 * BS)[0] is not None
    ids_a = manager.get_block_ids("a")
    ids_b = manager.get_block_ids("b")
    assert {_region(i) for i in ids_a[DSA_GROUP]} == {0}
    assert {_region(i) for i in ids_b[DSA_GROUP]} == {1}
    # The drafter's blocks stay in the shared region.
    assert all(0 < i < S for i in ids_a[DRAFT_GROUP] + ids_b[DRAFT_GROUP])
    assert regions.get_num_free_blocks_of(0) == D - 3
    assert regions.get_num_free_blocks_of(1) == D - 2
    assert manager.block_pool.get_num_free_blocks() == S - 1 - 5

    manager.free(a)
    manager.free(b)
    assert regions.get_num_free_blocks_of(0) == D
    assert regions.get_num_free_blocks_of(1) == D
    assert manager.block_pool.get_num_free_blocks() == S - 1


def test_prefix_hit_and_eviction_per_owner():
    manager = _manager()
    regions = manager.coordinator.dsa_regions
    # Owner salting (scheduler._assign_dsa_owner) keeps the hashes apart.
    a = _req("a", 1, 4 * BS, salt="|dsadp1")
    _alloc(manager, a, 4 * BS)
    a.num_computed_tokens = 4 * BS
    a_dsa = manager.get_block_ids("a")[DSA_GROUP]
    manager.free(a)

    # Same owner, same prefix: hits the owner's region blocks.
    b = _req("b", 1, 4 * BS + 1, salt="|dsadp1")
    _, hit = _alloc(manager, b, 4 * BS + 1)
    assert hit == 4 * BS
    assert manager.get_block_ids("b")[DSA_GROUP][:4] == a_dsa
    manager.free(b)

    # The region pools are separate: owner 0 never sees owner 1's blocks,
    # even with identical hashes.
    c = _req("c", 0, 4 * BS + 1, salt="|dsadp1")
    _, hit = _alloc(manager, c, 4 * BS + 1)
    assert hit == 0
    assert {_region(i) for i in manager.get_block_ids("c")[DSA_GROUP]} == {0}
    manager.free(c)

    # Filling owner 0's region evicts nothing of owner 1's cache.
    d = _req("d", 0, D * BS)
    assert _alloc(manager, d, D * BS)[0] is not None
    manager.free(d)
    e = _req("e", 1, 4 * BS + 1, salt="|dsadp1")
    _, hit = _alloc(manager, e, 4 * BS + 1)
    assert hit == 4 * BS
    manager.free(e)

    # Filling owner 1's region evicts owner 1's cached blocks.
    f = _req("f", 1, D * BS)
    assert _alloc(manager, f, D * BS)[0] is not None
    manager.free(f)
    g = _req("g", 1, 4 * BS + 1, salt="|dsadp1")
    _, hit = _alloc(manager, g, 4 * BS + 1)
    assert hit == 0
    assert regions.regions[1].get_num_free_blocks() < D


def test_full_region_fails_only_its_owner():
    manager = _manager()
    a = _req("a", 0, D * BS)
    assert _alloc(manager, a, D * BS)[0] is not None
    b = _req("b", 0, BS)
    assert _alloc(manager, b, BS)[0] is None
    assert manager.dsa_region_full_owner == 0
    c = _req("c", 1, BS)
    assert _alloc(manager, c, BS)[0] is not None
    assert manager.dsa_region_full_owner is None


def _scheduler(tp: int = 1, **kwargs) -> Scheduler:
    base = create_scheduler(enable_prefix_caching=True, block_size=BS, **kwargs)
    if tp > 1:
        base.vllm_config.parallel_config.tensor_parallel_size = tp
    cfg = _config()
    base.vllm_config.cache_config.num_gpu_blocks = cfg.num_blocks
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=cfg,
        block_size=BS,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def _sched_req(rid: str, owner: int, num_tokens: int):
    req = make_request(
        rid,
        [hash(rid) % 1000 + i for i in range(num_tokens)],
        BS,
        sha256,
        sampling_params=SamplingParams(max_tokens=4),
    )
    req.dsa_owner = owner
    return req


def test_scheduler_full_region_skips_only_its_owner():
    sched = _scheduler()
    # Owner 0: a fills the region, b (behind it in FCFS order) cannot fit.
    # Owner 1: c is behind b and must still be admitted.
    a = _sched_req("a", 0, D * BS - 1)
    b = _sched_req("b", 0, 2 * BS)
    c = _sched_req("c", 1, 2 * BS)
    for r in (a, b, c):
        sched.add_request(r)
    out = sched.schedule()
    scheduled = {r.req_id for r in out.scheduled_new_reqs}
    assert scheduled == {"a", "c"}
    assert b.status == RequestStatus.WAITING


def test_scheduler_preempts_same_owner():
    sched = _scheduler()
    # Both owners run; owner 0's region then runs out while decoding.
    a = _sched_req("a", 0, 4 * BS)
    b = _sched_req("b", 0, 4 * BS)
    c = _sched_req("c", 1, 2 * BS)
    for r in (c, a, b):
        sched.add_request(r)
    out = sched.schedule()
    assert {r.req_id for r in out.scheduled_new_reqs} == {"a", "b", "c"}
    ids = [r.request_id for r in sched.running]
    sampled = [[1] for _ in ids]
    sched.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    # a and b each need a 5th block in region 0 (8 blocks, 8 held): the
    # last owner-0 request (b) is preempted, never owner 1's c.
    out = sched.schedule()
    assert b.status == RequestStatus.PREEMPTED
    assert c.status == RequestStatus.RUNNING
    assert a.status == RequestStatus.RUNNING


def test_region_sizes():
    # No shared token group: every page but the null block and the reserve
    # goes to the DSA region.
    assert region_sizes(1001, 4, 1, 0, 100) == (101, 900)
    # One shared token group (the drafter) at TP4: shared / DSA = 4 : 1.
    s, d = region_sizes(1001, 4, 1, 1, 0)
    assert (s, d) == (801, 200)
    # Explicit override.
    assert region_sizes(1001, 4, 1, 1, 0, region_blocks_override=300) == (701, 300)


def _dsa_dp_layout(monkeypatch):
    """A DSA-DP layout for apply_dsa_dp_regions (which configures the flags from
    the config); every flag it sets is restored after the test."""
    monkeypatch.setattr(dsa_dp, "supported", lambda vllm_config: True)
    for flag in ("ENABLED", "KV_ENABLED", "PREFIX_OWNER", "SCATTER_IN"):
        monkeypatch.setattr(dsa_dp, flag, getattr(dsa_dp, flag))


def test_apply_regions_sizes_configs(monkeypatch):
    _dsa_dp_layout(monkeypatch)
    groups = [
        KVCacheGroupSpec(["dsa.attn"], _mla()),
        KVCacheGroupSpec(
            ["kda"],
            MambaSpec(block_size=BS, shapes=((1, 1),), dtypes=(torch.float32,)),
        ),
        KVCacheGroupSpec(["draft.attn"], _mla(is_draft=True)),
    ]
    workers = [
        KVCacheConfig(num_blocks=1001, kv_cache_tensors=[], kv_cache_groups=groups)
        for _ in range(4)
    ]
    sched_cfg = KVCacheConfig(
        num_blocks=1001, kv_cache_tensors=[], kv_cache_groups=groups
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=10),
    )
    # The draft group's merged spec has lost ``is_draft``; the per-layer specs
    # decide.
    groups[2].kv_cache_spec = _mla()
    layer_specs = {
        "dsa.attn": _mla(),
        "kda": groups[1].kv_cache_spec,
        "draft.attn": _mla(is_draft=True),
    }
    dsa_dp_regions.apply_dsa_dp_regions(
        vllm_config, [layer_specs] * 4, workers, sched_cfg
    )
    # reserve = 10 seqs x 1 Mamba group x 6; (1001 - 1 - 60) / 5 = 188.
    for cfg in workers:
        assert cfg.num_blocks == 1001
        assert cfg.dsa_dp_group_ids == (0,)
        assert (cfg.dsa_dp_shared_blocks, cfg.dsa_dp_region_blocks) == (813, 188)
    assert sched_cfg.num_blocks == 813
    assert sched_cfg.dsa_dp_num_regions == 4


def test_apply_regions_rejects_mixed_group(monkeypatch):
    """A group holding DSA and draft layers cannot go to the regions."""
    _dsa_dp_layout(monkeypatch)
    mixed = UniformTypeKVCacheSpecs.from_specs(
        {"dsa.attn": _mla(), "draft.attn": _mla(is_draft=True)}
    )
    assert mixed is not None
    cfg = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["dsa.attn", "draft.attn"], mixed)],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    with pytest.raises(ValueError, match="mixes DSA"):
        dsa_dp_regions.apply_dsa_dp_regions(
            vllm_config,
            [{"dsa.attn": _mla(), "draft.attn": _mla(is_draft=True)}],
            [cfg],
            replace(cfg),
        )


def test_block_id_translation():
    cfg = _config()
    # Card 1 backs region 1 [S + D, S + 2D) at local pages [S, S + D).
    assert dsa_dp.to_local_block_id(3, S, D, 1) == 3
    assert dsa_dp.to_local_block_id(S + D, S, D, 1) == S
    assert dsa_dp.to_local_block_id(S + 2 * D - 1, S, D, 1) == S + D - 1
    assert dsa_dp.to_local_block_id(S, S, D, 1) == -1  # card 0's region
    assert dsa_dp.to_local_block_id(S + 1, S, D, 0) == S + 1
    assert dsa_dp.local_block_ids([0, 5, S, S + D + 2], cfg, 1) == [0, 5, S + 2]
    copies = [
        KVCacheBlockCopy(S + D, S + D + 1, DSA_GROUP, 0),
        KVCacheBlockCopy(S, S + 1, DSA_GROUP, 0),
        KVCacheBlockCopy(2, 3, DRAFT_GROUP, 0),
    ]
    assert dsa_dp.local_block_copies(copies, cfg, 1) == [
        KVCacheBlockCopy(S, S + 1, DSA_GROUP, 0),
        KVCacheBlockCopy(2, 3, DRAFT_GROUP, 0),
    ]

    # Block tables and slot mappings of the DSA group only.
    bt_dsa = torch.tensor([[S + D, S + D + 3, 0]], dtype=torch.int32)
    bt_draft = torch.tensor([[1, 2, 0]], dtype=torch.int32)
    slots = torch.tensor([[(S + D) * BS + 5, -1], [2 * BS + 5, -1]], dtype=torch.int64)
    tables, out = dsa_dp.localize_tables((bt_dsa, bt_draft), slots, cfg, 1)
    assert tables[DSA_GROUP].tolist() == [[S, S + 3, 0]]
    assert tables[DRAFT_GROUP].tolist() == [[1, 2, 0]]
    assert out.tolist() == [[S * BS + 5, -1], [2 * BS + 5, -1]]
    # Card 0: region 0 ids are already local.
    tables, _ = dsa_dp.localize_tables(
        (torch.tensor([[S, S + 7]]), None), (None, None), cfg, 0
    )
    assert tables[DSA_GROUP].tolist() == [[S, S + 7]]


def test_gate_off_allocator_unchanged():
    manager = _manager(regions=False)
    assert manager.coordinator.dsa_regions is None
    assert manager.coordinator.dsa_group_ids == frozenset()
    a = _req("a", 1, 3 * BS)
    assert _alloc(manager, a, 3 * BS)[0] is not None
    ids = manager.get_block_ids("a")
    # Stock pool: both groups draw from the one pool, ids 1.. in order.
    assert sorted(ids[DSA_GROUP] + ids[DRAFT_GROUP]) == list(range(1, 7))
    assert manager.dsa_region_full_owner is None
    assert not dsa_dp.KV_ENABLED


def test_store_block_id_round_trip():
    """The Mooncake store maps an owned request's DSA block ids to the owner's
    pages and reports failed loads back in scheduler ids."""
    cfg = _config()
    ids = [0, S + D, S + 2 * D - 1]
    local = dsa_dp.region_block_ids(ids, cfg, 1)
    assert local == [0, S, S + D - 1]
    assert dsa_dp.region_block_ids(local, cfg, 1, to_global=True) == ids
    with pytest.raises(AssertionError):
        dsa_dp.region_block_ids([S], cfg, 1)


# Prefix-hit owner choice (dsa_dp.PREFIX_OWNER).


@pytest.fixture
def prefix_owner(monkeypatch):
    monkeypatch.setattr(dsa_dp, "ENABLED", True)
    monkeypatch.setattr(dsa_dp, "KV_ENABLED", True)
    monkeypatch.setattr(dsa_dp, "PREFIX_OWNER", True)


class _CountingHasher:
    def __init__(self) -> None:
        self.inner = get_request_block_hasher(BS, sha256)
        self.calls = 0

    def __call__(self, request: Request):
        self.calls += 1
        return self.inner(request)


def _toks(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


def _turn(rid: str, tokens: list[int], hasher=None) -> Request:
    """A request as AgentX sends it: no session_id, one output token."""
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=hasher or get_request_block_hasher(BS, sha256),
    )


def _next_turn(rid: str, prev: Request, start: int, n_new: int = 15) -> Request:
    return _turn(rid, list(prev.all_token_ids) + _toks(start, n_new))


def _step(sched: Scheduler) -> None:
    out = sched.schedule()
    ids = list(out.num_scheduled_tokens)
    sched.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[
                [] if sched.requests[r].is_prefill_chunk else [7] for r in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _run(sched: Scheduler, reqs: list[Request]) -> None:
    for _ in range(50):
        if all(r.is_finished() for r in reqs):
            return
        _step(sched)
    raise AssertionError("requests did not finish")


def _hit(sched: Scheduler, request: Request) -> int:
    return sched.kv_cache_manager.get_computed_blocks(request)[1]


def _expected_load(sched: Scheduler) -> list[int]:
    load = [0] * TP
    for r in sched.requests.values():
        rem = r.num_prompt_tokens - r.num_computed_tokens
        if rem > 0 and not r.is_finished():
            load[r.dsa_owner] += int(dsa_dp.chunk_cost(rem, r.num_computed_tokens))
    return load


def test_prefix_owner_ignores_gpu_prefix_hits(prefix_owner):
    """Owners are chosen by load like DEP ranks: a card holding a request's
    prefix does not attract it (the store serves the prefix to any owner)."""
    sched = _scheduler(tp=TP)
    a1 = _turn("a1", _toks(1000, 65))
    sched.add_request(a1)
    assert a1.dsa_owner == 0
    _run(sched, [a1])
    # Idle cards take turns: b on card 1, then the filler on card 0.
    b = _turn("b", _toks(5000, 65))
    sched.add_request(b)
    assert b.dsa_owner == 1
    _run(sched, [b])
    filler = _turn("filler", _toks(9000, 65))
    sched.add_request(filler)
    assert filler.dsa_owner == 0
    assert sched._dsa_load[0] > sched._dsa_load[1] == 0
    # Card 0 holds a2's prefix, but card 1 is less loaded.
    a2 = _next_turn("a2", a1, 2000)
    sched.add_request(a2)
    assert a2.dsa_owner == 1
    assert a2.cache_salt is None


def test_prefix_owner_equal_hits_go_to_least_loaded(prefix_owner):
    sched = _scheduler(tp=TP)
    prompt = _toks(1000, 65)
    # The same prompt twice before either runs: card 1 is less loaded for the
    # second, and both regions end up caching it.
    p1, p2 = _turn("p1", prompt), _turn("p2", prompt)
    for r in (p1, p2):
        sched.add_request(r)
    assert (p1.dsa_owner, p2.dsa_owner) == (0, 1)
    _run(sched, [p1, p2])
    filler = _turn("filler", _toks(9000, 65))
    sched.add_request(filler)
    assert filler.dsa_owner == 0
    q = _next_turn("q", p1, 2000)
    sched.add_request(q)
    assert q.dsa_owner == 1
    assert _hit(sched, q) == 64


def test_prefix_owner_hashes_do_not_depend_on_owner(prefix_owner):
    sched = _scheduler(tp=TP)
    prompt = _toks(1000, 65)
    p1, p2 = _turn("p1", prompt), _turn("p2", prompt)
    for r in (p1, p2):
        sched.add_request(r)
    assert p1.dsa_owner != p2.dsa_owner
    ref = _turn("ref", prompt).block_hashes
    assert len(ref) == 4
    assert p1.block_hashes == p2.block_hashes == ref
    assert p1.cache_salt is None and p2.cache_salt is None


def test_prefix_owner_hashes_prompt_once(prefix_owner):
    sched = _scheduler(tp=TP)
    a1 = _turn("a1", _toks(1000, 65))
    sched.add_request(a1)
    _run(sched, [a1])
    hasher = _CountingHasher()
    a2 = _turn("a2", list(a1.all_token_ids) + _toks(2000, 31), hasher=hasher)
    assert hasher.calls == 1
    sched.add_request(a2)
    # Owner choice and admission reuse the request's hashes: no rehash.
    assert hasher.calls == 1


def test_prefix_owner_load_is_incremental(prefix_owner, monkeypatch):
    monkeypatch.setattr(dsa_dp, "CHUNK", 2 * BS)
    sched = _scheduler(tp=TP, long_prefill_token_threshold=2 * BS)
    reqs = [_turn(f"r{i}", _toks(1000 * (i + 1), 3 * BS + 1)) for i in range(3)]
    for r in reqs:
        sched.add_request(r)
        assert sched._dsa_load == _expected_load(sched)
    assert [r.dsa_owner for r in reqs] == [0, 1, 0]
    _step(sched)  # partial prefills
    assert any(0 < r.num_computed_tokens < r.num_prompt_tokens for r in reqs)
    assert sched._dsa_load == _expected_load(sched)
    victim = sched.running[-1]
    sched.running.remove(victim)
    sched._preempt_request(victim, 0.0)
    assert victim.num_computed_tokens == 0
    assert sched._dsa_load == _expected_load(sched)
    sched.finish_requests(reqs[0].request_id, RequestStatus.FINISHED_ABORTED)
    assert sched._dsa_load == _expected_load(sched)
    _run(sched, reqs[1:])
    assert sched._dsa_load == [0] * TP


def test_prefix_owner_gate_off_keeps_salted_owner(monkeypatch):
    monkeypatch.setattr(dsa_dp, "ENABLED", True)
    monkeypatch.setattr(dsa_dp, "KV_ENABLED", True)
    monkeypatch.setattr(dsa_dp, "PREFIX_OWNER", False)
    sched = _scheduler(tp=TP)
    assert sched._dsa_load is None
    hasher = _CountingHasher()
    prompt = _toks(1000, 65)
    r = _turn("r", prompt, hasher=hasher)
    sched.add_request(r)
    # Phase 1 owner choice: one salted rehash per candidate card plus the final.
    assert r.cache_salt == f"|dsadp{r.dsa_owner}"
    assert hasher.calls == 1 + TP + 1
    assert r.block_hashes != _turn("ref", prompt).block_hashes


def test_store_region_group_keys_owner_independent():
    """Mooncake store: the region group (pure MLA) uses the TP-rank-0 key
    namespace on every card, so with unsalted hashes a chunk saved by one owner
    is found and loaded by any other; KDA keys stay per rank."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        KeyMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
        MooncakeStoreWorker,
    )

    region = UniformTypeKVCacheSpecs.from_specs(
        {"dsa.attn": _mla(), "draft.attn": _mla(is_draft=True)}
    )
    kda = MambaSpec(block_size=BS, shapes=((1, 1),), dtypes=(torch.float32,))
    groups = [
        KVCacheGroupSpec(["dsa.attn", "draft.attn"], region),
        KVCacheGroupSpec(["kda"], kda),
    ]
    block_hash = BlockHash(b"\x01" * 32)
    keys = []
    for rank in range(4):
        w = SimpleNamespace(
            dcp_size=1,
            tp_size=4,
            num_kv_head=1,
            tp_rank=rank,
            hash_block_size=BS,
            store_tp_size=None,
            _kv_cache_groups=groups,
        )
        w._spec_tp_replication_factor = functools.partial(
            MooncakeStoreWorker._spec_tp_replication_factor, w
        )
        w._group_tp_replication_factors = (
            MooncakeStoreWorker._compute_group_tp_replication_factors(w)
        )
        dbs = MooncakeStoreWorker._build_token_databases(
            w, KeyMetadata("m", rank, 0, 0, 0), None
        )
        keys.append([db.store_layout.key_for(0, block_hash) for db in dbs])
    assert len({k[0] for k in keys}) == 1
    assert len({k[1] for k in keys}) == 4


@pytest.mark.parametrize("rank", range(4))
def test_mooncake_dsa_owner_from_region_block_ids(rank, monkeypatch):
    """The Mooncake worker can see a request first in an async load, before
    the model runner has its owner: the owner must come from the region of
    its DSA block ids. Boundary puts of a DSA block go only to the owner card,
    at its local page (the other cards do not hold the KV)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
        KVCacheStoreSendingThread,
        MooncakeStoreWorker,
    )

    monkeypatch.setattr(dsa_dp, "ENABLED", True)
    monkeypatch.setattr(dsa_dp, "DSA_LAYER_NAMES", {"dsa.attn"})
    monkeypatch.setattr(dsa_dp, "REQ_OWNER", {})
    s, d, tp, owner = 41, 8, 4, 2
    db = SimpleNamespace(
        block_size=BS,
        key_for=lambda h: f"k{h}",
        metadata=None,
        prepare_value_for_block=lambda b: ([b], [1]),
    )
    worker = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        dcp_size=1,
        _kv_cache_groups=[SimpleNamespace(layer_names=["dsa.attn"])],
        _dsa_kv_cfg=SimpleNamespace(dsa_dp_shared_blocks=s, dsa_dp_region_blocks=d),
        coord=SimpleNamespace(hash_block_size=BS),
        group_participates=[True],
        group_put_steps=[tp],
        token_databases=[db],
    )
    worker._dsa_dp_owner = MooncakeStoreWorker._dsa_dp_owner.__get__(worker)
    worker._dsa_dp_block_ids = MooncakeStoreWorker._dsa_dp_block_ids.__get__(worker)
    block = s + owner * d + 3
    assert worker._dsa_dp_owner(0, "r", [0, block]) == owner
    req = SimpleNamespace(
        req_id="r", boundary_puts=[(0, block, 2 * BS)], block_hashes=["h0", "h1"]
    )
    puts = KVCacheStoreSendingThread._boundary_puts(worker, req)
    if rank == owner:
        assert puts == [("kh1", [s + 3], [1], None)]
    else:
        assert puts == []


def test_prefix_owner_spreads_decoding_requests(prefix_owner):
    """On a decode engine every queued prompt cost is 0 once the prompts are
    computed; new requests then go to the card with the fewest live requests
    (decode steps pad every card to the busiest owner)."""
    sched = _scheduler(tp=TP)
    owners = []
    for i in range(2 * TP):
        params = SamplingParams(max_tokens=8)
        params.update_from_generation_config({}, eos_token_id=100)
        r = Request(
            request_id=f"d{i}",
            prompt_token_ids=_toks(1000 * (i + 1), BS),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BS, sha256),
        )
        sched.add_request(r)
        _step(sched)
        assert sched._dsa_load == [0] * TP
        owners.append(r.dsa_owner)
    assert owners == [0, 1, 0, 1]


@pytest.mark.parametrize("owners", [None, [0, 1, 1, 0, 2, 3, 3, 3], [3, 3, 0]])
def test_dep_order_segments(owners):
    """Full-graph decode steps: card r's segment holds exactly its requests,
    then padding requests; rows / inv are inverse permutations."""
    tp, q, per = 4, 2, 3
    o = None if owners is None else np.array(owners)
    rows, reqs, inv = dsa_dp.dep_order(o, per * tp * q, tp, q)
    assert sorted(reqs.tolist()) == list(range(per * tp))
    assert (rows[inv] == np.arange(rows.size)).all()
    if o is not None:
        for r in range(tp):
            seg = reqs[r * per : (r + 1) * per]
            real = seg[seg < o.size]
            assert (o[real] == r).all() and real.size == (o == r).sum()


def test_prefix_owner_follows_prefiller_owner(prefix_owner):
    """P/D: a request whose KV comes from a DSA-DP prefiller is owned by the card
    paired with the prefiller's owner (only that card pulls the DSA KV), even
    when another card is less loaded."""
    sched = _scheduler(tp=TP)
    busy = _turn("busy", _toks(1000, 3 * BS))
    sched.add_request(busy)
    assert busy.dsa_owner == 0
    r = _turn("pd", _toks(5000, 3 * BS))
    r.kv_transfer_params = {"remote_dsa_owner": 1, "tp_size": 4}
    sched.add_request(r)
    assert r.dsa_owner == 0  # rank 1 of TP4 -> rank 0 of TP2, the busy card


def test_owner_decode_cap(prefix_owner):
    """A card runs at most max_num_seqs / tp requests that decode locally (a DEP
    rank's share), so a full-graph decode step fits max_num_seqs; requests that
    decode remotely (P/D prefill) are not counted."""
    sched = _scheduler(tp=TP, max_num_seqs=2)
    reqs = []
    for i in range(3):
        r = _turn(f"c{i}", _toks(1000 * (i + 1), BS))
        r.kv_transfer_params = {"remote_dsa_owner": 0, "tp_size": TP}
        sched.add_request(r)
        reqs.append(r)
    remote = _turn("rd", _toks(9000, BS))
    remote.kv_transfer_params = {
        "remote_dsa_owner": 0,
        "tp_size": TP,
        "do_remote_decode": True,
    }
    sched.add_request(remote)
    out = sched.schedule()
    assert [r.dsa_owner for r in reqs] == [0, 0, 0]
    assert set(out.num_scheduled_tokens) == {"c0", "rd"}


def test_prefix_owner_ties_take_turns(prefix_owner):
    """Idle cards (no queued DSA cost, no live requests) are used in turn."""
    sched = _scheduler(tp=TP)
    owners = []
    for i in range(2 * TP):
        r = _turn(f"t{i}", _toks(1000 + 100 * i, 65))
        sched.add_request(r)
        owners.append(r.dsa_owner)
        _run(sched, [r])
    assert owners == list(range(TP)) * 2


@pytest.mark.parametrize("per_page", [1, 8])
def test_apply_regions_slot_packed_draft(monkeypatch, per_page):
    """The release drafter's sliding-window KV packed as slots of shared pages
    (gqa_draft.py, ``blocks_per_pool_block``) reserves the pages its window
    spans, not one page per slot."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    _dsa_dp_layout(monkeypatch)
    swa = SlidingWindowSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=16 * BS,
    )
    groups = [
        KVCacheGroupSpec(["dsa.attn"], _mla()),
        KVCacheGroupSpec(["draft.swa"], swa),
    ]
    groups[1].blocks_per_pool_block = per_page
    cfg = KVCacheConfig(num_blocks=1001, kv_cache_tensors=[], kv_cache_groups=groups)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=10),
    )
    sched = replace(cfg)
    dsa_dp_regions.apply_dsa_dp_regions(
        vllm_config, [{"dsa.attn": _mla(), "draft.swa": swa}], [cfg], sched
    )
    # 17 window slots per request: 17 pages unpacked, ceil(17 / 8) + 1 = 4 packed.
    per_seq = 17 if per_page == 1 else 4
    s, d = region_sizes(1001, 4, 1, 0, 10 * per_seq)
    assert (cfg.dsa_dp_shared_blocks, cfg.dsa_dp_region_blocks) == (s, d)


def test_apply_regions_caps_the_state_reserve(monkeypatch):
    """Hybrid pages unified with the FP32 KDA state hold thousands of tokens, so
    max_num_seqs sequences' state pages can exceed the card: the shared region
    then keeps half the pages and the DSA region the rest (not one page)."""
    _dsa_dp_layout(monkeypatch)
    groups = [
        KVCacheGroupSpec(["dsa.attn"], _mla()),
        KVCacheGroupSpec(
            ["kda"],
            MambaSpec(block_size=BS, shapes=((1, 1),), dtypes=(torch.float32,)),
        ),
    ]
    cfg = KVCacheConfig(num_blocks=241, kv_cache_tensors=[], kv_cache_groups=groups)
    sched = replace(cfg)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    dsa_dp_regions.apply_dsa_dp_regions(
        vllm_config,
        [{"dsa.attn": _mla(), "kda": groups[1].kv_cache_spec}],
        [cfg],
        sched,
    )
    # 32 x 6 = 192 state pages > (241 - 1) // 2 = 120: reserve 120, D = 120.
    assert (cfg.dsa_dp_shared_blocks, cfg.dsa_dp_region_blocks) == (121, 120)


K31_BS = 4096  # K3.1 decode: DSA pages unified with the FP32 KDA state


def _k31_vllm_config(max_model_len: int, max_num_seqs: int = 32):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        max_in_flight_tokens=8192,
    )


def test_k31_decode_capacity_counts_dsa_regions(monkeypatch):
    """K3.1 full-model TP4 decode (1423 pages per card, max_num_seqs 32): two
    KDA groups (align, one prefill checkpoint), the DSA group and the slot-packed
    release drafter. The reported capacity is the owner regions' DSA KV, not the
    shared region (KDA state, drafter) divided by a request's DSA blocks."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    _dsa_dp_layout(monkeypatch)
    kda = MambaSpec(
        block_size=K31_BS,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=K31_BS,
    )
    draft = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )
    groups = [
        KVCacheGroupSpec(["kda0"], kda),
        KVCacheGroupSpec(["kda1"], kda),
        KVCacheGroupSpec(["dsa.attn"], replace(_mla(), block_size=K31_BS)),
        KVCacheGroupSpec(["draft.swa"], draft),
    ]
    groups[3].blocks_per_pool_block = 64
    layer_specs = {g.layer_names[0]: g.kv_cache_spec for g in groups}
    cfg = KVCacheConfig(num_blocks=1423, kv_cache_tensors=[], kv_cache_groups=groups)
    sched = replace(cfg)
    dsa_dp_regions.apply_dsa_dp_regions(
        _k31_vllm_config(1 << 20), [layer_specs], [cfg], sched
    )
    # 32 requests x (2 KDA groups x 3 + 2 drafter pages) = 256 state pages.
    assert (sched.dsa_dp_shared_blocks, sched.dsa_dp_region_blocks) == (257, 1166)
    assert sched.num_blocks == 257

    # 1M-token requests: 256 DSA pages each from 4 x 1166 region pages (the
    # shared region alone gave 257 / 264 x 1M = 1,020,772 tokens, 0.97x).
    assert get_kv_cache_capacity(_k31_vllm_config(1 << 20), sched) == (
        4 * 1166 * K31_BS,
        4 * 1166 / 256,
    )
    for num_tokens in (100_000, 185_000):
        # The shared region holds the state of max_num_seqs requests.
        concurrency = get_max_concurrency_for_kv_cache_config(
            _k31_vllm_config(num_tokens), sched
        )
        assert concurrency == 32
        # Worker configs (num_blocks = S + D) report the same.
        assert concurrency == get_max_concurrency_for_kv_cache_config(
            _k31_vllm_config(num_tokens), cfg
        )


@pytest.mark.parametrize("num_tokens", [100_000, 185_000])
def test_dsa_dp_capacity_matches_region_allocation(num_tokens):
    """K3.1 decode pages (4096 tokens, 1166-page DSA region per card, TP4): a
    P/D request loaded through the connector takes cdiv(L, 4096) DSA pages, all
    in its owner card's region, and a KDA state page in the shared region. One
    card admits 1166 // cdiv(L, 4096) requests; the reported concurrency is the
    TP group's, within one request per card of rounding."""
    tp, d, s = 4, 1166, 1 + 200 * 3
    kda = MambaSpec(
        block_size=K31_BS,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=K31_BS,
    )
    cfg = KVCacheConfig(
        num_blocks=s,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["dsa.attn"], replace(_mla(), block_size=K31_BS)),
            KVCacheGroupSpec(["kda"], kda),
        ],
        dsa_dp_group_ids=(DSA_GROUP,),
        dsa_dp_shared_blocks=s,
        dsa_dp_region_blocks=d,
        dsa_dp_num_regions=tp,
    )
    manager = make_kv_cache_manager(
        cfg, max_model_len=1 << 20, enable_caching=True, hash_block_size=K31_BS
    )
    regions = manager.coordinator.dsa_regions
    pages = -(-num_tokens // K31_BS)
    admitted = 0
    while True:
        req = make_request(
            f"r{admitted}", [0] * num_tokens, K31_BS, sha256, cache_salt=str(admitted)
        )
        req.dsa_owner = 0
        if (
            manager.allocate_slots(
                req,
                num_new_tokens=0,
                num_external_computed_tokens=num_tokens,
                delay_cache_blocks=True,
            )
            is None
        ):
            break
        dsa_ids, kda_ids = manager.get_block_ids(req.request_id)
        assert len(dsa_ids) == pages
        assert {regions.region_of(i) for i in dsa_ids} == {0}
        assert 0 < sum(i != 0 for i in kda_ids) <= 3
        admitted += 1
    assert admitted == d // pages  # 46 (100k) / 25 (185k) per card
    assert [regions.get_num_free_blocks_of(o) for o in range(1, tp)] == [d] * 3
    reported = get_max_concurrency_for_kv_cache_config(
        _k31_vllm_config(num_tokens), cfg
    )
    assert tp * admitted <= reported < tp * (admitted + 1)
