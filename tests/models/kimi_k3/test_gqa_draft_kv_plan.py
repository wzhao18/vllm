# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache plan of Kimi K3.1 with the release (Qwen3, sliding-window GQA) DSpark
draft at the chosen deployments' layouts, on CPU: P TP8/DCP8, D TP16/DCP16 and
C512 TP4/DCP4, BLHNC packed rows, fp8 MLA, MXFP4 indexer keys, FP32 KDA state.

Without packing, the draft's sliding-window group takes a whole DSA row per
1024-token block (3-4 rows per decoding request, 19 per 16K prefill chunk) and,
not supporting fine-grained lookups, drops prefix-cache hits from 128 tokens to
the scheduler block. Packed, its 128-token blocks are slots of the rows: hits
stay at 128 tokens and a request's window takes a quarter to a third of a row.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.models.kimi_k3.test_packed_dsa_block import _kda, _latent
from tests.v1.core.test_prefix_caching import make_request
from vllm.config import CacheConfig
from vllm.models.kimi_k3.common.dsa_cache import (
    KimiK3IndexerBackend,
    indexer_cache_entry_bytes,
)
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    init_none_hash,
    partial_hash_hits_enabled,
    resolve_dcp_kv_block_size,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.utils import allocate_kv_cache, customize_attention_spec

pytestmark = pytest.mark.cpu_test

NUM_DSA, NUM_KDA, NUM_DRAFT = 24, 69, 5
DRAFT_KV_HEADS, DRAFT_HEAD_DIM, WINDOW = 16, 64, 2048
HIT_UNIT = 128
LAYOUT = KVCacheLayout.BLHNC

# name: (TP = DCP, packed DSA block, KDA groups pinned (0: unpinned), K, KV GiB)
# with the KV memory each role measured in the frozen sweep (C1, C512).
POINTS = {
    "P_TP8_DCP8": (8, 1024, 4, 7, 29.48),
    "D_TP16_DCP16": (16, 1024, 4, 7, 90.49),
    "C512_TP4_DCP4": (4, 4096, 0, 3, 90.39),
}
# Rows the frozen sweep measured with the preview MLA draft (block_len_per_layer
# and num_blocks of the NIXL registration): the specs below reproduce them.
PREVIEW_ROWS = {"P_TP8_DCP8": 1683, "D_TP16_DCP16": 5168, "C512_TP4_DCP4": 1290}


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


@pytest.fixture(autouse=True)
def _no_pins(monkeypatch):
    """Each plan sets the renders' pins: 4 KDA groups on P/D, none at C512."""
    monkeypatch.delenv("VLLM_KIMI_K3_PACKED_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("VLLM_PACKED_KV_STATE_GROUPS", raising=False)


def _config(block_size: int, prefix_match_unit: int | None = HIT_UNIT):
    cache_config = CacheConfig(block_size=block_size)
    cache_config.kv_cache_layout = LAYOUT.name
    cache_config.prefix_match_unit = prefix_match_unit
    return SimpleNamespace(
        cache_config=cache_config,
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )


def _specs(tp: int, block_size: int, k: int, draft: str) -> dict:
    """One rank's K3.1 specs (fp8 latent, MXFP4 indexer keys, FP32 KDA state)
    with no draft, the preview MLA draft or the release GQA draft, which runs at
    DCP 1: its KV is replicated over DCP and split over TP by KV head."""
    latent = replace(_latent("fp8"), block_size=block_size)
    index = customize_attention_spec(
        KimiK3IndexerBackend,
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=indexer_cache_entry_bytes(128, True),
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        ),
    )
    kda = replace(_kda(tp, k, "fp32"), block_size=block_size, mamba_cache_mode="align")
    specs: dict = {}
    for i in range(NUM_DSA):
        specs[f"model.layers.{i}.self_attn.attn"] = latent
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = index
    for i in range(NUM_KDA):
        specs[f"model.layers.{100 + i}.linear_attn"] = kda
    if draft == "preview":
        mla = replace(_latent("fp8"), block_size=block_size, is_draft=True)
        specs |= {f"draft.layers.{i}.attn": mla for i in range(NUM_DRAFT)}
    elif draft == "release":
        swa = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=DRAFT_KV_HEADS // tp,
            max_tp_shards=DRAFT_KV_HEADS,
            head_size=DRAFT_HEAD_DIM,
            dtype=torch.bfloat16,
            sliding_window=WINDOW,
            dcp_sharded=False,
        )
        specs |= {f"draft.layers.{i}.attn": swa for i in range(NUM_DRAFT)}
    return specs


def _plan(point: str, draft: str, packed: bool = True) -> list[KVCacheGroupSpec]:
    tp, block_size, pinned_groups, k, _ = POINTS[point]
    with pytest.MonkeyPatch.context() as env:
        if pinned_groups:
            env.setenv("VLLM_PACKED_KV_STATE_GROUPS", str(pinned_groups))
        return _group(_config(block_size), _specs(tp, block_size, k, draft), packed)


def _group(config, specs: dict, packed: bool) -> list[KVCacheGroupSpec]:
    if packed:
        return GQADraftKVCacheGrouping.get_kv_cache_groups(config, specs)
    return get_kv_cache_groups(config, specs)


def _draft_group(groups: list[KVCacheGroupSpec]) -> KVCacheGroupSpec:
    (group,) = [g for g in groups if g.layer_names[0].startswith("draft.")]
    return group


def _num_rows(point: str, groups: list[KVCacheGroupSpec]) -> int:
    return int(POINTS[point][4] * 2**30) // _get_kv_cache_bytes_per_block(
        groups, LAYOUT
    )


def _window_slots(spec: SlidingWindowSpec, in_flight: int) -> int:
    """Blocks a request holds for its draft window (the admission cap)."""
    return spec.max_admission_blocks_per_request(in_flight, 1 << 20)


def _fine_grained_hits(point: str, groups: list[KVCacheGroupSpec]) -> bool:
    dcp = POINTS[point][0]
    managers = [
        KVCacheSpecRegistry.get_manager_class(
            getattr(g.kv_cache_spec, "first_spec", g.kv_cache_spec), g.role
        )
        for g in groups
    ]
    return partial_hash_hits_enabled(groups, HIT_UNIT, dcp, managers)


@pytest.mark.parametrize("point", POINTS)
def test_specs_reproduce_the_measured_preview_rows(point):
    """The synthetic specs size the row as production did with the preview
    draft (its 5 MLA pages inside the DSA row)."""
    assert _num_rows(point, _plan(point, "preview")) == PREVIEW_ROWS[point]


@pytest.mark.parametrize("point", POINTS)
def test_release_draft_keeps_128_token_prefix_hits(point):
    """Unpacked, the draft's sliding-window group (1024- or 4096-token blocks,
    no fine-grained lookup) turns 128-token hits off engine-wide, so hits fall
    back to the scheduler block. Packed, its blocks are the hit unit."""
    dcp = POINTS[point][0]
    unpacked = _plan(point, "release", packed=False)
    groups = _plan(point, "release")
    scheduler_block = math.lcm(
        *(resolve_dcp_kv_block_size(g.kv_cache_spec, dcp) for g in groups)
    )
    assert not _fine_grained_hits(point, unpacked)
    assert _fine_grained_hits(point, groups)
    assert _fine_grained_hits(point, _plan(point, "preview"))
    assert _draft_group(groups).kv_cache_spec.block_size == HIT_UNIT
    # The draft does not change the scheduler block (8192 on P, else 16384).
    assert scheduler_block == math.lcm(
        *(resolve_dcp_kv_block_size(g.kv_cache_spec, dcp) for g in unpacked)
    )
    assert scheduler_block == {"P_TP8_DCP8": 8192}.get(point, 16384)


@pytest.mark.parametrize("point", POINTS)
def test_target_groups_match_the_preview_deployment(point):
    """The target's groups, block and row are those of the preview draft run
    minus the draft pages: the release draft adds a group but changes no
    target group (at C512 the KDA states keep the preview's 2 groups)."""
    groups = _plan(point, "release")
    target = [g for g in groups if g is not _draft_group(groups)]
    preview = _plan(point, "preview")
    assert [
        (g.kv_cache_spec.block_size, [n for n in g.layer_names if "draft" not in n])
        for g in target
    ] == [
        (g.kv_cache_spec.block_size, [n for n in g.layer_names if "draft" not in n])
        for g in preview
    ]
    kda = [g for g in target if isinstance(g.kv_cache_spec.first_spec, MambaSpec)]
    assert len(kda) == {"C512_TP4_DCP4": 2}.get(point, 4)
    assert _get_kv_cache_bytes_per_block(groups, LAYOUT) == (
        _get_kv_cache_bytes_per_block(target, LAYOUT)
    )


# Per point: rows per GiB vs the preview, draft slots per row, and the draft
# rows a decoding request (K+1 rows in flight, twice under async scheduling)
# and a 16K prefill chunk hold, unpacked -> packed.
EXPECTED = {
    "P_TP8_DCP8": dict(slots=48, decode=(4, 18), chunk=(19, 145)),
    "D_TP16_DCP16": dict(slots=72, decode=(4, 18), chunk=(19, 145)),
    "C512_TP4_DCP4": dict(slots=87, decode=(2, 18), chunk=(6, 145)),
}


@pytest.mark.parametrize("point", POINTS)
def test_draft_window_takes_a_fraction_of_a_row(point):
    tp, _, _, k, _ = POINTS[point]
    expected = EXPECTED[point]
    unpacked = _draft_group(_plan(point, "release", packed=False))
    groups = _plan(point, "release")
    draft = _draft_group(groups)
    slots = draft.blocks_per_pool_block
    assert unpacked.blocks_per_pool_block == 1
    assert slots == expected["slots"]

    decode_in_flight, chunk = 2 * (k + 1), 16384
    unpacked_spec = unpacked.kv_cache_spec.first_spec
    spec = draft.kv_cache_spec.first_spec
    # Before: one row per draft block.
    assert (
        _window_slots(unpacked_spec, decode_in_flight),
        _window_slots(unpacked_spec, chunk),
    ) == (expected["decode"][0], expected["chunk"][0])
    # After: slots of shared rows.
    assert (
        _window_slots(spec, decode_in_flight),
        _window_slots(spec, chunk),
    ) == (expected["decode"][1], expected["chunk"][1])
    decode_rows = _window_slots(spec, decode_in_flight) / slots
    assert decode_rows <= 0.375
    assert cdiv(_window_slots(spec, chunk), slots) <= 4

    # Rows: the release draft's rows are the target's alone, ~18% more than the
    # preview's, whose MLA draft pages widened every row.
    rows = _num_rows(point, groups)
    assert rows == _num_rows(point, _plan(point, "none"))
    assert rows / PREVIEW_ROWS[point] > 1.17


@pytest.mark.parametrize("point", POINTS)
def test_draft_slots_tile_the_rows(point):
    """The planner lays the draft's slots inside the rows: slot ``j`` is the
    ``j % S``-th equal stride of row ``j // S``, every layer's page inside it,
    and the backing allocation is the target's."""
    groups = _plan(point, "release")
    draft = _draft_group(groups)
    slots = draft.blocks_per_pool_block
    row_bytes = _get_kv_cache_bytes_per_block(groups, LAYOUT)
    num_rows = 100
    config = _config(POINTS[point][1])
    kv_cache_config = get_kv_cache_config_from_groups(
        config, groups, num_rows * row_bytes
    )
    assert kv_cache_config.num_blocks == num_rows
    assert len({t.size for t in kv_cache_config.kv_cache_tensors}) == 1
    (draft_tensor,) = [
        t
        for t in kv_cache_config.kv_cache_tensors
        if set(t.layers) <= set(draft.layer_names)
    ]
    page = draft.kv_cache_spec.first_spec.page_size_bytes
    assert draft_tensor.layers == draft.layer_names
    assert draft_tensor.block_stride * slots == row_bytes
    assert draft_tensor.block_stride >= NUM_DRAFT * page
    assert draft_tensor.block_stride % 128 == 0
    assert (draft_tensor.layer_stride, draft_tensor.offset) == (page, 0)
    assert kv_cache_config.num_blocks_of(draft_tensor) == num_rows * slots
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor is not draft_tensor:
            assert tensor.block_stride == row_bytes
            assert kv_cache_config.num_blocks_of(tensor) == num_rows
    # Every slot's pages stay inside its row.
    for slot in (0, slots - 1, slots, num_rows * slots - 1):
        start = slot * draft_tensor.block_stride
        end = start + NUM_DRAFT * page
        assert start // row_bytes == (end - 1) // row_bytes == slot // slots


def test_prefix_match_unit_unset_keeps_the_draft_block():
    """Without a prefix-match unit, hits land on the scheduler block anyway:
    the draft keeps its block, packed when several fit in a row."""
    tp, block_size, _, k, _ = POINTS["D_TP16_DCP16"]
    groups = GQADraftKVCacheGrouping.get_kv_cache_groups(
        _config(block_size, prefix_match_unit=None),
        _specs(tp, block_size, k, "release"),
    )
    draft = _draft_group(groups)
    assert draft.kv_cache_spec.block_size == block_size
    assert draft.blocks_per_pool_block == 12


def _scheduler_manager(
    point: str, draft: str, packed: bool = True
) -> tuple[KVCacheManager, int]:
    """The scheduler's KV cache manager over the planned groups, with KDA
    prompt checkpoints exported inside the chunk as K3.1 does."""
    tp, block_size, _, k, _ = POINTS[point]
    groups = _plan(point, draft, packed=packed)
    for group in groups:
        if isinstance(group.kv_cache_spec.first_spec, MambaSpec):
            group.kv_cache_spec = UniformTypeKVCacheSpecs.from_specs(
                {
                    name: replace(
                        spec,
                        num_prefill_checkpoint_blocks=1,
                        prefill_checkpoint_alignment=HIT_UNIT,
                    )
                    for name, spec in group.kv_cache_spec.kv_cache_specs.items()
                }
            )
    row_bytes = _get_kv_cache_bytes_per_block(groups, LAYOUT)
    kv_cache_config = generate_scheduler_kv_cache_config(
        [get_kv_cache_config_from_groups(_config(block_size), groups, 400 * row_bytes)]
    )
    scheduler_block = math.lcm(
        *(resolve_dcp_kv_block_size(g.kv_cache_spec, tp) for g in groups)
    )
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=HIT_UNIT,
        scheduler_block_size=scheduler_block,
        dcp_world_size=tp,
    )
    return manager, scheduler_block


def _next_turn(point: str, draft: str, packed: bool = True):
    """A 20000-token turn chunked at the scheduler block, then the next turn of
    the conversation: its hit length, the per-group hits, and the draft blocks
    the writer held at the end of its prefill and the turn's hit restores."""
    manager, scheduler_block = _scheduler_manager(point, draft, packed)
    length = 20000
    tokens = list(range(length))
    writer = make_request("writer", tokens, HIT_UNIT, sha256)
    bounds = [*range(scheduler_block, length, scheduler_block), length]
    for start, end in zip([0, *bounds], bounds):
        assert manager.allocate_slots(writer, end - start)
        writer.num_computed_tokens = end
    held = [ids for ids in manager.get_block_ids("writer")]
    manager.free(writer)
    turn = make_request("turn", tokens + list(range(9, 1809)), HIT_UNIT, sha256)
    _, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        turn.block_hashes, turn.get_max_cache_hit_length()
    )
    blocks, num_hit, _ = manager.get_computed_blocks(turn)
    return manager, num_hit, per_group, held, blocks


@pytest.mark.parametrize("point", POINTS)
def test_next_turn_hits_where_the_preview_deployment_did(point):
    """The next turn of a 20000-token conversation resumes where it did with
    the preview draft: at the prompt's last 128-token boundary (19968) on P and
    D, where the target keeps that checkpoint. The packed draft window never
    lowers a hit (it holds the 19968 tail at C512 too, where this chunking
    leaves the target's states at 16384); unpacked, every hit drops to the
    scheduler block."""
    preview_hit = _next_turn(point, "preview")[1]
    manager, num_hit, per_group, held, blocks = _next_turn(point, "release")
    _, unpacked_hit, _, unpacked_held, _ = _next_turn(point, "release", False)
    draft_id = len(per_group) - 1
    scheduler_block = {"P_TP8_DCP8": 8192}.get(point, 16384)

    assert num_hit == preview_hit == min(per_group[:draft_id])
    assert preview_hit == {"C512_TP4_DCP4": 16384}.get(point, 19968)
    assert per_group[draft_id] == 19968
    assert unpacked_hit == 19999 // scheduler_block * scheduler_block
    # The hit restores the writer's own draft blocks: the window's 16 blocks.
    window = [b.block_id for b in blocks.blocks[draft_id] if not b.is_null]
    assert len(window) == WINDOW // HIT_UNIT
    assert set(window) <= set(held[draft_id])

    # The writer's draft blocks at the end of its prefill: a row each when
    # unpacked, slots of a single row when packed.
    slots = manager.coordinator.single_type_managers[draft_id].block_pool
    rows = {b // slots.slots_per_block for b in held[draft_id] if b}
    assert len(rows) == 1
    assert len([b for b in unpacked_held[draft_id] if b]) == (
        {"C512_TP4_DCP4": 2}.get(point, 6)
    )


@pytest.mark.parametrize("point", POINTS)
def test_worker_views_put_each_draft_slot_inside_its_row(point):
    """The worker's allocation views the planned tensors: draft slot ``j`` of
    every draft layer addresses bytes of row ``j // S`` only, at the slot's
    stride, so a slot write never reaches another row (another group's)."""
    groups = _plan(point, "release")
    draft = _draft_group(groups)
    slots = draft.blocks_per_pool_block
    row_bytes = _get_kv_cache_bytes_per_block(groups, LAYOUT)
    num_rows = 3
    kv_cache_config = get_kv_cache_config_from_groups(
        _config(POINTS[point][1]), groups, num_rows * row_bytes
    )
    views = allocate_kv_cache(kv_cache_config, torch.device("cpu"), LAYOUT)
    base = views[groups[0].layer_names[0]].untyped_storage().data_ptr()
    page = draft.kv_cache_spec.first_spec.page_size_bytes
    for i, name in enumerate(draft.layer_names):
        view = views[name]
        assert view.untyped_storage().data_ptr() == base
        assert view.shape[0] == num_rows * slots
        for slot in (1, slots - 1, slots, num_rows * slots - 1):
            first = view[slot].data_ptr() - base
            last = first + view[slot].numel() * view.element_size() - 1
            assert (first, last + 1 - first) == (
                slot // slots * row_bytes
                + slot % slots * row_bytes // slots
                + i * page,
                page,
            )
            assert first // row_bytes == last // row_bytes == slot // slots
