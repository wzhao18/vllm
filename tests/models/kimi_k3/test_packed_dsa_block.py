# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1's packed DSA block for a P (TP8) / D (TP16) pair, on CPU.

Unpinned, ``packed_dsa_block_size`` follows TP and the cache dtypes: with FP32
KDA states, or NVFP4 latents plus MXFP4 indexer keys at K=7, P rounds up to
2048 tokens while D stays at 1024, and NIXL cannot pair their blocks. With
``VLLM_KIMI_K3_PACKED_BLOCK_SIZE`` both get the pin, and the real planner fits
the KDA states into ``VLLM_PACKED_KV_STATE_GROUPS`` groups beside the DSA row,
or raises at startup when there are too few. P and D then split the states
alike, so NIXL registers the same regions on both.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.models.kimi_k3.common import dsa_cache
from vllm.models.kimi_k3.common.dsa_cache import (
    KimiK3IndexerBackend,
    indexer_cache_entry_bytes,
    packed_dsa_block_size,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    DS_MLA_STATE_CONTENT_BYTES,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import customize_attention_spec

pytestmark = pytest.mark.cpu_test

NUM_DSA, NUM_KDA, NUM_DRAFT = 24, 69, 5
KDA_HEADS, KDA_HEAD_DIM, KDA_CONV = 96, 128, 4
INDEX_HEAD_DIM = 128
P_TP, D_TP = 8, 16
ALIGNMENT, PIN = 64, 1024
LATENT_DTYPES = {"fp8": "fp8", "nvfp4": "nvfp4_ds_mla"}
STATE_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

# (latent, indexer keys, KDA recurrent state, K): the unpinned block of P and D,
# and the fewest KDA groups whose states fit beside a pinned block's DSA row.
FORMAT_FIELDS = "latent,index,state,k,unpinned,min_groups"
K31_FORMATS = [
    ("fp8", "fp8", "bf16", 3, (1024, 512), (2, 1)),
    ("fp8", "fp8", "bf16", 7, (1024, 512), (2, 1)),
    ("fp8", "fp8", "fp32", 3, (2048, 1024), (3, 2)),
    ("fp8", "fp8", "fp32", 7, (2048, 1024), (3, 2)),
    ("fp8", "mxfp4", "bf16", 3, (1024, 512), (2, 1)),
    ("fp8", "mxfp4", "bf16", 7, (1024, 512), (2, 1)),
    ("fp8", "mxfp4", "fp32", 3, (2048, 1024), (4, 2)),
    ("fp8", "mxfp4", "fp32", 7, (2048, 1024), (4, 2)),
    ("nvfp4", "fp8", "bf16", 3, (1024, 512), (3, 2)),
    ("nvfp4", "fp8", "bf16", 7, (1024, 512), (3, 2)),
    ("nvfp4", "fp8", "fp32", 3, (2048, 1024), (5, 2)),
    ("nvfp4", "fp8", "fp32", 7, (2048, 1024), (5, 3)),
    ("nvfp4", "mxfp4", "bf16", 3, (1024, 512), (3, 2)),
    ("nvfp4", "mxfp4", "bf16", 7, (2048, 1024), (3, 2)),
    ("nvfp4", "mxfp4", "fp32", 3, (2048, 1024), (5, 3)),
    ("nvfp4", "mxfp4", "fp32", 7, (2048, 1024), (5, 3)),
]


@pytest.fixture(autouse=True)
def _k31_env(monkeypatch):
    """No pin unless a test sets one; MXFP4 keys without SM10x (only their row
    width matters here)."""
    monkeypatch.delenv("VLLM_KIMI_K3_PACKED_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("VLLM_PACKED_KV_STATE_GROUPS", raising=False)
    monkeypatch.setattr(
        dsa_cache,
        "dsa_indexer_uses_fp4",
        lambda config: config.attention_config.indexer_kv_dtype == "mxfp4",
    )


def _model_config(index: str, kv_transfer_config=None) -> SimpleNamespace:
    """What packed_dsa_block_size reads of K3.1's config."""
    text_config = SimpleNamespace(
        dsa_config=SimpleNamespace(
            index_head_dim=INDEX_HEAD_DIM,
            index_k_norm_type="layernorm",
            index_rope_dim=0,
        ),
        linear_attn_config={
            "full_attn_layers": list(range(NUM_DSA)),
            "kda_layers": list(range(NUM_KDA)),
        },
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        attention_config=SimpleNamespace(indexer_kv_dtype=index),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        kv_transfer_config=kv_transfer_config,
    )


def _latent(dtype: str) -> MLAAttentionSpec:
    """A DSA layer's latent page as MLAAttention declares it and the sparse
    FlashInfer MLA backend customizes it."""
    return customize_attention_spec(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE.get_class(),
        MLAAttentionSpec(
            block_size=PIN,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=576,
            dtype=torch.uint8,
            cache_dtype_str=dtype,
            kv_quant_mode=get_kv_quant_mode(dtype),
            state_content_bytes=DS_MLA_STATE_CONTENT_BYTES.get(dtype),
        ),
    )


def _kda(tp: int, k: int, state: str) -> MambaSpec:
    """A KDA layer's state page per rank: BF16 conv rows, then the heads."""
    return MambaSpec(
        block_size=PIN,
        shapes=MambaStateShapeCalculator.kda_state_shape(
            tp, KDA_HEADS, KDA_HEAD_DIM, conv_kernel_size=KDA_CONV, num_spec=k
        ),
        dtypes=(torch.bfloat16, STATE_DTYPES[state]),
    )


def _k31_specs(tp: int, k: int, latent: str, index: str, state: str) -> dict:
    """One rank's specs: per DSA layer a latent and an indexer page, the DSpark
    drafter's fp8 latent pages, and the KDA states."""
    lat = _latent(LATENT_DTYPES[latent])
    idx = customize_attention_spec(
        KimiK3IndexerBackend,
        MLAAttentionSpec(
            block_size=PIN,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=indexer_cache_entry_bytes(INDEX_HEAD_DIM, index == "mxfp4"),
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        ),
    )
    draft = replace(_latent("fp8"), is_draft=True)
    specs: dict = {}
    for i in range(NUM_DSA):
        specs[f"model.layers.{i}.self_attn.attn"] = lat
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = idx
    for i in range(NUM_DRAFT):
        specs[f"draft.layers.{i}.self_attn.attn"] = draft
    kda = _kda(tp, k, state)
    for i in range(NUM_KDA):
        specs[f"model.layers.{100 + i}.linear_attn"] = kda
    return specs


def _planner_config() -> SimpleNamespace:
    """What the KV cache planner reads, at the pinned block."""
    cache_config = CacheConfig(block_size=PIN)
    cache_config.kv_cache_layout = "BLHNC"
    return SimpleNamespace(
        cache_config=cache_config,
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )


@pytest.mark.parametrize(FORMAT_FIELDS, K31_FORMATS)
def test_pin_gives_p_and_d_the_same_block(
    monkeypatch, latent, index, state, k, unpinned, min_groups
):
    """Pinned, P and D both get the pin, and each logs it with the size it
    replaced."""

    def blocks() -> tuple[int | None, ...]:
        return tuple(
            packed_dsa_block_size(
                _model_config(index),
                _latent(LATENT_DTYPES[latent]).state_content_size_bytes,
                _kda(tp, k, state).page_size_bytes,
                ALIGNMENT,
            )
            for tp in (P_TP, D_TP)
        )

    logged: list[tuple] = []
    monkeypatch.setattr(
        dsa_cache.logger, "info_once", lambda msg, *args, **kw: logged.append(args)
    )
    assert blocks() == unpinned and not logged
    monkeypatch.setenv("VLLM_KIMI_K3_PACKED_BLOCK_SIZE", str(PIN))
    assert blocks() == (PIN, PIN)
    assert logged == [(PIN, unpinned[0]), (PIN, unpinned[1])]


@pytest.mark.parametrize("pin", ["1000", "-1024"])
def test_pin_must_be_a_positive_multiple_of_the_alignment(monkeypatch, pin):
    monkeypatch.setenv("VLLM_KIMI_K3_PACKED_BLOCK_SIZE", pin)
    with pytest.raises(ValueError, match="positive multiple of the block alignment"):
        packed_dsa_block_size(
            _model_config("fp8"), 576, _kda(P_TP, 3, "bf16").page_size_bytes, ALIGNMENT
        )


def test_pin_under_a_kv_transfer_needs_the_state_groups(monkeypatch):
    """Without VLLM_PACKED_KV_STATE_GROUPS each side's planner picks the fewest
    groups at its own TP, which differ for every format above, so a pinned P/D
    start raises before NIXL would refuse the pair; unpinned sizing is unchanged."""
    config = _model_config("fp8", kv_transfer_config=SimpleNamespace())
    state_bytes = _kda(P_TP, 3, "bf16").page_size_bytes
    assert packed_dsa_block_size(config, 576, state_bytes, ALIGNMENT) == 1024
    monkeypatch.setenv("VLLM_KIMI_K3_PACKED_BLOCK_SIZE", str(PIN))
    with pytest.raises(ValueError, match="needs VLLM_PACKED_KV_STATE_GROUPS"):
        packed_dsa_block_size(config, 576, state_bytes, ALIGNMENT)
    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", "2")
    assert packed_dsa_block_size(config, 576, state_bytes, ALIGNMENT) == PIN


@pytest.mark.parametrize("groups", [0, 2, 3, 4])
@pytest.mark.parametrize("tp", [P_TP, D_TP])
@pytest.mark.parametrize(FORMAT_FIELDS, K31_FORMATS)
def test_pinned_block_fits_the_state_groups_or_raises(
    monkeypatch, latent, index, state, k, unpinned, min_groups, tp, groups
):
    """The KDA states split into exactly VLLM_PACKED_KV_STATE_GROUPS groups (0:
    the fewest that fit) and the block stride stays the DSA row's; too few
    groups raise instead of widening the block."""
    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", str(groups))
    config = _planner_config()
    specs = _k31_specs(tp, k, latent, index, state)
    need = min_groups[(P_TP, D_TP).index(tp)]
    if 0 < groups < need:
        with pytest.raises(ValueError, match=f"need at least {need} groups"):
            get_kv_cache_groups(config, specs)
        return

    plan = get_kv_cache_groups(config, specs)
    kda = [g for g in plan if isinstance(g.kv_cache_spec.first_spec, MambaSpec)]
    dsa = [g for g in plan if g not in kda]
    assert len(kda) == (groups or need) and len(dsa) == 1
    assert sorted(n for g in kda for n in g.layer_names) == sorted(
        n for n, spec in specs.items() if isinstance(spec, MambaSpec)
    )
    layout = KVCacheLayout.BLHNC
    assert _get_kv_cache_bytes_per_block(plan, layout) == (
        _get_kv_cache_bytes_per_block(dsa, layout)
    )


@pytest.mark.parametrize(
    f"{FORMAT_FIELDS},groups",
    [
        (*fields, groups)
        for fields in K31_FORMATS
        for groups in (2, 3, 4)
        if groups >= max(fields[-1])
    ],
)
def test_p_and_d_split_the_states_alike(
    monkeypatch, latent, index, state, k, unpinned, min_groups, groups
):
    """NIXL asserts that P and D register the same regions, one per KDA state
    slot beside the attention ones: at a group count that fits both, the TP8
    prefiller and the TP16 decoder get the same groups, with the same layers in
    the same order, at the same block."""
    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", str(groups))

    def groups_of(tp: int) -> list[tuple[int, list[str]]]:
        specs = _k31_specs(tp, k, latent, index, state)
        plan = get_kv_cache_groups(_planner_config(), specs)
        return [(g.kv_cache_spec.block_size, g.layer_names) for g in plan]

    p = groups_of(P_TP)
    assert p == groups_of(D_TP) and len(p) == groups + 1
