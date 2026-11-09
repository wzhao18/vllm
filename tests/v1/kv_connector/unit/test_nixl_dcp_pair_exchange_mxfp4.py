# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange on an MXFP4 indexer cache (indexer_kv_dtype="mxfp4"): 68-byte
indexer rows, 64 E2M1 value bytes then 4 UE8M0 scale bytes, kept as the fp8
cache's kernel pages of 64 rows' values, then their scales. The worker's value
bytes from the row width, also through register_kv_caches (A3), and the fp8 tests'
row tables and exchanges on these rows (CPU), the Triton exchange (A6) and the real
MXFP4 key writer (A7) on GPU; with the fp8 MLA rows and, in a tree with the NVFP4
pair rows, nvfp4_ds_mla's.
"""

import inspect
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    NixlPairModeError,
    PairArrayKind,
    PairRegion,
    pair_layout,
    pair_units,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole
from vllm.v1.worker.utils import AttentionGroup

from . import test_nixl_dcp_pair_exchange as cpu
from . import test_nixl_dcp_pair_exchange_gpu as gpu
from . import test_nixl_pair_exchange_worker as worker_suite
from .test_nixl_pair_exchange_worker import fabric, k31_env  # noqa: F401

MXFP4_ROW, MXFP4_VALUES = 68, 64  # index_head_dim 128: 64 B E2M1 + 4 UE8M0
LATENT_ROW = {"fp8": 576, "nvfp4_ds_mla": 352}
NO_NVFP4 = "nvfp4_ds_mla: this tree has no NVFP4 pair rows (_k31_nvfp4_specs)"


@pytest.fixture
def mxfp4_rows(monkeypatch):
    """The fp8 tests' indexer rows switched to MXFP4's (68 B, 64 value bytes)."""
    for module in (cpu, worker_suite):
        monkeypatch.setattr(module, "INDEX_ROW_BYTES", MXFP4_ROW)
        monkeypatch.setattr(module, "INDEX_VALUE_BYTES", MXFP4_VALUES)
    monkeypatch.setattr(gpu, "INDEX_VALUE_BYTES", MXFP4_VALUES)


def latent_ready(latent: str) -> None:
    """Skip nvfp4_ds_mla latent rows in a tree without their pair rows."""
    if latent != "fp8" and not hasattr(cpu, "_k31_nvfp4_specs"):
        pytest.skip(NO_NVFP4)


def k31_specs(latent: str, num_dsa: int, num_draft: int) -> dict:
    """The packed K3.1 DSA group on 1024-token blocks with ``latent`` DSA rows (the
    drafter's stay fp8) and the current indexer rows."""
    latent_ready(latent)
    make = cpu._k31_dsa_specs if latent == "fp8" else cpu._k31_nvfp4_specs
    return make(1024, num_dsa, num_draft)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "width,values", [(132, 128), (68, 64), (100, None), (36, None)]
)
def test_pair_index_format_from_row_width(width, values):
    """The worker takes the value bytes of an indexer row from the width of the
    kernel pages bound to the layer: fp8's index_head_dim + 4 bytes hold
    index_head_dim, MXFP4's index_head_dim / 2 + index_head_dim / 32 hold
    index_head_dim / 2; any other width fails pair mode at registration."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    name = "model.layers.0.self_attn.indexer.k_cache"
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
    )
    view = torch.zeros(3, 64, width, dtype=torch.uint8)  # its kernel pages
    region = PairRegion(name, spec, view.data_ptr(), spec.page_size_bytes, 1 << 20)
    worker = object.__new__(NixlBaseConnectorWorker)
    text_config = SimpleNamespace(dsa_config=SimpleNamespace(index_head_dim=128))
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={name: SimpleNamespace(kv_cache=view)}
        ),
        model_config=SimpleNamespace(hf_text_config=text_config),
    )
    if values is None:
        with pytest.raises(NixlPairModeError, match="neither fp8"):
            worker._dcp_pair_index_format([region])
    else:
        assert worker._dcp_pair_index_format([region]) == (64, values)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_tokens,num_cached,num_draft,extra_d",
    [(2500, 0, 0, 1), (3000, 1, 5, 0)],
    ids=["odd-P-blocks,D-lookahead", "even-P-blocks,D-prefix-hit,drafter"],
)
def test_pull_then_exchange_on_mxfp4_rows(
    mxfp4_rows,
    k31_env,  # noqa: F811
    fabric,  # noqa: F811
    monkeypatch,
    num_tokens,
    num_cached,
    num_draft,
    extra_d,
):
    """A3 (test_pull_then_exchange_on_k31_geometry) with MXFP4 indexer rows, through
    the real worker: register_kv_caches takes 64 value bytes from the bound 68 B
    rows at K3.1's index_head_dim 128 (where it raised NixlPairModeError before) and
    builds the exchanger on them, and every D rank's pulled and exchanged pages hold
    their interleave-1 tokens' rows."""
    real = worker_suite.NixlPullConnectorWorker

    def k31_head_dim(vllm_config, *args, **kwargs):
        # The suite's config takes index_head_dim as fp8's value bytes.
        vllm_config.model_config.hf_text_config.dsa_config.index_head_dim = 128
        return real(vllm_config, *args, **kwargs)

    monkeypatch.setattr(worker_suite, "NixlPullConnectorWorker", k31_head_dim)
    test = worker_suite.test_pull_then_exchange_on_k31_geometry
    kwargs = dict(k31_env=None, fabric=fabric, num_tokens=num_tokens)
    kwargs.update(num_cached=num_cached, num_draft=num_draft, extra_d=extra_d)
    for name, value in (("monkeypatch", monkeypatch), ("groups", 2)):
        if name in inspect.signature(test).parameters:
            kwargs[name] = value  # (the pin track's KDA state groups)
    test(**kwargs)


@pytest.mark.cpu_test
@pytest.mark.parametrize("latent", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize("num_draft", [0, 5])
def test_row_table_on_mxfp4_planner_geometry(mxfp4_rows, num_draft, latent):
    """test_row_table_matches_k31_planner_geometry with MXFP4 indexer rows: every
    row the table addresses is the row the backends' views read, and a block's
    arrays tile its attention pages exactly."""
    rank = cpu._PlannedRank(k31_specs(latent, 24, num_draft), num_blocks=2)
    table = rank.table
    assert table.block_stride == rank.kv_cache_config.kv_cache_tensors[0].block_stride
    # H: 24 DSA layers x 512 rows x (latent + 68) B, plus 294,912 B per (fp8) draft
    # layer: with K3.1's 5, 9,388,032 B on fp8 latents (fp8 keys: 10,174,464 B)
    # and 6,635,520 B on nvfp4_ds_mla (fp8 keys: 7,421,952 B).
    row = LATENT_ROW[latent]
    assert table.half_bytes == 24 * 512 * (row + MXFP4_ROW) + num_draft * 294_912
    arrays = Counter({(PairArrayKind.ROW, row): 24})
    arrays[PairArrayKind.ROW, LATENT_ROW["fp8"]] += num_draft
    arrays[PairArrayKind.INDEX_VALUE, MXFP4_VALUES] = 24
    arrays[PairArrayKind.INDEX_SCALE, MXFP4_ROW - MXFP4_VALUES] = 24
    assert Counter((a.kind, a.width) for a in table.arrays) == arrays
    rows = np.arange(1024)
    page_bytes: dict[str, list[tuple[int, int]]] = {}
    for array in table.arrays:
        for b in (0, 1):
            np.testing.assert_array_equal(
                table.base_addr + table.row_offsets(array, [b], rows)[0],
                rank.row_addresses(array, b),
                err_msg=array.layer_name,
            )
        starts = table.base_addr + table.row_offsets(array, [1], rows)[0]
        page_bytes.setdefault(array.layer_name, []).extend(
            (start, start + array.width) for start in starts.tolist()
        )
    for name, intervals in page_bytes.items():
        intervals.sort()
        start = rank.views[name][1].data_ptr()
        ends = [start, *(end for _, end in intervals)]
        assert [s for s, _ in intervals] == ends[:-1], name
        assert ends[-1] == start + rank.specs[name].page_size_bytes, name


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "block_size,num_tokens,num_cached",
    [(64, 1500, 0), (64, 2600, 1), (128, 5000, 0), (128, 6000, 1)],
)
def test_exchange_on_mxfp4_planner_geometry(
    mxfp4_rows, block_size, num_tokens, num_cached
):
    """A1 (test_exchange_on_planner_geometry) with MXFP4 indexer rows."""
    cpu.test_exchange_on_planner_geometry(block_size, num_tokens, num_cached)


@gpu.requires_gpu
@pytest.mark.parametrize("far", [False, True], ids=["near", "far"])
@pytest.mark.parametrize("latent", ["fp8", "nvfp4_ds_mla"])
def test_triton_exchange_mxfp4_matches_torch_reference(mxfp4_rows, latent, far):
    """A6 (test_triton_exchange_matches_torch_reference) on the K3.1 geometry with
    MXFP4 indexer rows: 64 B value rows (16 B copies) and 4 B scale rows, next to
    fp8 or nvfp4_ds_mla latent rows."""
    latent_ready(latent)
    gpu.test_triton_exchange_matches_torch_reference(latent, far)


def _write_tokens(ops, rank, slots, inputs, norm) -> None:
    """test_nixl_dcp_pair_exchange_gpu._write_tokens with the indexer pages written
    by the real MXFP4 key writer (mxfp4_k_norm_quant_insert: LayerNorm, MXFP4,
    insert) into their kernel pages, as the runner binds them, and the latent pages
    by concat_and_cache_mla in their spec's cache dtype."""
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )

    num_tokens = slots.shape[0]
    scale = torch.ones(1, dtype=torch.float32, device=slots.device)
    filled = torch.empty(num_tokens, 16, dtype=torch.int32, device=slots.device)
    for name, spec in rank.specs.items():
        view = rank.views[name]
        if spec.cache_role != SparseCacheRole.INDEXER:
            kv_c, k_pe = (t[:num_tokens] for t in inputs[name])
            dtype = spec.cache_dtype_str
            ops.concat_and_cache_mla(kv_c, k_pe, view[:, 0], slots, dtype, scale)
            continue
        group = AttentionGroup(
            backend=DeepseekV32IndexerBackend,
            layer_names=[name],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            kernel_block_size=cpu.INDEX_KERNEL_ROWS,
            maps_slot_mapping=True,
        )
        pages = group.map_kv_cache_to_kernel_blocks(view).squeeze(1)
        group.kernel_slot_mapping = torch.empty_like(slots)[None]
        page_slots = group.map_to_kernel_slot_mapping(slots)
        keys = inputs[name][0][:num_tokens]
        args = (*norm, 1e-6, page_slots, pages, filled, num_tokens)
        mxfp4_k_norm_quant_insert(keys, *args)


@gpu.requires_gpu
@pytest.mark.parametrize("latent", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize(
    "num_tokens,num_cached",
    [(3 * 16384 + 5, 0), (3 * 16384 + 5, 1), (5 * 8192 + 1, 0)],
    ids=["odd_p_blocks", "d_prefix_hit", "even_p_blocks"],
)
def test_real_mxfp4_writer_place_and_exchange_like_direct_writes(
    mxfp4_rows, num_tokens, num_cached, latent
):
    """A7 with MXFP4 indexer keys: KV (fp8 or nvfp4_ds_mla DSA rows), MXFP4 keys
    and drafter rows written by the real writers at the interleave-1 slot mappings
    of 8 P ranks, then pulled as today and exchanged by the Triton exchanger, equal
    what the same writers put in 16 D ranks directly: the whole caches, byte for
    byte."""
    if not current_platform.is_device_capability_family(100):
        pytest.skip("the MXFP4 key writer converts to E2M1 on SM10x")
    ops = gpu._cache_ops()
    device = torch.device("cuda")
    specs = k31_specs(latent, 2, 1)
    gen = torch.Generator(device=device).manual_seed(num_tokens)
    inputs = {}
    for name, spec in specs.items():
        dims = (128,) if spec.cache_role == SparseCacheRole.INDEXER else (512, 64)
        inputs[name] = tuple(
            torch.randn(
                (num_tokens, dim), dtype=torch.bfloat16, device=device, generator=gen
            )
            for dim in dims
        )
    weight = torch.rand(128, device=device, generator=gen) + 0.5
    norm = weight, torch.randn(128, device=device, generator=gen) / 10
    num_d = cdiv(num_tokens, cpu.D_DCP * 1024)
    num_p = cdiv(num_tokens, cpu.P_DCP * 1024)
    d_ids, p_ids = cpu._block_tables(np.random.default_rng(num_tokens), num_d, num_p)

    p_ranks = [gpu._Rank(specs, num_p + 2, device) for _ in range(cpu.P_DCP)]
    for p, rank in enumerate(p_ranks):
        slots = gpu._slot_mapping(p, cpu.P_DCP, p_ids, num_tokens, device)
        _write_tokens(ops, rank, slots, inputs, norm)
    direct = [gpu._Rank(specs, num_d + 2, device) for _ in range(cpu.D_DCP)]
    pulled = [gpu._Rank(specs, num_d + 2, device) for _ in range(cpu.D_DCP)]
    cached_tokens = min(num_tokens, num_cached * cpu.D_DCP * 1024)
    for r in range(cpu.D_DCP):
        slots = gpu._slot_mapping(r, cpu.D_DCP, d_ids, num_tokens, device)
        _write_tokens(ops, direct[r], slots, inputs, norm)
        if cached_tokens:
            slots = gpu._slot_mapping(r, cpu.D_DCP, d_ids, cached_tokens, device)
            _write_tokens(ops, pulled[r], slots, inputs, norm)
    # The caches compared below start as zeros: the writers filled every layer,
    # and every token's latent row on exactly one D rank.
    for name in specs:
        assert all(rank.views[name].count_nonzero() for rank in direct), name
    latent_name = next(
        n for n, s in specs.items() if s.cache_role != SparseCacheRole.INDEXER
    )
    rows = [
        rank.views[latent_name][:, 0].flatten(0, 1).any(-1).sum() for rank in direct
    ]
    assert sum(int(n) for n in rows) == num_tokens

    cpu._pull_as_today(pulled, d_ids, num_cached, p_ranks, p_ids)
    units = pair_units([d_ids[num_cached:]], [num_cached], [p_ids], groups=[0])
    for p in range(cpu.P_DCP):
        pair = [pulled[p], pulled[p + cpu.P_DCP]]
        gpu._exchange_pair(
            gpu._triton_exchanger,
            [r.buf for r in pair],
            [r.table for r in pair],
            [pair_layout(r, cpu.D_DCP, cpu.P_DCP, 1024, 1) for r in (p, p + cpu.P_DCP)],
            units,
            2,
        )
    torch.accelerator.synchronize()
    for r in range(cpu.D_DCP):
        assert torch.equal(pulled[r].buf, direct[r].buf), f"D rank {r}"
