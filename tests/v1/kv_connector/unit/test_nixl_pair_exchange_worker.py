# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder side of the DCP pair exchange, on CPU.

A P DCP8 -> D DCP16 pull at interleave 1 runs through the real NIXL workers and
read path over the real planner's Kimi-K3.1 geometry, then the exchange (A3).
The runner hook that runs the exchange, its fail-closed guard and the reuse
guard are checked against the model runner's call order (A5). The exchanger is
a fake on the torch reference of ``dcp_pair_exchange``, paired across ranks as
its NCCL exchange pairs them.
"""

import contextlib
import copy
import functools
import itertools
import time
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit.test_nixl_dcp_pair_exchange import (
    INDEX_KERNEL_ROWS,
    INDEX_ROW_BYTES,
    INDEX_VALUE_BYTES,
    _canonical_tokens,
    _k31_dsa_specs,
    _row_tags,
)
from tests.v1.kv_connector.unit.utils import expand_strided_descs
from vllm.config import CacheConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_worker,
    dcp_pair_exchange,
    page_zeroer,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    NixlPairModeError,
    PairUnit,
    build_pair_row_table,
    pair_layout,
    pair_units,
    reference_pair_exchange,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.page_zeroer import PageZeroer
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
)
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR, ActiveKVConnector
from vllm.v1.worker.utils import AttentionGroup, allocate_kv_cache

pytestmark = pytest.mark.cpu_test

P_DCP, D_DCP = 8, 16
NUM_DSA, NUM_KDA = 24, 35
CANARY, STALE, P_CANARY, KDA_FILL = 0x5A, 0x3C, 0x77, 0xA0
# The transfer group the A5 stub workers exchange.
PAIR_GROUP = 2


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Many small tensor ops: keep them off OpenMP on a shared host."""
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(num_threads)


@pytest.fixture
def k31_env(monkeypatch):
    """Kimi-K3.1's KDA state groups and DS conv layout, as in the recipe."""
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", "2")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    mamba_utils.get_conv_state_layout.cache_clear()
    yield
    mamba_utils.get_conv_state_layout.cache_clear()


# ----------------------------------------------------------------------------
# Fakes: NIXL transport, exchanger, CUDA events
# ----------------------------------------------------------------------------


class _Memory:
    """Every rank's KV storage, which NIXL descriptors address."""

    def __init__(self):
        self.buffers: list[torch.Tensor] = []

    def holding(self, addr: int, length: int = 1) -> torch.Tensor:
        for buf in self.buffers:
            if buf.data_ptr() <= addr and addr + length <= buf.data_ptr() + len(buf):
                return buf
        raise AssertionError(f"{length} bytes at {addr:#x} are outside every cache")

    def bytes(self, addr: int, length: int) -> torch.Tensor:
        buf = self.holding(addr, length)
        start = addr - buf.data_ptr()
        return buf[start : start + length]


class _ByteMovingNixlWrapper:
    """FakeNixlWrapper of test_nixl_connector.py (which needs ray) that resolves
    the descriptor ids of each READ to the (address, length) pairs it prepared,
    and copies the bytes."""

    def __init__(self, agent_name: str, *args, memory: _Memory, log: list, **kwargs):
        self.name, self.memory, self.log = agent_name, memory, log
        self._handles = itertools.count(1)
        self._dlists: dict[int, np.ndarray] = {}
        self._xfers: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return agent_metadata.decode()

    def get_reg_descs(self, caches_data, memory_type: str):
        return caches_data

    def register_memory(self, descs, backends=None) -> None:
        pass

    def deregister_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str):
        return blocks_data

    def prep_xfer_dlist(self, agent_name: str, descs, mem_type=None) -> int:
        handle = next(self._handles)
        self._dlists[handle] = expand_strided_descs(descs)[:, :2].astype(np.int64)
        return handle

    def make_prepped_xfer(
        self, xfer_type, local_handle, local_ids, remote_handle, remote_ids, **kwargs
    ) -> int:
        assert xfer_type == "READ"
        handle = next(self._handles)
        self._xfers[handle] = (
            self._dlists[local_handle][np.asarray(local_ids, dtype=np.int64)],
            self._dlists[remote_handle][np.asarray(remote_ids, dtype=np.int64)],
        )
        return handle

    def transfer(self, handle: int) -> str:
        local, remote = self._xfers[handle]
        self.log.append(("read", self.name))
        for (dst, length), (src, src_length) in zip(
            local.tolist(), remote.tolist(), strict=True
        ):
            assert length == src_length
            self.memory.bytes(dst, length).copy_(self.memory.bytes(src, length))
        return "PROC"

    def check_xfer_state(self, handle: int) -> str:
        return "DONE"

    def get_xfer_telemetry(self, handle: int):
        return None

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def release_dlist_handle(self, handle: int) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    def remove_remote_agent(self, agent: str) -> None:
        pass

    def check_remote_metadata(self, agent: str) -> bool:
        return True


class _FakeEvent:
    def __init__(self, log: list, name: str):
        self.log, self.name = log, name

    def synchronize(self) -> None:
        self.log.append(("synchronize", self.name))

    def query(self) -> bool:
        # Still queued on the device: the drain decides by block overlap.
        return False


class _Fabric:
    """Pairs the fake exchangers as NCCL pairs DCP ranks r and r ^ W: the
    second of a pair to exchange runs the torch reference over both."""

    def __init__(self, memory: _Memory, log: list):
        self.memory, self.log = memory, log
        self.waiting: dict[int, tuple[_FakeExchanger, list[PairUnit]]] = {}
        self.builds: dict[int, dict] = {}

    def build(self, regions, *, index_kernel_rows, index_value_bytes, layout, **kw):
        """build_dcp_pair_exchanger, on the stage-2 row table."""
        table = build_pair_row_table(
            regions,
            index_kernel_rows=index_kernel_rows,
            index_value_bytes=index_value_bytes,
        )
        self.builds[layout.dcp_rank] = dict(
            regions=regions,
            index_kernel_rows=index_kernel_rows,
            index_value_bytes=index_value_bytes,
            layout=layout,
            **kw,
        )
        return _FakeExchanger(self, self.memory.holding(table.base_addr), table, layout)


class _FakeExchanger:
    """DcpPairExchanger's API on reference_pair_exchange."""

    def __init__(self, fabric: _Fabric, buf, table, layout):
        self.fabric, self.buf, self.table, self.layout = fabric, buf, table, layout
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1

    def for_layout(self, layout):
        """Another exchange stage of this rank, on the same buffers."""
        return type(self)(self.fabric, self.buf, self.table, layout)

    def exchange(self, units) -> None:
        units = list(units)
        rank = self.layout.dcp_rank
        self.fabric.log.append(("exchange", rank, units))
        partner = self.fabric.waiting.pop(self.layout.partner, None)
        if partner is None:
            self.fabric.waiting[rank] = (self, units)
            return
        other, other_units = partner
        assert other_units == units, "partners must exchange the same units"
        pair = sorted((self, other), key=lambda e: e.layout.parity)
        reference_pair_exchange(
            [e.buf for e in pair],
            [{u.group: e.table for u in units} for e in pair],
            units,
        )

    def record_event(self) -> _FakeEvent:
        return _FakeEvent(self.fabric.log, f"exchange {len(self.fabric.log)}")


@pytest.fixture
def fabric(monkeypatch):
    fabric = _Fabric(_Memory(), [])
    # The worker imports the builder from the module when pair mode starts.
    monkeypatch.setattr(
        dcp_pair_exchange, "build_dcp_pair_exchanger", fabric.build, raising=False
    )
    # The page zeroer stages block ids in pinned memory, which CPU hosts lack.
    monkeypatch.setattr(
        page_zeroer,
        "async_tensor_h2d",
        lambda data, device, dtype: torch.tensor(data, dtype=dtype, device=device),
    )
    return fabric


# ----------------------------------------------------------------------------
# Ranks: the real planner's packed K3.1 geometry, the real NIXL worker
# ----------------------------------------------------------------------------


def _kda_specs(block_size: int, tp: int, num_kda: int) -> dict[str, MambaSpec]:
    """K3.1's KDA state pages, shrunk: [Q|K|V] conv channels by conv rows, then
    the recurrent heads, both sharded by TP."""
    heads, head_dim, conv_rows = 96, 16, 4
    spec = MambaSpec(
        block_size=block_size,
        shapes=(
            (3 * heads * head_dim // tp, conv_rows),
            (heads // tp, head_dim, head_dim),
        ),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    return {f"model.layers.{100 + i}.linear_attn": spec for i in range(num_kda)}


def _bind_indexer(view: torch.Tensor) -> torch.Tensor:
    """The kernel-page view an indexer layer binds, as
    DeepseekV32IndexerCache.bind_kv_cache does."""
    return view.squeeze(1)


# The release DSpark draft (Qwen3 DFlash backbone): 5 sliding-window GQA layers of
# 16 KV heads of 64, numbered after the target's layers. It runs at DCP 1, so each
# rank holds every token of its TP share of the heads.
DRAFT_KV_HEADS, DRAFT_HEAD_SIZE, NUM_DRAFT_LAYERS = 16, 64, 5
TARGET_LAYERS = 200


def _gqa_draft_specs(
    block_size: int, tp: int, window: int
) -> dict[str, SlidingWindowSpec]:
    """The release draft's KV cache specs at TP ``tp``, as the V2 runner marks
    them under a DCP target (dcp_sharded=False)."""
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=DRAFT_KV_HEADS // tp,
        head_size=DRAFT_HEAD_SIZE,
        dtype=torch.bfloat16,
        sliding_window=window,
        dcp_sharded=False,
        is_draft=True,
    )
    return {
        f"model.layers.{TARGET_LAYERS + i}.self_attn.attn": spec
        for i in range(NUM_DRAFT_LAYERS)
    }


def _layer_spec(group: KVCacheGroupSpec, name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[name]
    return spec


class _Rank:
    """One rank: its packed KV cache as the real planner lays it out and the
    runner binds it, and its real NIXL pull worker on a fake transport."""

    def __init__(
        self,
        fabric: _Fabric,
        tp: int,
        rank: int,
        num_blocks: int,
        *,
        block_size: int,
        num_dsa: int = NUM_DSA,
        num_kda: int = NUM_KDA,
        num_draft: int = 0,
        extra_config: dict | None = None,
        bind_indexer: Callable[[torch.Tensor], torch.Tensor] | None = _bind_indexer,
        draft_window: int | None = None,
        decoder: bool | None = None,
        draft_hit_unit: int | None = None,
        dcp: int | None = None,
    ):
        # KDA layers first: the KV cache groups are then KDA, KDA, DSA, so the
        # attention group is not group 0. With ``draft_window``, the release
        # draft's sliding-window GQA layers follow; with ``draft_hit_unit``,
        # Kimi's grouping packs its blocks of that many tokens as row slots.
        self.specs = {
            **_kda_specs(block_size, tp, num_kda),
            **_k31_dsa_specs(block_size, num_dsa, num_draft),
            **(_gqa_draft_specs(block_size, tp, draft_window) if draft_window else {}),
        }
        self.gqa_draft = draft_window is not None
        self.decoder = tp == D_DCP if decoder is None else decoder
        cache_config = CacheConfig(block_size=block_size)
        cache_config.kv_cache_layout = "BLHNC"
        cache_config.num_gpu_blocks_override = num_blocks
        config = SimpleNamespace(
            cache_config=cache_config,
            attention_config=SimpleNamespace(hisparse_config=None),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        )
        if draft_hit_unit:
            cache_config.prefix_match_unit = draft_hit_unit
            groups = GQADraftKVCacheGrouping.get_kv_cache_groups(config, self.specs)
        else:
            groups = get_kv_cache_groups(config, self.specs)
        self.page_bytes = {
            name: _layer_spec(group, name).page_size_bytes
            for group in groups
            for name in group.layer_names
        }
        self.kv_cache_config = get_kv_cache_config_from_groups(config, groups, 1)
        self.views = allocate_kv_cache(
            self.kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC
        )
        storage = next(iter(self.views.values())).untyped_storage()
        self.buf = torch.empty(0, dtype=torch.uint8).set_(storage)
        fabric.memory.buffers.append(self.buf)
        self.dsa_group = next(
            g
            for g, group in enumerate(groups)
            if isinstance(group.kv_cache_spec.first_spec, MLAAttentionSpec)
        )
        self.dsa_layers = groups[self.dsa_group].layer_names
        self.draft_groups = [
            g
            for g, group in enumerate(groups)
            if isinstance(group.kv_cache_spec.first_spec, SlidingWindowSpec)
        ]
        # Tokens per draft block, and the draft blocks one row holds.
        self.draft_block_size = block_size
        self.draft_slots = 1
        if self.draft_groups:
            draft_group = groups[self.draft_groups[0]]
            self.draft_block_size = draft_group.kv_cache_spec.block_size
            self.draft_slots = draft_group.blocks_per_pool_block
        # As the runner binds them: indexers see 64-row kernel pages.
        self.kernel_pages = {
            name: self._indexer_group(name).map_kv_cache_to_kernel_blocks(view)
            for name, view in self.views.items()
            if self._is_indexer(name)
        }
        layers = {
            name: SimpleNamespace(kv_cache=bind_indexer(view))
            for name, view in self.kernel_pages.items()
            if bind_indexer is not None
        }

        self.tp, self.rank, self.block_size = tp, rank, block_size
        self.dcp = tp if dcp is None else dcp
        self.worker = self._start_worker(
            fabric, layers, {"kv_lease_check": False, **(extra_config or {})}
        )
        # _build_page_zeroer skips CPU caches (host pools); zero them here.
        self.worker._page_zeroer = PageZeroer(
            {
                g: [self.views[name] for name in group.layer_names]
                for g, group in enumerate(groups)
                if isinstance(group.kv_cache_spec.first_spec, AttentionSpec)
            },
            {},
        )

    def _is_indexer(self, name: str) -> bool:
        spec = self.specs[name]
        return isinstance(spec, MLAAttentionSpec) and (
            spec.cache_role == SparseCacheRole.INDEXER
        )

    def _indexer_group(self, name: str) -> AttentionGroup:
        return AttentionGroup(
            backend=DeepseekV32IndexerBackend,
            layer_names=[name],
            kv_cache_spec=self.specs[name],
            kv_cache_group_id=0,
            kernel_block_size=INDEX_KERNEL_ROWS,
            maps_slot_mapping=True,
        )

    def _start_worker(self, fabric: _Fabric, layers: dict, extra: dict):
        """The worker, built and registered as the runner does it."""
        tp = self.tp
        decoder = self.decoder
        speculative_config = None
        if self.gqa_draft:
            speculative_config = SimpleNamespace(
                draft_model_config=SimpleNamespace(
                    use_mla=False,
                    get_total_num_kv_heads=lambda: DRAFT_KV_HEADS,
                    get_total_num_hidden_layers=lambda: NUM_DRAFT_LAYERS,
                )
            )
        transfer_config = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_consumer" if decoder else "kv_producer",
            engine_id="D" if decoder else "P",
            kv_connector_extra_config=extra,
            kv_buffer_device="cuda",
        )
        vllm_config = SimpleNamespace(
            kv_transfer_config=transfer_config,
            cache_config=SimpleNamespace(
                block_size=self.block_size,
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
                enable_prefix_caching=True,
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=self.dcp,
                pipeline_parallel_size=1,
                cp_kv_cache_interleave_size=1,
            ),
            model_config=SimpleNamespace(
                use_mla=True,
                enable_sleep_mode=False,
                hf_text_config=SimpleNamespace(
                    dsa_config=SimpleNamespace(index_head_dim=INDEX_VALUE_BYTES)
                ),
                get_total_num_hidden_layers=lambda: TARGET_LAYERS,
            ),
            speculative_config=speculative_config,
            use_v2_model_runner=True,
            compilation_config=SimpleNamespace(static_forward_context=layers),
        )
        platform = SimpleNamespace(
            device_type="cuda",
            get_nixl_memory_type=lambda: None,
            is_rocm=lambda: False,
        )
        backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE.get_class()
        with contextlib.ExitStack() as stack:
            for name, value in (
                (
                    "NixlWrapper",
                    functools.partial(
                        _ByteMovingNixlWrapper, memory=fabric.memory, log=fabric.log
                    ),
                ),
                ("current_platform", platform),
                ("get_tensor_model_parallel_rank", lambda: self.rank),
                ("get_tensor_model_parallel_world_size", lambda: tp),
                ("get_current_attn_backends", lambda *args: [backend]),
                # Packed (BLHNC) attention caches never split blocks.
                ("get_current_attn_backends_and_specs", lambda *args: ([], [])),
                ("compute_nixl_compatibility_hash", lambda *args, **kw: "hash"),
            ):
                stack.enter_context(patch.object(base_worker, name, value))
            worker = NixlPullConnectorWorker(
                vllm_config, transfer_config.engine_id, self.kv_cache_config
            )
            worker.register_kv_caches(self.views)
        return worker

    # Rows, read and written through the views the attention backends read.

    def arrays(self) -> list[tuple[str, str, int]]:
        """(layer, part, row width) of each array of the DSA pages."""
        arrays = []
        for name in self.dsa_layers:
            if self._is_indexer(name):
                arrays.append((name, "value", INDEX_VALUE_BYTES))
                arrays.append((name, "scale", INDEX_ROW_BYTES - INDEX_VALUE_BYTES))
            else:
                arrays.append((name, "row", self.specs[name].head_size))
        return arrays

    def _indexer_part(self, name: str, part: str, block_ids: list[int]):
        """The blocks' kernel pages flattened, their ids, and the byte range of
        ``part`` in a page: the cache kernel stores values, then scales."""
        pages = self.kernel_pages[name]
        per_block = self.block_size // INDEX_KERNEL_ROWS
        # Kernel page j of block b is b * stride + j (map_to_kernel_block_table).
        stride = self.views[name].stride(0) // pages.stride(0)
        ids = torch.as_tensor(
            (np.asarray(block_ids)[:, None] * stride + np.arange(per_block)).ravel()
        )
        values = INDEX_KERNEL_ROWS * INDEX_VALUE_BYTES
        if part == "value":
            return pages, ids, (0, values)
        return pages, ids, (values, INDEX_KERNEL_ROWS * INDEX_ROW_BYTES)

    def read_rows(self, array: tuple[str, str, int], block_ids: list[int]):
        name, part, width = array
        if part == "row":
            return self.views[name][block_ids, 0].numpy()
        pages, ids, (lo, hi) = self._indexer_part(name, part, block_ids)
        flat = pages[ids].reshape(len(ids), -1)[:, lo:hi]
        return flat.reshape(len(block_ids), self.block_size, width).numpy()

    def write_rows(self, array, block_ids: list[int], rows: np.ndarray) -> None:
        name, part, width = array
        if not block_ids:
            return
        if part == "row":
            self.views[name][block_ids, 0] = torch.from_numpy(rows)
            return
        pages, ids, (lo, hi) = self._indexer_part(name, part, block_ids)
        flat = pages[ids].reshape(len(ids), -1)
        flat[:, lo:hi] = torch.from_numpy(rows).reshape(len(ids), hi - lo)
        pages[ids] = flat.reshape(pages[ids].shape)

    def page_mask(self, layers: list[str], block_ids: list[int]) -> torch.Tensor:
        """Bytes of the layers' pages in the given blocks."""
        mask = torch.zeros_like(self.buf, dtype=torch.bool)
        for name in layers:
            for b in block_ids:
                start = self.views[name][b].data_ptr() - self.buf.data_ptr()
                mask[start : start + self.page_bytes[name]] = True
        return mask

    def kda_groups(self) -> list[int]:
        return [
            g
            for g in range(len(self.kv_cache_config.kv_cache_groups))
            if g != self.dsa_group and g not in self.draft_groups
        ]

    def group_layers(self, group: int) -> list[str]:
        return self.kv_cache_config.kv_cache_groups[group].layer_names

    def draft_heads(self, name: str, block: int) -> np.ndarray:
        """``[heads, tokens, bytes]`` of a draft layer's page of ``block``."""
        page = self.views[name][block].contiguous()
        return page.view(torch.uint8).numpy().reshape(page.shape[0], page.shape[1], -1)

    def write_draft_heads(self, name: str, block: int, data: np.ndarray) -> None:
        view = self.views[name][block]
        view.view(torch.uint8).copy_(
            torch.from_numpy(data).reshape(view.view(torch.uint8).shape)
        )


def _handshake(
    decoder: _Rank, prefillers: dict[int, _Rank], remote_tp: int = P_DCP
) -> None:
    """The decoder rank's real handshake with the P ranks it reads from; a
    fake side channel serves their real handshake metadata. P runs DCP = TP."""

    class _SideChannel:
        rank: int

        def setsockopt(self, *args) -> None:
            pass

        def send(self, msg: bytes) -> None:
            _, _, self.rank = msgspec.msgpack.decode(msg)

        def recv_multipart(self) -> list[bytes]:
            payload = prefillers[self.rank].worker.xfer_handshake_metadata
            return [
                msgspec.msgpack.encode(payload),
                msgspec.msgpack.encode(time.perf_counter()),
            ]

    worker = decoder.worker
    with (
        patch.object(
            base_worker,
            "zmq_ctx",
            lambda *args: contextlib.nullcontext(_SideChannel()),
        ),
        patch.object(
            base_worker, "current_platform", SimpleNamespace(set_device=lambda _: None)
        ),
    ):
        agents, clock_offset = worker._nixl_handshake(
            "localhost", 1, remote_tp, "P", remote_tp
        )
    worker._remote_agents["P"] = agents
    worker._engine_clock_offset["P"] = clock_offset


def _per_group(rank: _Rank, dsa, kda: list) -> tuple:
    """A value per KV cache group: ``dsa`` for the DSA group, ``kda`` in order
    for the KDA groups."""
    values = dict(zip(rank.kda_groups(), kda))
    values[rank.dsa_group] = dsa
    return tuple(values[g] for g in range(len(values)))


def _recv_metadata(
    req_id: str, local, computed, remote, num_tokens: int, remote_tp: int = P_DCP
):
    """The pull of one request, as the scheduler sends it to every D rank."""
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id=req_id,
        local_block_ids=copy.deepcopy(local),
        local_num_computed_blocks=computed,
        awaiting_kvs=True,
        kv_transfer_params={
            "remote_engine_id": "P",
            "remote_request_id": f"{req_id}-P",
            "remote_host": "localhost",
            "remote_port": 1,
            "remote_num_tokens": num_tokens,
            "remote_block_ids": copy.deepcopy(remote),
            "tp_size": remote_tp,
            "dcp_size": remote_tp,
        },
    )
    return metadata


def _facade(worker) -> NixlPullConnector:
    connector = object.__new__(NixlPullConnector)
    connector.connector_worker = worker
    return connector


# ----------------------------------------------------------------------------
# A3: pull plus exchange through the real read path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_cached,num_draft,extra_d,groups",
    [
        (2500, 0, 0, 1, 2),
        (3000, 1, 5, 0, 2),
        (2100, 1, 0, 1, 2),
        (3000, 1, 5, 0, 3),
        (3000, 1, 5, 0, 4),
    ],
    ids=[
        "odd-P-blocks,D-lookahead",
        "even-P-blocks,D-prefix-hit,drafter",
        "odd-P-blocks,D-prefix-hit,D-lookahead",
        "even-P-blocks,D-prefix-hit,drafter,3-state-groups",
        "even-P-blocks,D-prefix-hit,drafter,4-state-groups",
    ],
)
def test_pull_then_exchange_on_k31_geometry(
    k31_env, fabric, monkeypatch, num_tokens, num_cached, num_draft, extra_d, groups
):
    """A3: every D rank's real read path pulls whole pages (attention from P
    rank r % 8, KDA state from r // 2) into the real planner's packed K3.1
    blocks, zeroing what it leaves unwritten; after the exchange every DSA row
    (latent, indexer values and scales, drafter) holds its interleave-1 token,
    the prefix-cache hit is kept, and the KDA state, row padding and every
    other block are untouched. Also with the 3 and 4 KDA state groups of the
    nvfp4_ds_mla and FP32-state recipes."""
    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", str(groups))
    block_size = 64
    num_d = cdiv(num_tokens, D_DCP * block_size) + extra_d
    num_p = cdiv(num_tokens, P_DCP * block_size)
    rng = np.random.default_rng(num_tokens)
    # Per rank pool: the request's DSA blocks, a KDA block per group, a spare.
    d_pool = rng.permutation(num_d + groups + 1).tolist()
    p_pool = rng.permutation(num_p + groups + 1).tolist()
    d_ids, d_kda = d_pool[:num_d], d_pool[num_d : num_d + groups]
    p_ids, p_kda = p_pool[:num_p], p_pool[num_p : num_p + groups]

    geometry = dict(block_size=block_size, num_draft=num_draft)
    prefillers = {
        p: _Rank(fabric, P_DCP, p, num_p + groups + 1, **geometry) for p in range(P_DCP)
    }
    arrays = prefillers[0].arrays()
    num_arrays = len(arrays)
    for p, rank in prefillers.items():
        rank.buf.fill_(P_CANARY)
        tokens = _canonical_tokens(p, P_DCP, block_size, num_p)
        held = (tokens < num_tokens)[..., None]
        for a, array in enumerate(arrays):
            valid = _row_tags(tokens, a, array[2])
            junk = _row_tags(tokens, a + num_arrays, array[2])
            rank.write_rows(array, p_ids, np.where(held, valid, junk))
        for g, block in zip(rank.kda_groups(), p_kda):
            rank.buf[rank.page_mask(rank.group_layers(g), [block])] = KDA_FILL + p

    pair = {"dcp_pair_exchange": True}
    decoders = [
        _Rank(fabric, D_DCP, r, num_d + groups + 1, extra_config=pair, **geometry)
        for r in range(D_DCP)
    ]
    for r, rank in enumerate(decoders):
        assert rank.arrays() == arrays
        rank.buf.fill_(CANARY)
        # A stale owner's bytes in the pull targets; the hit holds D's rows.
        rank.buf[rank.page_mask(rank.dsa_layers, d_ids[num_cached:])] = STALE
        for g, block in zip(rank.kda_groups(), d_kda):
            rank.buf[rank.page_mask(rank.group_layers(g), [block])] = STALE
        tokens = _canonical_tokens(r, D_DCP, block_size, num_cached)
        for a, array in enumerate(arrays):
            rank.write_rows(array, d_ids[:num_cached], _row_tags(tokens, a, array[2]))
        _handshake(rank, prefillers)

    rank0 = decoders[0]
    assert len(rank0.kda_groups()) == groups
    local = _per_group(rank0, d_ids[num_cached:], [[b] for b in d_kda])
    computed = _per_group(rank0, num_cached, [0] * groups)
    remote = _per_group(rank0, p_ids, [[b] for b in p_kda])
    for rank in decoders:
        rank.worker.start_load_kv(
            _recv_metadata("req", local, computed, remote, num_tokens)
        )
        results = rank.worker.get_transfer_results()
        assert results.finished_recving == {"req"} and not results.failed_recving

    # The next step: every rank runs the plan before its forward.
    units = pair_units(local, computed, remote, groups=[rank0.dsa_group])
    assert units[-1].has_partner == ((num_p - 2 * num_cached) % 2 == 0)
    for rank in decoders:
        step = NixlConnectorMetadata()
        step.reqs_to_pair_exchange = {"req": units}
        connector = _facade(rank.worker)
        connector.finalize_loaded_kv(step)
        connector.bind_connector_metadata(step)
    assert not fabric.waiting, "a rank exchanged without its partner"

    for r, rank in enumerate(decoders):
        build = fabric.builds[r]
        assert build["layout"] == pair_layout(r, D_DCP, P_DCP, block_size, 1)
        assert sorted(region.layer_name for region in build["regions"]) == sorted(
            rank.dsa_layers
        )
        # The indexer page format comes from the bound kernel views and config.
        assert (build["index_kernel_rows"], build["index_value_bytes"]) == (
            INDEX_KERNEL_ROWS,
            INDEX_VALUE_BYTES,
        )
        assert build["chunk_units"] == 4 and build["device"] == torch.device("cpu")
        assert rank.worker._pair_exchanger.first.warm_ups == 1
        assert rank.worker._pair_group == rank.dsa_group != 0

        tokens = _canonical_tokens(r, D_DCP, block_size, num_d)
        held = tokens < num_tokens
        for a, array in enumerate(arrays):
            got = rank.read_rows(array, d_ids)
            valid = _row_tags(tokens, a, array[2])
            junk = _row_tags(tokens, a + num_arrays, array[2])
            np.testing.assert_array_equal(
                got[held], valid[held], err_msg=f"rank {r} {array}"
            )
            past = got[~held]
            assert ((past == 0).all(-1) | (past == junk[~held]).all(-1)).all(), (
                f"rank {r} {array}: a row past L holds another token or stale bytes"
            )
        kda = torch.zeros_like(rank.buf, dtype=torch.bool)
        for g, block in zip(rank.kda_groups(), d_kda):
            kda |= rank.page_mask(rank.group_layers(g), [block])
        assert (rank.buf[kda] == KDA_FILL + r // 2).all(), f"rank {r}: KDA state"
        outside = ~(kda | rank.page_mask(rank.dsa_layers, d_ids))
        assert (rank.buf[outside] == CANARY).all(), f"rank {r}: padding or blocks"


# ----------------------------------------------------------------------------
# A3 with the release draft: its GQA sliding-window KV at DCP 1, pulled whole
# ----------------------------------------------------------------------------

NULL_FILL, DRAFT_WINDOW = 0x11, 192
# The prefix-cache hit unit the packed draft's blocks take (128 in production).
DRAFT_HIT_UNIT = 16


def _draft_page(name: str, heads: range, first_token: int, rank: _Rank, junk=False):
    """``[heads, block, bytes]`` of a draft page holding the given global heads
    for the tokens from ``first_token``: unique per (layer, head, token)."""
    layer = int(name.split(".")[2]) - TARGET_LAYERS
    tokens = np.arange(first_token, first_token + rank.draft_block_size)
    width = rank.draft_heads(name, 0).shape[-1]
    return np.stack(
        [
            _row_tags(tokens, 16 * layer + head + (128 if junk else 0), width)
            for head in heads
        ]
    )


@pytest.mark.parametrize(
    "p_tp,d_tp,num_tokens,num_cached,num_dsa,groups,hit_unit",
    [
        (P_DCP, D_DCP, 3000, 0, NUM_DSA, 4, None),
        (P_DCP, D_DCP, 1100, 1, 2, 2, None),
        (4, 4, 3000, 0, NUM_DSA, 4, None),
        (4, 4, 1100, 1, 2, 2, None),
        (P_DCP, D_DCP, 3000, 0, NUM_DSA, 4, DRAFT_HIT_UNIT),
        (P_DCP, D_DCP, 1100, 1, NUM_DSA, 4, DRAFT_HIT_UNIT),
        (4, 4, 3000, 0, NUM_DSA, 4, DRAFT_HIT_UNIT),
        (4, 4, 1100, 4, NUM_DSA, 4, DRAFT_HIT_UNIT),
    ],
    ids=[
        "P-tp8dcp8,D-tp16dcp16-pair,4-state-groups",
        "P-tp8dcp8,D-tp16dcp16-pair,D-hit-in-window,3-draft-groups",
        "P-tp4dcp4,D-tp4dcp4,4-state-groups",
        "P-tp4dcp4,D-tp4dcp4,D-hit-in-window,3-draft-groups",
        "P-tp8dcp8,D-tp16dcp16-pair,slots-24-to-48",
        "P-tp8dcp8,D-tp16dcp16-pair,slots-24-to-48,D-hit-in-window",
        "P-tp4dcp4,D-tp4dcp4,slots-12",
        "P-tp4dcp4,D-tp4dcp4,slots-12,D-hit-in-window",
    ],
)
def test_pull_release_draft_on_k31_geometry(
    k31_env,
    fabric,
    monkeypatch,
    p_tp,
    d_tp,
    num_tokens,
    num_cached,
    num_dsa,
    groups,
    hit_unit,
):
    """The release DSpark draft's KV (5 sliding-window GQA layers, 16 KV heads,
    DCP 1: every rank holds every token of its heads) on the chosen P/D shapes:
    P TP8/DCP8 -> D TP16/DCP16 with the DCP pair exchange (C1 ... C160), and
    TP4/DCP4 on both sides (C512's DP ranks). Each D rank's real handshake and
    read path pull its heads' slice of the draft pages from the P rank owning
    them (D head r of P rank r // 2, or rank r at equal TP) into exactly its
    window blocks: paired from the end of P's block list, after dropping P's
    speculative lookahead block, never reading P's null or out-of-window
    blocks. The target's DSA rows, KDA state and every other byte are as
    without a draft, and each P rank waits for exactly its readers. The
    chosen points pin 4 KDA state groups (VLLM_PACKED_KV_STATE_GROUPS).

    With ``hit_unit``, Kimi's grouping cuts the draft's blocks to the
    prefix-cache hit unit and packs them as slots of the rows
    (``blocks_per_pool_block``), twice as many per row on D TP16 as on P TP8:
    the draft's block IDs are slot IDs on each side, D's window slots get P's
    slots of the same positions, and no other byte of their rows changes."""
    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", str(groups))
    block_size = 64
    draft_block = hit_unit or block_size
    pair = d_tp == 2 * p_tp
    d_heads = DRAFT_KV_HEADS // d_tp
    p_heads = DRAFT_KV_HEADS // p_tp
    num_d = cdiv(num_tokens, d_tp * block_size)
    num_p = cdiv(num_tokens, p_tp * block_size)
    # Draft blocks by token position: P holds a few blocks before D's window
    # (out-of-window ones are null) and a lookahead block past its tokens; D
    # allocates only its window, after its prefix-cache hit.
    num_positions = cdiv(num_tokens, draft_block)
    hit = num_cached * d_tp * block_size
    p_first = max(0, (num_tokens - DRAFT_WINDOW) // draft_block - 1)
    window_first = max(0, num_tokens - DRAFT_WINDOW + 1) // draft_block
    d_first = max(hit // draft_block, window_first)
    d_positions = range(d_first, num_positions)
    p_positions = range(p_first, num_positions + 1)

    geometry = dict(
        block_size=block_size,
        num_dsa=num_dsa,
        num_kda=4,
        draft_window=DRAFT_WINDOW,
        draft_hit_unit=hit_unit,
    )
    # The planner groups the 5 draft layers with the DSA layers' repeat count:
    # one group next to K3.1's 24 DSA layers, three next to 2.
    num_draft_groups = 1 if num_dsa == NUM_DSA else 3

    rng = np.random.default_rng(num_tokens + d_tp)
    num_d_pool = num_d + groups + num_draft_groups * len(d_positions) + 1
    num_p_pool = 1 + num_p + groups + num_draft_groups * len(p_positions) + 1
    d_pool = rng.permutation(num_d_pool).tolist()
    # P block 0 stands for the null block of the out-of-window positions.
    p_pool = (1 + rng.permutation(num_p_pool - 1)).tolist()
    d_ids, d_kda = d_pool[:num_d], d_pool[num_d : num_d + groups]
    p_ids, p_kda = p_pool[:num_p], p_pool[num_p : num_p + groups]
    d_rest, p_rest = d_pool[num_d + groups :], p_pool[num_p + groups :]

    def draft_ids(rows: list[int], slots: int, num: int) -> list[list[int]]:
        """``num`` draft block IDs per draft group from the rows the target does
        not use: the rows, or the shuffled IDs of their slots when packed (row 0
        holds the null slot)."""
        ids = rows
        if slots > 1:
            slot_ids = [row * slots + k for row in rows if row for k in range(slots)]
            ids = rng.permutation(slot_ids).tolist()
        return [ids[i * num : (i + 1) * num] for i in range(num_draft_groups)]

    prefillers = {
        p: _Rank(fabric, p_tp, p, num_p_pool, decoder=False, **geometry)
        for p in range(p_tp)
    }
    assert len(prefillers[0].draft_groups) == num_draft_groups
    assert prefillers[0].draft_block_size == draft_block
    p_slots = prefillers[0].draft_slots
    assert (p_slots > 1) == bool(hit_unit)
    p_draft = [
        [0] * p_first + ids for ids in draft_ids(p_rest, p_slots, len(p_positions))
    ]
    arrays = prefillers[0].arrays()
    num_arrays = len(arrays)
    for p, rank in prefillers.items():
        rank.buf.fill_(P_CANARY)
        rank.buf[rank.page_mask(list(rank.specs), [0])] = NULL_FILL
        tokens = _canonical_tokens(p, p_tp, block_size, num_p)
        held = (tokens < num_tokens)[..., None]
        for a, array in enumerate(arrays):
            valid = _row_tags(tokens, a, array[2])
            junk = _row_tags(tokens, a + num_arrays, array[2])
            rank.write_rows(array, p_ids, np.where(held, valid, junk))
        for g, block in zip(rank.kda_groups(), p_kda):
            rank.buf[rank.page_mask(rank.group_layers(g), [block])] = KDA_FILL + p
        heads = range(p * p_heads, (p + 1) * p_heads)
        for g, blocks in zip(rank.draft_groups, p_draft):
            for name in rank.group_layers(g):
                for q in p_positions:
                    page = _draft_page(
                        name, heads, q * draft_block, rank, junk=q >= num_positions
                    )
                    rank.write_draft_heads(name, blocks[q], page)

    extra = {"dcp_pair_exchange": True} if pair else {}
    decoders = [
        _Rank(fabric, d_tp, r, num_d_pool, decoder=True, extra_config=extra, **geometry)
        for r in range(d_tp)
    ]
    d_slots = decoders[0].draft_slots
    if hit_unit:
        # P and D tile the same rows into different slot counts.
        assert d_slots == (2 * p_slots if pair else p_slots)
    d_draft = draft_ids(d_rest, d_slots, len(d_positions))
    for r, rank in enumerate(decoders):
        assert rank.draft_groups == prefillers[0].draft_groups
        assert rank.worker._dcp_replicated_draft_groups == set(rank.draft_groups)
        rank.buf.fill_(CANARY)
        rank.buf[rank.page_mask(rank.dsa_layers, d_ids[num_cached:])] = STALE
        for g, block in zip(rank.kda_groups(), d_kda):
            rank.buf[rank.page_mask(rank.group_layers(g), [block])] = STALE
        for g, blocks in zip(rank.draft_groups, d_draft):
            rank.buf[rank.page_mask(rank.group_layers(g), blocks)] = STALE
        tokens = _canonical_tokens(r, d_tp, block_size, num_cached)
        for a, array in enumerate(arrays):
            rank.write_rows(array, d_ids[:num_cached], _row_tags(tokens, a, array[2]))
        _handshake(rank, prefillers, remote_tp=p_tp)
        # The draft's heads are owned by P rank r * p_tp // d_tp, which the
        # handshake covers besides the target's DCP and KDA sources.
        draft_source = r * p_tp // d_tp
        sources = {r % p_tp, draft_source}
        assert {rank_ for _, rank_ in rank.worker._remote_agents["P"]} == sources
        if pair:
            assert rank.worker._pair_group == rank.dsa_group

    # Every P rank frees a request once each D rank reading from it notified.
    plans = [rank.worker.tp_mappings["P"] for rank in decoders]
    for p in range(p_tp):
        readers = [plan for plan in plans if p in plan.all_source_ranks]
        assert readers and all(
            plan.consumers_for(p) == len(readers) for plan in readers
        )

    def by_group(rank: _Rank, dsa, kda: list, draft: list) -> tuple:
        values = dict(zip(rank.kda_groups(), kda)) | dict(zip(rank.draft_groups, draft))
        values[rank.dsa_group] = dsa
        return tuple(values[g] for g in range(len(values)))

    rank0 = decoders[0]
    hit_in_window = d_first - window_first
    local = by_group(rank0, d_ids[num_cached:], [[b] for b in d_kda], d_draft)
    computed = by_group(
        rank0, num_cached, [0] * groups, [hit_in_window] * num_draft_groups
    )
    remote = by_group(rank0, p_ids, [[b] for b in p_kda], p_draft)
    for rank in decoders:
        rank.worker.start_load_kv(
            _recv_metadata("req", local, computed, remote, num_tokens, p_tp)
        )
        results = rank.worker.get_transfer_results()
        assert results.finished_recving == {"req"} and not results.failed_recving

    if pair:
        # The exchanger rewrites the target's DSA pages only.
        for r in range(d_tp):
            names = [region.layer_name for region in fabric.builds[r]["regions"]]
            assert sorted(names) == sorted(decoders[r].dsa_layers)
        units = pair_units(local, computed, remote, groups=[rank0.dsa_group])
        for rank in decoders:
            step = NixlConnectorMetadata()
            step.reqs_to_pair_exchange = {"req": units}
            connector = _facade(rank.worker)
            connector.finalize_loaded_kv(step)
            connector.bind_connector_metadata(step)
        assert not fabric.waiting, "a rank exchanged without its partner"

    for r, rank in enumerate(decoders):
        heads = range(r * d_heads, (r + 1) * d_heads)
        for g, blocks in zip(rank.draft_groups, d_draft):
            for name in rank.group_layers(g):
                for block, q in zip(blocks, d_positions, strict=True):
                    np.testing.assert_array_equal(
                        rank.draft_heads(name, block),
                        _draft_page(name, heads, q * draft_block, rank),
                        err_msg=f"rank {r} {name} position block {q}",
                    )
        tokens = _canonical_tokens(r, d_tp, block_size, num_d)
        held = tokens < num_tokens
        for a, array in enumerate(arrays):
            got = rank.read_rows(array, d_ids)
            valid = _row_tags(tokens, a, array[2])
            junk = _row_tags(tokens, a + num_arrays, array[2])
            np.testing.assert_array_equal(
                got[held], valid[held], err_msg=f"rank {r} {array}"
            )
            past = got[~held]
            assert ((past == 0).all(-1) | (past == junk[~held]).all(-1)).all()
        pulled = torch.zeros_like(rank.buf, dtype=torch.bool)
        for g, block in zip(rank.kda_groups(), d_kda):
            pulled |= rank.page_mask(rank.group_layers(g), [block])
        kda_source = r * p_tp // d_tp
        assert (rank.buf[pulled] == KDA_FILL + kda_source).all(), f"rank {r}: KDA"
        pulled |= rank.page_mask(rank.dsa_layers, d_ids)
        for g, blocks in zip(rank.draft_groups, d_draft):
            pulled |= rank.page_mask(rank.group_layers(g), blocks)
        assert (rank.buf[~pulled] == CANARY).all(), f"rank {r}: padding or blocks"


# ----------------------------------------------------------------------------
# A5: the runner hook, the fail-closed guard and the reuse guard
# ----------------------------------------------------------------------------


class _LoggingExchanger:
    def __init__(self, log: list):
        self.log = log

    def exchange(self, units) -> None:
        self.log.append(("exchange", list(units)))

    def record_event(self) -> _FakeEvent:
        return _FakeEvent(self.log, f"event {len(self.log)}")


def _pair_worker(log: list | None = None) -> NixlPullConnectorWorker:
    """A worker with pair mode built for group PAIR_GROUP when given a log,
    else off."""
    worker = object.__new__(NixlPullConnectorWorker)
    if log is not None:
        worker._pair_exchanger = _LoggingExchanger(log)
        worker._pair_group = PAIR_GROUP
    return worker


def _runner_connector(kv_connector, disabled: bool = False) -> ActiveKVConnector:
    """The V2 runner's connector wrapper around ``kv_connector``."""
    active = object.__new__(ActiveKVConnector)
    active.kv_connector = kv_connector
    active._disabled = disabled
    active._pending_load_kwargs = None
    return active


def _step(metadata) -> SimpleNamespace:
    return SimpleNamespace(kv_connector_metadata=metadata, has_sync_kv_loads=False)


def _pair_step(group: int = PAIR_GROUP, **blocks: list[int]) -> NixlConnectorMetadata:
    metadata = NixlConnectorMetadata()
    metadata.reqs_to_pair_exchange = {
        req: [PairUnit(group, b, True) for b in ids] for req, ids in blocks.items()
    }
    return metadata


def test_skipped_hook_fails_before_the_forward():
    """A5: a step whose units were not exchanged stops at bind, in pre_forward,
    before a forward could read its blocks in P's row order. Run first, the
    hook exchanges every request's units in metadata order."""
    log: list = []
    runner = _runner_connector(_facade(_pair_worker(log)))
    step = _step(_pair_step(a=[3, 4], b=[7]))
    with pytest.raises(RuntimeError, match="did not run before the forward"):
        runner.pre_forward(step)
    assert log == []

    runner.finalize_loaded_kv(step)
    runner.pre_forward(step)
    assert log == [("exchange", [PairUnit(2, b, True) for b in (3, 4, 7)])]
    # Every step's units need their own run.
    with pytest.raises(RuntimeError, match="did not run before the forward"):
        runner.pre_forward(_step(_pair_step(c=[9])))


def test_multi_connector_forwards_finalize_loaded_kv():
    """A5: MultiConnector hands each child its own metadata, so the NIXL
    child's exchange runs and its bind guard passes."""
    log: list = []
    other = MagicMock(spec=KVConnectorBase_V1)
    multi = object.__new__(MultiConnector)
    multi._connectors = [other, _facade(_pair_worker(log))]
    multi._extra_async_saves = {}
    other_metadata, nixl_metadata = object(), _pair_step(a=[5])
    step = _step(MultiKVConnectorMetadata(metadata=(other_metadata, nixl_metadata)))
    runner = _runner_connector(multi)

    runner.finalize_loaded_kv(step)
    runner.pre_forward(step)
    other.finalize_loaded_kv.assert_called_once_with(other_metadata)
    assert log == [("exchange", [PairUnit(2, 5, True)])]


def test_without_units_finalize_is_a_noop():
    """A5: with pair mode off, or no pull to finish, the hook changes nothing:
    no exchange, no worker state, the metadata untouched."""
    worker = _pair_worker()
    runner = _runner_connector(_facade(worker))
    metadata = _recv_metadata("r", ([1],), (0,), ([2],), 16)
    metadata.reqs_to_send["s"] = 1.0
    before = copy.deepcopy(vars(metadata))

    runner.finalize_loaded_kv(_step(metadata))
    runner.pre_forward(_step(metadata))
    worker._drain_pair_events(metadata)
    assert vars(metadata) == before
    assert vars(worker) == {}

    # Disabled (dummy runs), no metadata, no connector, the base class.
    kv_connector = MagicMock(spec=KVConnectorBase_V1)
    _runner_connector(kv_connector, disabled=True).finalize_loaded_kv(_step(metadata))
    _runner_connector(kv_connector).finalize_loaded_kv(_step(None))
    kv_connector.finalize_loaded_kv.assert_not_called()
    NO_OP_KV_CONNECTOR.finalize_loaded_kv(_step(metadata))
    KVConnectorBase_V1.finalize_loaded_kv(kv_connector, metadata)


def test_units_without_an_exchanger_fail_closed():
    """Units reaching a worker that built no exchanger are a P/D config skew
    every rank sees alike: all raise, none recomputes or hangs."""
    with pytest.raises(RuntimeError, match="built no exchanger"):
        _facade(_pair_worker()).finalize_loaded_kv(_pair_step(a=[1]))


@pytest.mark.parametrize(
    "other_group,blocks",
    [(PAIR_GROUP + 1, [5]), (PAIR_GROUP, [4])],
    ids=["unit-of-another-group", "block-twice"],
)
def test_units_the_exchanger_cannot_own_fail_closed(other_group, blocks):
    """The exchanger rewrites blocks by id alone: a unit of another group (a
    KDA block row, say) or a block exchanged twice would corrupt blocks it does
    not own. Every rank sees the same units, so all raise before any NCCL
    round, and the step never reaches its forward."""
    log: list = []
    runner = _runner_connector(_facade(_pair_worker(log)))
    metadata = _pair_step(a=[3, 4])
    metadata.reqs_to_pair_exchange.update(
        _pair_step(other_group, b=blocks).reqs_to_pair_exchange
    )
    with pytest.raises(RuntimeError, match=f"blocks of transfer group {PAIR_GROUP}"):
        runner.finalize_loaded_kv(_step(metadata))
    assert log == []
    with pytest.raises(RuntimeError, match="did not run before the forward"):
        runner.pre_forward(_step(metadata))


def test_start_load_kv_raises_a_peer_mismatch_before_reads():
    """A5: a P/D interleave or mode mismatch a handshake thread stored stops
    the decoder on the main thread at its next start_load_kv, before any READ
    or exchange wait, instead of a later recompute."""
    worker = _pair_worker([])
    worker._peer_config_mismatch = base_worker.NixlPeerConfigMismatch("skew")
    metadata = _recv_metadata("r", ([1],), (0,), ([2],), 16)
    log: list = []
    worker._pair_events = (
        (_pair_step(a=[1]), _FakeEvent(log, "earlier"), frozenset({1})),
    )
    # A stub without a NIXL agent: posting a READ would raise AttributeError.
    with pytest.raises(base_worker.NixlPeerConfigMismatch, match="skew"):
        worker.start_load_kv(metadata)
    assert log == []


def test_execute_model_finalizes_loaded_kv_first():
    """A5: the V2 runner runs the hook at the top of a real step, before
    update_requests zeroes new blocks and applies copy-on-write copies (which
    may read the exchanged blocks), also in a step without tokens, and not in
    dummy runs."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    class _Stop(Exception):
        pass

    runner = MagicMock()
    runner.update_requests.side_effect = _Stop
    runner.gather_batch_req_state.side_effect = _Stop
    step = SimpleNamespace(num_scheduled_tokens={"r": 1}, total_num_scheduled_tokens=1)
    with pytest.raises(_Stop):
        GPUModelRunner.execute_model(runner, step)
    names = [name for name, _, _ in runner.mock_calls]
    assert names.index("kv_connector.finalize_loaded_kv") < names.index(
        "update_requests"
    )
    runner.kv_connector.finalize_loaded_kv.assert_called_once_with(step)

    # A step without tokens (e.g. every request still loading) returns early.
    runner.reset_mock()
    runner.update_requests.side_effect = None
    runner.encoder_cache = runner.aux_output_connector = None
    empty = SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    GPUModelRunner.execute_model(runner, empty)
    names = [name for name, _, _ in runner.mock_calls]
    assert names.index("kv_connector.finalize_loaded_kv") < names.index(
        "kv_connector.no_forward"
    )
    runner.kv_connector.finalize_loaded_kv.assert_called_once_with(empty)

    runner.reset_mock()
    with pytest.raises(_Stop):
        GPUModelRunner.execute_model(runner, step, dummy_run=True)
    runner.kv_connector.finalize_loaded_kv.assert_not_called()


def test_reuse_guard_waits_for_earlier_exchanges(k31_env, fabric):
    """A5: an abort can free a request's blocks while its exchange is queued,
    and a later step's pull can READ into them: start_load_kv waits for the
    earlier exchange before its first READ, once. The step's own pulls do not
    wait: they cannot target blocks of requests alive when it was scheduled.
    A pull into other blocks does not wait either (DSTEP: a full wait drains
    the async-scheduling lookahead after every admission)."""
    small = dict(block_size=64, num_dsa=2, num_kda=3)
    prefillers = {p: _Rank(fabric, P_DCP, p, 8, **small) for p in (1, 3)}
    decoder = _Rank(
        fabric, D_DCP, 3, 8, extra_config={"dcp_pair_exchange": True}, **small
    )
    _handshake(decoder, prefillers)
    connector, log = _facade(decoder.worker), fabric.log
    dsa = decoder.dsa_group

    def step(req_id: str, block: int, units: list[PairUnit]) -> list[str]:
        """One step: the hook, then the pull of ``req_id`` into DSA block
        ``block``, posted after the forward. Returns the kinds of exchange,
        event and READ calls, in order."""
        log.clear()
        metadata = _recv_metadata(
            req_id,
            _per_group(decoder, [block], [[6], [7]]),
            _per_group(decoder, 0, [0, 0]),
            _per_group(decoder, [1], [[2], [3]]),
            64,
        )
        metadata.reqs_to_pair_exchange = {"a": units} if units else {}
        connector.finalize_loaded_kv(metadata)
        connector.bind_connector_metadata(metadata)
        decoder.worker.start_load_kv(metadata)
        return [entry[0] for entry in log]

    # Request a's exchange (block 5) runs in the step that posts b's pull.
    assert step("b", 4, [PairUnit(dsa, 5, False)]) == ["exchange", "read", "read"]
    # a is aborted and its block goes to c: c's READs wait for a's exchange.
    # e's pull targets block 6: a's exchange (block 5) is no hazard to it.
    assert step("e", 6, []) == ["read", "read"]
    assert step("c", 5, []) == ["synchronize", "read", "read"]
    assert step("d", 3, []) == ["read", "read"]


def test_pair_mode_startup_checks(k31_env, fabric):
    """Pair mode's register-time checks name the problem."""
    small = dict(block_size=64, num_dsa=2, num_kda=3)
    pair = {"dcp_pair_exchange": True}
    with pytest.raises(NixlPairModeError, match="indexer layer .* not bound"):
        _Rank(fabric, D_DCP, 0, 4, extra_config=pair, bind_indexer=None, **small)
    # Bound, but not as rows: its kernel page size cannot be read off it.
    with pytest.raises(NixlPairModeError, match="indexer layer .* not bound"):
        _Rank(
            fabric,
            D_DCP,
            0,
            4,
            extra_config=pair,
            bind_indexer=lambda view: _bind_indexer(view).flatten(-2),
            **small,
        )
    # Without KDA state, packed MLA rows register whole: no region is a page.
    with pytest.raises(NixlPairModeError, match="not a NIXL region of their own"):
        _Rank(fabric, D_DCP, 0, 4, extra_config=pair, **{**small, "num_kda": 0})
    with pytest.raises(ValueError, match="chunk_units"):
        _Rank(
            fabric,
            D_DCP,
            0,
            4,
            extra_config={**pair, "dcp_pair_exchange_chunk_units": 0},
            **small,
        )
    assert not fabric.builds
    # Pair mode off builds nothing.
    assert _Rank(fabric, D_DCP, 0, 4, **small).worker._pair_exchanger is None
    assert not fabric.builds
