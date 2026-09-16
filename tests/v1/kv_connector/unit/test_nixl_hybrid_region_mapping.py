# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec


def _hybrid_worker() -> NixlPushConnectorWorker:
    worker = object.__new__(NixlPushConnectorWorker)
    worker.dcp_size = 1
    worker.pp_size = 1
    worker.use_mla = True
    worker._has_mamba = True
    worker._physical_blocks_per_logical_kv_block = 1
    worker.use_host_buffer = False
    worker.kv_cache_layout = "NHD"
    worker.host_buffer_kv_cache_layout = "NHD"
    worker.backend_name = "FLASHMLA"
    worker._is_hma_required = True
    worker._group_spec_types = (MLAAttentionSpec, MambaSpec)
    worker.block_len_per_layer = [16, 16]
    worker.block_stride_per_layer = [16, 16]
    worker.region_num_blocks = [4, 4]
    worker.region_group_ids = [0, 0]
    worker.region_names = ["main.0", "main.1"]
    worker._region_is_mla = [True, True]
    worker.device_id = 0
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.get_engine_info.return_value = EngineTransferInfo(
        remote_tp_size=1,
        remote_block_size=4,
        remote_block_len=16,
        remote_physical_blocks_per_logical=1,
        remote_dcp_size=1,
    )
    worker.transfer_topo.tp_ratio.return_value = 1
    worker.transfer_topo.block_size_ratio.return_value = 1
    worker.transfer_topo.is_kv_replicated.return_value = False
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=False)
    worker.dst_num_blocks = {"remote": 4}
    worker._validate_asymmetric_dcp = lambda *_: False
    return worker


def _remote_metadata() -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id="remote",
        agent_metadata=b"remote",
        kv_caches_base_addr=[3000, 2000, 1000],
        device_id=0,
        num_blocks=4,
        block_lens=[16, 16, 16],
        block_strides=[16, 16, 16],
        kv_cache_layout="NHD",
        block_size=4,
        ssm_sizes=(8, 8),
        attn_backend_name="FLASHMLA",
        physical_blocks_per_logical_kv_block=1,
        region_num_blocks=[4, 4, 4],
        region_group_ids=[0, 0, 0],
        region_names=["draft.0", "main.1", "main.0"],
        region_mem_types=["VRAM", "VRAM", "VRAM"],
    )


@pytest.mark.cpu_test
def test_hybrid_handshake_rejects_unaligned_physical_region_counts():
    """Model the observed 24-region producer and 29-region consumer."""
    worker = _hybrid_worker()

    with pytest.raises(
        AssertionError,
        match="Hybrid MLA kernel-granularity block lengths must match",
    ):
        worker._validate_remote_agent_handshake(_remote_metadata(), 1)


@pytest.mark.cpu_test
def test_push_region_alignment_removes_decode_only_draft_regions():
    worker = _hybrid_worker()
    metadata = _remote_metadata()
    plan = SimpleNamespace(
        source_ranks_per_group=([0], [0]), rank_offset_factor=0
    )

    local_fa = worker._build_fa_local([100, 200], block_size_ratio=1)
    assert len(local_fa) == 8

    worker._align_push_remote_regions(metadata)

    assert metadata.kv_caches_base_addr == [1000, 2000]
    assert metadata.block_lens == [16, 16]
    assert metadata.block_strides == [16, 16]
    assert metadata.region_num_blocks == [4, 4]
    assert metadata.region_group_ids == [0, 0]
    assert metadata.region_names == ["main.0", "main.1"]
    assert metadata.region_mem_types == ["VRAM", "VRAM"]
    worker._validate_remote_agent_handshake(metadata, 1)

    remote_fa = worker._build_fa_remote(plan, metadata, block_size_ratio=1)
    assert len(remote_fa) == len(local_fa)
    assert remote_fa[:, 0].tolist() == [
        1000,
        1016,
        1032,
        1048,
        2000,
        2016,
        2032,
        2048,
    ]
