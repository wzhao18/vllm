# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

import vllm.plugins as plugins_module
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


class DummyConnectorMetadata(KVConnectorMetadata):
    def __init__(
        self,
        block_hashes_by_req: dict[str, list[BlockHash]],
        block_ids_by_req: dict[str, tuple[list[int], ...]],
    ):
        self.block_hashes_by_req = block_hashes_by_req
        self.block_ids_by_req = block_ids_by_req


class DummyKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return (0, False)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        block_hashes_by_req = getattr(scheduler_output, "block_hashes_by_req", None)
        assert block_hashes_by_req is not None, (
            "DummyKVConnector expected 'block_hashes_by_req' on scheduler_output"
        )
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        block_ids_by_req = block_state.block_ids
        return DummyConnectorMetadata(
            block_hashes_by_req=block_hashes_by_req,
            block_ids_by_req=block_ids_by_req,
        )

    def start_load_kv(self, kv_caches, finished_req_ids):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


def _my_plugin():
    """Registers the dummy KV connector and overrides _build_kv_connector_meta"""
    KVConnectorFactory.register_connector(
        "DummyKVConnector",
        __name__,
        DummyKVConnector.__name__,
    )

    def _custom_build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        block_hashes_by_req: dict[str, list[BlockHash]] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            block_hashes_by_req[req_id] = request.block_hashes

        scheduler_output.block_hashes_by_req = block_hashes_by_req  # type: ignore[attr-defined]
        return connector.build_connector_meta(scheduler_output)

    Scheduler._build_kv_connector_meta = _custom_build_kv_connector_meta


@pytest.fixture
def _load_plugin():
    """Load the fake plugin through the real load_general_plugins() path."""
    ep = MagicMock()
    ep.name = "dummy_kv_connector_plugin"
    ep.value = f"{__name__}:_my_plugin"
    ep.load.return_value = _my_plugin

    # Reset the global guard so load_general_plugins() actually runs.
    plugins_module.plugins_loaded = False
    with patch("importlib.metadata.entry_points", return_value=[ep]):
        plugins_module.load_general_plugins()
        yield
    # Reset again so other tests are not affected.
    plugins_module.plugins_loaded = False
    KVConnectorFactory._registry.pop("DummyKVConnector", None)


def test_connector_receives_block_hashes(_load_plugin):
    block_size = 16
    num_tokens = 48  # 3 full blocks worth of tokens
    scheduler = create_scheduler(
        use_kv_connector="DummyKVConnector", block_size=block_size
    )
    requests = create_requests(
        num_requests=3, num_tokens=num_tokens, block_size=block_size
    )
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # Verify the connector metadata was built with block hashes.
    meta = output.kv_connector_metadata
    assert isinstance(meta, DummyConnectorMetadata)
    assert len(meta.block_hashes_by_req) == 3

    for req in requests:
        assert req.request_id in meta.block_hashes_by_req
        # Each request has num_tokens / block_size = 3 full block hashes.
        assert len(meta.block_hashes_by_req[req.request_id]) == (
            num_tokens // block_size
        )
        assert meta.block_hashes_by_req[req.request_id] == req.block_hashes

    expected_block_ids = {
        req.req_id: req.block_ids for req in output.scheduled_new_reqs
    }
    assert meta.block_ids_by_req == expected_block_ids
    assert output.kv_connector_block_state is None


def test_connector_receives_cached_block_ids_at_block_boundary(_load_plugin):
    block_size = 16
    scheduler = create_scheduler(
        use_kv_connector="DummyKVConnector",
        block_size=block_size,
        max_num_seqs=1,
        max_num_batched_tokens=block_size - 1,
        max_model_len=2 * block_size,
    )
    (request,) = create_requests(num_requests=1, num_tokens=block_size)
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    second_output = scheduler.schedule()

    assert second_output.scheduled_cached_reqs.new_block_ids == [None]
    meta = second_output.kv_connector_metadata
    assert isinstance(meta, DummyConnectorMetadata)
    assert meta.block_ids_by_req[
        request.request_id
    ] == scheduler.kv_cache_manager.get_block_ids(request.request_id)
