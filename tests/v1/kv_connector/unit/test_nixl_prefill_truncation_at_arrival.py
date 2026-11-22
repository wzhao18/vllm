# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The P-side prefill truncation (drop the trailing backoff tokens the decoder
recomputes) must happen at request arrival, before a later connector keys its
arrival work (the Mooncake store's arrival lookup) on the prompt. Truncating
only at ``on_new_request`` made every Dynamo-fed P request's arrival lookup one
token stale, so it was redone at the first schedule."""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)


class _FakeRequest:
    def __init__(self, num_prompt_tokens: int, params: dict | None):
        self.prompt_token_ids = list(range(num_prompt_tokens))
        self._all_token_ids = list(self.prompt_token_ids)
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_embeds = None
        self.kv_transfer_params = params
        self.skip_reading_external_kv = False
        self.max_tokens = 16

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)


def _nixl_scheduler(has_mamba: bool = True) -> NixlBaseConnectorScheduler:
    sched = object.__new__(NixlBaseConnectorScheduler)
    sched._has_mamba = has_mamba
    sched._bounded_replay = False
    sched.vllm_config = SimpleNamespace(num_prefill_lookahead_tokens=0)
    return sched


def _nixl_connector(sched: NixlBaseConnectorScheduler) -> NixlConnector:
    conn = object.__new__(NixlConnector)
    conn.connector_scheduler = sched
    return conn


class _RecordingConnector:
    """Stands in for the Mooncake store: records the prompt length its
    arrival lookup would key on."""

    def __init__(self):
        self.arrival_num_tokens: list[int] = []

    def on_request_arrival(self, request) -> None:
        self.arrival_num_tokens.append(request.num_tokens)


def test_p_request_truncated_at_arrival_once():
    sched = _nixl_scheduler()
    req = _FakeRequest(1000, {"do_remote_decode": True})
    sched.on_request_arrival(req)
    assert req.num_tokens == 999
    assert req.num_prompt_tokens == 999
    assert req.max_tokens == 1
    # The scheduler's on_new_request must not truncate again.
    sched._truncate_request_for_prefill(req)
    assert req.num_tokens == 999


def test_non_p_requests_untouched_at_arrival():
    sched = _nixl_scheduler()
    for params in (None, {}, {"do_remote_prefill": True}):
        req = _FakeRequest(1000, params)
        sched.on_request_arrival(req)
        assert req.num_tokens == 1000


def test_no_backoff_without_mamba():
    sched = _nixl_scheduler(has_mamba=False)
    req = _FakeRequest(1000, {"do_remote_decode": True})
    sched.on_request_arrival(req)
    assert req.num_tokens == 1000


def test_multiconnector_arrival_lookup_sees_truncated_prompt():
    """NIXL before the store (production order): the store's arrival work
    keys on the same prompt length the scheduler later schedules."""
    store = _RecordingConnector()
    multi = object.__new__(MultiConnector)
    multi._connectors = [_nixl_connector(_nixl_scheduler()), store]
    req = _FakeRequest(35260, {"do_remote_decode": True})
    multi.on_request_arrival(req)
    assert store.arrival_num_tokens == [35259]
    assert req.num_tokens == 35259
