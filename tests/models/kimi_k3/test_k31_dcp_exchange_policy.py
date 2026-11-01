# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the K3.1 DcpExchange decode is built (``dcp_exchange.decode.wanted``)."""

from types import SimpleNamespace

import pytest

import vllm.platforms
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode


def _config(tp: int, dcp: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=dcp
        )
    )


@pytest.fixture
def sm(monkeypatch):
    def set_cap(major: int, minor: int):
        platform = SimpleNamespace(
            get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
        )
        monkeypatch.setattr(vllm.platforms, "current_platform", platform, raising=False)

    return set_cap


@pytest.mark.parametrize(
    "tp,dcp,cap,mode,expected",
    [
        (16, 16, (10, 7), None, True),  # production recipe: auto on
        (16, 8, (10, 7), None, False),  # DCP < TP: auto keeps the stock path
        (8, 4, (10, 7), None, False),
        (16, 1, (10, 7), None, False),  # no DCP
        (16, 16, (10, 0), None, False),  # auto is SM107 only
        (16, 16, (10, 7), "0", False),
        (16, 16, (10, 0), "1", True),  # forced
        (16, 8, (10, 7), "1", True),  # forced: DcpxDecode rejects the layout
    ],
)
def test_wanted(monkeypatch, sm, tp, dcp, cap, mode, expected):
    sm(*cap)
    if mode is None:
        monkeypatch.delenv(decode.ENV, raising=False)
    else:
        monkeypatch.setenv(decode.ENV, mode)
    assert decode.wanted(_config(tp, dcp)) is expected
