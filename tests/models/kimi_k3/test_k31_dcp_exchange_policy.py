# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the K3.1 DcpExchange decode is built (``dcp_exchange.decode.wanted``)."""

from types import SimpleNamespace

import pytest

import vllm.platforms
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode


def _config(tp: int, dcp: int, kv_role: str | None = None):
    kv_transfer = None
    if kv_role is not None:
        kv_transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=dcp
        ),
        kv_transfer_config=kv_transfer,
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
        (16, 16, (10, 7), None, False),  # off unless set
        (16, 16, (10, 7), "auto", True),
        (16, 8, (10, 7), "auto", False),  # DCP < TP: auto keeps the stock path
        (8, 4, (10, 7), "auto", False),
        (16, 1, (10, 7), "auto", False),  # no DCP
        (16, 16, (10, 0), "auto", False),  # auto is SM107 only
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


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        ("kv_producer", False),  # a prefill-only instance never decodes
        ("kv_consumer", True),
        ("kv_both", True),
    ],
)
def test_wanted_kv_role(monkeypatch, sm, kv_role, expected):
    sm(10, 7)
    monkeypatch.setenv(decode.ENV, "1")
    assert decode.wanted(_config(16, 16, kv_role)) is expected


def test_parts_semicolons():
    """Harness env lists are comma-separated, so the parts may use semicolons."""
    parts = {"topk": (1, 1 << 30), "o": (97, 4096)}
    assert decode._parse_parts("topk;o=97-4096") == parts
    assert decode._parse_parts("topk,o=97-4096") == parts
