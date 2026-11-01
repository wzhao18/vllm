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


def _launcher(resident: dict[int, int], fit_wave: bool = True):
    """A ``DcpxDecode`` stand-in for ``launch_params``: 212 SMs (VR200), the kernel's
    co-resident clusters per cluster size from ``resident``."""
    ns = SimpleNamespace(
        ex=SimpleNamespace(resident_clusters=lambda c, segmax: resident[c]),
        sms=212,
        smem=228 * 1024,
        segmax_override=-1,
        fit_wave=fit_wave,
    )
    ns._segmax = lambda rows, c, ml: decode.DcpxDecode._segmax(ns, rows, c, ml)
    return lambda rows, ml: decode.DcpxDecode.launch_params(ns, rows, ml)


# TP4 / DCP4 at max_model_len 1M: 262144 local keys (the WRITE bitmap floor is 2 CTAs)
LOCAL_1M_DCP4 = 1 << 18


@pytest.mark.parametrize(
    "rows,expected",
    [
        (8, 8),  # small batches keep the cluster-by-batch rule
        (64, 8),
        (160, 4),  # 160 rows fit the 176 co-resident 4-CTA clusters
        (176, 4),
        (184, 2),  # beyond them: one wave of 2-CTA clusters, not a second turn
        (192, 2),
        (256, 2),
        (448, 2),  # beyond every wave: the WRITE floor (persistent turns)
    ],
)
def test_topk_cluster_fits_one_wave(rows, expected):
    launch = _launcher({8: 88, 4: 176, 2: 400, 1: 800})
    cluster, segmax = launch(rows, LOCAL_1M_DCP4)
    assert cluster == expected
    assert segmax == 0  # 1M-token rows never fit shared memory


def test_topk_cluster_rule_without_wave_fit():
    """``FIT_WAVE`` off (benchmarks): the former cluster-by-batch rule, which ran
    192 rows in two turns of 4-CTA clusters."""
    launch = _launcher({8: 88, 4: 176, 2: 400, 1: 800}, fit_wave=False)
    assert launch(192, LOCAL_1M_DCP4)[0] == 4


def test_topk_cluster_short_model_can_reach_one():
    """Below 128K local keys the cluster may drop to one CTA per row."""
    launch = _launcher({8: 88, 4: 176, 2: 300, 1: 800})
    assert launch(384, 100_000)[0] == 1
