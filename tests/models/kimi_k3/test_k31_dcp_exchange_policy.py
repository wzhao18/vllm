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
    def set_cap(major: int, minor: int, mx: bool = True):
        platform = SimpleNamespace(
            get_device_capability=lambda: SimpleNamespace(major=major, minor=minor),
            is_cuda=lambda: True,
        )
        monkeypatch.setattr(vllm.platforms, "current_platform", platform, raising=False)
        import vllm.utils.deep_gemm as deep_gemm

        monkeypatch.setattr(deep_gemm, "mqa_logits_use_mx_scales", lambda: mx)

    return set_cap


@pytest.mark.parametrize(
    "tp,dcp,cap,mx,expected",
    [
        (4, 4, (10, 7), True, True),  # on by policy (no switch): C512's TP4 = DCP4
        (8, 8, (10, 7), True, True),
        (16, 16, (10, 7), True, False),  # DCP16 (MERGE_DCP): the #63 merge
        (4, 4, (10, 0), True, False),  # measured on SM107 only
        (4, 4, (10, 7), False, False),  # no bf16 MX-scaled logits
        (16, 8, (10, 7), True, False),  # DCP < TP: the stock path
        (8, 4, (10, 7), True, False),
        (16, 1, (10, 7), True, False),  # no DCP
        (16, 16, (10, 0), True, False),  # measured on SM107 only
        (16, 16, (10, 7), False, False),  # no bf16 MX-scaled logits
    ],
)
def test_wanted(sm, tp, dcp, cap, mx, expected):
    sm(*cap, mx=mx)
    assert decode.wanted(_config(tp, dcp)) is expected


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        ("kv_producer", False),  # a prefill-only instance never decodes
        ("kv_consumer", True),
        ("kv_both", True),
    ],
)
def test_wanted_kv_role(sm, kv_role, expected):
    sm(10, 7)
    assert decode.wanted(_config(4, 4, kv_role)) is expected


def test_stock_tie_rule():
    """Ties at the K-th key go by ascending global id, the stock merge's rule (no
    other tie order is selectable); the DCP sizes that keep the #63 merge."""
    import inspect

    from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import DcpExchange

    assert "tie_order" not in inspect.signature(DcpExchange.topk_merge).parameters
    assert decode.ONE_ROUND_ROWS == 256
    # DCP16 (C1-C256): the #63 merge at every row (faster at rows 8-512, measured)
    assert decode.MERGE_DCP == {16}


def _launcher(resident: dict[int, int], fit_wave: bool = True):
    """A ``DcpxDecode`` stand-in for ``launch_params``: 212 SMs (VR200), the kernel's
    co-resident clusters per cluster size from ``resident``."""
    ns = SimpleNamespace(
        ex=SimpleNamespace(resident_clusters=lambda c, segmax: resident[c]),
        sms=212,
        smem=228 * 1024,
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
    # 8 rows of 8 CTAs stage their 32768-key segments; longer steps re-read from L2
    assert segmax == (32768 if rows == 8 else 0)


def test_topk_cluster_rule_without_wave_fit():
    """``FIT_WAVE`` off (benchmarks): the former cluster-by-batch rule, which ran
    192 rows in two turns of 4-CTA clusters."""
    launch = _launcher({8: 88, 4: 176, 2: 400, 1: 800}, fit_wave=False)
    assert launch(192, LOCAL_1M_DCP4)[0] == 4


def test_topk_cluster_short_model_can_reach_one():
    """Below 128K local keys the cluster may drop to one CTA per row."""
    launch = _launcher({8: 88, 4: 176, 2: 300, 1: 800})
    assert launch(384, 100_000)[0] == 1


# GR100 (212 SMs, 228 KiB shared memory per SM): co-resident top-k clusters per
# (CTAs per row, staged keys), from topk_resident_clusters.
GR100 = {
    (8, 0): 92, (8, 8192): 68, (8, 16384): 46, (8, 32768): 46,
    (4, 0): 184, (4, 8192): 138, (4, 16384): 92, (4, 32768): 92,
    (2, 0): 424, (2, 8192): 318, (2, 16384): 212, (2, 32768): 212,
    (1, 0): 848, (1, 8192): 636, (1, 16384): 424, (1, 32768): 424,
}  # fmt: skip


@pytest.mark.parametrize(
    "rows,expected",
    [
        (64, (8, 8192)),  # 64 rows fit the 68 co-resident 8-CTA clusters
        (72, (4, 16384)),  # the rule's 8-CTA clusters would take two turns
        (88, (4, 16384)),
        (96, (2, 32768)),  # past the 92 co-resident 4-CTA clusters
        (128, (2, 32768)),
        (256, (2, 0)),  # the rule's clusters already fit one wave
        (512, (1, 0)),
    ],
)
def test_topk_cluster_dcp16_one_wave(rows, expected):
    """DCP16 at max_model_len 1M (65536 local keys, staged up to 200 rows): the rule
    ran 72-128 rows in two turns (W16, 39-42 us per layer against 23-26 us at the
    clusters that fit one wave)."""
    ns = SimpleNamespace(
        ex=SimpleNamespace(resident_clusters=lambda c, segmax: GR100[c, segmax]),
        sms=212,
        smem=228 * 1024,
        fit_wave=True,
    )
    ns._segmax = lambda r, c, ml: decode.DcpxDecode._segmax(ns, r, c, ml)
    assert decode.DcpxDecode.launch_params(ns, rows, 1 << 16) == expected
