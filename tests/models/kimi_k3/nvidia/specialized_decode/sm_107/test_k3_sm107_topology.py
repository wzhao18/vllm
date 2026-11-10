# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode (SM107) on CPU: the topology descriptor
(``runtime.topology``). A table of deployments checks the mode (TP_EXPERTS /
EP_GROUP / DEP), the residual layout and the refusal reason; then the gate's
pass-through (``unsupported`` / ``specialized_decode_config``) and the
``sp_topology`` compatibility alias. No GPU, no CuTe DSL."""

import dataclasses
import re
from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.topology import (
    Layout,
    Mode,
    Topology,
    refusal,
    rubin_megamoe_world_sizes,
)

DG = "deep_gemm_mega_moe"
KK = "kimi_k3_mega_moe"
TP_X = Mode.TP_EXPERTS
EPG = Mode.EP_GROUP
DEP = Mode.DEP
REP = Layout.REPLICATED
SHD = Layout.ROW_SHARDED
LOC = Layout.LOCAL


def fake(
    tp, dcp=1, dp=1, ep=False, backend=DG, pp=1, spec=3, query_lens=None, mnbt=None
):
    """The vllm_config fields ``Topology.from_config`` reads (``mnbt``:
    max_num_batched_tokens, None: unset, so ``moe_tokens`` stays 0)."""
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        ),
        kernel_config=SimpleNamespace(moe_backend=backend),
        speculative_config=(
            None if spec is None else SimpleNamespace(num_speculative_tokens=spec)
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=128),
    )
    if mnbt is not None:
        config.scheduler_config.max_num_batched_tokens = mnbt
    config.get_decode_cudagraph_query_lens = lambda q: (
        query_lens if query_lens is not None else {q: 128}
    )
    return config


# (id, config kwargs, dsa, mode, layout, refusal substring or None)
TABLE = [
    # -- TP_EXPERTS: no expert parallelism (TP8 / TP16, DCP 1 or TP) -------------
    ("tp8", dict(tp=8), True, TP_X, REP, None),
    ("tp8-dcp8", dict(tp=8, dcp=8), True, TP_X, REP, None),
    (
        "tp8-dcp8-dp2",
        dict(tp=8, dcp=8, dp=2),
        True,
        TP_X,
        REP,
        "data parallelism without expert parallelism (DP2 x TP8",
    ),
    ("tp8-dp4", dict(tp=8, dp=4), True, TP_X, REP, "(DP4 x TP8: stock shards"),
    ("tp16", dict(tp=16), True, TP_X, REP, None),
    ("tp16-dcp16", dict(tp=16, dcp=16), True, TP_X, REP, None),
    (
        "tp16-dcp16-dp2",
        dict(tp=16, dcp=16, dp=2),
        True,
        TP_X,
        REP,
        "data parallelism without expert parallelism (DP2 x TP16",
    ),
    ("tp8-q1", dict(tp=8, dcp=8, spec=None), True, TP_X, REP, None),
    ("tp16-q8-kk", dict(tp=16, dcp=16, spec=7, backend=KK), True, TP_X, REP, None),
    ("tp8-k3-mla", dict(tp=8, dcp=8), False, TP_X, REP, None),
    ("tp16-k3-mla", dict(tp=16), False, TP_X, REP, None),
    (
        "tp16-dcp8",
        dict(tp=16, dcp=8),
        True,
        TP_X,
        REP,
        "decode context parallel other than 1 or TP (DCP8 at TP16",
    ),
    (
        "tp8-dcp4",
        dict(tp=8, dcp=4),
        True,
        TP_X,
        REP,
        "decode context parallel other than 1 or TP (DCP4 at TP8",
    ),
    (
        "tp4-dcp4-noep",
        dict(tp=4, dcp=4, dp=4),
        True,
        TP_X,
        REP,
        "TP other than 8/16 without expert parallelism (TP4",
    ),
    ("tp4-noep", dict(tp=4), True, TP_X, REP, "without expert parallelism (TP4"),
    ("tp2-noep", dict(tp=2), True, TP_X, REP, "without expert parallelism (TP2"),
    ("tp1-dp8-noep", dict(tp=1, dp=8), True, TP_X, LOC, "parallelism (TP1"),
    ("tp32-noep", dict(tp=32), True, TP_X, REP, "without expert parallelism (TP32"),
    ("tp8-pp2", dict(tp=8, pp=2), True, TP_X, REP, "pipeline parallelism"),
    # -- EP_GROUP: expert parallelism over TP > 1 groups -------------------------
    # C512's decode role (the SP step): served exactly as before.
    ("c512", dict(tp=4, dcp=4, dp=4, ep=True), True, EPG, SHD, None),
    ("c512-kk", dict(tp=4, dcp=4, dp=4, ep=True, backend=KK), True, EPG, SHD, None),
    ("tp4dcp4-dp2", dict(tp=4, dcp=4, dp=2, ep=True), True, EPG, SHD, None),
    (
        "tp4dcp4-dp2-kk",
        dict(tp=4, dcp=4, dp=2, ep=True, backend=KK),
        True,
        EPG,
        SHD,
        None,
    ),
    ("tp4dcp4-dp8", dict(tp=4, dcp=4, dp=8, ep=True), True, EPG, SHD, None),
    ("c512-q1", dict(tp=4, dcp=4, dp=4, ep=True, spec=None), True, EPG, SHD, None),
    (
        "tp4dcp4-dp8-kk",
        dict(tp=4, dcp=4, dp=8, ep=True, backend=KK),
        True,
        EPG,
        SHD,
        "kimi_k3_mega_moe at EP32: no Rubin MegaMoE tuning record",
    ),
    # The replicated step (runtime.group_ep): every other TP4 / TP8 / TP16 group.
    ("tp4-dcp1-ep", dict(tp=4, dcp=1, dp=4, ep=True), True, EPG, SHD, None),
    (
        "tp4-dcp2-ep",
        dict(tp=4, dcp=2, dp=4, ep=True),
        True,
        EPG,
        SHD,
        "decode context parallel other than 1 or TP (DCP2 at TP4",
    ),
    ("tp8dcp8-dp2-ep", dict(tp=8, dcp=8, dp=2, ep=True), True, EPG, SHD, None),
    ("tp8-dp4-ep32", dict(tp=8, dcp=1, dp=4, ep=True), True, EPG, SHD, None),
    ("tp8dcp8-dp4-ep32", dict(tp=8, dcp=8, dp=4, ep=True), True, EPG, SHD, None),
    ("tp16dcp16-ep", dict(tp=16, dcp=16, ep=True), True, EPG, SHD, None),
    ("tp16-dp2-ep32", dict(tp=16, dcp=1, dp=2, ep=True), True, EPG, SHD, None),
    (
        "tp8dcp8-dp2-kk",
        dict(tp=8, dcp=8, dp=2, ep=True, backend=KK),
        True,
        EPG,
        SHD,
        None,
    ),
    (
        "tp8dcp8-dp4-kk",
        dict(tp=8, dcp=8, dp=4, ep=True, backend=KK),
        True,
        EPG,
        SHD,
        "kimi_k3_mega_moe at EP32: no Rubin MegaMoE tuning record",
    ),
    (
        "tp16-dcp8-ep",
        dict(tp=16, dcp=8, dp=2, ep=True),
        True,
        EPG,
        SHD,
        "decode context parallel other than 1 or TP (DCP8 at TP16",
    ),
    (
        "c512-deep-gemm",
        dict(tp=4, dcp=4, dp=4, ep=True, backend="deep_gemm"),
        True,
        EPG,
        SHD,  # stock SP: DP > 1
        "expert parallelism on MoE backend deep_gemm",
    ),
    (
        "tp8-dp1-ep-cutlass",
        dict(tp=8, dcp=8, ep=True, backend="flashinfer_cutlass"),
        True,
        EPG,
        REP,  # stock non-SP EP: DP1 without native MegaMoE
        "expert parallelism on MoE backend flashinfer_cutlass",
    ),
    (
        "c512-k3-mla",
        dict(tp=4, dcp=4, dp=4, ep=True),
        False,
        EPG,
        SHD,
        "expert parallelism on models without DSA layers",
    ),
    (
        "tp2-ep",
        dict(tp=2, dcp=2, dp=8, ep=True),
        True,
        EPG,
        SHD,
        "TP2 under expert parallelism",
    ),
    (
        "c512-pp2",
        dict(tp=4, dcp=4, dp=4, ep=True, pp=2),
        True,
        EPG,
        REP,  # stock SP needs PP1
        "pipeline parallelism",
    ),
    # -- DEP: expert parallelism over TP1 ----------------------------------------
    (
        "dep16",
        dict(tp=1, dp=16, ep=True),
        True,
        DEP,
        LOC,
        "mode DEP not implemented on sm_107 yet (TP1 x DP16, EP16)",
    ),
    (
        "dep32",
        dict(tp=1, dp=32, ep=True),
        True,
        DEP,
        LOC,
        "mode DEP not implemented on sm_107 yet (TP1 x DP32, EP32)",
    ),
    (
        "dep16-kk",
        dict(tp=1, dp=16, ep=True, backend=KK),
        True,
        DEP,
        LOC,
        "mode DEP not implemented on sm_107 yet (TP1 x DP16, EP16)",
    ),
    (
        "dep32-kk",
        dict(tp=1, dp=32, ep=True, backend=KK),
        True,
        DEP,
        LOC,
        "kimi_k3_mega_moe at EP32: no Rubin MegaMoE tuning record",
    ),
    (
        "dep16-deep-gemm",
        dict(tp=1, dp=16, ep=True, backend="deep_gemm"),
        True,
        DEP,
        LOC,
        "expert parallelism on MoE backend deep_gemm",
    ),
    (
        "dep16-k3-mla",
        dict(tp=1, dp=16, ep=True),
        False,
        DEP,
        LOC,
        "expert parallelism on models without DSA layers",
    ),
]


@pytest.mark.parametrize(
    "kwargs,dsa,mode,layout,reason",
    [row[1:] for row in TABLE],
    ids=[row[0] for row in TABLE],
)
def test_topology_table(kwargs, dsa, mode, layout, reason):
    topology = Topology.from_config(fake(**kwargs))
    assert topology.mode is mode, topology
    assert topology.layout is layout, topology
    got = refusal(topology, dsa=dsa, pp=kwargs.get("pp", 1))
    if reason is None:
        assert got is None, got
    else:
        assert got is not None and reason in got, got


def test_table_size_and_ids_unique():
    assert len(TABLE) >= 40
    assert len({row[0] for row in TABLE}) == len(TABLE)


def test_descriptor_fields():
    """C512: TP4 = DCP4 x DP4, EP16 over DeepGEMM MegaMoE, q4, stock SP; the SP step."""
    topology = Topology.from_config(fake(tp=4, dcp=4, dp=4, ep=True))
    assert topology == Topology(
        g=4, dcp=4, dp=4, ep=16, moe_backend=DG, q_set=(4,), stock_sp=True
    )
    assert topology.sp_step and topology.expert_parallel
    assert str(topology).startswith("ep_group TP4 DCP4 x DP4 EP16")
    assert topology.moe_tokens == 0  # (no max_num_batched_tokens in the fake)
    assert topology.step_layout is SHD and topology.layout_at(4) is SHD
    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.g = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs,ep,stock_sp",
    [
        (dict(tp=8, dcp=8, dp=2), 1, False),  # no EP: dp kept, ep 1
        (dict(tp=16, ep=True), 16, True),  # native MegaMoE at DP1
        (dict(tp=8, ep=True, backend="deep_gemm"), 8, False),  # non-SP EP
        (dict(tp=8, dp=2, ep=True, backend="deep_gemm"), 16, True),  # DP > 1
        (dict(tp=1, dp=16, ep=True), 16, False),  # DEP: no group to shard over
    ],
)
def test_ep_and_stock_sp(kwargs, ep, stock_sp):
    topology = Topology.from_config(fake(**kwargs))
    assert (topology.ep, topology.stock_sp) == (ep, stock_sp)
    assert topology.dp == kwargs.get("dp", 1)


def test_sp_step_only_tp4_dcp4():
    """The SP step is EP_GROUP at TP4 = DCP4 with the stock SP layout only."""
    steps = {
        row[0]
        for row in TABLE
        if Topology.from_config(fake(**row[1])).sp_step
    }
    assert steps == {
        "c512",
        "c512-kk",
        "tp4dcp4-dp2",
        "tp4dcp4-dp2-kk",
        "tp4dcp4-dp8",
        "tp4dcp4-dp8-kk",
        "c512-q1",
        "c512-k3-mla",
    }


def test_q_set():
    assert Topology.from_config(fake(tp=8, spec=None)).q_set == (1,)
    assert Topology.from_config(fake(tp=8, spec=7)).q_set == (8,)
    dynamic = fake(tp=8, spec=3, query_lens={4: 32, 2: 128})
    assert Topology.from_config(dynamic).q_set == (2, 4)


def test_rubin_megamoe_records():
    """The shipped Rubin MegaMoE records cover EP4 / EP8 / EP16 (C512 on
    kimi_k3_mega_moe); EP32 has none until W9 tunes it."""
    sizes = rubin_megamoe_world_sizes()
    assert {4, 8, 16} <= sizes
    assert 32 not in sizes


def test_step_layout_per_mode():
    """The specialized step's residual layout: C512's SP step keeps stock's row
    shard (c); every other EP_GROUP topology replicates (b); TP_EXPERTS replicated;
    DEP local. One layout per deployment (every captured R)."""
    expect = {
        "c512": SHD,
        "tp4dcp4-dp2": SHD,
        "tp4-dcp1-ep": REP,
        "tp8dcp8-dp2-ep": REP,
        "tp8-dp4-ep32": REP,
        "tp16dcp16-ep": REP,
        "tp16-dp2-ep32": REP,
        "tp8-dcp8": REP,
        "tp16-dcp16": REP,
        "dep16": LOC,
    }
    rows = {row[0]: row[1] for row in TABLE}
    for name, layout in expect.items():
        topology = Topology.from_config(fake(**rows[name]))
        assert topology.step_layout is layout, name
        assert {topology.layout_at(r) for r in (1, 4, 64)} == {layout}, name
    with pytest.raises(ValueError):
        Topology.from_config(fake(**rows["c512"])).layout_at(0)


@pytest.mark.parametrize(
    "kwargs,mnbt,tokens",
    [
        (dict(tp=8, dcp=8, dp=2, ep=True), 8192, 1024),  # stock SP: cdiv over TP8
        (dict(tp=16, dcp=16, ep=True), 8192, 512),
        (dict(tp=4, dcp=4, dp=4, ep=True), 2048, 512),
        (dict(tp=8, dcp=8, dp=2), 8192, 0),  # no EP: no MegaMoE
        (dict(tp=1, dp=16, ep=True), 8192, 8192),  # DEP: no SP shard
    ],
)
def test_moe_tokens(kwargs, mnbt, tokens):
    """The MegaMoE capacity per rank the stock KimiMoE builds (max_num_tokens)."""
    assert Topology.from_config(fake(**kwargs, mnbt=mnbt)).moe_tokens == tokens


@pytest.mark.parametrize(
    "kwargs,mnbt,reason",
    [
        # EP4 has only the 2048-token inference records: a genphase capacity misses.
        (
            dict(tp=4, dcp=1, ep=True, backend=KK),
            512,
            "kimi_k3_mega_moe at EP4: no Rubin MegaMoE tuning record for (world "
            "size 4, capacity 128 tokens per rank: mxfp4_mxfp8 genphase)",
        ),
        (dict(tp=4, dcp=1, ep=True, backend=KK), 8192, None),  # 2048: exact
        (dict(tp=8, dcp=8, dp=2, ep=True, backend=KK), 4096, None),  # EP16 256
        (dict(tp=8, dcp=8, ep=True, backend=KK), 2400, None),  # EP8 300: buckets
        (dict(tp=8, dcp=8, dp=2, ep=True), 512, None),  # DeepGEMM: no records read
    ],
)
def test_rubin_megamoe_capacity(kwargs, mnbt, reason):
    """``kimi_k3_mega_moe`` resolves its knobs by (world size, dtype, kernel variant)
    of its capacity, then the token buckets: a miss is refused at load naming the
    world size and capacity (instead of the KnobLookupError at model build)."""
    topology = Topology.from_config(fake(**kwargs, mnbt=mnbt))
    got = refusal(topology, dsa=True)
    if reason is None:
        assert got is None, got
    else:
        assert got is not None and got.startswith(reason), got


# -- the gate's pass-through (runtime.unsupported / specialized_decode_config) ----
@pytest.fixture
def gate(monkeypatch):
    """A Rubin (SM107) device, the deployment's KDA dtypes, stock trtllm-gen and the
    env defaults the gate reads (as test_k3_sm107_sp_gating)."""
    from vllm import envs
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107 import runtime
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cc: tuple(cc) == (10, 7)
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 100
    )
    monkeypatch.setattr(
        runtime, "kda_dtypes", lambda config: (torch.bfloat16, torch.float32)
    )
    monkeypatch.setattr(flashinfer_mla_sparse, "use_prims_ts", lambda c, d: False)
    for name in ("VLLM_ENABLE_GEMM_RS", "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM"):
        monkeypatch.setattr(envs, name, False, raising=False)
    return runtime


def deployment(**kwargs):
    from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_sp_gating import (  # noqa: E501
        deployment as c512_deployment,
    )

    return c512_deployment(**kwargs)


@pytest.mark.parametrize(
    "kwargs,overrides,prefix",
    [
        (dict(tp=1, dcp=1, dp=16), {}, "mode DEP not implemented on sm_107 yet"),
        (dict(tp=8, dcp=8, dp=2), {}, "ReplaySSM / RecoverSSM (--use-replayssm) out"),
        (
            dict(tp=8, dcp=8, dp=2),
            {"cache_config.use_replayssm": False},
            "expert-parallel attention groups (EP_GROUP, replicated step) on more "
            "than 64 verify rows per attention group",
        ),
        (
            dict(tp=1, dcp=1, dp=32),
            {"kernel_config.moe_backend": KK},
            "kimi_k3_mega_moe at EP32",
        ),
        (dict(tp=16, dcp=8, dp=1, ep=False), {}, "decode context parallel other"),
    ],
)
def test_gate_raises_with_the_topology_reason(gate, kwargs, overrides, prefix):
    config = deployment(**kwargs, **overrides)
    reason = gate.unsupported(config)
    assert reason is not None and reason.startswith(prefix), reason
    with pytest.raises(
        ValueError,
        match=re.escape(f"Kimi K3 specialized decode does not support {prefix}"),
    ):
        gate.specialized_decode_config(config)


def test_gate_keeps_c512(gate):
    """C512 (TP4 = DCP4 x DP4, EP16 MegaMoE) still passes; its config carries the
    topology, DP and EP, and the SP step."""
    config = deployment()
    assert gate.unsupported(config) is None
    assert gate.sp_topology(config)
    decode = gate.specialized_decode_config(config)
    assert decode.mode is Mode.EP_GROUP and decode.sequence_parallel
    assert (decode.dp, decode.ep, decode.topology.g) == (4, 16, 4)


# The replicated EP_GROUP step's deployment: no ReplaySSM (W7), 16 requests x q4 = 64
# verify rows per group.
GROUP = {"cache_config.use_replayssm": False, "scheduler_config.max_num_seqs": 16}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tp=8, dcp=8, dp=2),  # DP2 x TP8 = DCP8, EP16
        dict(tp=8, dcp=1, dp=4),  # DP4 x TP8, EP32 (DeepGEMM MegaMoE)
        dict(tp=16, dcp=16, dp=1),  # TP16 = DCP16, EP16
        dict(tp=16, dcp=1, dp=2),  # DP2 x TP16, EP32
        dict(tp=4, dcp=1, dp=4),  # DP4 x TP4 (DCP1), EP16
    ],
)
def test_gate_serves_the_replicated_step(gate, kwargs):
    """EP_GROUP outside C512's SP role passes as the replicated step: its config
    carries the EP_GROUP mode without sequence parallelism, the replicated step
    layout, DP / EP and every captured row (4..64)."""
    config = deployment(**kwargs, **GROUP)
    assert gate.unsupported(config) is None
    assert not gate.sp_topology(config) and gate.ep_group_topology(config)
    decode = gate.specialized_decode_config(config)
    assert decode.mode is Mode.EP_GROUP and not decode.sequence_parallel
    assert decode.topology.step_layout is REP and decode.topology.layout is SHD
    assert (decode.dp, decode.ep) == (kwargs["dp"], kwargs["tp"] * kwargs["dp"])
    assert decode.captured == tuple(range(4, 65, 4)) and not decode.replayssm


@pytest.mark.parametrize(
    "overrides,cause",
    [
        ({"cache_config.cache_dtype": "fp8_ds_mla"}, "KV cache dtypes other than per"),
        ({"attention_config.backend": "FLASHMLA_SPARSE"}, "FLASHINFER_MLA_SPARSE"),
        ({"kernel_config.flashinfer_sparse_mla_kernel": "prims_ts"}, "prims_ts"),
        ({"speculative_config": None}, "no speculation (q1"),
        ({"scheduler_config.max_num_seqs": 32}, "max_num_seqs x q = 128; the wide"),
        ({"speculative_config.num_speculative_tokens": 7}, "max_num_seqs x q = 128"),
        ({"cache_config.use_replayssm": True}, "the EP_GROUP replicated step runs no"),
    ],
)
def test_gate_refuses_the_replicated_step(gate, overrides, cause):
    """Each narrowing of the replicated step raises at load with its reason."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    settings = {**GROUP, **overrides}
    if settings.get("attention_config.backend") == "FLASHMLA_SPARSE":
        settings["attention_config.backend"] = AttentionBackendEnum.FLASHMLA_SPARSE
    no_spec = "speculative_config" in settings
    settings.pop("speculative_config", None)
    config = deployment(tp=8, dcp=8, dp=2, **settings)
    if no_spec:  # (after the decode sizes resolved)
        config.speculative_config = None
    reason = gate.unsupported(config)
    assert reason is not None and cause in reason, reason
    with pytest.raises(ValueError, match="Kimi K3 specialized decode does not"):
        gate.specialized_decode_config(config)


def test_gate_refuses_replicated_step_states_and_env(gate, monkeypatch):
    """A BF16 KDA recurrent state (W6), the fused GEMM-RS o_proj and a non-Rubin
    GPU are refused for the replicated step too."""
    from vllm import envs
    from vllm.platforms import current_platform

    config = deployment(tp=8, dcp=8, dp=2, **GROUP)
    with monkeypatch.context() as patch:
        patch.setattr(
            gate, "kda_dtypes", lambda config: (torch.bfloat16, torch.bfloat16)
        )
        assert "a BF16 recurrent state: W6" in gate.unsupported(config)
    with monkeypatch.context() as patch:
        patch.setattr(envs, "VLLM_ENABLE_GEMM_RS", True, raising=False)
        assert "VLLM_ENABLE_GEMM_RS" in gate.unsupported(config)
    with monkeypatch.context() as patch:
        patch.setattr(current_platform, "is_device_capability", lambda cc: False)
        assert "Rubin SM107 only" in gate.unsupported(config)


def test_sp_topology_alias_matches_sp_step(gate):
    for row in TABLE:
        config = fake(**row[1])
        assert gate.sp_topology(config) == Topology.from_config(config).sp_step, row[0]


def test_decode_config_mode_without_topology():
    """Hand-built DecodeConfigs (tests) follow ``sequence_parallel``."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (
        DecodeConfig,
    )

    base = dict(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=16, q=8,
        capture_sizes=(8,),
    )  # fmt: skip
    assert DecodeConfig(**base).mode is Mode.TP_EXPERTS
    assert DecodeConfig(**base, sequence_parallel=True).mode is Mode.EP_GROUP
