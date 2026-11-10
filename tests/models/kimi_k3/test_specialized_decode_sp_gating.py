# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode on CPU: the sequence-parallel (SP) gating of C512's
decode role (TP4 = DCP4 x DP4, EP16 MegaMoE, DSpark q4, per-tensor FP8 KV on
trtllm-gen, FP32 KDA state, ReplaySSM), its coverage contract (``shapes``), and the
verify builds' declared-stock / raise behaviour under SP. TP8 / TP16 keep their
gating."""

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.kimi_k3.nvidia.specialized_decode import runtime
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
    DecodeConfig,
    sp_topology,
    specialized_decode_config,
    unsupported,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (
    DECLARED_STOCK,
    Choice,
    Context,
    Coverage,
    Path,
    Shape,
    Uncovered,
    Unsupported,
    shard_rows,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Every C512 decode graph: 4..512 rows (max-num-seqs 128 x q4).
C512_ROWS = tuple(range(4, 513, 4))


def deployment(tp=4, dcp=4, dp=4, ep=True, **overrides):
    """A vllm_config stand-in for C512's decode role; ``overrides`` replace
    ``section.field`` values (e.g. ``{"cache_config.cache_dtype": "fp8_ds_mla"}``)."""
    text = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        num_experts=896,
        num_experts_per_token=16,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        dsa_config=SimpleNamespace(),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            data_parallel_size=dp,
            pipeline_parallel_size=1,
            enable_expert_parallel=ep,
            use_ubatching=False,
        ),
        model_config=SimpleNamespace(
            hf_text_config=text,
            quantization_config=None,
            enable_sleep_mode=False,
            max_model_len=1 << 20,
            dtype=torch.bfloat16,
            use_fp64_gumbel=False,
        ),
        lora_config=None,
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=list(C512_ROWS),
            max_cudagraph_capture_size=512,
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=3,
            enable_adaptive_verification=False,
            rejection_sample_method="synthetic",
        ),
        cache_config=SimpleNamespace(
            use_replayssm=True,
            use_kda_recoverssm=False,
            cache_dtype="fp8",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
        kernel_config=SimpleNamespace(
            moe_backend="deep_gemm_mega_moe", flashinfer_sparse_mla_kernel="auto"
        ),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=128),
        additional_config={"decode_backend": "latency_specialized"},
    )
    for path, value in overrides.items():
        section, name = path.split(".")
        setattr(getattr(config, section), name, value)
    return config


def tp16(**overrides):
    """K3.1's TP16 = DCP16 decode role (no EP, no ReplaySSM, DSpark q8)."""
    config = deployment(tp=16, dcp=16, dp=1, ep=False, **overrides)
    config.cache_config.use_replayssm = overrides.get(
        "cache_config.use_replayssm", False
    )
    config.speculative_config.num_speculative_tokens = 7
    config.scheduler_config.max_num_seqs = 32
    return config


@pytest.fixture(autouse=True)
def rubin(monkeypatch):
    """A Rubin (SM107) device, the deployment's KDA dtypes, stock trtllm-gen and the
    env defaults the gating reads."""
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cc: tuple(cc) == (10, 7)
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 100
    )
    monkeypatch.setattr(
        runtime, "kda_dtypes", lambda config: (torch.bfloat16, torch.float32)
    )
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse

    monkeypatch.setattr(flashinfer_mla_sparse, "use_prims_ts", lambda c, d: False)
    for name in ("VLLM_ENABLE_GEMM_RS", "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM"):
        monkeypatch.setattr(envs, name, False, raising=False)


def test_c512_is_the_sp_topology():
    """C512's decode role is the SP topology and passes the gate; its decode config
    carries SP, DP4, EP16 and the ReplaySSM records."""
    config = deployment()
    assert sp_topology(config)
    assert unsupported(config) is None
    decode = specialized_decode_config(config)
    assert (decode.sequence_parallel, decode.dp, decode.ep, decode.replayssm) == (
        True,
        4,
        16,
        True,
    )
    assert (decode.q, decode.dcp, decode.captured) == (4, 4, C512_ROWS)


def test_c512_on_kimi_k3_mega_moe_is_the_sp_topology():
    """The Rubin MegaMoE backend (a native MegaMoE) keeps C512's decode role SP."""
    config = deployment(**{"kernel_config.moe_backend": "kimi_k3_mega_moe"})
    assert sp_topology(config)
    assert unsupported(config) is None


@pytest.mark.parametrize(
    "tp,dcp,ep,backend",
    [
        (4, 4, False, "deep_gemm_mega_moe"),  # TP4 without EP: no SP
        (4, 4, True, "deep_gemm"),  # SP needs MegaMoE
        (4, 2, True, "deep_gemm_mega_moe"),  # DCP != TP
        (4, 1, True, "deep_gemm_mega_moe"),
        (8, 8, True, "deep_gemm_mega_moe"),  # other TP sizes keep EP refused
        (16, 16, True, "deep_gemm_mega_moe"),
    ],
)
def test_other_topologies_are_not_sp(tp, dcp, ep, backend):
    """Only TP4 = DCP4 over expert-parallel MegaMoE is SP; everything else keeps the
    TP8 / TP16 gate, which refuses TP4, EP and ReplaySSM as before."""
    backend = {"kernel_config.moe_backend": backend}
    config = deployment(tp=tp, dcp=dcp, ep=ep, **backend)
    assert not sp_topology(config)
    assert unsupported(config) is not None


def test_moe_token_shard_is_not_sp(monkeypatch):
    """The MoE token shard replaces SP in the stock decoder: not the SP topology."""
    from vllm.models.kimi_k3.nvidia import model

    monkeypatch.setattr(model, "use_moe_token_shard", lambda config: True)
    assert not sp_topology(deployment())
    assert "TP other than 8/16" in unsupported(deployment())


def test_tp16_gate_unchanged():
    """TP16 = DCP16 (K3.1's decode role) passes as before, with the SP fields at their
    defaults; ReplaySSM and EP stay refused there."""
    assert unsupported(tp16()) is None
    decode = specialized_decode_config(tp16())
    assert (decode.sequence_parallel, decode.dp, decode.ep, decode.replayssm) == (
        False,
        1,
        1,
        False,
    )
    assert decode == DecodeConfig(
        max_context=decode.max_context,
        mla_core="auto",
        dsa_core="auto",
        dcp=16,
        q=8,
        capture_sizes=decode.capture_sizes,
        captured=decode.captured,
    )
    assert "ReplaySSM" in unsupported(tp16(**{"cache_config.use_replayssm": True}))
    config = tp16()
    config.parallel_config.enable_expert_parallel = True
    assert unsupported(config).startswith("expert parallelism")


@pytest.mark.parametrize(
    "overrides,cause",
    [
        ({"cache_config.cache_dtype": "fp8_ds_mla"}, "per-tensor fp8"),
        ({"cache_config.cache_dtype": "auto"}, "per-tensor fp8"),
        (
            {"attention_config.backend": AttentionBackendEnum.FLASHMLA_SPARSE},
            "FLASHINFER_MLA_SPARSE",
        ),
        ({"kernel_config.flashinfer_sparse_mla_kernel": "prims_ts"}, "prims_ts"),
        ({"speculative_config.num_speculative_tokens": 7}, "DSpark with 3"),
        ({"speculative_config.method": "eagle3"}, "DSpark"),
        ({"scheduler_config.max_num_seqs": 256}, "512 verify rows"),
    ],
)
def test_sp_refuses_other_than_c512(overrides, cause):
    """Under SP every setting off C512's decode role raises with its own reason."""
    reason = unsupported(deployment(**overrides))
    assert reason is not None and cause in reason, reason


def test_sp_refuses_without_dsa_or_speculation():
    config = deployment()
    config.model_config.hf_text_config.dsa_config = None
    assert "without DSA layers" in unsupported(config)
    config = deployment()
    config.speculative_config = None
    assert "DSpark" in unsupported(config)


def test_sp_refuses_auto_prims_ts(monkeypatch):
    """The "auto" kernel resolving to prims_ts (under DCP where it runs) is refused:
    the SP DSA core runs trtllm-gen, so stock must too."""
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse

    monkeypatch.setattr(flashinfer_mla_sparse, "use_prims_ts", lambda c, d: True)
    assert "prims_ts" in unsupported(deployment())


def test_sp_refuses_gemm_rs(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_GEMM_RS", True)
    assert "VLLM_ENABLE_GEMM_RS" in unsupported(deployment())


def test_sp_refuses_off_rubin(monkeypatch):
    monkeypatch.setattr(current_platform, "is_device_capability", lambda cc: False)
    assert "Rubin SM107" in unsupported(deployment())


def test_sp_refuses_bf16_kda_state(monkeypatch):
    monkeypatch.setattr(
        runtime, "kda_dtypes", lambda config: (torch.bfloat16, torch.bfloat16)
    )
    assert "FP32 recurrent" in unsupported(deployment())


def test_sp_raises_through_the_config():
    """The engine sees the SP reason as the specialized decode's ValueError."""
    with pytest.raises(ValueError, match="sequence parallelism on"):
        specialized_decode_config(
            deployment(**{"cache_config.cache_dtype": "fp8_ds_mla"})
        )


def context(**fields):
    base = dict(
        arch="rubin",
        tp=4,
        rank=0,
        dcp=4,
        sms=148,
        q=4,
        max_rows=512,
        max_context=1 << 20,
        bf16_state=False,
    )
    return Context(**{**base, **fields})


def test_context_defaults_keep_tp16():
    """The SP fields default off, so TP8 / TP16 contexts are unchanged."""
    ctx = context(tp=16, dcp=16)
    assert (ctx.dp, ctx.ep, ctx.sequence_parallel, ctx.heads_per_rank) == (
        1,
        1,
        False,
        0,
    )


def test_shard_rows():
    ctx = context(dp=4, ep=16, sequence_parallel=True, heads_per_rank=24)
    assert [shard_rows(Shape(m, 4), ctx) for m in (4, 128, 512)] == [1, 32, 128]
    with pytest.raises(Unsupported, match=r"sp_shard at M6\(q2\)"):
        shard_rows(Shape(6, 2), ctx)


def test_unsupported_names_family_shape_cause():
    error = Unsupported("dsa", Shape(324, 4), "no query geometry")
    assert str(error) == "Kimi K3 SP decode: dsa at M324(q4): no query geometry"
    assert isinstance(error, ValueError)
    assert "verify at every shape" in str(Unsupported("verify", None, "x"))


def covered(rows=C512_ROWS):
    coverage = Coverage(4)
    for m in rows:
        coverage.specialize(m, {"dsa": Choice(Path.GENERIC, note="sp")})
        for family in sorted(DECLARED_STOCK):
            coverage.declare(m, family, "stock")
    return coverage


def test_coverage_complete():
    coverage = covered()
    coverage.assert_complete(C512_ROWS)
    assert "declared(stock)" in coverage.lines()[0]


@pytest.mark.parametrize(
    "spoil,expect",
    [
        (lambda c: c.plan.pop(512), "M512(q4): not planned"),
        (lambda c: c.stock(256, "no box"), "M256(q4): stock (no box)"),
        (
            lambda c: c.plan[8].update(kda=Choice(Path.STOCK, note="seam")),
            "M8(q4): kda=stock(seam)",
        ),
        (lambda c: c.declare(4, "dsa", "stock"), "dsa=declared(stock) is not"),
        (lambda c: c.plan.update({100: {}}), "M100(q4): no family planned"),
    ],
)
def test_coverage_names_every_gap(spoil, expect):
    """A missing or stock-planned shape, a family on the stock seam, or a declared
    family outside DECLARED_STOCK fails the build with the shape named."""
    coverage = covered()
    spoil(coverage)
    with pytest.raises(RuntimeError, match="not fully specialized") as error:
        coverage.assert_complete((*C512_ROWS, 100))
    assert expect in str(error.value)


def test_declare_keeps_a_stock_shape():
    coverage = Coverage(4)
    coverage.stock(4, "plan None")
    coverage.declare(4, "embed", "stock")
    assert coverage.plan[4] is None


def tp_group(monkeypatch, world_size):
    """Rank 0 of a ``world_size`` TP group (``vllm.distributed.get_tp_group``)."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=world_size, rank_in_group=0, device_group=None
        ),
    )


@pytest.fixture
def lm_head_rank(monkeypatch):
    """A Rubin TP4 rank without a sharded LM head (rank 0) behind a plain logits
    processor (LogitsProcessor's defaults: no scale, soft cap or head dtype)."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 7))
    tp_group(monkeypatch, 4)
    processor = SimpleNamespace(
        scale=1.0, soft_cap=None, logits_as_input=False, head_dtype=None
    )
    return SimpleNamespace(lm_head=SimpleNamespace(), logits_processor=processor)


def test_verify_declared_stock_under_synthetic(lm_head_rank):
    """C512 rejects synthetically: the stock sampler, declared; nothing is built."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.verify import (
        VerifyArgmax,
    )

    assert VerifyArgmax.build(deployment(), lm_head_rank) is None


def test_verify_raises_when_not_covered_under_sp(lm_head_rank, monkeypatch):
    """Real rejection under SP: an uncovered build raises ``Unsupported``. TP16 = DCP16
    with speculation (the NQ-NOFALLBACK target) refuses it (``Uncovered``); other
    topologies keep the stock sampler (None)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.verify import (
        VerifyArgmax,
    )

    config = deployment(**{"speculative_config.rejection_sample_method": "standard"})
    with pytest.raises(
        Unsupported, match="verify at every shape: an LM head without vocabulary"
    ):
        VerifyArgmax.build(config, lm_head_rank)
    config = deployment(
        tp=8, dcp=8, dp=1, **{"speculative_config.rejection_sample_method": "standard"}
    )
    tp_group(monkeypatch, 8)
    assert VerifyArgmax.build(config, lm_head_rank) is None
    config = tp16(**{"speculative_config.rejection_sample_method": "standard"})
    tp_group(monkeypatch, 16)
    with pytest.raises(Uncovered, match="an LM head without vocabulary shards"):
        VerifyArgmax.build(config, lm_head_rank)


def test_verify_raises_off_rubin_under_sp(lm_head_rank, monkeypatch):
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.verify import (
        VerifyArgmax,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    config = deployment(**{"speculative_config.rejection_sample_method": "standard"})
    with pytest.raises(Unsupported, match="Rubin SM107"):
        VerifyArgmax.build(config, lm_head_rank)
    assert VerifyArgmax.build(tp16(), lm_head_rank) is None


def test_sampled_verify_fp64_gumbel_raises_under_sp(monkeypatch):
    """FP64 Gumbel draws: under SP ``Unsupported`` (before the TP group is read); at
    TP16 = DCP16 with speculation refused (``Uncovered``); elsewhere the stock
    sampler (None)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sampled_verify import (
        SampledVerify,
    )

    config = deployment(**{"model_config.use_fp64_gumbel": True})
    with pytest.raises(Unsupported, match="fp64 Gumbel"):
        SampledVerify.build(config, None)
    config = deployment(tp=8, dcp=8, dp=1, **{"model_config.use_fp64_gumbel": True})
    tp_group(monkeypatch, 8)
    assert SampledVerify.build(config, None) is None
    config = tp16(**{"model_config.use_fp64_gumbel": True})
    tp_group(monkeypatch, 16)
    with pytest.raises(Uncovered, match="FP64 Gumbel"):
        SampledVerify.build(config, None)
