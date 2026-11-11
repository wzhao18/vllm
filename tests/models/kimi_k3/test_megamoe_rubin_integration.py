# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_backend="kimi_k3_mega_moe" wiring for Kimi K3, CPU only:

- the backend value: accepted, normalized, a native MegaMoE backend, Kimi K3 only;
- the automatic format policy (MXFP8 x MXFP4 up to 512 tokens/rank, else NVFP4) and
  the kernel variant / shipped knob record of the server's bucket;
- ``KimiK3MegaMoEExperts`` registers DeepGEMM MegaMoE's parameter names, shapes and
  dtypes at the K3.1 shapes, loads checkpoint shards as stored, converts to the
  kernel layout (raising on non-representable NVFP4 weights on every rank), exposes
  EPLB views, masks padding and passes the server's bucket;
- the Kimi K3 model picks the experts class per backend; the DeepGEMM header
  install skips kimi_k3_mega_moe;
- the S1 token_comm drop is recorded in VENDOR.md.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import (
    MEGA_MOE_BACKENDS,
    NATIVE_MEGA_MOE_BACKENDS,
    KIMI_K3_MEGA_MOE,
    KernelConfig,
    validate_kimi_k3_mega_moe_model,
)
from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as dg
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X
from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W

# Kimi K3.1 routed MoE: 896 experts, latent hidden 3584, intermediate 3072, top-16.
K31 = dict(num_experts=896, hidden=3584, intermediate=3072, top_k=16)
SITU = dg.MegaMoEActivation("situ", alpha=4.0, beta=25.0)
PKG_DIR = Path(X.__file__).resolve().parent


def _vllm_config(mnbt=8192):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.fixture
def ep(monkeypatch):
    """A fake EP group (world, rank) and an SM107 device for the experts module."""
    state = SimpleNamespace(world_size=16, rank_in_group=0, cpu_group=None)
    state.device_group = state
    monkeypatch.setattr(X, "get_ep_group", lambda: state)
    monkeypatch.setattr(X, "check_kimi_k3_mega_moe_supported", lambda: None)
    X.select_rubin_bucket.cache_clear()
    return state


@pytest.fixture
def small_bucket(monkeypatch):
    """The shipped knobs have no record for the tiny CPU problem: stand in the EP8
    2048-token record of the problem's precision (the data path is under test)."""

    def bucket(problem, world_size, tokens):
        knobs = C.resolve_knobs(
            C.SHIPPED_KNOBS, problem=C.RubinMoeProblem(quant_kind=problem.quant_kind),
            variant="inference", world_size=8, max_tokens=2048,
        )[0]  # fmt: skip
        return C.RubinBucket(X.rubin_kernel_variant(tokens), tokens, knobs), ()

    monkeypatch.setattr(X, "select_rubin_bucket", bucket)


def _experts(vllm_config, *, num_experts, hidden, intermediate, top_k, ep, **kw):
    e_local = num_experts // ep.world_size
    return X.KimiK3MegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=e_local,
        experts_start_idx=ep.rank_in_group * e_local,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=SITU,
        weight_format=dg.MEGA_MOE_MXFP4,
        prefix=kw.pop("prefix", "model.layers.3.mlp.experts"),
        **kw,
    )


# -- backend value ---------------------------------------------------------------------


def test_backend_value_is_a_native_mega_moe_backend():
    assert KernelConfig(moe_backend="kimi-k3-mega-moe").moe_backend == KIMI_K3_MEGA_MOE
    assert KIMI_K3_MEGA_MOE in NATIVE_MEGA_MOE_BACKENDS
    assert KIMI_K3_MEGA_MOE in MEGA_MOE_BACKENDS  # LoRA refuses it like the others
    with pytest.raises(ValueError):
        KernelConfig(moe_backend="kimi_k3_megamoe")


@pytest.mark.parametrize(
    "arch",
    [
        "KimiK3ForConditionalGeneration",
        "KimiK31ForConditionalGeneration",
        "KimiLinearForCausalLM",
    ],
)
def test_backend_accepts_kimi_k3(arch):
    validate_kimi_k3_mega_moe_model(KIMI_K3_MEGA_MOE, [arch])


def test_backend_rejects_other_models():
    with pytest.raises(ValueError, match="only supported for Kimi K3"):
        validate_kimi_k3_mega_moe_model(KIMI_K3_MEGA_MOE, ["DeepseekV4ForCausalLM"])
    validate_kimi_k3_mega_moe_model("deep_gemm_mega_moe", ["DeepseekV4ForCausalLM"])


def test_backend_gates_kimi_native_mega_paths():
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=KIMI_K3_MEGA_MOE),
        parallel_config=SimpleNamespace(all2all_backend="naive"),
    )
    assert kimi_model.can_shard_sequence_parallel_shared_expert(cfg)


# -- format policy ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "kind", "variant"),
    [
        (1, "mxfp4_mxfp8", "genphase"),
        (160, "mxfp4_mxfp8", "genphase"),  # C512 decode: 640 MNBT / TP4 SP
        (256, "mxfp4_mxfp8", "genphase"),
        (512, "mxfp4_mxfp8", "genphase"),
        (513, "nvfp4", "genphase"),
        (1024, "nvfp4", "genphase"),
        (1025, "nvfp4", "inference"),
        (2048, "nvfp4", "inference"),  # TP8 prefill token shard: 16384 / 8
        (16384, "nvfp4", "inference"),
    ],
)
def test_format_policy(tokens, kind, variant):
    assert X.RUBIN_MX_MAX_TOKENS_PER_RANK == 512
    assert X.rubin_quant_kind(tokens) == kind
    assert X.rubin_kernel_variant(tokens) == variant


def test_format_policy_rejects_empty_server():
    with pytest.raises(ValueError):
        X.rubin_quant_kind(0)
    with pytest.raises(ValueError):
        X.rubin_kernel_variant(0)


@pytest.mark.parametrize(
    ("world", "tokens"), [(16, 160), (16, 512), (16, 4096), (8, 2048), (8, 768)]
)
def test_server_bucket_resolves_shipped_knobs(world, tokens):
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(tokens))
    bucket, _ = X.select_rubin_bucket(problem, world, tokens)
    assert bucket.max_tokens == tokens
    assert bucket.variant == X.rubin_kernel_variant(tokens)
    # every admission rule of the kernel config holds for the resolved knobs
    C.RubinMoeConfig.make(problem, bucket, rank=0, world_size=world)


@pytest.mark.parametrize(
    ("world", "tokens"), [(16, 160), (16, 512), (16, 2048), (8, 2048)]
)
def test_shared_overlap_sm_cap_admits_every_server_bucket(world, tokens):
    """reserve_sms_for_shared_expert's bucket (grid capped, fallback clusters
    dropped) passes the kernel admission rules for every shipped server record,
    including the mixed-CGA EP16 2048 one, and reaches the kernel config."""
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(tokens))
    bucket, _ = X.select_rubin_bucket(problem, world, tokens)
    module = SimpleNamespace(bucket=bucket, kernel_weights=None, prefix="layer")
    X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)
    capped = module.bucket
    assert capped.max_sm_count == X.RUBIN_SHARED_OVERLAP_MAX_SMS
    assert capped.knobs.fallback_cluster_shape_mn is None
    assert capped.variant == bucket.variant
    assert capped.max_tokens == bucket.max_tokens
    cfg = C.RubinMoeConfig.make(problem, capped, rank=0, world_size=world)
    assert cfg.max_sm_count == X.RUBIN_SHARED_OVERLAP_MAX_SMS
    module.kernel_weights = object()
    with pytest.raises(RuntimeError):
        X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)


def test_server_bucket_without_knob_record_raises():
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind="mxfp4_mxfp8")
    with pytest.raises(C.KnobLookupError):
        X.select_rubin_bucket(problem, 4, 256)  # no EP4 genphase record


# -- experts module --------------------------------------------------------------------


def _params(module):
    return {
        name: (tuple(p.shape), p.dtype)
        for name, p in module.named_parameters(recurse=False)
    }


@pytest.mark.parametrize("mnbt", [256, 8192])
def test_parameters_match_deep_gemm_mega_moe(monkeypatch, ep, mnbt):
    """Same names, shapes and dtypes as DeepGemmMegaMoEExperts at the K3.1 shapes (EP16,
    56 local experts), so the Kimi K3 expert mapping loads either."""
    monkeypatch.setattr(dg, "check_deep_gemm_mega_moe_supported", lambda *a: None)
    monkeypatch.setattr(dg.current_platform, "device_type", "meta", raising=False)
    ep.rank_in_group = 5
    args = dict(
        num_experts=896,
        num_local_experts=56,
        experts_start_idx=5 * 56,
        top_k=16,
        hidden_size=3584,
        intermediate_size=3072,
        activation=SITU,
        weight_format=dg.MEGA_MOE_MXFP4,
        max_num_tokens=mnbt,
    )
    with torch.device("meta"):
        ref = dg.DeepGemmMegaMoEExperts(_vllm_config(), prefix="a.experts", **args)
        ours = X.KimiK3MegaMoEExperts(_vllm_config(), prefix="b.experts", **args)
    assert _params(ours) == _params(ref)
    assert _params(ours)["w13_weight"] == ((56, 6144, 1792), torch.int8)
    for name, param in ours.named_parameters(recurse=False):
        assert param.weight_loader == ours.weight_loader
        assert getattr(param, "quant_method", None) == (
            "block" if name.endswith("weight_scale") else None
        )
    assert ours.top_k == 16 and ours.max_num_tokens == mnbt
    assert ours.quant_kind == ("mxfp4_mxfp8" if mnbt <= 512 else "nvfp4")
    assert ours.has_fused_shared_experts is False
    assert X.KimiK3MegaMoEExperts.weight_loader.supports_moe_loading


def test_constructor_refuses_what_it_cannot_run(ep):
    cfg = dict(num_experts=896, hidden=3584, intermediate=3072, top_k=16, ep=ep)
    with torch.device("meta"):
        with pytest.raises(NotImplementedError, match="shared expert"):
            _experts(_vllm_config(), shared_experts=torch.nn.Identity(), **cfg)
        with pytest.raises(NotImplementedError, match="redundant"):
            _experts(_vllm_config(), num_logical_experts=448, **cfg)
        with pytest.raises(ValueError, match="Duplicate layer name"):
            vc = _vllm_config()
            _experts(vc, **cfg)
            _experts(vc, **cfg)


def test_off_sm107_raises(monkeypatch):
    monkeypatch.setattr(X.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(X.current_platform, "is_device_capability", lambda cc: False)
    with pytest.raises(NotImplementedError, match="SM107"):
        X.check_kimi_k3_mega_moe_supported()


# Small problem for the CPU data-path tests: 8 experts over EP2, H = I = 128.
SMALL = dict(num_experts=8, hidden=128, intermediate=128, top_k=2)


def _leg(gen, rows, cols, kmax=-3, spread=6):
    codes = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
    k = torch.randint(kmax - spread, kmax + 1, (rows, cols // 32), generator=gen)
    return codes, (k + 127).to(torch.uint8)


def _load_small(module, gen):
    """Load every expert's w1/w3/w2 codes + scales through the Kimi K3 MegaMoE
    expert mapping names; returns the per-expert shards."""
    h, i = SMALL["hidden"], SMALL["intermediate"]
    shards = {}
    for expert in range(SMALL["num_experts"]):
        for shard, (rows, cols) in (("w1", (i, h)), ("w3", (i, h)), ("w2", (h, i))):
            codes, scales = _leg(gen, rows, cols)
            shards[expert, shard] = (codes, scales)
            prefix = "w2" if shard == "w2" else "w13"
            for suffix, tensor in (("weight", codes), ("weight_scale", scales)):
                name = f"model.layers.3.mlp.experts.{prefix}_{suffix}"
                param = getattr(module, f"{prefix}_{suffix}")
                ok = param.weight_loader(
                    param, tensor, name, shard_id=shard, expert_id=expert,
                    return_success=True,
                )  # fmt: skip
                local = module.experts_start_idx <= expert < module.experts_end_idx
                assert ok is local
    return shards


def test_weight_loader_keeps_checkpoint_layout(ep, small_bucket):
    ep.world_size, ep.rank_in_group = 2, 1
    module = _experts(_vllm_config(), ep=ep, **SMALL)
    shards = _load_small(module, torch.Generator().manual_seed(0))
    w13, s13, w2, s2 = module.canonical_weights()
    for local in range(4):
        expert = 4 + local
        w1, w3, d = (shards[expert, s] for s in ("w1", "w3", "w2"))
        assert torch.equal(w13[local], torch.cat([w1[0], w3[0]]))
        assert torch.equal(s13[local], torch.cat([w1[1], w3[1]]))
        assert torch.equal(w2[local], d[0]) and torch.equal(s2[local], d[1])
    # a wrongly shaped shard is an error, not a partial copy
    param = module.w2_weight_scale
    with pytest.raises(ValueError, match="shard shape"):
        param.weight_loader(
            param, torch.zeros(128, 3, dtype=torch.uint8),
            "x.experts.w2_weight_scale", shard_id="w2", expert_id=4,
        )  # fmt: skip
    with pytest.raises(ValueError, match="MXFP4"):
        param.weight_loader(
            param, torch.zeros(()), "x.experts.w2_weight_scale_2", "w2", 4
        )


class _FakeRuntime:
    def __init__(self, kind):
        self.kind, self.calls = kind, []

    def prepare_weights(self, w13, s13, w2, s2, *, layer_id):
        return W.prepare_weights(
            self.kind, w13, s13, w2, s2, hidden=128, intermediate=128,
            layer_id=layer_id, expert_offset=4,
        )  # fmt: skip

    def forward(self, hidden, ids, weights, kernel_weights, *, bucket):
        self.calls.append((ids.clone(), weights, kernel_weights, bucket))
        return torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.bfloat16)


@pytest.mark.parametrize(("mnbt", "kind"), [(64, "mxfp4_mxfp8"), (4096, "nvfp4")])
def test_process_weights_converts_and_frees(monkeypatch, ep, small_bucket, mnbt, kind):
    ep.world_size, ep.rank_in_group = 2, 1
    module = _experts(_vllm_config(), ep=ep, max_num_tokens=mnbt, **SMALL)
    assert module.quant_kind == kind
    _load_small(module, torch.Generator().manual_seed(1))
    runtime = _FakeRuntime(kind)
    monkeypatch.setattr(module, "get_runtime", lambda: runtime)
    reduced = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, op, group: reduced.append(int(t))
    )
    module.process_weights_after_loading()
    assert reduced == [1]
    w = module.kernel_weights
    assert w.quant_kind == kind
    W.validate_kernel_weights(
        w, quant_kind=kind, experts=4, hidden=128, intermediate=128
    )
    assert all(p.numel() == 0 for p in module.parameters())
    views = module.get_expert_weights()
    assert len(views) == (7 if kind == "nvfp4" else 4)
    assert all(v.shape[0] == 4 and v.is_contiguous() for v in views)
    with pytest.raises(RuntimeError, match="reloading"):
        module.weight_loader(module.w2_weight, torch.zeros(1), "w2_weight", "w2", 4)

    # forward: padding rows masked to -1, the server's one bucket every call
    hidden = torch.randn(3, 128).to(torch.bfloat16)
    ids = torch.tensor([[4, 5], [6, 7], [0, 1]], dtype=torch.int32)
    wts = torch.rand(3, 2)
    pad = torch.tensor([False, True, False, True])
    out = module(hidden, wts, ids, is_padding=pad)
    assert out.shape == (3, 128)
    staged_ids, staged_wts, kernel_weights, bucket = runtime.calls[-1]
    assert staged_ids.tolist() == [[4, 5], [-1, -1], [0, 1]]
    assert staged_ids.dtype == torch.int32 and staged_wts is wts
    assert kernel_weights is w and bucket is module.bucket
    with pytest.raises(ValueError, match="sized for"):
        module(torch.zeros(mnbt + 1, 128, dtype=torch.bfloat16), wts, ids)


def test_non_representable_nvfp4_raises_on_every_rank(monkeypatch, ep, small_bucket):
    ep.world_size, ep.rank_in_group = 2, 1
    module = _experts(_vllm_config(), ep=ep, max_num_tokens=4096, **SMALL)
    _load_small(module, torch.Generator().manual_seed(2))
    w13s = module.w13_weight_scale.data.view(torch.uint8)
    w13s[0, 0, 0], w13s[0, 0, 1] = 127 + 10, 127 - 10  # spread 20 > 17
    monkeypatch.setattr(module, "get_runtime", lambda: _FakeRuntime("nvfp4"))
    reduced = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, op, group: reduced.append(int(t))
    )
    with pytest.raises(W.NonRepresentableError):
        module.process_weights_after_loading()
    assert reduced == [0] and module.kernel_weights is None


def test_forward_before_conversion_raises(ep):
    with torch.device("meta"):
        module = _experts(
            _vllm_config(), ep=ep, num_experts=896, hidden=3584, intermediate=3072,
            top_k=16,
        )  # fmt: skip
    with pytest.raises(RuntimeError, match="process_weights_after_loading"):
        module(torch.zeros(1, 3584), torch.zeros(1, 16), torch.zeros(1, 16))


# -- model wiring ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "cls"),
    [
        ("deep_gemm_mega_moe", dg.DeepGemmMegaMoEExperts),
        ("kimi_k3_mega_moe", X.KimiK3MegaMoEExperts),
    ],
)
def test_model_picks_experts_class(backend, cls):
    cfg = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend=backend))
    assert kimi_model.mega_moe_experts_cls(cfg) is cls


def test_moe_token_shard_accepts_rubin(monkeypatch):
    monkeypatch.setattr(kimi_model, "pd_prefill_worker", lambda cfg: True)
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=KIMI_K3_MEGA_MOE),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=8,
        ),
    )
    assert kimi_model.use_moe_token_shard(cfg)


def test_deep_gemm_header_install_skips_rubin(monkeypatch):
    from vllm.models.kimi_k3.nvidia import megamoe
    from vllm.utils import deep_gemm

    def boom():
        raise AssertionError("DeepGEMM imported for kimi_k3_mega_moe")

    monkeypatch.setattr(deep_gemm, "_import_deep_gemm", boom)
    cfg = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend=KIMI_K3_MEGA_MOE))
    assert megamoe.prepare_megamoe(torch.nn.Module(), cfg) is None


def test_prepare_rubin_megamoe_without_rubin_layers_is_a_noop():
    assert X.prepare_rubin_megamoe(torch.nn.Sequential(torch.nn.Linear(2, 2))) == 0


# -- vendored S1 -----------------------------------------------------------------------


def test_vendor_records_s1_token_comm():
    text = (PKG_DIR / "VENDOR.md").read_text()
    manifest = dict(
        (path, digest)
        for digest, path in re.findall(r"^([0-9a-f]{64})  (sources/\S+)$", text, re.M)
    )
    assert manifest["sources/communication/nvlink_domain/token_comm.py"] == (
        "725b855edb022d5e3a3ee7c077342012d3aa9eb9c0169e8c9753b3df2dde0b81"
    )
    local = text.split("## Local differences", 1)[1].split("\n## ", 1)[0]
    assert "3a47d4d53" in local and "wzhao/kimi-k3-megamoe-sm107" in local
    assert "4731d31ad1871b24e6a22c1d6fad3bee33794a54ae0ec192632dc13d6f8d4134" in local
