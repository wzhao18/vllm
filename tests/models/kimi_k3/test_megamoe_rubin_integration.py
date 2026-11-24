# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_backend="kimi_k3_mega_moe" wiring for Kimi K3, CPU only:

- the backend value: accepted, normalized, a native MegaMoE backend, Kimi K3 only;
- the automatic format policy (MXFP8 x MXFP4 up to 512 tokens/rank, else NVFP4), the
  ``megamoe_quant`` override (incl. W4A4 MXFP4) and the kernel variant / shipped knob
  record of the server's bucket;
- ``KimiK3MegaMoEExperts`` registers DeepGEMM MegaMoE's parameter names, shapes and
  dtypes at the K3.1 shapes, loads checkpoint shards as stored, converts to the
  kernel layout (raising on non-representable NVFP4 weights on every rank), exposes
  EPLB views, masks padding and passes the smallest EP-uniform bucket per forward
  (decode bucket at the token_comm decode-landing limit, else the server's bucket);
- the Kimi K3 model picks the experts class per backend; the DeepGEMM header
  install skips kimi_k3_mega_moe;
- the S1 token_comm drop is recorded in VENDOR.md.
"""

import dataclasses
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


def _vllm_config(mnbt=8192, dp=1):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt),
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        compilation_config=SimpleNamespace(static_forward_context={}),
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=1)
        ),
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


@pytest.mark.parametrize(
    ("requested", "kind"),
    [("nvfp4", "nvfp4"), ("mxfp4_mxfp8", "mxfp4_mxfp8"), ("mxfp4_mxfp4", "mxfp4")],
)
def test_format_policy_override(requested, kind):
    """additional_config megamoe_quant overrides the automatic policy at any size."""
    for tokens in (1, 8192):
        assert X.rubin_quant_kind(tokens, requested=requested) == kind
    C.RubinMoeProblem(quant_kind=kind)


def test_format_policy_override_rejects_unknown():
    for bad in ("mxfp4", "mxfp8", "w4a4"):
        with pytest.raises(ValueError, match="megamoe_quant"):
            X.rubin_quant_kind(8192, requested=bad)


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
    ("tokens", "caps"),
    [
        (2048, (256, 2048)),  # decode server: MNBT 8192 / TP4 (C512)
        (4096, (256, 4096)),
        (1025, (256, 1025)),
        (1024, (1024,)),  # genphase server: one bucket
        (512, (512,)),
        (256, (256,)),
        (160, (160,)),
    ],
)
def test_bucket_capacities(tokens, caps):
    assert X.rubin_bucket_capacities(tokens, 16, prefill_worker=False) == caps
    assert X.rubin_bucket_capacities(4096, 2, prefill_worker=False) == (2048, 4096)
    # P/D prefill workers keep their one bucket
    assert X.rubin_bucket_capacities(tokens, 16, prefill_worker=True) == (tokens,)
    with pytest.raises(ValueError):
        X.rubin_bucket_capacities(2048, 0, prefill_worker=False)


def test_decode_bucket_limit_matches_token_comm():
    """The decode bucket sits exactly at the token_comm trace-time gates (S1 decode
    landing and S0 dynamic token_in claim), read from the vendored source."""
    src = (
        PKG_DIR / "src/sources/communication/nvlink_domain/token_comm.py"
    ).read_text()
    gates = []
    for name in ("decode_landing_max_routes", "token_in_dynamic_claim_max_routes"):
        (value,) = re.findall(rf"{name}: ClassVar\[int\] = (\d+)", src)
        gates.append(int(value))
    # both decode paths are on at the bucket: it sits at the tighter gate (a368c5f66
    # raised the landing gate to 16384 routes; the dynamic claim stays at 4096)
    assert min(gates) == X.RUBIN_DECODE_BUCKET_MAX_ROUTES


@pytest.mark.parametrize(("world", "tokens"), [(16, 2048), (16, 4096), (8, 2048)])
def test_server_buckets_share_the_main_knob_record(world, tokens):
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(tokens))
    buckets, notes = X.select_rubin_buckets(
        problem, world, tokens, prefill_worker=False
    )
    main, _ = X.select_rubin_bucket(problem, world, tokens)
    nv = problem.quant_kind == "nvfp4"
    assert [b.max_tokens for b in buckets] == [256, tokens]
    # prefill-sized NVFP4 buckets run the dedup combine (same tile knobs)
    assert buckets[-1] == (X._prefill_dedup(main) if nv else main)
    assert buckets[0].variant == "inference" and buckets[0].knobs == main.knobs
    assert any("decode bucket" in n for n in notes)
    for b in buckets:  # every admission rule holds at both capacities
        C.RubinMoeConfig.make(problem, b, rank=0, world_size=world)
    # a P/D prefill worker keeps its one bucket
    assert X.select_rubin_buckets(problem, world, tokens, prefill_worker=True)[0] == (
        X._prefill_dedup(main) if nv else main,
    )
    # a genphase server keeps its one bucket
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(512))
    buckets = X.select_rubin_buckets(problem, 16, 512, prefill_worker=False)[0]
    assert len(buckets) == 1


def _inference_record_capacities():
    """(world, quant kind, record capacity) of every shipped inference knob record."""
    data = C.load_knob_source(C.SHIPPED_KNOBS)
    return sorted(
        {
            (e["world_size"], e["dtype"], e["max_tokens"])
            for e in data["entries"]
            if e.get("kernel_variant") == "inference"
        }
    )


@pytest.mark.parametrize(("world", "kind", "tokens"), _inference_record_capacities())
def test_decode_bucket_admits_every_shipped_inference_record(world, kind, tokens):
    """Wherever a decoding server would resolve a shipped inference record (any EP
    size, precision, capacity), the decode bucket on that record passes every
    admission rule (select_rubin_buckets checks it and raises otherwise)."""
    if tokens <= X.RUBIN_GENPHASE_MAX_TOKENS_PER_RANK:
        tokens = X.RUBIN_GENPHASE_MAX_TOKENS_PER_RANK + 1  # resolves the same record
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=kind)
    buckets, _ = X.select_rubin_buckets(problem, world, tokens, prefill_worker=False)
    decode = (
        X.RUBIN_WIDE_EP_DECODE_TOKENS
        if world in X.RUBIN_INFERENCE_ONLY_EP_SIZES
        else 256
    )
    assert [b.max_tokens for b in buckets] == [decode, tokens]
    C.RubinMoeConfig.make(problem, buckets[0], rank=0, world_size=world)


@pytest.mark.parametrize(
    ("role", "world", "mnbt", "tp", "token_sharded", "caps"),
    [
        # C512 prefill: DP4 x TP4 (EP16), MNBT 32768, sequence parallel
        ("kv_both", 16, 32768, 4, True, (8192,)),
        # C512 prefill if it ran without the 1/TP shard
        ("kv_both", 16, 32768, 4, False, (32768,)),
        # C256 prefill: TP8 (EP8) MoE token shard, MNBT 16384
        ("kv_both", 8, 16384, 8, True, (2048,)),
        # C512 decode: DP4 x TP4 (EP16), MNBT 8192, sequence parallel
        ("kv_consumer", 16, 8192, 4, True, (256, 2048)),
        # the same decode launched kv_both on the latency-specialized decode backend
        ("kv_both+latency_specialized", 16, 8192, 4, True, (256, 2048)),
        # a kv_producer is a prefill worker whatever its decode backend
        ("kv_producer+latency_specialized", 16, 8192, 4, True, (2048,)),
        # unified serving decodes too
        (None, 16, 8192, 4, True, (256, 2048)),
    ],
)
def test_deployment_buckets_per_role(
    monkeypatch, ep, role, world, mnbt, tp, token_sharded, caps
):
    """The experts module's buckets at the deployed roles' capacities: the P/D
    prefill workers (both kimi_k3_mega_moe) keep one bucket on the shipped record
    (prefill-sized NVFP4: dedup combine), the decode role adds the 256 decode bucket
    on the same record."""
    ep.world_size = world
    tokens = kimi_model.mega_moe_max_num_tokens(mnbt, tp, token_sharded)
    config = _vllm_config(mnbt=mnbt, dp=4)
    if role is not None:
        role, _, backend = role.partition("+")
        config.kv_transfer_config = SimpleNamespace(
            is_kv_producer=role in ("kv_producer", "kv_both"),
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
        )
        if backend:
            config.additional_config = {"decode_backend": backend}
    with torch.device("meta"):
        module = _experts(config, ep=ep, max_num_tokens=tokens, **K31)
    assert tuple(b.max_tokens for b in module.buckets) == caps
    main, _ = X.select_rubin_bucket(module.problem, world, tokens)
    # prefill-sized NVFP4 buckets: the dedup combine on the same tile knobs
    want = X._prefill_dedup(main) if module.problem.quant_kind == "nvfp4" else main
    assert module.bucket == want
    assert all(b.knobs in (main.knobs, want.knobs) for b in module.buckets)
    for b in module.buckets:
        C.RubinMoeConfig.make(module.problem, b, rank=0, world_size=world)


@pytest.mark.parametrize(
    ("world", "tokens"), [(16, 160), (16, 512), (16, 2048), (8, 2048)]
)
def test_shared_overlap_sm_cap_admits_every_server_bucket(monkeypatch, world, tokens):
    """reserve_sms_for_shared_expert's bucket (grid capped, fallback clusters
    dropped) passes the kernel admission rules for every shipped server record,
    including the mixed-CGA EP16 2048 one, and reaches the kernel config."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import die_local

    monkeypatch.setattr(die_local, "current_die_sm_mask32", lambda: None)
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(tokens))
    bucket, _ = X.select_rubin_bucket(problem, world, tokens)
    buckets, _ = X.select_rubin_buckets(problem, world, tokens, prefill_worker=False)
    module = SimpleNamespace(
        bucket=bucket, buckets=buckets, kernel_weights=None, prefix="layer",
        prefill_worker=False,
    )
    X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)
    capped = module.bucket
    assert module.sequence_parallel  # forward gets 1/TP shards (ep_tokens)
    # every compiled bucket (the decode bucket included) is capped
    assert len(module.buckets) == len(buckets)
    assert all(
        b.max_sm_count == X.RUBIN_SHARED_OVERLAP_MAX_SMS
        and b.knobs.fallback_cluster_shape_mn is None
        for b in module.buckets
    )
    assert capped == module.buckets[-1]
    assert capped.max_sm_count == X.RUBIN_SHARED_OVERLAP_MAX_SMS
    assert capped.knobs.fallback_cluster_shape_mn is None
    assert capped.variant == bucket.variant
    assert capped.max_tokens == bucket.max_tokens
    cfg = C.RubinMoeConfig.make(problem, capped, rank=0, world_size=world)
    assert cfg.max_sm_count == X.RUBIN_SHARED_OVERLAP_MAX_SMS
    module.kernel_weights = object()
    with pytest.raises(RuntimeError):
        X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)


def test_server_bucket_without_knob_record_borrows_nearest():
    """No EP4 genphase record: the nearest tuned contract's knobs are borrowed
    (juntian #21, for redundant experts / EP sizes without records), with a note."""
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind="mxfp4_mxfp8")
    bucket, notes = X.select_rubin_bucket(problem, 4, 256)
    assert bucket.max_tokens >= 256
    assert any("borrowed" in note for note in notes)


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
        # Redundant experts (juntian #21): physical slot p holds logical p % 448.
        redundant = _experts(_vllm_config(), num_logical_experts=448, **cfg)
        assert redundant.num_redundant_experts == 448
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

    def prepare_weights(self, w13, s13, w2, s2, *, layer_id, experts=None):
        return W.prepare_weights(
            self.kind, w13, s13, w2, s2, hidden=128, intermediate=128,
            layer_id=layer_id, expert_offset=4,
        )  # fmt: skip

    def forward(
        self, hidden, ids, weights, kernel_weights, *, bucket, out=None,
        is_padding=None, ep_tokens=None, owner=None, eplb=None,
    ):  # fmt: skip
        assert out is None  # the stock layer lets the runtime allocate
        self.eplb = eplb
        if is_padding is not None:  # the stager's folded padding mask
            ids = torch.where(is_padding.unsqueeze(1), -1, ids)
        assert ep_tokens == hidden.shape[0]  # DP1: the rank's own (EP-uniform) count
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

    # forward: padding rows masked to -1, the smallest bucket holding the tokens
    assert [b.max_tokens for b in module.buckets] == (
        [2048, 4096] if mnbt == 4096 else [64]
    )
    assert module.bucket is module.buckets[-1]
    hidden = torch.randn(3, 128).to(torch.bfloat16)
    ids = torch.tensor([[4, 5], [6, 7], [0, 1]], dtype=torch.int32)
    wts = torch.rand(3, 2)
    pad = torch.tensor([False, True, False, True])
    out = module(hidden, wts, ids, is_padding=pad)
    assert out.shape == (3, 128)
    staged_ids, staged_wts, kernel_weights, bucket = runtime.calls[-1]
    assert staged_ids.tolist() == [[4, 5], [-1, -1], [0, 1]]
    assert staged_ids.dtype == torch.int32 and staged_wts is wts
    assert kernel_weights is w and bucket is module.buckets[0]
    assert runtime.eplb is None
    # EPLB: the logical ids and the padding mask go to the stager unchanged, with the
    # layer's EPLB tensors (the stager maps and records; no separate launches).
    st = module.eplb_state
    st.logical_to_physical_map = torch.arange(16).view(8, 2)
    st.logical_replica_count = torch.ones(8, dtype=torch.int64)
    st.expert_load_view = torch.zeros(4, dtype=torch.int32)
    st.should_record_tensor = torch.tensor(True)
    st.num_unpadded_tokens_tensors = [torch.tensor(2)]
    module(hidden, wts, ids, is_padding=pad)
    assert runtime.calls[-1][0].tolist() == [[4, 5], [-1, -1], [0, 1]]
    e = runtime.eplb
    assert e.logical_to_physical_map is st.logical_to_physical_map
    assert e.logical_replica_count is st.logical_replica_count
    assert e.expert_load_view is st.expert_load_view
    assert e.record_enabled is st.should_record_tensor
    assert e.num_unpadded_tokens is st.num_unpadded_tokens_tensors[0]
    st.logical_to_physical_map = None
    if mnbt == 4096:
        big = torch.zeros(2049, 128, dtype=torch.bfloat16)
        module(big, torch.rand(2049, 2), torch.zeros(2049, 2, dtype=torch.int32))
        assert runtime.calls[-1][3] is module.bucket
    with pytest.raises(ValueError, match="sized for"):
        module(torch.zeros(mnbt + 1, 128, dtype=torch.bfloat16), wts, ids)


def test_bucket_selection_is_ep_uniform(monkeypatch, ep, small_bucket):
    """DP > 1: equal DP batch sizes -> the smallest bucket holding the tokens; unequal
    ones (the same counts on every rank) -> the largest bucket on every rank; no DP
    metadata -> error (a rank-local choice could deadlock the lock-step kernel)."""
    ep.world_size, ep.rank_in_group = 2, 1
    module = _experts(_vllm_config(dp=4), ep=ep, max_num_tokens=4096, **SMALL)
    small, large = module.buckets
    ctx = SimpleNamespace(dp_metadata=None)
    monkeypatch.setattr(X, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(X, "get_forward_context", lambda: ctx)
    with pytest.raises(RuntimeError, match="DP metadata"):
        module.select_bucket(52)
    counts = torch.tensor([208, 208, 208, 208])
    ctx.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=counts)
    assert module.select_bucket(52) is small
    assert module.select_bucket(2048) is small
    assert module.select_bucket(2049) is large
    # each forward has its own DPMetadata
    ctx.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=counts.clone())
    ctx.dp_metadata.num_tokens_across_dp_cpu[2] = 200
    assert module.select_bucket(52) is large
    with pytest.raises(ValueError, match="sized for"):
        module.select_bucket(4097)


def test_ep_tokens_is_the_ep_mean(monkeypatch, ep, small_bucket):
    """ep_tokens (the S2 epilogue choice) is EP-uniform: the rank's own count at DP1,
    else the EP mean of the DP batch sizes, sharded 1/TP under sequence parallelism
    (reserve_sms_for_shared_expert), None without DP metadata; once per forward."""
    ep.world_size, ep.rank_in_group = 2, 1
    assert _experts(_vllm_config(), ep=ep, **SMALL).ep_tokens(37) == 37
    module = _experts(_vllm_config(dp=4), ep=ep, max_num_tokens=4096, **SMALL)
    ctx = SimpleNamespace(dp_metadata=None)
    monkeypatch.setattr(X, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(X, "get_forward_context", lambda: ctx)
    assert module.ep_tokens(52) is None
    ctx.dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([17400, 17400, 17000, 17801])
    )
    assert module.ep_tokens(17000) == 17401  # not sequence parallel: the DP mean
    module.sequence_parallel = True
    monkeypatch.setattr(X, "get_tensor_model_parallel_world_size", lambda: 4)
    # shards 4350 / 4350 / 4250 / 4451 -> mean 4350.25 -> 4351, on every rank
    ctx.dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=ctx.dp_metadata.num_tokens_across_dp_cpu.clone()
    )
    assert module.ep_tokens(4250) == module.ep_tokens(4451) == 4351


def test_dp_uniform_check_runs_once_per_forward(monkeypatch, ep, small_bucket):
    """Every MoE layer of a forward shares one DP-uniform check (host overhead):
    the counts are reduced once per DPMetadata, i.e. once per forward."""
    ep.world_size, ep.rank_in_group = 2, 1
    layers = [
        _experts(
            _vllm_config(dp=4),
            ep=ep,
            max_num_tokens=4096,
            prefix=f"model.layers.{i}.mlp.experts",
            **SMALL,
        )
        for i in range(3)
    ]

    class Counts:
        def __init__(self, values):
            self.values, self.reductions = torch.tensor(values), 0

        def min(self):
            self.reductions += 1
            return self.values.min()

        def max(self):
            return self.values.max()

    ctx = SimpleNamespace(dp_metadata=None)
    monkeypatch.setattr(X, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(X, "get_forward_context", lambda: ctx)
    for values, index in (([52] * 4, 0), ([52, 52, 40, 52], 1), ([52] * 4, 0)):
        counts = Counts(values)
        ctx.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=counts)
        for module in layers:
            assert module.select_bucket(52) is module.buckets[index]
        assert counts.reductions == 1


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


# -- native NVFP4 checkpoints (ModelOpt export) --------------------------------------

# Key and dtype facts of nvidia/Kimi-K3.1-Pro-NVFP4 (safetensors headers, Nov 2026):
# experts.{e}.{w1,w3,w2}.{weight U8 (rows, K/2), weight_scale F8_E4M3 (rows, K/16),
# weight_scale_2 F32 (), input_scale F32 ()}.
NVFP4_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def _nvfp4_experts(vllm_config, *, ep, **kw):
    cfg = dict(SMALL)
    cfg.update(kw)
    cfg.setdefault("max_num_tokens", 4096)
    e_local = cfg["num_experts"] // ep.world_size
    return X.KimiK3MegaMoEExperts(
        vllm_config,
        num_experts=cfg.pop("num_experts"),
        num_local_experts=e_local,
        experts_start_idx=ep.rank_in_group * e_local,
        top_k=cfg.pop("top_k"),
        hidden_size=cfg.pop("hidden"),
        intermediate_size=cfg.pop("intermediate"),
        activation=SITU,
        weight_format=dg.MEGA_MOE_NVFP4,
        prefix=cfg.pop("prefix", "model.layers.3.mlp.experts"),
        **cfg,
    )


def _nvfp4_shards(gen, *, ws2=None):
    """Every expert's checkpoint tensors {(expert, shard, suffix): tensor}."""
    h, i = SMALL["hidden"], SMALL["intermediate"]
    out = {}
    for expert in range(SMALL["num_experts"]):
        g13 = ws2(expert) if ws2 else float(2.0 ** -int(torch.randint(10, 14, (1,), generator=gen)))
        for shard, (rows, cols) in (("w1", (i, h)), ("w3", (i, h)), ("w2", (h, i))):
            out[expert, shard, "weight"] = torch.randint(
                0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8
            )
            out[expert, shard, "weight_scale"] = torch.randint(
                0, 0x7F, (rows, cols // 16), generator=gen, dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            g = g13 if shard != "w2" else 2.0**-12
            out[expert, shard, "weight_scale_2"] = torch.tensor(g, dtype=torch.float32)
            out[expert, shard, "input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    return out


def _load_through_mapping(module, shards, layer_prefix="model.layers.3.mlp"):
    """Feed checkpoint keys through the Kimi K3 MegaMoE expert mapping (the model's
    load_weights substring match + rename) into the module's weight loader."""
    mapping = kimi_model.make_kimi_k3_mega_moe_expert_params_mapping(
        SMALL["num_experts"], module.weight_format
    )
    params = dict(module.named_parameters(prefix=f"{layer_prefix}.experts"))
    loaded = 0
    for (expert, shard, suffix), tensor in shards.items():
        name = f"{layer_prefix}.experts.{expert}.{shard}.{suffix}"
        for param_name, weight_name, expert_id, shard_id in mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params[name]
            ok = param.weight_loader(
                param, tensor, name, shard_id=shard_id, expert_id=expert_id,
                return_success=True,
            )  # fmt: skip
            loaded += bool(ok)
            break
        else:
            raise AssertionError(f"{name} matched no mapping entry")
    return loaded


def test_nvfp4_checkpoint_format_from_quant_config():
    """The MegaMoE format follows the ModelOpt quantized_layers entry of the block's
    experts (spelled as in the K3.1 NVFP4 export); other checkpoints stay MXFP4."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    prefix = "language_model.model.layers.5.block_sparse_moe"
    quant = {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "ignore": ["*block_sparse_moe.gate*", "lm_head"],
        "quantized_layers": {
            f"{prefix}.experts": {"quant_algo": "NVFP4", "group_size": 16},
            f"{prefix}.shared_experts.gate_proj": {"quant_algo": "MXFP8"},
            "language_model.model.layers.6.block_sparse_moe.experts": {
                "quant_algo": "MXFP8"
            },
        },
    }
    cfg = ModelOptMixedPrecisionConfig.from_config(quant)
    fmt = kimi_model.kimi_k3_mega_moe_weight_format
    assert fmt(cfg, prefix) is dg.MEGA_MOE_NVFP4
    assert fmt(None, prefix) is dg.MEGA_MOE_MXFP4
    with pytest.raises(NotImplementedError, match="MXFP8"):
        fmt(cfg, "language_model.model.layers.6.block_sparse_moe")


def test_nvfp4_mapping_covers_the_export_keys():
    mapping = kimi_model.make_kimi_k3_mega_moe_expert_params_mapping(
        896, dg.MEGA_MOE_NVFP4
    )
    pre = "language_model.model.layers.5.block_sparse_moe.experts"
    for shard, prefix in (("w1", "w13"), ("w3", "w13"), ("w2", "w2")):
        for suffix in NVFP4_SUFFIXES:
            name = f"{pre}.811.{shard}.{suffix}"
            hits = [m for m in mapping if m[1] in name]
            assert hits and hits[0][2:] == (811, shard)
            assert name.replace(hits[0][1], hits[0][0]) == f"{pre}.{prefix}_{suffix}"
    # the MXFP4 mapping is unchanged
    mx = kimi_model.make_kimi_k3_mega_moe_expert_params_mapping(2)
    assert ("experts.w13_weight", "experts.1.w3.weight_packed", 1, "w3") in mx


def test_nvfp4_parameters_and_dummy_values(ep, small_bucket):
    """NVFP4 layers run the NVFP4 kernel at every size (no MX at <= 512 tokens), hold
    the export's tensors as stored, and dummy-load finite power-of-two scales."""
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )

    ep.world_size, ep.rank_in_group = 16, 3
    with torch.device("meta"):
        big = _nvfp4_experts(
            _vllm_config(), ep=ep, num_experts=896, hidden=3584, intermediate=3072,
            top_k=16, max_num_tokens=64,
        )  # fmt: skip
    assert big.quant_kind == "nvfp4" and big.problem.quant_kind == "nvfp4"
    assert _params(big) == {
        "w13_weight": ((56, 6144, 1792), torch.uint8),
        "w13_weight_scale": ((56, 6144, 224), torch.float8_e4m3fn),
        "w13_weight_scale_2": ((56, 2), torch.float32),
        "w13_input_scale": ((56, 2), torch.float32),
        "w2_weight": ((56, 3584, 1536), torch.uint8),
        "w2_weight_scale": ((56, 3584, 192), torch.float8_e4m3fn),
        "w2_weight_scale_2": ((56,), torch.float32),
        "w2_input_scale": ((56,), torch.float32),
    }
    ep.world_size, ep.rank_in_group = 2, 0
    small = _nvfp4_experts(_vllm_config(), ep=ep)
    for name, param in small.named_parameters(recurse=False):
        initialize_single_dummy_weight(param, name=name)
    for name in ("w13_weight_scale", "w2_weight_scale"):
        values = getattr(small, name).float()
        assert bool((values == 1.0).all()), name
    assert bool((small.w13_weight_scale_2 == 2.0**-13).all())
    assert bool((small.w2_input_scale == 1.0).all())
    assert int(small.w13_weight.max()) > 0  # random codes


def test_nvfp4_load_and_process(monkeypatch, ep, small_bucket):
    ep.world_size, ep.rank_in_group = 2, 1
    module = _nvfp4_experts(_vllm_config(), ep=ep)
    shards = _nvfp4_shards(torch.Generator().manual_seed(4))
    assert _load_through_mapping(module, shards) == 4 * 3 * len(NVFP4_SUFFIXES)
    w13, s13, w2, s2 = module.canonical_weights()
    for local in range(4):
        expert = 4 + local
        part = {k[1:]: v for k, v in shards.items() if k[0] == expert}
        assert torch.equal(w13[local], torch.cat([part["w1", "weight"], part["w3", "weight"]]))
        assert torch.equal(
            s13[local],
            torch.cat([part["w1", "weight_scale"], part["w3", "weight_scale"]]).view(
                torch.uint8
            ),
        )
        assert torch.equal(w2[local], part["w2", "weight"])
        assert float(module.w13_weight_scale_2[local, 1]) == float(
            part["w3", "weight_scale_2"]
        )

    class Runtime:
        def prepare_nvfp4_checkpoint_weights(self, *tensors):
            return W.prepare_nvfp4_checkpoint_weights(*tensors, hidden=128, intermediate=128)

    monkeypatch.setattr(module, "get_runtime", lambda: Runtime())
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op, group: None)
    expect_fc1 = module.w13_weight_scale_2[:, 0].clone()
    module.process_weights_after_loading()
    w = module.kernel_weights
    W.validate_kernel_weights(w, quant_kind="nvfp4", experts=4, hidden=128, intermediate=128)
    assert torch.equal(w.fc1_alpha, expect_fc1)
    assert torch.equal(w.fc2_alpha, torch.full((4,), 2.0**-12))
    assert all(p.numel() == 0 for p in module.parameters())
    assert len(module.get_expert_weights()) == 7


@pytest.mark.parametrize("case", ["gate_up_scale_2", "input_scale"])
def test_nvfp4_refuses_math_it_cannot_run(monkeypatch, ep, small_bucket, case):
    """Unequal w1 / w3 weight_scale_2 or non-unit input scales raise on every rank
    instead of loading a different computation."""
    ep.world_size, ep.rank_in_group = 2, 0
    module = _nvfp4_experts(_vllm_config(), ep=ep)
    shards = _nvfp4_shards(torch.Generator().manual_seed(5))
    if case == "gate_up_scale_2":
        shards[2, "w3", "weight_scale_2"] = torch.tensor(2.0**-9)
    else:
        shards[1, "w2", "input_scale"] = torch.tensor(0.5)
    _load_through_mapping(module, shards)
    monkeypatch.setattr(module, "get_runtime", lambda: object())
    reduced = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, op, group: reduced.append(int(t))
    )
    err = ValueError if case == "gate_up_scale_2" else NotImplementedError
    with pytest.raises(err, match="weight_scale_2|input_scale"):
        module.process_weights_after_loading()
    assert reduced == [0] and module.kernel_weights is None


def test_nvfp4_compact_scales_wiring(monkeypatch, ep, small_bucket):
    """additional_config megamoe_compact_scales: an NVFP4 export with paired per-16
    scales loads, compacts every layer (prepare_rubin_megamoe's install), launches
    each layer on its own expanded planes and hands EPLB the compact planes."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import scale_staging as S

    ep.world_size, ep.rank_in_group = 2, 1
    cfg = _vllm_config()
    cfg.additional_config = {"megamoe_compact_scales": True}
    seen = []

    class Runtime:
        def prepare_nvfp4_checkpoint_weights(self, *tensors):
            return W.prepare_nvfp4_checkpoint_weights(
                *tensors, hidden=128, intermediate=128
            )

        def forward(self, hidden, ids, weights, kw, **_):
            seen.append(planes(kw).clone())
            return torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.bfloat16)

    def planes(w):
        return torch.cat([w.fc1[1].view(torch.uint8), w.fc2[1].view(torch.uint8)], 1)

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op, group: None)
    modules, want = [], []
    for layer in (3, 5):
        module = _nvfp4_experts(cfg, ep=ep, prefix=f"model.layers.{layer}.mlp.experts")
        assert module.compact_scales
        shards = _nvfp4_shards(torch.Generator().manual_seed(layer))
        for key, t in shards.items():
            if key[2] == "weight_scale":  # an MXFP4 cast: equal per-16 pairs
                pairs = t.view(torch.uint8).view(t.shape[0], -1, 2)
                pairs[..., 1] = pairs[..., 0]
        _load_through_mapping(module, shards, f"model.layers.{layer}.mlp")
        monkeypatch.setattr(module, "get_runtime", lambda: Runtime())
        module.process_weights_after_loading()
        want.append(planes(module.kernel_weights))
        modules.append(module)
    assert S.install(modules, None) is modules[0].scale_staging is not None
    for module in modules:
        module(torch.zeros(3, 128, dtype=torch.bfloat16), torch.rand(3, 2),
               torch.zeros(3, 2, dtype=torch.int32))  # fmt: skip
    assert len(seen) == 2 and all(torch.equal(s, w) for s, w in zip(seen, want))
    views = modules[1].get_expert_weights()
    # (4 experts, half of the 256 x 8 FC1 / 128 x 8 FC2 per-16 planes)
    assert len(views) == 7 and (views[1].shape, views[3].shape) == ((4, 1024), (4, 512))


def test_mxfp4_layer_refuses_nvfp4_scales(ep, small_bucket):
    ep.world_size, ep.rank_in_group = 2, 1
    module = _experts(_vllm_config(), ep=ep, **SMALL)
    param = module.w13_weight_scale
    with pytest.raises(ValueError, match="MXFP4"):
        param.weight_loader(param, torch.ones(()), "x.experts.w13_weight_scale_2", "w1", 4)


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
    """S0/S1 (decode landing, dynamic token_in claim) are upstream in the FlashInfer
    a368c5f66 drop: token_comm.py is listed in the manifest and is not a local
    difference any more."""
    text = (PKG_DIR / "VENDOR.md").read_text()
    manifest = dict(
        (path, digest)
        for digest, path in re.findall(r"^([0-9a-f]{64})  (sources/\S+)$", text, re.M)
    )
    assert "sources/communication/nvlink_domain/token_comm.py" in manifest
    upstream = text.split("## Upstream", 1)[1].split("\n## ", 1)[0]
    assert "a368c5f66" in upstream and "S0/S1" in upstream


@pytest.mark.parametrize("world", [28, 32])
def test_wide_ep_decode_bucket_has_its_own_record(world):
    """EP28/EP32 decode servers: inference kernel at every capacity, a 128-token
    decode bucket with its own (N128) record below the main bucket."""
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(2048))
    buckets, notes = X.select_rubin_buckets(problem, world, 2048, prefill_worker=False)
    assert [b.max_tokens for b in buckets] == [X.RUBIN_WIDE_EP_DECODE_TOKENS, 2048]
    # EP32: the decode bucket runs the refreshed GenPhase kernel (decode scheduler);
    # EP28 keeps the inference kernel's N128 record.
    decode_variant = "genphase" if world in X.RUBIN_GENPHASE_DECODE_EP_SIZES else "inference"
    assert [b.variant for b in buckets] == [decode_variant, "inference"]
    assert tuple(buckets[0].knobs.mma_tiler_mnk)[1] == 128
    assert any("its own record" in n for n in notes)
    prefill, _ = X.select_rubin_buckets(problem, world, 2048, prefill_worker=True)
    assert [b.max_tokens for b in prefill] == [2048]


def test_knob_world_size_emulates_wide_ep_on_a_slice():
    """A 1-node EP4 slice emulating EP32 per-GPU work selects the EP32 buckets and
    records (bench hook); without the hook EP4 keeps its own (prefill) records."""
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(2048))
    ep32, _ = X.select_rubin_buckets(problem, 32, 2048, prefill_worker=False)
    emulated, _ = X.select_rubin_buckets(
        problem, 4, 2048, prefill_worker=False, knob_world_size=32
    )
    assert emulated == ep32
    assert emulated[0].max_tokens == X.RUBIN_WIDE_EP_DECODE_TOKENS
    assert emulated[0].variant == "genphase"
    assert tuple(emulated[0].knobs.schedule_policy)[0] == "decode"
    native, _ = X.select_rubin_buckets(problem, 4, 2048, prefill_worker=False)
    assert native != ep32


@pytest.mark.parametrize("world", [4, 16])
def test_prefill_worker_nvfp4_runs_dedup_combine(world):
    """Prefill-sized NVFP4 buckets run the refreshed dedup-combine kernel with the
    same tile knobs and standalone-warp token back; the decode bucket keeps its kernel."""
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind="nvfp4")
    main, _ = X.select_rubin_bucket(problem, world, 8192)
    (bucket,), _ = X.select_rubin_buckets(problem, world, 8192, prefill_worker=True)
    assert bucket.variant == X.RUBIN_DEDUP_VARIANT
    assert bucket.knobs.token_back_mode == X.RUBIN_DEDUP_TOKEN_BACK_MODE
    assert dataclasses.replace(bucket.knobs, token_back_mode="epi_warps") == (
        dataclasses.replace(main.knobs, token_back_mode="epi_warps")
    )
    C.RubinMoeConfig.make(problem, bucket, rank=0, world_size=world)
    decode, _ = X.select_rubin_buckets(problem, world, 8192, prefill_worker=False)
    assert [b.variant for b in decode] == ["inference", X.RUBIN_DEDUP_VARIANT]
    assert decode[0].max_tokens < X.RUBIN_DEDUP_MIN_TOKENS
    with pytest.raises(ValueError, match="combine warps"):
        C.RubinMoeConfig.make(
            problem,
            dataclasses.replace(bucket, knobs=main.knobs),
            rank=0,
            world_size=world,
        )
