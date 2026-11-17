# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1's release DSpark draft (Inferact/Kimi-K3.1-Pro-DSpark, the GQA drafter
``dspark_gqa.K3GqaDSparkForCausalLM``) under the specialized decode.

CPU (pytest, no kernel launch): the routing of a speculators ``dspark`` checkpoint
on a Kimi K3.1 target, the drafter checks of ``runtime.unsupported``, the SP decode's
drafter declaration, the GQA tables' tactics, and the coverage of every drafter batch
the chosen deployments capture (results/chosen_configs_nov_15: the decode roles'
capture lists, max_num_seqs and k) by the strict pieces' tables.

GPU tests: ``test_dspark_gqa_fastpath_gpu.py``.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia import dense_draft  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode import runtime  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    draft,
    sp_attention,
)

K7_SIZES = sorted({7 * b for b in range(1, 33)} | {8 * b for b in range(1, 65)})
K3_SIZES = [
    6, 8, 15, 16, 18, 20, 21, 24, 30, 32, 39, 40, 45, 48, 54, 56, 63, 64, 66, 68, 69,
    72, 75, 76, 78, 80, 81, 84, 87, 88, 90, 92, 93, 96, 100, 102, 104, 108, 112, 114,
    116, 120, 124, 126, 128, 132, 136, 138, 144, 150, 152, 156, 160, 162, 168, 174,
    176, 180, 184, 186, 192, 200, 204, 208, 216, 224, 228, 232, 240, 248, 252, 256,
    264, 272, 276, 288, 300, 304, 312, 320, 324, 336, 348, 352, 360, 368, 372, 384,
    400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640,
]  # fmt: skip
# K3_SIZES at draft length 5 (results/chosen_configs_nov_15/release_draft: gen_dsrel.py
# --k-override C128-1P1D=5,C160-1P1D-llh=5): its verify batches x 6, its drafter
# batches x 5, cap 160 x 6.
K5_SIZES = [
    10, 12, 24, 25, 30, 35, 36, 40, 48, 50, 60, 65, 72, 75, 80, 84, 90, 96, 102, 105,
    108, 110, 114, 115, 120, 125, 126, 130, 132, 135, 138, 140, 144, 145, 150, 155, 156,
    160, 162, 168, 170, 174, 180, 186, 190, 192, 198, 200, 204, 210, 216, 220, 228, 230,
    234, 240, 250, 252, 260, 264, 270, 276, 280, 288, 290, 300, 306, 310, 312, 320, 324,
    336, 340, 342, 348, 360, 372, 378, 380, 384, 396, 400, 408, 414, 420, 432, 440, 450,
    456, 460, 468, 480, 486, 500, 504, 520, 522, 528, 540, 552, 558, 560, 576, 580, 600,
    620, 624, 640, 648, 672, 696, 720, 744, 768, 792, 800, 816, 840, 864, 888, 912, 936,
    960,
]  # fmt: skip
# The decode roles of the chosen points (results/chosen_configs_nov_15/configs/<point>/
# agentx/render/config.yaml): (TP, k, max_num_seqs, max capture size, capture sizes);
# "-k5": the k5 renders of C128 / C160 (release_draft/k5/configs).
POINTS = {
    "C1": (16, 7, 32, 256, [s for s in K7_SIZES if s <= 256]),
    "C16-1P2D": (16, 7, 64, 512, K7_SIZES),
    "C72-1P3D": (16, 7, 64, 512, K7_SIZES),
    "C48-1P1D": (16, 7, 48, 384, [s for s in K7_SIZES if s <= 384]),
    "C64": (16, 7, 64, 512, K7_SIZES),
    "C128-1P1D": (16, 3, 160, 640, K3_SIZES),
    "C160-1P1D-llh": (16, 3, 160, 640, K3_SIZES),
    "C128-1P1D-k5": (16, 5, 160, 960, K5_SIZES),
    "C160-1P1D-llh-k5": (16, 5, 160, 960, K5_SIZES),
    "C512": (4, 3, 128, 512, [4 * b for b in range(1, 129)]),
}


def _config(tp, k, seqs, cap, sizes):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=cap
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
    )


def _batches(point):
    tp, k, seqs, cap, sizes = POINTS[point]
    return tp, k, draft.draft_batches(_config(tp, k, seqs, cap, sizes), k)


def test_chosen_points_drafter_batches():
    """The drafter batches the chosen deployments capture (draft_batches on their
    decode capture lists): every request count up to max_num_seqs at k7 (but the ones
    no k7 / k8 multiple rounds to), B2-B160 at k3 TP16, B2-B128 at C512."""
    tp, k, batches = _batches("C1")
    assert batches == list(range(1, 33))
    tp, k, batches = _batches("C64")
    assert batches[0] == 1 and batches[-1] == 64 and 33 not in batches
    tp, k, batches = _batches("C128-1P1D")
    assert (k, batches[0], batches[-1]) == (3, 2, 160) and 1 not in batches
    tp, k, batches = _batches("C128-1P1D-k5")
    assert (k, batches[0], batches[-1], len(batches)) == (5, 2, 160, 90)
    assert 1 not in batches
    tp, k, batches = _batches("C512")
    assert (tp, k, batches[0], batches[-1]) == (4, 3, 2, 128)


@pytest.mark.parametrize("point", [p for p, v in POINTS.items() if v[0] == 16])
def test_tp16_points_strict_tables_cover_every_batch(point):
    """NQ-NOFALLBACK at TP16 = DCP16: every captured drafter batch of the point has
    fused Markov sampling, the sharded context step at its context rows and Lamport
    all-reduce sites at its query rows (the build refuses otherwise)."""
    tp, k, batches = _batches(point)
    top = max(batches)
    markov = draft.batch_table(draft.closed(draft.GQA_FUSED_MARKOV[tp], top))
    context = draft.closed(draft.GQA_FAST_CONTEXT[tp], top * (k + 1))
    sites = draft.closed(draft.GQA_FAST_AR[tp], top * k)
    for b in batches:
        assert markov.get(b), (point, b)
        assert draft.covered(context, b * (k + 1), k), (point, b)
        assert draft.covered(sites, b * k, k), (point, b)


def test_c512_strict_tables_cover_every_batch():
    """NQ-NOFALLBACK at C512's SP decode role (TP4 = DCP4, DL3 B2-B128 per DP rank):
    every captured drafter batch has the vocabulary-parallel Markov sampling (the
    fused kernels cannot hold TP4 at Markov rank 512), the sharded context step and
    Lamport all-reduce sites."""
    tp, k, batches = _batches("C512")
    top = max(batches)
    markov = draft.batch_table(draft.closed(draft.GQA_VP_MARKOV[tp], top))
    context = draft.closed(draft.GQA_FAST_CONTEXT[tp], top * (k + 1))
    sites = draft.closed(draft.GQA_FAST_AR[tp], top * k)
    assert tp not in draft.GQA_FUSED_MARKOV
    for b in batches:
        assert markov.get(b), b
        assert draft.covered(context, b * (k + 1), k), b
        assert draft.covered(sites, b * k, k), b
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import markov

    rank512 = (163840, 512)
    limits = (212, 232448, 233472)  # VR200
    assert markov.supports(rank512, rank512, 16, limits)
    assert not markov.supports(rank512, rank512, 4, limits)


def test_gqa_shapes():
    assert draft.gqa_shapes(16) == {
        "qkv": (512, 7168),
        "o_proj": (7168, 384),
        "gate_up": (1792, 7168),
        "down": (7168, 896),
        "ctx_shard": (448, 35840),
        "ctx_kv": (640, 7168),
    }
    assert draft.gqa_shapes(4) == {
        "qkv": (2048, 7168),
        "o_proj": (7168, 1536),
        "gate_up": (7168, 7168),
        "down": (7168, 3584),
        "ctx_shard": (1792, 35840),
        "ctx_kv": (2560, 7168),
    }


def _tactics(tp):
    tables = [draft.GQA_PROJECTIONS.get(tp, {})]
    tables += [t for _, t in draft.GQA_PROJECTIONS_WIDE.get(tp, ())]
    tables.append(draft.GQA_RING_PROJECTIONS.get(tp, {}))
    return [(shape, t) for table in tables for shape, t in table.items()]


@pytest.mark.parametrize("tp", [4, 8, 16])
def test_gqa_tactics_valid(tp):
    """Every GQA split-K tactic is one of the drafter's shapes at its TP and splits
    its K in whole 128-column tiles (the kernel drops a remainder silently); the
    tensor-memory tactics only for K 7168."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        CTA_K,
    )

    shapes = set(draft.gqa_shapes(tp).values())
    for shape, tactic in _tactics(tp):
        assert shape in shapes, (tp, shape)
        if len(tactic) == 4:
            assert shape[1] % (CTA_K * tactic[2]) == 0, (shape, tactic)
        else:
            assert shape[1] == 7168, (shape, tactic)


def test_gqa_projection_tactic_bins():
    """The <= 8-row table, then the wide bins by row count, cuBLAS above them."""
    shape = (1792, 7168)
    assert draft.gqa_projection_tactic(shape, 16, 7) == draft.PROJECTIONS[16][shape]
    assert draft.gqa_projection_tactic(shape, 16, 14) == (64, 16, 4, 8)
    assert draft.gqa_projection_tactic(shape, 16, 448) is None
    assert draft.gqa_projection_tactic((1, 1), 16, 7) is None


def test_gqa_tables_do_not_touch_preview_declaration():
    """The SP decode declares the preview drafter stock but for FAST_DCP: the GQA_*
    tables (the release drafter's own build) never break that check."""
    assert "FAST_DCP" in sp_attention.check_stock_drafter(4)


def _spec(architectures, method="dspark", k=7):
    hf = SimpleNamespace(architectures=list(architectures), kv_lora_rank=None)
    model = SimpleNamespace(
        hf_config=hf, hf_text_config=hf, architectures=list(architectures)
    )

    def update_arch_():
        model.architectures = list(hf.architectures)

    return SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        draft_model_config=model,
        attention_backend=None,
        kv_cache_dtype=None,
        update_arch_=update_arch_,
    )


def test_route_release_draft(monkeypatch):
    """A speculators dspark checkpoint (Qwen3DSparkModel) on a Kimi K3.1 (KDA hybrid)
    target loads as K3GqaDSparkForCausalLM, with FLASH_ATTN / auto KV defaults;
    idempotent; an MLA draft keeps its own class."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    target = SimpleNamespace(linear_attn_config={})
    spec = _spec(["Qwen3DSparkModel"])
    dense_draft.configure_dense_draft(spec, target)
    assert spec.draft_model_config.architectures == ["K3GqaDSparkModel"]
    assert dense_draft.is_gqa_dspark(spec)
    assert spec.kv_cache_dtype == "auto"
    dense_draft.configure_dense_draft(spec, target)
    assert spec.draft_model_config.architectures == ["K3GqaDSparkModel"]
    mla = _spec(["K3DSparkModel"])
    mla.draft_model_config.hf_text_config = SimpleNamespace(kv_lora_rank=512)
    dense_draft.configure_dense_draft(mla, target)
    assert mla.draft_model_config.architectures == ["K3DSparkModel"]
    assert not dense_draft.is_gqa_dspark(mla)
    # Not a KDA hybrid target (no linear_attn_config): untouched.
    plain = _spec(["Qwen3DSparkModel"])
    dense_draft.configure_dense_draft(plain, SimpleNamespace())
    assert plain.draft_model_config.architectures == ["Qwen3DSparkModel"]


def test_registry_maps_release_class():
    from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

    module, cls = _SPECULATIVE_DECODING_MODELS["K3GqaDSparkModel"]
    assert (module, cls) == (
        "vllm.models.kimi_k3.nvidia.dspark_gqa",
        "K3GqaDSparkForCausalLM",
    )


def _vllm_config(architectures, tp=16, dcp=16, k=7):
    return SimpleNamespace(
        speculative_config=_spec(architectures, k=k),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=dcp
        ),
    )


@pytest.mark.parametrize(
    "architectures,tp,sp,refused",
    [
        (["K3GqaDSparkModel"], 16, False, False),
        (["K3DSparkModel"], 16, False, False),
        (["Qwen3DSparkModel"], 16, False, True),
        (["Qwen3DSparkModel"], 8, False, False),  # (not strict: TP8 = DCP8)
        (["Qwen3DSparkModel"], 4, True, True),  # (C512's SP decode role)
        (["K3GqaDSparkModel"], 4, True, False),
    ],
)
def test_drafter_unsupported(monkeypatch, architectures, tp, sp, refused):
    """NQ-NOFALLBACK: on strict deployments only the specialized drafters' classes."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_device_capability", lambda cap: True)
    config = _vllm_config(architectures, tp=tp, dcp=tp, k=3 if sp else 7)
    reason = runtime.drafter_unsupported(config, sp)
    assert (reason is not None) == refused, reason


def _drafter_config(target_tp, target_dcp, draft_dcp):
    """A drafter's vllm_config as DFlashSpeculator hands it to a GQA draft: its own
    parallel config at ``draft_dcp`` (1), the target's in the speculative config."""
    target = SimpleNamespace(
        tensor_parallel_size=target_tp, decode_context_parallel_size=target_dcp
    )
    own = SimpleNamespace(
        tensor_parallel_size=target_tp, decode_context_parallel_size=draft_dcp
    )
    spec = SimpleNamespace(target_parallel_config=target)
    return SimpleNamespace(parallel_config=own, speculative_config=spec)


@pytest.mark.parametrize("tp,dcp", [(16, 16), (4, 4), (8, 8)])
def test_deployment_config_is_the_targets(tp, dcp):
    # NQ-NOFALLBACK: a GQA draft's own config runs at DCP 1, so strictness keyed on
    # it would never hold at TP16 = DCP16 (or C512's TP4 = DCP4); the deployment
    # config carries the target's DCP.
    config = _drafter_config(tp, dcp, 1)
    deployment = draft.deployment_config(config)
    assert deployment.parallel_config.decode_context_parallel_size == dcp
    assert config.parallel_config.decode_context_parallel_size == 1  # untouched copy
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import target_topology

    assert target_topology(True, 16, draft.deployment_config(
        _drafter_config(16, 16, 1)).parallel_config.decode_context_parallel_size, 8)


def test_deployment_config_without_target():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=16),
        speculative_config=SimpleNamespace(target_parallel_config=None),
    )
    assert draft.deployment_config(config) is config


@pytest.mark.parametrize("loaded", [False, True])
def test_context_kv_buffers_built_before_fast_path(monkeypatch, loaded):
    """``process_weights_after_loading`` builds the context step's fused K/V buffers
    (``_build_fused_kv_buffers``, as the stock lazy build does: under
    ``weights_already_processed``) before ``GqaDraftFastPath.build`` wraps the qkv
    projections, when the load skipped them (dummy weights: no ``load_weights``);
    a real load (buffers built in ``load_weights``) is left as it is."""
    from contextlib import nullcontext

    from vllm.model_executor.utils import is_weights_pre_processed
    from vllm.models.kimi_k3.nvidia import cute_cache, dspark_gqa, mxfp8

    calls = []

    class Inner:
        def _build_fused_kv_buffers(self):
            calls.append(("buffers", is_weights_pre_processed()))
            self._num_attn_layers = 5
            self._fused_kv_weight = object()

    inner = Inner()
    if loaded:
        inner._num_attn_layers = 5
        inner._fused_kv_weight = "loaded"

    def build(vllm_config, model):
        calls.append(("build", getattr(model.model, "_fused_kv_weight", None)))
        return "fast"

    monkeypatch.setattr(mxfp8, "enable_kimi_k3_mxfp8_dispatch", lambda model: 0)
    monkeypatch.setattr(cute_cache, "shared_phase", lambda name: nullcontext())
    monkeypatch.setattr(draft.GqaDraftFastPath, "build", staticmethod(build))
    model = SimpleNamespace(model=inner, _vllm_config=None, _fast_path=None)
    dspark_gqa.K3GqaDSparkForCausalLM.process_weights_after_loading(model)
    assert model._fast_path == "fast"
    if loaded:
        assert calls == [("build", "loaded")]
    else:
        assert calls[0] == ("buffers", True)
        assert calls[1][0] == "build" and calls[1][1] is not None


@pytest.mark.parametrize("requires", [False, True])
@pytest.mark.parametrize(
    "bypass",
    ["acceptance_estimator", "use_confidence_head", "row_mapping", "d2t", "fp64"],
)
def test_speculator_markov_bypass_refused_when_strict(requires, bypass):
    """A speculator-side bypass of the fused Markov drafting raises for a draft model
    that requires it (the strict release drafter) and keeps the stock loop otherwise."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = object.__new__(DSparkSpeculator)
    spec._draft_topk = None
    spec.use_confidence_head = bypass == "use_confidence_head"
    spec.acceptance_estimator = object() if bypass == "acceptance_estimator" else None
    spec.draft_watermarker = None
    spec.sample_from_anchor = True
    spec.num_query_per_req = spec.num_speculative_steps = 3
    spec.use_fp64_gumbel = bypass == "fp64"
    spec._d2t_scatter_index = object() if bypass == "d2t" else None
    spec.draft_logits = SimpleNamespace(
        values=None, row_mapping=object() if bypass == "row_mapping" else None
    )
    spec.model = SimpleNamespace(
        requires_fused_markov=requires,
        sampled_markov_draft=lambda *a: pytest.fail("fused path must not run"),
    )
    if requires:
        with pytest.raises(RuntimeError, match="requires fused Markov"):
            spec._fused_greedy_markov(2, None)
    else:
        assert spec._fused_greedy_markov(2, None) is False
