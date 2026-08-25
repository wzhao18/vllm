# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _kimi_flashinfer_test_config(
    *,
    max_num_batched_tokens: int = 64,
    tensor_parallel_size: int = 1,
    additional_config: dict | None = None,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
        load_config=SimpleNamespace(load_format="auto"),
        additional_config=additional_config or {},
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
        ),
    )


def test_kimi_flashinfer_mega_moe_mapping_includes_nvfp4_metadata():
    from vllm.models.kimi_k3.nvidia.model import (
        make_kimi_k3_mega_moe_expert_params_mapping,
    )

    mapping = make_kimi_k3_mega_moe_expert_params_mapping(
        1, include_nvfp4_metadata=True
    )

    assert [entry[1].rsplit(".", 1)[-1] for entry in mapping[:4]] == [
        "weight_scale_2",
        "input_scale",
        "weight_scale",
        "weight",
    ]
    assert (
        "experts.w13_weight",
        "experts.0.w1.weight",
        0,
        "w1",
    ) in mapping
    assert (
        "experts.w13_weight_scale_2",
        "experts.0.w1.weight_scale_2",
        0,
        "w1",
    ) in mapping
    assert (
        "experts.w13_input_scale",
        "experts.0.w3.input_scale",
        0,
        "w3",
    ) in mapping
    assert (
        "experts.w2_weight_scale_2",
        "experts.0.w2.weight_scale_2",
        0,
        "w2",
    ) in mapping

    expected_destinations = {
        "weight": "experts.w13_weight",
        "weight_scale": "experts.w13_weight_scale",
        "weight_scale_2": "experts.w13_weight_scale_2",
        "input_scale": "experts.w13_input_scale",
    }
    for checkpoint_suffix, destination in expected_destinations.items():
        checkpoint_name = f"layers.1.mlp.experts.0.w1.{checkpoint_suffix}"
        matched = next(entry for entry in mapping if entry[1] in checkpoint_name)
        assert matched[0] == destination


def test_kimi_flashinfer_mega_moe_loader_preserves_nvfp4_metadata():
    from vllm.models.kimi_k3.nvidia.model import KimiK3FlashInferMegaMoEExperts

    experts = KimiK3FlashInferMegaMoEExperts(
        _kimi_flashinfer_test_config(),
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        activation="situ",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )

    assert experts.w13_weight_scale.shape == (2, 256, 8)
    assert experts.w2_weight_scale.shape == (2, 128, 8)
    assert experts.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert experts.w2_weight_scale.dtype == torch.float8_e4m3fn

    for parameter, name, shard, value in (
        (experts.w13_weight_scale_2, "w13_weight_scale_2", "w1", 3.0),
        (experts.w13_weight_scale_2, "w13_weight_scale_2", "w3", 7.0),
        (experts.w13_input_scale, "w13_input_scale", "w1", 11.0),
        (experts.w13_input_scale, "w13_input_scale", "w3", 13.0),
        (experts.w2_weight_scale_2, "w2_weight_scale_2", "w2", 17.0),
        (experts.w2_input_scale, "w2_input_scale", "w2", 19.0),
    ):
        assert experts.weight_loader(
            parameter,
            torch.tensor(value),
            name,
            shard_id=shard,
            expert_id=2,
            return_success=True,
        )

    assert torch.equal(experts.w13_weight_scale_2[0], torch.tensor([3.0, 7.0]))
    assert torch.equal(experts.w13_input_scale[0], torch.tensor([11.0, 13.0]))
    assert experts.w2_weight_scale_2[0].item() == 17.0
    assert experts.w2_input_scale[0].item() == 19.0


@pytest.mark.parametrize("integrated_shared", [False, True])
def test_kimi_flashinfer_mega_moe_builds_modelopt_scale_algebra(
    monkeypatch, integrated_shared
):
    import flashinfer.moe_ep as moe_ep

    from vllm.models.kimi_k3.nvidia import model as kimi_model

    experts = kimi_model.KimiK3FlashInferMegaMoEExperts(
        _kimi_flashinfer_test_config(),
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        activation="situ",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )
    experts.w13_weight_scale_2.data.copy_(torch.tensor([[5.0, 5.0], [7.0, 7.0]]))
    experts.w2_weight_scale_2.data.copy_(torch.tensor([11.0, 13.0]))
    experts.w13_input_scale.data.copy_(torch.tensor([[2.0, 1.0], [3.0, 1.0]]))
    experts.w2_input_scale.data.copy_(torch.tensor([4.0, 2.0]))
    experts._check_runtime_supported = lambda: None
    if integrated_shared:
        experts.set_integrated_shared_expert(
            SimpleNamespace(hidden_size=128, intermediate_size=256)
        )

    ep_group = SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)
    monkeypatch.setattr(kimi_model, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(
        kimi_model, "current_stream", lambda: SimpleNamespace(cuda_stream=7)
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    captured: dict[str, Any] = {}

    class FakeMegaLayer(torch.nn.Module):
        pass

    def make_layer(bootstrap, fleet_params, weights, *, backend):
        captured.update(
            bootstrap=bootstrap,
            fleet_params=fleet_params,
            weights=weights,
            backend=backend,
        )
        return FakeMegaLayer()

    monkeypatch.setattr(moe_ep, "MoEEpLayer", make_layer)

    experts.finalize_weights()

    kernel = captured["backend"].megakernel
    assert kernel.fc1_alpha is None
    assert kernel.fc2_alpha is None
    assert kernel.fc1_norm_const is None
    torch.testing.assert_close(
        experts._flashinfer_fc1_alpha, torch.tensor([15.0, 21.0])
    )
    torch.testing.assert_close(
        experts._flashinfer_fc2_alpha, torch.tensor([44.0, 52.0])
    )
    torch.testing.assert_close(
        experts._flashinfer_fc1_norm_const, torch.tensor([0.25, 0.25])
    )
    assert kernel.input_norm_const == pytest.approx(1.0 / 3.0)
    assert kernel.activation == "situ"
    assert kernel.situ_beta == 4.0
    assert kernel.situ_linear_beta == 25.0
    assert kernel.in_kernel_fc2_reduce
    assert kernel.combine_dtype == "bf16"
    expected_knobs = {
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 512,
        "max_active_clusters": 60,
        "epi_flag_batch": (2, 4),
        "load_balance_mode": "atomic_counter",
        "mma_tiler_mnk": (256, 128, 256),
        "flag_batch": 4,
        "token_back_mode": "standalone_warps",
    }
    if integrated_shared:
        expected_knobs.pop("max_active_clusters")
        assert kernel.shared_hidden_size == 128
        assert kernel.shared_intermediate_size == 256
    assert kernel.knobs == expected_knobs
    assert captured["weights"].w13_scale.dtype == torch.float8_e4m3fn
    assert captured["weights"].w13_scale.shape == (2, 256, 8)
    assert experts.w13_weight is None
    assert experts.w2_weight is None


def test_kimi_flashinfer_mega_moe_overrides_kernel_knobs():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_mega_moe_knobs,
    )

    default_knobs = get_kimi_nvfp4_mega_moe_knobs(_kimi_flashinfer_test_config())
    assert default_knobs["max_active_clusters"] == 60

    config = _kimi_flashinfer_test_config(
        additional_config={
            "kimi_nvfp4_mega_moe_knobs": {
                "max_active_clusters": 72,
                "mma_tiler_mnk": [256, 256, 256],
            }
        }
    )

    knobs = get_kimi_nvfp4_mega_moe_knobs(config)

    assert knobs["max_active_clusters"] == 72
    assert knobs["mma_tiler_mnk"] == (256, 256, 256)


def test_kimi_integrated_mega_moe_uses_full_hardware_grid_by_default():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_integrated_mega_moe_knobs,
    )

    knobs = get_kimi_nvfp4_integrated_mega_moe_knobs(
        _kimi_flashinfer_test_config()
    )
    assert "max_active_clusters" not in knobs

    config = _kimi_flashinfer_test_config(
        additional_config={
            "kimi_nvfp4_mega_moe_knobs": {"max_active_clusters": 64}
        }
    )
    knobs = get_kimi_nvfp4_integrated_mega_moe_knobs(config)
    assert knobs["max_active_clusters"] == 64


def test_kimi_fused_shared_expert_uses_routed_grid_complement():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_fused_shared_expert_num_sms,
    )

    config = _kimi_flashinfer_test_config()
    assert get_kimi_nvfp4_fused_shared_expert_num_sms(config, 148) == 28

    config = _kimi_flashinfer_test_config(
        additional_config={"kimi_nvfp4_mega_moe_knobs": {"max_active_clusters": 64}}
    )
    assert get_kimi_nvfp4_fused_shared_expert_num_sms(config, 148) == 20


def test_kimi_fused_shared_expert_num_sms_override():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_fused_shared_expert_num_sms,
    )

    config = _kimi_flashinfer_test_config(
        additional_config={"kimi_nvfp4_fused_shared_expert_num_sms": 16}
    )
    assert get_kimi_nvfp4_fused_shared_expert_num_sms(config, 148) == 16


def test_kimi_nvfp4_shared_expert_restores_checkpoint_layout():
    from vllm.models.kimi_k3.nvidia.model import KimiFusedSharedExpert

    hidden_size = 8
    intermediate_size = 16
    expert = KimiFusedSharedExpert.__new__(KimiFusedSharedExpert)
    torch.nn.Module.__init__(expert)
    expert.hidden_size = hidden_size
    expert.intermediate_size = intermediate_size
    packed = torch.nn.Parameter(torch.empty(2 * intermediate_size, hidden_size))
    expert.gate_up_proj = SimpleNamespace(weight=packed)
    down = torch.randn(hidden_size, intermediate_size)
    expert.down_proj = SimpleNamespace(weight=down)
    gate = torch.randn(intermediate_size, hidden_size)
    up = torch.randn(intermediate_size, hidden_size)

    expert._load_gate_up_weight(packed, gate, 0)
    expert._load_gate_up_weight(packed, up, 1)

    canonical_fc1, canonical_fc2 = expert._canonical_weights()
    torch.testing.assert_close(canonical_fc1, torch.cat((gate, up)))
    assert canonical_fc1.is_contiguous()
    assert canonical_fc2 is down


def test_kimi_flashinfer_mega_moe_quantized_combine_configuration():
    from vllm.models.kimi_k3.nvidia.model import (
        KimiK3FlashInferMegaMoEExperts,
    )

    config = _kimi_flashinfer_test_config(
        additional_config={"kimi_nvfp4_mega_moe_combine_dtype": "mxfp8"}
    )
    experts = KimiK3FlashInferMegaMoEExperts(
        config,
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        activation="situ",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )

    assert experts._mega_moe_combine_dtype == "mxfp8"
    assert experts._mega_moe_knobs["token_back_mode"] == "reuse_dispatch_warps"
    assert experts._mega_moe_knobs["non_ubulk_fc2_store"]


def test_kimi_flashinfer_mega_moe_staged_bf16_combine_configuration():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_mega_moe_in_kernel_reduce,
    )

    config = _kimi_flashinfer_test_config(
        additional_config={"kimi_nvfp4_mega_moe_in_kernel_fc2_reduce": False}
    )

    assert not get_kimi_nvfp4_mega_moe_in_kernel_reduce(config)


@pytest.mark.parametrize(
    ("with_shared_expert", "integrate_shared_expert"),
    [(False, True), (True, True), (True, False)],
)
def test_kimi_flashinfer_mega_moe_forward_preserves_tensor_contract(
    monkeypatch, with_shared_expert, integrate_shared_expert
):
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    experts = kimi_model.KimiK3FlashInferMegaMoEExperts.__new__(
        kimi_model.KimiK3FlashInferMegaMoEExperts
    )
    torch.nn.Module.__init__(experts)
    experts.max_num_tokens = 8
    experts.capture_fn = None
    experts._flashinfer_fc1_alpha = torch.ones(2)
    experts._flashinfer_fc2_alpha = torch.ones(2)
    experts._flashinfer_fc1_norm_const = torch.ones(2)
    experts._integrated_shared_expert = None
    experts.synchronize_first_launch = lambda: None
    experts.finalize_weights = lambda: None
    monkeypatch.setattr(kimi_model, "is_forward_context_available", lambda: False)

    captured = {}

    class FakeMegaLayer(torch.nn.Module):
        def forward(self, tensors):
            captured["tensors"] = tensors
            return tensors.hidden_states.clone()

    experts._flashinfer_layer = FakeMegaLayer()
    hidden_states = torch.randn(3, 8)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    topk_weights = torch.rand(3, 2)
    shared_inputs = None

    if with_shared_expert:
        shared_inputs = object()

        class FakeSharedSession:
            def integrated_inputs(self, weights):
                assert weights is shared_weights
                return shared_inputs

        class FakeSharedExpert(torch.nn.Module):
            def _nvfp4_session(self, states):
                assert states is hidden_states
                return FakeSharedSession()

            def _nvfp4_weights(self):
                return shared_weights

        shared_weights = object()
        experts._integrated_shared_expert = FakeSharedExpert()

    output = experts(
        hidden_states,
        topk_weights,
        topk_ids,
        activation_clamp=None,
        integrate_shared_expert=integrate_shared_expert,
    )

    assert output.data_ptr() != hidden_states.data_ptr()
    assert torch.equal(output, hidden_states)
    assert captured["tensors"].hidden_states is hidden_states
    assert captured["tensors"].topk_ids is topk_ids
    assert captured["tensors"].topk_weights is topk_weights
    assert captured["tensors"].fc1_alpha is experts._flashinfer_fc1_alpha
    assert captured["tensors"].fc2_alpha is experts._flashinfer_fc2_alpha
    assert captured["tensors"].fc1_norm_const is experts._flashinfer_fc1_norm_const
    expected_shared_inputs = shared_inputs if integrate_shared_expert else None
    assert captured["tensors"].mega_shared_inputs is expected_shared_inputs


def test_kimi_flashinfer_mega_moe_uses_sequence_parallel_token_capacity():
    from vllm.models.kimi_k3.nvidia.model import KimiK3FlashInferMegaMoEExperts

    experts = KimiK3FlashInferMegaMoEExperts(
        _kimi_flashinfer_test_config(
            max_num_batched_tokens=16384,
            tensor_parallel_size=8,
        ),
        num_experts=16,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        activation="situ",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )

    assert experts.max_num_tokens == 2048


def test_kimi_nvfp4_shared_expert_buckets_session_capacity(monkeypatch):
    import flashinfer.moe_ep as moe_ep

    from vllm.models.kimi_k3.nvidia import model as kimi_model

    experts = kimi_model.KimiFusedSharedExpert.__new__(kimi_model.KimiFusedSharedExpert)
    experts.max_num_tokens = 8192
    experts.overlap_max_num_tokens = 256
    experts.hidden_size = 128
    experts.intermediate_size = 256
    experts.num_sms = 28
    experts.situ_beta = 4.0
    experts.situ_linear_beta = 25.0
    experts._nvfp4_sessions = {}

    capacities = []

    class FakeSession:
        def __init__(self, *, max_num_tokens, **kwargs):
            capacities.append(max_num_tokens)

    monkeypatch.setattr(moe_ep, "Nvfp4CutedslSharedExpertSession", FakeSession)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=148),
    )

    session_65 = experts._nvfp4_session(torch.empty(65, 128))
    session_129 = experts._nvfp4_session(torch.empty(129, 128))
    session_200 = experts._nvfp4_session(torch.empty(200, 128))
    session_257 = experts._nvfp4_session(torch.empty(257, 128))
    session_320 = experts._nvfp4_session(torch.empty(320, 128))

    assert session_65 is not session_129
    assert session_129 is session_200
    assert session_257 is not session_200
    assert session_257 is session_320
    assert capacities == [128, 256, 512]


def test_kimi_moe_stages_shared_input_before_integrated_kernel(monkeypatch):
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    calls = []
    shared_output = torch.full((2, 8), 3.0)

    class SharedExpert(kimi_model.KimiFusedSharedExpert):
        def stage_nvfp4(self, hidden_states, *, integrated=False):
            calls.append(("stage", integrated, hidden_states.shape[0]))

        def _nvfp4_session(self, hidden_states):
            return SimpleNamespace(output=lambda: shared_output)

    class RoutedExperts(kimi_model.KimiK3FlashInferMegaMoEExperts):
        def forward(self, hidden_states, *args, integrate_shared_expert=True, **kwargs):
            calls.append(("routed", integrate_shared_expert))
            return torch.full_like(hidden_states, 2.0)

    class OutputTransform(torch.nn.Module):
        def forward(self, hidden_states, residual=None):
            return hidden_states + residual

    shared = SharedExpert.__new__(SharedExpert)
    torch.nn.Module.__init__(shared)
    shared.overlap_max_num_tokens = 256
    routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)

    moe = kimi_model.KimiMoE.__new__(kimi_model.KimiMoE)
    torch.nn.Module.__init__(moe)
    moe.use_in_kernel_shared_fc12 = True
    moe.in_kernel_shared_fc12_max_tokens = 1024
    moe.use_mega_moe = True
    moe.shared_experts = shared
    moe.experts = routed
    moe.routed_output_transform = OutputTransform()
    moe._shared_expert_events = (object(), object())
    object.__setattr__(
        moe,
        "_maybe_overlap_router_and_down_proj",
        lambda hidden_states: (
            hidden_states,
            torch.ones(hidden_states.shape[0], 1),
            torch.zeros(hidden_states.shape[0], 1, dtype=torch.int64),
        ),
    )

    def execute(fn0, fn1, *args, **kwargs):
        assert kwargs == {"launch_aux_first": True}
        result1 = fn1()
        result0 = fn0()
        return result0, result1

    monkeypatch.setattr(kimi_model, "maybe_execute_in_parallel", execute)
    monkeypatch.setattr(kimi_model, "kimi_shared_expert_stream", lambda: object())

    output = moe(torch.ones(2, 8))

    assert calls == [("stage", True, 2), ("routed", True)]
    torch.testing.assert_close(output, torch.full((2, 8), 5.0))
