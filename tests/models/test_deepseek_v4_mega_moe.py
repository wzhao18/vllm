# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MegaMoEExperts,
    make_deepseek_v4_expert_params_mapping,
)
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 MegaMoE requires CUDA",
)


def test_deepseek_v4_mega_moe_expert_mapping():
    mapping = make_deepseek_v4_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]


def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
    raw = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)

    decoded = DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float(raw)

    assert torch.equal(decoded.view(torch.int32), raw.to(torch.int32) << 23)
    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 0.5
    assert decoded[2].item() == 1.0
    assert decoded[3].item() == 2.0


@pytest.mark.parametrize("use_kimi", [False, True])
def test_deep_gemm_mega_moe_capture_precedes_eplb(monkeypatch, use_kimi):
    experts_cls = DeepseekV4MegaMoEExperts
    if use_kimi:
        from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts

        experts_cls = KimiK3MegaMoEExperts

    experts = experts_cls.__new__(experts_cls)
    torch.nn.Module.__init__(experts)
    if use_kimi:
        experts.synchronize_first_launch = lambda: None
    experts.prefix = "model.layers.3.ffn.experts"
    experts.max_num_tokens = 4
    experts.capture_fn = None
    experts.get_symm_buffer = lambda: object()
    experts.eplb_state = SimpleNamespace(
        logical_to_physical_map=torch.empty(1),
        expert_load_view=torch.empty(1),
        logical_replica_count=torch.empty(1),
        should_record_tensor=torch.empty(1),
        num_unpadded_tokens_tensors=None,
    )

    topk_ids = torch.tensor([[1, 2], [3, 4]])
    captured: list[tuple[int, torch.Tensor]] = []
    bind_routed_experts_capturer(
        SimpleNamespace(modules=lambda: [experts]),
        SimpleNamespace(capture=lambda layer_id, ids: captured.append((layer_id, ids))),
    )

    class MappingReached(Exception):
        pass

    def map_ids(**kwargs):
        assert captured == [(3, topk_ids)]
        raise MappingReached

    monkeypatch.setattr(
        f"{experts_cls.__module__}.eplb_map_to_physical_and_record",
        map_ids,
    )
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm", lambda: SimpleNamespace()
    )

    with pytest.raises(MappingReached):
        experts(
            torch.empty(2, 8),
            torch.empty(2, 2),
            topk_ids,
            activation_clamp=None,
        )


def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
    )

    nonlocal_weight = torch.ones(128, 64, dtype=torch.uint8)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_weight,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((128, 64), 3, dtype=torch.uint8)
    w3 = torch.full((128, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 64), 11, dtype=torch.uint8)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :128], w1)
    assert torch.equal(experts.w13_weight[0, 128:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


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


def test_kimi_flashinfer_mega_moe_builds_modelopt_scale_algebra(monkeypatch):
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
    assert kernel.knobs == {
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 512,
        "max_active_clusters": 60,
        "epi_flag_batch": (2, 4),
        "load_balance_mode": "atomic_counter",
        "mma_tiler_mnk": (256, 128, 256),
        "flag_batch": 4,
        "token_back_mode": "standalone_warps",
    }
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


def test_kimi_fused_shared_expert_uses_routed_grid_complement():
    from vllm.models.kimi_k3.nvidia.model import (
        get_kimi_nvfp4_fused_shared_expert_num_sms,
    )

    config = _kimi_flashinfer_test_config()
    assert get_kimi_nvfp4_fused_shared_expert_num_sms(config, 148) == 28

    config = _kimi_flashinfer_test_config(
        additional_config={
            "kimi_nvfp4_mega_moe_knobs": {"max_active_clusters": 64}
        }
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
        additional_config={
            "kimi_nvfp4_mega_moe_in_kernel_fc2_reduce": False
        }
    )

    assert not get_kimi_nvfp4_mega_moe_in_kernel_reduce(config)


@pytest.mark.parametrize("preschedule", [False, True])
@pytest.mark.parametrize("routed_first", [False, True])
def test_kimi_mega_moe_overlaps_replicated_shared_experts(
    monkeypatch, preschedule, routed_first
):
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    moe = kimi_model.KimiMoE.__new__(kimi_model.KimiMoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.use_flashinfer_mega_moe = True
    moe.use_fused_mega_moe_tail = False
    moe.skip_shared_expert_for_profiling = False
    moe.preschedule_shared_expert = preschedule
    moe.use_fused_shared_expert = routed_first
    moe.fused_shared_expert_routed_first = routed_first
    moe._shared_expert_events = (object(), object())

    class RoutedExperts(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states + 2

    class SharedExperts(torch.nn.Module):
        shard_sequence_parallel = False

        def forward(self, hidden_states):
            return hidden_states + 3

    class RoutedOutputTransform(torch.nn.Module):
        def forward(self, hidden_states, residual=None):
            assert residual is not None
            return hidden_states + residual

    moe.experts = RoutedExperts()
    moe.shared_experts = SharedExperts()
    moe.routed_output_transform = RoutedOutputTransform()
    object.__setattr__(
        moe,
        "_maybe_overlap_router_and_down_proj",
        lambda hidden_states: (
            hidden_states + 1,
            torch.ones(hidden_states.shape[0], 1),
            torch.zeros(hidden_states.shape[0], 1, dtype=torch.int64),
            None,
        ),
    )

    stream = object()
    captured = {}

    def fake_execute(fn0, fn1, event0, event1, aux, **kwargs):
        captured["stream"] = aux
        captured["events"] = (event0, event1)
        captured["kwargs"] = kwargs
        return fn0(), fn1()

    monkeypatch.setattr(kimi_model, "maybe_execute_in_parallel", fake_execute)
    monkeypatch.setattr(kimi_model, "aux_stream", lambda: stream)
    monkeypatch.setattr(kimi_model, "kimi_shared_expert_stream", lambda: stream)
    monkeypatch.setattr(kimi_model.envs, "VLLM_DISABLE_SHARED_EXPERTS_STREAM", False)
    monkeypatch.setattr(
        kimi_model.envs, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256
    )

    hidden_states = torch.ones(4, 8)
    output = moe(hidden_states)

    assert captured["stream"] is stream
    assert captured["events"] == moe._shared_expert_events
    assert captured["kwargs"] == {"launch_aux_first": not routed_first}
    torch.testing.assert_close(output, hidden_states * 2 + 6)


def test_kimi_fused_moe_tail_matches_two_projection_sum():
    from vllm.models.kimi_k3.nvidia.model import KimiFusedMoETail

    shared_size = 6
    routed_size = 4
    hidden_size = 8
    tail = KimiFusedMoETail.__new__(KimiFusedMoETail)
    torch.nn.Module.__init__(tail)
    tail.norm = None

    class Projection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, hidden_states):
            return hidden_states @ self.weight.t(), None

    shared_weight = torch.randn(hidden_size, shared_size)
    routed_weight = torch.randn(hidden_size, routed_size)
    tail.proj = Projection(torch.cat((shared_weight, routed_weight), dim=1))

    shared_activations = torch.randn(3, shared_size)
    routed_hidden_states = torch.randn(3, routed_size)
    expected = (
        shared_activations @ shared_weight.t()
        + routed_hidden_states @ routed_weight.t()
    )

    actual = tail(routed_hidden_states, shared_activations)

    torch.testing.assert_close(actual, expected)


def test_kimi_fused_moe_tail_loader_packs_input_shards():
    from vllm.models.kimi_k3.nvidia.model import (
        load_kimi_fused_moe_tail_shard,
    )

    packed = torch.zeros(8, 10)
    shared = torch.randn(8, 6)
    routed = torch.randn(8, 4)

    load_kimi_fused_moe_tail_shard(packed, shared, "shared")
    load_kimi_fused_moe_tail_shard(packed, routed, "routed")

    torch.testing.assert_close(packed, torch.cat((shared, routed), dim=1))


def test_kimi_fused_moe_input_matches_two_projections():
    from vllm.models.kimi_k3.nvidia.model import KimiFusedMoEInputProjection

    hidden_size = 8
    routed_size = 4
    shared_size = 6
    projection = KimiFusedMoEInputProjection.__new__(
        KimiFusedMoEInputProjection
    )
    torch.nn.Module.__init__(projection)
    projection.routed_hidden_size = routed_size
    projection.shared_gate_up_size = 2 * shared_size

    class Projection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, hidden_states):
            return hidden_states @ self.weight.t(), None

    routed_weight = torch.randn(routed_size, hidden_size)
    gate_weight = torch.randn(shared_size, hidden_size)
    up_weight = torch.randn(shared_size, hidden_size)
    projection.proj = Projection(
        torch.cat((routed_weight, gate_weight, up_weight), dim=0)
    )
    hidden_states = torch.randn(3, hidden_size)

    routed, gate_up = projection(hidden_states)

    torch.testing.assert_close(routed, hidden_states @ routed_weight.t())
    torch.testing.assert_close(
        gate_up,
        torch.cat(
            (hidden_states @ gate_weight.t(), hidden_states @ up_weight.t()),
            dim=-1,
        ),
    )


def test_kimi_fused_moe_input_loader_packs_output_shards():
    from vllm.models.kimi_k3.nvidia.model import (
        load_kimi_fused_moe_input_shard,
    )

    routed = torch.randn(4, 8)
    gate = torch.randn(6, 8)
    up = torch.randn(6, 8)
    packed = torch.zeros(16, 8)

    load_kimi_fused_moe_input_shard(packed, routed, "routed")
    load_kimi_fused_moe_input_shard(packed, gate, "gate")
    load_kimi_fused_moe_input_shard(packed, up, "up")

    torch.testing.assert_close(packed, torch.cat((routed, gate, up), dim=0))


def test_kimi_shared_output_projection_matches_activation_and_linear():
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.models.kimi_k3.nvidia.model import KimiSharedOutputProjection

    module = KimiSharedOutputProjection.__new__(KimiSharedOutputProjection)
    torch.nn.Module.__init__(module)
    module.act_fn = SituAndMul(beta=1.25, linear_beta=0.5)

    class Projection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, hidden_states):
            return hidden_states @ self.weight.t(), None

    weight = torch.randn(8, 6)
    module.down_proj = Projection(weight)
    gate_up = torch.randn(4, 12)

    expected = module.act_fn(gate_up) @ weight.t()

    torch.testing.assert_close(module(gate_up), expected)


def test_kimi_fused_shared_expert_packs_gate_up_blocks_and_matches_reference():
    from vllm.models.kimi_k3.nvidia.model import KimiFusedSharedExpert

    hidden_size = 8
    intermediate_size = 64
    module = KimiFusedSharedExpert.__new__(KimiFusedSharedExpert)
    torch.nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.intermediate_size = intermediate_size
    module.situ_beta = 4.0
    module.situ_linear_beta = 25.0
    packed = torch.nn.Parameter(torch.empty(2 * intermediate_size, hidden_size))
    module.gate_up_proj = SimpleNamespace(weight=packed)
    down_weight = torch.randn(hidden_size, intermediate_size)
    module.down_proj = SimpleNamespace(weight=down_weight)

    gate_weight = torch.randn(intermediate_size, hidden_size)
    up_weight = torch.randn(intermediate_size, hidden_size)
    module._load_gate_up_weight(packed, gate_weight, 0)
    module._load_gate_up_weight(packed, up_weight, 1)

    fc1_weight, fc2_weight = module._weights()
    assert fc1_weight.shape == (2 * intermediate_size, hidden_size)
    assert fc1_weight.is_contiguous()
    assert fc2_weight.shape == (hidden_size, intermediate_size)
    assert fc2_weight.is_contiguous()

    hidden_states = torch.randn(3, hidden_size)
    gate = hidden_states @ gate_weight.t()
    up = hidden_states @ up_weight.t()
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    expected = (gate * up) @ down_weight.t()

    torch.testing.assert_close(module.forward_native(hidden_states), expected)


def test_kimi_fused_shared_expert_uses_direct_deepgemm_input():
    from vllm.models.kimi_k3.nvidia.model import _KimiDeepGemmSharedSession

    session = _KimiDeepGemmSharedSession.__new__(_KimiDeepGemmSharedSession)
    session.buffer = object()
    session.dummy_l1 = torch.empty(1)
    session.dummy_l2 = torch.empty(1)
    session.output = torch.empty(2, 8)
    session.num_sms = 28
    captured = {}

    def bf16_mega_moe(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    session.deep_gemm = SimpleNamespace(
        bf16_mega_moe=bf16_mega_moe,
    )
    hidden_states = torch.randn(2, 8)
    gate_up_weight = torch.randn(12, 8)
    down_weight = torch.randn(8, 6)

    output = session.run(
        hidden_states,
        gate_up_weight,
        down_weight,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    assert output is session.output
    assert captured["args"][0] is session.output
    assert captured["args"][1] is session.dummy_l1
    assert captured["args"][2] is session.dummy_l2
    assert captured["args"][3] is session.buffer
    assert captured["kwargs"]["shared_l1_acts"] is hidden_states
    assert captured["kwargs"]["shared_l1_weights"] is gate_up_weight
    assert captured["kwargs"]["shared_l2_weights"] is down_weight
    assert captured["kwargs"]["num_sms"] == 28


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="DeepGEMM BF16 MegaMoE requires SM100",
)
def test_kimi_fused_shared_expert_deepgemm_matches_native(default_vllm_config):
    from vllm.models.kimi_k3.nvidia.model import KimiFusedSharedExpert

    torch.manual_seed(0)
    module = KimiFusedSharedExpert(
        hidden_size=256,
        intermediate_size=256,
        max_num_tokens=64,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
    ).to(device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda") / 32
    up = torch.randn_like(gate) / 32
    down = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda") / 32
    module._load_gate_up_weight(module.gate_up_proj.weight, gate, 0)
    module._load_gate_up_weight(module.gate_up_proj.weight, up, 1)
    module.down_proj.weight.data.copy_(down)
    hidden_states = torch.randn(16, 256, dtype=torch.bfloat16, device="cuda")

    expected = module.forward_native(hidden_states)
    actual = module.forward_cuda(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)


@pytest.mark.parametrize("with_shared_expert", [False, True])
def test_kimi_flashinfer_mega_moe_forward_preserves_tensor_contract(
    monkeypatch, with_shared_expert
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
    )

    assert output.data_ptr() != hidden_states.data_ptr()
    assert torch.equal(output, hidden_states)
    assert captured["tensors"].hidden_states is hidden_states
    assert captured["tensors"].topk_ids is topk_ids
    assert captured["tensors"].topk_weights is topk_weights
    assert captured["tensors"].fc1_alpha is experts._flashinfer_fc1_alpha
    assert captured["tensors"].fc2_alpha is experts._flashinfer_fc2_alpha
    assert captured["tensors"].fc1_norm_const is experts._flashinfer_fc1_norm_const
    assert captured["tensors"].mega_shared_inputs is shared_inputs


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
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    experts = kimi_model.KimiFusedSharedExpert.__new__(
        kimi_model.KimiFusedSharedExpert
    )
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
        def __init__(self, *, num_tokens, **kwargs):
            capacities.append(num_tokens)

    monkeypatch.setattr(
        kimi_model, "_KimiFlashInferNvfp4SharedSession", FakeSession
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=148),
    )

    session_129 = experts._nvfp4_session(torch.empty(129, 128))
    session_200 = experts._nvfp4_session(torch.empty(200, 128))
    session_257 = experts._nvfp4_session(torch.empty(257, 128))

    assert session_129 is session_200
    assert session_257 is not session_200
    assert capacities == [256, 512]


def test_kimi_nvfp4_shared_expert_caches_thunks_per_stream(monkeypatch):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import nvfp4
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    session = kimi_model._KimiFlashInferNvfp4SharedSession.__new__(
        kimi_model._KimiFlashInferNvfp4SharedSession
    )
    session.active_num_tokens = 2
    session.buffer = SimpleNamespace(output_activation=torch.empty(4, 8))
    session._thunks = {}
    transformed_weights = (
        (torch.empty(1), torch.empty(1)),
        (torch.empty(1), torch.empty(1)),
    )
    stream = [11]
    created_on = []
    launched_on = []

    def make_thunk(*args):
        created_on.append(stream[0])

        def launch():
            launched_on.append(stream[0])

        return launch

    monkeypatch.setattr(nvfp4, "nvfp4_mega_launch_thunk", make_thunk)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=stream[0]),
    )

    session.run(transformed_weights)
    session.run(transformed_weights)
    stream[0] = 22
    session.run(transformed_weights)

    assert created_on == [11, 22]
    assert launched_on == [11, 11, 22]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, :32] = 0
    hidden_states[1, 32:64] = 1.0e-6
    hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_weights = topk_weights.clone()

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_masks_padding():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    is_padding = torch.tensor(
        [False, True, False, False, True, False, True],
        device=device,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_idx[is_padding] = -1
    ref_topk_weights = topk_weights.clone()
    ref_topk_weights[is_padding] = 0.0

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )
