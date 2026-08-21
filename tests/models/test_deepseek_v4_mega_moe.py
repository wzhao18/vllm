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
    *, max_num_batched_tokens: int = 64, tensor_parallel_size: int = 1
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
        load_config=SimpleNamespace(load_format="auto"),
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
    assert kernel.knobs == {
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 512,
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


def test_kimi_flashinfer_mega_moe_forward_preserves_tensor_contract(monkeypatch):
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
