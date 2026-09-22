# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def nvfp4_experts():
    from vllm.models.kimi_k3.nvidia.fi_moe import KimiK3FlashInferMegaMoEExperts

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=65),
        compilation_config=SimpleNamespace(static_forward_context={}),
        load_config=SimpleNamespace(load_format="auto"),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            tensor_parallel_size=8,
        ),
    )
    return KimiK3FlashInferMegaMoEExperts(
        config,
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


@pytest.mark.parametrize("expert_id", [1, 2, 3])
def test_nvfp4_checkpoint_metadata_is_not_loaded_as_weights(nvfp4_experts, expert_id):
    """Overlapping checkpoint suffixes must reach the correct local metadata shard."""
    from vllm.models.kimi_k3.nvidia.model import (
        make_kimi_k3_mega_moe_expert_params_mapping,
    )

    experts = nvfp4_experts
    mapping = make_kimi_k3_mega_moe_expert_params_mapping(4, nvfp4=True)
    values = {
        "weight_scale_2": 2.0,
        "input_scale": 3.0,
        "weight_scale": 4.0,
        "weight": 5,
    }
    for shard in ("w1", "w3", "w2"):
        for suffix, value in values.items():
            checkpoint_name = f"layers.0.mlp.experts.{expert_id}.{shard}.{suffix}"
            param_name, _, mapped_id, mapped_shard = next(
                entry for entry in mapping if entry[1] in checkpoint_name
            )
            expected_prefix = "w2" if shard == "w2" else "w13"
            assert param_name == f"experts.{expected_prefix}_{suffix}"
            param = getattr(experts, param_name.removeprefix("experts."))
            shape = param.shape[1:]
            if shard != "w2":
                shape = shape[1:] if len(shape) == 1 else (shape[0] // 2, *shape[1:])
            loaded = torch.full(shape, value, dtype=param.dtype)
            before = param.detach().clone()
            accepted = experts.weight_loader(
                param, loaded, param_name, mapped_shard, mapped_id, return_success=True
            )
            assert accepted == (expert_id >= 2)
            if expert_id < 2:
                torch.testing.assert_close(param.float(), before.float())
            else:
                local = expert_id - 2
                result = param[local]
                if shard in ("w1", "w3"):
                    half = 0 if shard == "w1" else 1
                    result = result[half] if result.ndim == 1 else result.chunk(2)[half]
                torch.testing.assert_close(result.float(), loaded.float())


def test_nvfp4_finalization_preserves_checkpoint_scales(nvfp4_experts, monkeypatch):
    """Fold global activation scales into epilogues without changing packed weights."""
    import sys
    from dataclasses import dataclass
    from types import ModuleType

    from vllm.models.kimi_k3.nvidia import fi_moe

    experts = nvfp4_experts
    experts.w13_weight_scale_2.data.copy_(torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
    experts.w2_weight_scale_2.data.copy_(torch.tensor([5.0, 7.0]))
    experts.w13_input_scale.data.fill_(11.0)
    experts.w2_input_scale.data.fill_(13.0)
    original_weights = experts.w13_weight
    captured = {}

    @dataclass(init=False)
    class Config:
        situ_beta: float

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Layer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        def _ensure_workspace(self):
            pass

    fake = ModuleType("flashinfer.moe_ep")
    for name, value in {
        "BootstrapConfig": SimpleNamespace,
        "FleetParams": SimpleNamespace,
        "MegaConfig": SimpleNamespace,
        "MoEWeightPack": SimpleNamespace,
        "MoEEpLayer": Layer,
        "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig": Config,
    }.items():
        setattr(fake, name, value)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake)
    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        fi_moe,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    experts.finalize_weights()
    layer = experts._flashinfer_layer
    experts.finalize_weights()
    assert experts._flashinfer_layer is layer
    torch.testing.assert_close(captured["weights"].w13, original_weights)
    torch.testing.assert_close(experts._fc1_alpha, torch.tensor([22.0, 33.0]))
    torch.testing.assert_close(experts._fc2_alpha, torch.tensor([65.0, 91.0]))
    torch.testing.assert_close(experts._fc1_norm_const, torch.full((2,), 1 / 13.0))
    config = captured["backend"].megakernel
    assert config.input_norm_const == pytest.approx(1 / 11.0)
    assert (config.activation, config.situ_beta, config.situ_linear_beta) == (
        "situ",
        4,
        25,
    )
    assert captured["fleet_params"].max_tokens_per_rank == 9
    assert experts.w13_weight is None
