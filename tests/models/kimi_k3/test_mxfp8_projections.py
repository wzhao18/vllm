# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online MXFP8 for the Kimi K3.1 BF16 projections: target preset, kernels vs
references, the packed fused-DSA GEMM and the specialized-decode guards."""

import pytest
import torch
import torch.nn.functional as F

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.models.kimi_k3.nvidia.mxfp8_pack import unswizzle_mxfp8_scale
from vllm.models.kimi_k3.quant_presets import K31_DEP_MXFP8_TARGETS
from vllm.platforms import current_platform

PACKED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
}
L = "language_model.model.layers"
TARGETED = [
    f"{L}.1.self_attn.in_proj_qkvgfab",
    f"{L}.1.self_attn.o_proj",
    f"{L}.4.self_attn.fused_qkv_a_proj",
    f"{L}.4.self_attn.q_b_proj",
    f"{L}.4.self_attn.g_proj",
    f"{L}.4.self_attn.o_proj",
    f"{L}.4.self_attn.dsa_indexer.wq_index",
    f"{L}.4.self_attn.dsa_indexer.wkv_index",
    f"{L}.4.block_sparse_moe.shared_experts.gate_up_proj",
    f"{L}.4.block_sparse_moe.shared_experts.down_proj",
    f"{L}.4.block_sparse_moe.routed_expert_down_proj",
    f"{L}.4.block_sparse_moe.routed_expert_up_proj",
    f"{L}.0.mlp.gate_up_proj",
    f"{L}.0.mlp.down_proj",
]
UNTARGETED = [
    f"{L}.4.self_attn.kv_b_proj",
    f"{L}.1.self_attn.f_b_proj",
    f"{L}.4.block_sparse_moe.gate",
    f"{L}.4.self_attention_res_proj",
    f"{L}.4.mlp_res_proj",
    "language_model.model.output_attn_res_proj",
    "language_model.lm_head",
]

needs_mxfp8_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="MXFP8 GEMMs need SM100+",
)


def _online(targets=K31_DEP_MXFP8_TARGETS) -> OnlineQuantizationConfig:
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=targets))
    config.packed_modules_mapping = PACKED
    return config


@pytest.mark.parametrize("name", TARGETED)
def test_dep_preset_targets_bf16_projections(name):
    resolved = _online().resolve_quant_method_cls(LinearBase, name)
    assert resolved is not None and resolved[1] == "mxfp8"


@pytest.mark.parametrize("name", UNTARGETED)
def test_dep_preset_leaves_other_layers(name):
    assert _online().resolve_quant_method_cls(LinearBase, name) is None


def test_targets_compose_with_activation_only_moe_override():
    args = QuantizationConfigArgs(
        moe={"activation": "mxfp8"}, targets=K31_DEP_MXFP8_TARGETS
    )
    assert args.moe is not None and args.moe.weight is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantizationConfigArgs(moe="mxfp8", targets=K31_DEP_MXFP8_TARGETS)
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantizationConfigArgs(linear="mxfp8", targets=K31_DEP_MXFP8_TARGETS)


@pytest.mark.parametrize("n,k", [(576, 7168), (160, 7168), (2272, 7168), (1, 128)])
def test_unswizzle_inverts_swizzle(n, k):
    scales = torch.randint(0, 255, (n, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8)
    swizzled = swizzle_mxfp8_scale(scales, M=n, K=k)
    assert torch.equal(unswizzle_mxfp8_scale(swizzled, n, k), scales)


def test_specialized_decode_rejects_online_linear_quant():
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
        online_linear_quantization,
    )

    assert not online_linear_quantization(None)
    assert not online_linear_quantization(
        QuantizationConfigArgs(moe={"activation": "mxfp8"})
    )
    assert online_linear_quantization(QuantizationConfigArgs(linear="mxfp8"))
    assert online_linear_quantization(
        QuantizationConfigArgs(
            moe={"activation": "mxfp8"}, targets=K31_DEP_MXFP8_TARGETS
        )
    )


# ---- GPU: kernels vs references -------------------------------------------


def _mxfp8_kernels():
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
        FlashInferTrtllmMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.pytorch import (
        TorchMxFp8LinearKernel,
    )

    return [
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
        FlashInferTrtllmMxfp8LinearKernel,
        TorchMxFp8LinearKernel,
    ]


def _linear(prefix, n, k, quant_config, weight, kernel_cls=None):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    layer = ReplicatedLinear(
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
    )
    if kernel_cls is not None:
        from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
            Mxfp8LinearLayerConfig,
        )

        layer.quant_method.kernel = kernel_cls(Mxfp8LinearLayerConfig((n, k)))
    if layer.weight.is_meta:
        layer = layer.to_empty(device="cuda")
    layer = layer.to("cuda")
    layer.weight.data.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _dequant(x, scales):
    k = x.shape[-1]
    blocks = x.float().view(*x.shape[:-1], k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (blocks * torch.exp2(scales.float() - 127.0).unsqueeze(-1)).view(x.shape)


# (N, K) of the DEP16 (TP1) and TP16 projections, rows cut to keep the test fast
# where the full N only repeats tiles.
SHAPES = [
    (2272, 7168),  # [q_a | kv_a | wk | weights_proj]
    (4096, 1536),  # wq_index (q_b | wq_index: 22528 x 1536)
    (3216, 7168),  # KDA in_proj_qkvgfab at TP16 (padded)
    (7168, 768),  # o_proj at TP16
    (7168, 3584),  # latent up
    (3584, 7168),  # latent down
]


@needs_mxfp8_gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 7, 64, 160, 1000])
@pytest.mark.parametrize("kernel_idx", range(4))
def test_online_mxfp8_linear_matches_references(
    default_vllm_config, dist_init, n, k, m, kernel_idx
):
    """Each SM10x MXFP8 kernel equals the FP32 product of the dequantized MXFP8
    operands (the same stored inputs) and stays within MXFP8 error of BF16."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    kernel_cls = _mxfp8_kernels()[kernel_idx]
    ok, why = kernel_cls.is_supported()
    if ok:
        from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
            Mxfp8LinearLayerConfig,
        )

        ok, why = kernel_cls.can_implement(Mxfp8LinearLayerConfig((n, k)))
    if ok and kernel_cls.__name__.startswith("FlashInferTrtllm") and k % 256:
        ok, why = False, "TRTLLM MXFP8 needs K % 256 == 0"
    if not ok:
        pytest.skip(f"{kernel_cls.__name__}: {why}")
    torch.manual_seed(n + k + m)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    layer = _linear("model.layers.4.self_attn.o_proj", n, k, _online(), weight,
                    kernel_cls)  # fmt: skip
    out = layer(x)[0].float()

    wq, ws = mxfp8_e4m3_quantize(weight)
    xq, xs = mxfp8_e4m3_quantize(x)
    exact = _dequant(xq, xs.view(m, -1)) @ _dequant(wq, ws.view(n, -1)).t()
    # FP32 accumulation of exact FP8 products, BF16 output rounding.
    torch.testing.assert_close(out, exact, rtol=1.6e-2, atol=1e-2 * exact.abs().max())
    bf16 = F.linear(x, weight).float()
    rel = ((out - bf16).norm() / bf16.norm()).item()
    assert rel < 0.06, rel  # two E4M3 roundings: ~2 x 2^-4 / sqrt(3)


@needs_mxfp8_gpu
@pytest.mark.parametrize("m", [1, 4, 32, 160, 1000])
def test_packed_mxfp8_matches_per_linear(default_vllm_config, dist_init, m):
    """The packed [q_a | kv_a | wk | weights_proj] and [q_b | wq_index] MXFP8
    GEMMs equal the per-linear MXFP8 GEMMs exactly; the linears keep working on
    their views."""
    from vllm.models.kimi_k3.nvidia.mxfp8_pack import PackedMxfp8Linear, can_pack_mxfp8

    torch.manual_seed(m)
    online = _online()
    for k, parts in ((7168, (2112, 160)), (1536, (1152, 4096))):
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                   for n in parts]  # fmt: skip
        prefixes = ("model.layers.4.self_attn.fused_qkv_a_proj",
                    "model.layers.4.self_attn.dsa_indexer.wkv_index")  # fmt: skip
        linears = [_linear(p, n, k, online, w)
                   for p, n, w in zip(prefixes, parts, weights)]  # fmt: skip
        assert can_pack_mxfp8(linears)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        separate = torch.cat([linear(x)[0] for linear in linears], dim=-1)
        packed = PackedMxfp8Linear(linears)
        assert torch.equal(packed(x), separate)
        after = torch.cat([linear(x)[0] for linear in linears], dim=-1)
        assert torch.equal(after, separate)
        # The linears' weights now live in the packed weight.
        base = packed.holder.weight.untyped_storage().data_ptr()
        assert all(
            linear.weight.untyped_storage().data_ptr() == base for linear in linears
        )


@needs_mxfp8_gpu
def test_bf16_weight_refuses_quantized_linear(default_vllm_config, dist_init):
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        bf16_weight,
    )

    weight = torch.randn(256, 7168, device="cuda", dtype=torch.bfloat16)
    quantized = _linear("model.layers.4.self_attn.o_proj", 256, 7168, _online(),
                        weight)  # fmt: skip
    with pytest.raises(ValueError, match="needs BF16 projections"):
        bf16_weight(quantized)
    plain = _linear("model.layers.4.self_attn.kv_b_proj", 256, 7168, _online(),
                    weight)  # fmt: skip
    assert bf16_weight(plain) is plain.weight
