# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP (non-SP) specialized decode's build-time format checks
(``runtime.providers.mxfp4_experts`` / ``check_bf16_projections``, the drafter's
``runtime.draft.check_draft_formats``): its experts kernels read MXFP4 routed experts
(``kind::mxf8f6f4`` over UE8M0 scales per 32 elements, the FlashInfer TRT-LLM MXFP4 x
MXFP8 layout) and its projections BF16 [N, K] weights. Any other format (an NVFP4
checkpoint's experts: E4M3 scales per 16 elements plus global scales; ModelOpt MXFP8
projections) must raise at build, naming it, instead of being programmed as if it
were MXFP4 / BF16. CPU only: modules are built without a model."""

import pytest
import torch
from torch import nn

pytest.importorskip("cutlass")

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
)
from vllm.model_executor.layers.linear import (  # noqa: E402
    LinearMethodBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E402, E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.mxfp4 import (  # noqa: E402
    Mxfp4MoEMethod,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    draft,
    providers,
)

E, N, K = 4, 64, 3584  # experts, w13 rows, hidden (the TP experts' latent width)


def method(cls, backend=None):
    """A quant method instance without its constructor (no config / platform)."""
    m = cls.__new__(cls)
    if backend is not None:
        m.mxfp4_backend = backend
    return m


def routed(quant_method, block, scale_dtype, extra=()):
    """A routed-experts module as stock leaves it after loading: packed E2M1
    w13 [E, N, K / 2] / w2 [E, K, N / 2] and one scale byte per ``block`` elements."""
    layer = nn.Module()
    layer.quant_method = quant_method
    layer.layer_name = "model.layers.1.block_sparse_moe.experts"
    shapes = {"w13": (E, N, K), "w2": (E, K, N)}
    for name, (e, rows, cols) in shapes.items():
        weight = torch.zeros(e, rows, cols // 2, dtype=torch.uint8)
        scale = torch.zeros(e, rows, cols // block, dtype=torch.uint8)
        setattr(layer, f"{name}_weight", nn.Parameter(weight, requires_grad=False))
        setattr(
            layer,
            f"{name}_weight_scale",
            nn.Parameter(scale.view(scale_dtype), requires_grad=False),
        )
    for name in extra:
        setattr(layer, name, nn.Parameter(torch.ones(E), requires_grad=False))
    return layer


TRTLLM = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e4m3fn])
def test_mxfp4_experts_production(scale_dtype):
    """Stock's MXFP4 experts (``Mxfp4MoEMethod`` on the TRT-LLM MXFP4 x MXFP8 backend,
    which keeps the UE8M0 bytes in a float8_e4m3fn view) pass, in PersistentExperts'
    argument order."""
    layer = routed(method(Mxfp4MoEMethod, TRTLLM), 32, scale_dtype)
    w13, s13, w2, s2 = providers.mxfp4_experts(layer)
    assert w13 is layer.w13_weight and s13 is layer.w13_weight_scale
    assert w2 is layer.w2_weight and s2 is layer.w2_weight_scale


def test_mxfp4_experts_refuses_nvfp4():
    """An NVFP4 checkpoint's experts (compressed-tensors W4A4 NVFP4: E4M3 scales per
    16 elements, global scales) raise, naming the method."""
    nvfp4 = method(CompressedTensorsW4A4Nvfp4MoEMethod)
    globals_ = ("w13_weight_global_scale", "w2_weight_global_scale")
    layer = routed(nvfp4, 16, torch.float8_e4m3fn, globals_)
    with pytest.raises(ValueError, match="CompressedTensorsW4A4Nvfp4MoEMethod"):
        providers.mxfp4_experts(layer)


@pytest.mark.parametrize(
    "block, extra, cause",
    [
        (16, (), "one scale per 32 elements"),  # 16-element blocks
        (32, ("w13_weight_scale_2",), "global scale w13_weight_scale_2"),
        (32, ("w13_input_scale",), "global scale w13_input_scale"),
        (32, ("w13_bias",), "expert bias w13_bias"),
    ],
)
def test_mxfp4_experts_refuses_layout(block, extra, cause):
    """Under the MXFP4 method, anything but one UE8M0 byte per 32 elements and no
    global scale / bias raises."""
    layer = routed(method(Mxfp4MoEMethod, TRTLLM), block, torch.uint8, extra)
    with pytest.raises(ValueError, match=cause):
        providers.mxfp4_experts(layer)


@pytest.mark.parametrize(
    "backend",
    [
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        Mxfp4MoeBackend.MARLIN,
    ],
)
def test_mxfp4_experts_refuses_backend(backend):
    """MXFP4 experts on another backend (another weight / scale layout, or BF16
    activations) raise, naming the backend."""
    layer = routed(method(Mxfp4MoEMethod, backend), 32, torch.uint8)
    with pytest.raises(ValueError, match=backend.value):
        providers.mxfp4_experts(layer)


class FakeQuantized(LinearMethodBase):
    """A checkpoint-quantized linear method (ModelOpt MXFP8-like)."""

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        raise NotImplementedError


def linear(prefix, weight, quant_method):
    m = ReplicatedLinear.__new__(ReplicatedLinear)
    nn.Module.__init__(m)
    m.weight = nn.Parameter(weight, requires_grad=False)
    m.quant_method, m.prefix = quant_method, prefix
    return m


def decoder(o_proj_method, o_proj_dtype=torch.bfloat16, kv_b_method=None):
    """A decoder layer with BF16 projections except (optionally) o_proj / kv_b_proj."""
    plain = method(UnquantizedLinearMethod)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.in_proj_qkvgfab = linear(
        "layers.1.self_attn.in_proj_qkvgfab", torch.zeros(8, 16, dtype=torch.bfloat16),
        plain,
    )  # fmt: skip
    layer.self_attn.o_proj = linear(
        "layers.1.self_attn.o_proj", torch.zeros(16, 8, dtype=o_proj_dtype),
        o_proj_method,
    )  # fmt: skip
    layer.self_attn.kv_b_proj = linear(
        "layers.1.self_attn.kv_b_proj", torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
        kv_b_method or FakeQuantized(),
    )  # fmt: skip
    layer.self_attn.conv1d = linear(  # FP32 in stock (KimiDeltaAttention)
        "layers.1.self_attn.conv1d", torch.zeros(8, 1, 4, dtype=torch.float32), plain
    )
    layer.mlp = nn.Module()
    layer.mlp.shared_experts = nn.Module()
    layer.mlp.shared_experts.down_proj = linear(
        "layers.1.mlp.shared_experts.down_proj",
        torch.zeros(16, 8, dtype=torch.bfloat16), plain,
    )  # fmt: skip
    return layer


def test_bf16_projections_pass():
    """BF16 projections pass; ``kv_b_proj`` (absorbed into W_UK / W_UV by stock, never
    read) may be anything."""
    providers.check_bf16_projections(decoder(method(UnquantizedLinearMethod)))


def test_bf16_projections_refuse_quantized():
    """A checkpoint-quantized projection (FP8 weight) raises, naming method and
    layer."""
    layer = decoder(FakeQuantized(), torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"FakeQuantized.*self_attn\.o_proj"):
        providers.check_bf16_projections(layer)


def test_bf16_projections_refuse_non_bf16():
    """An unquantized projection in another dtype raises too."""
    layer = decoder(method(UnquantizedLinearMethod), torch.float16)
    with pytest.raises(ValueError, match="float16"):
        providers.check_bf16_projections(layer)


def test_conv1d_keeps_stock_dtype():
    """The KDA conv1d taps are FP32 in stock (pass); quantized or BF16 they raise."""
    plain = method(UnquantizedLinearMethod)
    layer = decoder(plain)
    layer.self_attn.conv1d.weight.data = layer.self_attn.conv1d.weight.data.bfloat16()
    with pytest.raises(ValueError, match="unquantized torch.float32 conv1d"):
        providers.check_bf16_projections(layer)
    layer.self_attn.conv1d.weight.data = layer.self_attn.conv1d.weight.data.float()
    layer.self_attn.conv1d.quant_method = FakeQuantized()
    with pytest.raises(ValueError, match="FakeQuantized"):
        providers.check_bf16_projections(layer)


def drafter(dtype):
    model = nn.Module()
    model.model = nn.Module()
    head = model.model.markov_head = nn.Module()
    head.markov_w1 = nn.Embedding(32, 8, dtype=dtype)
    head.markov_w2 = nn.Embedding(32, 8, dtype=dtype)
    return model


def test_draft_formats():
    """The drafter's Markov head must be BF16 (the fused Markov kernels' tables)."""
    draft.check_draft_formats(drafter(torch.bfloat16))
    with pytest.raises(ValueError, match="markov_w1 torch.float32"):
        draft.check_draft_formats(drafter(torch.float32))


def test_context_projection_refuses_quantized():
    """``PreparedContext`` reads the context projection through ``bf16_weight``."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.context import (
        PreparedContext,
    )

    model = nn.Module()
    model.context_proj = linear(
        "model.context_proj", torch.zeros(16, 8, dtype=torch.float8_e4m3fn),
        FakeQuantized(),
    )  # fmt: skip
    with pytest.raises(ValueError, match="BF16 projections"):
        PreparedContext(model, None, 1, 0, ())


def test_mxfp4_experts_accepts_production_layer():
    """No false refusal: the stock routed-experts layer production builds (``Mxfp4Config``
    -> ``Mxfp4MoEMethod``, moe-backend auto -> FLASHINFER_TRTLLM_MXFP4_MXFP8, loaded and
    processed as ``test_specialized_decode_wide_experts_gpu.production_experts``
    builds it) passes, and the check hands back the tensors runtime.decode passes to
    ``PersistentExperts``."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM10x GPU (FlashInfer TRT-LLM MXFP4 experts)")
    from tests.models.kimi_k3 import test_specialized_decode_wide_experts_gpu as wide

    dev = torch.device("cuda", torch.cuda.current_device())
    case = wide.production_experts(wide.logical_weights(dev), dev)
    layer = case["layer"]
    checked = providers.mxfp4_experts(layer)
    for got, want in zip(checked, case["weights"]):
        assert got.data_ptr() == want.data_ptr() and got.numel() == want.numel()
