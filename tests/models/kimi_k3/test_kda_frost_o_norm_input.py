# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level regression: the KDA prefill core output must reach the o_norm.

``KimiK3DeltaAttention.forward`` in the TP > 1 prefill configuration (fused GEMM-AR
set on a BF16 o_proj, so the MXFP8 o_norm is off and ``o_mxfp8`` is None; FROST /
persistent core on a pure prefill step) with every buffer that ``kda.py`` allocates with
``torch.empty`` poisoned (NaN, or a finite value), against the stock FlashKDA core
on the same layer. On int53 the FROST / persistent core left ``core_attn_out``
unwritten and the in-place gated o_norm normalized that uninitialized memory:
NaN / non-reproducible TP8 + DCP8 prefill steps (BUG-DCP8-1).

The MXFP8 o_norm route is covered as well: DEP / TP1 at ``_forward`` level, and
TP > 1 with an MXFP8 o_proj, whose fused GEMM-RS/AR takes the MXFP8 o_norm output.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="FlashKDA / FROST / persistent KDA prefill target SM10x",
)

D = 128
H = 12  # local heads of the 96-head KDA at TP8
P = H * D
HIDDEN = 1024
LENS = [1029, 300, 37]
SLOTS = 8
SLOT_IDS = [5, 2, 7]
HAS_INIT = [True, False, True]


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


class _PoisonedTorch:
    """kda.py's ``torch`` with ``empty`` returning poisoned floating buffers."""

    def __init__(self, value: float):
        self._value = value

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        t = torch.empty(*args, **kwargs)
        if t.is_floating_point() and t.element_size() >= 2:
            t.fill_(self._value)
        return t


class _FakeGemmRsAr:
    """The TP fused GEMM-RS/AR o_proj on one GPU (all-reduce of one rank)."""

    def __init__(self):
        self.calls = 0

    def should_run(self, x: torch.Tensor) -> bool:
        return True

    def apply(self, x, linear, residual=None, residual_row0=0, copy_out=True):
        self.calls += 1
        return linear(x)[0]


def _inputs(seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

    T = sum(LENS)
    width = 4 * P + D + H
    pad = (-width) % 8  # the real in_proj pads its rows to 16 B
    return SimpleNamespace(
        T=T,
        hidden=randn(T, HIDDEN, dtype=torch.bfloat16),
        w_in=randn(width + pad, HIDDEN, dtype=torch.bfloat16, scale=HIDDEN**-0.5),
        pad=pad,
        w_fb=randn(P, D, dtype=torch.bfloat16, scale=D**-0.5),
        w_o=randn(HIDDEN, P, dtype=torch.bfloat16, scale=P**-0.5),
        conv_w=randn(3 * P, 1, 4, scale=0.5),
        A_log=randn(H, scale=0.5),
        dt_bias=randn(P, scale=0.1),
        norm_w=1 + randn(D, scale=0.1),
        conv_state=randn(SLOTS, 3 * P, 3, dtype=torch.bfloat16, scale=0.5),
        state=randn(SLOTS, H, D, D, scale=0.05),
    )


def _layer(x, core: str, gemm_rs_ar):
    """The KimiK3DeltaAttention fields forward() / _forward() read, TP-sharded
    shapes; ``core`` = "persistent" | "frost" | "flashkda" (stock)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    import vllm._flashkda_C  # noqa: F401

    with set_current_vllm_config(VllmConfig()):  # o_norm picks forward_cuda here
        norm = FusedRMSNormGated(D, eps=1e-5, activation="sigmoid").cuda()
    with torch.no_grad():
        norm.weight.copy_(x.norm_w)
    conv_state = x.conv_state.clone()
    if not is_conv_state_dim_first():
        # SD storage: _forward transposes it back to (lines, dim, width - 1).
        conv_state = conv_state.transpose(1, 2).contiguous()
    N = len(LENS)
    ws = torch.ops._flashkda_C.get_workspace_size(x.T, H, N)
    layer = SimpleNamespace(
        prefix="test.kda",
        hidden_size=HIDDEN,
        head_dim=D,
        local_num_heads=H,
        local_projection_size=P,
        in_proj_padding=x.pad,
        _projection_events=None,
        _projection_aux_stream=None,
        _projection_overlap_max_tokens=0,
        _prefill_conv_in_proj=None,
        in_proj_qkvgfab=lambda h: (h @ x.w_in.t(),),
        f_b_proj=lambda f: (f @ x.w_fb.t(),),
        o_proj=lambda a: (a @ x.w_o.t(),),
        gemm_rs_ar=gemm_rs_ar,
        _o_norm_mxfp8_ok=None,
        o_norm=norm,
        kv_cache=(conv_state, x.state.clone()),
        conv1d=SimpleNamespace(weight=x.conv_w, bias=None),
        A_log=x.A_log,
        dt_bias=x.dt_bias,
        gate_lower_bound=-5.0,
        kda_spec_decode_backend="triton",
        kda_decode_backend="triton",
        decode_conv1d_weight=None,
        decode_norm_weight=None,
        use_recoverssm=False,
        kda_prefill_backend="flashkda",
        _frost=core in ("persistent", "frost"),
        _kda_persistent=core == "persistent",
        _flashkda_buffer_specs=(
            ((1, x.T, H, D), torch.bfloat16),
            ((N, H, D, D), torch.float32),
            ((N, H, D, D), torch.float32),
            ((ws,), torch.uint8),
        ),
        _segscan_bufs=None,
        _checkpoint_exporter=None,
    )
    layer._o_norm_mxfp8 = MethodType(KimiK3DeltaAttention._o_norm_mxfp8, layer)
    layer._forward = MethodType(KimiK3DeltaAttention._forward, layer)
    return layer


def _metadata():
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata

    T = sum(LENS)
    offs = [0]
    for n in LENS:
        offs.append(offs[-1] + n)
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    return KimiK3KDAMetadata(
        num_prefills=len(LENS),
        num_prefill_tokens=T,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=T,
        has_initial_state=torch.tensor(HAS_INIT, device="cuda"),
        non_spec_query_start_loc=cu_cpu.cuda(),
        non_spec_state_indices_tensor=torch.tensor(
            SLOT_IDS, dtype=torch.int32, device="cuda"
        ),
        non_spec_query_start_loc_cpu=cu_cpu,
    )


@pytest.fixture
def kda_env(monkeypatch):
    from vllm.models.kimi_k3.nvidia import kda as kda_mod
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    metadata = _metadata()
    monkeypatch.setattr(
        kda_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test.kda": metadata}),
    )

    def poison(value: float):
        monkeypatch.setattr(kda_mod, "torch", _PoisonedTorch(value))

    yield poison
    monkeypatch.setattr(kda_mod, "torch", torch)
    reset_workspace_manager()


def _core_supported(core: str) -> bool:
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    dev = torch.device("cuda")
    if core == "persistent":
        return sp_kda.frost_supported(dev) and sp_kda.persistent_supported(dev)
    return sp_kda.frost_supported(dev)


@pytest.mark.parametrize("core", ["persistent", "frost"])
@torch.inference_mode()
def test_tp_prefill_forward_matches_stock(kda_env, core):
    """TP > 1 prefill (GEMM-AR o_proj, BF16 o_norm in place on core_attn_out):
    finite, independent of the poison (nothing reads uninitialized memory), and
    equal to the stock FlashKDA core up to its operand rounding (2e-2 rel rms,
    as test_kda_frost_prefill), final states included."""
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    if not _core_supported(core):
        pytest.skip(f"{core} KDA prefill core unavailable")
    x = _inputs()
    positions = torch.arange(x.T, device="cuda")

    def run(c: str, poison: float):
        kda_env(poison)
        rs_ar = _FakeGemmRsAr()
        layer = _layer(x, c, rs_ar)
        out = KimiK3DeltaAttention.forward(layer, x.hidden, positions)
        torch.accelerator.synchronize()
        assert rs_ar.calls == 1  # the TP o_proj ran, so o_mxfp8 was None
        assert layer._o_norm_mxfp8_ok is False
        return out.clone(), layer.kv_cache[1].clone()

    out_nan, state_nan = run(core, float("nan"))
    out_fin, state_fin = run(core, 3.0e4)
    ref, ref_state = run("flashkda", float("nan"))

    assert torch.isfinite(ref).all()
    assert torch.isfinite(out_nan).all(), "o_norm read uninitialized core_attn_out"
    torch.testing.assert_close(out_nan, out_fin, atol=0, rtol=0)
    torch.testing.assert_close(state_nan, state_fin, atol=0, rtol=0)
    assert _rel_rms(out_nan, ref) < 2e-2
    rows = torch.tensor(SLOT_IDS, device="cuda")
    assert _rel_rms(state_nan[rows], ref_state[rows]) < 2e-2


@pytest.mark.parametrize("core", ["persistent", "frost"])
@torch.inference_mode()
def test_mxfp8_o_norm_prefill_reads_the_core_output(kda_env, core):
    """The DEP route (TP1: MXFP8 prefill o_norm, o_mxfp8 set) at _forward level:
    the MXFP8 o_proj input is independent of the poison and dequantizes to the
    stock FlashKDA layer's."""
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

    if not _core_supported(core):
        pytest.skip(f"{core} KDA prefill core unavailable")
    x = _inputs(1)
    T = x.T
    proj = x.hidden @ x.w_in.t()
    mixed_qkv, g_proj, f_a, beta = proj.split([3 * P, P, D, H, x.pad], dim=-1)[:4]
    g1 = (f_a @ x.w_fb.t()).view(1, T, H, D)
    g2 = g_proj.view(T, H, D)

    def run(c: str, poison: float):
        kda_env(poison)
        layer = _layer(x, c, None)
        core_attn_out = torch.full(
            (1, T, H, D), poison, dtype=torch.bfloat16, device="cuda"
        )
        o_mxfp8 = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
        KimiK3DeltaAttention._forward(
            layer,
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta.unsqueeze(0),
            core_attn_out=core_attn_out,
            o_mxfp8=o_mxfp8,
        )
        torch.accelerator.synchronize()
        rows = (T + 127) // 128 * 128
        s = unswizzle_mxfp8_scale(o_mxfp8.scale, rows, P)[:T].float()
        deq = o_mxfp8.data.float().view(T, P // 32, 32) * torch.exp2(
            s.view(T, P // 32, 1) - 127
        )
        return o_mxfp8.data.view(torch.uint8).clone(), deq.view(T, P)

    codes_nan, deq_nan = run(core, float("nan"))
    codes_fin, _ = run(core, 3.0e4)
    _, deq_ref = run("flashkda", float("nan"))
    assert torch.isfinite(deq_nan).all()
    torch.testing.assert_close(codes_nan, codes_fin, atol=0, rtol=0)
    assert _rel_rms(deq_nan, deq_ref) < 3e-2


class _FakeMxfp8GemmRsAr(_FakeGemmRsAr):
    """The fused GEMM-RS/AR o_proj of an MXFP8 o_proj: takes the MXFP8 activation
    the gated o_norm wrote (records what it was handed)."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def should_run(self, x) -> bool:
        return True

    def apply(self, x, linear, residual=None, residual_row0=0, copy_out=True):
        self.inputs.append(type(x).__name__)
        return super().apply(x, linear, residual, residual_row0, copy_out)


def _dequant_activation(qa) -> torch.Tensor:
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale

    T, K = qa.orig_shape
    rows = (T + 127) // 128 * 128
    s = unswizzle_mxfp8_scale(qa.scale, rows, K)[:T].float()
    deq = qa.data.float().view(T, K // 32, 32) * torch.exp2(s.view(T, K // 32, 1) - 127)
    return deq.view(T, K)


@pytest.mark.parametrize("core", ["persistent", "frost"])
@torch.inference_mode()
def test_tp_prefill_mxfp8_o_norm_feeds_gemm_rs(kda_env, core, monkeypatch):
    """TP > 1 prefill with an MXFP8 o_proj: the gated o_norm writes the MXFP8
    activation and the fused GEMM-RS/AR consumes it (no BF16 o_norm output, no
    separate quantize); finite, poison-independent, and equal to the BF16 TP
    route (in-place o_norm, then the o_proj's own MXFP8 quantization) up to that
    quantization."""
    from vllm.models.kimi_k3.nvidia import mxfp8 as mxfp8_mod
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    if not _core_supported(core):
        pytest.skip(f"{core} KDA prefill core unavailable")
    x = _inputs()
    positions = torch.arange(x.T, device="cuda")

    def run(c: str, poison: float, mx: bool):
        kda_env(poison)
        monkeypatch.setattr(mxfp8_mod, "takes_mxfp8_activation", lambda *linears: mx)
        rs_ar = _FakeMxfp8GemmRsAr()
        layer = _layer(x, c, rs_ar)
        if mx:
            layer.o_proj = lambda a: (_dequant_activation(a) @ x.w_o.float().t(),)
        out = KimiK3DeltaAttention.forward(layer, x.hidden, positions)
        torch.accelerator.synchronize()
        assert rs_ar.calls == 1
        assert layer._o_norm_mxfp8_ok is mx
        assert rs_ar.inputs == (["QuantizedActivation"] if mx else ["Tensor"])
        return out.float().clone()

    out_nan = run(core, float("nan"), True)
    out_fin = run(core, 3.0e4, True)
    ref = run(core, float("nan"), False)
    assert torch.isfinite(out_nan).all(), "the MXFP8 o_norm read uninitialized memory"
    torch.testing.assert_close(out_nan, out_fin, atol=0, rtol=0)
    assert _rel_rms(out_nan, ref) < 3e-2
