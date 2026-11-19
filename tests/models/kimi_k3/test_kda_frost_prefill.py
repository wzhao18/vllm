# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA prefill core on cuDNN FROST (specialized_prefill/kda.py) against
FlashKDA, the core it replaces: same output and final state up to BF16
operand rounding order."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="cuDNN FROST KDA targets SM10x",
)

D = 128


def _rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("lens,heads", [([37], 4), ([300, 1, 1029], 8), ([2051, 777], 96)])
def test_frost_prefill_matches_flashkda(lens, heads):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    assert sp_kda.frost_supported(torch.device("cuda"))
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H].unsqueeze(0)  # strided
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    out = torch.empty(1, T, H, D, dtype=bf, device="cuda")
    fs = torch.empty_like(h0)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda")
    torch.ops._flashkda_C.fwd(
        q, k, v, g, beta.contiguous(), D**-0.5, out, ws, A_log, dt_bias.view(H, D),
        -5.0, h0, fs, cu, None, None,
    )
    o, state = sp_kda.frost_prefill(q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu)
    torch.accelerator.synchronize()
    assert o.shape == out.shape and state.shape == fs.shape
    assert not torch.isnan(o).any()
    assert _rel_rms(o, out) < 2e-2
    assert _rel_rms(state, fs) < 2e-2


@pytest.mark.parametrize(
    "lens,heads", [([37], 8), ([300, 1, 1029], 8), ([2051, 777, 64], 96), ([5461] * 3, 96)]
)
def test_persistent_prefill_matches_flashkda(lens, heads):
    """The persistent core (LPT chain pieces with FP32 state handoffs; the
    3 x 5461 x 96-head shape splits chains across CTAs) against FlashKDA."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H].unsqueeze(0)  # strided
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    out = torch.empty(1, T, H, D, dtype=bf, device="cuda")
    fs = torch.empty_like(h0)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda")
    torch.ops._flashkda_C.fwd(
        q, k, v, g, beta.contiguous(), D**-0.5, out, ws, A_log, dt_bias.view(H, D),
        -5.0, h0, fs, cu, None, None,
    )
    h0_ref = h0.clone()
    o, state = sp_kda.persistent_prefill(
        q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu, cu_cpu
    )
    torch.accelerator.synchronize()
    assert torch.equal(h0, h0_ref)
    assert o.shape == out.shape and state.shape == fs.shape
    assert not torch.isnan(o).any()
    assert _rel_rms(o, out) < 2e-2
    assert _rel_rms(state, fs) < 2e-2


def test_frost_strided_beta():
    """frost_prefill passes beta as a row-strided view of the projection:
    FROST must read it exactly like a contiguous copy."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    assert sp_kda.frost_supported(torch.device("cuda"))
    torch.manual_seed(1)
    T, H, bf = 300, 8, torch.bfloat16
    cu = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
    q, k, v, g = (torch.randn(T, H, D, device="cuda").to(bf) for _ in range(4))
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H]
    kw = dict(scale=D**-0.5, initial_state=torch.zeros(2, H, D, D, device="cuda"),
              output_final_state=True, use_qk_l2norm_in_kernel=True,
              use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=-5.0,
              a_log=torch.zeros(H, device="cuda"), dt_bias=torch.zeros(H, D, device="cuda"),
              plan_name="kda_frost")
    ref = sp_kda._frost_fn(q, k, v, g, beta.contiguous(), cu, **kw)[0]
    got = sp_kda._frost_fn(q, k, v, g, beta, cu, **kw)[0]
    assert torch.equal(got, ref)


def test_frost_requires_cudnn_frontend_133(monkeypatch):
    """An older cuDNN frontend on SM10x is an install error, not a silent
    FlashKDA fallback."""
    import cudnn

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    monkeypatch.setattr(sp_kda, "_frost_fn", None)
    monkeypatch.setattr(cudnn, "__version__", "1.32.1")
    with pytest.raises(RuntimeError, match="older than 1.33"):
        sp_kda.frost_supported(torch.device("cuda"))
