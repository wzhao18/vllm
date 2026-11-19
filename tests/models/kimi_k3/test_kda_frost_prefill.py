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

    if not sp_kda.frost_supported(torch.device("cuda")):
        pytest.skip("cuDNN frontend >= 1.33 (kda_frost) not installed")
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
