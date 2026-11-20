# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-homed KDA prefill buffers (specialized_prefill.kda_dielocal): the
g|f_a|beta projection written into the die-homed rows and the persistent core
writing into the die-homed output are bitwise equal to the default
allocations, and the buffers are reused across calls."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)


def _two_dies() -> bool:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    return dies.domain_count(torch.cuda.current_device()) == 2


@pytest.mark.parametrize("tokens", [4099, 2048])
def test_proj_rows_scaled_mm_bitwise(tokens):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_dielocal

    if not _two_dies():
        pytest.skip("single-die device")
    dev = torch.device("cuda")
    k, n = 1024, 1664
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randint(-3, 4, (tokens, k), device=dev, generator=g).float()
    w = torch.randint(-3, 4, (n, k), device=dev, generator=g).float()
    sx = torch.randint(118, 122, (((tokens + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    args = (x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t())
    kw = dict(
        scale_a=sx.view(torch.float8_e8m0fnu),
        scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )
    ref = torch._scaled_mm(*args, **kw)
    rows = kda_dielocal.proj_rows(tokens, n, dev)
    assert rows.shape == (tokens, n) and rows.is_contiguous()
    got = torch._scaled_mm(*args, **kw, out=rows)
    assert got.data_ptr() == rows.data_ptr()
    assert torch.equal(got, ref)
    # Same buffer for a shorter call; regrown for a longer one.
    assert kda_dielocal.proj_rows(tokens - 1, n, dev).data_ptr() == rows.data_ptr()
    longer = kda_dielocal.proj_rows(tokens + 4096, n, dev)
    assert longer.shape == (tokens + 4096, n)


def test_core_out_persistent_prefill_bitwise():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_dielocal

    if not _two_dies():
        pytest.skip("single-die device")
    dev = torch.device("cuda")
    if not sp_kda.persistent_supported(dev):
        pytest.skip("CuTe DSL not installed")
    torch.manual_seed(0)
    lens, H, D, bf = [300, 1, 1029], 8, 128, torch.bfloat16
    T, N = sum(lens), len(lens)
    cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, gg = (torch.randn(1, T, H, D, device=dev).to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device=dev) * 0.5).to(bf)
    beta = torch.randn(1, T, H, device=dev).to(bf)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H * D, device=dev) * 0.1
    h0 = torch.randn(N, H, D, D, device=dev) * 0.1
    args = (q, k, v, gg, beta, A_log, dt_bias, -5.0, h0, cu, cu_cpu)
    o_ref, s_ref = sp_kda.persistent_prefill(*args)
    out = kda_dielocal.core_out(T, H, D, dev)
    o, s = sp_kda.persistent_prefill(*args, out=out)
    assert o.data_ptr() == out.data_ptr()
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
