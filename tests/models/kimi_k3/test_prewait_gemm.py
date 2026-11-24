# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-wait weight-streaming decode GEMM (specialized decode): numerics vs torch."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.get_device_capability()[1] != 7,
    reason="sm_107 (Rubin) only",
)


@pytest.mark.parametrize("t", [32, 64, 96])
@pytest.mark.parametrize("n,out_dtype", [(4480, torch.float32), (3584, torch.bfloat16), (1000, torch.float32)])
def test_prewait_gemm_matches_torch(t, n, out_dtype):
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm as pg

    torch.manual_seed(t + n)
    k = 7168
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    packed = pg.pack_weight(w)
    out = torch.full((t, n), float("nan"), device="cuda", dtype=out_dtype)
    pg.gemm(out, x, packed, n)
    ref = (x.float() @ w.float().t())
    if out_dtype == torch.bfloat16:
        torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), atol=2e-2, rtol=1.6e-2)
    else:
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)
    # deterministic (rank-ordered split-K sum)
    out2 = torch.empty_like(out)
    pg.gemm(out2, x, packed, n)
    assert torch.equal(out, out2)


@pytest.mark.parametrize("t,ns", [(32, 8), (64, 7), (96, 6)])
@pytest.mark.parametrize("pace", [0, 1440])
def test_prewait_gemm_coresident_bitwise(t, ns, pace):
    """The co-resident form (8 warps, <= 64 registers) with burst or paced pre-wait weight
    issue gives the same bits as the 1-CTA-per-SM form (same K order, same split-K sum)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm as pg

    torch.manual_seed(t)
    n, k = 4480, 7168
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    packed = pg.pack_weight(w)
    ref = torch.empty(t, n, device="cuda", dtype=torch.float32)
    pg.gemm(ref, x, packed, n, (4, 1, 6, 6, 3, 0, 0))
    out = torch.full_like(ref, float("nan"))
    pg.gemm(out, x, packed, n, (4, 1, ns, 2, 1, pace, 1))
    assert torch.equal(out, ref)
    torch.testing.assert_close(out, x.float() @ w.float().t(), atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("t", [33, 34, 40, 63, 65, 68, 70, 72, 95])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_prewait_gemm_padded_tile_bitwise(t, out_dtype):
    """t live tokens on the next tile (64 / 96): every live row has the bits of the
    full tile run on the zero-padded input, and nothing is stored past row t."""
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm as pg

    torch.manual_seed(t)
    n, k = 4480, 7168
    tile = pg.tile(t)
    assert tile == (64 if t <= 64 else 96) and pg.supported(t, k)
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    packed = pg.pack_weight(w)
    padded = torch.zeros(tile, k, device="cuda", dtype=torch.bfloat16)
    padded[:t] = x
    ref = torch.empty(tile, n, device="cuda", dtype=out_dtype)
    pg.gemm(ref, padded, packed, n)
    buf = torch.full((tile, n), float("nan"), device="cuda", dtype=out_dtype)
    out = buf[:t]
    pg.gemm(out, x, packed, n)
    assert torch.equal(out, ref[:t])
    assert torch.isnan(buf[t:]).all()
