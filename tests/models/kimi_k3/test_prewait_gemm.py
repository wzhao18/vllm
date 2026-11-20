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


@pytest.mark.parametrize("t", [32, 64])
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
