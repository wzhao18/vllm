# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QueryNorm with the KV insert (sdec local_query) vs stock: ``rms_norm`` of the
kv_a columns then ``concat_and_cache_mla`` (fp8, per-tensor k_scale), bitwise on
the cache; q_c rows identical to the QueryNorm compiled without the insert."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="sdec CuTe kernels need sm_100 family",
)

PITCH, BLOCKS, BLOCK = 2240, 32, 64


@pytest.mark.parametrize("heads,batch", [(12, 3), (12, 32), (12, 96), (6, 5), (24, 64)])
def test_querynorm_kv_insert_vs_stock(heads, batch):
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (
        compile_norm,
    )

    device = torch.device("cuda")
    torch.manual_seed(heads * 100 + batch)
    eps = 1e-6
    projected = (torch.randn(batch, PITCH, device=device) * 2).to(torch.bfloat16)
    projected[0, 1536:2048] = 0  # all-zero kv_a row
    w_q = (torch.rand(1536, device=device) + 0.5).to(torch.bfloat16)
    w_kv = (torch.rand(512, device=device) + 0.5).to(torch.bfloat16)
    k_scale = torch.tensor([0.37], device=device)
    slots = torch.randperm(BLOCKS * BLOCK, device=device)[:batch].to(torch.int64)
    slots[batch // 2] = -1  # padding row: skipped

    junk = torch.randint(0, 255, (BLOCKS, BLOCK, 576), device=device)
    cache_ref = junk.to(torch.uint8)
    cache = cache_ref.clone()
    kv = torch.empty(batch, 512, dtype=torch.bfloat16, device=device)
    ops.rms_norm(kv, projected[:, 1536:2048], w_kv, eps)
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], cache_ref, slots, "fp8", k_scale
    )

    layout = tuple(cache.shape), tuple(cache.stride()), cache.dtype
    plain = compile_norm(batch, heads, PITCH)
    fused = compile_norm(batch, heads, PITCH, layout)
    qc_ref = torch.empty(batch, 1536, dtype=torch.bfloat16, device=device)
    qc = torch.empty_like(qc_ref)
    s = stream(device)
    args = to_cute(projected), to_cute(w_q, 16)
    e = cutlass.Float32(eps)
    plain(*args, to_cute(qc_ref), e, s)
    fused(
        *args, to_cute(qc), e, to_cute(w_kv, 16), to_cute(cache),
        to_cute(slots, 8), to_cute(k_scale, 4), s,
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert torch.equal(qc, qc_ref)
    assert torch.equal(cache, cache_ref)
