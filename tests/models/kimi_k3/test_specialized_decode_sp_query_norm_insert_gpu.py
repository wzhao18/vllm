# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode, local query rows: QueryNorm (dsa.overlap) with this rank's KV
insert against the stock ops it replaces (``rms_norm(kv_a)`` +
``concat_and_cache_mla(..., "fp8", k_scale)``): the q_c rows bitwise those of the
plain QueryNorm, unowned slots untouched, k_pe bytes bitwise, and the kv_c bytes
equal but where the kv_a sum of squares' FP32 order moves the BF16 row by an ulp
(rare; one FP8 step at most). One GPU."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

PITCH, UNIT, FP8_ROW, EPS, HEADS = 5344, 64, 576, 1e-6, 24


@pytest.mark.parametrize("rows", [68, 128, 384])
def test_query_norm_kv_insert_matches_stock(rows):
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
    )

    dev, bf = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(rows)
    projected = (torch.randn(rows, PITCH, generator=gen, device=dev) * 3).to(bf)
    w_qa = (1 + 0.1 * torch.randn(1536, generator=gen, device=dev)).to(bf)
    w_kv = (1 + 0.1 * torch.randn(512, generator=gen, device=dev)).to(bf)
    k_scale = torch.tensor([0.0625], device=dev)
    blocks = -(-rows // UNIT) + 2
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:rows]
    slots[torch.arange(rows, device=dev) % 4 != 1] = -1  # other ranks' tokens
    fill = torch.randint(0, 256, (blocks, UNIT, FP8_ROW), generator=gen, device=dev)
    cache = fill.to(torch.uint8)
    stock = cache.clone()

    eps = cutlass.Float32(EPS)
    plain = compile_norm(rows, HEADS, PITCH)
    fused = compile_norm(rows, HEADS, PITCH, (tuple(cache.shape), cache.stride()))
    qc, ref_qc = (torch.empty(rows, 1536, dtype=bf, device=dev) for _ in range(2))
    s = stream(torch.device(dev))
    common = to_cute(projected), to_cute(w_qa)
    plain(*common, to_cute(ref_qc), eps, s)
    fused(*common, to_cute(qc), eps, to_cute(w_kv), to_cute(cache),
          to_cute(slots, 8), to_cute(k_scale, 4), eps, s)  # fmt: skip
    kv = torch.empty(rows, 512, dtype=bf, device=dev)
    ops.rms_norm(kv, projected[:, 1536:2048].contiguous(), w_kv, EPS)
    ops.concat_and_cache_mla(kv, projected[:, 2048:2112], stock, slots, "fp8", k_scale)
    torch.cuda.synchronize()

    assert torch.equal(qc, ref_qc)
    owned = slots[slots >= 0]
    flat, ref = cache.view(-1, FP8_ROW), stock.view(-1, FP8_ROW)
    others = torch.ones(flat.shape[0], dtype=torch.bool, device=dev)
    others[owned] = False
    assert torch.equal(flat[others], ref[others])  # unowned slots untouched
    assert torch.equal(flat[owned, 512:], ref[owned, 512:])  # k_pe
    got, want = flat[owned, :512], ref[owned, :512]
    differ = got != want
    steps = (got.view(torch.int8).int() - want.view(torch.int8).int()).abs()
    print(f"rows {rows}: {int(differ.sum())} of {differ.numel()} kv_c bytes differ")
    assert float(differ.float().mean()) < 2e-3
    assert int(steps[differ].max(initial=0)) <= 1
