# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the drafter's LM-head kernels on the TP8 vocabulary
shard ([163840 / 8, 7168], DFn), as ``runtime.draft.FAST_LM_HEAD[8]`` /
``WIDE_LM_HEAD[8]`` serve them: the BF16-logits GEMV (``PreparedLmHeadLogits``, 1-7
rows; an FP32 K reorder) and the evict-first GEMM (``PreparedWideLmHead``, 17-32 rows,
FP32 accumulation in K order) against the FP32 product, one BF16 rounding apart, and
each deterministic across calls."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

VOCAB_RANK, HIDDEN = 163840 // 8, 7168


@pytest.mark.parametrize("rows", [1, 4, 7, 17, 24, 32])
def test_tp8_drafter_lm_head(rows):
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    assert draft.batch_table(draft.FAST_LM_HEAD[8]).get(7)
    assert draft.batch_table(draft.WIDE_LM_HEAD[8]).get(32)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    gen = torch.Generator(device=dev).manual_seed(rows)
    weight = (torch.randn(VOCAB_RANK, HIDDEN, generator=gen, device=dev) / 40).bfloat16()
    x = torch.randn(rows, HIDDEN, generator=gen, device=dev).bfloat16()
    assert lmhead.supports(weight)
    if rows <= 8:
        head = lmhead.PreparedLmHeadLogits(weight, dev)
        out = head.run(x).clone()
        again = head.run(x).clone()
    else:
        head = lmhead.PreparedWideLmHead(weight, dev)
        out = head.run(x).clone()
        again = head.run(x).clone()
    torch.cuda.synchronize()
    ref = (x.float() @ weight.float().t()).bfloat16()
    assert torch.equal(out, again)
    # one BF16 ulp of the value, or of the row's largest logit near zero
    exact = x.float() @ weight.float().t()
    tol = 2**-7 * torch.maximum(exact.abs(), 2**-8 * exact.abs().amax(1, keepdim=True))
    assert bool(((out.float() - exact).abs() <= tol).all())
    assert bool(((ref.float() - exact).abs() <= tol).all())
