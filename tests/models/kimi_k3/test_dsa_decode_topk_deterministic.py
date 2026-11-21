# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA decode top-k under sparse_indexer_topk_backend="trtllm_gvr2" (the
Kimi K3.1 P role) must be exact AND deterministic: the same decode logits give
the same column set in the same order on every call.

A P step with at most 1 + num_speculative_tokens new tokens over a cached prefix
is a uniform decode step, so its indexer runs the decode top-k. The 16-bit GVR
V2 decode engine wrote every row in a different order on each call and picked a
different set among the ties at the cutoff, and the sparse MLA is
order-sensitive: the same 1-4 token request repeated gave different logprobs
(BUG-STRESS-2). The rows below are the decode rows of such a step, selected as
sparse_attn_indexer selects them (``_decode_topk_backend`` + ``get_indexer_topk``).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="BF16 DeepGEMM decode logits / DeepSelect target SM10x",
)

K = 2048


def _decode_logits(
    batch: int, next_n: int, context: int, kind: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 decode logits of ``batch`` requests x ``next_n`` rows at ``context``
    tokens (row ends context - next_n + 1 .. context), DeepGEMM's 1024 B row
    stride, and the (B, next_n) per-row ends of native spec decode."""
    torch.manual_seed(batch * 7 + next_n + context)
    ends = context - next_n + 1 + torch.arange(next_n, dtype=torch.int32)
    ends = ends.repeat(batch, 1).cuda()
    cols = (context + 511) // 512 * 512
    x = torch.randn(batch * next_n, cols, device="cuda") * 3
    if kind == "zero_prefix":
        x[:, : context - 64] = 0.0  # zero KV: the prefix scores all tie at 0
    return x.to(torch.bfloat16), ends.contiguous()


def _sorted_scores(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    v = x.float().gather(1, idx.long().clamp(min=0))
    return v.masked_fill(idx < 0, float("-inf")).sort(dim=1, descending=True).values


@pytest.mark.parametrize(
    "batch,next_n,context,kind",
    [
        (1, 1, 4096, "random"),
        (1, 2, 65536, "random"),
        (1, 4, 262144, "random"),
        (4, 4, 16384, "random"),
        (2, 3, 16384, "zero_prefix"),
    ],
)
@torch.inference_mode()
def test_trtllm_gvr2_decode_topk_is_deterministic(batch, next_n, context, kind):
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.model_executor.layers.sparse_attn_indexer import _decode_topk_backend

    x, ends = _decode_logits(batch, next_n, context, kind)
    rows = batch * next_n
    topk = get_indexer_topk(_decode_topk_backend("trtllm_gvr2", 1))
    outs = []
    for _ in range(4):
        out = torch.full((rows, K), -2, dtype=torch.int32, device="cuda")
        topk(x, ends, next_n, out, K, context)
        outs.append(out)
    torch.accelerator.synchronize()
    for out in outs[1:]:
        assert torch.equal(out, outs[0]), (
            f"{int((out != outs[0]).any(1).sum())} of {rows} rows differ between "
            "two calls on the same logits"
        )
    # Exact: every row keeps K columns below its end, the row's top-K scores.
    assert bool((outs[0] >= 0).all())
    assert bool((outs[0] < ends.view(-1, 1)).all())
    col = torch.arange(x.shape[1], device="cuda")[None, :]
    masked = x.float().masked_fill(col >= ends.view(-1, 1), float("-inf"))
    ref = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :K]
    torch.testing.assert_close(
        _sorted_scores(x, outs[0]), _sorted_scores(masked, ref.int()), atol=0, rtol=0
    )
