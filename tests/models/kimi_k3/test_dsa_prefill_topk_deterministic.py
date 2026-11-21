# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA prefill top-k under sparse_indexer_topk_backend="trtllm_gvr2" (the
Kimi K3.1 P configs) must be exact AND deterministic: the same logits give the
same column set in the same order on every call.

BF16 logits tie at the top-2048 cutoff on almost every long-context row, and the
sparse MLA prefill is order-sensitive. The 16-bit GVR V2 prefill engine picked a
different tie set and row order per call, so a TP8 + DCP8 96K-prefix prefill
step was not reproducible (BUG-DCP8-2).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="BF16 DeepGEMM prefill logits / DeepSelect target SM10x",
)

K = 2048


def _logits(rows: int, prefix: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A DCP8 ShardQ rank's chunk: ``rows`` queries over a ``prefix``-token
    cached context plus their causal new tokens; DeepGEMM's 1024 B row stride."""
    torch.manual_seed(rows + prefix)
    ends = prefix + 1 + 8 * torch.arange(rows, dtype=torch.int32, device="cuda")
    cols = (int(ends.max()) + 511) // 512 * 512
    x = torch.randn(rows, cols, device="cuda") * 3
    if kind == "zero_prefix":
        x[:, :prefix] = 0.0  # zero KV: every prefix score ties at 0
    col = torch.arange(cols, device="cuda")[None, :]
    x = x.masked_fill(col >= ends[:, None].long(), float("-inf"))
    return x.to(torch.bfloat16).contiguous(), ends.contiguous()


def _sorted_scores(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    v = x.float().gather(1, idx.long().clamp(min=0))
    return v.masked_fill(idx < 0, float("-inf")).sort(dim=1, descending=True).values


@pytest.mark.parametrize(
    "rows,prefix,kind",
    [(2048, 98304, "random"), (2048, 98304, "zero_prefix"), (512, 16384, "random")],
)
@torch.inference_mode()
def test_trtllm_gvr2_prefill_topk_is_deterministic(rows, prefix, kind):
    from vllm.model_executor.layers.sparse_attn_indexer import _prefill_topk

    x, ends = _logits(rows, prefix, kind)
    outs = []
    for _ in range(4):
        out = torch.full((rows, K), -1, dtype=torch.int32, device="cuda")
        _prefill_topk(x, None, ends, out, K, "trtllm_gvr2")
        outs.append(out)
    torch.accelerator.synchronize()
    for out in outs[1:]:
        assert torch.equal(out, outs[0]), (
            f"{int((out != outs[0]).any(1).sum())} of {rows} rows differ between "
            "two calls on the same logits"
        )
    # Exact: the selected scores are the row's top-K scores.
    ref = torch.sort(x.float(), dim=1, descending=True, stable=True).indices[:, :K]
    torch.testing.assert_close(
        _sorted_scores(x, outs[0]), _sorted_scores(x, ref.int()), atol=0, rtol=0
    )
