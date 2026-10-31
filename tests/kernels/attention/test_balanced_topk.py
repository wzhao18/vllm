# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""balanced_topk: exact top-k of ragged BF16 decode logits vs torch.topk."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.balanced_topk import CHUNK, balanced_topk

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="balanced_topk is a CUDA (Triton) op"
)


def _row_lens(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
    if seq_lens.numel() == num_rows:
        return seq_lens.reshape(-1).long()
    off = torch.arange(next_n, device=seq_lens.device)
    return (seq_lens.reshape(-1, 1) - next_n + 1 + off).clamp(min=0).reshape(-1).long()


def _check(logits, seq_lens, next_n, out, topk):
    num_rows, width = logits.shape
    lens = _row_lens(seq_lens, next_n, num_rows).clamp(max=width).tolist()
    for r in range(num_rows):
        n = lens[r]
        got = out[r, :topk]
        k = min(n, topk)
        assert (got[k:] == -1).all(), f"row {r}: tail not -1"
        sel = got[:k].long()
        assert ((sel >= 0) & (sel < n)).all(), f"row {r}: index out of range"
        assert sel.unique().numel() == k, f"row {r}: duplicate indices"
        if n <= topk:
            assert torch.equal(sel.sort().values, torch.arange(n, device=sel.device))
            continue
        row = logits[r, :n].float()
        ref = row.topk(k).values.sort().values
        torch.testing.assert_close(row[sel].sort().values, ref, rtol=0, atol=0)


@pytest.mark.parametrize("next_n", [1, 4, 8])
@pytest.mark.parametrize("per_row_lens", [False, True])
@pytest.mark.parametrize("topk", [2048, 512])
def test_balanced_topk_skewed(next_n, per_row_lens, topk):
    torch.manual_seed(0)
    # AgentX-like skew: short, mid, one long row, a row shorter than topk,
    # an empty one, and one spanning a chunk boundary exactly.
    req_lens = [5, 0, topk, topk + 1, 37_888, CHUNK * 3, 131_072, 659_456]
    width = 663_552
    num_rows = len(req_lens) * next_n
    logits = (torch.randn(num_rows, width, device="cuda") * 3).to(torch.bfloat16)
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device="cuda")
    if per_row_lens:
        seq_lens = _row_lens(seq_lens, next_n, num_rows).to(torch.int32).view(-1, next_n)
    out = torch.full((num_rows, topk + 64), -7, dtype=torch.int32, device="cuda")
    balanced_topk(logits, seq_lens, next_n, out[:, :topk], topk)
    assert (out[:, topk:] == -7).all(), "wrote past topk"
    _check(logits, seq_lens, next_n, out, topk)


@pytest.mark.parametrize(
    "pattern", ["ties", "neg", "inf", "const", "adversarial", "far", "short_ties"]
)
def test_balanced_topk_value_patterns(pattern):
    """Heavy ties at the threshold, all-negative rows, -inf padding, a
    constant row (one tie spanning the whole row), and a row whose sample
    misjudges the threshold. Ties and const overflow the candidate list."""
    torch.manual_seed(1)
    topk, n, rows = 2048, 300_000, 4
    if pattern == "ties":
        x = torch.randint(-4, 4, (rows, n), device="cuda").float()
    elif pattern == "neg":
        x = -torch.rand(rows, n, device="cuda") * 100 - 1
    elif pattern == "inf":
        x = torch.randn(rows, n, device="cuda")
        x[:, ::3] = float("-inf")
    elif pattern == "const":
        x = torch.full((rows, n), 1.5, device="cuda")
    elif pattern == "far":
        # A few thousand huge values over a plateau just above the bulk: the
        # threshold sits at the plateau, the k-th key far above it.
        x = -torch.ones(rows, n, device="cuda")
        x[:, torch.randperm(n, device="cuda")[:6000]] = 1.0
        x[:, torch.randperm(n, device="cuda")[: topk + 100]] = 1e6
    elif pattern == "short_ties":
        # Rows shorter than the candidate capacity, all ties.
        n = 9000
        x = torch.randint(-2, 2, (rows, n), device="cuda").float()
    else:
        # Large distinct values exactly at the strided sample positions and
        # small ones elsewhere: the sampled threshold keeps fewer than topk
        # candidates, which must fall back to the exact whole-row search.
        from vllm.v1.attention.ops.balanced_topk import SAMPLES

        x = torch.rand(rows, n, device="cuda")
        i = torch.arange(SAMPLES, device="cuda")
        pos = (2 * i + 1) * n // (2 * SAMPLES)
        x[:, pos] = 100 + i.float() / 8
    logits = x.to(torch.bfloat16)
    seq_lens = torch.full((rows, 1), n, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    balanced_topk(logits, seq_lens, 1, out, topk)
    _check(logits, seq_lens, 1, out, topk)


def test_balanced_topk_cuda_graph():
    """Replays with new lengths and values (device-side work list)."""
    torch.manual_seed(2)
    topk, width, next_n = 2048, 400_000, 4
    logits = torch.empty(16 * next_n, width, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.empty(16, 1, dtype=torch.int32, device="cuda")
    out = torch.empty(16 * next_n, topk, dtype=torch.int32, device="cuda")
    logits.normal_()
    seq_lens.fill_(width)
    balanced_topk(logits, seq_lens, next_n, out, topk)  # compile
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        balanced_topk(logits, seq_lens, next_n, out, topk)
    for lens in ([width] * 16, [3000 * (i + 1) for i in range(16)], [100] * 15 + [width]):
        logits.normal_()
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32).view(-1, 1))
        g.replay()
        _check(logits, seq_lens, next_n, out, topk)
