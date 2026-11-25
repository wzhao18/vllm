# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-batched cluster AttnRes (``attn_res.rows.ClusterAttnResRows``, several
tokens per 7-CTA cluster, exchange rounds of up to two warp-reduce chunks) against
the one-token ``ClusterAttnRes`` (bitwise: output, updated prefix, saved blocks)
and the stock AttnRes op (tolerance of the stock test)."""

import pytest
import torch

from vllm.platforms import current_platform

HIDDEN = 7168
MAX_BLOCKS = 8
EPS = 1e-6

gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.get_device_properties(0).multi_processor_count <= 152,
    reason="specialized decode AttnRes kernels: Rubin (sm_10x, > 152 SMs)",
)


def _inputs(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, device="cuda", generator=g, dtype=torch.float32)
        return (x * scale).to(torch.bfloat16)

    return dict(
        prefix=randn(rows, HIDDEN),
        delta=randn(rows, HIDDEN, scale=0.5),
        blocks=randn(rows, MAX_BLOCKS, HIDDEN),
        norm=(1 + 0.1 * randn(HIDDEN).float()).to(torch.bfloat16),
        query=randn(HIDDEN, scale=HIDDEN**-0.5),
        out_norm=(1 + 0.1 * randn(HIDDEN).float()).to(torch.bfloat16),
    )


def _run(nb, write, final, tokens, x):
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.prepared import (  # noqa: E501
        compile_consumer,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        architecture,
        dynamic_view,
        stream,
        to_cute,
    )

    device = x["prefix"].device
    rows = x["prefix"].shape[0]
    prefix, blocks = x["prefix"].clone(), x["blocks"].clone()
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    code = compile_consumer(
        nb, True, write, final, 0, 0, architecture(device), tokens=tokens
    )
    weights = (to_cute(x[k].reshape(-1)) for k in ("norm", "query", "out_norm"))
    views = map(dynamic_view, (x["delta"], prefix, blocks))
    eps = cutlass.Float32(EPS)
    code(*views, *weights, dynamic_view(out), eps, eps, stream(device))
    torch.accelerator.synchronize()
    return out, prefix, blocks


# (nb, tokens): every token count up to the two-chunk round at 1-8 saved blocks.
CASES = [
    (nb, t)
    for nb in (0, 1, 2, 3, 5, 6, 8)
    for t in (2, 3, 4)
    if t * max(2 * nb, 4 if nb else 1) <= 64
]


def test_rows_tokens_policy():
    """Two tokens per cluster at 193-384 rows; at 385-512 rows only from 6 saved
    blocks; one token elsewhere; every picked count has a compiled variant."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.prepared import (  # noqa: E501
        rows_tokens,
        token_counts,
    )

    for nb in range(MAX_BLOCKS + 1):
        assert rows_tokens(128, nb) == 1 and rows_tokens(192, nb) == 1
        assert rows_tokens(256, nb) == 2 and rows_tokens(384, nb) == 2
        assert rows_tokens(512, nb) == (2 if nb >= 6 else 1)
        assert rows_tokens(1024, nb) == 1
        assert token_counts(nb) == {2}


@gpu
@pytest.mark.parametrize("final", [False, True], ids=["mix", "final"])
@pytest.mark.parametrize("nb,tokens", CASES)
def test_rows_bitwise_vs_one_token(nb, tokens, final):
    if final and nb == 0:
        pytest.skip("the final AttnRes has saved blocks")
    rows = 509  # not a multiple of any token count: the last cluster is partial
    x = _inputs(rows, seed=nb * 10 + tokens)
    write = -1 if final or nb == MAX_BLOCKS else nb
    ref = _run(nb, write, final, 1, x)
    got = _run(nb, write, final, tokens, x)
    for a, b, what in zip(got, ref, ("output", "prefix", "blocks")):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), what


@gpu
@pytest.mark.parametrize("nb,tokens", [(4, 4), (6, 2), (6, 3), (8, 2), (8, 4)])
def test_rows_vs_stock(nb, tokens):
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res

    if not current_platform.is_device_capability_family(100):
        pytest.skip("stock NVIDIA AttnRes: SM100 family")
    rows = 512
    x = _inputs(rows, seed=nb)
    write = nb if nb < MAX_BLOCKS else -1
    out, prefix, blocks = _run(nb, write, False, tokens, x)
    s_prefix, s_blocks = x["prefix"].clone(), x["blocks"].clone()
    expected = attn_res(
        s_prefix, x["delta"], s_blocks, x["norm"], x["query"], x["out_norm"],
        nb, write, EPS, EPS,
    )  # fmt: skip
    torch.testing.assert_close(out, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, s_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, s_blocks, atol=0, rtol=0)
