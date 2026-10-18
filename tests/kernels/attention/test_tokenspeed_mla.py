# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.tokenspeed_mla import (
    _plan_proportional_split_counts,
    _prepare_proportional_splits,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="TokenSpeed MLA requires CUDA"
)


@pytest.mark.parametrize("batch,query_len,bytes_", [(1, 5, 0), (8, 0, 0), (8, 5, 128)])
def test_split_planner_preserves_workspace_when_splitting_cannot_help(
    batch, query_len, bytes_
):
    query = torch.empty(batch, query_len, 96, 576, device="cuda")
    lengths = torch.zeros(batch, device="cuda", dtype=torch.int32)
    workspace = torch.empty(bytes_, device="cuda", dtype=torch.int8)
    scratch, counts, capacity = _prepare_proportional_splits(
        query, lengths, workspace, 512, 8
    )
    assert scratch is workspace
    assert counts is None and capacity is None


def test_split_planner_updates_counts_in_graph_without_overflow():
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    counts = torch.empty_like(lengths)

    def plan(budget):
        _plan_proportional_split_counts[(1,)](
            lengths, counts, 4, budget, 16, BLOCK_SIZE=128
        )

    plan(32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan(32)
    for values, expected in (
        ([0, 128, 4096, 262144], [1, 1, 1, 16]),
        ([65536] * 4, [8] * 4),
        ([3, 7, 11, 17], [3, 6, 10, 15]),
        ([0] * 4, [1] * 4),
    ):
        lengths.copy_(torch.tensor(values, dtype=torch.int32))
        graph.replay()
        assert counts.tolist() == expected
    lengths.fill_(1 << 20)
    plan(1 << 16)
    assert counts.tolist() == [16] * 4


@pytest.mark.parametrize("query_len", [1, 5])
def test_proportional_decode_reuses_workspace_during_graph_replay(query_len):
    tokenspeed = pytest.importorskip("tokenspeed_mla")
    if not current_platform.is_device_capability_family(100):
        pytest.skip("TokenSpeed MLA requires Blackwell")
    batch, heads, max_len = 8, 96, 16384
    query = torch.randn(
        batch, query_len, heads, 576, device="cuda", dtype=torch.bfloat16
    )
    query = query.to(torch.float8_e4m3fn)
    cache = torch.randn(
        batch * (max_len // 64), 64, 576, device="cuda", dtype=torch.bfloat16
    )
    cache = cache.to(torch.float8_e4m3fn)
    tables = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32).view(
        batch, -1
    )
    lengths = torch.full((batch,), max_len, device="cuda", dtype=torch.int32)
    workspace = torch.empty(256 << 20, device="cuda", dtype=torch.int8)
    out = torch.empty(batch, query_len, heads, 512, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        query=query,
        kv_cache=cache,
        block_tables=tables,
        seq_lens=lengths,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_seq_len=max_len,
        softmax_scale=192**-0.5,
        enable_pdl=False,
    )

    def decode():
        scratch, counts, capacity = _prepare_proportional_splits(
            query, lengths, workspace, 512, 8
        )
        assert counts is not None
        assert scratch.data_ptr() == workspace.data_ptr()
        assert counts.data_ptr() >= scratch.data_ptr() + scratch.numel()
        assert counts.data_ptr() % 16 == 0
        tokenspeed.tokenspeed_mla_decode(
            **kwargs,
            workspace_buffer=scratch,
            out=out,
            split_kv_counts=counts,
            max_split_kv=capacity,
        )

    decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    for values in ([129] * 7 + [16384], [8193] * batch, [65, 129, 257, 513] * 2):
        lengths.copy_(torch.tensor(values, dtype=torch.int32))
        expected = tokenspeed.tokenspeed_mla_decode(
            **kwargs,
            workspace_buffer=workspace,
            min_split_kv=8,
        ).clone()
        # The planner must refresh counts after scratch was reused by other work.
        workspace.fill_(127)
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)
