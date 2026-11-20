# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-batched MXFP4 decode indexer logits (dsa.fp4_index_logits) against the
stock DeepGEMM varlen logits (one sequence per verify row), on DCP-local row lengths
(interleave 1: rows of a request may share a local length)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability_family(100)
         and has_deep_gemm()),
    reason="needs SM100-family DeepGEMM",
)

PAGE, ROW, HEADS, DIM = 64, 68, 32, 128


def _case(reqs, q, ctx, dcp_rank, world=4, seed=0):
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, rows = "cuda", reqs * q
    glob = torch.randint(ctx // 2, ctx, (reqs,), device=dev, generator=g)
    glob = glob[:, None] + torch.arange(q, device=dev)[None, :]  # verify rows L..L+q-1
    lens = ((glob - dcp_rank + world - 1) // world).clamp(min=0).int().reshape(rows)
    pages = -(-(ctx + q + world) // world // PAGE) + 1  # fixed per (ctx, q): graph test
    nb = reqs * pages + 3
    gap = 2 * PAGE * ROW  # other layers' pages between two of ours (packed cache)
    store = torch.randint(0, 128, (nb, PAGE * ROW + gap), dtype=torch.uint8, device=dev,
                          generator=g)
    cache = store[:, : PAGE * ROW].view(nb, PAGE, ROW)
    bt = torch.randperm(nb, device=dev, generator=g)[: reqs * pages].int().view(reqs, pages)
    bt_rows = bt.repeat_interleave(q, 0).contiguous()
    query = (torch.randint(0, 256, (rows, HEADS, DIM // 2), dtype=torch.uint8, device=dev,
                           generator=g) & 0x77).contiguous()
    q_scale = torch.randint(0x7A, 0x80, (rows, HEADS, 4), dtype=torch.uint8, device=dev,
                            generator=g).view(torch.int32).view(rows, HEADS).contiguous()
    head_w = torch.randn(rows, HEADS, device=dev, generator=g).bfloat16()
    kv = kv_cache_as_quant_view(cache, DIM, True)
    ctx_rows = lens.view(rows, 1)
    idx = torch.arange(rows, dtype=torch.int32, device=dev)
    sched = get_paged_mqa_logits_metadata(ctx_rows, PAGE, get_num_sms(), idx)
    qv = query.view(torch.int8).view(rows, 1, HEADS, DIM // 2)
    qs, w = mqa_logits_mx_inputs(qv, q_scale.view(rows, 1, HEADS), head_w)
    max_model = pages * PAGE
    ref = fp8_fp4_paged_mqa_logits((qv, qs), kv, w, ctx_rows, bt_rows, sched, max_model,
                                   indices=idx)
    return dict(query=query, q_scale=q_scale, head_w=head_w, kv_cache=cache,
                row_lens=ctx_rows, block_table=bt_rows, max_len=max_model), ref, lens


def _check(got, ref, lens):
    col = torch.arange(ref.shape[1], device=ref.device)
    valid = col[None, :] < lens[:, None]
    assert got.shape[0] == ref.shape[0] and got.shape[1] >= ref.shape[1]
    assert got.stride(1) == 1 and (got.stride(0) * 2) % 1024 == 0
    g, r = got[:, : ref.shape[1]].float(), ref.float()
    assert bool(torch.isfinite(g[valid]).all())
    # BF16 logits of FP32 head sums taken in another order: within 2 BF16 ulps.
    tol = 2 * 2.0**-7 * r.abs().clamp(min=2.0**-20)
    bad = ((g - r).abs() > tol) & valid
    assert int(bad.sum()) <= int(valid.sum()) // 10000, int(bad.sum())


@pytest.mark.parametrize("reqs,q,ctx,rank", [(32, 4, 9000, 0), (96, 4, 6000, 3), (8, 2, 700, 1)])
def test_fp4_index_logits_matches_stock(reqs, q, ctx, rank):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.fp4_index_logits import (  # noqa: E501
        Fp4IndexLogits,
    )

    args, ref, lens = _case(reqs, q, ctx, rank)
    op = Fp4IndexLogits(reqs * q, q, "cuda")
    _check(op(**args), ref, lens)


def test_fp4_index_logits_graph_replay():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.fp4_index_logits import (  # noqa: E501
        Fp4IndexLogits,
    )

    args, ref, lens = _case(48, 4, 5000, 2, seed=1)
    op = Fp4IndexLogits(48 * 4, 4, "cuda")
    op(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args)
    # new lengths and keys under the same addresses: the schedule re-derives itself
    args2, ref2, lens2 = _case(48, 4, 5000, 2, seed=2)
    for k in ("query", "q_scale", "head_w", "row_lens", "block_table"):
        args[k].copy_(args2[k])
    args["kv_cache"].copy_(args2["kv_cache"])
    graph.replay()
    torch.cuda.synchronize()
    _check(out, ref2, lens2)
