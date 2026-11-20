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


def _case(reqs, q, ctx, dcp_rank, world=4, seed=0, lo=None, contig=False):
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, rows = "cuda", reqs * q
    glob = torch.randint(ctx // 2 if lo is None else lo, ctx, (reqs,), device=dev, generator=g)
    glob = glob[:, None] + torch.arange(q, device=dev)[None, :]  # verify rows L..L+q-1
    lens = ((glob - dcp_rank + world - 1) // world).clamp(min=0).int().reshape(rows)
    pages = -(-(ctx + q + world) // world // PAGE) + 1  # fixed per (ctx, q): graph test
    nb = reqs * pages + 3
    gap = 2 * PAGE * ROW  # other layers' pages between two of ours (packed cache)
    store = torch.randint(0, 128, (nb, PAGE * ROW + gap), dtype=torch.uint8, device=dev,
                          generator=g)
    cache = store[:, : PAGE * ROW].view(nb, PAGE, ROW)
    order = torch.arange(nb, device=dev) if contig else torch.randperm(nb, device=dev, generator=g)
    bt = order[: reqs * pages].int().view(reqs, pages)  # contig: as large KV-manager blocks
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


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(codes, scales):
    """E2M1 bytes [..., 64] (low nibble first), UE8M0 bytes [..., 4] -> FP32 [..., 128]."""
    table = torch.tensor(E2M1, device=codes.device)
    table = torch.cat([table, -table])
    nib = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
    values = table[nib].unflatten(-1, (4, 32))
    return (values * torch.exp2(scales.float() - 127).unsqueeze(-1)).flatten(-2)


def _fp32_logits(args, lens, q):
    """FP32 reference [rows, max_len]: sum_h w_h relu(q_h . k) of the dequantized
    rows (page = [64 x 64 B E2M1 | 64 x 4 UE8M0])."""
    cache, bt = args["kv_cache"], args["block_table"]
    pages = torch.as_strided(cache, (cache.shape[0], PAGE * ROW), (cache.stride(0), 1))
    rows = lens.shape[0]
    out = torch.zeros(rows, args["max_len"], device=lens.device)
    qd = _dequant(args["query"], args["q_scale"].view(torch.uint8).view(rows, HEADS, 4))
    w = args["head_w"].float()
    for b in range(rows // q):
        n = int(lens[b * q: b * q + q].max())
        if n == 0:
            continue
        pos = torch.arange(n, device=lens.device)
        pg = pages[bt[b * q, pos // PAGE].long()]
        off = (pos % PAGE).unsqueeze(1)
        codes = pg.gather(1, off * 64 + torch.arange(64, device=lens.device))
        sc = pg.gather(1, PAGE * 64 + off * 4 + torch.arange(4, device=lens.device))
        k = _dequant(codes, sc)  # [n, 128]
        s = torch.einsum("thd,nd->thn", qd[b * q: b * q + q], k).relu()
        out[b * q: b * q + q, :n] = torch.einsum("thn,th->tn", s, w[b * q: b * q + q])
    return out


def _check(got, ref, lens, args, q):
    """``got`` (ours) and ``ref`` (stock) against the FP32 reference on every row's
    valid columns: ours no further from it than BF16 rounding of the FP32 value
    allows the stock op (same max error bound, same mean)."""
    col = torch.arange(ref.shape[1], device=ref.device)
    valid = col[None, :] < lens[:, None]
    assert got.shape[0] == ref.shape[0] and got.shape[1] >= ref.shape[1]
    assert got.stride(1) == 1 and (got.stride(0) * 2) % 1024 == 0
    exact = _fp32_logits(args, lens, q)
    g = got[:, : ref.shape[1]].float()
    assert bool(torch.isfinite(g[valid]).all())
    e_ours = (g - exact).abs()[valid]
    e_stock = (ref.float() - exact).abs()[valid]
    bound = 2.0**-8 * exact.abs()[valid] + 1e-6 * exact.abs().max()  # half a BF16 ulp
    assert bool((e_ours <= bound * 1.01 + 1e-30).all()) or (
        e_ours.max() <= 1.01 * e_stock.max() and e_ours.mean() <= 1.1 * e_stock.mean()
    ), (e_ours.max().item(), e_stock.max().item(), e_ours.mean().item(),
        e_stock.mean().item())


@pytest.mark.parametrize("reqs,q,ctx,rank", [(32, 4, 9000, 0), (96, 4, 6000, 3), (8, 2, 700, 1)])
def test_fp4_index_logits_matches_stock(reqs, q, ctx, rank):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.fp4_index_logits import (  # noqa: E501
        Fp4IndexLogits,
    )

    args, ref, lens = _case(reqs, q, ctx, rank)
    op = Fp4IndexLogits(reqs * q, q, "cuda")
    _check(op(**args), ref, lens, args, q)


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
    _check(out, ref2, lens2, args, 4)
