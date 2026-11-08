# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA wide sparse core (specialized_decode/cutedsl/dsa/wide_core) against the
stock FlashInfer trtllm-gen sparse MLA decode + _finish_dcp_partials on one GPU:
the gathered q mailbox (96 real of 128 heads), per-tensor FP8 KV rows, DCP16 and DCP4
local counts (empty rows, 1 key, partial and full 128-key tiles, up to 2048), and
the persistent schedule (more items than SMs). Same math, accumulation order aside:
the BF16 latents must agree bit for bit on nearly every element and the LSEs to
FP32 rounding."""

import math

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100-family kernel (tcgen05)",
)

H, QH, D, NOPE, TOPK = 96, 128, 576, 512, 2048


def _case(counts, seed, nkv=1 << 16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = len(counts)
    kv = torch.randn(nkv, D, generator=g, device="cuda")
    k_scale = float(kv.abs().max()) / 448.0
    kv8 = (kv / k_scale).to(torch.float8_e4m3fn)
    q = torch.randn(rows, H, D, generator=g, device="cuda") * 2.0
    q_scale = float(q.abs().max()) / 448.0
    box = torch.zeros(rows, QH, D, dtype=torch.uint8, device="cuda")
    box[:, :H] = (q / q_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    sel = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    for r, n in enumerate(counts):
        if n:
            sel[r, :n] = torch.randperm(nkv, generator=g, device="cuda")[:n].int()
    cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
    bmm1 = q_scale * k_scale / math.sqrt(192)
    return box, sel, cnt, kv8, bmm1, k_scale


def _stock(box, sel, cnt, kv8, bmm1, bmm2):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    rows, dev = box.shape[0], box.device
    fims._reserve_kv_counters(dev, max(rows, 64), QH)
    out = torch.empty(rows, 1, QH, NOPE, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(rows, QH, dtype=torch.float32, device=dev)
    workspace = fims._get_workspace_buffer(
        dev, fims._required_workspace_bytes(16, 6, max(rows, 2048))
    )
    trtllm_batch_decode_with_kv_cache_mla(
        query=box.view(torch.float8_e4m3fn).unsqueeze(1),
        kv_cache=kv8.view(-1, 64, D).unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=sel.unsqueeze(1),
        seq_lens=cnt,
        max_seq_len=TOPK,
        out=out,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
        sparse_mla_top_k=TOPK,
        return_lse=True,
        lse=lse,
        **fims._kv_counters(rows, QH, dev),
    )
    fims._finish_dcp_partials(out[:, 0, :H], lse[:, :H], cnt)
    return out[:, 0, :H].clone(), lse[:, :H].clone()


def _wide(box, sel, cnt, kv8, bmm1, bmm2):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide_core import (
        WideCore,
    )

    rows = box.shape[0]
    # sentinel-filled outputs: every real-head word must be written
    out = torch.full((rows, QH, NOPE), float("nan"), dtype=torch.bfloat16, device="cuda")
    lse = torch.full((rows, QH), float("nan"), device="cuda")
    WideCore(box.device).run(box, sel, cnt, kv8, bmm1, bmm2, out, lse, H)
    torch.cuda.synchronize()
    return out[:, :H], lse[:, :H]


def _compare(counts, seed):
    box, sel, cnt, kv8, bmm1, bmm2 = _case(counts, seed)
    so, sl = _stock(box, sel, cnt, kv8, bmm1, bmm2)
    wo, wl = _wide(box, sel, cnt, kv8, bmm1, bmm2)
    assert bool(torch.isfinite(wo.float()).all()), "unwritten / non-finite latent"
    bad = []
    for r, n in enumerate(counts):
        if n == 0:
            if not (bool((wo[r] == 0).all()) and bool(torch.isneginf(wl[r]).all())):
                bad.append((r, n, "empty row not (0, -inf)"))
            continue
        equal = float((wo[r] == so[r]).float().mean())
        rel = float((wo[r].float() - so[r].float()).abs().max() / so[r].float().abs().max())
        dl = float((wl[r] - sl[r]).abs().max())
        print(f"row {r} n {n}: bf16-equal {equal:.5f} max rel {rel:.2e} lse {dl:.2e}")
        # (an accumulation-order flip of one P rounding changes one head: 1 / 96)
        if not (equal > 0.97 and rel < 2e-2 and dl < 1e-4):
            bad.append((r, n, equal, rel, dl))
    eq = float((wo == so).float().mean())
    print(f"all rows: bf16-equal {eq:.6f}")
    assert not bad, bad
    assert eq > 0.999, eq
    return eq


def test_wide_core_counts():
    """Every tile shape: empty rows, 1 key, a partial / full chunk and tile, two
    tiles, DCP4 lengths and the whole top-k. (At least 64 rows: below ~54 rows
    trtllm-gen splits each row's KV over CTAs and rounds BF16 partials, a schedule
    the wide core does not reproduce; it serves wide rows only.)"""
    counts = [0, 1, 7, 63, 64, 65, 100, 127, 128, 129, 155, 200, 256, 300, 440, 512, 580,
              1000, 2048, 0]  # fmt: skip
    _compare(counts * 4, 0)


@pytest.mark.parametrize("rows,lo,hi", [(64, 96, 155), (512, 96, 155), (300, 440, 580),
                                         (1024, 96, 155)])  # fmt: skip
def test_wide_core_production_rows(rows, lo, hi):
    """Production row counts (DCP16 / DCP4 local lengths), more items than SMs."""
    g = torch.Generator().manual_seed(rows)
    counts = torch.randint(lo, hi + 1, (rows,), generator=g).tolist()
    counts[rows // 3] = 0
    _compare(counts, rows)


def test_wide_core_graph_replay():
    """CUDA-graph capture and replay give the eager results."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide_core import (
        WideCore,
    )

    counts = [130] * 200 + [0, 5]
    box, sel, cnt, kv8, bmm1, bmm2 = _case(counts, 9)
    core = WideCore(box.device)
    rows = len(counts)
    out = torch.zeros(rows, QH, NOPE, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(rows, QH, device="cuda")
    core.run(box, sel, cnt, kv8, bmm1, bmm2, out, lse, H)
    torch.cuda.synchronize()
    ref_o, ref_l = out.clone(), lse.clone()
    out.zero_()
    lse.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        core.run(box, sel, cnt, kv8, bmm1, bmm2, out, lse, H)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[:, :H], ref_o[:, :H])
    assert torch.equal(lse[:, :H], ref_l[:, :H])
