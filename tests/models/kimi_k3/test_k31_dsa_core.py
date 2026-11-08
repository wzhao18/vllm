# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA sparse core (dsa_kernels.dsa_core) on one GPU (W = 1: every gathered head is
owned by this rank): the pushed partial rows and base-2 LSEs vs an FP32 reference
over the selected FP8 KV rows, for every split count and heads per CTA, with empty
rows, partial and full (2048) local lists, and the mailbox sentinel rules (no
negative zero in a pushed pair, EMPTY LSE for rows without entries); DCP16 emulated
on one GPU: the merged rows against the trtllm-gen chain, and every rank's pushed
partial rows elementwise against the stock trtllm-gen sparse MLA decode of the same
rank (its precision: E4M3 P at 448 x the running max of 128-key tiles, l from the
unquantized P, BF16 partials of trtllm-gen's KV CTA pieces), and the decode policy.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100-family kernel (mma.sync e4m3, clusters)",
)

H, QH, D, NOPE = 96, 128, 576, 512
EMPTY = -3.0e38


def _case(rows, counts, seed=0, nkv=40000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(nkv, D, generator=g, device="cuda") * 0.5
    k_scale = float(kv.abs().max()) / 448.0
    kv8 = (kv / k_scale).to(torch.float8_e4m3fn)
    q = torch.randn(rows, H, D, generator=g, device="cuda")
    q_scale = float(q.abs().max()) / 448.0
    q8 = (q / q_scale).to(torch.float8_e4m3fn)
    box = torch.zeros(rows, QH, D, dtype=torch.uint8, device="cuda")
    box[:, :H] = q8.view(torch.uint8)
    sel = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
    for r, n in enumerate(counts):
        if n:
            sel[r, :n] = torch.randperm(nkv, generator=g, device="cuda")[:n].int()
    cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
    return box, sel, cnt, kv8, q8, q_scale, k_scale


def _reference(q8, kv8, sel, counts, q_scale, k_scale, sm_scale):
    out, lse = [], []
    for r, n in enumerate(counts):
        if n == 0:
            out.append(torch.zeros(H, NOPE, device="cuda"))
            lse.append(torch.full((H,), EMPTY, device="cuda"))
            continue
        k = kv8[sel[r, :n].long()].float() * k_scale
        s = (q8[r].float() * q_scale) @ k.T * sm_scale  # [H, n], natural units
        out.append(torch.softmax(s, -1) @ k[:, :NOPE])
        lse.append(torch.logsumexp(s, -1) / math.log(2))
    return torch.stack(out), torch.stack(lse)


@pytest.mark.parametrize(
    "splits,heads_per_cta", [(s, h) for h in (16, 32) for s in (1, 2, 4, 8)]
)
def test_dsa_core_push(splits, heads_per_cta):
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore

    counts = [0, 1, 63, 64, 65, 130, 517, 2048]
    rows, cap = len(counts), 8
    box, sel, cnt, kv8, q8, q_scale, k_scale = _case(rows, counts)
    sm_scale = 1.0 / math.sqrt(192)
    core = DsaCore(torch.device("cuda"), cap=cap, heads=H, max_splits=8)
    # DcpBox layout of one rank: partial [2, 1, cap, H, 512] bf16, LSE [2, 1, cap, H]
    part = torch.zeros(2, 1, cap, H, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(2, 1, cap, H, dtype=torch.float32, device="cuda")
    peers = torch.tensor([part.data_ptr()], dtype=torch.int64, device="cuda")
    lpeers = torch.tensor([lse.data_ptr()], dtype=torch.int64, device="cuda")
    ref_o, ref_l = _reference(q8, kv8, sel, counts, q_scale, k_scale, sm_scale)
    for parity in (0, 1, 0):  # (repeat: no state survives a launch)
        part.view(torch.int32).fill_(-0x80000000)
        lse.view(torch.int32).fill_(-1)
        core.run(
            box,
            sel,
            cnt,
            kv8.view(torch.uint8),
            bmm1_scale=sm_scale * q_scale * k_scale,
            bmm2_scale=k_scale,
            part_peers=peers,
            lse_peers=lpeers,
            box_cap=cap,
            hpr=H,
            parity=parity,
            rank=0,
            rows=rows,
            splits=splits,
            heads_per_cta=heads_per_cta,
        )
        torch.cuda.synchronize()
        core.check()
        o = part[parity, 0, :rows].float()
        lv = lse[parity, 0, :rows]
        # every word written; no negative zero (0x8000 halves) in a pushed pair
        halves = part[parity, 0, :rows].view(torch.int16)
        assert not bool((halves == -0x8000).any())
        assert bool(torch.isfinite(o).all())
        for r, n in enumerate(counts):
            if n == 0:
                assert bool((o[r] == 0).all()) and bool((lv[r] == EMPTY).all())
                continue
            err = (o[r] - ref_o[r]).abs().max() / ref_o[r].abs().max()
            # (P in E4M3, as the stock FP8 MLA decode)
            assert err < 4e-2, (r, n, float(err))
            assert torch.allclose(lv[r], ref_l[r], rtol=1e-3, atol=2e-3), (r, n)
        # the other parity is never written
        other = part[1 - parity].view(torch.int32)
        assert bool((other == -0x80000000).all())


# --- DCP16 on one GPU: 16 emulated ranks (the core's 16 pushes to the head owners'
# boxes per rank), as PreparedDsa runs it (dsa_core.decode).
W, HPR, CAP = 16, 6, 64


def _dcp_case(rows, pattern, seed=1, nkv=1 << 16):
    """Every rank's selection of each row: 2048 keys spread over the 16 ranks'
    disjoint KV rows ("balanced"), or row 0 entirely on rank 3 (a saturated rank,
    count 0 on the others), row 1 on no rank (an empty row) and row 2 on two ranks
    only ("skewed"), or row r's keys on 1 + r % 8 ranks ("long": local counts of
    256-2048, 2-16 tiles per rank-row: every trtllm-gen piece size at 8-16 rows)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(nkv, D, generator=g, device="cuda") * 0.5
    k_scale = float(kv.abs().max()) / 448.0
    kv8 = (kv / k_scale).to(torch.float8_e4m3fn)
    q = torch.randn(rows, H, D, generator=g, device="cuda")
    q_scale = float(q.abs().max()) / 448.0
    q8 = (q / q_scale).to(torch.float8_e4m3fn)
    box = torch.zeros(CAP, QH, D, dtype=torch.uint8, device="cuda")
    box[:rows, :H] = q8.view(torch.uint8)
    owner = torch.arange(nkv, device="cuda") % W  # interleave 1
    sel = torch.full((W, rows, 2048), -1, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(W, rows, dtype=torch.int32, device="cuda")
    for r in range(rows):
        keys = torch.randperm(nkv, generator=g, device="cuda")[:2048]
        if pattern == "skewed" and r == 0:
            keys = torch.randperm(nkv // W, generator=g, device="cuda")[:2048] * W + 3
        elif pattern == "skewed" and r == 1:
            keys = keys[:0]
        elif pattern == "skewed" and r == 2:
            keys = keys[(owner[keys] == 5) | (owner[keys] == 11)]
        elif pattern == "long":
            ranks = 1 + r % 8
            pool = torch.randperm(nkv // W, generator=g, device="cuda")[:2048] * W
            keys = pool + (torch.arange(2048, device="cuda") % ranks + r) % W
        for rank in range(W):
            mine = keys[owner[keys] == rank].int()
            sel[rank, r, : mine.numel()] = mine
            cnt[rank, r] = mine.numel()
    return box, sel, cnt, kv8, q8, q_scale, k_scale


def _union_reference(q8, kv8, sel, cnt, q_scale, k_scale, sm_scale):
    """FP32 attention of every gathered head over the union of the ranks' rows."""
    out = []
    for r in range(q8.shape[0]):
        keys = torch.cat([sel[w, r, : int(cnt[w, r])] for w in range(W)]).long()
        if keys.numel() == 0:
            out.append(torch.zeros(H, NOPE, device="cuda"))
            continue
        k = kv8[keys].float() * k_scale
        s = (q8[r].float() * q_scale) @ k.T * sm_scale
        out.append(torch.softmax(s, -1) @ k[:, :NOPE])
    return torch.stack(out)


def _lse_merge(rows, lses, live):
    """[W, T, H, 512] partials and [W, T, H] base-2 LSEs merged over the sources
    ``live`` [W, T] (DcpValue's rule: the others are not read; trtllm-gen leaves
    garbage, even NaN, in a row of count 0)."""
    lses = torch.where(live[:, :, None], lses, torch.full_like(lses, -float("inf")))
    rows = torch.where(live[:, :, None, None], rows.float(), 0.0)
    top = lses.amax(0).clamp(min=-1e30)
    weights = torch.exp2(lses - top)
    weights = weights / weights.sum(0).clamp(min=1e-30)
    return (weights[..., None] * rows).sum(0)


def _core_dcp16(core, box, sel, cnt, kv8, scales, variant, parity=0, pdl=False):
    """The 16 ranks' cores pushing into 16 owner boxes; the merged rows of every
    gathered head [T, H, 512] and every rank's pushed rows [W, T, H, 512] (bf16) and
    base-2 LSEs [W, T, H]."""
    rows = sel.shape[1]
    part = [
        torch.zeros(2, W, CAP, HPR, 512, dtype=torch.bfloat16, device="cuda")
        for _ in range(W)
    ]
    lse = [torch.zeros(2, W, CAP, HPR, device="cuda") for _ in range(W)]
    for t in part:
        t.view(torch.int32).fill_(-0x80000000)
    for t in lse:
        t.view(torch.int32).fill_(-1)
    peers = torch.tensor([t.data_ptr() for t in part], device="cuda")
    lpeers = torch.tensor([t.data_ptr() for t in lse], device="cuda")
    splits, heads, split_min = variant
    for rank in range(W):
        core.run(box, sel[rank], cnt[rank], kv8.view(torch.uint8),
                 bmm1_scale=scales[0], bmm2_scale=scales[1], part_peers=peers,
                 lse_peers=lpeers, box_cap=CAP, hpr=HPR, parity=parity, rank=rank,
                 rows=rows, splits=splits, split_min=split_min,
                 heads_per_cta=heads, pdl=pdl)  # fmt: skip
    torch.cuda.synchronize()
    core.check()
    # owner o's box holds gathered heads o * HPR .. from every source s
    rows_ = torch.stack([p[parity, :, :rows] for p in part], 2)  # [W, T, 16, HPR, 512]
    lses = torch.stack([v[parity, :, :rows] for v in lse], 2)  # [W, T, 16, HPR]
    rows_ = rows_.reshape(W, rows, H, 512)
    lses = lses.reshape(W, rows, H)
    assert not bool((lses.view(torch.int32) == -1).any())  # every LSE pushed
    live = lses > EMPTY
    return _lse_merge(rows_, lses, live.any(-1)), rows_, lses


def _trtllm_dcp16(box, sel, cnt, kv8, scales, raw=False):
    """The trtllm-gen chain per rank (attn_fi.trtllm's call) merged by LSE over the
    ranks with entries (DcpValue drops rows of count 0); ``raw``: also every rank's
    rows [W, T, H, 512] (bf16) and base-2 LSEs [W, T, H]."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    dev = torch.device("cuda")
    rows = sel.shape[1]
    fims._reserve_kv_counters(dev, CAP, QH)
    pages = kv8.view(-1, 64, D)
    outs, lses = [], []
    for rank in range(W):
        out = torch.empty(rows, 1, QH, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, QH, device=dev)
        extra = fims._kv_counters(rows, QH, dev)
        extra["lse"] = lse
        trtllm_batch_decode_with_kv_cache_mla(
            query=box[:rows].view(torch.float8_e4m3fn).unsqueeze(1),
            kv_cache=pages.unsqueeze(1),
            workspace_buffer=fims._get_workspace_buffer(dev),
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=sel[rank].unsqueeze(1),
            seq_lens=cnt[rank],
            max_seq_len=2048,
            out=out,
            bmm1_scale=scales[0],
            bmm2_scale=scales[1],
            sparse_mla_top_k=2048,
            return_lse=True,
            **extra,
        )
        outs.append(out[:, 0, :H])
        lses.append(lse[:, :H])
    torch.cuda.synchronize()
    merged = _lse_merge(torch.stack(outs), torch.stack(lses), cnt > 0)
    return (merged, torch.stack(outs), torch.stack(lses)) if raw else merged


def _error(x, ref):
    return float((x.float() - ref).norm() / ref.norm().clamp(min=1e-30))


@pytest.mark.parametrize("pattern", ["balanced", "skewed"])
@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("variant", ["2x16s2", "1x16", "8x16s4", "4x32s2"])
def test_dsa_core_dcp16_merge_vs_trtllm(variant, rows, pattern):
    """DCP16 on one GPU (16 emulated ranks, interleave 1, q8 rows of 96 gathered
    heads): the 16 ranks' cores push their partial rows and base-2 LSEs into the
    owners' boxes, and the LSE merge of every gathered head over the sources is as
    close to FP32 attention over the union of the ranks' rows as the trtllm-gen
    chain's (attn_fi.trtllm's call, DcpValue's merge rule), including a row whose
    2048 entries sit on one rank (count 0 on the other 15), a row on no rank and a
    row on two ranks."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.decode import parse_variant
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore

    box, sel, cnt, kv8, q8, q_scale, k_scale = _dcp_case(rows, pattern)
    sm_scale = 1.0 / math.sqrt(192)
    scales = sm_scale * q_scale * k_scale, k_scale
    core = DsaCore(torch.device("cuda"), cap=CAP, heads=H, max_splits=8)
    ours, _, _ = _core_dcp16(core, box, sel, cnt, kv8, scales, parse_variant(variant))
    theirs = _trtllm_dcp16(box, sel, cnt, kv8, scales)
    ref = _union_reference(q8, kv8, sel, cnt, q_scale, k_scale, sm_scale)
    live = cnt.sum(0) > 0
    if pattern == "skewed":  # the row on no rank merges to zeros
        assert bool((ours[~live] == 0).all())
    ours_error, trtllm_error = (
        _error(ours[live], ref[live]),
        _error(theirs[live], ref[live]),
    )
    print(f"vs FP32 over the union: core {ours_error:.4f} trtllm {trtllm_error:.4f}")
    assert ours_error <= 1.1 * trtllm_error + 1e-3, (ours_error, trtllm_error)


@pytest.mark.parametrize("variant", ["2x16s2", "8x16s4"])
def test_dsa_core_graph_replay_alternates_parities(variant):
    """Two DSA layers on alternating DcpBox parities (each its own q mailbox and
    selection, with saturated, empty and skewed rows) captured in one CUDA graph
    with PDL launches, replayed 100 times: every replay pushes the eager launches'
    rows and LSEs bit for bit into the 16 owners' boxes, and a layer writes only its
    own parity, its rank's source slots and its rows."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.decode import parse_variant
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore

    rows, (splits, heads, split_min) = 16, parse_variant(variant)
    layers = [_dcp_case(rows, "skewed", seed=s) for s in (3, 4)]
    q_box = torch.stack([layer[0] for layer in layers])  # [2 parities, CAP, QH, D]
    kv8 = layers[0][3]
    scales = 1.0 / math.sqrt(192) * layers[0][5] * layers[0][6], layers[0][6]
    core = DsaCore(torch.device("cuda"), cap=CAP, heads=H, max_splits=8)
    shape = (2, W, CAP, HPR)
    part = torch.zeros(W, *shape, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(W, *shape, device="cuda")
    peers = torch.tensor([p.data_ptr() for p in part], device="cuda")
    lpeers = torch.tensor([v.data_ptr() for v in lse], device="cuda")
    rank = 5

    def run(parity):
        _, sel, cnt, *_ = layers[parity]
        core.run(q_box[parity], sel[rank], cnt[rank], kv8.view(torch.uint8),
                 bmm1_scale=scales[0], bmm2_scale=scales[1], part_peers=peers,
                 lse_peers=lpeers, box_cap=CAP, hpr=HPR, parity=parity, rank=rank,
                 rows=rows, splits=splits, split_min=split_min, heads_per_cta=heads,
                 pdl=True)  # fmt: skip

    def arm():
        part.view(torch.int32).fill_(-0x80000000)
        lse.view(torch.int32).fill_(-1)

    arm()
    run(0)
    torch.cuda.synchronize()
    core.check()
    written = lse.view(torch.int32) != -1  # [owner, parity, source, row, head]
    assert bool(written[:, 0, rank, :rows].all())  # every owner, rank 5's rows
    assert int(written.sum()) == W * rows * HPR  # nothing else, parity 1 untouched
    run(1)
    torch.cuda.synchronize()
    eager = part.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture):
        arm()
        run(0)
        run(1)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    core.check()
    assert torch.equal(part.view(torch.int16), eager[0].view(torch.int16))
    assert torch.equal(lse.view(torch.int32), eager[1].view(torch.int32))


LOG2E = math.log2(math.e)


def _emulate(q8, k8, bmm1, bmm2, kv_ctas):
    """The stock trtllm-gen sparse MLA decode of one row (FP32 emulation; probed bitwise
    against trtllm-gen on VR200 at rows 1-64, local counts 100-2048): its KV CTA split
    (``kv_ctas`` = ops.trtllm_kv_ctas of the batch), the row's T = ceil(n / 128) tiles
    in pieces of k = ceil(T / kv_ctas) consecutive tiles. Per piece, one base-2 running
    max step per tile, P = E4M3(448 x 2^(s - m)), the piece's P V and sum (from the
    unquantized P) rescaled by 2^(m_old - m), its unnormalized partial P V x bmm2 / 448
    rounded to BF16 (kv_ctas > 1); merged in FP32 with weights 2^(m_p - M), divided by
    L = sum l_p 2^(m_p - M), rounded to BF16. q8 [H, 576] and k8 [n, 576] are E4M3
    values (float). (rows bf16 [H, 512], base-2 LSE [H])."""
    s = (q8 @ k8.T) * (bmm1 * LOG2E)
    tiles = (k8.shape[0] + 127) // 128
    per = -(-tiles // kv_ctas)
    parts = []
    for p0 in range(0, tiles, per):
        m = torch.full((s.shape[0],), -float("inf"), device=s.device)
        acc = torch.zeros(s.shape[0], NOPE, device=s.device)
        l = torch.zeros(s.shape[0], device=s.device)
        for t in range(p0, min(tiles, p0 + per)):
            st = s[:, 128 * t : 128 * t + 128]
            mn = torch.maximum(m, st.amax(1))
            a = torch.exp2(m - mn)
            p = torch.exp2(st - mn[:, None])
            p8 = (p * 448.0).to(torch.float8_e4m3fn).float()
            acc = acc * a[:, None] + p8 @ k8[128 * t : 128 * t + 128, :NOPE]
            l = l * a + p.sum(1)
            m = mn
        x = acc * (bmm2 / 448.0)
        if kv_ctas > 1:
            x = x.to(torch.bfloat16).float()
        parts.append((x, m, l))
    M = torch.stack([m for _, m, _ in parts]).amax(0)
    L = sum(l * torch.exp2(m - M) for _, m, l in parts)
    o = sum(x * torch.exp2(m - M)[:, None] for x, m, _ in parts) / L[:, None]
    return o.to(torch.bfloat16), M + torch.log2(L)


def _ulps(a, b):
    """Elementwise distance of two bf16 tensors in units of bf16 ulps (of the
    larger magnitude, at least 2^-8 of the tensor's largest: values that cancel to
    near zero are compared at that scale); 0 where both are equal."""
    a, b = a.float(), b.float()
    floor = float(torch.maximum(a.abs().max(), b.abs().max())) * 2.0**-8
    scale = torch.maximum(a.abs(), b.abs()).clamp(min=max(floor, 1e-30))
    ulp = torch.exp2(torch.floor(torch.log2(scale)) - 7)
    return (a - b).abs() / ulp


@pytest.mark.parametrize("pattern", ["balanced", "skewed", "long"])
@pytest.mark.parametrize("rows", [8, 14, 16])
def test_dsa_core_matches_trtllm_per_rank(rows, pattern):
    """Stock precision, rank by rank (DCP16 q8 rows 8 / 14 / 16: trtllm-gen KV CTAs 8 /
    7 / 6 on VR200, the served 2x16s2): every rank's pushed partial rows and base-2
    LSEs against the stock trtllm-gen sparse MLA decode of the same rank's rows
    (attn_fi.trtllm's call), and both against the FP32 emulation of trtllm-gen's
    precision (``_emulate``), with local counts of up to 2048 (every piece size of
    trtllm-gen's KV CTA split at these rows). The old FP16-P core matched trtllm-gen
    on 53-68% of the elements."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.decode import (
        VARIANT,
        parse_variant,
    )
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore, trtllm_kv_ctas

    torch.backends.cuda.matmul.allow_tf32 = False
    variant = parse_variant(VARIANT[8].split(":")[1])
    box, sel, cnt, kv8, q8, q_scale, k_scale = _dcp_case(rows, pattern)
    sm_scale = 1.0 / math.sqrt(192)
    scales = sm_scale * q_scale * k_scale, k_scale
    core = DsaCore(torch.device("cuda"), cap=CAP, heads=H, max_splits=8)
    kv_ctas = trtllm_kv_ctas(rows, QH, 2048, core.sms)
    _, ours, our_lse = _core_dcp16(core, box, sel, cnt, kv8, scales, variant)
    _, theirs, their_lse = _trtllm_dcp16(box, sel, cnt, kv8, scales, raw=True)
    live = cnt > 0  # [W, T]
    a, b = ours[live], theirs[live].to(torch.bfloat16)  # [N, H, 512]
    equal = float((a.view(torch.int16) == b.view(torch.int16)).float().mean())
    ulps = _ulps(a, b)
    lse_err = float((our_lse[live] - their_lse[live]).abs().max())
    # the emulation of trtllm-gen's precision per row, against both
    kf, qf = kv8.float(), q8.float()
    emu_eq: dict = {"trtllm": [], "core": []}
    for w in range(W):
        for t in range(rows):
            n = int(cnt[w, t])
            if n == 0:
                continue
            e, _ = _emulate(qf[t], kf[sel[w, t, :n].long()], *scales, kv_ctas)
            e = e.view(torch.int16)
            tr = theirs[w, t].to(torch.bfloat16).view(torch.int16)
            emu_eq["trtllm"].append(float((e == tr).float().mean()))
            emu_eq["core"].append(float((e == ours[w, t].view(torch.int16)).float().mean()))
    summary = {k: round(sum(v) / len(v), 5) for k, v in emu_eq.items()}
    counts = cnt[live]
    p999 = float(ulps.flatten()[::4].quantile(0.999))  # (quantile: < 16M elements)
    print(
        f"DCP16 q8 M{rows} kv_ctas {kv_ctas} {pattern} (local counts {int(counts.min())}-"
        f"{int(counts.max())}): core vs trtllm-gen bitwise {equal:.5f}, ulps max "
        f"{float(ulps.max()):.1f} p99.9 {p999:.2f}, LSE max |diff| {lse_err:.2e}; "
        f"emulation bitwise vs {summary}"
    )
    assert summary["core"] >= 0.999, summary
    assert summary["trtllm"] >= 0.999, summary
    assert equal >= 0.999, equal
    assert p999 <= 1.0, p999
    assert lse_err <= 1e-4, lse_err


def test_dsa_core_serving_deadline():
    """The served core's q mailbox poll deadline (build()): 600 s, as DcpExchange's
    serving one, so a peer stalled on the host does not trap it."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode as dcpx
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

    assert decode.SERVING_SPIN_S == dcpx.SERVING_SPIN_S == 600.0


def test_trtllm_kv_ctas():
    """trtllm-gen's KV CTAs per row as probed on VR200 (212 SMs, 128 padded heads,
    2048 slots per row) and FlashInfer's rule elsewhere."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import trtllm_kv_ctas

    probed = {1: 8, 2: 8, 4: 8, 8: 8, 12: 8, 16: 6, 24: 4, 32: 3, 48: 2, 56: 1, 64: 1}
    for rows, ctas in probed.items():
        assert trtllm_kv_ctas(rows, 128, 2048, 212) == ctas, rows
    assert trtllm_kv_ctas(14, 128, 2048, 212) == 7
    assert trtllm_kv_ctas(8, 128, 2048, 148) == 8
    assert trtllm_kv_ctas(16, 128, 2048, 148) == 4
    assert trtllm_kv_ctas(1, 128, 512, 212) == 2
    with pytest.raises(ValueError):
        trtllm_kv_ctas(8, 32, 2048, 212)


def test_dsa_core_policy():
    """The core's tables (no switch): by q, then rows (a q without a table and rows
    past ROWS keep FlashInfer), split_min in the variant name; a test table for
    every q."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

    assert decode.parse_variant("2x16s2") == (2, 16, 2)
    assert decode.parse_variant("1x32") == (1, 32, 4)
    for bad in ("u", "3x16", "2x8", "x16", "2x16s0"):
        with pytest.raises(ValueError):
            decode.parse_variant(bad)
    dsac = decode.DsacDecode(torch.device("cuda"), H)
    assert dsac.rows == decode.ROWS == 16
    assert dsac.variant(8, 8) == dsac.variant(16, 8) == (2, 16, 2)
    assert dsac.variant(8, 4) is None  # no q4 table
    assert dsac.variant(24, 8) is None  # past ROWS
    every = decode.DsacDecode(torch.device("cuda"), H, rows=64, spec="1-8:1x16;9-64:4x32")
    assert every.variant(8, 4) == (1, 16, 4) and every.variant(64, 8) == (4, 32, 4)


def _model(kv_dtype="fp8", dcp=16):
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention

    layer = KimiK3DSAAttention.__new__(KimiK3DSAAttention)
    torch.nn.Module.__init__(layer)
    layer.dsa_indexer = SimpleNamespace(dcp_world_size=dcp)
    layer.mla_attn = SimpleNamespace(mla_attn=SimpleNamespace(kv_cache_dtype=kv_dtype))
    return torch.nn.ModuleList([layer])


@pytest.mark.parametrize(
    "kv_dtype,dcp,sp,q,reason",
    [
        ("fp8", 16, False, 8, None),  # C1 / C8 / C48 / C64: the core runs
        ("fp8", 16, False, 4, "verify length 4"),  # C160 / C256: no q4 table
        ("fp8", 4, True, 4, "sequence-parallel"),  # C512: SpDsaCore
        ("fp8", 1, False, 8, "no DCP"),
        ("nvfp4_ds_mla", 16, False, 8, "nvfp4_ds_mla"),  # capability: FP8 rows only
    ],
)
def test_dsa_core_build_policy(kv_dtype, dcp, sp, q, reason):
    """``unsupported``: the core is on wherever it can run (no switch); every other
    deployment keeps FlashInfer's chain, and build() then attaches nothing."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

    model = _model(kv_dtype, dcp)
    config = SimpleNamespace(sequence_parallel=sp, q=q)
    why = decode.unsupported(model, config)
    if reason is None:
        assert why is None
    else:
        assert why is not None and reason in why, why
        assert decode.build(model, config) is None
        assert getattr(model[0].dsa_indexer, "dsac", None) is None


def test_dsa_core_rows_source():
    """The core's rows (MergeRows): the #63 merge's steps and DcpExchange's steps
    (its top-k launches without PDL: the core polls its q tiles before its PDL
    wait)."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

    class Stage:  # IndexerStage's step predicate and rows
        source: tuple = (None, 0)

        def merge_source(self, batch):
            return self.source

        def merge_rows(self, *args):
            return None if self.source[0] is None else ("sel", "cnt")

    stage = Stage()
    source = decode.MergeRows(stage)
    for step, fits in (
        ((None, 0), False),
        (("merge", 8), True),
        (("exchange", 8), True),
        (("exchange", 64), True),
    ):
        stage.source = step
        assert source.fits(64) is fits, step
    assert source.rows(None, None, None, None, 64, q=8) == ("sel", "cnt")
