# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA sparse core (dsa_kernels.dsa_core) on one GPU (W = 1: every gathered head is
owned by this rank): the pushed partial rows and base-2 LSEs vs an FP32 reference
over the selected FP8 KV rows, for every split count and heads per CTA, with empty
rows, partial and full (2048) local lists, and the mailbox sentinel rules (no
negative zero in a pushed pair, EMPTY LSE for rows without entries). The multi-rank
path (DcpValue(pushed) merge, W_UV, gate) is benchmarks/kernels/k31_vr/dcp_dsa_core.py.
"""

import math

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
    kv = (torch.randn(nkv, D, generator=g, device="cuda") * 0.5)
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
    "splits,heads_per_cta",
    [(s, h) for h in (16, 32) for s in (1, 2, 4, 8)] + [(1, 0)],  # 0: tcgen05 variant
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
        core.run(box, sel, cnt, kv8.view(torch.uint8), bmm1_scale=sm_scale * q_scale * k_scale,
                 bmm2_scale=k_scale, part_peers=peers, lse_peers=lpeers, box_cap=cap,
                 hpr=H, parity=parity, rank=0, rows=rows, splits=splits,
                 heads_per_cta=heads_per_cta or 32, umma=heads_per_cta == 0)
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
            # (tcgen05: P in E4M3, as the stock FP8 MLA decode)
            assert err < (4e-2 if heads_per_cta == 0 else 1.5e-2), (r, n, float(err))
            assert torch.allclose(lv[r], ref_l[r], rtol=1e-3, atol=2e-3), (r, n)
        # the other parity is never written
        other = part[1 - parity].view(torch.int32)
        assert bool((other == -0x80000000).all())
