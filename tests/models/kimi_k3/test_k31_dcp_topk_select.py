# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The #63 DCP top-k select kernel (``DCPTopkSelectKernel``) on one GPU against a
torch reference of its contract: the top-K of the gathered candidates under (bf16
score desc, global id asc), written as global ids and as this rank's physical KV
rows in the kernel's fixed emission order (thread-major over the CTA's register
keys), plus the owned count. Tie-heavy rows (quantized scores) exercise the gid
radix passes; rows with fewer than K candidates keep all of them. ``dist``:
relu rows (exact zeros: score range > 512 ord16 values, the 9 + 7 bit score
passes), tail rows (each rank's top-N of 8250 logits: a narrow range, the one
score pass on ord16 - min) and rows spanning exactly 511 / 512 ord16 values
(either side of that switch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not has_cutedsl(),
    reason="The DCP top-k select needs SM100 and CuTe DSL",
)

W, N, K, BLOCK = 16, 192, 2048, 64
TB = 512  # the select's threads per CTA


def _ord16(x: torch.Tensor) -> torch.Tensor:
    bits = x.view(torch.int16).to(torch.int64) & 0xFFFF
    neg = (bits & 0x8000) != 0
    return torch.where(neg, (~bits) & 0xFFFF, bits | 0x8000)


def _case(rows: int, levels: int, seed: int, short: int = -1, dist: str = "relu"):
    """Gathered [W, rows, N + N/32] int32 words, and per (rank, row, slot) the
    candidate's ord16 and global id (-1: empty slot)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "tail":
        score = torch.randn((W, rows, 8250), generator=gen, device="cuda")
        score = score.topk(N, dim=2).values
        if levels:
            score = torch.floor(score * levels) / levels
        score = score.to(torch.bfloat16)
    elif dist.startswith("span"):
        # ord16 in [0xC000, 0xC000 + span] (bf16 2.0 .. ~4-8), both ends hit
        span = int(dist[4:])
        o = 0xC000 + torch.randint(
            0, span + 1, (W, rows, N), generator=gen, device="cuda"
        )
        o[0, :, 0] = 0xC000
        o[1, :, 0] = 0xC000 + span
        score = (o - 0x8000).to(torch.int16).view(torch.bfloat16)
    else:
        score = torch.randn((W, rows, N), generator=gen, device="cuda")
        if levels:
            score = torch.floor(score * levels) / levels
        score = torch.relu(score + 0.3).to(torch.bfloat16)
    base = torch.randint(0, 1 << 17, (W, rows, 1), generator=gen, device="cuda")
    # distinct local ids per (rank, row): a rank's candidates are distinct tokens
    local = (base + torch.arange(N, device="cuda") * 641) % (1 << 17)
    o16 = _ord16(score)
    valid = torch.ones_like(local, dtype=torch.bool)
    if short >= 0:  # row 0: only `short` candidates per rank
        valid[:, 0, short:] = False
    words = (o16 << 16) | (0xFFFF - (local & 0xFFFF))
    words = torch.where(valid, words, torch.zeros_like(words))
    plane = torch.zeros((W, rows, N // 32), dtype=torch.int64, device="cuda")
    hi = ((local >> 16) & 1) * valid
    for j in range(N):
        plane[:, :, j // 32] |= hi[:, :, j] << (j % 32)
    packed = torch.cat([words, plane], dim=2)
    packed = torch.where(packed >= 1 << 31, packed - (1 << 32), packed)
    gid = local * W + torch.arange(W, device="cuda").view(W, 1, 1)
    gid = torch.where(valid, gid, torch.full_like(gid, -1))
    return packed.to(torch.int32).contiguous(), o16, gid


def _reference(o16, gid, rank, block_table, stride_rows):
    gid_bits = (W * (1 << 17) - 1).bit_length()
    rows = o16.shape[1]
    ids, owned_rows, counts = [], [], []
    flat_n = W * N
    # register key i of thread t is flat candidate t + i * TB (rank-major flat)
    flat = torch.arange(flat_n, device="cuda")
    order = torch.argsort((flat % TB) * 64 + flat // TB)
    for t in range(rows):
        key = (o16[:, t, :] << gid_bits) | ((1 << gid_bits) - 1 - gid[:, t, :])
        key = torch.where(gid[:, t, :] >= 0, key, torch.zeros_like(key)).reshape(-1)
        nz = int((key != 0).sum())
        kk = min(K, nz)
        sel = torch.zeros(flat_n, dtype=torch.bool, device="cuda")
        sel[torch.topk(key, kk).indices] = True
        emitted = order[sel[order]]
        g = gid[:, t, :].reshape(-1)[emitted]
        ids.append(torch.cat([g, g.new_full((K - kk,), -1)]))
        own = g[g % W == rank]
        local = own // W
        prow = block_table[t, local // BLOCK] * stride_rows + local % BLOCK
        counts.append(prow.numel())
        owned_rows.append(torch.cat([prow, prow.new_full((K - prow.numel(),), -1)]))
    return torch.stack(ids), torch.stack(owned_rows), torch.tensor(counts)


@pytest.mark.parametrize(
    "rows,levels,short,dist",
    [
        (8, 0, -1, "relu"),
        (8, 4, -1, "relu"),
        (8, 1, 100, "relu"),
        (64, 16, -1, "relu"),
        (129, 2, 50, "relu"),
        (512, 0, -1, "relu"),
        (8, 0, -1, "tail"),
        (64, 8, -1, "tail"),
        (129, 0, 50, "tail"),
        (16, 0, -1, "span511"),
        (16, 0, -1, "span512"),
    ],
)
def test_select_matches_reference(rows, levels, short, dist):
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        _DCP_TOPK_SELECT_KERNEL,
        _RowsOutput,
    )

    rank, stride_rows = 5, 64
    packed, o16, gid = _case(
        rows, levels, seed=rows * 7 + levels, short=short, dist=dist
    )
    blocks = (1 << 17) // BLOCK + 1
    gen = torch.Generator(device="cuda").manual_seed(3)
    block_table = torch.randint(
        0, 1 << 20, (rows, blocks), generator=gen, device="cuda", dtype=torch.int32
    )
    out = torch.empty((rows, K), dtype=torch.int32, device="cuda")
    rows_out = torch.empty((rows, K), dtype=torch.int32, device="cuda")
    valid_out = torch.empty((rows,), dtype=torch.int32, device="cuda")
    req_id = torch.arange(rows, dtype=torch.int32, device="cuda")
    ro = _RowsOutput(req_id, block_table, rows_out, valid_out, rank, BLOCK, stride_rows)
    _DCP_TOPK_SELECT_KERNEL(packed, out, local_topk=N, topk=K, rows=ro)
    torch.cuda.synchronize()
    ref_ids, ref_rows, ref_counts = _reference(
        o16, gid, rank, block_table.long(), stride_rows
    )
    assert torch.equal(out.long().cpu(), ref_ids.cpu())
    assert torch.equal(valid_out.long().cpu(), ref_counts)
    assert torch.equal(rows_out.long().cpu(), ref_rows.cpu())
