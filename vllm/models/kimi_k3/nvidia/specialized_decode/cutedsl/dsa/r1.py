# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreparedDsa on the round-1 runtime (runtime/providers.py, a cap-128 overlay file
that stays as is): its DSA DcpBox (mla.dcp: one q mailbox and one BF16 partials
mailbox, no "proj" rows) and its shared scratch, whose last item is the BF16
latent of the removed CuTe sparse core instead of FlashInfer's buffers."""

from functools import cache

import torch

from .dcp_value import PARTIALS_DEFAULT, partials_for


class DsaBox:
    """``box`` (mla.dcp.DcpBox) as PreparedDsa reads a DcpBox: ``mailbox(batch)``
    is the box itself, a DcpMailbox of BF16 latent partials [2, DCP, CAP, H, 512]
    and LSE [2, DCP, CAP, H] (the same shapes and sentinels) and their peers'
    addresses; no batch exchanges "proj" rows (PARTIALS_DEFAULT: from 160 rows,
    above the box's rows)."""

    format = "bf16"

    def __init__(self, box):
        self.box = box
        self.size, self.rank, self.cap = box.size, box.rank, box.cap
        self.q, self.q_heads = box.q, box.q_heads
        self.partial, self.lse = box.partial, box.lse
        self.peers, self.lse_peers = box.peers, box.lse_peers

    def mailbox(self, batch):
        assert batch <= self.cap, (batch, self.cap)
        assert partials_for(PARTIALS_DEFAULT["dsa"], batch) == self.format, batch
        return self

    def value_weights(self, w_uv):
        return None

    def mc(self, parity):
        return self.box.mc(parity)


@cache
def fi_buffers(device, rows, topk, heads):
    """FlashInfer's flat top-k rows [rows, topk] and counts [rows] (int32), latent
    [rows * heads * 512] (BF16) and LSE [rows * heads] (FP32), shared by every DSA
    layer of the device (they run one at a time)."""
    i32 = dict(dtype=torch.int32, device=device)
    return (
        torch.full((rows, topk), -1, **i32),
        torch.zeros(rows, **i32),
        torch.empty(rows * heads * 512, dtype=torch.bfloat16, device=device),
        torch.empty(rows * heads, dtype=torch.float32, device=device),
    )
