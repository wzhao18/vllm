# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreparedDsa on the round-1 runtime (runtime/providers.py, a cap-128 overlay file
that stays as is): its DSA DcpBox (mla.dcp: one q mailbox and one BF16 partials
mailbox), with #108's "proj" partials in a DcpMailbox beside it where
VLLM_K31_DSA_PARTIALS asks for them, and its shared scratch, whose last item is the
BF16 latent of the removed CuTe sparse core instead of FlashInfer's buffers."""

import os
from functools import cache

import torch

from ..common.host import to_cute
from ..common.policy import batch_table
from .dcp_value import FORMATS, PROJECTED, DcpMailbox, gather_heads

# The DSA layers' DcpValue partials per batch of rows (partials_for): "bf16" (MR !8's
# BF16 latent mailbox; the default), "proj" (#108's value rows through W_UV, 4x fewer
# NVLink bytes for a gathered W_UV per layer) or a batch table of both, e.g.
# "1-64:bf16,65-512:proj". Read when the providers are built.
PARTIALS_ENV = "VLLM_K31_DSA_PARTIALS"


def dsa_partials(spec=None):
    """The checked partials policy ``spec`` (None: VLLM_K31_DSA_PARTIALS, unset or
    empty "bf16"): "bf16", "proj" or a batch table of them."""
    if spec is None:
        spec = os.environ.get(PARTIALS_ENV) or "bf16"
    if spec in FORMATS:
        return spec
    try:
        table = batch_table(spec)
    except ValueError:
        table = {}
    if not table or any(fmt not in FORMATS for fmt in table.values()):
        raise ValueError(
            f"{PARTIALS_ENV}={spec!r}: expected bf16, proj or a batch table of them "
            "(e.g. 1-64:bf16,65-512:proj)"
        )
    return spec


def projected_mailbox(box):
    """The "proj" DcpMailbox beside an MR !8 DcpBox ``box``: value rows [2, DCP, CAP,
    H, 128] BF16 and LSE [2, DCP, CAP, H] FP32, sentinel-filled. One per box (kept on
    it), shared by every DSA layer; created collectively by the first layer whose
    rows use it."""
    mail = getattr(box, "dsa_projected", None)
    if mail is None:
        device = box.q.device
        mail = DcpMailbox(box.group, device, box.size, box.cap, box.heads, PROJECTED)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=box.group)
        box.dsa_projected = mail
    return mail


class DsaBox:
    """``box`` (mla.dcp.DcpBox) as PreparedDsa and WideDsa read a DcpBox, for the
    ``batches`` it serves (default: every row up to its cap) under the ``partials``
    policy (dsa_partials; None: VLLM_K31_DSA_PARTIALS). ``mailbox(batch)`` of a
    "bf16" batch is the box itself: MR !8's BF16 latent partials [2, DCP, CAP, H, 512]
    and LSE [2, DCP, CAP, H] (the same shapes and sentinels) and their peers'
    addresses; of a "proj" batch the box's projected_mailbox."""

    format = "bf16"

    def __init__(self, box, batches=None, partials=None):
        self.box = box
        self.size, self.rank, self.cap = box.size, box.rank, box.cap
        self.q, self.q_heads = box.q, box.q_heads
        self.partial, self.lse = box.partial, box.lse
        self.peers, self.lse_peers = box.peers, box.lse_peers
        self.partials = spec = dsa_partials(partials)
        rows = range(1, self.cap + 1)
        # partials_for at every row, the table parsed once
        table = {} if spec in FORMATS else batch_table(spec)
        default = spec if spec in FORMATS else "bf16"
        self.formats = {b: table.get(b, default) for b in rows}
        served = sorted(batches) if batches else rows
        assert served[-1] <= self.cap, (served[-1], self.cap)
        self.mailboxes = {self.format: self}
        if any(self.formats[b] == PROJECTED for b in served):
            self.mailboxes[PROJECTED] = projected_mailbox(box)

    def mailbox(self, batch):
        assert batch <= self.cap, (batch, self.cap)
        return self.mailboxes[self.formats[batch]]

    @property
    def projected(self):
        """Whether some batch the box serves exchanges "proj" value rows."""
        return PROJECTED in self.mailboxes

    def gather_heads(self, w_uv):
        """Every rank's W_UV, gathered over the box's group (collective)."""
        return gather_heads(w_uv, self.size, self.box.group)

    def value_weights(self, layer):
        """The gathered W_UV (gather_heads) of an MLA ``layer`` that its "proj" batches
        read, with its CuTe view, or None when the box serves no "proj" batch. DCP x
        the local W_UV (BF16 DCP * H * 512 * 128: 12 MiB per K3.1 attention layer of
        96 heads), taken once per layer and kept on it: PreparedDsa and WideDsa share
        it. Collective: the providers build the same layers in the same order on
        every rank, before the KV cache is profiled."""
        if not self.projected:
            return None
        if getattr(layer, "W_UV_dcp_gathered", None) is None:
            gathered = self.gather_heads(layer.W_UV)
            layer.W_UV_dcp_gathered = gathered, to_cute(gathered)
        return layer.W_UV_dcp_gathered

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
