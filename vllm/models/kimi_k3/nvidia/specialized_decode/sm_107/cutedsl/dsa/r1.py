# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreparedDsa on the round-1 runtime (runtime/providers.py, a cap-128 overlay file
that stays as is): its DSA DcpBox (mla.dcp: one q mailbox and one BF16 partials
mailbox), with #108's "proj" partials in a DcpMailbox beside it on the rows
``AUTO_PARTIALS`` gives them, and its shared scratch, whose last item is the BF16
latent of the removed CuTe sparse core instead of FlashInfer's buffers."""

from functools import cache

import torch

from ..common.host import to_cute
from ..common.policy import batch_table
from .dcp_value import DSA_PARTIALS, FORMATS, PROJECTED, DcpMailbox, gather_heads

# The DSA layers' DcpValue partials per batch of rows (partials_for), chosen in code
# per shape and backend: "bf16" (MR !8's BF16 latent mailbox) or "proj" (#108's value
# rows through W_UV, 4x fewer NVLink bytes for a gathered W_UV per layer). Both round
# the partials to BF16 once (the merge is linear: only the rounding point moves, as an
# order change). On FLASHINFER_MLA_SPARSE (per-tensor FP8: the only backend "proj"
# serves) the one automatic policy, dcp_value.DSA_PARTIALS (PARTIALS_DEFAULT["dsa"]):
# rows 1-16 keep BF16 (the DSA sparse core, dsa_kernels.dsa_core, pushes BF16
# partials there, and "proj" ties BF16 at few rows) and every row from 17 on exchanges
# "proj", open-ended ("proj"'s merging CTAs stride the row tiles over at most one
# wave, ``dcp_value.DcpValue``, so it serves any rows). "proj" numerics are what
# production already ran (the C48-C256 recipes ran "proj", C1 / C8
# "1-16:bf16,17-512:proj"). FlashMLA sparse (fp8_ds_mla):
# "bf16" at every row (its DcpValue strides its row tiles above one wave too:
# ``mla.dcp.DcpValue.tile_stride``).
# The BF16 rows follow the sparse core's coverage of the deployment's verify length
# (``auto_partials``): at a q the core never serves, "proj" wins from the first row
# (VR200, 1 GPU, fp8 KV, DcpValue + partials, agentx contexts, us bf16 -> proj: q4 M4
# 10.71 -> 9.32, M8 11.47 -> 9.43, M12 13.53 -> 9.94, M16 14.52 -> 10.42; nqbench
# kern-dsa value). NQ-TUNE: the bf16 / proj crossover in the 16-rank step at the core's
# rows with the core on.
AUTO_PARTIALS = DSA_PARTIALS


def auto_partials(q=None):
    """The automatic FlashInfer-sparse partials of verify length ``q``: "bf16" on the
    rows the DSA sparse core serves at that q (it pushes BF16 partials), "proj" on
    every other row; AUTO_PARTIALS when q is unknown."""
    from ...dsa_kernels.dsa_core.decode import ROWS, VARIANT

    if q is None:
        return AUTO_PARTIALS
    return f"1-{ROWS}:bf16,{ROWS + 1}-:proj" if q in VARIANT else PROJECTED


def dsa_partials(spec=None, fp8=False, q=None):
    """The checked partials policy ``spec`` (None: the automatic one, auto_partials(q)
    on FLASHINFER_MLA_SPARSE (``fp8``), else "bf16"): "bf16", "proj" or a batch table
    of them."""
    if spec is None:
        spec = auto_partials(q) if fp8 else "bf16"
    if spec in FORMATS:
        return spec
    try:
        table = batch_table(spec)
    except ValueError:
        table = {}
    if not table or any(fmt not in FORMATS for fmt in table.values()):
        raise ValueError(
            f"DSA DcpValue partials {spec!r}: expected bf16, proj or a batch table of "
            "them (e.g. 1-16:bf16,17-:proj)"
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
    policy (dsa_partials; None: the automatic one of the backend, ``fp8``: FlashInfer
    sparse's per-tensor FP8, at verify length ``verify_q``). ``mailbox(batch)`` of a
    "bf16" batch is the box itself: MR !8's BF16 latent partials [2, DCP, CAP, H, 512]
    and LSE [2, DCP, CAP, H] (the same shapes and sentinels) and their peers'
    addresses; of a "proj" batch the box's projected_mailbox. A box created without
    the BF16 mailboxes (``DcpBox(partials=False)``: WideDsa's widened box) gets them
    (collectively) only when a served batch is "bf16"."""

    format = "bf16"

    def __init__(self, box, batches=None, partials=None, fp8=False, verify_q=None):
        self.box = box
        self.size, self.rank, self.cap = box.size, box.rank, box.cap
        self.q, self.q_heads = box.q, box.q_heads
        self.partials = spec = dsa_partials(partials, fp8, verify_q)
        rows = range(1, self.cap + 1)
        # partials_for at every row, the table parsed once
        table = {} if spec in FORMATS else batch_table(spec)
        default = spec if spec in FORMATS else "bf16"
        self.formats = {b: table.get(b, default) for b in rows}
        served = sorted(batches) if batches else rows
        assert served[-1] <= self.cap, (served[-1], self.cap)
        if box.partial is None and any(self.formats[b] == self.format for b in served):
            box.add_partials()  # (collective: every rank serves the same batches)
        self.partial, self.lse = box.partial, box.lse
        self.peers, self.lse_peers = box.peers, box.lse_peers
        self.mailboxes = {self.format: self}
        if any(self.formats[b] == PROJECTED for b in served):
            self.mailboxes[PROJECTED] = projected_mailbox(box)

    def mailbox(self, batch):
        assert batch <= self.cap, (batch, self.cap)
        fmt = self.formats[batch]
        assert fmt != self.format or self.partial is not None, (batch, fmt)
        return self.mailboxes[fmt]

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
