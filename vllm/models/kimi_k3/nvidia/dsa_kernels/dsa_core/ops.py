# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA sparse core under DCP for small M (csrc/dsa_core.cu).

One kernel per DSA layer from the gathered q mailbox (polled) and this rank's
DcpExchange top-k slots to the head owners' DcpBox O mailboxes: sparse MLA over the
local per-tensor FP8 KV rows for every gathered query head at the stock FP8 sparse
MLA decode's precision (trtllm-gen's: its KV CTA split of the batch, ``trtllm_kv_ctas``;
per KV CTA FP8 QK, FP32 softmax with one running max step per 128-key tile, E4M3 P at
448 x the running max, FP32 PV accumulation, BF16 unnormalized partials merged in
FP32), the normalized partial O (bf16) and its base-2 LSE
stored into ``DcpBox.partial`` / ``DcpBox.lse`` of each head's owner, the format
``DcpValue(pushed=True)`` merges. ``splits`` > 1 spreads a row's local tiles over
that many CTAs of one cluster, which LSE-merge their BF16 partials through distributed
shared memory before the push.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch

_EXT = None
LOG2E = math.log2(math.e)
_ERR = {1: "q mailbox poll timeout"}
GROUP_HEADS = 32


def load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).parent / "csrc"
    srcs = [csrc / "dsa_core.cu"]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    digest = hashlib.sha1(b"".join(s.read_bytes() for s in srcs)).hexdigest()[:10]
    # (infrastructure: where torch keeps the extension build; default its cache)
    build = os.environ.get("K31_DSAC_BUILD_DIR")
    if build:
        build = str(Path(build) / digest)
        Path(build).mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name=f"k31_dsa_core_sm{arch}_{digest}",
        sources=[str(s) for s in srcs],
        extra_cuda_cflags=[
            "-O3",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "-std=c++20",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=build,
    )
    return _EXT


def trtllm_kv_ctas(rows: int, q_heads: int, width: int, sms: int) -> int:
    """trtllm-gen's KV CTAs per row of the stock FP8 sparse MLA decode of ``rows``
    query rows of ``q_heads`` (padded) heads over ``width`` slots per row on ``sms``
    SMs: FlashInfer's computeCtaAndClusterConfig for the sparse MLA generation kernel
    (more than 32 heads: KeepsMmaAb, tileSizeQ 64, stepKv 128): at most
    ceil(min(width, 2048) / 256), at most sms // (rows x q_heads / 64), at least 1.
    trtllm-gen splits a row's ceil(n / 128) tiles over them in pieces of
    ceil(tiles / C) consecutive tiles. Probed bitwise on VR200 (212 SMs, 128 padded
    heads, width 2048; scratchpad probe_split3): C = 8 at rows 1-12, 6 at 16, 4 at 24,
    3 at 32, 2 at 48, 1 (no split) at 56 and 64, each row's pieces as above for local
    counts 100-2048."""
    if not 32 < q_heads <= 128 or q_heads % 64:
        raise ValueError(f"trtllm_kv_ctas: {q_heads} padded query heads (64 or 128 only)")
    most = -(-min(width, 2048) // 256)
    return max(1, min(most, sms // (rows * (q_heads // 64))))


class DsaCore:
    """State of the DSA sparse core for one device (shared by every DSA layer; the
    launches must be stream-ordered). ``cap``: most rows per launch; ``heads``: the
    gathered query heads (DCP x local heads, a multiple of 32)."""

    def __init__(
        self,
        device: torch.device,
        cap: int = 64,
        heads: int = 96,
        max_splits: int = 8,
        spin_s: float = 5.0,
    ) -> None:
        if heads % GROUP_HEADS:
            raise ValueError(f"DsaCore: {heads} gathered heads, not a multiple of 32")
        # (heads per CTA: 32, or 16 for more CTAs on a small batch)
        self.ext = load_ext()
        with torch.cuda.device(device):
            self.ext.prepare()  # (shared-memory opt-ins, never inside a capture)
        self.device, self.cap, self.heads = device, cap, heads
        self.groups = heads // GROUP_HEADS
        self.max_splits = max_splits
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.spin_ns = int(spin_s * 1e9)

    def run(
        self,
        q_box: torch.Tensor,
        sel: torch.Tensor,
        cnt: torch.Tensor,
        kv: torch.Tensor,
        *,
        bmm1_scale: float,
        bmm2_scale: float,
        part_peers: torch.Tensor,
        lse_peers: torch.Tensor,
        box_cap: int,
        hpr: int,
        parity: int,
        rank: int,
        rows: int,
        splits: int = 1,
        split_min: int = 4,
        kv_ctas: int | None = None,
        heads_per_cta: int = 32,
        lse_wide: bool = False,
        pdl: bool = True,
        dbg: torch.Tensor | None = None,
    ) -> None:
        """``q_box``: the DcpBox q mailbox of this layer's parity [cap, q_heads, 576]
        (FP8 bytes); ``sel`` [>= rows, >= K] int32 / ``cnt`` [>= rows]: this rank's
        selected flat KV rows and their counts (DcpExchange.topk_merge); ``kv``: the
        flat per-tensor FP8 KV rows [N, 576]; ``bmm1_scale`` = softmax scale x q scale x
        k scale, ``bmm2_scale`` = k scale; ``part_peers`` / ``lse_peers``: DcpBox.peers
        / lse_peers; ``parity``: the DcpBox parity of this layer; ``lse_wide``: the LSE
        box in the [2, CAP, H, W] layout of DcpValue's wide / spread merge.
        ``splits`` > 1: rows of at most ``split_min`` 64-key chunks still run on one CTA
        (no cluster merge), longer ones over the ``splits`` CTAs of their cluster (128-key
        pieces s, s + splits, ..). ``kv_ctas``: trtllm-gen's KV CTAs per row at this
        batch (default ``trtllm_kv_ctas`` of ``rows`` rows of the mailbox's heads over
        ``sel``'s width on this device: the stock decode of the same rows)."""
        if rows > self.cap:
            raise ValueError(f"DsaCore: {rows} rows > cap {self.cap}")
        if splits not in (1, 2, 4, 8) or splits > self.max_splits:
            raise ValueError(f"DsaCore: splits {splits} not 1, 2, 4 or 8 (<= max)")
        if kv_ctas is None:
            kv_ctas = trtllm_kv_ctas(rows, q_box.shape[1], sel.shape[1], self.sms)
        self.ext.run(
            q_box,
            sel,
            cnt,
            kv,
            bmm1_scale * LOG2E,
            bmm2_scale,
            part_peers,
            lse_peers,
            box_cap,
            hpr,
            parity,
            rank,
            rows,
            int(lse_wide),
            splits,
            split_min,
            kv_ctas,
            heads_per_cta,
            self.err,
            self.spin_ns,
            1 if pdl else 0,
            dbg,
        )

    def check(self) -> None:
        code = int(self.err.item())
        if code:
            raise RuntimeError(f"DsaCore error {code}: {_ERR.get(code, '?')}")
