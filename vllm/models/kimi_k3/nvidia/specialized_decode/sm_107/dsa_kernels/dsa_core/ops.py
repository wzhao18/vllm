# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA sparse core under DCP for small M (csrc/dsa_core.cu).

One kernel per DSA layer from the gathered q mailbox (polled) and this rank's
FP32 DCP top-k slots to the head owners' DcpBox O mailboxes: sparse MLA over the
local per-tensor FP8 KV rows for every gathered query head (FP8 QK, FP32 softmax,
FP16 PV, FP32 accumulation), the normalized partial O (bf16) and its base-2 LSE
stored into ``DcpBox.partial`` / ``DcpBox.lse`` of each head's owner, the format
``DcpValue(pushed=True)`` merges. ``splits`` > 1 spreads a row's local chunks over
that many CTAs of one cluster, which LSE-merge them through distributed shared memory
before the push.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch

_EXT = None
LOG2E = math.log2(math.e)
_ERR = {1: "q mailbox poll timeout", 2: "QK MMA wait timeout", 3: "PV MMA wait timeout"}
GROUP_HEADS = 32
# Test only (fault-sensitivity of whole-step checks): K31_DSAC_FAULT=1 runs the core
# with a quarter of its softmax scale.
FAULT = os.environ.get("K31_DSAC_FAULT") == "1"


def load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).parent / "csrc"
    srcs = [csrc / "dsa_core.cu", csrc / "dsa_core_umma.cu"]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    digest = hashlib.sha1(b"".join(s.read_bytes() for s in srcs)).hexdigest()[:10]
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
        ]
        + (["-Xptxas=-v"] if os.environ.get("K31_DSAC_PTXAS_V") else []),
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=build,
        verbose=bool(os.environ.get("K31_DSAC_VERBOSE")),
    )
    return _EXT


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
        heads_per_cta: int = 32,
        umma: bool = False,
        lse_wide: bool = False,
        pdl: bool = True,
        dbg: torch.Tensor | None = None,
    ) -> None:
        """``q_box``: the DcpBox q mailbox of this layer's parity [cap, q_heads, 576]
        (FP8 bytes); ``sel`` [>= rows, >= K] int32 / ``cnt`` [>= rows]: this rank's
        selected flat KV rows and their counts (IndexerStage.merge_rows); ``kv``: the
        flat per-tensor FP8 KV rows [N, 576]; ``bmm1_scale`` = softmax scale x q scale x
        k scale, ``bmm2_scale`` = k scale; ``part_peers`` / ``lse_peers``: DcpBox.peers
        / lse_peers; ``parity``: the DcpBox parity of this layer; ``lse_wide``: the LSE
        box in the [2, CAP, H, W] layout of DcpValue's wide / spread merge.
        ``splits`` > 1: rows of at most ``split_min`` 64-key chunks still run on one CTA
        (no cluster merge), longer ones over the ``splits`` CTAs of their cluster."""
        if rows > self.cap:
            raise ValueError(f"DsaCore: {rows} rows > cap {self.cap}")
        if splits not in (1, 2, 4, 8) or splits > self.max_splits:
            raise ValueError(f"DsaCore: splits {splits} not 1, 2, 4 or 8 (<= max)")
        if FAULT:  # test only: a wrong softmax scale (whole-step checks must see it)
            bmm1_scale *= 0.25
        if umma:  # tcgen05 variant: CTA (latent half, row), every gathered head
            self.ext.run_umma(
                q_box, sel, cnt, kv, bmm1_scale * LOG2E, bmm2_scale, part_peers,
                lse_peers, box_cap, hpr, parity, rank, rows, int(lse_wide), self.err,
                self.spin_ns,
                1 if pdl else 0, dbg,
            )  # fmt: skip
            return
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
