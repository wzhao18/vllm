# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A candidate DSA sparse MLA core for the wide DCP rows (csrc/wide_core.cu), not
wired into the decode: FlashInfer trtllm-gen stays the production sparse core at
every row (measured faster, below).

A persistent tcgen05 kernel over the gathered q mailbox [rows, q_heads, 576] (per-
tensor FP8, the first ``heads`` of each row real) and this rank's flat FP8 KV rows
(a compacted prefix of ``cnt`` slots per row) -> the latent [rows, >= heads, 512]
(bf16) and LSE [rows, >= heads] (fp32), in the format of trtllm-gen's sparse MLA
decode + the stock ``_finish_dcp_partials`` (rows without local entries: latent 0,
LSE -inf). A producer warp loads Q (2D TMA tiles) and the selected KV rows
(``tile::gather4``) into a 4-chunk ring; 16 compute warps run QK, the base-2 online
softmax over 128-key tiles and PV for one latent half per CTA, the two halves of a
row on neighbouring CTAs.

Exactness: it computes trtllm-gen's math on rows trtllm-gen runs unsplit (>= 54
rows at 128 padded heads on 212 SMs): E4M3 Q K^T with FP32 accumulation, P =
E4M3(2^(s - m) * 448) at the running max of 128-key tiles, FP32 l of the unquantized
P, E4M3 P V into FP32, one BF16 rounding (probe: an emulation with exactly these
choices matches trtllm-gen's BF16 output bit for bit on 99.9-100 % of elements, LSE
to 4e-6; this kernel vs trtllm-gen: 99.99 % equal, LSE within 2e-6;
tests/models/kimi_k3/test_k31_dsa_wide_core.py). Below ~54 rows trtllm-gen splits
each row's KV over CTAs and rounds BF16 partials, which this kernel does not.

Speed (1 VR200, 96 real heads, AgentX-like local counts, L2 flushed, us per call,
trtllm-gen chain -> this kernel): DCP16 q4 M64 10.6 -> 14.5, M512 28.8 -> 50.9,
M1024 45.2 -> 90.1; DCP4 (C512 SP) M512 41.0 -> 112.2. The per-item critical path
is load latency, not bandwidth (the same times with an L2-resident KV): each CTA can
hold one item's Q + KV (~127 KB of 227 KB) and the next item's Q only after the
current QK, so the item period is ~ one loaded-GPU memory round trip (~5 us) plus
QK / softmax, as trtllm-gen's per-wave period (~6 us, one row per 2-CTA pair). Beating
it needs about two items in flight per SM: a 2-CTA (cta_group::2) split of Q by
heads and K by keys, or both latent halves of a row on one CTA."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

_EXT = None


def load_ext():
    """Build (once per source digest) and load the extension (SM100 family, 10.0f:
    runs on SM100 / SM103 / SM107)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "csrc" / "wide_core.cu"
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
    build = os.environ.get("K31_DSAW_BUILD_DIR")
    if build:
        build = str(Path(build) / digest)
        Path(build).mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name=f"k31_dsa_wide_core_{digest}",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100f,code=sm_100f",
            "-std=c++20",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=build,
        verbose=bool(os.environ.get("K31_DSAW_VERBOSE")),
    )
    return _EXT


def supported(device: torch.device) -> bool:
    """The kernel needs tcgen05 (SM100 family)."""
    return torch.cuda.get_device_capability(device)[0] == 10


class WideCore:
    """The wide sparse core of one device (stateless across launches; every DSA
    layer may share it)."""

    def __init__(self, device: torch.device):
        if not supported(device):
            raise RuntimeError(
                f"Kimi K3.1 DSA wide core needs an SM100-family GPU, got "
                f"{torch.cuda.get_device_capability(device)}"
            )
        self.ext = load_ext()
        with torch.cuda.device(device):
            self.ext.prepare()  # (shared-memory opt-in, never inside a capture)
        self.device = device
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count

    def run(
        self, q, sel, cnt, kv, bmm1, bmm2, out, lse, heads, rows=None, pdl=True, dbg=None,
        dbg_cta=0,
    ):
        """``q`` [>= rows, q_heads, 576] FP8 bytes; ``sel`` [>= rows, W] int32 flat
        KV rows (compacted prefix), ``cnt`` [>= rows] int32; ``kv`` [N, 576] FP8 rows;
        ``bmm1`` = softmax scale x q scale x k scale, ``bmm2`` = k scale (FlashInfer's
        trtllm-gen scales); ``out`` [>= rows, >= heads, 512] bf16, ``lse`` [>= rows,
        >= heads] fp32 (trtllm-gen's base-2 LSE). ``dbg`` (benches): int64 [32 * 8]
        globaltimer stamps of CTA ``dbg_cta``'s first 32 items (start, data ready,
        QK done, softmax done, PV done, epilogue done)."""
        rows = q.shape[0] if rows is None else rows
        self.ext.run(
            q.view(torch.uint8),
            sel,
            cnt,
            kv.view(torch.uint8),
            float(bmm1),
            float(bmm2),
            out,
            lse,
            heads,
            rows,
            self.sms,
            1 if pdl else 0,
            dbg,
            dbg_cta,
        )
