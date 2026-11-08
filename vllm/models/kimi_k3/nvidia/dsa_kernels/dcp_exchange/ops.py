# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-free DCP exchanges for Kimi K3.1 DSA decode (see csrc/dcp_exchange.cu).

``DcpExchange`` owns the symmetric-memory mailboxes for one DCP group:

* ``topk_merge``: exact global top-K by histogram threshold. Input: this
  rank's local bf16 indexer logits [T, >= local_len]; output: this rank's
  selected local tokens (KV slots through ``block_table``, or local
  positions) and their count. 1 NVLink round (``one_round_window`` keys at
  16-bit resolution) or 2 rounds (coarse 1/8-octave window + 16-key refine),
  plus one tie round when ties at the K-th key straddle the cut on >= 2
  ranks. Ties are broken by global token id (the stock stable merge's rule),
  and the output is deterministic: tokens above the K-th key in position
  order, then the selected ties in position order.
* ``o_exchange``: push partial outputs [T, H, D] + LSE to the head owners and
  merge the W partials (FP32, one bf16 rounding). D = 128 after W_UV, or the
  512-dim latent.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

_EXT = None

_COARSE_WORDS = 20
_FINE_WORDS = 8
_TIE_LIST = 64  # kTieList
_TIE_WORDS = 32  # kTieWords

# stats counters (kSt* in the kernel)
STAT_NAMES = (
    "rows", "one_round", "two_round_refine", "fallback_r8", "tie_round",
    "tie_radix", "tie_write_pass", "select_all_valid",
)
_ERR_CODES = {
    1: "1R poll timeout", 2: "coarse poll timeout", 3: "refine poll timeout",
    4: "refine: no K-th key", 5: "local length exceeds SMEM segment",
    6: "R8 poll timeout", 7: "R4 poll timeout", 8: "R4: no K-th key",
    10: "O LSE poll timeout", 11: "O data poll timeout", 12: "row-0 token timeout",
    13: "global length >= 2^24", 14: "tie list poll timeout",
    15: "tie radix poll timeout", 16: "tie radix: no cut",
    17: "local segment exceeds the WRITE bitmap (131072 keys)",
    18: "1R window broadcast timeout",
    19: "act broadcast timeout",
    20: "top-k WRITE: CTA counts differ from the window counts",
    21: "top-k pre-pass: window start differs from the merge kernel's",
}

# A top-k launch with keys in global memory (segmax 0) takes the balanced pre-pass
# (csrc topk_pp_*_kernel: the coarse histogram, window counts and candidates of the
# rows from a grid over their keys) when a row's CTA could stream more than
# ``TOPK_PREPASS_ROW_KEYS`` keys itself. The in-kernel passes are issue-latency bound
# (one 256-thread CTA per row segment: ~3 passes at ~8 GB/s over the longest
# segment); the pre-pass spreads the long rows' keys over the whole GPU and leaves the
# merge kernel the exchange and the WRITE.
#
# Rows of at most ``topk_prepass_min`` local keys still run their own passes in the
# merge kernel (pp_row(); the pre-pass kernels cost about one pass over the keys they
# cover plus three launches): 16K keys per CTA when every row is co-resident, 32K
# when the rows take more than one persistent turn (the merge kernel is long then).
# Measured on VR200 (q4 AgentX contexts, real-logit marginals, us per call; pre-pass
# for rows above 0 / 8K / 16K / 32K keys per CTA, "-": not measured):
#   DCP4  (2 CTAs per row) 64 rows 45.7 / 46.8 / - / 56.4, 256 rows 59.5 / 57.5 / - / 63.1,
#         512 rows (2 turns) 104.3 / 100.7 / - / 86.5;
#   DCP8  256 rows (2 CTAs) 55.8 / 56.8 / 52.5 / 57.7, 512 rows 77.5 / 74.6 / 66.3 / 64.0,
#         1024 rows (2 turns) 150.1 / 147.6 / 142.7 / 118.9;
#   DCP16 local lengths (4 ranks, contexts / 4) 256 rows 44.2 / 41.6 / 47.0 / 54.8,
#         512 rows 56.9 / 54.7 / 52.9 / 62.9, 1024 rows (2 turns) 101.0 / 91.0 / 91.3 / 86.0
#         (no pre-pass: 53.6 / 59.8 / 88.8).
TOPK_PREPASS_ROW_KEYS = 1 << 14
TOPK_PREPASS_TURN_KEYS = 1 << 15


def topk_prepass(cluster: int, segmax: int, max_local_len: int) -> bool:
    """Whether a top-k launch of ``cluster`` CTAs per row with ``segmax`` staged keys
    (0: keys in global memory) takes the pre-pass (``TOPK_PREPASS_ROW_KEYS``)."""
    return segmax == 0 and -(-max_local_len // cluster) > TOPK_PREPASS_ROW_KEYS


def topk_prepass_min(cluster: int, rows: int, resident: int) -> int:
    """Local keys up to which a row of a pre-pass launch (``rows`` rows, ``resident``
    co-resident clusters of ``cluster`` CTAs) runs its own passes."""
    per_cta = TOPK_PREPASS_ROW_KEYS if rows <= resident else TOPK_PREPASS_TURN_KEYS
    return per_cta * cluster


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "csrc" / "dcp_exchange.cu"
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    import hashlib

    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
    # (infrastructure: where torch keeps the extension build; default its cache)
    build = os.environ.get("K31_DCPX_BUILD_DIR")
    if build:
        build = str(Path(build) / digest)
        Path(build).mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name=f"k31_dcp_exchange_sm{arch}_{digest}",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "--use_fast_math",
            "-std=c++20",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=build,
    )
    return _EXT


def _window_words(nw1: int) -> int:
    return ((2 + nw1 // 2) + 3) & ~3


class DcpExchange:
    """Symmetric mailboxes + device epochs for one DCP group.

    ``cap``: max rows (tokens) per call; ``cap1``: max rows that use the
    1-round window (rows >= cap1 use 2 rounds); ``heads``: total query heads;
    ``o_dims``: the O-exchange head dims to support (128 post-W_UV, 512 latent).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        cap: int = 1536,
        cap1: int = 256,
        one_round_window: int = 512,
        heads: int = 96,
        o_dims: tuple[int, ...] = (128, 512),
        topk: int = 2048,
        interleave: int = 1,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.ext = _load_ext()
        self.group = group
        self.device = device
        self.W = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.cap = cap
        self.cap1 = cap1 if one_round_window else 0
        self.nw1 = one_round_window
        self.K = topk
        self.interleave = interleave
        self.heads = heads
        assert heads % self.W == 0
        self.hpr = heads // self.W

        # ---- top-k mailbox (u32 words)
        W = self.W
        # 1R slot per rank: the window message, then (centred window) 128 words of
        # 256-key bin counts and a 132-word refine message
        ww = _window_words(one_round_window) + 260 if one_round_window else 4
        sizes = {
            "w1": 2 * max(self.cap1, 1) * W * ww,
            "c1": 2 * cap * W * _COARSE_WORDS,
            "f1": 2 * cap * W * _FINE_WORDS,
            "r8": 2 * cap * W * 128,
            "r4": 2 * cap * W * 8,
            "tl": 2 * cap * W * _TIE_LIST,
            "tr": 2 * cap * W * _TIE_WORDS,
            "tk": 2 * W * 4,
        }
        self.off = {}
        o = 0
        for k in sizes:
            v = sizes[k]
            self.off[k] = o
            o += (v + 63) & ~63
        self.tk_box = symm_mem.empty(o, dtype=torch.int32, device=device)
        self.tk_box.fill_(-1)
        h = symm_mem.rendezvous(self.tk_box, group)
        self.tk_peers = torch.tensor(
            [int(x) for x in h.buffer_ptrs], dtype=torch.int64, device=device
        )
        self.tk_mc = int(getattr(h, "multicast_ptr", 0) or 0)
        self.tk_epoch = torch.zeros(cap, dtype=torch.int32, device=device)  # per row
        self.tk_ticket = torch.zeros(1, dtype=torch.int32, device=device)
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        self.mailbox_bytes = 4 * o
        # top-k pre-pass scratch (per row: coarse histogram, window counts, candidate
        # list; csrc topk_pp_*_kernel), zero between uses
        self.tk_pp = torch.zeros(
            int(self.ext.topk_pp_words(cap)), dtype=torch.int32, device=device
        )

        # ---- O mailboxes, one per head dim
        self.ox = {}
        for D in o_dims:
            data_words = 2 * cap * W * self.hpr * D // 2
            lse_words = 2 * cap * W * self.hpr
            box = symm_mem.empty(data_words + lse_words, dtype=torch.int32, device=device)
            box.fill_(-1)
            hh = symm_mem.rendezvous(box, group)
            peers = torch.tensor(
                [int(x) for x in hh.buffer_ptrs], dtype=torch.int64, device=device
            )
            self.ox[D] = dict(
                box=box,
                peers=peers,
                off_data=0,
                off_lse=data_words,
                epoch=torch.zeros(1, dtype=torch.int32, device=device),
                ticket=torch.zeros(1, dtype=torch.int32, device=device),
                grid=int(self.ext.max_coresident_ox(D)),
            )
        torch.cuda.synchronize(device)
        dist.barrier(group)

    # ------------------------------------------------------------------ top-k
    @staticmethod
    def cluster_for(T: int, max_local_len: int, segmax: int = 8192, sms: int = 212) -> int:
        need = max(1, -(-max_local_len // segmax))
        par = max(1, (2 * sms) // max(T, 1))
        c = max(need, min(par, 16))
        p2 = 1
        while p2 < c:
            p2 *= 2
        return min(p2, 16)

    def topk_merge(
        self,
        logits: torch.Tensor,
        local_len: torch.Tensor,
        global_len: torch.Tensor,
        out: torch.Tensor,
        out_count: torch.Tensor,
        *,
        max_local_len: int,
        block_table: torch.Tensor | None = None,
        block_size: int = 1,
        next_n: int = 1,
        block_stride: int = 0,
        req_of_row: torch.Tensor | None = None,
        cluster: int | None = None,
        segmax: int = 8192,
        one_round: bool = True,
        window: str = "top",
        small_ctx: int = 8192,
        dbg: torch.Tensor | None = None,
        stats: torch.Tensor | None = None,
        interleave: int | None = None,
        prepass: bool | None = None,
        prepass_min: int | None = None,
    ) -> None:
        """``stats``: optional int32[8] row counters (``STAT_NAMES``), added to.
        ``prepass`` / ``prepass_min``: the pre-pass and the local keys up to which a
        row still runs its own passes (None: ``topk_prepass`` / ``topk_prepass_min``;
        the outputs are the same either way).
        Ties at the K-th key are selected by ascending global token id (the stock
        merge's rule; one extra round when they straddle ranks).
        ``window`` (1-round rows): "top" ends each rank's window at its local
        maximum; "center" centres it at the rank's local estimate of the K-th key
        (its local ceil(K * L / G)-th key). Rows whose K-th key falls outside a
        window take the octave rounds, so both are exact.

        Invariant: launches of one ``DcpExchange`` must be stream-ordered on
        every rank (no PDL overlap); the kernel keeps the 2-parity mailboxes
        safe by always exchanging on row 0."""
        T = logits.shape[0]
        if cluster is None:
            # Small batches: a cluster of CTAs per row (keys staged in SMEM). Large
            # batches: one CTA per row re-reading the (L2-resident) logits, so every
            # row is co-resident and the NVLink rounds of all rows overlap.
            cluster = 1
            while cluster < 8 and T * cluster * 2 <= 800:
                cluster *= 2
        # the WRITE bitmaps hold 131072 keys per CTA
        while cluster < 16 and -(-max_local_len // cluster) > 131072:
            cluster *= 2
        assert -(-max_local_len // cluster) <= 131072, "local context too long"
        if -(-max_local_len // cluster) > segmax:
            segmax = 0  # keys re-read from global memory
        pre = topk_prepass(cluster, segmax, max_local_len) if prepass is None else prepass
        pre = pre and segmax == 0
        if prepass_min is None:
            res = self.resident_clusters(cluster, segmax, True) if pre else 0
            prepass_min = topk_prepass_min(cluster, T, res)
        pp_min = prepass_min
        self.ext.topk_merge(
            logits,
            local_len,
            global_len,
            out,
            out_count,
            block_table,
            block_size,
            next_n,
            block_stride,
            req_of_row,
            0,  # (ties by global id; the kernel's rank-order mode is not exposed)
            self.tk_box.data_ptr(),
            self.tk_mc,
            self.tk_peers,
            self.W,
            self.rank,
            self.K,
            self.cap,
            self.cap1 if one_round else 0,
            self.nw1 if one_round else 0,
            {"top": 0, "center": 1}[window],
            self.off["w1"],
            self.off["c1"],
            self.off["f1"],
            self.off["r8"],
            self.off["r4"],
            self.off["tl"],
            self.off["tr"],
            self.off["tk"],
            self.interleave if interleave is None else interleave,
            small_ctx,
            self.tk_epoch,
            self.tk_ticket,
            self.err,
            cluster,
            segmax,
            dbg,
            stats,
            self.tk_pp if pre else None,
            pp_min,
        )

    def resident_clusters(self, cluster: int, segmax: int, pre: bool) -> int:
        """Top-k clusters of ``cluster`` CTAs (``segmax`` staged keys each; ``pre``:
        a pre-pass launch, ``topk_prepass``) that fit on the GPU at once; a launch
        walks rows beyond it in turns."""
        return int(self.ext.topk_resident_clusters(cluster, segmax, pre))

    def set_spin_timeout(self, seconds: float) -> None:
        """Cross-rank spin deadline of the kernels launched (or captured) from now
        on; past it a kernel records an error code and traps. ``<= 0``: wait
        forever. The process default is 5 s (tests, warmup: a missing peer fails
        fast)."""
        self.ext.set_spin_ns(int(seconds * 1e9) if seconds > 0 else 0)

    def check(self) -> None:
        """Raise if a kernel recorded an error (debug / warmup; syncs)."""
        code = int(self.err.item())
        if code:
            raise RuntimeError(f"DcpExchange error {code}: {_ERR_CODES.get(code, '?')}")

    # ------------------------------------------------------------------ O
    def o_exchange(
        self,
        y: torch.Tensor,
        lse: torch.Tensor,
        out: torch.Tensor,
        *,
        lse_base_e: bool = True,
        out_lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        D = y.shape[2]
        s = self.ox[D]
        self.ext.o_exchange(
            y,
            lse,
            lse_base_e,
            out,
            out_lse,
            s["box"].data_ptr(),
            s["peers"],
            self.W,
            self.rank,
            self.hpr,
            self.cap,
            s["off_data"],
            s["off_lse"],
            s["epoch"],
            s["ticket"],
            self.err,
            s["grid"],
        )
        return out


# ---------------------------------------------------------------- DCP layout
def local_to_global(pos: torch.Tensor, rank: int, world: int, interleave: int) -> torch.Tensor:
    """Local token position on ``rank`` -> global token id (interleaved DCP)."""
    return ((pos // interleave) * world + rank) * interleave + pos % interleave


def local_len_of(global_len: torch.Tensor, rank: int, world: int, interleave: int) -> torch.Tensor:
    """Tokens of a ``global_len`` context that ``rank`` owns."""
    full = global_len // (world * interleave)
    rem = global_len - full * world * interleave
    extra = (rem - rank * interleave).clamp(0, interleave)
    return full * interleave + extra
