# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA sparse core (``DsaCore``) in ``PreparedDsa``'s DCP decode.

On pure-decode steps whose top-k DcpExchange serves (``DcpxDecode.counts_for``: the
same step predicate as the indexer op, so every rank and the MLA layer agree), at
most ``rows`` rows, ``PreparedDsa.run_dcp`` replaces

    MlaQuery (q publish + gathered-q poll) .. DcpExchange.topk_merge
    -> FlashInfer sparse decode over the gathered q mailbox (trtllm-gen + its
       reduction / LSE glue)
    -> DcpValue (push, merge, W_UV, gate)

by

    MlaQuery (q publish only) .. DcpExchange.topk_merge
    -> DsaCore (polls its q tiles, sparse MLA, O + LSE pushed to the head owners)
    -> DcpValue(pushed=True) (merge, W_UV, gate)

Switch: ``VLLM_K31_DSA_CORE`` = 1 (default: per-tensor FP8 KV under DCP wherever
DcpExchange's top-k is) | 0; ``VLLM_K31_DSA_CORE_ROWS`` = most rows (default 64, see
ROWS; at most the DcpBox cap of 64); ``VLLM_K31_DSA_CORE_VARIANT`` = the core variant by rows
(``batch_table`` syntax, see VARIANT); ``VLLM_K31_DSA_CORE_SPREAD`` = rows whose
pushed DcpValue spreads its merge (see SPREAD). ``VLLM_K31_DSA_CORE_STATS=1`` logs the kernel's error word
at exit.
"""

from __future__ import annotations

import atexit
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_K31_DSA_CORE"
ROWS_ENV = ENV + "_ROWS"
# KV splits per (row, 32 heads) by rows: spread a small batch's local chunks over
# more CTAs (the last one LSE-merges them). DCP16 layer bench (kernel RESULTS).
# Most rows the core serves by default (the DcpBox cap). 16-rank step A H S S H A on
# one allocation (dummy weights, AgentX, indexer fill; A = trtllm chain, H = these
# defaults without the spread merge, S = with it, as shipped), us/step A -> S: B1 q8
# 5705 -> 5416, B8 q8 11664 -> 11447, B1 q4 5852 -> 5601, B8 q4 9449 -> 8825.
# Caveat: a row whose selection sits on one rank (a ~100K-way tie at the cut, e.g. all
# index keys zero) walks its chunks serially: DCP4 skew bench B8 q8 97 -> 122 us/layer.
ROWS = 64
VARIANT_ENV = ENV + "_VARIANT"
# Core variant by rows: "u" = the tcgen05 core (CTA per (row, latent half), all 96
# gathered heads as M = 128), "<S>x<HG>" = the mma.sync core with S KV splits (one
# cluster; rows of <= 4 chunks still on one CTA) per (row, HG = 16 / 32 gathered
# heads). DCP16 layer bench (16 x VR200, AgentX, us per layer incl. logits + top-k,
# max rank; B1 q8 / B1 q4 / B8 q4 / B8 q8): trtllm chain 65.5 / 61.3 / 71.2 / 74.8;
# 1x16 51.0 / 48.7 / 59.0 / 76.8; 1x32 54.9 / 52.4 / 63.7 / 82.8; u 55.4 / 53.2 / 63.0
# / 78.0; 2x16 50.4 / 48.4 / 66.2 / 91.7 (kernel RESULTS).
# Up to 16 rows two splits per (row, 16 heads): free on balanced rows (rows of <= 4
# chunks run unsplit) and half the serial chunks when one rank holds most of a row's
# entries (ties at the cut go to ranks in rank order: zero index keys put ~2040 of 2048
# on rank 0). 9-layer real-weight bench without the index fill (that skewed case), us
# per 9-layer step, these defaults vs the trtllm chain: B1 884 vs 844, B8 agentx 1738
# vs 1496 (+16 %), mix 1638 vs 1416. Real indexer logits are balanced (96-155 entries
# per rank-row at DCP16), where 1x16 at 64 rows wins (step ABBA under ROWS); set
# VLLM_K31_DSA_CORE_ROWS=32 to keep 33-64 rows on the trtllm chain if skewed
# selections matter. Open: a split that adapts to the row length without idle CTAs.
VARIANT = "1-16:2x16,17-64:1x16"
SPREAD_ENV = ENV + "_SPREAD"
# Rows whose pushed DcpValue spreads every row's merge over its CTAs (Rubin schedule,
# LSE box [CAP, H, DCP]; spec verify rows only) instead of one merge CTA per (head, 16
# rows). DCP16 layer bench with 1x16: B8 q4 59.0 -> 55.9, B8 q8 76.8 -> 73.7 us. Step
# A H S S H A (see ROWS), us/step H -> S: B8 q8 11511 -> 11447 (-64), B8 q4 8889 ->
# 8825 (-64); B1 rows (<= 16) do not spread. Applies only where the device has more
# than 152 SMs (PreparedDsa.run_dcp). 9-layer real-weight R.1 check with it on passes
# (rel rms as off) and still flags an injected softmax-scale fault.
SPREAD = "17-64:1"


def parse_variant(name: str) -> tuple[int, int]:
    """(splits, heads per CTA; 0 = the tcgen05 core) of a variant name."""
    if name == "u":
        return 1, 0
    splits, _, heads = name.partition("x")
    if int(splits) not in (1, 2, 4, 8) or int(heads) not in (16, 32):
        raise ValueError(f"{VARIANT_ENV}: bad variant {name!r}")
    return int(splits), int(heads)


def wanted(dcpx) -> bool:
    mode = os.environ.get(ENV, "1").strip().lower()
    if mode in ("0", "off", "false") or dcpx is None:
        return False
    return "topk" in dcpx.parts


class DsacDecode:
    def __init__(self, dcpx, device: torch.device, heads: int):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
            batch_table,
        )

        from .ops import DsaCore

        self.dcpx = dcpx
        self.rows = min(int(os.environ.get(ROWS_ENV, str(ROWS))), dcpx.cap)
        spec = os.environ.get(VARIANT_ENV, VARIANT).replace(";", ",")  # (";": shells)
        self.variant = {b: parse_variant(v) for b, v in batch_table(spec).items()}
        spread = os.environ.get(SPREAD_ENV, SPREAD).replace(";", ",")
        self.spread = {b for b, v in batch_table(spread).items() if v}
        max_splits = max([1, *(v[0] for v in self.variant.values())])
        self.core = DsaCore(device, cap=self.rows, heads=heads, max_splits=max_splits)
        logger.info(
            "Kimi K3.1 DSA decode: sparse core on (rows <= %d, variants %s)",
            self.rows,
            spec,
        )
        if os.environ.get(ENV + "_STATS") == "1":
            atexit.register(self._report)

    def _report(self) -> None:
        try:
            err = int(self.core.err.item())
        except Exception as e:  # (CUDA already torn down)
            logger.warning("Kimi K3.1 DSA core stats unavailable: %s", e)
            return
        print(f"K31 DSAC stats: err {err}", flush=True)

    def counts(self, prefix: str, rows: int) -> torch.Tensor | None:
        """This step's local top-k counts if the core serves it, else None."""
        if rows > self.rows:
            return None
        return self.dcpx.counts_for(prefix, rows)

    def run(self, prepared, batch, sel, counts, cache, box, parity, lse_wide=False):
        """``lse_wide``: the pushed DcpValue spreads its merge (LSE box [CAP, H, DCP])."""
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

        md = get_forward_context().attn_metadata[prepared.layer.layer_name]
        kv, _ = flat_kv_row_view(cache.view(torch.uint8), md.block_size)
        prepared.impl.record_logical_topk_ready()
        self.core.run(
            box.q[parity],
            sel,
            counts,
            kv,
            bmm1_scale=prepared.core_bmm1,
            bmm2_scale=prepared.core_bmm2,
            part_peers=box.peers,
            lse_peers=box.lse_peers,
            box_cap=box.cap,
            hpr=box.heads,
            parity=parity,
            rank=box.rank,
            rows=batch,
            lse_wide=lse_wide,
            **self.launch(batch),
        )

    def launch(self, batch: int) -> dict:
        splits, heads = self.variant.get(batch, (1, 32))
        return dict(splits=splits, heads_per_cta=heads or 32, umma=heads == 0)


def build(model, dcpx) -> DsacDecode | None:
    """Build the core and attach it to every DSA indexer (``PreparedDsa`` picks it up
    on a per-tensor FP8 cache)."""
    if not wanted(dcpx):
        return None
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention

    layers = [m for m in model.modules() if isinstance(m, KimiK3DSAAttention)]
    if not layers:
        return None
    device = layers[0].mla_attn.mla_attn.W_UK_T.device
    dsac = DsacDecode(dcpx, device, dcpx.heads)
    for layer in layers:
        layer.dsa_indexer.dsac = dsac
    return dsac
