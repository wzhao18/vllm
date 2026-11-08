# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA sparse core (``DsaCore``) in ``PreparedDsa``'s DCP decode.

On DCP decode steps whose top-k ``IndexerStage.merge_rows`` serves (``MergeRows``:
this rank's compacted flat KV rows and their counts from the #63 merge, or from
DcpExchange's top-k on the steps it admits, ``merge_source``, whose step predicate
every rank shares), at most ``ROWS`` rows of a verify length in ``VARIANT``,
``PreparedDsa.run_dcp`` replaces

    MlaQuery (q publish) + QueryWait (gathered-q poll) .. merge_rows
    -> FlashInfer's trtllm-gen sparse decode over the gathered q mailbox (+ its
       reduction / LSE kernels)
    -> DcpValue (push, merge, W_UV, gate)

by

    MlaQuery (q publish) .. merge_rows
    -> DsaCore (polls its q tiles, sparse MLA, O + LSE pushed to the head owners)
    -> DcpValue(pushed=True) (merge, W_UV, gate)

The core computes the stock FP8 sparse MLA decode's (trtllm-gen's) precision at every
stage (csrc/dsa_core.cu): trtllm-gen's KV CTA split of the batch's rows
(ops.trtllm_kv_ctas: a row's 128-key tiles in pieces of ceil(tiles / C) consecutive
tiles), per piece E4M3 Q K with FP32 accumulation, FP32 base-2 softmax with one
running max step per 128-key tile, P = E4M3(448 x 2^(s - m)) with the row sum from
the unquantized P, FP32 P V accumulation, the pieces' BF16-rounded unnormalized
partials merged in FP32.

Policy (``build``, no switch): on wherever it can run, i.e. a decode under DCP (TP =
DCP > 1, not the sequence-parallel decode, which runs its own DSA path) over a
per-tensor FP8 MLA cache on an SM100-family GPU, at the verify lengths of ``VARIANT``;
elsewhere (nvfp4_ds_mla, no DCP, SP) FlashInfer's chain, logged at startup.

The core polls its q tiles before its PDL wait, which is safe because the kernel right
before it never releases its dependents early: the #63 merge's rescue kernel, or
DcpExchange's top-k (launched without PDL).
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Most rows the core serves (at most the DcpBox cap of 64). 16-rank step A H S S H A on
# one allocation (dummy weights, AgentX, indexer fill; A = trtllm chain, H = these
# defaults without the spread merge, S = with it), us/step A -> S: B1 q8 5705 -> 5416,
# B8 q8 11664 -> 11447, B1 q4 5852 -> 5601, B8 q4 9449 -> 8825.
# Caveat: a row whose selection sits on one rank (a ~100K-way tie at the cut, e.g. all
# index keys zero) walks its tiles serially: DCP4 skew bench B8 q8 97 -> 122 us/layer.
# Here 16 (DL7 M8 / M16, see VARIANT): from 24 rows no variant holds up where one rank
# holds 600 or more of a request's entries.
ROWS = 16
# Core variant by rows, per verify length q: "<S>x<HG>[s<M>]" = S KV splits (one
# cluster; rows of at most M 64-key chunks, default 4, on one CTA) per (row, HG = 16 /
# 32 gathered heads). A q without a table keeps FlashInfer's chain.
# Round 3 Step 0a (FP16-P core, before the stock-precision P), one SM107 GPU, DCP16
# rank shapes (96 gathered heads, q8, slots over 1M FP8 KV rows), CUPTI us per call,
# the trtllm-gen chain (+ reduction + LSE) / 1x16 / 2x16s2 / 4x16s2 / 8x16s4, where the
# 8 rows of one request hold X local entries and the other rows 96-155 ("bal"):
#   M8   bal 13.47 / 9.48 / 9.45 / 8.73 / 10.43; X 256: 13.54 / 12.75 / 10.59 / 15.84
#        / 14.18; 384: 13.61 / 17.47 / 12.98 / 20.45 / 25.50; 600: 14.74 / 26.13 /
#        17.69 / 25.20 / 32.30; 2048: 17.17 / 78.56 / 43.72 / 49.28 / 47.25
#   M16  bal 14.34 / 9.65 / 9.71 / 16.25 / 17.66; 256: 14.44 / 12.85 / 10.69 / 17.70 /
#        21.10; 384: 14.65 / 17.56 / 13.07 / 21.21 / 29.72; 600: 15.71 / 26.22 / 17.78 /
#        25.34 / 34.05
# and DcpValue (push + merge) 8.96 / 11.82 us at M8 / M16 against DcpValue(pushed)
# 7.00 / 9.59, QueryWait 1.84 (side stream). 2x16s2 at 1-16 rows: -6.0 / -6.9 us per
# layer balanced, -2.6 / -3.8 at 384 entries, +1.0 / -0.2 at 600, +24 at 2048.
# Stock precision (the core computes trtllm-gen's per-tile E4M3 P and BF16 partials,
# csrc/dsa_core.cu), re-measured on one VR200 (kern-dsa bench_core, CUDA graphs, us
# per call, DCP16 rank shapes q8), trtllm-gen chain / stock-precision 2x16s2 (/ the
# FP16-P core it replaced): M8 bal 14.44 / 9.17 (/ 8.36), x256 15.10 / 9.80 (/ 9.87),
# x384 15.13 / 14.14 (/ 12.22), x600 16.77 / 18.14 (/ 16.59), x2048 19.12 / 42.80
# (/ 42.74); M16 bal 15.70 / 9.43 (/ 8.69), x256 15.59 / 9.83, x384 15.71 / 14.16,
# x600 16.81 / 18.14, x2048 18.21 / 42.81. Still -5.3 / -6.3 us per layer on the
# balanced rows of real indexer logits (96-155 local entries per rank-row); a row whose
# selection sits on one rank (x600 and up) is slower, as before.
# With trtllm-gen's KV CTA pieces (ops.trtllm_kv_ctas: rows of more tiles than KV CTAs
# keep a running max across a piece's tiles; bitwise stock on such rows too), same
# bench, trtllm-gen chain / core with pieces (/ per-tile core): M8 bal 14.25 / 9.46
# (/ 9.15), x256 15.01 / 10.04, x384 15.10 / 14.60, x600 16.87 / 18.58, x2048 18.88 /
# 45.10; M16 bal 15.63 / 9.73 (/ 9.42), x256 15.50 / 10.13, x384 15.65 / 14.62, x600
# 16.59 / 18.60, x2048 18.60 / 49.45 (envs lane exact4): -4.8 / -5.9 us per layer on
# balanced rows.
VARIANT = {8: "1-16:2x16s2"}
# The served core's q mailbox poll deadline (s; ``build``): a peer stalled on the host
# (cold JIT, GC, a P/D handshake) then only delays this rank, as with DcpExchange's
# serving deadline (600 s) and DcpBox's polls (none). DsaCore's own 5 s stays for tests.
SERVING_SPIN_S = 600.0


def parse_variant(name: str) -> tuple[int, int, int]:
    """(splits, heads per CTA, split_min) of a variant name."""
    base, _, split_min = name.partition("s")
    splits, _, heads = base.partition("x")
    try:
        parsed = int(splits), int(heads), int(split_min or 4)
    except ValueError:
        parsed = 0, 0, 0
    if parsed[0] not in (1, 2, 4, 8) or parsed[1] not in (16, 32) or parsed[2] < 1:
        raise ValueError(f"DSA sparse core: bad variant {name!r}")
    return parsed


def rows_table(spec: str) -> dict:
    from ...specialized_decode.cutedsl.common.policy import batch_table

    return batch_table(spec)


class MergeRows:
    """The core's (sel, cnt) source on PreparedDsa's DCP steps: the layer's ``stage``
    (IndexerStage.merge_rows: this rank's flat KV rows [rows, K], a compacted prefix,
    and their counts [rows]) from the #63 merge, or on the steps DcpExchange's top-k
    admits its slots and counts (DcpxDecode.counts_for)."""

    def __init__(self, stage):
        self.stage = stage

    def fits(self, batch: int) -> bool:
        """Whether the core serves the step's ``batch`` rows: ``merge_rows`` gives
        them (``merge_source``: every rank takes the same branch)."""
        source, _ = self.stage.merge_source(batch)
        return source in ("merge", "exchange")

    def rows(self, *args, q: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """(sel, cnt) of the step: ``merge_rows(*args, q)``, on a step ``fits``
        accepted."""
        merged = self.stage.merge_rows(*args, q)
        assert merged is not None, "merge_rows declined a step fits accepted"
        return merged


class DsacDecode:
    """The core of one device, shared by every DSA layer (stream-ordered launches):
    ``heads`` gathered query heads (DCP x local heads), at most ``cap`` rows (the
    DcpBox's). ``rows`` and ``spec`` (one variant table for every q) replace ROWS and
    VARIANT (tests and benchmarks)."""

    def __init__(
        self,
        device: torch.device,
        heads: int,
        cap: int = 64,
        rows: int | None = None,
        spec: str | None = None,
    ):
        from .ops import DsaCore

        self.rows = min(ROWS if rows is None else rows, cap)
        self.spec = spec  # every q's variant table, else VARIANT[q]
        self.tables: dict[int, dict] = {}
        self.core = DsaCore(device, cap=self.rows, heads=heads, max_splits=8)
        logger.info(
            "Kimi K3.1 DSA decode: sparse core on (rows <= %d, variants %s)",
            self.rows,
            VARIANT if self.spec is None else f"{self.spec} at every q",
        )

    def variant(self, batch: int, q: int) -> tuple[int, int, int] | None:
        """The core variant (``parse_variant``) at ``batch`` rows of q-row requests
        (every DSA layer alike), or None: FlashInfer's chain."""
        if batch > self.rows:
            return None
        if q not in self.tables:
            spec = self.spec if self.spec is not None else VARIANT.get(q, "")
            table = rows_table(spec.replace(";", ","))
            self.tables[q] = {b: parse_variant(n) for b, n in table.items()}
        return self.tables[q].get(batch)

    def run(
        self,
        variant,
        q_box,
        sel,
        counts,
        kv,
        scales,
        box,
        hpr,
        parity,
        batch,
        lse_wide=False,
    ):
        """``variant``: ``self.variant``'s; ``q_box``: this parity's q mailbox;
        ``sel`` / ``counts``: MergeRows.rows; ``kv``: the flat FP8 KV rows;
        ``scales``: FlashInfer's (bmm1, bmm2) of the layer; ``box``: the DSA DcpBox
        (``hpr`` heads per rank); ``lse_wide``: the pushed DcpValue spreads its
        merge (LSE box [CAP, H, DCP])."""
        bmm1, bmm2 = scales
        splits, heads, split_min = variant
        self.core.run(
            q_box,
            sel,
            counts,
            kv,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            part_peers=box.peers,
            lse_peers=box.lse_peers,
            box_cap=box.cap,
            hpr=hpr,
            parity=parity,
            rank=box.rank,
            rows=batch,
            splits=splits,
            split_min=split_min,
            heads_per_cta=heads,
            lse_wide=lse_wide,
        )


def unsupported(model, config=None) -> str | None:
    """Why the deployment of ``model`` (``config``: its specialized DecodeConfig) runs
    FlashInfer's chain instead of the core, or None (the core runs)."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.platforms import current_platform

    layers = [m for m in model.modules() if isinstance(m, KimiK3DSAAttention)]
    if not layers:
        return "no DSA layers"
    if layers[0].dsa_indexer.dcp_world_size < 2:
        return "no DCP"
    if config is not None and config.sequence_parallel:
        return "the sequence-parallel decode (its own DSA path, SpDsaCore)"
    if config is not None and config.q not in VARIANT:
        return f"verify length {config.q} (variant table only at q {sorted(VARIANT)})"
    dtype = layers[0].mla_attn.mla_attn.kv_cache_dtype
    if dtype == "nvfp4_ds_mla":
        return "an nvfp4_ds_mla MLA cache (the core reads per-tensor FP8 rows)"
    if not current_platform.is_device_capability_family(100):
        return "not an SM100-family GPU (mma.sync e4m3, clusters)"
    return None


def build(model, config=None) -> DsacDecode | None:
    """Build the core where it can run (``unsupported``; ``config``: the specialized
    DecodeConfig) and attach it to every DSA indexer (``PreparedDsa`` picks it up on a
    per-tensor FP8 cache under DCP). DcpExchange (built at weight load) has registered
    its top-k hooks by now: on layers that have one, the core also takes the
    exchange's steps."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention

    layers = [m for m in model.modules() if isinstance(m, KimiK3DSAAttention)]
    for ordinal, layer in enumerate(layers):
        layer.dsa_indexer.dsac_layer = ordinal
    reason = unsupported(model, config)
    if reason is not None:
        if layers and layers[0].dsa_indexer.dcp_world_size > 1:
            logger.info_once("Kimi K3.1 DSA decode: sparse core off (%s)", reason)
        return None
    first = layers[0]
    device = first.mla_attn.mla_attn.W_UK_T.device
    heads = first.num_local_heads * first.dsa_indexer.dcp_world_size
    dsac = DsacDecode(device, heads)
    for layer in layers:
        layer.dsa_indexer.dsac = dsac
    dsac.core.spin_ns = int(SERVING_SPIN_S * 1e9)
    return dsac
