# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA sparse core (``DsaCore``) in ``PreparedDsa``'s DCP decode.

On DCP decode steps whose top-k ``IndexerStage.merge_rows`` serves (``MergeRows``:
this rank's compacted flat KV rows and their counts from the #63 merge, or from
DcpExchange's top-k on the steps it admits, ``merge_source``, whose step predicate
every rank shares), at most ``rows`` rows, ``PreparedDsa.run_dcp`` replaces

    MlaQuery (q publish) + QueryWait (gathered-q poll) .. merge_rows
    -> FlashInfer's trtllm-gen sparse decode over the gathered q mailbox (+ its
       reduction / LSE kernels)
    -> DcpValue (push, merge, W_UV, gate)

by

    MlaQuery (q publish) .. merge_rows
    -> DsaCore (polls its q tiles, sparse MLA, O + LSE pushed to the head owners)
    -> DcpValue(pushed=True) (merge, W_UV, gate)

Switch: ``VLLM_K31_DSA_CORE`` = 0 (default) | 1; ``VLLM_K31_DSA_CORE_ROWS`` = most
rows (default ROWS; at most the DcpBox cap of 64); ``VLLM_K31_DSA_CORE_VARIANT`` = the
core variant by rows (``batch_table`` syntax) of every q (default VARIANT[q]: rows of
a q without a table keep FlashInfer); ``VLLM_K31_DSA_CORE_SKIP_LAYERS`` = DSA layers
(ordinals 0 .. 23, ';' or ',' separated) that keep FlashInfer;
``VLLM_K31_DSA_CORE_SPIN_S`` = how long a served core's q mailbox poll waits for a
peer before it traps (default 600 s; 0: forever; see DEFAULT_SPIN_S);
``VLLM_K31_DSA_CORE_SPREAD`` = rows whose pushed DcpValue spreads its merge (see
SPREAD). ``VLLM_K31_DSA_CORE_STATS=1`` logs the kernel's error word at exit, and (with
the core on or off) per DSA layer and rows the histogram of each step's largest local
count (``CountStats``).

The core polls its q tiles before its PDL wait, which is safe only while the kernel
right before it does not release its dependents early: the #63 merge's rescue kernel,
or DcpExchange's top-k above ``K31_DCPX_PDL`` rows (``MergeRows`` refuses the core on
the exchange's steps of at most that many rows: K31_DCPX_PDL=0 gives it all of them).
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
# Here 16 (DL7 M8 / M16, see VARIANT): from 24 rows no variant holds up where one rank
# holds 600 or more of a request's entries.
ROWS = 16
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
# "<S>x<HG>s<M>": rows of at most M chunks run on one CTA (default 4).
# Here by q (the rows of q-row requests). Round 3 Step 0a, one SM107 GPU, DCP16 rank
# shapes (96 gathered heads, q8, slots over 1M FP8 KV rows), CUPTI us per call, the
# trtllm-gen chain (+ reduction + LSE) / 1x16 / 2x16s2 / 4x16s2 / 8x16s4, where the
# 8 rows of one request hold X local entries and the other rows 96-155 ("bal"):
#   M8   bal 13.47 / 9.48 / 9.45 / 8.73 / 10.43; X 256: 13.54 / 12.75 / 10.59 / 15.84
#        / 14.18; 384: 13.61 / 17.47 / 12.98 / 20.45 / 25.50; 600: 14.74 / 26.13 /
#        17.69 / 25.20 / 32.30; 2048: 17.17 / 78.56 / 43.72 / 49.28 / 47.25
#   M16  bal 14.34 / 9.65 / 9.71 / 16.25 / 17.66; 256: 14.44 / 12.85 / 10.69 / 17.70 /
#        21.10; 384: 14.65 / 17.56 / 13.07 / 21.21 / 29.72; 600: 15.71 / 26.22 / 17.78 /
#        25.34 / 34.05
#   M24  bal 14.98 / 9.83 / 16.19 (2 waves); 384: 15.25 / 17.64 / 16.99; 600: 15.91 /
#        26.30 / 17.89
# and DcpValue (push + merge) 8.96 / 11.82 us at M8 / M16 against DcpValue(pushed)
# 7.00 / 9.59, QueryWait 1.84 (side stream). 2x16s2 at 1-16 rows: -6.0 / -6.9 us per
# layer balanced, -2.6 / -3.8 at 384 entries, +1.0 / -0.2 at 600, +24 at 2048.
VARIANT = {8: "1-16:2x16s2"}
SKIP_ENV = ENV + "_SKIP_LAYERS"
STATS_ENV = ENV + "_STATS"
# CountStats' bins of a step's largest local count (the core walks its longest row's
# 64-key chunks serially): up to each of these, and above the last.
COUNT_BINS = (64, 128, 192, 256, 384, 640, 1024)
SPREAD_ENV = ENV + "_SPREAD"
# Rows whose pushed DcpValue spreads every row's merge over its CTAs (Rubin schedule,
# LSE box [CAP, H, DCP]; spec verify rows only) instead of one merge CTA per (head, 16
# rows). DCP16 layer bench with 1x16: B8 q4 59.0 -> 55.9, B8 q8 76.8 -> 73.7 us. Step
# A H S S H A (see ROWS), us/step H -> S: B8 q8 11511 -> 11447 (-64), B8 q4 8889 ->
# 8825 (-64); B1 rows (<= 16) do not spread. Applies only where the device has more
# than 152 SMs (PreparedDsa.run_dcp). 9-layer real-weight R.1 check with it on passes
# (rel rms as off) and still flags an injected softmax-scale fault.
SPREAD = "17-64:1"
SPIN_ENV = ENV + "_SPIN_S"
# The served core's q mailbox poll deadline (s; ``build``): a peer stalled on the host
# (cold JIT, GC, a P/D handshake) then only delays this rank, as with DcpExchange's
# serving deadline (VLLM_K31_DCP_EXCHANGE_SPIN_S, 600 s) and DcpBox's polls (none).
# DsaCore's own 5 s stays for tests.
DEFAULT_SPIN_S = 600.0


def serving_spin_ns() -> int:
    """The served core's q mailbox poll deadline in ns (``SPIN_ENV``; 0 or less: a
    deadline no poll reaches)."""
    seconds = float(os.environ.get(SPIN_ENV, str(DEFAULT_SPIN_S)))
    return int(seconds * 1e9) if seconds > 0 else 1 << 62


def parse_variant(name: str) -> tuple[int, int, int]:
    """(splits, heads per CTA; 0 = the tcgen05 core, split_min) of a variant name."""
    if name == "u":
        return 1, 0, 4
    name, _, split_min = name.partition("s")
    splits, _, heads = name.partition("x")
    if int(splits) not in (1, 2, 4, 8) or int(heads) not in (16, 32):
        raise ValueError(f"{VARIANT_ENV}: bad variant {name!r}")
    return int(splits), int(heads), int(split_min or 4)


def rows_table(spec: str) -> dict:
    from ...specialized_decode.cutedsl.common.policy import batch_table

    return batch_table(spec)


def layer_list(spec: str) -> frozenset[int]:
    """DSA layer ordinals of a ';' / ',' separated list."""
    return frozenset(int(x) for x in spec.replace(";", ",").split(",") if x.strip())


def wanted() -> bool:
    return os.environ.get(ENV, "0").strip().lower() in ("1", "on", "true")


class MergeRows:
    """The core's (sel, cnt) source on PreparedDsa's DCP steps: the layer's ``stage``
    (IndexerStage.merge_rows: this rank's flat KV rows [rows, K], a compacted prefix,
    and their counts [rows]) from the #63 merge, or on the steps DcpExchange's top-k
    admits (VLLM_K31_DCP_EXCHANGE) its slots and counts, which Wei's DsacDecode read
    (DcpxDecode.counts_for). ``pdl_rows``: the most rows whose DcpExchange top-k
    launches with PDL (dcp_exchange.decode.topk_pdl_rows), whose steps keep
    FlashInfer (see the module docstring)."""

    def __init__(self, stage, pdl_rows: int = 0):
        self.stage, self.pdl_rows = stage, pdl_rows

    def fits(self, batch: int) -> bool:
        """Whether the core serves the step's ``batch`` rows: ``merge_rows`` gives
        them (``merge_source``: every rank takes the same branch), and not from a
        top-k launched with PDL."""
        source, rows = self.stage.merge_source(batch)
        if source == "exchange":
            return rows > self.pdl_rows
        return source == "merge"

    def rows(self, *args, q: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """(sel, cnt) of the step: ``merge_rows(*args, q)``, on a step ``fits``
        accepted."""
        merged = self.stage.merge_rows(*args, q)
        assert merged is not None, "merge_rows declined a step fits accepted"
        return merged


class DsacDecode:
    """The core of one device, shared by every DSA layer (stream-ordered launches):
    ``heads`` gathered query heads (DCP x local heads), at most ``cap`` rows (the
    DcpBox's). ``rows``, ``spec`` (every q's variant table) and ``skip`` (DSA
    layers) override their environment switches (tests); ``pdl_rows``: MergeRows'."""

    def __init__(
        self,
        device: torch.device,
        heads: int,
        cap: int = 64,
        rows: int | None = None,
        spec: str | None = None,
        skip: str | None = None,
        pdl_rows: int = 0,
    ):
        from .ops import DsaCore

        rows = int(os.environ.get(ROWS_ENV, str(ROWS))) if rows is None else rows
        self.rows = min(rows, cap)
        # every q's variant table, else VARIANT[q]
        self.spec = os.environ.get(VARIANT_ENV) if spec is None else spec
        self.tables: dict[int, dict] = {}
        self.skip = layer_list(os.environ.get(SKIP_ENV, "") if skip is None else skip)
        spread = os.environ.get(SPREAD_ENV, SPREAD).replace(";", ",")
        self.spread = {b for b, v in rows_table(spread).items() if v}
        self.core = DsaCore(device, cap=self.rows, heads=heads, max_splits=8)
        self.pdl_rows = pdl_rows
        logger.info(
            "Kimi K3.1 DSA decode: sparse core on (rows <= %d, variants %s, skipped "
            "DSA layers %s)",
            self.rows,
            VARIANT if self.spec is None else f"{self.spec} at every q",
            sorted(self.skip) or "none",
        )
        if pdl_rows > 0:
            logger.warning(
                "Kimi K3.1 DSA sparse core: not on DcpExchange top-k steps of <= %d "
                "rows, whose top-k launches with PDL (K31_DCPX_PDL; the core polls "
                "its q tiles before its PDL wait); K31_DCPX_PDL=0 gives it them",
                pdl_rows,
            )
        if os.environ.get(STATS_ENV) == "1":
            atexit.register(self._report)

    def _report(self) -> None:
        try:
            err = int(self.core.err.item())
        except Exception as e:  # (CUDA already torn down)
            logger.warning("Kimi K3.1 DSA core stats unavailable: %s", e)
            return
        print(f"K31 DSAC stats: err {err}", flush=True)

    def variant(self, batch: int, q: int, layer: int) -> tuple[int, int, int] | None:
        """The core variant (``parse_variant``) of DSA layer ``layer`` (ordinal) at
        ``batch`` rows of q-row requests, or None: FlashInfer's chain."""
        if batch > self.rows or layer in self.skip:
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
            heads_per_cta=heads or 32,
            umma=heads == 0,
            lse_wide=lse_wide,
        )


class CountStats:
    """``VLLM_K31_DSA_CORE_STATS=1`` (the core on or off): per (DSA layer, rows) the
    histogram over steps of the largest local count of the step's rows (the #63
    merge's counts on this rank; COUNT_BINS), accumulated on the device inside the
    captured graphs and printed at exit. The core's time follows its longest row."""

    def __init__(self, device: torch.device):
        self.edges = torch.tensor(COUNT_BINS, dtype=torch.int32, device=device)
        self.one = torch.ones(1, dtype=torch.int64, device=device)
        self.hist: dict[tuple[int, int], torch.Tensor] = {}
        atexit.register(self.report)

    def reserve(self, layer: int, rows: int) -> None:
        """The histogram of (layer, rows), allocated before any capture."""
        if (layer, rows) not in self.hist:
            bins = len(COUNT_BINS) + 1
            self.hist[layer, rows] = self.one.new_zeros(bins)

    def record(self, layer: int, rows: int, counts: torch.Tensor) -> None:
        hist = self.hist.get((layer, rows))
        if hist is not None:
            top = counts[:rows].amax().view(1)
            hist.index_add_(0, torch.bucketize(top, self.edges), self.one)

    def report(self) -> None:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        names = [f"<={b}" for b in COUNT_BINS] + [f">{COUNT_BINS[-1]}"]
        try:
            hists = {k: v.tolist() for k, v in sorted(self.hist.items())}
        except Exception as e:  # (CUDA already torn down)
            logger.warning("Kimi K3.1 DSA count stats unavailable: %s", e)
            return
        for (layer, rows), counts in hists.items():
            if any(counts):
                bins = " ".join(f"{n}:{c}" for n, c in zip(names, counts) if c)
                print(f"K31 DSAC counts rank {rank} layer {layer} rows {rows}: {bins}")


_STATS: dict = {}


def count_stats(device: torch.device) -> CountStats | None:
    """The device's CountStats under VLLM_K31_DSA_CORE_STATS=1, else None."""
    if os.environ.get(STATS_ENV) != "1":
        return None
    if device not in _STATS:
        _STATS[device] = CountStats(device)
    return _STATS[device]


def build(model) -> DsacDecode | None:
    """Build the core (``VLLM_K31_DSA_CORE=1``, DCP) and attach it to every DSA
    indexer (``PreparedDsa`` picks it up on a per-tensor FP8 cache with the #63
    merge). DcpExchange (built at weight load) has registered its top-k hooks by
    now: on layers that have one, the core also takes the exchange's steps."""
    from vllm.model_executor.layers.sparse_attn_indexer import DCP_DECODE_TOPK_HOOKS
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention

    layers = [m for m in model.modules() if isinstance(m, KimiK3DSAAttention)]
    for ordinal, layer in enumerate(layers):  # (SKIP_LAYERS)
        layer.dsa_indexer.dsac_layer = ordinal
    if not wanted() or not layers or layers[0].dsa_indexer.dcp_world_size < 2:
        return None
    first = layers[0]
    device = first.mla_attn.mla_attn.W_UK_T.device
    heads = first.num_local_heads * first.dsa_indexer.dcp_world_size
    pdl_rows = 0
    prefixes = {layer.dsa_indexer.k_cache.prefix for layer in layers}
    if prefixes & DCP_DECODE_TOPK_HOOKS.keys():
        from ..dcp_exchange.decode import topk_pdl_rows

        pdl_rows = topk_pdl_rows()
    dsac = DsacDecode(device, heads, pdl_rows=pdl_rows)
    for layer in layers:
        layer.dsa_indexer.dsac = dsac
    dsac.core.spin_ns = serving_spin_ns()
    return dsac
