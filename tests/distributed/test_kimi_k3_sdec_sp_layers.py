# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP specialized decode, per layer, against the stock layer on one node
(4 Rubin GPUs: TP4 = DCP4 = the SP group, DP1; C512's decode attention at full
size: hidden 7168, 96 heads (24 per rank), q4 DSpark verify rows R = 4 n, n = 1..128
requests per DP group, 0.25..32 per GPU).

Every case builds a real stock layer (``KimiK3DSAAttention`` on FLASHINFER_MLA_SPARSE
trtllm-gen, per-tensor FP8 MLA KV, MXFP4 or FP8 indexer; ``KimiK3DeltaAttention``
with an FP32 state and RecoverSSM records, ``--use-replayssm``), random weights
(each rank its own shard), and runs one decode step through

  stock   sp_all_gather(x_shard)[:R] -> layer forward (o_proj not reducing, as the
          SP decoder sets it) -> sp_reduce_scatter      (the stock decoder's SP seam)
  ours    runtime.sp_attention.SpAttentionBlock.run(x_shard): SpComm all-gather ->
          DecodeProjection -> SpDsaCore / SpKdaCore -> SpOproj with its 4-rank
          reduce-scatter

on the same input rows, the same metadata and each on its own copy of the caches:

* ``test_sp_dsa_block_matches_stock_layer``: the gated attention rows entering o_proj
  and this rank's output shard (``_check_dsa_rows``: rows whose global top-k set is the
  same on both paths, every row whose length fits the 2048-key top-k among them,
  elementwise within the distance the two paths' query rounding reaches through
  trtllm-gen's E4M3 query and P, ``DSA_DIRECT``; every row, after taking out the FP64
  effect of near-tie top-k swaps (each path's FP64 reference over the keys it selected),
  within stock's own error against FP64; each path against FP64 attention over its
  selected cached keys, the two within 1.5x of each other either way,
  ``_check_dsa_reference``), decomposed: our FP8 query codes vs stock's (one step apart
  at most), our decode kernel call on stock's inputs vs stock's (where stock chunks its
  launches, both against FP64 within 1.5x of each other), the same flat KV rows,
  the kernel on stock's inputs but one factor of ours each (the FP8 query, the new MLA
  rows, the flat row order), and our merge + gate on stock's partials and valid counts
  (``DSA_CROSS`` reported, ``DSA_DIRECT`` held), the new MLA rows (FP8 codes one step
  apart where the RMSNorm or the in-projection rounds differently; the rope part
  exact against our own k_pe, ``_check_rope_part``), the new indexer keys
  (few codes different) and every other cache byte as it was; then the block replayed in
  a CUDA graph bit for bit as its eager run. Contexts per request cycle through short
  (all keys) and long ones, one of 128K tokens. The top-k runs the #63 merge or, as C512
  deploys it (its policy: the top-k exchange, one round to 256 rows), DcpExchange
  on both paths.
* ``test_sp_kda_block_matches_stock_layer`` (two steps) and
  ``test_sp_kda_recoverssm_multistep`` (``CHAIN_STEPS`` steps, the specialized block
  replayed in one CUDA graph): RecoverSSM over consecutive steps of the same
  requests with partial acceptance (random 1..4 accepted per request and step, the
  stock Triton commit after each step on each path): per step the records (keys,
  raw gates, corrections), the conv window, the o_norm-gated rows entering o_proj and
  the shard; after each commit the committed FP32 state of every request. The two
  paths' in-projections round differently (BF16 GEMM order), so everything is
  compared within BF16 rounding bounds; the committed state's error must not grow
  past its bound over the steps.
* every case checks the startup coverage of its blocks (each R a specialized
  choice) and that refused configurations raise ``shapes.Unsupported`` with the
  cause: 192 gathered DSA heads, a DCP2 context on the DCP4 box, a BF16 KDA state,
  a non-SP o-projection, a row past the mailboxes.

Rows: ``*_ROWS`` (a subset spanning every policy seam) per test; the ``slow_test``
variants sweep all 128 captured R. Tolerances are BF16 / FP32 reduction-order bounds
(the stock reduce-scatter sums BF16 partials over NCCL; ours in FP32, rounded once).
Needs 4 Rubin (SM107) GPUs with NVLink multicast; CuTe compiles dominate a cold
cache."""

import math
import os
import time
import zlib
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.network_utils import get_open_port

WORLD, Q, HEADS, WIDTH = 4, 4, 24, 7168
CAP = 512  # max_num_seqs 128 x q4
ALL_ROWS = tuple(range(4, CAP + 1, 4))
# DSA: front / fork query overlap (16 | 20), the 64-row seam, DcpExchange's one /
# two rounds (256 | 260), the cap.
DSA_ROWS = (4, 16, 20, 64, 68, 128, 256, 260, 512)
# KDA record cores by requests n (cutedsl.kda.sp.RECORD_KINDS at 24 heads): KdaSplit
# n1, KdaWide n8 / n16 / n35, KdaWS n81 / n128.
KDA_ROWS = (4, 32, 64, 128, 140, 324, 512)
CHAIN_ROWS, CHAIN_STEPS = (4, 64, 140, 512), 8
TOPK, UNIT, EPS = 2048, 64, 1e-6
FP8_ROW, INDEX_ROWS = 576, {"fp8": 132, "mxfp4": 68}
MAX_LEN = 1 << 17
LONG = MAX_LEN - 64  # the u128k request
SHORT = (600, 1500, 2050, 4321, 3000, 9000)
# C512's decode conv layout (its DSA top-k exchange needs no environment).
DEPLOYED_ENV = {"VLLM_SSM_CONV_STATE_LAYOUT": "DS"}
TIMEOUT = 3600  # one launch, every rank (CuTe compiles on a cold cache)


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    capabilities = {torch.cuda.get_device_capability(i) for i in range(WORLD)}
    return capabilities == {(10, 7)} and _has_multicast_support()


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        not _rubin_node(),
        reason="needs 4 Rubin (SM107) GPUs with NVLink multicast (C512's TP4 group)",
    ),
]


def _generator(*key, device="cuda"):
    """The same stream on every rank for ``key`` (inputs shared by the TP group)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(repr(key).encode()))
    return gen


def _rows_error(got, want):
    """Per-row relative L2 error of two [rows, width] tensors."""
    got, want = got.double(), want.double()
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-30)


def _msg(prefix):
    """An ``assert_close`` message naming the case."""
    return lambda message: f"{prefix}: {message}"


def _keep_input(kept):
    """A forward pre-hook keeping a copy of the module's input (the rows entering
    the stock o_proj)."""

    def hook(module, args):
        kept.append(args[0].clone())

    return hook


def _check_rows(name, got, want, median=1e-2, tail=3e-2, worst=0.15):
    """The median and 98th percentile of the per-row error, and no row far off."""
    errors = _rows_error(got, want)
    quantiles = torch.quantile(errors.float(), torch.tensor([0.5, 0.98]).cuda())
    print(
        f"{name}: per-row error median {quantiles[0]:.2e} p98 {quantiles[1]:.2e} max "
        f"{errors.max():.2e}"
    )
    assert quantiles[0] <= median and quantiles[1] <= tail, (name, quantiles)
    assert float(errors.max()) <= worst, (name, errors.max())


# The DSA A/B, ours against stock. The two paths' queries round differently (q_c
# RMSNorm, Q_B and W_UK sums in another FP32 order: BF16 flips), so a few of their
# E4M3 query codes differ by one step (``_check_q_codes``), and trtllm-gen's E4M3 P
# turns those into P code flips: rows over the same keys differ by ~0.5-2% (C512
# SP layer test at 0da2d21, R4, 4 exact rows per rank: per-row median 0.50-1.00e-2,
# p98 1.85e-2, max 1.87e-2, elementwise within 2e-2; the shard's single exact row
# 1.21-1.25e-2; at 45a6a74 R8 / R16 the same exact rows reach 2.09e-2 and 2.23e-2,
# stock's own error against FP64 being 3.3-3.6e-2 there). The indexer queries round
# differently too, so where a row's global top-k is a subset (contexts past 2048
# keys) both paths take an exact top-k of logits that differ by rounding and may
# swap near-tie keys: such a row moves by the swapped keys' attention weight (one of
# 2048 near-uniform keys: ~1/sqrt(2048) = 2.2% relative per key), not by rounding.
# So (``_check_dsa_rows``):
# * rows whose global top-k set is the same on both paths (every exact row): median
#   1.6e-2 (over ``DSA_MEDIAN_ROWS`` or more), p98 3e-2 and elementwise 2e-2
#   (``DSA_DIRECT``, ``DSA_SAME``);
# * every row, the paths' distance less the FP64 effect of their selections (each
#   path's FP64 reference over the keys it selected, ``_dsa_selected_reference``)
#   within stock's own per-row error against FP64 over its keys (``DSA_BAND``: the
#   paths disagree by no more than stock deviates from exact math; measured 0.15-
#   0.65 of it on exact rows), each path's aggregate error within 1.5x of the other's
#   (``_check_dsa_reference``), the same global top-k length (min(length, 2048)) on
#   both and at most ``DSA_SWAP`` of the keys a row can swap (min(2048, length -
#   2048): the keys outside its top-k bound the swaps) swapped (a near-tie swap moves
#   a few keys; a lost chunk, round or rank share, or a top-k that ignores the
#   logits, moves most of them; a 2125-token row has 77 keys to swap, so a fixed
#   share of the 2048 selected would pass any selection there);
# * no same-set row past 0.15 (a wrong head, scale, LSE base, W_UV or gate moves
#   rows by 10-100%; on differing rows the residual bound above catches it).
DSA_DIRECT = dict(median=1.6e-2, tail=3e-2, worst=0.15)
# The same-set median follows the paths' query distance (``_check_q_codes``: the
# dequantized relative error of our FP8 query codes against stock's), which moves
# with the R-dependent flip share: at 6e67e34, R256-R512 (0.04% codes flipped, query
# 0.8e-3, the core on our query median 0 per (row, head)) 4.96-5.08e-3: the floor of
# the other rounding points (P codes, merge, W_UV, gate); R20-R64 (1.0-1.25%, query
# 3.9-4.4e-3) and R68 / R128 (1.83%, 5.29-5.30e-3; the core on our query alone
# 1.27-1.35e-2 per (row, head) there against 0.23-0.79e-2 at R20-R64) up to 1.74e-2.
# In quadrature over the floor that is at most 3.2 x the query distance (every rank
# and arm); R68's rows keep their residual within 0.62-0.69 of stock's own FP64
# error (our error against FP64 0.998 of stock's). So the median is held to the
# larger of ``DSA_DIRECT``'s and hypot(``DSA_FLOOR``, ``DSA_Q_GAIN`` x query
# distance): the fixed 1.6e-2 below query 3.8e-3, 2.2e-2 at R68's 5.29e-3.
DSA_FLOOR, DSA_Q_GAIN = 5e-3, 4.0
DSA_SAME = dict(atol=2e-2, rtol=2e-2)
DSA_BAND, DSA_SWAP = 1.0, 0.1
# Fewest same-set rows whose median is held to ``DSA_DIRECT``'s (R24 rank 0's shard:
# 5 same-set rows at 1.61-1.75e-2, residual 0.51 of stock's error).
DSA_MEDIAN_ROWS = 16
# Our merge + gate on stock's partials and counts against stock: only BF16 rounding
# points differ. Stock rounds each corrected partial, its NCCL reduce-scatter sums
# BF16 partials (2-3 roundings), then W_UV and the gate round; ours ("proj") rounds
# the W_UV value rows, merges in FP32 and rounds once: ~5-6 BF16 roundings (2^-9
# relative, RMS 1.1e-3 each) apart, ~3e-3 per row (a CPU emulation of the merge
# alone: 0.39%). Derived, not measured: reported against these (``_report_rows``)
# until a GPU run calibrates them; the arm is held to ``DSA_DIRECT`` (on stock's
# own partials it can only be closer to stock than the whole path).
DSA_CROSS = dict(median=6e-3, tail=1.2e-2, worst=1.5e-2)


def _e4m3_steps(a, b):
    """Per-element distance in E4M3 rounding steps of two FP8 byte tensors: codes
    in value order with zero (+0 = -0) a half step from each smallest subnormal, so
    a value that rounds to -min on one path and +min on the other (a sign flip at
    zero), or to 0 on one and +-min on the other, is one step apart."""

    def twice_ordinal(codes):
        codes = codes.view(torch.uint8).int()
        magnitude = codes & 0x7F
        half = torch.where(magnitude > 0, 2 * magnitude - 1, 0)
        return torch.where(codes >= 0x80, -half, half)

    return ((twice_ordinal(a) - twice_ordinal(b)).abs() + 1) // 2


def _check_q_codes(name, ours, stock, scales):
    """Our gathered FP8 query against the stock decode's ([R, DCP x H, 576] E4M3
    bytes each). ``ours`` is what our sparse decode read (``_TapQuery``), not the
    mailbox after the step: DcpValue re-arms the q mailbox with Q_SENT (0xFF bytes,
    dsa/dcp_value.py ``rearm``) once the decode has read it. The layouts are the
    same: token-major rows; head block d = DCP rank d's heads (ours: MlaQuery's
    publish row ``tok * q_heads + rank * heads + head``, mla/query.py; stock: the
    DCP query gather, an all_gather over dim 1, v1/attention/ops/dcp.py); per head
    512 NoPE (q_nope W_UK) then 64 RoPE; each side divided by the layer's per-tensor
    ``_q_scale`` (``scales``: (ours, stock) dequant scales; trtllm-gen's bmm1_scale
    multiplies it back on both). So BF16 rows that differ only where the two sums
    round differently: at most one code step apart and few codes different (as the
    new MLA rows, ``fp8_codes_adjacent``). On a mismatch the head / row probe
    names the stock (row, head) each of a few of ours is closest to."""
    ours, stock = ours.view(torch.uint8), stock.view(torch.uint8)
    assert ours.shape == stock.shape, (name, ours.shape, stock.shape)
    sentinel = int((ours.reshape(-1, 16) == 0xFF).all(1).sum())
    ours_scale, stock_scale = (float(s) for s in scales)

    def dequant(codes, scale):
        return codes.view(torch.float8_e4m3fn).float() * scale

    a, b = dequant(ours, ours_scale), dequant(stock, stock_scale)
    finite = torch.isfinite(a)
    a = torch.where(finite, a, 0.0)

    def relative(x, y):
        return float((x - y).norm() / y.norm().clamp_min(1e-30))

    steps = _e4m3_steps(ours, stock)
    differ = int((steps > 0).sum())
    rows = int((steps > 0).flatten(1).any(1).sum())
    print(
        f"{name} FP8 query: {differ} of {steps.numel()} codes differ "
        f"({differ / steps.numel():.2e}), in {rows} of {steps.shape[0]} rows, max "
        f"{int(steps.max())} step(s); dequantized relative error {relative(a, b):.2e} "
        f"(NoPE {relative(a[..., :512], b[..., :512]):.2e}, RoPE "
        f"{relative(a[..., 512:], b[..., 512:]):.2e}); scales ours {ours_scale:.6g} "
        f"stock {stock_scale:.6g}; ours non-finite {int((~finite).sum())}, Q_SENT "
        f"chunks {sentinel}"
    )
    if int(steps.max()) > 1:
        # Which stock (row, head) is each of a few of ours closest to: an offset
        # head (a rank-block or head-major order), another row, or none (garbage).
        r, h = ours.shape[:2]
        flat_a, flat_b = a.flatten(0, 1), b.flatten(0, 1)
        picks = sorted(
            {(t, g) for t in (0, r - 1) for g in (0, 1, HEADS - 1, HEADS, h - 1)}
        )
        index = torch.tensor([t * h + g for t, g in picks], device=a.device)
        distance = torch.cdist(flat_a[index], flat_b) / flat_b.norm(dim=1).clamp_min(
            1e-30
        )
        best = distance.argmin(1).tolist()
        probe = ", ".join(
            f"({t},{g})->({j // h},{j % h}) {float(distance[i, j]):.2e}"
            for i, ((t, g), j) in enumerate(zip(picks, best))
        )
        # The NoPE / RoPE halves swapped, or the scale off by a constant factor.
        swapped = relative(a[..., :64], b[..., 512:])
        ratio = float((a * b).sum() / (b * b).sum().clamp_min(1e-30))
        print(
            f"{name} FP8 query probe (ours (row,head) -> best stock (row,head) rel): "
            f"{probe}; ours RoPE-first vs stock RoPE {swapped:.2e}; least-squares "
            f"ours/stock factor {ratio:.4g}"
        )
    assert sentinel == 0, (name, "the q mailbox read after its re-arm", sentinel)
    # One BF16 rounding apart (summation order) moves a normal E4M3 code by at most one
    # step; more only where the W_UK dot product cancels to a small value: the inputs'
    # BF16 flips move it by an absolute amount on the scale of its terms, several
    # E4M3 steps of a small result. So codes more than one step apart must be small
    # and rare, each at most one BF16 ulp of the largest |q| apart (the rounding
    # scale of the terms; a step count is no bound there: at 6e67e34 R72 had 92 of
    # 3981312 codes past one step, max 3 steps at |q| <= 0.0625 of a 5.5 row max, so
    # at most 3 x 2^-8 = 1.2e-2 apart, under the ulp 2^-5 = 3.1e-2; the former
    # two-step cap allowed 2 x 2^-4 = 0.125 at the 1/8 row max), and the dequantized
    # rows within BF16 rounding (below).
    far = steps > 1
    if bool(far.any()):
        mags = torch.stack([a[far].abs(), b[far].abs()]).max(0).values
        apart = float((a[far] - b[far]).abs().max())
        top = float(b.abs().max())
        ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
        print(
            f"{name} FP8 query: {int(far.sum())} codes > 1 step apart (max "
            f"{int(steps.max())}), max |q| there {float(mags.max()):.3g}, max apart "
            f"{apart:.3g} (row max |q| {top:.3g}, its BF16 ulp {ulp:.3g})"
        )
        assert apart <= ulp, (name, "far codes apart", apart, ulp)
        assert int(far.sum()) <= 1e-3 * steps.numel(), (name, int(far.sum()))
        biggest = float(mags.max())
        assert biggest <= 0.125 * top, (name, biggest)
    assert differ <= 0.05 * steps.numel(), (name, differ)
    # One-step flips spread over codes of every magnitude: the dequantized distance
    # is sqrt(flipped share) x the RMS step relative to the RMS code. A step is 2^-4
    # to 2^-3 of a normal code; measured 0.038-0.040 x sqrt(share) at R4-R68 (share
    # 0.34-1.83%, growing with the rows: relative 2.25e-3 to 5.29e-3), so a fixed
    # bound fails larger batches; flips concentrated on the largest codes, or two-
    # step ones, approach 2^-3.
    share = differ / steps.numel()
    assert relative(a, b) <= 2**-4 * share**0.5, (name, relative(a, b), share)
    return relative(a, b)


class _TapQuery:
    """Records the FP8 query our sparse decode reads (``fi``: FlashInferCore; its
    first ``decode`` call of the block run): PreparedDsa.dcp_core passes the q
    mailbox rows ``box.q[parity, :batch]`` after the query's join (every rank's
    heads published), the latest point they hold the query: the DcpValue kernel
    after the decode re-arms them with Q_SENT. ``heads``: the real heads (the
    mailbox pads to trtllm-gen's query heads). ``rows``: the same call's flat KV
    rows and counts (this rank's share of our global top-k)."""

    def __init__(self, fi, heads):
        self.fi, self.heads, self.q, self.rows = fi, heads, None, None

    def __enter__(self):
        original = self.fi.decode

        def decode(query, cache, slots, valid, block_size):
            if self.q is None:
                self.q = query.view(torch.uint8)[:, : self.heads].clone()
                self.rows = slots.clone(), valid.clone()
            return original(query, cache, slots, valid, block_size)

        self.fi.decode = decode
        return self

    def __exit__(self, *exc):
        del self.fi.decode


def _stock_core_launches(impl, rows, heads):
    """trtllm-gen launches the stock sparse core split ``rows`` query rows of ``heads``
    heads into (FlashInferMLASparseImpl._forward_mqa_kernel with LSE: row chunks of
    the workspace's softmax-stats slab capacity)."""
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fms

    capacity = fms.trtllm_gen_lse_rows_capacity(
        impl._workspace_buffer.numel(), fms.trtllm_gen_query_heads(heads)
    )
    assert capacity > 0, (rows, heads, capacity)
    return -(-rows // min(capacity, fms._TRTLLM_GEN_MAX_REQUESTS))


def _core_reference(impl, stock, cache, keyed):
    """FP64 sparse MLA of stock's core inputs (``stock``: the _StockCore record, its
    FP8 query and this rank's flat KV rows and counts; ``cache``: the MLA cache it
    read) on the ``keyed`` rows: latent [rows, H, 512] with stock's bmm1 / bmm2
    scales on the FP8 codes (the kernel's inputs, so only its arithmetic differs)."""
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    kv_rows = flat_kv_row_view(cache.view(torch.float8_e4m3fn), UNIT)[0]
    query = stock["q"].view(torch.float8_e4m3fn)
    rows = keyed.nonzero().flatten().tolist()
    out = torch.empty(
        len(rows), query.shape[1], 512, dtype=torch.float64, device=query.device
    )
    for i, row in enumerate(rows):
        n = int(stock["counts"][row])
        keys = kv_rows[stock["rows"][row, :n].long()].double()
        probs = torch.softmax(query[row].double() @ keys.T * impl.bmm1_scale, -1)
        out[i] = probs @ keys[:, :512] * impl.bmm2_scale
    return out


def _check_core_replay(name, replay, record, impl, cache):
    """Our FlashInferCore.decode on stock's FP8 query, KV rows, flat rows and counts
    against stock's own trtllm-gen call (``replay``: latent [R, H, 512], LSE [R, H]
    over the real heads; ``record``: the _StockCore record, stock's inputs and its
    latent and LSE; ``impl``: the stock layer's FlashInferMLASparseImpl; ``cache``:
    stock's MLA cache) on the rows with keys on this rank (stock zeroes the others):
    the same kernel on the same inputs.

    Stock launches row chunks of its workspace's softmax-stats slab capacity
    (``_stock_core_launches``); ours runs all R rows in one launch. trtllm-gen picks
    its split-KV per launch from the launch's rows, so a short tail chunk (R208 =
    191 + 17) reduces the same keys in another order: latent per (row, head) up to
    ~3e-2 apart from stock (each ~2e-2 from FP64; hti/p2/sparse-mla-r208), while
    stock's own single launch over all R is bitwise ours. So: one stock launch ->
    within 1e-3 of stock (the same call); always, both against FP64 attention over
    the same FP8 codes (``_core_reference``) within 1.5x of each other either way;
    the LSE within 1e-3 of stock's (chunking moves it ~1e-6)."""
    keyed = record["counts"] > 0
    latent, lse = replay[0][keyed], replay[1][keyed]
    want, want_lse = record["latent"][keyed], record["lse"][keyed]
    live = torch.isfinite(want_lse)
    bitwise = torch.equal(latent, want) and torch.equal(lse[live], want_lse[live])
    errors = _rows_error(latent.flatten(0, 1), want.flatten(0, 1))
    shift = float((lse[live] - want_lse[live]).abs().max()) if live.any() else 0.0
    rows, heads = record["q"].shape[:2]
    launches = _stock_core_launches(impl, rows, heads)
    ref = _core_reference(impl, record, cache, keyed)[:, : want.shape[1]]
    total = lambda x: float((x.double() - ref).norm() / ref.norm())  # noqa: E731
    ours_total, stock_total = total(latent), total(want)
    print(
        f"{name} core replay (stock's query, rows, counts; stock {launches} "
        f"launch(es)): bitwise {bitwise}, latent per (row, head) max "
        f"{errors.max():.2e}, LSE max {shift:.2e}; vs FP64 ours {ours_total:.2e} "
        f"stock {stock_total:.2e} (ratio {ours_total / max(stock_total, 1e-30):.3f})"
    )
    assert shift <= 1e-3, (name, shift)
    if launches == 1:
        assert float(errors.max()) <= 1e-3, (name, errors.max())
    assert ours_total <= 1.5 * stock_total + 1e-6, (name, ours_total, stock_total)
    assert ours_total >= stock_total / 1.5 - 1e-6, (name, ours_total, stock_total)


def _slot_sets(ours, stock, batch):
    """This rank's flat KV rows per row, ours vs stock's (``ours``, ``stock``:
    (rows, counts), a compacted prefix of ``counts`` per row) over the first
    ``batch`` rows -> (the same set, the same order, the rows both list, our count,
    stock's count) per row (int64; every key is on one rank: the counts sum to the
    global ones over the DCP group)."""
    (rows, counts), (want, want_counts) = ours, stock
    assert min(counts.shape[0], want_counts.shape[0]) >= batch, (counts.shape, batch)
    width = max(rows.shape[1], want.shape[1])
    pad = torch.iinfo(torch.int64).max

    def prefix(flat, valid):
        flat, valid = flat[:batch].long(), valid[:batch].long()
        keep = torch.arange(flat.shape[1], device=flat.device) < valid.view(-1, 1)
        flat = torch.where(keep, flat, pad)
        return torch.nn.functional.pad(flat, (0, width - flat.shape[1]), value=pad)

    a, b = prefix(rows, counts), prefix(want, want_counts)
    same_order = (a == b).all(1)
    a, b = a.sort(1)[0], b.sort(1)[0]
    same_set = (a == b).all(1)
    at = torch.searchsorted(b, a).clamp_max(width - 1)
    common = ((b.gather(1, at) == a) & (a != pad)).sum(1)
    return (
        same_set,
        same_order,
        common,
        counts[:batch].long(),
        want_counts[:batch].long(),
    )


def _global_sets(local, group):
    """The DCP group's view of ``_slot_sets`` (collective): per row whether every
    rank has the same set, the global overlap (keys both selected over the larger
    count) and the global counts (ours, stock's)."""
    import torch.distributed as dist

    same_set, _, common, counts, want_counts = local
    tally = torch.stack([(~same_set).long(), common, counts, want_counts])
    dist.all_reduce(tally, group=group)
    differ, common, counts, want_counts = tally
    overlap = common.double() / torch.maximum(counts, want_counts).clamp_min(1)
    return differ == 0, overlap, counts, want_counts


def _check_slots(name, local, exact):
    """This rank's flat KV rows (``local``: ``_slot_sets``): over every key
    (``exact``) the same set; reports how many rows list them in the same order
    (trtllm-gen's E4M3 P is quantized against the running row max, so the order is
    a numerical input) and the other rows' overlap. -> the rows with the same set."""
    same_set, same_order, common, counts, want_counts = local
    batch = same_set.shape[0]
    overlap = common.double() / torch.maximum(counts, want_counts).clamp_min(1)
    other = ~same_set
    print(
        f"{name} flat rows: {int(same_set.sum())} of {batch} rows the same set, "
        f"{int(same_order.sum())} in the same order; other rows' overlap min "
        f"{float(overlap[other].min()) if other.any() else 1.0:.3f}"
    )
    wrong = exact[:batch] & ~same_set
    assert not bool(wrong.any()), (name, "exact rows", wrong.nonzero().flatten())
    return same_set


def _check_selection(name, same, overlap, counts, lengths):
    """The global top-k of every row on both paths (``same``, ``overlap``,
    ``counts``: ``_global_sets``): min(length, TOPK) keys each, the same set where
    the length fits the top-k, elsewhere at most ``DSA_SWAP`` of the keys the row
    can swap (min(TOPK, length - TOPK)) swapped (exact top-k of logits that round
    differently: near-tie swaps only)."""
    want = lengths.clamp_max(TOPK).long()
    room = (lengths.long() - want).clamp_max(TOPK)
    ours, stock = counts
    differ = ~same
    swapped = ((1 - overlap) * torch.maximum(ours, stock).double()).round().long()
    print(
        f"{name} global top-k: {int(same.sum())} of {same.numel()} rows the same set "
        f"({int((lengths <= TOPK).sum())} over every key); differing rows "
        f"{differ.nonzero().flatten().tolist()}, overlap min "
        f"{float(overlap[differ].min()) if differ.any() else 1.0:.4f}, keys swapped "
        f"{swapped[differ].tolist()} of {room[differ].tolist()}"
    )
    assert torch.equal(ours, want) and torch.equal(stock, want), (
        name, "top-k length", ours.tolist(), stock.tolist(), want.tolist(),
    )  # fmt: skip
    assert bool(same[lengths <= TOPK].all()), (name, "rows over every key")
    beyond = swapped > DSA_SWAP * room
    assert not bool(beyond.any()), (
        name, "keys swapped", beyond.nonzero().flatten().tolist(),
        swapped[beyond].tolist(), room[beyond].tolist(),
    )  # fmt: skip


def _check_dsa_rows(name, got, want, refs, same, query):
    """Ours vs stock per row (``got``, ``want``) with each path's FP64 reference over
    the keys it selected (``refs``: ours', stock's; ``_dsa_selected_reference``)
    and whether the row's global top-k set is the same on both (``same``):
    ``DSA_DIRECT`` (median, p98; the median bound raised to hypot(``DSA_FLOOR``,
    ``DSA_Q_GAIN`` x ``query``) where larger, ``query`` the FP8 query's dequantized
    distance from ``_check_q_codes``) and ``DSA_SAME`` (elementwise) on the same-set
    rows; on every row the distance less the selections' FP64 effect within
    ``DSA_BAND`` of stock's own error, and no same-set row past ``DSA_DIRECT``'s
    worst.
    Printed: the rows past the p98 bound by kind, and each differing row's
    distance, swap effect, residual and stock's error."""
    ours_ref, stock_ref = refs
    got, want = got.double(), want.double()
    norm = want.norm(dim=-1).clamp_min(1e-30)
    errors = (got - want).norm(dim=-1) / norm
    swap = (ours_ref - stock_ref).norm(dim=-1) / norm
    residual = ((got - want) - (ours_ref - stock_ref)).norm(dim=-1) / norm
    band = _rows_error(want, stock_ref)
    tail = DSA_DIRECT["tail"]
    differ = ~same
    print(
        f"{name}: {int(same.sum())} of {same.numel()} rows the same top-k set; past "
        f"{tail:.0e}: {int((errors[same] > tail).sum())} same-set rows, "
        f"{int((errors[differ] > tail).sum())} of {int(differ.sum())} differing rows; "
        f"residual / stock's FP64 error max {float((residual / band).max()):.3f}"
    )
    if differ.any():
        worst = differ.nonzero().flatten()
        worst = worst[errors[worst].argsort(descending=True)][:8].tolist()
        print(
            f"{name} differing rows (row: distance, FP64 swap effect, residual, stock "
            "vs FP64): "
            + ", ".join(
                f"{t}: {errors[t]:.2e} {swap[t]:.2e} {residual[t]:.2e} {band[t]:.2e}"
                for t in worst
            )
        )
    if same.any():
        torch.testing.assert_close(
            got[same].float(), want[same].float(), **DSA_SAME,
            msg=_msg(f"{name} (rows over the same keys)"),
        )  # fmt: skip
        levels = torch.tensor([0.5, 0.98], dtype=torch.float64, device=errors.device)
        quantiles = torch.quantile(errors[same], levels)
        print(
            f"{name} same-set rows: per-row error median {quantiles[0]:.2e} p98 "
            f"{quantiles[1]:.2e} max {errors[same].max():.2e}"
        )
        # The median of a handful of rows (a rank's shard: R / DCP rows) is one draw
        # of the per-row spread (exact rows measured 0.5-2.23e-2): there every row
        # is held to the tail bound instead.
        worst = float(errors[same].max())
        median = max(DSA_DIRECT["median"], math.hypot(DSA_FLOOR, DSA_Q_GAIN * query))
        if int(same.sum()) >= DSA_MEDIAN_ROWS:
            assert quantiles[0] <= median, (name, quantiles, median)
        else:
            assert worst <= DSA_DIRECT["tail"], (name, "few same-set rows", worst)
        assert quantiles[1] <= DSA_DIRECT["tail"], (name, quantiles)
        assert worst <= DSA_DIRECT["worst"], (name, "same-set row", worst)
    # A differing row's distance is its swapped keys' FP64 effect (R24 rank 2 row 6,
    # 10 keys swapped: distance 0.156, FP64 swap effect 0.158, residual 1.73e-2
    # against stock's 3.22e-2), so it is held by the residual: a wrong head, scale,
    # LSE base, W_UV or gate leaves a 10-100% residual.
    beyond = residual > DSA_BAND * band
    assert not bool(beyond.any()), (
        name, "residual past stock's error", beyond.nonzero().flatten().tolist(),
        residual[beyond].tolist(), band[beyond].tolist(),
    )  # fmt: skip


def _report_rows(name, got, want, median, tail, worst):
    """Prints the per-row error distribution against bounds not yet calibrated on
    GPU (no assert)."""
    errors = _rows_error(got, want)
    quantiles = torch.quantile(errors.float(), torch.tensor([0.5, 0.98]).cuda())
    within = (
        float(quantiles[0]) <= median
        and float(quantiles[1]) <= tail
        and float(errors.max()) <= worst
    )
    print(
        f"{name}: per-row error median {quantiles[0]:.2e} p98 {quantiles[1]:.2e} max "
        f"{errors.max():.2e} ({'within' if within else 'ABOVE'} the derived "
        f"{median:.0e}/{tail:.1e}/{worst:.1e})"
    )


def _report_core(name, factor, got, stock, rows):
    """One decode kernel call (``got``: latent [R, H, 512], LSE [R, H]) against
    stock's partials on ``rows``, per (row, head) (printed: the factor's share of
    the direct distance)."""
    if not bool(rows.any()):
        print(f"{name} core, {factor}: no rows")
        return
    errors = _rows_error(got[0][rows].flatten(0, 1), stock[0][rows].flatten(0, 1))
    lse, want = got[1][rows], stock[1][rows]
    live = torch.isfinite(want)
    shift = float((lse[live] - want[live]).abs().max()) if live.any() else 0.0
    print(
        f"{name} core, {factor}: latent per (row, head) median {errors.median():.2e} "
        f"max {errors.max():.2e}, LSE max {shift:.2e} ({int(rows.sum())} rows)"
    )


def _check_dsa_reference(name, got, want, refs):
    """Ours and stock each against FP64 attention over the keys that path selected
    as it cached them (every key of the request on the rows whose length fits the
    top-k), from the unquantized query of the same input rows (``refs``: ours',
    stock's; ``_dsa_selected_reference``).
    Held: the aggregate errors within 1.5x of each other both ways (a wrong head,
    scale, LSE base or gate is a 10-100% error; the two paths' rounding-only
    distance, <= 2e-2 per exact row, against stock's ~3% error moves the ratio by
    at most ~1 +- 0.6 under the triangle limit, ~1.15 if independent). Printed: the
    per-row ratios and the paths' distance against stock's error (uncalibrated)."""
    ours_ref, stock_ref = refs
    ours, stock = _rows_error(got, ours_ref), _rows_error(want, stock_ref)
    total = lambda x, ref: float((x.double() - ref).norm() / ref.norm())  # noqa: E731
    ours_total, stock_total = total(got, ours_ref), total(want, stock_ref)
    apart = total(got, want.double())
    print(
        f"{name} vs FP64 ({got.shape[0]} rows over all keys): ours {ours_total:.2e} "
        f"stock {stock_total:.2e} (ratio {ours_total / max(stock_total, 1e-30):.3f}), "
        f"ours vs stock {apart:.2e}; per row ours/stock min "
        f"{(ours / stock.clamp_min(1e-30)).min():.3f} max "
        f"{(ours / stock.clamp_min(1e-30)).max():.3f}, stock median "
        f"{stock.median():.2e} max {stock.max():.2e}"
    )
    print(f"{name} vs FP64: ours vs stock / stock's error {apart / stock_total:.3f}")
    assert ours_total <= 1.5 * stock_total + 1e-3, (name, ours_total, stock_total)
    assert ours_total >= stock_total / 1.5 - 1e-3, (name, ours_total, stock_total)


class _StockCore:
    """Records the stock decode's sparse core call (FlashInferMLASparseImpl.
    _forward_mqa_kernel, one per step) while the stock layer runs: its gathered FP8
    query [R, DCP x H, 576], this rank's flat KV rows and counts, the partial latent
    [R, DCP x H, 512] and LSE [R, DCP x H]."""

    def __init__(self, impl):
        self.impl, self.seen, self.calls, self.dtypes = impl, {}, 0, []

    def __enter__(self):
        original = self.impl._forward_mqa_kernel

        def spy(q, kv_cache, topk_indices, seq_lens, **kwargs):
            out, lse = original(q, kv_cache, topk_indices, seq_lens, **kwargs)
            assert lse is not None, "the stock DCP decode returns its LSE"
            query = torch.cat(q, dim=-1) if isinstance(q, tuple) else q
            self.calls += 1
            self.dtypes.append(query.dtype)
            self.seen.update(
                q=query.clone(), rows=topk_indices.clone(), counts=seq_lens.clone(),
                latent=out.clone(), lse=lse.clone(),
            )  # fmt: skip
            return out, lse

        self.impl._forward_mqa_kernel = spy
        return self

    def __exit__(self, *exc):
        del self.impl._forward_mqa_kernel


class _FeedCore:
    """Wraps our FlashInferCore.decode (``fi``) for one block run: the decode kernel
    on stock's inputs (``stock``: the _StockCore record; ``kv``: our MLA cache, the
    (page, entry) of the step's new rows and stock's codes of them) with one factor
    of ours at a time, then stock's partial latent, LSE and valid counts in place of
    ours for our merge and gate. ``seen``: "rows" (our flat rows and counts), and
    per call ((latent, LSE) over the real heads) "replay" (all stock's), "query"
    (our FP8 query), "kv" (our new MLA rows), "order" (our flat rows and counts)."""

    def __init__(self, fi, stock, kv):
        self.fi, self.stock, self.kv, self.seen = fi, stock, kv, {}

    def __enter__(self):
        original, stock, seen = self.fi.decode, self.stock, self.seen
        rows_cache, at, stock_kv = self.kv
        stock_q = stock["q"].view(torch.uint8)
        heads, flat = stock_q.shape[1], (stock["rows"], stock["counts"])

        def decode(query, cache, slots, valid, block_size):
            q = query.view(torch.uint8)[:, :heads]
            ours_q, ours_kv = q.clone(), rows_cache[at].clone()
            seen["rows"] = slots.clone(), valid.clone()
            calls = {
                "replay": (stock_q, stock_kv, flat),
                "query": (ours_q, stock_kv, flat),
                "kv": (stock_q, ours_kv, flat),
                "order": (stock_q, stock_kv, (slots, valid)),
            }
            for factor, (q_codes, kv_codes, (rows, counts)) in calls.items():
                q.copy_(q_codes)
                rows_cache[at] = kv_codes
                latent, lse = original(query, cache, rows, counts, block_size)
                seen[factor] = latent[:, :heads].clone(), lse[:, :heads].clone()
            latent[:, :heads].copy_(stock["latent"])
            lse[:, :heads].copy_(stock["lse"])
            valid[: stock["counts"].shape[0]].copy_(stock["counts"])
            return latent, lse

        self.fi.decode = decode
        return self

    def __exit__(self, *exc):
        del self.fi.decode


def _cached_keys(cache, table, request, length, scale, rank, group):
    """[length, 576] FP64 MLA rows of a request's tokens 0..length-1 as the DCP
    ranks cached them (interleave 1: token p on rank p % WORLD at local p // WORLD;
    per-tensor FP8 times ``scale``). Collective over ``group``."""
    import torch.distributed as dist

    local = torch.arange(rank, length, WORLD, device="cuda") // WORLD
    rows = torch.zeros(-(-length // WORLD), FP8_ROW, device="cuda")
    codes = cache[table[request, local // UNIT].long(), local % UNIT]
    rows[: local.numel()] = codes.view(torch.float8_e4m3fn).float() * scale
    parts = [torch.empty_like(rows) for _ in range(WORLD)]
    dist.all_gather(parts, rows, group=group)
    return torch.stack(parts, 1).reshape(-1, FP8_ROW)[:length].double()


def _reference_query(attn, x):
    """The unquantized FP64 query [rows, heads, 576] (NoPE through W_UK, then RoPE)
    and output gate [rows, heads x 128] of this rank's heads from the layer's packed
    in-projection of the input rows ``x``."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        pack_dsa_projection,
    )

    wrapper = attn.mla_attn
    layer = wrapper.mla_attn
    heads = layer.W_UK_T.shape[0]
    proj = x.double() @ pack_dsa_projection(attn).double().T
    q_a = proj[:, :1536]
    eps = wrapper.q_a_layernorm.variance_epsilon
    q_c = q_a * torch.rsqrt(q_a.pow(2).mean(-1, keepdim=True) + eps)
    q_c = q_c * wrapper.q_a_layernorm.weight.double()
    query = (q_c @ wrapper.q_b_proj.weight.double().T).view(-1, heads, 192)
    nope = torch.einsum("thd,hdl->thl", query[..., :128], layer.W_UK_T.double())
    query = torch.cat([nope, query[..., 128:]], -1)
    gate = torch.sigmoid(proj[:, 2112 : 2112 + heads * 128])
    return query, gate


def _gated_rows(attn, latent, gate):
    """FP64 latent [rows, heads, 512] -> the gated o_proj input rows."""
    layer = attn.mla_attn.mla_attn
    out = torch.einsum("thl,hlv->thv", latent, layer.W_UV.double())
    return out.reshape(latent.shape[0], -1) * gate


def _dsa_reference(attn, x, lengths, exact, caches, table, rank, n):
    """FP64 references of the rows over every key (``exact``), each path over its own
    MLA cache (``caches``: ours, stock): this rank's gated o_proj input rows and the
    exact rows of its output shard (o_proj of every rank's reference rows, summed).
    The query is unquantized from the layer's packed in-projection of ``x``.
    Collective over the TP group. (Cross-checks ``_dsa_selected_reference``.)"""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    group = get_tp_group().device_group
    layer = attn.mla_attn.mla_attn
    rows = exact.nonzero().flatten()
    query, gate = _reference_query(attn, x[rows])
    heads = query.shape[1]
    w_o = attn.o_proj.weight.double()
    requests = (rows // Q).tolist()
    spans = {req: int(lengths[req * Q + Q - 1]) for req in sorted(set(requests))}
    scale = layer._k_scale_float
    outs = []
    for cache in caches:
        keys = {
            req: _cached_keys(cache, table, req, span, scale, rank, group)
            for req, span in spans.items()
        }
        latent = query.new_empty(rows.numel(), heads, 512)
        for i, (row, req) in enumerate(zip(rows.tolist(), requests)):
            k = keys[req][: int(lengths[row])]
            probs = torch.softmax(query[i] @ k.T * layer.impl.scale, -1)
            latent[i] = probs @ k[:, :512]
        outs.append(_gated_rows(attn, latent, gate))
    shards = torch.stack([out @ w_o.T for out in outs])  # [2, E, 7168]
    dist.all_reduce(shards, group=group)
    mine = (rows >= rank * n) & (rows < (rank + 1) * n)
    return tuple(outs), (shards[0][mine], shards[1][mine])


def _dsa_selected_reference(attn, x, selections, caches, rank, n, chunk=16):
    """FP64 references of every row, each path over the keys its global top-k
    selected as it cached them (``selections``: this rank's flat KV rows and counts,
    ours and stock's, a compacted prefix per row; ``caches``: the MLA caches, ours
    and stock's): this rank's gated o_proj input rows and its output shard rows (as
    ``_dsa_reference``). Every rank gathers each row's selected key rows of every
    rank (``chunk`` rows at a time). Collective over the TP group."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    group = get_tp_group().device_group
    layer = attn.mla_attn.mla_attn
    rows = x.shape[0]
    query, gate = _reference_query(attn, x)
    heads = query.shape[1]
    w_o = attn.o_proj.weight.double()
    scale, softmax_scale = layer._k_scale_float, layer.impl.scale
    outs = []
    for (slots, counts), cache in zip(selections, caches):
        flat = flat_kv_row_view(cache.view(torch.float8_e4m3fn), UNIT)[0]
        latent = query.new_empty(rows, heads, 512)
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            here = counts[start:stop].long()
            width = here.max().view(1).clone()
            dist.all_reduce(width, op=dist.ReduceOp.MAX, group=group)
            width = max(int(width), 1)
            index = slots[start:stop].long()
            index = torch.nn.functional.pad(
                index, (0, max(0, width - index.shape[1])), value=-1
            )[:, :width]
            span = torch.arange(width, device=index.device)
            keep = span < here.view(-1, 1)
            keys = flat[torch.where(keep, index, 0)].float() * scale
            keys = torch.where(keep.unsqueeze(-1), keys, 0.0).contiguous()
            parts = [torch.empty_like(keys) for _ in range(WORLD)]
            dist.all_gather(parts, keys, group=group)
            every = [torch.empty_like(here) for _ in range(WORLD)]
            dist.all_gather(every, here.contiguous(), group=group)
            keys = torch.cat(parts, 1).double()  # [rows, WORLD x width, 576]
            valid = torch.cat([span < c.view(-1, 1) for c in every], 1)
            scores = torch.einsum("thd,tkd->thk", query[start:stop], keys)
            scores = scores * softmax_scale
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            probs = torch.softmax(scores, -1)
            latent[start:stop] = torch.einsum("thk,tkl->thl", probs, keys[..., :512])
        outs.append(_gated_rows(attn, latent, gate))
    shards = torch.stack([out @ w_o.T for out in outs])  # [2, R, 7168]
    dist.all_reduce(shards, group=group)
    mine = slice(rank * n, (rank + 1) * n)
    return tuple(outs), (shards[0][mine], shards[1][mine])


# --------------------------------------------------------------------------- launch


def _launch(case: str, *args) -> None:
    """``case`` on 4 spawned ranks (one per GPU); every rank must exit cleanly within
    one deadline (a hung mailbox fails the test, never CI)."""
    import torch.multiprocessing as mp

    port = get_open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, port, case, args))
        for rank in range(WORLD)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + TIMEOUT
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [p.pid for p in processes if p.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert not hung, f"{case}: ranks {hung} did not finish in {TIMEOUT} s (a hang?)"
    assert [p.exitcode for p in processes] == [0] * WORLD, case


def _worker(rank: int, port: int, case: str, args: tuple) -> None:
    """One TP4 = DCP4 rank: the vLLM groups (and their symmetric memory), the
    workspace and the DCP top-k merge's buffers, then ``case``."""
    os.environ.update(DEPLOYED_ENV)
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    current_platform.import_kernels()
    import vllm.kernels.vllm_c  # noqa: F401  (the vllm_c RMSNorm provider)

    config = _dsa_config(args[0]) if case == "dsa" else _kda_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=WORLD,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=WORLD, decode_context_model_parallel_size=WORLD
        )
        group = get_tp_group().device_group
        try:  # (older torch: symmetric memory is enabled per group)
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        init_workspace_manager(device)
        try:
            if case == "dsa":
                from vllm.v1.attention.ops.dcp_topk_merge import init_dcp_topk_merge

                assert init_dcp_topk_merge(config, TOPK)
                _dsa_case(config, rank, device, *args)
            else:
                _kda_case(config, rank, device, *args)
        finally:
            torch.accelerator.synchronize()
            reset_workspace_manager()
    cleanup_dist_env_and_memory()


def _context(rank, device, **overrides):
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context

    fields = dict(
        arch="rubin", tp=WORLD, rank=rank, dcp=WORLD,
        sms=torch.cuda.get_device_properties(device).multi_processor_count, q=Q,
        max_rows=CAP, max_context=MAX_LEN, bf16_state=False, dp=1, ep=WORLD,
        sequence_parallel=True, heads_per_rank=HEADS,
    )  # fmt: skip
    fields.update(overrides)
    return Context(**fields)


class _Step:
    """The parts of ``sp_attention.SpStep`` an ``SpAttentionBlock`` reads at build:
    the context, rows, mailboxes, DSA box / scratch, records mode and the shared
    scratch buffers (``SpStep.scratch``)."""

    def __init__(self, ctx, rows, device, comm, records="none", box=None, dsa=None):
        self.ctx, self.rows, self.device, self.comm = ctx, rows, device, comm
        self.q, self.records = ctx.q, records
        self.dsa_box, self.dsa_scratch, self.dsa_layers = box, dsa, 0
        self.buffers: dict = {}
        self.shard_cap = rows[-1] // ctx.tp

    def scratch(self, name, shape):
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
            sp_attention,
        )

        return sp_attention.SpStep.scratch(self, name, shape)


def _sp_block(layer_idx, attention, step):
    """The SP block of one stock layer (ordinal 0: parity 0), and the shared
    coverage check: every row of the step a specialized (not stock) choice."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_attention
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Path

    decoder = SimpleNamespace(layer_idx=layer_idx, self_attn=attention)
    block = sp_attention.SpAttentionBlock(decoder, step, 0)
    for r in step.rows:
        choices = {"ag": step.comm.choice(r), **block.choices(r)}
        assert set(choices) == {"ag", f"{block.kind}_in", block.kind, "oproj_rs"}
        stock = [
            f for f, c in choices.items() if c.path not in (Path.GENERIC, Path.TUNED)
        ]
        assert not stock, (r, stock)
    return block


def _barrier():
    """All ranks done with the previous kernels (a TP collective between two SP
    blocks of the same mailbox parity, as the model's final sp_all_gather is)."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    torch.accelerator.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)


# ------------------------------------------------------------------------------ DSA


def _dsa_config(indexer):
    """The DSA layer's vLLM config at TP4 = DCP4 (interleave 1), FLASHINFER_MLA_
    SPARSE trtllm-gen on per-tensor FP8 KV, the ``indexer`` key cache, q4 verify
    rows; the metadata builders read DeepSeek-V2-Lite's MLA sizes (kv_lora 512, rope
    64) as test_specialized_decode_dsa.py's layer harness does."""
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = create_vllm_config(
        model_name=dsa.LAYER_MODEL,
        max_model_len=MAX_LEN,
        block_size=UNIT,
        max_num_seqs=CAP // Q,
        max_num_batched_tokens=8192,
        hf_config_override={"index_topk": TOPK},
    )
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = WORLD
    parallel.decode_context_parallel_size = WORLD
    parallel.cp_kv_cache_interleave_size = 1
    config.cache_config.cache_dtype = "fp8"
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    config.attention_config.indexer_kv_dtype = indexer
    config.kernel_config.flashinfer_sparse_mla_kernel = "trtllm"
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=Q - 1,
        enable_adaptive_verification=False,
        parallel_drafting=False,
    )
    return config


def _dsa_attention(config, rank, topk_buffer):
    """A K3.1 DSA layer at C512's sizes (96 heads over TP4, hidden 7168, 32 index
    heads of 128), random weights (this rank's shards), MLA weights processed, the
    SP decoder's non-reducing o_proj."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text = KimiLinearConfig(
        hidden_size=WIDTH, num_attention_heads=WORLD * HEADS, q_lora_rank=1536,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        mla_use_nope=True, mla_use_output_gate=True, rms_norm_eps=EPS,
        dsa_config={
            "dsa_layers": [4], "index_topk": TOPK, "index_head_dim": 128,
            "index_num_heads": 32, "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
    )  # fmt: skip
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attn = KimiK3DSAAttention(
                text, hidden_size=WIDTH, num_heads=WORLD * HEADS, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, q_lora_rank=1536,
                kv_lora_rank=512, cache_config=config.cache_config,
                prefix="model.layers.3.self_attn", topk_indices_buffer=topk_buffer,
            )  # fmt: skip
    finally:
        torch.set_default_dtype(torch.float32)
    torch.manual_seed(5 + rank)
    with torch.no_grad():
        for name, param in attn.named_parameters():
            if name.endswith("bias"):
                param.copy_(torch.randn_like(param) / 10)
            elif "norm" in name:
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                param.copy_(torch.randn_like(param) * param.shape[-1] ** -0.5)
    attn.mla_attn.mla_attn.process_weights_after_loading(torch.bfloat16)
    attn.o_proj.reduce_results = False  # (KimiDecoderLayer under SP)
    return attn


def _page_table(lengths, blocks, gen):
    """Each request's pages over a fixed ``blocks``-page cache, in random order."""
    pages = [max(-(-n // UNIT), 1) for n in lengths]
    assert sum(pages) <= blocks, (sum(pages), blocks)
    order = torch.randperm(blocks, generator=gen, device="cuda").int()
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[start : start + n]
        start += n
    return table


def _contexts(reqs):
    """Tokens before the step per request: short ones (all keys fit the top-k),
    long ones and one of 128K (from two requests on)."""
    return [LONG if r == 1 else SHORT[r % len(SHORT)] + 37 * r for r in range(reqs)]


def _local(contexts, rank):
    return [(c + Q - rank + WORLD - 1) // WORLD for c in contexts]


def _pages(cache, row):
    """Each packed indexer page's bytes [blocks, UNIT x row]."""
    return torch.as_strided(
        cache, (cache.shape[0], cache.shape[1] * row), (cache.stride(0), 1)
    )


def _index_token_bytes(cache, slots, row, key_bytes):
    """(page, byte offset) of every byte the tokens at ``slots`` own in a packed
    indexer cache: their key codes, then their 4 scale bytes."""
    page, entry = (slots // UNIT).long(), (slots % UNIT).long()
    keys = key_bytes * entry.view(-1, 1) + torch.arange(key_bytes, device="cuda")
    scales = UNIT * key_bytes + 4 * entry.view(-1, 1) + torch.arange(4, device="cuda")
    return page, torch.cat([keys, scales], 1)


KPE = slice(2048, 2112)  # k_pe in the packed DSA projection (after q_a | kv_c)


def _check_rope_part(where, ours, theirs, ours_pe, k_scale):
    """The rope part (64 FP8 codes of the un-rotated k_pe: K3.1 DSA is NoPE) of this
    rank's new MLA rows. Ours inserts it exactly as stock's writer does
    (concat_and_cache_mla: BF16 / k_scale, E4M3 round-to-nearest) from our own
    in-projection row, so it must equal that bit for bit. k_pe is a raw projection
    output: our DecodeProjection (warp / cluster split-K, or cuBLAS) and stock's
    fused_qkv_a_proj (cuBLAS) sum K = 7168 in different FP32 orders, so a rare k_pe
    element rounds to the neighbouring BF16 value and, where that crosses an E4M3
    rounding boundary, lands one code step from stock's (C512 at 075259e: R16, rank 1
    only, in every variant; its kernel effect alone, ours rows on stock's inputs:
    LSE 3.8e-6, the size of one code step on one key). So against stock: each code
    within one BF16 step of our k_pe, and few of them different."""

    def e4m3(values):  # both writers convert RN with saturation (cvt satfinite)
        return values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)

    want = e4m3(ours_pe.float() / k_scale)
    assert torch.equal(ours, want), f"{where}: rope part vs our k_pe"
    bits = ours_pe.view(torch.int16)
    reach = torch.zeros_like(theirs, dtype=torch.bool)
    for step in (-1, 0, 1):  # (a magnitude step either way, same sign)
        reach |= e4m3((bits + step).view(torch.bfloat16).float() / k_scale) == theirs
    differ = int((ours != theirs).sum())
    if differ:
        print(f"{where} rope part: {differ} of {ours.numel()} codes one BF16 step off")
    assert bool(reach.all()), f"{where}: rope part beyond one BF16 step of our k_pe"
    assert differ <= 0.05 * ours.numel(), (where, "rope part", differ)


def _dsa_case(config, rank, device, indexer, topk_mode, rows, refusals):
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from vllm import _custom_ops as ops
    from vllm import ir
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.common.ops.sequence_parallel import (
        sp_all_gather,
        sp_reduce_scatter,
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import indexer_k_norm_quant_insert
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.sp_comm import (
        SpComm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sp import (
        sp_dsa_box,
        sp_dsa_scratch,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.worker.block_table import get_block_table_width

    bf = torch.bfloat16
    index_row = INDEX_ROWS[indexer]
    key_bytes = index_row - 4
    insert = mxfp4_k_norm_quant_insert if indexer == "mxfp4" else (
        indexer_k_norm_quant_insert
    )  # fmt: skip
    topk_buffer = torch.full((8192, TOPK), -1, dtype=torch.int32, device=device)
    attn = _dsa_attention(config, rank, topk_buffer)
    layer, idx = attn.mla_attn.mla_attn, attn.dsa_indexer
    if topk_mode == "exchange":  # C512's stock hook, attached as the model does
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.decode import (
            DcpxDecode,
        )

        dcpx = DcpxDecode(config, device, WORLD * HEADS, idx.topk_tokens)
        dcpx.attach(attn)
        dcpx.warm()
        dcpx.serve()

    # One cache per path, sized for the widest case (the kernels compile for it).
    widest = max(rows) // Q
    blocks = sum(max(-(-n // UNIT), 1) for n in _local(_contexts(widest), rank)) + 2
    caches = {
        path: (
            dsa.packed_cache(blocks, UNIT, FP8_ROW, UNIT),
            dsa.packed_cache(blocks, UNIT, index_row, dsa.GAP),
        )
        for path in ("stock", "ours")
    }
    spec = layer.get_kv_cache_spec(config)
    builder = layer.attn_backend.get_builder_cls()(
        spec, [layer.layer_name], config, device
    )
    index_spec = idx.k_cache.get_kv_cache_spec(config)
    index_builder = DeepseekV32IndexerMetadataBuilder(
        index_spec, [idx.k_cache.prefix], config, device,
        block_table_width=get_block_table_width(
            index_spec.max_num_blocks_per_req(config, MAX_LEN), UNIT
        ),
    )  # fmt: skip
    names = layer.layer_name, idx.k_cache.prefix

    def bind(path):
        layer.kv_cache, idx.k_cache.kv_cache = caches[path]

    # The SP block, built (collectively) once for every case's rows.
    ctx = _context(rank, device)
    group = get_tp_group().device_group
    comm = SpComm(group, device, WORLD, rank, width=WIDTH, rows_cap=CAP)
    box = sp_dsa_box(group, device, rank, heads=HEADS, cap=CAP, interleave=1)
    step = _Step(
        ctx, tuple(rows), device, comm, box=box, dsa=sp_dsa_scratch(device, CAP)
    )
    bind("ours")
    block = _sp_block(3, attn, step)
    block.core.prepare_cache(caches["ours"][0])

    for r in rows:
        reqs, n = r // Q, r // WORLD
        gen = _generator("dsa", r, rank)  # this rank's pages and old keys
        shared = _generator("dsa-step", r)  # the step's inputs (every rank)
        contexts = _contexts(reqs)
        local = _local(contexts, rank)
        mla_table = _page_table(local, blocks, gen)
        index_table = _page_table(local, blocks, gen)
        old = [range(c) for c in contexts]
        old_mla = dsa.owned_slots(mla_table, old, WORLD, rank)
        old_index = dsa.owned_slots(index_table, old, WORLD, rank)
        stock_mla, stock_index = caches["stock"]
        stock_mla.zero_()
        stock_index.zero_()
        count = old_mla.numel()
        unit = torch.ones(1, dtype=torch.float32, device=device)
        kv_c = torch.randn(count, 512, device=device, dtype=bf, generator=gen)
        k_pe = torch.randn(count, 64, device=device, dtype=bf, generator=gen)
        ops.concat_and_cache_mla(kv_c, k_pe, stock_mla, old_mla, "fp8", unit)
        norm = torch.ones(128, dtype=bf, device=device)
        chunk = 16384
        scratch = torch.empty(chunk, TOPK, dtype=torch.int32, device=device)
        for start in range(0, count, chunk):
            keys = torch.randn(
                min(chunk, count - start), 128, device=device, dtype=bf, generator=gen
            )
            rows_here = keys.shape[0]
            insert(
                keys, norm, torch.zeros_like(norm), EPS,
                old_index[start : start + rows_here], stock_index,
                scratch[:rows_here], rows_here,
            )  # fmt: skip
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)
        new = [range(c, c + Q) for c in contexts]
        new_mla = dsa.owned_slots(mla_table, new, WORLD, rank)
        new_index = dsa.owned_slots(index_table, new, WORLD, rank)
        md = builder.build(
            0, dsa.common_metadata(contexts, Q, mla_table, new_mla, WORLD, rank)
        )
        index_md = index_builder.build(
            0, dsa.common_metadata(contexts, Q, index_table, new_index, WORLD, rank)
        )
        forward = create_forward_context(
            dict(zip(names, (md, index_md))),
            config,
            slot_mapping=dict(zip(names, (new_mla, index_md.slot_mapping))),
        )
        x = torch.randn(r, WIDTH, device=device, dtype=bf, generator=shared)
        x_shard = x[rank * n : (rank + 1) * n].contiguous()
        positions = torch.tensor([p for pos in new for p in pos], device=device)
        lengths = torch.tensor(
            [c + j + 1 for c in contexts for j in range(Q)], device=device
        )
        exact = lengths <= TOPK

        # Stock: the SP decoder's seam around the layer forward, its sparse core's
        # inputs and partials recorded.
        entering: list = []
        hook = attn.o_proj.register_forward_pre_hook(_keep_input(entering))
        bind("stock")
        # (its RMSNorms as a worker without inductor runs them: vllm_c, one rounding)
        norms = ir.ops.rms_norm.set_priority(["vllm_c", "native"])
        with override_forward_context(forward), norms, _StockCore(layer.impl) as spy:
            gathered = sp_all_gather(x_shard)[:r]
            want = sp_reduce_scatter(attn(positions=positions, hidden_states=gathered))
        hook.remove()
        stock = spy.seen
        assert spy.calls == 1 and spy.dtypes == [torch.float8_e4m3fn], spy.dtypes
        assert stock["q"].shape == (r, WORLD * HEADS, FP8_ROW), stock["q"].shape

        # Ours: the SP block on the same rows, metadata and its own caches.
        bind("ours")
        prepared = block.core.prepared
        run = SimpleNamespace(slot_mapping={layer.layer_name: new_mla}, q=Q)
        tap = _TapQuery(prepared.fi, WORLD * HEADS)
        with override_forward_context(forward), tap:
            got = block.run(x_shard, [caches["ours"][0]], md, run).clone()
        torch.accelerator.synchronize()
        attn_in = block.attn_out[:r].clone()
        # (our in-projection's k_pe of this rank's new rows: the rope part's source)
        ours_pe = block.projected[:r][new_mla >= 0, KPE].clone()
        assert tap.q is not None and tap.q.shape[0] == r, "our sparse decode's query"
        ours_q = tap.q
        # (dequant scales: ours multiplies by 1 / _q_scale, stock divides by it)
        q_scales = 1 / prepared.query_stage.scales[0], layer._q_scale
        where = f"DSA R{r} ({indexer}, {topk_mode}) rank {rank}"

        # The step's new MLA rows of this rank as each path wrote them.
        (ours_mla, ours_index), (stock_mla, stock_index) = (
            caches["ours"],
            caches["stock"],
        )
        written = new_mla[new_mla >= 0]
        at = written // UNIT, written % UNIT
        a, b = ours_mla[at].clone(), stock_mla[at].clone()

        # Every collective before any check (a failing rank must not strand the
        # others): the global top-k sets of the two paths, the FP64 references
        # (every rank has the same rows: the exact rows, request 0's 600 tokens at
        # least, over every key; every row over each path's selected keys), then one
        # block run whose decode kernel runs on stock's inputs with one factor of
        # ours at a time (on this rank alone) and whose merge and gate take stock's
        # partials and counts.
        assert tap.rows is not None, "our sparse decode's flat rows"
        flat = {
            "ours": tuple(t[:r] for t in tap.rows),
            "stock": (stock["rows"][:r], stock["counts"][:r]),
        }
        local_sets = _slot_sets(flat["ours"], flat["stock"], r)
        same, overlap, *counts = _global_sets(local_sets, group)
        mla_caches = caches["ours"][0], caches["stock"][0]
        ref_in, _ = _dsa_reference(
            attn, x, lengths, exact, mla_caches, mla_table, rank, n
        )
        sel_in, sel_shard = _dsa_selected_reference(
            attn, x, (flat["ours"], flat["stock"]), mla_caches, rank, n
        )
        feed = _FeedCore(prepared.fi, stock, (ours_mla, at, b))
        _barrier()
        with override_forward_context(forward), feed:
            merged_shard = block.run(x_shard, [caches["ours"][0]], md, run).clone()
        torch.accelerator.synchronize()
        merged_in = block.attn_out[:r].clone()
        _barrier()

        # The selected-key reference over every key is the all-keys one (it reads
        # the flat rows as the decode kernels do).
        for path, sel, full in zip(("ours", "stock"), sel_in, ref_in):
            apart = _rows_error(sel[exact], full)
            assert float(apart.max()) <= 1e-9, (where, path, "reference", apart)
        # Decomposed first (printed even where the whole path fails): the query (our
        # FP8 codes vs stock's), the decode kernel call (stock's inputs through
        # ours), the flat rows (same sets), the global top-k sets, the kernel on
        # stock's inputs with one factor of ours (query, new KV rows, row order:
        # printed), the merge and gate on stock's partials and counts.
        seen = feed.seen
        assert set(seen) == {"rows", "replay", "query", "kv", "order"}, set(seen)
        query = _check_q_codes(where, ours_q, stock["q"], q_scales)
        stock_core = stock["latent"], stock["lse"]
        _check_core_replay(where, seen["replay"], stock, layer.impl, stock_mla)
        same_set = _check_slots(where, local_sets, exact)
        _check_selection(where, same, overlap, counts, lengths)
        keyed = stock["counts"][:r] > 0
        _report_core(where, "our FP8 query", seen["query"], stock_core, keyed)
        _report_core(where, "our new MLA rows", seen["kv"], stock_core, keyed)
        _report_core(
            where, "our flat row order (same set)", seen["order"], stock_core,
            keyed & same_set,
        )  # fmt: skip
        for label, ours_rows, theirs in (
            ("o_proj input", merged_in, entering[0]),
            ("shard", merged_shard, want),
        ):
            label = f"{where} {label}, our merge + gate on stock's partials"
            _report_rows(label, ours_rows, theirs, **DSA_CROSS)
            _check_rows(label, ours_rows, theirs, **DSA_DIRECT)
        # Ours vs stock: the whole path, same-set rows as exact ones, every row
        # within stock's error once the selections' FP64 effect is taken out.
        mine = slice(rank * n, (rank + 1) * n)
        _check_dsa_rows(
            f"{where} o_proj input", attn_in, entering[0], sel_in, same, query
        )
        _check_dsa_rows(f"{where} shard", got, want, sel_shard, same[mine], query)
        # Accuracy: each path against FP64 attention over the keys it selected.
        _check_dsa_reference(f"{where} o_proj input", attn_in, entering[0], sel_in)
        _check_dsa_reference(f"{where} shard", got, want, sel_shard)

        # The step's cache writes: this rank's new MLA rows and indexer keys (the
        # arms wrote stock's new rows into ours: ``a`` holds ours as written).
        _check_rope_part(where, a[:, 512:], b[:, 512:], ours_pe, layer._k_scale_float)
        assert dsa.fp8_codes_adjacent(a[:, :512], b[:, :512]), where
        assert (a[:, :512] != b[:, :512]).sum() <= 0.05 * a[:, :512].numel(), where
        ours_mla[at], stock_mla[at] = 0, 0
        assert torch.equal(ours_mla, stock_mla), f"{where}: other MLA bytes"
        keyed = new_index[new_index >= 0]
        page, offsets = _index_token_bytes(ours_index, keyed, index_row, key_bytes)
        pages = _pages(ours_index, index_row), _pages(stock_index, index_row)
        a, b = (p[page.view(-1, 1), offsets] for p in pages)
        assert (a != b).sum() <= 0.05 * a.numel(), f"{where}: new indexer keys"
        for p in pages:
            p[page.view(-1, 1), offsets] = 0
        assert torch.equal(*pages), f"{where}: other indexer bytes"
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)  # (the new rows as stock wrote them, for the replay)

        # The block captured and replayed: the eager run's shard bit for bit.
        eager = got.clone()
        _barrier()
        graph = torch.cuda.CUDAGraph()
        with (
            override_forward_context(forward),
            graph_capture(device) as capture,
            torch.cuda.graph(graph, stream=capture.stream),
        ):
            replayed = block.run(x_shard, [caches["ours"][0]], md, run)
        _barrier()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(replayed, eager), (where, (replayed - eager).abs().max())
        _barrier()

    if refusals:
        _dsa_refusals(attn, block, step, rank, device)


def _dsa_refusals(attn, block, step, rank, device):
    """Configurations the SP DSA path refuses at build, naming the cause."""
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sp import (
        SpDsaCore,
        sp_dsa_box,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    group = get_tp_group().device_group
    size = torch.distributed.get_world_size(group)
    # sp.sp_dsa_box names the factors: "<TP> x <local heads> gathered query heads".
    refused = f"{size} x 48 gathered query heads: trtllm-gen sparse MLA reads at most"
    with pytest.raises(Unsupported, match=refused):
        sp_dsa_box(group, device, rank, heads=48, cap=CAP, interleave=1)
    dcp2 = _context(rank, device, dcp=2)
    with pytest.raises(Unsupported, match=r"DCP4 box under TP4 / DCP2"):
        SpDsaCore(
            attn, step.rows, step.dsa_scratch, box=step.dsa_box, parity=1, ctx=dcp2
        )
    with pytest.raises(Unsupported, match="above the DcpBox's 512 rows"):
        SpDsaCore(
            attn, (*step.rows, 516), step.dsa_scratch, box=step.dsa_box, parity=1,
            ctx=_context(rank, device),
        )  # fmt: skip
    with pytest.raises(Unsupported, match="not a prepared row count"):
        block.core.choice(step.rows[0] + 2)


@pytest.mark.parametrize("topk", ["merge", "exchange"])
@pytest.mark.parametrize("indexer", ["mxfp4", "fp8"])
def test_sp_dsa_block_matches_stock_layer(indexer, topk):
    """SpAttentionBlock (DSA) vs the stock KimiK3DSAAttention under the SP seam at
    ``DSA_ROWS`` (module docstring); C512 deploys the MXFP4 indexer with DcpExchange."""
    pytest.importorskip("cutlass")
    _launch("dsa", indexer, topk, DSA_ROWS, True)


@pytest.mark.slow_test
def test_sp_dsa_block_matches_stock_layer_all_rows():
    """The DSA A/B at every captured R (4..512), as C512 deploys it."""
    pytest.importorskip("cutlass")
    _launch("dsa", "mxfp4", "exchange", ALL_ROWS, False)


# ------------------------------------------------------------------------------ KDA


def _kda_config():
    """The KDA layer's vLLM config: TP4 = DCP4, a BF16 conv / FP32 recurrent state,
    RecoverSSM records (--use-replayssm with q4 DSpark), 128 requests; the stock
    layer's Triton prefill (decode is what is compared)."""
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from tests.v1.attention.utils import create_vllm_config

    config = create_vllm_config(
        model_name=dsa.LAYER_MODEL,
        max_model_len=MAX_LEN,
        block_size=UNIT,
        max_num_seqs=CAP // Q,
        max_num_batched_tokens=8192,
    )
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = WORLD
    parallel.decode_context_parallel_size = WORLD
    parallel.cp_kv_cache_interleave_size = 1
    cache = config.cache_config
    cache.mamba_ssm_cache_dtype = "float32"
    cache.use_replayssm = True
    cache.use_kda_recoverssm = True
    config.additional_config = {"kda_prefill_backend": "triton"}
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=Q - 1,
        enable_adaptive_verification=False,
        parallel_drafting=False,
    )
    return config


def _kda_attention(config, rank):
    """A K3.1 KDA layer at C512's sizes (96 heads over TP4, hidden 7168, width-4
    conv, bounded gate), random weights (this rank's shards; the in-projection's
    pad rows zero, as the loader leaves them), the SP decoder's o_proj."""
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text = KimiLinearConfig(
        hidden_size=WIDTH, rms_norm_eps=1e-5, hidden_act="silu",
        linear_attn_config={
            "kda_layers": [2], "full_attn_layers": [4], "head_dim": 128,
            "num_heads": WORLD * HEADS, "short_conv_kernel_size": 4,
            "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        },
    )  # fmt: skip
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attn = KimiK3DeltaAttention(text, config, prefix="model.layers.1.self_attn")
    finally:
        torch.set_default_dtype(torch.float32)
    assert attn.use_recoverssm and attn.recoverssm_backend == "triton"
    g = torch.Generator(device="cuda").manual_seed(11 + rank)

    def randn(t, scale):
        return torch.randn(t.shape, generator=g, device="cuda") * scale

    dim = HEADS * 128
    with torch.no_grad():
        w = attn.in_proj_qkvgfab.weight
        w.copy_(randn(w, WIDTH**-0.5))
        if attn.in_proj_padding:
            w[-attn.in_proj_padding :] = 0
        attn.f_b_proj.weight.copy_(randn(attn.f_b_proj.weight, 128**-0.5))
        attn.conv1d.weight.copy_(randn(attn.conv1d.weight, 0.5))
        attn.dt_bias.copy_(randn(attn.dt_bias, 0.5))
        attn.A_log.copy_(randn(attn.A_log, 0.5))
        attn.o_norm.weight.copy_(1 + randn(attn.o_norm.weight, 0.1))
        attn.o_proj.weight.copy_(randn(attn.o_proj.weight, dim**-0.5))
        if attn.decode_conv1d_weight is not None:  # (the loader's decode copies)
            conv = attn.conv1d.weight.squeeze(1)
            for i in range(3):
                attn.decode_conv1d_weight[i].copy_(conv[i * dim : (i + 1) * dim].T)
        if attn.decode_norm_weight is not None:
            attn.decode_norm_weight.copy_(attn.o_norm.weight)
    attn.o_proj.reduce_results = False  # (KimiDecoderLayer under SP)
    return attn


def _kda_caches(slots, gen):
    """The DS conv window [slot, 3 dim, q + 2] (BF16) and the FP32 state."""
    dim = HEADS * 128
    conv = torch.randn(slots, 3 * dim, Q + 2, generator=gen, device="cuda")
    state = torch.randn(slots, HEADS, 128, 128, generator=gen, device="cuda") * 0.1
    return [conv.to(torch.bfloat16), state]


def _kda_case(config, rank, device, rows, steps, graph, refusals):
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.common.ops.sequence_parallel import (
        sp_all_gather,
        sp_reduce_scatter,
    )
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.sp_comm import (
        SpComm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm import (
        prepare_recoverssm,
    )

    assert is_conv_state_dim_first()  # C512's DS conv layout
    attn = _kda_attention(config, rank)
    ctx = _context(rank, device)
    decoder = SimpleNamespace(layer_idx=1, self_attn=attn)
    records = prepare_recoverssm([decoder], ctx, defer=False)
    assert records == "triton", records
    widest = max(rows) // Q
    slots = 2 * widest + 2
    gen = _generator("kda-caches", rank)
    caches = {"stock": _kda_caches(slots, gen)}
    caches["ours"] = [t.clone() for t in caches["stock"]]
    group = get_tp_group().device_group
    comm = SpComm(group, device, WORLD, rank, width=WIDTH, rows_cap=CAP)
    step = _Step(ctx, tuple(rows), device, comm, records=records)
    block = _sp_block(1, attn, step)
    block.core.prepare_caches(caches["ours"])

    def metadata(reqs, checkpoints):
        r = reqs * Q
        return KimiK3KDAMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=reqs, num_spec_decode_tokens=r, num_actual_tokens=r,
            spec_query_start_loc=torch.arange(
                0, r + 1, Q, dtype=torch.int32, device=device
            ),
            spec_state_indices_tensor=checkpoints.view(reqs, 1),
            num_accepted_tokens=torch.ones(reqs, dtype=torch.int32, device=device),
            uniform_spec_sequence_length=Q,
        )  # fmt: skip

    def commit(path, accepted, checkpoints, starts):
        attn.kv_cache = caches[path]
        KDARecoverSSMCommitContext.create(
            [attn], spec_query_len=Q, max_num_reqs=CAP // Q, backend="triton"
        ).commit(accepted, checkpoints, starts)

    rec = attn.recoverssm_records
    for r in rows:
        reqs, n = r // Q, r // WORLD
        # Both paths start every case from the same states (the previous case's
        # commits left them within its bounds, not equal).
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)
        order = torch.randperm(
            slots - 1, generator=_generator("slots", r, device="cpu")
        )
        checkpoints = (order[:reqs] + 1).to(torch.int32).to(device)
        md = metadata(reqs, checkpoints)
        starts = md.spec_query_start_loc
        forward = create_forward_context({attn.prefix: md}, config)
        x_shard = torch.empty(n, WIDTH, dtype=torch.bfloat16, device=device)
        run = SimpleNamespace(slot_mapping={}, q=Q)
        replay = replayed = None
        if graph:  # the specialized block captured once, replayed every step
            _barrier()
            replay = torch.cuda.CUDAGraph()
            with (
                graph_capture(device) as capture,
                torch.cuda.graph(replay, stream=capture.stream),
            ):
                replayed = block.run(x_shard, caches["ours"], md, run)
            _barrier()
        state_errors = []
        for t in range(steps):
            where = f"KDA R{r} step {t} rank {rank}"
            x = torch.randn(
                r, WIDTH, dtype=torch.bfloat16, device=device,
                generator=_generator("kda-x", r, t),
            )  # fmt: skip
            x_shard.copy_(x[rank * n : (rank + 1) * n])
            # Stock: the SP decoder's seam around the layer forward (its records).
            entering: list = []
            hook = attn.o_proj.register_forward_pre_hook(_keep_input(entering))
            attn.kv_cache = caches["stock"]
            with override_forward_context(forward):
                gathered = sp_all_gather(x_shard)[:r]
                positions = torch.zeros(r, dtype=torch.long, device=device)
                want = sp_reduce_scatter(
                    attn(hidden_states=gathered, positions=positions)
                )
            hook.remove()
            stock_rec = (rec.correction[:reqs].clone(), rec.key_gate[:reqs].clone())
            # Ours: the SP block (its records into the layer's record rows).
            if replay is not None:
                replay.replay()
                got = replayed
            else:
                with override_forward_context(forward):
                    got = block.run(x_shard, caches["ours"], md, run)
            torch.accelerator.synchronize()
            # (no top-k: every row is compared, within the in-projections' BF16
            # rounding difference carried through the conv and the recurrence)
            bounds = dict(median=1e-2, tail=2e-2, worst=5e-2)
            _check_rows(
                f"{where} o_proj input", block.attn_out[:r], entering[0], **bounds
            )
            _check_rows(f"{where} shard", got, want, **bounds)
            # Records and conv windows: the two in-projections round differently
            # (BF16 GEMM order), so within BF16 bounds, at the stock row layout.
            correction, key_gate = rec.correction[:reqs], rec.key_gate[:reqs]
            torch.testing.assert_close(
                key_gate.float(), stock_rec[1].float(), rtol=2**-6, atol=2e-2,
                msg=_msg(f"{where} keys / raw gates"),
            )  # fmt: skip
            error = float((correction - stock_rec[0]).norm() / stock_rec[0].norm())
            assert error <= 2e-2, (where, "corrections", error)
            live = checkpoints.long()
            a, b = caches["ours"][0][live].float(), caches["stock"][0][live].float()
            torch.testing.assert_close(
                a, b, rtol=2**-7, atol=1e-2, msg=_msg(f"{where} conv window")
            )
            # Both commit their own records (the same accepted counts).
            accepted = torch.randint(
                1, Q + 1, (reqs,), generator=_generator("accepted", r, t, device="cpu")
            )
            accepted[0] = Q
            if reqs > 1:
                accepted[1] = 1
            accepted = accepted.to(torch.int32).to(device)
            commit("ours", accepted, checkpoints, starts)
            rec.correction[:reqs].copy_(stock_rec[0])
            rec.key_gate[:reqs].copy_(stock_rec[1])
            commit("stock", accepted, checkpoints, starts)
            torch.accelerator.synchronize()
            ours_state, stock_state = caches["ours"][1][live], caches["stock"][1][live]
            error = float(
                (ours_state.double() - stock_state.double()).norm()
                / stock_state.double().norm()
            )
            state_errors.append(error)
            assert error <= 2e-2, (where, "committed state", error)
            others = torch.ones(slots, dtype=torch.bool, device=device)
            others[live] = False
            for i in range(2):  # slots of no request untouched on both paths
                assert torch.equal(
                    caches["ours"][i][others], caches["stock"][i][others]
                )
            _barrier()
        print(f"KDA R{r} rank {rank}: committed state error per step {state_errors}")
        # BF16 input rounding enters every step; the recurrence must not amplify it.
        assert max(state_errors) <= 4 * state_errors[0] + 1e-3, state_errors

    if refusals:
        _kda_refusals(attn, block, step, rank, device)


def _kda_refusals(attn, block, step, rank, device):
    """Configurations the SP KDA path and the SP o-projection refuse at build."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import SpKdaCore
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.sp import SpOproj
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    bf16 = _context(rank, device, bf16_state=True)
    with pytest.raises(Unsupported, match="FP32 SSM state"):
        SpKdaCore(attn, step.rows, records="triton", ctx=bf16)
    with pytest.raises(Unsupported, match="not whole requests"):
        SpKdaCore(attn, (6,), records="triton", ctx=step.ctx)
    off = _context(rank, device, sequence_parallel=False)
    with pytest.raises(Unsupported, match="SP o-projection at TP4"):
        SpOproj(attn.o_proj.weight, step.comm, step.rows, off)
    with pytest.raises(Unsupported, match=r"not a multiple of TP4 in 1\.\.512"):
        SpOproj(attn.o_proj.weight, step.comm, (516,), step.ctx)
    missing = next(r for r in ALL_ROWS if r not in step.rows)
    with pytest.raises(Unsupported, match="not a served row count"):
        block.core.choice(missing)


def test_sp_kda_block_matches_stock_layer():
    """SpAttentionBlock (KDA, RecoverSSM records) vs the stock KimiK3DeltaAttention
    under the SP seam at ``KDA_ROWS``, two steps with partial acceptance."""
    pytest.importorskip("cutlass")
    _launch("kda", KDA_ROWS, 2, False, True)


def test_sp_kda_recoverssm_multistep():
    """``CHAIN_STEPS`` RecoverSSM steps with partial acceptance, the specialized
    block replayed from one CUDA graph: records, conv window and committed FP32
    state against the stock layer and commit at every step."""
    pytest.importorskip("cutlass")
    _launch("kda", CHAIN_ROWS, CHAIN_STEPS, True, False)


@pytest.mark.slow_test
def test_sp_kda_block_matches_stock_layer_all_rows():
    """The KDA A/B at every captured R (4..512)."""
    pytest.importorskip("cutlass")
    _launch("kda", ALL_ROWS, 2, False, False)
