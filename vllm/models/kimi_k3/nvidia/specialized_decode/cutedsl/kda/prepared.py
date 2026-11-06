# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared KDA core. One packed projection supplies Q/K/V, the output gate, F_A and
beta; per-batch policies select the gate schedule and the projection row pitch. Verify
batches the caller lists in ``prepared`` also get ``KdaVerifyPrepared``, which reads
the operands an in-projection epilogue prepared (``run(..., prepared=...)``) and polls
the projection's F_A rows: the caller arms them (every BF16 element ``FA_SENT``) before
the first run and gives that projection buffer to this path only; each run re-arms
them."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Choice, Path, rows_policy
from ..common.host import fake, layout, prepare_shared_carveout, stream, to_cute
from ..common.policy import batch_table
from .kernel import KdaSplit
from .verify import KdaVerifyCols, KdaVerifyPrepared, KdaVerifyRows
from .verify_stream import KdaVerifyStream
from .verify_stream3 import KdaVerifyStream3
from .verify_wide import KdaVerifyHalves, KdaVerifyWide

# KdaSplit schedule "early.lanes.occupancy.cluster.evict" per batch by heads (TP8 12,
# TP16 6), from GB300 sweeps: early predecessor-independent loads (MMA gate modes only;
# 2 also stages the state rows in shared memory, 3 also the F_B operands), lanes per
# state row, minimum CTAs per SM, CTAs per head (0: defaults) and L2 evict-first state
# traffic.
SCHEDULE = {12: "16-32:3.16.3.2.1,64:3.16.4.2.1", 6: "16:1.0.0,32-64:3.16.3.2"}
# Per compute capability: overrides on top of SCHEDULE. Rubin (10.7, 212 SMs) TP16
# B1-8, from Rubin sweeps of the in-projection -> KdaSplit chain: staged state rows
# (evict-first), lanes 16, three CTAs per SM, eight-CTA clusters up to B4, then four.
DEVICE_SCHEDULE = {(10, 7): {6: "1-4:2.16.3.8.1,5-8:2.16.3.4.1"}}
# BF16 recurrent state (--mamba-ssm-cache-dtype bfloat16): overrides on top of those,
# from the same Rubin chain sweeps: four-CTA clusters win at B2-4. B5-8 keep the FP32
# geometry: behind an AttnRes-like predecessor and before the o-projection (the serving
# chain), two CTAs per head with staged F_B operands (3.16.3.2.1, fastest without the
# o-projection) are 0.48 us per layer slower at B8 and 0.67 at B7.
BF16_DEVICE_SCHEDULE = {(10, 7): {6: "2-4:2.16.3.4.1"}}
# Verify (q > 1) schedules by request count B, from GB300
# sweeps over q 4-8: no staging; eight-CTA clusters, then fewer as B grows.
SPEC_SCHEDULE = {
    12: "1:1.8.2.8,2-4:1.8.3.8,5-12:1.8.3.4,13-16:1.8.3.2",
    6: "1:1.8.2.8,2:1.8.0.8,3-12:1.8.3.8,13-16:1.8.3.4",
}
# Per compute capability: verify schedules on top of SPEC_SCHEDULE. Rubin TP16 above 16
# requests (the GB300 table stops at 16, leaving the default geometry: 16 lanes, four
# CTAs per head, no occupancy bound), from a 1-GPU sweep of KdaSplit verify alone (69
# layers, cold state slots, BF16 state), us per layer at B32 / B64: q8 34.4 / 64.4 ->
# 18.1 / 27.9, q4 15.6 / 25.9 -> 12.1 / 18.6 (lanes 8, three CTAs per SM, four per
# head); same arithmetic (8 or 16 lanes per row reduce in the same tree). B4: two CTAs
# per SM instead of three (GB300's 1.8.3.8 packs the 192 CTAs onto ~72 SMs and the
# token loop takes ~3 us); 1-GPU in-projection -> KdaSplit -> o-projection-footprint
# chain, the o-projection stand-in's PDL wait returned 11.75 -> 10.50 us after the
# AttnRes stand-in (us/call over the chain 14.34 -> 12.52). B4 also marks its
# checkpoint stores L2 evict-first (the same values; that chain, BF16 state: 10.46 ->
# 9.98 us, 13.00 -> 12.52 us/call; B2 gains nothing from it).
DEVICE_SPEC_SCHEDULE = {(10, 7): {6: "4:1.16.2.8.1,17-64:1.8.3.4"}}
# Compute capabilities whose verify steps activate only the gates a CTA uses and each
# beta once per CTA (``KdaSplit(lean=True)``, bitwise identical). Rubin TP16 B8 x q8,
# 1-GPU in-projection -> KdaSplit -> o-projection-footprint chain: 14.76 -> 14.66 us.
LEAN_VERIFY = {(10, 7)}
# Per compute capability: verify request counts whose KdaSplit normalizes q / k at
# operand load (``KdaSplit(load_norm=True)``, bitwise identical) instead of in a pass
# and a CTA barrier of its own. Rubin TP16 q8, the same chain, the o-projection
# stand-in's PDL wait returned (us after the AttnRes stand-in): B2 9.82 -> 9.62, B4
# 10.46 -> 10.35.
VERIFY_LOAD_NORM: dict[tuple[int, int], str] = {(10, 7): "2-4:1"}
# Per compute capability: state rows per recurrence thread of the prepared verify
# (``KdaVerifyRows``, bitwise) by request count B, "rows[.spare warps]"; 1
# (absent) keeps one row per thread. Rubin TP16 B8 x q8, in-projection -> verify ->
# o-projection-footprint chain: two rows per thread plus a spare warp (three CTAs per
# SM, as with one row; 168 registers would let four pack) 14.25 -> 13.93 us, token loop
# 7640 -> 6340 cycles. Four rows (247 registers) pack four per SM and lose; capping by
# an explicit shared-memory size delays the CTAs' residency by ~4 us.
VERIFY_ROWS: dict[tuple[int, int], str] = {(10, 7): "8:2.1"}
# Per compute capability: the prepared verify's recurrence transposed
# (``KdaVerifyCols``, bitwise) by request count B, the value its spare warps (86
# registers: one caps CTAs at three per SM); takes precedence over VERIFY_ROWS. Rubin
# TP16 B8 x q8, the same chain: 13.50 -> 12.49 us per call vs VERIFY_ROWS (one LDS.128
# per operand per lane-token).
VERIFY_COLS: dict[tuple[int, int], str] = {(10, 7): "8:1"}
# Per compute capability: request counts whose plain (not prepared) verify runs one
# 16-warp CTA per (request, head) (1: ``KdaVerifyWide``) or a cluster of two 8-warp
# CTAs (2: ``KdaVerifyHalves``) instead of KdaSplit; both bitwise. 3 / 4: the chunked
# verify (``KdaVerifyStream`` / ``KdaVerifyStream3``, q <= 4, BF16 state; not
# bitwise). Rubin TP16, 1-GPU verify alone (69 layers, cold slots), us per layer
# KdaSplit / Wide / Halves: q8 B16 9.60 / 10.62 / 8.93, B24 16.88 / 11.10 / 11.98,
# B32 18.13 / 11.21 / 12.13, B48 24.76 / 21.51 / 19.75, B64 27.44 / 22.21 / 23.73; q4
# B16 6.27 / 6.40 / 5.57, B24 10.57 / 6.82 / 7.27, B32 11.93 / 7.10 / 7.49, B48 16.76
# / 13.58 / 12.58, B64 18.31 / 14.10 / 14.80 (Halves fill SMs where one CTA per head
# leaves a short second wave).
# B16 is Wide: beside its neighbours (1-GPU chain: AttnRes stand-in, in-projection,
# verify, o-projection footprint; the o-projection's PDL wait returned, us after the
# AttnRes) KdaSplit / Halves / Wide q8 21.74 / 22.61 / 21.52, q4 10.62 / 11.10 / 10.21;
# Halves also trails Wide at B24 / B32 there (q8 +0.85 / +1.07, q4 +1.04 / +0.99).
VERIFY_WIDE: dict[tuple[int, int], str] = {(10, 7): "16-32:1,33-48:2,49-64:1"}
# Per compute capability and verify length q: rows read on top of DEVICE_SPEC_SCHEDULE
# ({q: {heads: spec}}) and of VERIFY_ROWS / VERIFY_COLS / VERIFY_WIDE / VERIFY_LOAD_NORM
# ({q: spec}), which key on the request count alone. A row here moves only that q's
# kernels (the same request count runs other rows at other q).
# Rubin TP16 q4 (DL3), from a 1-GPU sweep of the in-projection -> verify chain between
# tail / AttnRes / o-projection stand-ins (o-projection PDL-wait release after the
# AttnRes end, us per layer, every arm bitwise): 6 requests two CTAs per SM instead of
# three (1.8.3.8 -> 1.8.2.8: 11.78 -> 10.46), 14 requests likewise (1.8.3.4 -> 1.8.2.4:
# 14.53 -> 13.86); 12 requests on KdaVerifyWide (KdaSplit 1.8.3.8 16.19 -> 13.54).
# 33-34 requests on KdaVerifyWide instead of VERIFY_WIDE's Halves (graph us per layer,
# cuBLAS in-projection: 30.00 -> 28.90, 30.13 -> 29.03); Halves stays at 36-48 (Wide
# +0.1..+0.5) and Wide at 50-64 (Halves +0.1..+2.0).
# 65-256 requests on KdaVerifyStream3 (4: the chunked verify with tensor-core row dots;
# not bitwise: u_t / o_t from dots against the step's base state, FP64-band checked)
# instead of Wide / Halves / the default-geometry KdaSplit (no row above 64). 1-GPU
# verify alone, 69 layers, own cold BF16 slots, us per layer, Wide / Halves / Stream3:
# B33 6.27 / 7.08 / 6.70 (Wide: one wave of 198 CTAs), B36 10.92 / 11.21 / 8.94, B48
# 11.66 / 12.38 / 9.27, B64 12.28 / 13.88 / 10.29, B96 18.65 / - / 16.26, B128 25.03 /
# - / 19.44, B192 37.22 / - / 28.51, B256 49.13 / - / 37.25 (KdaSplit default at B96 /
# 128 / 192: 36.38 / 47.60 / 69.85). 35-64 requests stay on VERIFY_WIDE's Halves /
# Wide, as the cap-128 serving overlay (ov-cap128-bca26213) runs them (its numbers are
# not recorded in the tree).
# BF16 state, 36-64 requests at q4 on Stream3 too ("36-256:4", was "65-256:4" with
# the cap-128 overlay's Halves / Wide at 35-64): the VR200 kernel bench (nqbench
# kern-kda, section b, 1-GPU verify alone, 69 layers, own cold slots, us per layer)
# Halves or Wide / Stream3: B36 11.23 / 9.06, B40 11.51 / 9.27, B48 12.34 / 9.65,
# B56 11.98 (Wide) / 9.79, B64 12.45 (Wide) / 10.57: -15..-22%, the margin the
# verify-alone numbers above showed (B36 / B48 / B64 +2.27 / +3.11 / +1.99 us). B33-35
# stay on Wide (B33 / B34 Wide 6.29 / 6.34 vs Stream3 6.52 / 6.60; 35 requests are
# still one wave of 210 Wide CTAs on 212 SMs, Halves' 420 CTAs two).
# q8 B12 on KdaVerifyWide (it ran the 1.8.3.8 KdaSplit, the only q8 row between B8 and
# B16 off the wide kernels): same bench, KdaSplit / Wide 13.51 / 8.83 us (-35%; Halves
# 7.34 alone, but Halves trails Wide in the serving chain at B16-32, see VERIFY_WIDE).
# The bench's other BF16 retune hints stay unapplied: Halves at q4 B8 / B12 / B14 / B16
# and q8 B8 / B16 lead by 0.31-0.38 us (q4) / 0.52-1.48 us (q8) verify alone, less
# than the serving-chain shift against Halves at B16 (VERIFY_WIDE's numbers, q4 / q8:
# Halves 0.83 / 1.69 us ahead of Wide alone, 0.89 / 1.09 us behind in the chain;
# 0.70 / 0.67 ahead of KdaSplit alone, 0.48 / 0.87 behind in the chain).
# Stream3 (and Stream) need a BF16 state (``BF16_ONLY``): an FP32 state reads
# FP32_VERIFY_WIDE below instead.
DEVICE_SPEC_SCHEDULE_Q: dict = {(10, 7): {4: {6: "6:1.8.2.8,14:1.8.2.4"}}}
VERIFY_Q: dict = {
    "rows": {},
    "cols": {},
    "wide": {(10, 7): {4: "12:1,33-35:1,36-256:4", 8: "12:1"}},
    "load_norm": {},
}
# FP32 recurrent state (--mamba-ssm-cache-dtype float32, the production precision):
# the plain verify kernel by request count per q, in place of VERIFY_WIDE and VERIFY_Q
# "wide" (whose crossovers were tuned with a BF16 state; an FP32 state doubles the
# state bytes per verify, which moves them). Values as VERIFY_WIDE's (0 KdaSplit at
# its SPEC_SCHEDULE / DEVICE_SPEC_SCHEDULE(_Q) geometry, 1 Wide, 2 Halves; never the
# BF16-only 3 / 4); requests below the first row and between rows run KdaSplit, above
# the last row the last row's kernel (``wide_kind``). From the VR200 kernel bench
# (nqbench kern-kda, section b: 1-GPU verify alone, 69 layers, own cold FP32 slots,
# us per layer, KdaSplit / Wide / Halves):
#   q4 B1-8   2.90 / 6.73 / 5.23 .. 4.67 / 7.08 / 5.42  KdaSplit (unchanged)
#      B12    7.71 / 7.33 / 6.25   Wide (unchanged; Halves' lead 1.08 is below its
#                                  serving-chain shift vs Wide at B16, 1.72)
#      B14    5.60 / 7.20 / 6.15   KdaSplit at 1.8.2.4 (unchanged)
#      B16-34 Wide: B16 6.83 / 7.15 / 6.22, B24 11.90 / 7.98 / 9.01, B32 12.83 / 8.64 /
#             9.65, B34 12.97 / 8.75 / 9.93 (unchanged; at B16 KdaSplit's / Halves'
#             leads, 0.32 / 0.93, are below their serving-chain shifts vs Wide, 0.54
#             / 1.72; B35: one Wide wave of 210 CTAs, was Halves)
#      B36-40 KdaSplit 1.8.3.4: B36 13.05 / 14.28 / 14.59, B40 13.40 / 14.69 / 16.12
#             (was Halves: -10.5% / -16.9%)
#      B41-48 Wide: B48 19.03 / 15.48 / 18.30 (was Halves: -15.4%); KdaSplit's step
#             between B40 and B48 is unmeasured, so B41-47 take Wide, which runs two
#             even waves from B36 to B70 (B36 14.28, B65 17.13)
#      B49+   Wide: B64 20.75 / 16.95 / 19.23 .. B512 148.81 / 127.19 / 142.02 (the
#             others, Halves and KdaSplit at the default geometry or 1.8.3.4,
#             trail by 10-15%; unchanged: FP32_WIDE)
#   q8 B1-8   KdaSplit: 4.70 / 11.04 / 7.91 .. 7.49 / 11.85 / 8.24 (unchanged)
#      B12    Wide: 13.40 / 12.09 / 9.31 (was KdaSplit: -9.8%; Halves as at q4 B12)
#      B16-32 Wide: B16 11.52 / 11.77 / 9.60, B32 21.93 / 13.81 / 15.07 (unchanged)
#      B33-48 Wide: B48 31.96 / 25.24 / 27.98 (was Halves: -9.8%)
#      B49+   Wide: B64 34.19 / 27.09 / 29.34 .. B256 161.85 / 106.08 / 109.03
#             (Halves trails by 3-8%; unchanged)
# q values without a row here keep VERIFY_WIDE / VERIFY_Q with ``FP32_WIDE`` in place
# of the BF16-only kernels.
FP32_VERIFY_WIDE: dict = {
    (10, 7): {4: "12:1,16-35:1,36-40:0,41-512:1", 8: "12:1,16-256:1"},
}
# Plain verify kernels by VERIFY_WIDE / VERIFY_Q "wide" value (0: KdaSplit).
WIDE_KINDS = (KdaVerifyWide, KdaVerifyHalves, KdaVerifyStream, KdaVerifyStream3)
WIDE_NAMES = ("kdasplit", "wide", "halves", "stream", "stream3")
# Values whose kernel holds the base state as BF16 (KdaVerifyStream / Stream3: S0
# staged as BF16 pairs and, in Stream3, fed exactly to BF16 tensor-core MMAs; their
# constructor asserts a BF16 state). An FP32 state (--mamba-ssm-cache-dtype float32)
# runs FP32_WIDE at those rows instead: KdaVerifyWide, bitwise KdaSplit's verify, which
# reads and stores the state through ``load_state`` / ``store_state`` (FP32 rows) and
# scales with the requests (grid (1, heads, B)); with a BF16 state it led Halves and
# the default-geometry KdaSplit at every measured row above 48 (numbers above: B64
# 12.28 / 13.88, B96 18.65 vs KdaSplit 36.38). An FP32 chunked verify (S0 staged as
# FP32, three-term BF16 splits of S0 into the row-dot MMAs) is not written.
# The FP32 crossovers are FP32_VERIFY_WIDE's (measured verify alone).
# NQ-TUNE: FP32 state in the serving chain (AttnRes, in-projection, verify,
# o-projection) at the FP32_VERIFY_WIDE crossovers (q4 B12 / B16 Halves vs Wide, B36-48
# KdaSplit vs Wide, q8 B12 Halves vs Wide); q4 B9-11 / B13 / B15 and q8 B9-11 / B13-15
# (KdaSplit, unmeasured); SPEC_SCHEDULE / DEVICE_SPEC_SCHEDULE(_Q) and VERIFY_ROWS /
# COLS at B8 swept with an FP32 state; the stock S1 crossover (STOCK_ROWS) too.
BF16_ONLY, FP32_WIDE = (3, 4), 1
_VERIFY = {
    "rows": VERIFY_ROWS, "cols": VERIFY_COLS, "wide": VERIFY_WIDE,
    "load_norm": VERIFY_LOAD_NORM,
}  # fmt: skip


def verify_table(name, capability, tokens):
    """The request-keyed verify table ``name`` ("rows", "cols", "wide", "load_norm")
    at ``capability`` with VERIFY_Q's rows for verify length ``tokens`` on top."""
    base = _VERIFY[name].get(capability, "")
    extra = VERIFY_Q[name].get(capability, {}).get(tokens, "")
    return batch_table(",".join(filter(None, (base, extra))))


def wide_table(capability, tokens, bf16_state):
    """The request-keyed plain verify kernel table at verify length ``tokens`` for
    the state dtype: FP32_VERIFY_WIDE's row for an FP32 state where it has one, else
    VERIFY_WIDE with VERIFY_Q's "wide" rows on top."""
    fp32 = None if bf16_state else FP32_VERIFY_WIDE.get(capability, {}).get(tokens)
    if fp32 is not None:
        return batch_table(fp32)
    return verify_table("wide", capability, tokens)


def wide_kind(requests, tokens, capability, bf16_state=False, gate_mode=0, heads=6):
    """The plain (not prepared) verify kernel at ``requests`` x ``tokens``: 0 KdaSplit,
    else the ``wide_table`` value (``WIDE_KINDS[value - 1]``): FP32_VERIFY_WIDE for an
    FP32 state, VERIFY_WIDE / VERIFY_Q "wide" for a BF16 one.

    Open-ended: requests above the table's last row take that row's kernel (each
    wide kernel's grid is (CTAs per head, heads, B): no request cap), so any captured
    batch runs the kernel that scales instead of the default-geometry KdaSplit (~1.9x
    Wide's time at q4 B96-192, see above). ``BF16_ONLY`` values need a BF16 state and
    q <= 4; otherwise ``FP32_WIDE`` runs. The wide kernels run gate mode 0 without
    fast math (their asserts): other layers (TP8: gate mode 1, 12 heads) keep
    KdaSplit."""
    if tokens == 1 or gate_mode != 0 or heads == 12:
        return 0
    table = wide_table(capability, tokens, bf16_state)
    kind = table.get(requests)
    if kind is None and table and requests > max(table):
        kind = table[max(table)]
    if kind in BF16_ONLY and (not bf16_state or tokens > 4):
        kind = FP32_WIDE
    return kind or 0


def schedule_of(heads, requests, gate_mode, tokens, capability, bf16_state, rows=True):
    """KdaSplit's schedule at ``requests`` x ``tokens`` (SCHEDULE / SPEC_SCHEDULE and
    the device tables on top); ``rows``: the state's rows are whole and contiguous
    (staging copies them)."""
    table = SCHEDULE if tokens == 1 else SPEC_SCHEDULE
    spec = table.get(heads, "")
    if tokens == 1:
        tables = (DEVICE_SCHEDULE, BF16_DEVICE_SCHEDULE)[: 1 + bf16_state]
        overrides = (t.get(capability, {}).get(heads, "") for t in tables)
        spec = ",".join(filter(None, (spec, *overrides)))
    else:
        override = DEVICE_SPEC_SCHEDULE.get(capability, {}).get(heads, "")
        by_q = DEVICE_SPEC_SCHEDULE_Q.get(capability, {}).get(tokens, {})
        spec = ",".join(filter(None, (spec, override, by_q.get(heads, ""))))
    schedule = batch_table(spec)
    schedule = schedule.get(requests, (0, 0, 0)) if gate_mode != 2 else (0,) * 3
    if schedule[0] >= 2 and not rows:  # staging needs whole rows
        schedule = (1, *schedule[1:4])  # no evict hint without staging
    return schedule


@cache
def executable(specs, heads, gate_mode, has_bias, device, tokens=1, prepared=False):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    requests, bf16_state = specs[0][0][0] // tokens, specs[9][2] == torch.bfloat16
    capability = torch.cuda.get_device_capability(device)
    rows = specs[9][1][-2:] == (128, 1)
    schedule = schedule_of(
        heads, requests, gate_mode, tokens, capability, bf16_state, rows
    )
    options = (heads, requests, gate_mode, has_bias, schedule, tokens, bf16_state)
    kernel = make_kernel(options, capability, prepared)
    scalars = (cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        compiled = cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )
    carveout = getattr(kernel, "shared_carveout", None)  # several staged CTAs per SM
    if carveout is not None:
        compiled = prepare_shared_carveout(compiled, device, carveout)
    return compiled


def make_kernel(options, capability, prepared=False) -> KdaSplit:
    """The kernel object ``executable`` compiles for ``options`` = (heads, requests,
    gate_mode, has_bias, schedule, tokens, bf16_state) (no GPU work: the constructors'
    shape / dtype asserts run here)."""
    heads, requests, gate_mode, _, _, tokens, bf16_state = options
    rows = verify_table("rows", capability, tokens).get(requests, 1)
    rows = rows if isinstance(rows, tuple) else (rows,)
    cols = verify_table("cols", capability, tokens).get(requests)
    wide = wide_kind(requests, tokens, capability, bf16_state, gate_mode, heads)
    if prepared and cols is not None:  # the transposed recurrence
        return KdaVerifyCols(*options, cols)
    if prepared and rows[0] > 1:  # several state rows per thread, operands loaded once
        return KdaVerifyRows(*options, *rows)
    if prepared:  # the gates and betas arrive activated (the in-projection epilogue)
        return KdaVerifyPrepared(*options)
    if wide:  # one or two CTAs per (request, head), or the chunked verify
        return WIDE_KINDS[wide - 1](*options)
    lean = tokens > 1 and capability in LEAN_VERIFY
    norm = verify_table("load_norm", capability, tokens).get(requests)
    return KdaSplit(*options, lean, bool(norm))


def decode_conv_weight(weight):
    """The [3, width, D] (Q, K, V) decode layout of the [3 * D, 1, width] conv
    weight."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


class PreparedKda:
    def __init__(self, attention, policies, conv_dim_first, prepared=()):
        """``prepared``: verify policy keys (requests, tokens) that also run on
        prepared operands."""
        bias, dt = attention.conv1d.bias, attention.dt_bias
        self.heads = attention.A_log.numel()
        self.dim, self.has_bias = self.heads * 128, bias is not None
        conv_w = decode_conv_weight(attention.conv1d.weight)
        bias, fb = dt if bias is None else bias, attention.f_b_proj.weight
        norm = attention.o_norm.weight.detach().float()  # BF16 slows the fused kernel
        self.weights = (conv_w, bias, fb, attention.A_log, dt, norm)
        self.device, self.policies = dt.device, policies
        scalars = (attention.gate_lower_bound, 128**-0.5, attention.o_norm.eps)
        self.scalars = tuple(map(cutlass.Float32, scalars))
        self.conv_dim_first, self.kernels = conv_dim_first, {}
        self.prepared, self.prepared_kernels = frozenset(prepared), {}
        # KdaVerifyPrepared's F_A reader counts per request (its last reader re-arms)
        requests = max((r for r, _ in self.prepared), default=0)
        self.readers = torch.zeros(requests, dtype=torch.int32, device=self.device)

    def _operands(self, caches, projected, slots, out, accepted=None):
        """Kernel tensors in launch order: 13 before the scalars, then F_B (and the
        verify step's accepted counts)."""
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        ops = (p(0, 3 * d), w, bias, conv, p(4 * d, 128), p(4 * d + 128, h), a_log, dt)
        tail = (fb,) if accepted is None else (fb, accepted)
        return ops + (slots, state, out, p(3 * d, d), norm) + tail

    def prepare_caches(self, caches):
        """Policy keys: a batch, or (requests, tokens) for verify steps; prepared
        operands: [requests * tokens, 4 * dim + 8] FP32."""
        for key, (pitch, mode) in self.policies.items():
            requests, tokens = key if isinstance(key, tuple) else (key, 1)
            batch, meta = requests * tokens, dict(device="meta")
            projected = torch.empty((batch, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(batch, self.dim, dtype=torch.bfloat16, **meta)
            accepted = torch.empty(requests, dtype=torch.int32, **meta)
            shape = (requests, tokens) if tokens > 1 else (batch,)
            slots = torch.empty(shape, dtype=torch.int32, **meta)
            accepted = accepted if tokens > 1 else None
            operands = self._operands(caches, projected, slots, out, accepted)
            specs = tuple(map(layout, operands))
            options = (self.heads, mode, self.has_bias, self.device.index, tokens)
            self.kernels[key] = executable(specs, *options)
            if key in self.prepared:
                width = 4 * self.dim + 8
                prepared = torch.empty((batch, width), dtype=torch.float32, **meta)
                specs = (layout(prepared), *specs[1:], layout(self.readers[:requests]))
                self.prepared_kernels[key] = executable(specs, *options, True)
        return self

    def run(self, projected, out, *, caches, slots, accepted=None, prepared=None):
        """``accepted`` [>= B] (verify steps): ``slots`` is [>= B, q] contiguous;
        ``prepared``: the verify operands of ``KdaVerifyPrepared`` (its key must be in
        ``self.prepared``; ``projected``'s F_A rows armed, see the module docstring)."""
        batch = projected.shape[0]
        key, rows = batch, batch
        if accepted is not None:
            rows = batch // slots.shape[1]
            key, accepted = (rows, slots.shape[1]), accepted[:rows]
            if slots.stride() != (slots.shape[1], 1):  # e.g. a block-table slice
                slots = slots.contiguous()
        tensors = self._operands(caches, projected, slots[:rows], out, accepted)
        kernel = self.kernels[key]
        if prepared is not None:
            tensors = (prepared, *tensors[1:], self.readers[:rows])
            kernel = self.prepared_kernels[key]
        views = tuple(to_cute(t, 16) for t in tensors)
        kernel(*views[:13], *self.scalars, *views[13:], stream(self.device))
        return out


# Verify shapes above the tuned rows (``runtime.wide``) whose KDA core runs the stock
# KDA forward from the in-projection rows (seam S1) instead of KdaSplit, by q ("*":
# any).
# Set from in-graph A/Bs: generic KdaSplit at B32-64 sits 7.7-8.3x its byte floor
# (sol5), the stock crossover is being measured (kdarec); empty: KdaSplit everywhere.
STOCK_ROWS: dict = {"*": ""}


def verify_kernel(
    requests, tokens, capability, bf16_state=False, gate_mode=0, heads=None
):
    """The plain (not prepared) verify kernel ``executable`` builds for ``requests`` x
    ``tokens`` (coverage notes): "wide", "halves", "stream3" (``wide_kind``: by the
    state dtype) or "kdasplit". ``gate_mode`` as the KDA policies (TP8 verify: 1);
    ``heads`` per rank (default: TP8's 12 with gate mode 1, else TP16's 6; the SP
    decode passes its 24)."""
    if heads is None:
        heads = 12 if gate_mode else 6
    return WIDE_NAMES[
        wide_kind(requests, tokens, capability, bf16_state, gate_mode, heads)
    ]


def choose(shape, ctx):
    """The KDA core of a wide decode shape (the runtime's family hook): KdaSplit (or
    the wide verify kernels, ``wide_kind``, by the deployment's state dtype) at any
    requests and q (GENERIC), or the stock KDA core (STOCK, seam S1)."""
    if rows_policy(STOCK_ROWS, shape, default=0):
        return Choice(Path.STOCK, note="S1")
    capability = (10, 7) if ctx.arch == "rubin" else None
    gate_mode = int(ctx.tp == 8)  # the verify policies' gate mode (providers)
    kernel = verify_kernel(
        shape.requests, shape.q, capability, ctx.bf16_state, gate_mode
    )
    state = "bf16" if ctx.bf16_state else "fp32"
    return Choice(Path.GENERIC, note=f"{kernel} {shape.requests}x{shape.q} {state}")
