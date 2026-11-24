# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MegaMoE layer's dense BF16 GEMMs on the SP shard (C512 decode role,
``runtime.sp_attention``): the stock ``KimiMoE.forward`` of the sequence-parallel
BF16 MegaMoE layer, step for step, with its four dense GEMMs on ``gemm.sp_policy``'s
measured tactics (``decode.DecodeProjection``) instead of the stock linears' cuBLAS
call:

  stock step                                       here
  router gate + grouped top-k (main stream)        the stock ``_router``
  routed latent down (aux stream, n <= 256)        latent_down  [3584, 7168]
  MegaMoE experts (+ shared on the side stream)    the stock ``experts``
  shared gate_up -> SiTU -> down                   shared_gu [12288, 7168] -> the stock
                                                   act -> shared_down [7168, 6144]
  latent RMSNorm, up with the shared beta-add      the stock norm -> latent_up
                                                   (``residual.addmm_``, torch kinds);
                                                   on the Rubin MegaMoE the norm runs in
                                                   its top-k reduce (stock rms_norm math)

Same streams and events as stock, the same operands, FP32 accumulation and one BF16
rounding per GEMM (the beta-add adds the BF16 shared output in FP32 before its one
rounding, as stock's cuBLAS epilogue); only the GEMM kernels (and so their FP32
summation order) differ. A layer this path cannot reproduce exactly (quantized or
packed MoE linears, the online-MXFP8 inputs, a token-sharded or TP-sharded shared
expert, a GEMM-RS/AR epilogue) raises ``Unsupported`` at build: never a stock
fallback for a planned row."""

import torch

from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

from ...mxfp8 import is_mxfp8_linear
from .. import prewait_gemm
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.gemm.mxfp8 import Mxfp8Projection
from ..cutedsl.gemm.sp_policy import sp_projection_tactics
from ..cutedsl.moe.route_stage import RouteStage, route_stage_cause
from ..shapes import Choice, Path, Shape, Unsupported, rows_policy
from .providers import bf16_weight

# Stock: the routed down projection runs on the aux stream up to this many tokens
# (``model._ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD``); read from the model module.
FAMILIES = ("latent_down", "shared_gu", "shared_down", "latent_up")
# On the MegaMoE side stream (``_shared_overlap``) these take ``sp_policy``'s
# "<family>_side" table (torch kinds: what fits beside the capped MegaMoE grid).
SHARED = ("shared_gu", "shared_down")
# Router gate | routed latent down as ONE GEMM over the load-time concatenated
# [896 + 3584, 7168] weight (the two stock linears become row views of it), FP32
# output: the router logits are the FP32 columns 0:896 exactly as stock's cuBLAS
# BF16 -> FP32 router GEMM, the latent the round-to-nearest BF16 of the FP32 columns
# 896: as stock's BF16 latent down (summation order aside). One kernel
# (``moe.route_stage``) then runs stock's grouped top-k on the logits and the
# MegaMoE stager on the latent, so the chain to MegaMoE is GEMM -> stage, not
# router GEMM -> split-K reduce -> top-k -> (latent down join) -> stage.
# Shard rows on the pinned GEMM (``ltpin``). In-step (D3-32 NVFP4 slice, step
# median vs the router || latent down pair): n32 / n64 -34..-39 / -52..-64 us per
# 24-layer step; n96 +28..+54 (the GEMM at 96 rows is no faster than the pair), so
# n96 keeps the stock pair:
# Rows with a split-K tactic (M tile, N tile, split, stages) run the concatenated GEMM
# on ``gemm.splitk.SplitKGemm`` (FP32 out; weights requested before the PDL wait, so
# they stream while the SP reduce runs; no separate split-K reduce launch) instead of
# the pinned cuBLASLt kernel. n32: 1 GPU 8.54 vs ltpin 9.22 us (flushed L2, after a
# producer); in-step D3-32 NVFP4 slice r8 (fastab ABBA x2, same node) 3967.5 ->
# 3902.8 us per 24-layer step (-65 us). n64 (64.64.2.6; 1 GPU 10.11 vs 9.74, but the
# weights stream during the reduce in-step): r16 4639.5 -> 4605.8 (-34), r8 with both
# -68.5 (another node), r24 unchanged. n96 (64.96.2.5; 1 GPU 11.58 us vs ~9.9 for the
# router || latent down pair, but in-step the concat also moves r24 onto route_stage):
# r24 5344.5 -> 5312.2 us per 24-layer step (-32 us, fastab ABBA x2 same node, rhea).
# Shard rows 1-24 (SP DCP4 decode at <= 24 requests per DP rank) take the n32
# tactic too (one 64-row M tile; TMA zero-fills the rows past n). Without it they
# kept the stock router || latent down pair, the stock top-k and the unstaged
# MegaMoE launch: DC (DCP4 x DP4 EP16) B24 per DP rank 17.53 ms vs B28 14.78 ms
# per step (CUPTI, per rank: MegaMoE GenPhase 9.33 vs 6.18 ms, nvjet router/latent
# GEMMs +1.2 ms, single_group_topk +0.37 ms).
# 65-80 rows: one 80-row token tile instead of two 64-row tiles (DDSA step-up bench r2,
# 1 GPU cold chain, us/call 64.64.2.6 / 64.80.2.5: R65 15.53 / 9.39, R72 15.98 / 9.39,
# R80 16.06 / 9.46; same split, same max error vs FP32).
# 33-40 rows: one 64-row token tile (64.64.2.6) instead of two 32-row tiles, which
# read the weights twice (CUPTI DC k7 B16 -> B17, n32 -> n34: +590 us of split-K per
# step; DC k3 B40 17.731 > B44 17.675 ms, rgst-a). Same split (2), so the same split
# summation order; the token tile only changes which CTA computes a row.
ROUTER_LATENT = {"*": "1-32:64.32.2.9,33-64:64.64.2.6,65-80:64.80.2.5,81-96:64.96.2.5"}
# Rows whose concatenated GEMM runs on ``prewait_gemm`` (weights streamed before the
# PDL wait) instead of the ROUTER_LATENT kernel. Same-node in-step A/B, D3-32 NVFP4
# slice, split-K front vs prewait (fastab xy1/xy2/x96, Rhea): n32 split-K faster
# (r8 +22 us/step on prewait), n64 prewait faster (r16 -19 us/step), n96 prewait
# faster (r24 -9.3 +- 6.6 us/step, 3 rounds).
# Rows between the tiles run the next tile (``prewait_gemm.tile``: 33-64 on the 64-token
# tile, 65-96 on the 96-token one; the TMA zero-fills the tile's rows past n, so each
# row's sums are the tile's bit for bit). Without it the shard rows just past 64 fell
# to the split-K front (CUPTI DC k7 B32 -> B34, 64 -> 68 shard rows, frozen 928a:
# wsg_k 17.7 -> SplitKGemm 64.80.2.5 19.8 us per call, +193 us per step).
PREWAIT_ROWS = ((33, 96),)


def prewait_rows(n: int) -> bool:
    return any(lo <= n <= hi for lo, hi in PREWAIT_ROWS)
# Rows whose latent up (stock's beta-add ``shared_out.addmm_(final, W^T)``) runs the
# token-wide 2-CTA GEMM (``gemm.wide``, ``partials``) with K split over the given
# number of CTA pairs and leaves FP32 split partials: the next layer's SP all-gather
# boundary forms the output as stock's cuBLAS beta-add does (bf16(bf16(partials
# summed in split order) + residual), ``attn_res.sp_fused.SplitDelta``), so there
# is no split exchange and no separate sum. Weights are requested before the PDL wait; the MegaMoE top-k reduce
# releases its dependents at entry (its completion follows the shared expert's
# stores). 1 GPU (cold weights, behind a 4 us predecessor, us per call, vs nvjet
# badd): n32 4.56 vs 7.79, n64 4.95 vs 8.96, n96 5.59 vs 7.28.
# n128 / 160 / 192 (q8 B64 / B80 / B96, one MMA of N = n tokens) at split 3 too
# (168 CTAs; cuBLAS badd ran there: DB k7 B64 latent_up:cublas). n224 has no legal
# split on 212 SMs: split 3 fails the uneven-region ring check (``wide._exchange_fits``),
# split 4 is 224 CTAs (> 212, ``PreparedWide`` raises); it keeps cuBLAS.
LATENT_UP_PARTS = {"*": "32:3,64:3,96:3,128:3,160:3,192:3"}


class SplitKFront:
    """The router | latent down GEMM on the split-K kernel for the rows of
    ``tactics`` ({rows: tactic}); ``LtPin``'s ``plans`` / ``run`` interface."""

    def __init__(self, packed, tactics):
        from ..cutedsl.gemm.projection import PreparedProjection

        self.plans = dict(tactics)
        self.proj = PreparedProjection(packed, tactics, output_dtype=torch.float32)
        self.proj.prepare()

    def run(self, w, x, out):
        return self.proj.run(x, out)


class FrontGemms:
    """Per-rows dispatch of the concatenated GEMM over ``LtPin`` / ``SplitKFront``."""

    def __init__(self, gemms):
        self.by_rows = {r: g for g in gemms for r in g.plans}
        self.plans = dict.fromkeys(self.by_rows)

    def run(self, w, x, out):
        return self.by_rows[x.shape[0]].run(w, x, out)


def _stock_threshold():
    from ...model import _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD

    return _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD


def sp_mlp_cause(mlp):
    """Why ``mlp`` (a stock KimiMoE) is not the BF16 SP MegaMoE layer this path
    reproduces, or None."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not getattr(mlp, "use_mega_moe", False) or getattr(mlp, "moe_token_shard", 0):
        return "not MegaMoE on the SP shard"
    if not getattr(mlp, "use_latent_moe", False):
        return "no latent MoE projections"
    if getattr(mlp, "_packed_moe", None) is not None:
        return "packed MXFP8 MoE projections"
    if mlp._shared_mxfp8_input():
        return "online-MXFP8 layer input"
    shared, transform = mlp.shared_experts, mlp.routed_output_transform
    if shared is None or transform is None:
        return "no shared expert / routed output transform"
    if getattr(shared, "shard_sequence_parallel", True):
        return "a TP-sharded shared expert"
    if getattr(shared, "gemm_rs_ar", None) is not None:
        return "a GEMM-RS/AR shared down projection"
    if transform._norm_mxfp8():
        return "online-MXFP8 latent up input"
    mx = shared_mxfp8(mlp)
    if mx is None:
        return "one of the shared expert's gate_up / down online MXFP8, not both"
    if not mx and shared._act_mxfp8_ok():
        return "online-MXFP8 shared down input"
    if mx and not shared._act_mxfp8_ok():
        return "MXFP8 shared expert whose activation cannot write the MXFP8 input"
    linears = (mlp.routed_expert_down_proj, transform.up_proj)
    if not mx:
        linears += (shared.gate_up_proj, shared.down_proj)
    for linear in linears:
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            return f"quantized {type(linear.quant_method).__name__} linear"
        if getattr(linear, "bias", None) is not None:
            return "a linear with bias"
    return None


def shared_mxfp8(mlp):
    """True when both shared-expert projections are online MXFP8 (the NVFP4
    checkpoint), False when neither is, None when mixed."""
    shared = mlp.shared_experts
    flags = {is_mxfp8_linear(shared.gate_up_proj), is_mxfp8_linear(shared.down_proj)}
    return flags.pop() if len(flags) == 1 else None


class SpMlp:
    """One MegaMoE layer's MLP on this rank's shard (module docstring):
    ``run(h [n, 7168])`` returns a new [n, 7168] BF16 tensor (stock
    ``mlp(h)``, the dense GEMMs' summation order aside). ``rows``: every captured
    shard row count n."""

    def __init__(self, mlp, rows, ctx, index):
        cause = sp_mlp_cause(mlp)
        if cause is not None:
            raise Unsupported("moe", None, f"layer {index}: {cause}")
        self.mlp, self.rows = mlp, tuple(sorted(set(rows)))
        shared, transform = mlp.shared_experts, mlp.routed_output_transform
        self.mxfp8 = bool(shared_mxfp8(mlp))
        # Router | latent down concatenation (before any GEMM reads the weights:
        # the stock linears become views of the packed weight).
        front = {r: rows_policy(ROUTER_LATENT, Shape(r, 1), default=None)
                 for r in self.rows}  # fmt: skip
        concat_rows = tuple(r for r, v in front.items() if v is not None)
        self.router_latent = self.route_stage = None
        self.prewait_router_latent = None
        if (
            concat_rows
            and router_concat_cause(mlp) is None
            and route_stage_cause(mlp) is None
        ):
            from ..cutedsl.gemm.ltpin import LtPin

            packed = pack_router_latent(mlp)
            n, k = packed.shape
            pinned = tuple(r for r in concat_rows if front[r] == "ltpin")
            split = {r: front[r] for r in concat_rows if isinstance(front[r], tuple)}
            gemms = []
            if pinned:
                gemms.append(LtPin("router_latent", n, k, pinned, out32=True,
                                   device=packed.device))  # fmt: skip
            if split:
                gemms.append(SplitKFront(packed, split))
            self.router_latent = (packed, FrontGemms(gemms), mlp.gate.weight.shape[0])
            self.route_stage = RouteStage(mlp, self.router_latent[1].plans)
            # Pre-wait weight-streaming GEMM at the token counts it serves (its own
            # tile-image copy: the stock linears stay views of the row-major concat).
            if any(
                prewait_rows(r) and prewait_gemm.supported(r, k) for r in concat_rows
            ):
                self.prewait_router_latent = prewait_gemm.pack_weight(packed)
        try:
            weights = {
                "latent_down": bf16_weight(mlp.routed_expert_down_proj),
                "latent_up": bf16_weight(transform.up_proj),
            }
            if not self.mxfp8:
                weights["shared_gu"] = bf16_weight(shared.gate_up_proj)
                weights["shared_down"] = bf16_weight(shared.down_proj)
        except ValueError as error:  # (re-raised with the layer)
            raise Unsupported("moe", None, f"layer {index}: {error}") from error
        self.gemms = {}
        side = bool(mlp._shared_overlap)  # the shared expert beside MegaMoE
        # Rubin MegaMoE + MXFP8 shared expert: the wide-EP decode bucket runs the
        # shared expert inside the GenPhase kernel (same MXFP8 operands: the shared
        # gate_up input scratch, the stock weights' bytes). Per-shape: rows above the
        # bucket keep the side-stream chain.
        fused_cap = None
        if self.mxfp8 and _rubin_megamoe(mlp.experts):
            from ...megamoe_rubin.experts import RUBIN_WIDE_EP_DECODE_TOKENS

            down = shared.down_proj
            if mlp.experts.enable_fused_shared(
                down.output_size_per_partition, down.input_size_per_partition
            ):
                fused_cap = max(self.rows[-1], RUBIN_WIDE_EP_DECODE_TOKENS)
        # One resident copy of the shared gate_up: the fused FC1 views the stock
        # weight striped in place (before any GEMM below binds it).
        self.striped = fused_cap is not None and stripe_shared_gate_up(mlp)
        with torch.accelerator.device_index(weights["latent_down"].device.index):
            if self.mxfp8:  # NVFP4 checkpoint: MXFP8 W8A8 shared expert (gemm.mxfp8)
                self.gemms["shared_gu"] = Mxfp8Projection(
                    shared.gate_up_proj, self.rows, family="shared_gu", cap=fused_cap
                )
                self.gemms["shared_down"] = Mxfp8Projection(
                    shared.down_proj, self.rows, family="shared_down"
                )
            for family, weight in weights.items():
                n, k = weight.shape
                table = f"{family}_side" if side and family in SHARED else family
                tactics = sp_projection_tactics(table, n, k, self.rows, ctx)
                self.gemms[family] = DecodeProjection(
                    weight, self.rows, tactics, family=table
                )
            parts = {r: rows_policy(LATENT_UP_PARTS, Shape(r, 1), default=None)
                     for r in self.rows}  # fmt: skip
            self.parts = {r: s for r, s in parts.items() if s is not None}
            self.latent_up_parts = None
            if self.parts:
                from ..cutedsl.gemm.wide import PreparedWide

                self.latent_up_parts = PreparedWide(
                    weights["latent_up"], self.parts, partials=True
                ).prepare()
        self.latent = weights["latent_down"].shape[0]
        if self.mxfp8:
            self.inter2 = self.gemms["shared_gu"].n
            self.hidden = self.gemms["shared_down"].n
        else:
            self.inter2 = weights["shared_gu"].shape[0]
            self.hidden = weights["shared_down"].shape[0]
        self.threshold = _stock_threshold()
        self.fused_shared = None
        if fused_cap is not None:
            from ...megamoe_rubin.session import FusedSharedArgs
            from ...megamoe_rubin.weights import prepare_mxfp8_shared_weights

            gu, dn = self.gemms["shared_gu"], self.gemms["shared_down"]
            fc1, fc2 = prepare_mxfp8_shared_weights(
                gu.weight, gu.scale, dn.weight, dn.scale,
                hidden=self.hidden, intermediate=self.inter2 // 2,
                striped=self.striped,
            )  # fmt: skip
            buf = gu.input.buf
            self.fused_shared = mlp.experts.warmup_fused_shared(
                FusedSharedArgs(buf.data, buf.scale, fc1, fc2), placed=self.striped
            )
        # Rubin MegaMoE: the latent RMSNorm folds into the kernel's top-k reduce (one
        # kernel and one [n, latent] round trip fewer per layer); compiled here, before
        # capture. Other MegaMoE backends keep the separate stock norm.
        self.fused_norm = None
        norm = transform.norm
        if norm is not None and _rubin_megamoe(mlp.experts):
            from ...megamoe_rubin.experts import prepare_output_rmsnorm

            prepare_output_rmsnorm(mlp.experts, norm)
            self.fused_norm = norm

    def choices(self, n):
        """The four GEMM families' choices at ``n`` shard rows."""
        out = {f: self.gemms[f].choice(n) for f in FAMILIES}
        concat = self.router_latent
        if concat is not None and n in concat[1].plans:
            out["latent_down"] = Choice(
                Path.TUNED, tactic=("ltpin", None), note="router|latent_down ltpin"
            )
        return out

    def mx_input(self):
        """The MXFP8 shared gate_up's input scratch (``gemm.mxfp8.Mxfp8Input``) a
        producer may fill (``run(prequantized=True)``), or None (BF16 shared)."""
        return self.gemms["shared_gu"].input if self.mxfp8 else None

    def _shared(self, h, prequantized=False):
        shared, gemms, n = self.mlp.shared_experts, self.gemms, h.shape[0]
        x = gemms["shared_gu"].input.view(n) if prequantized else h
        gate_up = gemms["shared_gu"].run(x, h.new_empty((n, self.inter2)))
        if self.striped:  # rows above the fused bucket: [gate | up] for the act
            from ...megamoe_rubin.weights import deinterleave_gate_up_cols_16

            gate_up = deinterleave_gate_up_cols_16(gate_up)
        if self.mxfp8:
            # stock's online-MXFP8 shared expert: SiTU-and-mul writes the down
            # projection's MXFP8 input (KimiMLP._act_mxfp8), bitwise act -> quantize
            down = gemms["shared_down"]
            act = down.input.view(n)
            shared.act_mxfp8_into(gate_up, act)
            return down.run(act, h.new_empty((n, self.hidden)))
        act = shared.act_fn(gate_up)
        return gemms["shared_down"].run(act, h.new_empty((n, self.hidden)))

    def run(self, h: torch.Tensor, prequantized: bool = False, parts: bool = False):
        """``prequantized``: the shared gate_up's MXFP8 input scratch already holds
        ``h`` quantized (``mx_input``; the post-AttnRes wrote it). ``parts``: the
        caller hands the output to an SP all-gather boundary compiled for this layer's
        latent up splits, so rows in ``self.parts`` return a ``SplitDelta``."""
        if prequantized and not self.mxfp8:
            raise Unsupported("moe", h.shape[0], "an MXFP8 input for a BF16 shared")
        mlp, gemms = self.mlp, self.gemms
        n = h.shape[0]
        if n not in self.gemms["latent_down"].kinds or h.dim() != 2:
            raise Unsupported("moe", n, f"shard {tuple(h.shape)} not prepared")

        def down():
            return gemms["latent_down"].run(h, h.new_empty((n, self.latent)))

        fused_norm = self.fused_norm
        route = self.route_stage
        if route is not None and route.ready(n):
            # Router | latent down as one FP32 GEMM, then one kernel stages its
            # top-k routes and the quantized latent into MegaMoE (route_stage).
            packed, gemm, _ = self.router_latent
            y = h.new_empty((n, packed.shape[0]), dtype=torch.float32)
            pw = self.prewait_router_latent
            if (
                pw is not None
                and prewait_rows(n)
                and prewait_gemm.supported(n, packed.shape[1])
            ):
                prewait_gemm.gemm(y, h, pw, packed.shape[0])
            else:
                gemm.run(packed, h, y)
            # Staged before the shared expert forks to its stream: nothing
            # competes with the MegaMoE front.
            staged = route.stage(y)

            def experts(shared=None):
                return route.launch(staged, fused_norm, shared)

        else:
            (router_output, topk_ids), routed = maybe_execute_in_parallel(
                lambda: mlp._router(h),
                down,
                mlp._down_proj_events[0],
                mlp._down_proj_events[1],
                mlp._down_proj_stream if n <= self.threshold else None,
            )

            def experts(shared=None):
                kwargs = {}
                if fused_norm is not None:  # backends without the fused norm take no kwarg
                    kwargs["output_rmsnorm"] = fused_norm
                if shared is not None:
                    kwargs["fused_shared"] = shared
                return mlp.experts(routed, router_output, topk_ids, **kwargs)

        if self.fused_shared is not None and mlp.experts.fused_shared(n):
            if not prequantized:
                gemms["shared_gu"].input.quantize(h)
            final, shared_out = experts(self.fused_shared)
        elif mlp._shared_overlap:
            final, shared_out = maybe_execute_in_parallel(
                experts,
                lambda: self._shared(h, prequantized),
                mlp._shared_events[0],
                mlp._shared_events[1],
                mlp._shared_stream,
            )
        else:
            final = experts()
            shared_out = self._shared(h, prequantized)
        norm = mlp.routed_output_transform.norm
        if norm is not None and fused_norm is None:
            final = norm(final)
        if parts and n in self.parts:
            from ..cutedsl.attn_res.sp_fused import SplitDelta

            split = self.parts[n]
            partials = final.new_empty((split * n, self.hidden), dtype=torch.float32)
            self.latent_up_parts.run(final.contiguous(), partials)
            return SplitDelta(partials, shared_out, split)
        return gemms["latent_up"].run(final, residual=shared_out)


def stripe_shared_gate_up(mlp) -> bool:
    """Before the fused shared expert is prepared and before any GEMM binds the
    weight: stripe the stock MXFP8 shared ``gate_up_proj`` in place (16-row gate/up
    stripes with their scales, ``weights.stripe_mxfp8_gate_up_``), home it per die
    where the fused bucket is die-local (the copy replaces the parameter's storage),
    and mark the stock ``KimiMLP`` so its own forwards de-stripe the GEMM output. The
    fused FC1 then views this one copy instead of a second striped copy (W4:
    2I x H FP8 per MoE layer). False (nothing changed: the fused FC1 keeps its own
    striped copy) when the weight is a view of a packed projection or the shared
    expert is not the stock ``KimiMLP``."""
    from ...model import KimiMLP
    from ...mxfp8 import _kernel_layout
    from ...megamoe_rubin.weights import stripe_mxfp8_gate_up_
    from ..cutedsl.gemm.mxfp8 import mxfp8_weight

    shared, experts = mlp.shared_experts, mlp.experts
    if not isinstance(shared, KimiMLP) or getattr(mlp, "_packed_moe", None) is not None:
        return False
    lin = shared.gate_up_proj
    if shared.gate_up_striped:  # (a rebuilt runtime: striped and homed already)
        return True
    weight, scale = mxfp8_weight(lin, "shared_gu")
    if not lin.weight_scale.data.is_contiguous():
        raise RuntimeError("shared gate_up scale is not contiguous (cannot stripe in place)")
    n, k = weight.shape
    stripe_mxfp8_gate_up_(weight, scale, hidden=k, intermediate=n // 2)
    homed = experts.home_fused_shared_rows(weight)
    if homed is not weight:
        lin.weight.data = homed.t() if _kernel_layout(lin) == "kn" else homed
    shared.gate_up_striped = True
    return True


def _rubin_megamoe(experts) -> bool:
    from ...megamoe_rubin.experts import KimiK3MegaMoEExperts

    return isinstance(experts, KimiK3MegaMoEExperts)


def router_concat_cause(mlp):
    """Why the router | latent down concatenation would not reproduce stock's router
    GEMM at the SP shard rows (> 16: stock's BF16 -> FP32 cuBLAS tier), or None."""
    from ..cutedsl.gemm.ltpin import shape_name

    gate = getattr(mlp, "gate", None)
    if gate is None:
        return "no router gate linear"
    weight = gate.weight
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return f"router weight {weight.dtype}"
    if getattr(gate, "bias", None) is not None:
        return "router bias"
    if gate.out_dtype != torch.float32 or not getattr(gate, "allow_cublas_router_gemm", False):
        return "router is not the BF16 -> FP32 cuBLAS GEMM"
    for flag in ("allow_fp32_router_gemm", "allow_ll_fp32w_gemm",
                 "allow_bf16x3_router_gemm", "allow_rocm_bf16x3_router_gemm"):  # fmt: skip
        if getattr(gate, flag, False):
            return f"router tier {flag}"
    down = mlp.routed_expert_down_proj.weight
    if down.dtype != torch.bfloat16 or down.shape[1] != weight.shape[1]:
        return "latent down weight"
    n, k = weight.shape[0] + down.shape[0], weight.shape[1]
    if shape_name(n, k, "bf16", out32=True) != "router_latent":
        return f"no pinned router | latent down GEMM for [{n}, {k}]"
    return None


def pack_router_latent(mlp):
    """One [router | latent down] BF16 weight; the stock router gate and routed down
    projection become row views of it (no second copy)."""
    packed = getattr(mlp, "packed_router_latent", None)
    if packed is not None:
        return packed
    linears = (mlp.gate, mlp.routed_expert_down_proj)
    packed = torch.cat([linear.weight.data for linear in linears])
    row = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        linear.weight.data = packed[row : row + rows]
        row += rows
    mlp.packed_router_latent = packed
    return packed


def moe_choice(mlps, n):
    """The ``moe_dense`` coverage note at ``n``: the first layer's four GEMM kinds."""
    first = next(iter(mlps.values()))
    choices = first.choices(n)
    kinds = {
        f: c.tactic[0] if isinstance(c.tactic, tuple) else c.note
        for f, c in choices.items()
    }
    paths = {c.path for c in choices.values()}
    path = min(paths, key=lambda p: p.value)  # "generic" before "tuned"
    note = " ".join(f"{f}:{k}" for f, k in kinds.items())
    return Choice(path, tuple(sorted(kinds.items())), note=note)
