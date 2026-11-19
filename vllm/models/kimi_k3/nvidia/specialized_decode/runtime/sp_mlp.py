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
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.gemm.mxfp8 import Mxfp8Projection
from ..cutedsl.gemm.sp_policy import sp_projection_tactics
from ..cutedsl.moe.moe_front import MoeFront, config_for
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
ROUTER_LATENT = {"*": "25-72:ltpin"}


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
        concat_rows = tuple(
            r for r in self.rows
            if rows_policy(ROUTER_LATENT, Shape(r, 1), default=None) == "ltpin"
        )  # fmt: skip
        self.router_latent = self.route_stage = None
        if (
            concat_rows
            and router_concat_cause(mlp) is None
            and route_stage_cause(mlp) is None
        ):
            from ..cutedsl.gemm.ltpin import LtPin

            packed = pack_router_latent(mlp)
            n, k = packed.shape
            self.router_latent = (
                packed,
                LtPin("router_latent", n, k, concat_rows, out32=True,
                      device=packed.device),
                mlp.gate.weight.shape[0],
            )  # fmt: skip
            self.route_stage = RouteStage(mlp, self.router_latent[1].plans)
        # The whole front (router | latent down GEMM + top-k + staging) as one
        # kernel (moe.moe_front) at the shard rows it has a launch shape for.
        self.front, self.front_rows = None, frozenset()
        front_rows = tuple(r for r in self.rows if config_for(r) is not None)
        if (
            front_rows
            and router_concat_cause(mlp) is None
            and route_stage_cause(mlp) is None
        ):
            self.front = MoeFront(mlp, front_rows, pack_router_latent(mlp))
            self.front_rows = frozenset(front_rows)
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
        with torch.accelerator.device_index(weights["latent_down"].device.index):
            if self.mxfp8:  # NVFP4 checkpoint: MXFP8 W8A8 shared expert (gemm.mxfp8)
                self.gemms["shared_gu"] = Mxfp8Projection(
                    shared.gate_up_proj, self.rows, family="shared_gu"
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
        self.latent = weights["latent_down"].shape[0]
        if self.mxfp8:
            self.inter2 = self.gemms["shared_gu"].n
            self.hidden = self.gemms["shared_down"].n
        else:
            self.inter2 = weights["shared_gu"].shape[0]
            self.hidden = weights["shared_down"].shape[0]
        self.threshold = _stock_threshold()
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
        if n in self.front_rows:
            out["latent_down"] = Choice(
                Path.TUNED, tactic=("moe_front", None), note="router|latent_down+top-k+stage"
            )
        elif concat is not None and n in concat[1].plans:
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
        if self.mxfp8:
            # stock's online-MXFP8 shared expert: SiTU-and-mul writes the down
            # projection's MXFP8 input (KimiMLP._act_mxfp8), bitwise act -> quantize
            down = gemms["shared_down"]
            act = down.input.view(n)
            shared.act_mxfp8_into(gate_up, act)
            return down.run(act, h.new_empty((n, self.hidden)))
        act = shared.act_fn(gate_up)
        return gemms["shared_down"].run(act, h.new_empty((n, self.hidden)))

    def run(self, h: torch.Tensor, prequantized: bool = False) -> torch.Tensor:
        """``prequantized``: the shared gate_up's MXFP8 input scratch already holds
        ``h`` quantized (``mx_input``; the post-AttnRes wrote it)."""
        if prequantized and not self.mxfp8:
            raise Unsupported("moe", h.shape[0], "an MXFP8 input for a BF16 shared")
        mlp, gemms = self.mlp, self.gemms
        n = h.shape[0]
        if n not in self.gemms["latent_down"].kinds or h.dim() != 2:
            raise Unsupported("moe", n, f"shard {tuple(h.shape)} not prepared")

        def down():
            return gemms["latent_down"].run(h, h.new_empty((n, self.latent)))

        fused_norm = self.fused_norm
        route, front = self.route_stage, self.front
        if n in self.front_rows and front.ready(h):
            # GEMM + top-k + staging in one kernel; staged before the shared expert
            # forks, as route_stage.
            staged = front.stage(h)

            def experts():
                return front.launch(staged, fused_norm)

        elif route is not None and route.ready(n):
            # Router | latent down as one FP32 GEMM, then one kernel stages its
            # top-k routes and the quantized latent into MegaMoE (route_stage).
            packed, gemm, _ = self.router_latent
            y = h.new_empty((n, packed.shape[0]), dtype=torch.float32)
            gemm.run(packed, h, y)
            # Staged before the shared expert forks to its stream: nothing
            # competes with the MegaMoE front.
            staged = route.stage(y)

            def experts():
                return route.launch(staged, fused_norm)

        else:
            (router_output, topk_ids), routed = maybe_execute_in_parallel(
                lambda: mlp._router(h),
                down,
                mlp._down_proj_events[0],
                mlp._down_proj_events[1],
                mlp._down_proj_stream if n <= self.threshold else None,
            )

            def experts():
                if fused_norm is None:
                    return mlp.experts(routed, router_output, topk_ids)
                return mlp.experts(
                    routed, router_output, topk_ids, output_rmsnorm=fused_norm
                )

        if mlp._shared_overlap:
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
        return gemms["latent_up"].run(final, residual=shared_out)


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
