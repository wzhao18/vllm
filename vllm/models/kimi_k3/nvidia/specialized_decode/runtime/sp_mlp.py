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
from ..shapes import Choice, Unsupported
from .providers import bf16_weight

# Stock: the routed down projection runs on the aux stream up to this many tokens
# (``model._ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD``); read from the model module.
FAMILIES = ("latent_down", "shared_gu", "shared_down", "latent_up")
# On the MegaMoE side stream (``_shared_overlap``) these take ``sp_policy``'s
# "<family>_side" table (torch kinds: what fits beside the capped MegaMoE grid).
SHARED = ("shared_gu", "shared_down")


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
            )  # fmt: skip
            buf = gu.input.buf
            self.fused_shared = mlp.experts.warmup_fused_shared(
                FusedSharedArgs(buf.data, buf.scale, fc1, fc2)
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
        return {f: self.gemms[f].choice(n) for f in FAMILIES}

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

        (router_output, topk_ids), routed = maybe_execute_in_parallel(
            lambda: mlp._router(h),
            down,
            mlp._down_proj_events[0],
            mlp._down_proj_events[1],
            mlp._down_proj_stream if n <= self.threshold else None,
        )
        fused_norm = self.fused_norm

        def experts():
            if fused_norm is None:
                return mlp.experts(routed, router_output, topk_ids)
            return mlp.experts(
                routed, router_output, topk_ids, output_rmsnorm=fused_norm
            )

        if self.fused_shared is not None and mlp.experts.fused_shared(n):
            if not prequantized:
                gemms["shared_gu"].input.quantize(h)
            final, shared_out = mlp.experts(
                routed, router_output, topk_ids, output_rmsnorm=fused_norm,
                fused_shared=self.fused_shared,
            )  # fmt: skip
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
        return gemms["latent_up"].run(final, residual=shared_out)


def _rubin_megamoe(experts) -> bool:
    from ...megamoe_rubin.experts import KimiK3MegaMoEExperts

    return isinstance(experts, KimiK3MegaMoEExperts)


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
