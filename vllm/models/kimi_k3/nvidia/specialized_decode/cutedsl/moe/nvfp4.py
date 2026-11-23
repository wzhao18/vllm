# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE layers of the NVFP4 checkpoint (``modelopt_mixed``) on the TP (non-SP) decode:
MoE tensor parallel, no expert parallelism (TP16 = DCP16, BUG-BENCH-1).

That checkpoint's MoE layer differs from the K3 / MXFP4 one the tuned providers
(``SharedFront`` -> ``PersistentExperts`` -> ``PreparedTail``) were built for in two
places, both handled here with the stock layer's own kernels on its own weights:

- routed experts: NVFP4 W4A4 (E2M1 weights, E4M3 scales per 16 elements, per-expert
  global scales; ``ModelOptNvFp4FusedMoE`` on the FlashInfer TRT-LLM-Gen backend).
  ``Nvfp4Experts`` runs the layer's own experts kernel exactly as the stock runner
  does: its monolithic TRT-LLM-Gen entry (vLLM ``TrtLlmNvFp4ExpertsMonolithic`` ->
  FlashInfer's public ``trtllm_fp4_block_scale_moe``) on the router logits, so the
  routing (sigmoid + correction bias top 16 inside the kernel, BF16 weights), the
  activation quantization with the layer's input global scale, the folded alphas and
  the SiTU alpha / beta are the stock ones, bitwise. Where the layer defers its top-k
  finalize (``LatentMoERunner``'s tail fusion: TP8 / TP16, small batches) the output
  is the experts' ``UnfinalizedMoEOutput`` and the wide tail's LatentRows sums the
  top k (FP32, k order) as the stock tail op does.
- shared experts: online MXFP8 W8A8 (32-element E8M0 blocks). ``Mxfp8SharedExpert``
  is stock ``KimiMLP.forward`` at TP (``reduce_results=False``: this rank's partial):
  ``gemm.mxfp8.Mxfp8Projection`` gate_up (BF16 input quantized bitwise as FlashInfer's
  ``mxfp8_quantize``, stock's per-R MXFP8 kernel), ``KimiMLP.act_mxfp8_into`` (the
  stock SiTU-and-mul + MXFP8 quantize op writing the down projection's input), then
  the MXFP8 down projection.

``Nvfp4MoE`` is the layer's provider for every captured MoE batch: the router and
latent down projection (BF16: at most 64 rows ``Nvfp4Front``, the MXFP4 layers' tuned
front GEMM publishing logits and latent shards, gathered as the stock layer's
[M, 896] FP32 logits and [M, 3584] BF16 latent; above, the stock
``KimiMoE._maybe_overlap_router_and_down_proj``, as the wide rows' "stock front"), the NVFP4 routed experts' partial into the wide MoE tail (``tail_rows.PreparedTailRows``: two-shot latent all-reduce,
RMS norm, up projection, output ring the next AttnRes polls), and the MXFP8 shared
expert on the side stream with its reduce-scatter push. A layer this path cannot
reproduce raises ``Unsupported`` at build (``nvfp4_moe_cause``); nothing falls back to
the stock layer at run."""

import torch

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from ...shapes import Choice, Path, Unsupported

TOPK, LATENT, HIDDEN = 16, 3584, 7168


def nvfp4_experts_cause(routed):
    """Why ``routed`` (a MoE layer's ``RoutedExperts``) is not the stock NVFP4 W4A4
    experts ``Nvfp4Experts`` runs, or None."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    method = getattr(routed, "quant_method", None)
    if not isinstance(method, ModelOptNvFp4FusedMoE):
        return f"{type(method).__name__} is not the ModelOpt NVFP4 experts method"
    if method.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
        return f"NVFP4 backend {method.nvfp4_backend.value} (needs FLASHINFER_TRTLLM)"
    if method.use_a16:
        return "W4A16 NVFP4 experts (the decode runs W4A4)"
    if method.moe_quant_config is None or method.moe_kernel is None:
        return "the experts' kernel is not built (weights not processed)"
    if not method.uses_input_scales:
        return "no activation input scales"
    if not method.is_monolithic:
        return "modular experts (the stock layer routes outside the kernel)"
    config = routed.moe_config
    if config.moe_parallel_config.use_ep or routed.expert_map is not None:
        return "expert parallelism (the TP decode needs every expert local)"
    if routed.local_num_experts != routed.global_num_experts:
        return f"{routed.local_num_experts} of {routed.global_num_experts} experts"
    if routed.activation != MoEActivation.SITU:
        return f"activation {routed.activation} (Kimi K3: SiTU)"
    if config.moe_parallel_config.enable_eplb:
        return "EPLB"
    if routed.apply_router_weight_on_input:
        return "router weights applied on the input"
    if config.experts_per_token != TOPK:
        return f"top-{config.experts_per_token} (needs top-{TOPK})"
    if config.hidden_dim != LATENT or (config.hidden_dim_unpadded or LATENT) != LATENT:
        return f"hidden {config.hidden_dim} (latent {LATENT}, unpadded)"
    for name in ("w13_bias", "w2_bias"):
        if getattr(routed, name, None) is not None:
            return f"expert bias {name}"
    return None


class Nvfp4Experts:
    """The layer's stock NVFP4 experts on its router logits (module docstring).
    ``run`` returns this rank's routed partial: [M, 3584] BF16 (the top-k weighted sum
    over the rank's share of the intermediate dimension, before the TP reduction, the
    routed norm and the up projection), or, at the rows the layer defers, the experts'
    ``UnfinalizedMoEOutput``."""

    def __init__(self, routed):
        cause = nvfp4_experts_cause(routed)
        if cause is not None:
            where = getattr(routed, "layer_name", None) or type(routed).__name__
            raise Unsupported("moe", None, f"NVFP4 routed experts {where}: {cause}")
        self.routed = routed

    def deferred(self, rows):
        """Whether the experts leave the top-k finalize to the tail at ``rows``."""
        return self.routed.moe_config.should_defer_moe_finalize(rows)

    def run(self, latent, logits):
        """``latent`` [M, 3584] BF16 (the routed experts' input), ``logits`` [M, 896]
        FP32 (the router's)."""
        routed = self.routed
        return routed.quant_method.apply_monolithic(routed, latent, logits)


def mxfp8_shared_cause(shared):
    """Why ``shared`` (a MoE layer's ``KimiMLP``) is not the TP online-MXFP8 shared
    expert ``Mxfp8SharedExpert`` reproduces, or None."""
    from ....mxfp8 import is_mxfp8_linear

    if shared is None:
        return "no shared expert"
    linears = (shared.gate_up_proj, shared.down_proj)
    if not all(is_mxfp8_linear(linear) for linear in linears):
        return "not online MXFP8 (gate_up and down)"
    if getattr(shared, "shard_sequence_parallel", False):
        return "a sequence-parallel sharded shared expert"
    if getattr(shared, "gemm_rs_ar", None) is not None:
        return "a GEMM-RS/AR shared down projection"
    if shared.down_proj.reduce_results:
        return "a reducing down projection (the tail reduces the shared partial)"
    if not shared._act_mxfp8_ok():
        return "the activation cannot write the down projection's MXFP8 input"
    return None


class Mxfp8SharedExpert:
    """Stock ``KimiMLP.forward`` of a TP online-MXFP8 shared expert (module docstring)
    at the prepared ``rows``: ``run(x [R, 7168] BF16, out [R, 7168] BF16)`` writes this
    rank's unreduced partial."""

    def __init__(self, shared, rows, *, cap=None):
        from ..gemm.mxfp8 import Mxfp8Projection

        cause = mxfp8_shared_cause(shared)
        if cause is not None:
            where = getattr(shared.gate_up_proj, "prefix", "shared experts")
            raise Unsupported("moe", None, f"MXFP8 shared expert {where}: {cause}")
        self.shared, self.rows = shared, tuple(sorted(set(rows)))
        # Families of their own: their MXFP8 input scratch is shared by every MoE
        # layer's projection of that family (the SP decode's shared_gu / shared_down
        # take another K).
        self.gate_up = Mxfp8Projection(
            shared.gate_up_proj, self.rows, family="tp_shared_gu", cap=cap
        )
        self.down = Mxfp8Projection(
            shared.down_proj, self.rows, family="tp_shared_down", cap=cap
        )
        self.inter2, self.hidden = self.gate_up.n, self.down.n
        if self.down.k * 2 != self.inter2 or self.gate_up.k != self.hidden:
            raise Unsupported(
                "moe", None, f"MXFP8 shared expert gate_up [{self.inter2}, "
                f"{self.gate_up.k}] / down [{self.hidden}, {self.down.k}]",
            )  # fmt: skip
        cap = cap or self.rows[-1]
        device = shared.gate_up_proj.weight.device
        self.gate_up_out = torch.empty(
            (cap, self.inter2), dtype=torch.bfloat16, device=device
        )

    def run(self, x, out):
        n = x.shape[0]
        gate_up = self.gate_up.run(x, self.gate_up_out[:n])
        act = self.down.input.view(n)
        self.shared.act_mxfp8_into(gate_up, act)
        return self.down.run(act, out)

    def choice(self, rows):
        gu, dn = self.gate_up.choice(rows), self.down.choice(rows)
        return f"{gu.note} -> act_mxfp8 -> {dn.note}"


def nvfp4_moe_cause(module):
    """Why a MoE layer (stock ``KimiMoE``) is not the NVFP4 layer ``Nvfp4MoE`` runs,
    or None. None of its parts may be another format: a BF16 / MXFP4 layer keeps the
    tuned providers."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(module, "use_mega_moe", False):
        return "MegaMoE (expert parallel)"
    if getattr(module, "moe_token_shard", 0):
        return "a token-sharded MoE"
    if getattr(module, "_packed_moe", None) is not None:
        return "packed MoE projections"
    runner = getattr(module, "experts", None)
    routed = getattr(runner, "routed_experts", None)
    if routed is None:
        return "no routed experts"
    cause = nvfp4_experts_cause(routed)
    if cause is not None:
        return cause
    cause = mxfp8_shared_cause(module.shared_experts)
    if cause is not None:
        return f"shared expert: {cause}"
    transform = getattr(module, "routed_output_transform", None)
    down = getattr(module, "routed_expert_down_proj", None)
    if transform is None or transform.norm is None or down is None:
        return "no latent MoE projections with the routed norm"
    for linear in (down, transform.up_proj, module.gate):
        method = getattr(linear, "quant_method", None)
        if method is not None and not isinstance(method, UnquantizedLinearMethod):
            return f"quantized {type(method).__name__} latent / router linear"
        if linear.weight.dtype != torch.bfloat16:
            return f"{linear.weight.dtype} latent / router weight"
    return None


def is_nvfp4_moe(module):
    """A MoE layer with NVFP4 routed experts (the NVFP4 checkpoint), whether or not
    ``Nvfp4MoE`` can run it (it refuses the others at build)."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    routed = getattr(getattr(module, "experts", None), "routed_experts", None)
    return isinstance(getattr(routed, "quant_method", None), ModelOptNvFp4FusedMoE)


# Front GEMM rows: the tuned box holds 64 tokens (``tail_runtime.CAP``).
FRONT_MAX = 64


class Nvfp4Front:
    """The BF16 router + latent-down front of an NVFP4 MoE layer at <= 64 rows: the
    MXFP4 layers' front GEMM (``front.compile_publish``: this rank's [router shard;
    latent-down shard] rows, ``providers.pack_front``) multicasts every token's logits
    and latent columns into the model's front box; ``RouteSitu`` (polled, copy role +
    ``logits_only`` route role) gathers them into the stock layer's operands: the
    latent [M, 3584] BF16 (each value the GEMM's FP32 sum rounded once, as the stock
    ``routed_expert_down_proj``) and the FP32 logits [M, 896] (the stock router's
    output dtype). The stock experts then route from these logits exactly as from the
    stock router's: the only difference is the GEMMs' FP32 summation order (an
    order-only change; K split over the cluster instead of cuBLAS's)."""

    MAX = FRONT_MAX

    def __init__(self, module, packed, box, batches):
        import cutlass as c

        from ..common.host import to_cute
        from .front import compile_publish, compile_route, front_ks

        self.batches = frozenset(b for b in batches if b <= FRONT_MAX)
        if not self.batches:
            raise Unsupported("moe", None, "NVFP4 front: no batch of at most 64 rows")
        tp, rank = box.tp, box.rank
        valid, shard = 896 // tp, LATENT // tp
        n, device = packed.shape[0], packed.device
        if n != -(-valid // 16) * 16 + shard or packed.shape[1] != HIDDEN:
            shape = tuple(packed.shape)
            raise Unsupported("moe", None, f"NVFP4 front: packed rows {shape}")
        rows = box.buffer.shape[1]
        if max(self.batches) > rows:
            raise Unsupported("moe", None, f"NVFP4 front: {rows}-row front box")
        self.device, dev = device, device.index
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        wide = sms > 152  # the *_WIDE front tables (Rubin-class SM counts)
        self.packed = packed  # (kept alive: the CuTe view does not hold it)
        packed_view = to_cute(packed)
        cap, shared = max(self.batches), 6144 // tp
        f32, bf16 = torch.float32, torch.bfloat16

        def empty(shape, dtype=bf16):
            return torch.empty(shape, dtype=dtype, device=device)

        # the stock layer's operands (logits FP32, latent BF16), and the routing
        # kernel's unused route / SiTU outputs (``logits_only`` writes none of them)
        self.logits, self.latent = empty((cap, 896), f32), empty((cap, LATENT))
        unused = (empty((cap, TOPK), torch.int32), empty((cap, TOPK)))
        unused += (empty((cap, 2 * shared), f32), empty((cap, shared)))
        self.unused = unused
        bias = module.gate.e_score_correction_bias
        self.bias = bias  # (kept alive)
        box_view = to_cute(box.buffer)
        self.box = (box_view, c.Int64(box.mc))
        self.box += (c.Int32(rank * valid), c.Int32(rank * shard))
        scalars = tuple(
            map(c.Float32, (module.routed_scaling_factor, 1.0, 1.0))
        )  # (SiTU betas: no SiTU role)
        renorm = module.moe_renormalize
        self.plans = {}
        for b in sorted(self.batches):
            ks = front_ks(b, n, wide)
            front = compile_publish(n, n - shard, valid, b, dev, ks, wide=wide)
            route = compile_route(
                b, shared, renorm, (True, True, False), dev, True, False,
                logits_only=True,
            )  # fmt: skip
            logits, latent = self.logits[:b], self.latent[:b]
            ids, weights, gate, h = (t[:b] for t in unused)
            args = tuple(map(to_cute, (logits, bias, ids, weights)))
            args += (box_view, to_cute(latent), to_cute(gate), to_cute(h), *scalars)
            self.plans[b] = front, route, args, (latent, logits), packed_view

    def route(self, x):
        """``x`` [M, 7168] BF16 -> (latent [M, 3584] BF16, logits [M, 896] FP32),
        on the current stream."""
        from ..common.host import stream, to_cute

        b = x.shape[0]
        plan = self.plans.get(b)
        if plan is None:
            prepared = sorted(self.plans)
            raise Unsupported("moe", b, f"NVFP4 front: not prepared ({prepared})")
        front, route, args, outputs, packed = plan
        s = stream(self.device)
        front(to_cute(x), packed, *self.box, args[0], s)
        route(*args, s)
        return outputs


class Nvfp4MoE:
    """One NVFP4 MoE layer's provider for every captured MoE batch (module
    docstring). ``rows``: the layer's wide tail (``tail_rows.PreparedTailRows``) over
    the model's ``TailRowsBox``; ``state``: the model's ``MoEScratch`` (side stream and
    its events)."""

    def __init__(
        self, module, state, rows, batches, *, prefetch=None, layer_idx=None,
        front=None, tail=None,
    ):  # fmt: skip
        """``tail``: (``tail_runtime.MoeTailWeights`` with ``permuted``, the model's
        ``MoeTailBox``): its batches (<= 64 rows) run the front-mode MoeTail over the
        experts' deferred output into the box's output ring, instead of the wide
        tail; None: the wide tail at every batch."""
        cause = nvfp4_moe_cause(module)
        if cause is not None:
            raise Unsupported("moe", None, f"NVFP4 MoE layer: {cause}")
        self.module, self.state, self.rows = module, state, rows
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.batches = frozenset(batches)
        self.front = front  # ``Nvfp4Front`` (its batches), or None: the stock front
        self.experts = Nvfp4Experts(module.experts.routed_experts)
        # (the tail sums the top k itself where the experts defer it)
        late = [b for b in self.batches if self.experts.deferred(b)]
        missing = [b for b in late if b not in getattr(rows, "deferred", {})]
        if missing:
            raise Unsupported(
                "moe", missing[0], f"NVFP4 MoE: no deferred tail for rows {missing}"
            )
        self.tail, self.tail_box, self.tail_batches = None, None, frozenset()
        if tail is not None:
            self.tail, self.tail_box = tail
            self.tail_batches = frozenset(self.tail.codes)
            if not self.tail.permuted:
                raise Unsupported("moe", None, "NVFP4 MoE: the tail must be permuted")
            # (the MoeTail reads the permuted rows: the experts must defer there)
            deferred = self.experts.deferred
            eager = sorted(b for b in self.tail_batches if not deferred(b))
            if eager or not self.tail_batches <= self.batches:
                raise Unsupported(
                    "moe", (eager or [None])[0], f"NVFP4 MoE tail rows {eager}: the "
                    "experts finalize there (the MoeTail reads their deferred rows)",
                )  # fmt: skip
        cap = max(self.batches)
        self.shared = Mxfp8SharedExpert(module.shared_experts, self.batches, cap=cap)
        self.shared_out = torch.empty(
            (cap, HIDDEN), dtype=torch.bfloat16, device=state.device
        )

    def _shared(self, x):
        shared = self.shared_out[: x.shape[0]]
        self.shared.run(x, shared)
        self.rows.push(shared)

    def run(self, x, out, *, consume=None):
        """The layer on ``x`` [M, 7168]: its output lands in the wide tail's ring, which
        ``consume`` (the next AttnRes's ``run_ring``) reads."""
        from .prepared import PreparedMoE

        b, module = x.shape[0], self.module
        if b not in self.batches:
            prepared = sorted(self.batches)
            raise Unsupported("moe", b, f"NVFP4 MoE: not prepared ({prepared})")
        if consume is None:
            raise RuntimeError("NVFP4 MoE: the output must be consumed from its ring")
        if b in self.tail_batches:
            return self.run_tail(x, out, consume)
        # (the shared rows' RS push on the side stream; joined before NormRows)
        with PreparedMoE.beside(self, lambda: self._shared(x)) as join:
            if self.front is not None and b in self.front.batches:
                latent, logits = self.front.route(x)
            else:
                latent, logits, _ = module._maybe_overlap_router_and_down_proj(x)
            partial = self.experts.run(latent, logits)
            if self.prefetch is not None:  # (as PreparedMoE.finish: after the experts)
                self.prefetch.fork("next_attention", self.layer_idx, b)
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out

    def run_tail(self, x, out, consume):
        """A ``tail`` batch: front, the stock experts (deferred) and the shared expert
        (side stream, joined before the tail: the front-mode MoeTail reads its rows
        and pushes their shards itself), then the MoeTail into the box's ring."""
        from .prepared import PreparedMoE

        b, module = x.shape[0], self.module
        shared = self.shared_out[:b]
        with PreparedMoE.beside(self, lambda: self.shared.run(x, shared)):
            if self.front is not None and b in self.front.batches:
                latent, logits = self.front.route(x)
            else:
                latent, logits, _ = module._maybe_overlap_router_and_down_proj(x)
            partial = self.experts.run(latent, logits)
            if self.prefetch is not None:  # (as PreparedMoE.finish: after the experts)
                self.prefetch.fork("next_attention", self.layer_idx, b)
        if not isinstance(partial, UnfinalizedMoEOutput):
            raise RuntimeError(f"NVFP4 MoE B{b}: the experts finalized (not deferred)")
        weights = partial.expert_weights[:b]
        index = partial.expanded_idx_to_permuted_idx[:b]
        if weights.dtype != torch.bfloat16 or index.dtype != torch.int32:
            raise RuntimeError(
                f"NVFP4 MoE B{b}: routing weights {weights.dtype} / map {index.dtype}"
            )
        rows = partial.gemm2_permuted
        self.tail.run(self.tail_box, rows, weights, shared, index=index)
        if consume(out) is not out:
            raise RuntimeError("The MoE tail output must be consumed from its ring")
        return out

    def choice(self, rows):
        tuned = self.front is not None and rows in self.front.batches
        front = "tuned front (WsGemm + logits gather)" if tuned else "stock front"
        tail = "MoeTail (permuted)" if rows in self.tail_batches else "wide tail"
        key = ("tuned-front" if tuned else "stock-front",)
        key += ("moetail" if rows in self.tail_batches else "rows",)
        return Choice(
            Path.GENERIC,
            ("nvfp4", *key),
            f"{front}, NVFP4 TRT-LLM-Gen experts (stock monolithic), MXFP8 shared "
            f"({self.shared.choice(rows)}), {tail}",
        )
