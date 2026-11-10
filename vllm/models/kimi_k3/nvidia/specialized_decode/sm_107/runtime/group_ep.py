# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The replicated EP_GROUP specialized decode step (layout (b)): expert-parallel
native MegaMoE over attention groups of G = TP 4 / 8 / 16 ranks (h = 96 / G = 24 /
12 / 6 heads per rank), DCP 1 or G, any DP, at most ``EP_GROUP_MAX_ROWS`` rows per
group (``runtime.ep_group_unsupported``). C512's TP4 = DCP4 role keeps the
row-sharded SP step (``runtime.sp_attention``, layout (c)); this step serves every
other EP_GROUP topology (``Topology.step_layout``).

Stock (``use_sequence_parallel``: every EP_GROUP topology over native MegaMoE) keeps
the residual stream row-sharded over the group (``sp_shard``: rank r owns rows
[r n, (r + 1) n), n = ceil(R / G), padded) and runs each layer as AttnRes on the
shard -> all-gather -> attention on all R rows (h heads per rank) -> o_proj partial
-> reduce-scatter -> AttnRes on the shard -> MoE on the shard. This step keeps every
group rank's residual stream replicated (all R rows) and runs each layer as

  stage              provider                                      family
  embedding          ``PreparedEmbedding`` (vocab-shard gather)     embed
                     + vLLM TP all-reduce                          DECLARED embed_reduce
  AttnRes (pre)      ``PreparedAttnRes`` (ClusterAttnRes, plain     attn_res
                     input) on the R rows
  in-projection      ``DecodeProjection`` (``sp_policy``)           dsa_in / kda_in
  core               ``SpDsaCore`` (DCP = G: DcpBox over the        dsa / kda
                     group, trtllm-gen inside the specialized
                     chain as C512's) / ``TpDsaCore`` (DCP1:
                     PreparedDsa; its trtllm-gen sparse MLA is
                     DECLARED sparse_mla) / ``SpKdaCore`` ("none":
                     the plain verify)
  o_proj partial     ``DecodeProjection`` [R, h x 128] x W_o^T      oproj
  all-reduce         vLLM ``sp_reduce_scatter`` +                 DECLARED oproj_reduce
                     ``sp_all_gather``
  AttnRes (post)     ``PreparedAttnRes.for_post`` on the R rows     attn_res
  MoE / dense MLP    stock SP module on stock's ``sp_shard`` of the DECLARED moe / dense
                     R rows (``sp_padding_mask``), then vLLM       DECLARED mlp_gather
                     ``sp_all_gather`` back to the R rows

then the final ``PreparedAttnRes`` (with the backbone's final norm, as the TP8 /
TP16 step's) on the R rows. Outside the step's forward the LM head, sampler,
verification under synthetic rejection and the DSpark drafter are declared stock as
in C512's SP step (the drafter's pieces follow ``runtime.draft``'s TP tables; at
DCP = G its attention is strict). The declared families are exactly
``GROUP_DECLARED`` (``sparse_mla`` only at DCP1), listed per captured R, and every
required boundary family must be present per R (``required_families``): a family
that cannot serve a captured R raises ``shapes.Unsupported`` at build, a FULL
capture outside the captured rows raises in ``plan``.

PENDING SIGN-OFF (orchestrator / user): ``moe`` declares the whole stock KimiMoE
module (FP32 router, BF16 shared expert, latent down / norm / up GEMMs and the routed
MegaMoE), as sm_100's hybrid step and C512's SP step run it. Design section 2.2 ports
the EP-front (sm_100 ``EpMoeLayer`` / ``pack_front``) so that only the routed
MegaMoE callable stays declared; that port is not done here. If the declared seam is
meant as the routed MegaMoE callable only, this mode needs the EP-front port before
it ships.

Precision is stock's at every stage: the embedding gather copies the shard rows and
the TP all-reduce is stock's; the o_proj partial sums K in FP32 and rounds once to
BF16 (stock's BF16 GEMM, another order), the reduction is stock's own
reduce-scatter (the same collective on the same partial rows: ``custom_reduce_scatter``
-> ``mnnvl_lamport_reduce_scatter``, FP32 rank-order sum rounded once; the critique's
P0-8) and the all-gather only copies; AttnRes is row-wise, so ClusterAttnRes on the
replicated rows computes stock's shard rows (the TP8 / TP16 step's kernel, stock's
roundings); the MoE is stock's module on stock's shard, the same rows, padding and
mask. The attention cores' precision is the SP step's (``dsa.sp`` / ``kda.sp``
docstrings; the 12-head KDA verify runs TP8's gate mode). The aux hidden streams are
the AttnRes prefix rows (stock's "prefix_only" mode); the AttnRes-stream aux mode
(VLLM_KIMI_K3_AUX_ATTN_RES_STREAM) is refused at load.

Collectives (P1-16): the only collective outside the TP group is MegaMoE's EP
dispatch / combine, inside the stock MoE module, called once per MoE layer per
step on every path: this step, the stock forward of an eager / profiling / dummy
step (``plan`` returns None outside FULL captures) and a DP replica's padded dummy
graph (the runner replays the captured graph of the padded size; padding requests
are stock's: NULL KDA slots, empty MLA requests, ``sp_padding_mask``). Inside the
group every predicate of ``plan`` is the same on every rank.

Weight reload (P1-17): the projections read the stock parameters (DSA: the packed
storage ``pack_dsa_projection`` rebinds the stock linears to), the cores re-derive
their copies (``process_weights_after_loading``).

NQ-TUNE: every family is GENERIC (no VR200 table at TP8 / TP16 under EP); the
replicated layout costs one all-gather more per layer than layout (c) (W8 measures
(b) vs (c) per R).
"""

import torch

from vllm import envs
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)

from ..cutedsl.attn_res.prepared import PreparedAttnRes
from ..cutedsl.common.embedding import PreparedEmbedding
from ..cutedsl.dsa.prepared import PreparedDsa, dsa_on_flashinfer
from ..cutedsl.dsa.sp import SP_KERNELS, SpDsaCore, sp_dsa_box, sp_dsa_scratch
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.gemm.sp_policy import GROUPS, sp_projection_tactics
from ..cutedsl.kda.sp import SpKdaCore
from ..recoverssm import prepare_recoverssm
from ..shapes import Choice, Coverage, Path, Shape, Unsupported
from . import EP_GROUP_MAX_ROWS
from .providers import attention_kind, bf16_weight, dsa_scratch, pack_dsa_projection

logger = init_logger(__name__)

# The attention kinds a group block serves (K3 MLA layers: refused at load).
GROUP_KINDS = ("dsa", "kda")
# Hidden width of the residual stream.
WIDTH = 7168
# The stock modules / collectives this step may keep (``Path.DECLARED``), checked
# per captured R by ``Coverage.assert_complete``; spelled out (not derived from
# ``shapes.DECLARED_STOCK``) so a new stock family never passes by inheritance.
# Inside the forward: the vLLM collectives (the embedding's TP all-reduce, the
# o_proj reduce-scatter + all-gather, the MLP shard all-gather), the stock MoE module
# (PENDING SIGN-OFF, module docstring), the stock dense MLP and, at DCP1 only, the
# trtllm-gen sparse MLA kernel (``GroupStep.allowed``). Outside it (as C512's SP
# step): the LM head, sampler, synthetic-rejection verify and the DSpark drafter.
# Per-layer and final AttnRes and the embedding gather are specialized: no
# ``attn_res`` / ``embed`` declaration is accepted.
GROUP_DECLARED = frozenset(
    {
        "embed_reduce",
        "oproj_reduce",
        "mlp_gather",
        "moe",
        "dense",
        "sparse_mla",
        "lm_head",
        "sampler",
        "verify",
        "drafter",
    }
)


def group_layer(decoder, step):
    """Raises unless ``decoder`` is a stock layer the replicated step serves: a DSA /
    KDA attention whose o_proj does not reduce and has no fused GEMM-RS / AR (stock
    reduce-scatters after it, ``sp_reduce_scatter``), in a sequence-parallel layer;
    its MLP the stock MegaMoE (expert parallel, no token shard, a replicated shared
    expert, a MegaMoE buffer for the step's largest shard) or a replicated dense
    MLP."""
    index, attention = decoder.layer_idx, decoder.self_attn
    kind = attention_kind(attention)
    where = f"layer {index}"
    if kind not in GROUP_KINDS:
        raise Unsupported(kind, None, f"{where}: {kind} attention (DSA / KDA only)")
    if not getattr(decoder, "use_sequence_parallel", False):
        raise Unsupported(
            "group_step", None,
            f"{where}: not a sequence-parallel layer (the MoE runs on stock's row "
            "shard)",
        )  # fmt: skip
    if attention.o_proj.reduce_results:
        raise Unsupported("oproj", None, f"{where}: a reducing stock o_proj")
    if getattr(attention, "gemm_rs_ar", None) is not None:
        raise Unsupported(
            "oproj", None,
            f"{where}: a fused GEMM-RS / AR o_proj (VLLM_ENABLE_GEMM_RS / "
            "VLLM_KIMI_K3_GEMM_AR): not the partial + reduce-scatter this step runs",
        )  # fmt: skip
    mlp = decoder.mlp
    if not decoder.is_moe_layer:
        if getattr(mlp, "shard_sequence_parallel", False):
            raise Unsupported(
                "dense", None,
                f"{where}: a TP-sharded dense MLP "
                "(VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT)",
            )  # fmt: skip
        if getattr(mlp, "gemm_rs_ar", None) is not None:
            raise Unsupported("dense", None, f"{where}: a fused GEMM-RS / AR dense MLP")
        return
    if not getattr(mlp, "use_mega_moe", False) or getattr(mlp, "moe_token_shard", 0):
        raise Unsupported("moe", None, f"{where}: not MegaMoE on the stock row shard")
    shared = getattr(mlp, "shared_experts", None)
    if getattr(shared, "shard_sequence_parallel", False):
        raise Unsupported(
            "moe", None,
            f"{where}: a TP-sharded shared expert "
            "(VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT): the declared MoE is stock's "
            "replicated BF16 shared expert on the shard",
        )  # fmt: skip
    tokens = getattr(mlp.experts, "max_num_tokens", 0)
    if tokens < step.shard_cap:
        raise Unsupported(
            "moe", Shape(step.rows[-1], step.q),
            f"{where}: MegaMoE max_num_tokens {tokens} < {step.shard_cap} shard rows",
        )  # fmt: skip


class TpDsaCore:
    """One K3.1 DSA layer at DCP1 in the replicated step: ``PreparedDsa`` over this
    rank's h heads and its full KV cache, as the TP8 DCP1 deployments run it (the
    fused decode indexer at stock K3 precision, the exact top-k, MlaQuery, the
    FlashInfer trtllm-gen sparse MLA kernel stock runs, called directly: declared
    ``sparse_mla``; MlaValue and the gate) -> [R, h x 128], the stock o_proj input.
    ``rows``: every captured R (whole q-token requests)."""

    def __init__(self, attention, rows, step):
        indexer, layer = attention.dsa_indexer, attention.mla_attn.mla_attn
        q = self.q = step.q
        if indexer.dcp_world_size != 1:
            raise Unsupported(
                "dsa", None, f"DCP{indexer.dcp_world_size} indexer on the DCP1 core"
            )
        if not dsa_on_flashinfer(attention):
            raise Unsupported(
                "dsa", None,
                f"the {type(layer.impl).__name__} backend (FLASHINFER_MLA_SPARSE "
                "trtllm-gen only; FlashMLA sparse: W6)",
            )  # fmt: skip
        if getattr(layer.impl, "use_prims_ts", False):
            raise Unsupported("dsa", None, "the prims_ts sparse MLA kernel")
        self.rows = rows = tuple(sorted(set(rows)))
        for r in rows:
            if r % q:
                raise Unsupported("dsa", Shape(r, q), f"not whole q{q} requests")
        self.heads = layer.W_UK_T.shape[0]
        self.pitch = pitch = pack_dsa_projection(attention).shape[0]
        scratch = dsa_scratch(step, max(rows))
        try:
            self.prepared = PreparedDsa(
                attention,
                dict.fromkeys(rows, pitch),
                scratch,
                queries=(q,),
                kernels=SP_KERNELS,
            )
        except ValueError as error:  # (re-raised with the family: never stock)
            raise Unsupported("dsa", None, str(error)) from error
        for r in rows:
            core = self.prepared.cores.get((r, q))
            if core != SP_KERNELS[0]:
                raise Unsupported("dsa", Shape(r, q), f"resolved sparse core {core}")

    def prepare_cache(self, kv):
        self.prepared.prepare_cache(kv)
        return self

    def run(self, projected, out, *, cache, slots, index_slots, metadata, q):
        rows = projected.shape[0]
        if q != self.q or rows not in self.rows:
            raise Unsupported("dsa", Shape(rows, q), "not a prepared row count")
        return self.prepared.run(
            projected, out, cache=cache, slots=slots, index_slots=index_slots,
            metadata=metadata, q=q,
        )  # fmt: skip

    def choice(self, rows):
        if rows not in self.rows:
            raise Unsupported("dsa", Shape(rows, self.q), "not a prepared row count")
        note = self.prepared.coverage_note(rows, self.q)
        return Choice(
            Path.GENERIC, tactic=("dcp1", SP_KERNELS[0]),
            note=f"dcp1 {self.heads}h {note}",
        )  # fmt: skip


class GroupAttentionBlock:
    """One DSA / KDA layer on the replicated rows (module docstring):
    ``run(x [R, 7168])`` returns the layer's attention output on all R rows, stock
    ``sp_all_gather(sp_reduce_scatter(o_proj(self_attn(x))))[:R]`` (summation order
    aside), a fresh tensor (the next AttnRes may keep it as the prefix sum)."""

    def __init__(self, decoder, step: "GroupStep"):
        attention, ctx, rows = decoder.self_attn, step.ctx, step.rows
        self.kind = kind = attention_kind(attention)
        self.index = decoder.layer_idx
        where = f"layer {self.index}"
        try:  # BF16 projections only (a quantized layer is refused, not misread)
            if kind == "dsa":
                weight = pack_dsa_projection(attention)
            else:
                weight = bf16_weight(attention.in_proj_qkvgfab)
            output = bf16_weight(attention.o_proj)
        except ValueError as error:  # (re-raised with the layer)
            raise Unsupported(kind, None, f"{where}: {error}") from error
        pitch, k = weight.shape
        self.width = output.shape[1]  # heads x 128 (the o_proj input)
        if ctx.heads_per_rank and self.width != ctx.heads_per_rank * 128:
            raise Unsupported(
                "oproj", None,
                f"{where}: o_proj K {self.width} != {ctx.heads_per_rank} heads x 128",
            )  # fmt: skip
        group = (ctx.tp, ctx.dcp)
        family = f"{kind}_in"
        with torch.accelerator.device_index(step.device.index):
            tactics = sp_projection_tactics(family, pitch, k, rows, ctx)
            out_tactics = sp_projection_tactics("o_proj", WIDTH, self.width, rows, ctx)
        self.projection = DecodeProjection(
            weight, rows, tactics, family=family, group=group
        )
        self.oproj = DecodeProjection(
            output, rows, out_tactics, family="oproj", group=group
        )
        if kind == "dsa":
            self.core = step.dsa_core(attention)
            self.slots = attention.mla_attn.mla_attn.layer_name  # (as run_dsa)
            self.index_prefix = attention.dsa_indexer.k_cache.prefix
        else:
            self.core = SpKdaCore(attention, rows, records=step.records, ctx=ctx)
        if self.core.pitch != pitch:
            raise Unsupported(
                kind, None,
                f"{where}: core pitch {self.core.pitch} != in-projection N {pitch}",
            )  # fmt: skip
        cap = rows[-1]
        self.projected = step.scratch(f"{kind}.projected", (cap, pitch))
        self.attn_out = step.scratch(f"attn_out.{self.width}", (cap, self.width))
        self.partial = step.scratch("oproj.partial", (cap, WIDTH))
        # (every tactic takes the operands: the AttnRes rows, the scratch rows)
        normed = step.scratch("normed", (cap, WIDTH))
        self.projection.check_operands(normed, self.projected)
        self.oproj.check_operands(self.attn_out, self.partial)

    def run(self, x, caches, metadata, step):
        """``x`` [R, 7168] BF16: the stock ``_pre_attn_norm`` output on all R rows."""
        rows = x.shape[0]
        projected = self.projected[:rows]
        self.projection.run(x, projected)
        attn_out = self.attn_out[:rows]
        if self.kind == "dsa":
            # The packed indexer's write slots come from its own metadata
            # (``providers.Attention.run_dsa``).
            indexer = get_forward_context().attn_metadata[self.index_prefix]
            self.core.run(
                projected, attn_out, cache=caches[0],
                slots=step.slot_mapping[self.slots],
                index_slots=indexer.slot_mapping, metadata=metadata, q=step.q,
            )  # fmt: skip
        else:
            self.core.run(projected, attn_out, caches=caches, metadata=metadata)
        partial = self.partial[:rows]
        self.oproj.run(attn_out, partial)
        # Stock's reduce-scatter of the partial (its padding to whole shards, its
        # collective), then the all-gather back to every rank's R rows.
        return sp_all_gather(sp_reduce_scatter(partial))[:rows]

    def choices(self, r):
        """This block's families at ``r`` rows (each raises if ``r`` is not
        prepared)."""
        return {
            f"{self.kind}_in": self.projection.choice(r),
            self.kind: self.core.choice(r),
            "oproj": self.oproj.choice(r),
        }


class GroupStep:
    """Builds the replicated EP_GROUP blocks on a ``SpecializedDecode`` (``runtime``,
    built with ``Context.ep_group`` and no sequence parallelism) and runs its
    captured steps (module docstring). All collective allocations (the DSA DcpBox at
    DCP = G, the DSA layers' W_UV gathers) happen here, in layer order on every
    rank, before the KV cache is profiled."""

    # The family ``plan`` / ``scratch`` name in ``Unsupported`` (``dep.DepStep``:
    # "dep_step").
    family = "group_step"

    def __init__(self, runtime):
        self.runtime = runtime
        self.ctx = ctx = runtime.context
        self.q, self.device = ctx.q, runtime.device
        backbone = runtime.backbone
        if not ctx.ep_group or ctx.sequence_parallel:
            raise Unsupported(
                "group_step", None,
                f"ep_group {ctx.ep_group}, sequence parallel {ctx.sequence_parallel} "
                "(the replicated EP_GROUP step only)",
            )  # fmt: skip
        if ctx.tp not in GROUPS or ctx.dcp not in (1, ctx.tp):
            raise Unsupported(
                "group_step", None,
                f"TP{ctx.tp} / DCP{ctx.dcp} (groups of {GROUPS}, DCP 1 or TP)",
            )  # fmt: skip
        if not getattr(backbone, "use_sequence_parallel", False):
            raise Unsupported(
                "group_step", None,
                "the backbone is not sequence-parallel (stock's MoE row shard)",
            )  # fmt: skip
        if not getattr(backbone, "use_attn_res", False):
            raise Unsupported(
                "attn_res", None, "the group step runs AttnRes models only"
            )
        self.rows = rows = tuple(sorted(set(runtime.captured)))
        if not rows:
            raise Unsupported("group_step", None, "no captured FULL decode graphs")
        for r in rows:
            shape = Shape(r, self.q)
            if r % self.q:
                raise Unsupported("group_step", shape, f"not whole q{self.q} requests")
            if r > EP_GROUP_MAX_ROWS:
                raise Unsupported(
                    "group_step", shape,
                    f"more than {EP_GROUP_MAX_ROWS} rows per attention group (the "
                    "wide rows need W4)",
                )  # fmt: skip
        # Stock's shard of the largest step: ceil(R / G) rows (sp_shard pads).
        self.shard_cap = -(-rows[-1] // ctx.tp)
        self.layers = layers = runtime.layers
        for decoder in layers:
            group_layer(decoder, self)
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        self.buffers: dict[str, torch.Tensor] = {}
        kinds = {attention_kind(d.self_attn) for d in layers}
        self.dsa_box = self.dsa_scratch = None
        self.dsa_layers = 0
        if "dsa" in kinds and ctx.dcp > 1:
            parallel = get_current_vllm_config().parallel_config
            self.dsa_box = sp_dsa_box(
                runtime.group, self.device, ctx.rank, heads=ctx.heads_per_rank,
                cap=rows[-1], interleave=parallel.cp_kv_cache_interleave_size,
            )  # fmt: skip
            self.dsa_scratch = sp_dsa_scratch(self.device, rows[-1])
        self.records = "none"
        if "kda" in kinds:
            # The load gate refuses ReplaySSM / RecoverSSM here (W7); a layer that
            # still carries records would be misread by the plain core: raise.
            records = prepare_recoverssm(layers, ctx, defer=False)
            if records != "none":
                raise Unsupported(
                    "kda", None,
                    f"RecoverSSM records ({records}) in the replicated EP_GROUP step "
                    "(W7)",
                )  # fmt: skip
        self.blocks = {d.layer_idx: GroupAttentionBlock(d, self) for d in layers}
        self.dense = any(not d.is_moe_layer for d in layers)
        self.moe = any(d.is_moe_layer for d in layers)
        self.kinds = kinds
        self.build_attn_res(backbone, rows[-1])
        try:  # the vocab-shard gather (+ stock's TP all-reduce), as TP8 / TP16
            self.embedding = PreparedEmbedding(backbone.embed_tokens, ctx.tp, ctx.rank)
        except ValueError as error:
            raise Unsupported("embed", None, str(error)) from error
        # sparse_mla is a declared seam at DCP1 only (at DCP = G the trtllm-gen call
        # is inside the specialized DCP chain, as C512's SP step).
        self.allowed = GROUP_DECLARED - ({"sparse_mla"} if ctx.dcp > 1 else set())
        self.table = self.plan_coverage()
        self.table.assert_complete(rows, self.allowed)
        if ctx.rank == 0:
            for line in self.table.lines():
                logger.info("Kimi K3 EP_GROUP decode coverage %s", line)
        logger.info_once(
            "Kimi K3 EP_GROUP decode (replicated layout: TP%d DCP%d over DP%d, EP%d, "
            "%d heads per rank, q%d, rows %d-%d, %d graphs): %d blocks (%s), the MoE "
            "on stock's row shard (<= %d rows per rank)",
            ctx.tp, ctx.dcp, ctx.dp, ctx.ep, ctx.heads_per_rank, self.q, rows[0],
            rows[-1], len(rows), len(self.blocks), "/".join(sorted(kinds)),
            self.shard_cap, scope="global",
        )  # fmt: skip

    def build_attn_res(self, backbone, cap):
        """Every AttnRes site on ``PreparedAttnRes`` (plain input; the TP8 / TP16
        step's ClusterAttnRes): before each layer (the first without a delta: the
        embedding rows start the prefix), after each attention (``for_post``: a
        block-write layer takes the attention output as the new prefix) and the final
        one with the backbone's final norm. The sites entering an aux layer (or the
        final one) also store the updated prefix rows: stock's "prefix_only" aux
        stream."""
        layers = self.layers
        if self.aux_layers and getattr(backbone, "_aux_attn_res_stream", False):
            raise Unsupported(
                "attn_res", None,
                "the AttnRes-stream aux hidden states "
                "(VLLM_KIMI_K3_AUX_ATTN_RES_STREAM): the specialized AttnRes stores "
                "the prefix rows only",
            )  # fmt: skip
        try:
            self.before = {
                d.layer_idx: PreparedAttnRes.for_target(d, has_delta=d is not layers[0])
                for d in layers
            }
            self.post = {d.layer_idx: PreparedAttnRes.for_post(d) for d in layers}
            self.final = PreparedAttnRes.for_target(backbone, final=True)
        except ValueError as error:  # (re-raised with the family: never stock)
            raise Unsupported("attn_res", None, str(error)) from error
        end = layers[-1].layer_idx + 1
        for index in self.aux_layers:
            if index == end:
                self.final.bind_aux()
            elif index in self.before:
                self.before[index].bind_aux()
            else:
                raise Unsupported(
                    "attn_res", None, f"aux layer {index} outside this stage"
                )
        nb = backbone.num_attn_res_blocks
        empty = dict(dtype=torch.bfloat16, device=self.device)
        self.prefix = torch.empty((cap, WIDTH), **empty)
        self.res_blocks = torch.empty((cap, nb, WIDTH), **empty)
        self.moe_in = self.scratch("moe_in", (cap, WIDTH))

    def scratch(self, name, shape):
        """A BF16 buffer shared by the blocks (they run one at a time)."""
        buffer = self.buffers.get(name)
        if buffer is None:
            buffer = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            self.buffers[name] = buffer
        elif tuple(buffer.shape) != tuple(shape):
            raise Unsupported(
                self.family, None, f"scratch {name} {tuple(buffer.shape)} vs {shape}"
            )
        return buffer

    def dsa_core(self, attention):
        """The DSA core of the next DSA layer (layer order): SpDsaCore on the
        group's DcpBox at DCP = G (its parity alternates by DSA ordinal), TpDsaCore
        at DCP1."""
        if self.ctx.dcp == 1:
            return TpDsaCore(attention, self.rows, self)
        parity = self.dsa_layers % 2  # the DcpBox's own alternation
        self.dsa_layers += 1
        return SpDsaCore(
            attention, self.rows, self.dsa_scratch, box=self.dsa_box, parity=parity,
            ctx=self.ctx,
        )  # fmt: skip

    def declared(self):
        """The stock modules and collectives the step keeps (``Path.DECLARED``),
        family -> note."""
        ctx = self.ctx
        config = get_current_vllm_config()
        method = getattr(config.speculative_config, "rejection_sample_method", None)
        backend = config.kernel_config.moe_backend
        declared = {
            "embed_reduce": f"vLLM TP{ctx.tp} all-reduce of the embedding gather "
            "(stock VocabParallelEmbedding's collective)",
            "oproj_reduce": "vLLM sp_reduce_scatter of the o_proj partial + "
            "sp_all_gather (stock's reduce-scatter precision; the gather copies)",
            "mlp_gather": "vLLM sp_shard of the R rows (sp_padding_mask) + "
            "sp_all_gather of the MLP output back to the R rows",
            "lm_head": "stock (outside the step's forward)",
            "sampler": "stock (outside the step's forward)",
            "drafter": f"stock DSpark module; runtime.draft's TP{ctx.tp} tables "
            "specialize its pieces (own coverage log; attention strict at DCP = TP)",
        }
        if "dsa" in self.kinds and ctx.dcp == 1:
            declared["sparse_mla"] = (
                "FlashInfer trtllm-gen sparse MLA decode at DCP1: the kernel stock "
                "runs, called by the DSA core"
            )
        if self.moe:
            declared["moe"] = (
                f"stock KimiMoE module ({backend}: FP32 router, BF16 shared expert, "
                f"latent down / norm / up, routed MegaMoE EP{ctx.ep}) on stock's "
                "sp_shard (PENDING SIGN-OFF: EP-front port, module docstring)"
            )
        if self.dense:
            declared["dense"] = "stock dense MLP on stock's sp_shard"
        if method == "synthetic":
            declared["verify"] = "stock (synthetic rejection)"
        return declared

    def plan_coverage(self):
        """Every captured R's families: the layout, every block's choice (each
        raises unless it serves R; the first block of each kind's is logged) and the
        declared stock modules. Without synthetic rejection ``verify`` is
        VerifyArgmax, which covers TP4 / TP8 / TP16 or raises when the model builds
        it."""
        coverage = Coverage(self.q)
        declared = self.declared()
        required = self.required_families()
        for r in self.rows:
            shard = -(-r // self.ctx.tp)
            note = f"layout (b): replicated rows, MoE on a {shard}-row shard"
            choices = {"layout": Choice(Path.GENERIC, "replicated", note=note)}
            choices["embed"] = Choice(
                Path.GENERIC, "gather", note=f"PreparedEmbedding TP{self.ctx.tp}"
            )
            choices["attn_res"] = Choice(
                Path.GENERIC, "plain", note="ClusterAttnRes pre / post / final"
            )
            for block in self.blocks.values():
                for family, choice in block.choices(r).items():
                    choices.setdefault(family, choice)
            if "verify" not in declared:
                choices["verify"] = Choice(Path.GENERIC, note="VerifyArgmax")
            coverage.specialize(r, choices)
            for family, note in declared.items():
                coverage.declare(r, family, note)
            planned = coverage.plan[r] or {}
            missing = sorted(required - set(planned))
            if missing:
                raise Unsupported(
                    "group_step", Shape(r, self.q),
                    f"boundaries without a provider: {', '.join(missing)}",
                )  # fmt: skip
        return coverage

    def required_families(self):
        """The boundary families every captured R must plan (a specialized choice
        or a declared seam): the embedding and its reduction, AttnRes, each attention
        kind's in-projection / core / o_proj, the o_proj reduction, the MLPs and
        their shard gather, the DCP1 sparse MLA and the verification."""
        required = {"layout", "embed", "embed_reduce", "attn_res", "oproj_reduce"}
        required |= {"mlp_gather", "verify"}
        for kind in self.kinds:
            required |= {f"{kind}_in", kind, "oproj"}
        if "dsa" in self.kinds and self.ctx.dcp == 1:
            required.add("sparse_mla")
        if self.moe:
            required.add("moe")
        if self.dense:
            required.add("dense")
        return frozenset(required)

    def plan(self, positions, intermediate_tensors):
        """This step's plan (``decode.Step``), or None for a step outside a FULL
        decode capture (eager / profiling / warmup steps, PIECEWISE captures: they run
        the stock forward, which calls the same MegaMoE collective per MoE layer). A
        FULL capture this step cannot serve raises with the cause, never captures the
        stock forward."""
        from .decode import Step, decode_metadata

        b, family = positions.shape[0], self.family
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            raise Unsupported(family, b, "a capture without a forward context")
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        if intermediate_tensors is not None:
            raise Unsupported(family, b, "pipeline-parallel stage inputs")
        by_layer = context.attn_metadata
        found = list(by_layer.values()) if isinstance(by_layer, dict) else []
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in found) else 1
        shape = Shape(b, q)
        if b not in self.rows:
            raise Unsupported(
                family, shape,
                f"a FULL decode capture outside the captured rows {self.rows[0]}-"
                f"{self.rows[-1]}",
            )  # fmt: skip
        if q != self.q:
            raise Unsupported(
                family, shape, f"a FULL decode capture that is not a q{self.q} "
                "verify",
            )  # fmt: skip
        layers = self.layers
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in layers]
        missing = [d.layer_idx for d, md in zip(layers, metadata) if md is None]
        if missing:
            raise Unsupported(
                family, shape,
                f"layers {missing[:4]} ({len(missing)}) lack uniform q{q} verify "
                "metadata",
            )  # fmt: skip
        return Step(q, metadata, context.slot_mapping, positions)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step: the
        stock ``KimiLinearModel.forward`` on replicated rows with the specialized
        embedding gather, AttnRes sites and group blocks, and stock's MoE / dense MLP
        on stock's row shard. The final AttnRes applies the backbone's final norm
        (as the TP8 / TP16 step's); the aux streams stay pre-norm."""
        from .decode import caches_of

        backbone = self.runtime.backbone
        x = inputs_embeds
        if x is None:
            ids = input_ids
            x = torch.empty(
                (ids.shape[0], WIDTH), dtype=torch.bfloat16, device=ids.device
            )
            self.embedding.run(ids, x)
        rows = x.shape[0]
        if rows not in self.rows:
            raise Unsupported(
                "group_step", Shape(rows, step.q), "not a captured row count"
            )
        # As the stock SP forward: MegaMoE reads the shard's padding rows from the
        # forward context when the stock deployment skips them.
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            context = get_forward_context()
            context.is_padding = sp_padding_mask(context.is_padding, x)
        remote = backbone.collect_remote_aux_hidden_states(None)
        aux_out = [torch.empty_like(x) for _ in self.aux_layers]
        aux = dict(zip(self.aux_layers, aux_out))
        prefix, blocks = self.prefix[:rows], self.res_blocks[:rows]
        normed = self.scratch("normed", (self.rows[-1], WIDTH))[:rows]
        moe_in = self.moe_in[:rows]
        hidden, index = x, self.layers[0].layer_idx - 1
        for layer, md in zip(self.layers, step.metadata):
            index = layer.layer_idx
            # _pre_attn_norm: the delta is the previous MLP output (the first layer:
            # the embedding rows start the prefix).
            self.before[index].run(hidden, prefix, blocks, normed, aux.get(index))
            caches = caches_of(layer.self_attn)
            attended = self.blocks[index].run(normed, caches, md, step)
            # _post_attn_norm on the gathered attention output.
            self.post[index].run(attended, prefix, blocks, moe_in)
            # The stock SP MLP on stock's row shard, gathered back to the R rows.
            hidden = sp_all_gather(layer.mlp(sp_shard(moe_in)))[:rows]
        out = torch.empty_like(x)
        self.final.run(hidden, prefix, blocks, out, aux.get(index + 1))
        streams = remote + aux_out
        return (out, streams) if streams else out

    def prepare_kv(self):
        """Compile every block's core for the bound KV cache (the profiling cache,
        then the real one), before any capture."""
        from .decode import caches_of

        for decoder in self.layers:
            block, caches = self.blocks[decoder.layer_idx], caches_of(decoder.self_attn)
            if block.kind == "dsa":
                block.core.prepare_cache(caches[0])
            else:
                block.core.prepare_caches(caches)

    @torch.no_grad()
    def process_weights_after_loading(self):
        """After a weight reload (the backbone refilled in place): re-derive the
        built blocks' copies, in layer order on every rank (the DSA "proj" W_UV
        gather at DCP = G is collective). The projections read the stock
        parameters (the packed DSA in-projection is the stock linears' storage)."""
        for decoder in self.layers:
            block = self.blocks[decoder.layer_idx]
            if block.kind == "kda":
                block.core.process_weights_after_loading()
                continue
            layer = decoder.self_attn.mla_attn.mla_attn
            gathered = getattr(layer, "W_UV_dcp_gathered", None)
            box = getattr(block.core, "box", None)
            if gathered is not None and box is not None:
                fresh = box.gather_heads(layer.W_UV)
                gathered[0].copy_(fresh)

    def coverage(self) -> Coverage:
        """The startup coverage table (R -> provider per family)."""
        return self.table
