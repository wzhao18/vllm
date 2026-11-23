# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sequence-parallel (SP) specialized decode step: K3.1's C512 decode role (TP4 =
DCP4 = SP4 attention groups over DP4, EP16 MegaMoE, DSpark q4; ``runtime.sp_topology``).

Stock keeps the residual stream row-sharded over the TP group (``sp_shard``: rank r
owns rows [r n, (r + 1) n), n = R / 4) and runs each layer as AttnRes on the shard ->
all-gather -> attention on all R rows (24 of 96 heads per rank) -> o_proj ->
reduce-scatter -> AttnRes on the shard -> MLP on the shard. This step keeps that
layout; each layer boundary is two fused kernels (``attn_res.sp_fused``, family
``sp_boundary``): ``SpAttnResGather`` (the pre-attention AttnRes, its rows multicast
into every rank's all-gather box) and ``SpReduceAttnRes`` (the reduce-scatter owner
sum as the post-attention AttnRes delta), around one ``SpAttentionBlock``:

  stage            provider                                          family
  AttnRes + AG     ``SpAttnResGather`` (multicast, Lamport)          sp_boundary
  in-projection    ``DecodeProjection`` over the gathered rows        dsa_in / kda_in
  core             ``SpDsaCore`` (DSA) / ``SpKdaCore`` (KDA)          dsa / kda
  o_proj           ``SpOproj.produce`` (partial, or pushed by "warp") oproj_rs
  RS + AttnRes     ``SpReduceAttnRes`` (push, owner sum, re-arm)     sp_boundary

Each MegaMoE layer's MLP runs ``runtime.sp_mlp.SpMlp``: the stock layer's forward
step for step, its four dense BF16 GEMMs (latent down / up, shared gate_up / down) on
``gemm.sp_policy``'s measured tactics (family ``moe_dense``).

Everything else is the stock module, called as the stock forward calls it
(``Path.DECLARED``, ``shapes.DECLARED_STOCK``): the embedding, the MegaMoE router /
experts / norms and the dense MLP on the shard (with stock's ``sp_padding_mask``), the
aux streams, the final AttnRes and the norm. The final row all-gather is
``SpFinalGather`` (one multicast Lamport kernel; stock ``sp_all_gather``'s rows). The verification (synthetic
rejection), LM head, sampler and the preview (MLA) DSpark drafter stay stock too, except
the drafter's MLA attention: ``runtime.draft`` FAST_DCP's TP4 entry runs it on
cutedsl.draft.attention DraftDcp (publish-only MlaQuery, polled cores with length-keyed
splits, DcpValue), strict; no other preview drafter table has a TP4 entry
(SP_DRAFTER_TABLES, checked at build). The release (GQA) DSpark drafter is not a
declared stock module: its own build (``runtime.draft.GqaDraftFastPath``) is strict
here, every piece of every captured drafter batch specialized or the load refused.

R = 4 n for n = 1..128 requests per DP group (0.25..32 requests per GPU): every
captured R is a multiple of TP4, so SP never pads. Every captured R is served by every
block, or the build raises ``shapes.Unsupported`` naming the family, R and cause; a
FULL decode capture this step cannot plan raises too (``SpStep.plan``). Nothing on
this path falls back to the stock forward.

DSA-DP (``runtime.dsa_dp_topology``: TP4 at DCP1 over the same EP MegaMoE, "DSA DP,
GEMM TP"): every DSA layer is the stock owner-segment block, declared (family
``dsa_dp``): stock ``_pre_attn_norm`` on the shard -> ``ops.dsa_dp.run_segment`` (the
shard's rows pushed to their owner card, the layer with all 96 heads on the owner's
DEP_PERM segment and the owner's per-segment DSA metadata, the outputs pushed back) ->
stock ``_post_attn_norm``, then the layer's ``SpMlp``. KDA layers, MLPs, the drafter,
the embedding scatter and the final gather are the SP step's as above. The runner
keeps the DSA-DP DEP order (``DEP_PERM``, refreshed before every replay) and builds
the DSA metadata per segment; captured rows are whole requests per TP rank (q x TP).
Steps whose owners do not fit a captured graph run eager (``dep_eager``): the stock
forward, counted by ``StockSteps``.

Mailbox parities: ``SpComm`` alternates by SP block ordinal (DSA and KDA together),
the DSA DcpBox by DSA-layer ordinal. With an odd block count the last block of a step
and the first of the next share a parity; the final ``SpFinalGather`` (a TP
collective: every rank polls every rank's rows) separates them.

RecoverSSM: "deferred" records (``recoverssm.prepare_recoverssm``): the commit after
sampling only stages each request's accepted count, and the next step's KDA core folds
it into the checkpoint in its own pass (KdaWide, ``kda.sp.FOLDS``, the stock commit's
math bit for bit) or, before a core that does not fold, the step's first KDA block
flushes every layer's pending commit. ``KimiK3ModelState`` binds the runner-state side
(``SpecializedDecode.bind_model_state``).

NQ-TUNE: every family is GENERIC (no VR200 table at TP4 / DCP4 / 24 heads); see each
provider's module docstring for its sweep.
"""

import torch

from vllm import envs
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard

from ...dense_draft import is_gqa_dspark
from ...ops.attn_res import attn_res
from ..cutedsl.attn_res.sp_fused import SpBoundary
from ..cutedsl.common.sp_comm import SpComm, SpFinalGather
from ..cutedsl.dsa.sp import SpDsaCore, sp_dsa_box, sp_dsa_scratch
from ...mxfp8 import is_mxfp8_linear
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.gemm.mxfp8 import Mxfp8Projection
from ..cutedsl.gemm.sp_policy import sp_projection_tactics
from ..cutedsl.kda.sp import SpKdaCore
from ..cutedsl.oproj.sp import SpOproj
from ..recoverssm import prepare_recoverssm
from ..shapes import Choice, Coverage, Path, Shape, Unsupported, shard_rows
from .providers import attention_kind, bf16_weight, pack_dsa_projection
from .sp_dense import SpDenseMlp
from .sp_mlp import SpMlp, moe_choice

logger = init_logger(__name__)

# The attention kinds an SP block serves (K3 MLA layers are refused: sp_unsupported).
SP_KINDS = ("dsa", "kda")
# Hidden width of the residual stream (the SP mailboxes' row width).
WIDTH = 7168


def sp_layer(decoder, step):
    """Raises unless ``decoder`` is a stock layer in the SP layout ``step`` serves:
    its attention a DSA / KDA layer whose o_proj does not reduce (stock reduce-scatters
    after it), its MLP the stock MegaMoE (expert parallel, no token shard) or dense
    module, with a MegaMoE buffer for the step's largest shard."""
    index, attention = decoder.layer_idx, decoder.self_attn
    kind = attention_kind(attention)
    where = f"layer {index}"
    if kind not in SP_KINDS:
        raise Unsupported(kind, None, f"{where}: {kind} attention (DSA / KDA only)")
    if not getattr(decoder, "use_sequence_parallel", False):
        raise Unsupported("sp_step", None, f"{where}: not a sequence-parallel layer")
    if getattr(step, "dsa_dp", False) and kind == "dsa":
        # The stock DSA-DP layer (owner segment, push in / out): no o_proj reduce.
        if not getattr(decoder, "_dsa_scatter", False):
            raise Unsupported(
                "dsa_dp", None,
                f"{where}: not a DSA-DP scatter layer (ops.dsa_dp SCATTER_IN)",
            )  # fmt: skip
    elif attention.o_proj.reduce_results:
        raise Unsupported("oproj_rs", None, f"{where}: a reducing stock o_proj")
    elif getattr(attention, "gemm_rs_ar", None) is not None:
        raise Unsupported(
            "oproj_rs", None, f"{where}: a fused GEMM + reduce-scatter stock o_proj"
        )
    if not decoder.is_moe_layer:
        return
    mlp = decoder.mlp
    if not getattr(mlp, "use_mega_moe", False) or getattr(mlp, "moe_token_shard", 0):
        raise Unsupported("moe", None, f"{where}: not MegaMoE on the SP shard")
    tokens = getattr(mlp.experts, "max_num_tokens", 0)
    if tokens < step.shard_cap:
        raise Unsupported(
            "moe", Shape(step.rows[-1], step.q),
            f"{where}: MegaMoE max_num_tokens {tokens} < {step.shard_cap} shard rows",
        )  # fmt: skip


def sp_boundary(decoder, comm, block, rows, mlp=None, delta_parts=()):
    """The layer's fused SP boundaries (``attn_res.sp_fused.SpBoundary``): the stock
    ``_pre_attn_norm`` / ``_post_attn_norm`` weights, block pattern and epsilons. An
    MXFP8 in-projection (the NVFP4 checkpoint's KDA layers) takes the all-gather as
    MXFP8; an ``SpMlp`` with an MXFP8 shared expert (``mlp``) takes its gate_up input
    quantized by the post-AttnRes."""
    pre = (
        (
            decoder.self_attention_res_norm.weight,
            decoder.self_attention_res_proj.weight,
            decoder.input_layernorm.weight,
        ),
        (
            decoder.self_attention_res_norm.variance_epsilon,
            decoder.input_layernorm.variance_epsilon,
        ),
    )
    post = (
        (
            decoder.mlp_res_norm.weight,
            decoder.mlp_res_proj.weight,
            decoder.post_attention_layernorm.weight,
        ),
        (
            decoder.mlp_res_norm.variance_epsilon,
            decoder.post_attention_layernorm.variance_epsilon,
        ),
    )
    write = decoder.block_write_idx if decoder.is_block_write_layer else -1
    pushes = {r: block.oproj.pushes(r) for r in rows}
    n_max = rows[-1] // comm.tp
    first = decoder.layer_idx == 0
    gather_mx = isinstance(block.projection, Mxfp8Projection)
    out_mx = None if mlp is None else mlp.mx_input()
    return SpBoundary(
        comm, pre, post, decoder.prev_valid_blocks, write, n_max, pushes, first,
        gather_mx=gather_mx, out_mx=out_mx, delta_parts=delta_parts,
    )  # fmt: skip



def dense_mlps(layers, shards, device) -> dict:
    """The SP decode's dense (non-MoE) layers as SpDenseMlp for every captured shard
    row count ``shards``; above sp_dense.MAX_ROWS SpDenseMlp raises Unsupported at
    build (no stock dense module on a captured shape)."""
    return {
        d.layer_idx: SpDenseMlp(d.mlp, shards, device, d.layer_idx)
        for d in layers
        if not d.is_moe_layer
    }

class SpAttentionBlock:
    """One DSA / KDA layer's all-gather -> in-projection -> core -> o_proj with its
    reduce-scatter (module docstring): ``run(x_shard [n, 7168])`` returns the layer's
    attention output shard [n, 7168] (stock ``sp_reduce_scatter(o_proj(self_attn(
    sp_all_gather(x_shard))))``, summation order aside). ``ordinal``: the block's index
    among the step's SP blocks (its ``SpComm`` parity)."""

    def __init__(self, decoder, step: "SpStep", ordinal: int):
        attention, ctx, rows = decoder.self_attn, step.ctx, step.rows
        self.kind = kind = attention_kind(attention)
        self.index, self.parity = decoder.layer_idx, ordinal % 2
        where = f"layer {self.index}"
        family = f"{kind}_in"
        # BF16 projections, or the online-MXFP8 KDA projections of the NVFP4
        # (modelopt_mixed) checkpoint (``gemm.mxfp8``); any other quantized layer is
        # refused, not misread.
        mx_in = kind == "kda" and is_mxfp8_linear(attention.in_proj_qkvgfab)
        mx_out = kind == "kda" and is_mxfp8_linear(attention.o_proj)
        try:
            if kind == "dsa":
                weight = pack_dsa_projection(attention)
            elif not mx_in:
                weight = bf16_weight(attention.in_proj_qkvgfab)
            output = None if mx_out else bf16_weight(attention.o_proj)
        except ValueError as error:  # (re-raised with the layer)
            raise Unsupported(kind, None, f"{where}: {error}") from error
        with torch.accelerator.device_index(step.device.index):
            if mx_in:
                self.projection = Mxfp8Projection(
                    attention.in_proj_qkvgfab, rows, family=family, cap=rows[-1]
                )
                pitch = self.projection.n
            else:
                pitch, k = weight.shape
                tactics = sp_projection_tactics(family, pitch, k, rows, ctx)
                self.projection = DecodeProjection(
                    weight, rows, tactics, family=family
                )
            if mx_out:
                output = Mxfp8Projection(
                    attention.o_proj, rows, family="o_proj", cap=rows[-1]
                )
        if kind == "dsa":
            parity = step.dsa_layers % 2  # the DcpBox's own alternation
            step.dsa_layers += 1
            self.core = SpDsaCore(
                attention, rows, step.dsa_scratch, box=step.dsa_box, parity=parity,
                ctx=ctx,
            )  # fmt: skip
            self.slots = attention.mla_attn.mla_attn.layer_name  # (as run_dsa)
            self.index_prefix = attention.dsa_indexer.k_cache.prefix
        else:
            # The MXFP8 o_proj's input written by the KDA core itself where it can
            # (``SpKdaCore.mx_rows``), not by a standalone quantization of attn_out.
            self.core = SpKdaCore(
                attention, rows, records=step.records, ctx=ctx, mx_out=mx_out
            )
        self.mx_input = output.input if mx_out else None
        if self.core.pitch != pitch:
            raise Unsupported(
                kind, None,
                f"{where}: core pitch {self.core.pitch} != in-projection N {pitch}",
            )  # fmt: skip
        # heads x 128 (the o_proj input)
        self.width = output.k if isinstance(output, Mxfp8Projection) else output.shape[1]
        self.oproj = SpOproj(output, step.comm, rows, ctx)
        cap = rows[-1]
        self.projected = step.scratch(f"{kind}.projected", (cap, pitch))
        self.attn_out = step.scratch(f"attn_out.{self.width}", (cap, self.width))
        for parity in range(step.comm.parities):  # every tactic takes the operands
            self.projection.check_operands(step.comm.gather[parity], self.projected)

    def attend(self, gathered, caches, metadata, step):
        """In-projection + core over the all-gathered rows ``gathered`` [R, 7168]
        (``attn_res.sp_fused.SpAttnResGather`` filled them), then the o-projection
        half (``SpOproj.produce``): returns this rank's partial to push, or None when
        the o-projection pushed it."""
        rows = gathered.shape[0] if isinstance(gathered, torch.Tensor) else gathered.rows
        projected = self.projected[:rows]
        self.projection.run(gathered, projected)
        attn_out = self.core_output(rows)
        if self.kind == "dsa":
            indexer = get_forward_context().attn_metadata[self.index_prefix]
            self.core.run(
                projected, attn_out, cache=caches[0],
                slots=step.slot_mapping[self.slots],
                index_slots=indexer.slot_mapping, metadata=metadata, q=step.q,
            )  # fmt: skip
        else:
            self.core.run(projected, attn_out, caches=caches, metadata=metadata)
        return self.oproj.produce(attn_out, self.parity)

    def run(self, x_shard, caches, metadata, step):
        """``x_shard`` [n, 7168] BF16: the stock ``_pre_attn_norm`` output on this
        rank's shard; returns a new [n, 7168] BF16 shard (the next AttnRes may keep it
        as the prefix sum, so it is never a shared buffer)."""
        comm = self.comm
        n = x_shard.shape[0]
        rows = n * comm.tp
        gathered = comm.all_gather(x_shard, self.parity)
        projected = self.projected[:rows]
        self.projection.run(gathered, projected)
        attn_out = self.core_output(rows)
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
        shard = x_shard.new_empty((n, WIDTH))
        return self.oproj.run(attn_out, shard, self.parity)

    def core_output(self, rows):
        """The core's output at ``rows``: BF16 ``attn_out``, or the MXFP8 o_proj's
        input view where the KDA core writes it (``SpKdaCore.writes_mxfp8``)."""
        if self.kind == "kda" and self.core.writes_mxfp8(rows):
            return self.mx_input.view(rows)
        return self.attn_out[:rows]

    @property
    def comm(self) -> SpComm:
        return self.oproj.comm

    def choices(self, r):
        """This block's families at ``r`` rows (each raises if ``r`` is not
        prepared)."""
        return {
            f"{self.kind}_in": self.projection.choice(r),
            self.kind: self.core.choice(r),
            "oproj_rs": self.oproj.choice(r),
        }


# ``runtime.draft`` TP-keyed tables the SP decode specializes the preview (MLA) drafter
# with at TP4: its MLA attention (DraftDcp). Every other preview drafter piece is the
# declared stock module. (The release drafter's GQA_* tables are its own build's.)
SP_DRAFTER_TABLES = frozenset({"FAST_DCP"})
DRAFTER_NOTE = "stock DSpark module; its MLA attention on DraftDcp (strict)"


def check_stock_drafter(tp):
    """The preview DSpark drafter is the declared stock module but for its attention:
    raises if a ``runtime.draft`` TP-keyed table of the preview drafter other than
    SP_DRAFTER_TABLES would install a specialized drafter piece at ``tp`` (the
    declaration would then be false), or one of SP_DRAFTER_TABLES lacks its TP entry
    (the attention would stay stock)."""
    from . import draft

    tables = {
        name: table
        for name, table in vars(draft).items()
        if name.isupper()
        and not name.startswith("GQA_")
        and isinstance(table, dict)
        and table
        and all(isinstance(key, int) for key in table)
    }
    covered = sorted(
        name
        for name, table in tables.items()
        if tp in table and name not in SP_DRAFTER_TABLES
    )
    if covered:
        raise Unsupported(
            "drafter", None,
            f"runtime.draft tables {covered} have TP{tp} entries: the SP decode "
            f"declares the drafter stock but for {sorted(SP_DRAFTER_TABLES)}",
        )  # fmt: skip
    missing = sorted(n for n in SP_DRAFTER_TABLES if tp not in tables.get(n, {}))
    if missing:
        raise Unsupported(
            "drafter", None,
            f"runtime.draft tables {missing} have no TP{tp} entry: the SP decode "
            "runs the drafter's attention on DraftDcp",
        )  # fmt: skip
    return tuple(sorted(tables))


def sp_embedding(embedding, ctx):
    """(weight, start, vocab) of the vocabulary shard ``SpEmbedScatter`` gathers from
    (stock ``VocabParallelEmbedding``: rank r holds ids [r shard, (r + 1) shard) below
    ``org_vocab_size``), or None where its layout is not that one (the stock module
    then runs, declared)."""
    weight = getattr(embedding, "weight", None)
    shards = getattr(embedding, "shard_indices", None)
    if (
        weight is None
        or shards is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.shape[1] != WIDTH
        or not weight.is_contiguous()
        or getattr(embedding, "num_added_embeddings", 0)
        or shards.org_vocab_start_index != ctx.rank * weight.shape[0]
        or shards.padded_org_vocab_start_index != shards.org_vocab_start_index
        or shards.padded_org_vocab_end_index - shards.org_vocab_start_index
        != weight.shape[0]
    ):
        return None
    return weight.detach(), shards.org_vocab_start_index, embedding.org_vocab_size


class SpStep:
    """Builds the SP blocks on a ``SpecializedDecode`` (``runtime``, built with
    ``sequence_parallel``) and runs its captured steps (module docstring). All
    collective allocations (the SP mailboxes, the DSA DcpBox, the DSA layers' W_UV
    gathers) happen here, in layer order on every rank, before the KV cache is
    profiled."""

    # DSA-DP (``runtime.dsa_dp_topology``): the stock DSA-DP layers' indices.
    dsa_dp = False
    stock_dsa: frozenset = frozenset()

    def __init__(self, runtime):
        self.runtime = runtime
        self.ctx = ctx = runtime.context
        self.q, self.device = ctx.q, runtime.device
        backbone = runtime.backbone
        # DSA-DP (TP4 at DCP1): the DSA layers are the stock owner-segment block.
        self.dsa_dp = dsa_dp = bool(getattr(runtime, "dsa_dp", False))
        if not ctx.sequence_parallel or ctx.tp != ctx.dcp and not (
            dsa_dp and ctx.dcp == 1
        ):
            raise Unsupported(
                "sp_step", None,
                f"TP{ctx.tp} / DCP{ctx.dcp}, sequence parallel {ctx.sequence_parallel}"
                f", DSA-DP {dsa_dp}",
            )  # fmt: skip
        if dsa_dp:
            from ...ops import dsa_dp as dsa_dp_ops

            if not dsa_dp_ops.ENABLED:
                raise Unsupported(
                    "dsa_dp", None,
                    "the DSA-DP layout without ops.dsa_dp configured (the runner's "
                    "DEP order and per-segment DSA metadata)",
                )  # fmt: skip
        if not getattr(backbone, "use_sequence_parallel", False):
            raise Unsupported("sp_step", None, "the backbone is not sequence-parallel")
        if not getattr(backbone, "use_attn_res", False):
            raise Unsupported("attn_res", None, "the SP step runs AttnRes models only")
        self.rows = rows = tuple(sorted(set(runtime.captured)))
        if not rows:
            raise Unsupported("sp_step", None, "no captured FULL decode graphs")
        # DSA-DP: whole q-token requests per TP rank (ops.dsa_dp.dep_order).
        unit = self.q * ctx.tp if dsa_dp else self.q
        for r in rows:
            shape = Shape(r, self.q)
            if r % unit:
                raise Unsupported(
                    "sp_step", shape,
                    f"not whole q{self.q} requests"
                    + (f" per TP{ctx.tp} rank (DSA-DP)" if dsa_dp else ""),
                )  # fmt: skip
            shard_rows(shape, ctx)
        self.shard_cap = rows[-1] // ctx.tp
        if self.shard_cap >= 256:  # normalize: one stock RMSNorm launch per shard
            raise Unsupported(
                "sp_step", None, f"{self.shard_cap} shard rows (final norm < 256)"
            )
        self.layers = layers = runtime.layers
        for decoder in layers:
            sp_layer(decoder, self)
        # The stock DSA-DP layers (declared ``dsa_dp``); every other layer is an SP
        # block.
        self.stock_dsa = frozenset(
            d.layer_idx
            for d in layers
            if dsa_dp and attention_kind(d.self_attn) == "dsa"
        )
        if layers and layers[0].layer_idx in self.stock_dsa:
            raise Unsupported(
                "dsa_dp", None,
                "a DSA first layer (the embedding scatter feeds an SP block)",
            )  # fmt: skip
        for d in layers:
            if d.layer_idx in self.stock_dsa:
                # As the stock load (KimiLinearForCausalLM.load_weights without the
                # specialized decode): the fused DSA forward where it applies.
                d.self_attn.pack_projections()
        blocks = [d for d in layers if d.layer_idx not in self.stock_dsa]
        # The release (GQA) drafter's pieces are its own strict build's; the preview
        # (MLA) drafter is the declared stock module but for its attention.
        spec = get_current_vllm_config().speculative_config
        self.gqa_drafter = is_gqa_dspark(spec)
        self.draft_tables = () if self.gqa_drafter else check_stock_drafter(ctx.tp)
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        self.buffers: dict[str, torch.Tensor] = {}
        owner = runtime.group, self.device, ctx.tp, ctx.rank
        self.comm = SpComm(*owner, width=WIDTH, rows_cap=rows[-1])
        # [hidden | aux...] rows of the final all-gather (every aux layer is this
        # rank's: no pipeline stages on the SP path).
        packed = WIDTH * (1 + len(self.aux_layers))
        self.final_gather = SpFinalGather(*owner, width=packed, rows_cap=rows[-1])
        # Early aux columns. In the stock prefix-only capture an aux stream is
        # prefix_sum + the layer's MLP output: exactly the prefix row the next layer's
        # pre-AttnRes stores, which stays until that layer's reduce. A side stream
        # multicasts it into the final box meanwhile ({gather layer: packed column});
        # the other columns go with the final gather.
        self.early_aux: dict[int, int] = {}
        prefix_only = not (backbone._aux_attn_res_stream and backbone.use_attn_res)
        indices = {d.layer_idx for d in layers}
        for k, a in enumerate(self.aux_layers):
            if (
                prefix_only
                and a in indices
                and a != backbone.start_layer
                and a not in self.stock_dsa  # (its stock pre-AttnRes keeps no row)
            ):
                self.early_aux[a] = WIDTH * (1 + k)
        self.tail_cols = (0,) + tuple(
            WIDTH * (1 + k)
            for k, a in enumerate(self.aux_layers)
            if a not in self.early_aux
        )
        self.side = torch.cuda.Stream(self.device) if self.early_aux else None
        for r in rows:
            self.final_gather.prepare(r)
            if self.early_aux:
                widths = (WIDTH,) * len(self.tail_cols)
                self.final_gather.prepare(r, self.tail_cols, widths)
            for col in self.early_aux.values():
                self.final_gather.prepare_push(r, col, WIDTH)
        self.embedding = sp_embedding(backbone.embed_tokens, ctx)
        kinds = {attention_kind(d.self_attn) for d in blocks}
        self.dsa_box = self.dsa_scratch = None
        self.dsa_layers = 0
        if "dsa" in kinds:
            parallel = get_current_vllm_config().parallel_config
            self.dsa_box = sp_dsa_box(
                runtime.group, self.device, ctx.rank, heads=ctx.heads_per_rank,
                cap=rows[-1], interleave=parallel.cp_kv_cache_interleave_size,
            )  # fmt: skip
            self.dsa_scratch = sp_dsa_scratch(self.device, rows[-1])
        self.records = "none"
        if "kda" in kinds:
            # "deferred": the KDA cores fold the commit (KimiK3ModelState binds it).
            self.records = prepare_recoverssm(layers, ctx)
        # (ordinal over the SP blocks only: the SpComm users; the stock DSA-DP
        # layers exchange through their own Lamport buffers)
        self.blocks = {
            d.layer_idx: SpAttentionBlock(d, self, ordinal)
            for ordinal, d in enumerate(blocks)
        }
        self.dense = any(not d.is_moe_layer for d in layers)
        self.moe = any(d.is_moe_layer for d in layers)
        # The MegaMoE layers' dense BF16 GEMMs on this rank's shard rows (sp_mlp).
        shards = tuple(sorted({r // ctx.tp for r in rows}))
        self.mlps = {
            d.layer_idx: SpMlp(d.mlp, shards, ctx, d.layer_idx)
            for d in layers
            if d.is_moe_layer
        }
        # The layer-0 dense MLP (replicated weights) as one die-local weight stream,
        # for every captured shard size: SpDenseMlp raises Unsupported at build above
        # sp_dense.MAX_ROWS (never the stock dense module on a captured shape).
        self.dense_mlps = dense_mlps(layers, shards, self.device)
        # MoE layers whose latent up leaves FP32 split partials that the next layer's
        # all-gather boundary sums (``SpMlp.parts``): not the last layer (the final
        # AttnRes) nor one whose output is also a BF16 aux hidden state.
        self.parts_ok: dict[int, bool] = {}
        delta_parts: dict[int, tuple] = {}
        order = [d.layer_idx for d in layers]
        for index, nxt in zip(order, order[1:]):
            mlp = self.mlps.get(index)
            aux_bf16 = index + 1 in self.aux_layers and index + 1 not in self.early_aux
            # (a stock DSA-DP layer's pre-AttnRes takes BF16 rows)
            if mlp is not None and mlp.parts and not aux_bf16 and nxt in self.blocks:
                self.parts_ok[index] = True
                delta_parts[nxt] = tuple(sorted(set(mlp.parts.values())))
        # Each layer's AttnRes pair fused with the SP all-gather / reduce-scatter
        # (``attn_res.sp_fused``): two kernels per layer boundary instead of four.
        self.bounds = {
            d.layer_idx: sp_boundary(
                d, self.comm, self.blocks[d.layer_idx], rows,
                self.mlps.get(d.layer_idx), delta_parts.get(d.layer_idx, ()),
            )  # fmt: skip
            for d in blocks
        }
        self.table = self.plan_coverage()
        self.table.assert_complete(rows)
        if ctx.rank == 0:
            for line in self.table.lines():
                logger.info("Kimi K3 SP decode coverage %s", line)
        drafter = (
            "the release (GQA) DSpark drafter's pieces are its strict build's "
            "(runtime.draft GqaDraftFastPath)"
            if self.gqa_drafter
            else f"the DSpark drafter stays the stock module but for its attention "
            f"(TP{ctx.tp} entries only in FAST_DCP of {len(self.draft_tables)} "
            "drafter tables)"
        )
        logger.info_once(
            "Kimi K3 SP decode (TP%d = DCP%d over DP%d, EP%d, q%d, rows %d-%d, %d "
            "graphs): %d SP blocks (%s), %d stock DSA-DP layers, RecoverSSM %s; SP "
            "mailboxes %.0f MiB symmetric; %s",
            ctx.tp, ctx.dcp, ctx.dp, ctx.ep, self.q, rows[0], rows[-1], len(rows),
            len(self.blocks), "/".join(sorted(kinds)), len(self.stock_dsa),
            self.records,
            self.comm.nbytes() / 2**20, drafter,
            scope="global",
        )  # fmt: skip

    def scratch(self, name, shape):
        """A BF16 buffer shared by the blocks (they run one at a time)."""
        buffer = self.buffers.get(name)
        if buffer is None:
            buffer = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            self.buffers[name] = buffer
        elif tuple(buffer.shape) != tuple(shape):
            raise Unsupported(
                "sp_step", None, f"scratch {name} {tuple(buffer.shape)} vs {shape}"
            )
        return buffer

    def declared(self):
        """The stock modules the SP step keeps (``Path.DECLARED``), family -> note."""
        ctx = self.ctx
        spec = get_current_vllm_config().speculative_config
        method = getattr(spec, "rejection_sample_method", None)
        declared = {
            "attn_res": "the final AttnRes: stock attn_res on the shard",
            "lm_head": "stock",
            "sampler": "stock",
        }
        if self.embedding is None:
            declared["embed"] = "stock VocabParallelEmbedding"
        if not self.gqa_drafter:
            declared["drafter"] = DRAFTER_NOTE
        if self.moe:
            declared["moe"] = (
                f"stock MegaMoE router / experts / norms on the shard (EP{ctx.ep}); "
                "its dense GEMMs: moe_dense"
            )
        if self.dense and not self.dense_mlps:
            declared["dense"] = "stock dense MLP on the shard"
        if method == "synthetic":
            declared["verify"] = "stock (synthetic rejection)"
        if self.stock_dsa:
            declared["dsa_dp"] = (
                f"{len(self.stock_dsa)} DSA layers: stock DSA-DP owner segment "
                "(pre-AttnRes, dsa_dp.run_segment push in / all heads / push out, "
                "post-AttnRes); their MLPs: moe_dense"
            )
        return declared

    def plan_coverage(self):
        """Every captured R's families: every block's choice (each raises unless it
        serves R; the first block of each kind's is logged), and the declared stock
        modules. Without synthetic rejection ``verify`` is VerifyArgmax, which covers
        TP4 or raises when the model builds it."""
        coverage = Coverage(self.q)
        declared = self.declared()
        ctx = self.ctx
        for r in self.rows:
            choices = {"ag": self.comm.choice(r)}
            choices["final_gather"] = Choice(
                Path.GENERIC, "multicast", note="SpFinalGather (Lamport, one kernel)"
            )
            if self.embedding is not None:
                choices["embed"] = Choice(
                    Path.GENERIC, "scatter",
                    note="vocab-parallel gather pushed as the SP reduce-scatter "
                    "(SpEmbedScatter)",
                )  # fmt: skip
            choices["sp_boundary"] = Choice(
                Path.GENERIC, "cluster",
                note="AttnRes + SP AG / RS fused (attn_res.sp_fused)",
            )  # fmt: skip
            for block in self.blocks.values():
                for family, choice in block.choices(r).items():
                    choices.setdefault(family, choice)
            if self.mlps:
                choices["moe_dense"] = moe_choice(self.mlps, r // ctx.tp)
            for dense in self.dense_mlps.values():
                choices["dense"] = dense.choice(r // ctx.tp)
            if "verify" not in declared:
                choices["verify"] = Choice(Path.GENERIC, note="VerifyArgmax")
            coverage.specialize(r, choices)
            for family, note in declared.items():
                coverage.declare(r, family, note)
        return coverage

    def plan(self, positions, intermediate_tensors):
        """This step's plan (``decode.Step``), or None for a step outside a FULL
        decode capture (eager / profiling / warmup steps, PIECEWISE captures: they run
        the stock forward, compiled kernels never run there). A FULL capture this
        step cannot serve raises with the cause, never captures the stock forward."""
        from .decode import Step, decode_metadata

        b = positions.shape[0]
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            raise Unsupported("sp_step", b, "a capture without a forward context")
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        if intermediate_tensors is not None:
            raise Unsupported("sp_step", b, "pipeline-parallel stage inputs")
        by_layer = context.attn_metadata
        found = list(by_layer.values()) if isinstance(by_layer, dict) else []
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in found) else 1
        shape = Shape(b, q)
        if b not in self.rows:
            raise Unsupported(
                "sp_step", shape,
                f"a FULL decode capture outside the captured rows {self.rows[0]}-"
                f"{self.rows[-1]}",
            )  # fmt: skip
        if q != self.q:
            raise Unsupported(
                "sp_step",
                shape,
                f"a FULL decode capture that is not a q{self.q} verify",
            )
        layers = self.layers
        if self.stock_dsa:
            from ...ops import dsa_dp

            # FULL: the runner's FULL decode capture (its DEP order, run_segment).
            # NONE with this step's owned rows: a capture over the eager runner
            # (``vllm bench forward``), where the stock layer runs its owned-row
            # exchange (``_dsa_scatter_rows``) as the stock forward would.
            mode = dsa_dp.graph_mode()
            eager_rows = mode == CUDAGraphMode.NONE and dsa_dp.STEP.num_tokens >= 0
            if mode != CUDAGraphMode.FULL and not eager_rows:
                raise Unsupported(
                    "dsa_dp", shape,
                    f"a FULL capture whose DSA-DP mode is {mode} without the step's "
                    "DEP order or owned rows",
                )  # fmt: skip
        # The stock DSA-DP layers read the owner segment's metadata themselves.
        metadata = [
            None
            if d.layer_idx in self.stock_dsa
            else decode_metadata(d.self_attn, context, q, b)
            for d in layers
        ]
        missing = [
            d.layer_idx
            for d, md in zip(layers, metadata)
            if md is None and d.layer_idx not in self.stock_dsa
        ]
        if missing:
            raise Unsupported(
                "sp_step", shape,
                f"layers {missing[:4]} ({len(missing)}) lack uniform q{q} verify "
                "metadata",
            )  # fmt: skip
        return Step(q, metadata, context.slot_mapping, positions)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step: the stock
        ``KimiLinearModel.forward`` under SP with the SP blocks for its attention."""
        from .decode import caches_of

        runtime = self.runtime
        backbone = runtime.backbone
        x = inputs_embeds
        if x is None and self.embedding is None:
            x = backbone.embed_input_ids(input_ids)
        rows = (x if x is not None else input_ids).shape[0]
        if rows not in self.rows:
            raise Unsupported(
                "sp_step", Shape(rows, step.q), "not a captured row count"
            )
        # As the stock SP forward: MegaMoE reads the shard's padding rows from the
        # forward context when the stock deployment skips them.
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            context = get_forward_context()
            rows_like = x if x is not None else input_ids
            context.is_padding = sp_padding_mask(context.is_padding, rows_like)
        if x is None:
            # This rank's shard rows of the embedding, straight from the vocabulary
            # shards (SpEmbedScatter, the reduce-scatter parity of block 1: block 0
            # re-arms nothing it uses; module docstring of common.sp_comm).
            weight, start, vocab = self.embedding
            shard = backbone.embed_tokens.weight.new_empty((rows // self.ctx.tp, WIDTH))
            parity = 1 % self.comm.parities
            hidden = self.comm.embed_scatter(
                input_ids, weight, start, vocab, shard, parity
            )
        else:
            hidden = sp_shard(x)
        remote = backbone.collect_remote_aux_hidden_states(None)
        aux = [hidden] if backbone.start_layer in self.aux_layers else []
        residual = hidden.new_empty(
            (hidden.size(0), backbone.num_attn_res_blocks, hidden.size(1))
        )
        prefix_sum, hidden = hidden, None
        # The prefix sum is updated in place (the embedding shard first): the fused
        # boundaries store each updated / new prefix row into it.
        early = self.early_aux
        main = torch.cuda.current_stream(self.device)
        for layer, md in zip(self.layers, step.metadata):
            index = layer.layer_idx
            bound = self.bounds.get(index)
            if bound is None:  # a stock DSA-DP layer (declared ``dsa_dp``)
                hidden, prefix_sum, residual = self.run_dsa_dp(
                    layer, step, hidden, prefix_sum, residual
                )
            else:
                block = self.blocks[index]
                gathered = bound.gather(hidden, prefix_sum, residual, block.parity)
                col = early.get(index)
                if col is not None:  # prefix_sum holds the previous layer's aux row
                    self.side.wait_stream(main)
                    with torch.cuda.stream(self.side):
                        self.final_gather.push(prefix_sum, col)
                partial = block.attend(gathered, caches_of(layer.self_attn), md, step)
                if col is not None:  # the reduce updates prefix_sum in place
                    main.wait_stream(self.side)
                out = prefix_sum.new_empty(prefix_sum.shape)
                hidden = bound.reduce(prefix_sum, residual, out, block.parity, partial)
            prequantized = bound is not None and bound.out_mx is not None
            mlp = self.mlps.get(index)
            dense = self.dense_mlps.get(index)
            if dense is not None and dense.serves(hidden.shape[0]):
                hidden = dense.run(hidden)
            elif mlp is None:
                hidden = layer.mlp(hidden)
            else:  # (its MXFP8 shared gate_up input: written by bound.reduce)
                hidden = mlp.run(
                    hidden, prequantized=prequantized,
                    parts=self.parts_ok.get(index, False),
                )  # fmt: skip
            if index + 1 in self.aux_layers and index + 1 not in early:
                aux.append(
                    backbone._capture_aux_hidden_stream(
                        index, prefix_sum, hidden, residual
                    )
                )
        final = backbone.output_attn_res_norm
        hidden = attn_res(
            prefix_sum,
            hidden,
            residual,
            final.weight,
            backbone.output_attn_res_proj.weight.squeeze(0),
            None,
            num_blocks=backbone.num_attn_res_blocks,
            block_write_idx=-1,
            eps=final.variance_epsilon,
            output_norm_eps=0.0,
        )
        # The backbone's final norm on this rank's rows, before the gather: a shard
        # has < 256 rows, so the stock RMSNorm kernel reduces each row as
        # ``runtime.normalize`` does in its 255-row chunks (bit-equal rows, one
        # small kernel instead of two chunks and a cat over the gathered rows).
        hidden = runtime.normalize(hidden)
        # The final row all-gather on the multicast box (``SpFinalGather``; stock
        # ``sp_all_gather``'s rows, sanitized words).
        if early:
            width = hidden.shape[-1]
            sources = tuple(zip((hidden, *aux), self.tail_cols))
            packed = self.final_gather.run(None, cols=sources)[:rows]
            hidden, *aux = packed.split(width, dim=-1)
        elif aux:
            width = hidden.shape[-1]
            packed = self.final_gather.run(torch.cat([hidden, *aux], dim=-1))[:rows]
            hidden, *aux = packed.split(width, dim=-1)
        else:
            hidden = self.final_gather.run(hidden.contiguous())[:rows]
        aux = remote + aux
        return (hidden, aux) if aux else hidden

    def run_dsa_dp(self, layer, step, hidden, prefix_sum, residual):
        """A stock DSA-DP layer's attention half on this rank's shard, as the stock
        decoder's ``_dsa_scatter`` FULL-graph path runs it: the pre-AttnRes, the
        owner-segment exchange and attention (``ops.dsa_dp.run_segment``; under an
        eager-runner capture the layer's owned-row exchange), the post-AttnRes. Returns (the MLP input, the prefix sum, the saved blocks)."""
        from ...ops import dsa_dp

        x, prefix_sum, residual = layer._pre_attn_norm(hidden, residual, prefix_sum)
        if dsa_dp.graph_mode() == CUDAGraphMode.FULL:
            x = dsa_dp.run_segment(layer.self_attn, step.positions, x)
        else:  # (plan: an eager-runner capture with this step's owned rows)
            x = layer._dsa_scatter_rows(step.positions, x)
        return layer._post_attn_norm(x, residual, prefix_sum)

    def prepare_kv(self):
        """Compile every block's core for the bound KV cache (the profiling cache,
        then the real one), before any capture."""
        from .decode import caches_of

        for decoder in self.layers:
            if decoder.layer_idx in self.stock_dsa:
                continue
            block, caches = self.blocks[decoder.layer_idx], caches_of(decoder.self_attn)
            if block.kind == "dsa":
                block.core.prepare_cache(caches[0])
            else:
                block.core.prepare_caches(caches)

    @torch.no_grad()
    def process_weights_after_loading(self):
        """After a weight reload (the backbone refilled in place): re-derive the
        built blocks' copies, in layer order on every rank (the DSA "proj" W_UV
        gather is collective). The packed DSA in-projection is the stock linears'
        storage (``pack_dsa_projection``), so it is current already."""
        for decoder in self.layers:
            if decoder.layer_idx in self.stock_dsa:
                continue
            block = self.blocks[decoder.layer_idx]
            if block.kind == "kda":
                block.core.process_weights_after_loading()
                continue
            layer = decoder.self_attn.mla_attn.mla_attn
            gathered = getattr(layer, "W_UV_dcp_gathered", None)
            if gathered is not None:
                fresh = block.core.box.gather_heads(layer.W_UV)
                gathered[0].copy_(fresh)

    def coverage(self) -> Coverage:
        """The startup coverage table (R -> provider per family)."""
        return self.table
