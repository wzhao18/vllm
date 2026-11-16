# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sequence-parallel (SP) specialized decode step: K3.1's C512 decode role (TP4 =
DCP4 = SP4 attention groups over DP4, EP16 MegaMoE, DSpark q4; ``runtime.sp_topology``).

Stock keeps the residual stream row-sharded over the TP group (``sp_shard``: rank r
owns rows [r n, (r + 1) n), n = R / 4) and runs each layer as all-gather -> attention
on all R rows (24 of 96 heads per rank) -> o_proj -> reduce-scatter -> AttnRes on the
shard -> MLP on the shard. This step keeps that layout and replaces only each DSA /
KDA layer's all-gather .. reduce-scatter with one ``SpAttentionBlock``:

  stage            provider                                          family
  all-gather       ``SpComm.all_gather`` (multicast, Lamport)        ag
  in-projection    ``DecodeProjection`` over the gathered rows        dsa_in / kda_in
  core             ``SpDsaCore`` (DSA) / ``SpKdaCore`` (KDA)          dsa / kda
  o_proj + RS      ``SpOproj`` (4-rank reduce-scatter)                oproj_rs

Everything else is the stock module, called as the stock forward calls it
(``Path.DECLARED``, ``shapes.DECLARED_STOCK``): the embedding, ``_pre_attn_norm`` /
``_post_attn_norm`` (AttnRes) on the shard, the MegaMoE / dense MLP on the shard (with
stock's ``sp_padding_mask``), the aux streams, the final AttnRes, the final
``sp_all_gather`` and the norm. The verification (synthetic rejection), LM head,
sampler and the DSpark drafter stay stock too, except the drafter's MLA attention:
``runtime.draft`` FAST_DCP's TP4 entry runs it on cutedsl.draft.attention DraftDcp
(publish-only MlaQuery, polled cores with length-keyed splits, DcpValue), strict;
no other drafter table has a TP4 entry (SP_DRAFTER_TABLES, checked at build).

R = 4 n for n = 1..128 requests per DP group (0.25..32 requests per GPU): every
captured R is a multiple of TP4, so SP never pads. Every captured R is served by every
block, or the build raises ``shapes.Unsupported`` naming the family, R and cause; a
FULL decode capture this step cannot plan raises too (``SpStep.plan``). Nothing on
this path falls back to the stock forward.

Mailbox parities: ``SpComm`` alternates by SP block ordinal (DSA and KDA together),
the DSA DcpBox by DSA-layer ordinal. With an odd block count the last block of a step
and the first of the next share a parity; the stock final ``sp_all_gather`` (a TP
collective) separates them.

RecoverSSM: "triton" records (``recoverssm.prepare_recoverssm(defer=False)``), the
stock row layout and the stock Triton commit after sampling. The "deferred" commit
needs the stock ``KimiK3ModelState`` to call ``recoverssm.bind_model_state``, which it
does not; until it does, deferring would gain nothing anyway (``kda.sp.REPLAY`` is
empty).

NQ-TUNE: every family is GENERIC (no VR200 table at TP4 / DCP4 / 24 heads); see each
provider's module docstring for its sweep.
"""

import torch

from vllm import envs
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)

from ....ops.attn_res import attn_res
from ..cutedsl.common.sp_comm import SpComm
from ..cutedsl.dsa.sp import SpDsaCore, sp_dsa_box, sp_dsa_scratch
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.gemm.sp_policy import sp_projection_tactics
from ..cutedsl.kda.sp import SpKdaCore
from ..cutedsl.oproj.sp import SpOproj
from ..recoverssm import prepare_recoverssm
from ..shapes import Choice, Coverage, Path, Shape, Unsupported, shard_rows
from .providers import attention_kind, bf16_weight, pack_dsa_projection

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
    if attention.o_proj.reduce_results:
        raise Unsupported("oproj_rs", None, f"{where}: a reducing stock o_proj")
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
        try:  # BF16 projections only (a quantized layer is refused, not misread)
            if kind == "dsa":
                weight = pack_dsa_projection(attention)
            else:
                weight = bf16_weight(attention.in_proj_qkvgfab)
            output = bf16_weight(attention.o_proj)
        except ValueError as error:  # (re-raised with the layer)
            raise Unsupported(kind, None, f"{where}: {error}") from error
        pitch, k = weight.shape
        family = f"{kind}_in"
        with torch.accelerator.device_index(step.device.index):
            tactics = sp_projection_tactics(family, pitch, k, rows, ctx)
        self.projection = DecodeProjection(weight, rows, tactics, family=family)
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
            self.core = SpKdaCore(attention, rows, records=step.records, ctx=ctx)
        if self.core.pitch != pitch:
            raise Unsupported(
                kind, None,
                f"{where}: core pitch {self.core.pitch} != in-projection N {pitch}",
            )  # fmt: skip
        self.width = output.shape[1]  # heads x 128 (the o_proj input)
        self.oproj = SpOproj(output, step.comm, rows, ctx)
        cap = rows[-1]
        self.projected = step.scratch(f"{kind}.projected", (cap, pitch))
        self.attn_out = step.scratch(f"attn_out.{self.width}", (cap, self.width))
        for parity in range(step.comm.parities):  # every tactic takes the operands
            self.projection.check_operands(step.comm.gather[parity], self.projected)

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
        shard = x_shard.new_empty((n, WIDTH))
        return self.oproj.run(attn_out, shard, self.parity)

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


# ``runtime.draft`` TP-keyed tables the SP decode specializes the drafter with at TP4:
# its MLA attention (DraftDcp). Every other drafter piece is the declared stock module.
SP_DRAFTER_TABLES = frozenset({"FAST_DCP"})
DRAFTER_NOTE = "stock DSpark module; its MLA attention on DraftDcp (strict)"


def check_stock_drafter(tp):
    """The DSpark drafter is the declared stock module but for its attention: raises
    if a ``runtime.draft`` TP-keyed table other than SP_DRAFTER_TABLES would install
    a specialized drafter piece at ``tp`` (the declaration would then be false), or
    one of SP_DRAFTER_TABLES lacks its TP entry (the attention would stay stock)."""
    from . import draft

    tables = {
        name: table
        for name, table in vars(draft).items()
        if name.isupper()
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


class SpStep:
    """Builds the SP blocks on a ``SpecializedDecode`` (``runtime``, built with
    ``sequence_parallel``) and runs its captured steps (module docstring). All
    collective allocations (the SP mailboxes, the DSA DcpBox, the DSA layers' W_UV
    gathers) happen here, in layer order on every rank, before the KV cache is
    profiled."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.ctx = ctx = runtime.context
        self.q, self.device = ctx.q, runtime.device
        backbone = runtime.backbone
        if not ctx.sequence_parallel or ctx.tp != ctx.dcp:
            raise Unsupported(
                "sp_step", None,
                f"TP{ctx.tp} / DCP{ctx.dcp}, sequence parallel {ctx.sequence_parallel}",
            )  # fmt: skip
        if not getattr(backbone, "use_sequence_parallel", False):
            raise Unsupported("sp_step", None, "the backbone is not sequence-parallel")
        if not getattr(backbone, "use_attn_res", False):
            raise Unsupported("attn_res", None, "the SP step runs AttnRes models only")
        self.rows = rows = tuple(sorted(set(runtime.captured)))
        if not rows:
            raise Unsupported("sp_step", None, "no captured FULL decode graphs")
        for r in rows:
            shape = Shape(r, self.q)
            if r % self.q:
                raise Unsupported("sp_step", shape, f"not whole q{self.q} requests")
            shard_rows(shape, ctx)
        self.shard_cap = rows[-1] // ctx.tp
        self.layers = layers = runtime.layers
        for decoder in layers:
            sp_layer(decoder, self)
        self.draft_tables = check_stock_drafter(ctx.tp)
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        self.buffers: dict[str, torch.Tensor] = {}
        owner = runtime.group, self.device, ctx.tp, ctx.rank
        self.comm = SpComm(*owner, width=WIDTH, rows_cap=rows[-1])
        kinds = {attention_kind(d.self_attn) for d in layers}
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
            # "triton": the stock record layout and commit (no stock hook needed).
            self.records = prepare_recoverssm(layers, ctx, defer=False)
        self.blocks = {
            d.layer_idx: SpAttentionBlock(d, self, ordinal)
            for ordinal, d in enumerate(layers)
        }
        self.dense = any(not d.is_moe_layer for d in layers)
        self.moe = any(d.is_moe_layer for d in layers)
        self.table = self.plan_coverage()
        self.table.assert_complete(rows)
        if ctx.rank == 0:
            for line in self.table.lines():
                logger.info("Kimi K3 SP decode coverage %s", line)
        logger.info_once(
            "Kimi K3 SP decode (TP%d = DCP%d over DP%d, EP%d, q%d, rows %d-%d, %d "
            "graphs): %d SP blocks (%s), RecoverSSM %s; SP mailboxes %.0f MiB "
            "symmetric; the DSpark drafter stays the stock module but for its "
            "attention (TP%d entries only in FAST_DCP of %d drafter tables)",
            ctx.tp, ctx.dcp, ctx.dp, ctx.ep, self.q, rows[0], rows[-1], len(rows),
            len(self.blocks), "/".join(sorted(kinds)), self.records,
            self.comm.nbytes() / 2**20, ctx.tp, len(self.draft_tables),
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
            "embed": "stock VocabParallelEmbedding",
            "attn_res": "stock attn_res on the shard",
            "final_gather": "stock sp_all_gather",
            "lm_head": "stock",
            "sampler": "stock",
            "drafter": DRAFTER_NOTE,
        }
        if self.moe:
            declared["moe"] = f"stock MegaMoE on the shard (EP{ctx.ep})"
        if self.dense:
            declared["dense"] = "stock dense MLP on the shard"
        if method == "synthetic":
            declared["verify"] = "stock (synthetic rejection)"
        return declared

    def plan_coverage(self):
        """Every captured R's families: every block's choice (each raises unless it
        serves R; the first block of each kind's is logged), and the declared stock
        modules. Without synthetic rejection ``verify`` is VerifyArgmax, which covers
        TP4 or raises when the model builds it."""
        coverage = Coverage(self.q)
        declared = self.declared()
        for r in self.rows:
            choices = {"ag": self.comm.choice(r)}
            for block in self.blocks.values():
                for family, choice in block.choices(r).items():
                    choices.setdefault(family, choice)
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
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in layers]
        missing = [d.layer_idx for d, md in zip(layers, metadata) if md is None]
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
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        rows = x.shape[0]
        if rows not in self.rows:
            raise Unsupported(
                "sp_step", Shape(rows, step.q), "not a captured row count"
            )
        # As the stock SP forward: MegaMoE reads the shard's padding rows from the
        # forward context when the stock deployment skips them.
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            context = get_forward_context()
            context.is_padding = sp_padding_mask(context.is_padding, x)
        hidden = sp_shard(x)
        remote = backbone.collect_remote_aux_hidden_states(None)
        aux = [hidden] if backbone.start_layer in self.aux_layers else []
        residual = hidden.new_empty(
            (hidden.size(0), backbone.num_attn_res_blocks, hidden.size(1))
        )
        prefix_sum, hidden = hidden, None
        for layer, md in zip(self.layers, step.metadata):
            index = layer.layer_idx
            hidden, prefix_sum, residual = layer._pre_attn_norm(
                hidden, residual, prefix_sum
            )
            block = self.blocks[index]
            hidden = block.run(hidden, caches_of(layer.self_attn), md, step)
            hidden, prefix_sum, residual = layer._post_attn_norm(
                hidden, residual, prefix_sum
            )
            hidden = layer.mlp(hidden)
            if index + 1 in self.aux_layers:
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
        if aux:
            width = hidden.shape[-1]
            packed = sp_all_gather(torch.cat([hidden, *aux], dim=-1))[:rows]
            hidden, *aux = packed.split(width, dim=-1)
        else:
            hidden = sp_all_gather(hidden)[:rows]
        aux = remote + aux
        return runtime.normalize((hidden, aux) if aux else hidden)

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
        gather is collective). The packed DSA in-projection is the stock linears'
        storage (``pack_dsa_projection``), so it is current already."""
        for decoder in self.layers:
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
