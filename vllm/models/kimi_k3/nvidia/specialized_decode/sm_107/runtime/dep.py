# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP specialized decode step: data-parallel attention (TP1 = DCP1, all 96 heads
on every rank) over expert-parallel native MegaMoE (EP16 / EP32 = DP16 / DP32),
K3.1 only, DSpark q 2..8 (``runtime.dep_unsupported``). Phase A of the DEP plan: the
first end-to-end DEP step; phase B items are named where they apply.

Stock at g = 1 (``use_sequence_parallel`` False: no row shard) runs each layer on the
rank's own R rows (R = n q, n <= ``runtime.DEP_MAX_REQUESTS`` requests) with no TP
collective: AttnRes -> attention (96 heads) -> o_proj (the whole output: TP1) ->
AttnRes -> the MoE / dense MLP on all R rows (MegaMoE ``max_num_tokens`` =
``max_num_batched_tokens``, the runner's ``is_padding``). This step runs the same
composition as

  stage              provider                                      family
  embedding          ``PreparedEmbedding`` (TP1: the whole vocab;   embed
                     its TP all-reduce is the identity at world
                     size 1: no collective)
  AttnRes (pre)      ``PreparedAttnRes`` (ClusterAttnRes) on the    attn_res
                     R rows
  in-projection      ``DecodeProjection`` (``sp_policy``, key       dsa_in / kda_in
                     (1, 1, N, K): 14560 / 49376 x 7168)
  core               ``group_ep.TpDsaCore`` (PreparedDsa at DCP1,   dsa / kda
                     96 heads: the fused indexer (32 heads),
                     MlaQuery into the 128-head padded FP8 query,
                     the trtllm-gen sparse MLA kernel stock runs,
                     DECLARED sparse_mla, MlaValue) /
                     ``SpKdaCore`` ("none": the plain stock-math
                     verify, KdaSplitStock / KdaVerifyLoopStock)
  o_proj             ``DecodeProjection`` [R, 12288] x W_o^T (no    oproj
                     reduction; split-K at 1-16 rows: K / 128 >
                     64 rules warp split-K out)
  AttnRes (post)     ``PreparedAttnRes.for_post`` on the R rows     attn_res
  MoE / dense MLP    the stock KimiMoE module (FP32 router, BF16    DECLARED moe / dense
                     shared expert, latent down / norm / up,
                     routed MegaMoE EP16 / EP32) / the stock
                     replicated dense MLP, on all R rows

then the final ``PreparedAttnRes`` (with the backbone's final norm) on the R rows.
Outside the step's forward the LM head (stock cuBLAS over the 163840 columns), the
sampler and the verification (stock rejection sampler: TP1 argmax / sampled verify
are B1 / B2) and the DSpark drafter (stock module at G = 1: W10) are declared stock.
The declared families are exactly ``DEP_DECLARED``, listed per captured R; every
required boundary family (``required_families``) is planned per R or the build
raises ``shapes.Unsupported``; a FULL capture outside the captured rows raises in
``plan``.

PENDING SIGN-OFF (as ``group_ep``): ``moe`` declares the whole stock KimiMoE module;
B7 (the EP-front rewrite) leaves only the routed MegaMoE callable declared.

Precision is stock's at every stage (summation order aside): the projections sum K
in FP32 and round once to BF16 (stock's BF16 GEMMs); AttnRes is row-wise
(ClusterAttnRes, stock's roundings); the DSA core is the TP DCP1 chain at stock K3
precision (FP32 indexer logits, exact top-k) with the query zero-padded to 128 heads
exactly as stock's ``_pad_query_heads`` for trtllm-gen (heads are independent); the
KDA core is the stock Triton verify's math (FP32 state). The MoE / dense MLPs, the
LM head, the sampler and the drafter are stock's own modules.

Collectives (plan section 5): the only collective of a DEP step is MegaMoE's EP
dispatch / combine inside the stock MoE module, called once per MoE layer per step
on every path: this step, the stock forward of an eager / profiling / warmup step
(``plan`` returns None outside FULL captures) and a DP rank's dummy step (the runner
replays the same captured FULL graph with padding metadata: slot -1 rows skipped by
the MlaQuery insert and DsaIndex, NULL KDA state slots, count-0 trtllm-gen rows).
The step never touches ``forward_context.is_padding`` (stock DEP reads the runner's
mask; no ``sp_padding_mask``). The TP group has size 1: no o_proj reduction, no
DcpBox, no mailbox.

NQ-TUNE: every family is GENERIC (B3: the split 96-head MLA query; B4: KDA tables at
96 heads; B5: the DEP projection tables)."""

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from ..cutedsl.common.embedding import PreparedEmbedding
from ..recoverssm import prepare_recoverssm
from ..shapes import Choice, Coverage, Path, Shape, Unsupported
from .group_ep import GROUP_KINDS, WIDTH, GroupAttentionBlock, GroupStep
from .providers import attention_kind

logger = init_logger(__name__)

# The stock modules this step keeps (``Path.DECLARED``), checked per captured R by
# ``Coverage.assert_complete``; spelled out so a new stock family never passes by
# inheritance. No collective is declared (TP1: no embedding all-reduce, o_proj
# reduction or MLP gather); the attention cores, projections, AttnRes and the
# embedding are specialized.
DEP_DECLARED = frozenset(
    {
        "moe",
        "dense",
        "sparse_mla",
        "lm_head",
        "sampler",
        "verify",
        "drafter",
    }
)
# The DEP attention width per rank: all 96 heads of 128 (TP1).
DEP_HEADS = 96


def dep_layer(decoder, step):
    """Raises unless ``decoder`` is a stock layer the DEP step serves: a DSA / KDA
    attention without a fused GEMM-RS / AR o_proj, in a layer that is not
    sequence-parallel; its MLP the stock MegaMoE (expert parallel, no token shard, a
    MegaMoE buffer for the step's largest R) or a replicated dense MLP. (At TP1 the
    o_proj's ``reduce_results`` reduces over a world of one: no collective.)"""
    index, attention = decoder.layer_idx, decoder.self_attn
    kind = attention_kind(attention)
    where = f"layer {index}"
    if kind not in GROUP_KINDS:
        raise Unsupported(kind, None, f"{where}: {kind} attention (DSA / KDA only)")
    if getattr(decoder, "use_sequence_parallel", False):
        raise Unsupported(
            "dep_step", None,
            f"{where}: a sequence-parallel layer (stock DEP has no row shard)",
        )  # fmt: skip
    if getattr(attention, "gemm_rs_ar", None) is not None:
        raise Unsupported(
            "oproj", None,
            f"{where}: a fused GEMM-RS / AR o_proj: not the plain o_proj this step "
            "runs",
        )  # fmt: skip
    mlp = decoder.mlp
    if not decoder.is_moe_layer:
        if getattr(mlp, "shard_sequence_parallel", False):
            raise Unsupported("dense", None, f"{where}: a TP-sharded dense MLP")
        if getattr(mlp, "gemm_rs_ar", None) is not None:
            raise Unsupported("dense", None, f"{where}: a fused GEMM-RS / AR dense MLP")
        return
    if not getattr(mlp, "use_mega_moe", False) or getattr(mlp, "moe_token_shard", 0):
        raise Unsupported(
            "moe", None, f"{where}: not MegaMoE on the rank's rows (no token shard)"
        )
    shared = getattr(mlp, "shared_experts", None)
    if getattr(shared, "shard_sequence_parallel", False):
        raise Unsupported("moe", None, f"{where}: a TP-sharded shared expert")
    tokens = getattr(mlp.experts, "max_num_tokens", 0)
    if tokens < step.rows[-1]:
        raise Unsupported(
            "moe", Shape(step.rows[-1], step.q),
            f"{where}: MegaMoE max_num_tokens {tokens} < {step.rows[-1]} rows",
        )  # fmt: skip


class DepAttentionBlock(GroupAttentionBlock):
    """One DSA / KDA layer on the rank's own rows: ``run(x [R, 7168])`` returns the
    stock ``o_proj(self_attn(x))`` (TP1: the whole output, no reduction; summation
    order aside), the step's o_proj scratch rows (the post AttnRes consumes them
    before the next layer's o_proj writes them)."""

    def run(self, x, caches, metadata, step):
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
        out = self.partial[:rows]
        self.oproj.run(attn_out, out)
        return out


class DepStep(GroupStep):
    """Builds the DEP blocks on a ``SpecializedDecode`` (``runtime``, built with
    ``Context.dep``) and runs its captured steps (module docstring). Reuses the
    replicated EP_GROUP step's AttnRes sites, scratch, DCP1 DSA core
    (``group_ep.TpDsaCore``), ``plan``, ``prepare_kv`` and weight reload; nothing
    here allocates a collective buffer."""

    family = "dep_step"

    def __init__(self, runtime):
        self.runtime = runtime
        self.ctx = ctx = runtime.context
        self.q, self.device = ctx.q, runtime.device
        backbone = runtime.backbone
        if not ctx.dep or ctx.ep_group_step:
            raise Unsupported(
                "dep_step", None,
                f"dep {ctx.dep}, ep_group {ctx.ep_group}, sequence parallel "
                f"{ctx.sequence_parallel} (the DEP step only)",
            )  # fmt: skip
        if (ctx.tp, ctx.dcp) != (1, 1):
            raise Unsupported(
                "dep_step", None, f"TP{ctx.tp} / DCP{ctx.dcp} (DEP is TP1 = DCP1)"
            )
        if ctx.heads_per_rank != DEP_HEADS:
            raise Unsupported(
                "dep_step", None,
                f"{ctx.heads_per_rank} heads per rank (K3.1 DEP: {DEP_HEADS})",
            )  # fmt: skip
        if getattr(backbone, "use_sequence_parallel", False):
            raise Unsupported(
                "dep_step", None,
                "the backbone is sequence-parallel (stock DEP keeps the rank's rows)",
            )  # fmt: skip
        if not getattr(backbone, "use_attn_res", False):
            raise Unsupported("attn_res", None, "the DEP step runs AttnRes models only")
        self.rows = rows = tuple(sorted(set(runtime.captured)))
        if not rows:
            raise Unsupported("dep_step", None, "no captured FULL decode graphs")
        for r in rows:
            if r % self.q:
                shape = Shape(r, self.q)
                raise Unsupported("dep_step", shape, f"not whole q{self.q} requests")
        # (no EP_GROUP_MAX_ROWS cap: the load gate caps the requests at the fused
        # indexer's 512; every other DEP family is uncapped in rows)
        self.shard_cap = rows[-1]  # (the MoE runs on every row: no shard)
        self.layers = layers = runtime.layers
        for decoder in layers:
            dep_layer(decoder, self)
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        self.buffers: dict[str, torch.Tensor] = {}
        kinds = {attention_kind(d.self_attn) for d in layers}
        self.dsa_box = self.dsa_scratch = None  # (DCP1: no DcpBox)
        self.dsa_layers = 0
        self.records = "none"
        if "kda" in kinds:
            # ReplaySSM / RecoverSSM is refused at load (B9); a layer that still
            # carries records would be misread by the plain core: raise.
            records = prepare_recoverssm(layers, ctx, defer=False)
            if records != "none":
                raise Unsupported(
                    "kda", None, f"RecoverSSM records ({records}) at DEP (B9)"
                )
        self.blocks = {d.layer_idx: DepAttentionBlock(d, self) for d in layers}
        self.dense = any(not d.is_moe_layer for d in layers)
        self.moe = any(d.is_moe_layer for d in layers)
        self.kinds = kinds
        self.build_attn_res(backbone, rows[-1])
        try:  # TP1: the whole vocabulary, its all-reduce the identity
            self.embedding = PreparedEmbedding(backbone.embed_tokens, ctx.tp, ctx.rank)
        except ValueError as error:
            raise Unsupported("embed", None, str(error)) from error
        self.allowed = DEP_DECLARED
        self.table = self.plan_coverage()
        self.table.assert_complete(rows, self.allowed)
        if ctx.rank == 0:
            for line in self.table.lines():
                logger.info("Kimi K3 DEP decode coverage %s", line)
        logger.info_once(
            "Kimi K3 DEP decode (TP1 x DP%d, EP%d, %d heads per rank, q%d, rows %d-%d, "
            "%d graphs): %d blocks (%s), the stock MoE module on every row",
            ctx.dp, ctx.ep, ctx.heads_per_rank, self.q, rows[0], rows[-1], len(rows),
            len(self.blocks), "/".join(sorted(kinds)), scope="global",
        )  # fmt: skip

    def declared(self):
        """The stock modules the step keeps (``Path.DECLARED``), family -> note."""
        ctx = self.ctx
        config = get_current_vllm_config()
        backend = config.kernel_config.moe_backend
        declared = {
            "lm_head": "stock processor._get_logits (cuBLAS, 163840 columns at TP1; "
            "B6: the TP1 LM head)",
            "sampler": "stock (outside the step's forward)",
            "verify": "stock rejection sampler (DEP: TP1 argmax / sampled verify "
            "pending B1 / B2)",
            "drafter": "stock DSpark module at G = 1 (W10: the specialized drafter "
            "at TP1)",
        }
        if "dsa" in self.kinds:
            declared["sparse_mla"] = (
                "FlashInfer trtllm-gen sparse MLA decode at DCP1 (96 query heads "
                "zero-padded to 128 as stock): the kernel stock runs, called by the "
                "DSA core"
            )
        if self.moe:
            declared["moe"] = (
                f"stock KimiMoE module ({backend}: FP32 router, BF16 shared expert, "
                f"latent down / norm / up, routed MegaMoE EP{ctx.ep}) on the rank's R "
                "rows (PENDING SIGN-OFF: B7 EP-front)"
            )
        if self.dense:
            declared["dense"] = "stock replicated dense MLP on the rank's R rows (B8)"
        return declared

    def plan_coverage(self):
        """Every captured R's families: the layout, the embedding, AttnRes, every
        block's choice (each raises unless it serves R) and the declared stock
        modules; a required boundary without a provider raises."""
        coverage = Coverage(self.q)
        declared = self.declared()
        required = self.required_families()
        for r in self.rows:
            note = "local rows (TP1), the MoE on every row"
            choices = {"layout": Choice(Path.GENERIC, "local", note=note)}
            choices["embed"] = Choice(
                Path.GENERIC, "gather", note="PreparedEmbedding TP1 (whole vocab)"
            )
            choices["attn_res"] = Choice(
                Path.GENERIC, "plain", note="ClusterAttnRes pre / post / final"
            )
            for block in self.blocks.values():
                for family, choice in block.choices(r).items():
                    choices.setdefault(family, choice)
            coverage.specialize(r, choices)
            for family, note in declared.items():
                coverage.declare(r, family, note)
            planned = coverage.plan[r] or {}
            missing = sorted(required - set(planned))
            if missing:
                raise Unsupported(
                    "dep_step", Shape(r, self.q),
                    f"boundaries without a provider: {', '.join(missing)}",
                )  # fmt: skip
        return coverage

    def required_families(self):
        """The boundary families every captured R must plan: the embedding, AttnRes,
        each attention kind's in-projection / core / o_proj, the DSA sparse MLA, the
        MLPs, the verification and the drafter."""
        required = {"layout", "embed", "attn_res", "verify", "drafter"}
        for kind in self.kinds:
            required |= {f"{kind}_in", kind, "oproj"}
        if "dsa" in self.kinds:
            required.add("sparse_mla")
        if self.moe:
            required.add("moe")
        if self.dense:
            required.add("dense")
        return frozenset(required)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step: the
        stock ``KimiLinearModel.forward`` at g = 1 with the specialized embedding,
        AttnRes sites and blocks, and the stock MoE / dense MLP on every row. The
        runner's ``is_padding`` is stock DEP's: left untouched."""
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
                self.family, Shape(rows, step.q), "not a captured row count"
            )
        remote = backbone.collect_remote_aux_hidden_states(None)
        aux_out = [torch.empty_like(x) for _ in self.aux_layers]
        aux = dict(zip(self.aux_layers, aux_out))
        prefix, blocks = self.prefix[:rows], self.res_blocks[:rows]
        normed = self.scratch("normed", (self.rows[-1], WIDTH))[:rows]
        moe_in = self.moe_in[:rows]
        hidden, index = x, self.layers[0].layer_idx - 1
        for layer, md in zip(self.layers, step.metadata):
            index = layer.layer_idx
            self.before[index].run(hidden, prefix, blocks, normed, aux.get(index))
            caches = caches_of(layer.self_attn)
            attended = self.blocks[index].run(normed, caches, md, step)
            self.post[index].run(attended, prefix, blocks, moe_in)
            # The stock MoE (MegaMoE EP dispatch / combine: the step's only
            # collective) or dense MLP on every row.
            hidden = layer.mlp(moe_in)
        out = torch.empty_like(x)
        self.final.run(hidden, prefix, blocks, out, aux.get(index + 1))
        streams = remote + aux_out
        return (out, streams) if streams else out
