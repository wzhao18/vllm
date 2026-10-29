# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock Kimi K3 backbone: one routing
decision per step, then every layer's prepared providers in one loop."""

import time
from functools import partial
from typing import Any, NamedTuple

import torch
from cutlass import Int32

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.distributed import tensor_model_parallel_all_reduce as all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ..cutedsl.attn_res.prepared import PreparedAttnRes
from ..cutedsl.attn_res.prepared import choose as attn_res_choose
from ..cutedsl.attn_res.sharded import ShardBox
from ..cutedsl.common.embedding import PreparedEmbedding
from ..cutedsl.common.graph_switch import GraphSwitch
from ..cutedsl.common.host import dynamic_view, to_cute
from ..cutedsl.common.host import stream as current_stream
from ..cutedsl.common.policy import batch_table
from ..cutedsl.common.twoshot import TwoShot
from ..cutedsl.kda.head import KdaSide
from ..cutedsl.kda.prepared import choose as kda_choose
from ..cutedsl.kda.prepared import verify_kernel
from ..cutedsl.mla.prepared import SWITCH
from ..cutedsl.mla.prepared import choose as mla_choose
from ..cutedsl.mla.rows import compile_rows
from ..cutedsl.moe.dense import PreparedDenseMLP
from ..cutedsl.moe.experts_runtime import (
    PersistentExperts,
    place,
    route_mailbox,
    rubin64,
)
from ..cutedsl.moe.front import SharedFront
from ..cutedsl.moe.front_rows import FrontRows
from ..cutedsl.moe.prepared import (
    WIDE_EXPERTS,
    WIDE_FRONT,
    MoEScratch,
    PreparedMoE,
    deferred_rows,
    stock_partials_ok,
)
from ..cutedsl.moe.prepared import choose as moe_choose
from ..cutedsl.moe.small_batch import chain as small_batch_chain
from ..cutedsl.moe.tail_rows import PreparedTailRows
from ..cutedsl.moe.tail_runtime import MoeTailBox, PreparedTail
from ..cutedsl.oproj.prepared import OprojMailbox
from ..cutedsl.oproj.prepared import choose as oproj_choose
from ..shapes import Choice, Context, Coverage, Path, Shape, rows_policy
from .prefetch import BEFORE_HEAD, BEFORE_TMEM_MLA, PrefetchController
from .providers import (
    BATCHES,
    GENERIC_ROWS,
    PREPARED_MAX,
    Attention,
    attention_kind,
    pack_dsa_projection,
    pack_front,
)
from .seams import StockAttnRes

# Rubin (compute capability 10.7) TP16 batches whose step head and dense MLP replace the
# framework all-reduce (FlashInfer two-shot) + a non-PDL AttnRes launch with one-shot
# Lamport mailboxes polled by the next AttnRes: the embedding gather publishes each row
# from its owner rank into a one-slot mailbox (exact: the other ranks' rows are zeros),
# and the dense down projection publishes its BF16 partials into TP slots that the
# AttnRes sums in FP32 in rank order and rounds once.
logger = init_logger(__name__)

# 65-256 rows too (axhb-sw3, 16 ranks, mirrored, 93 layers, 32k, us/step: B9 -11, B17
# -19, B24 -14, B28 -37, B32 -5): the stock embedding + FlashInfer two-shot all-reduce
# (~36 us at 256 rows) leave the step head.
RING_HEAD = {16: "1-256:1"}
# Rubin verify graphs above the tuned rows (token rows by q, "*": any) that run the
# specialized decode (generic kernels and stock seams, ``wide``); others run the stock
# forward. A point leaves this table when its in-graph step measures slower than the
# stock forward.
WIDE_ROWS = {"*": "65-768:1"}
RING_DENSE = {16: "1-8:1"}


def head_batches(attention, q):
    """Batches whose captured steps run this layer's KdaHead. A deployment with verify
    length q > 1 captures every FULL graph as a uniform q-token verify step: the head
    runs there only if it has a verify kernel for (batch, q) (``verify_kernels``), or
    says so itself (``runs(batch, q)``)."""
    head = attention.head
    if head is None:
        return ()
    runs = getattr(head, "runs", None)
    if runs is not None:
        return tuple(b for b in head.batches if runs(b, q))
    verify = getattr(head, "verify_kernels", {})
    return tuple(b for b in head.batches if q == 1 or (b, q) in verify)


def ring_batches(table, tp, device, batches):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return frozenset()
    on = batch_table(table.get(tp, ""))
    return frozenset(b for b in batches if on.get(b))


class Step(NamedTuple):
    q: int  # tokens per request
    metadata: list  # per layer
    slot_mapping: Any
    positions: Any = None  # the step's token positions (the stock seams read them)


def attention_layer(module):
    """The registered attention layer of a decoder's attention: a K3.1 DSA layer's
    inner sparse MLA layer, else the module itself."""
    return module.mla_attn.mla_attn if hasattr(module, "dsa_indexer") else module


def layer_name(module):
    module = attention_layer(module)
    return getattr(module, "prefix", getattr(module, "layer_name", None))


def decode_metadata(attention, context, q, b):
    """One layer's metadata if this step is a pure decode (q = 1) or a uniform
    verify step of b tokens, else None."""
    md = context.attn_metadata
    md = md.get(layer_name(attention)) if isinstance(md, dict) else None
    if md is None:
        return None
    if q > 1:
        return md if is_uniform_verify(md, q, b) else None
    if md.num_prefills or getattr(md, "num_spec_decodes", 0):
        return None
    return md


def is_uniform_verify(md, q, b):
    """Whether every request of the b // q graph rows verifies q tokens (FULL graphs
    pad whole requests: KDA stages NULL state rows, MLA zero-length decodes)."""
    if md.num_prefills:
        return False
    if not hasattr(md, "num_spec_decodes"):
        return md.num_decode_tokens == md.num_decodes * q == b
    rows, tokens = md.spec_state_indices_tensor, md.num_spec_decode_tokens
    full = rows is not None and rows.shape[0] * q == b
    return full and not md.num_decodes and tokens == md.num_spec_decodes * q


def caches_of(attention):
    cache = getattr(attention_layer(attention), "kv_cache", ())
    return [cache] if isinstance(cache, torch.Tensor) else list(cache)


class SpecializedDecode:
    """Built once the weights are loaded; owns every prepared provider, the
    mailboxes and the decode buffers. Reads the backbone, never modifies it."""

    def __init__(self, backbone, config):
        self.backbone, self.buffers = backbone, {}
        q = config.q
        # Verify steps decode q tokens per request at M = requests * q tokens; FULL
        # verify graphs round each capture size up to whole requests.
        rounded = {round_up(s, q) for s in config.capture_sizes}
        verify = {m for m in rounded if m <= PREPARED_MAX}
        # The batches of the tuned providers (every family up to 64 rows).
        self.tuned, self.q = tuple(sorted({*BATCHES, *verify})), q
        self.queries = (1, q) if q > 1 else (1,)
        # The longest sequence the deployment serves bounds the MLA split count.
        self.max_context, self.mla_core = config.max_context, config.mla_core
        self.dsa_core = config.dsa_core
        # Decode context parallel over the TP ranks: MLA KV is sequence-sharded.
        self.dcp, self.dcp_box, self.mla_layers = config.dcp, None, 0
        self.dsa_dcp_box, self.dsa_layers = None, 0
        self.bf16_state = config.bf16_state  # BF16 KDA recurrent state (KdaHead)
        group = get_tp_group()
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.group = group.device_group
        self.device = torch.device(current_platform.current_device())
        properties = torch.cuda.get_device_properties(self.device)
        cc = (properties.major, properties.minor)
        # The runner's FULL decode graphs. On Rubin, those of verify steps above the
        # tuned rows (``wide``) run the generic kernels and stock seams instead of the
        # stock forward: ``Attention.run_wide`` (S2 MLA, S3 o-projection) and the stock
        # MoE layer (S4); the o-projection seam needs every rank's full rows (no
        # sequence parallelism).
        self.captured = config.captured
        seams = not getattr(backbone, "use_sequence_parallel", False)
        wide = cc == (10, 7) and q > 1 and seams
        rows = [m for m in self.captured if m > PREPARED_MAX and wide]
        self.wide = tuple(m for m in rows if rows_policy(WIDE_ROWS, Shape(m, q), 0))
        self.batches = tuple(sorted({*self.tuned, *self.wide}))
        self.capacity = self.batches[-1]  # rows of every decode buffer
        self.context = Context(
            arch="rubin" if cc == (10, 7) else "gb300",
            tp=self.tp,
            rank=self.rank,
            dcp=self.dcp,
            sms=properties.multi_processor_count,
            q=q,
            max_rows=self.capacity,
            max_context=self.max_context,
            bf16_state=self.bf16_state,
        )
        self.coverage: Coverage | None = None
        # Wide batches whose AttnRes state is row-sharded (``attn_res.prepared
        # .choose``): only where the fused wide o-projection and the wide MoE tail also
        # serve the shape, since they publish into and read from the row owners.
        # Decided before any provider is built: every hook is a pure function of the
        # shape and the context.
        modes = {m: attn_res_choose(Shape(m, q), self.context) for m in self.wide}
        self.sharded = frozenset(
            m
            for m, mode in modes.items()
            if mode.tactic == ("rows", "sharded")
            and oproj_choose(Shape(m, q), self.context)
            and moe_choose(Shape(m, q), self.context)
        )
        # Wide batches whose AttnRes sites run the stock op (``seams.StockAttnRes``).
        self.stock_attn_res = frozenset(
            m for m, mode in modes.items() if mode.path is Path.STOCK
        )
        # Their mailboxes (row owners' slots and ring, all-gather boxes): comm3's
        # ``attn_res.sharded.ShardBox``, built before any provider binds it.
        self.shard_box = None
        if self.sharded:
            owner = self.group, self.device, self.tp, self.rank
            self.shard_box = ShardBox(*owner, self.capacity)

        layers = backbone.layers[backbone.start_layer : backbone.end_layer]
        self.layers = layers = tuple(layers)
        self.kda_side, self.kda_heads = KdaSide(self.device), 0
        # Batches whose captured step switches between two bodies (prepare_kv).
        self.switched: dict[int, int] = {}
        self.switch: GraphSwitch | None = None
        self.build_providers()
        # Verify batches whose MLA layers run as single-query rows (mla.rows): their
        # per-row (tables, local lengths, totals), written by verify_rows each step.
        self.rows_batches = frozenset(
            b for a in self.attention.values() for b in getattr(a, "rows", ())
        )
        self.rows_meta: dict = {}
        self.embedding = PreparedEmbedding(backbone.embed_tokens, self.tp, self.rank)
        self.before = {
            d.layer_idx: PreparedAttnRes.for_target(
                d,
                has_delta=d is not layers[0],
                carveout=self.attention[d.layer_idx].before_carveout,
            )
            for d in layers
        }
        # One-slot embedding mailbox polled by the first AttnRes (RING_HEAD).
        self.head_batches = ring_batches(RING_HEAD, self.tp, self.device, self.batches)
        if self.head_batches:
            owner = self.group, self.device, self.tp, self.rank
            # Sized to the largest ring batch: wide rows (RING_HEAD above 64) too.
            cap = max(PREPARED_MAX, *self.head_batches)
            self.head_box = OprojMailbox(*owner, (1, 1, cap, 7168))
            self.embedding.bind_ring(self.head_box.mc)
            self.before[layers[0].layer_idx].bind_ring(self.head_box.buffer[0])
        # The final AttnRes also applies the backbone's final norm.
        self.final = PreparedAttnRes.for_target(backbone, final=True)
        # The stock AttnRes sites of ``stock_attn_res`` batches.
        self.before_stock, self.final_stock = {}, None
        if self.stock_attn_res:
            self.before_stock = {
                d.layer_idx: StockAttnRes.for_target(d, has_delta=d is not layers[0])
                for d in layers
            }
            self.final_stock = StockAttnRes.for_final(backbone)
        # The next (or final) AttnRes consumes each MoE output from its MoE tail output
        # row.
        self.fused_before = {}
        followers = [self.before[f.layer_idx] for f in layers[1:]] + [self.final]
        for layer, consumer in zip(layers, followers):
            moe = self.feed_forward[layer.layer_idx]
            if isinstance(moe, PreparedMoE):
                consumer.bind_ring(moe.tail.tail_box.out.buffer)
                if moe.rows is not None:  # wide batches: the wide tail's output ring
                    consumer.bind_ring_rows(moe.state.rows.out.buffer)
                self.fused_before[layer.layer_idx] = consumer
            elif layer.layer_idx in self.dense_rings:
                # Bitwise with the FlashInfer two-shot all-reduce it replaces.
                ring = self.dense_rings[layer.layer_idx][0].buffer[0]
                consumer.bind_ring(ring, sanitize=True)
                self.fused_before[layer.layer_idx] = consumer
        # The AttnRes entering each aux layer (or the final one) also stores the
        # updated prefix: the stock aux stream (prefix sum plus the last MLP output).
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        end = layers[-1].layer_idx + 1
        for index in self.aux_layers:
            (self.final if index == end else self.before[index]).bind_aux()
        if self.sharded:  # after the rings and the aux variants
            for consumer in (*self.before.values(), self.final):
                consumer.bind_sharded(self.shard_box, self.sharded, self.rank, self.tp)
        empty = partial(torch.empty, device=self.device, dtype=torch.bfloat16)
        rows = self.capacity
        self.hidden = (empty((rows, 7168)), empty((rows, 7168)))
        self.prefix = empty((rows, 7168))
        self.blocks = empty((rows, backbone.num_attn_res_blocks, 7168))

    def build_providers(self):
        """Every layer's attention and feed-forward providers, the model's mailboxes
        and the prefetch side stream. All collective allocations, weight packing and
        kernel compiles finish here, before KV binding and CUDA graph capture."""
        config, layers = self.backbone.config, self.layers
        beta = config.activation_situ_beta
        linear_beta = config.activation_situ_linear_beta
        tp, rank, device, batches = self.tp, self.rank, self.device, self.batches
        owner = self.group, device, tp, rank
        packed = {
            d.layer_idx: pack_front(d.mlp, rank, tp) for d in layers if d.is_moe_layer
        }
        for d in layers:  # before the prefetch tables take the weight addresses
            if attention_kind(d.self_attn) == "dsa":
                pack_dsa_projection(d.self_attn)
        prefetch = self.prefetch = PrefetchController(self, layers, packed)
        # The MoE layers' providers serve the tuned batches and the wide ones their
        # ``choose`` admits; other wide batches run the stock MoE layer (S4).
        admitted = (m for m in self.wide if moe_choose(Shape(m, self.q), self.context))
        self.moe_batches = moe_batches = (*self.tuned, *admitted)
        state = MoEScratch(
            tp, device, moe_batches, owner=(self.group, device, tp, rank)
        )
        # Wide MoE batches run ``PreparedMoE.run_rows`` (the tuned providers below).
        tuned_moe = tuple(b for b in moe_batches if b <= PREPARED_MAX)
        wide_moe = tuple(b for b in moe_batches if b > PREPARED_MAX)
        mailbox, tail_box, moe_layers = (
            OprojMailbox(*owner),
            MoeTailBox(*owner),
            list(packed),
        )
        twoshot = TwoShot(*owner)
        front_box = OprojMailbox(*owner, (1, 64, 3584 + 2 * 896))
        small_batch = small_batch_chain.SmallBatchBox(*owner)
        self.attention: dict[int, Attention] = {}
        self.feed_forward: dict[int, Any] = {}
        self.dense_rings: dict[int, Any] = {}
        dense_batches = ring_batches(RING_DENSE, tp, device, batches)
        placed: dict[int, bool] = {}  # MoE layer -> routed weights die-placed
        for layer in layers:
            index, module = layer.layer_idx, layer.mlp
            boxes = (mailbox, small_batch, twoshot)
            self.attention[index] = Attention(layer, self, prefetch, *boxes)
            if not layer.is_moe_layer:
                reduce = all_reduce if module.down_proj.reduce_results else None
                ring = None
                if reduce is not None and dense_batches:
                    # [TP, cap, 7168] slots; this rank publishes into slot ``rank``.
                    box = OprojMailbox(*owner, (1, tp, PREPARED_MAX, 7168))
                    self.dense_rings[index] = box, dense_batches
                    ring = box.mc + rank * PREPARED_MAX * 7168 * 2
                # Rows above GENERIC_ROWS run the stock dense MLP (cuBLAS wins there).
                prepared = [b for b in batches if b <= GENERIC_ROWS]
                self.feed_forward[index] = PreparedDenseMLP(
                    module, prepared, reduce, ring, dense_batches if ring else (),
                    rows=min(self.capacity, GENERIC_ROWS),
                )  # fmt: skip
                continue
            args = (module, packed[index], front_box, state.front, tuned_moe, beta)
            shared = dict(plan=state.plan, counters=state.shared, q=self.q)
            shared["routes"] = route_mailbox(state.experts["workspace"])
            front = SharedFront(*args, linear_beta, **shared)
            routed = module.experts.routed_experts
            dies = place(routed)  # Rubin: w13 / w2 alternating dies, die matching
            placed[index] = bool(dies)
            if dies:
                logger.info_once(
                    "Kimi K3 MoE: routed w13 / w2 on alternating dies, die matching "
                    "(%d of %d SMs on die 1)",
                    sum(dies),
                    len(dies),
                    scope="process",  # every rank: place() decides per rank
                )
            w13, w2 = (
                w.view(torch.uint8) for w in (routed.w13_weight, routed.w2_weight)
            )
            weights = w13, routed.w13_weight_scale, w2, routed.w2_weight_scale
            experts = PersistentExperts(
                *weights, beta=beta, linear_beta=linear_beta, dies=dies, q=self.q
            )
            norm = module.routed_expert_norm.weight
            up = module.routed_expert_up_proj.weight
            op = module.experts._k3_latent_moe_tail_op
            placement = tail_box, moe_layers.index(index), len(moe_layers)
            tail = PreparedTail(op, norm, up, tuned_moe, placement)
            rows = None
            if wide_moe:
                if not all(stock_partials_ok(module, b) for b in wide_moe):
                    raise RuntimeError(
                        f"wide MoE batches {wide_moe}: no stock partials"
                    )
                eps = op.contract.rms_eps
                ordinal = moe_layers.index(index)
                rows = PreparedTailRows(
                    tp, rank, up, norm, eps, wide_moe, state.rows, ordinal,
                    sharded=(self.sharded, self.shard_box),
                    deferred=deferred_rows(module, wide_moe),
                    experts_push=wide_moe if WIDE_EXPERTS else (),
                )  # fmt: skip
            front_rows = None
            if wide_moe and WIDE_EXPERTS and WIDE_FRONT:
                front_rows = FrontRows(
                    module, packed[index], state.front_rows, state.front, wide_moe,
                    beta, linear_beta, plan=state.plan, q=self.q,
                )  # fmt: skip
            moe = PreparedMoE(
                front, experts, tail, state, prefetch=prefetch, layer_idx=index,
                rows=rows, module=module, front_rows=front_rows,
            )  # fmt: skip
            moe.small_batch = small_batch_chain.SmallBatchChain(
                small_batch, module, experts, tail, index, beta, linear_beta, prefetch
            )
            self.feed_forward[index] = moe
        # A layer left die-oblivious beside die-placed ones runs a slower experts kernel
        # in every graph, and every rank waits for it: say so once per rank, with all of
        # them.
        oblivious = sorted(i for i, p in placed.items() if not p)
        if oblivious and len(oblivious) < len(placed):
            logger.warning(
                "Kimi K3 MoE die placement: %d of %d MoE layers die-oblivious on rank "
                "%d (layers %s)", len(oblivious), len(placed), rank, oblivious,
            )  # fmt: skip
        # MoE tails followed by a layer whose input projection preloads its weights
        # before its PDL wait (a KdaHead; at verify q > 1 also a tensor-memory MLA
        # in-projection) release that layer's AttnRes early.
        for layer, after in zip(layers, layers[1:]):
            attention = self.attention[after.layer_idx]
            heads = head_batches(attention, self.q)
            if not heads and self.q > 1:
                heads = attention.tmem_batches
            moe = self.feed_forward.get(layer.layer_idx)
            if heads and isinstance(moe, PreparedMoE):
                moe.tail.tail_weights.use_heads(heads)

    def prepare_kv(self):
        """Compile each layer's cores for its bound KV cache layout (the profiling
        cache, then the real one)."""
        for layer in self.layers:
            caches = caches_of(layer.self_attn)
            self.attention[layer.layer_idx].prepare_layout(caches)
        # KdaHead and the tensor-memory MLA in-projection preload their own weights:
        # no next-attention prefetch before them
        for layer, after in zip(self.layers, self.layers[1:]):
            attention = self.attention[after.layer_idx]
            if not layer.is_moe_layer:
                continue
            heads = head_batches(attention, self.q)
            if heads:
                self.prefetch.skip_before(layer.layer_idx, heads, BEFORE_HEAD)
            if attention.tmem_batches:
                batches = attention.tmem_batches
                self.prefetch.skip_before(layer.layer_idx, batches, BEFORE_TMEM_MLA)
        # Step switch batches (SWITCH): every MLA layer has both cores prepared.
        mla = [a for a in self.attention.values() if a.kind == "mla"]
        self.switched = {
            b: bound
            for b, bound in SWITCH.items()
            if mla
            and all(
                a.core_long is not None
                and b in a.core_long.lse_fold
                and b in a.core.rubin_cores
                for a in mla
            )
        }
        if self.switched and self.switch is None:
            self.switch = GraphSwitch(self.device)
        if self.coverage is None:
            self.coverage = self.plan_coverage()
            for line in self.coverage.lines() if self.rank == 0 else ():
                logger.info("Kimi K3 decode coverage %s", line)

    def plan_coverage(self):
        """Each captured decode graph's path: the families' choices where the step runs
        specialized, else why it runs the stock forward."""
        coverage, q = Coverage(self.q), self.q
        rows = self.captured or tuple(b for b in self.batches if b % q == 0)
        for b in rows:
            if b not in self.batches or b % q:
                why = (
                    "not in WIDE_ROWS"
                    if b > PREPARED_MAX and self.context.arch == "rubin" and q > 1
                    else f"no prepared providers above {PREPARED_MAX} rows"
                )
                coverage.stock(b, why)
            elif b in self.wide:
                coverage.specialize(b, self.wide_choices(b, q))
            else:
                coverage.specialize(b, self.choices(b, q))
        return coverage

    def wide_choices(self, b, q):
        """The families' choices at a wide batch (``Attention.run_wide``, S4)."""
        generic, stock = Path.GENERIC, Path.STOCK
        kinds = {a.kind for a in self.attention.values()}
        if b in self.stock_attn_res:
            attn_res = Choice(stock, note="stock")
        else:
            mode = "sharded" if b in self.sharded else "plain"
            attn_res = Choice(generic, note=mode)
        choices = {"embed": Choice(stock), "attn_res": attn_res}
        gemm = "splitk" if b <= GENERIC_ROWS else "cublas"
        if "kda" in kinds:
            core = kda_choose(Shape(b, q), self.context)
            choices["kda"] = Choice(core.path, note=f"{gemm} + {core.note}")
        mla = next((a for a in self.attention.values() if a.kind == "mla"), None)
        if mla is not None and b in mla.mla_wide:
            choices["mla"] = mla_choose(Shape(b, q), self.context)
        elif mla is not None:
            choices["mla"] = Choice(stock, note=f"{gemm} + S2")
        attention = next(iter(self.attention.values()), None)
        if attention is not None and b in attention.oproj_wide:
            choices["oproj"] = oproj_choose(Shape(b, q), self.context)
        else:
            choices["oproj"] = Choice(stock, note="S3")
        if any(isinstance(m, PreparedMoE) for m in self.feed_forward.values()):
            if b in self.moe_batches:
                choices["moe"] = moe_choose(Shape(b, q), self.context)
            else:
                choices["moe"] = Choice(stock, note="S4")
        if any(isinstance(m, PreparedDenseMLP) for m in self.feed_forward.values()):
            dense = Choice(generic, note="splitk") if b <= GENERIC_ROWS else None
            choices["dense"] = dense or Choice(stock, note="cublas")
        return choices

    def choices(self, b, q):
        """The families' choices at a prepared batch: TUNED where the tables were
        measured (q8 verify or plain decode at 8 / 16 / 32 / 64 rows), else GENERIC."""
        path = Path.TUNED if q in (1, 8) and b in BATCHES else Path.GENERIC
        kda = next((a for a in self.attention.values() if a.kind == "kda"), None)
        mla = next((a for a in self.attention.values() if a.kind == "mla"), None)
        choices = {"embed": Choice(path, note="ring" if b in self.head_batches else "")}
        if kda is not None:
            head = kda.head
            if head is not None and q > 1 and (b, q) in head.verify_kernels:
                note = "head_verify"
            elif head is not None and q == 1 and b in head.kernels:
                note = "head"
            elif kda.fold64 is not None and q > 1 and b == 8 * q:  # as Attention.run
                note = f"fold {b // q}x{q}"
            else:
                capability = torch.cuda.get_device_capability(self.device)
                note = f"{verify_kernel(b // q, q, capability)} {b // q}x{q}"
            choices["kda"] = Choice(path, note=note)
        if mla is not None:
            rows = q > 1 and b in mla.rows
            choices["mla"] = Choice(path, note="rows" if rows else "")
        attention = kda or mla
        if attention is not None:
            output = attention.output
            codes = output.codes.get(b, ())
            two = "two-shot" if b in output.twoshots else "one-shot"
            choices["oproj"] = Choice(path, note="fused" if len(codes) == 1 else two)
        moe = next(
            (m for m in self.feed_forward.values() if isinstance(m, PreparedMoE)), None
        )
        if moe is not None:
            experts = moe.experts
            if b in experts.rubin:
                note = "image8"
            elif rubin64(experts.tp, b, experts.device):
                note = "rubin64"
            else:
                note = "grouped"
            choices["moe"] = Choice(path, note=note)
        return choices

    def embed_input_ids(self, ids):
        rows = ids.shape[0] if ids.ndim == 1 and ids.is_contiguous() else 0
        if not 1 <= rows <= PREPARED_MAX:
            return self.backbone.embed_input_ids(ids)
        table = self.backbone.embed_tokens.weight
        return self.embedding.run(ids, table.new_empty((ids.shape[0], table.shape[1])))

    def plan(self, positions, intermediate_tensors):
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (``intermediate_tensors``);
        - token counts without prepared providers (above 64 or not in ``batches``);
        - steps outside a CUDA graph capture (profiling, warmup and the warmup run
          before each capture), so every stock kernel runs before its capture;
        - PIECEWISE steps, keyed by token count alone and so captured as stock;
        - steps with prompt rows (prefills, chunked prefills, fresh prompts);
        - speculative steps whose requests do not all verify q tokens (partial
          drafts, verify rows mixed with plain decode rows).

        Every predicate is the same on all TP ranks: the mailboxes of the specialized
        decode need every rank to take the same path.
        """
        b = positions.shape[0]
        if intermediate_tensors is not None or b not in self.batches:
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        by_layer = context.attn_metadata
        rows: list[Any] = list(by_layer.values()) if isinstance(by_layer, dict) else []
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in rows) else 1
        if b % q or (b in self.wide and q == 1):  # wide batches: verify steps only
            return None
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in self.layers]
        if any(md is None for md in metadata):
            return None
        return Step(q, metadata, context.slot_mapping, positions)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step. Batches of
        ``switched`` (captured steps only) capture the step twice under an IF/ELSE
        node: body 0 with the custom MLA cores while every request's local length is
        within the bound, body 1 with the long-context cores (mla.prepared.SWITCH)."""
        x = inputs_embeds if inputs_embeds is not None else input_ids
        bound = self.switched.get(x.shape[0]) if step.q == 1 else None
        if bound is None or not torch.cuda.is_current_stream_capturing():
            return self.run_path(step, input_ids, inputs_embeds)
        # Both bodies write the same outputs; nothing is allocated inside them.
        b = x.shape[0]
        empty = partial(torch.empty, device=self.device, dtype=torch.bfloat16)
        if inputs_embeds is None:
            x = empty((b, 7168))
        out = empty((b, 7168))
        buffers = x, out, [empty((b, 7168)) for _ in self.aux_layers]
        lengths = self.local_lengths(step, b)
        self.switch.begin(lengths, bound)
        seconds = []
        for path in (0, 1):
            start = time.perf_counter()
            with self.switch.body(path):
                result = self.run_path(step, input_ids, inputs_embeds, path, buffers)
            seconds.append(time.perf_counter() - start)
        if any(self.switch.allocations[-2:]):
            raise RuntimeError(
                "Kimi K3 decode switch: a body allocated memory "
                f"({self.switch.allocations[-2:]} allocations)"
            )
        logger.info_once(
            "Kimi K3 decode: B%d step captured as a switch on the local length <= %d"
            " (body captures %.2f s, %.2f s)",
            b,
            bound,
            *seconds,
        )
        return result

    def local_lengths(self, step, b):
        """The step's DCP-local request lengths (int32 [b]), read by the switch."""
        for layer, md in zip(self.layers, step.metadata):
            if self.attention[layer.layer_idx].kind == "mla":
                lengths = md.decode.seq_lens
                return lengths[: min(b, lengths.numel())]
        raise RuntimeError("Kimi K3 decode switch: no MLA layer")

    def verify_rows(self, step, b):
        """Per-row MLA metadata of a verify step whose MLA layers run as single-query
        rows (mla.rows): one kernel per step, before the layers."""
        if step.q == 1 or b not in self.rows_batches:
            return
        md = next(
            md
            for layer, md in zip(self.layers, step.metadata)
            if self.attention[layer.layer_idx].kind == "mla"
        )
        decode = md.decode
        table, totals = decode.block_table, decode.dcp_tot_seq_lens
        requests = min(decode.seq_lens.numel(), table.shape[0], totals.numel())
        key = "mla.rows", b, table.shape[1]
        if key not in self.buffers:
            i32 = dict(dtype=torch.int32, device=self.device)
            self.buffers[key] = (
                torch.zeros((b, table.shape[1]), **i32),  # (mla.rows: entries past
                torch.empty((b,), **i32),  # a row's prefix keep valid pages)
                torch.empty((b,), **i32),
            )
        rows, box = self.buffers[key], self.dcp_box
        kernel = compile_rows(step.q, b, box.size, box.rank, box.interleave)
        kernel(
            dynamic_view(table, 4),
            dynamic_view(totals, 4),
            dynamic_view(rows[0], 4),
            *(to_cute(t, 4) for t in rows[1:]),
            Int32(requests),
            current_stream(self.device),
        )
        self.rows_meta[b] = rows

    def run_path(self, step, input_ids, inputs_embeds, path=0, buffers=None):
        """One body of the step: ``path`` 1 runs the MLA layers' long-context cores;
        ``buffers``: preallocated (x, out, aux_out) (the switch's bodies)."""
        x, ids = inputs_embeds, input_ids
        rows = x if x is not None else ids
        self.verify_rows(step, rows.shape[0])
        head_ring = (
            x is None
            and ids.ndim == 1
            and ids.is_contiguous()
            and ids.shape[0] in self.head_batches
        )
        if head_ring:
            # The rows go to the first AttnRes's mailbox; x only carries the shape.
            self.embedding.run_ring(ids)
            if buffers is not None:
                x = buffers[0]
            else:
                x = torch.empty(
                    (ids.shape[0], 7168), dtype=torch.bfloat16, device=ids.device
                )
        elif x is None and buffers is not None:
            x = self.embedding.run(ids, buffers[0])
        elif x is None:
            x = self.embed_input_ids(input_ids)
        if buffers is not None:
            out, aux_out = buffers[1], list(buffers[2])
        else:
            out = torch.empty_like(x)
            # The stock aux streams: the AttnRes prefix entering each aux layer.
            aux_out = [torch.empty_like(x) for _ in self.aux_layers]
        aux = dict(zip(self.aux_layers, aux_out))
        b, q, slots = x.shape[0], step.q, step.slot_mapping
        prefix, blocks = self.prefix[:b], self.blocks[:b]
        first, second = self.hidden[0][:b], self.hidden[1][:b]
        hidden, pre_ready = x, False
        stock_res = b in self.stock_attn_res  # AttnRes sites on the stock op
        for layer, md in zip(self.layers, step.metadata):
            index, caches = layer.layer_idx, caches_of(layer.self_attn)
            attn_in = first  # the attention's input rows
            if not pre_ready and head_ring and layer is self.layers[0]:
                self.before[index].run_ring(prefix, blocks, first, aux.get(index))
            elif stock_res and not pre_ready:
                # (A site the previous MoE tail's ring consumer already ran is done:
                # running the stock op on its output would apply the site twice.)
                stock = self.before_stock[index]
                attn_in = stock.run(hidden, prefix, blocks, aux.get(index))
            elif not pre_ready:
                self.before[index].run(hidden, prefix, blocks, first, aux.get(index))
            # The MoE input rows: ``second``, or the stock post AttnRes's output.
            moe_in = self.attention[index].run(
                attn_in, second, prefix, blocks, caches, md, slots, q, path,
                positions=step.positions,
            )  # fmt: skip
            consumer = self.fused_before.get(index)
            if index in self.dense_rings and b not in self.dense_rings[index][1]:
                consumer = None
            stock_moe = b in self.wide and b not in self.moe_batches
            if stock_moe:  # the next AttnRes reads the plain MLP output
                consumer = None
            pre_ready = consumer is not None
            ff = self.feed_forward[index]
            if stock_moe and isinstance(ff, PreparedMoE):
                hidden = layer.mlp(moe_in)  # S4: the stock MoE layer
            elif b > GENERIC_ROWS and isinstance(ff, PreparedDenseMLP):
                hidden = layer.mlp(moe_in)  # the stock dense MLP (cuBLAS)
            elif consumer is not None:
                target = out if consumer is self.final else first
                stream = aux.get(index + 1)
                consume = partial(consumer.run_ring, prefix, blocks, aux=stream)
                hidden = self.feed_forward[index].run(moe_in, target, consume=consume)
            else:
                hidden = self.feed_forward[index].run(moe_in, first)
                first, second = second, first
        if stock_res and not pre_ready:  # the stock final AttnRes + the stock norm
            final = self.final_stock.run(hidden, prefix, blocks, aux.get(index + 1))
            out = self.normalize(final)
        elif not pre_ready:
            self.final.run(hidden, prefix, blocks, out, aux.get(index + 1))
        self.prefetch.join()
        self.kda_side.join()
        return (out, aux_out) if aux_out else out

    def normalize(self, output):
        """A stock step's output with the backbone's final norm applied, as the
        specialized decode's last AttnRes applies it; the aux streams stay pre-norm.

        Runs in chunks of at most 255 rows: the stock RMSNorm kernel reduces with
        1024-thread blocks below 256 rows and 256-thread blocks above, so chunking keeps
        the reduction order ``compute_logits`` uses for the few sampled rows, and
        prefill logits match stock.
        """
        hidden, aux = output if isinstance(output, tuple) else (output, None)
        norm = self.backbone.norm
        if hidden.ndim == 2 and hidden.shape[0] >= 256:
            chunks = range(0, hidden.shape[0], 255)
            hidden = torch.cat([norm(hidden[i : i + 255], None) for i in chunks])
        else:
            hidden = norm(hidden, None)
        return hidden if aux is None else (hidden, aux)
