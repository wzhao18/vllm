# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP o-projection of an online-MXFP8 ``o_proj`` (the NVFP4 ``modelopt_mixed``
checkpoint's KDA layers) with the BF16 path's all-reduce and post-attention AttnRes.

``PreparedOproj`` fuses a BF16 weight-stationary GEMM with its exchange; an MXFP8
weight instead runs the stock layer's GEMM, then pushes its partial into the same
mailboxes, which the same consumers read:

1. ``gemm.mxfp8.Mxfp8Projection``: the KDA core output quantized to MXFP8 (bitwise
   FlashInfer's ``mxfp8_quantize``, as stock's linear), then the MXFP8 GEMM of stock's
   kind at R rows: this rank's partial [R, 7168] in FP32, rounded once to BF16 (stock's
   ``o_proj`` output before its all-reduce, bit for bit);
2. ``OprojPush``: the partial's 16-byte chunks into the mailbox the BF16 producer
   fills at that batch, as it fills them (-0 halves made +0, the Lamport sentinel):
   "oneshot" multicasts into slot ``rank`` of the parity's [TP, 64, 7168] box,
   "twoshot" stores into the column owner's receive slot (``common.twoshot.push``),
   "owners" (row-sharded AttnRes) into the row owner's slot;
3. the BF16 path's consumer kernels unchanged (``compile_oproj``): the TP slots summed
   in FP32 in rank order and rounded once, the AttnRes, the next MoE front's x_moe rows
   and epoch hint (``front``), the re-arms. Two-shot batches (``TWOSHOT``) and the wide
   rows (``choose``) take the consumer whose column owners reduce (``reduced=False``):
   no MXFP8 producer reduces.

The delta is stock's ``o_proj`` + all-reduce up to the summation order (FP32, one
rounding); the AttnRes is the BF16 path's. No fused, polled or KdaHead schedule
(KdaHead fuses a BF16 in-projection, ``providers.Attention``)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Unsupported
from ..attn_res.sharded import compile_gather, owned
from ..common.host import dynamic_view, fake, prepare_shared_carveout, stream, to_cute
from ..common.primitives import (
    at,
    launch,
    load_vector,
    multicast_words,
    no_neg_zero,
    store_relaxed,
    thread,
)
from ..common.twoshot import push
from ..gemm.mxfp8 import Mxfp8Projection
from .prepared import (
    OPROJ_ROWS,
    compile_oproj,
    consumer_carveout,
    oproj_vote,
    twoshot_batched,
    twoshot_batches,
)

WIDTH = 7168  # output features (the hidden size)
CHUNKS = WIDTH // 8  # 16-byte chunks per row
THREADS = 256
MODES = ("oneshot", "twoshot", "owners")


class OprojPush:
    """One thread per 16-byte chunk of an R-row BF16 partial: the chunk's words, -0
    halves made +0, into the mailbox of ``mode`` (module docstring). Arguments:
    (partial [R, 7168], box, target, slot, stream) with box / target / slot:
    "oneshot" the [TP, cap, 7168] parity view, its multicast address, this rank;
    "twoshot" the [TP, cap, 7168 / TP] receive view, the owners' bases, this rank;
    "owners" the [TP, per_cap, 7168] owners' slots, their bases, this rank."""

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"o-projection push mode {mode!r} (modes {MODES})")
        self.mode = mode

    @cute.jit
    def __call__(self, *args):
        grid = (args[0].shape[0] * CHUNKS + THREADS - 1) // THREADS
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        partial, box, target, slot = args
        tid, _, _, cta = thread()
        cute.arch.griddepcontrol_wait()  # the GEMM's partial
        cute.arch.griddepcontrol_launch_dependents()
        rows: cutlass.Constexpr = partial.shape[0]
        g = cta * THREADS + tid
        if g < rows * CHUNKS:
            token, column = g // CHUNKS, g % CHUNKS * 8
            loaded = load_vector(at(partial, token, column))
            words = tuple(no_neg_zero(word) for word in loaded)
            if cutlass.const_expr(self.mode == "oneshot"):
                offset = cute.crd2idx((slot, token, column), box.layout)
                multicast_words(target + Int64(offset) * 2, words)
            elif cutlass.const_expr(self.mode == "twoshot"):
                push(box, target, slot, token, column, words)
            else:
                per: cutlass.Constexpr = (rows + box.shape[0] - 1) // box.shape[0]
                owner = token // per
                offset = cute.crd2idx((slot, token - owner * per, column), box.layout)
                store_relaxed(Int64(target[owner]) + Int64(offset) * 2, words)


@cache
def compile_push(mode, rows, tp, cap, device_index):
    """``OprojPush(mode)`` for an R-row partial; ``cap``: the box's rows (per owner
    for "owners")."""
    with torch.accelerator.device_index(device_index):
        partial = fake((rows, WIDTH))
        if mode == "oneshot":
            box, target = fake((tp, cap, WIDTH)), Int64(0)
        else:
            columns = WIDTH // tp if mode == "twoshot" else WIDTH
            box, target = fake((tp, cap, columns)), fake((tp,), None, torch.int64)
        kernel = OprojPush(mode)
        return cute.compile(kernel, partial, box, target, Int32(0), make_fake_stream())


class Mxfp8Oproj:
    """``PreparedOproj``'s interface (``run``, ``codes``, ``twoshots``) for a KDA
    layer whose ``o_proj`` is online MXFP8 (module docstring). ``batches``: the
    tuned (<= 64 rows) and wide rows; ``front`` / ``twoshot`` / ``x_buffer`` /
    ``sharded`` as ``PreparedOproj``; ``partial``: a [cap, 7168] BF16 scratch (one per
    model: layers run one after another); ``projection``: the MXFP8 GEMM (default:
    ``decoder.self_attn.o_proj``'s own weight)."""

    def __init__(
        self, decoder, mailbox, batches, front=None, twoshot=None, x_buffer=None,
        sharded=None, *, partial, projection=None,
    ):  # fmt: skip
        batches = tuple(sorted(set(batches)))
        if projection is None:
            projection = Mxfp8Projection(
                decoder.self_attn.o_proj, batches, family="o_proj", cap=batches[-1]
            )
        self.gemm = projection
        if projection.n != WIDTH:
            raise Unsupported("o_proj", None, f"MXFP8 o_proj N {projection.n}")
        if partial.shape[0] < batches[-1] or partial.shape[1:] != (WIDTH,):
            raise Unsupported("o_proj", batches[-1], f"partial {tuple(partial.shape)}")
        self.partial = partial
        norm, out_norm = decoder.mlp_res_norm, decoder.post_attention_layernorm
        weights = (norm.weight, decoder.mlp_res_proj.weight.squeeze(0), out_norm.weight)
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.eps = tuple(cutlass.Float32(n.variance_epsilon) for n in (norm, out_norm))
        parity, rank, tp = decoder.layer_idx % 2, mailbox.rank, mailbox.tp
        self.device = device = partial.device
        index, k = device.index, projection.k
        nb = decoder.prev_valid_blocks + decoder.is_block_write_layer
        delta = not decoder.is_block_write_layer
        self.front = None if front is None else (*map(to_cute, front[:2]), front[2])
        local_box, mc = mailbox.slot(parity)
        self.twoshots = twoshot_batches(tp) if twoshot is not None else set()
        two = None
        if twoshot is not None:
            recv, peers, gather, gather_mc = twoshot.views(parity)
            two = (to_cute(recv), to_cute(peers), to_cute(gather), Int64(gather_mc))
        # batch -> (push kernel, its operands, consumer kernel, consumer head, tail);
        # ``codes`` (coverage): batch -> (push, consumer)
        self.plans: dict[int, tuple] = {}
        self.owners: dict[int, tuple] = {}  # row-sharded wide batches
        self.codes: dict[int, tuple] = {}
        rank_cute = Int32(rank)
        wide = [b for b in batches if b > OPROJ_ROWS]
        if wide:
            cap = x_buffer.shape[0] if x_buffer is not None else 0
            if twoshot is None or cap < wide[-1]:
                raise Unsupported(
                    "o_proj", wide[-1], f"wide MXFP8 o-projection without a two-shot "
                    f"box of {wide[-1]} rows (cap {cap})",
                )  # fmt: skip
            box = twoshot.rows_box(cap)
            recv, peers, gather, gather_mc = box.views(parity)
            operands = (to_cute(recv), to_cute(peers), rank_cute)
            tail = (to_cute(recv), Int64(gather_mc), rank_cute)
            shard = sharded[1] if sharded is not None else None
            for b in wide:
                if shard is not None and b in sharded[0]:
                    shape = (k, b, tp, nb, delta, False, False, shard.per_cap)
                    owners = to_cute(shard.owners.buffer)
                    codes = (
                        compile_push("owners", b, tp, shard.per_cap, index),
                        compile_oproj("consumer_owners", *shape),
                    )
                    head = (owners, to_cute(shard.owner_peers), rank_cute)
                    self.owners[b] = codes, head, owners, shard
                    self.codes[b] = codes
                    continue
                shape = (k, b, tp, nb, delta, False, False, cap, True)
                codes = (
                    compile_push("twoshot", b, tp, cap, index),
                    compile_oproj("consumer", *shape),  # its owners reduce
                )
                self.plans[b] = (*codes, operands, to_cute(gather), tail)
                self.codes[b] = codes
        for b in batches:
            if b > OPROJ_ROWS:
                continue
            publish = front is not None and b in front[2]
            twoshot_b = b in self.twoshots
            vote = oproj_vote(tp, b, device) and not twoshot_b
            batched = twoshot_b and twoshot_batched(tp, b, device)
            shape = (k, b, tp, nb, delta, False, publish, 64, twoshot_b)
            consumer = compile_oproj(
                "consumer", *shape, False, False, False, batched, False, False, 1, vote
            )
            carve = consumer_carveout(tp, b, device) if twoshot_b else None
            if carve is not None:
                prepare_shared_carveout(consumer, index, carve)
            if twoshot_b:
                pusher = compile_push("twoshot", b, tp, 64, index)
                operands = two[:2] + (rank_cute,)
                head, tail = two[2], (two[0], two[3], rank_cute)
            else:
                pusher = compile_push("oneshot", b, tp, 64, index)
                operands = (to_cute(local_box), Int64(mc), rank_cute)
                head, tail = to_cute(local_box), ()
            self.plans[b] = (pusher, consumer, operands, head, tail)
            self.codes[b] = (pusher, consumer)

    def run(self, local, prefix, blocks, out, after_head=False, polled=False):
        batch, s = local.shape[0], stream(self.device)
        if after_head or polled:
            raise Unsupported(
                "o_proj", batch, "MXFP8 o-projection behind KdaHead or polling x"
            )
        partial = self.partial[:batch]
        self.gemm.run(local, partial)
        if batch in self.owners:
            return self.run_owned(partial, prefix, blocks, out, s)
        pusher, consumer, operands, head, tail = self.plans[batch]
        pusher(to_cute(partial), *operands, s)
        front = (None, None)
        if self.front is not None and batch in self.front[2]:
            front = self.front[:2]
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), *front, *self.eps)
        consumer(head, *rows, *tail, s)
        return out, prefix, blocks

    def run_owned(self, partial, prefix, blocks, out, s):
        """Row-sharded (``PreparedOproj.run_owned``): rows pushed to their owners,
        this rank's rows reduced and AttnRes'd into gather box 0, every row gathered
        into ``out``."""
        codes, head, owners, shard = self.owners[partial.shape[0]]
        rows_total = partial.shape[0]
        first, count = owned(rows_total, shard.tp, shard.rank)
        codes[0](to_cute(partial), *head, s)
        if count:
            mine = slice(first, first + count)
            views = (dynamic_view(prefix[mine]), dynamic_view(blocks[mine]))
            rows = (*views, *self.views, views[0], None, None, *self.eps)
            mc = Int64(shard.gathers[0].mc)
            codes[1](owners, *rows, mc, Int32(first), s)
        gather = compile_gather(rows_total, shard.cap, self.device.index)
        gather(to_cute(shard.gathers[0].buffer), to_cute(out), s)
        return out, prefix, blocks

    def choice_note(self, batch):
        two = batch in self.twoshots or batch > OPROJ_ROWS
        mode = "owners" if batch in self.owners else "two-shot" if two else "one-shot"
        return f"mxfp8 {self.gemm.engine(batch)} + push {mode}"
