# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's KDA core (C512: TP4, 24 heads per rank, FP32 state, q = 4).

``SpKdaCore`` runs one KDA layer's verify on the TP group's R = 4 n rows (n requests
of q tokens, n = 1..128 per DP group, 0.25..32 per GPU) from the packed in-projection
rows ``[q | k | v | g | f_a | beta | pad]`` to the gated, RMS-normed output that the
stock o_proj consumes (stock: the RecoverSSM verify, ``KimiK3DeltaAttention.
_recoverssm_verify``, then ``o_norm``).

RecoverSSM (``--use-replayssm``, ``recoverssm.prepare_recoverssm``'s mode): the verify
leaves each request's FP32 checkpoint as it is, rolls the conv window and writes the
records the commit after sampling folds the accepted tokens in with: the correction
[slot, H, q, V] (FP32) and the key and raw gate [slot, H, q, 2K] (BF16).

==========  ======================================================================
mode        cores (ported from the upstream specialized decode)
==========  ======================================================================
"triton"    record cores, records at the step row (the stock layout), the stock
            Triton commit after sampling (unchanged)
"deferred"  record cores at the request's record slot (``record_rows``); KdaWide
            folds the pending commit into its own pass with the stock Triton
            commit's math bit for bit (``wide`` docstring), a replaying core
            (``REPLAY``) in its own form; before the other cores the step's first
            KDA core folds every layer's pending commit
            (``KDARecoverSSMDeferred.flush``)
"none"      no records: the plain verify with q state slots (``prepared.
            PreparedKda``, the TP8 / TP16 kernels at 24 heads)
==========  ======================================================================

Precision: every record core has the stock Triton verify's math (``recover.KdaSplit``
docstring): BF16 conv / SiLU outputs, raw gates and recurrence outputs, FP32 state and
recurrence, the Triton formulas and instructions; only FP32 reduction order differs.
The replaying cores (``fused``: chunked WY form) also round the recurrence in FP32 in
another form, so ``REPLAY`` is empty until that is signed off.

Record cores by requests n at 24 heads (``items = 24 n``), measured on VR200 (SM107):
q4, FP32 state, "triton" records, one CUDA graph of 69 layers over 8 disjoint slot
sets (state working set above L2), every core at n = 1..16, 24, 32, 36, 40, 48, 52,
56, 60, 64, 68..72, 76, 80, 88, 96 (us per layer, the winner first):
n1 / n2 KdaSplit 3.8 / 4.7 (KdaWide 5.6); n3..n7 KdaWide 5.7..6.2 (KdaSplit
6.0..11.3); n8..n15 KdaWide 6.2..9.1 (KdaWS 10.2..11.2); n16..n70 KdaWide (n32 18.1
against KdaStream 19.8, n52 27.4 against KdaWS 29.6, n70 36.2 against 36.5); from n71
(1704 items, past eight per SM) KdaWS (n71 39.1 against 39.3, n80 42.3 against 43.2,
n96 45.7 against 49.5). KdaStream wins nowhere. The cores differ from each other only
in FP32 reduction order (not bit for bit; each one's error against the stock verify is
the same, the chain test's bound). REPLAY (KdaFused / KdaFusedWS) in "deferred" mode
once its precision is accepted.

"deferred" (the serving mode: every request has 1..q pending tokens to fold), the same
graph with the pending commits, us per layer (KdaWS without its flush, which adds a
full state pass): n8 KdaWide 8.6 (KdaFlow 10.5); n16 KdaWide 13.3 (KdaFlow 14.0);
from n24 KdaFlow, which folds in its own single state pass: n24 20.2 (KdaWide 24.5),
n32 21.8 (26.1), n48 29.8 (39.2), n64 38.9 (51.7), n71 44.2 (56.7; KdaWS 41.3 + flush),
n96 53.6 (74.3; KdaWS 48.7 + flush), n128 71.0 (95.2; KdaWS 62.6 + flush).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Choice, Path, Shape, Unsupported
from ..common.host import fake, layout, stream, to_cute
from ..common.policy import batch_table
from . import chain, flow, fused_ws, ring, specialized
from .fused import KdaFused
from .fused_ws import KdaFusedWS
from .recover import KdaSplit
from .specialized import KdaWS
from .stream import KdaStream
from .wide import KdaWide

K = 128
# Record core per request count n, by heads per rank (items = heads x n, see above).
RECORD_KINDS = {24: "1-2:split,3-20:wide,21-:flow"}
# Record KdaSplit geometry ("early.lanes.occupancy.cluster", as prepared.SPEC_SCHEDULE)
# by n; empty: the batch defaults (eight-CTA clusters at n1, then four; 16 lanes).
SP_SPEC_SCHEDULE = {24: ""}
# "deferred": requests whose core replays the pending commit in its own pass
# ("fused_ws" where its shared memory fits, else "fused"); empty: every step folds the
# pending commit first (flush) and runs the record core (stock precision).
REPLAY = {24: ""}
# KdaFused CTAs per (request, head) by items (upstream FUSED_SPLIT).
FUSED_SPLIT = ((48, 8), (96, 4), (192, 2), (1 << 30, 1))
KINDS = ("split", "ws", "wide", "stream", "fused", "fused_ws", "flow", "chain")
# "deferred": cores that fold the pending commit themselves (no flush): the replaying
# cores, and KdaWide, whose prologue runs the stock Triton commit's math bit for bit.
FOLDS = ("fused", "fused_ws", "wide", "flow", "chain", "ring")
DIE_KINDS = ("flow", "ring")  # die-matched items, MXFP8 outputs
MODES = ("deferred", "triton", "none")


def record_kind(heads, requests):
    """The record core at ``requests`` (RECORD_KINDS), or None."""
    return batch_table(RECORD_KINDS.get(heads, "")).get(requests)


def replays(heads, requests):
    return bool(batch_table(REPLAY.get(heads, "")).get(requests, 0))


def stream_layout(specs, tokens):
    """(conv dim-first, window) when KdaWS / KdaStream / KdaFusedWS can cp.async this
    step's operands (16-byte aligned rows and slots), else None (upstream
    ``prepared.stream_layout``)."""
    (_, x_stride, _), conv = specs[0], specs[3]
    shape, stride, dtype = conv
    size = torch.empty((), dtype=dtype).element_size()
    window = shape[-1]
    if stride[-1] == 1 and stride[-2] == window:
        dim_first = True
    elif stride[-2] == 1:
        dim_first = False
    else:
        return None
    aligned = (stride[0] * size) % 16 == 0 and (x_stride[0] * 2) % 16 == 0
    if not dim_first:
        aligned = aligned and (stride[-1] * size) % 16 == 0
    return (dim_first, window) if aligned and window >= tokens + 2 else None


def smem_limit(device):
    """Dynamic shared memory one CTA may opt into on ``device``."""
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def persistent_ctas(items, sms, floor=2):
    """Persistent CTAs (one per SM) with an equal item count, at least ``floor``."""
    per = max(floor, -(-items // sms))
    return -(-items // per)


def fits_fused_ws(specs, tokens, device):
    """Whether KdaFusedWS can stage this step (cp.async layout, shared memory)."""
    streamed = stream_layout(specs, tokens)
    if streamed is None:
        return False
    conv_size = torch.empty((), dtype=specs[3][2]).element_size()
    size = fused_ws.shared_bytes(tokens, specs[15][0][2], *streamed, conv_size)
    return size <= min(fused_ws.SMEM_LIMIT, smem_limit(device))


def make_core(kind, specs, heads, tokens, record, has_bias, device):
    """The kernel object of ``kind`` for a step's operand layouts ``specs`` (no GPU
    work); raises ``Unsupported`` where the kind cannot run them. KdaFlow options
    ride on the kind: "flow+dies" (die-matched items), "+mx" (MXFP8 outputs)."""
    kind, *options = kind.split("+")
    requests = specs[0][0][0] // tokens
    items, mode = heads * requests, 0  # gate mode 0: one chain of eight F_B MMAs
    props = torch.cuda.get_device_properties(device)
    sms, limit = props.multi_processor_count, smem_limit(device)
    shape = Shape(requests * tokens, tokens)
    if kind == "split":
        table = batch_table(SP_SPEC_SCHEDULE.get(heads, ""))
        schedule = table.get(requests, (0, 0, 0))
        return KdaSplit(heads, requests, mode, has_bias, schedule, tokens, record)
    if kind == "wide":
        return KdaWide(heads, tokens, record, mode, has_bias, 2 * sms)
    if kind == "fused":
        split = next(s for bound, s in FUSED_SPLIT if items < bound)
        return KdaFused(heads, tokens, mode, has_bias, split)
    streamed = stream_layout(specs, tokens)
    if streamed is None:
        cause = "conv cache / projection rows not 16-byte cp.async aligned"
        raise Unsupported("kda", shape, f"{kind}: {cause}")
    conv_size = torch.empty((), dtype=specs[3][2]).element_size()
    if kind == "fused_ws":
        if not fits_fused_ws(specs, tokens, device):
            raise Unsupported("kda", shape, "KdaFusedWS's shared memory does not fit")
        ctas = persistent_ctas(items, sms)
        return KdaFusedWS(heads, tokens, mode, has_bias, *streamed, ctas)
    if kind in ("flow", "chain", "ring"):
        if not record or not 2 <= tokens <= 4:
            cause = "KdaFlow runs RecoverSSM record steps of 2..4 tokens only"
            raise Unsupported("kda", shape, cause)
        budget = flow.smem_budget(device)
        stages = flow.max_stages(tokens, *streamed, conv_size, budget)
        if not stages:
            cause = "KdaFlow needs a dim-first window and its shared memory"
            raise Unsupported("kda", shape, cause)
        ctas = min(items, sms)
        if -(-items // ctas) > flow.max_items_per_cta(requests):
            raise Unsupported("kda", shape, "KdaFlow: a CTA would span three heads")
        if kind == "ring":
            if len(specs) < 23:  # the deferred operands and the operand blocks
                raise Unsupported("kda", shape, f"KdaRing runs deferred records only ({len(specs)} operands)")
            if ring.shared_bytes(tokens) > budget:
                raise Unsupported("kda", shape, "KdaRing's shared memory does not fit")
            return ring.KdaRing(heads, tokens, mode, has_bias, *streamed, ctas, items,
                                dies="dies" in options, mx="mx" in options)  # fmt: skip
        if kind == "chain":
            if len(specs) != 20:  # with the record rows, pending counts, scratch
                raise Unsupported("kda", shape, "KdaChain runs deferred records only")
            return chain.KdaChain(
                heads, tokens, mode, has_bias, *streamed, ctas, items, stages
            )
        return flow.KdaFlow(heads, tokens, mode, has_bias, *streamed, ctas, stages,
                            dies="dies" in options, mx="mx" in options)  # fmt: skip
    if kind == "ws":
        size = specialized.shared_bytes(tokens, *streamed, conv_size)
        if size > min(specialized.SMEM_LIMIT, limit):
            raise Unsupported("kda", shape, f"KdaWS needs {size} B shared memory")
        ctas = persistent_ctas(items, sms)
        return KdaWS(heads, tokens, record, mode, has_bias, *streamed, ctas)
    if kind == "stream":  # two CTAs per SM, an equal item count each
        ctas = persistent_ctas(items, 2 * sms, 1)
        return KdaStream(heads, tokens, record, mode, has_bias, *streamed, ctas)
    raise Unsupported("kda", shape, f"unknown KDA core {kind!r}")


_CHAIN_SCRATCH: dict = {}
_CHAIN_RETIRED: list = []


def chain_scratch(device, items, grow=True):
    """KdaChain's operand-block scratch rows for items items: one buffer per
    device shared by every layer (the layers run in stream order), sized at build."""
    buffer = _CHAIN_SCRATCH.get(device)
    if buffer is None or buffer.shape[0] < items:
        if not grow:
            raise Unsupported("kda", None, f"KdaChain scratch for {items} items")
        shape = chain.scratch_shape(items)
        buffer = torch.empty(shape, dtype=torch.float32, device=device)
        _CHAIN_SCRATCH[device] = buffer  # (graphs captured on a smaller one keep it)
        _CHAIN_RETIRED.append(buffer)
    return buffer[:items]


_RING_SCRATCH: dict = {}


def ring_scratch(device, items, tokens, grow=True):
    """KdaRing's operand blocks for items items: one buffer per device shared by
    every layer (each layer's prep kernel fills it right before its ring), sized
    at build."""
    buffer = _RING_SCRATCH.get(device)
    shape = ring.scratch_shape(items, tokens)
    if buffer is None or buffer.shape[0] < items or buffer.shape[1] != shape[1]:
        if not grow:
            raise Unsupported("kda", None, f"KdaRing scratch for {items} items")
        buffer = torch.empty(shape, dtype=torch.float32, device=device)
        _RING_SCRATCH[device] = buffer  # (graphs captured on a smaller one keep it)
        _CHAIN_RETIRED.append(buffer)
    return buffer[:items]


@cache
def executable(specs, kind, heads, tokens, record, has_bias, device):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    kernel = make_core(kind, specs, heads, tokens, record, has_bias, device)
    scalars = (cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


def decode_conv_weight(weight):
    """The [3, width, D] (Q, K, V) decode layout of the [3 * D, 1, width] conv
    weight (its dtype: the stock FP32 conv weight stays FP32)."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


class SpKdaCore:
    """One KDA layer's verify core at every captured R (``rows``) of the SP decode.

    ``records``: ``recoverssm.prepare_recoverssm``'s mode (module docstring). Every R
    is served by a specialized core; a shape or layout the cores cannot run raises
    ``shapes.Unsupported`` at build (``__init__`` / ``prepare_caches``), never a stock
    fallback."""

    def __init__(self, attention, rows, *, records, ctx):
        if records not in MODES:
            raise Unsupported("kda", None, f"RecoverSSM records mode {records!r}")
        if not ctx.sequence_parallel:
            raise Unsupported("kda", None, "SpKdaCore serves the SP decode only")
        self.mode, self.q = records, ctx.q
        self.heads = heads = attention.A_log.numel()
        self.dim, self.has_bias = heads * K, attention.conv1d.bias is not None
        if attention.head_dim != K or attention.conv_size != 4:
            raise Unsupported("kda", None, "head dim 128 and a width-4 conv only")
        if ctx.heads_per_rank and ctx.heads_per_rank != heads:
            cause = f"{heads} KDA heads per rank, the context says {ctx.heads_per_rank}"
            raise Unsupported("kda", None, cause)
        if attention.spec_query_len != self.q or not 2 <= self.q <= 8:
            cause = f"verify length q{self.q} (layer: q{attention.spec_query_len})"
            raise Unsupported("kda", None, cause)
        if ctx.bf16_state or attention.get_state_dtype()[1] != torch.float32:
            raise Unsupported("kda", None, "the SP KDA core runs an FP32 SSM state")
        if attention.gate_lower_bound is None:
            raise Unsupported("kda", None, "a bounded KDA gate (gate_lower_bound)")
        # packed in-projection columns (12448 at C512): N of the BF16 [N, K] weight
        # or of the online-MXFP8 one (``gemm.mxfp8``: its own [N, K] FP8 rows)
        from ..gemm.mxfp8 import projection_rows

        self.pitch = projection_rows(attention.in_proj_qkvgfab)
        if self.pitch < 4 * self.dim + K + heads or self.pitch % 8:
            cause = f"in-projection pitch {self.pitch} (16-byte rows, >= 4 D + 128 + H)"
            raise Unsupported("kda", None, cause)
        self.rows = tuple(sorted(rows))
        for r in self.rows:
            if r <= 0 or r % self.q:
                shape = Shape(r, self.q)
                raise Unsupported("kda", shape, "rows are not whole requests")
        self.layer = attention
        self.device = dt = attention.dt_bias.device
        conv_w = decode_conv_weight(attention.conv1d.weight)
        bias, fb = attention.conv1d.bias, attention.f_b_proj.weight
        bias = attention.dt_bias if bias is None else bias  # (unread placeholder)
        norm = attention.o_norm.weight.detach().float()  # exact; BF16 slows the cores
        self.weights = (conv_w, bias, fb, attention.A_log, attention.dt_bias, norm)
        scalars = (attention.gate_lower_bound, 128**-0.5, attention.o_norm.eps)
        self.scalars = tuple(map(cutlass.Float32, scalars))
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        self.conv_dim_first = is_conv_state_dim_first()
        self.kinds, self.kernels, self.specs = {}, {}, {}
        self.plain = None
        # The o_proj's MXFP8 input scratch (gemm.mxfp8.Mxfp8Input), set before
        # prepare_caches: KdaFlow then writes its outputs there (run returns the view).
        self.mx_input = None
        self.die_ops = None
        if records == "none":
            if getattr(attention, "recoverssm_records", None) is not None:
                raise Unsupported("kda", None, "mode 'none' on a RecoverSSM layer")
            from .prepared import PreparedKda, verify_kernel

            capability = torch.cuda.get_device_capability(dt)
            keys = {(r // self.q, self.q): (self.pitch, 0) for r in self.rows}
            self.plain = PreparedKda(attention, keys, self.conv_dim_first)
            for r in self.rows:
                n = r // self.q
                self.kinds[r] = verify_kernel(n, self.q, capability, False, 0, heads)
            return
        found = getattr(attention, "recoverssm_records", None)
        if found is None:
            cause = f"mode {records!r} on a layer without records"
            raise Unsupported("kda", None, cause)
        installed = getattr(attention, "deferred_recoverssm", False)
        if records == "deferred" and not installed:
            raise Unsupported("kda", None, "deferred mode before install_layers")
        self.records = (found.correction, found.key_gate)
        slots = self.records[0].shape[0]  # request rows ("triton") / record slots
        if self.records[0].shape[1:] != (heads, self.q, K):
            raise Unsupported("kda", None, f"records {tuple(self.records[0].shape)}")
        for r in self.rows:
            n, shape = r // self.q, Shape(r, self.q)
            if records == "triton" and n > slots:
                raise Unsupported("kda", shape, f"{n} requests > {slots} record rows")
            replay = records == "deferred" and replays(heads, n)
            kind = "fused_ws" if replay else record_kind(heads, n)
            if kind is None:
                raise Unsupported("kda", shape, f"no record core at {heads} heads")
            self.kinds[r] = kind

    @torch.no_grad()
    def process_weights_after_loading(self):
        """Refill the derived conv weight and FP32 norm after a weight reload."""
        if self.plain is not None:
            conv_w, *_, norm = self.plain.weights
        else:
            conv_w, *_, norm = self.weights
        conv_w.copy_(decode_conv_weight(self.layer.conv1d.weight))
        norm.copy_(self.layer.o_norm.weight)

    def _operands(self, caches, projected, slots, out, accepted, deferred=()):
        """Kernel tensors in launch order (upstream ``PreparedKda._operands``): 13
        before the scalars, then F_B, the accepted counts, the records and (deferred)
        the record rows and pending counts."""
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        ops = (p(0, 3 * d), w, bias, conv, p(4 * d, K), p(4 * d + K, h), a_log, dt)
        tail = (fb, accepted, *self.records, *deferred)
        return ops + (slots, state, out, p(3 * d, d), norm) + tail

    def prepare_caches(self, caches):
        """Compile every R's core for the bound KV cache (before graph capture)."""
        if self.plain is not None:
            self.plain.prepare_caches(caches)
            return self
        state = caches[1]
        if self.mode == "deferred":
            from .commit import supported

            if not supported(state, self.layer.gate_lower_bound):
                cause = "the deferred FP32 commit cannot fold this state cache"
                raise Unsupported("kda", None, cause)
        meta = dict(device="meta")
        self.die_ops = None
        if self.mode == "deferred" and set(DIE_KINDS) & set(self.kinds.values()):
            n_cap = max(self.rows) // self.q
            self.die_ops = flow.die_operands(state, n_cap, self.heads)
        for r in self.rows:
            n, kind = r // self.q, self.kinds[r]
            projected = torch.empty((r, self.pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty((r, self.dim), dtype=torch.bfloat16, **meta)
            accepted = torch.empty(n, dtype=torch.int32, **meta)
            slots = torch.empty((n, 1), dtype=torch.int32, **meta)  # the checkpoints
            deferred: tuple = ()
            if self.mode == "deferred":
                count = self.records[0].shape[0]  # one pending count per record slot
                pending = torch.empty(count, dtype=torch.int32, **meta)
                deferred = (torch.empty(n, dtype=torch.int32, **meta), pending)
            flavor = kind
            if kind in DIE_KINDS and self.die_ops is not None:
                deferred = (*deferred, *self.die_ops)
                flavor += "+dies"
            mx_kind = kind in DIE_KINDS and self.mode == "deferred"
            if mx_kind and self.mx_input is not None:
                view = self.mx_input.view(r)
                mx = (view.data.view(torch.uint8), view.scale)
                deferred = (*deferred, *(torch.empty_like(t, **meta) for t in mx))
                flavor += "+mx"
            if kind == "chain":
                items = self.heads * n
                chain_scratch(self.device, items)  # allocated before any capture
                scratch = torch.empty(chain.scratch_shape(items), **meta)
                deferred = (*deferred, scratch)
            if kind == "ring" and self.mode == "deferred":
                items = self.heads * n
                ring_scratch(self.device, items, self.q)  # allocated before any capture
                scratch = torch.empty(ring.scratch_shape(items, self.q), **meta)
                deferred = (*deferred, scratch)
            operands = self._operands(caches, projected, slots, out, accepted, deferred)
            specs = tuple(map(layout, operands))
            if kind == "fused_ws" and not fits_fused_ws(specs, self.q, self.device):
                self.kinds[r] = kind = "fused"  # KdaFused: no persistent page staging
            if kind not in DIE_KINDS:
                flavor = kind
            options = (self.heads, self.q, True, self.has_bias, self.device.index)
            self.kernels[r] = executable(specs, flavor, *options)
            self.specs[r] = specs
        return self

    def run(self, projected, out, *, caches, metadata):
        """``projected`` [R, pitch] BF16 (stride (pitch, 1)); ``out`` [R, 3072] BF16
        (o_norm-gated, the o_proj input); reads ``metadata.spec_state_indices_tensor``
        [>= R / q, 1] (q without records), ``num_accepted_tokens`` and, deferred,
        ``recoverssm_deferred``."""
        rows = projected.shape[0]
        n = rows // self.q
        slots = metadata.spec_state_indices_tensor
        accepted = metadata.num_accepted_tokens
        if self.plain is not None:
            return self.plain.run(
                projected, out, caches=caches, slots=slots, accepted=accepted
            )
        kernel = self.kernels.get(rows)
        if kernel is None:
            raise Unsupported("kda", Shape(rows, self.q), "not a prepared row count")
        extra: tuple = ()
        if self.mode == "deferred":
            deferred = getattr(metadata, "recoverssm_deferred", None)
            if deferred is None:
                cause = (
                    "no deferred RecoverSSM metadata: the runner state is not bound "
                    "(recoverssm.bind_model_state)"
                )
                raise Unsupported("kda", Shape(rows, self.q), cause)
            if self.kinds[rows] not in FOLDS:
                deferred.flush()  # this core verifies from committed checkpoints
            extra = (deferred.record_rows[:n], deferred.pending)
            if self.kinds[rows] in DIE_KINDS and self.die_ops is not None:
                extra = (*extra, *self.die_ops)
            if self.kinds[rows] in DIE_KINDS and self.mx_input is not None:
                mx_out = self.mx_input.view(rows)
                extra = (*extra, mx_out.data.view(torch.uint8), mx_out.scale)
            if self.kinds[rows] == "chain":
                extra = (*extra, chain_scratch(self.device, self.heads * n, grow=False))
            if self.kinds[rows] == "ring":
                scratch = ring_scratch(self.device, self.heads * n, self.q, grow=False)
                extra = (*extra, scratch)
        slots = slots[:n]
        if slots.stride() != (slots.shape[1], 1):  # e.g. an eager block-table slice
            slots = slots.contiguous()
        tensors = self._operands(caches, projected, slots, out, accepted[:n], extra)
        specs = tuple(map(layout, tensors))
        if specs != self.specs[rows]:  # CuTe takes a mismatched layout silently
            pairs = enumerate(zip(specs, self.specs[rows]))
            changed = [i for i, (a, b) in pairs if a != b]
            cause = f"operand layouts {changed} differ from the compiled ones"
            raise Unsupported("kda", Shape(rows, self.q), cause)
        views = tuple(to_cute(t, 16) for t in tensors)
        kernel(*views[:13], *self.scalars, *views[13:], stream(self.device))
        if extra and self.kinds[rows] in DIE_KINDS and self.mx_input is not None:
            return self.mx_input.view(rows)
        return out

    def choice(self, rows):
        """The coverage entry at ``rows``: GENERIC (no 24-head table is measured)."""
        kind = self.kinds.get(rows)
        if kind is None:
            raise Unsupported("kda", Shape(rows, self.q), "not a served row count")
        state = f"fp32 {self.mode}" if self.plain is None else "fp32 plain"
        note = f"{kind} {rows // self.q}x{self.q} {self.heads}h {state}"
        return Choice(Path.GENERIC, tactic=kind, note=note)
