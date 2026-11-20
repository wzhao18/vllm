# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's layer boundaries as two fused kernels (C512 / D3-32 SP decode,
``runtime.sp_attention``; TP4 = DCP4 = SP4, rank r owns shard rows [r n, (r + 1) n)).

Stock runs four latency-bound kernels per layer between the MLP GEMMs and the
attention in-projection / after the o-projection: the AttnRes before attention, the
SP all-gather, the SP reduce-scatter and the AttnRes after attention. Here each pair
is one ``ClusterAttnRes`` (one 7-CTA cluster per shard row, 1024 columns per CTA):

- ``SpAttnResGather`` (pre-AttnRes + all-gather): the AttnRes of the MLP output
  (``_pre_attn_norm``) stores its normed row by NVLink multicast into row
  r n + token of every rank's ``SpComm.gather[parity]`` (BF16 -0 halves made +0, so no
  published word is the Lamport sentinel), then Lamport-polls the same 1024 columns of
  rows o n + token of every rank o: when the kernel ends the gather holds all R rows
  (stock ``sp_all_gather``), read plainly by the in-projection.
- ``SpReduceAttnRes`` (reduce-scatter + post-AttnRes): (``push``) first stores this
  rank's o-projection partial rows o n + token (its 1024 columns) into owner o's
  receive slot [rank] (``positive_zero32`` words, as ``SpReduceScatter``), then polls
  its own TP slots of shard row token, sums them in FP32 in rank order, rounds once to
  BF16 and re-arms them: that sum is the AttnRes delta (``_post_attn_norm``); a
  block-write layer takes it as the new prefix row. Finally it re-arms rows o n + token
  of ``gather[parity]`` (the in-projection consumed them).

Each CTA only touches its own 1024 columns of its token's rows, so no grid-wide
striding; every CTA polls rows that other ranks' CTAs of the same (column, token) store
in the same kernel, so the whole grid must be co-resident on every rank (checked at
build: 7 n CTAs within the per-SM occupancy, held by a launch bound sized
for n_max: ``_co_resident``).

Precision: the AttnRes is ``ClusterAttnRes``'s (stock precision: FP32 statistics, the
softmax over the saved blocks and the prefix, one BF16 rounding of the normed output;
the summation order differs from the stock op). The reduce-scatter sum is stock's
(FP32 in rank order, one rounding; -0 sums become +0). The all-gathered words are the
AttnRes output's (-0 halves +0: a GEMM operand sign of zero).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..common import mx_rows
from ..common.host import architecture, dynamic_fake, fake, prepare_shared_carveout
from ..common.mailbox import poll, rearm
from ..common.primitives import at, launch, load_words, store_relaxed
from ..common.sp_comm import positive_zero32
from .kernel import ClusterAttnRes

CLUSTER = 7  # CTAs per row (1024 columns each)
# (kind, nb) -> (CTAs per SM, SMs, registers, smem allowance) of the compiled codes
RESIDENCY: dict = {}
THREADS = 128


def _launch(kernel, args):
    """One 7-CTA cluster per shard row. ``kernel.min_blocks`` (``_co_resident``): the
    launch bound that keeps the whole grid co-resident at n_max rows."""
    grid = (CLUSTER, args[1].shape[0], 1)
    if cutlass.const_expr(kernel.min_blocks > 0):
        launch(
            kernel.kernel, args, grid, THREADS, CLUSTER,
            min_blocks_per_mp=kernel.min_blocks,
        )  # fmt: skip
    else:
        launch(kernel.kernel, args, grid, THREADS, CLUSTER)


def _co_resident(n_max, device):
    """CTAs per SM the grid of n_max 7-CTA clusters needs to be co-resident (every CTA
    polls rows other ranks' CTAs store in the same kernel). The launch bound caps the
    registers per thread at 64 K / (128 x bound): the 93-layer AttnRes (8 saved
    blocks) compiles to 104 registers = 4 CTAs per SM, 848 < 896 CTAs at n_max 128 on
    212 SMs; bound 5 (<= 102 registers) fits. Same instructions (register allocation
    only), so the same numerics; ``_check_clusters`` still verifies the result."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return -(-CLUSTER * n_max // sms)


class SpAttnResGather(ClusterAttnRes):
    """Pre-AttnRes + SP all-gather (module docstring). Args: delta [n, 7168] (the MLP
    output; for a layer without delta, the prefix itself), prefix [n, 7168], blocks
    [n, NB, 7168], norm, query, out_norm [7168], output (unused row view), eps,
    out_eps, the gather multicast address, rank, the local gather view [cap, 7168]."""

    def __init__(self, nb, tp, *, has_delta, write, mx=False, parts=0, **options):
        # ``mx``: publish the normed rows as MXFP8 into the MXFP8 box (``SpComm``
        # gather data + scale regions; ``common.mx_rows``) for an MXFP8 in-projection;
        # args 9.. are then (data mc, scale mc, rank, local data, local scales).
        # ``parts``: the delta is a beta-add GEMM's FP32 split-K partials (arg 0,
        # [parts * n, 7168]) and its BF16 residual (the last arg, [n, 7168]); see
        # ``ClusterAttnRes.sum_parts``.
        out = dict(out_mx="publish") if mx else dict(out_mc=True)
        super().__init__(
            nb, slots=0, has_delta=has_delta, write=write, delta_parts=parts, **out,
            **options,
        )  # fmt: skip
        self.tp, self.mx = tp, mx

    @cute.jit
    def __call__(self, *args):
        _launch(self, args)

    @cute.kernel
    def kernel(self, args):
        delta, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        if cutlass.const_expr(self.delta_parts > 0):
            delta = (delta, args[len(args) - 1])
        part, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        n = prefix.shape[0]
        bufs = self.setup(cutlass.utils.SmemAllocator())
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        state = self.state()
        rows, ids = (prefix, blocks, norm, query, out_norm), (part, token, tid)
        self.pre(state, bufs, rows, eps, ids)
        col = (part * 128 + tid) * 8
        if cutlass.const_expr(self.mx):
            mc, smc, rank, data, scale = args[9:14]
            out = (mc, rank * n, smc)
            self.post(state, bufs, delta, rows, out, eps, out_eps, ids, ())
            # Every rank's MXFP8 row of this token, these 1024 columns: published?
            mx_rows.poll(data, scale, token, n, self.tp, col, 7168)
        else:
            mc, rank, gather = args[9:12]
            self.post(state, bufs, delta, rows, (mc, rank * n), eps, out_eps, ids, ())
            # Every rank's row of this token, these 1024 columns: published?
            poll(at(gather, token, col), n * 7168, self.tp)
        cute.arch.griddepcontrol_wait()


class SpReduceAttnRes(ClusterAttnRes):
    """SP reduce-scatter owner sum + post-AttnRes (module docstring). Args: the local
    receive view [TP, cap / TP, 7168] (the Lamport delta ring), prefix, blocks, norm,
    query, out_norm, output [n, 7168], eps, out_eps, partial [R, 7168] (read with
    ``push``), the owners' receive bases [TP] (int64), rank, the local gather view."""

    def __init__(self, nb, tp, *, has_delta, push, gather_mx=False, out_mx=False,
                 **options):  # fmt: skip
        # ``gather_mx``: this layer's all-gather was MXFP8 (re-arm that layout; the
        # last arg is then (local data, local scales)). ``out_mx``: also store the
        # output row as MXFP8 for the MXFP8 projection that reads it (``output`` is
        # then (out, data address, scale address)).
        super().__init__(
            nb, slots=tp, has_delta=has_delta, write=-1, sanitize=True,
            out_mx="store" if out_mx else None, **options,
        )  # fmt: skip
        self.tp, self.push, self.gather_mx = tp, push, gather_mx

    @cute.jit
    def __call__(self, *args):
        _launch(self, args)

    @cute.kernel
    def kernel(self, args):
        ring, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        partial, peers, rank = args[9:12]
        rest = args[12:]
        if cutlass.const_expr(self.gather_mx):
            gather, rest = (rest[0], rest[1]), rest[2:]
        else:
            gather, rest = rest[0], rest[1:]
        if cutlass.const_expr(self.out_mx is not None):
            output = (output, rest[0], rest[1])
        part, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        n = prefix.shape[0]
        col = (part * 128 + tid) * 8
        bufs = self.setup(cutlass.utils.SmemAllocator())
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        state = self.state()
        rows, ids = (prefix, blocks, norm, query, out_norm), (part, token, tid)
        self.pre(state, bufs, rows, eps, ids)
        if cutlass.const_expr(self.push):
            cute.arch.griddepcontrol_wait()  # the o-projection partial is complete
            cap = ring.shape[1]
            for o in cutlass.range_constexpr(self.tp):
                words = load_words(at(partial, o * n + token, col), count=4)
                clean = tuple(positive_zero32(w) for w in words)
                offset = Int64((rank * cap + token) * 7168 + col) * 2
                store_relaxed(Int64(peers[o]) + offset, clean)
        self.post(state, bufs, ring, rows, output, eps, out_eps, ids, ())
        if cutlass.const_expr(self.gather_mx):
            data, scale = gather
            for o in cutlass.range_constexpr(self.tp):
                mx_rows.rearm(data, scale, o * n + token, col, 7168)
        else:
            for o in cutlass.range_constexpr(self.tp):
                rearm(at(gather, o * n + token, col))
        cute.arch.griddepcontrol_wait()


def _options(device):
    """The Rubin variants of ``ClusterAttnRes`` (``attn_res.prepared``)."""
    from .prepared import STORES_FIRST_SMS

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rubin = sms > STORES_FIRST_SMS
    return dict(stores_first=rubin, spread_push=rubin, poll_warp=rubin)


def _occupancy(code, device):
    """(CTAs per SM, SMs, registers, max dynamic smem attribute) at this kernel's own
    shared memory (``kernel.SMEM_BYTES``; the function's max-dynamic attribute is
    the allowance, not the request)."""
    import cuda.bindings.driver as cuda

    from ..common.host import _checked, kernel_registers, resolve_kernel
    from .kernel import SMEM_BYTES

    _, function = resolve_kernel(code, device)
    attribute = cuda.CUfunction_attribute
    (allowed,) = _checked(cuda.cuFuncGetAttribute(
        attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function
    ))  # fmt: skip
    (per_sm,) = _checked(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function, THREADS, SMEM_BYTES
    ))  # fmt: skip
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return per_sm, sms, kernel_registers(code, device), allowed


def _check_clusters(code, device, n_max, what):
    """Every CTA of the grid co-resident (they poll one another across ranks)."""
    per_sm, sms, registers, allowed = _occupancy(code, device)
    if CLUSTER * n_max > per_sm * sms:
        from ...shapes import Unsupported

        raise Unsupported(
            "sp_comm", None,
            f"{what}: {n_max} 7-CTA clusters cannot be co-resident ({per_sm} CTAs "
            f"per SM x {sms} SMs; {registers} registers, smem allowance {allowed})",
        )  # fmt: skip
    return per_sm, sms, registers, allowed


@cache
def compile_gather(nb, has_delta, write, tp, cap, n_max, device, mx=False, parts=0):
    """``SpAttnResGather`` for a layer key (nb, has_delta, write, mx, parts) at any
    n <= n_max."""
    with torch.accelerator.device_index(device):
        kernel = SpAttnResGather(
            nb, tp, has_delta=has_delta, write=write, mx=mx, parts=parts,
            min_blocks=_co_resident(n_max, device), **_options(device),
        )  # fmt: skip
        row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
        delta = dynamic_fake(2, torch.float32) if parts else row
        args = (delta, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
        if mx:
            args += (Int64(0), Int64(0), Int32(0), Int64(0), Int64(0))
        else:
            args += (Int64(0), Int32(0), fake((cap, 7168)))
        if parts:
            args += (row,)
        arch = architecture(torch.device("cuda", device))
        code = cute.compile(
            kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
        )
        RESIDENCY[("gather", nb, mx) + ((parts,) if parts else ())] = _check_clusters(
            code, device, n_max, "SP AttnRes + all-gather"
        )
    return code


@cache
def compile_reduce(nb, has_delta, push, tp, cap, n_max, device, gather_mx=False,
                   out_mx=False):  # fmt: skip
    """``SpReduceAttnRes`` for a layer key at any n <= n_max."""
    with torch.accelerator.device_index(device):
        kernel = SpReduceAttnRes(
            nb, tp, has_delta=has_delta, push=push, gather_mx=gather_mx,
            out_mx=out_mx, min_blocks=_co_resident(n_max, device),
            **_options(device),
        )  # fmt: skip
        row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
        ring = fake((tp, cap // tp, 7168))
        args = (ring, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
        args += (dynamic_fake(2), fake((tp,), None, torch.int64), Int32(0))
        args += (Int64(0), Int64(0)) if gather_mx else (fake((cap, 7168)),)
        if out_mx:
            args += (Int64(0), Int64(0))
        arch = architecture(torch.device("cuda", device))
        code = cute.compile(
            kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
        )
        RESIDENCY["reduce", nb, gather_mx, out_mx] = _check_clusters(
            code, device, n_max, "SP reduce-scatter + AttnRes"
        )
        # The next kernel (the router | latent down GEMM) streams its weights before
        # its dependency wait; its CTAs can only land beside this one if the SM keeps
        # the full shared-memory carve-out (CuTe's launcher drops the preference).
        code = prepare_shared_carveout(code, device, 100)
    return code


class SplitDelta:
    """A beta-add GEMM's output left for the next boundary to form: FP32 split-K
    ``partials`` [parts * n, 7168] (split s in rows [s n, (s + 1) n)) and the BF16
    ``residual`` [n, 7168]; the delta is bf16(bf16(sum of the partials in split
    order) + residual), stock's beta-add arithmetic (``ClusterAttnRes.sum_parts``)."""

    def __init__(self, partials, residual, parts):
        rows = residual.shape[0]
        if partials.shape != (parts * rows, residual.shape[1]) or not (
            partials.is_contiguous() and residual.is_contiguous()
        ):
            raise ValueError(
                f"SplitDelta: partials {tuple(partials.shape)} for {parts} x "
                f"{tuple(residual.shape)}"
            )
        self.partials, self.residual, self.parts = partials, residual, parts
        self.shape = residual.shape


class SpBoundary:
    """One decoder layer's two fused SP boundaries (host side). ``pre`` / ``post``:
    (norm, query, out_norm) BF16 [7168] weights and (eps, out_eps) of the stock
    ``_pre_attn_norm`` / ``_post_attn_norm``; ``nb`` the layer's prev_valid_blocks,
    ``write`` its block_write_idx (-1: not a block-write layer). Every kernel is
    compiled (and its co-residency checked) here, for shard rows up to ``n_max``;
    ``first``: the step's first layer (its pre-AttnRes has no delta).

    ``gather_mx``: the layer's in-projection is MXFP8: the all-gather carries the
    MXFP8 rows (``SpComm.mx_view``; half the bytes, no quantization kernel).
    ``out_mx``: an MXFP8 input scratch (``gemm.mxfp8.Mxfp8Input``) the post-AttnRes
    output row is also written into, quantized (the MLP's MXFP8 shared gate_up)."""

    def __init__(self, comm, pre, post, nb, write, n_max, push_rows, first=False,
                 gather_mx=False, out_mx=None, delta_parts=()):  # fmt: skip
        from ..common.host import to_cute

        self.comm, self.nb, self.write = comm, nb, write
        self.gather_mx, self.out_mx = gather_mx, out_mx
        device = comm.device.index
        tp, cap = comm.tp, comm.rows_cap
        self.pre_w = tuple(to_cute(w.reshape(-1)) for w in pre[0])
        self.pre_eps = tuple(map(cutlass.Float32, pre[1]))
        self.post_w = tuple(to_cute(w.reshape(-1)) for w in post[0])
        self.post_eps = tuple(map(cutlass.Float32, post[1]))
        self.out_args = ()
        if out_mx is not None:
            if out_mx.k != 7168 or out_mx.buf.rows < n_max:
                from ...shapes import Unsupported

                raise Unsupported(
                    "sp_boundary", None, f"MXFP8 output scratch [{out_mx.buf.rows}, "
                    f"{out_mx.k}] for {n_max} shard rows of 7168",
                )  # fmt: skip
            buf = out_mx.buf
            buf.scale.zero_()  # padding rows of a tile: finite (never stored) rows
            self.out_args = (Int64(buf.data.data_ptr()), Int64(buf.scale.data_ptr()))
        self.codes = {}
        # The step's first layer has no delta (the embedding is the prefix).
        self.codes[("pre", not first)] = compile_gather(
            nb, not first, write, tp, cap, n_max, device, gather_mx
        )
        # ``delta_parts``: split counts of a previous layer's beta-add GEMM that leaves
        # its output as FP32 partials (``SplitDelta``); summed here.
        for parts in sorted(set(delta_parts)):
            self.codes[("pre", True, parts)] = compile_gather(
                nb, True, write, tp, cap, n_max, device, gather_mx, parts
            )
        block = write >= 0
        for push in sorted({bool(p) for p in push_rows.values()}):
            self.codes[("post", push)] = compile_reduce(
                nb + int(block), not block, push, tp, cap, n_max, device,
                gather_mx, out_mx is not None,
            )  # fmt: skip
        self.push_rows = push_rows  # R -> the reduce kernel pushes the partial

    def gather(self, delta, prefix, blocks, parity):
        """The pre-AttnRes of ``delta`` (None: none) over this rank's shard, all-
        gathered: returns gather[parity][:R] (complete when the kernel ends), or its
        MXFP8 form (``SpComm.mx_view``) with ``gather_mx``."""
        from ..common.host import dynamic_view, stream

        comm = self.comm
        n = prefix.shape[0]
        tail = ()
        if isinstance(delta, SplitDelta):
            code = self.codes[("pre", True, delta.parts)]
            source = delta.partials
            tail = (dynamic_view(delta.residual),)
        else:
            code = self.codes[("pre", delta is not None)]
            source = prefix if delta is None else delta
        rows = map(dynamic_view, (source, prefix, blocks, prefix))
        source_v, prefix_v, blocks_v, dummy = rows
        if self.gather_mx:
            mc, smc, data, scale = comm.mx_addresses(parity)
            code(
                source_v, prefix_v, blocks_v, *self.pre_w, dummy, *self.pre_eps,
                mc, smc, comm._rank, data, scale, *tail, stream(comm.device),
            )  # fmt: skip
            return comm.mx_view(parity, n * comm.tp)
        gather, mc, _, _ = comm.views(parity)
        code(
            source_v, prefix_v, blocks_v, *self.pre_w, dummy, *self.pre_eps,
            mc, comm._rank, gather, *tail, stream(comm.device),
        )  # fmt: skip
        return comm.gather[parity, : n * comm.tp]

    def reduce(self, prefix, blocks, out, parity, partial=None):
        """The owner sum of this rank's TP receive slots (pushing ``partial`` [R, 7168]
        first when given) + the post-AttnRes into ``out`` [n, 7168] (and, with
        ``out_mx``, its MXFP8 form into that scratch); a block-write layer stores the
        sum as the new prefix row (in ``prefix``)."""
        from ..common.host import dynamic_view, stream

        comm = self.comm
        n = prefix.shape[0]
        gather, _, recv, peers = comm.views(parity)
        if self.gather_mx:
            gather = comm.mx_addresses(parity)[2:]
        else:
            gather = (gather,)
        code = self.codes[("post", partial is not None)]
        if partial is None:
            partial = comm.partial[: n * comm.tp]
        views = map(dynamic_view, (prefix, blocks, out, partial))
        prefix_v, blocks_v, out_v, partial_v = views
        code(
            recv, prefix_v, blocks_v, *self.post_w, out_v, *self.post_eps,
            partial_v, peers, comm._rank, *gather, *self.out_args,
            stream(comm.device),
        )  # fmt: skip
        return out
