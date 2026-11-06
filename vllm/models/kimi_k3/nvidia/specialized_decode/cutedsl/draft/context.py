# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's context step: target hidden states -> per-layer context KV.

Stock: ``context_proj`` is a replicated [7168, 5 x 7168] BF16 GEMM (514 MB read by
every rank, every step), then ``context_norm``, then the replicated ``context_kv_proj``
and six small kernels (permute, grouped RMSNorm, permute, positions, RoPE, grouped FP8
insert).

Here each rank projects its 448-column shard of ``context_proj`` (32 MB), and
``CtxGather`` multicasts the shard rows into every rank's Lamport row mailbox [CAP,
7168], polls the full rows and applies ``context_norm`` (one CTA per row; re-arms what
it read). After the stock ``context_kv_proj`` GEMM, ``CtxKV`` writes every layer's
latent: the per-layer kv_a RMSNorm, the k_pe RoPE and the FP8 insert of the rows whose
slot this rank owns (one CTA per (row, layer)), with the stock rounding points (BF16
after the norm and after the RoPE, FP8 of value / scale).

Where the shard GEMM is a split-K projection, it runs as ``GatherSplitKGemm``: the
owner CTA rounds its tile to BF16 once (negative zeros made positive, as CtxGather
sanitizes) and multicasts the rows straight into this rank's columns of the mailbox
rows; ``CtxGather(polled=True)`` then only polls, normalizes and re-arms, releasing its
PDL dependents (the context_kv_proj GEMM's weight loads) right after its wait.

Above CAP rows CtxGather and CtxKV take the row count from their inputs (one kernel
each for every wider row count; the mailbox holds the largest). CtxGather's one CTA per
row polls the row the other ranks' same-row CTAs publish, so that grid must be
co-resident; above the rows the device holds at once (VR200: 212, one 256-thread CTA
per SM) the row gather is the persistent ``CtxGather(ctas=...)``: one wave of CTAs,
each publishing every row it owns (rows c, c + ctas, ...) before it polls any, so any
row count completes. Per row it runs the per-row CTA's instructions (the same publish,
FP32 sums in the same order, the same rounding): bitwise the same rows. Every context
row count is covered (no stock context step on the target topology).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Shape, full_capture, refuse
from ..common.host import (
    architecture,
    check_residency,
    dynamic_fake,
    dynamic_view,
    fake,
    resident_ctas,
    stream,
    to_cute,
)
from ..common.mailbox import SENTINEL, poll
from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    fma,
    launch,
    load_words,
    multicast_words,
    no_neg_zero,
    pack_fp8,
    pack_pair,
    store_relaxed,
    store_words,
    warp_sum,
)
from ..gemm.splitk import SplitKTactic
from ..gemm.splitk_ring import RingProjection, RingSplitKGemm, _fake

HIDDEN = 7168
CAP = 64  # context rows per step (requests x (1 + draft length)) of the static kernels


@asm(Float32, pure=True)
def rsqrt_approx(value):
    """Rsqrtf (MUFU.RSQ), as the stock RMSNorm kernel."""
    return "rsqrt.approx.f32 $0,$1;", "=f,f", (Float32(value),)


@asm(Float32, pure=True)
def div_rn(a, b):
    return "div.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm()
def store_bytes8(address, words):
    """8 bytes (two words) at a 64-bit global ``address``."""
    return "st.global.v2.b32 [$0],{$1,$2};", "l,r,r,~{memory}", (Int64(address), *words)


@asm()
def store_fp8x2(address, low, high):
    """FP8 E4M3 (satfinite) of ``low``, ``high`` as two bytes at ``address``."""
    code = "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair,$2,$1; "
    code += "st.global.b16 [$0],pair; }"
    return code, "l,f,f,~{memory}", (Int64(address), Float32(low), Float32(high))


class CtxGather:
    """One 256-thread CTA per context row: publish this rank's ``shard``-column slice
    (sanitized: no -0.0), poll the full row (224 threads x four 16 B chunks), RMSNorm
    with ``gamma`` as vLLM's rms_norm (FP32 sum of squares, rsqrtf, x * r * w rounded
    once), re-arm the row. ``rows`` None: the shard rows' count.

    ``ctas`` (shape-dynamic rows): a persistent grid of ``ctas`` co-resident CTAs, CTA
    c on rows c, c + ctas, ...: it publishes the slices of all its rows, then polls,
    normalizes and re-arms them one by one (a barrier between rows: the warp sums'
    shared memory). A CTA waits only on rows every rank's CTAs published before their
    first wait, so the one wave completes at any row count."""

    def __init__(self, rows, shard, polled=False, ctas=None):
        self.rows, self.shard, self.polled, self.ctas = rows, shard, polled, ctas

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(self.ctas is not None):
            launch(self.kernel, args, self.ctas, 256)
        else:
            rows = self.rows
            if cutlass.const_expr(rows is None):
                rows = args[0].shape[0]
            launch(self.kernel, args, rows, 256)

    @cute.jit
    def publish(self, y, mc, rank, row, tid):
        """This rank's slice of ``row`` (no -0.0) into every rank's mailbox row."""
        chunks: cutlass.Constexpr = self.shard // 8
        if tid < chunks:
            words = load_words(
                y.iterator + row * self.shard + 8 * tid, count=4, mutable=True
            )
            words = tuple(no_neg_zero(w) for w in words)
            column = rank * self.shard + 8 * tid
            address = mc + (Int64(row) * HIDDEN + column) * 2
            multicast_words(address, words, kind="f32")  # bit-exact store

    @cute.jit
    def finish(self, mailbox, gammas, eps, out, sums, row, tid):
        """Poll ``row``'s full mailbox row, RMSNorm it into ``out``, re-arm it."""
        lane, warp = tid % 32, tid // 32
        total = Float32(0)
        # Chunks t, t + 224, ... of the row's 896 (threads past 224 re-poll chunk 0's
        # and add nothing).
        owner = tid < 224
        base = mailbox.iterator + row * HIDDEN + min(tid, 223) * 8
        got = poll(base, 224 * 8, 4)
        if owner:
            for i in cutlass.range_constexpr(4):
                for k in cutlass.range_constexpr(4):
                    for h in (False, True):
                        x = bf16(got[i, k], h)
                        total = fma(x, x, total)
        total = warp_sum(total)
        if lane == 0:
            sums[warp] = total
        cute.arch.sync_threads()
        variance = Float32(0)
        for w in cutlass.range_constexpr(8):
            variance += sums[w]
        scale = rsqrt_approx(div_rn(variance, Float32(HIDDEN)) + eps)
        if owner:
            for i in cutlass.range_constexpr(4):
                col = tid * 8 + 224 * 8 * i
                weights = gammas[i]
                normed = []
                for k in cutlass.range_constexpr(4):
                    pair = []
                    for h in (False, True):
                        x, w = bf16(got[i, k], h), bf16(weights[k], h)
                        pair.append(x * scale * w)
                    normed.append(bf16x2(pair[0], pair[1]))
                store_words(out.iterator + row * HIDDEN + col, tuple(normed))
                store_relaxed(base + 224 * 8 * i, (Uint32(SENTINEL),) * 4)

    @cute.kernel
    def kernel(self, args):
        y, mailbox, mc, rank, gamma, eps, out = args
        first, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        sums = alloc(cutlass.utils.SmemAllocator(), Float32, 8)
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.polled):  # the shard rows came from the GEMM
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.ctas is None):  # one CTA per row
            if cutlass.const_expr(not self.polled):
                self.publish(y, mc, rank, first, tid)
            # The norm weights load while the row's chunks are in flight.
            gammas = [
                load_words(gamma.iterator + min(tid, 223) * 8 + 224 * 8 * i, count=4)
                for i in range(4)
            ]
            self.finish(mailbox, gammas, eps, out, sums, first, tid)
        else:  # persistent: every owned row published before the first poll
            count = y.shape[0]
            # The stride from the launch grid (not ``self.ctas``): the probe and the
            # launched kernel are the same code (the same registers and occupancy).
            stride, _, _ = cute.arch.grid_dim()
            if cutlass.const_expr(not self.polled):
                for row in cutlass.range(first, count, stride, unroll=1):
                    self.publish(y, mc, rank, row, tid)
            gammas = [
                load_words(gamma.iterator + min(tid, 223) * 8 + 224 * 8 * i, count=4)
                for i in range(4)
            ]
            for row in cutlass.range(first, count, stride, unroll=1):
                self.finish(mailbox, gammas, eps, out, sums, row, tid)
                cute.arch.sync_threads()  # every warp read ``sums`` before the next
        if cutlass.const_expr(not self.polled):
            cute.arch.griddepcontrol_launch_dependents()


class CtxKV:
    """One 64-thread CTA per (context row, draft layer): if this rank owns the row's
    slot, RMSNorm the layer's 512 latent columns (vLLM rms_norm arithmetic, per-layer
    weight, its 64-thread block's sum of squares in cub's order: each warp's down
    tree, then warp 0 + warp 1), RoPE the 64 k_pe columns (GPT-J pairs, FP32, BF16
    result), and store both as FP8 E4M3 (value / layer scale) into the layer's paged
    cache. ``rows`` None: the latent rows' count."""

    def __init__(self, rows, layers):
        self.rows, self.layers = rows, layers

    @cute.jit
    def __call__(self, *args):
        rows = self.rows
        if cutlass.const_expr(rows is None):
            rows = args[0].shape[0]
        launch(self.kernel, args, (rows, self.layers, 1), 64)

    @cute.kernel
    def kernel(self, args):
        kv, positions, slots, caches, gammas, scales, cos_sin, eps = args[0:8]
        page, page_stride = args[8:10]
        row, layer, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        sums = alloc(cutlass.utils.SmemAllocator(), Float32, 2)
        cute.arch.griddepcontrol_wait()
        slot = slots[row]
        if slot >= 0:
            width: cutlass.Constexpr = 576
            source = kv.iterator + row * (width * self.layers) + layer * width
            words = load_words(source + tid * 8, count=4, mutable=True)
            total = Float32(0)
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    x = bf16(words[k], h)
                    total = fma(x, x, total)
            for offset in (1, 2, 4, 8, 16):  # cub's warp reduction (lane 0's sum)
                total = total + cute.arch.shuffle_sync_down(total, offset)
            if lane == 0:
                sums[warp] = total
            cute.arch.sync_threads()
            rr = rsqrt_approx(div_rn(sums[0] + sums[1], Float32(512)) + eps)
            scale = scales[layer]
            address = caches[layer] + (slot // page) * page_stride + (slot % page) * 576
            weights = load_words(gammas.iterator + layer * 512 + tid * 8, count=4)
            fp8 = []
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    x, w = bf16(words[k], h), bf16(weights[k], h)
                    normed = Float32(BFloat16(x * rr * w))
                    fp8.append(div_rn(normed, scale))
            packed = (pack_fp8(tuple(fp8[0:4])), pack_fp8(tuple(fp8[4:8])))
            store_bytes8(address + tid * 8, packed)
            if tid < 32:  # rope pair tid: columns 512 + 2 tid, + 1
                pair = load_words(source + 512 + 2 * tid, count=1, mutable=True)[0]
                x, y = bf16(pair, False), bf16(pair, True)
                position = positions[row]
                cos = cos_sin[position, tid]
                sin = cos_sin[position, 32 + tid]
                # vLLM's x * cos - y * sin and y * cos + x * sin, contracted as nvcc
                # contracts them (the second product fused into the sum).
                rx = Float32(BFloat16(fma(x, cos, -(y * sin))))
                ry = Float32(BFloat16(fma(x, sin, y * cos)))
                store_fp8x2(
                    address + 512 + 2 * tid, div_rn(rx, scale), div_rn(ry, scale)
                )
        cute.arch.griddepcontrol_launch_dependents()


@cache
def compile_gather(rows, shard, device, polled=False, ctas=None):
    """``rows`` None: any row count (above CAP; dynamic views); ``ctas``: the
    persistent grid (``rows`` None)."""
    if rows:
        y, box, out = fake((rows, shard)), fake((CAP * HIDDEN,)), fake((rows, HIDDEN))
    else:
        y, box, out = dynamic_fake(2), fake((cute.sym_int(),)), dynamic_fake(2)
    args = (y, box, Int64(0), Int32(0), fake((HIDDEN,)), Float32(0), out)
    with torch.accelerator.device_index(device):
        kernel = CtxGather(rows, shard, polled, ctas)
        return cute.compile(kernel, *args, make_fake_stream())


class GatherSplitKGemm(RingSplitKGemm):
    """``RingSplitKGemm`` whose rows land in a [rows, HIDDEN] row mailbox: ``ring`` is
    the multicast address of row 0 at this rank's first shard column."""

    @cute.experimental.jit
    def publish(
        self, staged: cute.Tensor, mC: cute.Tensor, ring: cutlass.Int64,
        bidx: cutlass.Int32, n_idx: cutlass.Int32, epi_tid: cutlass.Int32,
    ):  # fmt: skip
        """Each epilogue thread multicasts one 8-column fragment of one row."""
        cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
        chunks = self.cta_m // 8
        token = epi_tid // chunks
        chunk = epi_tid - token * chunks
        width, rows = mC.layout.shape[0], mC.layout.shape[1]
        n, m = n_idx * self.cta_n + token, bidx * self.cta_m + chunk * 8
        if token < self.cta_n and n < rows and m < width:
            source = staged.iterator + token * self.cta_m + chunk * 8
            low = load_words(source, count=4, shared=True, result_type=Float32,
                             word_type="f32")  # fmt: skip
            high = load_words(source + 4, count=4, shared=True, result_type=Float32,
                              word_type="f32")  # fmt: skip
            values = (*low, *high)
            words = tuple(
                pack_pair(values[2 * j], values[2 * j + 1], policy="latent")
                for j in range(4)
            )
            offset = Int64(n) * Int64(HIDDEN) + Int64(m)
            multicast_words(ring + offset * 2, words, kind="bf16x2")


@cache
def compile_gather_projection(tactic, arch):
    y = _fake(0)
    kernel = GatherSplitKGemm(tactic=SplitKTactic(*tactic))
    args = (_fake(1), _fake(1), y, y, Int64(0), Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class GatherProjection(RingProjection):
    """``RingProjection`` on ``GatherSplitKGemm``."""

    def prepare(self):
        arch = architecture(self.device)
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                self.kernels[batch] = compile_gather_projection(tactic, arch)
                self.extra[batch] = self.weight_map(tactic[0])
        return self


@cache
def compile_kv(rows, layers, max_position, device):
    """``rows`` None: any row count (above CAP; dynamic views)."""
    f32 = torch.float32
    count = rows or cute.sym_int()
    args = (
        fake((rows, 576 * layers)) if rows else dynamic_fake(2),
        fake((count,), None, torch.int64, 8),
        fake((count,), None, torch.int64, 8),
        fake((layers,), None, torch.int64, 8),
        fake((layers, 512)),
        fake((layers,), None, f32, 4),
        fake((max_position, 64), None, f32, 4),
        Float32(0),
        Int64(0),
        Int64(0),
    )
    with torch.accelerator.device_index(device):
        return cute.compile(CtxKV(rows, layers), *args, make_fake_stream())


class PreparedContext:
    """The drafter's sharded context projection + Lamport row gather (+ norm), and the
    fused context-KV epilogue, for the drafter graphs' context row counts ``rows`` (up
    to CAP with kernels per row count, above it with the shape-dynamic ones: one CTA
    per row while the device holds them at once, the persistent gather above)."""

    def __init__(self, model, group, tp, rank, rows):
        self.tp, self.rank = tp, rank
        # NQ-NOFALLBACK: set by runtime.draft on the target topology: a captured
        # context step of a served row runs these kernels or raises.
        self.strict = False
        weight = model.context_proj.weight
        shard = weight.shape[0] // tp
        self.shard = weight.detach()[rank * shard : (rank + 1) * shard]
        self.gamma = model.context_norm.weight.detach()
        self.eps = Float32(model.context_norm.variance_epsilon)
        self.device = weight.device
        index = self.device.index
        self.rows = tuple(r for r in rows if r <= CAP)
        self.wide = tuple(r for r in rows if r > CAP)
        # Wide rows: one CTA per row while the shape-dynamic kernel's grid is
        # co-resident (the static ones are at most CAP CTAs), the persistent gather
        # (one co-resident wave) above.
        self.gathers = {}
        self.persistent = ()
        if self.wide:
            dynamic = compile_gather(None, shard, index)
            resident = resident_ctas(dynamic, index, 256)
            self.persistent = tuple(r for r in self.wide if r > resident)
            strided = self.persistent_gather(shard) if self.persistent else None
            for r in self.wide:
                self.gathers[r] = strided if r in self.persistent else dynamic
        self.buffer = symm_mem.empty(
            (max((CAP, *self.wide)) * HIDDEN,), dtype=torch.bfloat16, device=self.device
        )
        handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("Kimi K3 DSpark context gather needs multicast memory")
        self.buffer.view(torch.int32).fill_(-0x80000000)
        torch.accelerator.synchronize(self.device)
        torch.distributed.barrier(group=group)
        for r in self.rows:
            self.gathers[r] = compile_gather(r, shard, index)
        # The shard GEMM: the split-K projection where its policy covers the rows.
        from ...runtime.draft import projection

        self.projection = projection(self.shard, tp, self.rows)
        # Rows whose split-K shard GEMM publishes into the mailbox itself.
        self.ring = None
        self.polled = {}
        if self.projection is not None:
            tactics = self.projection.policies
            self.ring = GatherProjection(self.shard, tactics).prepare()
            for r in tactics:
                self.polled[r] = compile_gather(r, shard, index, True)
            self.shape_rows = torch.empty(
                (CAP, shard), dtype=torch.bfloat16, device=self.device
            )
        self.kv = {}
        self.view = to_cute(self.buffer)
        self.dynamic = dynamic_view(self.buffer)

    def persistent_gather(self, shard):
        """The persistent CtxGather: one wave of the device's resident CTAs (the
        kernel's own occupancy; ``ctas`` only sets the launch grid), checked."""
        index = self.device.index
        probe = compile_gather(None, shard, index, ctas=1)
        ctas = resident_ctas(probe, index, 256)
        if ctas < 1:
            raise RuntimeError("Kimi K3 DSpark context gather: no resident CTA")
        compiled = compile_gather(None, shard, index, ctas=ctas)
        what = f"Kimi K3 DSpark persistent context gather ({ctas} CTAs)"
        check_residency(compiled, index, ctas, 256, what)
        return compiled

    def combine(self, hidden):
        """context_norm(context_proj(hidden)) [rows, 7168], or None when not covered."""
        rows = hidden.shape[0]
        if not torch.cuda.is_current_stream_capturing():
            return None  # eager: no graph, the stock context projection
        if rows not in self.gathers:
            if self.strict and full_capture():  # NQ-NOFALLBACK
                refuse("DSpark drafter context step", "no row gather", Shape(rows))
            return None
        out = torch.empty((rows, HIDDEN), dtype=hidden.dtype, device=hidden.device)
        if rows in self.polled:
            ring = self.mc + self.rank * self.shard.shape[0] * 2
            self.ring.run_ring(hidden, self.shape_rows[:rows], ring)
            self.polled[rows](
                to_cute(self.shape_rows[:rows]),
                self.view,
                Int64(self.mc),
                Int32(self.rank),
                to_cute(self.gamma),
                self.eps,
                to_cute(out),
                stream(self.device),
            )
            return out
        if self.projection is not None and rows in self.projection.kernels:
            local = hidden.new_empty((rows, self.shard.shape[0]))
            self.projection.run(hidden, local)
        else:
            local = torch.nn.functional.linear(hidden, self.shard)
        self.gather(local, out)
        return out

    def gather(self, local, out):
        """CtxGather of the shard rows ``local`` [rows, shard] into ``out`` [rows,
        7168] (every rank calls it with its own shard rows)."""
        rows = local.shape[0]
        wide = rows in self.wide
        view = dynamic_view if wide else to_cute
        self.gathers[rows](
            view(local),
            self.dynamic if wide else self.view,
            Int64(self.mc),
            Int32(self.rank),
            to_cute(self.gamma),
            self.eps,
            view(out),
            stream(self.device),
        )

    def prepare_kv(self, model, max_position):
        layers = len(model.layers)
        index = self.device.index
        self.kv = {r: compile_kv(r, layers, max_position, index) for r in self.rows}
        for r in self.wide:
            self.kv[r] = compile_kv(None, layers, max_position, index)

    def store_kv(self, model, kv, positions, slots, caches, scales, page, page_stride):
        rows = kv.shape[0]
        kernel = self.kv.get(rows)
        if kernel is None:
            if self.strict and full_capture():  # NQ-NOFALLBACK
                refuse("DSpark drafter context KV", "no fused epilogue", Shape(rows))
            return False
        rotary = model.layers[0].self_attn.rotary_emb
        view = dynamic_view if rows in self.wide else to_cute
        kernel(
            view(kv),
            view(positions, 8),
            view(slots, 8),
            to_cute(caches, 8),
            to_cute(model._context_kv_norm_weights),
            to_cute(scales, 4),
            to_cute(rotary.cos_sin_cache, 4),
            Float32(model._context_rms_norm_eps),
            Int64(page),
            Int64(page_stride),
            stream(self.device),
        )
        return True
