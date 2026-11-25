# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S0a experiment (SOL decode architecture v0, designs/ARCHITECTURE.md section 4):
``RubinGroupedExpertsKernel.body`` inside an in-kernel layer loop.

``S0aExperts(layers=0)`` is today's kernel (one call per launch, PDL wait) with
benchmark stamps whose trace slab follows the grouped call counter (slab =
generation % ``slabs``). ``S0aExperts(layers=L)`` is the persistent arm: one launch
runs L calls back to back on the same SMs. Per layer every CTA polls the layer's
epoch tag (data as flag, ``ld.acquire.gpu``) instead of ``griddepcontrol.wait``;
TMEM is allocated once and the descriptors prefetched once; the mbarriers are
re-initialized between layers after a drain commit (the role functions restart their
slot / phase walks at 0 each call), off the critical path (before the poll). The
census word alternates by layer parity (the two 8-byte halves of today's 16-byte
census: no new memory). The stub route producer is the layer's last finishing CTA
(acq_rel arrival count): it spins ``delay_ns`` (D) and then releases the next
layer's tag, stamping the release time. Benchmarks only (not wired into the model)."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    alloc,
    asm,
    commit,
    fence_tmem,
    mailbox_store,
    release_tmem,
    thread,
    view,
)
from ..common.tensormap import GridMap, map_address
from .experts_rubin import stamp_at
from .experts_rubin64 import RubinGroupedExpertsKernel, census_add, require_below

# trace slab per (layer, CTA): 16 slots of 16 B (``stamp_at``)
SLAB = 512


@asm(Uint32)
def load_acquire(pointer):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (pointer,)


@asm(Uint32)
def add_acq_rel(pointer, value):
    return (
        "atom.acq_rel.gpu.global.add.u32 $0,[$1],$2;",
        "=r,l,r,~{memory}",
        (pointer, Uint32(value)),
    )


@asm()
def store_release(pointer, value):
    return "st.release.gpu.global.u32 [$0],$1;", "l,r,~{memory}", (pointer, value)


@asm(Uint64)
def global_ns():
    return "mov.u64 $0, %globaltimer;", "=l", ()


class S0aExperts(RubinGroupedExpertsKernel):
    """``layers`` 0: today's kernel + stamps (trace slab = generation % ``slabs``);
    ``layers`` L > 0: the persistent loop (trace slab = layer); ``delay_ns``: D;
    ``trace``: the stamp buffer's address (slabs x grid x 512 B, then the producer's
    release stamps, 16 B per layer)."""

    def __init__(self, *args, layers=0, delay_ns=0, slabs=16, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.trace != 0 and self.dies
        self.layers, self.delay_ns, self.slabs = layers, delay_ns, slabs
        self.release_base = slabs * self.grid * SLAB
        assert layers == 0 or self.publish == "twoshot"
        self.slab, self.census = (0, 0)  # set per call / layer while tracing
        self.off_drain = self.off_misc + 2936  # free misc bytes (2920-3200)

    def set_layer(self, layer):
        """Trace-time: this layer's trace slab and census half (kept off ``self``
        stores in the loop body, which the DSL would carry as loop state)."""
        self.slab = Int64(layer) * (self.grid * SLAB)
        self.census = (layer & 1) * 8

    @cute.jit
    def stamp(self, slot: cutlass.Constexpr[int]):
        stamp_at(Int64(self.trace) + self.slab + Int64(thread()[3]) * SLAB, slot)

    @cute.jit
    def ticket(self, workspace):
        before = Uint64(0)
        if (thread()[1] == 2) & (thread()[2] == 0):
            increment = Uint64(1) + (Uint64(self.die()) << Uint64(32))
            pointer = workspace.iterator + self.ws_census + self.census
            before = census_add(pointer, increment)
        return before

    @cute.jit
    def arrive(self, sm, workspace, before):
        info = view(sm.iterator, self.off_misc + 2688, Int32, 4)
        if (thread()[1] == 2) & (thread()[2] == 0):
            counts = before + Uint64(1)
            arrivals = Uint32(counts & Uint64(4294967295))
            ones = Uint32(counts >> Uint64(32))
            require_below(arrivals, Uint32(self.grid))
            die = self.die()
            k = Int32(arrivals) - Int32(ones)
            if die == 1:
                k = Int32(ones)
            if arrivals == Uint32(self.grid - 1):
                base = workspace.iterator + self.ws_census + self.census
                mailbox_store(base, (Uint32(4294967295), Uint32(4294967295)))
            info[0], info[1], info[3] = (die, Int32(arrivals), k)

    @cute.jit
    def barriers_init(self, sm):
        """``initialize``'s mbarrier counts (thread 0)."""
        barriers = view(sm.iterator, self.off_misc, Uint64, 65)
        acc = 2 * self.stages
        for i in cutlass.range_constexpr(acc + 36):
            count = 4 if acc + 16 <= i < acc + 32 else 1
            if i < self.stages:
                count = 33
            cute.arch.mbarrier_init(barriers.iterator + i, count)
        queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        claims = view(sm.iterator, self.off_misc + self.bar_claims, Uint64, 3)
        for i in cutlass.range_constexpr(16):
            cute.arch.mbarrier_init(queue.iterator + i, 1)
            cute.arch.mbarrier_init(pmeta.iterator + i, 1)
        for i in cutlass.range_constexpr(3):
            cute.arch.mbarrier_init(claims.iterator + i, 1)
        cute.arch.mbarrier_init_fence()

    @cute.jit
    def __call__(self, *args):
        options = dict(grid=(self.grid, 1, 1), block=(384, 1, 1), cluster=(1, 1, 1))
        maps = self.encode_maps(args[-3], args[-2])
        if cutlass.const_expr(self.layers == 0):
            kernel = self.kernel_maps(args[:-3], *maps)
            kernel.launch(stream=args[-1], use_pdl=True, **options)
        else:
            kernel = self.kernel_loop(args[:-4], args[-4], *maps)
            kernel.launch(stream=args[-1], use_pdl=True, **options)

    @cute.kernel
    def kernel_maps(self, args, w13_map: GridMap, w2_map: GridMap):
        self.body_traced(args, (map_address(w13_map), map_address(w2_map)))

    @cute.kernel
    def kernel_loop(self, args, flags, w13_map: GridMap, w2_map: GridMap):
        self.body_loop(args, flags, (map_address(w13_map), map_address(w2_map)))

    @cute.jit
    def roles(self, sm, args, tmem, distinct, generation, ranges, scalars):
        """``body``'s warp roles after the grouping (unchanged)."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *_ = args
        tid, warp, lane, cta = thread()
        experts = self.tables(sm)[0]
        if warp == 0:
            if lane == 0:
                weights = (sm, (self.maps_[0], self.maps_[1]), w13_scales, w2_scales)
                self.stream(*weights, ranges, experts, (workspace, generation & 1))
        else:
            if warp == 1:
                self.consume(sm, tmem, ranges)
                if lane == 0:
                    self.stamp(9)
            elif warp == 2:
                self.load_hidden(sm, workspace, ranges, generation & 1)
                if lane == 0:
                    self.stamp(12)
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            elif warp == 3:
                self.gather_x(sm, x, ranges, batch)
                if lane == 0:
                    self.stamp(14)
                self.rearm(workspace, generation)
                if lane == 0:
                    self.claims(sm, workspace, ranges, generation & 1)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            else:
                self.epilogue(
                    sm, workspace, output, tmem, ranges, batch, generation & 1, scalars
                )
                if tid == 128:
                    self.stamp(10)

    @cute.jit
    def counters_done(self, workspace, routed):
        """``body``'s end-of-call counters (thread 0)."""
        cta = thread()[3]
        distinct, call_count, generation = (routed[0], routed[1], routed[2])
        counters = view(workspace.iterator, self.ws_counter, Int32, self.slots)
        counters[cta] = call_count + 1
        if cta < self.slots - self.grid:
            counters[self.grid + cta] = call_count + 1
        grouped = view(workspace.iterator, self.ws_calls, Int32, self.slots + 2)
        grouped[cta] = generation + 1
        if cta < self.slots - self.grid:
            grouped[self.grid + cta] = generation + 1
        if cta == 0:
            grouped[self.slots + (generation & 1)] = distinct

    @cute.jit
    def body_traced(self, args, maps):
        """T: ``RubinGroupedExpertsKernel.body`` with stamps in slab generation % slabs
        (the wait return is kept in a register until the slab is known)."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        self.maps_ = maps
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        arrival = self.ticket(workspace)
        cute.arch.griddepcontrol_wait()
        waited = global_ns()
        routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        self.group(sm, ids, workspace, batch, arrival=arrival)
        distinct, generation = (routed[0], routed[2])
        self.slab = Int64(generation % self.slabs) * (self.grid * SLAB)
        if tid == 0:
            here = Int64(self.trace) + self.slab + Int64(cta) * SLAB
            mailbox_store(here + 16, (Uint32(waited & Uint64(4294967295)),))
            mailbox_store(here + 20, (Uint32(waited >> Uint64(32)),))
            self.stamp(2)
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        ranges = self.matched_ranges(sm, distinct)
        if tid == 0:
            self.stamp(11)
        self.roles(sm, args, tmem, distinct, generation, ranges, scalars)
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            self.stamp(13)
            self.counters_done(workspace, routed)

    @cute.jit
    def body_loop(self, args, flags, maps):
        """P: the layer loop. ``flags``: Int32 [2 L] (tags, then arrival counts),
        zeroed before the launch; tag 0 is released in front of the kernel."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        self.maps_ = maps
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        drain = view(sm.iterator, self.off_drain, Uint64, 1)
        if tid == 0:
            cute.arch.mbarrier_init(drain.iterator, 1)
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        tags = flags.iterator
        recv, peers, rslot = scalars[3:6]
        for layer in cutlass.range(self.layers, unroll=1):
            self.set_layer(layer)  # a call: no attribute store inside the dynamic loop
            if layer > 0:  # every role ended the previous layer (drained): re-arm
                if tid == 0:
                    self.barriers_init(sm)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
            arrival = self.ticket(workspace)
            if tid == 0:
                seen = load_acquire(tags + layer)
                while seen == Uint32(0):
                    seen = load_acquire(tags + layer)
                self.stamp(1)
            cute.arch.sync_threads()
            self.group(sm, ids, workspace, batch, arrival=arrival)
            if tid == 0:
                self.stamp(2)
            distinct, generation = (routed[0], routed[2])
            fence_tmem()
            ranges = self.matched_ranges(sm, distinct)
            if tid == 0:
                self.stamp(11)
            box = (recv, peers, Int32((layer % 3) * self.tp) + rslot)
            self.roles(sm, args, tmem, distinct, generation, ranges, (*scalars[:3], *box))
            if warp == 1:  # every tcgen05 commit of the layer has arrived
                if lane == 0:
                    commit(drain.iterator)
                self.sync(drain.iterator, layer & 1)
            fence_tmem(before=True)
            cute.arch.sync_threads()
            if tid == 0:
                self.stamp(13)
                self.counters_done(workspace, routed)
                count = add_acq_rel(tags + self.layers + layer, 1)
                if count == Uint32(self.grid - 1):
                    if layer + 1 < self.layers:
                        if cutlass.const_expr(self.delay_ns > 0):
                            start = global_ns()
                            now = global_ns()
                            while now - start < Uint64(self.delay_ns):
                                now = global_ns()
                        release = Int64(self.trace) + self.release_base
                        release += Int64(layer + 1) * 16
                        stamp_at(release, 0)
                        store_release(tags + layer + 1, Uint32(1))
        release_tmem(tmem, 512, warp, 3)


class S0aTagKernel:
    """The stub route producer in front of a layer (T) or of the persistent kernel
    (P, tag 0): releases its PDL dependents at entry, spins ``spin_ns``, stamps the
    release (trace + release_base + 16 ``index``) and stores flags[index] = 1."""

    def __init__(self, trace, release_base, spin_ns):
        self.trace, self.release_base, self.spin_ns = trace, release_base, spin_ns

    @cute.jit
    def __call__(self, flags, index, stream):
        self.kernel(flags, index).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, flags, index: Int32):
        tid = thread()[0]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid == 0:
            if cutlass.const_expr(self.spin_ns > 0):
                start = global_ns()
                now = global_ns()
                while now - start < Uint64(self.spin_ns):
                    now = global_ns()
            release = Int64(self.trace) + self.release_base + Int64(index) * 16
            stamp_at(release, 0)
            store_release(flags.iterator + index, Uint32(1))


def compile_s0a(
    experts, batch, grid, device, slots, dies, static, zone, trace, layers, delay_ns
):  # fmt: skip
    """``experts_runtime.compile_rubin64`` (two-shot publish) for ``S0aExperts``;
    ``layers`` > 0 adds the flags tensor (Int32 [2 layers]) before the addresses."""
    import torch
    from cutlass.cute.runtime import make_fake_stream

    from ..common.host import fake
    from .experts import PLAN_BYTES

    moe = S0aExperts(
        experts, grid, slots, publish="twoshot", pool=(zone, 6), dies=dies,
        static_groups=static, trace=trace, layers=layers, delay_ns=delay_ns,
    )  # fmt: skip
    flat = lambda n, dtype=cutlass.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (fake((48, 64, 224)), fake((16,), None, cutlass.Int64), Int32(0))
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16, cutlass.BFloat16)
    tensors = (flat(PLAN_BYTES), flat(batch * 16, Int32), flat(256, align=64))
    tensors += (*scales, flat(moe.workspace_bytes), out)
    scalars = (Int32(batch), *map(cutlass.Float32, (100, 0.25, 0.04)), *box)
    if layers:
        assert 2 * layers <= 64
        scalars += (flat(64, Int32),)
    scalars += (Int64(0), Int64(0))
    with torch.accelerator.device_index(device):
        return moe, cute.compile(moe, *tensors, *scalars, make_fake_stream())


def compile_tag(trace, release_base, spin_ns, device):
    import torch
    from cutlass.cute.runtime import make_fake_stream

    from ..common.host import fake

    kernel = S0aTagKernel(trace, release_base, spin_ns)
    flags = fake((64,), (1,), Int32, 16)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, flags, Int32(0), make_fake_stream())
