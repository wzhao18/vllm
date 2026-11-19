# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode (D3-32, TP4): reduce-scatter push probe.

The SP reduce-scatter pushes each rank's BF16 o-projection partial [R, 7168] to the
row owners' receive slots (``SpComm.recv``; 3 x R / 4 rows over NVLink per rank) and
each owner Lamport-polls its TP slots. The fused boundary (``SpReduceAttnRes``) pushes
with 16-byte relaxed system stores from every thread. This probe times, in one CUDA
graph of 93 push + poll + re-arm rounds, the push forms:

  v4    16-byte st.relaxed.sys per thread (the current form)
  v8    32-byte st.relaxed.sys.v8 per thread
  bulk  2-KiB segments staged in shared memory (sanitized), cp.async.bulk to the owner

usage: torchrun --nproc-per-node 4 bench_kimi_k3_sdec_sp_push.py [--rows 16,128,384]
       [--modes v4,v8,bulk]
"""

import argparse
import statistics

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    check_residency,
    fake,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.mailbox import rearm
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    launch,
    load_vector,
    load_words,
    published,
    shared_address,
    store_relaxed,
    zeros,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.sp_comm import (
    positive_zero32,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import (
    poll_slots,
)

WIDTH = 7168
CAP = 512
THREADS = 256
SEG = 1024  # columns per bulk segment (2 KiB)
SENTINEL = -0x80000000


@asm()
def store_v8(pointer, words):
    regs = ",".join(f"${i + 1}" for i in range(8))
    code = f"st.relaxed.sys.global.v8.b32 [$0], {{{regs}}};"
    return code, "l" + ",r" * 8 + ",~{memory}", (pointer, *words)


@asm()
def bulk_store(dst, src, size: cutlass.Constexpr[int]):
    code = f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {size};"
    return code, "l,r,~{memory}", (dst, src)


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm()
def bulk_wait_read(pending: cutlass.Constexpr[int]):
    return f"cp.async.bulk.wait_group.read {pending};", "~{memory}", ()


@asm()
def bulk_wait(pending: cutlass.Constexpr[int]):
    return f"cp.async.bulk.wait_group {pending};", "~{memory}", ()


@asm()
def proxy_fence():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def half_barrier(index):
    return "bar.sync $0, 128;", "r,~{memory}", (Int32(index),)


@asm()
def multicast(address, words, kind: cutlass.Constexpr[str]):
    count = len(words)
    vec = f".v{count}" if count > 1 else ""
    regs = ",".join(f"${i + 1}" for i in range(count))
    code = f"multimem.st.relaxed.sys.global{vec}.{kind} [$0], {{{regs}}};"
    if count == 1:
        code = f"multimem.st.relaxed.sys.global.{kind} [$0], $1;"
    return code, "l" + ",r" * count + ",~{memory}", (address, *words)


class GatherProbe:
    """Multicast this rank's n rows into rows [rank n, (rank + 1) n) of every rank's
    gather box with ``op`` bytes per store (8 / 16 / 32), poll all R rows, re-arm."""

    def __init__(self, tp, rows, grid, op, half=False):
        # ``half``: publish only columns [0, 3584) of each row (half the bytes, the
        # MXFP8 all-gather's payload); the poll still checks every published chunk.
        self.tp, self.rows, self.grid, self.op = tp, rows, grid, op
        self.width = WIDTH // 2 if half else WIDTH

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        x, gather, mc, rank = args
        tp: cutlass.Constexpr = self.tp
        R: cutlass.Constexpr = self.rows
        n: cutlass.Constexpr = R // tp
        total: cutlass.Constexpr = self.grid * THREADS
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        cols: cutlass.Constexpr = self.op // 2  # BF16 columns per store
        per: cutlass.Constexpr = self.width // cols
        for i in cutlass.range_constexpr(-(-n * per // total)):
            c = g + i * total
            if c < n * per:
                row, column = c // per, c % per * cols
                offset = Int64((rank * n + row) * WIDTH + column) * 2
                if cutlass.const_expr(self.op == 8):
                    w = load_words(at(x, row, column), count=2)
                    multicast(mc + offset, tuple(positive_zero32(v) for v in w), "f32")
                elif cutlass.const_expr(self.op == 16):
                    w = load_vector(at(x, row, column))
                    multicast(mc + offset, tuple(positive_zero32(v) for v in w), "f32")
                else:
                    a = load_vector(at(x, row, column))
                    b = load_vector(at(x, row, column + 8))
                    w = tuple(positive_zero32(v) for v in (*a, *b))
                    multicast(mc + offset, w, "bf16x2")
        per16: cutlass.Constexpr = self.width // 8
        for i in cutlass.range_constexpr(-(-R * per16 // total)):
            c = g + i * total
            if c < R * per16:
                raw = zeros((1, 4), Uint32)
                row, column = c // per16, c % per16 * 8
                pending = Int32(1)
                while pending != 0:
                    words = load_vector(at(gather, row, column), cache="volatile")
                    for j in cutlass.range_constexpr(4):
                        raw[0, j] = words[j]
                    pending = Int32(0)
                    if not published(raw[0, j] for j in range(4)):
                        pending = Int32(1)
                rearm(at(gather, row, column))


class PushProbe:
    def __init__(self, tp, rows, grid, mode):
        self.tp, self.rows, self.grid, self.mode = tp, rows, grid, mode

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        partial, recv, peers, rank = args
        tp: cutlass.Constexpr = self.tp
        R: cutlass.Constexpr = self.rows
        n: cutlass.Constexpr = R // tp
        per: cutlass.Constexpr = WIDTH // 8
        total: cutlass.Constexpr = self.grid * THREADS
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        cap: cutlass.Constexpr = recv.shape[1]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.mode == "v4"):
            for i in cutlass.range_constexpr(-(-R * per // total)):
                c = g + i * total
                if c < R * per:
                    row, column = c // per, c % per * 8
                    words = load_vector(at(partial, row, column))
                    clean = tuple(positive_zero32(w) for w in words)
                    owner = row // n
                    offset = Int64((rank * cap + row - owner * n) * WIDTH + column) * 2
                    store_relaxed(Int64(peers[owner]) + offset, clean)
        if cutlass.const_expr(self.mode == "v8"):
            per2: cutlass.Constexpr = WIDTH // 16
            for i in cutlass.range_constexpr(-(-R * per2 // total)):
                c = g + i * total
                if c < R * per2:
                    row, column = c // per2, c % per2 * 16
                    a = load_vector(at(partial, row, column))
                    b = load_vector(at(partial, row, column + 8))
                    clean = tuple(positive_zero32(w) for w in (*a, *b))
                    owner = row // n
                    offset = Int64((rank * cap + row - owner * n) * WIDTH + column) * 2
                    store_v8(Int64(peers[owner]) + offset, clean)
        if cutlass.const_expr(self.mode == "bulk"):
            smem = cutlass.utils.SmemAllocator()
            stage = alloc(smem, Uint32, (2, 2, SEG // 2), align=128)
            half, local = tid // 128, tid % 128
            units: cutlass.Constexpr = R * (WIDTH // SEG)
            steps: cutlass.Constexpr = -(-units // (2 * self.grid))
            for i in cutlass.range_constexpr(steps):
                s = i % 2
                u = (i * self.grid + cta) * 2 + half
                if cutlass.const_expr(i >= 2):
                    if local == 0:
                        bulk_wait_read(1)
                    half_barrier(1 + half)
                if u < units:
                    row, column = u // (WIDTH // SEG), u % (WIDTH // SEG) * SEG
                    words = load_vector(at(partial, row, column + local * 8))
                    for j in cutlass.range_constexpr(4):
                        stage[s, half, local * 4 + j] = positive_zero32(words[j])
                proxy_fence()
                half_barrier(1 + half)
                if (u < units) & (local == 0):
                    row, column = u // (WIDTH // SEG), u % (WIDTH // SEG) * SEG
                    owner = row // n
                    offset = Int64((rank * cap + row - owner * n) * WIDTH + column) * 2
                    src = shared_address(stage.iterator) + (s * 2 + half) * SEG * 2
                    bulk_store(Int64(peers[owner]) + offset, src, SEG * 2)
                    bulk_commit()
            if local == 0:
                bulk_wait(0)
        # The owner: poll this rank's TP slots of its n rows, re-arm them.
        for i in cutlass.range_constexpr(-(-n * per // total)):
            c = g + i * total
            if c < n * per:
                row, column = c // per, c % per * 8
                raw = zeros((tp, 4), Uint32)
                poll_slots(recv, raw, row, column)
                for slot in cutlass.range_constexpr(tp):
                    rearm(at(recv, slot, row, column))


def compile_probe(rows, tp, cap, grid, mode, device):
    with torch.accelerator.device_index(device):
        partial = fake((rows, WIDTH))
        recv = fake((tp, cap // tp, WIDTH))
        peers = fake((tp,), None, torch.int64)
        kernel = PushProbe(tp, rows, grid, mode)
        code = cute.compile(
            kernel, partial, recv, peers, Int32(0), make_fake_stream(),
            options="--gpu-arch=sm_107a",
        )  # fmt: skip
        check_residency(code, device, grid, THREADS, f"push probe {mode}")
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="16,128,384")
    p.add_argument("--modes", default="v4,v8,bulk")
    p.add_argument("--gather-ops", default="8,16,16h")
    p.add_argument("--layers", type=int, default=93)
    p.add_argument("--replays", type=int, default=30)
    args = p.parse_args()
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.accelerator.set_device_index(device.index)
    group = dist.group.WORLD
    try:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import sp_comm

    comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
    for rows in [int(r) for r in args.rows.split(",")]:
        n = rows // world
        gen = torch.Generator().manual_seed(7 + rank)
        partial = (torch.randn((rows, WIDTH), generator=gen) * 0.05).bfloat16()
        partial.view(torch.int32)[:, 5::97] = SENTINEL  # (+0, -0): sanitized to 0
        partial = partial.to(device)
        grid = sp_comm.polling_grid(rows, WIDTH, comm.sms)
        for mode in args.modes.split(","):
            try:
                code = compile_probe(rows, world, CAP, grid, mode, device.index)
            except Exception as error:  # (a form ptxas refuses: report, go on)
                if rank == 0:
                    print(f"R{rows} {mode}: compile failed: {error}"[:400], flush=True)
                continue

            def run(count, parity0=0):
                for layer in range(count):
                    _, _, recv, peers = comm.views((parity0 + layer) % 2)
                    code(to_cute(partial), recv, peers, comm._rank, stream(device))

            # correctness: one round, then compare this rank's slots before re-arm
            # is not observable (the kernel re-arms); check the re-arm instead.
            dist.barrier()
            run(2)
            torch.accelerator.synchronize(device)
            armed = bool((comm.recv.view(torch.int32) == SENTINEL).all())
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(args.layers)
            torch.accelerator.synchronize(device)
            times = []
            for _ in range(args.replays):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
            med = statistics.median(times[3:])
            t = torch.tensor([med], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"R{rows:>4} n{n:>3} {mode:>5}: {t.item() / args.layers:7.2f} us "
                      f"per push+poll round (grid {grid}, re-armed {armed})", flush=True)
            del graph
        for opt in [o for o in args.gather_ops.split(",") if o]:
            op, half = int(opt.rstrip("h")), opt.endswith("h")
            x = partial[:n].contiguous()
            try:
                with torch.accelerator.device_index(device.index):
                    kernel = GatherProbe(world, rows, grid, op, half)
                    gcode = cute.compile(
                        kernel, fake((n, WIDTH)), fake((CAP, WIDTH)), Int64(0),
                        Int32(0), make_fake_stream(), options="--gpu-arch=sm_107a",
                    )  # fmt: skip
                    check_residency(gcode, device.index, grid, THREADS, "gather")
            except Exception as error:
                if rank == 0:
                    print(f"R{rows} mc{op}: compile failed: {error}"[:400], flush=True)
                continue

            def grun(count):
                for layer in range(count):
                    gv, mc, _, _ = comm.views(layer % 2)
                    gcode(to_cute(x), gv, mc, comm._rank, stream(device))

            dist.barrier()
            grun(2)
            torch.accelerator.synchronize(device)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                grun(args.layers)
            torch.accelerator.synchronize(device)
            times = []
            for _ in range(args.replays):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
            med = statistics.median(times[3:])
            t = torch.tensor([med], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"R{rows:>4} n{n:>3}  mc{opt:<3}: {t.item() / args.layers:7.2f} us "
                      f"per multicast+poll round (grid {grid})", flush=True)
            del graph
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
