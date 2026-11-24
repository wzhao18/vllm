# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, the wide o-projection producer (``OprojUmmaRows``:
weight-stationary tcgen05 over token chunks, two-shot reduce-scatter into the
column owners, owner reduce and multicast gather) over 4 ranks of one node.

x's tiles stream from ``OprojUmmaRows.issuers`` issuing lanes (one per ring stage
with 64-row chunks, one with 80 / 96-row chunks by default).
Per case (rows 72 / 128 / 256 / 512, chunks 64 / 80 / 96) every rank's x [M, 768] and
weight [7168, 768] (BF16, random) through the producer with one issuing lane and with
one lane per ring stage:

- the gathered all-reduce rows bitwise between the two (same tiles, stages, MMAs);
- vs the reference (each rank's BF16 partial, summed over the ranks in FP32, one BF16
  rounding) within the MMA's FP32 accumulation order (a few BF16 ulp).

Each form is also replayed REPS times (gather re-armed between launches, ranks
released by a barrier, CUDA events around the launch on every rank); rank 0 prints the
min and median over the replays of the slowest rank's time (one issuing lane vs one
per ring stage) so the issuing-lane change is measured at kernel level. Timing is reported,
not asserted (barrier-exit skew across the ranks dominates a single launch).

Run on a node with 4 Rubin (SM107) GPUs and multicast memory; skipped elsewhere."""

import os
import time

import pytest
import torch
import torch.distributed as dist

WORLD = 4
H, K, CAP = 7168, 768, 512
CASES = ((72, 80), (128, 64), (256, 96), (512, 64))
TIMEOUT = 600
REPS = 20
SENTINEL = -0x80000000


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    if {torch.cuda.get_device_capability(i) for i in range(WORLD)} != {(10, 7)}:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        not _rubin_node(), reason="needs 4 Rubin (SM107) GPUs with multicast memory"
    ),
]


def _compile(batch, chunk, issuers):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.producer import (
        OprojUmmaRows,
    )

    x, w = fake((batch, K), (K, 1)), fake((H, K), (K, 1))
    recv, peers = fake((WORLD, CAP, H // WORLD)), fake((WORLD,), None, torch.int64)
    box = fake((1, CAP, H), (CAP * H, H, 1))
    mc, slot = cutlass.Int64(0), cutlass.Int32(0)
    kernel = OprojUmmaRows(K, chunk=chunk, issuers=issuers)
    assert kernel.issuers == issuers
    return cute.compile(kernel, x, w, recv, peers, slot, box, mc, mc, make_fake_stream())


def _stages(chunk):
    return {64: 8, 80: 7, 96: 6}[chunk]


def _timed(code, args, gather, dev):
    """min / median over REPS launches of the slowest rank's event time (us)."""
    times = []
    for _ in range(REPS):
        gather.view(torch.int32).fill_(SENTINEL)
        torch.accelerator.synchronize()
        dist.barrier()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        code(*args)
        end.record()
        torch.accelerator.synchronize()
        slowest = torch.tensor([begin.elapsed_time(end) * 1e3], device=dev)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        times.append(slowest.item())
    times.sort()
    return times[0], times[len(times) // 2]


def _default_issuers(chunk):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.producer import (
        OprojUmmaRows,
    )

    return OprojUmmaRows(768, chunk=chunk).issuers


def _worker(rank, port, result):
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import (
        TwoShot,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        weight_map,
    )

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        device_id=dev,
    )
    group = dist.group.WORLD
    try:  # (older torch: symmetric memory is enabled per group)
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    box = TwoShot(group, dev, WORLD, rank, width=H, cap=CAP, parities=2)
    recv, peers, gather, gather_mc = box.views(0)
    gen = torch.Generator(device=dev).manual_seed(11 + rank)
    x_buffer = (torch.randn(CAP, K, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    weight = (torch.randn(H, K, generator=gen, device=dev) / 30).to(torch.bfloat16)
    xs = [torch.empty_like(x_buffer) for _ in range(WORLD)]
    dist.all_gather(xs, x_buffer)
    ws = [torch.empty_like(weight) for _ in range(WORLD)]
    dist.all_gather(ws, weight)
    for batch, chunk in CASES:
        start = time.monotonic()
        tile = min(-(-batch // chunk) * chunk, CAP)
        tma = weight_map(x_buffer[:tile], chunk)
        head = (
            to_cute(recv), to_cute(peers), cutlass.Int32(rank), to_cute(gather),
            cutlass.Int64(gather_mc), cutlass.Int64(tma.data_ptr()),
        )  # fmt: skip
        outs, timing = [], {}
        for issuers in (1, _stages(chunk)):
            code = _compile(batch, chunk, issuers)
            gather.view(torch.int32).fill_(SENTINEL)
            torch.accelerator.synchronize()
            dist.barrier()
            code(to_cute(x_buffer[:batch]), to_cute(weight), *head, stream(dev))
            torch.accelerator.synchronize()
            dist.barrier()
            outs.append(gather[0, :batch].clone())
            args = (to_cute(x_buffer[:batch]), to_cute(weight), *head, stream(dev))
            timing[issuers] = _timed(code, args, gather, dev)
        single, lanes = outs
        default = _default_issuers(chunk)
        assert default == (_stages(chunk) if chunk == 64 else 1), default
        assert torch.equal(single.view(torch.int16), lanes.view(torch.int16)), batch
        ref = torch.zeros(batch, H, dtype=torch.float32, device=dev)
        for xr, wr in zip(xs, ws):
            part = torch.mm(xr[:batch].float(), wr.float().t()).to(torch.bfloat16)
            ref = ref + part.float()
        torch.testing.assert_close(lanes.float(), ref, atol=2e-2, rtol=2e-2)
        del tma
        if rank == 0:
            print(f"TP4 oproj rows M{batch} chunk {chunk}: ok "
                  f"{time.monotonic() - start:.1f} s; us min/median 1 lane "
                  f"{timing[1][0]:.2f}/{timing[1][1]:.2f}, 1 per stage "
                  f"{timing[_stages(chunk)][0]:.2f}/{timing[_stages(chunk)][1]:.2f}")  # fmt: skip
    dist.barrier()
    dist.destroy_process_group()
    result.put((rank, "ok"))


def test_tp4_oproj_rows_issuers():
    """OprojUmmaRows with one vs the default x issuing lanes (module docstring)."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    port = int(os.environ.get("MASTER_PORT", "29575"))
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, port, result)) for i in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT
    done = []
    while len(done) < WORLD:
        if time.monotonic() > deadline:
            for p in procs:
                p.kill()
            pytest.fail(f"TP4 oproj rows: no result in {TIMEOUT} s (a hang?)")
        if any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            pytest.fail("TP4 oproj rows: a rank failed (see its traceback)")
        try:
            done.append(result.get(timeout=5))
        except Exception:
            continue
    for p in procs:
        p.join(timeout=60)
    assert sorted(r for r, _ in done) == list(range(WORLD))
