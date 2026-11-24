# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, the wide MoE tail's LatentRows with NormRows' work
(``moe.tail_rows.LATENT_NORM``, ``LatentRows(norm=True)``) over 4 ranks of one node.

Per case (rows 72 / 128 / 256 / 512 on a 512-row ``TailRowsBox``), every rank's routed
latent partial [M, 3584] and shared partial [M, 7168] (BF16, random) through the wide
tail twice: LatentRows + NormRows (separate kernels) and the fused LatentRows. Checks:

- fused vs separate: ``shared`` (FP32), ``xnorm`` (BF16) and the output ring bitwise
  on every rank (same words, same FP32 operation order);
- the fused path vs the reference ops of the stated math (``shared`` bitwise, xnorm
  within one BF16 ulp, the ring BF16(up + shared));
- the fused tail replayed from a CUDA graph: the ring bitwise its eager rows.

Run on a node with 4 Rubin (SM107) GPUs and multicast memory (single process
launches 4 workers); skipped elsewhere."""

import gc
import os
import time

import pytest
import torch
import torch.distributed as dist

WORLD = 4
H, LATENT, CAP, EPS = 7168, 3584, 512, 1e-5
SHARD = H // WORLD
ROWS = (72, 128, 256, 512)
TIMEOUT = 600


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


def _bf16_no_neg_zero(x):
    out = x.to(torch.bfloat16)
    return torch.where(out == 0, torch.zeros_like(out), out)


def _run(rows, box, partial, shared, batch):
    ring = box.out.buffer[0, :batch]
    ring.view(torch.int16).fill_(-1)
    box.xnorm[:batch].view(torch.int16).fill_(-1)
    box.shared[:batch].fill_(float("nan"))
    torch.accelerator.synchronize()
    dist.barrier()
    rows.push(shared)
    rows.run(partial, batch)
    torch.accelerator.synchronize()
    dist.barrier()
    return (
        box.shared[:batch].clone(),
        box.xnorm[:batch].clone(),
        box.up[:batch].clone(),
        ring.clone(),
    )


def _case(fused, separate, box, gamma, rank, batch):
    dev = gamma.device
    gen = torch.Generator(device=dev).manual_seed(1000 + 17 * rank + batch)
    partial = (torch.randn(batch, LATENT, generator=gen, device=dev) * 0.5).to(
        torch.bfloat16
    )
    shared = (torch.randn(batch, H, generator=gen, device=dev) * 0.25).to(
        torch.bfloat16
    )
    want = _run(separate, box, partial, shared, batch)
    got = _run(fused, box, partial, shared, batch)
    for name, a, b in zip(("shared", "xnorm", "up", "ring"), got, want):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (batch, name)
    got_shared, xnorm, up, ring = got

    partials = [torch.empty_like(partial) for _ in range(WORLD)]
    dist.all_gather(partials, partial)
    shareds = [torch.empty_like(shared) for _ in range(WORLD)]
    dist.all_gather(shareds, shared)
    mine = slice(rank * SHARD, (rank + 1) * SHARD)
    acc = torch.zeros(batch, SHARD, dtype=torch.float32, device=dev)
    for s in shareds:
        acc = acc + s[:, mine].float()
    assert torch.equal(got_shared, acc), (batch, "shared ref")
    latent = torch.zeros(batch, LATENT, dtype=torch.float32, device=dev)
    for p in partials:
        latent = latent + p.float()
    latent = latent.to(torch.bfloat16).float()
    inv = torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + EPS)
    ref_xnorm = ((latent * inv) * gamma.float()).to(torch.bfloat16)
    torch.testing.assert_close(xnorm, ref_xnorm, atol=0, rtol=2**-7)
    out = _bf16_no_neg_zero(up + got_shared)
    outs = [torch.empty_like(out) for _ in range(WORLD)]
    dist.all_gather(outs, out)
    assert torch.equal(ring, torch.cat(outs, 1)), (batch, "ring ref")

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    gc.collect()
    gc.disable()  # (no symmetric buffer freed inside the capture)
    try:
        with torch.cuda.stream(side), torch.cuda.graph(
            graph, stream=side, capture_error_mode="thread_local"
        ):
            fused.push(shared)
            fused.run(partial, batch)
    finally:
        gc.enable()
    live = box.out.buffer[0, :batch]
    for replay in range(3):
        live.view(torch.int16).fill_(-1)
        torch.accelerator.synchronize()
        dist.barrier()
        graph.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        assert torch.equal(live, ring), (batch, replay)
    del graph


def _worker(rank, port, result):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.tail_rows import (
        PreparedTailRows,
        TailRowsBox,
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
    gen = torch.Generator(device=dev).manual_seed(7)  # replicated weights
    up = (torch.randn(H, LATENT, generator=gen, device=dev) / 60).to(torch.bfloat16)
    gamma = (torch.rand(LATENT, generator=gen, device=dev) + 0.5).to(torch.bfloat16)
    box = TailRowsBox(group, dev, WORLD, rank, CAP)
    fused = PreparedTailRows(WORLD, rank, up, gamma, EPS, ROWS, box, 0, norm_rows=ROWS)
    separate = PreparedTailRows(WORLD, rank, up, gamma, EPS, ROWS, box, 0, norm_rows=())
    assert fused.norm_rows == set(ROWS) and not separate.norm_rows
    for batch in ROWS:
        start = time.monotonic()
        _case(fused, separate, box, gamma, rank, batch)
        if rank == 0:
            print(f"TP4 latent+norm M{batch}: ok {time.monotonic() - start:.1f} s")
    dist.barrier()
    dist.destroy_process_group()
    result.put((rank, "ok"))


def test_tp4_latent_norm_rows():
    """LatentRows with NormRows' work vs the separate kernels (module docstring)."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    port = int(os.environ.get("MASTER_PORT", "29573"))
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
            pytest.fail(f"TP4 latent+norm: no result in {TIMEOUT} s (a hang?)")
        if any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            pytest.fail("TP4 latent+norm: a rank failed (see its traceback)")
        try:
            done.append(result.get(timeout=5))
        except Exception:
            continue
    for p in procs:
        p.join(timeout=60)
    assert sorted(r for r, _ in done) == list(range(WORLD))
