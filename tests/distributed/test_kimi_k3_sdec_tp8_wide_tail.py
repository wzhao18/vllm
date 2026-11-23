# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, the wide MoE tail at TP8 (``moe.tail_rows``,
``PreparedTailRows``) over 8 ranks on two 4-GPU nodes of one NVLink domain: the tail
the TP8 = DCP8 wide verify rows run behind the stock router, latent-down, shared and
routed experts (MXFP4: ``moe.prepared.WIDE_TAIL``; NVFP4: ``moe.nvfp4.Nvfp4MoE``), at
rows 72 / 128 / 256 / 768 / 1536 on the model's 1536-row ``TailRowsBox``.

Per case, every rank's routed latent partial [M, 3584] and shared partial [M, 7168]
(BF16, random) through PushShared, LatentRows, NormRows, the up projection and
OutRows, against the reference ops of the stated math:

- ``shared`` (FP32, this rank's 896 columns): the TP slots summed in rank order in
  FP32: bitwise;
- ``xnorm`` (BF16): RMS norm (``rsqrt.approx``) of the latent all-reduce (the TP slots
  in rank order in FP32, one BF16 rounding) times gamma: within one BF16 ulp;
- ``up`` (FP32): xnorm x this rank's up shard^T as cuBLAS computes it: bitwise;
- the output ring (every rank's 896 columns, multicast): BF16(up + shared), one
  rounding, -0 made +0: bitwise on every rank;
- the tail replayed from a CUDA graph: the ring bitwise its eager rows.

Launch as tests/distributed/test_kimi_k3_sdec_tp8_wide.py: one pytest per node over
two nodes (``srun -N2 --ntasks-per-node=1``), MASTER_ADDR / MASTER_PORT set; skipped
elsewhere."""

import gc
import os
import time

import pytest
import torch
import torch.distributed as dist

LOCAL, NODES = 4, 2
WORLD = LOCAL * NODES
H, LATENT, CAP, EPS = 7168, 3584, 1536, 1e-5
SHARD = H // WORLD
ROWS = (72, 128, 256, 768, 1536)
TIMEOUT = 600


def _launch():
    if os.environ.get("SLURM_NNODES") != str(NODES):
        return None
    if "SLURM_NODEID" not in os.environ or "MASTER_ADDR" not in os.environ:
        return None
    port = int(os.environ.get("MASTER_PORT", "29571"))
    return int(os.environ["SLURM_NODEID"]), os.environ["MASTER_ADDR"], port


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < LOCAL:
        return False
    if {torch.cuda.get_device_capability(i) for i in range(LOCAL)} != {(10, 7)}:
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
        _launch() is None or not _rubin_node(),
        reason="needs 2 nodes x 4 Rubin (SM107) GPUs of one NVLink domain, "
        "launched with srun -N2 and MASTER_ADDR",
    ),
]


def _bf16_no_neg_zero(x):
    out = x.to(torch.bfloat16)
    return torch.where(out == 0, torch.zeros_like(out), out)


def _case(rows, box, up, gamma, rank, batch):
    dev = up.device
    gen = torch.Generator(device=dev).manual_seed(1000 + 17 * rank + batch)
    partial = (torch.randn(batch, LATENT, generator=gen, device=dev) * 0.5).to(
        torch.bfloat16
    )
    shared = (torch.randn(batch, H, generator=gen, device=dev) * 0.25).to(
        torch.bfloat16
    )
    ring = box.out.buffer[0, :batch]
    ring.view(torch.int16).fill_(-1)
    torch.accelerator.synchronize()
    dist.barrier()
    rows.push(shared)
    rows.run(partial, batch)
    torch.accelerator.synchronize()
    dist.barrier()

    partials = [torch.empty_like(partial) for _ in range(WORLD)]
    dist.all_gather(partials, partial)
    shareds = [torch.empty_like(shared) for _ in range(WORLD)]
    dist.all_gather(shareds, shared)
    mine = slice(rank * SHARD, (rank + 1) * SHARD)
    # shared: the TP slots of this rank's columns in rank order, FP32
    acc = torch.zeros(batch, SHARD, dtype=torch.float32, device=dev)
    for s in shareds:
        acc = acc + s[:, mine].float()
    got_shared = box.shared[:batch]
    assert torch.equal(got_shared, acc), (batch, "shared")
    # xnorm: the latent all-reduce (rank order, FP32, one rounding), RMS, gamma
    latent = torch.zeros(batch, LATENT, dtype=torch.float32, device=dev)
    for p in partials:
        latent = latent + p.float()
    latent = latent.to(torch.bfloat16).float()
    inv = torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + EPS)
    ref_xnorm = ((latent * inv) * gamma.float()).to(torch.bfloat16)
    xnorm = box.xnorm[:batch]
    torch.testing.assert_close(xnorm, ref_xnorm, atol=0, rtol=2**-7)
    # up: this rank's up shard over xnorm, as cuBLAS computes it
    ref_up = torch.mm(xnorm, rows.up_shard.t(), out_dtype=torch.float32)
    assert torch.equal(box.up[:batch], ref_up), (batch, "up")
    # the ring: every rank's BF16(up + shared) columns
    out = _bf16_no_neg_zero(box.up[:batch] + got_shared)
    outs = [torch.empty_like(out) for _ in range(WORLD)]
    dist.all_gather(outs, out)
    assert torch.equal(ring, torch.cat(outs, 1)), (batch, "ring")
    eager = ring.clone()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    gc.collect()
    gc.disable()  # (no symmetric buffer freed inside the capture)
    try:
        with torch.cuda.stream(side), torch.cuda.graph(
            graph, stream=side, capture_error_mode="thread_local"
        ):
            rows.push(shared)
            rows.run(partial, batch)
    finally:
        gc.enable()
    for replay in range(3):
        ring.view(torch.int16).fill_(-1)
        torch.accelerator.synchronize()
        dist.barrier()
        graph.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        assert torch.equal(ring, eager), (batch, replay)
    del graph


def _worker(local, node, addr, port, result):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.tail_rows import (
        PreparedTailRows,
        TailRowsBox,
    )

    rank = node * LOCAL + local
    torch.accelerator.set_device_index(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{addr}:{port}",
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
    rows = PreparedTailRows(WORLD, rank, up, gamma, EPS, ROWS, box, 0)
    for batch in ROWS:
        start = time.monotonic()
        _case(rows, box, up, gamma, rank, batch)
        if rank == 0:
            print(f"TP8 wide tail M{batch}: ok {time.monotonic() - start:.1f} s")
    dist.barrier()
    dist.destroy_process_group()
    result.put((rank, "ok"))


def test_tp8_wide_moe_tail():
    """The TP8 wide MoE tail over two nodes (module docstring)."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    node, addr, port = _launch()
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(i, node, addr, port, result))
        for i in range(LOCAL)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT
    done = []
    while len(done) < LOCAL:
        if time.monotonic() > deadline:
            for p in procs:
                p.kill()
            pytest.fail(f"TP8 wide tail: no result in {TIMEOUT} s (a hang?)")
        if any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            pytest.fail("TP8 wide tail: a rank failed (see its traceback)")
        try:
            done.append(result.get(timeout=5))
        except Exception:
            continue
    for p in procs:
        p.join(timeout=60)
    assert sorted(r for r, _ in done) == [node * LOCAL + i for i in range(LOCAL)]
