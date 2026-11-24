# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, the wide o-projection producer (``OprojUmmaRows``:
weight-stationary tcgen05 over token chunks, two-shot reduce-scatter into the
column owners, owner reduce and multicast gather) over 4 ranks of one node.

x's tiles stream from one issuing lane per ring stage (``OprojUmmaRows.issuers``).
Per case (rows 72 / 128 / 256 / 512, chunks 64 / 80 / 96) every rank's x [M, 768] and
weight [7168, 768] (BF16, random) through the producer with one issuing lane and with
the default lanes:

- the gathered all-reduce rows bitwise between the two (same tiles, stages, MMAs);
- vs the reference (each rank's BF16 partial, summed over the ranks in FP32, one BF16
  rounding) within the MMA's FP32 accumulation order (a few BF16 ulp).

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
    assert kernel.issuers == (kernel.stages if issuers is None else issuers)
    return cute.compile(kernel, x, w, recv, peers, slot, box, mc, mc, make_fake_stream())


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
        outs = []
        for issuers in (1, None):
            code = _compile(batch, chunk, issuers)
            gather.view(torch.int32).fill_(SENTINEL)
            torch.accelerator.synchronize()
            dist.barrier()
            code(to_cute(x_buffer[:batch]), to_cute(weight), *head, stream(dev))
            torch.accelerator.synchronize()
            dist.barrier()
            outs.append(gather[0, :batch].clone())
        single, lanes = outs
        assert torch.equal(single.view(torch.int16), lanes.view(torch.int16)), batch
        ref = torch.zeros(batch, H, dtype=torch.float32, device=dev)
        for xr, wr in zip(xs, ws):
            part = torch.mm(xr[:batch].float(), wr.float().t()).to(torch.bfloat16)
            ref = ref + part.float()
        torch.testing.assert_close(lanes.float(), ref, atol=2e-2, rtol=2e-2)
        del tma
        if rank == 0:
            print(f"TP4 oproj rows M{batch} chunk {chunk}: ok "
                  f"{time.monotonic() - start:.1f} s")  # fmt: skip
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
