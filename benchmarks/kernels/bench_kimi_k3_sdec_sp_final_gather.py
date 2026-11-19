# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpFinalGather (the SP step's final row all-gather) vs stock sp_all_gather (NCCL):
correctness and CUDA-graph time per call. torchrun --nproc-per-node <TP>."""

import os
import statistics
import sys

import torch
import torch.distributed as dist

WIDTH = 7168
SENTINEL = -0x80000000


def timed(fn, reps=20):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    ts = []
    for _ in range(reps):
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 100)  # us per call
    t = torch.tensor([statistics.median(ts[3:])], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    device = torch.device(f"cuda:{local}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    try:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import sp_comm

    widths = [int(w) * WIDTH for w in (sys.argv[1] if len(sys.argv) > 1 else "1,6").split(",")]
    for width in widths:
        final = sp_comm.SpFinalGather(group, device, world, rank, width=width, rows_cap=512)
        for rows in (16, 128, 384):
            n = rows // world
            x = torch.randn(n, width, device=device).bfloat16()
            x.view(torch.int32)[:, 7::101] = SENTINEL
            if rank == 0:
                print(f"W{width // WIDTH} R{rows}: compile+run", flush=True)
            got = final.run(x)
            torch.cuda.synchronize()
            ref = torch.empty((rows, width), dtype=x.dtype, device=device)
            dist.all_gather_into_tensor(ref, x)
            ref.view(torch.int32)[ref.view(torch.int32) == SENTINEL] = 0
            ok = torch.equal(got.view(torch.int32), ref.view(torch.int32))
            armed = bool((final.box.view(torch.int32) == SENTINEL).all())
            t_new = timed(lambda: final.run(x))
            t_nccl = timed(lambda: dist.all_gather_into_tensor(ref, x))
            if rank == 0:
                print(f"W{width // WIDTH} R{rows}: equal {ok} armed {armed} "
                      f"SpFinalGather {t_new:.1f} us  NCCL {t_nccl:.1f} us", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
