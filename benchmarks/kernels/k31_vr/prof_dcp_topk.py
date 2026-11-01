# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase timestamps + cluster-size sweep of the DCP top-k merge (dev tool)."""

import os
import statistics

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    DcpExchange,
    local_len_of,
)


def main():
    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    ex = DcpExchange(dist.group.WORLD, dev, cap=1536, cap1=256, interleave=896)
    K = 2048
    for T, ctx in [tuple(int(v) for v in x.split(":")) for x in os.environ.get("K31_PROF_SHAPES", "4:148000,32:190000,128:190000,512:190000,768:600000").split(",")]:
        G = torch.full((T,), ctx, dtype=torch.int32, device=dev)
        Lr = local_len_of(G.long(), rank, W, 896).to(torch.int32)
        Lmax = int(Lr.max())
        logits = torch.randn(T, (Lmax + 511) // 512 * 512, device=dev).to(torch.bfloat16)
        out = torch.empty(T, K, dtype=torch.int32, device=dev)
        cnt = torch.empty(T, dtype=torch.int32, device=dev)
        dbg = torch.zeros(32, dtype=torch.int64, device=dev)
        for one in (True, False):
            if one and T > 256:
                continue
            for cl in [int(c) for c in os.environ.get("K31_PROF_CL", "1,2,4,8").split(",")]:
                fn = lambda: ex.topk_merge(logits, Lr, G, out, cnt, max_local_len=Lmax,  # noqa: E731
                                           cluster=cl, one_round=one)
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(20):
                        fn()
                ts = []
                for _ in range(10):
                    dist.barrier()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); g.replay(); e1.record(); e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000 / 20)
                # one eager call with timestamps
                dist.barrier(); torch.cuda.synchronize()
                dbg.zero_()
                ex.topk_merge(logits, Lr, G, out, cnt, max_local_len=Lmax, cluster=cl,
                              one_round=one, dbg=dbg)
                torch.cuda.synchronize()
                d = dbg.tolist()
                ph = [(i, (d[i] - d[0]) / 1000) for i in range(1, 16) if d[i]]
                if rank == 0:
                    print(f"T{T} 1R={one} cluster{cl}: graph {statistics.median(ts):.1f} us; "
                          f"phases(us from start) {[(i, round(x, 1)) for i, x in ph]}", flush=True)
                del g
                if os.environ.get("K31_PROF_GRAPHDBG"):
                    # timestamps of the last call of a graph replay, every rank
                    fn2 = lambda: ex.topk_merge(logits, Lr, G, out, cnt, max_local_len=Lmax,  # noqa: E731
                                                cluster=cl, one_round=one, dbg=dbg)
                    g2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g2):
                        for _ in range(20):
                            fn2()
                    dist.barrier(); torch.cuda.synchronize()
                    g2.replay(); torch.cuda.synchronize()
                    allg = [torch.zeros_like(dbg) for _ in range(W)]
                    dist.all_gather(allg, dbg)
                    if rank == 0:
                        for r_, d_ in enumerate(allg):
                            d_ = d_.tolist()
                            ph = [(i, round((d_[i] - d_[0]) / 1000, 1)) for i in range(1, 16) if d_[i]]
                            print(f"  graph rank {r_}: {ph}", flush=True)
                    del g2
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
