# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP top-k merge across the wide-decode row counts (dev tool, torchrun).

Times ``DcpExchange.topk_merge`` with the decode wiring's launch parameters
(``DcpxDecode.launch_params``: the cluster-by-batch rule, with and without the
one-wave fit) and a fixed-cluster sweep, at AgentX q8 batches (rows = 8 x requests),
interleave 1, the model's local bound (1M / DCP), positions output, ``tie_order``
"rank" and the centred one-round window, as the TP4/DCP4 decode runs them.

usage: torchrun --nproc-per-node 4 bench_dcp_topk_rows.py <decode_ctx_vectors.json>
env: K31_TR_BATCHES (requests, default 16,18,20,22,24,26,28,32), K31_TR_CLUSTERS
(fixed sweep, default 2,4,8), K31_TR_REPS. Prints one line per (rows, arm): median /
max-rank us per call, the cluster, staged keys, co-resident clusters, and whether the
selection equals the first arm's (per-row slot sets and counts).
"""

import json
import os
import statistics
import sys
import types

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode as dxd
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    DcpExchange,
    local_len_of,
)

K = 2048
Q = 8


def ctx_for(vecs, B):
    g = vecs["q8_C64"]
    if str(B) in g:
        return g[str(B)]
    out = []
    for b in sorted((int(k) for k in g), reverse=True):
        while len(out) + b <= B:
            out += g[str(b)]
        if len(out) == B:
            break
    return out[:B]


def policy(ex, dev, fit_wave):
    props = torch.cuda.get_device_properties(dev)
    ns = types.SimpleNamespace(
        ex=ex,
        sms=props.multi_processor_count,
        smem=int(getattr(props, "shared_memory_per_multiprocessor", 0) or 228 * 1024),
        segmax_override=-1,
        fit_wave=fit_wave,
    )
    ns._segmax = lambda rows, c, ml: dxd.DcpxDecode._segmax(ns, rows, c, ml)
    return lambda rows, ml: dxd.DcpxDecode.launch_params(ns, rows, ml)


def main():
    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    with open(sys.argv[1]) as f:
        vecs = json.load(f)
    ex = DcpExchange(dist.group.WORLD, dev, cap=2048, cap1=256, interleave=1,
                     one_round_window=256, o_dims=())
    old, new = policy(ex, dev, False), policy(ex, dev, True)
    model_local = -(-(1 << 20) // W)
    reps = int(os.environ.get("K31_TR_REPS", "15"))
    batches = [int(b) for b in os.environ.get(
        "K31_TR_BATCHES", "16,18,20,22,24,26,28,32").split(",")]
    sweep = [int(c) for c in os.environ.get("K31_TR_CLUSTERS", "2,4,8").split(",")]
    for B in batches:
        T = B * Q
        ctxs = torch.tensor(ctx_for(vecs, B), dtype=torch.int64, device=dev)
        G = (ctxs[:, None] - Q + 1 + torch.arange(Q, device=dev)[None, :]).reshape(-1)
        Lr = local_len_of(G, rank, W, 1).to(torch.int32)
        G32 = G.to(torch.int32)
        Lmax = int(Lr.max())
        stride = (Lmax + 63) // 64 * 64
        gen = torch.Generator(device=dev)
        gen.manual_seed(1234 + rank + 17 * B)
        logits = torch.randn(T, stride, generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty(T, K, dtype=torch.int32, device=dev)
        cnt = torch.empty(T, dtype=torch.int32, device=dev)
        ml = min(model_local, stride)
        arms = [("old", *old(T, ml)), ("fit", *new(T, ml))]
        for c in sweep:
            arms.append((f"c{c}", c, dxd.DcpxDecode._segmax(
                types.SimpleNamespace(sms=torch.cuda.get_device_properties(dev)
                                      .multi_processor_count,
                                      smem=228 * 1024, segmax_override=-1), T, c, ml)))
        ref = None
        for name, cluster, segmax in arms:
            kw = dict(max_local_len=ml, cluster=cluster, segmax=segmax, tie_order="rank",
                      one_round=True, window="center", small_ctx=1 << 30)
            try:
                ex.topk_merge(logits, Lr, G32, out, cnt, **kw)
                torch.cuda.synchronize()
                ex.check()
            except Exception as e:  # noqa: BLE001 - e.g. a cluster shape that cannot launch
                if rank == 0:
                    print(f"T{T} {name}: cl{cluster} seg{segmax} failed: {e!r}", flush=True)
                continue
            sets = (torch.sort(out, dim=1).values.clone(), cnt.clone())
            same = ref is None or (torch.equal(sets[0], ref[0]) and torch.equal(sets[1], ref[1]))
            ref = ref or sets
            fn = lambda kw=kw: ex.topk_merge(logits, Lr, G32, out, cnt, **kw)  # noqa: E731
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            ts = []
            for _ in range(reps):
                dist.barrier()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                graph.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / 10)
            del graph
            med = torch.tensor([statistics.median(ts)], device=dev)
            allm = [torch.zeros_like(med) for _ in range(W)]
            dist.all_gather(allm, med)
            allm = sorted(float(x) for x in allm)
            sd = torch.tensor([int(same)], device=dev)
            dist.all_reduce(sd, op=dist.ReduceOp.MIN)
            nres = ex.resident_clusters(cluster, segmax)
            if rank == 0:
                rec = dict(T=T, B=B, arm=name, cluster=cluster, segmax=segmax, nres=nres,
                           us_med=round(statistics.median(allm), 2),
                           us_max=round(allm[-1], 2), same=bool(sd.item()),
                           ctx_max=int(G.max()), mtok=int(ctxs.sum()))
                print("ROW " + json.dumps(rec), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
