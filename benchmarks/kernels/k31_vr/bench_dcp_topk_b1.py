# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP top-k merge at the small-batch decode shapes (dev tool, torchrun).

Times ``DcpExchange.topk_merge`` with the decode wiring's launch parameters
(``DcpxDecode.launch_params``, block-table-free positions, ties by global id)
for each variant in ``K31_TK_VARIANTS``:

* ``r8``: the int1 default (octave round R8 -> R4 -> 16-key refine: 3 rounds)
* ``1rc``: one round, window centred at the local estimate of the K-th key
* ``1rt``: one round, window ending at the local maximum

Logits: ``K31_TK_LOGITS`` = a .pt of global bf16 rows ({"rows": [tensor, ...]},
e.g. real indexer logits re-assembled from a dump), tiled to T rows; else
N(0, 1). Prints per-rank median / max over ranks, path counters and (with
``K31_TK_PHASES=1``) the leader's phase timestamps of the last graph call.
"""

import os
import statistics

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    STAT_NAMES,
    DcpExchange,
    local_len_of,
    local_to_global,
)

K = 2048
IL = int(os.environ.get("K31_TK_IL", "896"))
VARIANTS = {
    "r8": dict(one_round=False, small_ctx=1 << 30),
    "1rc": dict(one_round=True, window="center", small_ctx=0),
    "1rt": dict(one_round=True, window="top", small_ctx=0),
}


def launch_params(rows: int, max_local: int, sms: int) -> tuple[int, int]:
    cluster = 1
    while cluster < 8 and rows * cluster * 2 <= 800:
        cluster *= 2
    seg = (-(-max_local // cluster) + 7) // 8 * 8
    per_sm = -(-rows * cluster // sms)
    fits = (228 * 1024 // per_sm - 20 * 1024) // 2
    return cluster, (max(seg, 8192) if seg <= min(48 * 1024, fits) else 0)


def main():
    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ex = DcpExchange(dist.group.WORLD, dev, cap=1536, cap1=256, interleave=IL,
                     one_round_window=int(os.environ.get("K31_TK_NW1", "512")), o_dims=())
    src = os.environ.get("K31_TK_LOGITS")
    rows_src = torch.load(src)["rows"] if src else None
    dump = os.environ.get("K31_TK_DUMP")  # "<prefix>.r{r}.<suffix>" of a 4-rank IL 1024 dump
    if dump:
        parts = [torch.load(dump.format(r=r), weights_only=False) for r in range(4)]
        rows_src = []
        for t in range(parts[0]["logits"].shape[0]):
            Gt = int(parts[0]["glob"][t])
            row = torch.empty(Gt, dtype=torch.bfloat16)
            for r, d in enumerate(parts):
                n = int(d["local"][t])
                row[local_to_global(torch.arange(n), r, 4, 1024)] = d["logits"][t, :n]
            rows_src.append(row)
        crop = int(os.environ.get("K31_TK_CROP", "0"))
        if crop:  # prefixes of the long rows as shorter contexts
            rows_src = [r[: crop + i] for i, r in enumerate(rows_src) if r.numel() > crop]
    shapes = os.environ.get("K31_TK_SHAPES", "8:139000,64:192000,4:148000")
    variants = os.environ.get("K31_TK_VARIANTS", "r8,1rc").split(",")
    model_local = int(os.environ.get("K31_TK_MAX_LOCAL", str(-(-1048576 // (W * IL)) * IL)))
    reps = int(os.environ.get("K31_TK_REPS", "15"))
    for shape in shapes.split(","):
        T, ctx = (int(v) for v in shape.split(":"))
        if rows_src is not None:
            full_rows = [rows_src[i % len(rows_src)] for i in range(T)]
            Gs = [int(r.numel()) for r in full_rows]
        else:
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + T)
            Gs = [ctx - T + 1 + i for i in range(T)]
            mix = os.environ.get("K31_TK_CTXS")  # requests' contexts, T / len rows each
            if mix and ctx == 0:
                cl_ = [int(c) for c in mix.split(",")]
                qn = T // len(cl_)
                Gs = [c - qn + 1 + i for c in cl_ for i in range(qn)]
            full_rows = [torch.randn(G, generator=g, device=dev).to(torch.bfloat16)
                         for G in Gs]
        G = torch.tensor(Gs, dtype=torch.int32, device=dev)
        Lr = local_len_of(G.long(), rank, W, IL).to(torch.int32)
        Lmax = int(Lr.max())
        stride = (Lmax + 63) // 64 * 64
        local = torch.full((T, stride), float("nan"), dtype=torch.bfloat16, device=dev)
        for t, row in enumerate(full_rows):
            row = row.to(dev)
            pos = torch.arange(int(Lr[t]), device=dev)
            local[t, : pos.numel()] = row[local_to_global(pos, rank, W, IL)]
        cluster, segmax = launch_params(T, min(model_local, stride), sms)
        if os.environ.get("K31_TK_CLUSTER"):  # sweep: CTAs per row (keys in SMEM)
            cluster = int(os.environ["K31_TK_CLUSTER"])
            segmax = max(8192, (-(-stride // cluster) + 7) // 8 * 8)
        out = torch.empty(T, K, dtype=torch.int32, device=dev)
        cnt = torch.empty(T, dtype=torch.int32, device=dev)
        ref = None
        for v in variants:
            kw = dict(VARIANTS[v], max_local_len=min(model_local, stride),
                      cluster=cluster, segmax=segmax)
            stats = torch.zeros(8, dtype=torch.int32, device=dev)
            ex.topk_merge(local, Lr, G, out, cnt, stats=stats, **kw)
            torch.cuda.synchronize()
            ex.check()
            o = (out.clone(), cnt.clone())
            same = ref is None or (torch.equal(o[0], ref[0]) and torch.equal(o[1], ref[1]))
            ref = ref or o
            fn = lambda kw=kw: ex.topk_merge(local, Lr, G, out, cnt, **kw)  # noqa: E731
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
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
                ts.append(e0.elapsed_time(e1) * 1000 / 20)
            med = torch.tensor([statistics.median(ts)], device=dev)
            allm = [torch.zeros_like(med) for _ in range(W)]
            dist.all_gather(allm, med)
            allm = sorted(float(x) for x in allm)
            sd = torch.tensor([int(same)], device=dev)
            dist.all_reduce(sd, op=dist.ReduceOp.MIN)
            dist.all_reduce(stats)
            if rank == 0:
                st = {n: int(x) for n, x in zip(STAT_NAMES, stats.tolist()) if int(x)}
                print(f"T{T} G{min(Gs)}-{max(Gs)} cl{cluster} seg{segmax} {v:4s}: "
                      f"{statistics.median(allm):.2f} / {allm[-1]:.2f} us (median / max rank)"
                      f"  same-as-first={bool(sd.item())}  paths {st}", flush=True)
            del graph
            if os.environ.get("K31_TK_PHASES"):
                dbg = torch.zeros(32, dtype=torch.int64, device=dev)
                fn2 = lambda kw=kw: ex.topk_merge(  # noqa: E731
                    local, Lr, G, out, cnt, dbg=dbg, **kw)
                g2 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g2):
                    for _ in range(20):
                        fn2()
                dist.barrier()
                torch.cuda.synchronize()
                g2.replay()
                torch.cuda.synchronize()
                allg = [torch.zeros_like(dbg) for _ in range(W)]
                dist.all_gather(allg, dbg)
                if rank == 0:
                    for r_ in (0, W - 1):
                        d_ = allg[r_].tolist()
                        ph = [(i, round((d_[i] - d_[0]) / 1000, 2))
                              for i in range(1, 32) if d_[i]]
                        print(f"    phases rank {r_}: {ph}", flush=True)
                del g2
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
