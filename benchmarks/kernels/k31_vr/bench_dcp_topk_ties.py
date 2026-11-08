# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost of the stock tie rule in DcpExchange.topk_merge at production decode shapes.

Ties at the K-th key go by ascending global token id (the stock merge's rule,
``tie_order="global"``: one extra NVLink round when the ties straddle two or more
ranks) or by (rank, position) (``"rank"``: no extra round, the same selected score
multiset but not the stock key set). One process per GPU (DCP = world size), the
production decode wiring: DcpxDecode.launch_params (fit-wave), interleave 1,
max_model_len 1M, the centred 256-key one-round window on rows <= 256, KV-slot output
through a block table, AgentX contexts of the q (decode_ctx_vectors.json).

Per (rows, logits) the two tie orders alternate A B B A ... (``--reps`` CUDA graphs of
10 back-to-back calls each, us per call, max over ranks); the record also holds the
path counters of one call (tie rounds taken) and whether the two orders selected the
same key sets. Logits: ``randn``, or ``real`` (each row resampled from dumped bf16
indexer logit rows: ties at the K-th key as in serving).

  torchrun --nproc-per-node 4 bench_dcp_topk_ties.py --ctx decode_ctx_vectors.json \
      --q 4 --rows 512 --dists real,randn --dump-dir DIR
"""

import argparse
import glob
import json
import os
import statistics
import types

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode as dxd
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    STAT_NAMES,
    DcpExchange,
    local_len_of,
)

K = 2048
MODEL_LEN = 1 << 20
BLOCK = 64
CTX_KEY = {4: "q4_C48_C160_C256", 8: "q8_C1_C8"}


def ctx_for(vecs, key, B):
    g = vecs[key]
    if str(B) in g:
        return list(g[str(B)])
    out = []
    while len(out) < B:
        for b in sorted((int(k) for k in g), reverse=True):
            while len(out) + b <= B:
                out += g[str(b)]
            if len(out) == B:
                break
        if len(out) < B:
            out += g["1"]
    return out[:B]


def policy(ex, dev):
    props = torch.cuda.get_device_properties(dev)
    ns = types.SimpleNamespace(
        ex=ex,
        sms=props.multi_processor_count,
        smem=int(getattr(props, "shared_memory_per_multiprocessor", 0) or 228 * 1024),
        fit_wave=True,
        segmax_override=-1,  # (older trees read it)
    )
    ns._segmax = lambda rows, c, ml: dxd.DcpxDecode._segmax(ns, rows, c, ml)
    return lambda rows, ml: dxd.DcpxDecode.launch_params(ns, rows, ml)


def real_pools(dump_dir, dev):
    rows = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu")
        lg, loc = d["logits"], d["local"]
        for r in range(lg.shape[0]):
            v = lg[r, : int(loc[r])].float()
            v = v[torch.isfinite(v)]
            if v.numel() > 1024:
                rows.append(v)
    n = max(v.numel() for v in rows)
    pool = torch.zeros(len(rows), n)
    for i, v in enumerate(rows):
        pool[i, : v.numel()] = v
    lens = torch.tensor([v.numel() for v in rows])
    return pool.to(dev), lens.to(dev)


def make_logits(dist_name, T, stride, rank, seed, dev, pools):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed * 1000 + rank)
    if dist_name == "randn":
        return torch.randn(T, stride, generator=gen, device=dev).to(torch.bfloat16)
    pool, lens = pools
    g0 = torch.Generator(device=dev)
    g0.manual_seed(seed)  # the same source row on every rank
    src = torch.randint(0, pool.shape[0], (T,), generator=g0, device=dev)
    out = torch.empty(T, stride, dtype=torch.bfloat16, device=dev)
    for r0 in range(0, T, 64):
        s = src[r0 : r0 + 64]
        u = torch.rand(s.numel(), stride, generator=gen, device=dev)
        idx = (u * lens[s][:, None].float()).long().clamp_max(pool.shape[1] - 1)
        out[r0 : r0 + 64] = torch.gather(pool[s], 1, idx).to(torch.bfloat16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--q", type=int, choices=sorted(CTX_KEY), required=True)
    ap.add_argument("--rows", default="8,16,64")
    ap.add_argument("--dists", default="real,randn")
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    with open(args.ctx) as f:
        vecs = json.load(f)
    rows_list = [int(x) for x in args.rows.split(",")]
    cap = max(64, max(rows_list))
    ex = DcpExchange(
        dist.group.WORLD,
        dev,
        cap=cap,
        cap1=min(cap, getattr(dxd, "ONE_ROUND_ROWS", 256)),
        interleave=1,
        one_round_window=256,
        o_dims=(),
    )
    ex.set_spin_timeout(60.0)
    lp = policy(ex, dev)
    pools = real_pools(args.dump_dir, dev) if "real" in args.dists else None
    model_local = -(-MODEL_LEN // W)
    stats = torch.zeros(8, dtype=torch.int32, device=dev)
    key = CTX_KEY[args.q]
    for T in rows_list:
        B = max(1, T // args.q)
        ctxs = torch.tensor(ctx_for(vecs, key, B), dtype=torch.int64, device=dev)
        G = (ctxs[:, None] - args.q + 1 + torch.arange(args.q, device=dev)[None, :])
        G = G.reshape(-1)[:T]
        Lr = local_len_of(G, rank, W, 1).to(torch.int32)
        G32 = G.to(torch.int32)
        Lmax = int(Lr.max())
        stride = (Lmax + 511) // 512 * 512
        ml = min(model_local, stride)
        cluster, segmax = lp(T, ml)
        nblk = -(-Lmax // BLOCK) + 1
        gperm = torch.Generator(device=dev)
        gperm.manual_seed(77)
        bt = torch.randperm(T * nblk, generator=gperm, device=dev).int().view(T, nblk)
        req = (torch.arange(T, device=dev) // args.q).int()
        for dname in args.dists.split(","):
            logits = make_logits(dname, T, stride, rank, 1234 + T, dev, pools)
            outs, graphs, recs = {}, {}, {}
            for tie in ("rank", "global"):
                out = torch.empty(T, K, dtype=torch.int32, device=dev)
                cnt = torch.empty(T, dtype=torch.int32, device=dev)
                kw = dict(
                    max_local_len=ml,
                    cluster=cluster,
                    segmax=segmax,
                    tie_order=tie,
                    one_round=True,
                    window="center",
                    small_ctx=1 << 30,
                    block_table=bt,
                    block_size=BLOCK,
                    block_stride=BLOCK,
                    req_of_row=req,
                )

                def call(st=None, o=out, c=cnt, kw=kw):
                    ex.topk_merge(logits, Lr, G32, o, c, stats=st, **kw)

                stats.zero_()
                call(st=stats)
                torch.cuda.synchronize()
                ex.check()
                s = stats.clone()
                dist.all_reduce(s)
                c = cnt.clone()
                dist.all_reduce(c)
                ok = bool(torch.equal(c, torch.minimum(G32, torch.full_like(G32, K))))
                keys = [set(out[t, : int(cnt[t])].tolist()) for t in range(T)]
                outs[tie] = keys
                recs[tie] = dict(ok=ok, stats=dict(zip(STAT_NAMES, s.tolist())))
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10):
                        call()
                graphs[tie] = graph
            same = sum(a == b for a, b in zip(outs["rank"], outs["global"]))
            same_t = torch.tensor([same], device=dev)
            dist.all_reduce(same_t, op=dist.ReduceOp.MIN)
            times = {"rank": [], "global": []}
            order = ["rank", "global", "global", "rank"]
            for i in range(args.reps):
                tie = order[i % 4]
                dist.barrier()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                graphs[tie].replay()
                e1.record()
                e1.synchronize()
                times[tie].append(e0.elapsed_time(e1) * 1000 / 10)
            med = {k: statistics.median(v) for k, v in times.items()}
            allm = [None] * W
            dist.all_gather_object(allm, med)
            if rank == 0:
                rec = dict(
                    W=W,
                    q=args.q,
                    rows=T,
                    dist=dname,
                    cluster=cluster,
                    segmax=segmax,
                    Gmax=int(G.max()),
                    rank_us=round(max(m["rank"] for m in allm), 2),
                    global_us=round(max(m["global"] for m in allm), 2),
                    rows_same_keys_min_over_ranks=int(same_t),
                    rank_rec=recs["rank"],
                    global_rec=recs["global"],
                )
                rec["delta_us"] = round(rec["global_us"] - rec["rank_us"], 2)
                print("TIES " + json.dumps(rec), flush=True)
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            del graphs
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
