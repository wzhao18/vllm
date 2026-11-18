# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the Kimi K3.1 decode DSA exact top-k over DCP-sharded indexer logits
(dsa_kernels.dcp_exchange DcpExchange.topk_merge, the decode wiring's launch shape).

    torchrun --nproc-per-node 4 benchmarks/kernels/bench_k31_dcp_topk.py \
        --reqs 4,32,96 --q 4 --ctx 135000

Per case: us per call (CUDA graph of --calls launches, replayed --reps times, max over
ranks of the median), the per-kernel CUPTI split (torch profiler), the merge kernel's
per-row phase stamps (dbg) and the path counters (stats).
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import decode as dd
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    STAT_NAMES,
    DcpExchange,
    local_len_of,
)

K = 2048
PHASES = {
    0: "row start", 1: "pass A", 2: "S1", 16: "estimate", 17: "window counts",
    4: "1R send start", 19: "1R header", 20: "1R sent", 5: "1R polled",
    21: "1R accumulated", 22: "1R resolved", 23: "1R done", 6: "quota", 7: "WRITE",
    12: "bitmap", 24: "offsets", 15: "enumerated", 11: "ties", 9: "s9", 10: "s10",
    14: "s14", 25: "s25", 26: "s26", 27: "s27", 28: "s28", 8: "row end",
}


def make_logits(case, T, width, Lloc, gen, dev):
    x = torch.randn(T, width, generator=gen, device=dev)
    if case == "normal":
        pass
    elif case == "relu":  # indexer-like: sum of ReLU scores, many small, heavy right tail
        x = torch.relu(x + 0.3) * 2.0
    elif case == "hot_recent":  # recent tokens score much higher (local position ~ L)
        pos = torch.arange(width, device=dev)[None, :]
        hot = (pos >= (Lloc[:, None] - 512)) & (pos < Lloc[:, None])
        x = x + hot * 6.0
    elif case == "lognormal":
        x = torch.exp(x * 0.8)
    else:
        raise ValueError(case)
    return x.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="4,32,96")
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=135000)
    ap.add_argument("--spread", type=float, default=0.0, help="ctx uniform in ctx*(1 +- spread)")
    ap.add_argument("--cases", default="normal,relu,hot_recent")
    ap.add_argument("--max-model-len", type=int, default=1 << 20)
    ap.add_argument("--max-local", type=int, default=0, help="override the launch bound")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--spin", type=float, default=5.0, help="peer spin limit (s; ncu: long)")
    ap.add_argument("--warm", type=int, default=0, help="8K^3 bf16 matmuls before each replay")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    props = torch.cuda.get_device_properties(dev)
    reqs = [int(x) for x in a.reqs.split(",")]
    cap = 512
    ex = DcpExchange(dist.group.WORLD, dev, cap=cap, cap1=cap, one_round_window=256,
                     heads=96, o_dims=(), topk=K, interleave=1)
    ex.set_spin_timeout(a.spin)
    wa = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    wb = torch.empty_like(wa)
    ns =SimpleNamespace(ex=ex, sms=props.multi_processor_count,
                         smem=int(getattr(props, "shared_memory_per_multiprocessor", 0) or 228 * 1024),
                         fit_wave=True)
    ns._segmax = lambda r, c, ml: dd.DcpxDecode._segmax(ns, r, c, ml)
    model_local = -(-a.max_model_len // W)
    max_local = a.max_local or model_local
    width = model_local
    bs = 64
    results = []
    for B in reqs:
        T = B * a.q
        gen = torch.Generator(device=dev)
        gen.manual_seed(1234 + B)  # same contexts on every rank
        if a.spread > 0:
            u = torch.rand(B, generator=gen, device=dev) * 2 - 1
            Greq = (a.ctx * (1 + a.spread * u)).to(torch.int32)
        else:
            Greq = torch.full((B,), a.ctx, dtype=torch.int32, device=dev)
        j = torch.arange(a.q, device=dev, dtype=torch.int32)
        G = (Greq[:, None] - (a.q - 1 - j)[None, :]).reshape(-1).contiguous()
        L = local_len_of(G, rank, W, 1).to(torch.int32).contiguous()
        nblk = -(-width // bs)
        bt = torch.arange(B * nblk, dtype=torch.int32, device=dev).view(B, nblk)
        req_of_row = (torch.arange(T, device=dev, dtype=torch.int32) // a.q).contiguous()
        out = torch.empty(T, K, dtype=torch.int32, device=dev)
        cnt = torch.empty(T, dtype=torch.int32, device=dev)
        cluster, segmax = dd.DcpxDecode.launch_params(ns, T, max_local)
        for case in a.cases.split(","):
            g2 = torch.Generator(device=dev)
            g2.manual_seed(99 + 7 * rank + B)
            logits = make_logits(case, T, width, L, g2, dev)

            def call(dbg=None, stats=None):
                ex.topk_merge(logits, L, G, out, cnt, max_local_len=max_local,
                              cluster=cluster, segmax=segmax, block_table=bt,
                              block_size=bs, block_stride=bs, req_of_row=req_of_row,
                              one_round=True, window="center", small_ctx=1 << 30,
                              dbg=dbg, stats=stats)

            for _ in range(3):
                call()
            torch.cuda.synchronize()
            ex.check()
            tot = cnt.sum().clone()
            dist.all_reduce(tot)
            ok_count = int(tot) == T * K
            # path counters + phase stamps (eager, one call)
            stats = torch.zeros(8, dtype=torch.int32, device=dev)
            dbg = torch.zeros(T * 64, dtype=torch.int64, device=dev)
            dist.barrier()
            for _ in range(a.warm):
                torch.matmul(wa, wa, out=wb)
            call(dbg=dbg, stats=stats)
            torch.cuda.synchronize()
            d = dbg.view(T, 64).cpu()
            t0 = int(d[:, 0][d[:, 0] > 0].min()) if (d[:, 0] > 0).any() else 0
            phases = {}
            for i, name in PHASES.items():
                v = d[:, i]
                v = v[v > 0]
                if len(v):
                    rel = (v - t0).double() / 1e3
                    phases[f"{i}:{name}"] = (round(float(rel.median()), 2), round(float(rel.max()), 2), len(v))
            # graph timing
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(a.calls):
                        call()
            torch.cuda.synchronize()
            times = []
            for _ in range(a.reps):
                dist.barrier()
                torch.cuda.synchronize()
                for _ in range(a.warm):  # keep the SM clocks up (the GPU idles between reps)
                    torch.matmul(wa, wa, out=wb)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1e3 / a.calls)
            times.sort()
            med = torch.tensor([times[len(times) // 2]], device=dev)
            dist.all_reduce(med, op=dist.ReduceOp.MAX)
            kern = {}
            if a.profile:
                from torch.profiler import ProfilerActivity, profile

                dist.barrier()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        g.replay()
                    torch.cuda.synchronize()
                for ev in prof.key_averages():
                    if getattr(ev, "self_device_time_total", 0) > 0:
                        nm = ev.key
                        n = ev.count
                        kern[nm[:60]] = (round(ev.self_device_time_total / max(n, 1), 2), n)
            torch.cuda.synchronize()
            ex.check()
            st = torch.zeros(8, dtype=torch.int32, device=dev)
            st.copy_(stats)
            dist.all_reduce(st)
            r = dict(B=B, T=T, case=case, cluster=cluster, segmax=segmax, max_local=max_local,
                     us_per_call=round(float(med), 2), ok_count=ok_count,
                     stats=dict(zip(STAT_NAMES, st.tolist())), phases=phases, kernels=kern,
                     Lmean=float(L.float().mean()))
            results.append(r)
            if rank == 0:
                print(json.dumps(r), flush=True)
            del g
    if rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
