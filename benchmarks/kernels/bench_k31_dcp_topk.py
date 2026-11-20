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
PHASES_V2 = {24: "c:pass1", 25: "c:offsets", 26: "c:pass2", 0: "start", 9: "loads issued", 10: "sampled", 1: "theta",
             2: "cand", 12: "fine hist", 13: "e", 14: "window", 15: "band pick",
             16: "band ids", 3: "msg", 4: "polled", 19: "r:hdr", 20: "r:tau", 21: "r:groups",
             22: "r:class", 5: "resolved", 6: "tie round", 23: "w:bitmap(last)",
             7: "radix", 17: "gtt", 18: "write scan", 27: "r:pre", 28: "slots", 8: "end"}
STATS_V2 = ("rows", "one_round", "tie_round", "radix", "select_all", "full_pass", "retheta", "rebase")
PHASES_SP = {0: "start", 9: "landed+max", 10: "sampled", 1: "theta", 2: "pass", 13: "e", 14: "msg built",
             3: "sent", 4: "polled", 5: "resolved", 6: "radix", 7: "selected", 8: "end"}
STATS_SP = ("rows", "one_round", "id_radix", "key_radix", "select_all", "-", "-", "-")


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
    ap.add_argument("--evict-mb", type=int, default=0,
                    help="in-step emulation: stream this many MB through L2 before each call (other "
                         "kernels' code and data evict the top-k code), then re-touch the logits")
    ap.add_argument("--evict-touch", type=int, default=0,
                    help="after --evict-mb, re-read the top-k's small data (mailboxes, block table, "
                         "lengths, epochs, outputs) so only code stays evicted")
    ap.add_argument("--skew-us", type=float, default=0.0,
                    help="the last rank sleeps this long before each call (in-step rank skew)")
    ap.add_argument("--impl", default="merge", choices=["merge", "v2", "sp"])
    ap.add_argument("--check", type=int, default=1, help="v2: compare with topk_merge")
    ap.add_argument("--k", type=int, default=2048, help="top-k (1-GPU ncu emulation: K / DCP)")
    a = ap.parse_args()
    global K
    K = a.k

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
            if case.startswith("dump:"):  # logits / lengths dumped from a decode step
                dmp = torch.load(case[5:].replace("{rank}", str(rank)))
                assert dmp["logits"].shape[0] == T, (dmp["logits"].shape, T)
                L = dmp["local"].to(dev, torch.int32).contiguous()
                G = dmp["glob"].to(dev, torch.int32).contiguous()
                logits = torch.full((T, width), float("nan"), dtype=torch.bfloat16, device=dev)
                logits[:, : dmp["logits"].shape[1]] = dmp["logits"].to(dev)
                if rank == 0:
                    print("dump stats", dmp["stats"].tolist(), "L", int(L.min()), int(L.max()), flush=True)
            else:
                logits = make_logits(case, T, width, L, g2, dev)

            segcap = 0

            ev = (torch.empty(a.evict_mb << 18, dtype=torch.int32, device=dev) if a.evict_mb else None)
            ev2 = torch.empty_like(ev) if ev is not None else None

            def call(dbg=None, stats=None):
                if ev is not None:
                    ev2.copy_(ev)
                    torch.amax(logits, dim=1)
                    if a.evict_touch:
                        for t_ in (ex.v2_box, bt, L, G, req_of_row, ex.v2_epoch, out, cnt):
                            torch.amax(t_)
                if a.skew_us > 0 and rank == W - 1:
                    torch.cuda._sleep(int(a.skew_us * 2400))
                if a.impl in ("v2", "sp"):
                    ex.topk_v2(logits, L, G, out, cnt, block_table=bt,
                               block_size=bs, block_stride=bs, req_of_row=req_of_row,
                               stats=stats, dbg=dbg, single_path=a.impl == "sp")
                    return
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
            mism = -1
            if a.impl in ("v2", "sp") and a.check:
                o2, c2 = out.clone(), cnt.clone()
                ex.topk_merge(logits, L, G, out, cnt, max_local_len=max_local,
                              cluster=cluster, segmax=segmax, block_table=bt,
                              block_size=bs, block_stride=bs, req_of_row=req_of_row,
                              one_round=True, window="center", small_ctx=1 << 30)
                torch.cuda.synchronize()
                ex.check()
                bad = (c2 != cnt) | (o2 != out).any(1)
                m = torch.tensor([int(bad.sum())], device=dev)
                dist.all_reduce(m)
                mism = int(m)
            # path counters + phase stamps (eager, one call)
            stats = torch.zeros(8, dtype=torch.int32, device=dev)
            dbg = torch.zeros(T * 64, dtype=torch.int64, device=dev)
            phmap = PHASES_SP if a.impl == "sp" else PHASES_V2 if a.impl == "v2" else PHASES
            dist.barrier()
            for _ in range(a.warm):
                torch.matmul(wa, wa, out=wb)
            call(stats=stats)
            torch.cuda.synchronize()
            # stamps of the last of a.calls back-to-back graph calls: the exchange aligns the
            # ranks after the first call, so no eager launch skew between processes
            gd = torch.cuda.CUDAGraph()
            sd = torch.cuda.Stream()
            sd.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sd):
                call(dbg=dbg)
                with torch.cuda.graph(gd, stream=sd):
                    for _ in range(a.calls):
                        call(dbg=dbg)
            torch.cuda.synchronize()
            dist.barrier()
            gd.replay()
            torch.cuda.synchronize()
            d = dbg.view(T, 64).cpu()
            t0 = int(d[:, 0][d[:, 0] > 0].min()) if (d[:, 0] > 0).any() else 0
            phases = {}
            if a.impl in ("v2", "sp"):  # SM clock from (clock64, globaltimer) at row start / end
                c0, c1, t0n, t1n = d[:, 32].double(), d[:, 40].double(), d[:, 0].double(), d[:, 8].double()
                okr = t1n > t0n
                phases["ncand"] = (float(d[:, 31].double().median()), 0, 0)
                for bi in range(3):  # candidate batches: this thread's loads landed (clock64)
                    v = d[:, 24 + bi].double()
                    okb = v > 0
                    if okb.any():
                        phases[f"cb{bi}:loaded"] = (round(float(((v - d[:, 32].double())[okb]).median()) / 2.4e3, 2), 0, 0)
                phases["nband"] = (float(d[:, 30].double().median()), 0, 0)
                ghz = float(((c1 - c0)[okr] / (t1n - t0n)[okr]).median())
                phases["ghz"] = (round(ghz, 3), 0, int(okr.sum()))
                # per-row clock64 deltas from the row start (globaltimer is too coarse)
                for i, name in phmap.items():
                    v = d[:, 32 + i].double()
                    ok_i = v > 0
                    if ok_i.any() and i:
                        rel = (v - c0)[ok_i] / ghz / 1e3
                        phases[f"c{i}:{name}"] = (round(float(rel.median()), 2), round(float(rel.max()), 2), int(ok_i.sum()))
            for i, name in phmap.items():
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
                     us_per_call=round(float(med), 2), ok_count=ok_count, impl=a.impl,
                     segcap=segcap, mismatch_rows=mism,
                     stats=dict(zip(STATS_SP if a.impl == "sp" else STATS_V2 if a.impl == "v2" else STAT_NAMES, st.tolist())), phases=phases, kernels=kern,
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
