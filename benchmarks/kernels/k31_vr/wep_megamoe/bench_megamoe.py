# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 MegaMoE at wide-EP per-rank shapes, on one node.

Emulates the per-rank work of EP16 / EP32 (and smaller) decode on ``--nprocs``
local GPUs: each rank owns ``E`` experts (56 = EP16, 28 = EP32, 14 = EP64 of
the 896-expert model), routes ``T`` tokens to top-16 of ``nprocs * E`` experts,
so it receives ``16 T`` token-experts, exactly as one rank of the full EP group.
Only the number of peers (dispatch/combine fan-out) differs from the target.

Shapes: latent hidden 3584, intermediate 3072, MXFP4 weights (UE8M0 / 32),
MXFP8 activations, SiTU (alpha 4, beta 25), no fused shared expert -- the
K3.1 DEP / DCPk x DPm decode configuration (``DeepGemmMegaMoEExperts``).

Timing: one CUDA graph of ``--launches`` back-to-back kernel launches (each
streams every local expert's weights, far above L2), replayed ``--reps`` times;
per-launch time is the max over ranks of (event time / launches). Optional
CUPTI per-kernel durations (``--cupti``).

Byte floor: ``E * 17.55 MB / BW`` (``--floor-tbps``, default 13.4 TB/s, the
W0 / MODEL.md definition, 73.3 us at E=56).

Run (one node, 4 GPUs, inside the K3 VR image):
  python3 bench_megamoe.py --experts-per-rank 56,28,14 --tokens 64,96 --json out.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import importlib

import torch
import torch.distributed as dist


def import_dg():
    """The DeepGEMM vLLM uses: site-packages first, then the vendored copy."""
    try:
        return importlib.import_module("deep_gemm")
    except ImportError:
        return importlib.import_module("vllm.third_party.deep_gemm")


def dg_utils(dg):
    return importlib.import_module(dg.__name__ + ".utils")

HIDDEN = 3584
INTER = 3072
TOPK = 16
BYTES_PER_EXPERT = (
    2 * INTER * HIDDEN // 2  # w13 fp4
    + 2 * INTER * HIDDEN // 32  # w13 ue8m0
    + HIDDEN * INTER // 2  # w2 fp4
    + HIDDEN * INTER // 32  # w2 ue8m0
)


def make_weights(dg, num_experts: int, seed: int, localize: bool = False):
    """MXFP4 weights for ``num_experts`` experts in the MegaMoE layout.

    Casting 56 experts of BF16 is slow, so a few distinct experts are cast and
    the rest are copies in their own memory (identical byte traffic)."""
    per_token_cast_to_fp4 = dg_utils(dg).per_token_cast_to_fp4

    g = torch.Generator(device="cuda").manual_seed(seed)
    num_distinct = min(num_experts, 4)

    def cast(n: int, k: int):
        w = torch.empty((num_experts, n, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((num_experts, n, k // 32), device="cuda", dtype=torch.float)
        for i in range(num_distinct):
            bf = 0.1 * torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=g)
            w[i], sf[i] = per_token_cast_to_fp4(bf, use_ue8m0=True, gran_k=32)
        for i in range(num_distinct, num_experts):
            w[i].copy_(w[i % num_distinct])
            sf[i].copy_(sf[i % num_distinct])
        sf = dg.transform_sf_into_required_layout(sf, n, k, (1, 32), num_experts)
        return w, sf

    l1 = cast(2 * INTER, HIDDEN)
    l2 = cast(HIDDEN, INTER)
    torch.cuda.empty_cache()
    l1, l2 = dg.transform_weights_for_mega_moe(l1, l2, activation="situ")
    if localize:
        # Split N across the device locality domains (MLOPart): each die's SMs
        # stream the weight half in their own HBM.
        l1 = (dg.localize(l1[0]), l1[1])
        l2 = (dg.localize(l2[0]), l2[1])
        dg.destroy_localizer()
        torch.cuda.empty_cache()
    return l1, l2


def make_routing(mode: str, rank: int, nranks: int, tokens: int, num_experts: int, seed: int):
    """``[tokens, 16]`` expert ids and weights.

    balanced: global token g picks experts (16 g + j) mod E_total -> every expert
      gets exactly the same count (the ``vllm bench forward`` dummy routing).
    random: top-16 of i.i.d. Gaussian scores (multinomial-like spread).
    zipf: expert popularity ~ 1/(rank+1)^1.25 (hot experts), sampled per token
      without replacement.
    """
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    if mode == "balanced":
        gid = rank * tokens + torch.arange(tokens, device="cuda")
        idx = (gid[:, None] * TOPK + torch.arange(TOPK, device="cuda")[None, :]) % num_experts
    elif mode == "random":
        scores = torch.randn((tokens, num_experts), device="cuda", generator=g)
        idx = scores.topk(TOPK, dim=-1).indices
    elif mode == "zipf":
        p = 1.0 / torch.arange(1, num_experts + 1, device="cuda", dtype=torch.float) ** 1.25
        perm = torch.randperm(num_experts, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))
        p = p[perm]
        idx = torch.multinomial(p.expand(tokens, -1), TOPK, replacement=False, generator=g)
    else:
        raise ValueError(mode)
    w = torch.rand((tokens, TOPK), device="cuda", generator=g) + 0.1
    w = w / w.sum(-1, keepdim=True)
    return idx.to(torch.int64), w.to(torch.float32)


def run_case(dg, group, rank, nranks, args, epr: int, tokens: int, mode: str, buf, weights):
    per_token_cast_to_fp8 = dg_utils(dg).per_token_cast_to_fp8

    num_experts = epr * nranks
    torch.manual_seed(rank)
    x = torch.randn((tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    idx, tw = make_routing(mode, rank, nranks, tokens, num_experts, args.seed)
    buf.x[:tokens].copy_(x_fp8)
    buf.x_sf[:tokens].copy_(x_sf)
    buf.topk_idx[:tokens].copy_(idx)
    buf.topk_weights[:tokens].copy_(tw)
    y = torch.empty((tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
    l1, l2 = weights

    def launch():
        dg.fp8_fp4_mega_moe(
            y, l1, l2, buf,
            activation="situ", activation_clamp=None, fast_math=True,
            activation_alpha=4.0, activation_beta=25.0,
        )

    if args.pdl:
        dg.set_pdl(True)
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    dist.barrier(group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.launches):
            launch()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group)

    samples = []
    for _ in range(args.samples):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier(group)
        torch.cuda.synchronize()
        s.record()
        for _ in range(args.reps):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1e3 / (args.reps * args.launches))
    med = statistics.median(samples)
    t = torch.tensor([med], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    t_max = t.item()
    t_min = torch.tensor([med], device="cuda", dtype=torch.float64)
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN, group=group)

    # Received token-experts and touched experts on this rank.
    gathered = [torch.empty_like(idx) for _ in range(nranks)]
    dist.all_gather(gathered, idx, group=group)
    allidx = torch.cat(gathered)
    lo, hi = rank * epr, (rank + 1) * epr
    local = allidx[(allidx >= lo) & (allidx < hi)]
    recv = torch.tensor([local.numel(), torch.unique(local).numel(),
                         torch.bincount(local - lo, minlength=epr).max().item()],
                        device="cuda", dtype=torch.float64)
    recv_max = recv.clone()
    dist.all_reduce(recv_max, op=dist.ReduceOp.MAX, group=group)

    cupti = None
    if args.cupti:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
        durs = [ev.device_time for ev in prof.events()
                if "mega_moe" in ev.name and ev.device_time > 0]
        if durs:
            cupti = statistics.median(durs)
        c = torch.tensor([cupti or 0.0], device="cuda", dtype=torch.float64)
        dist.all_reduce(c, op=dist.ReduceOp.MAX, group=group)
        cupti = c.item()

    y_out = None
    if args.save_y or args.ref_y:
        y.zero_()
        launch()
        torch.cuda.synchronize()
        y_out = y.clone().cpu()

    trace = None
    if args.trace:
        # Per-SM %globaltimer stamps (trace header only).
        nmax = buf.x.shape[0]
        assert tokens <= nmax - 8
        stamps = buf.x[nmax - 8:].contiguous().view(torch.int64).flatten()
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        stamps.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)
        graph.replay()  # stamps of the last of `--launches` back-to-back launches (steady state)
        torch.cuda.synchronize()
        trace = buf.x[nmax - 8:].contiguous().view(torch.int64).flatten()[: nsm * 16].view(nsm, 16).cpu()
        dist.barrier(group)

    floor = epr * BYTES_PER_EXPERT / (args.floor_tbps * 1e12) * 1e6
    block_m = dg.get_block_m_for_mega_moe(nranks, num_experts, buf.num_max_tokens_per_rank,
                                          tokens, TOPK, "fp8xfp4") \
        if hasattr(dg, "get_block_m_for_mega_moe") else None
    del graph
    out = dict(
        experts_per_rank=epr, nranks=nranks, num_experts=num_experts, tokens=tokens,
        routing=mode, pdl=bool(args.pdl), localize=bool(args.localize), block_m=block_m,
        us=t_max, us_min_rank=t_min.item(), cupti_us=cupti, samples=samples,
        recv_token_experts_max=int(recv_max[0].item()),
        touched_experts_max=int(recv_max[1].item()),
        max_tokens_per_expert=int(recv_max[2].item()),
        floor_us=floor, x_floor=t_max / floor,
        eff_tbps=epr * BYTES_PER_EXPERT / (t_max * 1e-6) / 1e12,
    )
    if trace is not None:
        out["trace"] = trace
    if y_out is not None:
        out["y"] = y_out
    return out


def worker(local_rank: int, nprocs: int, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=nprocs,
                            device_id=torch.device(f"cuda:{local_rank}"))
    group = dist.new_group(list(range(nprocs)))
    dg = import_dg()
    if args.header:
        from pathlib import Path

        try:
            from vllm.models.kimi_k3.nvidia.megamoe.overlay import install
        except ImportError:  # an env without this branch: the copy next to the bench
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from overlay import install

        hdrs = dict(h.split("=", 1) for h in args.header)
        cache = Path(os.environ.get("VLLM_CACHE_ROOT", "/tmp")) / "kimi_k3_megamoe"
        root = install(dg, {k: Path(v) for k, v in hdrs.items()}, cache)
        if local_rank == 0:
            print(f"DeepGEMM overlay {root}: {sorted(hdrs)}", flush=True)

    eprs = [int(v) for v in args.experts_per_rank.split(",")]
    toks = [int(v) for v in args.tokens.split(",")]
    modes = args.routing.split(",")
    results = []
    for epr in eprs:
        weights = make_weights(dg, epr, args.seed, bool(args.localize))
        buf = dg.get_symm_buffer_for_mega_moe(
            group, epr * nprocs, max(toks) + (8 if args.trace else 0), TOPK, HIDDEN, INTER,
            mma_type="fp8xfp4", activation="situ")
        for mode in modes:
            for t in toks:
                r = run_case(dg, group, local_rank, nprocs, args, epr, t, mode, buf, weights)
                results.append(r)
                if local_rank == 0:
                    cup = "" if r["cupti_us"] is None else f", cupti {r['cupti_us']:6.1f}"
                    print(f"E/rank {epr:3d} EP{nprocs} T {t:4d} {mode:8s} bm {r['block_m']} | "
                          f"{r['us']:7.1f} us (min rank {r['us_min_rank']:6.1f}{cup}) | "
                          f"floor {r['floor_us']:5.1f} -> {r['x_floor']:.2f}x, {r['eff_tbps']:.1f} TB/s | "
                          f"recv {r['recv_token_experts_max']} touched {r['touched_experts_max']} "
                          f"maxpe {r['max_tokens_per_expert']}", flush=True)
        buf.destroy()
        del weights
        torch.cuda.empty_cache()
        dist.barrier(group)
    if args.save_y:
        torch.save([(r["experts_per_rank"], r["tokens"], r["routing"], r["y"]) for r in results],
                   f"{args.save_y}.rank{local_rank}.pt")
    if args.ref_y:
        ref = {(e, t, m): y for e, t, m, y in torch.load(f"{args.ref_y}.rank{local_rank}.pt")}
        bad = 0
        for r in results:
            y_ref = ref.get((r["experts_per_rank"], r["tokens"], r["routing"]))
            if y_ref is None:
                continue
            same = torch.equal(r["y"], y_ref)
            diff = (r["y"].float() - y_ref.float()).abs().max().item()
            rel = diff / max(y_ref.float().abs().max().item(), 1e-30)
            r["bitwise_equal"], r["max_abs_diff"] = same, diff
            bad += not same
            print(f"[rank{local_rank}] check E{r['experts_per_rank']} T{r['tokens']} {r['routing']}: "
                  f"bitwise {'EQUAL' if same else 'DIFF'} max|d| {diff:.3e} (rel {rel:.2e})", flush=True)
        t = torch.tensor([bad], device="cuda")
        dist.all_reduce(t, group=group)
        if local_rank == 0:
            print(f"check: {int(t.item())} mismatching case-ranks", flush=True)
    for r in results:
        r.pop("y", None)
    if args.trace:
        traces = [{k: v for k, v in r.items() if k != "samples"} for r in results]
        torch.save(traces, f"{args.trace}.rank{local_rank}.pt")
    for r in results:
        r.pop("trace", None)
    if local_rank == 0 and args.json:
        meta = dict(argv=sys.argv, time=time.strftime("%F %T"),
                    device=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0).multi_processor_count,
                    dg_file=dg.__file__, env={k: v for k, v in os.environ.items() if k.startswith(("DG_", "K31"))})
        with open(args.json, "w") as f:
            json.dump(dict(meta=meta, results=results), f, indent=1)
    dist.barrier(group)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nprocs", type=int, default=4)
    p.add_argument("--experts-per-rank", default="56,28,14")
    p.add_argument("--tokens", default="16,32,64,96,128")
    p.add_argument("--routing", default="balanced")
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--floor-tbps", type=float, default=13.4)
    p.add_argument("--pdl", type=int, default=0)
    p.add_argument("--cupti", type=int, default=0)
    p.add_argument("--header", action="append", default=[],
                   help="replace a DeepGEMM header: deep_gemm/impls/x.cuh=/path/x.cuh")
    p.add_argument("--trace", default="",
                   help="read per-SM phase stamps (trace header) and save them to this .pt")
    p.add_argument("--save-y", default="", help="save each case's output (per rank) to <path>.rank<i>.pt")
    p.add_argument("--ref-y", default="", help="compare each case's output to a --save-y reference")
    p.add_argument("--localize", type=int, default=0,
                   help="localize expert weights into the device locality domains")
    p.add_argument("--port", type=int, default=29517)
    p.add_argument("--json", default="")
    args = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.nprocs, args), nprocs=args.nprocs)


if __name__ == "__main__":
    main()
