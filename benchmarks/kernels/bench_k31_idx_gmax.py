# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA decode: fused top-k producer (idx_gmax) vs DeepGEMM paged MQA logits.

One GPU, the DCP4 decode shape: R requests x 4 verify rows, local context ~L keys, MXFP4 (or
FP8) indexer cache with 64-token pages, 32 heads. Checks the logits are bitwise equal to
DeepGEMM's and gmax / NaN counts equal a torch reference; times both kernels in CUDA graphs.

  python benchmarks/kernels/bench_k31_idx_gmax.py --reqs 32,64,96 --ctx 33000 [--fp8]
"""

import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.dsa_kernels import idx_gmax
from vllm.utils import deep_gemm as dgu


def keys_of(x: torch.Tensor) -> torch.Tensor:
    raw = x.view(torch.int16).to(torch.int32) & 0xFFFF
    k = torch.where(raw >= 0x8000, raw ^ 0xFFFF, raw ^ 0x8000)
    nan = (raw & 0x7FFF) > 0x7F80
    return torch.where(nan, torch.zeros_like(k), k), nan


def make(reqs: int, ctx: int, fp4: bool, seed: int, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    rows = reqs * 4
    page = 64
    dbytes = 64 if fp4 else 128
    lens_r = torch.randint(ctx - 2000, ctx + 1, (reqs,), generator=g, device=dev, dtype=torch.int32)
    ctx_rows = (lens_r.repeat_interleave(4) - 3 + torch.arange(4, device=dev, dtype=torch.int32).repeat(reqs))
    nbr = (int(lens_r.max()) + page - 1) // page
    nb = reqs * nbr + 8
    kv = torch.randint(0, 256, (nb, page * (dbytes + 4)), generator=g, device=dev, dtype=torch.int32).to(torch.uint8)
    sf = kv[:, page * dbytes :].view(nb, page, 4)
    if fp4:
        sf.fill_(0x7F)  # UE8M0 1.0
    else:
        kv[:, : page * dbytes].view(torch.float8_e4m3fn).copy_(
            (torch.randn((nb, page * dbytes), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn))
        sf.view(torch.float32).fill_(1.0)
    kv = kv.view(nb, page, 1, dbytes + 4)
    # each 4-byte row slice above must alias the fused layout: [page * D values | page * 4 scales]
    perm = torch.randperm(nb, generator=g, device=dev)[: reqs * nbr].view(reqs, nbr).to(torch.int32)
    bt = torch.zeros((rows, nbr), dtype=torch.int32, device=dev)
    bt[:reqs] = perm
    idx = torch.arange(reqs, device=dev, dtype=torch.int32).repeat_interleave(4)
    if fp4:
        q = torch.randint(0, 256, (rows, 1, 32, 64), generator=g, device=dev, dtype=torch.int32).to(torch.uint8)
        qs = torch.full((rows, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device=dev)
    else:
        q = (torch.randn((rows, 1, 32, 128), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        qs = None
    w = (torch.randn((rows, 32), generator=g, device=dev) * 0.1).to(torch.bfloat16)
    return dict(q=q, qs=qs, kv=kv, w=w, ctx=ctx_rows.view(rows, 1).contiguous(), bt=bt, idx=idx,
                maxlen=int(lens_r.max()) + 64)


def graph_us(fn, iters=20, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(iters):
            fn()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(out)[len(out) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="32,64,96")
    ap.add_argument("--ctx", type=int, default=33000)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    fp4 = not a.fp8
    for reqs in [int(x) for x in a.reqs.split(",")]:
        d = make(reqs, a.ctx, fp4, a.seed, dev)
        rows = reqs * 4
        meta = dgu.get_paged_mqa_logits_metadata(d["ctx"], 64, nsm, indices=d["idx"])
        q_sf, w = dgu.mqa_logits_mx_inputs(d["q"] if fp4 else d["q"], d["qs"], d["w"])
        qv = d["q"].view(torch.int8) if fp4 else d["q"]

        def stock():
            return dgu.fp8_fp4_paged_mqa_logits((qv, q_sf), d["kv"], w, d["ctx"], d["bt"], meta, d["maxlen"],
                                                indices=d["idx"])

        gm = torch.zeros((rows, idx_gmax.gmax_cols(d["maxlen"])), dtype=torch.int16, device=dev)
        nanc = torch.zeros((rows,), dtype=torch.int32, device=dev)

        def fused():
            return idx_gmax.paged_logits_gmax((qv, q_sf), d["kv"], w, d["ctx"], d["bt"], meta, d["maxlen"],
                                              d["idx"], gm, nanc)

        ref = stock()
        out = fused()
        torch.cuda.synchronize()
        L = d["ctx"].view(-1)
        cols = torch.arange(ref.shape[1], device=dev)
        valid = cols[None, :] < L[:, None]
        rb, ob = ref.view(torch.int16), out.view(torch.int16)
        lmism = int(((rb != ob) & valid).sum())
        # gmax = the group's max logit as raw bf16 (NaN if the group holds one; -inf
        # outside [0, L)); compared as order-preserving keys, a NaN group as 0xFFFF
        k, nan = keys_of(out)
        k = torch.where(nan, torch.full_like(k, 0xFFFF), k)
        k = torch.where(valid, k, torch.zeros_like(k))
        ng = (ref.shape[1] + 7) // 8
        kp = torch.nn.functional.pad(k, (0, ng * 8 - k.shape[1]))
        gref = kp.view(rows, ng, 8).amax(-1)
        gvalid = (torch.arange(ng, device=dev)[None, :] * 8) < L[:, None]
        kg, ngn = keys_of(gm[:, :ng].contiguous())
        kg = torch.where(ngn, torch.full_like(kg, 0xFFFF), kg)
        gmism = int(((kg != gref) & gvalid).sum())
        nref = torch.zeros_like(nanc[:rows])
        nmism = int((nanc[:rows] != nref).sum())
        t_stock = graph_us(stock)
        nanc.zero_()
        t_fused = graph_us(fused)
        print(json.dumps(dict(reqs=reqs, rows=rows, fp4=fp4, logits_mism=lmism, gmax_mism=gmism, nan_mism=nmism,
                              stock_us=round(t_stock, 2), fused_us=round(t_fused, 2),
                              delta_us=round(t_fused - t_stock, 2))), flush=True)


if __name__ == "__main__":
    main()
