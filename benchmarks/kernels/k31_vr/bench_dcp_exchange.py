# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP16 decode exchange microbenchmark for Kimi K3.1 DSA (VR200).

torchrun, one process per GPU (16 = 4 nodes x 4 GPUs = DCP16). For each
AgentX decode bucket (B, q) with the AgentX per-request context vector it
times, per rank, inside CUDA graphs (N back-to-back calls per replay, median
over replays; logits L2-warm as written by the indexer, plus an L2-flushed
variant for the top-k):

top-k (indexer logits -> this rank's local slots + count):
  ours_1r    histogram threshold, 1 NVLink round (512-key window)
  ours_2r    histogram threshold, 2 rounds (coarse + refine)
  stock      DeepSelect local top-2048 + pack + NCCL all_gather + stable
             select + triton_filter_and_convert_dcp_index
O (partial O + LSE -> owner heads after W_UV, [T, 6, 128]):
  ours_wuv   W_UV bmm on all 96 heads + push/merge of 128-dim rows
  ours_lat   push/merge of 512-dim latent rows + W_UV bmm on 6 heads
  stock      direct DCP A2A LSE reduce (latent) + W_UV bmm on 6 heads
Q:
  stock_q        this rank's 6 heads (q_b + W_UK + fp8) + DirectDCPQGatherWorkspace
                 (NVLS multicast direct gather, the VR recipe's
                 VLLM_USE_DIRECT_DCP_Q_GATHER=1 path), padded to 128 heads; the
                 NCCL all_gather variant is also timed
  local_q        q_b GEMM (all 96 heads) + W_UK bmm + fp8 cast (no exchange)
value (the DSA DcpValue stage, bench_dcp_value.py; VLLM_K31_DSA_PARTIALS):
  value_prod     as served: run_value "bf16" up to 64 rows, MR !8's DcpValue above
  value_proj     W_UV bmm over the 96 gathered heads + DcpValue(preprojected)
                 through the "proj" DcpMailbox (_contig: a contiguous W_UV)

Writes one JSON line per (shape, path) with per-rank times to --out.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (
    STAT_NAMES,
    DcpExchange,
    local_len_of,
)

K = 2048
INTERLEAVE = 896
HEADS = 96


def ctx_vectors(path):
    with open(path) as f:
        return json.load(f)


def ctx_for(vecs, B, q):
    g = vecs["q8_C64"] if q == 8 else vecs["q4_C48_C160_C256"]
    if str(B) in g:
        return g[str(B)]
    out = []
    for b in sorted((int(k) for k in g), reverse=True):
        while len(out) + b <= B:
            out += g[str(b)]
        if len(out) == B:
            break
    return out[:B]


def graph_time(fn, n_inner, n_rep, pre=None):
    """Median per-call us over n_rep replays of a graph of n_inner calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            if pre:
                pre()
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            if pre:
                pre()
            fn()
    ts = []
    for _ in range(n_rep):
        dist.barrier()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / n_inner)
    del g
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--batches", default="1,8,16,32,64,96,128,192")
    ap.add_argument("--qs", default="4,8")
    ap.add_argument("--paths", default="topk,o,q")
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--flush", type=int, default=1)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    ex = DcpExchange(dist.group.WORLD, dev, cap=1536, cap1=256, interleave=INTERLEAVE)
    hpr = HEADS // W
    vecs = ctx_vectors(args.vecs)
    paths = set(args.paths.split(","))
    value = None
    if "value" in paths:  # bench_dcp_value.py, next to this script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bench_dcp_value

        rows = [int(b) * int(q) for q in args.qs.split(",")
                for b in args.batches.split(",")]
        value = bench_dcp_value.ValueBench(dist.group.WORLD, dev, W, rank, hpr,
                                           INTERLEAVE, bench_dcp_value.wide_cap(rows))
    flush_buf = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def flush():
        flush_buf.zero_()

    t_flush = graph_time(flush, 5, 5)

    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        dcp_topk_id_bits,
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    results = []

    def emit(shape, path, us, extra=None):
        t = torch.tensor([us], device=dev)
        allt = [torch.zeros_like(t) for _ in range(W)]
        dist.all_gather(allt, t)
        per = [float(x) for x in allt]
        rec = dict(shape=shape, path=path, per_rank_us=per,
                   median_us=statistics.median(per), max_us=max(per))
        if extra:
            rec.update(extra)
        results.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    max_model_len = 1 << 20
    for q in [int(x) for x in args.qs.split(",")]:
        for B in [int(x) for x in args.batches.split(",")]:
            T = B * q
            ctxs = torch.tensor(ctx_for(vecs, B, q), dtype=torch.int64, device=dev)
            G = (ctxs[:, None] - q + 1 + torch.arange(q, device=dev)[None, :]).reshape(-1)
            G32 = G.to(torch.int32)
            Lr = local_len_of(G, rank, W, INTERLEAVE).to(torch.int32)
            Lmax = int(local_len_of(torch.tensor([int(G.max())]), 0, W, INTERLEAVE))
            stride = (Lmax + 511) // 512 * 512
            shape = dict(B=B, q=q, T=T, W=W, mean_ctx=float(ctxs.float().mean()),
                         max_local=Lmax, local_sum=int(Lr[::q].sum()))
            gen = torch.Generator(device=dev)
            gen.manual_seed(1234 + rank)
            logits = torch.randn(T, stride, generator=gen, device=dev).to(torch.bfloat16)
            # paged layout for slot conversion: 64-token kernel blocks
            bs = 64
            nblk = (Lmax + bs - 1) // bs
            block_table = (torch.arange(B * nblk, dtype=torch.int32, device=dev)
                           .reshape(B, nblk))
            out = torch.empty(T, K, dtype=torch.int32, device=dev)
            cnt = torch.empty(T, dtype=torch.int32, device=dev)

            if paths & {"topk", "topk_ours"}:
                for mode, one in (("ours_1r", True), ("ours_2r", False)):
                    if one and T > ex.cap1:
                        continue
                    ex.err.zero_()
                    fn = lambda one=one: ex.topk_merge(  # noqa: E731
                        logits, Lr, G32, out, cnt, max_local_len=Lmax,
                        block_table=block_table, block_size=bs, next_n=q, one_round=one)
                    # path counters of one call (rows per path, this rank)
                    st = torch.zeros(8, dtype=torch.int32, device=dev)
                    ex.topk_merge(logits, Lr, G32, out, cnt, max_local_len=Lmax,
                                  block_table=block_table, block_size=bs, next_n=q,
                                  one_round=one, stats=st)
                    torch.cuda.synchronize()
                    ex.check()
                    us = graph_time(fn, args.inner, args.reps)
                    usf = (graph_time(fn, 5, args.reps, pre=flush) - t_flush) if args.flush else None
                    torch.cuda.synchronize()
                    ex.check()
                    emit(shape, "topk_" + mode, us,
                         dict(flushed_us=usf, err=int(ex.err.item()),
                              stats=dict(zip(STAT_NAMES, st.tolist())),
                              mean_local_count=float(cnt.float().mean())))
            if paths & {"topk", "topk_stock"}:
                loc = torch.empty(T, K, dtype=torch.int32, device=dev)
                packed = torch.empty(T, K, 2, dtype=torch.float32, device=dev)
                gathered = torch.empty(T, W * K, 2, dtype=torch.float32, device=dev)
                req_id = torch.arange(T, device=dev, dtype=torch.int32) // q
                id_bits = dcp_topk_id_bits(max_model_len)

                ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)
                lg32 = torch.empty(T, stride, dtype=torch.float32, device=dev)
                maxlen = int(Lr.max())

                def stock_local():
                    # default VR decode path (topk_backend auto): fp32 upcast +
                    # persistent radix top-k over each row's local keys
                    lg32.copy_(logits)
                    torch.ops._C.persistent_topk(lg32, Lr, loc, ws, K, maxlen)

                def stock_merge():
                    pack_dcp_topk_candidates_cutedsl(lg32, loc, packed, rank, W,
                                                     INTERLEAVE, None)
                    _ag(packed, gathered, W)
                    stable_topk_from_gathered_candidates_cutedsl(
                        gathered, K, out=loc, score_bits=16, id_bits=id_bits)

                def _ag(src, dst, W):
                    buf = torch.empty(W, *src.shape, dtype=src.dtype, device=dev)
                    dist.all_gather_into_tensor(buf, src)
                    dst.view(T, W, K, 2).copy_(buf.permute(1, 0, 2, 3))

                def stock_filter():
                    triton_filter_and_convert_dcp_index(
                        req_id, block_table, loc, W, rank, INTERLEAVE,
                        BLOCK_SIZE=bs, NUM_TOPK_TOKENS=K, return_valid_counts=True)

                def stock_all():
                    stock_local(); stock_merge(); stock_filter()

                try:
                    t_loc = graph_time(stock_local, args.inner, args.reps)
                    t_mrg = graph_time(stock_merge, args.inner, args.reps)
                    t_fil = graph_time(stock_filter, args.inner, args.reps)
                    t_all = graph_time(stock_all, args.inner, args.reps)
                    t_allf = (graph_time(stock_all, 5, args.reps, pre=flush) - t_flush) if args.flush else None
                    emit(shape, "topk_stock", t_all,
                         dict(local_topk_us=t_loc, merge_us=t_mrg, filter_us=t_fil, flushed_us=t_allf))
                except Exception as e:  # noqa: BLE001
                    if rank == 0:
                        print(f"stock topk failed: {e!r}", flush=True)
                del packed, gathered

            if "o" in paths:
                ex.err.zero_()
                o_lat = torch.randn(T, 128, 512, generator=gen, device=dev).to(torch.bfloat16)[:, :HEADS]
                lse = torch.randn(T, 128, generator=gen, device=dev)[:, :HEADS]
                w_uv = (torch.randn(HEADS, 512, 128, generator=gen, device=dev) * 0.05).to(torch.bfloat16)
                y = torch.empty(HEADS, T, 128, dtype=torch.bfloat16, device=dev)
                o_out = torch.empty(T, hpr, 128, dtype=torch.bfloat16, device=dev)
                lat_out = torch.empty(T, hpr, 512, dtype=torch.bfloat16, device=dev)
                w_own = w_uv[rank * hpr:(rank + 1) * hpr]

                def ours_wuv():
                    torch.bmm(o_lat.transpose(0, 1), w_uv, out=y)
                    ex.o_exchange(y.transpose(0, 1), lse, o_out, lse_base_e=False)

                def ours_wuv_x():
                    ex.o_exchange(y.transpose(0, 1), lse, o_out, lse_base_e=False)

                def wuv_only():
                    torch.bmm(o_lat.transpose(0, 1), w_uv, out=y)

                def ours_lat():
                    ex.o_exchange(o_lat, lse, lat_out, lse_base_e=False)
                    torch.bmm(lat_out.transpose(0, 1), w_own)

                emit(shape, "o_ours_wuv", graph_time(ours_wuv, args.inner, args.reps),
                     dict(exchange_us=graph_time(ours_wuv_x, args.inner, args.reps),
                          wuv_bmm_us=graph_time(wuv_only, args.inner, args.reps)))
                emit(shape, "o_ours_latent", graph_time(ours_lat, args.inner, args.reps))
                torch.cuda.synchronize()
                if int(ex.err.item()):
                    print(f"rank {rank} o err {int(ex.err.item())}", flush=True)
                try:
                    from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace

                    if not hasattr(main, "_a2a"):
                        main._a2a = DirectDCPA2AWorkspace(dist.group.WORLD, dev, 1536, hpr, 512)
                    a2a = main._a2a
                    o_c = o_lat.contiguous()
                    l_c = lse.contiguous()

                    def stock_o():
                        r = a2a.lse_reduce(o_c, l_c, False)
                        torch.bmm(r.transpose(0, 1), w_own)

                    emit(shape, "o_stock", graph_time(stock_o, args.inner, args.reps))
                except Exception as e:  # noqa: BLE001
                    if rank == 0:
                        print(f"stock o failed: {e!r}", flush=True)
                del o_lat, y

            if "q" in paths:
                # Stock VR decode Q path (VLLM_USE_DIRECT_DCP_Q_GATHER=1): this rank's
                # 6 heads of q_b + W_UK + fp8 cast, then the NVLS multicast direct
                # gather into [T, 96 (padded 128), 576]. Ours: every rank computes all
                # 96 heads from the replicated q_c (no exchange).
                qc = torch.randn(T, 1536, generator=gen, device=dev).to(torch.bfloat16)
                wqb = (torch.randn(1536, HEADS * 192, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
                wuk = (torch.randn(HEADS, 128, 512, generator=gen, device=dev) * 0.05).to(torch.bfloat16)
                wqb_own = wqb.view(1536, HEADS, 192)[:, rank * hpr:(rank + 1) * hpr].reshape(1536, hpr * 192).contiguous()
                wuk_own = wuk[rank * hpr:(rank + 1) * hpr].contiguous()
                qloc = torch.empty(T, hpr, 576, dtype=torch.float8_e4m3fn, device=dev)
                qall = torch.empty(W, T, hpr, 576, dtype=torch.float8_e4m3fn, device=dev)
                qabs = torch.empty(T, HEADS, 576, dtype=torch.bfloat16, device=dev)
                qabs_own = torch.empty(T, hpr, 576, dtype=torch.bfloat16, device=dev)
                if not hasattr(main, "_qg"):
                    from vllm.v1.attention.ops.dcp import DirectDCPQGatherWorkspace

                    main._qg = DirectDCPQGatherWorkspace(
                        dist.group.WORLD, dev, 1536, hpr, 576, dtype=torch.float8_e4m3fn,
                        padded_num_heads=128)
                qg = main._qg

                def own_q():
                    qb = (qc @ wqb_own).view(T, hpr, 192)
                    torch.bmm(qb[..., :128].transpose(0, 1), wuk_own,
                              out=qabs_own[..., :512].transpose(0, 1))
                    qabs_own[..., 512:] = qb[..., 128:]
                    qloc.copy_(qabs_own)

                def stock_q():
                    own_q()
                    qg.gather(qloc)

                def stock_q_gather_only():
                    qg.gather(qloc)

                def stock_q_nccl():
                    own_q()
                    dist.all_gather_into_tensor(qall.view(torch.uint8), qloc.view(torch.uint8))

                def local_q():
                    qb = (qc @ wqb).view(T, HEADS, 192)
                    torch.bmm(qb[..., :128].transpose(0, 1), wuk,
                              out=qabs[..., :512].transpose(0, 1))
                    qabs[..., 512:] = qb[..., 128:]
                    qabs.to(torch.float8_e4m3fn)

                emit(shape, "q_stock_direct_gather", graph_time(stock_q, args.inner, args.reps),
                     dict(gather_only_us=graph_time(stock_q_gather_only, args.inner, args.reps),
                          own_heads_compute_us=graph_time(own_q, args.inner, args.reps)))
                emit(shape, "q_stock_nccl_allgather", graph_time(stock_q_nccl, args.inner, args.reps))
                emit(shape, "q_local_rebuild", graph_time(local_q, args.inner, args.reps))
            if value is not None:
                # per-row valid counts: this rank's share of a balanced top-k
                valid = torch.minimum(Lr, (G.clamp(max=K) // W).clamp(min=1))
                timer = graph_time, emit, shape, (args.inner, args.reps)
                bench_dcp_value.run_value_paths(value, T, q, ctxs, valid, *timer)
            del logits
            torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
