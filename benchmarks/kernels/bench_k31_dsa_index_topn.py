# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP local top-N chain of a K3.1 DSA layer on one GPU, at a DCP16 rank's
shapes (q8 rows per request, ``--local`` keys per request on this rank, N = 192),
over an MXFP4 indexer cache:

  stock  DeepGEMM MXFP4 paged logits -> DeepSelect top-N -> #63 pack (B' rows)
  fused  IndexerScoreFp4 -> IndexerLocalTopk (B' rows)

One CUDA graph of ``--layers`` chains back to back (PDL as launched in the step);
us per layer (median of replays) and the CUPTI kernel durations (mean per layer).
Checks first that both give the same local top-N sets and bit-equal logits.

    python3 benchmarks/kernels/bench_k31_dsa_index_topn.py --requests 1,2,8
"""

import argparse
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def graph_us(fn, layers, reps):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(layers):
                fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / layers)
    ts.sort()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    per = defaultdict(float)
    for e in prof.events():
        if e.device_type.name == "CUDA":
            per[e.name[:40] + "~" + e.name[-50:]] += e.device_time / (3 * layers)
    return ts[len(ts) // 2], dict(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="1,2,8")
    ap.add_argument("--local", type=int, default=8250)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    from tests.models.kimi_k3.test_specialized_decode_dsa_fp4_index import (
        deepgemm_fp4_logits,
        fused_fp4_inputs,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_pack

    q, n = 8, 192
    for reqs in map(int, args.requests.split(",")):
        contexts = [args.local - 1] * reqs  # row j sees context + 1 + j keys
        x = fused_fp4_inputs(contexts, q, reqs)
        rows = reqs * q
        decode = SimpleNamespace(
            seq_lens=x.ends.view(rows, 1), block_table=x.table.repeat_interleave(q, 0)
        )
        indexer = SimpleNamespace(
            k_cache=SimpleNamespace(kv_cache=x.index),
            topk_tokens=2048,
            max_model_len=(args.local + 64) * 16,
            dcp_world_size=16,
        )
        fused = FusedIndexer(
            indexer, [(rows, q)], torch.device("cuda"), max(64, rows), local_topk=n,
            packed=True,
        )  # fmt: skip
        pitch = fused.pitch
        out = torch.empty((rows, n + n // 32), dtype=torch.int32, device="cuda")

        from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata

        lens = x.ends.view(rows, 1).contiguous()
        idx = torch.arange(rows, dtype=torch.int32, device="cuda")
        sched = get_paged_mqa_logits_metadata(lens, 64, get_num_sms(), idx)

        from vllm.model_executor.layers.sparse_attn_indexer import (
            kv_cache_as_quant_view,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
            fp4_query_views,
        )
        from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

        qp, qs = fp4_query_views(x.q_rows, rows, rows)
        dg_q = (qp.view(torch.int8).unsqueeze(1), qs.unsqueeze(1))
        dg_kv = kv_cache_as_quant_view(x.index, 128, True)
        dg_table = x.table.repeat_interleave(q, 0).contiguous()

        def stock():
            logits = fp8_fp4_paged_mqa_logits(
                dg_q, dg_kv, x.weights, lens, dg_table, sched,
                max_model_len=pitch, indices=idx,
            )  # fmt: skip
            dcp_topk_pack(logits, x.ends, n, out=out, max_local_len=fused.max_local_len)

        def new():
            fused.run_dcp(x.q_rows, x.weights, decode, rows, q)

        score_only = FusedIndexer(
            indexer, [(rows, q)], torch.device("cuda"), max(64, rows), local_topk=n,
        )  # fmt: skip

        def score():
            score_only.run_dcp(x.q_rows, x.weights, decode, rows, q)

        def deepgemm():
            fp8_fp4_paged_mqa_logits(
                dg_q, dg_kv, x.weights, lens, dg_table, sched,
                max_model_len=pitch, indices=idx,
            )  # fmt: skip

        stock()
        packed, logits, _ = fused.run_dcp(x.q_rows, x.weights, decode, rows, q)
        ref = deepgemm_fp4_logits(x, q, pitch)
        torch.cuda.synchronize()
        bad_logits = bad_sets = 0
        for t in range(rows):
            e = int(x.ends[t])
            got, want = logits[t, :e].view(torch.int16), ref[t, :e].view(torch.int16)
            bad_logits += int((got != want).sum())
            a = (out[t, :n].long() & 0xFFFFFFFF) >> 16
            b = (packed[t, :n].long() & 0xFFFFFFFF) >> 16
            if not torch.equal(a.sort().values, b.sort().values):
                bad_sets += 1
                print(f"  row {t}: stock {a.sort().values[:6].tolist()} fused "
                      f"{b.sort().values[:6].tolist()} counts {int(fused.valid[t])}")
        bad = (bad_logits, bad_sets)
        print(f"CHECK requests {reqs} rows {rows}: mismatches {bad}", flush=True)
        for name, fn in (("stock", stock), ("fused", new), ("score", score), ("deepgemm", deepgemm)):
            us, per = graph_us(fn, args.layers, args.reps)
            kern = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(per.items()))
            print(f"REC {name} requests {reqs} rows {rows}: {us:.2f} us/layer "
                  f"({us * 24 / 1000:.3f} ms/step) kernels [{kern}]", flush=True)


if __name__ == "__main__":
    main()
