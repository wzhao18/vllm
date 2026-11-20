# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN: decode indexer logits, stock DeepGEMM varlen (one key stream per verify
row) vs dsa.fp4_index_logits (one per request), CUDA-graph timed, L2 flushed per
replay batch, at the deployed per-rank shape (DCP4 local lengths of ~135K contexts).

  python3 bench_dattn_idx_fp4.py --reqs 32,64,96 --ctx 135168
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tests/models/kimi_k3"))
from test_specialized_decode_fp4_index_logits_gpu import _case, _check  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.fp4_index_logits import (  # noqa: E402,E501
    Fp4IndexLogits,
)


def graph_time(fn, iters=20, reps=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(reps):
        torch.empty(512 << 20, dtype=torch.uint8, device="cuda").zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / iters)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="32,64,96")
    ap.add_argument("--ctx", type=int, default=135168)
    ap.add_argument("--q", type=int, default=4)
    a = ap.parse_args()
    for reqs in [int(x) for x in a.reqs.split(",")]:
        q, rows = a.q, reqs * a.q
        args, ref, lens = _case(reqs, q, a.ctx, 0)
        op = Fp4IndexLogits(rows, q, "cuda")
        got = op(**args)
        _check(got, ref, lens)
        kv = kv_cache_as_quant_view(args["kv_cache"], 128, True)
        ctx_rows = args["row_lens"]
        idx = torch.arange(rows, dtype=torch.int32, device="cuda")
        sched = get_paged_mqa_logits_metadata(ctx_rows, 64, get_num_sms(), idx)
        qv = args["query"].view(torch.int8).view(rows, 1, 32, 64)
        qs, w = mqa_logits_mx_inputs(qv, args["q_scale"].view(rows, 1, 32), args["head_w"])
        t_dg = graph_time(lambda: fp8_fp4_paged_mqa_logits(
            (qv, qs), kv, w, ctx_rows, args["block_table"], sched, args["max_len"],
            indices=idx))
        t_new = graph_time(lambda: op(**args))
        mb = float(lens.view(reqs, q).amax(1).sum()) * 68 / 1e6
        print(f"R{rows} keys {mb:6.1f} MB: stock DG {t_dg:6.2f} us | fp4_index_logits "
              f"{t_new:6.2f} us ({t_dg / t_new:4.2f}x, {mb / t_new:5.2f} TB/s) check ok",
              flush=True)


if __name__ == "__main__":
    main()
