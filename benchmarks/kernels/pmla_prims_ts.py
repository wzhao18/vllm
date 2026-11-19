# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer-main prims_ts sparse MLA on sm_107 at the prefill shape (comparator).

Run with PYTHONPATH=<flashinfer main checkout> so `flashinfer` resolves to the
checkout (SM107 support, FlashInfer #8186/#8322). One query row per token, 96 or
128 heads, top-k 2048 rows of a 576 B FP8 cache (random pages), BF16 output.
Checks 8 tokens against an FP32 torch reference.
usage: PYTHONPATH=... python pmla_prims_ts.py --tokens 16384 [--json out]
"""

import argparse
import json
import math
import statistics

import torch

H, L, TOPK = 96, 512, 2048


def make(F, ctx, dev, seed=17):
    g = torch.Generator(device=dev).manual_seed(seed)
    page = 64
    pages = math.ceil((ctx + F) / page) * 3
    kv = (torch.randn(pages * page, 576, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    idx = torch.randint(0, pages * page, (F, TOPK), device=dev, generator=g, dtype=torch.int32)
    q = (torch.randn(F, 128, 576, device=dev, generator=g) * 0.15).to(torch.float8_e4m3fn)
    return kv, idx, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import flashinfer
    from flashinfer.attention.prims_ts import (
        BatchSparseMLADecodePagedTSWrapper,
        get_prims_ts_sparse_mla_decode_workspace_size,
    )

    dev = torch.device("cuda")
    F = a.tokens
    res = {"tokens": F, "flashinfer": getattr(flashinfer, "__file__", "?")}
    layers = [make(F, 72900, dev, 17 + i) for i in range(3)]
    seq = torch.full((F,), TOPK, dtype=torch.int32, device=dev)
    scale = 0.0722
    for heads in (96, 128):
        try:
            rows = min(a.rows, F)
            kw = dict(max_topk=TOPK, max_q_len=1, packed_query=True,
                      q_dtype=torch.float8_e4m3fn, qk_rope_head_dim=64,
                      kv_cache_format="native", return_lse=False)
            size = get_prims_ts_sparse_mla_decode_workspace_size(dev, rows, heads, **kw)
            plan = BatchSparseMLADecodePagedTSWrapper(
                torch.empty(size, dtype=torch.uint8, device=dev))
            plan.plan(dev, rows, heads, **kw)
            qo = torch.arange(rows + 1, dtype=torch.int32, device=dev)
            po = torch.empty(F, heads, L, device=dev, dtype=torch.bfloat16)
            qhs = [q[:, :heads].contiguous() for _, _, q in layers]

            def run(i):
                kv, idx, _ = layers[i % 3]
                for s0 in range(0, F, rows):
                    s1 = min(F, s0 + rows)
                    plan.run(qhs[i % 3][s0:s1], kv, idx[s0:s1], seq[s0:s1], qo_indptr=qo,
                             softmax_scale=scale, output_scale=1.0, out=po[s0:s1],
                             lse=None, validate=False)

            run(0)
            torch.cuda.synchronize()
            kv, idx, q = layers[0]
            errs = []
            for t in range(0, F, F // 8):
                k = kv[idx[t].long()].float()
                s = (q[t, :heads].float() @ k.T) * scale
                ref = torch.softmax(s, -1) @ k[:, :L]
                errs.append(((po[t].float() - ref).abs().max() / ref.abs().max()).item())
            times = []
            for i in range(10):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                run(i)
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1e3)
            res[f"h{heads}"] = dict(us=round(statistics.median(times[1:]), 1),
                                    max_rel_err=round(max(errs), 5))
        except Exception as e:  # noqa: BLE001
            res[f"h{heads}"] = dict(error=repr(e)[:500])
        print(heads, res[f"h{heads}"], flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
