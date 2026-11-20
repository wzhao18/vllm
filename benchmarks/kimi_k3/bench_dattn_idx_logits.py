# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN decode indexer logits microbench (MXFP4 keys, DeepGEMM paged MQA logits).

Production (sdec decode) calls the SM100 varlen form: one batch entry per verify row
(next_n = 1), so every key block is read once per row (4x per request at k = 3).
This bench times that form against the request-batched form (next_n = rows per
request: one key stream per request) at the deployed per-rank shape and checks the
two give the same logits.

  python3 bench_dattn_idx_logits.py --reqs 96 --next 4 --ctx 33000
"""

import argparse

import torch

from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)

UNIT, ROW, HEADS, DIM = 64, 68, 32, 128


def graph_time(fn, iters=20, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    best = []
    for _ in range(reps):
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best.append(a.elapsed_time(b) * 1e3 / iters)
    best.sort()
    return best[len(best) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="32,64,96")
    ap.add_argument("--next", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=33000)
    ap.add_argument("--forms", default="varlen,next")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    for reqs in [int(x) for x in a.reqs.split(",")]:
        n = a.next
        rows = reqs * n
        # per row local context (causal verify rows): ctx - n + 1 + j
        lens = torch.tensor(
            [a.ctx - n + 1 + j + (r * 37) % 64 for r in range(reqs) for j in range(n)],
            dtype=torch.int32,
            device=dev,
        )
        max_len = int(lens.max())
        max_blocks = -(-max_len // UNIT)
        nb = reqs * max_blocks + 1
        perm = torch.randperm(nb, device=dev).int()
        bt_req = perm[: reqs * max_blocks].view(reqs, max_blocks).contiguous()
        # bytes < 128: UE8M0 scales <= 1 wherever the scale bytes sit (finite logits)
        cache = torch.randint(0, 128, (nb, UNIT, ROW), dtype=torch.uint8, device=dev)
        kv = kv_cache_as_quant_view(cache, DIM, True)
        q = torch.randint(-128, 127, (reqs, n, HEADS, DIM // 2), dtype=torch.int8, device=dev)
        qs = torch.full((reqs, n, HEADS), 0x7F7F7F7F, dtype=torch.int32, device=dev)
        w = torch.randn(rows, HEADS, device=dev, dtype=torch.bfloat16)
        max_model_len = max_blocks * UNIT
        nsm = get_num_sms()
        out = {}
        for form in a.forms.split(","):
            try:
                if form == "varlen":
                    ctx = lens.view(rows, 1).contiguous()
                    bt = bt_req.repeat_interleave(n, 0).contiguous()
                    idx = torch.arange(rows, dtype=torch.int32, device=dev)
                    sched = get_paged_mqa_logits_metadata(ctx, UNIT, nsm, idx)
                    qv, qsc = q.view(rows, 1, HEADS, DIM // 2), qs.view(rows, 1, HEADS)
                    qsc, ww = mqa_logits_mx_inputs(qv, qsc, w)
                    fn = lambda: fp8_fp4_paged_mqa_logits(  # noqa: E731
                        (qv, qsc), kv, ww, ctx, bt, sched, max_model_len, indices=idx)
                elif form == "next":
                    ctx = lens.view(reqs, n)[:, -1].contiguous()
                    sched = get_paged_mqa_logits_metadata(ctx, UNIT, nsm)
                    qsc, ww = mqa_logits_mx_inputs(q, qs, w)
                    fn = lambda: fp8_fp4_paged_mqa_logits(  # noqa: E731
                        (q, qsc), kv, ww, ctx, bt_req, sched, max_model_len)
                elif form == "next2d":
                    ctx = lens.view(reqs, n).contiguous()
                    sched = get_paged_mqa_logits_metadata(ctx, UNIT, nsm)
                    qsc, ww = mqa_logits_mx_inputs(q, qs, w)
                    fn = lambda: fp8_fp4_paged_mqa_logits(  # noqa: E731
                        (q, qsc), kv, ww, ctx, bt_req, sched, max_model_len)
                lg = fn()
                torch.cuda.synchronize()
                out[form] = lg
                t = graph_time(fn)
                gb = reqs * a.ctx * ROW / 1e9
                print(f"R{rows} reqs {reqs} {form:7s} {t:7.2f} us  keys/req {gb * 1e3 / t:6.2f} TB/s"
                      f"  (x{rows if form == 'varlen' else reqs} streams)", flush=True)
            except Exception as e:  # report and go on
                print(f"R{rows} {form} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
        if "varlen" in out:
            ref = out["varlen"].float()
            col = torch.arange(ref.shape[1], device=dev)
            valid = col[None, :] < lens[:, None]
            for k, v in out.items():
                if k == "varlen":
                    continue
                v = v.float().view(rows, -1)[:, : ref.shape[1]]
                d = (v - ref).abs()[valid]
                print(f"R{rows} {k} vs varlen: max|d| {d.max().item():.3g} "
                      f"bitwise {bool((d == 0).all())} finite {bool(torch.isfinite(ref[valid]).all())}",
                      flush=True)


if __name__ == "__main__":
    main()
