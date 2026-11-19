# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN sparse MLA decode microbench (1 GPU): FlashInfer trtllm-gen sparse MLA as
the SP decode calls it (dsa.attn_fi.FlashInferCore.trtllm) at the D3-32 per-GPU
shapes: R = 4 x requests rows, 128 (96 padded) FP8 query heads, this rank's DCP4 KV
shard (~ 2048 / 4 local keys per row, a compacted prefix of the merge's rows), FP8
per-tensor KV [rows, 576] in 64-row pages, LSE returned.

``--share``: the fraction of a row's local keys shared with the request's other
verify rows (consecutive positions' top-k sets overlap); 0 = independent rows.
Prints us per call (cold L2: a 1 GiB write between iterations) and the unique KV
bytes the call must read (the dedup floor) vs the per-row gathered bytes.
``--ncu``: one launch per shape (wrap the script in ncu)."""

import argparse
import json

import torch


def timed(fn, iters=30, flush=None):
    times = []
    for _ in range(iters + 3):
        if flush is not None:
            flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times = sorted(times[3:])
    return times[len(times) // 2]


def case(rows, local, share, ctx_rows, gen, q=4, heads=128, width=1024):
    """Slots [rows, width] (compacted prefix of ``local`` valid entries per row) over
    a cache of ``ctx_rows`` per request."""
    dev = "cuda"
    reqs = rows // q
    base = torch.arange(reqs, device=dev).repeat_interleave(q) * ctx_rows
    slots = torch.full((rows, width), -1, dtype=torch.int32, device=dev)
    shared_n = int(local * share)
    for r in range(reqs):
        common = torch.randperm(ctx_rows, generator=gen, device=dev)[: local * q]
        for j in range(q):
            own = common[shared_n + j * (local - shared_n) : shared_n + (j + 1) * (local - shared_n)]  # noqa: E501
            idx = torch.cat([common[:shared_n], own])[:local]
            slots[r * q + j, :local] = (idx + base[r * q + j]).int()
    valid = torch.full((rows,), local, dtype=torch.int32, device=dev)
    uniq = torch.unique(slots[slots >= 0]).numel()
    return slots, valid, uniq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="128,256,384,512")
    p.add_argument("--local", type=int, default=512)
    p.add_argument("--share", default="0,0.8")
    p.add_argument("--ctx", type=int, default=135000 // 4)
    p.add_argument("--ncu", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _kv_counters

    dev, heads = "cuda", 128
    gen = torch.Generator(device=dev).manual_seed(0)
    flush = None if a.ncu else torch.empty(1 << 28, device=dev, dtype=torch.float32)
    rows_list = [int(r) for r in a.rows.split(",")]
    reqs_max = max(rows_list) // 4
    ctx_rows = a.ctx
    kv = torch.randn(reqs_max * ctx_rows, 576, device=dev).to(torch.float8_e4m3fn)
    pages = kv.view(-1, 64, 576)
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    res = {}
    for rows in rows_list:
        query = torch.randn(rows, heads, 576, device=dev).to(torch.float8_e4m3fn)
        latent = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, heads, dtype=torch.float32, device=dev)
        for share in (float(s) for s in a.share.split(",")):
            slots, valid, uniq = case(rows, a.local, share, ctx_rows, gen)
            extra = _kv_counters(rows, heads, query.device)
            extra["lse"] = lse

            def run():
                trtllm_batch_decode_with_kv_cache_mla(
                    query=query.unsqueeze(1),
                    kv_cache=pages.unsqueeze(1),
                    workspace_buffer=workspace,
                    qk_nope_head_dim=128,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    block_tables=slots.unsqueeze(1),
                    seq_lens=valid,
                    max_seq_len=slots.shape[1],
                    out=latent.unsqueeze(1),
                    bmm1_scale=0.07,
                    bmm2_scale=1.0,
                    sparse_mla_top_k=slots.shape[1],
                    return_lse=True,
                    **extra,
                )

            if a.ncu:
                run()
                torch.cuda.synchronize()
                continue
            us = timed(run, flush=flush)
            key = f"R{rows}_share{share}"
            res[key] = dict(
                us=round(us, 2),
                gathered_MB=round(rows * a.local * 576 / 1e6, 1),
                unique_MB=round(uniq * 576 / 1e6, 1),
                mma_GF_128h=round(rows * 128 * a.local * 1088 * 2 / 1e9, 1),
            )
            print(key, json.dumps(res[key]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
