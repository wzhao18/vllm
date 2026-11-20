"""ops1 DSpark drafter-step ops (single GPU, cold L2 CUPTI):
DS.markov : rank-512 Markov head per position = BF16 GEMM [B*positions, 512] x [512, vocab_shard] (cuBLAS).
DS.attn   : SWA GQA over the last 2048 tokens, BF16 KV, head_dim 64, q_len k=3 per request:
            FlashInfer BatchPrefillWithPagedKVCacheWrapper (page 64, causal, distinct pages per request).
usage: draft_bench.py cases.json out.jsonl i/n
"""
import json
import sys
import traceback

import torch
import torch.nn.functional as F

from mbc import Out, env_info, log, shard, time_us

cases = shard([c for c in json.load(open(sys.argv[1]))
               if c["family"] == "drafter_verify" and c["shape"]["op"] in ("DS.markov", "DS.attn")], sys.argv[3])
out = Out(sys.argv[2])
dev = torch.device("cuda")
out.row(kind="env", **env_info())
WS = torch.empty(512 << 20, dtype=torch.uint8, device=dev)


def emit(c, t, m, notes):
    out.row(kind="case", case=c["id"], family=c["family"], ops=c["ops"], topologies=c["topo_ids"],
            ckpts=c["ckpts"], shape=c["shape"], measured_us=round(t, 3), method=m, t_sol_us=c["t_sol_us"],
            t_ach_us=c["t_ach_us"], notes=notes)


for c in cases:
    s = c["shape"]
    try:
        if s["op"] == "DS.markov":
            Mr = s["B"] * s["positions"]
            x = torch.randn(Mr, s["rank"], device=dev, dtype=torch.bfloat16)
            w = torch.randn(s["vocab_shard"], s["rank"], device=dev, dtype=torch.bfloat16) * 0.05
            t, m = time_us(lambda: F.linear(x, w))
            emit(c, t, m, f"cuBLAS F.linear M={Mr} N={s['vocab_shard']} K={s['rank']} (logit GEMM only; "
                          "argmax/softmax epilogue not timed)")
        else:
            import flashinfer
            B, kq, Hq, Hk, D, win = s["B"], s["k"], s["q_heads"], s["kv_heads"], s["head_dim"], s["window"]
            P = 64
            ppr = win // P
            kv = torch.randn(B * ppr, 2, P, Hk, D, device=dev, dtype=torch.bfloat16)
            q = torch.randn(B * kq, Hq, D, device=dev, dtype=torch.bfloat16)
            qo = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * kq
            ind = torch.arange(0, B * ppr, dtype=torch.int32, device=dev)
            indptr = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * ppr
            last = torch.full((B,), P, dtype=torch.int32, device=dev)
            wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(WS, "NHD")
            wr.plan(qo, indptr, ind, last, Hq, Hk, D, P, causal=True, q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16)
            t, m = time_us(lambda: wr.run(q, kv))
            note = f"flashinfer {t:.1f}"
            try:
                from vllm.vllm_flash_attn import flash_attn_varlen_func
                kc, vc = kv[:, 0].contiguous(), kv[:, 1].contiguous()
                bt = ind.view(B, ppr)
                seqk = torch.full((B,), win, dtype=torch.int32, device=dev)
                fa = lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=kq, cu_seqlens_q=qo,  # noqa: E731
                                                    max_seqlen_k=win, seqused_k=seqk, causal=True,
                                                    block_table=bt)
                tf, _ = time_us(fa)
                note += f"; vllm_flash_attn {tf:.1f}"
                t = min(t, tf)
            except Exception as e:  # noqa: BLE001
                note += "; vllm_flash_attn err " + repr(e)[:150]
            emit(c, t, m, f"min of FlashInfer paged prefill / vllm_flash_attn varlen (paged 64): {note}; "
                          f"{B} req x q{kq}, {win} KV tokens, {Hq}/{Hk} heads d{D}; "
                          f"KV {B * win * Hk * D * 4 / 1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out.row(kind="case", case=c["id"], family=c["family"], shape=s, error=repr(e)[:400])
    torch.cuda.empty_cache()
log("draft shard done")
