# K3.1 VR C64 — freeze e (Hecate, 2026-11-13)

Code: `k31vr-freeze-20261113e` (wzhao/k31-vr-hti-integ c62faa6933). Config: `srt-configs/c64.yaml` (1P1D, 6 nodes:
prefill TP8+DCP8, decode TP16+DCP16; batch-spx RoCE; identical to the run's config.yaml). Concurrency from
`workflow.env.resolved`: CONC=64, CONC_LIST=64, DURATION=3600.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 60.2 min profiling)** | **21,002** | **250** | 3.34 / 4.01 ms | 0.316 / 0.855 s |
| Pre-registered projection | ~21.0K | 233-245 | — | — |
| Previous sweep nq4kmc C64 (official) | ~20.5K | 233 | — | — |

Same windows vs nq4kmc C64 (`winbench.py`, 24 GPUs): 0-55 min 21,276 @ 248 vs 20,980 @ 233; 30-45 min 25,274 @ 259
vs 25,027 @ 239; 45-55 min 25,073 @ 234 vs 24,874 @ 222.

- 12,044 requests OK / 5 errors; profiling window 16:07:53..17:08:02; job 1055046.
- Validity: P/D health PASS (347,615 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries, sparse core on); container keepalive overlap NONE; decode GPU KV cache 50,351,812 tokens.
- Accuracy: gsm8k C64 on freeze e is queued (added to this directory when it lands).

Artifacts: `agentx/` = job a1055046 output. Files > 5 MB omitted; they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1055046-integaf10frz-c64-spx-1113150633`.
