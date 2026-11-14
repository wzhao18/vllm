# K3.1 VR C1 — freeze e (Hecate, 2026-11-13)

Code: `k31vr-freeze-20261113e` (wzhao/k31-vr-hti-integ c62faa6933). Config: `srt-configs/c1.yaml` (1P1D, 6 nodes:
prefill TP8+DCP8, decode TP16+DCP16; batch-spx RoCE; identical to the run's config.yaml). Concurrency from
`workflow.env.resolved`: CONC=1, CONC_LIST=1, DURATION=3600.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 59.9 min profiling)** | **1,347** | **805** | 1.21 / 1.24 ms | 0.235 / 0.328 s |
| Pre-registered projection | ~1.35K | 807-847 | — | — |
| Previous sweep nq4kmc C1 (0-55 min) | 1,348 | 807 | 1.21 / 1.24 ms | 0.27 / 0.38 s |

Same windows vs nq4kmc C1 (`winbench.py`, 24 GPUs): 0-55 min 1,357 @ 804 vs 1,348 @ 807; 30-45 min 1,572 @ 809 vs
1,546 @ 812; 45-55 min 1,904 @ 805 vs 1,897 @ 806.

- 412 requests OK / 2 errors; profiling window 17:51:51..18:51:45; job 1057686.
- Validity: P/D health PASS (11,651 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries, sparse core on); container keepalive overlap NONE; decode GPU KV cache 52,106,161 tokens.
- Accuracy: gsm8k C1 (real rejection, 1319 samples) strict 0.9651 / flexible 0.9644 (job a1050538, freeze d
  0f558a4414, before freeze e's decode-memory fixes, which do not touch the math).

Artifacts: `agentx/` = job a1057686 output. Files > 5 MB omitted; they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1057686-integaf10frz-pareto-c1-spx-1113171737`.
