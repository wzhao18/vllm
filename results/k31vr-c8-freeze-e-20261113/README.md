# K3.1 VR C8 — freeze e (Hecate, 2026-11-13)

Code: `k31vr-freeze-20261113e` (wzhao/k31-vr-hti-integ c62faa6933). Config: `srt-configs/c8.yaml` (1P1D, 6 nodes:
prefill TP8+DCP8, decode TP16+DCP16; batch-spx RoCE; identical to the run's config.yaml). Concurrency from
`workflow.env.resolved`: CONC=8, CONC_LIST=8, DURATION=3600.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 59.9 min profiling)** | **2,960** | **553** | 1.31 / 1.81 ms | 0.185 / 0.307 s |
| Pre-registered projection | ~2.9K | 548-575 | — | — |
| Previous sweep nq4kmc C8 (0-55 min) | 2,929 | 548 | 1.31 / 1.82 ms | 0.21 / 0.39 s |

Same windows vs nq4kmc C8 (`winbench.py`, 24 GPUs): 0-55 min 2,962 @ 551 vs 2,929 @ 548; 30-45 min 3,941 @ 536 vs
3,780 @ 543; 45-55 min 2,663 @ 634 vs 2,769 @ 624.

- 1,475 requests OK / 2 errors; profiling window 17:45:00..18:44:52; job 1057685.
- Validity: P/D health PASS (43,021 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries, sparse core on); container keepalive overlap NONE; decode GPU KV cache 52,106,161 tokens.
- Accuracy: gsm8k C8 (real rejection, 1319 samples) strict 0.9689 / flexible 0.9682 on the tree-fix code (job
  a1045081, before freeze e's decode-memory fixes, which do not touch the math).

Artifacts: `agentx/` = job a1057685 output. Files > 5 MB omitted; they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1057685-integaf10frz-pareto-c8-spx-1113170913`.
