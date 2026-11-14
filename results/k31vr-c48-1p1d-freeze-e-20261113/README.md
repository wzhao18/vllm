# K3.1 VR C48 (1P1D) — freeze e (Hecate, 2026-11-13)

Code: `k31vr-freeze-20261113e` (wzhao/k31-vr-hti-integ c62faa6933). Config: `srt-configs/c48-1p1d.yaml` (1P1D, 6 nodes:
prefill TP8+DCP8, decode TP16+DCP16; batch-spx RoCE; identical to the run's config.yaml). Concurrency from
`workflow.env.resolved`: CONC=48, CONC_LIST=48, DURATION=3600.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 60.1 min profiling)** | **15,593** | **305** | 2.54 / 3.28 ms | 0.297 / 0.791 s |
| Pre-registered projection | ~15.5K | 298-313 | — | — |
| Previous sweep nq4kmc C48 (0-55 min) | 15,492 | 298 | 2.55 / 3.36 ms | 0.34 / 0.97 s |

Same windows vs nq4kmc C48 (`winbench.py`, 24 GPUs): 0-55 min 15,540 @ 305 vs 15,492 @ 298; 30-45 min 17,106 @ 310
vs 16,857 @ 312; 45-55 min 17,792 @ 296 vs 18,054 @ 282.

- 8,316 requests OK / 1 error; profiling window 17:07:35..18:07:41; job 1055047.
- Validity: P/D health PASS (242,165 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries, sparse core on); container keepalive overlap NONE; decode GPU KV cache 51,228,987 tokens.
- Accuracy: gsm8k C48 (real rejection, 1319 samples) strict 0.9727 / flexible 0.9719 on the tree-fix code (job
  a1050535, before freeze e's decode-memory fixes, which do not touch the math).

Artifacts: `agentx/` = job a1055047 output. Files > 5 MB omitted; they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1055047-integaf10frz-pareto-c48-1p1d-spx-1113161739`.
