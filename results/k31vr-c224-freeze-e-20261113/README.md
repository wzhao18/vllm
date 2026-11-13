# K3.1 VR C224 — freeze e (Hecate, 2026-11-13)

Code: `k31vr-freeze-20261113e` (wzhao/k31-vr-hti-integ c62faa6933 = 0f558a4414 + decode memory fixes lseslab
e74a6a3730 + dcpdirect 8e83596b41). Config: `srt-configs/c224.yaml` (2P1D, 8 nodes: 2x prefill TP8+DCP8,
decode TP16+DCP16; batch-spx RoCE). Concurrency comes from the workflow env (`workflow.env.resolved`:
CONC=224, CONC_LIST=224, DURATION=3600); the config is otherwise identical to the C256 render it replaces
(only the job name differs).

C224 replaces C256 in this sweep: at C256 the trace's burst ~5 min into profiling fills decode KV to 97-99.9%
and decode throughput collapses for ~20-25 min (root cause under investigation). C224 peaks at 90.2% decode KV
during the burst and stays healthy (steady state ~40-47%).

| Window | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **Official (60.5 min profiling)** | **42,908** | **119** | 6.80 / 8.40 ms | 0.693 / 1.360 s |
| 0-55 min | 42,647 | 120 | 6.78 / 8.34 ms | 0.69 / 1.34 s |
| 30-45 min | 45,996 | 141 | 6.39 / 7.08 ms | 0.68 / 1.29 s |
| 45-55 min | 47,276 | 143 | 6.29 / 6.98 ms | 0.65 / 1.29 s |

- 33,313 requests OK / 13 errors (0.04%); profiling window 13:28:36..14:29:04; job 1052665.
- Validity: P/D health PASS (985,421 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries); container keepalive overlap NONE; decode GPU KV cache 82.07 GiB.
- Pre-registered projection 39.3K tok/s/GPU full hour (written during the run, before the result): +9%.
- vs previous sweep nq4kmc C256 (45,180 @ 97 over 0-55 min): 94% of the throughput at +24% interactivity and
  37% lower TTFT p90; vs the nq4kmc curve interpolated at interactivity 119: +7%.
- Accuracy: no gsm8k run at C224 on freeze e yet. Same 2P1D deployment on freeze d: gsm8k C160 0.974,
  C256 0.968 (real rejection, 1319 samples).

Artifacts: `agentx/` = job a1052665 output. Files > 5 MB omitted (OTel traces/logs, frontend log,
per-request profile_export.jsonl, large worker logs); they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1052665-integaf10frz-pareto-c224-spx-1113120146`.
