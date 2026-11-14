# K3.1 VR C512 — final freeze sweep point (Hecate, 2026-11-13)

Code: wzhao/k31-vr-hti-integ 0f558a4414 (freeze d: accuracy fixes; freeze e's later decode-memory fixes target the
TP16 2P1D decode, not this DP4 deployment). Config: `srt-configs/c512.yaml` (8 nodes: prefill DP4xTP4 DCP4, decode
DP4xTP4 SP; batch-spx RoCE; identical to the run's config.yaml). Concurrency from `workflow.env.resolved`:
CONC=512, CONC_LIST=512, DURATION=3600. This is the C512 point on the final pareto (`results/k31vr-pareto-20261113/`);
it supersedes `results/k31vr-c512-integfix-20261112/` (an earlier freeze, 76,401 @ 86).

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 60.5 min profiling)** | **76,544** | **87** | 10.75 / 11.53 ms | 3.151 / 5.012 s |
| Previous sweep nq4kmc C512 | ~61.7K | 66 | — | — |

- 65,002 requests OK / 25 errors; profiling window 04:29:46..05:30:15; job 1050536 (a rerun that excluded hecate0289
  after a frontend stall invalidated the first attempt, job 1045937).
- Validity: P/D health PASS (282,968 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries); container keepalive overlap NONE; decode GPU KV cache 18,335,100 tokens per DP engine (x4).
- Accuracy: gsm8k C512 (real rejection, 1319 samples) strict 0.9682 / flexible 0.9674 (`gsm8k/`, job a1050538).

Artifacts: `agentx/` = job a1050536 output, `gsm8k/` = job a1050538 eval output. Files > 5 MB omitted; they remain on
Hecate under `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/`.
