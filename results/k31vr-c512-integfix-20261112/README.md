# K3.1 VR C512 — fixed freeze (Hecate, 2026-11-12)

Code: `k31vr-freeze-20261112b` (af10ccc2c4 + MLA KV-capture fix f0e6a6a8ab). Config: `srt-configs/c512.yaml`
(8 nodes: prefill DP4xTP4 DCP4, decode DP4xTP4 SP; batch-spx RoCE).

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| Deployed reference | 61,580 | 66.5 | — | — |
| af10 freeze (before fix) | 74,235 | 84.5 | — | — / 7.43 s |
| **This run (fixed freeze)** | **76,401** | **86.4** | 10.81 / 11.58 ms | 3.16 / 5.19 s |

- AgentX DURATION 3600, CONC 512, 60.5 min profiling window: 64,890 requests OK, 22 aiperf no-content errors.
- Validity: 0 compiles after ready (32/32 JIT_WARMUP_VERIFIED), 0 failed KV transfers, run_in_alloc rc=0.
- Accuracy: gsm8k (real rejection, 1319 samples) strict 0.9704 / flexible 0.9697 (`gsm8k/`).

Artifacts: `agentx/` = job a1042171 output, `gsm8k/` = job a1041737 eval output. Files > 5 MB omitted
(OTel traces/logs, frontend log, per-request profile_export.jsonl, prefill worker logs); they remain on
Hecate under `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/`.
