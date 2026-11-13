# K3.1 VR C160 — freeze d (Hecate, 2026-11-13)

Code: wzhao/k31-vr-hti-integ 0f558a4414 (freeze d: accuracy fixes, before the freeze e decode-memory fixes).
Config: `srt-configs/c160.yaml` (2P1D, 8 nodes: 2x prefill TP8+DCP8, decode TP16+DCP16). The AgentX run used
the batch-xdr (IB) twin of that render; it differs only in job name and NIC lists. Concurrency from
`workflow.env.resolved`: CONC=160, CONC_LIST=160, DURATION=3600.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **This run (official, 60.4 min profiling)** | **31,174** | **164** | 5.25 / 6.09 ms | 0.777 / 1.626 s |
| Previous sweep nq4kmc C160 | ~31.0K | 157 | — | — |

- 24,288 requests OK / 7 errors; profiling window 02:26:51..03:27:17; job 1047927 (batch-xdr).
- Validity: P/D health PASS (714,775 successful KV transfers, 0 failed, 0 handshake failures, 0 prefill lease
  expiries); container keepalive overlap NONE.
- Accuracy: gsm8k (real rejection, 1319 samples) strict 0.9742 / flexible 0.9735 (`gsm8k/`, job 1050538, spx).

Artifacts: `agentx/` = job a1047927 output, `gsm8k/` = job a1050538 eval output. Files > 5 MB omitted; they
remain on Hecate under `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/`.
