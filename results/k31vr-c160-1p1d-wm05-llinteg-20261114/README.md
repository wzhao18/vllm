# K3.1 VR C160 on 1P1D + decode watermark 0.05 (Hecate, 2026-11-14)

Code: low-latency branch `wzhao/k31-vr-ll-integ` dea4ebd06f (freeze e + CuTe compile cache + F1 wide-DSA query fork +
F3a KDA-verify rows). Config: `srt-configs/c160-1p1d-wm05.yaml` = the official C160 render with ONLY (a) the prefill pool
2 instances -> 1 (6 nodes / 24 GPUs: 1x prefill TP8+DCP8, decode TP16+DCP16, k=3) and (b) decode `watermark: 0.05`
(user 2026-11-14: config changes approved to push the pareto; watermark 0.05 on all configs onward). CONC=160, DURATION=3600.

| Run | GPUs | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|---|
| **This run (1P1D + wm 0.05, official 60.5 min profiling)** | 24 | **39,782** | **167.1** | 5.23 / 5.99 ms | 1.41 / 4.24 s |
| Official C160 (2P1D, freeze d, a1047927) | 32 | 31,174 | 164 | 5.25 / 6.09 ms | 0.78 / 1.63 s |

- **+27.6% tok/s/GPU at equal-or-better interactivity.** Pre-registered projection (`hti/lowlat/C160_1P1D_PROJECTION.md`):
  ~40.5K @ 162-166 (accept >= 37.4K @ >= 158) -> **accepted, 98% of projection**. Mechanism: the second prefill instance
  was mostly idle at C160; one instance carries the fresh-prefill demand, so the same served tok/s needs 24 instead of 32 GPUs.
  Cost: TTFT p90 1.6 s -> 4.2 s (one prefill queue), as projected (1.5-3x).
- 23,666 requests OK / 8 errors; profiling window 10:26:23..11:26:51; job 1086151 (batch-xdr).
- Validity: P/D health PASS (697,561 successful NIXL KV transfers, 0 handshake failures, 0 prefill lease expiries; two
  Mooncake store events with failed keys (1 and 64 keys), recompute path); container keepalive overlap NONE.
- Accuracy: gsm8k (real rejection, 1319 samples) strict 0.9659 / flexible 0.9651 (`gsm8k/`, job 1089039, xdr).

Artifacts: `agentx/` = job a1086151 output, `gsm8k/` = job a1089039 eval output. Files > 5 MB omitted; they remain on
Hecate under `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/`.
