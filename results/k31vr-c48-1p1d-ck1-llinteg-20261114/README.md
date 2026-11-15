# K3.1 VR C48 1P1D checkpoint CK-1 on the low-latency branch (Hecate, 2026-11-14)

Code: `wzhao/k31-vr-ll-integ` dea4ebd06f (freeze e + CuTe compile cache + F1 wide-DSA query fork + F3a KDA-verify rows).
Config: the official C48 1P1D render with only the code tree and the CuTe cache dir changed (no watermark; run before the
watermark-0.05 rule). CONC=48, DURATION=3600, 6 nodes / 24 GPUs.

| Run | tok/s/GPU | Interactivity (1000/p90 ITL) | ITL p50 / p90 | TTFT p50 / p90 |
|---|---|---|---|---|
| **CK-1 (this run, 60.1 min profiling)** | **15,580** | **309.6** | 2.57 / 3.23 ms | 0.31 / 0.77 s |
| Official C48 1P1D (freeze e, a1055047) | 15,593 | 305 | 2.55 / 3.28 ms | — / 0.79 s |

- **Missed its pre-registered projection** (315-322 interactivity, 15.9-16.4K tok/s/GPU): +1.5% interactivity, flat y.
  Root cause (measured from per-batch-size step times, `hti/lowlat/gap2/ck1_rootcause/`):
  - ~65% of the miss is a projection error: the M96 bench delta was applied to every step, but p90 ITL is set by a mix of
    batch sizes (M96 is only 21% of the p90 band's step time) and 74% of all C48 decode step time is at batch <= 8, where
    F1/F3a do nothing; shape by shape the bench caps the gain at ~312.6.
  - ~32% is a +1.0-1.6% run-level slowdown at batch 4-8 where the code is unchanged. The CuTe compile cache is ruled out
    by a same-node step A/B (base / cache-loaded / cache-empty all within 8 us at M32-M96, `hti/lowlat/results/cc-step-*`);
    the remaining suspects are node/fabric (xdr vs spx) and per-rank MoE SM split.
  - F1 and F3a delivered roughly their bench deltas where they apply (M96: -0.53 ms code-only vs -0.55 ms bench).
- 8,312 requests OK / 1 error; profiling window 02:36:39..03:36:44; job 1068230 (batch-xdr).
- Validity: P/D health PASS (242,045 successful NIXL KV transfers, 0 handshake failures); keepalive overlap NONE.
- Accuracy: gsm8k (real rejection, 1319 samples) strict 0.9682 / flexible 0.9682 (`gsm8k/`, job 1068231, xdr).

Artifacts: `agentx/` = job a1068230 output, `gsm8k/` = job a1068231 eval output. Files > 5 MB omitted.
