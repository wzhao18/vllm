# c256 --watermark 0.05 (decode) investigation: pre-registered expectations (written 2026-11-13 17:00, before launch)
Render integaf10frz-pareto-c256-wm05-{spx,xdr} = integaf10frz-pareto-c256 + decode `watermark: 0.05` (names otherwise only).
Mechanism under test: the watermark is applied to WAITING (async KV-fetch admission) and PREEMPTED (resume) requests only
(kv_cache_manager.py:575-583, 614, 651-653); running requests may grow into it. 5% of 4687 blocks = 234 blocks.
Hypothesis: keeping growth headroom prevents the first preemption at the 5-min burst, so no local recompute.
| Observable | Expect if hypothesis holds | Freeze e c256 (watermark 0) |
|---|---|---|
| decode "stock step (prompt rows (mixed prefill + decode))" lines | 0 (or a handful) | >= 2048 between 11:06:30 and 11:26:46 |
| "N with all of them" (zero-computed resumes) | 0 | 129 |
| decode KV peak during the burst | <= ~95% (admission stops at 95% free-block level) | 99.9% |
| client slow (>200 ms) engine intervals, min 6-25 | ~1-3% of wall time, single-step | 54-72% |
| decode request-steps/s during the burst | within ~5% of steady state at the same running | ~1,640-2,130 vs ~4,780 |
| TTFT p90 during the burst | rises (queuing at decode admission) | 3.5-15.5 s p50 range |
| full-hour tok/s/GPU / interactivity | ~45K / ~95-100 (nq4kmc c256 45.2K @ 97); capacity cost of the 5% reserve ~ -8 requests | 35.3K @ 34 |
If mixed steps still appear: the 2-block/step KDA state boundary demand exceeds the reserve, or a resume path bypasses the watermark; analyze with the logged counters.
