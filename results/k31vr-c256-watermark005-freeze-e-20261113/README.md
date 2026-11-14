# K3.1 VR C256 with decode `--watermark 0.05` — investigation (Hecate, 2026-11-13)

Code: freeze e `k31vr-freeze-20261113e` (c62faa6933). Config: the C256 2P1D render with exactly one change, decode
`watermark: 0.05` (agentx/config.yaml vs `srt-configs/c224.yaml`: only that line plus the job name; CONC=256 from
`workflow.env.resolved`). Purpose: test whether keeping 5% of decode KV blocks free for growth (the watermark applies to
WAITING/PREEMPTED admissions only) prevents the C256 burst stall (decode preemption -> local recompute of ~120K-token
prompts in eager mixed steps). Pre-registered expectations: `PROJECTION.md` (written before launch).

| Run | tok/s/GPU | Interactivity | ITL p50/p90 | TTFT p50/p90 |
|---|---|---|---|---|
| **This run, official (60.5 min)** | **45,253** | **104** | 7.96 / 9.64 ms | 1.46 / 2.92 s |
| This run, 0-55 min (clean, see validity) | 45,259 | 103 | 7.91 / 9.68 ms | 1.44 / 2.89 s |
| C256 freeze e, watermark 0 (official) | 35,252 | 34 | 8.17 / 29.23 ms | 1.32 / 19.68 s |
| nq4kmc C256 (previous sweep, 0-55 min) | 45,180 | 97 | 8.66 / 10.27 ms | 0.88 / 2.13 s |

Burst mechanism (decode log, the 5-min trace burst): decode KV peak 92.8% (vs 99.9%), 0 stock mixed prefill+decode
steps (vs >= 2,048), 0 zero-computed resumes (vs 129); decode gen tok/s rose to 12.5-12.7K at ~155 running (vs
collapse to 3.6K). Same windows vs nq4kmc: 30-45 min 48,718 @ 110 vs 48,923 @ 104; 45-55 min 50,723 @ 117 vs
50,635 @ 108. All projections met.

Validity: 36,575 OK / 12 errors; P/D health PASS (1,085,812 KV transfers, 0 failed, 0 handshake failures, 0 lease
expiries). Container keepalive overlap: YES, 22:46:22-22:46:48 (~26 s, 64 short steps killed by the guard) inside the
last ~3.5 min of profiling (21:49:26..22:49:55): the dispatcher's 9000 s claim expired before this long-startup run ended
and runner2 respawned keepalives. The 0-55 min window ends 22:44:26, before the overlap, and matches the official number.

Artifacts: `agentx/` = job a1060197 output (files > 5 MB omitted; they remain on Hecate under
`/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a1060197-integaf10frz-pareto-c256-wm05-spx-1113201815`).
