# k31ws workstream status files

Each workstream keeps exactly one file, `k31ws/status/<ID>.json`. Owners update it after every result and at least every ~45 min of work. Write the whole file with the Write tool; no heredocs.

`k31ws/tools/render.py` turns all status files plus `k31ws/pool.json` into the tracker page `k31ws/tracker.html`. The tracker is the single source of truth for the campaign.

```json
{
  "id": "D-TOPK",
  "title": "Decode: DSA exact top-k + DCP merge",
  "side": "decode | prefill | both | infra",
  "owner": "workflow agent label",
  "state": "analysis | implementing | measuring | blocked | landed | done",
  "updated": "2026-11-18T13:05-08:00",
  "shape": "D3-32 r24: 384 verify rows per DP rank, L=135K, k=3 (per GPU)",
  "baseline_ms": 2.30,
  "current_ms": 2.30,
  "achievable_ms": 0.22,
  "target_ms": 0.24,
  "step_ms": 30.5,
  "basis": "MEASURED in-step (source path / run id) | PROJECTED | MODEL",
  "gap_analysis": "Paragraph: where the time goes now, measured evidence (ncu / profile numbers with sources), why it is above the model.",
  "inefficiencies": ["one line per root cause, measured or labelled hypothesis"],
  "rubin_features": ["VR200 features this workstream uses or plans to use (names only; no internal numbers)"],
  "plan": ["ordered steps to reach target"],
  "work": [{"t": "13:05", "what": "what was done", "result": "measured outcome with source"}],
  "next": "the very next action",
  "blockers": "empty or what blocks progress (and what was done about it)",
  "branch": "wzhao/k31ws-<id> @ <sha>",
  "evidence": ["paths / run dirs"]
}
```

Meaning of the fields:
- **ms values:** per decode step for decode workstreams, and per prefill forward for prefill workstreams, at `shape`.
- **step_ms:** the whole-step time at the same shape, which gives the workstream's share.
- **baseline_ms:** frozen at the campaign start. Gap closed = (baseline - current) / (baseline - achievable).
- **achievable_ms:** the perf model's achievable time for the same ops at the same shape (perf-model/out/optables.json or model_rows.py).
- **target_ms:** at most 1.10 x achievable, unless the gap_analysis proves with measured hardware evidence that the model is wrong. That is then a model fix, recorded in the work log.

Public-page rule: never put NVIDIA-internal hardware numbers (from the private hwdb) in status files. Name features only.
