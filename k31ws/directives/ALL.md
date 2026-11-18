# Directives for all workstreams (read at start and before every milestone)

- 2026-11-18 user: be smart about nodes. The 8-node EP32 hold queues for hours (one request is always pending). Prepare 8-node work in advance and do everything possible on 1-4 GPUs first: kernel microbench at the deployed per-GPU shapes, 1-node emulations (one DP rank of D3-32: DCP4 group + 28 experts/GPU; one node of DEP16).
- Target for every kernel family: at most 1.1x the perf model's achievable at the deployed shape. "Better than before" is not done. If the model is provably wrong, prove it with measured hardware evidence and record it as a model fix.
- Use Vera Rubin (sm_107) features: consult /Users/weizha/k31ws-hwdb/INDEX.md and your brief hwdb/briefs/<ID>.md. Ask the HW researcher by writing /Users/weizha/k31ws-hwdb/requests/<ID>-<n>.md.
