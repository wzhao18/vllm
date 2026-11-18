# Directives for all workstreams (read at start and before every milestone)

- 2026-11-18 user: be smart about nodes. The 8-node EP32 hold queues for hours (one request is always pending). Prepare 8-node work in advance and do everything possible on 1-4 GPUs first: kernel microbench at the deployed per-GPU shapes, 1-node emulations (one DP rank of D3-32: DCP4 group + 28 experts/GPU; one node of DEP16).
- Target for every kernel family: at most 1.1x the perf model's achievable at the deployed shape. "Better than before" is not done. If the model is provably wrong, prove it with measured hardware evidence and record it as a model fix.
- Use Vera Rubin (sm_107) features: consult /Users/weizha/k31ws-hwdb/INDEX.md and your brief hwdb/briefs/<ID>.md. Ask the HW researcher by writing /Users/weizha/k31ws-hwdb/requests/<ID>-<n>.md.
- 2026-11-18 15:40 MEASURED DECODE BASELINE (replaces the seeded EP16-transferred split). D3-32 r24 per-family gap, from the Rhea 1-node profile on wzhao/k31-sp-gemm projected to 93 layers. The projected step is 29.78 ms vs 30.48 measured on 8 nodes; r8 and r16 are also within 2-4%. Source: perf-model/microbench/results/rhea1n_d3x32/README.md (+ parsed/). Decode owners: set baseline_ms, current_ms and achievable_ms from it (r24 exposed / achievable, ms):

  | family | exposed | achievable | owner |
  |---|---|---|---|
  | BF16 GEMMs | 8.25 | 4.74 | D-GEMM |
  | top-k + merge | 2.59 | 0.23 | D-TOPK |
  | MegaMoE (substituted from FI EP32 standalone) | 7.69 | 5.35 | D-MOE |
  | q path / sparse MLA / DCP value | 2.68 | 1.00 | D-ATTN |
  | indexer logits | 0.94 | 0.43 | D-ATTN |
  | KDA core (KdaWS) | 3.40 | 1.81 | D-KDA |
  | small ops | 1.28 | 0.15 | D-FUSE |
  | SP AG/RS | 2.06 | 1.28 | D-FUSE |
  | drafter (substituted) | 0.89 | 0.28 | D-FUSE |

  Also use the r8 and r16 rows. Harness: perf-model/microbench/rhea1n (run_gap.sh); it is a 24-layer view, because sdec hard-codes the 896-expert MoE front.
- OVERLAY SAFETY (incident 2026-11-18): overlays are hard-linked trees (cp -al), so their files share inodes with base-4fbaa15, the k31gap overlays and the extracted image tree. NEVER modify a file inside an overlay in place: no '>' or '>>' redirection onto an existing file, no open(..., 'w'/'a') on existing files, no 'sed -i' on them. Change code only through mkov.sh (rsync replaces files with new inodes). Build outputs go to NQ_BUILD / TORCH_EXTENSIONS_DIR, never into the overlay.
