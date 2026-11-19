# INFRA top priority (2026-11-18 14:15, user)
Produce NVFP4 in-step baselines NOW: merge EN (wzhao/k31ws-en 6f164db270) into wzhao/k31ws-int, then run the 1-node D3-32 DP-rank slice at r8/r16/r24 and the DEP16 1-node slice at F16K/F8K on the NVFP4 view, profiled, with per-family breakdown vs nvfp4_modelopt model rows. Write k31ws/measured/<ts>-nvfp4-{decode,prefill}.json and post per-family baselines in findings/INFRA.md so every owner re-bases. Retire every MXFP4 number from your status (base-mx runs are not a baseline).

## 2026-11-18 13:50 PST coordinator: order of the NVFP4 baselines (user asked for an ETA)
1. Prefill DEP16 NVFP4 slice (EP4, 56 experts/GPU, F8K/F16K): runnable now (EP4 NVFP4 inference knobs 2048/8192 exist). Run it on int-cd9229c-1 as soon as its unit tests pass; post per-family baseline.
2. Decode D3-32 NVFP4 slice: blocked: the NVFP4 MegaMoE kind has no EP4 decode-bucket and no EP32 knob records in the vendored megamoe_rubin. Merge D-MOE vendored FI a368c5f66 (wzhao/k31ws-d-moe, once its v1 unit tests pass) into int, or use the FI tree via WS_PYTHONPATH, and run the decode slice at r8/r16/r24 on NVFP4 the moment the MoE path runs. Coordinate with D-MOE.

## 2026-11-18 15:50 PST coordinator
- The NVFP4 DECODE baseline is still missing and is your top priority: reconcile D-MOE (a9b8c85bd3) with EN in wzhao/k31ws-int (megamoe vendored-source manifest tests), then run step_d on the NVFP4 view at r8/r16/r24, profiled, with the per-family table, and post it like the prefill baseline. Decode owners re-base on it.

## 2026-11-18 18:25 PST coordinator
- Multi-node runs must never cold-compile: before any 8n/4n run on a new overlay, run a 1-node step slice (same configs) on that overlay to warm the shared CuTe/Triton/extension caches, and verify zero JIT compiles in its log. Add this as an automatic pre-step in step_multi.sh.
- Integration backlog: D-ATTN, D-FUSE, D-MOE, P-GEMM, P-KDA, P-MOE are landed-ready. Merge them in order of measured gain, one trial per merge, then re-measure the NVFP4 slices and post the new reference numbers.

## 2026-11-18 18:55 PST coordinator
- Check D-KDA hypothesis: does the decode step slice exercise production deferred-commit mode (pending > 0, RecoverSSM fold, state write) or run with pending = 0? If not representative, fix the slice so KDA is measured as in production, and re-post the decode reference.

## 2026-11-18 19:40 PST coordinator
- Release DSpark drafter config (e4e6e626: config.json, config.py, README, no weights) copied from Hecate to Rhea at /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/models/dspark-release. Build dspark-release-L24 (aux layer ids remapped to the L24 target like dspark-preview-L24) and make it the default drafter of step_d/step_multi; re-post the NVFP4 decode reference with the release drafter (the ~4 ms preview tail goes away).

## 2026-11-18 20:05 PST coordinator: TOP PRIORITY
- Build dspark-release-L24 (and the 93-layer release view for step_multi), make it the default drafter everywhere (step_d, step_p, step_multi), and re-post the NVFP4 decode and prefill references on int with the release drafter (profiled, per-family). Owners re-base on those. Mirror the release config to Hecate.

## 2026-11-18 20:45 PST coordinator
- D-KDA CONFIRMED from code that the decode step slice does NOT exercise the deferred commit (dummy/capture batches bind recoverssm_request_slots = None), so KDA (and the step) is understated vs production. Fix the slice so KDA runs as in production (pending > 0, fold + state write), together with the release-drafter rebuild, and re-post the NVFP4 decode reference. Real 8-node runs show the same gap? Check them too.
