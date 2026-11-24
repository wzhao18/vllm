# K3.1 Vera Rubin (VR200) pareto configs (srt-slurm)

srt-slurm renders used on Hecate (batch-spx, RoCE) for the K3.1 VR AgentX pareto and its gsm8k
accuracy gates, for code tag `k31vr-freeze-20261112b` (= af10ccc2c4 + MLA KV-capture fix f0e6a6a8ab).

- `<point>.yaml`: AgentX benchmark (DURATION 3600, synthetic acceptance as deployed).
- `gsm8k/<point>.yaml`: eval-only twin of the same deployment (real block rejection,
  FlashInfer autotune off) for the gsm8k accuracy gate.

| point | topology |
|---|---|
| c1, c8, c48-1p1d, c64 | 1P1D, 6 nodes: prefill TP8+DCP8, decode TP16+DCP16 |
| c160, c224 | 2P1D, 8 nodes: 2x prefill TP8+DCP8, decode TP16+DCP16 |
| c512 | 8 nodes: prefill DP4xTP4 (DCP4), decode DP4xTP4 SP |

The vLLM code is the Lustre tree `hti/p2/integ-af10ccc2c4` on PYTHONPATH (patched in place with the
fix; `mla_attention.py` md5 cd4070d5c87bfce0a885f01f2b28dd6a). The render names and the `vllm:`
identity string still say `integaf10frz` / af10ccc2c4 because the renders predate the fix; paths
are Hecate Lustre paths.

## 2026-11-13: C224 replaces C256

`c224.yaml` replaces `c256.yaml` (AgentX). The config is the C256 render with only the job name changed; the
concurrency is set by the workflow env (CONC=224, CONC_LIST=224; see `results/k31vr-c224-freeze-e-20261113/`).
C256 was dropped because the trace's burst fills decode KV and collapses decode throughput for ~20-25 min.
`gsm8k/c256.yaml` stays as the gsm8k twin of the same 2P1D deployment (the eval concurrency does not change
the server). Code for freeze e: tag `k31vr-freeze-20261113e` (c62faa6933), patched into the same Lustre tree.
Results: `results/k31vr-c224-freeze-e-20261113/`, `results/k31vr-c160-freeze-d-20261113/`.

## 2026-11-24: P/D with DEP8 + dense DWDP prefill, TP4 x DP4 DSA-DP decode, Inferact Dynamo router

`pd-dep8dwdp-tp4dp4.yaml` (AgentX) and `gsm8k/pd-dep8dwdp-tp4dp4.yaml` (its gsm8k twin) are the K3.1 VR200 P/D deployment
our runs converged on, expressed for this branch and its open agentx PRs. 6 worker nodes (24 GPUs) plus infra.

| role | layout | settings |
|---|---|---|
| prefill, 2 nodes | 1 x DEP8 (TP1, DP8, EP8), `--dwdp dense` | EPLB R=8 (window / step 500, `symm_mem`, async); `megamoe_quant` `mxfp4_mxfp4`; MNBT 8192; mns 24; GMU 0.97; DPLB `--prefill-context-cost-c0 900000 --prefill-context-cost-ref-len 100000` (`--long-prefill-token-threshold 0`); `--prefill-hrrn` (max wait 30 s); MultiConnector NIXL + Mooncake store with `load_same_step`; `cudagraph_mode` NONE; `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048` |
| decode, 4 nodes | TP4 x DP4, EP16; DSA-DP (automatic at TP > 1, DCP 1); specialized decode | EPLB R=16 (window 1000, step 10000); mns 32; FULL_AND_PIECEWISE capture 6..192 step 6; GMU 0.93; watermark 0.05; `--max-async-kv-loads-per-step 2`; `flashinfer_sparse_mla_kernel` trtllm |
| both | | DSpark k5 (probabilistic draft, block rejection); `trtllm_gvr2_nondet` top-k; FP8 KV, MXFP4 indexer KV, FP32 KDA state, BF16 dense, MXFP4 MoE; prefix-match-unit 128; cumem allocator |
| frontend | Inferact Dynamo | `--router-mode kv --load-aware --router-session-affinity-ttl-secs 1 --router-prefill-cost 6.8e-2,2.9e-8,0,1.1e-3 --router-decode-balance`; `DYN_ROUTER_GLOBAL_PREFIX_TTL_SECS=3600`, `DYN_ROUTER_TEMPERATURE=0`; backend admission limits 4096 / 4096 / 8192 |

Code:
- vLLM: `wzhao/k31-agentx-dev` f9f1b28658 + agentx #25-#42. The layout needs #41 (NIXL from the TP1 prefiller's 16384-token
  blocks into the decoder's 4096-token blocks) and #42 (full-model `--dwdp dense` fits only when sharded at construction);
  the flags need #26 (`--max-async-kv-loads-per-step`), #31 (DPLB cost knobs), #32 (`megamoe_quant mxfp4_mxfp4`), #35
  (`--prefill-hrrn`), #39 (`trtllm_gvr2_nondet`) and #40 (`load_same_step`). The configs point at a Lustre tree `k31vr/trees/k31-agentx-dev-pd` holding that
  stack with its compiled extensions, in place of `/opt/k3/ov-p` / `/opt/k3/ov-d` (as the frozen configs do); build it
  like the other trees (sm_107a; #33 pins the DeepSelect CUTLASS v4.8.1 build, #34 needs `nvidia-cudnn-frontend` >= 1.33).
- Dynamo: `Inferact/dynamo` 9ac6228943 (PR #410, `juntian/k31-rubin-router` = 3dd0744 K3.1 release + active-prefill
  sums, the Mooncake global-prefix store model, `--router-prefill-cost` with the cached-token coefficient,
  `--router-decode-balance`). `dynamo.source` pins it: srtctl builds that commit with maturin in the container
  (`install: true`); the compute nodes need read access to `Inferact/dynamo`. Without it, build the two wheels
  (`ai_dynamo`, `ai_dynamo_runtime`, aarch64) from that commit elsewhere, `pip install` them in the image and set
  `dynamo.install: false`. Every PYTHONPATH here leaves out `/opt/k3/dynamo`, which would shadow the pinned build with
  the image's 1.6.1+k31.

Notes:
- Paths, NIC lists and the store's `device_name` are Hecate's, as in the other configs. Our VR200 nodes used 8 NICs per node
  for the store (`mlx5_0,1,2,5,6,7,8,11`) and 2 x 260 GB store-only segments per decode node (`DLOAN_GB 520`,
  `DLOAN_SPLIT 2` here).
- The draft is the release (SWA) DSpark checkpoint, mounted at `/raid/common-models/kimi-k3.1-pro-dspark`.
- Concurrency comes from the workflow env; c380 keeps prefill at capacity (higher only queues in front of it).
- Status: not launched on Hecate yet. On our VR200 nodes, the integration tree (this branch + agentx #25-#39) scores
  GSM8K strict 0.9810 aggregated (TP4 x DP4); the full-model P/D run of this exact layout on this branch is pending
  agentx #42 (dense DWDP at construction: without it, full-model prefill runs out of memory while it is built).
