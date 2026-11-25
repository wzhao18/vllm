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
- Dynamo: branch `dynamo-k31-vr` of this repo at bdec0a54ce (same tree as `Inferact/dynamo` 9ac6228943, PR #410:
  3dd0744 K3.1 release + active-prefill sums, the Mooncake global-prefix store model, `--router-prefill-cost` with the
  cached-token coefficient, `--router-decode-balance`). `dynamo.source` pins it: srtctl builds that commit with maturin
  in the container (`install: true`). Without access from the compute nodes, build the two wheels (`ai_dynamo`,
  `ai_dynamo_runtime`, aarch64) from that commit elsewhere, `pip install` them in the image and set
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

## 2026-11-24: the same P/D deployment on the NVIDIA NVFP4 checkpoint

`pd-dep8dwdp-tp4dp4-nvfp4.yaml` and `gsm8k/pd-dep8dwdp-tp4dp4-nvfp4.yaml` are `pd-dep8dwdp-tp4dp4.yaml` and its gsm8k twin
on `nvidia/Kimi-K3.1-Pro-NVFP4` (ModelOpt mixed: NVFP4 routed experts, MXFP8 KDA and shared-expert projections, BF16
MLA / DSA, router, layer-0 MLP and LM head). Only these differ:

| | BF16 checkpoint configs | NVFP4 checkpoint configs |
|---|---|---|
| model | `moonshotai/Kimi-K3` | `nvidia/Kimi-K3.1-Pro-NVFP4` |
| `--quantization-config` | `{"moe":{"activation":"mxfp8"}}` on both roles | none (the checkpoint's NVFP4 W4A4 experts) |
| prefill `additional-config` | `decode_backend auto`, `megamoe_quant mxfp4_mxfp4` | `decode_backend auto` (NVFP4 kernel; NVFP4 prefill uses the per-route combine, `megamoe_prefill_combine` `dedup` selects the old one) |
| decode `additional-config` | `decode_backend latency_specialized` | `+ megamoe_compact_scales false` (compact NVFP4 scales stay on prefill only) |
| store `cache_prefix` | `k31pd-dep8dwdp-...` | `k31pd-nvfp4-dep8dwdp-...` |

Decode differs from the BF16 configs in two more settings, both measured on the full model:

| decode setting | value | why |
|---|---|---|
| `sparse-indexer-topk-backend` (and `kernel-config`) | `balanced` | **Known perf issue, do not use `trtllm_gvr2_nondet` on decode.** GVR V2's self-sampling `main` kernel (the route every 48-row x 64k+ decode batch takes, also on SM107) degenerates to one CTA scanning the whole row when a row has no usable threshold: rows that are entirely NaN / constant (seen in real traffic on long requests), or with fewer than k positive logits. Real decode logits (16 cards, 42 rows, median 81k-195k, max 730k, k 2048): GVR 13-25 us/call on clean cards but 107-750 us on cards holding such a row, i.e. up to 18 ms per step for the whole lockstepped decode group; `balanced` 21-29 us/call on every card (0.69 ms/step worst card); `deep_select` also degrades on NaN rows (up to 587 us/call). Prefill keeps `trtllm_gvr2_nondet`. |
| `max-num-seqs` | 28 | Decode step vs per-engine batch is flat up to ~28 (18.4-19.8 ms) and jumps at 30-32 (22.5-28+ ms): capping at 28 keeps TPS p10 above 170 while the decode tier still has headroom (median running 17-25 per engine at c370). |

Load: AgentX concurrency 370 (one P DEP8 instance + one TP4 x DP4 decode instance, 24 GPUs).

Node placement on our VR200 nodes: keep `vr-r41-cn04` as the frontend / infra node only (etcd, Mooncake master, Dynamo
frontend, AgentX client). Its GPU0 runs BF16 GEMMs ~9% slower than its peers, and every decode or prefill engine is a
lockstepped EP group, so one slow GPU sets the step time of the whole group.

Code: this branch with agentx #25-#42, #44 to #47 merged (the refreshed-kernel prefill MegaMoE buckets and the NVFP4
per-route prefill combine, compact NVFP4 expert scales #44, full-model `--dwdp dense` for the checkpoint's MXFP8 linears
#42, the long-prompt NaN fix #47).

Measured on our VR200 nodes (11-24, full model, P DEP8 on 2 nodes + D TP4 x DP4 on 4 nodes, AgentX c370 3600 s,
DSpark k5 synthetic AL 3.81):

| | this config | same, D `trtllm_gvr2_nondet` + `max-num-seqs` 32 at c380 |
|---|---|---|
| requests/s | 15.14 | 14.91 |
| (input + output) tok/s/GPU | 90.3k | 88.2k |
| TPS p10 / p50 | 178 / 204 | 131 / 178 |
| ITL p50 / p90 | 4.91 / 5.60 ms | 5.61 / 7.63 ms |
| TTFT p50 / p90 / p99 | 2.5 / 6.7 / 18.3 s | 2.3 / 6.1 / 15.7 s |
| NIXL failed transfers | 0 | 0 |

GSM8K (gsm8k twin, block rejection, full model): strict 0.9788 / flexible 0.9780 (±0.004; the GVR decode config scored 0.9795 / 0.9788).

Fixed (#47): the NaN rows came from prefill. The Mamba state pre-copy ran before the Mooncake same-step load of the
prefix hit had landed, so the prompt was computed on stale KDA state and NaN state reached the decoder. With #47, same
config: 15.74 req/s, 87.0k tok/s/GPU, TPS p10 / p50 176 / 186, TTFT p50 / p90 2.2 / 5.8 s (first 2383 s), GSM8K
strict 0.9780 / flexible 0.9773. `balanced` stays: its cost is flat on every row.
