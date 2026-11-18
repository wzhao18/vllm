# Workload and KV-capacity constants for the K3.1 SOL model

Machine-readable: `workload.json`. It is built by `workload_src/build_workload.py` from `workload_src/derived/*`. The scripts that make the derived files are next to it: `analyze_agentx.py`, `hist_moments.py`, `prefill_loggers.py`, `prefill_steps.py`, `st_headers.py` and `weights_breakdown.py`.

The sources are listed below. Everything is read-only.
- **Requests:** aiperf `profile_export.jsonl(.gz)` from `results/k31vr-frozen-sweep-20261114/runs/*/agentx/logs/agentic/conc_*/aiperf_artifacts/`.
  - 16 runs, C1 to C256. C512 kept only the summary file.
  - Population: `benchmark_phase == "profiling"`, not cancelled. That gives 270,887 requests.
  - Dataset: `semianalysis_cc_traces_weka_062126`, 3600 s, closed loop.
- **Engine logs:** the prefill and decode `*_w*.out(.gz)` files in the same runs.
  - Two KV-layout lines come from the release-drafter run `results/k31vr-release-draft-sweep-20261117/runs/C1-a1118776`.
- **Weights:** the headers of all 96 safetensors shards of `kimi-k3.1-pro-release-d9e99b07` (Hecate `/lustre/fsw/coreai_comparch_inferencex/weizha/models/`), 497,340 tensors. The DSpark header and config are in `.../k31vr/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626/`.
- **Code:** the `vllm/` tree of this worktree, with file and line given at each use.

Pooled numbers are dominated by the C ≥ 96 runs. The sessions differ by concurrency: C1 runs one long session with ISL 285K. From C48 upward the per-run means sit within ±10% of the pooled values (table in §6).

## 1. Requests (pooled)

| quantity | mean | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| ISL (prompt tokens) | 144,975 | 22,494 | 48,780 | 99,706 | 185,528 | 311,156 | 474,269 | 744,730 | 950,780 |
| cached prefix (server `prompt_cache_read`) | 139,518 | 0 | 43,904 | 94,080 | 181,120 | 307,584 | 469,248 | 742,400 | 928,640 |
| fresh prefill = ISL − cached | **5,456** | 439 | 867 | 1,967 | 3,353 | 8,654 | 17,763 | 74,421 | 950,780 |
| OSL | **1,048** | 111 | 196 | 444 | 1,048 | 2,423 | 4,021 | 9,566 | 59,903 |
| decode context, weighted by OSL (ISL + OSL/2) | **189,265** | 35,157 | 77,836 | 147,636 | 243,618 | 389,462 | 577,096 | 762,152 | |

- **Prefix hit:** 96.24% of prompt tokens are hits (token-weighted). 12.2% of requests have zero hits, and those cold requests carry **47%** of all fresh tokens (`hist_cached_prefix_fresh_token_weighted`, bin 0).
- **Cache-read granularity:** 128 tokens (the gcd of all cache reads). The code's prefix-hash unit is 64 tokens (`gqa_draft.py` `_DRAFT_BLOCK_ALIGN`).
- **Fresh-token mass by request fresh size:**

  | fresh size per request | < 4K | 4-16K | 16-64K | ≥ 64K |
  |---|---|---|---|---|
  | share of fresh tokens | 24% | 21% | 24% | 31% |

  81% of requests have fresh < 4K. The prefill workload is a mix of many short appends and a few cold, long prompts.
- **Prefix context seen by prefill:** the fresh-token-weighted mean cached prefix is **72.9K**, and the mean end context is 156.7K.
  - Indexer (dense) pairs per fresh token: **114,780**, from E[f·c + f(f+1)/2] / E[f].
  - Sparse attention keys per fresh token with top-k 2048: **~2,003**.
- **Decode context:** E[Σ ctx over output tokens] / E[OSL] = 189,265 tokens. The decode-time-weighted mean is 188,378 tokens. Use **~189K as the mean resident context of a decoding request**, not ISL (145K), because long-OSL requests also have long ISL.
- **Source mix:**

  | source | share of requests | ISL | OSL | fresh | hit rate |
  |---|---|---|---|---|---|
  | main | 40.8% | 191K | 1,368 | 4.7K | 97.5% |
  | subagent | 44.8% | 61K | 724 | 5.1K | 91.6% |
  | flat | 14.4% | 276K | 1,144 | 8.5K | 96.9% |

- **Structural growth per follow-up turn** (same conversation, ISL − previous ISL): median 1,633 and mean 3,074. The measured fresh mean of 5.46K is about 2-3x that. The excess comes from cold requests, evictions and partial-block tails.

## 2. Think time and the closed loop

- **Per-request idle per session slot:** C/λ − mean latency (Little's law per run) = **16.2 s** (request-weighted mean over the C ≥ 16 runs, range 15.2-19.0 s). This matches the expected 14-17 s.
- **Rates:** a session slot issues one request every (idle + latency). At C ≥ 96, only **30-40% of C** is in flight. Examples:

  | run | requests in flight |
  |---|---|
  | C128 | 36.7 |
  | C256 | 102.5 |

- The same-conversation gap (next start − previous end) has a median of 2.7 s and a mean of 28 s. It includes parents waiting on their subagents, so it is not the think time.

## 3. Prefill as scheduled

- **max_num_batched_tokens:**

  | instance | max_num_batched_tokens |
  |---|---|
  | prefill TP8+DCP8 (all 1P/2P runs) | 16,384 |
  | C512 prefill (DP4 × TP4/DCP4) | 32,768 |
  | decode | 8,192 |

- **C512 step counters** (`[context.py:894] DCP ShardQ prefill ... steps, tokens`, the only runs that log them): a mean of **15.2-15.9K tokens/step** and **2.1 steps/s** per DP4 engine, about 150M tokens per engine per run.
- **Prefix hits on the prefill engines:** local HBM hits fall as concurrency rises and Mooncake covers the rest. Steady state = last 75% of the 10 s logger windows; full list in `prefill_engines_steady_state`.

  | run | local HBM hit | Mooncake hit (% of the remainder) |
  |---|---|---|
  | C8 | 96% | 0% |
  | C64 | 81% | 82% |
  | C160 1P1D | 43-47% | 92% |
  | C256 2P1D | 33-36% | 94% |
  | C512 | 18-19% | 94% |

- **Prefill engine prompt rate** (computed tokens): the mean is 17-37K tok/s per TP8 engine. Bursts reach 105-149K tok/s. Running is 0.4-2.7 and waiting is 0.2-16.

## 4. KV and state bytes (exact, whole model, before sharding)

### DSA KV per token per layer (×24 layers)

| part | bytes | layout | source |
|---|---|---|---|
| MLA latent, FP8 (standard) | **576** | 512 + 64 FP8 E4M3, one per-tensor scale | `mla.py` `get_kv_cache_spec`; log "standard fp8 KV cache format"; decode block_len 589,824 = 1024 × 576 |
| indexer key, MXFP4 | **68** | 128 × FP4 = 64 B + 4 UE8M0 scales | release decode NIXL log "24 x index_value 64 B, 24 x index_scale 4 B"; block_len 69,632 |
| **per token, all 24 layers** | **15,456** | 13,824 MLA + 1,632 indexer | |
| alt: fp8_ds_mla + MXFP4 indexer | 17,376 | 656-byte DS layout | `mla.py` state_content_bytes |
| alt: FP8 MLA + FP8 indexer | 16,992 | 132-byte indexer | |
| deployed packed row | 15,606 | +150 B/token slack | "1024-row blocks 15980544 B apart" |

### KDA state per request (×69 layers, independent of length)

| part | per layer | ×69 | note |
|---|---|---|---|
| recurrent [96,128,128] FP32 | 6,291,456 | 434.1 MB | `mamba_utils.py` `kda_state_shape`; logs 393,216 per rank at TP16 |
| conv window [36864 ch, 3 cols] FP32 (SOL rule) | 442,368 | 30.5 MB | conv_kernel 4, so 3 history columns over q/k/v |
| **SOL state** | 6,733,824 | **464.6 MB** | one state per request |
| conv widened to 3+k cols, FP32 (k=3/5/7) | 884,736 / 1,179,648 / 1,474,560 | | `kda_state_shape(num_spec)` |
| deployed conv: BF16 (mamba_cache_dtype auto), 3+k cols | 442,368 / 589,824 / 737,280 | | logs: 46,080 B/rank TP16 k7; 110,592 B/rank TP4 k3 |
| deployed resident page sets (align mode) | | **2 + k + 1** sets (k7: 4.85 GB; 0 spec sets with `use-replayssm`) | `kv_cache_interface.py` MambaSpec; `mamba/abstract.py` |

### Drafter (release DSpark e4e6e626) sliding-window KV

- **Size:** 5 layers × 16 KV heads × 64 × (K,V) × BF16 = **20,480 B/token**.
- **Window:** 2,048 tokens, so the SOL is **41.9 MB per request**. The deployed bound is about 48.5 MB, with the FA2 tile slack plus one 128-token page.
- **Layout source:** decode block_len 32,768 = a 128-token page × 1 KV head per rank at TP16.
- The frozen 2026-11-14 sweep still ran the preview MLA drafter, which has its own MLA cache.

### Per request, examples (SOL: FP8 MLA + MXFP4 indexer + SOL KDA + drafter window)

| context | DSA KV | + KDA 0.465 GB + drafter 0.042 GB = SOL total | deployed TP16 k7 (packed rows + 10 BF16 page sets + drafter) |
|---|---|---|---|
| ISL p50 99.7K | 1.54 GB | 2.05 GB | 6.45 GB |
| ISL mean 145K | 2.24 GB | 2.75 GB | 7.16 GB |
| decode mean 189K | 2.93 GB | **3.43 GB** | 7.85 GB |
| ISL p90 311K | 4.81 GB | 5.32 GB | 9.75 GB |

**The deployed allocator is far from the SOL.**
- vLLM logs `GPU KV cache size = num_blocks / blocks_per_request(1M) × 1M`. Inverting that gives the state reservation per request:

  | deployment | blocks per 1M-token request | state rows | state per request | vs SOL state + drafter |
  |---|---|---|---|---|
  | TP16+DCP16 k7 | 107 (64 DSA rows) | 40 KDA rows (10 page sets × 4 rows of 15.98 MB) + 3 drafter rows | **11.0 GB** | 22x the 0.51 GB |
  | C512 DP4×TP4 k3, `use-replayssm` | 70 | 6 (3 sets × 2 rows) | 1.8 GB | |

- At TP16 the KDA state reservation is larger than the DSA KV of a 189K-token request. The SOL model should use the SOL rows and report the deployed numbers as an implementation gap.

### P→D transfer per request

- **Size:** a whole-prompt DSA KV of 2.24 GB at the mean ISL, plus 0.465 GB of KDA state, plus the 0.042 GB drafter window.
- **Relative to the prefill work:** that is about 27x the KV of the 5.46K fresh tokens the prefill computed, so it is ISL × 15,456 B, not fresh × 15,456 B.

### Sharding

- **DSA KV:** stored once per DCP group, interleaved per token. TP without DCP replicates the latent on every TP rank. Under DSA-DP (PRs #20/#23) it is stored once on the owner card. Under DP attention it lives on the request's rank.
- **KDA state:** split by heads across TP (96/TP heads per rank). It is never sharded across CP ranks; it is replicated over non-TP DCP ranks.
- **Drafter KV:** split by KV heads up to TP16, replicated beyond that.

## 5. Weights (release d9e99b07, from shard headers; GB = 1e9 B)

| component | bytes | params | sharding |
|---|---|---|---|
| routed experts, MXFP4 (packed + UE8M0 scales) | **1,446.46 GB** | 2,722.7 B | EP: 92 MoE layers × 896 × 17,547,264 B |
| KDA attention (q/k/v/g/o 12288-wide, b, f_a/f_b, FP32 conv) | 61.26 GB | 30.62 B | TP; 887.8 MB per layer |
| shared experts, BF16 (2 × 3072 fused) | 24.31 GB | 12.16 B | TP/replicated; 264 MB per layer |
| DSA MLA attention (q_a, q_b, kv_a, kv_b, o, output gate) | 11.15 GB | 5.57 B | TP over heads; 464.4 MB per layer |
| MoE latent up/down (7168↔3584), BF16 | 9.46 GB | 4.73 B | TP/replicated; 102.8 MB per layer |
| lm_head / embed_tokens, BF16 | 2.35 + 2.35 GB | 1.17 + 1.17 B | vocab-parallel |
| layer-0 dense MLP (33792) | 1.45 GB | 0.73 B | TP |
| router (896 × 7168 + bias) | 1.18 GB | 0.59 B | replicated |
| indexer projections (wq_b, wk, weights_proj) | 0.36 GB | 0.18 B | replicated/TP; 14.9 MB per layer |
| norms, attn-res | 0.01 GB | | replicated |
| **text model total** | **1,560.32 GB** | | (vision tower 0.90 GB more, which vLLM loads) |
| DSpark drafter (BF16) | 4.96 GB | | shares embed and lm_head; fc 0.51 GB, 5 layers × 0.82 GB, Markov head 0.34 GB |

- **NVFP4 checkpoint alternative:** routed experts 1,531.5 GB (0.5625 B/param), shared experts in MXFP8 12.53 GB, KDA projections in MXFP8 ~31.5 GB.
- **Layer mapping:** checkpoint layers are 0-based. The DSA layers are [3,7,...,91,92], which is config [4,...,92,93] 1-based. Layer 0 is the dense-MLP KDA layer.
- **Measured loaded weights per GPU** ("Model loading took"):

  | role | GiB per GPU | note |
  |---|---|---|
  | decode TP16+DCP16 | 131.2 | release drafter |
  | decode TP16+DCP16 | 132.4 | preview drafter |
  | prefill TP8+DCP8 | 205.9 | |
  | C512 prefill DP4×TP4 | 154.6 | |
  | C512 decode | 60.1 | |

- **The sharded math leaves a gap on decode TP16.** EP16 experts plus TP16 dense come to about 97.5 GB = 90.8 GiB. The ~40 GiB gap on TP16 is replicated pieces such as the drafter, router and buffers.
- **HBM:** 278.54 GiB per device, 272.48 GiB free at startup. Decode TP16 has 88.3 GiB available for KV at gpu_memory_utilization 0.90, after 155.2 GiB of weights + non-torch memory, 7.3 GiB of activations and 1.4 GiB of graphs. All per-role figures are in `hbm` and `engine_kv_memory`.

## 6. Per-run table (profiling records)

| run | C | requests | ISL mean | ISL p50 | fresh mean | hit | OSL mean | decode ctx (OSL-wtd) | req/s | latency s | idle s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1 | 404 | 285,366 | 294,692 | 5,598 | 98.04% | 1,753 | 371,751 | 0.112 | 2.52 | 6.4 |
| C8 | 8 | 1,457 | 169,807 | 110,941 | 4,291 | 97.47% | 1,166 | 242,392 | 0.406 | 1.99 | 17.73 |
| C16 1P2D | 16 | 2,791 | 159,265 | 111,651 | 4,598 | 97.11% | 1,029 | 211,342 | 0.775 | 1.68 | 18.98 |
| C48 1P1D | 48 | 8,293 | 161,140 | 95,931 | 5,115 | 96.83% | 1,056 | 204,532 | 2.30 | 3.32 | 17.55 |
| C64 | 64 | 11,967 | 149,503 | 89,357 | 5,089 | 96.60% | 1,042 | 194,142 | 3.31 | 4.16 | 15.16 |
| C72 1P3D | 72 | 14,085 | 145,618 | 85,960 | 5,392 | 96.30% | 1,068 | 191,132 | 3.90 | 2.80 | 15.65 |
| C96 1P2D | 96 | 16,937 | 145,009 | 93,525 | 5,511 | 96.20% | 1,057 | 188,325 | 4.69 | 3.79 | 16.67 |
| C96 1P2D k3 | 96 | 16,826 | 144,329 | 93,059 | 5,261 | 96.36% | 1,054 | 187,384 | 4.67 | 4.08 | 16.49 |
| C112 1P1D | 112 | 18,398 | 139,990 | 92,746 | 5,317 | 96.20% | 1,040 | 181,239 | 5.08 | 6.11 | 15.93 |
| C128 1P1D | 128 | 20,162 | 137,889 | 95,836 | 5,487 | 96.02% | 1,061 | 179,831 | 5.56 | 6.60 | 16.41 |
| C128 1P1D llh | 128 | 19,815 | 137,893 | 96,058 | 5,540 | 95.98% | 1,060 | 179,616 | 5.47 | 7.36 | 16.05 |
| C160 1P1D | 160 | 23,020 | 144,533 | 103,136 | 5,749 | 96.02% | 1,063 | 190,354 | 6.36 | 8.96 | 16.21 |
| C160 1P1D llh | 160 | 23,603 | 145,521 | 103,441 | 5,612 | 96.14% | 1,063 | 191,093 | 6.51 | 7.90 | 16.70 |
| C160 2P1D | 160 | 24,224 | 147,646 | 105,185 | 5,574 | 96.22% | 1,063 | 192,362 | 6.68 | 6.73 | 17.21 |
| C224 2P1D | 224 | 32,539 | 147,317 | 106,009 | 5,409 | 96.33% | 1,022 | 192,241 | 8.97 | 8.83 | 16.13 |
| C256 2P1D | 256 | 36,366 | 142,538 | 102,352 | 5,542 | 96.11% | 1,004 | 185,197 | 10.02 | 10.23 | 15.32 |

## 7. Caveats

- **Cached vs fresh is measured, not structural.** The cached/fresh split is the server-reported `usage_prompt_cache_read_tokens` (HBM + Mooncake) under the deployed Mooncake capacity. An infinite cache would bring fresh down toward the per-turn growth (median about 1.6K). Use the measured 5.46K for the AgentX SOL point.
- **Conv-state dtype:** the user rule says FP32 conv state. The deployed engines store the conv state in BF16, because `--mamba-cache-dtype` is unset (auto), and only the SSM state in FP32. The SOL rows use FP32.
- **C512 profile export:** C512 has no per-request export. Its ISL and OSL summary agrees with the pool: `workload_distribution_summary.txt` gives an ISL mean of 133K and an OSL mean of 1,002.
