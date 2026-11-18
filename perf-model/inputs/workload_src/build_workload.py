#!/usr/bin/env python3
"""Build k31vr/solmodel/inputs/workload.json from the derived inputs in workload_src/derived/.

Usage: python3 -I build_workload.py   (paths are relative to this file)

Derived inputs (regenerate with the scripts next to this file):
  derived/agentx_stats.json   analyze_agentx.py over results/k31vr-frozen-sweep-20261114/runs (aiperf profile_export)
  derived/hist_moments.json   hist_moments.py, same population, pooled histograms and attention-work moments
  derived/prefill_loggers.json prefill_loggers.py: prefill engine 10 s logger lines (prompt tok/s, local vs external hit)
  derived/prefill_steps.json  prefill_steps.py: ShardQ per-window step/token counters, max_num_batched_tokens,
                              available KV memory and logged KV capacity per engine
  derived/k31_release_tensors.json.gz  st_headers.py over every safetensors shard header of
                              Kimi-K3.1-Pro release d9e99b07 (hecate .../weizha/models/kimi-k3.1-pro-release-d9e99b07)
"""
import gzip, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "derived")
OUT = os.path.join(HERE, "..", "workload.json")

def jl(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)

st = jl("agentx_stats.json")
hm = jl("hist_moments.json")
pl = jl("prefill_loggers.json")
ps = jl("prefill_steps.json")
with gzip.open(os.path.join(D, "k31_release_tensors.json.gz"), "rt") as f:
    T = json.load(f)

P = st["POOLED"]

def r(x, n=1):
    return None if x is None else round(x, n)

# ---------------------------------------------------------------- workload
per_run = []
for run, v in st.items():
    if run == "POOLED":
        continue
    C = int(re.match(r"C(\d+)", run).group(1))
    lam = v["req_per_s"]
    inflight = v["mean_inflight_requests_littles_law"]
    per_run.append({
        "run": run, "concurrency": C, "n_profiling_requests": v["n_profiling"],
        "isl_mean": r(v["isl"]["mean"], 0), "isl_p50": r(v["isl"]["p50"], 0), "isl_p90": r(v["isl"]["p90"], 0),
        "cached_mean": r(v["cache_read"]["mean"], 0), "fresh_mean": r(v["fresh"]["mean"], 0),
        "fresh_p50": r(v["fresh"]["p50"], 0), "hit_rate_token_weighted": r(v["hit_rate_token_weighted"], 4),
        "zero_hit_request_frac": r(v["fresh_frac_requests_zero_hit"], 3),
        "osl_mean": r(v["osl"]["mean"], 0), "osl_p50": r(v["osl"]["p50"], 0),
        "decode_ctx_osl_weighted_mean": r(v["decode_ctx_token_weighted"]["mean"], 0),
        "req_per_s": r(lam, 3), "mean_request_latency_s": r(inflight / lam, 2),
        "closed_loop_idle_per_request_s": r((C - inflight) / lam, 2),
        "mean_inflight_requests": r(inflight, 1),
        "fresh_tok_per_s": r(v["fresh_tok_per_s"], 0), "out_tok_per_s": r(v["out_tok_per_s"], 0),
        "ttft_p50_s": r(v["ttft_s"]["p50"], 2), "ttft_p90_s": r(v["ttft_s"]["p90"], 2),
        "source": "results/k31vr-frozen-sweep-20261114/runs/" + v["source"],
    })
per_run.sort(key=lambda x: (x["concurrency"], x["run"]))
idle = [x["closed_loop_idle_per_request_s"] for x in per_run if x["concurrency"] >= 16]
idle_w = sum(x["closed_loop_idle_per_request_s"] * x["n_profiling_requests"] for x in per_run if x["concurrency"] >= 16) / \
    sum(x["n_profiling_requests"] for x in per_run if x["concurrency"] >= 16)

prefix_engines = []
for run, eng in pl.items():
    for f, v in eng.items():
        prefix_engines.append({"run": run, "engine_log": f, "steady_mean_prompt_tok_per_s": r(v["ss_mean_prompt_tps"], 0),
                               "steady_mean_running": r(v["ss_mean_running"], 2), "steady_mean_waiting": r(v["ss_mean_waiting"], 2),
                               "local_hbm_prefix_hit_pct": r(v["ss_mean_local_hit_pct"], 1),
                               "external_mooncake_hit_pct_of_remaining": r(v["ss_mean_ext_hit_pct"], 1)})

engine_mem = []
for run, eng in ps.items():
    for f, v in eng.items():
        if v["avail_kv_gib"] is None:
            continue
        e = {"run": run, "engine_log": f, "role": v["role"], "max_num_batched_tokens": v["max_num_batched_tokens"],
             "available_kv_gib_per_gpu": v["avail_kv_gib"], "logged_kv_capacity_tokens_at_max_model_len": v["gpu_kv_cache_tokens"]}
        if v["steps"]:
            e.update({"shardq_steps": v["steps"], "shardq_tokens": v["tokens"],
                      "mean_tokens_per_step": r(v["mean_tokens_per_step"], 0),
                      "window_p10_p50_p90_tokens_per_step": [r(v["window_tokens_per_step_p10"], 0), r(v["window_tokens_per_step_p50"], 0), r(v["window_tokens_per_step_p90"], 0)],
                      "steps_per_s": r(v["steps_per_10s_mean"] / 10.0, 2)})
        engine_mem.append(e)

# ---------------------------------------------------------------- KV / state constants
N_DSA, N_KDA = 24, 69
MLA_FP8 = 512 + 64              # standard fp8 MLA: 576 fp8 bytes, per-tensor scale (no per-token scale)
MLA_FP8_DS = 656                # fp8_ds_mla: 512 fp8 + 4 fp32 tile scales + 64 bf16 rope
IDX_MXFP4 = 64 + 4              # 128 dims fp4 + 4 UE8M0 scales (block 32)
IDX_FP8 = 128 + 4               # fp8 keys + 1 fp32 scale
per_tok = N_DSA * (MLA_FP8 + IDX_MXFP4)
KDA_HEADS, KDA_D, KDA_CONV_CH, KDA_CONV_K = 96, 128, 3 * 96 * 128, 4
kda_rec = KDA_HEADS * KDA_D * KDA_D * 4

def kda_conv(cols, b):
    return KDA_CONV_CH * cols * b

def kda_req(conv_cols, conv_bytes, page_sets=1):
    return N_KDA * (kda_rec + kda_conv(conv_cols, conv_bytes)) * page_sets

DR_LAYERS, DR_KVH, DR_HD, DR_WIN = 5, 16, 64, 2048
dr_tok = DR_LAYERS * DR_KVH * DR_HD * 2 * 2  # K+V, BF16
kv = {
    "_note": "All bytes are for the whole model and one request/token, before any sharding. B = bytes. Layer counts from release config.json text_config: dsa_config.dsa_layers (24, 1-based 4..92,93) and linear_attn_config.kda_layers (69).",
    "dsa_kv_per_token_per_layer": {
        "mla_latent_fp8_standard": {"bytes": MLA_FP8, "layout": "kv_lora 512 + rope 64, all FP8 E4M3, one per-tensor scale (no per-token scale)",
                                    "source": "vllm/models/kimi_k3/nvidia/mla.py get_kv_cache_spec (state_content_bytes None unless fp8_ds_mla); engine logs 'Using standard fp8 KV cache format'; decode NIXL log 'arrays 24 x row 576 B'; block_len 589824 = 1024 x 576"},
        "indexer_key_mxfp4": {"bytes": IDX_MXFP4, "layout": "index_head_dim 128 as FP4 E2M1 (64 B) + 4 UE8M0 scales (block 32)",
                              "source": "release decode NIXL log 'arrays ... 24 x index_value 64 B, 24 x index_scale 4 B'; block_len 69632 = 1024 x 68"},
        "alt_mla_fp8_ds_mla": {"bytes": MLA_FP8_DS, "layout": "512 FP8 + 4 x FP32 scales + 64 BF16 rope", "source": "mla.py state_content_bytes=656"},
        "alt_indexer_fp8": {"bytes": IDX_FP8, "layout": "128 FP8 + FP32 scale (DeepSeek V3.2 indexer cache format)"},
    },
    "dsa_kv_per_token_all_layers": {
        "sol_fp8_mla_plus_mxfp4_indexer": per_tok,
        "mla_part": N_DSA * MLA_FP8, "indexer_part": N_DSA * IDX_MXFP4,
        "alt_fp8_mla_plus_fp8_indexer": N_DSA * (MLA_FP8 + IDX_FP8),
        "alt_fp8_ds_mla_plus_mxfp4_indexer": N_DSA * (MLA_FP8_DS + IDX_MXFP4),
        "deployed_packed_row_bytes_per_token": 15980544 / 1024,
        "deployed_packed_row_note": "release decode logs: '1024-row blocks 15980544 B apart' (24 x (576+64+4) x 1024 = 15,826,944 + 153,600 B per row block = +150 B/token packing slack, ~1%)",
    },
    "kda_state_per_request_per_layer": {
        "recurrent_fp32": {"bytes": kda_rec, "shape": "[96 heads, 128, 128] FP32", "source": "mamba_utils.py kda_state_shape/kda_state_dtype (mamba-ssm-cache-dtype float32); logs mamba_ssm_size 393216 per rank at TP16 = 6 heads"},
        "conv_window_fp32_sol": {"bytes": kda_conv(KDA_CONV_K - 1, 4), "shape": "[3 x 96 x 128 = 36864 ch, 3 cols] FP32",
                                 "note": "SOL / user rule: FP32 conv state, window = conv_kernel-1 = 3 columns"},
        "conv_with_spec_cols_fp32": {"bytes_per_k": {str(k): kda_conv(KDA_CONV_K - 1 + k, 4) for k in (3, 5, 7)},
                                     "note": "kda_state_shape widens the conv state to 3 + num_spec columns"},
        "conv_deployed_bf16_with_spec_cols": {"bytes_per_k": {str(k): kda_conv(KDA_CONV_K - 1 + k, 2) for k in (3, 5, 7)},
                                              "source": "deployed mamba_cache_dtype=auto -> BF16 (model dtype); logs mamba_ssm_size conv 46080 B/rank at TP16 k7 = 2304 ch x 10 x 2 B, 110592 at TP4 k3 = 9216 x 6 x 2"},
    },
    "kda_state_per_request_all_layers": {
        "sol_one_state_fp32_conv3": kda_req(3, 4),
        "one_state_fp32_conv_3plus_k": {str(k): kda_req(3 + k, 4) for k in (3, 5, 7)},
        "deployed_page_set_bf16_conv_3plus_k": {str(k): kda_req(3 + k, 2) for k in (3, 5, 7)},
        "deployed_resident_align_mode": {str(k): kda_req(3 + k, 2, 2 + k + 1) for k in (3, 5, 7)},
        "deployed_resident_note": "mamba_cache_mode=align (prefix caching on) keeps 2 + num_speculative_blocks (=k unless use_replayssm) + num_prefill_checkpoint_blocks (1) page sets per request (v1/kv_cache_interface.py MambaSpec.max_memory_usage_bytes; model_executor/layers/mamba/abstract.py). use_replayssm=False by default (vllm/config/cache.py:380). A SOL design keeps 1 state (+ replay/recompute of the verify window); per-token verify states would be (1+k) sets.",
        "independent_of_context_length": True,
    },
    "drafter_swa_kv": {
        "per_token_bytes": dr_tok, "layout": "5 layers x 16 KV heads x 64 x (K,V) x BF16",
        "window_tokens": DR_WIN, "sol_per_request_bytes": dr_tok * DR_WIN,
        "deployed_bound_per_request_bytes": dr_tok * (DR_WIN + 256 - 64 + 128),
        "deployed_bound_note": "window + FA2 key-tile slack (tile 256 - hit unit 64, gqa_draft.py _hit_unit_block) + one 128-token page; k+1 verify rows ignored",
        "source": "DSpark e4e6e626 config.json (num_key_value_heads 16, head_dim 64, sliding_window 2048, 5 layers, bf16); release decode logs block_len 32768 = 128 tok x 1 KV head/rank x 64 x 2 x 2 B at TP16 ('GqaWindow on 128-token pages')",
        "frozen_sweep_note": "the 20261114 frozen sweep used the preview MLA drafter (kimi-k3.1-pro-dspark-preview, TOKENSPEED_MLA), which keeps its own MLA cache (dense_draft.py); the release GQA drafter replaces it",
    },
    "sharding_rules": {
        "dsa_kv": "stored once per DCP group, tokens interleaved across DCP ranks (cp-kv-cache-interleave-size 1); TP without DCP replicates the latent KV on every TP rank (MQA latent); DSA-DP (PRs #20/#23) stores it once per TP group on the owner card; DEP/DP attention: on the request's DP rank",
        "kda_state": "sharded by heads across TP (96/TP heads per rank); replicated across DCP ranks that are not TP ranks (MambaSpec.max_num_blocks_per_req: never CP-sharded); under DP attention the whole state is on the request's rank",
        "drafter_kv": "sharded by the 16 KV heads across TP up to 16; replicated beyond 16 (logs 'NIXL draft at DCP 1 (16 KV heads)')",
    },
    "deployed_capacity_reconciliation": {
        "_note": "vLLM logs 'GPU KV cache size' = num_blocks / blocks_per_request(at max_model_len 1,048,576) x 1,048,576 (v1/core/kv_cache_utils.py get_max_concurrency_for_kv_cache_config). Inverting it gives the pool rows a request takes beyond its DSA rows = the KDA/drafter state reservation of the deployed allocator.",
        "decode_TP16_DCP16_k7_release_gqa_drafter": {
            "num_blocks_per_rank": 5932, "row_bytes_per_rank": 15980544, "logged_tokens": 58132269,
            "blocks_per_1M_request": round(5932 / (58132269 / 1048576), 1), "dsa_rows": 64,
            "state_rows": "43 = 10 KDA page sets (align: 2 + k7 + 1 ckpt) x 4 rows + 3 drafter-window rows",
            "state_bytes_per_request_all_ranks": 43 * 15980544 * 16,
            "vs_sol_state": round(43 * 15980544 * 16 / (kda_req(3, 4) + dr_tok * DR_WIN), 1),
            "source": "k31vr-release-draft-sweep-20261117/runs/C1-a1118776/agentx/logs/hecate0245_decode_w0.out",
        },
        "decode_TP16_DCP16_k7_frozen_preview_drafter": {
            "num_blocks_per_rank": 4995, "logged_tokens": 50361895, "blocks_per_1M_request": round(4995 / (50361895 / 1048576), 1),
            "dsa_rows": 64, "state_rows": "40 = 10 KDA page sets x 4 rows",
            "source": "frozen C64-a1093680 hecate0338_decode_w0.out",
        },
        "decode_DP4_TP4_DCP4_k3_replayssm_C512": {
            "num_blocks_per_rank": 1290, "row_bytes_per_rank_approx": round(90.39 * 2**30 / 1290), "logged_tokens": 19323757,
            "blocks_per_1M_request": round(1290 / (19323757 / 1048576), 1), "dsa_rows": 64,
            "state_rows": "6 = 3 KDA page sets (align with use-replayssm: 2 + 0 + 1) x 2 rows",
            "state_bytes_per_request_all_ranks": round(6 * 90.39 * 2**30 / 1290 * 4),
            "source": "frozen C512-a1093684 hecate0320_decode_w0.out; config.yaml decode 'use-replayssm: true'",
        },
        "takeaway": "the deployed TP16 low-latency decode reserves ~11.0 GB of state per request (10.2 GB KDA in 40 rows + 0.77 GB drafter in 3 rows; 22x the 0.51 GB SOL state+drafter), more than the DSA KV of a 189K-token context (2.9 GB); the C512 replayssm config reserves ~1.8 GB. The SOL model should use kda_state_per_request_all_layers.sol_one_state_fp32_conv3 (+ drafter) and treat the deployed numbers as an implementation gap.",
    },
    "examples_per_request_bytes": {},
    "pd_transfer_per_request_bytes": {},
}
for name, ctx in (("isl_mean", P["isl"]["mean"]), ("decode_ctx_osl_weighted_mean", P["decode_ctx_token_weighted"]["mean"]),
                  ("isl_p50", P["isl"]["p50"]), ("isl_p90", P["isl"]["p90"]), ("ctx_1M_max_model_len", 1048576)):
    dsa = ctx * per_tok
    kv["examples_per_request_bytes"][name] = {
        "context_tokens": round(ctx), "dsa_kv": round(dsa),
        "kda_sol": kda_req(3, 4), "drafter_sol": dr_tok * DR_WIN,
        "total_sol": round(dsa + kda_req(3, 4) + dr_tok * DR_WIN),
        "total_deployed_k7": round(ctx * 15980544 / 1024 + kda_req(10, 2, 10) + dr_tok * (DR_WIN + 320)),
    }
for name, ctx in (("isl_mean", P["isl"]["mean"]), ("isl_p50", P["isl"]["p50"])):
    kv["pd_transfer_per_request_bytes"][name] = {
        "context_tokens": round(ctx), "dsa_kv": round(ctx * per_tok), "kda_one_state_sol": kda_req(3, 4),
        "drafter_window": dr_tok * DR_WIN,
        "note": "P->D pull of the whole prompt KV (cached + fresh) + final KDA state + drafter window; per fresh prefill token this is ISL/fresh = %.1fx the fresh-token KV" % (P["isl"]["mean"] / P["fresh"]["mean"]),
    }

# ---------------------------------------------------------------- weights
cat = collections.Counter(); par = collections.Counter()
dsa0 = set(int(x) for x in json.load(open(os.path.join(D, "weights_breakdown.json")))["dsa_layers_0based"])
def dt_params(dt, shape, nbytes):
    n = 1
    for s in shape:
        n *= s
    return n
def category(k):
    if k.startswith("vision_tower") or k.startswith("mm_projector"):
        return "vision_tower_and_projector_bf16"
    if "embed_tokens" in k:
        return "embed_tokens_bf16"
    if "lm_head" in k:
        return "lm_head_bf16"
    m = re.match(r"language_model\.model\.layers\.(\d+)\.(.*)", k)
    if not m:
        return "final_norms_bf16"
    i, n = int(m.group(1)), m.group(2)
    if ".experts." in n:
        return "moe_routed_experts_mxfp4"
    if "routed_expert_" in n:
        return "moe_latent_up_down_norm_bf16"
    if "shared_experts" in n:
        return "moe_shared_experts_bf16"
    if "block_sparse_moe.gate" in n:
        return "moe_router_bf16"
    if n.startswith("mlp."):
        return "layer0_dense_mlp_bf16"
    if "indexer" in n:
        return "dsa_indexer_proj_bf16"
    if n.startswith("self_attn"):
        return ("dsa_mla_attn_bf16" if i in dsa0 else "kda_attn_bf16_f32")
    return "layer_norms_and_attn_res_bf16"
for k, (dt, shape, nb) in T.items():
    c = category(k)
    cat[c] += nb
    if c == "moe_routed_experts_mxfp4":
        if k.endswith("weight_packed"):
            par[c] += shape[0] * shape[1] * 2   # 2 FP4 values per byte
    else:
        par[c] += dt_params(dt, shape, nb)
text_total = sum(v for k, v in cat.items() if not k.startswith("vision"))
EXP_PARAMS = 3 * 3072 * 3584
weights = {
    "_note": "Release Kimi-K3.1-Pro d9e99b07, bytes summed from every safetensors shard header (497,340 tensors). GB = 1e9 B. Sharding classes say how a topology divides each class.",
    "source": "hecate:/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07/model-0000*-of-000096.safetensors headers (st_headers.py); derived/k31_release_tensors.json.gz",
    "by_category_bytes": dict(sorted(cat.items(), key=lambda x: -x[1])),
    "by_category_params": dict(par),
    "text_model_total_bytes": text_total,
    "sharding_class": {
        "moe_routed_experts_mxfp4": "EP (896 experts; 92 MoE layers x 896 x 17,547,264 B)",
        "moe_latent_up_down_norm_bf16": "TP or replicated (per layer 7168x3584 up + 3584x7168 down)",
        "moe_shared_experts_bf16": "TP or replicated (2 shared experts fused: 6144 = 2 x 3072)",
        "moe_router_bf16": "replicated (896 x 7168 + FP32 bias)",
        "kda_attn_bf16_f32": "TP (q/k/v/g 12288x7168, o 7168x12288, b 96x7168, f_a 128x7168, f_b 12288x128, FP32 conv weights 3 x 12288x4, A_log, dt_bias)",
        "dsa_mla_attn_bf16": "TP over heads for q_b/kv_b/o/g; q_a/kv_a replicated or TP (q_a 1536x7168, q_b 18432x1536, kv_a 576x7168, kv_b 24576x512, o 7168x12288, output gate g 12288x7168)",
        "dsa_indexer_proj_bf16": "replicated or TP (wq_b 4096x1536, wk 128x7168, weights_proj 32x7168)",
        "layer0_dense_mlp_bf16": "TP (gate/up 33792x7168, down 7168x33792)",
        "embed_tokens_bf16": "vocab-parallel over TP or replicated (163840 x 7168)",
        "lm_head_bf16": "vocab-parallel over TP (163840 x 7168)",
        "vision_tower_and_projector_bf16": "loaded by vLLM for the multimodal arch; not on the text path",
    },
    "per_layer_bytes": {
        "routed_expert_one_expert_mxfp4": 17547264,
        "routed_experts_one_layer_mxfp4": 896 * 17547264,
        "moe_layer_nonexpert_bf16": 2 * 51380224 + 7168 + 3 * 88080384 + 12845056 + 3584,
        "kda_attention_layer": 4 * 176160768 + 176160768 + 1376256 + 1835008 + 3145728 + 3 * 196608 + 49152 + 512 + 512,
        "dsa_attention_layer_mla": 176160768 * 2 + 22020096 + 56623104 + 8257536 + 25165824 + 1024 + 3072,
        "dsa_indexer_layer": 12582912 + 1835008 + 458752 + 512,
        "layer0_dense_mlp": 3 * 484442112,
    },
    "nvfp4_checkpoint_alt": {
        "routed_experts_bytes": 92 * 896 * EXP_PARAMS * (0.5 + 1 / 16),
        "routed_experts_bytes_note": "NVFP4 group 16 with FP8 scales = 0.5625 B/param vs MXFP4 0.53125 (global scales ignored); see inputs/checkpoints.json",
        "shared_experts_mxfp8_bytes": par["moe_shared_experts_bf16"] * (1 + 1 / 32),
        "kda_projections_mxfp8_bytes_approx": (par["kda_attn_bf16_f32"] - 69 * (3 * 12288 * 4 + 12288 * 128 + 128 + 12288 + 128)) * (1 + 1 / 32),
    },
    "drafter_dspark_e4e6e626": {
        "file_bytes": 4959959810,
        "bf16_tensors": {
            "fc_aux_35840_to_7168": 7168 * 35840 * 2,
            "per_layer": {"q_proj": 6144 * 7168 * 2, "k_proj": 1024 * 7168 * 2, "v_proj": 1024 * 7168 * 2, "o_proj": 7168 * 6144 * 2,
                          "mlp_gate_up_down": 3 * 14336 * 7168 * 2},
            "layers": 5,
            "markov_head_w1_w2": 2 * 163840 * 512 * 2,
            "confidence_head": 7680 * 2 + 2,
        },
        "shares_target": ["embed_tokens", "lm_head"],
        "aux_hidden_state_layer_ids": [76, 80, 84, 88, 92],
        "source": "hecate .../cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626/{config.json, model.safetensors header}",
    },
    "measured_loaded_weights_gib_per_gpu": [
        {"role": "decode TP16+DCP16 (release GQA drafter)", "gib": 131.18, "source": "k31vr-release-draft-sweep-20261117/runs/C1-a1118776 hecate0245_decode_w0.out 'Model loading took'"},
        {"role": "decode TP16+DCP16 (preview MLA drafter)", "gib": 132.42, "source": "frozen C64-a1093680 hecate0338_decode_w0.out"},
        {"role": "prefill TP8+DCP8", "gib": 205.85, "source": "frozen C64-a1093680 hecate0334_prefill_w0.out.gz"},
        {"role": "C512 prefill DP4 x TP4/DCP4 (16 GPUs)", "gib": 154.62, "source": "frozen C512-a1093684 hecate0313_prefill_w0.out.gz"},
        {"role": "C512 decode DP4 x TP4/DCP4 (16 GPUs)", "gib": 60.11, "source": "frozen C512-a1093684 hecate0320_decode_w0.out"},
    ],
}

hbm = {
    "device_total_gib": 278.54, "free_at_startup_gib": 272.48,
    "source": "engine logs 'Free memory on device (272.48/278.54 GiB) on startup' (all VR200 runs)",
    "deployed_gpu_memory_utilization": {"prefill": 0.92, "decode": 0.90, "C512 decode": 0.93},
    "deployed_non_kv_examples": [
        {"role": "decode TP16+DCP16 release", "consumed_weights_plus_nontorch_gib": 155.15, "peak_activation_gib": 7.25, "cudagraph_gib": 1.43, "available_kv_gib": 88.29},
        {"role": "prefill TP8+DCP8", "consumed_weights_plus_nontorch_gib": 218.32, "peak_activation_gib": 8.46, "cudagraph_gib": 1.63, "available_kv_gib": 29.48},
        {"role": "C512 prefill DP4xTP4", "consumed_weights_plus_nontorch_gib": 169.4, "peak_activation_gib": 7.95, "cudagraph_gib": 0.43, "available_kv_gib": 78.9},
        {"role": "C512 decode DP4xTP4", "consumed_weights_plus_nontorch_gib": 161.13, "peak_activation_gib": 7.52, "cudagraph_gib": 5.37, "available_kv_gib": 90.39},
    ],
    "note": "non-torch includes NCCL/symmetric-memory buffers and the Mooncake/NIXL registrations; 'consumed' minus 'Model loading took' = non-torch + runtime buffers (~22-24 GiB decode TP16, ~12 GiB prefill TP8, ~15 GiB C512 prefill, ~101 GiB C512 decode incl. dloan/side buffers)",
}

workload = {
    "_meta": {
        "built_by": "k31vr/solmodel/inputs/workload_src/build_workload.py",
        "date": "2026-11-17",
        "population": "aiperf profile_export.jsonl(.gz) records with benchmark_phase == 'profiling' and was_cancelled == false, all 16 runs of results/k31vr-frozen-sweep-20261114 that kept the per-request export (C512 kept only the summary). Pooled = all records of those runs (dominated by C>=96).",
        "dataset": "semianalysis_cc_traces_weka_062126 (AgentX 'agentx profile', duration 3600 s, closed loop)",
        "units": "tokens unless stated; s = seconds; B = bytes",
    },
    "pooled": {
        "n_requests": P["n_profiling"],
        "isl": P["isl"], "osl": P["osl"], "cached_prefix_tokens": P["cache_read"], "fresh_prefill_tokens": P["fresh"],
        "prefix_hit_rate_token_weighted": P["hit_rate_token_weighted"],
        "zero_hit_request_frac": P["fresh_frac_requests_zero_hit"],
        "cache_read_granularity_tokens": P["cache_read_gcd_tokens"],
        "source_kind_mix": P["source_kind_mix"],
        "by_source_kind": {k[len("by_kind_"):]: v for k, v in P.items() if k.startswith("by_kind_")},
        "fresh_size_bucket_request_frac": P["fresh_size_bucket_request_frac"],
        "fresh_size_bucket_token_mass": P["fresh_size_bucket_token_mass"],
        "fresh_token_weighted_mean_cached_prefix": P["fresh_weighted_mean_prefix"],
        "fresh_token_weighted_mean_end_context": P["fresh_weighted_mean_end_ctx"],
        "decode_context_osl_weighted": P["decode_ctx_token_weighted"],
        "decode_context_decode_time_weighted": P["decode_ctx_time_weighted"],
        "first_turn_in_window_frac": P["first_turn_in_window_frac"],
        "first_turn_isl_mean": P["first_turn_isl_mean"], "first_turn_fresh_mean": P["first_turn_fresh_mean"],
        "same_conversation_turn_growth": {
            "isl_minus_prev_isl": P["structural_new_tokens_vs_prev_turn_in"],
            "isl_minus_prev_isl_plus_osl": P["structural_new_tokens_vs_prev_turn_in_plus_out"],
            "note": "negative values = context compaction/truncation between turns; the median new-input growth per follow-up turn is ~1.6K tokens, i.e. the measured fresh mean (5.46K) carries ~2-3x structural growth from partial-block tails, misses (12% zero-hit), and evictions",
        },
        "same_conversation_gap_s": P["think_time_s_start_minus_prev_end"],
        "same_conversation_gap_note": "next-turn start minus previous-turn end within one conversation_id; includes parent waits on subagents, so it is NOT the per-request think time; use closed_loop_idle_per_request_s",
    },
    "think_time": {
        "closed_loop_idle_per_request_s": {"mean_request_weighted_C>=16": round(idle_w, 2), "min": min(idle), "max": max(idle)},
        "definition": "C / lambda - mean request latency (Little's law per run): the idle time each of the C closed-loop session slots spends between a response and its next request",
        "use": "request rate per session slot = 1 / (idle + latency); in-flight requests = C x latency / (idle + latency) (30-40% of C at C>=96)",
    },
    "attention_work_moments": {
        k: v for k, v in hm.items() if not k.startswith("hist")
    },
    "attention_work_note": "mean_dense_attn_pairs_prefill = E[fresh*cached + fresh*(fresh+1)/2] (indexer logits per request); dense_pairs_per_fresh_token = average keys the indexer scores per fresh token; mean_sparse_attn_pairs_prefill_topk2048 = E[sum over fresh tokens of min(ctx, 2048)] (~2003 per fresh token); decode_ctx_per_output_token_mean = E[sum over output tokens of ctx]/E[OSL]",
    "histograms": {k: v for k, v in hm.items() if k.startswith("hist")},
    "per_run": per_run,
    "prefill_engines_steady_state": prefix_engines,
    "prefill_engines_note": "steady state = last 75% of the 10 s logger windows. local_hbm_prefix_hit_pct = vLLM 'Prefix cache hit rate' (HBM); external = 'External prefix cache hit rate' (Mooncake, % of tokens not hit locally that were found). Prompt tok/s is computed (fresh) tokens.",
    "prefill_scheduling": {
        "max_num_batched_tokens": {"prefill TP8+DCP8 (all 1P/2P runs)": 16384, "prefill DP4 x TP4/DCP4 (C512)": 32768, "decode": 8192},
        "c512_shardq_counters": [e for e in engine_mem if e.get("shardq_steps")],
        "note": "only the C512 prefill logs carry the per-window 'DCP ShardQ prefill ... steps, tokens' counter: ~15.2-15.9K tokens per step (mean), ~2.1 steps/s per DP4 engine at mnbt 32768. TP8 prefill runs pack chunks up to 16384 tokens; fresh token mass is 50% in requests with > 16K fresh tokens, so long requests split into 16K/32K chunks while the 81% of requests with < 4K fresh tokens share a step.",
    },
    "engine_kv_memory": engine_mem,
    "kv_state_bytes": kv,
    "weights": weights,
    "hbm": hbm,
}
with open(OUT, "w") as f:
    json.dump(workload, f, indent=1, default=float)
print("wrote", os.path.abspath(OUT))
print("per_tok", per_tok, "kda_sol", kda_req(3, 4), "kda_dep_k7", kda_req(10, 2, 10), "drafter", dr_tok * DR_WIN)
print("text_total", text_total / 1e9)
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {k:40s} {v/1e9:10.3f} GB  params {par[k]/1e9:8.3f} B")
print("idle", idle_w, min(idle), max(idle))
