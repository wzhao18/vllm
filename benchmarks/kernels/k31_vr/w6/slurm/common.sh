# W6 step-bench common args (W0 harness, stock DEP decode config). source; then "${COMMON[@]}".
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; WD=$W/w0dep; W6=$W/w6
MODEL=/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1/rainhollow-test-weights-1024_v1
DRAFT=/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3-dspark
COMMON=(--model $MODEL --trust-remote-code --language-model-only --load-format dummy --tensor-parallel-size 1
  --enable-expert-parallel --moe-backend deep_gemm_mega_moe --attention-backend FLASHINFER_MLA_SPARSE
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --quantization-config '{"moe":{"activation":"mxfp8"}}'
  --use-replayssm --max-model-len 1048576 --max-num-batched-tokens 8192 --kv-len-jitter 0 --seed 0 --iters 20 --samples 5)
spec() { # spec <k>
  echo "{\"method\":\"dspark\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$1,\"attention_backend\":\"TOKENSPEED_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
}
kcfg() { # kcfg <topk backend>
  echo "{\"flashinfer_sparse_mla_kernel\":\"trtllm\",\"enable_flashinfer_autotune\":false,\"sparse_indexer_topk_backend\":\"$1\"}"
}
# Arms: name -> "<env tag> <topk backend> <attention-config json>"
declare -A ARM
ARM[stock]="w0dep-a050f29 auto {\"tokenspeed_mla_min_split_kv\":16}"
ARM[auto]="${W6_TAG:-w6-dev} auto {\"tokenspeed_mla_min_split_kv\":16}"
ARM[bal]="${W6_TAG:-w6-dev} balanced {\"tokenspeed_mla_min_split_kv\":16}"
# M5: length-proportional drafter split-KV (needs the patched tokenspeed in the env)
ARM[m5s32]="${W6_M5_TAG:-w6-m5} balanced {\"tokenspeed_mla_max_split_kv\":32}"
ARM[m5s64]="${W6_M5_TAG:-w6-m5} balanced {\"tokenspeed_mla_max_split_kv\":64}"
# M5 with the split budget of the uniform floor (16 x requests), at most 64 per request
ARM[m5b32]="${W6_M5B_TAG:-w6-m5b} balanced {\"tokenspeed_mla_min_split_kv\":16,\"tokenspeed_mla_max_split_kv\":32}"
ARM[m5c32]="${W6_M5C_TAG:-w6-m5c} balanced {\"tokenspeed_mla_min_split_kv\":16,\"tokenspeed_mla_max_split_kv\":32}"
ARM[m5b]="${W6_M5B_TAG:-w6-m5b} balanced {\"tokenspeed_mla_min_split_kv\":16,\"tokenspeed_mla_max_split_kv\":64}"
# Review fixes (aeb5f5ed8): m5c32 + per-batch cap of the drafter splits at the Int32 workspace limit
ARM[m5rf]="${W6_RF_TAG:-w6-rf} balanced {\"tokenspeed_mla_min_split_kv\":16,\"tokenspeed_mla_max_split_kv\":32}"
# + uniform split when the longest request has <= 2 x min_split_kv KV tiles (no slots for short batches)
ARM[m5rf2]="${W6_RF2_TAG:-w6-rf2} balanced {\"tokenspeed_mla_min_split_kv\":16,\"tokenspeed_mla_max_split_kv\":32}"
