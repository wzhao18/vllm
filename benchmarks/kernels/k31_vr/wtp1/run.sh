#!/bin/bash
# usage: run.sh <jobid> <nnodes> <name> <layout> <wq> <kv> <shapes set> [extra bench args]
#  layout: d4x4 (TP4/DCP4 x DP4, EP16) | d4x1 (TP4/DCP4, EP4; 1 node) | dep16 | dep4 | tp16 (TP16/DCP16 stock) | tp16ls (latency_specialized)
#  wq: bf16 | mxfp8 ; kv: fp8 (trtllm-gen) | f8pts (fp8, prims_ts) | fp4 (nvfp4_ds_mla + MXFP4 idx, prims_ts)
#  shapes set: main | cap | all | x<r>  (q from Q env, default 4)
#  current DEP recipe (W6 M1+M5 m5rf): TOPK=balanced MINSPLIT=16 MAXSPLIT=32 (defaults); ACFG_EXTRA=',"k":v' appends
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; D=$W/wtp1/tools
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models
J=$1; NN=$2; NAME=$3; LAY=$4; WQ=$5; KVV=$6; SET=$7; shift 7
Q=${Q:-4}; K=$((Q - 1))
R=$W/wtp1/results; mkdir -p $R
case $LAY in
  d4x4) PAR=(--tensor-parallel-size 4 --decode-context-parallel-size 4 --data-parallel-size 4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe); G=4;;
  d4x1) PAR=(--tensor-parallel-size 4 --decode-context-parallel-size 4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe); G=4;;
  d2x2) PAR=(--tensor-parallel-size 2 --decode-context-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --moe-backend deep_gemm_mega_moe); G=2;;
  dep16) PAR=(--tensor-parallel-size 1 --data-parallel-size 16 --enable-expert-parallel --moe-backend deep_gemm_mega_moe); G=1;;
  dep4) PAR=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe); G=1;;
  tp16) PAR=(--tensor-parallel-size 16 --decode-context-parallel-size 16); G=16;;
  tp16ls) PAR=(--tensor-parallel-size 16 --decode-context-parallel-size 16 --additional-config '{"decode_backend":"latency_specialized"}'); G=16;;
  *) echo bad layout; exit 2;;
esac
[ -n "$LAYOUT_ARGS" ] || LAYOUT_ARGS="--cp-kv-cache-interleave-size 1"
case $KVV in
  fp8) KVA=(--kv-cache-dtype fp8); KER=trtllm; IDX=fp8; export D4_PTS=0;;
  f8pts) KVA=(--kv-cache-dtype fp8); KER=prims_ts; IDX=fp8; export D4_PTS=1;;
  fp4) KVA=(--kv-cache-dtype nvfp4_ds_mla); KER=prims_ts; IDX=mxfp4; export D4_PTS=1;;
  n4f8) KVA=(--kv-cache-dtype nvfp4_ds_mla); KER=prims_ts; IDX=fp8; export D4_PTS=1;;
esac
DRAFTQ=""
case $WQ in
  bf16) QC='{"moe":{"activation":"mxfp8"}}';;
  mxfp8) QC='{"moe":{"activation":"mxfp8"},"targets":{"re:.*\\.self_attn\\.(in_proj_qkvgfab|fused_qkv_a_proj|q_b_proj|g_proj|o_proj)$":"mxfp8","re:.*\\.self_attn\\.dsa_indexer\\.(wq_index|wkv_index)$":"mxfp8","re:.*\\.block_sparse_moe\\.shared_experts\\.(gate_up_proj|down_proj)$":"mxfp8","re:.*\\.block_sparse_moe\\.routed_expert_(up|down)_proj$":"mxfp8","re:.*\\.layers\\.\\d+\\.mlp\\.(gate_up_proj|down_proj)$":"mxfp8"}}'
    DRAFTQ=',"quantization":"mxfp8"';;
esac
SPEC="{\"method\":\"dspark\",\"model\":\"$M/kimi-k3-dspark\",\"num_speculative_tokens\":$K,\"attention_backend\":\"TOKENSPEED_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"$DRAFTQ}"
SH=$R/$NAME.shapes.json
python3 $D/mkshapes.py $D/decode_ctx_vectors.json $SH $G $Q $SET > $R/$NAME.shapes.txt
MODEL=${MODEL:-$M/kimi-k3.1/rainhollow-test-weights-1024_v1}
bash $D/launch.sh $J $NN $R/$NAME.log --model $MODEL --trust-remote-code --language-model-only --load-format dummy \
  "${PAR[@]}" $LAYOUT_ARGS --dcp-comm-backend a2a \
  --attention-backend FLASHINFER_MLA_SPARSE "${KVA[@]}" --mamba-ssm-cache-dtype bfloat16 \
  --attention-config "{\"indexer_kv_dtype\":\"$IDX\",\"tokenspeed_mla_min_split_kv\":${MINSPLIT:-16},\"tokenspeed_mla_max_split_kv\":${MAXSPLIT:-32}${ACFG_EXTRA}}" \
  --kernel-config "{\"flashinfer_sparse_mla_kernel\":\"$KER\",\"enable_flashinfer_autotune\":false,\"sparse_indexer_topk_backend\":\"${TOPK:-balanced}\"}" \
  --quantization-config "$QC" --use-replayssm \
  --max-model-len 1048576 --max-num-batched-tokens ${MNBT:-8192} --gpu-memory-utilization ${GMU:-0.90} \
  --speculative-config "$SPEC" --routing ${ROUTING:-balanced} \
  --shapes $SH --kv-len-jitter 0 --seed 0 --iters 20 --samples 5 --output-json $R/$NAME.json "$@"
