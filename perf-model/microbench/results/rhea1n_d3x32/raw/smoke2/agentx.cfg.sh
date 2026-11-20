# cfg2bench /lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/tools/rhea_d3x32_dp1.json arm=ls drafter=/lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/models/dspark-preview-L24 q=4
# NOTE: config drafter /raid/common-models/kimi-k3.1-pro-dspark replaced by /lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/models/dspark-preview-L24
# capture: measured rows only (4 sizes, full list 128; --full-capture for the production list): sizes [96, 128, 384, 512]
# capture: verify rows [96, 128, 384, 512] (by-products [96, 384]), drafter batches [32, 43, 128] (by-products [43]); capacity anchors: largest verify row 512, largest drafter batch 128 (= the full list's)
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RPC_TIMEOUT=600000
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_USE_DIRECT_DCP_A2A=1
export VLLM_USE_DIRECT_DCP_KV_GATHER=1
export VLLM_USE_DIRECT_DCP_Q_GATHER=1
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_USE_V2_MODEL_RUNNER=1
NQ_Q=4
NQ_TP=4
NQ_DP=1
NQ_WORLD=4
NQ_MNS=128
NQ_MAXCAP=512
NQ_PRODLIMIT=512
NQ_CAP=96,128,384,512
NQ_FULLCAP=4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268,272,276,280,284,288,292,296,300,304,308,312,316,320,324,328,332,336,340,344,348,352,356,360,364,368,372,376,380,384,388,392,396,400,404,408,412,416,420,424,428,432,436,440,444,448,452,456,460,464,468,472,476,480,484,488,492,496,500,504,508,512
NQ_ROWS=128
NQ_ARGS=(--model /lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/models/k31-view-L24 --load-format dummy --attention-backend FLASHINFER_MLA_SPARSE --attention-config '{"indexer_kv_dtype": "mxfp4"}' --cp-kv-cache-interleave-size 1 --cpu-distributed-timeout-seconds 1800 --data-parallel-size 1 --dcp-comm-backend a2a --decode-context-parallel-size 4 --distributed-timeout-seconds 1800 --enable-ep-weight-filter --enable-expert-parallel --gpu-memory-utilization 0.93 --kernel-config '{"flashinfer_sparse_mla_kernel": "trtllm"}' --kv-cache-dtype fp8 --language-model-only --mamba-ssm-cache-dtype float32 --max-model-len 1048576 --max-num-batched-tokens 8192 --moe-backend deep_gemm_mega_moe --no-enable-flashinfer-autotune --quantization-config '{"moe": {"activation": "mxfp8"}}' --sparse-indexer-topk-backend deep_select --tensor-parallel-size 4 --trust-remote-code --use-replayssm --max-num-seqs 128 --max-cudagraph-capture-size 512 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [96, 128, 384, 512]}' --speculative-config '{"method": "dspark", "model": "/lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/models/dspark-preview-L24", "num_speculative_tokens": 3, "attention_backend": "TOKENSPEED_MLA", "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.08}' --additional-config '{"decode_backend": "latency_specialized"}')
