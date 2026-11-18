# mb lane prefill: shared bench args (sourced by queue tasks). Release prefill roles from
# e2e/renders/dsrel-c48-1p1d-xdr (P1: TP8+DCP8 EP8) and dsrel-c512-xdr (P2: DP4 x TP4=DCP4, EP16), serving-only
# flags dropped (connectors, prefix-cache policy, numa, frontend, mamba checkpoint policy); dummy weights, real config.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; L=$W/mb/prefill; T=$L/tools
MV=$W/e2e/model-view/kimi-k3.1-pro-k3type
DRAFT=$W/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626
spec() { echo "{\"method\":\"dspark\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$1,\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$2}"; }
BASE=(--model $MV --trust-remote-code --language-model-only --load-format dummy
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 --gpu-memory-utilization ${PF_GMU:-0.92}
  --enable-expert-parallel --moe-backend kimi_k3_mega_moe --no-enable-flashinfer-autotune
  --attention-backend FLASHINFER_MLA_SPARSE --sparse-indexer-topk-backend trtllm_gvr2
  --quantization-config '{"moe":{"activation":"mxfp8"}}' --attention-config '{"indexer_kv_dtype":"mxfp4"}'
  --max-model-len 1048576 --distributed-timeout-seconds 1800
  --kv-len-jitter 0 --routing ${PF_ROUTING:-balanced} --seed 0 --iters ${PF_ITERS:-5} --samples ${PF_SAMPLES:-3})
P1=("${BASE[@]}" -tp 8 -dcp 8 --cp-kv-cache-interleave-size 1 --dcp-comm-backend a2a --block-size 1024
  --max-num-batched-tokens 16384 --max-num-seqs 32 --speculative-config "$(spec 7 4.16)")
P2=("${BASE[@]}" -tp 4 -dcp 4 -dp 4 --cp-kv-cache-interleave-size 1 --dcp-comm-backend a2a --enable-ep-weight-filter
  --use-replayssm --additional-config '{"decode_backend":"auto"}'
  --max-num-batched-tokens 32768 --max-num-seqs 64 --speculative-config "$(spec 3 3.08)")
P3=("${BASE[@]}" -tp 1 -dp 16 --enable-ep-weight-filter
  --max-num-batched-tokens 16384 --max-num-seqs 32 --speculative-config "$(spec 3 3.08)")
# cases (per DP rank / attention group): F = 16384 chunk tokens of ~5,456-fresh requests at L_ref 78,360 (KV length
# includes the query; prefix-cache hit), the optables reference regime; plus a fresh 16K chunk (no prefix)
C16=(--case pf16k_L78k=2*5461@78360,5462@78360 --case pf16k_fresh=16384)
C32=(--case pf32k_L78k=4*5461@78360,2*5462@78360)
pst() { echo "$(date +%T) $*" | tee -a $L/results/status.txt; }
guard() {  # guard <minutes needed>
  if [ -n "$NQ_DEADLINE" ] && [ $(( $(date +%s) + $1 * 60 )) -gt $NQ_DEADLINE ]; then pst "skip (time): need $1 min"; return 1; fi
}
