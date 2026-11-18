#!/bin/bash
# ops1 lane (single-GPU kernel microbench vs SOL model), on a 4-node nq4 slot (the runner2 holds serve nq4 only):
#   nodes 1-2: all in-scope GEMM cases, 8 shards (BF16 cuBLAS / Kimi low-latency CuTe, BF16 BMM, MXFP8 cuBLASLt + CuTe DSL)
#   node 3   : attention-side families, one per GPU (DSA indexer + top-k, sparse MLA, KDA prefill/decode, AttnRes)
#   node 4   : DSpark drafter ops (DS.markov, DS.attn)
# Every process is one GPU (CUDA_VISIBLE_DEVICES), CUPTI cold-L2 timing. Never scancels.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; L=$W/mb-ops1; J=$SLURM_JOB_ID
IFS=, read -r H1 H2 H3 H4 _ <<< "$NQ_HOSTS"
TS=$(date +%m%d%H%M); R=$L/results
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh; export ENROOT_CACHE_PATH=$K/enroot-cache
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
echo "$(date +%T) ops1 all4 on $NQ_HOSTS job $J ts $TS"
run() {  # run <host> <outdir> <cname> <env assignments...> -- <inner args...>
  local h=$1 o=$2 cn=$3; shift 3; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p $o
  env "${envs[@]}" timeout 4800 srun --overlap --jobid $J -N1 -n1 -w $h --container-image=$IMG --container-name=$cn \
    --no-container-mount-home --container-mounts=$W:$W,$K:$K,$M:$M --export=ALL \
    bash $L/src/inner.sh $o "$@" > $o/inner-$h.log 2>&1
  echo "$(date +%T) $h $(basename $o) rc=$?"
}
C=$L/src/cases_ops1.json
run $H1 $R/gemm-$TS ops1-g1 OPS1_G0=0 OPS1_NT=8 -- gemm_bench.py $C &
run $H2 $R/gemm-$TS ops1-g2 OPS1_G0=4 OPS1_NT=8 -- gemm_bench.py $C &
run $H3 $R/attn-$TS ops1-a OPS1_TIMEOUT=4500 -- attn_bench.py $C &
run $H4 $R/draft-$TS ops1-d OPS1_TIMEOUT=1800 -- draft_bench.py $C &
wait
for d in gemm attn draft; do echo "== $d: $(cat $R/$d-$TS/rows.g*.jsonl 2>/dev/null | grep -c '"kind": "case"') case rows, $(cat $R/$d-$TS/rows.g*.jsonl 2>/dev/null | grep -c '"error"') errors"; done
echo "$(date +%T) ops1 all4 done"
