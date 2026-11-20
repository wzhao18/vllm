#!/bin/bash
# ops1 lane (microbench vs SOL model): follow-up checks: BF16 cuBLAS unaligned-N vs padded, big-GEMM run-to-run repeat with clocks. One node, 3 GPUs.
# MXFP8 cuBLASLt + CuTe DSL), 4 GPUs of one node in parallel, CUPTI cold-L2 timing. Never scancels.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; L=$W/mb-ops1; J=$SLURM_JOB_ID
H=${NQ_HOSTS%%,*}; TAG=verify-$(date +%m%d%H%M); OUT=$L/results/$TAG; mkdir -p $OUT
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh; export ENROOT_CACHE_PATH=$K/enroot-cache
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
echo "$(date +%T) ops1 gemm on $H job $J out $OUT"
OPS1_NG=3 timeout 1200 srun --overlap --jobid $J -N1 -n1 -w $H --container-image=$IMG --container-name=ops1-mbv \
  --no-container-mount-home --container-mounts=$W:$W,$K:$K,$M:$M \
  bash $L/src/inner.sh $OUT verify_bench.py > $OUT/inner.log 2>&1
echo "$(date +%T) rc=$? $(cat $OUT/rows.g*.jsonl | grep -c '"kind": "case"') case rows"
tail -12 $OUT/inner.log
