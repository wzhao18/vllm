#!/bin/bash
# queue task for holds/runner_once.sbatch POOL=nqk-draft (1 node, 4 GPUs): the drafter kernel microbenchmarks (GEMM
# tactic tables vs cuBLAS incl. the split-K sweep, Markov rank 512 / 256 fused vs the stock loop, DraftDcp vs the stock
# drafter DCP path) on the code under test = overlay nq-786ce08, inside the round-4 image. Self timeout 2h40m, then
# scancels its job. Results: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/results/kern-draft/
#   (SUMMARY.txt, status.txt, gemm/<linear>/*.csv, markov/markov.csv, dcp/dcp.csv, dcp_single/*.csv, *.log)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
KD=$NB/kern/draft
export NQ_OV=$NB/ov/nq-786ce08
export NQ_BUILD=$NB/cache/build/nq-786ce08 ENROOT_CACHE_PATH=$K/enroot-cache
OUT=$NB/results/kern-draft
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
mkdir -p $OUT
echo "$(date +%T) kern-draft task: job ${SLURM_JOB_ID:?SLURM_JOB_ID} nodes $SLURM_JOB_NODELIST overlay $NQ_OV" | tee -a $OUT/status.txt
H=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
timeout ${KD_TIMEOUT:-9600} srun --overlap --jobid $SLURM_JOB_ID -N1 -n1 -w $H \
  --container-image=$IMG --container-name=k31nq-kern-draft --no-container-mount-home \
  --container-mounts=$W:$W,$K:$K,$M:$M \
  bash $KD/inner.sh $OUT > $OUT/inner.log 2>&1
rc=$?
echo "$(date +%T) kern-draft task end rc=$rc (124 = self timeout)" | tee -a $OUT/status.txt
[ -f $OUT/SUMMARY.txt ] || python3 $KD/summarize.py $OUT > $OUT/SUMMARY.txt 2>&1
scancel $SLURM_JOB_ID
exit $rc
