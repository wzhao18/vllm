#!/bin/bash
# queue task for holds/runner_once.sbatch POOL=nqk-kda (1 node, 4 GPUs, self-limited to 2 h 45 min): the kda group
# kernel microbench of the nq precision (FP32 KDA state; BF16 state as the previous reference) on the code under test
# overlay nq-786ce08 (branch wzhao/k31-nq-sdec @ 786ce08), inside the round-4 image. Stage (the user's call), e.g.:
#   cp $NB/kern/kda/task.sh $W/queue/pending/50-nqk-kda-786ce08.sh
# Results: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/results/kern-kda/
#   SUMMARY.txt (per case x rows: spec / stock / ratio / tactic; WORSE, RETUNE, MISSING lists; FP32 vs BF16),
#   SUMMARY.csv, ALL.csv, <tag>.csv + <tag>.log per process, inner.log, status.txt
# A previous results dir is moved aside (kern-kda.prev-<stamp>) so CSVs are never appended across runs.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
KD=$NB/kern/kda
OUT=${NQK_OUT:-$NB/results/kern-kda}
BUDGET=${NQK_BUDGET:-9900}  # 2 h 45 min for the whole task, scancel included
T0=$(date +%s)
export NQ_OV=${NQ_OV:-$NB/ov/nq-786ce08}
export NQ_BUILD=${NQ_BUILD:-$NB/cache/build/$(basename $NQ_OV)}
export ENROOT_CACHE_PATH=$K/enroot-cache
IMG=${NQ_IMG:-$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh}
J=${SLURM_JOB_ID:?SLURM_JOB_ID}
finish() { echo "$(date +%T) kda task end ($1), $(( $(date +%s) - T0 )) s; scancel $J"; scancel $J; exit $1; }
[ -d $OUT ] && mv $OUT $OUT.prev-$(date +%Y%m%d-%H%M%S)
mkdir -p $OUT
echo "$(date +%T) kda task: job $J nodes $SLURM_JOB_NODELIST overlay $NQ_OV build $NQ_BUILD out $OUT budget ${BUDGET}s"
if [ ! -d $NQ_OV/vllm/models/kimi_k3/nvidia/specialized_decode ] || [ ! -f $KD/bench_kda.py ]; then
  echo "NOT RUN: overlay or bench missing" | tee $OUT/SUMMARY.txt; finish 2
fi
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
CARGS=(--container-image=$IMG --container-name=nqk-kda --no-container-mount-home
       --container-mounts=$W:$W,$K:$K,$M:$M)
# inner budget: the task budget less container start, the merge fallback and scancel (10 min)
INNER=$((BUDGET - 600))
timeout $((INNER + 120)) srun --overlap --jobid $J -N1 -n1 "${CARGS[@]}" \
  bash $KD/run_inner.sh $OUT $INNER > $OUT/inner.log 2>&1
rc=$?
echo "$(date +%T) inner rc=$rc (124: cut by the task timeout)"
if [ ! -s $OUT/SUMMARY.txt ]; then  # the inner merge did not run: merge here (pure python, no torch)
  timeout 120 python3 $KD/bench_kda.py merge --out $OUT > $OUT/merge.log 2>&1 || echo "host merge failed: see merge.log"
fi
tail -n 40 $OUT/SUMMARY.txt 2>/dev/null
finish $rc
