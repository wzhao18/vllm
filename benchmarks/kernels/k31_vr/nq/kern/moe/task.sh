#!/bin/bash
# queue task for holds/runner_once.sbatch POOL=nqk-moe (1 node, 4 GPUs): the MoE-group kernel microbenchmark of the
# code under test (overlay nq-786ce08) at TP16 = DCP16 per-rank shapes, 64..2048 rows: wide routed experts (single
# launch <= 768 rows, chunked above) vs the stock FlashInfer TRT-LLM MXFP4 x MXFP8 experts, FrontRows, the wide tail
# (tail_rows), the wide o-projection and AttnRes vs their stock seams (bench_moe_kern.py; inner.sh has the phases).
# Self timeout 2h45m, then scancel of this allocation.
# Results: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/results/kern-moe/{SUMMARY.txt,all.csv,*.csv,logs/}
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
D=$NB/kern/moe
OUT=$NB/results/kern-moe
export NQ_OV=${NQ_OV:-$NB/ov/nq-786ce08}
export NQ_BUILD=$NB/cache/build/$(basename $NQ_OV)
export ENROOT_CACHE_PATH=$K/enroot-cache
T0=$(date +%s)
export KM_DEADLINE=$(( T0 + 165 * 60 - 180 ))  # 2h45m minus the summary / scancel margin
mkdir -p $OUT
echo "kern-moe task $(date) job ${SLURM_JOB_ID:-none} nodes ${SLURM_JOB_NODELIST:-?} overlay $NQ_OV ($(head -1 $NQ_OV/SOURCE_REV 2>/dev/null))" | tee -a $OUT/task.log
if [ ! -d "$NQ_OV/vllm" ] || [ ! -f $D/bench_moe_kern.py ]; then
  echo "kern-moe: missing overlay or bench" | tee -a $OUT/task.log
  [ -n "$SLURM_JOB_ID" ] && scancel $SLURM_JOB_ID
  exit 2
fi
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
CARGS=(--container-image=$IMG --container-name=k31nq-kern-moe --no-container-mount-home
       --container-mounts=$W:$W,$K:$K,$M:$M)
timeout --kill-after=60 $(( KM_DEADLINE - $(date +%s) + 60 )) \
  srun --overlap --jobid $SLURM_JOB_ID -N1 -n1 "${CARGS[@]}" bash $D/inner.sh $OUT >> $OUT/inner.log 2>&1
rc=$?
echo "kern-moe inner rc=$rc after $(( $(date +%s) - T0 ))s" | tee -a $OUT/task.log
# (pure python: also runs outside the image)
python3 $D/bench_moe_kern.py summary --out $OUT > $OUT/summary.host.log 2>&1 || true
grep -A40 "^WORSE" $OUT/SUMMARY.txt 2>/dev/null | head -60 | tee -a $OUT/task.log
scancel $SLURM_JOB_ID
exit $rc
