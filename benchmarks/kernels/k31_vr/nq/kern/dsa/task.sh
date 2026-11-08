#!/bin/bash
# queue task for holds/runner_once.sbatch (POOL nqk-dsa, 1 node, 4 GPUs): the nq DSA kernel microbenchmarks of the code
# under test (overlay nq-786ce08 = wzhao/k31-nq-sdec 786ce08), Kimi K3.1 specialized decode DSA layers at DCP16 rank
# shapes, fp8 MLA KV (trtllm-gen), verify q4 / q8, rows up to 2048:
#   core   1 GPU x 4 processes: DSA sparse core (on by policy at q8 rows <= 16, every variant) vs the trtllm-gen chain, plus
#          prims_ts fp8 and the previous nvfp4_ds_mla (prims_ts) reference, rows <= 64 (5 count patterns) and 96..2048
#   value  torchrun 4 GPUs (4 real ranks + 12 emulated): DcpValue bf16 (PreparedDsa / WideDsa), forced row-tile
#          strides, "proj" (bmm + DcpValue), DcpValue(pushed[/spread]) vs the stock A2A LSE reduce + W_UV + gate
#   front  torchrun 4 GPUs: WideDsa MlaQuery (publish + gathered poll; nvfp4 ref) vs the stock S2-DSA q path
#   summary  tables, WORSE list, CSV
# Every stage runs inside the round-4 image via srun --overlap on this job's node with the nq_gputest_inner.sh
# environment (kern/dsa/inner.sh), resumes after a hang or a kill, and the task ends by 2h45m, then scancels the job.
# Results: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/results/kern-dsa/{SUMMARY.txt,summary.csv,
#          arms.csv,core/,value/,front/,*.log}
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench; J=${SLURM_JOB_ID:?SLURM_JOB_ID}
KD=$NB/kern/dsa; R=$NB/results/kern-dsa; mkdir -p $R
export NQ_OV=$NB/ov/nq-786ce08
export NQ_BUILD=$NB/cache/build/nq-786ce08 ENROOT_CACHE_PATH=$K/enroot-cache
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
T0=$(date +%s); BUDGET=${NQK_BUDGET:-9900}; END=$(( T0 + BUDGET - 360 ))
# Self timeout: whatever happens, the job is released at 2h45m.
( sleep $BUDGET; echo "$(date +%T) task budget ${BUDGET}s reached: scancel $J"; scancel $J ) &
WATCH=$!
HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
CARGS=(--container-image=$IMG --container-name=k31nqk-dsa --no-container-mount-home
       --container-mounts=$W:$W,$K:$K,$M:$M)
[ -d $NQ_OV/vllm ] || { echo "no overlay $NQ_OV"; scancel $J; exit 2; }
echo "$(date +%T) kern-dsa on $HOST (job $J), overlay $NQ_OV ($(head -1 $NQ_OV/SOURCE_REV 2>/dev/null)), until $(date -d @$END +%T)"

# stage <name> <cap s> <reserve s for the later stages>
stage() {
  local name=$1 cap=$2 reserve=$3 now dl lim
  now=$(date +%s); dl=$(( now + cap )); lim=$(( END - reserve )); [ $dl -gt $lim ] && dl=$lim
  if [ $dl -le $(( now + 300 )) ]; then echo "$(date +%T) stage $name SKIPPED: no time left" | tee -a $R/STAGES.txt; return; fi
  echo "$(date +%T) stage $name until $(date -d @$dl +%T)" | tee -a $R/STAGES.txt
  timeout $(( dl - now + 240 )) srun --overlap --jobid $J -N1 -n1 -w $HOST "${CARGS[@]}" \
    bash $KD/inner.sh $name $R $dl > $R/$name.srun.log 2>&1
  echo "$(date +%T) stage $name rc=$?" | tee -a $R/STAGES.txt
}

stage core 2400 3900
stage value 4200 1500
stage front 2100 300
timeout 600 srun --overlap --jobid $J -N1 -n1 -w $HOST "${CARGS[@]}" bash $KD/inner.sh summary $R 0 \
  > $R/summary.srun.log 2>&1 || python3 $KD/summarize.py $R/core $R/value $R/front --csv $R/summary.csv \
  --arms $R/arms.csv > $R/SUMMARY.txt 2>&1
cat $R/SUMMARY.txt
echo "$(date +%T) kern-dsa done ($(( $(date +%s) - T0 )) s)"
kill $WATCH 2>/dev/null
scancel $J
