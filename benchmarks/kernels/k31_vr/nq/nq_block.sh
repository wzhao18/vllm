#!/bin/bash
# nq_block.sh <plan file>: run an nq A/B plan on one 12-node block (queue task of holds/runner.sbatch POOL=mx12, or any
# 12-node allocation via NQ_JOB): three concurrent 4-node TP16 slots A/B/C (one NVL72 segment), each running its plan
# lines in order with nq_ab.sh, then one summary over the block, then scancel of the block (NQ_NO_SCANCEL=1 keeps it).
# Plan lines: <slot A|B|C> <point> <old|new> <rows> <order> <ctx> [extra bench args; @R = the slot's results dir]
#   or '<slot> gputest': the GPU correctness stage (nq_gputest.sh) on the slot's first node (4 GPUs) and second node
#   (2 GPUs) before the slot's A/B lines; the other slots start their A/B meanwhile. Results in <R>/gputest/GPUTEST.txt.
# Order: prechecks (overlay, 12 nodes, every plan line resolves), then the task lock (a requeued copy of a task that
# already ran exits 0 instead of running again: holdd2.sh puts tasks of a cancelled block back in queue/pending; a
# failed precheck takes no lock), then idle GPUs (BUSY: abort, lock released, block not cancelled), containers, slots.
# env: NQ_TASK (lock name), NQ_TAG (results dir name), NQ_OV (code-under-test overlay root), NQ_IMG, NQ_TIMEOUT,
#      NQ_RUN_EST (see nq_ab.sh), NQ_MARGIN (s kept free at the block's end, default 600), NQ_BUSY_WAIT (s to wait for
#      the keepalive to yield, default 600), NQ_TEST_TIMEOUT / NQ_TEST_FILE_TIMEOUT (nq_gputest.sh).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLAN=$1; J=${NQ_JOB:-$SLURM_JOB_ID}
[ -f "$PLAN" ] && [ -n "$J" ] || { echo "usage: nq_block.sh <plan> (inside a block, or NQ_JOB=<jobid>)"; exit 2; }
TASK=${NQ_TASK:-$(basename $PLAN .plan)}
echo "nq block $TASK on $J precheck $(date) overlay ${NQ_OV:-base} plan $PLAN"
# --- prechecks: no lock, no results dir yet
if [ -n "$NQ_OV" ]; then
  [ -f $NQ_OV/vllm/__init__.py ] && [ -f $NQ_OV/vllm/models/kimi_k3/__init__.py ] || {
    echo "overlay $NQ_OV is not a whole vllm tree (stage_overlay.sh)"; exit 2; }
fi
NODES=($(scontrol show hostnames $(squeue -h -j $J -o %N)))
[ ${#NODES[@]} -ge 12 ] || { echo "need 12 nodes, job $J has ${#NODES[@]}"; exit 3; }
# every plan line's config resolves before any GPU time is spent (an unclassified production arg is an error)
grep -v '^\s*#' $PLAN | awk 'NF && $2 != "gputest"' | while read -r slot point drafter rows order ctx rest; do
  python3 $D/cfg2bench.py $point --arm ls --drafter $drafter --rows $rows > /dev/null || exit 2
done || { echo "plan does not resolve"; exit 2; }
if grep -v '^\s*#' $PLAN | awk '$2 == "gputest"' | grep -q . && [ -z "$NQ_OV" ]; then
  echo "plan has a gputest stage but no overlay (NQ_OV): the base env carries no tests"; exit 2
fi
# --- lock
mkdir -p $NB/locks
mkdir $NB/locks/$TASK 2>/dev/null || { echo "$TASK already ran ($(cat $NB/locks/$TASK/job 2>/dev/null)): nothing to do"; exit 0; }
echo $J > $NB/locks/$TASK/job
R=$NB/results/${NQ_TAG:-$TASK-$J}; mkdir -p $R; cp $PLAN $R/plan.txt
exec > >(tee -a $R/block.log) 2>&1
echo "nq block $TASK on $J start $(date) overlay ${NQ_OV:-base} plan $PLAN"
[ -n "$NQ_OV" ] && { echo "overlay rev:"; sed 's/^/  /' $NQ_OV/SOURCE_REV 2>/dev/null; }
left=$(squeue -h -j $J -o %L)
sec=$(echo "$left" | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3) s+=3600*$(n-2); if(n>=4) s+=86400*$(n-3); print s}')
export NQ_DEADLINE=$(( $(date +%s) + sec - ${NQ_MARGIN:-600} ))
echo "time left $left -> deadline $(date -d @$NQ_DEADLINE +%T)"
declare -A SLOT=([A]=$(IFS=,; echo "${NODES[*]:0:4}") [B]=$(IFS=,; echo "${NODES[*]:4:4}") [C]=$(IFS=,; echo "${NODES[*]:8:4}"))
for s in A B C; do echo "slot $s: ${SLOT[$s]}"; done
ALL=$(IFS=,; echo "${NODES[*]:0:12}")
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES
# --- idle GPUs: the hold keepalive (holds/keepalive.py) yields once this task is in queue/running; a co-tenant GPU
# context during warmup hangs DeepGEMM's barrier and skews timing, so a block that stays busy is not measured
busy_nodes() {
  local out
  out=$(srun --overlap --jobid $J -N12 -w $ALL --ntasks-per-node=1 bash -c \
    'echo "NQBUSY $(hostname) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"' 2>/dev/null)
  [ $(echo "$out" | grep -c "^NQBUSY ") = 12 ] || { echo "query-failed"; return; }
  echo "$out" | awk '$1 == "NQBUSY" && $3 != 0 {printf "%s:%s ", $2, $3}'
}
t0=$(date +%s)
while :; do
  busy=$(busy_nodes)
  [ -z "$busy" ] && break
  if [ $(( $(date +%s) - t0 )) -ge ${NQ_BUSY_WAIT:-600} ]; then
    echo "BUSY: GPU processes still on the block after ${NQ_BUSY_WAIT:-600} s ($busy): not measuring" | tee $R/SUMMARY.txt
    rm -rf $NB/locks/$TASK  # (a requeued copy may run on an idle block)
    echo "nq block $TASK aborted $(date) (block not cancelled)"; exit 4
  fi
  echo "$(date +%T) GPU processes on the block ($busy), waiting"; sleep 15
done
echo "GPUs idle on all 12 nodes after $(( $(date +%s) - t0 )) s"
# pyxis containers first, per slot in parallel (fresh nodes: the image import can outlast the NCCL rendezvous)
export ENROOT_CACHE_PATH=$K/enroot-cache
IMG=${NQ_IMG:-$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh}
for s in A B C; do
  srun --overlap --jobid $J -N4 -w ${SLOT[$s]} --ntasks-per-node=1 --container-image=$IMG --container-name=${NQ_CNAME:-k31nq-step} \
    --no-container-mount-home --container-mounts=$W:$W,$K:$K,$M:$M true > $R/container-$s.log 2>&1 &
done
wait
run_slot() {
  local s=$1 nl=${SLOT[$1]}
  grep -v '^\s*#' $PLAN | awk -v s=$s '$1 == s' | while read -r slot point drafter rows order ctx rest; do
    if [ "$point" = gputest ]; then
      echo "$(date +%T) slot $s: GPU correctness stage"
      IFS=, read -r h1 h2 _ <<< "$nl"
      bash $D/nq_gputest.sh $J $h1 $h2 $R/gputest < /dev/null
      continue
    fi
    echo "$(date +%T) slot $s: $point $drafter $rows $order $ctx $rest"
    bash $D/nq_ab.sh $J $nl $R/$s $point $drafter $rows $order $ctx -- ${rest//@R/$R/$s} < /dev/null
  done
}
# NQ-TUNE: NQ_STAGGER (120 s) is a guess; measure each slot's time to its first [forward] line with 0 / 120 s.
# staggered slot starts: the first ls process of a fresh overlay builds the DCP-exchange / DSA-core extensions under
# one lock in $NQ_BUILD and compiles the CuTe kernels; the others then mostly load
for s in A B C; do run_slot $s > $R/slot-$s.log 2>&1 & sleep ${NQ_STAGGER:-120}; done
wait
{
  echo "nq block $TASK job $J overlay ${NQ_OV:-base}: $(head -1 $NQ_OV/SOURCE_REV 2>/dev/null)"
  if [ -f $R/gputest/GPUTEST.txt ]; then echo "--- GPU correctness stage"; cat $R/gputest/GPUTEST.txt; fi
  for s in A B C; do
    [ -d $R/$s ] || continue
    echo "--- slot $s"
    python3 $D/nq_summary.py $R/$s --csv $R/$s.csv
    [ -f $R/$s/SKIPPED ] && { echo "SKIPPED (block time): $(tr '\n' ';' < $R/$s/SKIPPED)"; }
    ls $R/$s/DEAD-* > /dev/null 2>&1 && echo "DEAD at startup: $(cd $R/$s && ls DEAD-* | tr '\n' ' ')"
  done
} > $R/SUMMARY.txt 2>&1
cat $R/SUMMARY.txt
echo "nq block $TASK end $(date)"
if [ "${NQ_NO_SCANCEL:-0}" != 1 ]; then echo "scancel $J"; scancel $J; fi
