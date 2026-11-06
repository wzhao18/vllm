#!/bin/bash
# inner.sh <out dir>: inside the nq e2e image on one 4-GPU node, the MoE-group kernel microbenchmark
# (bench_moe_kern.py) of the code under test NQ_OV. Environment = nq_gputest_inner.sh (env/k31vr_env.sh with
# K31VR_VLLM_TAG=jh-242fb49cd4, the overlay first on PYTHONPATH, VLLM_SSM_CONV_STATE_LAYOUT=DS, per-overlay build dirs).
#   phase 1 (1 GPU each, concurrently): experts uniform / req4 / req8 routes on GPUs 0 / 1 / 2, attnres on GPU 3
#   phase 2 (torchrun, 4 real ranks + 12 emulated, sequentially): front, layer, tail, oproj
#   then the summary (SUMMARY.txt, all.csv).
# Every section runs to its budget: a run that dies (watchdog on a hung case, crash) is restarted with the
# journal (cases done / hung are skipped) until it exits 0, its attempts or its budget run out, or KM_DEADLINE.
# A sticky CUDA error (illegal address, ...) journals its case as FAIL and exits 18: the restart resumes past it.
# env: NQ_OV (required), KM_DEADLINE (epoch s, required), KM_QUICK=1 (fewer variants), KM_ROWS (override rows).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; NB=$W/nqbench
OUT=$1
: "${NQ_OV:?NQ_OV}" "${KM_DEADLINE:?KM_DEADLINE}"
export K31VR_VLLM_TAG=jh-242fb49cd4
source $W/env/k31vr_env.sh
export PYTHONPATH=$NQ_OV:${PYTHONPATH#*:}:$W/e2e/env/pydeps:$W/wide/tools:$W/merge/tools/shim
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
export NQ_BUILD=${NQ_BUILD:-$NB/cache/build/$(basename $NQ_OV)}
export K31_DCPX_BUILD_DIR=$NQ_BUILD/dcpx K31_DSAC_BUILD_DIR=$NQ_BUILD/dsac
mkdir -p $OUT/logs $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR
B=$NB/kern/moe/bench_moe_kern.py
EXTRA=(); [ "${KM_QUICK:-0}" = 1 ] && EXTRA+=(--quick); [ -n "$KM_ROWS" ] && EXTRA+=(--rows "$KM_ROWS")
cd $OUT
now() { date +%s; }
echo "KMOE host $(hostname) start $(date) deadline $(date -d @$KM_DEADLINE 2>/dev/null || echo $KM_DEADLINE)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
python3 -c "import vllm, os; print('KMOE vllm from', os.path.dirname(vllm.__file__))"
python3 -c "import vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime as e; print('KMOE experts_runtime', e.__file__)"

# run_section <name> <budget s> <max attempts> <cmd...>: restart on failure (resume via the journal) within the budget
run_section() {
  local name=$1 budget=$2 tries=$3; shift 3
  local end=$(( $(now) + budget )); [ $end -gt $(( KM_DEADLINE - 120 )) ] && end=$(( KM_DEADLINE - 120 ))
  local i rc=1 left
  for i in $(seq 1 $tries); do
    left=$(( end - $(now) ))
    if [ $left -lt 60 ]; then echo "KMOE $name: no time left (attempt $i)"; break; fi
    echo "KMOE $(date +%T) $name attempt $i (timeout ${left}s): $*"
    timeout --kill-after=30 $left "$@" >> $OUT/logs/$name.log 2>&1
    rc=$?
    echo "KMOE $(date +%T) $name attempt $i rc=$rc"
    [ $rc = 0 ] && break
    [ $rc = 124 ] && break  # the budget ran out: the journal keeps what finished
    sleep 5
  done
  echo "KMOE section $name rc=$rc" >> $OUT/status.txt
}

# ---- phase 1: single-GPU sections, concurrently on the 4 GPUs (budget: until 75 min after start or deadline)
P1=$(( 75 * 60 ))
for spec in "0 experts uniform" "1 experts req4" "2 experts req8"; do
  set -- $spec
  ( export CUDA_VISIBLE_DEVICES=$1
    run_section experts-$3 $P1 8 python3 $B experts --out $OUT --routes $3 --tag=-$3 --case-timeout 1500 "${EXTRA[@]}" ) &
done
( export CUDA_VISIBLE_DEVICES=3
  run_section attnres $P1 8 python3 $B attnres --out $OUT --case-timeout 600 "${EXTRA[@]}" ) &
wait
python3 $B summary --out $OUT > $OUT/logs/summary-phase1.log 2>&1

# ---- phase 2: 4 real ranks + 12 emulated (torchrun), sequentially
export CUDA_VISIBLE_DEVICES=0,1,2,3
TR=(python3 -m torch.distributed.run --standalone --nproc-per-node 4)
run_section front 900 6 "${TR[@]}" $B front --q 4 --out $OUT --case-timeout 600 "${EXTRA[@]}"
run_section layer 1500 6 "${TR[@]}" $B layer --q 4 --out $OUT --case-timeout 900 "${EXTRA[@]}"
run_section tail 1200 6 "${TR[@]}" $B tail --out $OUT --case-timeout 600 "${EXTRA[@]}"
run_section oproj 1200 6 "${TR[@]}" $B oproj --out $OUT --case-timeout 600 "${EXTRA[@]}"

python3 $B summary --out $OUT > $OUT/logs/summary.log 2>&1
cat $OUT/status.txt
echo "KMOE end $(date)"
