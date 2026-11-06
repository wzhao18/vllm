#!/bin/bash
# run_inner.sh <out dir> <budget s>: inside the round-4 image on one node, the kda kernel microbench on its 4 GPUs.
# Env = nq_gputest_inner.sh's (env/k31vr_env.sh with K31VR_VLLM_TAG=jh-242fb49cd4, the overlay NQ_OV first on
# PYTHONPATH in place of the base env, DS conv layout, extension builds under NQ_BUILD).
#   GPU0: section a (KdaHead vs pair vs stock, iso + chain), then the NQ-TUNE 3 early-tile sweep q1=* and verify=*
#         (one process per value, own CuTe cache dir under $NQ_BUILD: the swept constants may not change the key)
#   GPU1: section b shard 0/3 (verify kernels vs stock S1)
#   GPU2: section b shard 1/3, then the kdasplit1834 arm alone in its own CuTe cache dir (same KdaSplit specs as the
#         kdasplit arm, only the schedule differs: a shared cache could hand it the kdasplit binary)
#   GPU3: section c (in-projection cuBLAS / cuBLASLt / split-K), section b shard 2/3, then the multi=* sweep values
# A process that hits a sticky CUDA error records it on the case and exits 86; it is restarted (up to 3 times) and
# resumes after that case. Every process stops starting cases CASE_MARGIN s before the budget ends; then merge.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
NB=$W/nqbench
KD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; BUDGET=${2:-9000}
T0=$(date +%s); HARD_END=$((T0 + BUDGET)); CASE_END=$((HARD_END - ${CASE_MARGIN:-900}))
export K31VR_VLLM_TAG=jh-242fb49cd4
source $W/env/k31vr_env.sh
export PYTHONPATH=${NQ_OV:?NQ_OV}:${PYTHONPATH#*:}:$W/e2e/env/pydeps:$W/wide/tools:$W/merge/tools/shim
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
export K31_DCPX_BUILD_DIR=${NQ_BUILD:?NQ_BUILD}/dcpx K31_DSAC_BUILD_DIR=$NQ_BUILD/dsac
export TORCH_EXTENSIONS_DIR=$NQ_BUILD/nqk-kda-ext
mkdir -p $OUT $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR $TORCH_EXTENSIONS_DIR
cd /tmp
echo "NQK host $(hostname) start $(date) budget ${BUDGET}s overlay $NQ_OV ($(head -1 $NQ_OV/SOURCE_REV 2>/dev/null))"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
python3 -c "import vllm, os; print('NQK vllm from', os.path.dirname(vllm.__file__))"
: > $OUT/status.txt

run() {  # run <gpu> <tag> <bench args...>: skipped when fewer than 5 min of case time remain; rc 86 (sticky CUDA
         # error, the case is recorded) restarts the process, which resumes after the recorded cases
  local gpu=$1 tag=$2; shift 2
  local start=$(date +%s) now rc attempt
  for attempt in 0 1 2 3; do
    now=$(date +%s)
    if [ $((CASE_END - now)) -lt 300 ]; then echo "NQK $tag SKIPPED (budget)" | tee -a $OUT/status.txt; return; fi
    CUDA_VISIBLE_DEVICES=$gpu timeout $((HARD_END - now)) python3 $KD/bench_kda.py run --out $OUT --tag $tag \
      --deadline-s $((CASE_END - now)) "$@" >> $OUT/$tag.log 2>&1
    rc=$?
    echo "NQK $tag gpu$gpu attempt $attempt rc=$rc secs=$(( $(date +%s) - start ))" | tee -a $OUT/status.txt
    [ $rc -eq 86 ] || return
  done
}

sweep() {  # sweep <gpu> <q1=N|verify=N|multi=N>: one early-tile value in its own CuTe cache dir (inside $NQ_BUILD)
  local t=${2/=/}
  mkdir -p $NQ_BUILD/cutedsl-early-$t
  CUTE_DSL_CACHE_DIR=$NQ_BUILD/cutedsl-early-$t run $1 a-early-$t --sections a --modes chain --heads-only --early $2
}

(
  run 0 a-main --sections a --modes iso,chain
  for spec in verify=18 verify=24 q1=13 q1=21 verify=15 q1=9; do sweep 0 $spec; done  # nearest-first priority
) &
run 1 b-s0 --sections b --shard 0/3 --skip-arms kdasplit1834 &
(
  run 2 b-s1 --sections b --shard 1/3 --skip-arms kdasplit1834
  mkdir -p $NQ_BUILD/cutedsl-1834
  CUTE_DSL_CACHE_DIR=$NQ_BUILD/cutedsl-1834 run 2 b-1834 --sections b --arms kdasplit1834
) &
(
  run 3 c --sections c
  run 3 b-s2 --sections b --shard 2/3 --skip-arms kdasplit1834
  for spec in multi=18 multi=24 multi=15; do sweep 3 $spec; done
) &
wait
python3 $KD/bench_kda.py merge --out $OUT > $OUT/merge.log 2>&1
echo "NQK merge rc=$? end $(date)" | tee -a $OUT/status.txt
tail -n 60 $OUT/merge.log
