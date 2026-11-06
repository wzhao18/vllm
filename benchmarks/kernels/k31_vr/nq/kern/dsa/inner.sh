#!/bin/bash
# inner.sh <stage core|value|front|summary> <results dir> <deadline epoch s>: one stage of the nq DSA kernel benches
# inside the nq e2e image on one node (4 GPUs), the environment of nq_gputest_inner.sh (env/k31vr_env.sh on the base env
# vllm-jh-242fb49cd4 = FlashInfer 0.7.3.dev20261031 and its JIT dirs, the overlay NQ_OV first on PYTHONPATH, DS conv
# layout, extension builds under NQ_BUILD). Each bench process runs under a watchdog (no new record or log line for
# NQK_STALL s, default 900: killed) and is resumed (its JSONL skips finished keys and records the key it hung on) until
# it finishes, the deadline passes, or two attempts in a row add nothing.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
STAGE=$1; R=$2; DEADLINE=$3
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export K31VR_VLLM_TAG=jh-242fb49cd4
source $W/env/k31vr_env.sh
export PYTHONPATH=$NQ_OV:${PYTHONPATH#*:}:$W/e2e/env/pydeps:$W/wide/tools:$W/merge/tools/shim
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
export K31_DCPX_BUILD_DIR=${NQ_BUILD:?NQ_BUILD}/dcpx K31_DSAC_BUILD_DIR=$NQ_BUILD/dsac
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p $R $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR
cd $NQ_OV
STALL=${NQK_STALL:-900}

lines() { cat "$@" 2>/dev/null | wc -l; }

# watched <log> <jsonl> <deadline> cmd...: 0 done, 124 deadline, 125 stalled, else the command's rc
watched() {
  local log=$1 out=$2 dl=$3; shift 3
  setsid "$@" >> $log 2>&1 &
  local pid=$! last=$(date +%s) seen=-1 now n
  while kill -0 $pid 2>/dev/null; do
    sleep 15; now=$(date +%s)
    n=$(( $(lines $out) + $(lines $log) ))
    [ $n != $seen ] && { seen=$n; last=$now; }
    if [ $now -ge $dl ] || [ $(( now - last )) -ge $STALL ]; then
      echo "NQK $(date +%T) killing $* ($([ $now -ge $dl ] && echo deadline || echo "stalled ${STALL}s"))" | tee -a $log
      kill -TERM -$pid 2>/dev/null; sleep 10; kill -KILL -$pid 2>/dev/null; wait $pid 2>/dev/null
      [ $now -ge $dl ] && return 124 || return 125
    fi
  done
  wait $pid
}

# resumed <tag> <jsonl> <deadline> cmd...: watched attempts until done / deadline / two empty attempts
resumed() {
  local tag=$1 out=$2 dl=$3; shift 3
  local log=$R/$tag.log empty=0 a=1 rc before
  while [ $(date +%s) -lt $(( dl - 60 )) ]; do
    before=$(lines $out)
    echo "NQK $(date +%T) $tag attempt $a: $*" | tee -a $log
    watched $log $out $dl "$@"; rc=$?
    echo "NQK $(date +%T) $tag attempt $a rc=$rc records $before -> $(lines $out)" | tee -a $log
    [ $rc = 0 ] && grep -q "bench done" $log && return 0
    [ $rc = 124 ] && return 124
    if [ $(lines $out) = $before ]; then empty=$((empty + 1)); else empty=0; fi
    [ $empty -ge 2 ] && { echo "NQK $tag: two attempts without a record, giving up" | tee -a $log; return 1; }
    a=$((a + 1))
  done
  return 124
}

echo "NQK host $(hostname) stage $STAGE until $(date -d @$DEADLINE +%T 2>/dev/null || echo $DEADLINE):" \
  "$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tr '\n' ' ')"
python3 -c "import vllm, os; print('NQK vllm from', os.path.dirname(vllm.__file__))"
TR="python3 -m torch.distributed.run --standalone --nproc-per-node 4"
case $STAGE in
  core)  # 4 single-GPU processes at once: small rows q4 / q8, wide rows q4 / q8
    mkdir -p $R/core; i=0
    # The DsaCore extension (nvcc, minutes, silent) once, before the watched processes.
    CUDA_VISIBLE_DEVICES=0 timeout 1800 python3 -c "from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops \
import load_ext; load_ext(); print('NQK DsaCore extension ready')" >> $R/core-build.log 2>&1 \
      || echo "NQK DsaCore extension build failed (dsac arms will fail): see core-build.log"
    for spec in "small 4" "small 8" "wide 4" "wide 8"; do
      set -- $spec
      ( export CUDA_VISIBLE_DEVICES=$i
        resumed core-$1-q$2 $R/core/$1-q$2.jsonl $DEADLINE \
          python3 $D/bench_core.py --out $R/core/$1-q$2.jsonl --part $1 --qs $2 ${NQK_CORE_ARGS:-} ) &
      i=$((i + 1))
    done
    wait ;;
  value)
    mkdir -p $R/value
    resumed value $R/value/value.jsonl $DEADLINE \
      $TR $D/bench_value.py --out $R/value/value.jsonl ${NQK_VALUE_ARGS:-} ;;
  front)
    mkdir -p $R/front
    resumed front $R/front/front.jsonl $DEADLINE \
      $TR $D/bench_front.py --out $R/front/front.jsonl ${NQK_FRONT_ARGS:-} ;;
  summary)
    python3 $D/summarize.py $R/core $R/value $R/front --csv $R/summary.csv --arms $R/arms.csv \
      > $R/SUMMARY.txt 2>&1; cat $R/SUMMARY.txt ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
