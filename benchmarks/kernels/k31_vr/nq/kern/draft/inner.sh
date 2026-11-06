#!/bin/bash
# inner.sh <out dir>: inside the nq e2e image on one node (4 GPUs), the drafter kernel benches of the overlay NQ_OV.
# Environment exactly as tools/nq_gputest_inner.sh (env/k31vr_env.sh at K31VR_VLLM_TAG=jh-242fb49cd4, overlay first on
# PYTHONPATH, VLLM_SSM_CONV_STATE_LAYOUT=DS, build dirs under $NQ_BUILD). Phases (hard timeouts; each bench also has a
# soft --budget so it ends with its tables written):
#   A  (55 min) GEMM chosen + sweep, one process per linear, 3 lanes on GPUs 0 / 2 / 3; tokenspeed single-rank core on GPU 1
#   B  (50 min) Markov fused vs stock loop, torchrun 4
#   C  (40 min) DraftDcp vs stock drafter DCP, torchrun 4
# then summarize.py -> SUMMARY.txt.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
OUT=$1
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export K31VR_VLLM_TAG=jh-242fb49cd4
source $W/env/k31vr_env.sh
export PYTHONPATH=$NQ_OV:${PYTHONPATH#*:}:$W/e2e/env/pydeps:$W/wide/tools:$W/merge/tools/shim
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
export K31_DCPX_BUILD_DIR=${NQ_BUILD:?NQ_BUILD}/dcpx K31_DSAC_BUILD_DIR=$NQ_BUILD/dsac
mkdir -p $OUT $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR
cd $NQ_OV
st() { echo "$(date +%T) $*" | tee -a $OUT/status.txt; }
st "host $(hostname): $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tr '\n' ' ')"
python3 -c "import vllm, os; print('vllm from', os.path.dirname(vllm.__file__))" 2>&1 | tee -a $OUT/status.txt
head -1 $NQ_OV/SOURCE_REV >> $OUT/status.txt
PA=${KD_PHASE_A:-3300}; PB=${KD_PHASE_B:-3000}; PC=${KD_PHASE_C:-2400}

# ---- phase A: GEMMs (3 lanes) + single-rank tokenspeed core
lane() {  # lane <gpu> <seconds> <linear...>
  local gpu=$1 secs=$2; shift 2
  local per=$(( secs / $# ))
  for lin in "$@"; do
    local t0=$(date +%s) rc=75 left
    # rc 75 (common.STICKY_EXIT): sticky CUDA error -> restart in a fresh process, which
    # resumes from $OUT/gemm/$lin/raw.csv and skips the crashed arm
    while [ $rc -eq 75 ]; do
      left=$(( per - ($(date +%s) - t0) ))
      [ $left -gt 120 ] || break
      CUDA_VISIBLE_DEVICES=$gpu timeout $(( left + 180 )) python3 $D/bench_draft_gemm.py --out $OUT/gemm/$lin \
        --linears $lin --budget $(( left - 60 )) >> $OUT/gemm_$lin.log 2>&1
      rc=$?
      [ $rc -eq 75 ] && st "phaseA gemm $lin gpu$gpu sticky CUDA error: restart (resume)"
    done
    st "phaseA gemm $lin gpu$gpu rc=$rc secs=$(( $(date +%s) - t0 ))"
  done
}
st "phase A start"
# resume state is per task run: an earlier run's GEMM results are set aside (summarize.py skips *.prev.*)
if [ -d $OUT/gemm ]; then
  PREV=prev.$(date +%Y%m%d-%H%M%S); mv $OUT/gemm $OUT/gemm.$PREV
  for f in $OUT/gemm_*.log; do [ -f "$f" ] && mv "$f" "$f.$PREV"; done
  st "moved an earlier run's gemm results to gemm.$PREV"
fi
SA=$(( PA - 300 ))
lane 0 $SA fused_qkv_a ctx_shard q_b_old &
lane 2 $SA gate_up context_kv o_proj &
lane 3 $SA lm_head down q_b o_proj_old &
( t0=$(date +%s); CUDA_VISIBLE_DEVICES=1 timeout $(( SA + 120 )) python3 $D/bench_draft_dcp.py --single --out $OUT/dcp_single \
    --budget $(( SA - 600 )) > $OUT/dcp_single.log 2>&1
  st "phaseA dcp_single gpu1 rc=$? secs=$(( $(date +%s) - t0 ))" ) &
wait
st "phase A end"

# ---- phase B: Markov (torchrun 4)
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout $PB python3 -m torch.distributed.run --nproc-per-node 4 --master-port 29531 \
  $D/bench_draft_markov.py --out $OUT/markov --budget $(( PB - 600 )) > $OUT/markov.log 2>&1
st "phaseB markov rc=$? secs=$(( $(date +%s) - t0 ))"
pkill -f bench_draft_markov.py 2>/dev/null; sleep 5

# ---- phase C: DraftDcp (torchrun 4)
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout $PC python3 -m torch.distributed.run --nproc-per-node 4 --master-port 29532 \
  $D/bench_draft_dcp.py --out $OUT/dcp --budget $(( PC - 480 )) > $OUT/dcp.log 2>&1
st "phaseC dcp rc=$? secs=$(( $(date +%s) - t0 ))"
pkill -f bench_draft_dcp.py 2>/dev/null

python3 $D/summarize.py $OUT > $OUT/SUMMARY.txt 2>&1
# the per-bench WORSE blocks (stdout tables) are in the logs; SUMMARY.txt merges every CSV
st "done"
