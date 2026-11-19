#!/bin/bash
# step_inner.sh <out prefix> [extra bench args]: runs INSIDE the K3.1 image via wsgpu.sh -> wsenv.sh (PYTHONPATH =
# overlay, Lustre caches, CUDA_VISIBLE_DEVICES = the locked GPUs, NQ_BUILD per overlay). Reads <out>.cfg.sh
# (cfg2bench.py) and <out>.shapes.json (nq_shapes.py) and launches NQ_WORLD local ranks of nq_step.py (one per locked
# GPU; RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* set here, so nq_step.py's SLURM rendezvous is skipped).
# env: NQ_EMULATE_EP (expert-count emulation, default 32), NQ_ITERS, NQ_SAMPLES, NQ_JIT_WARMUP=1 (keep the server JIT warmups).
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OUT=$1; shift
source $OUT.cfg.sh
[ -n "$FLASHINFER_JIT_DIR" ] && export FLASHINFER_WORKSPACE_BASE=$(dirname $FLASHINFER_JIT_DIR)
mkdir -p $NQ_BUILD/dcpx $NQ_BUILD/dsac 2>/dev/null
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
[ "$NG" = "$NQ_WORLD" ] || { echo "step_inner: $NG GPUs locked, config needs $NQ_WORLD ranks"; exit 2; }
echo "NQ host $(hostname) gpus $CUDA_VISIBLE_DEVICES overlay $WS_OV build $NQ_BUILD"
echo "NQ env $(env | grep -E '^(VLLM_|NCCL_|K31_|K3_)' | sort | tr '\n' ' ')"
echo "NQ args ${NQ_ARGS[*]}"
nvidia-smi --query-gpu=index,memory.used,clocks.mem --format=csv,noheader | tr '\n' ' '; echo
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + RANDOM % 20000)) WORLD_SIZE=$NQ_WORLD
cd /tmp
PIDS=()
for ((r = 0; r < NQ_WORLD; r++)); do
  RANK=$r LOCAL_RANK=$r python3 $D/nq_step.py --capture-sizes "$NQ_CAP" --max-capture $NQ_MAXCAP \
    --full-capture-sizes "$NQ_FULLCAP" --measured-rows "$NQ_ROWS" --overlay "${WS_OVDIR:-$K/ov/$WS_OV}" \
    --emulate-ep ${NQ_EMULATE_EP:-32} ${NQ_JIT_WARMUP:+--jit-warmup} -- \
    "${NQ_ARGS[@]}" --shapes $OUT.shapes.json --kv-len-jitter 0 --routing balanced --seed 0 \
    --iters ${NQ_ITERS:-20} --samples ${NQ_SAMPLES:-5} --output-json $OUT.json "$@" > $OUT.rank$r.log 2>&1 &
  PIDS+=($!)
done
rc=0
# first failure kills the rest (a dead rank leaves the others hanging in a collective)
while :; do
  alive=0
  for i in "${!PIDS[@]}"; do
    p=${PIDS[$i]}; [ -z "$p" ] && continue
    if kill -0 $p 2>/dev/null; then alive=1; continue; fi
    wait $p; c=$?; PIDS[$i]=""
    if [ $c != 0 ]; then rc=$c; echo "NQ rank $i exit $c: killing the others"; for q in "${PIDS[@]}"; do [ -n "$q" ] && kill $q 2>/dev/null; done; fi
  done
  (( alive )) || break
  sleep 1
done
cat $OUT.rank0.log
for ((r = 1; r < NQ_WORLD; r++)); do grep -a "Traceback\|Error" $OUT.rank$r.log | head -5 | sed "s/^/rank$r: /"; done
exit $rc
