#!/bin/bash
# step_rank.sh <out prefix> [extra bench args]: one rank (srun task, 4 per node) of a multi-node step bench, inside the
# K3.1 image via wsenv.sh (srun ... bash wsenv.sh bash step_rank.sh ...). File rendezvous on <out>.rdv (nq_step.py).
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OUT=$1; shift
source $OUT.cfg.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p $NQ_BUILD/dcpx $NQ_BUILD/dsac 2>/dev/null
if [ "$SLURM_PROCID" = 0 ]; then
  echo "NQ rank0 $(hostname) ntasks $SLURM_NTASKS overlay $WS_OV build $NQ_BUILD"
  echo "NQ env $(env | grep -E '^(VLLM_|NCCL_|K31_|K3_)' | sort | tr '\n' ' ')"
  echo "NQ args ${NQ_ARGS[*]}"
fi
[ "$SLURM_LOCALID" = 0 ] && echo "NQ node $(hostname) gpus: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
cd /tmp
exec python3 $D/nq_step.py --init-file $OUT.rdv --capture-sizes "$NQ_CAP" --max-capture $NQ_MAXCAP \
  --full-capture-sizes "$NQ_FULLCAP" --measured-rows "$NQ_ROWS" --overlay "${WS_OVDIR:-$K/ov/$WS_OV}" \
  --emulate-ep ${NQ_EMULATE_EP:-0} -- \
  "${NQ_ARGS[@]}" --shapes $OUT.shapes.json --kv-len-jitter 0 --routing ${NQ_ROUTING:-balanced} --seed 0 \
  --iters ${NQ_ITERS:-20} --samples ${NQ_SAMPLES:-5} --output-json $OUT.json "$@"
