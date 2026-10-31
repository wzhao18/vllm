#!/bin/bash
# usage (on Hecate login): run16.sh <jobid> <nodes> <tag> <bench args...>
#   bench_megamoe.py --srun 1 on every GPU of <nodes> nodes of the held allocation (EP = 4 x nodes, real NVLink peers)
J=$1; N=$2; T=$3; shift 3
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
export ENROOT_CACHE_PATH=$K/enroot-cache
R=$W/wepmoe/results; mkdir -p $R $W/wepmoe/logs
export MASTER_ADDR=$(scontrol show hostnames $(squeue -h -j $J -o %N) | head -1) MASTER_PORT=$((29600 + RANDOM % 200))
srun --overlap --jobid $J -N$N --ntasks-per-node=4 --container-image=$IMG --container-name=wepmoe --no-container-mount-home \
  --container-mounts=$W:$W bash -c "source $W/env/k31vr_env.sh; export PYTHONPATH=\${WEPMOE_DG_PATH:+\$WEPMOE_DG_PATH:}\$PYTHONPATH; \
  [ \$SLURM_PROCID = 0 ] && nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu --format=csv,noheader; \
  cd $W/wepmoe/tools && python3 bench_megamoe.py --srun 1 --json $R/$T.json $*" > $W/wepmoe/logs/$T.log 2>&1
grep -E "^E/rank|check:|rror|imeout" $W/wepmoe/logs/$T.log | tail -40
