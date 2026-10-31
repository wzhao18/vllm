#!/bin/bash
# usage (on Hecate login): run1.sh <jobid> <tag> <bench args...>   -- runs bench_megamoe.py on 4 GPUs of the held node
J=$1; T=$2; shift 2
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
export ENROOT_CACHE_PATH=$K/enroot-cache
R=$W/wepmoe/results; mkdir -p $R
srun --overlap --jobid $J -N1 -n1 --container-image=$IMG --container-name=wepmoe --no-container-mount-home \
  --container-mounts=$W:$W bash -c "source $W/env/k31vr_env.sh; export PYTHONPATH=\${WEPMOE_DG_PATH:+\$WEPMOE_DG_PATH:}\$PYTHONPATH; \
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu --format=csv,noheader; \
  cd $W/wepmoe/tools && python3 bench_megamoe.py --json $R/$T.json $*" > $W/wepmoe/logs/$T.log 2>&1
tail -40 $W/wepmoe/logs/$T.log
