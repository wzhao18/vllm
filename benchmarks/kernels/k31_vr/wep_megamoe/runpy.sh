#!/bin/bash
# usage: runpy.sh <jobid> <cmd...>  -- run a short command inside the K3 VR image on the held node (no GPU timing)
J=$1; shift
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
export ENROOT_CACHE_PATH=$K/enroot-cache
srun --overlap --jobid $J -N1 -n1 --container-image=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh \
  --container-name=wepmoe --no-container-mount-home --container-mounts=$W:$W \
  bash -c "source $W/env/k31vr_env.sh; cd $W/wepmoe/tools; $*" 2>&1 | grep -v "^USDT\|deprecated"
