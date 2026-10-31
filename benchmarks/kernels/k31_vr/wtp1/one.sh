#!/bin/bash
# usage: one.sh <jobid> <gpu list e.g. 0 or 0,1> <log> <tag> <cmd...>   one task in the K3 VR image, cwd = wtp1/src-<tag>
J=$1; G=$2; L=$3; TAGV=$4; shift 4
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
export ENROOT_CACHE_PATH=$K/enroot-cache
srun --overlap --jobid $J -N 1 -n 1 --container-image=$IMG --container-name=k31wtp1 --no-container-mount-home \
  --container-mounts=$W:$W bash -c "export K31VR_VLLM_TAG=wtp1-$TAGV; source $W/env/k31vr_env.sh; export CUDA_VISIBLE_DEVICES=$G; cd $W/wtp1/src-$TAGV; $*" > $L 2>&1
rc=$?; echo "rc=$rc" >> $L; tail -40 $L; exit $rc
