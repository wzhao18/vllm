#!/bin/bash
# W6 launcher: one srun task per GPU. usage: launch.sh <jobid> <nnodes> <env tag> <log> <bench forward args...>
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models
J=$1; NN=$2; TAG=$3; LOG=$4; shift 4
export ENROOT_CACHE_PATH=$K/enroot-cache
mkdir -p $(dirname $LOG); RDV=$LOG.rdv; rm -f $RDV
echo "[launch] job $J nn $NN tag $TAG $(date) args: $*" > $LOG
srun --overlap --jobid $J -N $NN --ntasks-per-node=4 ${W6_NODELIST:+-w $W6_NODELIST} --kill-on-bad-exit=1 \
  --container-image=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh --container-name=${W6_CNAME:-k31w6} \
  --no-container-mount-home --container-mounts=$W:$W,$M:$M \
  bash -c "export K31VR_VLLM_TAG=$TAG; exec bash $W/w6/tools/inner.sh $RDV \"\$@\"" _ "$@" >> $LOG 2>&1
rc=$?; echo "[launch] rc=$rc $(date)" >> $LOG; exit $rc
