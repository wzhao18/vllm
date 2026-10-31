#!/bin/bash
# usage: launch.sh <jobid> <nnodes> <log> <bench forward args...>   (env D4_NODELIST optional)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models
J=$1; NN=$2; LOG=$3; shift 3
export ENROOT_CACHE_PATH=$K/enroot-cache
IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
mkdir -p $(dirname $LOG); RDV=$LOG.rdv; rm -f $RDV
echo "[launch] job $J nn $NN $(date) TAG=$TAG D4_PTS=$D4_PTS DCPX=$DCPX args: $*" > $LOG
# create the named container on every node first (fresh nodes: import can outlast the NCCL rendezvous)
srun --overlap --jobid $J -N $NN --ntasks-per-node=1 ${D4_NODELIST:+-w $D4_NODELIST} --container-image=$IMG \
  --container-name=k31wtp1 --no-container-mount-home --container-mounts=$W:$W,$M:$M true >> $LOG 2>&1
srun --overlap --jobid $J -N $NN --ntasks-per-node=4 ${D4_NODELIST:+-w $D4_NODELIST} --kill-on-bad-exit=1 \
  --container-image=$IMG --container-name=k31wtp1 \
  --no-container-mount-home --container-mounts=$W:$W,$M:$M \
  bash $W/wtp1/tools/inner.sh $RDV "$@" >> $LOG 2>&1
rc=$?; echo "[launch] rc=$rc $(date)" >> $LOG
grep -h "^\[forward\]\|GPU KV cache size\|Available KV cache\|Model loading took\|DcpExchange on\|Error\|error:" $LOG | sort -u | head -40
exit $rc
