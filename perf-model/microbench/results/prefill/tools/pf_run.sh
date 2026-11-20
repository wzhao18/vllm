#!/bin/bash
# mb lane prefill launcher (runs inside a runner2 task): one srun step, 4 ranks per node, on the task's own hosts.
# usage: pf_run.sh <hosts comma list> <nnodes> <out prefix> <bench forward args...>
# writes <out>.log (all ranks), <out>.json (bench --output-json is passed by the caller)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models
IMG=${PF_IMG:-$K/images/k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-mr4b8c04e28d7_d3c468ef795-kmm7989055e66_fi0.7.3.dev20261031_mc0.3.14_base3f1bcfd82b_r1-aarch64.sqsh}
HOSTS=$1; NN=$2; OUT=$3; shift 3
export ENROOT_CACHE_PATH=$K/enroot-cache
mkdir -p $(dirname $OUT); rm -f $OUT.rdv
echo "PFRUN job $SLURM_JOB_ID hosts $HOSTS nn $NN img $(basename $IMG) tree ${PF_TREE:-rel-1767cf7} layers ${K31W0_LAYERS:-all} moe ${PF_MOE_QUANT:-mxfp8} start $(date +%FT%T) args: $*" > $OUT.log
env $(env | grep -o "^SLURM_[A-Z_]*" | grep -v "^SLURM_CONF$\|^SLURM_JOB_ID$" | sed "s/^/-u /") \
  timeout ${PF_TIMEOUT:-3600} srun --overlap --jobid $SLURM_JOB_ID -w $HOSTS -N $NN -n $((NN*4)) --ntasks-per-node=4 \
  --kill-on-bad-exit=1 --container-image=$IMG --container-name=${PF_CNAME:-k31pf} --no-container-mount-home \
  --container-mounts=$W:$W,$M:$M bash $W/mb/prefill/tools/inner.sh $OUT.rdv "$@" >> $OUT.log 2>&1
rc=$?
echo "PFEND rc=$rc end=$(date +%FT%T)" >> $OUT.log
echo "$(basename $OUT) rc=$rc"
grep -a "^\[forward\]\|Rubin MegaMoE\|^PF:\|Traceback\|Error" $OUT.log | grep -v "^\s*$" | head -30
exit $rc
