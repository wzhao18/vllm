#!/bin/bash
# pf_run1.sh <jobid> <hosts (4, comma)> <out prefix> <shapes.json> [-- extra bench args]
# gap512: one 16-rank process (4 nodes x 4 GPUs, srun --overlap in allocation <jobid>) of the C512 PREFILL role
# (pf_cfg.py: nq-c512-xdr prefill args/env, DP4 x TP4=DCP4 ShardQ, EP16 MegaMoE, MXFP8 MoE act) through the dsrel
# nq harness (nq_inner.sh -> nq_step.py -> vllm bench forward), CUDA-graph replay timing, optional --profile-dir.
# Modeled on dsrel/tools/nq_run1.sh (same image, mounts, overlay/build env).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; T=$W/dsrel/tools; G=$W/gap512
REL=$W/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626
J=$1; NL=$2; OUT=$3; SHAPES=$4; shift 4
[ "$1" = "--" ] && shift
IMG=${NQ_IMG:-$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh}
export NQ_OV NQ_BUILD=${NQ_BUILD:?NQ_BUILD}
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
mkdir -p $(dirname $OUT); rm -f $OUT.rdv $OUT.json
NQ_DRAFT_BACKEND=FLASH_ATTN python3 $G/pf_cfg.py $REL > $OUT.cfg.sh || { echo "pf_cfg failed"; exit 2; }
cp $SHAPES $OUT.shapes.json
export ENROOT_CACHE_PATH=$K/enroot-cache
{
  echo "PFRUN point=c512-prefill shapes=$SHAPES extra=$(echo "$*" | tr ' ' '_') ov=${NQ_OV:-base} build=$NQ_BUILD img=$(basename $IMG) job=$J nodes=$NL start=$(date +%FT%T)"
  grep '^#' $OUT.cfg.sh
} > $OUT.log
timeout ${NQ_TIMEOUT:-4200} srun --overlap --jobid $J -N4 -w $NL --ntasks-per-node=4 --kill-on-bad-exit=1 \
  --container-image=$IMG --container-name=${NQ_CNAME:-k31nq-step} --no-container-mount-home \
  --container-mounts=$W:$W,$K:$K,$M:$M bash $T/nq_inner.sh $OUT "$@" >> $OUT.log 2>&1
rc=$?
echo "NQEND rc=$rc end=$(date +%FT%T)" >> $OUT.log
echo "$(basename $OUT) rc=$rc"
grep -a "^\[forward\].*median\|^NQ: KV\|Traceback\|Error:" $OUT.log | sed 's/, sample spread.*//' | head -40
exit $rc
