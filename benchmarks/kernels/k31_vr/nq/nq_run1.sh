#!/bin/bash
# nq_run1.sh <jobid> <nodelist (4 hosts)> <out prefix> <point> <stock|ls> <old|new|drafter dir> <rows> [ctx] [-- extra bench args]
# One arm process of the nq step A/B: TP16=DCP16 on 4 nodes x 4 GPUs (srun --overlap inside allocation <jobid>), the
# decode role of jh/noquant/cfg/nq-<point>-xdr.yaml (cfg2bench.py), verify rows <rows> (B*q, comma list) at contexts
# [ctx] (default agentx; nq_shapes.py). Writes <out>.log / .json / .cfg.sh / .shapes.json; prints the [forward] lines.
# env: NQ_OV (overlay root, '' = base kimi_k3), NQ_BUILD (default $NB/cache/build/<overlay basename|base>),
#      NQ_IMG (default: the nq e2e image), NQ_TIMEOUT (s, default 2700), NQ_CNAME (pyxis container name).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J=$1; NL=$2; OUT=$3; POINT=$4; ARM=$5; DRAFT=$6; ROWS=$7; shift 7
CTX=agentx; if [ $# -gt 0 ] && [ "$1" != "--" ]; then CTX=$1; shift; fi
[ "$1" = "--" ] && shift
# the nq e2e image: the configs' container key vllm/vllm-openai:cu134-nightly-3f1bcfd82bb2... resolves to this sqsh in
# jh/srtslurm-jh.yaml and e2e/k31r/srtslurm.yaml (the jh round-4 renders log 'Container: ...3f1bcfd82b-aarch64.sqsh');
# the DEP configs (jh/dep/srtslurm-dep.yaml) use internal-vr-mum-92ccd9647f instead
IMG=${NQ_IMG:-$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh}
OVTAG=base; [ -n "$NQ_OV" ] && OVTAG=$(basename $NQ_OV)
export NQ_OV NQ_BUILD=${NQ_BUILD:-$NB/cache/build/$OVTAG}
# runner tasks inherit the block's 12-node geometry (oracle/tools/step_run.sh:20)
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
mkdir -p $(dirname $OUT); rm -f $OUT.rdv $OUT.json
python3 $D/cfg2bench.py $POINT --arm $ARM --drafter $DRAFT --rows $ROWS > $OUT.cfg.sh || { echo "cfg2bench failed"; exit 2; }
Q=$(sed -n 's/^NQ_Q=//p' $OUT.cfg.sh)
CAPS="prodlimit=$(sed -n 's/^NQ_PRODLIMIT=//p' $OUT.cfg.sh) maxcap=$(sed -n 's/^NQ_MAXCAP=//p' $OUT.cfg.sh) mns=$(sed -n 's/^NQ_MNS=//p' $OUT.cfg.sh)"
python3 $D/nq_shapes.py $Q $OUT.shapes.json --rows $ROWS --ctx $CTX --point $POINT > $OUT.shapes.txt || exit 2
export ENROOT_CACHE_PATH=$K/enroot-cache
{
  echo "NQRUN point=$POINT arm=$ARM drafter=$DRAFT q=$Q ctx=$CTX rows=$ROWS $CAPS extra=$(echo "$*" | tr ' ' '_') ov=${NQ_OV:-base} img=$(basename $IMG) job=$J nodes=$NL start=$(date +%FT%T)"
  cat $OUT.shapes.txt; grep '^#' $OUT.cfg.sh
} > $OUT.log
timeout ${NQ_TIMEOUT:-2700} srun --overlap --jobid $J -N4 -w $NL --ntasks-per-node=4 --kill-on-bad-exit=1 \
  --container-image=$IMG --container-name=${NQ_CNAME:-k31nq-step} --no-container-mount-home \
  --container-mounts=$W:$W,$K:$K,$M:$M bash $D/nq_inner.sh $OUT "$@" >> $OUT.log 2>&1
rc=$?
echo "NQEND rc=$rc end=$(date +%FT%T)" >> $OUT.log
echo "$(basename $OUT) rc=$rc"
grep -a "^\[forward\].*median\|^NQ: specialized\|^NQ: drafter\|Traceback\|Error:" $OUT.log | sed 's/, sample spread.*//' | head -40
exit $rc
