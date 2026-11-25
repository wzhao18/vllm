#!/bin/bash
# m1_run.sh <outdir> <which> <reps>: login side, run inside a claimed 1-node hold by sol16-n1run.sh (srun = step of the hold).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
O=${1:?outdir}; mkdir -p $O; S=$K/sol16/M1BC/src
set -a; . $K/sol16/prod/SOL16_MEASURE.env; set +a
export WS_OV=$(cat $K/sol16/BASE_OV)
echo "m1_run $(date +%FT%T) job=$SLURM_JOB_ID WS_OV=$WS_OV out=$O which=$2 reps=$3"
srun -N1 -n1 --overlap --container-image=$IMG --container-mounts=/lustre:/lustre --no-container-mount-home --container-workdir=/tmp \
  bash $K/sol16/tools/sol16-wsenv.sh bash $S/m1_inner.sh $O ${2:-akcd} ${3:-3}
echo "=== rc=$? $(date +%T)"
