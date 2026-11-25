#!/bin/bash
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
set -a; . $K/sol16/prod/SOL16_MEASURE.env; set +a
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
OV=$(cat $K/sol16/BASE_OV); TAG=${M1TAG:-r1}
echo "ov $OV node $(hostname) $(date +%T)"
timeout 1200 srun -N1 -n1 --container-image=$IMG --container-mounts=/lustre:/lustre,$K/ov/$OV:/k31ov --no-container-mount-home \
  --container-workdir=/tmp --export=ALL,WS_OV=$OV bash $K/sol16/tools/sol16-wsenv.sh bash $K/sol16/MOEM1/run.sh $TAG
echo "=== rc=$? $(date +%T)"
