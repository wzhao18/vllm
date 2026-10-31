#!/bin/bash
# arms.sh <envtag> <outtag> <cases e.g. x6,16,28> <arm>... [-- extra bench args]
# arm = name:VAR=v,VAR=v (env for that arm; name:- = none). DEP4 (or LAY=dep16 NN=4) step, current DEP recipe
# (MXFP8 targets + drafter, fp4 KV + MXFP4 idx, prims_ts, balanced top-k, split 16-32), Q (default 8) AgentX, CUPTI per arm.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; J=${JOB:?JOB}
TAGE=$1; O=$2; SET=$3; shift 3
ARMS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARMS+=("$1"); shift; done; [ "$1" = "--" ] && shift
export TAG=w2dmf-$TAGE Q=${Q:-8}; if [ "${LAY:-dep4}" = dep4 ]; then export K31W0_LAYERS=${K31W0_LAYERS-8}; else unset K31W0_LAYERS; fi
mkdir -p $W/w2dmf/results/step/$O
i=0
for a in "${ARMS[@]}"; do
  i=$((i+1)); name=${a%%:*}; envs=${a#*:}; N=$O/$i-$name
  echo "== arm $i $name [$envs] $(date +%T)"
  ( if [ "$envs" != "-" ]; then IFS=,; for kv in $envs; do export "$kv"; done; unset IFS; fi
    bash $W/w2dmf/step/run.sh $J ${NN:-1} $N ${LAY:-dep4} mxfp8 fp4 $SET ${PROF:+--profile-dir $W/w2dmf/results/step/$N.prof} "$@" ) | grep "median\|rror\|KV cache\|Model loading" | head -12
done
