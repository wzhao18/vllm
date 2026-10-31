#!/bin/bash
# usage: arms.sh <jobid> <prefix> <hdrdir> "<arm list>" <bench args...>   (arm "stock" = no header)
J=$1; P=$2; D=$3; A=$4; shift 4
cd /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/wepmoe/tools
for v in $A; do
  if [ $v = stock ]; then H=""; else H="--header deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh=$D/$v.cuh"; fi
  ./run1.sh $J ${P}_$v "$@" $H > /dev/null
  echo "== $v"; grep -E "^E/rank|check:|rror|imeout" ../logs/${P}_$v.log | cut -c1-150 | head -40
done
