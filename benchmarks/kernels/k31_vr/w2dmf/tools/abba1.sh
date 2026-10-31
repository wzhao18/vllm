#!/bin/bash
# abba1.sh <job> <out> <cases> : DEP4 8-layer, NVFP4 projections (ledger build), base env vs w2dmf env (-loc / +loc), A B C C B A
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; export JOB=$1; O=$2; SET=${3:-x6,16,28}
E=${E:-a1}; export PROF=${PROF-1}
run() { bash $W/w2dmf/arms.sh $1 $O/$3 $SET "$2" ; }
for step in "base A:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0 a" "$E B:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0 b" "$E C:VLLM_K3_NVFP4_PROJ=1 c" \
            "$E C:VLLM_K3_NVFP4_PROJ=1 c2" "$E B:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0 b2" "base A:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0 a2"; do
  set -- $step; echo "#### $3 env=$1 arm=$2 $(date +%T)"; run $1 $2 $3
done
echo DONE $(date +%T)
