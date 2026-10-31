#!/bin/bash
# abba2.sh <job> <out> <cases> : C B D D B C on env a3 (NVFP4 projections). B = side stream only, C = + localize, D = C + router on a high-priority stream
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; export JOB=$1; O=$2; SET=${3:-x6,16,28}; E=${E:-a3}; export PROF=${PROF-1}
C=C:VLLM_K3_NVFP4_PROJ=1; B=B:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0; D=D:VLLM_K3_NVFP4_PROJ=1,K31_W2DMF_HP_ROUTER=1
for step in "$C c" "$B b" "$D d" "$D d2" "$B b2" "$C c2"; do
  set -- $step; echo "#### $2 arm=$1 $(date +%T)"; bash $W/w2dmf/arms.sh $E $O/$2 $SET "$1"
done
echo DONE $(date +%T)
