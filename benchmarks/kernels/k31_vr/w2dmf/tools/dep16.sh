#!/bin/bash
# dep16.sh <job> <envtag> <out> : full DEP16 (4 nodes, one NVL72 block) A C C A (+B), q8 AgentX r6/16/28, NVFP4 projections.
# A = base env (dac64f593, localize off), B = <envtag> side stream only, C = <envtag> default (side stream + localize)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; J=$1; E=$2; O=$3
LOCK=$W/w2dmf/results/step/$O.LOCK; mkdir $LOCK 2>/dev/null || { echo "already ran/running: $LOCK"; exit 0; }
for v in $(env | grep -o '^SLURM_[A-Z_]*' ); do [ $v = SLURM_JOB_ID ] || unset $v; done
export JOB=$J NN=4 LAY=dep16 PROF=1
A=A:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0; B=B:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=0; C=C:VLLM_K3_NVFP4_PROJ=1,VLLM_K3_MEGAMOE_LOCALIZE=1
for step in "base $A a" "$E $C c" "$E $B b" "$E $C c2" "base $A a2"; do
  set -- $step; echo "#### $3 env=$1 arm=$2 $(date +%T)"; bash $W/w2dmf/arms.sh $1 $O/$3 x6,16,28 "$2"
done
echo DONE $(date +%T)
