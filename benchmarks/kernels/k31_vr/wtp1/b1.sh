#!/bin/bash
# baseline at the DEP16 owner points with the current DEP recipe (W6 M1 balanced top-k + M5 drafter split, MXFP8, fp4 KV),
# DEP4 8-layer (2 DSA layers), per-rank CUPTI. usage: b1.sh <jobid> [prefix]
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; cd $W/wtp1/tools
export TAG=wtp1-${ENVT:-base1} K31W0_LAYERS=8 VLLM_K31_DSA_FUSED_DECODE_Q=${FQ:-0}
J=$1; P=${2:-b1}; R=$W/wtp1/results
Q=8 bash rr.sh $J 1 ${P}_q8 dep4 mxfp8 fp4 x16,28 --profile-dir $R/prof_${P}_q8
[ -n "$NOQ4" ] || Q=4 bash rr.sh $J 1 ${P}_q4 dep4 mxfp8 fp4 x16 --profile-dir $R/prof_${P}_q4
