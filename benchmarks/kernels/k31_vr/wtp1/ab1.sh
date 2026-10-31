#!/bin/bash
# A/B of the wep-tp1-attn levers on the DEP4 8-layer owner-point bench (q8 r16/r28, MXFP8 + fp4 KV, W6 M1+M5
# recipe). A = levers off (VLLM_TOKENSPEED_SM107_SPLIT=0), B = branch defaults. Arm order = $ARMS, default ABBA
# ("A B B2 A2"); the first booked A/B (ab3n1/ab3n2) ran ARMS="A B" per node, with an A/A control (ab2 offA/offB).
# usage: [ARMS="A B"] [PFX=..] ab1.sh <jobid> <envtag> (never re-push <envtag> while it runs)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; cd $W/wtp1/tools
export TAG=wtp1-$2 K31W0_LAYERS=8 VLLM_K31_DSA_FUSED_DECODE_Q=${FQ:-0}
J=$1; P=${PFX:-ab1}
for arm in ${ARMS:-A B B2 A2}; do
  case $arm in A*) export VLLM_TOKENSPEED_SM107_SPLIT=0;; B*) unset VLLM_TOKENSPEED_SM107_SPLIT;; esac
  Q=8 bash rr.sh $J 1 ${P}_${arm}_q8 dep4 mxfp8 fp4 x16,28 --profile-dir $W/wtp1/results/prof_${P}_${arm}_q8
done
