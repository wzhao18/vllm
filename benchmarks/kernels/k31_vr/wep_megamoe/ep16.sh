#!/bin/bash
# EP16 MegaMoE on 4 nodes (16 ranks, 56 experts/rank = the production EP16 shape): stock vs K3.1 header, +-localize, bitwise check.
# usage: ep16.sh <jobid> <prefix>
J=$1; P=${2:-ep16}
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; T=$W/wepmoe/tools; R=$W/wepmoe/results
H="--header deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh=$W/wepmoe/hdr/v5/prod.cuh"
A="--experts-per-rank 56 --tokens 4,8,48,64,128,224 --routing balanced,random --samples 5 --cupti 1"
cd $T
./run16.sh $J 4 ${P}_stock $A --save-y $R/${P}_ref
./run16.sh $J 4 ${P}_best $A $H --ref-y $R/${P}_ref
./run16.sh $J 4 ${P}_locstock $A --localize 1
./run16.sh $J 4 ${P}_locbest $A $H --localize 1
./run16.sh $J 4 ${P}_stock2 $A
./run16.sh $J 4 ${P}_best2 $A $H
