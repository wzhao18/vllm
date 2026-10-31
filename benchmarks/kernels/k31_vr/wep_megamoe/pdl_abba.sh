#!/bin/bash
# Stock vs K3.1 header A/B with PDL on (production: DeepGEMM PDL is on for SM100-family parts) and off,
# interleaved ABBA per PDL setting on one held node, plus a bitwise check at BLOCK_M > 64 / zero-token experts.
# usage: pdl_abba.sh <jobid> <prefix> <header .cuh>
J=$1; P=${2:-m3}; HDR=$3
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; R=$W/wepmoe/results
H="--header deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh=$HDR"
A="--experts-per-rank 56,28 --tokens 4,8,48,128,224 --routing balanced,random --samples 5 --cupti 1"
cd $W/wepmoe/tools
for pdl in 1 0; do
  ./run1.sh $J ${P}_p${pdl}_stockA $A --pdl $pdl > /dev/null
  ./run1.sh $J ${P}_p${pdl}_prodB $A --pdl $pdl $H > /dev/null
  ./run1.sh $J ${P}_p${pdl}_prodB2 $A --pdl $pdl $H > /dev/null
  ./run1.sh $J ${P}_p${pdl}_stockA2 $A --pdl $pdl > /dev/null
done
C="--experts-per-rank 56,28 --tokens 4,128,224 --routing balanced,random --samples 2 --pdl 1"
./run1.sh $J ${P}chk_stock $C --save-y $R/${P}chk_ref > /dev/null
./run1.sh $J ${P}chk_prod $C $H --ref-y $R/${P}chk_ref > /dev/null
for f in ${P}_p1_stockA ${P}_p1_prodB ${P}_p1_prodB2 ${P}_p1_stockA2 ${P}_p0_stockA ${P}_p0_prodB ${P}_p0_prodB2 ${P}_p0_stockA2 ${P}chk_prod; do
  echo "== $f"; grep -E "^E/rank|check:|rror|imeout" ../logs/$f.log | cut -c1-160 | head -40
done
echo PDL_ABBA_DONE
