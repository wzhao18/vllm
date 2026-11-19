#!/bin/bash
# P-KDA2 M2' : skel4 (1 chain/SM, BT64, bf16 INP, 2 MMA issuers, TMA raw ring). usage: run_m2p.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -Xptxas -v -o $O/skel4 $S/skel4.cu -lcuda > $O/build4.log 2>&1 || { echo BUILD4_FAIL; grep -m8 error $O/build4.log; exit 1; }
grep -E "spill|registers" $O/build4.log | head -2
B=$O/skel4
timeout 60 $B 116 80 3 | grep -E "RESULT|TIMEOUT|CU|CUDA" || echo "SMOKE_FAIL rc=$?"
for r in 1 2; do for fl in 80 16 0; do timeout 90 $B 116 $fl 7 | grep -E "RESULT|TIMEOUT"; done; done
# attribution (ideal prep + HBM): 4 no pass, 8 no MMA, 2 no out; real prep with fill
for fl in 20 24 18 26 88; do timeout 90 $B 116 $fl 7 | grep -E "RESULT|TIMEOUT"; done
for pf in "68 16" "51 16"; do timeout 90 $B 116 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
