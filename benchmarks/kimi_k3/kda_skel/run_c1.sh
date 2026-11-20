#!/bin/bash
# P-KDA2 C1 math calibration: skel2c (BT32 1 chain/SM, fip role layout: NPI prep instances, paced math fill)
# vs the fip kernel (314 us at F16K). usage: run_c1.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -Xptxas -v -o $O/skel2c $S/skel2c.cu -lcuda > $O/build.log 2>&1 || { echo BUILD_FAIL; grep -m8 error $O/build.log; exit 1; }
grep -E "spill|registers" $O/build.log | head -4
B=$O/skel2c
timeout 60 $B 0 1 5 232 0 2 | grep -E "RESULT|TIMEOUT|CU|CUDA|bad" || echo "SMOKE_FAIL rc=$?"
# args: mode pw npi steps flags iters pfill efill cfill
for r in 1 2; do
  for cfg in "1 1 0 0 0" "1 5 0 0 0" "1 5 31 6 0" "1 1 31 6 0" "2 4 39 6 0"; do
    set -- $cfg; timeout 120 $B 0 $1 $2 232 0 5 $3 $4 $5 | grep -E "RESULT|TIMEOUT"
  done
done
# fill sensitivity at the fip layout (striped) + die-local, ideal prep (16) and no MMA (8) attribution
for cfg in "1 5 15 6 0" "1 5 46 6 0" "1 5 31 12 0" "1 5 31 6 4"; do set -- $cfg; timeout 120 $B 0 $1 $2 232 0 5 $3 $4 $5 | grep -E "RESULT|TIMEOUT"; done
timeout 120 $B 1 1 5 232 0 5 31 6 0 | grep -E "RESULT|TIMEOUT"
for fl in 16 8 4; do timeout 120 $B 0 1 5 232 $fl 5 31 6 0 | grep -E "RESULT|TIMEOUT"; done
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
