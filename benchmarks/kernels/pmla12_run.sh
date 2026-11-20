#!/bin/bash
# pmla12_run.sh <outdir> [probe] [check] [skel]: P-MLA-2CTA round 4 (1 GPU)
#   probe: per-warp cp.async issue law (pmla12_gather.cu)
#   check: qko (stage landing + proxy fence before the S-free / O-ready waits) bitwise + F16K ABBA vs qko 9
#   skel:  empty protocol (no gather/math/MMA) and MMA-only, qko vs qko 9 (timing only)
OUT=$1; shift; WHAT=${*:-probe check skel}
D=$(cd "$(dirname "$0")" && pwd); OV=$(cd "$D/../.." && pwd); mkdir -p $OUT
cd $OV
for w in $WHAT; do
  case $w in
    probe) bash $D/pmla12_gather.sh $OUT/probe | grep -E "W[ 0-9]+ D|rc=" ;;
    check) python3 $D/pmla10_check.py 0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:9 2>&1 | grep -E "equal|median" ;;
    skel) python3 $D/pmla10_check.py --nocheck --logical-only 0:1:1:1 0:1:1:1:0:0:0:0:0:0:0:0:0:0:0:0:9 \
            0:1:1:0 0:1:1:0:0:0:0:0:0:0:0:0:0:0:0:0:9 2>&1 | grep -E "median" ;;
    *) python3 $D/pmla10_check.py --nocheck $w 2>&1 | grep -E "median" ;;
  esac
done 2>&1 | tee $OUT/log
