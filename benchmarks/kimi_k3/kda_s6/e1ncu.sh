#!/bin/bash
# P-KDA2-S6 r9 E1: same-node ncu (full set, lineinfo, clock-control none) of kvar variants at F16K, then per-range
# stall table (stall9.py, absolute samples) and the top long_sb / barrier / no_inst lines of the chain + prep ranges.
#   bash e1ncu.sh <tag> <var1.py> [var2.py ...]   (ranges/<var>.txt must exist)
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
VD=/k31ov/benchmarks/kimi_k3/kda_s6
TAG=$1; shift
R=$K/runs/P-KDA2-S6/ncu9-$TAG; mkdir -p $R
NCU=$K/tools/nsight/bin/ncu; [ -x $NCU ] || NCU=ncu
export CUTE_DSL_LINEINFO=1
for v in "$@"; do
  nm=${v%.py}
  export CUTE_DSL_CACHE_DIR=$R/dsl-$nm
  echo "== $nm ncu start $(date +%T)"
  timeout 1500 $NCU --set full --import-source yes --clock-control none -k regex:_pkd -c 1 -f -o $R/$nm \
    python3 $VD/fip_var.py $VD/kvar/$v --shapes f16k --time none --tag ncu > $R/$nm.app.log 2>&1
  echo "ncu rc=$?"; grep -E "==ERROR|Error|Traceback" $R/$nm.app.log | head -3
  $NCU -i $R/$nm.ncu-repz --page source --csv --print-source cuda,sass > $R/$nm.cs.csv 2>/dev/null
  $NCU -i $R/$nm.ncu-repz --page details --csv > $R/$nm.details.csv 2>/dev/null
  grep -E '"Duration"|"Executed Ipc Active"|"Issue Slots Busy"|"SM Frequency"|"Executed Instructions"' $R/$nm.details.csv \
    | awk -F'","' '{print $(NF-2), $NF}' | sed 's/"//g' | head -8
  python3 $VD/stall9.py $R/$nm.cs.csv $VD/ranges/$nm.txt $v
  for rg in COMPUTE EPILOGUE MMA_LOAD; do python3 $VD/stall9.py $R/$nm.cs.csv $VD/ranges/$nm.txt $v $rg long_sb 6 | sed -n '/top lines/,$p'; done
  for rg in prep_setup p0_rawTMA p1p2_beta_gate p5_gram p7_restore; do python3 $VD/stall9.py $R/$nm.cs.csv $VD/ranges/$nm.txt $v $rg long_sb 4 | sed -n '/top lines/,$p'; done
done
echo "E1 done $(date +%T)"
