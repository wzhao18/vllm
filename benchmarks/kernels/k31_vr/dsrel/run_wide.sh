#!/bin/bash
# run_wide.sh <outdir>: bench_wide_gemm.py with cuBLAS, then one process per config
# (a failing config's abort does not end the sweep).
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
O=$1; mkdir -p $O
python3 $D/bench_wide_gemm.py --configs none > $O/cublas.csv 2> $O/cublas.err
for c in $(seq 0 $((${NCFG:-11} - 1))); do
  python3 $D/bench_wide_gemm.py --configs $c > $O/c$c.csv 2> $O/c$c.err
  echo "config $c rc=$?" >> $O/rc.txt
done
