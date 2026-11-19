#!/bin/bash
# D-ATTN native sparse MLA microbench: $1 = overlay tag, $2 = run name, rest = bench args.
# Run on the Rhea login node: setsid nohup bash dattn_sp.sh <ov> <name> [args] &
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OV=$1; NAME=$2; shift 2
OUT=$K/runs/D-ATTN/sp; mkdir -p $OUT
bash $K/tools/wsgpu.sh -g 1 -m 15 -o $OV -n D-ATTN -- \
  bash -c "cd $K/ov/$OV && python3 benchmarks/kimi_k3/bench_dattn_sparse_native.py --out $OUT/$NAME.json $*" \
  > $OUT/$NAME.log 2>&1 < /dev/null
echo "END rc=$?" >> $OUT/$NAME.log
