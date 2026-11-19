#!/bin/bash
# D-ATTN native sparse MLA microbench, several variants in one 1-GPU allocation.
# dattn_spm.sh <ov> <name> "<args variant 1>" "<args variant 2>" ...
# Results: $K/runs/D-ATTN/sp/<name>.<i>.json, log <name>.log. Run on the Rhea login node with setsid nohup.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OV=$1; NAME=$2; shift 2
OUT=$K/runs/D-ATTN/sp; mkdir -p $OUT
CMD="cd $K/ov/$OV"
i=0
for v in "$@"; do
  CMD="$CMD; echo VARIANT $i: $v; python3 benchmarks/kimi_k3/bench_dattn_sparse_native.py --out $OUT/$NAME.$i.json $v 2>&1 | grep -E '^R[0-9]|Error|error|Traceback|File ' | tail -12"
  i=$((i+1))
done
bash $K/tools/wsgpu.sh -g 1 -m 20 -o $OV -n D-ATTN -- bash -c "$CMD" > $OUT/$NAME.log 2>&1 < /dev/null
echo "END rc=$?" >> $OUT/$NAME.log
