#!/bin/bash
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; OV=$K/ov/gex1; OUT=$K/runs/GEMM-EX1/res/r6e; mkdir -p $OUT; date
B="python3 $OV/benchmarks/kernels/ex1_bench_r6.py --reps 25"
A=own:256x192:2x1:1:256:1:4; C=own:256x192:2x2:1:256:1:4; D=own:256x256:2x1:1:256:1:4
timeout 500 $B --rows 384 --arms $A:s6:d4,$A:d4,$C:d4,$C:s6:d4,$C,$C:d1 --out $OUT/a.json > $OUT/a.log 2>&1
timeout 500 $B --rows 512 --arms $D:d4,$D --out $OUT/b.json > $OUT/b.log 2>&1
for g in a b; do grep -E "^\{|rror" $OUT/$g.log | cut -c1-470; done; date
