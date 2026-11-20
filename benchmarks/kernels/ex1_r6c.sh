#!/bin/bash
# GEMM-EX1 R6 (09:01 directive): in-step stamp analysis of step_d st6a + standalone A/C arms with first-wave stamps
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; OV=$K/ov/gex1; OUT=$K/runs/GEMM-EX1/res/r6c; mkdir -p $OUT; date
echo "== in-step stamps (since $1)"; python3 $OV/benchmarks/kernels/ex1_st2_ana.py $K/runs/GEMM-EX1/stamps $1 2>&1 | tail -12
B="python3 $OV/benchmarks/kernels/ex1_bench_r6.py --nvml"
A=own:256x192:2x1:1:256:1:4:s6; C=own:256x192:2x2:1:256:1:4:s6
timeout 500 $B --rows 384 --arms $A,$A:d1,$A:d4,$A:d16 --out $OUT/a.json > $OUT/a.log 2>&1
timeout 500 $B --rows 384 --arms $C,$C:d1,$C:d4,$C:d16 --out $OUT/c.json > $OUT/c.log 2>&1
for g in a c; do grep -E "^\{|rror" $OUT/$g.log | cut -c1-420; done; date
