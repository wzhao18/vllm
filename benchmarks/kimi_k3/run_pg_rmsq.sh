#!/bin/bash
# P-GEMM RMSQ: rms_norm_quantize multi-row: base (int12) vs this overlay, bitwise + timing
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/P-GEMM/rmsq/$(date +%m%d%H%M%S); mkdir -p $O
B=$K/ov/int12t-9628d05-1
echo "== BASE"; (cd $B && PYTHONPATH=$B python3 /k31ov/benchmarks/kimi_k3/pgemm_rmsq_bench.py $O/a.pt 2>&1 | grep -E "^\{|Error|error" | tail -12)
echo "== NEW"; (cd /k31ov && python3 benchmarks/kimi_k3/pgemm_rmsq_bench.py $O/b.pt 2>&1 | grep -E "^\{|Error|error" | tail -12)
echo "== BASE2"; (cd $B && PYTHONPATH=$B python3 /k31ov/benchmarks/kimi_k3/pgemm_rmsq_bench.py $O/a2.pt 2>&1 | grep -E "^\{" | head -4)
echo "== CMP"; python3 /k31ov/benchmarks/kimi_k3/pgemm_rmsq_bench.py --cmp $O/a.pt $O/b.pt
echo "== END $O"
