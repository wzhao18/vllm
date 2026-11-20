#!/bin/bash
# P-GEMM R4: prefetching glue kernels (gate_sigmoid_mul, latent rms_norm_cta, situ4) base (int12) vs this overlay
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/P-GEMM/glue3/$(date +%m%d%H%M%S); mkdir -p $O
B=$K/ov/int12t-9628d05-1
for arm in A B A2 B2; do
  if [ ${arm:0:1} = A ]; then D=$B; else D=/k31ov; fi
  echo "== $arm glue2"; (cd $D && PYTHONPATH=$D python3 /k31ov/benchmarks/kimi_k3/pgemm_glue2_bench.py $O/glue_$arm.json 2>&1 | grep -E "^\{|Error" | cut -c1-260 | tail -10)
  echo "== $arm situ"; (cd $D && PYTHONPATH=$D python3 /k31ov/benchmarks/kimi_k3/pgemm_situ_bench.py $O/situ_$arm.pt 2>&1 | grep -E "^\{|Error|error" | tail -16)
done
echo "== CMP"; python3 /k31ov/benchmarks/kimi_k3/pgemm_situ_bench.py --cmp $O/situ_A.pt $O/situ_B.pt
echo "== END $O"
