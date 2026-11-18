#!/bin/bash
# KDA prefill chain: unit test + per-kernel microbench of stock / split / fused at
# the deployed DEP16 shapes (F16K = 3 x 5.4K, F8K = 3 x 2.7K; 96 heads), then ncu
# of the recurrence (K2) and prepare (K1) at F16K.
# usage (inside wsgpu.sh): run_kda_prefill_ab.sh <overlay dir> <out dir>
set -x
OV=$1; OUT=$2; mkdir -p $OUT
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
NCU=$(command -v ncu); [ -z "$NCU" ] && NCU=$(find /usr/local /opt -name ncu -type f 2>/dev/null | head -1)
echo "ncu=$NCU"
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv
python3 -m pytest -q --noconftest -p no:cacheprovider $OV/tests/models/kimi_k3/test_kda_prefill_fused_conv.py 2>&1 | tail -30
for impl in stock split fusedconv fused; do
  python3 $B --impl $impl --tokens 16384 --json $OUT/f16k_$impl.json 2>&1 | grep -v Warning
  python3 $B --impl $impl --tokens 8192 --json $OUT/f8k_$impl.json 2>&1 | grep -v Warning
done
for pin in 1 2 3 4 5 6 7; do
  python3 $B --impl split --k2-pin $pin --tokens 16384 --json $OUT/f16k_split_pin$pin.json 2>&1 | grep -v Warning | head -4
done
if [ -n "$NCU" ]; then
  $NCU --target-processes all -k regex:"recurrence|prepare|qkv_conv|rmsnorm" -c 4 \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section Occupancy \
    --section WarpStateStats --section MemoryWorkloadAnalysis --section LaunchStats \
    --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum \
    -o $OUT/ncu_f16k_split -f python3 $B --impl split --tokens 16384 --iters 1 > $OUT/ncu.log 2>&1
  $NCU -i $OUT/ncu_f16k_split.ncu-rep --page details > $OUT/ncu_f16k_split.txt 2>&1
  $NCU -i $OUT/ncu_f16k_split.ncu-rep --page raw --csv > $OUT/ncu_f16k_split_raw.csv 2>&1
fi
