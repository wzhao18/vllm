#!/bin/bash
# D-KDA M3 diag: KdaRing per-kernel split (prep vs ring) via nsys, the non-MX build error, bitwise test.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-ring2
echo "== nsys mx n32 n96"
for n in 32 96; do
  timeout 600 nsys profile -o /tmp/dkda-ring2-$n --force-overwrite true -t cuda python3 $B --n $n --kinds flow ring --gap 1 --pending 4 --place local --mx > /dev/null 2>&1
  nsys stats -q -r cuda_gpu_kern_sum -f csv /tmp/dkda-ring2-$n.nsys-rep 2>/dev/null | awk -F, 'NR>1{gsub(/"/,""); printf "n'$n' %8.2f us avg x%-6s %s\n", $4/1000, $3, (index($9,"prep")>0 ? "PREP" : "CORE") " " substr($9,60,40)}' | grep cutlass | head -8
done
