#!/bin/bash
# D-KDA M3 diag: KdaRing per-kernel split (prep vs ring) via nsys, the non-MX build error, bitwise test.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-ring2
echo "== ring non-mx"
timeout 600 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local 2>&1 | grep -vE "^W1119|deprecated" | tail -6 | cut -c1-300
echo "== nsys mx n32 n96"
for n in 32 96; do
  timeout 600 nsys profile -o /tmp/dkda-ring2-$n --force-overwrite true -t cuda python3 $B --n $n --kinds flow ring --gap 1 --pending 4 --place local --mx > /dev/null 2>&1
  nsys stats -q -r cuda_gpu_kern_sum -f csv /tmp/dkda-ring2-$n.nsys-rep 2>/dev/null | awk -F, 'NR>1{gsub(/"/,""); printf "n'$n' %8.2f us avg x%-6s %s\n", $4/1000, $3, substr($9,1,90)}' | sort -k2 -rn | head -6
done
echo "== ptxas/regs"
ls $CUTE_DSL_CACHE_DIR 2>/dev/null | head -3
echo "== test ring bitwise"
timeout 900 python3 -m pytest --noconftest -q -rf -x $T -k "ring_is_bitwise and 33" 2>&1 | grep -E "passed|failed|^FAILED|Error|Mismatch|assert" | cut -c1-240 | head -8
