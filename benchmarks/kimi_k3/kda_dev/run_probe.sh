#!/bin/bash
# P-KDA2 probe arms: PROBE_ 1 = compute skips VFULL wait; 2 = compute/epilogue skip QK waits; 4 = loader issues V without waiting QK. Timing only (numerics wrong).
cd /k31ov
python3 benchmarks/kimi_k3/bench_fip_dev.py --rounds 3 --lens 5461,5461,5462 8192,8192 16384 --variants "{}" "{\"PROBE_\": 1}" "{\"PROBE_\": 4}" "{\"PROBE_\": 2}" "{\"PROBE_\": 3}" "{\"PROBE_\": 7}" 2>&1 | grep -E "graph us|Error|error|Traceback" | tail -40
echo PR_DONE
