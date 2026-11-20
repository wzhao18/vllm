#!/bin/bash
for b in 96 32; do python3 /k31ov/benchmarks/kernels/k31_vr/dsrel/bench_vp_markov_keys.py --batch $b 2>&1 | grep -E "^B|Error|error" | head -20; done
