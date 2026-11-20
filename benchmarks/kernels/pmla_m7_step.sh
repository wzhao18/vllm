#!/bin/bash
# P-MLA M7 in-step ABBA (Hecate login node, setsid nohup), gated on run134 bitwise.
# A = int12 (int12t-9628d05-1), B = int12 + M7 (RoPE LDG warps). Profiled F16K NVFP4 slice.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
A=int12t-9628d05-1; B=$1; P=$2; R=$K/runs/P-MLA/run134
for i in $(seq 1 360); do grep -q DONE $R/summary.txt 2>/dev/null && break; sleep 20; done
if ! grep -q "T=16384 bitwise=True" $R/summary.txt || grep -q "bitwise=False" $R/summary.txt; then echo "M7AB gate failed"; exit 1; fi
run() { bash $K/tools/step_p.sh -n P-MLA -q nvfp4 -m 40 -t $2 $1 1; echo "M7AB $2 rc=$? $(date +%T)"; }
run $A $P-a1; run $B $P-b1; run $B $P-b2; run $A $P-a2
echo "M7AB done $(date +%T)"
