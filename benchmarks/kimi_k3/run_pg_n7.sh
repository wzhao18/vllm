#!/bin/bash
# P-GEMM skeleton: hot-load probe, staged epilogue, multicast clusters.
cd /k31ov
B="python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel --ms 16384,8192 --tactic"
F='^\{|Error|error'
for T in \
  '[[512,256],[2,1],256,4,"n",0,0,"noepi"]' \
  '[[512,256],[2,1],256,4,"n",0,0,"hot+noepi"]' \
  '[[512,256],[2,1],256,4,"n",0,0,"sf2+noepi"]' \
  '[[512,256],[2,1],256,4,"n",0,0,"sf2+hot+noepi"]' \
  '[[512,256],[2,1],256,4,"n",0,0,"sf2"]' \
  '[[512,256],[4,1],256,4,"n",0,0,"staged"]' \
  '[[512,256],[2,1],256,4,"n",0,0,"sf2+staged"]' \
  '[[512,256],[2,4],256,4,"n",0,0,"noepi"]' \
  ; do timeout 600 $B "$T" 2>&1 | grep -E "$F" | grep -v '"cublaslt"' | sed 's/"tb": \[.*\(".\{0,300\}"\]\)/"tb": \1/' | cut -c1-600; done
