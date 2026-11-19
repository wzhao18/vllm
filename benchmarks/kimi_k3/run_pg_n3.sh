#!/bin/bash
# P-GEMM skeleton decomposition: mainloop-only (noepi) vs early TMEM release vs stock epilogue; L2 raster.
cd /k31ov
B="python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel --ms 16384,8192 --tactic"
F='^\{|Error|error'
for T in \
  '[[512,256],[4,1],256,8,"n",0,0,""]' \
  '[[512,256],[4,1],256,8,"n",0,0,"early"]' \
  '[[512,256],[4,1],256,8,"n",0,0,"noepi"]' \
  '[[512,256],[2,1],256,8,"n",0,0,"noepi"]' \
  '[[256,256],[2,1],256,8,"m",0,0,"noepi"]' \
  '[[512,256],[4,1],256,1,"n",0,0,"early"]' \
  '[[512,256],[4,1],256,4,"n",0,0,"early"]' \
  '[[512,256],[4,1],256,16,"n",0,0,"early"]' \
  '[[512,256],[4,1],256,8,"m",0,0,"early"]' \
  ; do timeout 600 $B "$T" 2>&1 | grep -E "$F" | grep -v '"cublaslt"' | cut -c1-260; done
timeout 600 $B '[[256,256],[2,1],256,8,"n",1,1]' --ms 16384 2>&1 | grep -o '"tb".*' | cut -c1-1200
