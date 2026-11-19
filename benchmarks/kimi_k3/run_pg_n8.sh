#!/bin/bash
# P-GEMM: sustained timing + SM clock/power (nvidia-smi) for cuBLASLt vs CuTe base; SASS opcode mix vs nvjet.
cd /k31ov
B="python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode clk --ms 16384 --tactic"
for T in '[[512,256],[2,1],256,4,"n",0,0,""]' '[[512,256],[2,1],256,4,"n",0,0,"hot+noepi"]'; do
  timeout 600 $B "$T" 2>&1 | grep -E '^\{|Error' | cut -c1-400
done
export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=/tmp/pg8_dump; rm -rf /tmp/pg8_dump; mkdir -p /tmp/pg8_dump
timeout 600 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel --ms 8192 --tactic '[[512,256],[2,1],256,4,"n",0,0,""]' > /dev/null 2>&1
P=$(ls -S /tmp/pg8_dump/*.ptx 2>/dev/null | head -1); echo "== ours: $P"
if [ -n "$P" ]; then
  ptxas -arch=sm_107a -v "$P" -o /tmp/pg8.cubin 2>&1 | grep -E 'registers|spill' | head -3
  nvdisasm /tmp/pg8.cubin > /tmp/pg8.sass 2>/dev/null
  grep -oE '^\s+/\*[0-9a-f]+\*/\s+[A-Z0-9_.]+' /tmp/pg8.sass | awk '{print $2}' | sed 's/\..*//' | sort | uniq -c | sort -rn | grep -E 'UTC|SYNCS|UBLKCP|UTMA|BAR|R2UR|ELECT|WARPSYNC|BRA' | head -20
fi
L=$(python3 -c "import glob,os;import nvidia.cublas as c;print(sorted(glob.glob(os.path.join(list(c.__path__)[0],'lib','libcublasLt.so*')))[0])" 2>/dev/null)
[ -z "$L" ] && L=$(ldconfig -p | grep -m1 libcublasLt.so | awk '{print $NF}')
echo "== nvjet from $L"
F=$(cuobjdump -lelf "$L" 2>/dev/null | head -0; echo nvjet_sm107_qqtst_256x256_128x6_2x4f_2cta_x2_Breuse_h_bz_Avec32UE8M0_Bvec32UE8M0_TNT)
timeout 600 cuobjdump -sass -fun "$F" "$L" > /tmp/nvj.sass 2>/dev/null
wc -l /tmp/nvj.sass
grep -oE '^\s+/\*[0-9a-f]+\*/\s+[A-Z0-9_.]+' /tmp/nvj.sass | awk '{print $2}' | sed 's/\..*//' | sort | uniq -c | sort -rn | grep -E 'UTC|SYNCS|UBLKCP|UTMA|BAR|R2UR|ELECT|WARPSYNC|BRA' | head -20
