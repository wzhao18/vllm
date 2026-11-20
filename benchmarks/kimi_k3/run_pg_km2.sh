#!/bin/bash
# P-GEMM KM2' (01:15 audit item 2): is the sm_107 block-scaled base operand-pipeline bound? M16384 K7168, shared
# gate_up (N12288) and KDA in_proj Q|K|V (N36864): mainloop-only (noepi) vs loads-only (noepi+nomma: same TMA ring
# and barriers, no MMA / no SF copies) vs L2-hot reloads (hot), 512x256 2-CTA tiles, clusters 2x4 (A multicast) / 2x1.
cd /k31ov
F='^\{|Error|error'
T='[[[512,256],[2,4],256,4,"m",0,0,"noepi"],
    [[512,256],[2,4],256,4,"m",0,0,"noepi+nomma"],
    [[512,256],[2,4],256,4,"m",0,0,"noepi+hot"],
    [[512,256],[2,4],256,4,"m",0,0,"noepi+nomma+hot"],
    [[512,256],[2,1],256,4,"m",0,0,"noepi"],
    [[512,256],[2,1],256,4,"m",0,0,"noepi+nomma"]]'
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader | head -2
for NS in 12288:172 36864:516; do
  N=${NS%:*}; SOL=${NS#*:}
  echo "== N $N (tensor-bound $SOL us at 16.8 PF)"
  timeout 900 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel --ms 16384 --n $N --sol-us $SOL \
    --tactics "$T" 2>&1 | grep -E "$F" | cut -c1-420
done
echo "== END"
