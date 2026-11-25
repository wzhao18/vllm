#!/bin/bash
# probe (a) interleaved-weights arms: GPU0/1 design ring, GPU2/3 today's ring, cudaMalloc weights
D=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/MOEM1; O=$D/out.$1; mkdir -p $O/bin
NV=/usr/local/cuda/bin/nvcc
echo "host $(hostname) $(date) K31_SPIN_GUARD=$K31_SPIN_GUARD WS_OV=$WS_OV"
$NV -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/bin/m1a_stream $D/src/m1a_stream.cu -lcuda > $O/bin/a.log 2>&1 || { echo BUILD FAIL; grep -i -m5 error $O/bin/a.log; }
md5sum $O/bin/m1a_stream | cut -c1-12
for g in 0 1 2 3; do arm=$([ $g -lt 2 ] && echo 0 || echo 3); (CUDA_VISIBLE_DEVICES=$g timeout 600 $O/bin/m1a_stream $arm il > $O/a.il.gpu$g.txt 2>&1; echo "rc=$?" >> $O/a.il.gpu$g.txt) & done; wait
echo "== END $(date)"
