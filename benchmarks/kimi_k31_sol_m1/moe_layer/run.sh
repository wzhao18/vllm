#!/bin/bash
# M1 moe-layer probes: build (sm_107a) + run (a) on GPU0 and GPU3 (2 process reps), (c)+(b) on GPU1, (d) on GPU2.
D=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/MOEM1; O=$D/out.$1; mkdir -p $O/bin
NV=/usr/local/cuda/bin/nvcc; $NV --version | tail -2
echo "host $(hostname) $(date) K31_SPIN_GUARD=$K31_SPIN_GUARD WS_OV=$WS_OV"
nvidia-smi --query-gpu=index,serial,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv,noheader
for p in m1a_stream m1c_handoff m1d_entry; do
  $NV -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/bin/$p $D/src/$p.cu -lcuda > $O/bin/$p.log 2>&1 || { echo "BUILD FAIL $p"; grep -m12 -i -B1 -A2 error $O/bin/$p.log; }
done
md5sum $O/bin/m1* | cut -c1-12
(CUDA_VISIBLE_DEVICES=0 timeout 600 $O/bin/m1a_stream > $O/a.gpu0.txt 2>&1; echo "a0 rc=$?" >> $O/a.gpu0.txt) &
(CUDA_VISIBLE_DEVICES=1 timeout 600 $O/bin/m1c_handoff > $O/c.gpu1.txt 2>&1; echo "c rc=$?" >> $O/c.gpu1.txt) &
(CUDA_VISIBLE_DEVICES=2 timeout 600 $O/bin/m1d_entry > $O/d.gpu2.txt 2>&1; echo "d rc=$?" >> $O/d.gpu2.txt) &
(CUDA_VISIBLE_DEVICES=3 timeout 600 $O/bin/m1a_stream > $O/a.gpu3.txt 2>&1; echo "a3 rc=$?" >> $O/a.gpu3.txt) &
wait
echo "== END $(date)"
