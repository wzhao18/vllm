#!/bin/bash
# m1_inner.sh <outdir> <which> <reps>: container side (via sol16-wsenv.sh). Builds and runs the boundary-chain M1 probes.
O=${1:?outdir}; W=${2:-akcd}; REPS=${3:-3}; S=$(dirname $(readlink -f $0)); cd $O
NV=/usr/local/cuda/bin/nvcc
echo "host $(hostname -s) $(date +%T) WS_OV=$WS_OV K31_SPIN_GUARD=$K31_SPIN_GUARD"; $NV --version | tail -1
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>/dev/null
for f in m1_single m1_round; do
  [ -x $O/$f ] && [ $O/$f -nt $S/$f.cu ] && continue
  $NV -O3 -std=c++17 -gencode arch=compute_100f,code=sm_100f -o $O/$f $S/$f.cu -lcuda > $O/$f.build.log 2>&1 || { cat $O/$f.build.log; exit 1; }
  /usr/local/cuda/bin/cuobjdump -sass $O/$f > $O/$f.sass 2>/dev/null
done
echo "built $(date +%T)"
if [[ $W == *[akcd]* ]]; then CUDA_VISIBLE_DEVICES=0 timeout 900 $O/m1_single $(echo $W | tr -cd akcd) $REPS; echo "m1_single rc=$?"; fi
if [[ $W == *b* ]]; then CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 900 $O/m1_round $REPS; echo "m1_round rc=$?"; fi
