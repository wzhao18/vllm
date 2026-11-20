#!/bin/bash
# pmla_l1cap.sh <run dir name>: LDGSTS gather rate vs unused dynamic smem (L1 carveout), 1 GPU
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1; mkdir -p $R; D=$(cd "$(dirname "$0")" && pwd)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $R/l1cap $D/pmla_l1cap.cu -lcuda > $R/build.log 2>&1 || { grep -m10 error $R/build.log > $R/summary.txt; exit 1; }
timeout 600 $R/l1cap > $R/summary.txt 2>&1; echo "rc=$?" >> $R/summary.txt
