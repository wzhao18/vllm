#!/bin/bash
# P-MLA run 77: h6 (unroll=1) full sstamp timeline (all roles), then no math+no MMA skeleton.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run77
mkdir -p $R
timeout 600 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs 0:0:0:0:1 --json $R/stamp.json > $R/stamp.log 2>&1
echo "rc=$?" >> $R/stamp.log
timeout 400 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs 1:0:1:1 --json $R/skel.json > $R/skel.log 2>&1
echo "rc=$?" >> $R/skel.log
echo DONE > $R/DONE
