#!/bin/bash
# P-MLA run 73: h6 loader loop unroll=1 (no LLVM 2x unroll): SASS gather count, tests, F16K base vs gather-stamped.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run73
mkdir -p $R
export PATH=$PATH:/usr/local/cuda/bin
d=$R/dump_0; mkdir -p $d
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$d timeout 600 python3 benchmarks/kernels/pmla_ptxas.py $d 255 h6 0 > $d/ptxas.log 2>&1
for f in $d/*.cubin; do cuobjdump -sass $f > $d/h6.sass 2>&1; break; done
echo "gather4 SASS: $(grep -c GATHER4 $d/h6.sass)" > $R/summary.txt
timeout 300 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h6" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" >> $R/summary.txt
timeout 900 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --h4-knobs 0,0:0:0:0:1:0:8,0 --json $R/f16k.json > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/f16k.log
echo DONE > $R/DONE
