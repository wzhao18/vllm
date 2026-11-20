#!/bin/bash
# D-ATTN SASS audit for the QueryUK race pattern (LDS consumed only after an mbarrier
# arrive) over the CuTe DSL kernels the given tree compiles in its GPU unit tests.
# usage: dattn_sass_audit.sh <out dir> <overlay tag> [test files...]   (1 GPU)
O=$1; TAG=$2; shift 2
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
D=$K/ov/$TAG
mkdir -p $O/dump $O/cc
export CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$O/dump VLLM_KIMI_K3_CUTE_CACHE_DIR=$O/cc
cd $D
export PYTHONPATH=$D:$PYTHONPATH
timeout 300 python3 /k31ov/benchmarks/kimi_k3/dattn_quk_probe.py sass --impl new --rows 512 \
  --out $O --max 0 > $O/quk.log 2>&1
echo "query_uk sass rc=$?"
for t in "$@"; do
  timeout 600 python3 -m pytest -q -x -p no:cacheprovider --noconftest $t > $O/$(basename $t .py).log 2>&1
  echo "$(basename $t) rc=$? $(grep -E '[0-9]+ (passed|failed|skipped)' $O/$(basename $t .py).log | tail -1)"
done
n=$(ls $O/dump/*.cubin 2>/dev/null | wc -l)
echo "cubins: $n"
python3 /k31ov/benchmarks/kimi_k3/sass_lds_release_scan.py $O/dump/*.cubin > $O/scan.txt 2>&1
grep -E '^== .*: [1-9]|SCAN TOTAL' $O/scan.txt | cut -c1-200
