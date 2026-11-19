#!/bin/bash
# D-ATTN: dump the keys-on-M sparse MLA PTX/cubin, ptxas -v and SASS local-memory sites.
# $1 = overlay tag, $2 = run name, rest = bench args. Run on the Rhea login node.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OV=$1; NAME=$2; shift 2
OUT=$K/runs/D-ATTN/sp/$NAME.dump; mkdir -p $OUT
bash $K/tools/wsgpu.sh -g 1 -m 15 -o $OV -n D-ATTN -- bash -c "
cd $K/ov/$OV
export CUTE_DSL_KEEP=ptx,cubin CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$OUT
cd $OUT && python3 $K/ov/$OV/benchmarks/kimi_k3/bench_dattn_sparse_native.py --rows 384 --no-trtllm --check 2 $*
ls -la $OUT | head -20
for f in \$(ls $OUT/*.ptx 2>/dev/null | xargs -r grep -l SparseMlaDecodeKM); do
  echo == ptxas \$f; /usr/local/cuda/bin/ptxas -arch=sm_107a -v \$f -o $OUT/km.cubin 2>&1 | tail -5
done
C=\$(ls $OUT/*.cubin 2>/dev/null | head -1); [ -f $OUT/km.cubin ] && C=$OUT/km.cubin
/usr/local/cuda/bin/cuobjdump -sass \$C > $OUT/km.sass 2>&1
echo LDL \$(grep -c LDL $OUT/km.sass) STL \$(grep -c STL $OUT/km.sass) MUFU \$(grep -c MUFU $OUT/km.sass) lines \$(wc -l < $OUT/km.sass)
" > $K/runs/D-ATTN/sp/$NAME.log 2>&1 < /dev/null
echo "END rc=$?" >> $K/runs/D-ATTN/sp/$NAME.log
