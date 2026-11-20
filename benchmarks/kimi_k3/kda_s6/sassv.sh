#!/bin/bash
# Compile kernel.py variants (f16k route) with the CuTe DSL cubin dump; report regs / spills / SASS size.
# env: TAG (out dir suffix), VARS ("name=file ..." file relative to kda_s6/kvar or "-" = installed)
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/sassv-${TAG:-x}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6
for v in $VARS; do
  nm=${v%%=*}; f=${v#*=}; [ "$f" != - ] && f=$VD/kvar/$f
  D=/tmp/kdasassv-$nm; rm -rf $D; mkdir -p $D
  CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache \
    python3 $VD/fip_var.py $f --shapes ${SHAPES:-f16k} --time none --tag $nm > $D/app.log 2>&1
  for c in $(ls $D/*.cubin 2>/dev/null | head -2); do
    nvdisasm -c $c > $R/$nm.sass 2>/dev/null
    echo "$nm $(cuobjdump -res-usage $c 2>/dev/null | grep -o 'REG:[0-9]*\|STACK:[0-9]*\|LOCAL:[0-9]*' | tr '\n' ' ') LDL $(grep -c 'LDL' $R/$nm.sass) STL $(grep -c 'STL' $R/$nm.sass) instr $(grep -cE '^ +/\*[0-9a-f]{4,}\*/' $R/$nm.sass) BSSY $(grep -c 'BSSY' $R/$nm.sass) WARPSYNC $(grep -c 'WARPSYNC' $R/$nm.sass)"
  done
  tail -2 $D/app.log
done
