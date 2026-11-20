#!/bin/bash
# Compile fip (f16k) with the CuTe DSL cubin dump and report regs / spills / SASS size.
# env: TAG (out dir suffix), AOV (optional overlay first on PYTHONPATH = arm A)
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/sass-${TAG:-x}
one() { # $1 = name, $2 = PYTHONPATH prefix or empty
  local D=/tmp/kdasass-$1; rm -rf $D; mkdir -p $D
  PYTHONPATH=$2$BASEPP CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache \
    python3 /k31ov/benchmarks/kimi_k3/kda_s6/fip_ab.py --shapes f16k --time none --tag $1 > $D/app.log 2>&1
  for c in $(ls $D/*.cubin 2>/dev/null | head -2); do
    nvdisasm -c $c > $R/$1.sass 2>/dev/null
    echo "$1 $(basename $c | cut -c1-40): $(cuobjdump -res-usage $c 2>/dev/null | grep -o 'REG:[0-9]*\|STACK:[0-9]*\|LOCAL:[0-9]*' | tr '\n' ' ') LDL $(grep -c 'LDL' $R/$1.sass) STL $(grep -c 'STL' $R/$1.sass) instr $(grep -cE '^ +/\*[0-9a-f]{4,}\*/' $R/$1.sass)"
  done
  grep -E "Error|error" $D/app.log | head -3
}
mkdir -p $R; BASEPP=$PYTHONPATH
[ -n "$AOV" ] && one A $K/ov/$AOV:
one B ""
