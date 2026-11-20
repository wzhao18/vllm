#!/bin/bash
# P-KDA2-S6 r9: compile the persistent-KDA kernel for one shape / variant with line info, dump SASS (nvdisasm -gi)
# and print the local-memory / global-store map per source line.
#   bash sassli.sh <tag> "<lens>" '<variant json>' [more variant json...]
# Each variant compiles in its own fresh cache dir; outputs runs/P-KDA2-S6/sassli/<tag>-v<i>.sass
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
TAG=$1; LENS=$2; shift 2
R=$K/runs/P-KDA2-S6/sassli; mkdir -p $R
i=0
for VAR in "$@"; do
  D=/tmp/kdasassli-$TAG-$i; rm -rf $D; mkdir -p $D
  CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache CUTE_DSL_LINEINFO=1 \
    timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_devb.py --rounds 1 --iters 1 --lens $LENS --variants "$VAR" \
    > $D/app.log 2>&1
  grep -E "Error|error|Traceback" $D/app.log | head -5
  c=$(ls -S $D/*.cubin 2>/dev/null | head -1)
  if [ -z "$c" ]; then c=$(find $D -name "*.cubin" | head -1); fi
  echo "== v$i $VAR cubin=$(basename "$c")"
  nvdisasm -gi -c "$c" > $R/$TAG-v$i.sass 2>/dev/null || nvdisasm -g -c "$c" > $R/$TAG-v$i.sass
  cuobjdump -res-usage "$c" 2>/dev/null | grep -o 'REG:[0-9]*\|STACK:[0-9]*\|SHARED:[0-9]*\|LOCAL:[0-9]*' | tr '\n' ' '; echo
  python3 benchmarks/kimi_k3/kda_s6r8/sassli.py $R/$TAG-v$i.sass
  i=$((i+1))
done
