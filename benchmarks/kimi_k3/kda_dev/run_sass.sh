#!/bin/bash
# P-KDA2: cubin stats per route: bash run_sass.sh <tag> <route:g|fag> [dev json] ...(pairs)
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; TAG=$1; shift; R=$K/runs/P-KDA2/sass-$TAG; mkdir -p $R
i=0; while [ $# -gt 0 ]; do RT=$1; DV=${2:-"{}"}; shift 2; i=$((i+1)); D=/tmp/pkda2sass-$TAG-$i; rm -rf $D; mkdir -p $D
  VLLM_KIMI_K3_CUTE_CACHE_DIR=$D/kc CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache timeout 600 python3 -u benchmarks/kimi_k3/kda_dev/compile_one.py $RT "$DV" > $D/app.log 2>&1; grep -E "Error|error|compiled" $D/app.log | head -2; find $D -name "*.cubin" | head -3
  echo "dump: $(find $D -name "*.cubin" | wc -l) cubins; $(ls $D | head -5 | tr "\n" " ")"
  for c in $(find $D -name "*.cubin" -size +50k 2>/dev/null | head -1); do nvdisasm -c $c > $R/v$i.sass 2>/dev/null; echo "v$i $RT $DV: $(cuobjdump -res-usage $c 2>/dev/null | grep -o "REG:[0-9]*\|STACK:[0-9]*\|LOCAL:[0-9]*" | tr "\n" " ") LDL $(grep -c LDL $R/v$i.sass) STL $(grep -c STL $R/v$i.sass) instr $(grep -cE "^ +/\*[0-9a-f]{4,}\*/" $R/v$i.sass)"; done; done
