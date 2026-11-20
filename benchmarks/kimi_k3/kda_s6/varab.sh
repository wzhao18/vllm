#!/bin/bash
# Same-node A/B of kernel.py variants on one overlay: arms V1 V2 ... then reversed.
# The first variant is the reference (outputs saved, others compared bit for bit).
# env: TAG, VARS ("name=file ..." with file relative to kda_s6/kvar or "-" = installed),
#      SHAPES, TIMES, ITERS
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/var-${TAG:-x}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6
SH=${SHAPES:-small,mixed,f16k}; TM=${TIMES:-f16k}; IT=${ITERS:-40}
arm() { # $1 = name, $2 = file, $3 = rep, rest = extra args
  local nm=$1 f=$2 rep=$3; shift 3
  [ "$f" != - ] && f=$VD/kvar/$f
  timeout 900 python3 $VD/fip_var.py $f --tag $nm$rep --shapes $SH --time $TM --iters $IT "$@" \
    > $R/$nm$rep.log 2>&1
  echo "rc=$?" >> $R/$nm$rep.log; grep -E "^$nm$rep|Error|rc=[1-9]" $R/$nm$rep.log | tail -7
}
set -- $VARS; REF=${1%%=*}
for rep in 1 2; do
  list="$VARS"; [ $rep = 2 ] && list=$(echo $VARS | tr ' ' '\n' | tac | tr '\n' ' ')
  for v in $list; do
    nm=${v%%=*}; f=${v#*=}
    if [ $nm = $REF ] && [ $rep = 1 ]; then arm $nm $f $rep --save $R/ref.pt; else arm $nm $f $rep --cmp $R/ref.pt; fi
  done
done
echo "== medians (us)"; grep -h "graph_us" $R/*.log | sed -E 's/^([^ ]+) +([^ ]+).*med ([0-9.]+).*/\2 \1 \3/' | sort
