#!/bin/bash
# Same-node arms of kernel.py variants with a py-spy dump of any arm still
# running after SPY seconds (compile vs GPU-wait diagnosis).
# env: TAG, VARS ("name=file ..." with file in kda_s6/kvar or "-"), SHAPES, TIMES,
#      REPS (1), TMO (arm timeout s, 900), SPY (s, 300)
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/var-${TAG:-x}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6
SH=${SHAPES:-small,mixed,2x8k,f16k}; TM=${TIMES:-f16k,2x8k}
arm() {
  local nm=$1 f=$2 rep=$3 extra=$4
  [ "$f" != - ] && f=$VD/kvar/$f
  timeout ${TMO:-900} python3 $VD/fip_var.py $f --tag $nm$rep --shapes $SH --time $TM --iters 40 $extra \
    > $R/$nm$rep.log 2>&1 &
  local p=$!
  ( sleep ${SPY:-300}; kill -0 $p 2>/dev/null && $K/tools/pyspy/py-spy dump --pid $p > $R/$nm$rep.spy 2>&1 ) &
  local s=$!
  wait $p; echo "rc=$?" >> $R/$nm$rep.log; kill $s 2>/dev/null
  grep -E "^$nm$rep |Error|rc=[1-9]" $R/$nm$rep.log | tail -6
  [ -s $R/$nm$rep.spy ] && { echo "-- spy $nm$rep"; grep -E "^ +[a-zA-Z_]+ \(" $R/$nm$rep.spy | head -8; }
}
set -- $VARS; REF=${1%%=*}
for rep in $(seq 1 ${REPS:-1}); do
  list="$VARS"; [ $((rep % 2)) = 0 ] && list=$(echo $VARS | tr ' ' '\n' | tac | tr '\n' ' ')
  for v in $list; do
    nm=${v%%=*}; f=${v#*=}
    if [ $nm = $REF ] && [ $rep = 1 ]; then arm $nm $f $rep "--save $R/ref.pt"; else arm $nm $f $rep "--cmp $R/ref.pt"; fi
  done
done
echo "== medians (us)"; grep -h "graph_us" $R/*.log | sed -E 's/^([^ ]+) +([^ ]+).*med ([0-9.]+).*/\2 \1 \3/' | sort
