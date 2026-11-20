#!/bin/bash
# Same-node ABBA of the fip kernel: A = overlay $AOV (first on PYTHONPATH), B = /k31ov.
# B is checked bit for bit against A. env: TAG, AOV, SHAPES, TIMES, ITERS
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/ab-${TAG:-x}; mkdir -p $R
AB=/k31ov/benchmarks/kimi_k3/kda_s6/fip_ab.py
SH=${SHAPES:-small,mixed,f16k}; TM=${TIMES:-f16k}; IT=${ITERS:-40}
arm() { # $1 = A|B, $2 = rep, rest = extra args
  local who=$1 rep=$2; shift 2
  if [ $who = A ]; then export PYTHONPATH=$K/ov/$AOV:$BASEPP; else export PYTHONPATH=$BASEPP; fi
  timeout 900 python3 $AB --tag $who$rep --shapes $SH --time $TM --iters $IT "$@" > $R/$who$rep.log 2>&1
  echo "rc=$?" >> $R/$who$rep.log; grep -E "^$who$rep|Error|rc=" $R/$who$rep.log | tail -7
}
BASEPP=$PYTHONPATH
arm A 1 --save $R/a.pt; arm B 1 --cmp $R/a.pt; arm B 2 --cmp $R/a.pt; arm A 2
