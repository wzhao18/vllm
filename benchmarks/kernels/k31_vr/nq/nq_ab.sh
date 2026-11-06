#!/bin/bash
# nq_ab.sh <jobid> <nodelist (4 hosts)> <results dir> <point> <old|new> <rows> [order] [ctx] [-- extra bench args]
# One A/B point: the arms of [order] (default ls,stock,stock,ls = ABBA) as separate processes on the same 4 nodes, same
# cfg2bench args/env except decode_backend, then nq_summary.py over the point's logs (per row: stock and ls medians,
# ls/stock, the ls coverage line and the drafter fast-path flags).
#   q comes from the point's config: c256/c160/c48-2p1d = q4 (K=3), c64/c48-1p1d/c8/c1 = q8 (K=7).
#   Rows <= the point's production limit (min(max-num-seqs * q, capture cap): cfg2bench.py --split) and rows above it
#   run as separate processes (part 'lo', tag <TAG>; part 'hi', tag <TAG>-hi), so the rows inside the limit always
#   run under the production max-num-seqs / capture cap; only the 'hi' processes raise them (NOT PRODUCTION note).
#   An arm that dies before its first [forward] line (startup refusal / crash) is not retried for the same process tag
#   (<TAG>[-hi], i.e. point, drafter, ctx, extra args and part): its later runs in this line are skipped (same code, same
#   config, same failure); other lines (other ctx / extra args / part) still run. The summary reports the error.
# env: as nq_run1.sh (NQ_OV, NQ_BUILD, NQ_IMG, NQ_TIMEOUT); NQ_DEADLINE (epoch s): skip runs that cannot finish
# (NQ_RUN_EST s each, default 1500) before it.
# NQ-TUNE: NQ_RUN_EST / NQ_TIMEOUT come from the bf16-state runs (6-9 min startup, 15-40 s per case) and the full
# capture list (~100 rows to prepare at c256, 20-30+ min startup at fp32 state). Each process now captures only its
# measured rows, their drafter batches and the capacity anchors (cfg2bench.py measured_capture: 9-14 sizes); remeasure
# the startup on the first block and lower NQ_RUN_EST. NQ_FULL_CAPTURE=1 (nq_run1.sh) restores the production list.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J=$1; NL=$2; R=$3; POINT=$4; DRAFT=$5; ROWS=$6; shift 6
ORDER=ls,stock,stock,ls; CTX=agentx
if [ $# -gt 0 ] && [ "$1" != "--" ]; then ORDER=$1; shift; fi
if [ $# -gt 0 ] && [ "$1" != "--" ]; then CTX=$1; shift; fi
[ "$1" = "--" ] && shift
EXTRA=("$@")
TAG=$POINT-$DRAFT-$CTX; [ ${#EXTRA[@]} -gt 0 ] && TAG=$TAG-x$(echo "${EXTRA[*]}" | tr -c 'A-Za-z0-9\n' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
mkdir -p $R
read -r LO HI < <(python3 $D/cfg2bench.py $POINT --split --rows $ROWS) || { echo "cfg2bench --split failed"; exit 2; }
[ -n "$LO" ] && [ -n "$HI" ] || { echo "cfg2bench --split gave no rows"; exit 2; }
for part in lo hi; do
  if [ $part = lo ]; then prows=$LO; ptag=$TAG; else prows=$HI; ptag=$TAG-hi; fi
  [ "$prows" = - ] && continue
  echo "part $part ($ptag): rows $prows"
  i=0
  for arm in ${ORDER//,/ }; do
    i=$((i + 1)); out=$R/$ptag-$i$arm; dead=$R/DEAD-$ptag-$arm
    if [ -f $dead ]; then echo "skip $out: $arm failed at startup in $(cat $dead)"; continue; fi
    if [ -n "$NQ_DEADLINE" ] && [ $(( $(date +%s) + ${NQ_RUN_EST:-1500} )) -gt $NQ_DEADLINE ]; then
      echo "skip $out: not enough block time left"; echo "$(basename $out) rows $prows" >> $R/SKIPPED; continue
    fi
    left=${NQ_TIMEOUT:-2700}
    [ -n "$NQ_DEADLINE" ] && left=$(( NQ_DEADLINE - $(date +%s) )) && [ $left -gt ${NQ_TIMEOUT:-2700} ] && left=${NQ_TIMEOUT:-2700}
    NQ_TIMEOUT=$left bash $D/nq_run1.sh $J $NL $out $POINT $arm $DRAFT $prows $CTX -- "${EXTRA[@]}"
    rc=$?
    # (not a timeout: a cold first build can outlast a short NQ_TIMEOUT without any code failure)
    if [ $rc != 0 ] && [ $rc != 124 ] && ! grep -aq "^\[forward\].*median" $out.log; then basename $out > $dead; fi
  done
done
python3 $D/nq_summary.py $R --tag $TAG | tee $R/$TAG.summary.txt
