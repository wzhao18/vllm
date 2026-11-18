#!/bin/bash
# wsq.sh (Rhea login node): the k31ws GPU pool queue in fair order with the scheduler plan (tools/wsplan.py),
# running holds with busy GPUs and reservations, then the last usage.log lines (start lines carry waited=, end lines dur=).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
python3 $K/tools/wsplan.py show
echo "  recent:"; tail -${1:-8} $K/pool/usage.log | sed "s/^/  /"
