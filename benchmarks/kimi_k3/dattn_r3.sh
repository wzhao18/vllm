#!/bin/bash
# D-ATTN round 3 microbench driver: dattn_r3.sh <out dir> <arm>... ; arm = kern:share[:extra,flags]
# e.g. dattn_r3.sh $K/runs/D-ATTN/r3/e1 h8:0.75 km:0.75:--no-trtllm+--h8knobs+1,0,0  (extra args: + separated)
set -u
OUT=$1; shift
mkdir -p "$OUT"
cd /k31ov
for arm in "$@"; do
  IFS=: read -r kern share extra <<< "$arm"
  tag=${kern}_s${share}${extra//[^A-Za-z0-9]/_}
  echo "== arm $arm"
  python3 benchmarks/kimi_k3/bench_dattn_sparse_native.py --kern "$kern" --share "$share" \
    --rows "${ROWS:-128,256,384}" ${extra//+/ } --out "$OUT/$tag.json" 2>&1 \
    | grep -v -E 'Warning|warn\(' | tail -n 8
done
