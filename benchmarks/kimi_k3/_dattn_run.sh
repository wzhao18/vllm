#!/bin/bash
# private D-ATTN variant runner (untracked): _dattn_run.sh <outdir> <set>
out=$1; set=$2
mkdir -p $out
B="--rows 384 --share 0.8 --no-trtllm"
case $set in
  A) V=("$B --stamps --cpa 0" "$B --stamps --cpa 2" "$B --stamps --cpa 3" "$B --stamps --gather-only --cpa 0" "$B --stamps --gather-only --cpa 3" "--rows 128,256,384 --share 0.8 --cpa 3") ;;
  B) V=("$B --stamps --cpa 0" "$B --stamps --cpa 3" "--rows 128,256,384 --share 0.8 --cpa 3")
     (cd /k31ov && python3 -m pytest -q --noconftest -x tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 4) ;;
  C) V=("$B --stamps --cpa 3" "$B --stamps --cpa 2" "--rows 128,256,384 --share 0.8")
     (cd /k31ov && python3 -m pytest -q --noconftest -x tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 4) ;;
  D) V=("$B --stamps --prefetch 0" "$B --stamps --prefetch 6" "$B --stamps --cpa 0" "$B --gather-only --stamps") ;;
  E) V=("$B --stamps" "$B --stamps --prefetch 8" "--rows 128,256,384 --share 0.8")
     (cd /k31ov && python3 -m pytest -q --noconftest -x tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 4) ;;
  F) V=("$B --stamps" "$B --stamps --cpa 4" "--rows 128,256,384 --share 0.8 --cpa 4 --no-trtllm") ;;
  M) V=("$B --stamps" "$B --stamps --cpa 3" "$B --stamps --cpa 2" "$B --stamps --cpa 4" "$B --gather-only --stamps --cpa 3" "--rows 128,256,384 --share 0.8 --cpa 3" "--rows 128,256,384 --share 0.8 --cpa 2 --no-trtllm") ;;
  P) V=("$B --stamps" "$B --stamps --no-gather" "$B --gather-only --stamps" "--rows 128,256,384 --share 0.8 --no-gather --no-trtllm" "--rows 128,256,384 --share 0.8")
     (cd /k31ov && python3 -m pytest -q --noconftest -x tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 4) ;;
  Q) V=("$B --stamps" "$B --stamps --no-gather" "--rows 128,256,384 --share 0.8")
     (cd /k31ov && python3 -m pytest -q --noconftest -x tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 4)
     (cd /k31ov && python3 -m pytest -q --noconftest -x -s tests/models/kimi_k3/test_specialized_decode_sp_local_query_gpu.py 2>&1 | grep -E "KV insert|passed|failed|Error|assert" | tail -n 12) ;;
  *) V=("$set") ;;
esac
i=0
for v in "${V[@]}"; do
  echo "VARIANT $i: $v"
  python3 /k31ov/benchmarks/kimi_k3/bench_dattn_sparse_native.py $v --out $out/v$i.json 2>&1 | grep -v "^\s*$" | grep "^R\|Error\|error\|Traceback\|assert" | cut -c1-300
  python3 -c "
import json
d=json.load(open('$out/v$i.json'))
for r,x in d.items():
  print(r,{k:v for k,v in x.items() if k.endswith('_us')})
  s=x.get('stamps_us_median_max')
  if s: print('ST',' '.join(f'{k}={v[0]}/{v[1]}' for k,v in s.items()))
" 2>&1 | tail -n 8
  i=$((i+1))
done
echo END
