#!/bin/bash
# D-KDA round 3 (a2): KdaFlow per-group cycle traces of the LAST (steady-state) launch,
# CTAs 0/53, full and skip arms, striped mx --gap 1 --pending 4.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3a-$$
for arm in "32:" "96:" "96:prep,gate" "96:prep,gate,store" "96:prep,gate,store,scan"; do
  n=${arm%%:*}; sk=${arm#*:}
  O=/tmp/dkda-r3a-$$-$n-${sk//,/_}.out
  timeout 600 python3 $B --n $n --kinds flow --gap 1 --pending 4 --place striped --mx --trace --iters 3 --layers 2 ${sk:+--skip $sk} > $O 2>&1
  echo "== flow trace n$n skip=[$sk] rc=$? traces=$(grep -c '^TRACE' $O)"
  grep "^TRACE cta 0 \|^TRACE cta 53 " $O | tac | awk '!seen[$3 $5]++' | cut -c1-330 | head -4
  grep "^PTRACE" $O | tac | awk '!seen[$3]++' | cut -c1-230 | head -1
  grep -E "rror|Traceback" $O | cut -c1-200 | head -3
done
