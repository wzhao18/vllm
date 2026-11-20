#!/bin/bash
# D-KDA round 3 (b): KdaFlow prologue breakdown (PTRACE, CTAs 0/53) at n32/n96 and
# the upper bounds of the prologue / prep levers: --skip gate, prep, gate+prep.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3b-$$
for n in 32 96; do
  O=/tmp/dkda-r3b-$$-$n.out
  timeout 600 python3 $B --n $n --kinds flow --gap 1 --pending 4 --place striped --mx --trace --iters 1 --layers 2 > $O 2>&1
  echo "== flow ptrace n$n rc=$?"
  grep "^PTRACE" $O | awk '!seen[$3]++' | cut -c1-260 | head -2
done
for sk in "" gate prep gate,prep; do
  echo "== flow skip=[$sk]"
  timeout 900 python3 $B --n 32 64 96 128 --kinds flow --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-62
done
