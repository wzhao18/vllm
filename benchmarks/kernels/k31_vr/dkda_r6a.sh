#!/bin/bash
# D-KDA round 6 (a): die-local (alternating-die 2 MiB chunks + die-matched claim) ladder of
# KdaCore v2 and KdaCoreQ Q3: full / no store / no scan math (= movement skeleton) /
# no scan + no store, n32/64/96/128 mx; KdaFlow v18 local reference; the bench's own
# per-layer neighbour cost (--gap -1) so intercepts can be split.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
C=/tmp/dkda-r6a-$$
run() {  # tag kind extra...
  local tag=$1; shift
  echo "-- $tag"
  CUTE_DSL_CACHE_DIR=$C-$tag timeout 900 python3 $B --gap 1 --pending 4 --mx "$@" 2>&1 \
    | grep -E "us/layer|rror|FAILED" | cut -c1-140
}
run nb --n 32 96 --kinds core --gap -1 --place local
run core-full --n 32 64 96 128 --kinds core --place local
run core-nostore --n 32 64 96 128 --kinds core --place local --skip store
run core-skel --n 32 64 96 128 --kinds core --place local --skip scan
run core-skel-nostore --n 32 64 96 128 --kinds core --place local --skip scan,store
run q3-full --n 32 64 96 128 --kinds coreq --qgroups 3 --place local
run q3-nostore --n 32 64 96 128 --kinds coreq --qgroups 3 --place local --skip store
run flow-full --n 32 96 128 --kinds flow --place local
