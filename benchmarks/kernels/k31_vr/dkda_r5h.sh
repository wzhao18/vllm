#!/bin/bash
# D-KDA round 5 (h): is the checkpoint store bound by its 128-byte row segments? KdaCore
# v2 (per-chunk 2-D TMA store boxes) vs the qstore arm (the warp's 16 KiB quarter as one
# contiguous bulk store per item); KdaCoreQ fold test vs stock (cache check fixed).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
C=/tmp/dkda-r5h-$$
for sk in "" qstore; do
  echo "-- core skip=[$sk] striped"
  CUTE_DSL_CACHE_DIR=$C-$sk timeout 900 python3 $B --n 32 96 128 --kinds core --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
echo "== coreq fold test vs stock (Q3)"
CUTE_DSL_CACHE_DIR=$C-t timeout 1500 python3 -m pytest --noconftest -q -rf -x $T -k "fold_matches and coreq and (33 or 96)" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|assert|differ|Unsupported" | cut -c1-300 | head -8
