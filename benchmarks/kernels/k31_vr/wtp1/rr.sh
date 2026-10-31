#!/bin/bash
# rr.sh <run.sh args...>: run.sh, retried once at gpu-memory-utilization $GMU_RETRY (0.85) if the first attempt ran out of memory.
D=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/wtp1/tools; R=$D/../results
bash $D/run.sh "$@"; rc=$?
if [ $rc != 0 ] && grep -q "out of memory\|OutOfMemory\|CUDA out of memory" $R/$3.log; then
  mv $R/$3.log $R/$3.oom1.log; GMU=${GMU_RETRY:-0.85} bash $D/run.sh "$@"; rc=$?
fi
exit $rc
