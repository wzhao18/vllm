#!/bin/bash
# run_gemm.sh <args of bench_draft_gemm.py>: rerun the bench while it exits with a
# sticky CUDA error (75); it resumes from <out>/raw.csv.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rc=75
while [ $rc -eq 75 ]; do
  python3 $D/bench_draft_gemm.py "$@"
  rc=$?
done
exit $rc
