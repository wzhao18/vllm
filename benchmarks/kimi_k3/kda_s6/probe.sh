#!/bin/bash
# quick per-shape probe with timeouts: $1 = overlay dir, rest = check args
D=$1; shift; cd $D || exit 1; export PYTHONPATH=$D:$PYTHONPATH; python3 -c "import vllm; print(vllm.__file__)"
for s in ${SHAPES:-small f16k}; do echo "== $D $s"; timeout ${TMO:-300} python3 /k31ov/benchmarks/kimi_k3/fip_s6_check.py --shapes $s --time $s "$@" 2>&1 | grep -v Warning | tail -4; echo "rc=$?"; done
