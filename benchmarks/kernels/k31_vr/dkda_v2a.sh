#!/bin/bash
# D-KDA v2a: KdaRing v2 (register page stream + prep/tail warps). regs, bitwise test vs KdaFlow, microbench.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
D=/tmp/dkda-dump-$$; mkdir -p $D
export CUTE_DSL_CACHE_DIR=/tmp/dkda-c-$$ CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D
echo "== bench n96 ring (compile + regs)"
timeout 900 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local --mx --iters 10 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-200 | head -6
echo "== res"; for f in $D/*.cubin; do cuobjdump -res-usage $f 2>/dev/null | grep -iE "Function|REG" | sed -E "s/Function ([^:]{0,40}).*:/F \1/" | cut -c1-170 | head -4; done | sort -u | head -8
unset CUTE_DSL_KEEP_CUBIN CUTE_DSL_DUMP_DIR
echo "== test bitwise n33"
timeout 900 python3 -m pytest --noconftest -q -rf -x $T -k "ring_is_bitwise and 33" 2>&1 | grep -E "passed|failed|^FAILED|Error|Mismatch|mismatch|assert" | cut -c1-220 | head -8
echo "== bench ring/flow local+striped n32..128"
timeout 1500 python3 $B --n 32 64 96 128 --kinds ring flow --gap 1 --pending 4 --place local striped --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-80
