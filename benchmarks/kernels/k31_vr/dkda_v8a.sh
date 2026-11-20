#!/bin/bash
# D-KDA v8a: KdaRing v8 (prologue: loads overlapped, gates split: pairs first item, scan warps the rest).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
D=/tmp/dkda-dump-$$; mkdir -p $D
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v8a-$$ CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D
echo "== bench n96 striped (compile + regs)"
timeout 900 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place striped --mx --iters 10 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120 | head -4
for f in $D/*.cubin; do cuobjdump -res-usage $f 2>/dev/null | grep -iE "REG" | cut -c1-100 | head -2; done | sort -u | head -3
unset CUTE_DSL_KEEP_CUBIN CUTE_DSL_DUMP_DIR
echo "== test bitwise n33 + fold-vs-stock ring n33"
timeout 900 python3 -m pytest --noconftest -q -rf $T -k "(ring_is_bitwise and 33) or (fold_matches and ring and 33)" 2>&1 | grep -E "passed|failed|^FAILED|Error|differ" | cut -c1-200 | head -8
echo "== bench ring striped+local n32 64 96 128"
timeout 900 python3 $B --n 32 64 96 128 --kinds ring --gap 1 --pending 4 --place striped local --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-80
echo "== trace n96 striped"
O=/tmp/dkda-v8a-$$.out
timeout 600 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place striped --mx --trace --iters 1 --layers 2 > $O 2>&1
for k in "pro cta 0 " "prep cta 0 pair 0" "prep cta 0 pair 1" "scan cta 0 "; do grep "RTRACE $k" $O | head -1 | cut -c1-280; done
grep -vE "RTRACE" $O | grep -E "rror|Traceback" | cut -c1-200 | head -4
