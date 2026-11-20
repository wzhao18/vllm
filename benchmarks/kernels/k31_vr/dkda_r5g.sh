#!/bin/bash
# D-KDA round 5 (g): KdaCoreQ (3 / 4 item groups, 4-row x 32-key scan, <=168 / 128
# registers) - fold test vs the stock commit (KdaFlow prep stand-in), timing vs KdaCore,
# cycle trace, ptxas registers (ncu launch stats), striped mx --gap 1 --pending 4.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r5g; mkdir -p $O
C=/tmp/dkda-r5g-$$
for q in 3 4; do
  echo "== QGROUPS=$q (tests at the default 3)"
  [ $q = 3 ] && CUTE_DSL_CACHE_DIR=$C-t$q timeout 1500 python3 -m pytest --noconftest -q -rf -x $T -k "fold_matches and coreq and (33 or 96) and graph" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|assert|differ|Unsupported" | cut -c1-300 | head -8
  echo "-- coreq Q$q striped"
  CUTE_DSL_CACHE_DIR=$C-b$q timeout 900 python3 $B --n 32 96 128 --kinds coreq --qgroups $q --gap 1 --pending 4 --place striped --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
  CUTE_DSL_CACHE_DIR=$C-tr$q timeout 900 python3 $B --n 96 --kinds coreq --qgroups $q --gap 1 --pending 4 --place striped --mx --trace --layers 2 --iters 1 2>&1 | grep -E "QTRACE" | tail -48 | grep " w 0 " | head -6 | cut -c1-200
  CUTE_DSL_CACHE_DIR=$C-b$q timeout 900 ncu --metrics launch__registers_per_thread,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,gpu__time_duration.sum -k regex:Kda -s 2 -c 1 --csv --page raw python3 $B --n 96 --kinds coreq --qgroups $q --gap 1 --pending 4 --place striped --mx --iters 1 --layers 4 2>/dev/null | tail -1 | tr ',' '\n' | tail -3 | tr '\n' ' '; echo
done
