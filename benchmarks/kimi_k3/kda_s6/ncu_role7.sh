#!/bin/bash
# ncu --set full of one F16K fip launch (installed kernel or a kvar variant) with CuTe DSL lineinfo;
# per-role instruction/stall split via ncu_role.py.  env: TAG, VAR (kvar file or -), ROLES (kernel.py line map)
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/ncu-${TAG:-r7}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$R/dslcache
f=${VAR:--}; [ "$f" != - ] && f=$VD/kvar/$f
ncu --set full --import-source yes -k regex:_pkd -c 1 -f -o $R/r \
  python3 $VD/fip_var.py $f --shapes f16k --time none --tag ncu > $R/app.log 2>&1
echo "ncu rc=$?"; grep -E "==ERR|Error" $R/app.log | head -3
ncu -i $R/r.ncu-repz --page source --csv --print-source cuda > $R/cuda.csv 2>/dev/null
python3 $VD/ncu_role.py $R/cuda.csv "$ROLES" prep 2>&1 | head -40
