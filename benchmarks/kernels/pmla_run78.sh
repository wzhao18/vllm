#!/bin/bash
# P-MLA run 78: FlashInfer-main prims_ts sparse MLA on sm_107 at F16K (comparator, lead s4).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-MLA/run78; mkdir -p $R
cp benchmarks/kernels/pmla_prims_ts.py $R/
cd $R
PYTHONPATH=$K/src/fi-main-855eece timeout 900 python3 pmla_prims_ts.py --tokens 16384 --json $R/a.json > $R/a.log 2>&1; echo "rc=$?" >> $R/a.log
if ! grep -q "us" $R/a.json 2>/dev/null; then PYTHONPATH=$K/src/fi-main-855eece:$K/venvs/tvmffi-new/site timeout 900 python3 pmla_prims_ts.py --tokens 16384 --json $R/b.json > $R/b.log 2>&1; echo "rc=$?" >> $R/b.log; fi
echo DONE > $R/DONE
