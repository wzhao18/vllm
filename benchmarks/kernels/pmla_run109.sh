#!/bin/bash
# P-MLA run 109: full specialized prefill sparse MLA test file on the landed-ready h8 dispatch.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run109
mkdir -p $R
timeout 900 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py > $R/test.log 2>&1
echo "tests rc=$? $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "FAILED|Error" $R/test.log | head -8 | cut -c1-250 >> $R/summary.txt
echo DONE > $R/DONE
