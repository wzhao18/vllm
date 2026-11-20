#!/bin/bash
# ncu --set full of one f16k launch of a kernel variant; exports the SASS source view CSV.
# env: TAG, VAR (file in kvar/ or "-")
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/ncus-${TAG:-x}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6; f=${VAR:--}; [ "$f" != - ] && f=$VD/kvar/$f
export CUTE_DSL_LINEINFO=1
ncu --set full --import-source yes -k regex:_pkd -c 1 -f -o $R/r \
  python3 $VD/fip_var.py $f --shapes f16k --time none --tag S > $R/app.log 2>&1
echo "ncu rc=$?"; grep -E "==ERR|Error" $R/app.log | head -3
ncu -i $R/r.ncu-repz --page source --csv --print-source sass > $R/sass.csv 2>/dev/null
ncu -i $R/r.ncu-repz --page details --section SpeedOfLight --section SchedulerStats 2>&1 \
  | grep -E "Duration|SM Frequency|Issued Warp|Issue Slots Busy" | head -6
ls -la $R/sass.csv
