#!/bin/bash
# ncu warp-state / scheduler / instruction-fetch comparison of kernel.py variants (f16k launch).
# env: TAG, VARS ("name=file ..." as in varab.sh), SECS (extra ncu args)
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6/ncuv-${TAG:-x}; mkdir -p $R
VD=/k31ov/benchmarks/kimi_k3/kda_s6
M=smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
M=$M,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
M=$M,smsp__issue_active.avg.per_cycle_active,smsp__inst_executed.sum,gpu__time_duration.sum
M=$M,sm__cycles_elapsed.avg.per_second
for v in $VARS; do
  nm=${v%%=*}; f=${v#*=}; [ "$f" != - ] && f=$VD/kvar/$f
  ncu --metrics $M -k regex:_pkd -c 1 -f --csv \
    python3 $VD/fip_var.py $f --shapes f16k --time none --tag $nm > $R/$nm.csv 2> $R/$nm.err
  echo "== $nm rc=$?"
  python3 - "$R/$nm.csv" <<'EOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 10]
h = rows[0]; mi = h.index("Metric Name"); vi = h.index("Metric Value")
out = []
for r in rows[1:]:
    n = r[mi].replace("smsp__average_warps_issue_stalled_", "st_").replace("_per_issue_active.ratio", "")
    out.append(f"{n}={r[vi]}")
print(" ".join(out))
EOF
done
