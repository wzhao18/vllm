#!/bin/bash
# Pipe / MIO utilization of saved ncu --set full reports (runs/P-KDA2-S6/ncus-<tag>/r.ncu-repz).
# args: tags
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2-S6
for t in "$@"; do
  echo "== $t"
  ncu -i $R/ncus-$t/r.ncu-repz --page raw --csv > /tmp/raw-$t.csv 2>&1
  python3 - /tmp/raw-$t.csv <<'EOF'
import csv, sys
r = list(csv.reader(open(sys.argv[1])))
h = next(x for x in r if "ID" in x or "Kernel Name" in x)
v = r[r.index(h) + 2] if len(r) > r.index(h) + 2 else r[-1]
keys = ("pipe_lsu", "pipe_xu", "pipe_alu", "pipe_fma", "pipe_fmaheavy", "pipe_uniform",
        "mio_inst_issued", "mio2rf", "pipe_shared_cycles", "data_pipe_lsu_wavefronts_mem_shared",
        "issue_active", "inst_executed_pipe_adu", "pipe_tensor")
for a, b in zip(h, v):
    if "pct_of_peak_sustained_active" in a and any(k in a for k in keys):
        print(f"{a[:72]:72s} {b}")
EOF
done
