#!/bin/bash
# pmla10_run.sh <outdir> [probe|ncu|both] [knob lists...]: P-MLA-2CTA round 2 (1 GPU).
#   probe: build + run pmla10_gather.cu (LDGSTS rate laws)
#   ncu:   ncu --set full on one sparse_mla_prefill_h8 launch per knob list at F16K L78K (default: base),
#          then a compact summary (duration, SM clock, throughput, issue, stall reasons) per report
OUT=$1; MODE=${2:-both}; shift 2; KNOBS=${*:-base}
D=$(cd "$(dirname "$0")" && pwd); OV=$(cd "$D/../.." && pwd); mkdir -p $OUT
if [ "$MODE" != ncu ]; then bash $D/pmla10_gather.sh $OUT/probe | tail -n 20; fi
[ "$MODE" = probe ] && exit 0
export CUTE_DSL_LINEINFO=1
cd $OV
for k in $KNOBS; do
  a=$k; [ "$k" = base ] && a=""
  tag=h8_${k//:/_}
  ncu --set full --import-source yes -k regex:SparseMlaPrefillH8 --launch-skip 1 -c 1 --clock-control none \
      -f -o $OUT/$tag python3 benchmarks/kernels/pmla10_ncu.py $a --n 3 > $OUT/$tag.log 2>&1
  echo "== $tag rc=$? $(grep -m3 'iter' $OUT/$tag.log | tr '\n' ' ')"
  ncu -i $OUT/$tag.ncu-repz --page details --csv > $OUT/$tag.details.csv 2>&1
  python3 - $OUT/$tag.details.csv <<'EOF'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0] if rows else []
keys = ("Duration", "SM Frequency", "Compute (SM) Throughput", "Memory Throughput", "L1/TEX Hit Rate",
        "Issue Slots Busy", "Issued Warp Per Scheduler", "No Eligible", "Active Warps Per Scheduler",
        "Eligible Warps Per Scheduler", "Warp Cycles Per Issued Instruction", "Executed Ipc Active",
        "L2 Hit Rate", "DRAM Throughput", "Shared Memory", "Registers Per Thread", "Max Bandwidth")
try:
    ni, nv, nu = hdr.index("Metric Name"), hdr.index("Metric Value"), hdr.index("Metric Unit")
except ValueError:
    print("no csv:", rows[:2]); sys.exit()
seen = set()
for r in rows[1:]:
    if len(r) > nv and any(r[ni].startswith(k) for k in keys) and r[ni] not in seen:
        seen.add(r[ni]); print(f"  {r[ni]}: {r[nv]} {r[nu]}")
EOF
  ncu -i $OUT/$tag.ncu-repz --page raw --csv --metrics regex:smsp__pcsamp_warps_issue_stalled_.*,regex:sm__pipe_.*_cycles_active.avg.pct_of_peak_sustained_active,regex:sm__inst_executed_pipe_.*.avg.pct_of_peak_sustained_active > $OUT/$tag.raw.csv 2>&1
  python3 - $OUT/$tag.raw.csv <<'EOF'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
if len(rows) < 3: print("no raw"); sys.exit()
h, v = rows[0], rows[2]
st = [(h[i], float(v[i])) for i in range(len(h)) if h[i].startswith("smsp__pcsamp_warps_issue_stalled_") and v[i].replace('.', '', 1).isdigit() and not h[i].endswith("_not_issued")]
tot = sum(x for _, x in st) or 1
print("  stalls %:", ", ".join(f"{k.replace('smsp__pcsamp_warps_issue_stalled_', '')} {100 * x / tot:.1f}" for k, x in sorted(st, key=lambda t: -t[1])[:9]))
pp = [(h[i], v[i]) for i in range(len(h)) if "pipe_" in h[i]]
pp = [(k, float(x)) for k, x in pp if x.replace('.', '', 1).isdigit() and float(x) > 3]
print("  pipes %:", ", ".join(f"{k.split('__')[1][:48]} {x:.0f}" for k, x in sorted(pp, key=lambda t: -t[1])[:12]))
EOF
done
