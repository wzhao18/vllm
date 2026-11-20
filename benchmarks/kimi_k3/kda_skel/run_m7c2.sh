#!/bin/bash
# P-KDA2 M7a producer stamps (skel7) + C2 math calibration (fip prep-math ablation timing + ncu inst counts).
# usage (inside the image, overlay with this checkout): run_m7c2.sh <ovdir> <outdir>
V=$1; O=$2; mkdir -p $O; S=$V/benchmarks/kimi_k3/kda_skel; B=$V/benchmarks/kimi_k3
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
echo M7A_START
$NVCC $F -Xptxas -v -o $O/skel7 $S/skel7.cu -lcuda > $O/build7.log 2>&1 || { echo BUILD7_FAIL; grep -m8 error $O/build7.log; }
grep -E "spill|registers" $O/build7.log | head -2
for fl in 2064 6160 2128 16 4112; do timeout 90 $O/skel7 116 $fl 5 | grep -E "RESULT|STAMPS|PRODUCER|TIMEOUT" | sed 's/ | HBM.*//' || echo "FAIL fl=$fl"; done
echo C2_TIME_START
cd $O
timeout 1500 python $B/bench_fip_abl.py --lens 5461,5461,5462 --probe PREPABL=1 --probe PREPABL=2 --probe PREPABL=4 --probe PREPABL=8 --probe PREPABL=15 2>&1 | grep -E "lens=|Error|error" | head -20
echo C2_NCU_START
timeout 1800 ncu -k regex:_pkd -c 84 --metrics smsp__inst_executed.sum,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_fma.sum --csv --print-units base \
  python $B/bench_fip_abl.py --lens 5461,5461,5462 --probe PREPABL=1 --probe PREPABL=2 --probe PREPABL=4 --probe PREPABL=8 --probe PREPABL=15 > $O/ncu.csv 2> $O/ncu.err
python - $O/ncu.csv <<'PY'
import csv, sys, collections
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 10]
h = rows[0]; iid = h.index("ID"); im = h.index("Metric Name"); iv = h.index("Metric Value")
d = collections.defaultdict(dict)
for r in rows[1:]:
    try: d[int(r[iid])][r[im]] = float(r[iv].replace(",", ""))
    except ValueError: pass
names = ["old", "ABL1", "ABL2", "ABL4", "ABL8", "ABL15"]
for vi, n in enumerate(names):
    ks = [k for k in sorted(d) if k // 14 == vi][4:]
    if not ks: continue
    avg = lambda m: sum(d[k].get(m, 0) for k in ks) / len(ks)
    print(f"NCU {n}: inst {avg('smsp__inst_executed.sum')/1e6:.1f}M fma {avg('smsp__inst_executed_pipe_fma.sum')/1e6:.1f}M lsu {avg('sm__inst_executed_pipe_lsu.sum')/1e6:.1f}M time {avg('gpu__time_duration.sum')/1e3:.1f} us(ns base) clk {avg('sm__cycles_elapsed.avg.per_second')/1e9:.2f} GHz (n={len(ks)})")
PY
tail -3 $O/ncu.err | cut -c1-200
echo ALL_DONE
