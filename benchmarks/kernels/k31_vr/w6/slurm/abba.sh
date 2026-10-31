#!/bin/bash
# W6 ABBA step bench. usage: abba.sh <jobid> <nnodes: 1 (8-layer DEP4) | 4 (full DEP16)> <q: 4|8> <armA> <armB> <shapes json> <outdir> [extra args]
# Runs A B B A (each arm a fresh process); profiles only in the first A and B.
D=$(dirname $(readlink -f $0)); source $D/common.sh
J=$1; NN=$2; Q=$3; A=$4; B=$5; SH=$6; R=$7; shift 7
mkdir -p $R
if [ $NN = 1 ]; then export K31W0_LAYERS=8; DP=4; else DP=16; fi
K=$((Q-1)); i=0
for arm in $A $B $B $A; do
  i=$((i+1)); read tag topk acfg <<< "${ARM[$arm]}"
  prof=(); [ $i -le 2 ] && prof=(--profile-dir $R/prof_${arm})
  bash $D/launch.sh $J $NN $tag $R/$i-$arm.log "${COMMON[@]}" --data-parallel-size $DP \
    --kernel-config "$(kcfg $topk)" --attention-config "$acfg" \
    --speculative-config "$(spec $K)" --gpu-memory-utilization ${GMU:-0.90} --routing balanced \
    --shapes $SH --output-json $R/$i-$arm.json "${prof[@]}" "$@"
done
python3 - $R $A $B <<'PY'
import json, sys, glob, os
R, A, B = sys.argv[1:4]; res = {}
for f in sorted(glob.glob(f"{R}/[1-4]-*.json")):
    arm = os.path.basename(f)[2:-5]
    for c in json.load(open(f)):
        res.setdefault(c["case"], {}).setdefault(arm, []).append(c["median_us"] / 1000)
arms = [A, B]
print("case", *[f"{a}(ms)" for a in arms], "B/A-1 %", sep="\t")
for case, v in res.items():
    m = {a: sum(x) / len(x) for a, x in v.items()}
    d = (m[B] / m[A] - 1) * 100 if len(m) == 2 else 0
    print(case, *[f"{m[a]:.3f} ({','.join(f'{x:.2f}' for x in v[a])})" for a in arms], f"{d:+.1f}", sep="\t")
PY
