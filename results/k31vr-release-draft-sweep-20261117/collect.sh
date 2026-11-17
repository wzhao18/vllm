#!/bin/bash
# collect.sh <label> <topology> <agentx outdir on Hecate> <gsm8k outdir on Hecate>: copy one release-draft sweep point
# (AgentX + its gsm8k twin; logs only, no dloan/otel dumps) into runs/<label>-a<job>/, compute metrics.json with
# frz_metrics.py (the frozen sweep's metric code, unchanged) and gsm8k/score.json (strict/flexible + real AL from the
# decode SpecDecoding counters). Run from this directory on the Mac.
set -eo pipefail
LABEL=$1; TOPO=$2; AGX=$3; EV=$4
H=$(cd "$(dirname "$0")" && pwd); J=$(basename "$AGX" | sed -E 's/^a([0-9]+)-.*/\1/')
R=$H/runs/$LABEL-a$J; mkdir -p $R/agentx $R/gsm8k
X="--exclude dloan --exclude otel --exclude tachometer --exclude *_timeslices.* --exclude __pycache__ --max-size=20m"
rsync -a $X hecate:$AGX/ $R/agentx/; echo "source: $AGX" > $R/agentx/SOURCE.txt
ssh hecate "python3 /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/dsrel/scratch/frz_metrics.py $AGX $LABEL" | tail -1 > $R/metrics.json
if [ -n "$EV" ]; then
  rsync -a $X hecate:$EV/ $R/gsm8k/; echo "source: $EV" > $R/gsm8k/SOURCE.txt
  python3 - "$R" "$EV" <<'PY'
import glob, json, re, sys
r, ev = sys.argv[1:3]
txt = "".join(open(f, errors="replace").read() for f in glob.glob(f"{r}/gsm8k/logs/eval*.out") + glob.glob(f"{r}/gsm8k/logs/eval_results/**/*.txt", recursive=True))
def score(kind):
    m = re.findall(rf"\|\s*{kind}\s*\|\s*\d+\|exact_match\|[^|]*\|([0-9.]+)", txt)
    return float(m[-1]) if m else None
acc = dr = 0
for f in glob.glob(f"{r}/gsm8k/logs/*decode_w0.out"):
    for a, d in re.findall(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens", open(f, errors="replace").read()):
        acc += int(a); dr += int(d)
cfg = open(f"{r}/gsm8k/config.yaml").read()
k = int(re.search(r'"num_speculative_tokens":\s*(\d+)', cfg).group(1))
al = round(1 + acc / (dr / k), 3) if dr else None
json.dump(dict(strict=score("strict-match"), flexible=score("flexible-extract"), real_al=al, k=k, accepted=acc, drafted=dr,
               run=ev, rejection="real (no synthetic acceptance)"), open(f"{r}/gsm8k/score.json", "w"))
print(open(f"{r}/gsm8k/score.json").read())
PY
fi
[ -f $R/meta.json ] || echo "{\"label\": \"$LABEL\", \"topology\": \"$TOPO\", \"chosen\": true, \"note\": \"\"}" > $R/meta.json
cat $R/metrics.json; du -sh $R
