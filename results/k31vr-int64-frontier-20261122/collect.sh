#!/bin/bash
# collect.sh <label> <topology> <agentx outdir on Hecate> [<gsm8k host> <gsm8k outdir>]: copy one AgentX point (+ its gsm8k
# twin; logs only, no dloan/otel dumps) into runs/<label>-a<job>/. Adapted from ../k31vr-release-draft-sweep-20261117/collect.sh:
#  - metrics.json: frz_metrics.py (unchanged), full profiling phase, run on Hecate via stdin;
#  - metrics_win10_60.json: win_metrics.py, minutes 10-60 (ledger convention; both 'end' and 'start' windowings);
#  - the gsm8k twin may live on another cluster (the f19 twin ran on Rhea), so its host is an argument;
#  - two extra excludes: aiperf-runtime/ (the client's uv venv + python, ~800 MB) and logs/iterlog/ (per-iteration engine
#    logs, ~90 MB of small files per run; kept on Hecate).
# rsync -rlt, not -a: the Lustre dirs are setgid (drwxrwsr-x) and the Mac rsync cannot set that bit.
# Run from this directory on the Mac.
set -eo pipefail
LABEL=$1; TOPO=$2; AGX=$3; EVH=$4; EV=$5
H=$(cd "$(dirname "$0")" && pwd); J=$(basename "$AGX" | sed -E 's/^a([0-9]+)-.*/\1/')
R=$H/runs/${LABEL// /-}-a$J; mkdir -p $R/agentx $R/gsm8k
r() { for i in 1 2 3 4 5; do rsync -rlt $X "$@" && return 0; echo "rsync retry $i" >&2; sleep 20; done; return 1; }
X="--exclude dloan --exclude otel --exclude tachometer --exclude *_timeslices.* --exclude __pycache__ --exclude aiperf-runtime --exclude iterlog --max-size=20m"
r hecate:$AGX/ $R/agentx/; echo "source: hecate:$AGX" > $R/agentx/SOURCE.txt
ssh -o BatchMode=yes hecate "python3 - $AGX $LABEL" < $H/frz_metrics.py | tail -1 > $R/metrics.json
ssh -o BatchMode=yes hecate "python3 - $AGX $LABEL" < $H/win_metrics.py | tail -1 > $R/metrics_win10_60.json
if [ -n "$EV" ]; then
  r $EVH:$EV/ $R/gsm8k/; echo "source: $EVH:$EV" > $R/gsm8k/SOURCE.txt
  python3 - "$R" "$EVH:$EV" <<'PY'
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
