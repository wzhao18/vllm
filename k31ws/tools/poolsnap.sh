#!/bin/bash
# poolsnap.sh (Mac): write k31ws/pool.json from Rhea squeue (all my jobs) for the tracker.
D=$(cd "$(dirname "$0")/.." && pwd)
ssh -o BatchMode=yes rhea 'squeue -h -u $USER -o "%i|%j|%T|%D|%S|%L"' 2>/dev/null | grep -v '^\*\*' | python3 -c '
import sys, json
h = []
for l in sys.stdin:
    p = l.strip().split("|")
    if len(p) == 6:
        h.append(dict(job=p[0], name=p[1], state=p[2], nodes=p[3], start=p[4], left=p[5]))
json.dump({"note": "Rolling holds from holdd-ws.sh (1-node class for kernel work, one standing 8-node request for EP32/DP8). GPUs are shared through wsgpu.sh locks.", "holds": h}, open(sys.argv[1], "w"), indent=1)
' $D/pool.json
