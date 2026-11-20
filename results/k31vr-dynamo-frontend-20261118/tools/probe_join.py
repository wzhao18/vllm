"""probe_join.py <run dir>: join fe probe records (probe-*.jsonl, keyed by Rust request id) with the frontend logs'
`request received` lines and report, per 2-minute window: requests entering Python, and percentiles of
  handoff = t_enter - received (Rust HTTP -> Python coroutine start), pre = t_pre - t_enter (Python preprocessing),
  first = t_first - t_pre (engine call -> first chunk back in Python)."""
import glob, json, re, sys, collections
from datetime import datetime
d = sys.argv[1]
rx = re.compile(rb'^(\S+Z).*request received request_id=([0-9a-f-]+)'); ansi = re.compile(rb'\x1b\[[0-9;]*m')
recv = {}
for p in sorted(glob.glob(d + "/fe_*.out")) + sorted(glob.glob(d + "/logs/*frontend_*.out")):
    for line in open(p, "rb"):
        if b"request received" not in line: continue
        m = rx.match(ansi.sub(b"", line))
        if m: recv.setdefault(m.group(2).decode(), datetime.strptime(m.group(1).decode()[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp())
recs = []
for p in glob.glob(d + "/probe-*.jsonl") + glob.glob(d + "/logs/probe/probe-*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if "rid" in r: recs.append(r)
print(f"{len(recv)} received, {len(recs)} probe records, {sum(1 for r in recs if r['rid'] in recv)} joined, "
      f"{sum(1 for r in recs if r['t_pre'] is None)} without t_pre, {sum(1 for r in recs if r['t_first'] is None)} without t_first")
q = lambda a, p: a[int(p * (len(a) - 1))] if a else float("nan")
win = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    w = int(r["t_enter"] // 120) * 120
    if r["rid"] in recv: win[w]["handoff"].append(r["t_enter"] - recv[r["rid"]])
    if r["t_pre"]: win[w]["pre"].append(r["t_pre"] - r["t_enter"])
    if r["t_pre"] and r["t_first"]: win[w]["first"].append(r["t_first"] - r["t_pre"])
    win[w]["n"].append(1)
for w in sorted(win):
    s = f"{datetime.fromtimestamp(w).strftime('%H:%M')} n={len(win[w]['n']):5d}"
    for k in ("handoff", "pre", "first"):
        v = sorted(win[w][k]); s += f" | {k} p50 {q(v,.5):7.3f} p90 {q(v,.9):7.3f} max {v[-1] if v else float('nan'):7.2f}"
    print(s)
