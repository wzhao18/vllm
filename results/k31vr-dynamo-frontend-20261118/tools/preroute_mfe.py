"""preroute_mfe.py <run dir>...: Dynamo frontend pre-route delay (HTTP `request received` -> prefill `Selected worker`) per
frontend process and combined, for single- and multi-frontend runs (all logs/*frontend_*.out). Requests in the first 40 %
of the run (warmup / primers) are dropped, as in preroute.py. Rates are per frontend over its own request span."""
import glob, re, statistics as st, sys
from datetime import datetime

rx_r = re.compile(rb'^(\S+Z).*request received request_id=([0-9a-f-]+)')
rx_s = re.compile(rb'^(\S+Z).*Selected worker.*? request_id="([0-9a-f-]+)"')
ansi = re.compile(rb'\x1b\[[0-9;]*m')
def ts(b): return datetime.strptime(b.decode()[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
import os
SKIP = float(os.environ.get('PREROUTE_SKIP', '0.4'))
q = lambda a, p: a[int(p * (len(a) - 1))] if a else float("nan")

def parse(path):
    recv, sel = {}, {}
    with open(path, "rb") as f:
        for line in f:
            if b"request received" not in line and b"Selected worker" not in line: continue
            line = ansi.sub(b"", line)
            m = rx_r.match(line)
            if m: recv.setdefault(m.group(2), ts(m.group(1))); continue
            m = rx_s.match(line)
            if m and m.group(2) not in sel: sel[m.group(2)] = ts(m.group(1))
    return recv, sel

for d in sys.argv[1:]:
    fs = sorted(glob.glob(d + "/logs/*frontend_*.out")) or sorted(glob.glob(d + "/fe_*.out"))
    allr, alls = {}, {}
    print(d.rstrip("/").split("/")[-1])
    for p in fs:
        recv, sel = parse(p)
        if not recv: print(f"  {p.split('/')[-1]:34s} no requests"); continue
        allr.update(recv); alls.update(sel)
        keep = sorted(recv, key=recv.get)[int(len(recv) * SKIP):]
        dl = sorted(sel[k] - recv[k] for k in keep if k in sel)
        t0, t1 = min(recv.values()), max(recv.values())
        print(f"  {p.split('/')[-1]:34s} n={len(recv):6d} rate={len(recv)/(t1-t0):5.2f}/s mean={st.mean(dl):6.3f}s p50={q(dl,.5):6.3f} p90={q(dl,.9):6.3f} p99={q(dl,.99):6.3f}")
    if allr:
        keep = sorted(allr, key=allr.get)[int(len(allr) * SKIP):]
        dl = sorted(alls[k] - allr[k] for k in keep if k in alls)
        t0, t1 = min(allr.values()), max(allr.values())
        print(f"  {'ALL':34s} n={len(allr):6d} rate={len(allr)/(t1-t0):5.2f}/s mean={st.mean(dl):6.3f}s p50={q(dl,.5):6.3f} p90={q(dl,.9):6.3f} p99={q(dl,.99):6.3f}")
