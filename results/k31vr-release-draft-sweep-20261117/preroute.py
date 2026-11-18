import re, sys, glob, statistics as st
from datetime import datetime
rx_r = re.compile(rb'^(\S+Z).*request received request_id=([0-9a-f-]+)')
rx_s = re.compile(rb'^(\S+Z).*Selected worker.*? request_id="([0-9a-f-]+)"')
def ts(b): return datetime.strptime(b.decode()[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
for d in sys.argv[1:]:
    fs = glob.glob(d + "/logs/*frontend_0.out")
    if not fs: continue
    recv, sel = {}, {}
    with open(fs[0], "rb") as f:
        for line in f:
            line = re.sub(rb'\x1b\[[0-9;]*m', b'', line)
            m = rx_r.match(line)
            if m: recv.setdefault(m.group(2), ts(m.group(1))); continue
            m = rx_s.match(line)
            if m and m.group(2) not in sel: sel[m.group(2)] = ts(m.group(1))
    dl = sorted(sel[k] - recv[k] for k in sel if k in recv)
    if not dl: print(d.split('/')[-1], "no data"); continue
    t0, t1 = min(recv.values()), max(recv.values())
    # profiling-ish window: drop first 40% of requests (warmup)
    dl2 = sorted(sel[k] - recv[k] for k in sorted(recv, key=recv.get)[int(len(recv)*0.4):] if k in sel)
    q = lambda a, p: a[int(p*(len(a)-1))]
    print(f"{d.split('/')[-1][:48]:48s} n={len(dl2):6d} rate={len(recv)/(t1-t0):5.1f}/s mean={st.mean(dl2):6.2f}s p50={q(dl2,.5):6.2f} p90={q(dl2,.9):6.2f}")
