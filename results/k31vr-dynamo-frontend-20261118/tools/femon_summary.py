"""femon_summary.py <femon-*.tsv>... [--from EPOCH] [--to EPOCH]: per frontend process, the busiest threads' CPU utilization
(cores) between consecutive 20 s samples, averaged over the window; process total too. Input from fe_mon.sh."""
import collections, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
t_from = float(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
t_to = float(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else 1e18
args = [a for a in args if not a.replace(".", "").isdigit()]
HZ = 100
for path in args:
    samp = collections.defaultdict(dict)  # (host,pid) -> t -> {tid: (comm, jiff)}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) != 6: continue
        t, h, pid, tid, comm, j = float(p[0]), p[1], p[2], p[3], p[4], int(p[5])
        if not (t_from <= t <= t_to): continue
        samp[(h, pid)].setdefault(t, {})[tid] = (comm, j)
    for (h, pid), ts in samp.items():
        times = sorted(ts)
        if len(times) < 2: continue
        dt = times[-1] - times[0]
        first, last = ts[times[0]], ts[times[-1]]
        util = {tid: ((last[tid][1] - first[tid][1]) / HZ / dt, last[tid][0]) for tid in last if tid in first}
        tot = sum(u for u, _ in util.values())
        # peak 20 s window of the busiest thread
        top = sorted(util.items(), key=lambda x: -x[1][0])[:4]
        best = top[0][0]
        peak = max(((ts[b][best][1] - ts[a][best][1]) / HZ / (b - a)) for a, b in zip(times, times[1:]) if best in ts[a] and best in ts[b])
        kind = "frontend" if any(c.startswith("python") or c.startswith("pt_") for _, c in util.values()) or tid else "?"
        print(f"{h} pid {pid}: {dt/60:5.1f} min, process {tot:5.2f} cores; top threads " +
              ", ".join(f"{c}:{u:.2f}" for _, (u, c) in top) + f"; busiest-thread peak 20s {peak:.2f}")
