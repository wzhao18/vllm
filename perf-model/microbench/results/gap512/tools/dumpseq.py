import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_parse as K

tr = sys.argv[1]
kj = json.load(open(tr.replace(".trace.json.gz", ".kernels.json")))
ev = json.load(gzip.open(tr, "rt"))["traceEvents"]
acts = sorted((e for e in ev if e.get("cat") in K.GPU_CATS and "dur" in e), key=lambda e: e["ts"])
reps = K.split_replays(acts, kj)
pre, fwd = reps[int(sys.argv[2]) if len(sys.argv) > 2 else 0]
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else len(fwd)
t0 = fwd[0]["ts"]
for i, e in enumerate(fwd[lo:hi], lo):
    a = e.get("args", {})
    print("%4d %9.1f %7.1f s%-4s %-14s %s" % (i, e["ts"] - t0, e["dur"], a.get("stream", e.get("tid")), str(a.get("grid", "")),
                                          K.short(e["name"])[:110]))
