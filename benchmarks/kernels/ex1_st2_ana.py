"""GEMM-EX1 R6 stamp analyzer (12 int32 per CTA at STAMP_OFF: smid, entry, pre-issued, wait-ret,
last TMA issue, first full, acc commit, epi end, last full, blk x|y<<16, rank in cluster, MMA done).
Usage: ex1_st2_ana.py <stamps dir> <min mtime epoch>"""
import glob, os, statistics as stt, sys, collections
import torch
OFF = 2048
def rel(v, t0):
    d = (v - t0) % (1 << 32)
    return (d - (1 << 32) if d >= (1 << 31) else d) / 1000.0
def summarize(buf):
    a = buf.view(torch.int32)[OFF:OFF + 12 * 1024].reshape(1024, 12).to(torch.int64) & 0xFFFFFFFF
    rows = [r.tolist() for r in a if r[1] != 0]
    if not rows: return None
    t0 = min(r[3] for r in rows if r[3])
    w1 = [r for r in rows if (r[10] >> 8) <= 1]
    w2 = [r for r in rows if (r[10] >> 8) > 1]
    def mx(rs, i): v = [rel(r[i], t0) for r in rs if r[i]]; return max(v) if v else float("nan")
    def md(rs, i): v = [rel(r[i], t0) for r in rs if r[i]]; return stt.median(v) if v else float("nan")
    return dict(n=len(rows), w2=len(w2), entry_p50=md(rows, 1), ff_p50=md(w1, 5), lf_max1=mx(w1, 8),
                mma_max1=mx(w1, 11), mma_p50=md(w1, 11), epi_p50=md(rows, 7), epi_max1=mx(w1, 7), epi_max=mx(rows, 7), lf_max=mx(rows, 8))
if __name__ == "__main__":
    d, t = sys.argv[1], float(sys.argv[2])
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(d + "/*.pt")):
        if os.path.getmtime(f) < t: continue
        o = torch.load(f)
        for k, buf in zip(o["keys"], o["data"]):
            s = summarize(buf)
            if s: per[str(k[2]) + " " + str(k[3][1])].append(s)
    for key, ss in sorted(per.items()):
        agg = {m: round(stt.median([s[m] for s in ss]), 2) for m in ss[0]}
        print(key, "sites", len(ss), agg, flush=True)
