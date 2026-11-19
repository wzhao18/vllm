# D-KDA calibration helper: aggregate an ncu mixed source page (--print-source cuda,sass --csv) by the
# enclosing python function of each SASS instruction's source line (flow.py), plus per-function opcode mix.
# usage: python3 dkda_srcagg.py flow.mix.csv flow.py
import ast
import collections
import csv
import sys

spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(ast.parse(open(sys.argv[2]).read()))
         if isinstance(n, ast.FunctionDef)]


def fn(line):
    best = None
    for a, b, name in spans:
        if a <= line <= b and (best is None or a > best[0]):
            best = (a, name)
    return best[1] if best else "other"


num = lambda x: float(x.replace(",", "")) if x.strip() not in ("", "-") else 0.0
agg = collections.defaultdict(lambda: [0.0, 0.0, collections.Counter()])
cur_file, cur, hdr = "", "other", None
for r in csv.reader(open(sys.argv[1])):
    if not r:
        continue
    if r[0] == "File Path":
        cur_file = r[1].rsplit("/", 1)[-1]
        continue
    if r[0] == "Line No":
        hdr = r
        ki, ii = r.index("Warp Stall Sampling (All Samples)"), r.index("Instructions Executed")
        continue
    if hdr is None or len(r) <= ii:
        continue
    if r[0].strip().isdigit():
        cur = fn(int(r[0])) if cur_file == "flow.py" else cur_file
        continue
    if r[2].startswith("0x"):
        a = agg[cur]
        n = num(r[ii])
        a[0] += n
        a[1] += num(r[ki])
        op = r[3].strip().split()
        op = [t for t in op if not t.startswith("@")]
        if op:
            a[2][op[0].split(".")[0]] += n
tot = [sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values())]
print("total warp-instr %.0f stall-samples %.0f" % tuple(tot))
for f, (n, s, ops) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
    top = " ".join("%s:%.0f%%" % (o, 100 * c / max(n, 1)) for o, c in ops.most_common(5))
    print("%-16s instr %9.0f (%4.1f%%) stalls %4.1f%%  %s" % (f[:16], n, 100 * n / max(tot[0], 1),
                                                            100 * s / max(tot[1], 1), top))
