"""Generate spread-poller variants: m1s.py from m1.py, d1s.py from d1.py.

Prep instances poll MB_RFREE / MB_SFREE (and MB_BRAW for beta) from fixed warps
(plw 0 / 2), which puts all five instances' pollers on the same SMSP.  The 's'
variants rotate the polling/TMA-issuing warp with the instance index."""
import sys

HERE = sys.argv[1] if len(sys.argv) > 1 else "."


def spread(src):
    lines = src.split("\n")
    out = []
    hits = {"rfree": 0, "sfree": 0, "qtma": 0, "beta": 0}
    for i, ln in enumerate(lines):
        nxt = "\n".join(lines[i + 1:i + 4])
        s = ln.strip()
        if s == "if plw == 0:" and "MB_RFREE" in nxt:
            ln = ln.replace("plw == 0", "plw == (inst & 3)"); hits["rfree"] += 1
        elif s == "if plw == 0:" and "MB_SFREE" in nxt:
            ln = ln.replace("plw == 0", "plw == (inst & 3)"); hits["sfree"] += 1
        elif s == "if plw == 0:" and "f_qr = cute.make_tensor(" in nxt:
            ln = ln.replace("plw == 0", "plw == (inst & 3)"); hits["qtma"] += 1
        elif s == "if plw == 2:" and ("MB_BRAW" in nxt or "BETA_TMA_" in nxt):
            ln = ln.replace("plw == 2", "plw == ((inst + 2) & 3)"); hits["beta"] += 1
        out.append(ln)
    return "\n".join(out), hits


for a, b in (("m1.py", "m1s.py"), ("d1.py", "d1s.py")):
    s, h = spread(open(f"{HERE}/{a}").read())
    open(f"{HERE}/{b}", "w").write(s)
    print(b, h)
