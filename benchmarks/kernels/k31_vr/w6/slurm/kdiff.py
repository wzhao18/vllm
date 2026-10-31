import json,sys
R=sys.argv[1]
def L(p):
    d=json.load(open(p))
    return d["kernels"] if isinstance(d,dict) else d
for case in sys.argv[2:]:
    a={k["name"]:k["us"] for k in L(f"{R}/prof_stock/{case}.rank0.kernels.json")}
    b={k["name"]:k["us"] for k in L(f"{R}/prof_m5rf/{case}.rank0.kernels.json")}
    print("==",case,"total",round(sum(a.values())),round(sum(b.values())))
    ks=sorted(set(a)|set(b),key=lambda k:-abs(b.get(k,0)-a.get(k,0)))
    for k in ks[:10]: print(f"{a.get(k,0):9.1f} {b.get(k,0):9.1f} {b.get(k,0)-a.get(k,0):+8.1f}  {k[:90]}")
