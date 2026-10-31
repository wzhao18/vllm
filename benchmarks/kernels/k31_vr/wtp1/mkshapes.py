"""DCP4xDEP4 / DEP16 / TP16 `vllm bench forward --shapes` files at equal per-GPU request counts.
usage: mkshapes.py <decode_ctx_vectors.json> <out.json> <group_gpus> <q> <set>
Per DP group (one bench-forward batch) B = group_gpus * r requests; AgentX contexts from the
q4_C48_C160_C256 vector of size B (time-weighted in-flight ctx quantiles); uniform cases match W0's
per-GPU (r, tokens/GPU) grid: B = g*r requests of ctx c give (B*c)/g tokens per GPU under DCPg."""
import json, sys
vec = json.load(open(sys.argv[1]))["q4_C48_C160_C256"]
out, g, q, which = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def agx(B):
    if str(B) in vec:
        return vec[str(B)]
    # no vector of that size: resample the 128-request one at B evenly spaced quantiles
    big = sorted(vec["128"]); n = len(big)
    return [big[min(n - 1, int((i + 0.5) * n / B))] for i in range(B)]
def case(name, ctxs):
    return {"name": name, "requests": [{"query_len": q, "kv_len": int(c), "num_draft_tokens": q - 1} for c in ctxs]}
cases = []
if which in ("main", "all"):
    for r in (4, 8, 16):
        B = g * r
        cases.append(case(f"q{q}_r{r}_agx", agx(B)))
    for r, ctx in ((1, 131072), (4, 65536), (8, 32768), (8, 131072), (8, 262144), (16, 65536), (16, 131072)):
        cases.append(case(f"q{q}_r{r}_u{ctx // 1024}k", [ctx] * (g * r)))
if which in ("cap", "all"):
    for r in (20, 24, 32):
        cases.append(case(f"q{q}_r{r}_agx", agx(g * r)))
if which.startswith("x"):  # x<r>: single AgentX case
    for r in which[1:].split(","):
        cases.append(case(f"q{q}_r{r}_agx", agx(g * int(r))))
json.dump({"cases": cases}, open(out, "w"), indent=1)
for c in cases:
    t = sum(x["kv_len"] for x in c["requests"])
    print(c["name"], len(c["requests"]), "Mtok/GPU %.3f" % (t / g / 1e6), "max", max(x["kv_len"] for x in c["requests"]))
