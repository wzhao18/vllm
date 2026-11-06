"""nq_shapes.py <q> <out.json> --rows 8,16,... [--ctx agentx,u192k] [--group auto|<name>] [--point c256]:
`vllm bench forward --shapes` cases for the nq step A/B: B = rows/q requests of q verify tokens
(q-1 drafts) per case, ascending rows (a case that fails late loses only the larger rows).

Contexts (b1/r2x/tools/make_shapes.py, plus the q8 c1/c8 group the audit asks for):
  agentx  AgentX in-flight contexts: decode_ctx_vectors.json group q8_C1_C8 (q8 c1/c8),
          q8_C64 (other q8 points) or q4_C48_C160_C256 (q4); B values without a vector take
          quantiles of the B=128 vector
  uNk     uniform N*1024 tokens (u192k)
  mix     alternating 3K / 150K (rows with no local top-k entry on some ranks)

--max-kv-tokens N (0 = off): a case whose contexts sum to more than N tokens has every context
scaled by N / sum (shape of the distribution kept), so B x ctx stays within what the decode
server's KV holds. Both arms read the same file. Each case's context summary is printed and
stored as the case's "ctx" entry; nq_step.py also checks every case against the KV capacity
the server would have and refuses one it would not admit.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("q", type=int)
ap.add_argument("out")
ap.add_argument("--rows", required=True)
ap.add_argument("--ctx", default="agentx")
ap.add_argument("--group", default="auto")
ap.add_argument("--point", default="")
ap.add_argument("--max-kv-tokens", type=int, default=0)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
vec = json.load(open(os.path.join(here, "decode_ctx_vectors.json")))
group = a.group
if group == "auto":
    if a.q == 8:
        group = "q8_C1_C8" if a.point in ("c1", "c8") else "q8_C64"
    else:
        group = "q4_C48_C160_C256"
ref = sorted(vec[group]["128"])
rows = sorted({int(r) for r in a.rows.split(",") if r})
bad = [r for r in rows if r % a.q]
assert not bad, f"rows {bad} are not multiples of q={a.q}"


def agentx(b):
    if str(b) in vec[group]:
        return sorted(vec[group][str(b)])
    n = len(ref)
    return [ref[min(n - 1, int((i + 0.5) * n / b))] for i in range(b)]


cases = []
for ctx in a.ctx.split(","):
    for m in rows:
        b = m // a.q
        if ctx == "agentx":
            lens = agentx(b)
        elif ctx.startswith("u") and ctx.endswith("k"):
            lens = [int(ctx[1:-1]) * 1024] * b
        elif ctx == "mix":
            lens = [(3000 if i % 2 == 0 else 150 * 1024) for i in range(b)]
        else:
            raise ValueError(ctx)
        lens = [max(k, 2 * a.q) for k in lens]
        scale = 1.0
        if a.max_kv_tokens and sum(lens) > a.max_kv_tokens:
            scale = a.max_kv_tokens / sum(lens)
            lens = [max(int(k * scale), 2 * a.q) for k in lens]
        reqs = [{"query_len": a.q, "kv_len": k + a.q, "num_draft_tokens": a.q - 1} for k in lens]
        s = sorted(lens)
        stats = {"B": b, "sum": sum(s), "min": s[0], "p50": s[b // 2], "mean": sum(s) // b,
                 "max": s[-1], "scale": round(scale, 4)}
        cases.append({"name": f"q{a.q}_B{b}_M{m}_{ctx}", "ctx": stats, "requests": reqs})
json.dump({"cases": cases}, open(a.out, "w"), indent=1)
print(f"{len(cases)} cases (group {group}):", " ".join(c["name"] for c in cases))
for c in cases:
    print(f"ctx {c['name']}: " + " ".join(f"{k} {v}" for k, v in c["ctx"].items()))
