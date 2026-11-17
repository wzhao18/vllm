#!/usr/bin/env python3
"""conc_variant.py <render dir> <conc>: turn a generated AgentX render (gen_dsrel.py output) into the same deployment at
another client concurrency. Only workflow.env.resolved changes: CONC, CONC_LIST and the _c<old>_ token of EXP_NAME /
RESULT_FILENAME; the engines (max-num-seqs, CUDA graphs, KV, Mooncake) are untouched, so decode max-num-seqs must
already cover the new CONC. Appends the change to DSREL.txt. Refuses gsm8k renders (their deployment needs no change)."""
import os, re, sys
d, conc = sys.argv[1], int(sys.argv[2])
w = os.path.join(d, "workflow.env.resolved")
t = open(w).read()
old = int(re.search(r"^CONC=(\d+)$", t, re.M).group(1))
assert "gsm8k" not in open(os.path.join(d, "DSREL.txt")).read().split("\n")[0] and "-ev-" not in os.path.basename(d.rstrip("/")), "agentx renders only"
assert re.search(r"^CONC_LIST=%d$" % old, t, re.M), "CONC_LIST != CONC"
n_tok = len(re.findall(r"_c%d_" % old, t))
t = re.sub(r"^CONC=%d$" % old, "CONC=%d" % conc, t, flags=re.M)
t = re.sub(r"^CONC_LIST=%d$" % old, "CONC_LIST=%d" % conc, t, flags=re.M)
t = t.replace("_c%d_" % old, "_c%d_" % conc)
open(w, "w").write(t)
with open(os.path.join(d, "DSREL.txt"), "a") as f:
    f.write(f"  conc_variant.py: CONC/CONC_LIST {old} -> {conc}, _c{old}_ -> _c{conc}_ x{n_tok} in workflow.env.resolved; "
            "engines unchanged\n")
print(f"{d}: CONC {old} -> {conc} ({n_tok} name tokens)")
