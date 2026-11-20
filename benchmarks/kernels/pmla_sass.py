# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SASS facts of h8 variants: local-memory ops (spills / stack), instruction counts,
MUFU / F2FP / BAR counts. Compiles each knob list with CUTE_DSL_KEEP_CUBIN into its own
dump dir, then cuobjdump -sass / -res-usage.
usage: python pmla_sass.py <outdir> <knobs a:b:...> [<knobs> ...]"""

import glob
import os
import re
import subprocess
import sys

out = sys.argv[1]
for ks in sys.argv[2:]:
    d = os.path.join(out, "k" + ks.replace(":", "_"))
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUTE_DSL_KEEP_CUBIN="1", CUTE_DSL_KEEP_PTX="1",
               CUTE_DSL_DUMP_DIR=d)
    code = ("from vllm.models.kimi_k3.nvidia.specialized_prefill import sparse_mla_h8 as m;"
            f"m._compiled(212, tuple(int(v) for v in '{ks}'.split(':')))")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True)
    if r.returncode:
        print(ks, "compile failed:", r.stderr[-600:], flush=True)
        continue
    cubins = sorted(glob.glob(os.path.join(d, "**", "*.cubin"), recursive=True),
                    key=os.path.getmtime)
    if not cubins:
        print(ks, "no cubin in", d, os.listdir(d)[:8], flush=True)
        continue
    cb = cubins[-1]
    sass = subprocess.run(["cuobjdump", "-sass", cb], capture_output=True,
                          text=True).stdout
    res = subprocess.run(["cuobjdump", "-res-usage", cb], capture_output=True,
                         text=True).stdout
    ins = [ln for ln in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln)]
    cnt = {}
    for op in ("STL", "LDL", "MUFU.EX2", "F2FP", "BAR", "LDTM", "STS", "LDS",
               "SYNCS", "FFMA2", "FADD2"):
        cnt[op] = sum(1 for ln in ins if re.search(r"\b" + re.escape(op), ln))
    ru = " ".join(re.findall(r"(REG:\d+|STACK:\d+|LOCAL:\d+|SHARED:\d+)", res))
    print(f"{ks}: sass_instr={len(ins)} {ru} " +
          " ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)
    with open(os.path.join(d, "sass.txt"), "w") as f:
        f.write(sass)
