# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ptxas resource usage (registers, spills) of the specialized_prefill sparse-MLA kernels.
Compiles with CUTE_DSL_KEEP_PTX/CUBIN into a dump dir, then runs ptxas -v on each PTX.
usage: CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=<dir> python pmla_ptxas.py <dir> [maxrreg] [h2|h4|h5] [knobs a:b:c]"""
import glob
import os
import subprocess
import sys

d = sys.argv[1]
maxr = sys.argv[2] if len(sys.argv) > 2 else "80"
import importlib  # noqa: E402

name = sys.argv[3] if len(sys.argv) > 3 else "h2"
mod = importlib.import_module(f"vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_{name}")
knobs = tuple(int(v) for v in sys.argv[4].split(":")) if len(sys.argv) > 4 else ()
fn = mod._compiled(212, False, knobs)
for a in dir(fn):
    if "ptx" in a.lower() or "cubin" in a.lower() or "sass" in a.lower():
        print("attr", a, flush=True)
for a in ("__sass__", "__ptx__"):
    try:
        v = getattr(fn, a)
        v = v() if callable(v) else v
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        open(os.path.join(d, name + a.strip("_") + ".txt"), "w").write(str(v))
        print("wrote", a, len(str(v)), flush=True)
    except Exception as e:  # noqa: BLE001
        print("no", a, repr(e)[:300], flush=True)
ptxs = glob.glob(os.path.join(d, "**", "*.ptx"), recursive=True)
print("ptx files", ptxs, flush=True)
for p in ptxs:
    for arch in ("sm_107a",):
        r = subprocess.run(["ptxas", f"-arch={arch}", "-v", f"-maxrregcount={maxr}", p, "-o",
                            p + ".cubin"], capture_output=True, text=True)
        print(p, arch, r.returncode, r.stdout[-3000:], r.stderr[-3000:], flush=True)
