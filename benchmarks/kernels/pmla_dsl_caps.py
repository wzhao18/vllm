# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print the CuTe DSL capabilities the prefill sparse MLA kernel depends on."""

import inspect
import os
import re

import cutlass
import cutlass.cute as cute

print("cutlass-dsl", getattr(cutlass, "__version__", "?"), cutlass.__file__)
root = os.path.dirname(cutlass.__file__)
hits = {}
for dirpath, _, files in os.walk(os.path.dirname(root)):
    if "cutlass" not in dirpath:
        continue
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        try:
            s = open(p, errors="ignore").read()
        except OSError:
            continue
        for key in ("OVERSIZED", "is_exclusive", "SHARED_MEMORY_MODE", "gather4",
                    "griddepcontrol", "redux"):
            if key in s:
                for m in re.finditer(r".{0,80}" + key + r".{0,80}", s):
                    hits.setdefault(key, []).append(f"{p}: {m.group(0).strip()}")
for k, v in hits.items():
    print("==", k, len(v))
    for line in v[:8]:
        print("  ", line)
try:
    from cutlass.experimental import primitives as nvvm

    print("tcgen05_alloc sig:", inspect.signature(nvvm.tcgen05_alloc))
except Exception as e:  # noqa: BLE001
    print("tcgen05_alloc:", repr(e))
try:
    print("launch sig:", inspect.signature(cute.core._KernelGenerationContext))
except Exception:
    pass
for name in ("launch",):
    for obj in (getattr(cutlass.cutlass_dsl, "KernelLauncher", None),):
        if obj is not None and hasattr(obj, name):
            print(name, inspect.signature(getattr(obj, name)))
