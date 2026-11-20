"""Run bench_fip_devb.py with the persistent KDA modules replaced from files.

usage: bench_fip_var.py KERNEL_FILE|DIR|- [bench_fip_devb.py args]
DIR holds kernel.py, forward.py and kda.py (a whole other code tree arm).
"""
import importlib
import importlib.util
import os
import sys

BASE = "vllm.models.kimi_k3.nvidia.specialized_prefill"
kfile = sys.argv.pop(1)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, attr = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], attr, mod)


if os.path.isdir(kfile):
    importlib.import_module(BASE + ".kda_persistent")
    _load(BASE + ".kda_persistent.kernel", os.path.join(kfile, "kernel.py"))
    _load(BASE + ".kda_persistent.forward", os.path.join(kfile, "forward.py"))
    _load(BASE + ".kda", os.path.join(kfile, "kda.py"))
elif kfile != "-":
    _load(BASE + ".kda_persistent.kernel", kfile)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bench_fip_devb  # noqa: E402

print(f"KERNEL {kfile}", flush=True)
bench_fip_devb.main()
