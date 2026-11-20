"""Run fip_ab.py with the persistent KDA kernel module replaced by a file.

usage: fip_var.py KERNEL_FILE|- [fip_ab.py args]   ("-" keeps the installed kernel)"""
import importlib.util
import os
import sys

NAME = "vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.kernel"
kfile = sys.argv.pop(1)
if kfile != "-":
    spec = importlib.util.spec_from_file_location(NAME, kfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod
    spec.loader.exec_module(mod)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fip_ab  # noqa: E402

if kfile != "-":
    import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fw  # noqa: E402

    assert fw._k is sys.modules[NAME], "kernel override not picked up"
fip_ab.main()
