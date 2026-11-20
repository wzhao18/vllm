"""Prep phase stamps of a stamped PREP_ONLY kernel variant (see kvar/mkstamp.py).

usage: prep_stamp.py KERNEL_FILE [--shape f16k]"""
import argparse
import importlib.util
import os
import sys

NAME = "vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.kernel"
ap = argparse.ArgumentParser()
ap.add_argument("kfile")
ap.add_argument("--shape", default="f16k")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location(NAME, a.kfile)
mod = importlib.util.module_from_spec(spec)
sys.modules[NAME] = mod
spec.loader.exec_module(mod)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import fip_ab  # noqa: E402
import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fw  # noqa: E402

fw._TPROBE = 1
fn, reset = fip_ab.runner(fip_ab.SHAPES[a.shape])
for _ in range(3):
    reset()
    fn()
torch.cuda.synchronize()
t = fw.LAST_TPROBE[0].cpu().view(-1, 64)[:, :60].view(-1, 5, 12)[:, :, :9].double()
ok = (t[:, :, 0] > 0) & (t[:, :, 8] > t[:, :, 0])
d = t[:, :, 1:] - t[:, :, :-1]
names = ["rfree", "sfree+ib", "gate", "decor", "gram_wait", "gram_st", "solve_w0", "tail"]
print(f"{a.shape}: {int(ok.sum())} instance samples")
for inst in range(5):
    m = ok[:, inst]
    if m.sum() == 0:
        continue
    med = d[m, inst].median(dim=0).values
    tot = (t[m, inst, 8] - t[m, inst, 0]).median().item()
    print(f"inst{inst} total {tot:6.0f} | " + " ".join(f"{n} {v:5.0f}" for n, v in zip(names, med.tolist())))
m = ok.view(-1)
dd = d.view(-1, 8)[m]
print("all   total %6.0f | " % (t.view(-1, 9)[m, 8] - t.view(-1, 9)[m, 0]).median().item()
      + " ".join(f"{n} {v:5.0f}" for n, v in zip(names, dd.median(dim=0).values.tolist())))
