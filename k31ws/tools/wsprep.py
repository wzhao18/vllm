#!/usr/bin/env python3
"""wsprep.py [--strict] [--cfg <out>.cfg.sh]: CPU-side preflight + extension prebuild for a queued k31ws run (FAST 2026-11-18).

Runs inside the K3.1 image (wsenv.sh) on a pool node's CPUs, with no GPU visible, while the request waits for GPUs
(wsgpu.sh -P). Two jobs:
  1. preflight: byte-compile every .py the overlay ships (SOURCE_REV.ws), import vllm from the overlay, import the
     Kimi K3 model / drafter / specialized decode + prefill packages, and (--cfg) parse the step bench's vLLM CLI
     arguments with vLLM's own parser. Failures exit 3 with the error: the run would crash on the GPUs.
     Without --strict only a shipped-file syntax error or a failing `import vllm` is fatal; model import failures
     are reported (some runs never import the model).
  2. prebuild: build (or just load) the digest-keyed torch extensions the model JIT-builds at startup
     (k3_flashkda_split, k3_kda_conv, dcp_exchange, dsa_core) so the GPU run finds them built. The builds use fixed
     -gencode flags; dcp_exchange / dsa_core pick the arch from the device, which is pinned to sm_107 here.
     Prebuild failures are reported, never fatal (the GPU run builds and reports for itself).
Output: one line per step ("wsprep: ok|FAIL|warn <what> <seconds>"); exit 0 ok, 3 preflight failed.
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T0 = time.time()
FAIL = []


def say(status, what, t):
    print(f"wsprep: {status} {what} {time.time() - t:.1f}s", flush=True)


def step(what, fn, fatal):
    t = time.time()
    try:
        fn()
        say("ok", what, t)
        return True
    except BaseException as e:  # noqa: BLE001 (report everything, incl. SystemExit from argparse)
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:400]
        say("FAIL" if fatal else "warn", f"{what}: {msg}", t)
        if fatal:
            FAIL.append(what)
        return False


def shipped_compile(root):
    import py_compile

    p = os.path.join(root, "SOURCE_REV.ws")
    if not os.path.isfile(p):
        return
    bad = []
    for line in open(p).read().splitlines()[1:]:
        st, _, f = line.partition(" ")
        if st in ("M", "A") and f.endswith(".py"):
            try:
                # compile only (no .pyc written into the hard-linked overlay)
                compile(open(os.path.join(root, f), "rb").read(), f, "exec")
            except SyntaxError as e:
                bad.append(f"{f}:{e.lineno}: {e.msg}")
    if bad:
        raise SyntaxError("; ".join(bad[:5]))


def import_vllm(root):
    import vllm

    real = os.path.realpath(root) + os.sep
    if not os.path.realpath(vllm.__file__).startswith(real):
        raise RuntimeError(f"vllm imported from {vllm.__file__}, not the overlay {root}")


def import_model():
    import importlib

    for m in ("vllm.models.kimi_k3.nvidia.model",
              "vllm.models.kimi_k3.nvidia.specialized_decode",
              "vllm.models.kimi_k3.nvidia.specialized_prefill"):
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name != m:  # a missing optional package is fine; a missing import inside it is not
                raise


def parse_cfg(cfg):
    import subprocess

    out = subprocess.run(["bash", "-c", f'source "{cfg}"; printf "%s\\0" "${{NQ_ARGS[@]}}"'],
                         capture_output=True, check=True).stdout.decode()
    args = [a for a in out.split("\0") if a]
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    parser.parse_args(args + ["--shapes", "/dev/null"])
    for i, a in enumerate(args):  # model / drafter dirs must exist
        if a == "--model" and i + 1 < len(args) and not os.path.isdir(args[i + 1]):
            raise FileNotFoundError(f"--model {args[i + 1]}")


def prebuild():
    import torch

    torch.cuda.get_device_capability = lambda *a, **k: (10, 7)  # the pool's VR200 (sm_107a)
    loaders = []
    try:
        from vllm.models.kimi_k3.nvidia.ops import kda_segscan
        loaders.append(("k3_flashkda_split", kda_segscan.load_extension))
    except Exception:
        pass
    try:
        from vllm.models.kimi_k3.nvidia.ops import kda_conv
        loaders.append(("k3_kda_conv", kda_conv._load_qkv_conv))
    except Exception:
        pass
    try:
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange import ops as dcpx
        loaders.append(("dcp_exchange", dcpx._load_ext))
    except Exception:
        pass
    try:
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import ops as dsac
        loaders.append(("dsa_core", dsac.load_ext))
    except Exception:
        pass
    for name, fn in loaders:
        step(f"prebuild {name}", fn, fatal=False)


def main():
    strict = "--strict" in sys.argv
    cfg = sys.argv[sys.argv.index("--cfg") + 1] if "--cfg" in sys.argv else ""
    root = os.environ.get("WS_OVDIR") or ""
    if root:
        step("shipped .py compile", lambda: shipped_compile(root), fatal=True)
        step("import vllm (overlay)", lambda: import_vllm(root), fatal=True)
    if not FAIL:
        step("import Kimi K3 model packages", import_model, fatal=strict)
    if cfg and not FAIL:
        step("vLLM args parse", lambda: parse_cfg(cfg), fatal=True)
    if not FAIL:
        prebuild()
    print(f"wsprep: {'FAIL ' + ', '.join(FAIL) if FAIL else 'done'} total {time.time() - T0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(3 if FAIL else 0)


if __name__ == "__main__":
    main()
