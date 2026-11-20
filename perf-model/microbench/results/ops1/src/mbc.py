"""ops1 lane common timing helpers (single GPU).

time_us(fn): FlashInfer CUPTI per-iteration GPU time (first kernel start -> last kernel end of one call),
cold L2 (flush between iterations), median. Falls back to CUDA-graph replay + events (L2 flush outside
the timed region is not possible there, so the fallback replays N calls back to back) when CUPTI fails.
"""
import json
import os
import statistics
import sys
import time

import torch

_CUPTI_OK = None


def env_info():
    info = {"gpu": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "host": os.uname().nodename, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pythonpath0": os.environ.get("PYTHONPATH", "").split(":")[0]}
    try:
        import flashinfer
        info["flashinfer"] = flashinfer.__version__ + " @ " + os.path.dirname(flashinfer.__file__)
    except Exception as e:  # noqa: BLE001
        info["flashinfer"] = repr(e)[:100]
    try:
        import vllm
        info["vllm"] = os.path.dirname(vllm.__file__)
    except Exception as e:  # noqa: BLE001
        info["vllm"] = repr(e)[:100]
    return info


def _events(fn, n=20, reps=7):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / n)
    del g
    return statistics.median(ts)


def time_us(fn, cold=True, repeat_ms=60):
    """Returns (us, method)."""
    global _CUPTI_OK
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if _CUPTI_OK is not False:
        try:
            from flashinfer.testing import bench_gpu_time_with_cupti
            t = bench_gpu_time_with_cupti(fn, dry_run_time_ms=10, repeat_time_ms=repeat_ms,
                                          cold_l2_cache=cold)
            _CUPTI_OK = True
            return statistics.median(t) * 1e3, "cupti" + ("-cold" if cold else "-warm")
        except Exception as e:  # noqa: BLE001
            if _CUPTI_OK is None:
                print("CUPTI unavailable, falling back to graph events:", repr(e)[:300], flush=True)
            _CUPTI_OK = False
    return _events(fn), "graph-events-warm"


class Out:
    def __init__(self, path):
        self.f = open(path, "a")

    def row(self, **kw):
        kw["ts"] = time.strftime("%H:%M:%S")
        line = json.dumps(kw)
        self.f.write(line + "\n")
        self.f.flush()
        print("ROW " + line, flush=True)


def shard(cases, spec):
    i, n = map(int, spec.split("/"))
    return [c for j, c in enumerate(cases) if j % n == i]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True, file=sys.stdout)
