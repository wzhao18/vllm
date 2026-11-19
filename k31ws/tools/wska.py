#!/usr/bin/env python3
"""wska.py: idle guard for one node of a k31ws pool hold on Hecate (XC 2026-11-18).

Hecate's site reaper cancels allocations whose GPUs look idle, so a pool hold that waits for work must keep its GPUs
busy, but never while a pool run uses the node. One persistent process per node (an srun --overlap step started by
holdka.sh, or by runner2's keepalive2.py for adopted holds); it never touches CUDA itself:
  busy  = a wsgpu.sh run file (pool/run/*) names this hold and this node ("NAME G JOB HOST GPUS") or this hold as a
          whole ("NAME 8n JOB"), or any process other than our burner has a CUDA context on a GPU of this node;
  idle  = not busy for WSKA_IDLE_S seconds (default 120): start a burner child (BF16 GEMMs on every GPU);
  as soon as the node is busy again the burner is SIGKILLed (its contexts and memory go away with it), so a pool run
  never shares a GPU with it. wsgpu.sh writes the run file before its srun starts, so the burner is gone (2 s poll)
  before the run creates its first context.
"""
import glob
import os
import signal
import subprocess
import sys
import time

import pynvml

K = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws"
JOB = os.environ.get("SLURM_JOB_ID", "")
HOST = os.uname().nodename.split(".")[0]
IDLE_S = float(os.environ.get("WSKA_IDLE_S", "120"))
BURN = r"""
import time, torch
xs = [torch.randn(8192, 8192, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(torch.cuda.device_count())]
while True:
    for _ in range(8):
        for x in xs:
            x @ x
    torch.cuda.synchronize()
"""


def log(msg):
    print(time.strftime("%F %T"), HOST, JOB, msg, flush=True)


def pool_busy():
    for p in glob.glob(K + "/pool/run/*"):
        try:
            with open(p) as f:
                w = f.read().split()
        except OSError:
            continue
        if len(w) >= 3 and w[2] == JOB and (w[1] == "8n" or (len(w) >= 4 and w[3] == HOST)):
            return True
    return False


def others(handles, mine):
    for h in handles:
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        except pynvml.NVMLError:
            return True  # be safe: unknown = busy
        if any(p.pid not in mine for p in procs):
            return True
    return False


def main():
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    child = None
    idle_since = time.time()
    log(f"wska start: {len(handles)} GPUs, idle threshold {IDLE_S:.0f} s")
    while True:
        mine = {child.pid} if child else set()
        busy = pool_busy() or others(handles, mine)
        now = time.time()
        if busy:
            idle_since = now
            if child:
                child.send_signal(signal.SIGKILL)
                child.wait()
                child = None
                log("node busy: burner stopped")
        else:
            if child and child.poll() is not None:
                log(f"burner exited rc={child.returncode}")
                child = None
                idle_since = now
            if child is None and now - idle_since >= IDLE_S:
                child = subprocess.Popen([sys.executable, "-c", BURN])
                log(f"node idle {now - idle_since:.0f} s: burner pid {child.pid}")
        time.sleep(2)


if __name__ == "__main__":
    main()
