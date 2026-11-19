#!/usr/bin/env python3
"""wsfree.py <g 1|2|4|8n> <minutes> [ovtag]: where a new wsgpu.sh request would land on THIS cluster now (XC 2026-11-18).

Runs on a cluster login node (called by the Mac dispatcher wsrun.sh, one ssh per decision). It appends a probe request
at the end of the fair queue and runs the same deterministic plan as wsgpu.sh (wsplan.py), so the answer is what
wsgpu.sh would do. Prints one line:
  <cluster> <ASSIGN|RESERVE|WAIT|NONE> ahead=<requests ahead> nodes=<pool nodes running> free=<free GPUs>
            pend=<pending pool nodes> ov=<0|1>
ASSIGN = a slot is free now; RESERVE = a node would drain to it; WAIT = queued behind others; NONE = no running hold fits.
For 8n: ASSIGN when a running 8-node hold8 has >= m+5 min left and no live lock, else WAIT (pend counts hold8 requests).
"""
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsplan  # noqa: E402

K = wsplan.K


def main():
    g, m = sys.argv[1], int(sys.argv[2])
    ov = sys.argv[3] if len(sys.argv) > 3 else ""
    cl = "hecate" if "hecate" in socket.gethostname() else "rhea"
    now, nodes, waits, running, resv = wsplan.state()
    try:
        pend_txt = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "weizha"), "-t", "PD", "-o", "%j|%D"],
                                  capture_output=True, text=True, timeout=30).stdout
    except Exception:
        pend_txt = ""
    free = sum(len(f) for _, _, _, f in nodes)
    has_ov = int(not ov or os.path.isdir(f"{K}/ov/{ov}"))
    if g == "8n":
        pend = sum(int(l.split("|")[1]) for l in pend_txt.splitlines() if ".hold8" in l)
        ok = "WAIT"
        for job, hosts, end, name in wsplan.squeue_holds():
            if ".hold8" in name and len(hosts) == 8 and end >= now + (m + 5) * 60:
                if not any(wsplan.live(f"{wsplan.P}/locks/{job}.{h}.{x}") for h in hosts for x in range(4)):
                    ok = "ASSIGN"
                    break
        n8 = sum(1 for p, f in wsplan.read_dir(wsplan.P + "/wait").items() if len(f) > 1 and f[1] == "8n")
        print(f"{cl} {ok} ahead={n8} nodes={len(nodes)} free={free} pend={pend} ov={has_ov}")
        return
    pend = sum(int(l.split("|")[1]) for l in pend_txt.splitlines() if "-k31ws.hold" in l)
    probe = dict(wf="__probe__", name="__probe__", g=int(g), t0=int(now) + 1, m=m)
    order, out = wsplan.plan(now, nodes, waits + [probe], running, resv)
    a = out.get("__probe__", ("WAIT",))[0]
    if a == "WAIT" and not any(e >= now + (m + 5) * 60 for _, _, e, _ in nodes):
        a = "NONE"
    ahead = next((i for i, w in enumerate(order) if w["wf"] == "__probe__"), len(order))
    print(f"{cl} {a} ahead={ahead} nodes={len(nodes)} free={free} pend={pend} ov={has_ov}")


if __name__ == "__main__":
    main()
