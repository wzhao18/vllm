#!/usr/bin/env python3
"""wsplan.py: the k31ws GPU pool scheduler (FAST 2026-11-18), called by wsgpu.sh every poll and by wsq.sh.

Every waiting wsgpu.sh computes the same deterministic plan from the shared state and takes only the slot the plan
gives to itself (the GPU flocks still arbitrate races):
  state  = running pool holds (squeue, cached in pool/squeue.cache for 10 s), live GPU locks (flock probe of
           pool/locks/<job>.<host>.<gpu>), waiting requests (pool/wait/<loginhost>.<pid>: NAME G T0 [M]),
           running requests per agent (pool/run/*: NAME ...).
  order  = fair round-robin over agents: round k takes every agent's k-th oldest request; inside a round, agents
           with fewer running jobs go first, then the older request.
  assign = in that order, each request takes a free aligned slot on a node that is not reserved for someone else and
           whose hold has >= M+5 min left: best fit (the node with the fewest free GPUs, then the hold that ends
           first), so 1-GPU work packs onto busy nodes and whole nodes stay whole.
  reserve= a multi-GPU request that cannot be placed reserves the non-reserved node with the most free GPUs (it drains
           to it); at most (nodes - 1) reservations while a smaller request waits, so small work always has a node.
           A reservation lasts while its waiter waits (pool/resv/<job>.<host> holds the waiter's wait file, which
           older wsgpu.sh versions honour as well).
usage: wsplan.py me <waitfile>   -> "ASSIGN <job> <host> <gpu,gpu> <hold-end-epoch>" | "RESERVE <job> <host>" | "WAIT <pos> <n>"
       wsplan.py show            -> human-readable queue + plan
"""
import fcntl
import os
import subprocess
import sys
import time

K = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws"
P = K + "/pool"
SQ_TTL = 10


def squeue_holds():
    """[(job, [hosts], end_epoch, name)] of running *-k31ws.hold1 jobs (cached)."""
    cache = P + "/squeue.cache"
    try:
        fresh = time.time() - os.stat(cache).st_mtime < SQ_TTL
    except OSError:
        fresh = False
    if fresh:
        with open(cache) as f:
            text = f.read()
    else:
        try:
            text = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "weizha"), "-t", "R",
                                   "-o", "%i|%N|%e|%j|%D"], capture_output=True, text=True, timeout=30).stdout
        except Exception:
            text = ""
        if text.strip():
            tmp = f"{cache}.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, cache)
        elif os.path.exists(cache):
            with open(cache) as f:
                text = f.read()
    holds = []
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) < 5 or "-k31ws.hold" not in p[3]:
            continue
        try:
            end = int(time.mktime(time.strptime(p[2], "%Y-%m-%dT%H:%M:%S")))
        except ValueError:
            continue
        holds.append((p[0], expand(p[1]), end, p[3]))
    return holds


def expand(nodelist):
    if "[" not in nodelist and "," not in nodelist:
        return [nodelist]
    try:
        out = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True,
                             timeout=30).stdout.split()
        return out or [nodelist]
    except Exception:
        return [nodelist]


def live(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def pid_alive(path):
    """wait/run files are <loginhost>.<pid>: on this host check the pid, elsewhere trust the file."""
    base = os.path.basename(path)
    host, _, pid = base.rpartition(".")
    if host != os.uname().nodename.split(".")[0]:
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def read_dir(d):
    out = {}
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        p = os.path.join(d, n)
        try:
            with open(p) as f:
                out[p] = f.read().split()
        except OSError:
            pass
    return out


def state():
    now = time.time()
    holds = squeue_holds()
    wait_files = read_dir(P + "/wait")
    # 8-node holds serve 1-4 GPU work while no 8n request waits (their GPUs idle until EP32 work runs)
    h8 = not any(len(f) > 1 and f[1] == "8n" and pid_alive(p) for p, f in wait_files.items())
    nodes = []  # (job, host, end, free set)
    for job, hosts, end, name in holds:
        if ".hold1" not in name and not (h8 and ".hold8" in name):
            continue
        for h in hosts:
            free = {g for g in range(4) if not live(f"{P}/locks/{job}.{h}.{g}")}
            nodes.append((job, h, end, free))
    waits = []
    for p, f in wait_files.items():
        if len(f) < 3 or not pid_alive(p):
            continue
        name, g, t0 = f[0], f[1], int(f[2])
        m = int(f[3]) if len(f) > 3 else 60
        if g == "8n":
            continue
        waits.append(dict(wf=p, name=name, g=int(g), t0=t0, m=m))
    running = {}
    for p, f in read_dir(P + "/run").items():
        if f and pid_alive(p):
            running[f[0]] = running.get(f[0], 0) + 1
    resv = {}
    for p, f in read_dir(P + "/resv").items():
        if f and os.path.exists(f[0]):
            resv[os.path.basename(p)] = f[0]
    return now, nodes, waits, running, resv


def fair_order(waits, running):
    by = {}
    for w in sorted(waits, key=lambda w: (w["t0"], w["wf"])):
        by.setdefault(w["name"], []).append(w)
    order, k = [], 0
    while True:
        rnd = [(running.get(n, 0), ws[k]["t0"], ws[k]["wf"], ws[k]) for n, ws in by.items() if len(ws) > k]
        if not rnd:
            return order
        order += [r[3] for r in sorted(rnd, key=lambda r: r[:3])]
        k += 1


def plan(now, nodes, waits, running, resv):
    free = {(j, h): set(f) for j, h, _, f in nodes}
    end = {(j, h): e for j, h, e, _ in nodes}
    order = fair_order(waits, running)
    # reservations of waiters that still wait (old and new wsgpu.sh): node -> wait file
    owner = {}
    for key, wf in resv.items():
        j, _, h = key.partition(".")
        if (j, h) in free and any(w["wf"] == wf for w in waits):
            owner[(j, h)] = wf
    small_waiting = any(w["g"] == 1 for w in waits)
    maxresv = max(1, len(free) - 1) if small_waiting else len(free)
    out = {}
    for w in order:
        g, ok_end = w["g"], now + (w["m"] + 5) * 60
        best = None
        for (j, h), f in free.items():
            if end[(j, h)] < ok_end or owner.get((j, h), w["wf"]) != w["wf"]:
                continue
            for s in range(0, 4, g):
                slot = set(range(s, s + g))
                if slot <= f:
                    key = (len(f), end[(j, h)], j, h, s)
                    if best is None or key < best[0]:
                        best = (key, j, h, slot)
                    break
        if best:
            _, j, h, slot = best
            free[(j, h)] -= slot
            out[w["wf"]] = ("ASSIGN", j, h, ",".join(map(str, sorted(slot))), end[(j, h)])
            continue
        mine = [n for n, o in owner.items() if o == w["wf"]]
        if not mine and g > 1 and len(owner) < maxresv:
            cands = [n for n in free if n not in owner and end[n] >= ok_end]
            if cands:
                n = max(cands, key=lambda n: (len(free[n]), end[n]))
                owner[n] = w["wf"]
                mine = [n]
        if mine:
            # the reserved node's free GPUs are kept for this request
            out[w["wf"]] = ("RESERVE", mine[0][0], mine[0][1])
            free[mine[0]] = set()
        else:
            out[w["wf"]] = ("WAIT",)
    return order, out


def main():
    now, nodes, waits, running, resv = state()
    order, out = plan(now, nodes, waits, running, resv)
    if sys.argv[1] == "me":
        wf = sys.argv[2]
        pos = next((i for i, w in enumerate(order) if w["wf"] == wf), len(order))
        a = out.get(wf, ("WAIT",))
        if a[0] == "ASSIGN":
            print("ASSIGN", a[1], a[2], a[3], a[4])
        elif a[0] == "RESERVE":
            print("RESERVE", a[1], a[2], pos + 1, len(order))
        else:
            print("WAIT", pos + 1, len(order))
        return
    t = time.strftime("%H:%M:%S")
    print(f"{t} pool: {len(nodes)} node(s) of running hold1 jobs; {len(waits)} waiting; running by agent {running}")
    for j, h, e, f in nodes:
        busy = sorted(set(range(4)) - f)
        rs = [w for n, w in resv.items() if n == f"{j}.{h}"]
        print(f"  {j} {h} ends {time.strftime('%H:%M', time.localtime(e))} ({int((e - now) / 60)} min left)"
              f" busy GPUs {busy}{' reserved' if rs else ''}")
    print("  queue (fair order): pos agent G waited-min M plan")
    for i, w in enumerate(order):
        a = out.get(w["wf"], ("WAIT",))
        print(f"  {i + 1:3d} {w['name']:8s} {w['g']} {int((now - w['t0']) / 60):4d} {w['m']:4d} {' '.join(map(str, a[:4]))}")


if __name__ == "__main__":
    main()
