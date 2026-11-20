# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan SASS for the dsa.query_uk race pattern: a shared-memory load (LDS) whose
destination registers are first read only AFTER an mbarrier arrive (SYNCS.ARRIVE /
USYNCS.ARRIVE) issued by the same warp. If that arrive frees the buffer the load read,
the buffer's next fill (TMA, cp.async or another warp's stores) can land before the
load has returned (QueryUK, D-ATTN 2026-11-20).

usage: sass_lds_release_scan.py <cubin or .sass> [...]   (nvdisasm -c on cubins)
Linear scan in program order (branches ignored): candidates to inspect, not proofs.
A FENCE.VIEW.ASYNC between the load and the arrive is reported but not cleared."""

import re
import subprocess
import sys

INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
LDS = re.compile(r"^(@!?U?P\d+\s+)?LDS(\.(?:U8|S8|U16|S16|32|64|128))?\s+(R\d+),")
REG = re.compile(r"\bR(\d+)\b")


def width(sfx):
    return {".64": 2, ".128": 4}.get(sfx or "", 1)


def insns(path):
    if path.endswith(".cubin"):
        text = subprocess.run(["nvdisasm", "-c", path], capture_output=True, text=True,
                              check=True).stdout
    else:
        text = open(path).read()
    func = "?"
    out = []
    for ln in text.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", ln) or re.match(r"\s*(\S+):\s*$", ln)
        if m and "_x_" not in m.group(1):
            func = m.group(1)
        m = INSN.search(ln)
        if m:
            out.append((func, m.group(1)))
    return out


def scan(path, window=400):
    seq = insns(path)
    hits = []
    for i, (func, ins) in enumerate(seq):
        m = LDS.match(ins)
        if not m:
            continue
        base = int(m.group(3)[1:])
        dst = {base + k for k in range(width(m.group(2)))}
        fence = False
        for j in range(i + 1, min(i + window, len(seq))):
            nxt = seq[j][1]
            op = nxt.split()[0] if not nxt.startswith("@") else nxt.split()[1]
            if op.startswith("FENCE.VIEW.ASYNC"):
                fence = True
            if "ARRIVE" in op and "SYNCS" in op:
                hits.append((func, i, j, ins, nxt, fence))
                break
            srcs = REG.findall(nxt.split(",", 1)[1] if "," in nxt else "")
            dsts = REG.findall(nxt.split(",", 1)[0])
            if dst & {int(r) for r in srcs}:
                break  # read before any arrive
            if not op.startswith(("ST", "RED", "ATOM")) and dst & {int(r) for r in dsts}:
                break  # overwritten
            if op in ("EXIT", "RET", "BRA") or op.startswith("BRA"):
                if op == "EXIT" or op == "RET":
                    break
    return hits


def main():
    total = 0
    for p in sys.argv[1:]:
        hits = scan(p)
        total += len(hits)
        print(f"== {p.split('/')[-1][:120]}: {len(hits)} LDS read after an arrive")
        for func, i, j, a, b, fence in hits[:12]:
            print(f"   [{i}] {a[:60]}  ->  [{j}] {b[:60]}{'  (fence between)' if fence else ''}")
    print(f"SCAN TOTAL {total}")


if __name__ == "__main__":
    main()
