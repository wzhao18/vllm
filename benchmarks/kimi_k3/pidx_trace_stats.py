# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Individual kernel events from a step_p chrome trace (.json.gz), in launch order.
Usage: python3 pidx_trace_stats.py <pattern> <trace.json.gz> [max]"""
import gzip
import json
import re
import sys

pat = re.compile(sys.argv[1], re.I)
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 40
d = json.load(gzip.open(sys.argv[2]))
ev = d["traceEvents"] if isinstance(d, dict) else d
ks = [e for e in ev if e.get("ph") == "X" and e.get("cat", "") in ("kernel", "Kernel")
      and pat.search(str(e.get("name", "")))]
ks.sort(key=lambda e: e["ts"])
t0 = ks[0]["ts"] if ks else 0
for e in ks[:mx]:
    a = e.get("args", {})
    print(f"{e['ts'] - t0:10.0f} {e['dur']:8.1f} grid={a.get('grid')} blk={a.get('block')} "
          f"str={a.get('stream')} {e['name'][-28:]}")
print("n", len(ks))
