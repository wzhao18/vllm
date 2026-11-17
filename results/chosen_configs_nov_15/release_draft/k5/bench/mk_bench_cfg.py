#!/usr/bin/env python3
"""mk_bench_cfg.py <nq-c160-xdr.json> <k5 render config.yaml> <out.json>: the bench-only step-bench config of the k5
coverage run (08-nq4-dsrel-k5full-c160-m75.sh): the nq bench's c160 decode config (nqbench/cfg/nq-c160-xdr.json, the
args cfg2bench classifies) with the decode role of a k5 render (release_draft/k5: dsrelk5-c160-1p1d-llh): k5,
max-num-seqs 160, max-cudagraph-capture-size 960 and its capture list. Not a production config. Standard library."""
import json
import re
import sys

src, render, out = sys.argv[1:4]
doc = json.load(open(src))
args = doc["roles"]["decode"]["args"]
text = open(render).read()
body = text[text.index("\n  decode:\n"):]


def arg(name):
    return re.search(rf"^      {name}: (.*)$", body, re.M).group(1)


k5_spec = json.loads(arg("speculative-config").strip("'"))
spec = json.loads(args["speculative-config"])
spec["num_speculative_tokens"] = k5_spec["num_speculative_tokens"]
comp = json.loads(args["compilation-config"])
comp["cudagraph_capture_sizes"] = json.loads(arg("compilation-config").strip("'"))["cudagraph_capture_sizes"]
args["speculative-config"] = json.dumps(spec)
args["compilation-config"] = json.dumps(comp, separators=(",", ":"))
args["max-num-seqs"] = int(arg("max-num-seqs"))
args["max-cudagraph-capture-size"] = int(arg("max-cudagraph-capture-size"))
doc["_k5_note"] = (f"bench-only: {src} with the decode role of {render}: k{spec['num_speculative_tokens']}, "
                   f"max-num-seqs {args['max-num-seqs']}, cap {args['max-cudagraph-capture-size']}, "
                   f"{len(comp['cudagraph_capture_sizes'])} capture sizes")
json.dump(doc, open(out, "w"), indent=1)
print(doc["_k5_note"])
