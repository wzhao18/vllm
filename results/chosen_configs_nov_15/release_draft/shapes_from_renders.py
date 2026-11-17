#!/usr/bin/env python3
"""shapes_from_renders.py [chosen_configs_nov_15 dir]: write release_draft/capture_shapes.json, the decode and prefill shapes of
the 8 release-draft renders (release_draft/configs/<point>/agentx/render/config.yaml; the gsm8k twins use the same deployment).
For each role it records TP / DCP / DP / EP, k, q = k + 1, max-num-seqs, max-cudagraph-capture-size, cudagraph_mode, the
captured sizes, the speculative-config and the draft attention backend. Coverage tests of the release drafter's specialized
pieces read it: every captured decode size of every chosen point (rows = requests x q verify, requests x k draft) must run
the specialized path or refuse at capture (NQ-NOFALLBACK). Standard library + PyYAML."""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
POINTS = ("C1", "C16-1P2D", "C72-1P3D", "C48-1P1D", "C64", "C128-1P1D", "C160-1P1D-llh", "C512")


def role(cfg, name):
    r = cfg["roles"][name]
    a = r["args"]
    spec = json.loads(a["speculative-config"])
    comp = json.loads(a.get("compilation-config", "{}"))
    add = json.loads(a.get("additional-config", "{}"))
    k = spec["num_speculative_tokens"]
    return {
        "instances": r["workers"], "nodes": r["nodes"], "gpus_per_instance": r["gpus"] // r["workers"] if r["workers"] else r["gpus"],
        "tensor_parallel_size": a.get("tensor-parallel-size"), "decode_context_parallel_size": a.get("decode-context-parallel-size"),
        "data_parallel_size": a.get("data-parallel-size", 1), "enable_expert_parallel": bool(a.get("enable-expert-parallel", False)),
        "decode_backend": add.get("decode_backend", "(default)"), "k": k, "q": k + 1,
        "max_num_seqs": a.get("max-num-seqs"), "max_num_batched_tokens": a.get("max-num-batched-tokens"),
        "max_cudagraph_capture_size": a.get("max-cudagraph-capture-size"),
        "cudagraph_mode": comp.get("cudagraph_mode", "(default)"), "cudagraph_capture_sizes": comp.get("cudagraph_capture_sizes"),
        "block_size": a.get("block-size"), "kv_cache_dtype": a.get("kv-cache-dtype"), "attention_backend": a.get("attention-backend"),
        "speculative_config": spec, "draft_attention_backend": spec.get("attention_backend", "auto (FLASH_ATTN)"),
        "dcp_pair_exchange": '"dcp_pair_exchange":true' in a.get("kv-transfer-config", ""),
    }


out = {"source": "release_draft/configs/<point>/agentx/render/config.yaml (= Hecate e2e/renders/dsrel-<base>-<fab>)", "points": {}}
for p in POINTS:
    cfg = yaml.safe_load(open(f"{ROOT}/release_draft/configs/{p}/agentx/render/config.yaml"))
    out["points"][p] = {"render": cfg["name"].split(".", 1)[1], "prefill": role(cfg, "prefill"), "decode": role(cfg, "decode")}
with open(f"{HERE}/capture_shapes.json", "w") as fh:
    json.dump(out, fh, indent=1)
    fh.write("\n")
for p, v in out["points"].items():
    d = v["decode"]
    print(f"{p:14s} decode TP{d['tensor_parallel_size']} DCP{d['decode_context_parallel_size']} DP{d['data_parallel_size']} x{d['instances']} "
          f"k{d['k']} mns {d['max_num_seqs']} max capture {d['max_cudagraph_capture_size']} sizes {len(d['cudagraph_capture_sizes'] or [])} "
          f"pair_exchange {d['dcp_pair_exchange']} backend {d['decode_backend']}")
