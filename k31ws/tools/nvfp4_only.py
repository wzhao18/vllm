#!/usr/bin/env python3
"""nvfp4_only.py: 2026-11-18 user rule - baseline and target are the NVFP4 checkpoint only. Removes MXFP4 numbers from
campaign.json (steps, findings) and marks every status file whose numbers are not all NVFP4-basis as ckpt=pending
(the tracker then hides gap closed for it)."""
import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = json.load(open(f"{ROOT}/campaign.json"))
c["steps"] = {
    "Prefill DEP16 NVFP4 (tok/s/GPU)": {"now": "not measured yet", "ach": "30.7K", "note": "F16K, L78K; NVFP4 baseline run pending (INFRA)"},
    "Decode r8 NVFP4 (tok/s/GPU)": {"now": "not measured yet", "ach": "2,855", "note": "D3-32; NVFP4 baseline pending"},
    "Decode r24 NVFP4 (tok/s/GPU)": {"now": "not measured yet", "ach": "6,280", "note": "D3-32; NVFP4 baseline pending"},
}
bad = ("MXFP4", "mxfp4", "MXFP8 MoE act", "Decode baseline MEASURED", "8.9", "13.9K", "1,178", "2,425")
c["findings"] = [f for f in c["findings"] if not any(b in f["text"] for b in bad)]
c["findings"].append({"t": "11-18 14:15", "text": "Rule (user): baseline and target are the NVFP4 checkpoint only; MXFP4 numbers are removed from the tracker. Gap closed is shown only for workstreams whose baseline, current and achievable are all NVFP4-basis (ckpt=nvfp4). NVFP4 in-step baselines (decode slice r8/16/24, DEP16 slice) are INFRA's top priority."})
json.dump(c, open(f"{ROOT}/campaign.json", "w"), indent=1)
for p in glob.glob(f"{ROOT}/status/*.json"):
    s = json.load(open(p))
    if s.get("ckpt") != "nvfp4":
        s["ckpt"] = "pending"
        json.dump(s, open(p, "w"), indent=1)
print("done")
