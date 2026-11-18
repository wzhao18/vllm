"""pf_cfg.py: the C512 PREFILL role (jh/noquant/cfg/nq-c512-xdr.yaml, DP4 x TP4=DCP4, EP16, ShardQ) -> the nq step
bench's cfg.sh (same format as cfg2bench.py, consumed by nq_inner.sh / nq_step.py), for the gap512 lane.

Reuses cfg2bench.build() on a copy of the config whose decode role is replaced by the prefill role:
  * the prefill-only serving/scheduler args (prefix-cache policy, mamba prefix checkpoints, long-prefill threshold,
    Mooncake/NIXL connector) are dropped (no scheduler, no prefix cache, no transfer in vllm bench forward);
  * arm 'stock' -> additional-config decode_backend auto (= the prefill role's own value);
  * cfg2bench adds a C512 decode-only kernel-config (flashinfer_sparse_mla_kernel trtllm); the prefill role has no
    kernel-config, so it is removed again here (production prefill = auto);
  * drafter = the release DSpark draft on FLASH_ATTN (as the dsrel decode harness), dummy weights, MXFP8 MoE
    activations from the role's quantization-config (kept as is).
usage: pf_cfg.py <drafter dir>  > out.cfg.sh
"""
import json
import sys

sys.path.insert(0, "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/dsrel/tools")
import cfg2bench as c  # noqa: E402

PREFILL_ONLY = [
    "prefix-cache-policy", "enable-mamba-shared-prefix-checkpoint", "mamba-conversation-max-turns",
    "mamba-conversation-max-turns-mode", "mamba-low-priority-snapshot-prompt-len", "mamba-internal-snapshot",
    "long-prefill-token-threshold", "long-prefill-token-threshold-adaptive", "mamba-extra-tail-checkpoint-tokens",
]
for k in PREFILL_ONLY:
    c.DROP[k] = "prefill serving/scheduler/prefix-cache only (gap512 prefill bench: no scheduler, no prefix cache)"

src = c.resolve("c512")
doc = c.load(src)
pre = doc["roles"]["prefill"]
doc2 = dict(doc)
doc2["roles"] = dict(doc["roles"])
doc2["roles"]["decode"] = {"args": dict(pre["args"]), "env": dict(pre.get("env") or {})}
out = c.build(doc2, "stock", sys.argv[1], [], full_capture=True)
argv = out["argv"]
if "--kernel-config" in argv:
    i = argv.index("--kernel-config")
    del argv[i:i + 2]
    out["notes"] = [n for n in out["notes"] if not n.startswith("C512: kernel-config")]
    out["notes"].append("gap512: prefill role has no kernel-config (production auto); cfg2bench's C512 decode "
                        "kernel-config removed")
out["notes"].insert(0, "gap512 PREFILL role of nq-c512-xdr (prefill args/env), not the decode role")
print(c.as_sh(out, src + "#prefill"))
