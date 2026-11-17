#!/usr/bin/env python3
"""vllm_check.py: CPU-only vLLM-level check of the release-draft renders' speculative-config. Run it with the vLLM tree the
renders use, inside the production image, on the dsrel dev node (no GPU; gpurun.sh with gpus ""):
    bash $W/dsrel/tools/gpurun.sh <tag> "" python3 $W/dsrel/cfg/tools/vllm_check.py [render glob]
For every dsrel render (prefill and decode role) it checks, with vLLM's own code:
  1. the speculative-config JSON parses the way SpeculativeConfig parses it: method, the attention_backend enum (or auto),
     and the synthetic AL resolved to per-position rates by SpeculativeConfig._resolve_synthetic_acceptance_rates;
  2. the release checkpoint (the render's draft mount host dir) loads through vLLM's config path (get_config) as the GQA
     DSpark draft the configs assume: Qwen3DSparkModel, 5 sliding-window (2048) layers, 96 q / 16 kv heads x 64, aux layers
     76..92, Markov rank 512, anchor sampling; and its per-rank heads at the render's TP divide evenly;
  3. the draft attention backend accepts the draft layers' selector config on Rubin (DeviceCapability(10, 7)): head 64, BF16
     KV ("auto": configure_dense_draft's dtype for GQA drafts), the render's KV block size, causal sliding window, a KV
     connector, no DCP (DSpark runs GQA drafts at DCP 1): AttentionBackend.validate_configuration, the call
     CudaPlatform.get_attn_backend_cls makes for a selected backend (reasons != [] raises at draft construction). The old
     TOKENSPEED_MLA value is checked as the negative control (it must be rejected).
Prints PASS/FAIL lines and VLLM_CHECK PASS|FAIL. Nothing is launched; no GPU is touched.
"""

import glob
import json
import os
import sys

import torch
import yaml

from vllm.config.speculative import SpeculativeConfig
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

RS = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders"
FAILS = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def selector(block_size):
    return dict(head_size=64, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=block_size, use_mla=False,
                has_sink=False, use_sparse=False, use_mm_prefix=False, use_per_head_quant_scales=False,
                device_capability=DeviceCapability(10, 7), attn_type="decoder", has_sliding_window=True, use_non_causal=False,
                use_batch_invariant=False, use_kv_connector=True, use_pcp=False, use_adaptive_verification=False,
                use_dcp=False, use_rswa=False)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else f"{RS}/dsrel-*"
    renders = sorted(d for d in glob.glob(pattern) if os.path.isfile(f"{d}/render/config.yaml") and ".prev-" not in d)
    ok(len(renders) > 0, f"{len(renders)} renders match {pattern}")
    drafts, backends, blocks, tps = {}, set(), set(), set()
    for d in renders:
        cfg = yaml.safe_load(open(f"{d}/render/config.yaml"))
        mounts = {m.split(":")[1]: m.split(":")[0] for m in cfg.get("extra_mount", [])}
        for role in ("prefill", "decode"):
            a = cfg["roles"][role]["args"]
            js = json.loads(a["speculative-config"])
            k = js["num_speculative_tokens"]
            be = js.get("attention_backend", "auto")
            parsed = None if be.lower() == "auto" else AttentionBackendEnum[be.upper()]
            backends.add(parsed.name if parsed else "auto")
            line = f"{os.path.basename(d)} {role}: dspark k={k} backend={parsed.name if parsed else 'auto'}"
            ok(js["method"] == "dspark" and js["draft_sample_method"] == "probabilistic", line)
            if js["rejection_sample_method"] == "synthetic":
                r = SpeculativeConfig._resolve_synthetic_acceptance_rates(k, None, js["synthetic_acceptance_length"])
                ok(abs(1 + sum(r) - js["synthetic_acceptance_length"]) < 1e-9,
                   f"{line}: synthetic AL {js['synthetic_acceptance_length']} -> rates {r}")
            else:
                ok(js["rejection_sample_method"] == "block" and "synthetic_acceptance_length" not in js, f"{line}: block rejection")
            host = mounts.get(js["model"])
            ok(host is not None, f"{line}: draft {js['model']} is an extra_mount target (host {host})")
            drafts[host] = drafts.get(host, 0) + 1
            blocks.add(a.get("block-size"))
            tps.add(a.get("tensor-parallel-size"))
    for host in drafts:
        c = get_config(host, trust_remote_code=True)
        arch = list(getattr(c, "architectures", []) or [])
        lt = list(getattr(c, "layer_types", []) or [])
        ok(arch == ["Qwen3DSparkModel"], f"draft {host}: vLLM config architectures {arch}")
        ok(lt == ["sliding_attention"] * 5 and c.sliding_window == 2048, f"draft: layer_types {lt}, sliding_window {c.sliding_window}")
        ok((c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (96, 16, 64),
           f"draft: heads q {c.num_attention_heads} kv {c.num_key_value_heads} dim {c.head_dim}")
        aux = getattr(c, "eagle_aux_hidden_state_layer_ids", None)
        ok(list(aux or []) == [76, 80, 84, 88, 92], f"draft: eagle_aux_hidden_state_layer_ids {aux}")
        ok(getattr(c, "markov_rank", None) == 512 and getattr(c, "sample_from_anchor", None) is True,
           f"draft: markov_rank {getattr(c, 'markov_rank', None)}, sample_from_anchor {getattr(c, 'sample_from_anchor', None)}")
        for tp in sorted(t for t in tps if t):
            ok(96 % tp == 0 and 16 % tp == 0, f"draft at TP{tp}: {96 // tp} q / {16 // tp} kv heads per rank")
    os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")  # as tests/conftest.py default_vllm_config
    from vllm.config import VllmConfig, set_current_vllm_config

    # the renders' knob value(s) must pass; the other GQA-capable knob values are reported; the old value must fail
    for be in sorted(backends | {"FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "TOKENSPEED_MLA"}):
        if be == "auto":
            continue
        cls = AttentionBackendEnum[be].get_class()
        for bs in sorted(b for b in blocks if b):
            with set_current_vllm_config(VllmConfig()):  # some backends' supports_combination read the current config
                reasons = cls.validate_configuration(**selector(bs))
            if be == "TOKENSPEED_MLA":
                ok(bool(reasons), f"negative control: {be} (the frozen renders' value) rejects the GQA draft layers "
                                  f"(block {bs}): {reasons}")
            elif be in backends:
                ok(not reasons, f"{be} (the renders' draft backend) accepts the GQA draft layers on sm_107 (head 64, bf16 KV, "
                                f"block {bs}, sliding window, KV connector, no DCP): reasons {reasons}")
            else:
                print(f"INFO alternative knob value {be}: validate_configuration reasons at block {bs}: {reasons or 'none (valid)'}")
            print(f"INFO {be} kernel block sizes outside a vLLM config context: {cls.get_supported_kernel_block_sizes()}")
    print(f"VLLM_CHECK {'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s), {len(renders)} renders, drafts {drafts}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
