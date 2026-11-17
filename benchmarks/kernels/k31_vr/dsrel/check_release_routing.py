# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check (no weights loaded) that a Kimi K3.1 deployment with the release DSpark draft
(a speculators ``dspark`` checkpoint) resolves the draft to
``dspark_gqa.K3GqaDSparkForCausalLM`` and its defaults (FLASH_ATTN, auto KV), and
that the specialized decode's drafter check accepts it, from the real configs.

    python check_release_routing.py <target model dir> <release draft dir>
"""

import sys


def main(target, draft):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.models.kimi_k3.nvidia.specialized_decode import runtime

    args = EngineArgs(
        model=target,
        trust_remote_code=True,
        tensor_parallel_size=1,  # (the routing does not depend on the topology)
        max_model_len=1048576,
        kv_cache_dtype="fp8",
        attention_backend="FLASHINFER_MLA_SPARSE",
        speculative_config={
            "method": "dspark",
            "model": draft,
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
        load_format="dummy",
    )
    config = args.create_engine_config()
    spec = config.speculative_config
    draft_config = spec.draft_model_config
    cls, arch = get_model_architecture(draft_config)
    print("draft architectures", draft_config.architectures)
    print("draft class", f"{cls.__module__}.{cls.__qualname__}", "arch", arch)
    print("draft attention backend", spec.attention_backend, "kv", spec.kv_cache_dtype)
    print("drafter_unsupported", runtime.drafter_unsupported(config, False))
    assert cls.__qualname__ == "K3GqaDSparkForCausalLM", cls
    assert runtime.drafter_unsupported(config, False) is None


if __name__ == "__main__":
    main(*sys.argv[1:3])
