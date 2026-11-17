# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the release DSpark draft (real checkpoint) the way the DSpark speculator does
(``load_dspark_model``, TP1, a stand-in target embedding / LM head) and check the
model class, the loaded geometry, the shared target modules and the context step's
math against the stock Qwen3 DSpark context step.

    python check_release_model.py <target model dir> <release draft dir>
"""

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def main(target, draft):
    import vllm._custom_ops as ops
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

    config = EngineArgs(
        model=target,
        trust_remote_code=True,
        max_model_len=1048576,
        kv_cache_dtype="fp8",
        attention_backend="FLASHINFER_MLA_SPARSE",
        speculative_config={
            "method": "dspark",
            "model": draft,
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
    ).create_engine_config()
    torch.cuda.set_device(0)
    with set_current_vllm_config(config):
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            backend="nccl",
            distributed_init_method="tcp://127.0.0.1:29655",
        )
        initialize_model_parallel(1)
        with torch.device("cuda"):
            embed = VocabParallelEmbedding(163840, 7168).bfloat16()
            head = ParallelLMHead(163840, 7168).bfloat16()
        target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed),
                                       lm_head=head)  # fmt: skip
        model = load_dspark_model(target_model, config)
    inner = model.model
    print("class", type(model).__module__, type(model).__qualname__)
    layer = inner.layers[0]
    print(
        "layers", len(inner.layers),
        "qkv", tuple(layer.self_attn.qkv_proj.weight.shape),
        "o", tuple(layer.self_attn.o_proj.weight.shape),
        "o reduce", layer.self_attn.o_proj.reduce_results,
        "down reduce", layer.mlp.down_proj.reduce_results,
    )  # fmt: skip
    print(
        "fc", tuple(inner.fc.weight.shape),
        "markov", tuple(inner.markov_head.markov_w2.weight.shape),
        "fused kv", tuple(inner._fused_kv_weight.shape),
        "confidence", inner.confidence_head is not None,
    )  # fmt: skip
    print(
        "shared embed",
        inner.embed_tokens is embed,
        "shared lm_head",
        model.lm_head is head,
    )
    print("hooks", all(hasattr(model, h) for h in (
        "greedy_markov_draft", "sampled_markov_draft", "gather_draft_logits",
        "combine_aux_rows", "context_overlap", "defer_context_join")),
        "fast path", model._fast_path)  # fmt: skip
    gen = torch.Generator(device="cuda").manual_seed(0)
    aux = torch.randn((64, 5 * 7168), generator=gen, device="cuda").bfloat16()
    combined = model.combine_hidden_states(aux)
    projected = F.linear(aux, inner.fc.weight)
    stock = torch.empty_like(projected)
    ops.rms_norm(stock, projected, inner.hidden_norm.weight, inner.config.rms_norm_eps)
    print("combine == stock fc + hidden_norm:", torch.equal(combined, stock))
    out = torch.empty_like(combined)
    rows = [h for h in aux.split(7168, dim=-1)]
    model.combine_aux_rows(rows, 0, 64, out)
    print("combine_aux_rows == combine:", torch.equal(out, combined))
    assert type(model).__qualname__ == "K3GqaDSparkForCausalLM"
    assert torch.equal(combined, stock) and torch.equal(out, combined)


if __name__ == "__main__":
    main(*sys.argv[1:3])
