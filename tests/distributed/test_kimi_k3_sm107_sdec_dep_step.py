# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DEP specialized decode (SM107, ``runtime.dep``), whole-step equivalence
on several VR200 nodes (skeleton: the harness and the checks; it runs only when
launched as below, on Rubin, with a K3.1 checkpoint).

Cases (4 GPUs per node):

  case        topology                          nodes  MoE backend
  dep16-dg    TP1 x DP16, EP16 MegaMoE          4      deep_gemm_mega_moe
  dep16-kk    TP1 x DP16, EP16 MegaMoE          4      kimi_k3_mega_moe (its EP16
                                                       record for the MNBT bucket)
  dep32-dg    TP1 x DP32, EP32 MegaMoE          8      deep_gemm_mega_moe

Each rank runs one ``LLM`` (``distributed_executor_backend="external_launcher"``,
DP rank from RANK) with the specialized decode on, the deployment's decode engine
args from ``K31_DECODE_ENGINE_ARGS`` (JSON: per-tensor fp8 KV, FLASHINFER_MLA_SPARSE
trtllm-gen, kernel config, DSpark q 2..8, BF16 conv / FP32 recurrent KDA state; no
--use-replayssm: B9) and the case's parallel sizes. The DP ranks get different loads
so that every mode of the collective symmetry (plan section 5) occurs:

* ``dp_rank % 4 == 0``: every prompt (decodes longest);
* ``dp_rank % 4 == 1``: one prompt (finishes early: its later steps are DP dummy
  steps replaying the padded captured graph while the others decode);
* ``dp_rank % 4 == 2``: a long prompt (``LONG_PROMPT_TOKENS``, chunked prefill: its
  prefill / mixed steps meet the others' decode steps; every rank then runs the
  stock forward, the same MegaMoE collective per MoE layer);
* ``dp_rank % 4 == 3``: two prompts.

Checks, per DP rank:

* startup: the decode's step is ``dep.DepStep``, its allowed seams exactly
  ``dep.DEP_DECLARED`` and its coverage table complete;
* greedy-token match: each greedy continuation decoded by the specialized step
  equals the argmax of the stock forward over the same tokens (a teacher-forced
  prefill), except near-ties (top-1 / chosen logprob gap <= ``TIE``);
* hidden-state tolerance: the decoded tokens' logprobs against the teacher-forced
  stock forward's: median |delta| <= ``MEDIAN``, max <= ``WORST`` nats;
* every rank finishes within ``TIMEOUT`` (a hung EP collective fails the case).

gsm8k on the DEP16 deployment is informational only (not part of this test).

Launch on every node of the allocation (node rank 0..N-1):

  K31_MODEL=/path/to/kimi-k3.1 K31_DECODE_ENGINE_ARGS='{"kv_cache_dtype": "fp8",
  ...}' torchrun --nnodes N --nproc-per-node 4 --node-rank $NODE --master-addr $HEAD
  --master-port 29512 -m pytest -x -q tests/distributed/
  test_kimi_k3_sm107_sdec_dep_step.py -k dep16-dg

Not covered here: a per-layer A/B of the DEP blocks against the stock layers (the
DSA layer at 96 heads, padded trtllm-gen query, against ``KimiK3DSAAttention``)."""

import faulthandler
import json
import os
import statistics

import pytest
import torch

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or "K31_MODEL" not in os.environ,
    reason="launched by torchrun on VR200 nodes with K31_MODEL (module docstring)",
)

GPUS_PER_NODE = 4
CASES = {
    "dep16-dg": dict(dp=16, backend="deep_gemm_mega_moe"),
    "dep16-kk": dict(dp=16, backend="kimi_k3_mega_moe"),
    "dep32-dg": dict(dp=32, backend="deep_gemm_mega_moe"),
}
MAX_NUM_SEQS = 16  # requests per DP rank (R = 16 q rows; the gate caps 512)
NEW_TOKENS = 48
LONG_PROMPT_TOKENS = 6000
TIE = 0.05  # nats: a top-1 the decode may legitimately flip
MEDIAN, WORST = 0.02, 0.25  # nats: decode vs prefill logprobs of the same token
TIMEOUT = 5400  # s per rank: model load, CuTe compiles, captures, generation
PROMPTS = [
    "The history of the printing press begins",
    "Write a Python function that merges two sorted lists.",
    "Explain why the sky is blue to a ten-year-old.",
    "List the planets of the solar system in order.",
    "A train leaves Paris at 9 am travelling at 300 km/h.",
    "Translate into French: the cat sleeps on the warm windowsill.",
    "def fibonacci(n):",
    "The three laws of thermodynamics state that",
]


def _engine_args(case):
    sizes = CASES[case]
    args = json.loads(os.environ.get("K31_DECODE_ENGINE_ARGS", "{}"))
    spec = args.get("speculative_config") or {}
    q = 1 + int(spec.get("num_speculative_tokens", 0))
    if q < 2:
        pytest.skip("the DEP decode verifies DSpark q 2..8 (no q1)")
    if args.get("use_replayssm"):
        pytest.skip("ReplaySSM / RecoverSSM at DEP is B9")
    additional = dict(args.pop("additional_config", None) or {})
    additional["decode_backend"] = "latency_specialized"
    kernel = dict(args.pop("kernel_config", None) or {})
    kernel["moe_backend"] = sizes["backend"]
    args.update(
        model=os.environ["K31_MODEL"],
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=sizes["dp"],
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        additional_config=additional,
        kernel_config=kernel,
        max_num_seqs=MAX_NUM_SEQS,
        seed=0,
    )
    return args, q


def _replica_prompts(dp_rank, tokenizer):
    """The DP rank's prompt token ids (module docstring: different loads per rank;
    only the load differs, every rank makes the same ``llm.generate`` calls)."""
    role = dp_rank % 4
    if role == 0:
        texts = PROMPTS
    elif role == 1:
        texts = PROMPTS[:1]
    elif role == 3:
        texts = PROMPTS[:2]
    else:
        filler = " ".join(PROMPTS)
        ids = tokenizer.encode(filler)
        long = (ids * (LONG_PROMPT_TOKENS // len(ids) + 1))[:LONG_PROMPT_TOKENS]
        return [long, tokenizer.encode(PROMPTS[3])]
    return [tokenizer.encode(p) for p in texts]


def _check_startup(llm):
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import dep

    worker = llm.llm_engine.model_executor.driver_worker.worker
    decode = worker.model_runner.model.specialized_decode
    assert decode is not None, "the specialized decode is not built"
    assert isinstance(decode.sp, dep.DepStep), type(decode.sp)
    assert decode.sp.allowed == dep.DEP_DECLARED, decode.sp.allowed
    decode.coverage.assert_complete(decode.captured, decode.sp.allowed)
    ctx = decode.context
    assert ctx.dep and (ctx.tp, ctx.dcp, ctx.heads_per_rank) == (1, 1, 96), ctx
    return decode


@pytest.mark.parametrize("case", list(CASES))
def test_dep_step_equivalence(case):
    world = CASES[case]["dp"]
    if int(os.environ["WORLD_SIZE"]) != world:
        pytest.skip(f"{case} needs {world} ranks ({world // GPUS_PER_NODE} nodes)")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the SM107 specialized decode runs on Rubin only")
    faulthandler.dump_traceback_later(TIMEOUT, exit=True)  # a hang fails the case
    try:
        _run_case(case)
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_case(case):
    import torch.distributed as dist

    from tests.distributed.test_kimi_k3_sm107_sdec_ep_group_step import (
        _all_passed,
        _teacher_forced,
    )
    from vllm import LLM, SamplingParams
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.topology import (
        Layout,
        Mode,
        Topology,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", device_id=torch.device("cuda", local_rank)
        )
    args, q = _engine_args(case)
    llm = LLM(**args)
    config = llm.llm_engine.vllm_config
    topology = Topology.from_config(config)
    assert topology.mode is Mode.DEP and topology.step_layout is Layout.LOCAL, topology
    assert (topology.g, topology.dcp, topology.dp) == (1, 1, CASES[case]["dp"])
    # Every rank makes the same two generate calls and only then asserts, after a
    # world-wide pass / fail all-reduce (as the EP_GROUP harness).
    problems = []
    try:
        _check_startup(llm)
    except AssertionError as error:
        problems.append(f"startup: {error}")
    dp_rank = config.parallel_config.data_parallel_rank
    prompts = _replica_prompts(dp_rank, llm.get_tokenizer())
    greedy = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, logprobs=0)
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], greedy, use_tqdm=False
    )
    continuations, decoded = [], []
    for output in outputs:
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        continuations.append(tokens)
        decoded.append(
            [row[t].logprob for t, row in zip(tokens, completion.logprobs)]
        )
    forced_rows = _teacher_forced(llm, prompts, continuations, SamplingParams)
    flips, deltas = [], []
    for tokens, lps, forced in zip(continuations, decoded, forced_rows):
        for i, (chosen, (top, top_lp, chosen_lp)) in enumerate(zip(tokens, forced)):
            if top != chosen and top_lp - chosen_lp > TIE:
                flips.append((i, chosen, top, top_lp - chosen_lp))
            deltas.append(abs(lps[i] - chosen_lp))
    if flips:
        problems.append(
            f"greedy tokens off the stock argmax beyond near-ties: {flips[:8]}"
        )
    if not deltas:
        problems.append("no decoded tokens to compare")
    elif statistics.median(deltas) > MEDIAN or max(deltas) > WORST:
        problems.append(
            f"logprob |delta| median {statistics.median(deltas):.4f} (<= {MEDIAN}), "
            f"max {max(deltas):.4f} (<= {WORST})"
        )
    # Every rank (idle, prefilling and decoding ones) reached here: the EP
    # collectives all joined.
    passed = _all_passed(not problems)
    assert not problems, f"{case} DP rank {dp_rank}: " + "; ".join(problems)
    assert passed, f"{case}: another rank failed (its own log names the cause)"
