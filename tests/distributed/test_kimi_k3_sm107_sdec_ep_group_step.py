# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 EP_GROUP specialized decode (SM107), whole-step equivalence on several
VR200 nodes (skeleton: the harness and the checks; it runs only when launched as
below, on Rubin, with a K3.1 checkpoint).

Cases (4 GPUs per node):

  case                 topology                       nodes  step (Topology.step_layout)
  dp2-tp4dcp4-ep8      DP2 x TP4 = DCP4, EP8 MegaMoE  2      SP step, layout (c)
  dp2-tp8dcp8-ep16     DP2 x TP8 = DCP8, EP16         4      replicated step, layout (b)

Each rank runs one ``LLM`` (``distributed_executor_backend="external_launcher"``,
DP rank from RANK) with the specialized decode on, the deployment's decode engine
args from ``K31_DECODE_ENGINE_ARGS`` (JSON: KV / indexer dtypes, attention backend,
kernel config, DSpark speculative config, mamba dtypes; the production recipe's) and
the case's parallel sizes, at most 64 verify rows per group (max_num_seqs = 64 / q).
Checks, per DP replica (its own prompts; replica 1 gets fewer, so it runs padded
dummy steps while replica 0 decodes: MegaMoE's EP collective must still join):

* startup: the decode's step is the topology's layout (``group_ep.GroupStep`` or
  ``sp_attention.SpStep``) and its coverage table is complete (every captured R
  specialized or a declared stock seam of ``GroupStep.allowed`` /
  ``DECLARED_STOCK``);
* greedy-token match: each greedy (temperature 0) continuation decoded by the
  specialized step equals the argmax of the stock forward over the same tokens
  (a teacher-forced prefill of prompt + continuation: prefills always run the stock
  forward), except near-ties (top-1 / chosen logprob gap <= ``TIE``);
* hidden-state tolerance: the decoded tokens' logprobs (the specialized step's final
  hidden states through the LM head) against the teacher-forced stock forward's
  logprobs of the same tokens at the same positions: median |delta| <= ``MEDIAN``,
  max <= ``WORST`` nats (BF16 / FP32 reduction-order differences between the decode
  and prefill kernels, not a precision change);
* every TP rank of a replica returns the same tokens; every rank finishes within
  ``TIMEOUT`` (a hung EP collective fails the case, never CI).

Launch on every node of the allocation (node rank 0..N-1):

  K31_MODEL=/path/to/kimi-k3.1 K31_DECODE_ENGINE_ARGS='{"kv_cache_dtype": "fp8",
  ...}' torchrun --nnodes N --nproc-per-node 4 --node-rank $NODE --master-addr $HEAD
  --master-port 29511 -m pytest -x -q tests/distributed/
  test_kimi_k3_sm107_sdec_ep_group_step.py -k dp2-tp8dcp8-ep16

Not covered here: the per-layer A/B of each block against the stock layer (the
1-node tests/distributed/test_kimi_k3_sm107_sdec_sp_layers.py harness at TP4; the
TP8 one is a follow-up), the DP4 x TP8 EP32 point (8 nodes, W2's third row)."""

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
    "dp2-tp4dcp4-ep8": dict(tp=4, dcp=4, dp=2),
    "dp2-tp8dcp8-ep16": dict(tp=8, dcp=8, dp=2),
}
MAX_GROUP_ROWS = 64  # runtime.EP_GROUP_MAX_ROWS
NEW_TOKENS = 48
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
        pytest.skip("the EP_GROUP decode verifies DSpark q 2..8 (no q1)")
    additional = dict(args.pop("additional_config", None) or {})
    additional["decode_backend"] = "latency_specialized"
    args.update(
        model=os.environ["K31_MODEL"],
        tensor_parallel_size=sizes["tp"],
        decode_context_parallel_size=sizes["dcp"],
        data_parallel_size=sizes["dp"],
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        additional_config=additional,
        max_num_seqs=MAX_GROUP_ROWS // q,
        seed=0,
    )
    return args, q


def _replica_prompts(dp_rank):
    """Replica 0 decodes every prompt, replica 1 two: while it idles, replica 0's
    steps meet its padded dummy steps in MegaMoE's EP collective. (Only the load
    differs: every replica makes the same ``llm.generate`` calls, two in all.)"""
    return PROMPTS if dp_rank == 0 else PROMPTS[:2]


def _consistent(group, obj):
    """Whether every rank of ``group`` holds ``obj`` (rank 0's broadcast)."""
    import torch.distributed as dist

    box = [obj]
    dist.broadcast_object_list(box, src=group.ranks[0], group=group.cpu_group)
    return box[0] == obj


def _check_startup(llm, topology_layout):
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (
        group_ep,
        sp_attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.topology import (
        Layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (
        DECLARED_STOCK,
    )

    worker = llm.llm_engine.model_executor.driver_worker.worker
    decode = worker.model_runner.model.specialized_decode
    assert decode is not None, "the specialized decode is not built"
    if topology_layout is Layout.ROW_SHARDED:
        assert isinstance(decode.sp, sp_attention.SpStep), type(decode.sp)
        decode.coverage.assert_complete(decode.captured, DECLARED_STOCK)
    else:
        assert isinstance(decode.sp, group_ep.GroupStep), type(decode.sp)
        assert decode.sp.allowed <= group_ep.GROUP_DECLARED, decode.sp.allowed
        decode.coverage.assert_complete(decode.captured, decode.sp.allowed)
    return decode


def _teacher_forced(llm, prompts, continuations, sampling_params):
    """The stock forward's log-softmax rows over each prompt + continuation (the
    replica's prefills, in ONE ``llm.generate`` call: under external_launcher DP the
    replicas stay in lockstep only inside a generate call, so every replica makes
    the same calls whatever its prompt count): per prompt, for each continuation
    position but the last (whose token the prefill does not see), (top-1 token, its
    logprob, the chosen token's logprob)."""
    params = sampling_params(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    requests = [
        {"prompt_token_ids": list(p) + list(c[:-1])}
        for p, c in zip(prompts, continuations)
    ]
    outputs = llm.generate(requests, params, use_tqdm=False)
    results = []
    for prompt_ids, continuation, out in zip(prompts, continuations, outputs):
        rows = out.prompt_logprobs[len(prompt_ids) :]
        result = []
        for chosen, row in zip(continuation, rows):
            top = max(row.items(), key=lambda item: item[1].logprob)
            result.append((top[0], top[1].logprob, row[chosen].logprob))
        results.append(result)
    return results


def _all_passed(ok):
    """Whether every rank of the world passed (a MIN all-reduce on the CPU group):
    a rank that fails never leaves the others in a collective it skipped."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_world_group

    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_world_group().cpu_group)
    return bool(flag.item())


@pytest.mark.parametrize("case", list(CASES))
def test_ep_group_step_equivalence(case):
    sizes = CASES[case]
    world = sizes["tp"] * sizes["dp"]
    if int(os.environ["WORLD_SIZE"]) != world:
        pytest.skip(f"{case} needs {world} ranks ({world // GPUS_PER_NODE} nodes)")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the SM107 specialized decode runs on Rubin only")
    faulthandler.dump_traceback_later(TIMEOUT, exit=True)  # a hang fails the case
    try:
        _run_case(case, sizes)
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_case(case, sizes):
    import torch.distributed as dist

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.topology import (
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
    assert (topology.g, topology.dcp, topology.dp) == (
        sizes["tp"], sizes["dcp"], sizes["dp"]
    ), topology  # fmt: skip
    # Every rank makes the same two generate calls (greedy, then the batched
    # teacher-forced prefills) and only then asserts, after a world-wide pass / fail
    # all-reduce: a check failing on one rank never strands the others in a DP /
    # EP collective (the failure shows instead of a TIMEOUT hang).
    problems = []
    try:
        _check_startup(llm, topology.step_layout)
    except AssertionError as error:
        problems.append(f"startup: {error}")
    dp_rank = config.parallel_config.data_parallel_rank
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.encode(p) for p in _replica_prompts(dp_rank)]
    greedy = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, logprobs=0)
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], greedy, use_tqdm=False
    )
    tp_group = get_tp_group()
    continuations, decoded = [], []
    for index, output in enumerate(outputs):
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        if not _consistent(tp_group, tokens):  # every TP rank of the replica agrees
            problems.append(f"prompt {index}: TP ranks disagree on the tokens")
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
    # Every rank (both replicas) reached here: the EP collectives all joined.
    passed = _all_passed(not problems)
    assert not problems, f"{case} replica {dp_rank}: " + "; ".join(problems)
    assert passed, f"{case}: another rank failed (its own log names the cause)"
