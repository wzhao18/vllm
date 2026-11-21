# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`logprob_token_ids` on a speculative-decoding server: every generated
token's `top_logprobs` are the requested ids (plus the sampled token), for
every list length 1..128, and the response formatting never fails on a step
without logprobs."""

import math

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import Logprob

# Any small model works; the chat template below is model independent.
MODEL_NAME = "Qwen/Qwen3-0.6B"
CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
# Repetitive prompt: n-gram drafts match from the first decode step on.
PROMPT = "one two three four five six seven eight " * 8
LENGTHS = list(range(1, 129))


def _token_id(token: str) -> int:
    assert token.startswith("token_id:"), token
    return int(token.removeprefix("token_id:"))


def _step(sampled: int, ids: list[int]) -> dict[int, Logprob]:
    """One step's logprobs as the engine returns them: sampled token first."""
    step = {sampled: Logprob(logprob=-0.5, rank=1)}
    for i, token_id in enumerate(ids):
        step.setdefault(token_id, Logprob(logprob=-1.0 - i, rank=i + 2))
    return step


@pytest.mark.parametrize("n", LENGTHS)
def test_chat_logprobs_return_requested_ids_per_step(n: int):
    serving = object.__new__(OpenAIServingChat)
    serving.return_tokens_as_token_ids = False
    ids = list(range(100, 100 + n))
    token_ids = [7, 8, 9, 10]
    result = serving._create_chat_logprobs(
        token_ids=token_ids,
        top_logprobs=[_step(t, ids) for t in token_ids],
        tokenizer=None,
        logprob_token_ids=ids,
        return_as_token_id=True,
    )
    assert result.content is not None and len(result.content) == len(token_ids)
    for token_id, content in zip(token_ids, result.content):
        assert _token_id(content.token) == token_id
        assert [_token_id(e.token) for e in content.top_logprobs] == [token_id, *ids]


@pytest.mark.parametrize("num_steps_with_logprobs", [0, 1, 3])
def test_logprobs_shorter_than_tokens_do_not_raise(num_steps_with_logprobs: int):
    """A step without logprobs is formatted like a step whose logprobs are None
    (token only) instead of raising IndexError (HTTP 500)."""
    token_ids = [7, 8, 9, 10]
    ids = [100, 101]
    steps = [_step(t, ids) for t in token_ids[:num_steps_with_logprobs]]

    chat = object.__new__(OpenAIServingChat)
    chat.return_tokens_as_token_ids = False
    result = chat._create_chat_logprobs(
        token_ids=token_ids,
        top_logprobs=steps,
        tokenizer=None,
        logprob_token_ids=ids,
        return_as_token_id=True,
    )
    assert result.content is not None and len(result.content) == len(token_ids)
    for i, content in enumerate(result.content):
        assert _token_id(content.token) == token_ids[i]
        assert bool(content.top_logprobs) == (i < num_steps_with_logprobs)

    completion = object.__new__(OpenAIServingCompletion)
    completion.return_tokens_as_token_ids = False
    result = completion._create_completion_logprobs(
        token_ids=token_ids,
        top_logprobs=steps,
        num_output_top_logprobs=len(ids),
        tokenizer=None,
        logprob_token_ids=ids,
        return_as_token_id=True,
    )
    assert len(result.tokens) == len(token_ids)
    assert [lp is not None for lp in result.token_logprobs] == [
        i < num_steps_with_logprobs for i in range(len(token_ids))
    ]


@pytest.fixture(scope="module")
def spec_server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "16",
        "--enforce-eager",
        "--max-logprobs",
        "128",
        "--chat-template",
        CHAT_TEMPLATE,
        "--speculative-config",
        '{"method": "ngram", "num_speculative_tokens": 3,'
        ' "prompt_lookup_max": 4, "prompt_lookup_min": 2}',
    ]
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1"}
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env) as server:
        yield server


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_spec_decode_logprob_token_ids_every_length(spec_server, stream: bool):
    """Lengths 1..128, greedy and sampled, through the speculative-decoding
    rejection sampler: HTTP 200, one logprobs entry per generated token, and on
    every step exactly the requested ids (plus the sampled token)."""
    import asyncio

    async with spec_server.get_async_client() as client:

        async def one(n: int, temperature: float):
            ids = list(range(100, 100 + n))
            kwargs = dict(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=12,
                temperature=temperature,
                logprobs=True,
                extra_body={
                    "logprob_token_ids": ids,
                    "return_tokens_as_token_ids": True,
                    "ignore_eos": True,
                },
            )
            contents = []
            if stream:
                async for chunk in await client.chat.completions.create(
                    stream=True, **kwargs
                ):
                    if chunk.choices and chunk.choices[0].logprobs is not None:
                        contents.extend(chunk.choices[0].logprobs.content or [])
            else:
                resp = await client.chat.completions.create(**kwargs)
                assert resp.usage.completion_tokens == 12
                contents = resp.choices[0].logprobs.content
            return ids, contents

        jobs = [one(n, t) for n in LENGTHS for t in (0.0, 0.8)]
        for ids, contents in await asyncio.gather(*jobs):
            assert len(contents) == 12, len(ids)
            for content in contents:
                sampled = _token_id(content.token)
                returned = [_token_id(e.token) for e in content.top_logprobs]
                assert returned[0] == sampled
                assert sorted(set(returned)) == sorted({sampled, *ids}), len(ids)
                assert len(returned) == len(set(returned))
                assert all(not math.isnan(e.logprob) for e in content.top_logprobs)
