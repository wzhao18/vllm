# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Retention rule ported from TensorRT-LLM's per-conversation block reuse
# policy (tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py,
# ConversationManager), Apache-2.0,
# Copyright (c) NVIDIA CORPORATION & AFFILIATES.
"""Per-conversation retention of recurrent (Mamba) prefix-cache states.

A multi-turn conversation registers new Mamba states every turn, and each
turn resumes from the previous one, so a turn's states are rarely hit again
once a few later turns exist. ``CacheConfig.mamba_conversation_max_turns``
keeps the states of the last N completed turns of a conversation and drops
older ones from the prefix cache (TensorRT-LLM ``block_reuse_config.policy:
per_conversation``, ``max_num_turns``). Attention blocks are not affected.
The conversation is ``Request.session_id`` or, failing that,
``SamplingParams.extra_args["conversation_id"]``.

Only one request per conversation is tracked at a time: a request that
arrives while another of its conversation is active is ignored, as in
TensorRT-LLM. Requests without a conversation id are never tracked.
"""

from collections import OrderedDict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field

from vllm.v1.request import Request


@dataclass
class _ConversationState:
    current_request_id: str | None = None
    turns: deque[frozenset[Hashable]] = field(default_factory=deque)


class ConversationStateTracker:
    def __init__(self, max_num_turns: int, max_conversations: int = 65536) -> None:
        assert max_num_turns > 0
        self.max_num_turns = max_num_turns
        self.max_conversations = max_conversations
        self._conversations: OrderedDict[str, _ConversationState] = OrderedDict()
        self.num_turns_dropped = 0

    @staticmethod
    def conversation_id(request: Request) -> str | None:
        """``Request.session_id``, else ``extra_args["conversation_id"]``."""
        conversation_id = request.session_id
        if not conversation_id:
            params = request.sampling_params
            extra_args = params.extra_args if params is not None else None
            conversation_id = (extra_args or {}).get("conversation_id")
        if not isinstance(conversation_id, str):
            return None
        return conversation_id.strip() or None

    def prepare(self, request: Request) -> None:
        """Make ``request`` its conversation's current one, unless another is."""
        conversation_id = self.conversation_id(request)
        if conversation_id is None:
            return
        state = self._conversations.get(conversation_id)
        if state is None:
            state = self._conversations[conversation_id] = _ConversationState()
            while len(self._conversations) > self.max_conversations:
                # Forgotten, not dropped: its states stay under normal eviction.
                self._conversations.popitem(last=False)
        self._conversations.move_to_end(conversation_id)
        if state.current_request_id is None:
            state.current_request_id = request.request_id

    def finish(
        self, request: Request, state_hashes: set[Hashable], completed: bool
    ) -> set[Hashable]:
        """Record a finished request's states; return the states to drop.

        ``completed``: the request finished its prefill, so it is a turn. Once
        a conversation holds more than ``max_num_turns`` turns, the oldest
        one's states are returned, minus any a retained turn also registered.
        """
        conversation_id = self.conversation_id(request)
        if conversation_id is None:
            return set()
        state = self._conversations.get(conversation_id)
        if state is None or state.current_request_id != request.request_id:
            return set()
        state.current_request_id = None
        dropped: set[Hashable] = set()
        if completed:
            state.turns.append(frozenset(state_hashes))
            if len(state.turns) > self.max_num_turns:
                oldest = state.turns.popleft()
                retained = frozenset().union(*state.turns)
                dropped = set(oldest - retained)
                self.num_turns_dropped += 1
        if not state.turns:
            del self._conversations[conversation_id]
        return dropped

    def clear(self) -> None:
        self._conversations.clear()
