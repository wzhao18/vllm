# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local AttnRes block buffer for Kimi K3.1 prefill.

The prefill AttnRes calls of one forward read ``block_residual`` [T, NB, H]
(BF16, token-major) 816 times per 93-layer forward: about half of all AttnRes
HBM traffic. A default allocation interleaves the two VR200 dies every 4 KiB,
so these reads run at the striped rate. Here the buffer is one persistent VMM
range whose physical memory is homed per 1024-token super-block: tokens
[1024 k, 1024 (k + 1)) live on die k % 2 (1024 tokens x NB rows x 14 KiB =
7 NB whole 2 MiB chunks). ``attn_res_pf`` CTAs claim tokens homed on their own
SM's die (``sm_die`` from the sdec latency probe) and steal the other die's
leftovers, so the block reads are die-local whatever the SM split.

The buffer is allocated at the first prefill forward (vLLM's profiling run at
max_num_batched_tokens, so the KV-cache sizing sees it) and regrown only if a
later forward is longer. Single-die devices use the default allocation."""

import functools

import torch

SUPER_TOKENS = 1024
CHUNK = 2 << 20


@functools.cache
def _two_dies(device: int) -> bool:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    return dies.domain_count(device) == 2


@functools.cache
def _sm_die(device: int) -> torch.Tensor:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    table = dies.sm_domains(device)
    t = torch.zeros(1024, dtype=torch.int8)
    t[: len(table)] = torch.tensor([max(d, 0) for d in table], dtype=torch.int8)
    return t.to(f"cuda:{device}")


class _Buffer:
    def __init__(self, device: int, tokens: int, nb: int, hidden: int):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

        self.nb = nb
        self.hidden = hidden
        self.tokens = -(-tokens // SUPER_TOKENS) * SUPER_TOKENS
        row = nb * hidden * 2
        super_bytes = SUPER_TOKENS * row
        assert super_bytes % CHUNK == 0 and SUPER_TOKENS * hidden * 2 % CHUNK == 0
        per_super = super_bytes // CHUNK
        nbytes = self.tokens * row
        domains = [(c // per_super) & 1 for c in range(nbytes // CHUNK)]
        raw = dies.localized_empty(nbytes, domains, device)
        self.blocks = raw[:nbytes].view(torch.bfloat16).view(self.tokens, nb, hidden)
        self.base = self.blocks.data_ptr()
        # Prefix rows [tokens, H] with the same token -> die homing
        # (1024 tokens x 14 KiB = 7 chunks).
        pbytes = self.tokens * hidden * 2
        pdomains = [(c // (SUPER_TOKENS * hidden * 2 // CHUNK)) & 1
                    for c in range(pbytes // CHUNK)]
        praw = dies.localized_empty(pbytes, pdomains, device)
        self.prefix = praw[:pbytes].view(torch.bfloat16).view(self.tokens, hidden)
        self.prefix_base = self.prefix.data_ptr()
        self.sm_die = _sm_die(device)
        self.claim = torch.zeros(4, dtype=torch.int32, device=f"cuda:{device}")


_buffers: dict[int, _Buffer] = {}


def block_residual(hidden_states: torch.Tensor, nb: int) -> torch.Tensor | None:
    """The die-local [T, nb, H] block buffer for this forward, or None on a
    single-die device (the caller allocates as before)."""
    device = hidden_states.device.index
    if not _two_dies(device):
        return None
    t, hidden = hidden_states.shape
    buf = _buffers.get(device)
    if buf is None or buf.nb != nb or buf.hidden != hidden or buf.tokens < t:
        _buffers.pop(device, None)
        buf = _buffers[device] = _Buffer(device, t, nb, hidden)
    return buf.blocks[:t]


def kernel_args(blocks: torch.Tensor):
    """(sm_die, claim) when ``blocks`` is (a token prefix of) the die-local
    buffer, else (None, None)."""
    buf = _buffers.get(blocks.device.index)
    if (
        buf is not None
        and blocks.data_ptr() == buf.base
        and blocks.stride(0) == buf.nb * buf.hidden
    ):
        return buf.sm_die, buf.claim
    return None, None


def prefix_out(prefix: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor | None:
    """Where the AttnRes should write the updated prefix: the die-local
    prefix buffer when blocks is the die-local block buffer and prefix
    is not already that buffer (else None: in place). The caller adopts the
    returned tensor as its prefix."""
    buf = _buffers.get(blocks.device.index)
    if buf is None or kernel_args(blocks)[0] is None:
        return None
    if prefix.data_ptr() == buf.prefix_base:
        return None
    return buf.prefix[: prefix.shape[0]]
