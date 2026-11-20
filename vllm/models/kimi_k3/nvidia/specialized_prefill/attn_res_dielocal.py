# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local AttnRes block buffer for Kimi K3.1 prefill.

The prefill AttnRes calls of one forward read ``block_residual`` [T, NB, H]
(BF16, token-major) 816 times per 93-layer forward: about half of all AttnRes
HBM traffic. A default allocation interleaves the two VR200 dies every 4 KiB,
so these reads run at the striped rate. Here the buffer is one persistent VMM
range whose physical memory is homed per 1024-token super-block: tokens
[2048 k, 2048 (k + 1)) live on die k % 2 (whole 2 MiB chunks for the
block rows, the prefix and the BF16 / MXFP8 outputs). ``attn_res_pf`` CTAs claim tokens homed on their own
SM's die (``sm_die`` from the sdec latency probe) and steal the other die's
leftovers, so the block reads are die-local whatever the SM split.

The buffer is allocated at the first prefill forward (vLLM's profiling run at
max_num_batched_tokens, so the KV-cache sizing sees it) and regrown only if a
later forward is longer. Single-die devices use the default allocation."""

import functools

import torch

SUPER_TOKENS = 2048
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
        self.device = device
        # Rotating pairs of die-local output buffers (AttnRes outputs consumed
        # within the layer: the KDA in_proj / DSA projections / MoE inputs).
        self.outs: dict[str, list[torch.Tensor]] = {}
        self.out_next: dict[str, int] = {}
        self.claim = torch.zeros(4, dtype=torch.int32, device=f"cuda:{device}")


    def out(self, kind: str, rows: int) -> torch.Tensor:
        """The next of two die-local [tokens, H] output buffers of ``kind``
        ('bf16' or 'fp8'), first ``rows`` rows."""
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

        pair = self.outs.get(kind)
        if pair is None:
            esize = 2 if kind == "bf16" else 1
            row = self.hidden * esize
            per_super = SUPER_TOKENS * row // CHUNK
            assert SUPER_TOKENS * row % CHUNK == 0
            nbytes = self.tokens * row
            domains = [(c // per_super) & 1 for c in range(nbytes // CHUNK)]
            dtype = torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fn
            pair = self.outs[kind] = [
                dies.localized_empty(nbytes, domains, self.device)[:nbytes]
                .view(dtype).view(self.tokens, self.hidden)
                for _ in range(2)
            ]
            self.out_next[kind] = 0
        i = self.out_next[kind]
        self.out_next[kind] = i ^ 1
        return pair[i][:rows]


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


@functools.cache
def _die_sms(device: int) -> tuple[int, int]:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    table = [max(d, 0) for d in dies.sm_domains(device)]
    return table.count(0), table.count(1)


def _die_tokens(rows: int, die: int) -> int:
    """Tokens of a ``rows``-token forward homed on ``die`` (super-block k on
    die k % 2, as the kernel's die_count)."""
    nsb = -(-rows // SUPER_TOKENS)
    full = (nsb - die + 1) >> 1
    c = full * SUPER_TOKENS
    if (nsb - 1) % 2 == die:
        c -= nsb * SUPER_TOKENS - rows
    return c


def die_split(blocks: torch.Tensor, rows: int) -> tuple[int, int]:
    """(xfer, xdie) for attn_res_lp: die ``xdie`` (the one with more SMs) also
    takes the last ``xfer`` tokens of the other die's range, so each die's
    token count follows its SM count (the call's latency-bound token chains
    end together). (0, 0) off the die-local buffer or on balanced parts."""
    if kernel_args(blocks)[0] is None:
        return 0, 0
    n0, n1 = _die_sms(blocks.device.index)
    if n0 == n1 or n0 + n1 == 0:
        return 0, 0
    rx = 0 if n0 > n1 else 1
    want = round(rows * max(n0, n1) / (n0 + n1))
    xfer = max(0, min(want - _die_tokens(rows, rx), _die_tokens(rows, rx ^ 1) - 1))
    return xfer, rx


def is_prefix(prefix: torch.Tensor, blocks: torch.Tensor) -> bool:
    """Whether ``prefix`` is the die-local prefix buffer of ``blocks``' forward
    (the AttnRes prefix the dataflow-C producers update in place)."""
    buf = _buffers.get(blocks.device.index)
    return (
        buf is not None
        and kernel_args(blocks)[0] is not None
        and prefix.data_ptr() == buf.prefix_base
        and prefix.dim() == 2
        and prefix.stride(0) == buf.hidden
    )


def out_buffer(blocks: torch.Tensor, kind: str, rows: int) -> torch.Tensor | None:
    """A die-local output buffer (rotating pair per kind) when ``blocks`` is
    the die-local block buffer, else None. Only for outputs consumed before
    the next-but-one AttnRes call of the same kind."""
    if kernel_args(blocks)[0] is None:
        return None
    return _buffers[blocks.device.index].out(kind, rows)
