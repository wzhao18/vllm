# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-homed KDA prefill buffers for Kimi K3.1.

The o_norm + MXFP8 pass reads the persistent core output O (T, H, 128) BF16
and the gate columns of the g|f_a|beta projection rows (T, N) BF16: about
0.8 GB per F16K layer. A default allocation interleaves the two VR200 dies
every 4 KiB; here both buffers are persistent VMM ranges whose physical memory
is homed per 2048-token super-block (tokens [2048 k, 2048 (k + 1)) on die
k % 2, the ``attn_res_dielocal`` granule), so each o_norm CTA's contiguous row
range lives on one die. Measured on VR200 (same kernel, same data): F16K
92 -> 81 us, F8K 48 -> 44 us.

The buffers are allocated at the first prefill forward (vLLM's profiling run
at max_num_batched_tokens, so the KV-cache sizing sees them) and regrown only
if a later forward is longer. Each is consumed inside the layer that writes
it, so one buffer per device and stream serves every layer. Single-die devices
use the default allocation.
"""

import functools

import torch

SUPER_TOKENS = 2048
CHUNK = 2 << 20


@functools.cache
def _two_dies(device: int) -> bool:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    return dies.domain_count(device) == 2


def _homed(tokens: int, row_bytes: int, device: int) -> torch.Tensor:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    nbytes = tokens * row_bytes
    super_bytes = SUPER_TOKENS * row_bytes
    domains = [
        ((c * CHUNK) // super_bytes) & 1 for c in range((nbytes + CHUNK - 1) // CHUNK)
    ]
    return dies.localized_empty(nbytes, domains, device)[:nbytes]


class _Buffer:
    def __init__(self, device: int, tokens: int, row_elems: int):
        self.tokens = -(-tokens // SUPER_TOKENS) * SUPER_TOKENS
        self.row_elems = row_elems
        self.data = (
            _homed(self.tokens, row_elems * 2, device)
            .view(torch.bfloat16)
            .view(self.tokens, row_elems)
        )


_buffers: dict[tuple[int, int, str], _Buffer] = {}


def _rows(kind: str, device: torch.device, tokens: int, row_elems: int):
    index = device.index if device.index is not None else 0
    if not _two_dies(index):
        return None
    key = (index, torch.cuda.current_stream(device).cuda_stream, kind)
    buf = _buffers.get(key)
    if buf is None or buf.row_elems != row_elems or buf.tokens < tokens:
        _buffers.pop(key, None)
        buf = _buffers[key] = _Buffer(index, tokens, row_elems)
    return buf.data[:tokens]


def core_out(tokens: int, heads: int, dim: int, device: torch.device):
    """Die-homed (1, tokens, heads, dim) BF16 output of the persistent core, or
    None on a single-die device."""
    rows = _rows("core", device, tokens, heads * dim)
    return None if rows is None else rows.view(1, tokens, heads, dim)


def proj_rows(tokens: int, cols: int, device: torch.device):
    """Die-homed (tokens, cols) BF16 rows of the g|f_a|beta projection, or None
    on a single-die device."""
    return _rows("proj", device, tokens, cols)
