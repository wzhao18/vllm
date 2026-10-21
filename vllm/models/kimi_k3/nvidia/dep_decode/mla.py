# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode's MLA core for a deployment's MLA layers: one split-KV work list
planned per step (``plan``), then the FP8 latent attention of each layer (see
``mla_decode.WorkListMla``)."""

import torch

from .mla_decode import WorkListMla


class MlaCore:
    """``WorkListMla`` bound to the stock MLA modules' scales and caches, optionally
    sharing work across the EP ranks (``enable_share``, see ``mla_share.py``)."""

    def __init__(self, module, max_tokens: int, q: int):
        self.heads, self.q = module.num_local_heads, q
        self.latent, self.rope = module.kv_lora_rank, module.qk_rope_head_dim
        self.inner = WorkListMla(self.heads, q, max_tokens, module.W_UV.device)
        self.share = None
        self.modules: list = []

    def enable_share(self, share, modules) -> None:
        """Share MLA work across the EP ranks for ``modules`` (the MLA layers, in
        order)."""
        self.share, self.modules = share, list(modules)
        self.layer_of = {id(m): i for i, m in enumerate(self.modules)}

    def begin_step(self, md) -> None:
        """Plan this step (every MLA layer reads the same lengths)."""
        if self.share is None:
            self.inner.plan(md.decode.seq_lens)
            return
        own_lens, _ = self.share.begin_step(md.decode.seq_lens, md.decode.block_table)
        self.inner.plan(own_lens)

    def bind(self, module) -> None:
        """Take the page size of ``module``'s bound cache (all MLA layers share it)."""
        self.inner.page_size = module.kv_cache.shape[1]

    def plan(self, seq_lens: torch.Tensor) -> None:
        self.inner.plan(seq_lens)

    def __call__(self, module, query, md) -> torch.Tensor:
        """Latent attention output [T, H, L] of an FP8 query [T, H, L + R]."""
        if self.share is not None:
            return self.share.attend(
                self.layer_of[id(module)],
                self.inner,
                query,
                module._attn_read_kv_cache(),
                module.impl.scale * module._q_scale_float * module._k_scale_float,
                module._k_scale_float,
            )
        return self.inner.decode(
            query,
            module._attn_read_kv_cache(),
            md.decode.block_table,
            md.decode.seq_lens,
            module.impl.scale * module._q_scale_float * module._k_scale_float,
            module._k_scale_float,
        )

    def warm_up(self, module, requests: int, q: int) -> None:
        """Compile the planner and the decode for ``requests`` requests of ``q``
        tokens: one-token requests on the first block of ``module``'s bound cache
        (read only)."""
        device = module.W_UV.device
        seq_lens = torch.ones(requests, dtype=torch.int32, device=device)
        block_table = torch.zeros((requests, 1), dtype=torch.int32, device=device)
        query = torch.zeros(
            (requests * q, self.heads, self.latent + self.rope),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.inner.plan(seq_lens)
        self.inner.decode(
            query, module._attn_read_kv_cache(), block_table, seq_lens, 1.0, 1.0
        )

    def bind_share(self) -> None:
        """Bind the shared work's caches and compile its kernels (one eager step of
        the protocol, outside graph capture)."""
        share = self.share
        share.bind_caches([m._attn_read_kv_cache() for m in self.modules])
        device = self.modules[0].W_UV.device
        requests = 1
        seq_lens = torch.ones(requests, dtype=torch.int32, device=device)
        block_table = torch.zeros((requests, 1), dtype=torch.int32, device=device)
        query = torch.zeros(
            (requests * self.q, self.heads, self.latent + self.rope),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        own_lens, own_bt = share.begin_step(seq_lens, block_table)
        self.inner.plan(own_lens)
        module = self.modules[0]
        share.attend(0, self.inner, query, module._attn_read_kv_cache(), 1.0, 1.0)
