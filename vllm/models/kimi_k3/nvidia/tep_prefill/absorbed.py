# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Absorbed MLA prefill for short continuations over a long cached context.

Agentic prefill mostly appends a few hundred tokens (a tool result) to 20K-200K
cached tokens. The stock path up-projects the whole cached context to per-head K/V
and runs a dense attention kernel, so its cost is nearly independent of the number
N of new tokens. In the absorbed form the new tokens' queries are projected into
the latent space (W_UK) and attend over the latent cache directly, like decode
rows: the DEP decode's work-list MLA kernel with N query rows per request (causal
among them, bottom-right), then the V up-projection (W_UV). Its cost scales with
N, so for small N it is several times cheaper.

A step whose prefills are all short runs only this path. A mixed step
(``HybridPlan``) runs the stock path for every prefill with the short requests'
context chunks removed, then overwrites those requests' rows with this path; each
request's rows are padded at the front to a query bucket (padded rows attend to
fewer keys and are discarded).
"""

import dataclasses

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

BUCKETS = (32, 128, 256, 512, 1024)
MAX_NEW = BUCKETS[-1]
MAX_REQUESTS = 8  # prefill requests per step on this path
LATENT, ROPE = 512, 64


def _capacity(q: int) -> int:
    """KV splits per request: few query tiles need splits to fill the GPU."""
    return 8 if q <= 32 else 4 if q <= 128 else 2 if q <= 256 else 1


@dataclasses.dataclass
class HybridPlan:
    """A mixed step: ``requests`` (prefill-relative indices) run absorbed, the rest
    stock. ``chunked_context`` is the step's chunked context without the chunks of
    those requests (their rows are left to the empty-context path), or None when no
    chunk remains."""

    requests: list[int]
    chunked_context: object | None


class AbsorbedPrefill:
    """The absorbed prefill of a deployment's MLA layers (shared by all layers)."""

    _shared: "AbsorbedPrefill | None" = None

    @classmethod
    def get(cls, module) -> "AbsorbedPrefill":
        if cls._shared is None:
            cls._shared = cls(module)
        return cls._shared

    def __init__(self, module):
        from vllm.models.kimi_k3.nvidia.dep_decode.mla_decode import WorkListMla

        device = torch.device("cuda", torch.cuda.current_device())
        self.heads = module.num_local_heads
        self.cores = {}
        for q in BUCKETS:
            self.cores[q] = WorkListMla(
                self.heads,
                q,
                MAX_REQUESTS * q,
                device,
                capacity=_capacity(q),
                max_waves=max(8, -(-MAX_REQUESTS * q * self.heads // 128 // 64)),
            )
        # One workspace for every bucket (a step uses one).
        size = max(core.workspace.numel() for core in self.cores.values())
        workspace = torch.empty(size, dtype=torch.int8, device=device)
        for core in self.cores.values():
            core.workspace = workspace[: core.workspace.numel()]
        self.page_size = None

    def eligible(self, prefill, fp8_prefill: bool) -> bool:
        """Whether every prefill of the step runs this path."""
        if not fp8_prefill or prefill.max_query_len > MAX_NEW:
            return False
        requests = prefill.query_start_loc.shape[0] - 1
        return 0 < requests <= MAX_REQUESTS

    def hybrid(self, prefill, fp8_prefill: bool) -> HybridPlan | None:
        """The step's mixed plan (computed once per step), None when no prefill
        would run this path."""
        plan = getattr(prefill, "_tep_hybrid", False)
        if plan is not False:
            return plan
        plan = None
        context = prefill.chunked_context
        if fp8_prefill and context is not None and prefill.query_lens_cpu is not None:
            lens = prefill.query_lens_cpu.tolist()
            contexts = context.context_lens_list
            short = [
                r
                for r, (n, c) in enumerate(zip(lens, contexts))
                if 0 < n <= MAX_NEW and c > 0
            ][:MAX_REQUESTS]
            if short:
                plan = HybridPlan(short, self._without(context, set(short)))
                logger.info_once("Kimi K3 TEP prefill runs hybrid absorbed MLA steps")
        prefill._tep_hybrid = plan
        return plan

    @staticmethod
    def _without(context, requests: set[int]):
        """``context`` without the chunks that hold only ``requests``."""
        keep, skip = [], []
        for chunk in context.chunks:
            span = range(chunk.request_slice.start, chunk.request_slice.stop)
            (skip if all(r in requests for r in span) else keep).append(chunk)
        if not keep:
            return None
        empty = list(context.empty_token_slices)
        for chunk in skip:
            if chunk.token_slice not in empty:
                empty.append(chunk.token_slice)
        return dataclasses.replace(context, chunks=keep, empty_token_slices=empty)

    def run(
        self,
        module,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions,
        cos_sin_cache,
        slot_mapping: torch.Tensor,
        prefill,
        out: torch.Tensor,
        requests: list[int] | None = None,
    ) -> None:
        """Attention of the prefill tokens (q [T, H, 192]) into out [T, H * 128]
        (only ``requests``' rows, prefill-relative, when given); also inserts their
        latent into the cache."""
        m = module
        starts = prefill.query_start_loc
        lens_cpu = prefill.query_lens_cpu
        if lens_cpu is None:
            lens_cpu = (starts[1:] - starts[:-1]).cpu()
        all_lens = lens_cpu.tolist()
        if requests is None:
            requests = list(range(len(all_lens)))
            rows = None  # every prefill token, in order
        else:
            offsets = [0]
            for n in all_lens:
                offsets.append(offsets[-1] + n)
            rows = torch.cat(
                [torch.arange(offsets[r], offsets[r + 1]) for r in requests]
            ).to(q.device, non_blocking=True)
        lens = [all_lens[r] for r in requests]

        def pick(x):
            return x if rows is None or x is None else x.index_select(0, rows)

        tokens = sum(lens)
        q_nope, q_pe = pick(q).split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
        # Absorb q_nope into the latent space: (H, T, P) x (H, P, L) -> (T, H, L).
        ql_nope = torch.bmm(q_nope.transpose(0, 1), m.W_UK_T).transpose(0, 1)
        query = m._decode_concat_cache(
            ql_nope,
            q_pe,
            pick(kv_c_normed),
            pick(k_pe),
            pick(positions),
            cos_sin_cache,
            pick(slot_mapping),
        )
        bucket = next(b for b in BUCKETS if b >= max(lens))
        core = self.cores[bucket]
        if self.page_size is None:
            self.page_size = m.kv_cache.shape[1]
            for c in self.cores.values():
                c.page_size = self.page_size
            logger.info_once("Kimi K3 TEP prefill runs absorbed MLA continuations")
        count = len(requests)
        # Row i * bucket + (bucket - N_i) + t holds request i's token t.
        index = torch.cat(
            [
                torch.arange(i * bucket + bucket - n, (i + 1) * bucket)
                for i, n in enumerate(lens)
            ]
        ).to(q.device, non_blocking=True)
        padded = query.new_zeros((count * bucket, self.heads, LATENT + ROPE))
        # (FP8 has no index_copy: move the bytes.)
        padded.view(torch.uint8).index_copy_(0, index, query[:tokens].view(torch.uint8))
        picked = torch.tensor(requests, device=q.device)
        seq_lens = (starts[picked + 1] - starts[picked]).to(torch.int32)
        context = prefill.chunked_context
        if context is not None:
            seq_lens = seq_lens + context.context_lens[picked].to(torch.int32)
        core.plan(seq_lens)
        latent = core.decode(
            padded,
            m._attn_read_kv_cache(),
            prefill.block_table[picked],
            seq_lens,
            m.impl.scale * m._q_scale_float * m._k_scale_float,
            m._k_scale_float,
        )
        result = latent.index_select(0, index)
        if rows is None:
            m._v_up_proj(result, out=out[:tokens])
        else:
            values = out.new_empty((tokens, out.shape[-1]))
            m._v_up_proj(result, out=values)
            out.index_copy_(0, rows, values)
