# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DcpExchange's top-k in the Kimi K3.1 DSA decode under DCP (any interleave).

One ``DcpxDecode`` per model (one set of symmetric mailboxes, every DSA layer in
turn; the device epochs make CUDA graph replays carry no host state). On pure-decode
steps of at most ``cap`` rows it replaces the stock local top-k + generic all-gather
merge + ``triton_filter_and_convert_dcp_index`` (or the #63 one-round DCP top-k
merge) by ``DcpExchange.topk_merge`` on the bf16 indexer logits. Its output is
already this rank's KV slots (through the MLA block table, as
``triton_filter_and_convert_dcp_index`` maps them) and their count, so it is
independent of the interleave. Hooks: the indexer op's decode branch
(``sparse_attn_indexer.DCP_DECODE_TOPK_HOOKS``), the sparse MLA layer's DCP
conversion (``SparseMLACommonImpl.dcp_local_topk_counts``) and
``PreparedDsa.dcp_core``.

The selection is the stock one: the exact global top-K of the logits, ties at the
K-th key by ascending global token id (the stock merge's rule; ``TIE_ORDER``).

Policy (``wanted``, no switch): built on a decode instance (not a prefill-only P/D
producer) under 1 < DCP == TP (head owner = rank) on SM107 (Rubin, where it was
measured), where DeepGEMM's MX-scaled MQA logits give bf16 logits; every pure-decode
step up to ``topk_cap`` rows. Other layouts keep the stock DCP path, logged.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Serving spin deadline (s): long enough that a rank stalled on the host (cold JIT, GC,
# a P/D handshake) only delays its peers, as a stock collective would. The kernels'
# own 5 s default applies to tests and the build-time warmup only.
SERVING_SPIN_S = 600.0
# Ties at the K-th key by ascending global token id: the stock merge's rule (one extra
# NVLink round only when the ties straddle two or more ranks). "rank" order (no extra
# round, the same selected score multiset) selects other keys than stock on ties,
# which bf16 logits make common. Cost of the stock rule at production shapes: envs
# lane bench_dcp_topk_ties (TIES-COST).
TIE_ORDER = "global"
# Rows that take the one-round top-k: each rank sends 16-bit counts of a 256-key window
# centred at its local estimate of the K-th key (its local ceil(K * L / G)-th key) and
# its 256-key bin counts; rows whose K-th key falls outside the windows take one refine
# round over the K-th key's bin, so the selection is exact either way. Old DCP16 q8
# step (AgentX): B1 5731 -> 5635 us with it.
ONE_ROUND_ROWS = 256
# Most bf16 keys a top-k CTA stages in shared memory (the kernel's 96 KB opt-in), and
# a bound on its static shared memory (bytes).
SMEM_KEYS = 48 * 1024
STATIC_SMEM = 20 * 1024
# 256-thread top-k CTAs per SM (Rubin: 1024 threads per SM): the exchange serves at
# most one wave of single-CTA rows (the persistent grid would walk more in turns).
TOPK_CTAS_PER_SM = 4
# Keep a step's top-k rows within one wave of co-resident clusters (smaller clusters
# above the co-resident count; ``DcpxDecode.launch_params``).
FIT_WAVE = True


def wanted(vllm_config) -> bool:
    """Whether the model builds a DcpxDecode (the module docstring's policy). A
    prefill-only instance never decodes."""
    parallel = vllm_config.parallel_config
    dcp = parallel.decode_context_parallel_size
    if dcp <= 1:
        return False
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    if (
        kv_transfer is not None
        and kv_transfer.is_kv_producer
        and not kv_transfer.is_kv_consumer
    ):
        return False
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    if cap is None or (cap.major, cap.minor) != (10, 7):
        logger.info_once(
            "Kimi K3.1 DSA decode: DcpExchange off (measured on SM107 only); stock "
            "DCP path"
        )
        return False
    if dcp != parallel.tensor_parallel_size:
        logger.info_once(
            "Kimi K3.1 DSA decode: DcpExchange off (needs DCP == TP, got DCP%d / "
            "TP%d); stock DCP path",
            dcp,
            parallel.tensor_parallel_size,
        )
        return False
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    if not current_platform.is_cuda() or not mqa_logits_use_mx_scales():
        logger.info_once(
            "Kimi K3.1 DSA decode: DcpExchange off (its top-k reads the bf16 logits of "
            "DeepGEMM's MX-scaled MQA logits); stock DCP path"
        )
        return False
    return True


class DcpxDecode:
    def __init__(self, vllm_config, device: torch.device, heads: int, topk: int):
        from vllm.distributed import get_dcp_group, get_tp_group

        from .ops import DcpExchange, _load_ext

        group = get_dcp_group()
        self.W, self.rank = group.world_size, group.rank_in_group
        if group.world_size != get_tp_group().world_size or (
            group.rank_in_group != get_tp_group().rank_in_group
        ):
            raise ValueError("DcpExchange decode needs DCP == TP (head owner = rank)")
        # Rows whose global context is below this would start at the octave round
        # (R8) instead of the 4-octave coarse window: every row does. Indexer logits
        # of long contexts span more than 4 octaves above the K-th key (TP4/DCP4,
        # 8 real layers, AgentX contexts: every row fell back to R8 after the coarse
        # window), so the coarse round is wasted: 55 -> 40 us per call at M8, 99 ->
        # 65 us at M128.
        self.small_ctx = 1 << 30
        self.fit_wave = FIT_WAVE
        # topk_merge reads bf16 logits: DeepGEMM's MX-scaled MQA logits kernels (SM100
        # family) return bf16 (``wanted`` checked them). Decided here, not per step:
        # once ``topk_active`` admits a step the MLA layer reads slots, so the
        # indexer cannot fall back.
        sched = vllm_config.scheduler_config
        spec = vllm_config.speculative_config
        q = 1 + (spec.num_speculative_tokens if spec is not None else 0)
        compile_cfg = vllm_config.compilation_config
        rows = getattr(compile_cfg, "max_cudagraph_capture_size", None) or 0
        rows = max(rows, min(sched.max_num_seqs * q, sched.max_num_batched_tokens))
        self.cap = min(max(rows, 64), 2048)
        parallel = vllm_config.parallel_config
        self.interleave = parallel.cp_kv_cache_interleave_size
        model_len = vllm_config.model_config.max_model_len
        span = self.W * self.interleave
        self.max_local_len = -(-model_len // span) * self.interleave
        self.K = topk
        cap1 = min(self.cap, ONE_ROUND_ROWS)
        # Every rank's extension before any rank enters the first rendezvous: torch's
        # JIT load builds one rank at a time on the build directory's lock, and a
        # rendezvous wait past the store timeout (a FileStore's: 300 s a key) leaves
        # the group's symmetric-memory sequence offset for every later rendezvous.
        _load_ext()
        group.barrier()
        self.ex = DcpExchange(
            group.device_group,
            device,
            cap=self.cap,
            cap1=cap1,
            # 256 keys: half the message and the resolve work of 512 (DCP16: B1 top-k
            # 20.2 -> 18.4 us); tau outside the windows costs one refine round
            one_round_window=256 if cap1 else 0,
            heads=heads,
            o_dims=(),
            topk=topk,
            interleave=self.interleave,  # global ids (tie_order="global")
        )
        props = torch.cuda.get_device_properties(device)
        self.sms = props.multi_processor_count
        self.smem = int(
            getattr(props, "shared_memory_per_multiprocessor", 0) or 228 * 1024
        )
        threads = int(getattr(props, "max_threads_per_multi_processor", 1024) or 1024)
        self.topk_cap = min(self.cap, self.sms * min(TOPK_CTAS_PER_SM, threads // 256))
        self.counts = torch.zeros(self.cap, dtype=torch.int32, device=device)
        self.global_rows = torch.zeros(self.cap, dtype=torch.int32, device=device)
        self.offsets = torch.arange(64, dtype=torch.int32, device=device)
        self.heads, self.hpr = heads, heads // self.W
        self.layers: dict[str, Any] = {}  # indexer prefix -> MLA layer
        self.indexers: dict[str, Any] = {}  # indexer prefix -> indexer
        # VLLM_DSA_BPRIME_CHECK (interleave 1): every exchanged top-k also against
        # the generic merge of the same logits, as the one-round DCP merge's.
        from vllm.v1.attention.ops.dcp_topk_merge.check import (
            dcp_topk_merge_check_enabled,
        )

        self.check_sat = None
        if self.interleave == 1 and dcp_topk_merge_check_enabled():
            self.check_sat = torch.zeros(self.cap, dtype=torch.int32, device=device)
            logger.warning_once(
                "DSA DCP top-k merge check: the DcpExchange top-k is checked too "
                "(its slots as global ids through the MLA block table)"
            )
        logger.info(
            "Kimi K3.1 DSA decode: DcpExchange top-k on (DCP%d, interleave %d, rows "
            "<= %d, one-round rows <= %d, ties by global id)",
            self.W,
            self.interleave,
            self.topk_cap,
            cap1,
        )

    # ------------------------------------------------------------------ policy
    def topk_active(self, prefix: str, rows: int) -> bool:
        """Pure-decode steps of uniform rows whose indexer and MLA metadata the
        exchange covers. Both the indexer op and the MLA layer decide with this, from
        the same step metadata, so they always agree (and so do all DCP ranks)."""
        if not 0 < rows <= self.topk_cap:
            return False
        from vllm.forward_context import get_forward_context

        md_all = get_forward_context().attn_metadata
        layer = self.layers.get(prefix)
        if not isinstance(md_all, dict) or layer is None:
            return False
        index_md, mla_md = md_all.get(prefix), md_all.get(layer.layer_name)
        if index_md is None or mla_md is None:
            return False
        decode = getattr(index_md, "decode", None)
        if (
            decode is None
            or index_md.num_prefills != 0
            or index_md.num_decode_tokens != rows
            or decode.global_seq_lens is None
            or decode.requires_padding
            or rows % decode.decode_lens.shape[0] != 0
            or getattr(mla_md, "shardq", None) is not None
            or decode.seq_lens.numel() != rows
        ):
            return False
        g = decode.global_seq_lens.numel()
        return g == rows or (0 < g and rows % g == 0 and rows // g <= 64)

    # ------------------------------------------------------------------ top-k
    def _global_rows(self, decode, rows: int) -> torch.Tensor | None:
        g = decode.global_seq_lens
        if g.numel() == rows:
            return g.reshape(-1)
        reqs = g.numel()
        n = rows // reqs
        out = self.global_rows[:rows].view(reqs, n)
        torch.add(g.view(-1, 1), self.offsets[:n] - (n - 1), out=out)
        out.clamp_(min=0)
        return out.view(-1)

    def decode_topk(self, prefix: str, logits: torch.Tensor, decode, out: torch.Tensor):
        """Indexer hook: this rank's selected KV slots into ``out`` [T, >= K] and
        their count into ``counts``. False leaves the step to the stock path."""
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

        rows = logits.shape[0]
        # ``topk_active`` admitted the step, so the MLA layer reads slots: no
        # fallback from here on.
        if logits.dtype != torch.bfloat16 or logits.stride(1) != 1:
            raise RuntimeError(f"DcpExchange top-k: bf16 logits, got {logits.dtype}")
        local = decode.seq_lens.reshape(-1)
        glob = self._global_rows(decode, rows)
        assert local.numel() == rows and glob is not None
        layer = self.layers[prefix]
        md = get_forward_context().attn_metadata[layer.layer_name]
        _, stride = flat_kv_row_view(layer.kv_cache, md.block_size)
        max_local = min(self.max_local_len, logits.shape[1])
        cluster, segmax = self.launch_params(rows, max_local)
        self.ex.topk_merge(
            logits,
            local,
            glob,
            out,
            self.counts[:rows],
            max_local_len=max_local,
            cluster=cluster,
            segmax=segmax,
            block_table=md.block_table,
            block_size=md.block_size,
            block_stride=stride,
            req_of_row=md.req_id_per_token[:rows],
            one_round=self.ex.cap1 > 0,
            window="center",
            tie_order=TIE_ORDER,
            small_ctx=self.small_ctx,
        )
        if self.check_sat is not None:
            self.check_topk(prefix, logits, decode, out, rows, md, stride)
        return True

    def check_topk(self, prefix, logits, decode, out, rows, md, stride) -> None:
        """VLLM_DSA_BPRIME_CHECK: this step's exchanged slots as global ids against
        the generic merge of the same logits (dcp_topk_merge.check: collective, no
        host syncs, so captured graphs keep it)."""
        from vllm.distributed import get_dcp_group
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import (
            _check_dcp_decode_topk_merge,
        )
        from vllm.v1.attention.ops.dcp_topk_merge.check import (
            exchange_slot_ids,
            merge_exchange_ids,
        )

        indexer = self.indexers[prefix]
        local = decode.seq_lens.reshape(-1)[:rows]
        mine = exchange_slot_ids(
            out[:rows, : self.K],
            self.counts[:rows],
            md.block_table,
            md.req_id_per_token[:rows],
            md.block_size,
            stride,
            local,
            self.rank,
            self.W,
        )
        every = get_dcp_group().all_gather(mine, dim=0).view(self.W, *mine.shape)
        reqs = decode.decode_lens.shape[0]
        _check_dcp_decode_topk_merge(
            prefix,
            logits,
            local,
            decode.seq_lens[:reqs],
            rows // reqs,
            merge_exchange_ids(every),
            0,
            indexer.topk_backend,
            indexer.topk_backend != "deep_select",
            get_forward_context().attn_metadata[prefix].max_seq_len,
            indexer.max_model_len,
            self.rank,
            self.W,
            16,  # bf16 logits
            sat_word=self.check_sat[:rows],
        )

    def launch_params(self, rows: int, max_local: int) -> tuple[int, int]:
        """(CTAs per row, shared-memory keys per CTA or 0) of a top-k launch.

        Small batches split each row over a cluster of CTAs (``rows x cluster`` up to
        ~400 CTA pairs). A step's rows must also fit in one wave of co-resident
        clusters: rows beyond it run in a second persistent turn, every NVLink round
        included, which made the TP4/DCP4 top-k jump from 66 to 100 us per call
        between 160 and 192 rows (cluster 4: fewer co-resident clusters than rows).
        So the cluster halves until the rows fit, down to the smallest cluster whose
        CTA segment fits the WRITE bitmap (``topk_merge``)."""
        cluster = 1  # as DcpExchange.topk_merge: CTAs per row by batch
        while cluster < 8 and rows * cluster * 2 <= 800:
            cluster *= 2
        floor = 1  # topk_merge raises the cluster to this for long contexts
        while floor < 16 and -(-max_local // floor) > 131072:
            floor *= 2
        cluster = max(cluster, floor)
        segmax = self._segmax(rows, cluster, max_local)
        if self.fit_wave:
            while cluster > floor and rows > self.ex.resident_clusters(cluster, segmax):
                cluster //= 2
                segmax = self._segmax(rows, cluster, max_local)
        return cluster, segmax

    def _segmax(self, rows: int, cluster: int, max_local: int) -> int:
        # Stage a row segment's keys in shared memory when the longest possible one
        # (the bound is the model length, not this step's contexts) fits with every
        # row's CTAs resident at once (else the persistent grid walks rows in turns).
        seg = (-(-max_local // cluster) + 7) // 8 * 8
        per_sm = -(-rows * cluster // self.sms)
        fits = (self.smem // per_sm - STATIC_SMEM) // 2
        return max(seg, 8192) if seg <= min(SMEM_KEYS, fits) else 0

    def counts_for(self, prefix: str, rows: int) -> torch.Tensor | None:
        """The slot counts of this step's exchanged top-k (the indexer's decode rows,
        of which the layer reads the first ``rows``), or None (stock path)."""
        from vllm.forward_context import get_forward_context

        md_all = get_forward_context().attn_metadata
        index_md = md_all.get(prefix) if isinstance(md_all, dict) else None
        if index_md is None:
            return None
        decoded = index_md.num_decode_tokens
        if not self.topk_active(prefix, decoded):
            return None
        # The indexer wrote slots for its rows: the stock conversion must not run.
        if rows > decoded:
            raise RuntimeError(f"DcpExchange top-k: {rows} rows > {decoded} decoded")
        return self.counts[:rows]

    # ------------------------------------------------------------------ wiring
    def attach(self, attention) -> None:
        """Wire one KimiK3DSAAttention's top-k hooks."""
        from vllm.model_executor.layers import sparse_attn_indexer as op

        indexer = attention.dsa_indexer
        layer = attention.mla_attn.mla_attn
        prefix = indexer.k_cache.prefix
        # Two-level candidate selection rewrites the indexer's output as global ids
        # (and the indexer op drops the hook for it): such a layer keeps the stock
        # top-k on both sides, decided once here.
        topk = getattr(indexer, "candidate_blocks", None) is None
        if topk and not _reads_counts(layer.impl):
            # FlashMLA sparse (fp8_ds_mla) filters the DCP ids in its own
            # forward_mqa: slots would be read as global ids there.
            logger.warning_once(
                "DcpExchange top-k: %s converts the DCP top-k itself; its layers "
                "keep the stock top-k",
                type(layer.impl).__name__,
            )
            topk = False
        if topk:
            self.layers[prefix] = layer
            self.indexers[prefix] = indexer
            op.DCP_DECODE_TOPK_HOOKS[prefix] = (
                lambda rows, p=prefix: self.topk_active(p, rows),
                lambda logits, decode, out, p=prefix: self.decode_topk(
                    p, logits, decode, out
                ),
            )
            layer.impl.dcp_local_topk_counts = lambda rows, p=prefix: self.counts_for(
                p, rows
            )

    def warm(self) -> None:
        """Configure the top-k kernel and its occupancy per launch shape outside any
        capture: one select-all row per (cluster, shared keys) the decode can use
        (every rank alike, so the row-0 token exchanges match)."""
        dev = self.counts.device
        logits = torch.zeros((1, 64), dtype=torch.bfloat16, device=dev)
        lens = torch.ones(1, dtype=torch.int32, device=dev)
        out = torch.empty((1, self.K), dtype=torch.int32, device=dev)
        rows = range(1, self.topk_cap + 1)
        shapes = {self.launch_params(r, self.max_local_len) for r in rows}
        for cluster, segmax in sorted(shapes):
            self.ex.topk_merge(
                logits, lens, lens, out, self.counts[:1], max_local_len=64,
                cluster=cluster, segmax=segmax,
            )  # fmt: skip
        torch.cuda.synchronize(dev)
        self.ex.check()

    def serve(self) -> None:
        """From here on (after the warmup) kernels wait SERVING_SPIN_S for a peer."""
        self.ex.set_spin_timeout(SERVING_SPIN_S)
        logger.info("DcpExchange peer wait limit: %g s", SERVING_SPIN_S)


def _reads_counts(impl) -> bool:
    """Whether the sparse MLA impl's decode reads ``dcp_local_topk_counts`` (the
    shared ``SparseMLACommonImpl.forward_mqa``)."""
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonImpl,
    )

    return type(impl).forward_mqa is SparseMLACommonImpl.forward_mqa


def build(model, vllm_config) -> DcpxDecode | None:
    """Build and attach the decode exchange to every DSA layer of ``model``."""
    if not wanted(vllm_config):
        return None
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention

    layers = [m for m in model.modules() if isinstance(m, KimiK3DSAAttention)]
    if not layers:
        return None
    first = layers[0]
    tp = vllm_config.parallel_config.tensor_parallel_size
    heads = first.num_local_heads * tp
    device = first.mla_attn.mla_attn.W_UK_T.device
    dcpx = DcpxDecode(vllm_config, device, heads, first.dsa_indexer.topk_tokens)
    for layer in layers:
        dcpx.attach(layer)
    dcpx.warm()
    dcpx.serve()
    return dcpx
