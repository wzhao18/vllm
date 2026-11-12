# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA attention backend."""

import bisect
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
    max_decode_query_len,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
    prepare_sparse_mla_safe_lengths,
)
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# CUDA's gridDim.z limit; trtllm-gen MLA decode launches one z-slice per request.
_TRTLLM_GEN_MAX_REQUESTS = 65535


class _FlashInferMLASparseBackendBase(AttentionBackend):
    """Common metadata for concrete FlashInfer sparse MLA backends."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 NoPE + 64 RoPE (with-rope layout); 512 = 512 NoPE only
        # (no-rope layout, qk_rope_head_dim == 0). Both share D_V = 512 and are
        # served by the TRTLLM-GEN sparse MLA kernel.
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # The kernels read the rows as pages of their kernel block size.
        (page,) = cls.get_supported_kernel_block_sizes(spec)
        assert isinstance(page, MultipleOf)
        return align_blocks_to_rows(spec, page.base)


class FlashInferMLASparseTRTLLMBackend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend using the TRTLLM-gen launcher."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4_ds_mla",  # prims_ts kernel only
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseTRTLLMMetadataBuilder"]:
        return FlashInferMLASparseTRTLLMMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            and parallel_config.decode_context_parallel_size > 1
        ):
            return (
                f"{cls.get_name()} does not support combined PCP+DCP; "
                "use FLASHMLA_SPARSE, which gathers each DCP KV shard before "
                "running the rank-local PCP prefill queries"
            )
        if kv_cache_dtype == "fp8_ds_mla":
            return f"{cls.get_name()} SM10 does not support fp8_ds_mla kv-cache dtype"
        if kv_cache_dtype == "nvfp4_ds_mla":
            from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

            if vllm_config.kernel_config.flashinfer_sparse_mla_kernel == "trtllm":
                return (
                    f"{cls.get_name()} reads nvfp4_ds_mla only with the prims_ts "
                    "kernel (kernel_config.flashinfer_sparse_mla_kernel=trtllm)"
                )
            if not has_flashinfer_prims_ts_sparse_mla():
                return (
                    f"{cls.get_name()} reads nvfp4_ds_mla with FlashInfer's "
                    "prims_ts sparse MLA decode, which the installed FlashInfer "
                    "lacks (needs a build with flashinfer-ai/flashinfer#7410)"
                )

        # FlashInfer MLA sparse SM10 kernel requires qk_nope_head_dim in [128, 192].
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = hf_text_config.qk_nope_head_dim
            qk_rope_head_dim = hf_text_config.qk_rope_head_dim
            kv_lora_rank = hf_text_config.kv_lora_rank
            if qk_rope_head_dim == 0:
                # Native no-rope MLA: FlashInfer only ships the one shape
                # (nope_mla_dimensions in flashinfer.mla._core).
                if qk_nope_head_dim != 256 or kv_lora_rank != 512:
                    return (
                        "FlashInfer native no-rope MLA requires "
                        "qk_nope_head_dim=256 and kv_lora_rank=512, but got "
                        f"qk_nope_head_dim={qk_nope_head_dim}, "
                        f"kv_lora_rank={kv_lora_rank}"
                    )
            elif qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None


class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend for SM120."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
        )

        return FlashInferMLASparseSM120Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            index_topk = getattr(hf_text_config, "index_topk", None)
            if index_topk is None:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires a model with "
                    "index_topk config"
                )
            if int(index_topk) != 2048:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; "
                    f"got {index_topk}"
                )
        return None


@dataclass
class FlashInferMLASparseMetadata(SparseMLACommonMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""


class FlashInferMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    metadata_cls = FlashInferMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )


class FlashInferMLASparseTRTLLMMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """Metadata builder for the SM100 TRT-LLM sparse MLA kernel."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Decode uses device request boundaries; prefill metadata is not graph-safe.
        return max_decode_query_len(vllm_config)

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = (
            self.dcp_world_size == 1
            and self.pcp_world_size == 1
            and vllm_config.attention_config.hisparse_config is None
        )

        # trtllm-gen's buffers are baked into CUDA graphs, so they reach their
        # final size before the first forward/capture (no lazy regrow): under
        # DCP the workspace holds the softmax-stats slab, and the multi-CTA KV
        # counters cover every launch. The layers' memory-profile pass
        # (FlashInferMLASparseImpl.reserve_workspace) already allocated them, so
        # the KV cache is sized without them; this is a no-op then. The prims_ts
        # kernel owns its own (small) plan workspaces.
        kv_cache_dtype = (
            getattr(kv_cache_spec, "cache_dtype_str", None)
            or vllm_config.cache_config.cache_dtype
        )
        if not use_prims_ts(vllm_config, kv_cache_dtype):
            reserve_trtllm_gen_buffers(
                vllm_config,
                device,
                self.dcp_world_size,
                self.dcp_shardq is not None,
            )

    def update_draft_decode_metadata(
        self, metadata: FlashInferMLASparseMetadata
    ) -> None:
        num_tokens = metadata.num_decode_tokens
        if num_tokens:
            compute_token_to_req_indices(
                metadata.query_start_loc,
                metadata.req_id_per_token,
                num_tokens,
                num_tokens,
            )


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None
# Shared trtllm-gen multi-CTA KV counters (see _reserve_kv_counters).
_fi_sparse_kv_counters: torch.Tensor | None = None

# trtllm-gen carves a softmax-stats slab from the workspace whenever LSE is
# requested (FlashInfer csrc/trtllm_fmha_kernel_launcher.cu, unchanged from
# v0.6.14 through current main):
#   sizeof(float2) * num_qo_heads * batch_size * round_up(max_q_len, 256)
#   + 1 MiB guard
# forward_mqa always passes q_len == 1, so each (head, token) pair costs
# round_up(1, 256) == 256 slots.
_TRTLLM_GEN_SOFTMAX_STAT_BYTES = 8  # sizeof(float2)
_TRTLLM_GEN_SOFTMAX_SLOTS_PER_TOKEN = 256  # round_up(q_len=1, 256)
_TRTLLM_GEN_SOFTMAX_GUARD_BYTES = 1024 * 1024

# Keep in sync with the VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default in
# vllm/envs.py.
_DEFAULT_WORKSPACE_BUFFER_SIZE = 394 * 1024 * 1024


def trtllm_gen_query_heads(num_heads: int) -> int:
    """Query heads the trtllm-gen MLA decode runs: it takes no head count strictly
    between 64 and 128 (e.g. 96 gathered under DCP8 at 12 heads per rank), so
    those are zero-padded to 128."""
    return 128 if 64 < num_heads < 128 else num_heads


def compute_trtllm_sparse_mla_workspace_bytes(
    base_workspace_bytes: int,
    dcp_world_size: int,
    num_heads_per_rank: int,
    max_num_batched_tokens: int,
) -> int:
    """Workspace bytes needed by the trtllm-gen sparse MLA decode kernel.

    Under DCP the query is all-gathered across the DCP group in the head dim
    and the kernel is asked for LSE, so trtllm-gen carves the softmax-stats
    slab described above from the workspace before its (batch-independent)
    counter and scratch regions. ``base_workspace_bytes`` must stay available
    for those regions, so the slab is added on top of it (see #50781).

    ``max_num_batched_tokens``: the most rows a step asks LSE for
    (``trtllm_gen_lse_max_rows``). Without DCP no LSE is requested, no slab is
    carved, and the base size is returned unchanged. A launch covers at most
    ``_TRTLLM_GEN_MAX_REQUESTS`` tokens, and launches run in order on one
    stream, so larger batches reuse the same slab.
    """
    if dcp_world_size <= 1:
        return base_workspace_bytes
    softmax_bytes = (
        _TRTLLM_GEN_SOFTMAX_STAT_BYTES
        * trtllm_gen_query_heads(num_heads_per_rank * dcp_world_size)
        * min(max_num_batched_tokens, _TRTLLM_GEN_MAX_REQUESTS)
        * _TRTLLM_GEN_SOFTMAX_SLOTS_PER_TOKEN
        + _TRTLLM_GEN_SOFTMAX_GUARD_BYTES
    )
    return base_workspace_bytes + softmax_bytes


def _required_workspace_bytes(
    dcp_world_size: int,
    num_heads_per_rank: int,
    max_num_batched_tokens: int,
) -> int:
    """Resolve the workspace size, honoring an explicit env override."""
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        _DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size,
        num_heads_per_rank,
        max_num_batched_tokens,
    )
    if not envs.is_set("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"):
        return computed
    env_bytes = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
    if env_bytes < computed:
        logger.warning_once(
            "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=%d is below the %d bytes "
            "computed for the sparse MLA workspace with dcp_world_size=%d, "
            "%d heads/rank and max_num_batched_tokens=%d. Respecting the "
            "override, but the trtllm-gen kernel may crash with a workspace "
            "overflow; set VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=%d or unset "
            "it to use the computed size.",
            env_bytes,
            computed,
            dcp_world_size,
            num_heads_per_rank,
            max_num_batched_tokens,
            computed,
        )
    return env_bytes


def _get_workspace_buffer(
    device: torch.device, min_bytes: int | None = None
) -> torch.Tensor:
    global _fi_sparse_workspace
    required = (
        min_bytes
        if min_bytes is not None
        else envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
    )
    if _fi_sparse_workspace is None or _fi_sparse_workspace.numel() < required:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_sparse_workspace = torch.zeros(
            required,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_workspace


# prims_ts: FlashInfer's task-scheduled sparse MLA decode
# (flashinfer.attention.prims_ts.BatchSparseMLADecodePagedTSWrapper), an
# experimental API. Each query row attends its own int32 storage-row indices,
# so it serves decode, MTP verify and sparse prefill rows alike.
_PRIMS_TS_FORMATS = {
    # D576 per-tensor E4M3 rows (NoPE + RoPE); the tensor descale is folded into
    # the softmax and output scales.
    "fp8": "native",
    "fp8_e4m3": "native",
    # Byte-identical to vLLM's nvfp4_ds_mla (352 B: E2M1 NoPE, unscaled E4M3
    # RoPE, E4M3 block scales).
    "nvfp4_ds_mla": "nvfp4.fp8rope",
}
# FlashInfer builds without flashinfer-ai/flashinfer#7984 choose the 2-CTA
# schedule (KV rows loaded and converted once per CTA pair) only for exactly 128
# query heads; other counts in (64, 128) run as two M64 tiles that load every KV
# row twice, 1.7x (fp8) to 2.7x (nvfp4) slower at 96 heads (DCP8 decode). There
# such queries are zero-padded to 128 heads. Builds with it schedule 96 heads on
# the same 2-CTA tile directly, which saves the padded query/output copies.
_PRIMS_TS_2CTA_HEADS = 128
# Rows per prims_ts call. Bounds the head-padded query/output of long prefill
# steps.
_PRIMS_TS_MAX_ROWS = 4096


def use_prims_ts(vllm_config: VllmConfig, kv_cache_dtype: str) -> bool:
    """Whether FLASHINFER_MLA_SPARSE runs this model's sparse MLA on prims_ts.

    "auto" uses prims_ts under DCP when it can run (faster than trtllm-gen
    for per-tensor fp8 prefill and decode on SM100 there) and trtllm-gen
    otherwise, including TP-only (DCP 1). "trtllm" keeps trtllm-gen for
    per-tensor fp8. nvfp4_ds_mla (no trtllm-gen kernel) always uses prims_ts.
    Raises if prims_ts is required but cannot run."""
    kernel = vllm_config.kernel_config.flashinfer_sparse_mla_kernel
    required = kv_cache_dtype == "nvfp4_ds_mla" or kernel == "prims_ts"
    if not required and (
        kernel == "trtllm"
        or vllm_config.parallel_config.decode_context_parallel_size <= 1
    ):
        return False
    reason = _prims_ts_unsupported_reason(vllm_config, kv_cache_dtype)
    if not required:
        return reason is None
    if reason is not None:
        raise ValueError(
            f"FLASHINFER_MLA_SPARSE cannot run the prims_ts kernel: {reason}"
        )
    return True


def _prims_ts_row_buckets(vllm_config: VllmConfig) -> tuple[int, ...]:
    """Row capacities to plan: every CUDA graph size (graph replays keep the
    schedule FlashInfer picks for their exact row count) and the powers of two
    up to the largest call, which serve every other row count."""
    max_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens, _PRIMS_TS_MAX_ROWS
    )
    sizes = {1 << i for i in range(max_rows.bit_length())} | {max_rows}
    graph_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
    sizes.update(s for s in graph_sizes if s <= max_rows)
    return tuple(sorted(sizes))


def _prims_ts_unsupported_reason(
    vllm_config: VllmConfig, kv_cache_dtype: str
) -> str | None:
    from vllm.utils.flashinfer import (
        has_flashinfer_prims_ts_sparse_mla,
        prims_ts_sparse_mla_format_error,
    )

    if kv_cache_dtype not in _PRIMS_TS_FORMATS:
        return f"kv-cache dtype {kv_cache_dtype!r} (needs fp8 or nvfp4_ds_mla)"
    text_config = vllm_config.model_config.hf_text_config
    dims = (text_config.kv_lora_rank, text_config.qk_rope_head_dim)
    if dims != (512, 64):
        return f"(kv_lora_rank, qk_rope_head_dim) {dims} (needs (512, 64))"
    if vllm_config.attention_config.hisparse_config is not None:
        return "HiSparse"
    if vllm_config.parallel_config.use_ubatching:
        # The shared plans and workspace serve one stream at a time.
        return "micro-batching (DBO / ubatching)"
    if not current_platform.is_device_capability_family(100):
        return "this GPU (needs SM100 / SM103)"
    if not has_flashinfer_prims_ts_sparse_mla():
        return (
            "the installed FlashInfer has no prims_ts sparse MLA decode "
            "(BatchSparseMLADecodePagedTSWrapper)"
        )
    return prims_ts_sparse_mla_format_error(_PRIMS_TS_FORMATS[kv_cache_dtype])


class _PrimsTSPlans:
    """Planned prims_ts wrappers, shared by all layers.

    A plan fixes FlashInfer's schedule (chosen from its row capacity) and
    compiles that schedule's kernel on first use, which takes seconds. So every
    row capacity of a geometry is planned at once, at startup (from the profile
    run's ``reserve_workspace``), and a call runs on the smallest capacity that
    holds its rows: no planning or compilation happens while serving, and the
    plan count stays fixed. Each plan is a packed-query plan of ``capacity``
    one-token requests, so its schedule is the one FlashInfer picks for
    ``capacity`` decode rows; a call with fewer rows passes the matching
    offsets (``qo_indptr`` = [0, 1, ..., rows, ..., rows]).

    The sparse MLA layers run one at a time on one stream, so every plan shares
    one workspace. A geometry that needs more workspace replaces it; plans keep
    the buffer they were planned with, so graphs captured with them stay valid.
    """

    def __init__(self) -> None:
        self._plans: dict[tuple, dict[int, tuple[object, torch.Tensor]]] = {}
        self._workspace: torch.Tensor | None = None

    def get(
        self,
        device: torch.device,
        buckets: tuple[int, ...],
        heads: int,
        kv_cache_format: str,
        max_topk: int,
        return_lse: bool,
    ) -> dict[int, tuple[object, torch.Tensor]]:
        """{capacity: (wrapper, qo_indptr of a full call)} for one geometry."""
        key = (device, buckets, heads, kv_cache_format, max_topk, return_lse)
        plans = self._plans.get(key)
        if plans is not None:
            return plans
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prims_ts sparse MLA was not planned before CUDA graph capture"
            )
        from flashinfer.attention.prims_ts import (
            BatchSparseMLADecodePagedTSWrapper,
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        kwargs = dict(
            max_topk=max_topk,
            max_q_len=1,
            packed_query=True,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            kv_cache_format=kv_cache_format,
            return_lse=return_lse,
        )
        size = max(
            get_prims_ts_sparse_mla_decode_workspace_size(device, rows, heads, **kwargs)
            for rows in buckets
        )
        if self._workspace is None or self._workspace.numel() < size:
            self._workspace = torch.empty(size, dtype=torch.uint8, device=device)
        start = time.perf_counter()
        plans = {}
        for rows in buckets:
            plan = BatchSparseMLADecodePagedTSWrapper(self._workspace)
            plan.plan(device, rows, heads, **kwargs)
            qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device=device)
            plans[rows] = (plan, qo_indptr)
        logger.info(
            "Planned FlashInfer prims_ts sparse MLA (%s, %d heads, top-k %d) for "
            "%d row capacities up to %d in %.1f s",
            kv_cache_format,
            heads,
            max_topk,
            len(buckets),
            buckets[-1],
            time.perf_counter() - start,
        )
        self._plans[key] = plans
        return plans


_prims_ts_plans = _PrimsTSPlans()


def _kv_counter_bytes(num_rows: int, num_heads: int, device: torch.device) -> int:
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    return get_trtllm_gen_multi_ctas_kv_counter_bytes(
        num_rows, num_heads, get_device_sm_count(device)
    )


def _reserve_kv_counters(device: torch.device, num_rows: int, num_heads: int) -> None:
    """Allocates the zeroed trtllm-gen multi-CTA KV counters for launches of up
    to ``num_rows`` x ``num_heads``, before CUDA graph capture. The kernel
    resets them after each launch, so all layers share them, as they share the
    workspace. Without them FlashInfer zero-allocates counters on every call."""
    global _fi_sparse_kv_counters
    nbytes = _kv_counter_bytes(num_rows, num_heads, device)
    if _fi_sparse_kv_counters is None or _fi_sparse_kv_counters.nbytes < nbytes:
        _fi_sparse_kv_counters = torch.zeros(nbytes, dtype=torch.uint8, device=device)


def trtllm_gen_lse_max_rows(vllm_config: VllmConfig, dcp_shardq: bool) -> int:
    """Most query rows of one step that ask trtllm-gen for LSE; it sizes the
    DCP softmax-stats slab (``compute_trtllm_sparse_mla_workspace_bytes``).

    Only the DCP path asks for LSE (``need_to_return_lse_for_decode``, for the
    DCP combine), for every MQA row of its step: up to max_num_batched_tokens.

    ``dcp_shardq``: the metadata builders plan DCP ShardQ steps
    (``SparseMLACommonMetadataBuilder.dcp_shardq``). Their prefill rows run in
    ``forward_mqa_shardq``, whose token-slice launches pass ``return_lse=False``
    (the slice's output goes straight back to the head owners). The DCP path,
    and with it LSE, then keeps only (``DcpShardQPlanner.plan``):

    * the leading decode and spec-verify rows of a ShardQ step
      (``MLAAttention._dcp_decode_rows``): at most max_num_seqs requests of at
      most the indexer's decode threshold;
    * the steps the planner leaves on the DCP path: steps without prefill rows,
      few-token steps (decode rows plus fewer than
      ``sparse_mla_shard_q_min_tokens`` prefill tokens), dense-MHA steps (MQA
      on their decode rows only) and padded steps (FULL CUDA graphs, at most
      the DCP workspace's tokens; DCP ShardQ excludes FULL mixed graphs). No
      step keeps it for a full gathered-context workspace: that holds at least
      max_model_len tokens, so one request always fits, and larger steps run
      in request groups.

    ``dcp_fallback_max_tokens`` bounds all of them (2048 rows for Kimi K3.1
    C512 prefill against 32768 tokens per step: 0.9 instead of 8.4 GiB per
    rank at 128 padded heads). ``FlashInferMLASparseImpl._forward_mqa_kernel``
    raises if a launch ever asks for more LSE rows than the slab holds.
    """
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    if not dcp_shardq:
        return max_tokens
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.ops.dcp_shardq.context import dcp_fallback_max_tokens

    return min(
        max_tokens,
        dcp_fallback_max_tokens(
            vllm_config,
            DeepseekV32IndexerMetadataBuilder.get_decode_threshold(vllm_config),
        ),
    )


def trtllm_gen_lse_rows_capacity(workspace_bytes: int, padded_heads: int) -> int:
    """Rows of ``padded_heads`` query heads whose softmax stats a workspace of
    ``workspace_bytes`` holds (the inverse of
    ``compute_trtllm_sparse_mla_workspace_bytes``)."""
    slab = workspace_bytes - _DEFAULT_WORKSPACE_BUFFER_SIZE
    slab -= _TRTLLM_GEN_SOFTMAX_GUARD_BYTES
    per_row = (
        _TRTLLM_GEN_SOFTMAX_STAT_BYTES
        * padded_heads
        * _TRTLLM_GEN_SOFTMAX_SLOTS_PER_TOKEN
    )
    return max(slab, 0) // per_row


def reserve_trtllm_gen_buffers(
    vllm_config: VllmConfig,
    device: torch.device,
    dcp_world_size: int,
    dcp_shardq: bool,
) -> None:
    """Allocate the shared trtllm-gen workspace and multi-CTA KV counters at
    their final size (both sized from the configuration). Called by the layers'
    memory-profile pass, before the KV cache is sized, and again (a no-op) by
    the metadata builders, which only exist once the KV cache does.

    ``dcp_shardq``: DCP ShardQ steps take the prefill rows off the DCP path,
    so the softmax-stats slab covers ``trtllm_gen_lse_max_rows`` rows only."""
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    if dcp_world_size > 1:
        required = _required_workspace_bytes(
            dcp_world_size,
            num_q_heads,
            trtllm_gen_lse_max_rows(vllm_config, dcp_shardq),
        )
        if _fi_sparse_workspace is not None and _fi_sparse_workspace.numel() < required:
            # Only when the builders keep the DCP path for prefill steps that
            # the layers' profile pass expected to run as ShardQ.
            logger.warning_once(
                "The trtllm-gen sparse MLA workspace grows from %d to %d bytes "
                "after the memory profile (DCP ShardQ is off for this layer "
                "group); the KV cache was sized without the difference.",
                _fi_sparse_workspace.numel(),
                required,
            )
        _get_workspace_buffer(device, required)
    _reserve_kv_counters(
        device, max_tokens, trtllm_gen_query_heads(num_q_heads * dcp_world_size)
    )


def _kv_counters(
    num_rows: int, num_heads: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """``multi_ctas_kv_counter_buffer`` for one launch when the reserved
    counters cover it; otherwise FlashInfer allocates its own."""
    counters = _fi_sparse_kv_counters
    if counters is None or counters.nbytes < _kv_counter_bytes(
        num_rows, num_heads, device
    ):
        return {}
    return {"multi_ctas_kv_counter_buffer": counters}


@triton.jit
def _pad_query_heads_kernel(
    src_ptr,
    dst_ptr,
    src_row_stride,
    row_words,
    padded_row_words,
    BLOCK: tl.constexpr,
):
    """One query row of 8-byte words, zero-extended to the padded heads."""
    row = tl.program_id(0).to(tl.int64)
    words = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        src_ptr + row * src_row_stride + words, mask=words < row_words, other=0
    )
    tl.store(
        dst_ptr + row * padded_row_words + words, value, mask=words < padded_row_words
    )


def _pad_query_heads(query: torch.Tensor, heads: int) -> torch.Tensor:
    """``query`` [T, H, D] in a new contiguous [T, ``heads``, D] buffer whose
    extra heads are zero (the head count a decode kernel runs), in one kernel."""
    num_tokens, num_heads, dim = query.shape
    padded = query.new_empty((num_tokens, heads, dim))
    element_size = query.element_size()
    if (
        query.stride(2) != 1
        or query.stride(1) != dim
        or query.stride(0) * element_size % 8
        or dim * element_size % 8
        or query.data_ptr() % 8
    ):
        padded[:, num_heads:].zero_()
        padded[:, :num_heads].copy_(query)
        return padded
    if num_tokens:
        padded_row_words = heads * dim * element_size // 8
        block = 1024
        _pad_query_heads_kernel[(num_tokens, triton.cdiv(padded_row_words, block))](
            query.view(torch.int64),
            padded.view(torch.int64),
            query.stride(0) * element_size // 8,
            num_heads * dim * element_size // 8,
            padded_row_words,
            BLOCK=block,
        )
    return padded


@triton.jit
def _decode_query_quant_pad_kernel(
    nope_ptr,
    pe_ptr,
    scale_ptr,
    out_ptr,
    nope_stride0,
    nope_stride1,
    pe_stride0,
    pe_stride1,
    num_heads,
    padded_heads,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """One token's ``HEADS_PER_PROGRAM`` heads of cat(ql_nope, q_pe) quantized
    to FP8 with a static per-tensor scale (``scaled_fp8_quant``'s arithmetic:
    x * (1 / scale), clamped, round-to-nearest) into a [T, padded_heads,
    NOPE + ROPE] buffer; heads past ``num_heads`` are written as zeros."""
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    live = (heads < num_heads)[:, None]
    stored = (heads < padded_heads)[:, None]
    inv_scale = tl.fdiv(1.0, tl.load(scale_ptr).to(tl.float32), ieee_rounding=True)
    fp8_max: tl.constexpr = 448.0
    out_row = out_ptr + (token * padded_heads + heads[:, None]) * (NOPE + ROPE)

    cols = tl.arange(0, NOPE)[None, :]
    x = tl.load(
        nope_ptr + token * nope_stride0 + heads[:, None] * nope_stride1 + cols,
        mask=live,
        other=0.0,
    ).to(tl.float32)
    x = tl.minimum(tl.maximum(x * inv_scale, -fp8_max), fp8_max)
    tl.store(out_row + cols, x.to(out_ptr.dtype.element_ty), mask=stored)

    cols = tl.arange(0, ROPE)[None, :]
    x = tl.load(
        pe_ptr + token * pe_stride0 + heads[:, None] * pe_stride1 + cols,
        mask=live,
        other=0.0,
    ).to(tl.float32)
    x = tl.minimum(tl.maximum(x * inv_scale, -fp8_max), fp8_max)
    tl.store(out_row + NOPE + cols, x.to(out_ptr.dtype.element_ty), mask=stored)


def decode_query_quant_pad(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    scale: torch.Tensor,
    padded_heads: int,
) -> torch.Tensor | None:
    """cat(``ql_nope`` [T, H, L], ``q_pe`` [T, H, R]) quantized to E4M3 with the
    static per-tensor ``scale`` in one kernel, into a new [T, ``padded_heads``,
    L + R] buffer whose extra heads are zero. Bitwise equal to the decode
    query's cat + ``scaled_fp8_quant`` + head pad (three kernels); None when the
    shapes are outside the kernel's (power-of-two L and R, unit inner stride)."""
    num_tokens, num_heads, nope = ql_nope.shape
    rope = q_pe.shape[-1]
    if (
        ql_nope.dtype not in (torch.bfloat16, torch.float16)
        or q_pe.dtype != ql_nope.dtype
        or q_pe.shape[:2] != ql_nope.shape[:2]
        or ql_nope.stride(2) != 1
        or q_pe.stride(2) != 1
        or nope & (nope - 1)
        or rope <= 0
        or rope & (rope - 1)
        or padded_heads < num_heads
        or scale.numel() != 1
        or current_platform.fp8_dtype() != torch.float8_e4m3fn
    ):
        return None
    out = ql_nope.new_empty(
        (num_tokens, padded_heads, nope + rope), dtype=torch.float8_e4m3fn
    )
    if num_tokens:
        heads_per_program = 4
        _decode_query_quant_pad_kernel[
            (num_tokens, triton.cdiv(padded_heads, heads_per_program))
        ](
            ql_nope,
            q_pe,
            scale,
            out,
            ql_nope.stride(0),
            ql_nope.stride(1),
            q_pe.stride(0),
            q_pe.stride(1),
            num_heads,
            padded_heads,
            NOPE=nope,
            ROPE=rope,
            HEADS_PER_PROGRAM=heads_per_program,
            num_warps=4,
        )
    return out


@triton.jit
def _finish_dcp_partials_kernel(
    out_ptr,
    lse_ptr,
    counts_ptr,
    heads,
    out_row_stride,
    lse_row_stride,
    HEAD_DIM: tl.constexpr,
    HEADS_BLOCK: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
):
    """One row of a trtllm-gen DCP partial: without top-k entries (count 0) its
    output is zeroed and its LSE set to -inf; other rows are not touched."""
    row = tl.program_id(0).to(tl.int64)
    if tl.load(counts_ptr + row) == 0:
        head = tl.arange(0, HEADS_BLOCK)
        lse_row = lse_ptr + row * lse_row_stride
        tl.store(lse_row + head, float("-inf"), mask=head < heads)
        out_row = out_ptr + row * out_row_stride
        zeros = tl.zeros([OUT_BLOCK], dtype=out_ptr.dtype.element_ty)
        for start in range(0, heads * HEAD_DIM, OUT_BLOCK):
            item = start + tl.arange(0, OUT_BLOCK)
            tl.store(out_row + item, zeros, mask=item < heads * HEAD_DIM)


def _finish_dcp_partials(
    out: torch.Tensor, lse: torch.Tensor, counts: torch.Tensor
) -> None:
    """In place, for trtllm-gen's DCP partials ``out`` [T, H, D] and ``lse``
    [T, H] (views of its head-padded buffers): rows without top-k entries
    (``counts`` 0; the DCP index conversion compacts a row's valid entries to
    its front and counts them, so these are the rows whose entries are all -1)
    get output 0 and LSE -inf, the empty-shard identity of the DCP combine. One
    small kernel that writes only the empty rows."""
    num_tokens, heads, head_dim = out.shape
    assert out.stride(2) == 1 and out.stride(1) == head_dim and lse.stride(1) == 1
    assert counts.shape == (num_tokens,) and counts.is_contiguous()
    if num_tokens == 0:
        return
    _finish_dcp_partials_kernel[(num_tokens,)](
        out,
        lse,
        counts,
        heads,
        out.stride(0),
        lse.stride(0),
        HEAD_DIM=head_dim,
        HEADS_BLOCK=triton.next_power_of_2(heads),
        OUT_BLOCK=4096,
    )


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    use_prims_ts: bool = False

    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    # Spec-decode rows attend to their own top-k slots, which the shared DCP index
    # filter localizes at any interleave (as for FlashMLA sparse).
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True
    lse_base_on_e: bool = False
    # ShardQ under DCP only: TP-only steps keep the head-sharded path
    # (Inferact/mum#66 also turns TP ShardQ on for this backend).
    supports_dcp_shardq: ClassVar[bool] = True
    # fp8 query quantization is required when using fp8 kv_cache,
    # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
    # for query and kv_cache (mixed bf16+fp8 is not supported).
    supports_quant_query_input: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )

        self._workspace_buffer: torch.Tensor | None = None
        # The last head-padded decode query quantize_decode_query wrote; held
        # (so no other tensor can take its address) until the kernel reads it.
        self._fused_query: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # Native no-rope MLA additionally requires a per-query-token active
        # top-k length tensor.
        self.is_nope_mla = self.qk_rope_head_dim == 0

        # The layer quantizes the query with its static per-tensor q_scale (a
        # dynamic scale cannot reach the kernel's host-scalar softmax scale in a
        # CUDA graph) before the DCP query all-gather.
        vllm_config = get_current_vllm_config()
        self.use_prims_ts = use_prims_ts(vllm_config, kv_cache_dtype)
        # Query heads that callers besides this layer's decode run prims_ts at
        # (e.g. a q mailbox padded to the 2-CTA heads); reserve_workspace plans
        # them with the layer's own.
        self.prims_ts_extra_heads: set[int] = set()
        if self.use_prims_ts:
            # prims_ts returns natural-log LSE; (0, -inf) for rows without
            # entries, the DCP merge identity.
            self.lse_base_on_e = True
            self.prims_ts_format = _PRIMS_TS_FORMATS[kv_cache_dtype]
            self.prims_ts_buckets = _prims_ts_row_buckets(vllm_config)
            logger.info_once(
                "FLASHINFER_MLA_SPARSE: FlashInfer prims_ts sparse MLA (%s cache)",
                kv_cache_dtype,
            )

    def _prepare_mqa_kernel(
        self,
        layer: AttentionLayer,
        device: torch.device,
    ) -> None:
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(device)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

    def _decode_query_heads(self, num_heads: int) -> int:
        """Query heads the decode kernel runs for ``num_heads`` real heads."""
        if self.use_prims_ts:
            return self._prims_ts_query_heads(num_heads)
        return trtllm_gen_query_heads(num_heads)

    def quantize_decode_query(
        self, ql_nope: torch.Tensor, q_pe: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor | None:
        """The FP8 decode query [T, H, L + R] in one kernel (concat, static
        quant and the decode kernel's head padding), or None to let the layer
        concat and quantize it. The result is the real-head view of a
        head-padded buffer that ``_padded_query`` hands to the kernel as is.

        Only where the query goes to this rank's kernel unchanged: no DCP
        query gather, no ShardQ exchange."""
        if self.dcp_world_size > 1 or self.use_shardq:
            return None
        num_heads = ql_nope.shape[1]
        padded = decode_query_quant_pad(
            ql_nope, q_pe, q_scale, self._decode_query_heads(num_heads)
        )
        if padded is None:
            return None
        self._fused_query = padded
        return padded[:, :num_heads]

    def _padded_query(self, query: torch.Tensor, heads: int) -> torch.Tensor:
        """``query`` [T, H, D] with ``heads`` heads (extra heads zero): a row
        slice of the buffer ``quantize_decode_query`` wrote when ``query`` is
        one of its views (no copy), else a padded copy."""
        fused = getattr(self, "_fused_query", None)
        if (
            fused is not None
            and fused.shape[1] == heads
            and query.dtype == fused.dtype
            and query.stride() == fused.stride()
            and query.shape[2] == fused.shape[2]
        ):
            row_bytes = fused.stride(0) * fused.element_size()
            offset = query.data_ptr() - fused.data_ptr()
            if offset >= 0 and offset % row_bytes == 0:
                start = offset // row_bytes
                if start + query.shape[0] <= fused.shape[0]:
                    return fused[start : start + query.shape[0]]
        return _pad_query_heads(query, heads)

    def _prims_ts_query_heads(self, num_heads: int) -> int:
        from vllm.utils.flashinfer import prims_ts_sparse_mla_schedules_native_heads

        if 64 < num_heads < _PRIMS_TS_2CTA_HEADS and (
            not prims_ts_sparse_mla_schedules_native_heads()
        ):
            return _PRIMS_TS_2CTA_HEADS
        return num_heads

    def _get_prims_ts_plans(self, device: torch.device, heads: int, max_topk: int):
        return _prims_ts_plans.get(
            device,
            self.prims_ts_buckets,
            self._prims_ts_query_heads(heads),
            self.prims_ts_format,
            max_topk,
            self.need_to_return_lse_for_decode,
        )

    def reserve_workspace(self) -> None:
        if not self.use_prims_ts:
            self._reserve_trtllm_gen()
            return
        heads = self.num_heads * self.dcp_world_size
        # ShardQ steps attend their token slice with every TP rank's heads.
        shardq_heads = self.num_heads * get_tensor_model_parallel_world_size()
        # Plan (and compile) every row capacity before serving starts.
        if self.topk_indices_buffer is not None:
            planned = {heads, shardq_heads} if self.use_shardq else {heads}
            for num_heads in planned | self.prims_ts_extra_heads:
                self._get_prims_ts_plans(
                    torch.device(
                        current_platform.device_type,
                        torch.accelerator.current_device_index(),
                    ),
                    num_heads,
                    self.topk_indices_buffer.shape[1],
                )
        # Claim the transient peak of one long prefill step in the profile run:
        # one chunk's head-padded query, output and LSE, and, for a step of
        # several chunks, the step output.
        # A ShardQ step attends only this rank's token slice, with every head.
        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        steps = [(max_tokens, heads)]
        if self.use_shardq:
            from vllm.v1.attention.ops.shardq import max_rank_tokens

            max_reqs = vllm_config.scheduler_config.max_num_seqs
            world = get_tensor_model_parallel_world_size()
            steps.append((max_rank_tokens(max_tokens, max_reqs, world), shardq_heads))
        peak = max(self._prims_ts_peak_bytes(*step) for step in steps)
        _ = torch.empty(peak, dtype=torch.uint8, device=current_platform.device_type)

    def _reserve_trtllm_gen(self) -> None:
        """The trtllm-gen path's share of the memory profile run, which skips
        attention:

        * persistent: the shared workspace (under DCP its softmax-stats slab
          covers trtllm_gen_lse_max_rows rows: max_num_batched_tokens, 8 GiB at
          128 padded heads and 32K tokens, or under DCP ShardQ the DCP path's
          bounded steps, 0.9 GiB at 2048 rows) and the multi-CTA KV counters,
          at their final size. Otherwise they come with the metadata builders,
          after the KV cache was sized;
        * transient: the attention working set of one full step: the layer's
          token-major query (model dtype, then its FP8 quantization) and the
          kernel's output, physical top-k and, on the DCP path, the gathered and
          head-padded query and the LSE; the larger of a ShardQ token slice
          (every head) and a DCP-path step.
        """
        vllm_config = get_current_vllm_config()
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        dcp_shardq = self.use_shardq and self.dcp_world_size > 1
        reserve_trtllm_gen_buffers(vllm_config, device, self.dcp_world_size, dcp_shardq)

        scheduler_config = vllm_config.scheduler_config
        max_tokens = scheduler_config.max_num_batched_tokens
        qk_dim = self.kv_lora_rank + self.qk_rope_head_dim
        model_itemsize = vllm_config.model_config.dtype.itemsize
        fp8_query = is_quantized_kv_cache(self.kv_cache_dtype)
        q_itemsize = 1 if fp8_query else model_itemsize
        topk = (
            self.topk_indices_buffer.shape[1]
            if self.topk_indices_buffer is not None
            else 0
        )
        layer_query = (
            max_tokens * self.num_heads * qk_dim * (model_itemsize + int(fp8_query))
        )

        def kernel_bytes(rows: int, heads: int, dcp_path: bool) -> int:
            padded = trtllm_gen_query_heads(heads)
            # bf16 output and int32 physical top-k of every row.
            per_row = padded * self.kv_lora_rank * 2 + topk * 4
            if dcp_path:
                # The all-gathered query, its head-padded copy and the LSE.
                per_row += (heads + padded) * qk_dim * q_itemsize + padded * 4
            elif padded != heads:
                per_row += padded * qk_dim * q_itemsize
            return rows * per_row

        world = get_tensor_model_parallel_world_size()
        steps = []
        if self.use_shardq:
            from vllm.v1.attention.ops.shardq import max_rank_tokens

            # shard_q hands the kernel its head-padded slice (no copy).
            rows = max_rank_tokens(max_tokens, scheduler_config.max_num_seqs, world)
            padded = trtllm_gen_query_heads(self.num_heads * world)
            steps.append(rows * (padded * self.kv_lora_rank * 2 + topk * 4))
        if self.dcp_world_size > 1:
            # The DCP path's rows: under DCP ShardQ its steps are bounded.
            rows = trtllm_gen_lse_max_rows(vllm_config, dcp_shardq)
            steps.append(
                kernel_bytes(rows, self.num_heads * self.dcp_world_size, True)
            )
        else:
            steps.append(kernel_bytes(max_tokens, self.num_heads, False))
        _ = torch.empty(layer_query + max(steps), dtype=torch.uint8, device=device)

    def _prims_ts_peak_bytes(self, num_tokens: int, heads: int) -> int:
        padded = self._prims_ts_query_heads(heads)
        rows = min(num_tokens, self.prims_ts_buckets[-1])
        peak = rows * padded * (self.kv_lora_rank + self.qk_rope_head_dim + 1024 + 4)
        if num_tokens > rows:
            peak += num_tokens * heads * self.kv_lora_rank * 2
        return peak

    def _forward_prims_ts(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        layer: AttentionLayer,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sparse MLA of every row of ``q`` [T, H, 576] (E4M3, quantized by the
        layer with ``q_scale``) over its ``valid_counts`` leading physical rows
        of ``topk_indices`` (-1 entries are masked)."""
        assert q.dtype == torch.float8_e4m3fn, q.dtype
        num_tokens, heads, _ = q.shape
        padded = self._prims_ts_query_heads(heads)
        # Storage rows; the backend aligns block strides to whole rows.
        is_fp8 = self.prims_ts_format == "native"
        rows, _ = flat_kv_row_view(
            kv_cache.view(torch.float8_e4m3fn if is_fp8 else torch.uint8), block_size
        )
        kv_descale = layer._k_scale_float if is_fp8 else 1.0
        softmax_scale = self.scale * layer._q_scale_float * kv_descale
        return_lse = self.need_to_return_lse_for_decode
        buckets = self.prims_ts_buckets
        # A call of one chunk (every decode step) returns views of the kernel's
        # head-padded output and LSE: the DCP combine reads them in place.
        out_heads = padded if num_tokens <= buckets[-1] else heads
        out = q.new_empty(
            (num_tokens, out_heads, self.kv_lora_rank), dtype=torch.bfloat16
        )
        lse = (
            q.new_empty((num_tokens, out_heads), dtype=torch.float32)
            if return_lse
            else None
        )
        plans = self._get_prims_ts_plans(q.device, heads, topk_indices.shape[1])
        for start in range(0, num_tokens, buckets[-1]):
            end = min(num_tokens, start + buckets[-1])
            n = end - start
            capacity = buckets[bisect.bisect_left(buckets, n)]
            plan, qo_indptr = plans[capacity]
            if n < capacity:
                qo_indptr = qo_indptr.clamp(max=n)
            query = q[start:end]
            chunk_out, chunk_lse = (
                out[start:end],
                None if lse is None else lse[start:end],
            )
            if padded != heads:
                # Zero query heads: computed, then dropped.
                query = self._padded_query(query, padded)
            if out_heads != padded:
                chunk_out = out.new_empty((n, padded, self.kv_lora_rank))
                if lse is not None:
                    chunk_lse = lse.new_empty((n, padded))
            plan.run(
                query,
                rows,
                topk_indices[start:end],
                valid_counts[start:end],
                qo_indptr=qo_indptr,
                softmax_scale=softmax_scale,
                output_scale=kv_descale,
                out=chunk_out,
                lse=chunk_lse,
                validate=False,
            )
            if out_heads != padded:
                # The copy moves 8-byte words (1 KiB output rows).
                out[start:end].view(torch.int64).copy_(
                    chunk_out.view(torch.int64)[:, :heads]
                )
                if lse is not None:
                    assert chunk_lse is not None
                    lse[start:end] = chunk_lse[:, :heads]
        self._fused_query = None
        return out[:, :heads], None if lse is None else lse[:, :heads]

    def autotune_hisparse_decode(self, layer: AttentionLayer) -> None:
        """Autotune the largest legal HiSparse decode batch."""
        assert isinstance(self.index_group, HiSparseMLAIndexGroup)
        cache = self.index_group.cache(self.index_group_index)
        assert self.topk_indices_buffer is not None

        runtime = cache.runtime
        kv_cache = runtime.hot.attention_cache
        num_tokens = runtime.max_num_reqs
        topk_tokens = self.topk_indices_buffer.shape[1]

        q_dtype = (
            current_platform.fp8_dtype()
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else kv_cache.dtype
        )
        q = torch.zeros(
            (
                num_tokens,
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=q_dtype,
            device=kv_cache.device,
        )
        topk_indices = (
            torch.arange(
                topk_tokens,
                dtype=torch.int32,
                device=kv_cache.device,
            )
            .expand(num_tokens, -1)
            .contiguous()
        )
        seq_lens = torch.full(
            (num_tokens,),
            topk_tokens,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._forward_mqa_kernel(
            q,
            kv_cache,
            topk_indices,
            seq_lens,
            layer=layer,
            block_size=runtime.hot.block_size,
            is_decode=True,
        )

    def shardq_query_heads(self, num_heads: int) -> int:
        if self.use_prims_ts:
            return num_heads
        return trtllm_gen_query_heads(num_heads)

    def _forward_mqa_kernel(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        layer: AttentionLayer,
        block_size: int,
        is_decode: bool,
        return_lse: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if return_lse is None:
            return_lse = self.need_to_return_lse_for_decode
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if q_pe.shape[-1] == 0 and ql_nope.is_contiguous():
                q = ql_nope
            else:
                q = torch.cat(q, dim=-1)

        if self.use_prims_ts:
            return self._forward_prims_ts(
                q, kv_cache, topk_indices, seq_lens, layer, block_size
            )

        self._prepare_mqa_kernel(layer, q.device)
        assert self._workspace_buffer is not None
        assert self.bmm1_scale is not None
        assert self.bmm2_scale is not None

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_cache.view(q.dtype), block_size
        )
        # Block sizes and strides have been aligned to 32 rows.
        kv_cache = kv_rows.view(
            -1, math.gcd(block_stride_rows, block_size, 64), kv_rows.shape[-1]
        )

        num_heads = q.shape[1]
        padded_heads = trtllm_gen_query_heads(num_heads)
        if padded_heads != num_heads:
            q = self._padded_query(q, padded_heads)
        self._fused_query = None

        # Single-token sparse decode. trtllm-gen requires the q_len_per_request
        # dim, but the sparse attention mask is fully per-token (each query token
        # carries its own top-k index row), so unsqueeze is sufficient and
        # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
        # deferred until MTP is validated end-to-end for this backend.
        query = q.unsqueeze(1)
        block_tables = topk_indices.unsqueeze(1)

        # page_table width = topk buffer width, which kpool widens past
        # index_topk (topk_tokens) and rounds up to a multiple of 128. The
        # kernel treats sparse_mla_top_k as the page-table *capacity* and bounds
        # the active per-query length by ``seq_lens`` (the compacted valid
        # count), so the -1 padding slots past seq_lens are never attended to.
        # Use the actual buffer width instead of the fixed topk_tokens, which
        # mismatches the page_table when index_kpool > 1.
        sparse_topk_capacity = topk_indices.shape[1]

        needs_empty_query_guard = self.need_to_return_lse_for_decode or isinstance(
            self.index_group, HiSparseMLAIndexGroup
        )
        topk_lens = None
        if self.is_nope_mla:
            # Resident TP queries have nonempty selections. Preserve empty-query
            # handling for DCP's local selections and host-backed HiSparse.
            topk_lens = seq_lens
            if needs_empty_query_guard:
                topk_lens = prepare_sparse_mla_safe_lengths(topk_indices, seq_lens)

        num_tokens = q.shape[0]
        out = q.new_empty(
            (num_tokens, padded_heads, self.kv_lora_rank), dtype=torch.bfloat16
        )
        lse = None
        if return_lse:
            # trtllm-gen carves each launch's softmax stats from the workspace
            # unchecked; the slab covers trtllm_gen_lse_max_rows rows only.
            launch_rows = min(num_tokens, _TRTLLM_GEN_MAX_REQUESTS)
            capacity = trtllm_gen_lse_rows_capacity(
                self._workspace_buffer.numel(), padded_heads
            )
            if launch_rows > capacity:
                raise RuntimeError(
                    f"trtllm-gen sparse MLA: a launch of {launch_rows} rows x "
                    f"{padded_heads} heads asks for LSE, but the workspace's "
                    f"softmax-stats slab holds {capacity} rows "
                    "(trtllm_gen_lse_max_rows). Running it would overwrite the "
                    "workspace's other regions."
                )
            lse = q.new_empty((num_tokens, padded_heads), dtype=torch.float32)

        # One trtllm-gen request per token, and requests map to gridDim.z: a
        # launch over more than 65535 tokens fails without raising and leaves
        # ``out`` unwritten (flashinfer-ai/flashinfer#3125). Launch row chunks.
        for start in range(0, num_tokens, _TRTLLM_GEN_MAX_REQUESTS):
            end = min(start + _TRTLLM_GEN_MAX_REQUESTS, num_tokens)
            extra_kwargs = _kv_counters(end - start, padded_heads, q.device)
            if topk_lens is not None:
                extra_kwargs["sparse_mla_top_k_lens"] = topk_lens[start:end]
            if lse is not None:
                extra_kwargs["lse"] = lse[start:end]
            trtllm_batch_decode_with_kv_cache_mla(
                query=query[start:end],
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=block_tables[start:end],
                seq_lens=seq_lens[start:end],
                max_seq_len=sparse_topk_capacity,
                out=out[start:end].unsqueeze(1),
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                sparse_mla_top_k=sparse_topk_capacity,
                return_lse=lse is not None,
                **extra_kwargs,
            )
        out = out[:, :num_heads]
        if lse is not None:
            lse = lse[:, :num_heads]
        if self.is_nope_mla and needs_empty_query_guard:
            empty_queries = seq_lens == 0
            if lse is not None:
                # DCP combine already suppresses outputs with zero LSE weight.
                lse.masked_fill_(empty_queries[:, None], float("-inf"))
            else:
                out.masked_fill_(empty_queries[:, None, None], 0.0)
        elif lse is not None:
            _finish_dcp_partials(out, lse, seq_lens)
        return out, lse
