# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""

import functools
import os
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Workspace upper bound for tokenspeed_mla_decode (per-device, lazy):
#   num_sms * num_heads * MAX_Q_LEN * (kv_lora_rank + 1) * sizeof(float32)
# Matches the kernel's `get_workspace_size` formula. MAX_Q_LEN=8 covers up to
# EAGLE3 / MTP-2 spec decoding query lengths; larger q_len fails the kernel's
# own buffer check.
_TOKENSPEED_MAX_Q_LEN = 8


# TokenSpeed MLA decode addresses its split-KV scratch with Int32 byte offsets
# (the LSE partials start at B * rows * splits * D * 4 bytes) and refuses a
# workspace of 2^31 bytes or more.
_TOKENSPEED_WORKSPACE_LIMIT = 1 << 31


def _var_split_rows(num_heads: int, query_len: int) -> int:
    """Query rows per request in the split-KV scratch (an upper bound: packed
    128-row tiles or folded heads)."""
    return max(128 * -(-num_heads * query_len // 128), num_heads * query_len)


def _var_split_cap(
    num_requests: int,
    num_heads: int,
    query_len: int,
    kv_lora_rank: int,
    max_split_kv: int,
) -> int:
    """``max_split_kv`` lowered, if needed, so that ``num_requests`` requests
    with that many splits each stay below the kernel's Int32 workspace
    limit: B x rows x splits x (D + 1) x 4 bytes < 2^31."""
    per_split = (
        max(num_requests, 1)
        * _var_split_rows(num_heads, query_len)
        * (kv_lora_rank + 1)
        * 4
    )
    return min(max_split_kv, (_TOKENSPEED_WORKSPACE_LIMIT - 1) // per_split)


def _get_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    min_split_kv: int,
    max_decode_tokens: int,
    max_split_kv: int = 0,
    max_decode_requests: int = 0,
    query_len: int = 1,
) -> torch.Tensor:
    from tokenspeed_mla import get_num_sm

    # TokenSpeed reduction scratch: B * H * Q * split_kv * (D + 1) * 4 bytes,
    # using the effective query layout and FP32 partial outputs plus LSE.
    # Automatic splitting lowers split_kv as the batch grows; split_kv=1
    # needs no reduction scratch, so an SM-based bound covers all batch sizes.
    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    if min_split_kv > 1:
        # A forced floor breaks that bound: reserve for the maximum batch.
        # Packed queries pad head rows to 128; reserve one tile per query token.
        needed = max(
            needed,
            max_decode_tokens
            * max(num_heads, 128)
            * min_split_kv
            * (kv_lora_rank + 1)
            * 4,
        )
        logger.info_once(
            "TokenSpeed MLA decode workspace: %.2f GiB", needed / (1 << 30)
        )
    if max_split_kv > 1:
        # Length-proportional splits: the grid holds max_split_kv splits of
        # every request's query rows. _var_split_cap lowers the split count
        # of large batches so that no batch needs more than the kernel's
        # Int32 byte-offset limit.
        needed = max(
            needed,
            min(
                max_decode_requests
                * _var_split_rows(num_heads, query_len)
                * max_split_kv
                * (kv_lora_rank + 1)
                * 4,
                _TOKENSPEED_WORKSPACE_LIMIT - 1,
            ),
        )
        logger.info_once(
            "TokenSpeed MLA length-proportional split-KV workspace: %.2f GiB",
            needed / (1 << 30),
        )
    return current_workspace_manager().get_simultaneous(((needed,), torch.int8))[0]


# SM107 (VR200) M256 FP8 decode: TokenSpeed's 4-CTA-cluster kernel
# (mla_decode_fp8_sm107.py, cluster launch control) written with a BF16 output. The
# public entry point only takes it for FP8 outputs. For the K3.1 DSpark drafter (64
# heads x q8, TP1 DEP ranks, AgentX contexts) it cuts the drafter's CUPTI kernel time
# per step from 3.04 to 2.71 ms (r16) and 5.21 to 4.47 ms (r28) vs the SM100 FP8 kernel
# + length-proportional split (DEP4 8-layer bench, 2 nodes), at the same error against
# an FP32 reference. Split-KV floor per batch (_sm107_split_floors): 32, and 24 from
# _SM107_LARGE_BATCH requests up (r28 q8 step ABBA: drafter 4.46 -> 4.36 ms per step;
# at 1-6 and 16 requests 24 is neutral to 4-7% slower in the kernel microbench).
# VLLM_TOKENSPEED_SM107_SPLIT sets one floor for every batch (0 disables the path).
# Only shapes that fill at least one 256-row tile (num_heads * q_len >= 256) take
# it: at 4 / 16 heads the tile is mostly padding and it is 18-32% slower than the SM100
# path.
_SM107_M_TILE = 256
_SM107_MAX_TILES = 4
_SM107_SPLIT = 32
_SM107_SPLIT_LARGE_BATCH = 24
_SM107_LARGE_BATCH = 24

# The private TokenSpeed helpers this path calls, with the parameters it passes.
# Releases differ: tokenspeed-mla 2.0.2 adds a required ``enable_packed_q`` (True =
# always pack rows into 256-row tiles, the behaviour of the later builds that dropped
# the flag). Any other signature disables the path instead of failing at the first
# decode.
_SM107_DECODE_KWARGS = frozenset(
    (
        "q_latent", "q_rope", "c_latent", "c_rope", "block_tables", "seq_lens",
        "workspace_buffer", "out", "page_size", "max_seq_len", "softmax_scale",
        "output_scale", "is_var_seq", "causal_mask", "window_left", "enable_pdl",
        "return_lse", "causal_seqs", "cp_world", "cp_rank", "local_visible_lens",
        "min_split_kv", "compute_capability",
    )
)  # fmt: skip
_SM107_COMPILE_KWARGS = frozenset(
    (
        "page_size", "num_heads", "seq_len_q", "is_var_seq", "causal_mask",
        "is_workspace_size_zero", "use_pdl", "use_fp16_softmax", "reducer_max_splits",
        "output_dtype",
    )
)  # fmt: skip


@functools.cache
def _sm107_decode_extra_kwargs() -> "dict[str, bool] | None":
    """Extra keyword arguments ``_sm107_fp8_out_decode`` needs in the installed
    TokenSpeed ({} or {"enable_packed_q": True}), or None when its SM107 helpers
    are missing or have a signature this path does not know."""
    import inspect

    try:
        from tokenspeed_mla import mla_decode as md
    except ImportError:
        return None
    names = (
        "_sm107_fp8_out_decode", "_get_compiled_sm107_mla_kernel",
        "_get_sm107_split_kv_and_workspace_size", "_get_reducer_max_splits",
        "get_max_active_clusters", "_SM107_FP16_SOFTMAX",
    )  # fmt: skip
    if not all(hasattr(md, n) for n in names):
        return None

    def required(params) -> set[str]:
        return {n for n, p in params.items() if p.default is inspect.Parameter.empty}

    try:
        dec = inspect.signature(md._sm107_fp8_out_decode).parameters
        comp = inspect.signature(md._get_compiled_sm107_mla_kernel).parameters
        split = inspect.signature(md._get_sm107_split_kv_and_workspace_size).parameters
    except (TypeError, ValueError):
        return None
    extra: dict[str, bool] = {}
    if "enable_packed_q" in dec:
        extra["enable_packed_q"] = True
    if (
        not set(dec) >= _SM107_DECODE_KWARGS
        or not required(dec) <= _SM107_DECODE_KWARGS | set(extra)
        or set(comp) != _SM107_COMPILE_KWARGS
        or len(split) != 7
    ):
        logger.info_once(
            "TokenSpeed MLA decode: the installed tokenspeed_mla SM107 helpers "
            "have an unknown signature; SM107 M256 decode disabled."
        )
        return None
    return extra


def _sm107_split_floors() -> tuple[int, int]:
    """(split-KV floor below _SM107_LARGE_BATCH requests, floor from there up)."""
    env = os.environ.get("VLLM_TOKENSPEED_SM107_SPLIT")
    if env is not None:
        return int(env), int(env)
    return _SM107_SPLIT, _SM107_SPLIT_LARGE_BATCH


def _sm107_batch_split(min_split: int, min_split_large: int | None, b: int) -> int:
    if min_split_large is not None and b >= _SM107_LARGE_BATCH:
        return min_split_large
    return min_split


@functools.cache
def _sm107_workspace_bound(
    max_requests: int,
    tiles: int,
    kv_lora_rank: int,
    max_seq_len: int,
    min_split: int,
    min_split_large: int | None = None,
) -> int:
    """Largest SM107 split-KV workspace (bytes) of any batch of at most
    ``max_requests`` requests and ``max_seq_len`` tokens (the planner's split
    count is non-decreasing in the sequence length), or -1 when one exceeds
    the kernel's Int32 offsets."""
    from tokenspeed_mla import mla_decode as md

    clusters = md.get_max_active_clusters(4) * 4
    worst = 0
    for b in range(1, max_requests + 1):
        try:
            _, size = md._get_sm107_split_kv_and_workspace_size(
                b,
                tiles,
                _SM107_M_TILE,
                kv_lora_rank,
                clusters,
                max_seq_len,
                _sm107_batch_split(min_split, min_split_large, b),
            )
        except ValueError:
            return -1
        worst = max(worst, size)
    return worst


def sm107_warmup(
    impl: "TokenspeedMLAImpl",
    page_size: int,
    q_lens: "set[int]",
    causal_modes: "tuple[bool, ...]",
    max_requests: int,
    max_seq_len: int,
) -> int:
    """Compile every SM107 decode kernel key (``_get_compiled_sm107_mla_kernel``)
    that ``impl._forward_sm107`` can reach: the key depends on the batch and
    max_seq_len only through the split count (no workspace at 1 split, else the
    reducer's power-of-two capacity). Returns the number of keys compiled."""
    from tokenspeed_mla import mla_decode as md

    if not impl._sm107_split:
        return 0
    heads = impl.num_heads
    clusters = md.get_max_active_clusters(4) * 4
    max_tiles = -(-max_seq_len // 128)
    tile_counts = sorted(
        {t for t in range(1, 257)}
        | {1 << i for i in range(max_tiles.bit_length() + 1)}
        | {max_tiles}
    )
    keys = set()
    for q_len in sorted(q_lens):
        if (
            not 0 < q_len <= impl._query_len
            or not _SM107_M_TILE <= heads * q_len <= _SM107_M_TILE * _SM107_MAX_TILES
        ):
            continue
        tiles = -(-heads * q_len // _SM107_M_TILE)
        for b in range(1, max_requests + 1):
            for t in tile_counts:
                if t > max_tiles:
                    continue
                try:
                    split, size = md._get_sm107_split_kv_and_workspace_size(
                        b,
                        tiles,
                        _SM107_M_TILE,
                        impl.kv_lora_rank,
                        clusters,
                        128 * t,
                        _sm107_batch_split(
                            impl._sm107_split,
                            getattr(impl, "_sm107_split_large", None),
                            b,
                        ),
                    )
                except ValueError:
                    continue
                for causal in causal_modes:
                    keys.add(
                        (q_len, causal, size == 0, md._get_reducer_max_splits(split))
                    )
    for q_len, causal, ws_zero, reducer in sorted(keys):
        md._get_compiled_sm107_mla_kernel(
            page_size=page_size,
            num_heads=heads,
            seq_len_q=q_len,
            is_var_seq=True,
            causal_mask=causal,
            is_workspace_size_zero=ws_zero,
            use_pdl=False,
            use_fp16_softmax=md._SM107_FP16_SOFTMAX,
            reducer_max_splits=reducer,
            output_dtype=torch.bfloat16,
        )
    return len(keys)


class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # The kernel accepts an explicit causal mask, so a non-causal DSpark
    # block can remain fused instead of being flattened to single tokens.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            MLACommonMetadata,
            supports_dcp_with_varlen=True,
        )


class TokenspeedMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "TOKENSPEED_MLA"

    @staticmethod
    def get_impl_cls() -> type["TokenspeedMLAImpl"]:
        return TokenspeedMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TokenspeedMLAMetadataBuilder"]:
        return TokenspeedMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

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
        # Surface a clear install hint up front rather than letting a raw
        # ModuleNotFoundError fire deep inside `forward_mqa` at first request.
        try:
            import tokenspeed_mla  # noqa: F401
        except ImportError:
            return (
                "tokenspeed_mla package is not installed. "
                "Install it with: `uv pip install tokenspeed-mla`"
            )

        # tokenspeed_mla CuTe DSL kernel is shape-specialized for DeepSeek R1
        # MLA dimensions (qk_nope=128, qk_rope=64, v=128). Reject anything else.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 0)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            v_head_dim = getattr(hf_text_config, "v_head_dim", 0)
            if qk_nope_head_dim != 128 or qk_rope_head_dim != 64 or v_head_dim != 128:
                return (
                    "tokenspeed_mla requires DeepSeek R1 MLA dimensions "
                    "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128), "
                    f"got ({qk_nope_head_dim}, {qk_rope_head_dim}, {v_head_dim})"
                )
        return None


class TokenspeedMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True
    # tokenspeed_mla_decode returns LSE in log2 units; its own DCP test merges
    # partial outputs with exp2(lse).
    lse_base_on_e: bool = False

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
        **mla_args,
    ) -> None:
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
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TokenspeedMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TokenspeedMLAImpl"
            )

        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TokenspeedMLAImpl requires an FP8 KV cache "
                "(--kv-cache-dtype fp8 or fp8_e4m3); "
                f"got kv_cache_dtype={self.kv_cache_dtype!r}."
            )

        config = get_current_vllm_config()
        self._min_split_kv = config.attention_config.tokenspeed_mla_min_split_kv
        self._max_split_kv = config.attention_config.tokenspeed_mla_max_split_kv
        spec = config.speculative_config
        self._query_len = 1 + (spec.num_speculative_tokens if spec else 0)
        self._max_decode_requests = max(
            config.scheduler_config.max_num_seqs,
            -(-(config.compilation_config.max_cudagraph_capture_size or 0)
              // self._query_len),
        )
        if self._max_split_kv > 1:
            import tokenspeed_mla

            if not hasattr(tokenspeed_mla, "tokenspeed_mla_var_split_counts"):
                raise ValueError(
                    "attention_config.tokenspeed_mla_max_split_kv needs a "
                    "tokenspeed_mla with length-proportional split-KV "
                    "(tokenspeed_mla_var_split_counts)"
                )
        self._max_decode_tokens = 0
        if self._min_split_kv > 1:
            query_len = self._query_len
            self._max_decode_tokens = max(
                min(
                    config.scheduler_config.max_num_batched_tokens,
                    config.scheduler_config.max_num_seqs * query_len,
                ),
                config.compilation_config.max_cudagraph_capture_size or 0,
            )

        # SM107 M256 decode (see _sm107_split_floors): enabled per call when the
        # shape fits; the workspace is sized once for the largest batch.
        self._sm107_split = 0
        self._sm107_split_large: int | None = None
        self._sm107_workspace = 0
        split, split_large = _sm107_split_floors()
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (10, 7)
            and split > 0
            and split_large > 0
            and _SM107_M_TILE
            <= num_heads * self._query_len
            <= _SM107_M_TILE * _SM107_MAX_TILES
            and _sm107_decode_extra_kwargs() is not None
        ):
            bound = _sm107_workspace_bound(
                self._max_decode_requests,
                -(-num_heads * self._query_len // _SM107_M_TILE),
                self.kv_lora_rank,
                config.model_config.max_model_len,
                split,
                split_large,
            )
            if bound >= 0:
                self._sm107_split = split
                self._sm107_split_large = split_large
                self._sm107_workspace = max(bound, 1)
                logger.info_once(
                    "TokenSpeed MLA decode: SM107 M256 kernel (BF16 out, split "
                    "floor %d, %d from %d requests, workspace %.2f GiB)",
                    split,
                    split_large,
                    _SM107_LARGE_BATCH,
                    bound / (1 << 30),
                )

        self.softmax_scale: float | None = None
        self.output_scale: float | None = None
        # NIXL resolves interleaving after model construction; retain the config
        # rather than caching its initial interleave size.
        self._parallel_config = config.parallel_config

        # Pre-JIT BF16 and FP8 prefill kernels here too — decode impl always
        # runs when tokenspeed is selected, prefill backend may not (user can
        # pair with flash_attn / trtllm). Idempotent.
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                d_v=self.v_head_dim,
                enable_pdl=False,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from tokenspeed_mla import tokenspeed_mla_decode

        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # supports_quant_query_input=True (set in MLACommonImpl) tells the
        # pipeline to concat+FP8-quantize Q upstream via _decode_concat_quant_fp8_op.
        # The kernel is shape-specialized for FP8 Q + FP8 KV, so anything else
        # here means the upstream quant didn't run and the kernel will produce
        # garbage.
        assert q.dtype == torch.float8_e4m3fn, (
            f"TokenspeedMLAImpl expected FP8 query (supports_quant_query_input=True), "
            f"got {q.dtype}. Pipeline isinstance(q, tuple)={isinstance(q, tuple)}, "
            f"q_scale={layer._q_scale_float}, k_scale={layer._k_scale_float}."
        )

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        block_tables = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        causal_seqs = attn_metadata.decode.dcp_tot_seq_lens

        # tokenspeed_mla_decode expects query shape
        # (num_decodes, q_len_per_request, num_heads, head_dim).
        if num_decode_tokens % num_decodes != 0:
            logger.warning_once(
                """TokenspeedMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.softmax_scale is None:
            # FP8 KV cache is mandatory for this backend, so q_scale/k_scale
            # always apply. softmax_scale is bmm1; output_scale is bmm2 — both
            # required to recover the correct attention output from the FP8
            # KV cache (V is stored as V_real/k_scale).
            self.softmax_scale = (
                self.scale * layer._q_scale_float * layer._k_scale_float
            )
            self.output_scale = layer._k_scale_float

        if (
            self._sm107_split
            and self.dcp_world_size == 1
            and q.shape[1] <= self._query_len  # DSpark drafts with query_len - 1
            and q.shape[1] * q.shape[2] >= _SM107_M_TILE  # one full M tile
            and q.shape[0] <= self._max_decode_requests
            and q.stride(1) == q.shape[2] * q.stride(2)
            and not self.need_to_return_lse_for_decode
        ):
            return self._forward_sm107(q, kv_c_and_k_pe_cache, attn_metadata), None

        var_split_cfg = self._max_split_kv > 1 and self.dcp_world_size == 1
        var_split = False
        max_split_kv = 0
        if var_split_cfg:
            # Large batches (e.g. 128 requests x 4 tokens or 64 x 8 at 64
            # heads) get fewer splits per request so that the workspace stays
            # within the kernel's Int32 offsets; below 2 splits the uniform
            # path takes over (it has the same limit at its own split count).
            max_split_kv = _var_split_cap(
                q.shape[0],
                q.shape[-2],
                q.shape[1],
                self.kv_lora_rank,
                self._max_split_kv,
            )
            if max_split_kv < self._max_split_kv:
                logger.info_once(
                    "TokenSpeed MLA length-proportional split-KV: %d requests "
                    "x %d tokens use at most %d splits per request (Int32 "
                    "workspace limit; configured %d)",
                    q.shape[0],
                    q.shape[1],
                    max_split_kv,
                    self._max_split_kv,
                )
            # No request has more than ceil(max_seq_len / 128) KV tiles, so
            # more split slots would only add empty CTAs (the variable-split
            # grid holds max_split_kv slots per request). Batches whose
            # longest request has at most 2 x min_split_kv tiles (4K tokens
            # at the floor 16) have no skew worth the extra slots (+2.5% step
            # at 128 x 2K, n1): keep the uniform split there.
            max_tiles = -(-attn_metadata.max_seq_len // 128)
            max_split_kv = min(max_split_kv, max_tiles)
            var_split = max_split_kv > 1 and max_tiles > 2 * max(
                self._min_split_kv, 1
            )
        workspace = _get_workspace(
            q.device,
            q.shape[-2],
            self.kv_lora_rank,
            self._min_split_kv,
            self._max_decode_tokens,
            self._max_split_kv if var_split_cfg else 0,
            self._max_decode_requests,
            self._query_len,
        )
        split_kwargs: dict = {"min_split_kv": self._min_split_kv}
        if var_split:
            split_kwargs = {
                "max_split_kv": max_split_kv,
                "block_split_kvs": self._block_split_kvs(
                    attn_metadata, q, max_split_kv
                ),
            }

        # vLLM kv_c_and_k_pe_cache is already (num_blocks, block_size, head_size).
        # tokenspeed_mla_decode wants 3D — pass as-is (no unsqueeze, unlike trtllm).
        return_lse = self.need_to_return_lse_for_decode
        kernel_out = tokenspeed_mla_decode(
            query=q,
            kv_cache=kv_c_and_k_pe_cache,
            workspace_buffer=workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            enable_pdl=False,
            return_lse=return_lse,
            causal_mask=attn_metadata.causal,
            causal_seqs=causal_seqs if self.dcp_world_size > 1 else None,
            cp_world=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_interleave_size=self._parallel_config.cp_kv_cache_interleave_size,
            **split_kwargs,
        )
        if return_lse:
            o, lse = kernel_out
            # DCP ranks with no local KV may return undefined outputs (e.g.,
            # NaNs). The downstream DCP combine path must mask empty
            # shards and ignore their outputs when merging partial attention.
            lse = lse.view(-1, lse.shape[-1])
        else:
            o, lse = kernel_out, None

        o = o.view(-1, o.shape[-2], o.shape[-1])
        return o, lse

    def _forward_sm107(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        """The SM107 M256 FP8 decode with a BF16 output (see _sm107_split_floors)."""
        from tokenspeed_mla import mla_decode as md

        assert attn_metadata.decode is not None
        decode = attn_metadata.decode
        B, q_len, H, _ = q.shape
        L = self.kv_lora_rank
        (workspace,) = current_workspace_manager().get_simultaneous(
            ((self._sm107_workspace,), torch.int8)
        )
        out = torch.empty((B, q_len, H, L), dtype=torch.bfloat16, device=q.device)
        kv = kv_c_and_k_pe_cache
        md._sm107_fp8_out_decode(
            q_latent=q[..., :L],
            q_rope=q[..., L:],
            c_latent=kv[..., :L],
            c_rope=kv[..., L:],
            block_tables=decode.block_table,
            seq_lens=decode.seq_lens,
            workspace_buffer=workspace,
            out=out,
            page_size=kv.shape[-2],
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            is_var_seq=True,
            causal_mask=attn_metadata.causal,
            window_left=-1,
            enable_pdl=False,
            return_lse=False,
            causal_seqs=None,
            cp_world=1,
            cp_rank=0,
            local_visible_lens=None,
            min_split_kv=_sm107_batch_split(
                self._sm107_split, getattr(self, "_sm107_split_large", None), B
            ),
            compute_capability=(10, 7),
            **_sm107_decode_extra_kwargs(),
        )
        return out.view(-1, H, L)

    def _block_split_kvs(
        self, attn_metadata: MLACommonMetadata, q: torch.Tensor, max_split_kv: int
    ) -> torch.Tensor:
        """Per-request split counts, computed once per step (the layers of a
        step share the decode metadata) and cached on it."""
        decode = attn_metadata.decode
        assert decode is not None
        cached = getattr(decode, "_tokenspeed_block_split_kvs", None)
        if cached is not None and cached[0] == max_split_kv:
            return cached[1]
        from tokenspeed_mla import tokenspeed_mla_var_split_counts

        # With a split floor, the batch keeps the uniform floor's total
        # (min_split_kv per request) and moves splits to its long requests.
        seq_lens = decode.seq_lens
        total = self._min_split_kv * seq_lens.numel() if self._min_split_kv > 1 else 0
        counts = tokenspeed_mla_var_split_counts(
            seq_lens, max_split_kv, q.shape[-2], q.shape[1], q.dtype, total
        )
        decode._tokenspeed_block_split_kvs = (max_split_kv, counts)
        return counts
