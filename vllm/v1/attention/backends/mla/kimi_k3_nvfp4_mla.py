# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA backend with native NVFP4 decode attention."""

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.kimi_k3_nvfp4_native import (
    FP4_MLA_BYTES_PER_TOKEN,
    FP4_MLA_TOKENS_PER_BLOCK,
    quantize_kimi_k3_nvfp4_query,
    run_kimi_k3_nvfp4_attention,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec


class KimiK3NVFP4Metadata(MLACommonMetadata):
    state_indices: torch.Tensor | None = None


class KimiK3NVFP4MetadataBuilder(MLACommonMetadataBuilder[KimiK3NVFP4Metadata]):
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            metadata_cls=KimiK3NVFP4Metadata,
            supports_dcp_with_varlen=True,
        )
        # The native writer quantizes Q itself for decode; prefill continues to
        # use the model's BF16 flash-attention path for newly supplied tokens.
        self.q_data_type = torch.bfloat16

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KimiK3NVFP4Metadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        metadata.state_indices = common_attn_metadata.persistent_state_indices
        if metadata.state_indices is None:
            raise ValueError(
                "Kimi-K3 NVFP4 requires a hybrid recurrent-state cache group."
            )
        return metadata


class KimiK3NVFP4MLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4_kimi_k3"]

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        spec = super().customize_spec(spec)
        if spec.cache_dtype_str != "nvfp4_kimi_k3":
            return spec
        return replace(
            spec,
            state_content_bytes=FP4_MLA_BYTES_PER_TOKEN,
            storage_block_size=FP4_MLA_TOKENS_PER_BLOCK,
        )

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [FP4_MLA_TOKENS_PER_BLOCK]

    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_NVFP4_MLA"

    @staticmethod
    def get_impl_cls() -> type["KimiK3NVFP4MLAImpl"]:
        return KimiK3NVFP4MLAImpl

    @staticmethod
    def get_builder_cls() -> type[KimiK3NVFP4MetadataBuilder]:
        return KimiK3NVFP4MetadataBuilder

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
        if head_size != 576:
            return f"Kimi-K3 NVFP4 MLA requires head_size=576, got {head_size}."
        return None


class KimiK3NVFP4MLAImpl(MLACommonImpl[KimiK3NVFP4Metadata]):
    can_return_lse_for_decode = True
    supports_dcp = True
    lse_base_on_e = True

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
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "Kimi-K3 NVFP4 MLA does not support ALiBi, sliding-window "
                "attention, or logits soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Kimi-K3 NVFP4 supports decoder attention only.")
        if (
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.v_head_dim,
        ) != (512, 64, 512):
            raise ValueError(
                "Kimi-K3 NVFP4 requires MLA dimensions (512, 64, 512), got "
                f"({self.kv_lora_rank}, {self.qk_rope_head_dim}, "
                f"{self.v_head_dim})."
            )
        # The model's fused cache-update epilogue owns FP4 Q quantization.
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: KimiK3NVFP4Metadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert attn_metadata.decode is not None
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        q_fp4, q_sf = quantize_kimi_k3_nvfp4_query(self, q)
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decode_tokens % num_decodes:
            raise ValueError("Kimi-K3 NVFP4 requires uniform decode query lengths.")
        query_len = num_decode_tokens // num_decodes
        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        if not attn_metadata.causal and query_len > 1:
            block_table = block_table.repeat_interleave(query_len, dim=0)
            seq_lens = seq_lens.repeat_interleave(query_len)
            query_len = 1
        output = torch.empty(
            (num_decode_tokens, q_fp4.shape[0] // num_decode_tokens, 512),
            dtype=torch.bfloat16,
            device=q_fp4.device,
        )
        run_kimi_k3_nvfp4_attention(
            self,
            q_fp4=q_fp4,
            q_sf=q_sf,
            cache=kv_c_and_k_pe_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
            query_len_per_seq=query_len,
            sm_scale=self.scale,
        )
        lse = self._kimi_k3_nvfp4_max + torch.log(self._kimi_k3_nvfp4_denom)
        return output, lse if self.need_to_return_lse_for_decode else None
