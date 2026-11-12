# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_kda_layer(
    worker: Worker, module: str = "vllm.models.kimi_k3.nvidia.kda"
) -> KimiK3DeltaAttention | None:
    # Kimi model construction already imports kda. Avoid importing it here.
    kda = sys.modules.get(module)
    if kda is None:
        return None

    compilation_config = getattr(
        worker.model_runner,
        "compilation_config",
        None,
    )
    static_context = getattr(compilation_config, "static_forward_context", None)
    if not isinstance(static_context, dict):
        return None
    return next(
        (
            layer
            for layer in static_context.values()
            if isinstance(layer, kda.KimiK3DeltaAttention)
        ),
        None,
    )


def _warm_attn_res(worker: Worker) -> None:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import (
        attn_res,
        get_attn_res_triton_warmup_profiles,
    )

    config = worker.model_config.hf_text_config
    block_size = getattr(config, "attn_res_block_size", None)
    if block_size is None:
        return

    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 2:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros(
        (1, max_blocks, hidden_size),
        dtype=dtype,
        device=device,
    )
    norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)
    qk_weight = torch.zeros_like(norm_weight)
    output_norm_weight = torch.zeros_like(norm_weight)

    for (
        num_blocks,
        has_delta,
        block_write_idx,
        apply_output_norm,
    ) in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta if has_delta else None,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight if apply_output_norm else None,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=eps if apply_output_norm else 0.0,
        )


def _warm_recurrent_kda(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda,
        get_fused_recurrent_kda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_kda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 speculative KDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_kda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


def _warm_gluon_chunk_kda(layer, input_dtype: torch.dtype) -> None:
    from vllm.models.kimi_k3.amd.ops.kda_prefill import (
        chunk_kda_prefill,
        get_gluon_chunk_kda_warmup_profiles,
    )

    if layer.kda_prefill_backend != "gluon":
        return
    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 Gluon chunk KDA prefill kernels.")
    h, d = int(layer.local_num_heads), int(layer.head_dim)
    for num_seqs, seq_len in get_gluon_chunk_kda_warmup_profiles():
        num_tokens = num_seqs * seq_len
        shape = (1, num_tokens, h, d)
        chunk_kda_prefill(
            q=torch.zeros(shape, dtype=input_dtype, device=state.device),
            k=torch.zeros(shape, dtype=input_dtype, device=state.device),
            v=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_g=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_beta=torch.zeros(
                (1, num_tokens, h), dtype=input_dtype, device=state.device
            ),
            A_log=layer.A_log,
            g_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            state_cache=state,
            state_indices=torch.zeros(num_seqs, dtype=torch.int32, device=state.device),
            has_initial_state=torch.zeros(
                num_seqs, dtype=torch.bool, device=state.device
            ),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.arange(
                0, num_tokens + 1, seq_len, dtype=torch.int32, device=state.device
            ),
            backend="gluon",
            out=torch.empty(shape, dtype=input_dtype, device=state.device),
        )


def _warm_prefill_glue(worker: Worker, layer: KimiK3DeltaAttention) -> None:
    """Prefill-only kernels that dummy runs never reach (KDA's _forward returns
    without attention metadata), once each at the serving strides, so no
    Triton kernel JIT-compiles during inference:
    - KDA: short conv (generic and tile), output gate norm (every TILES
      variant, prefill and projection-overlap gate strides), initial-state
      gather, checkpoint export, segment-scan chain;
    Cache line 0 is the null block: the state kernels skip it (no writes)."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        gather_initial_states,
    )
    from vllm.models.kimi_k3.nvidia.ops import (
        kda_conv,
        kda_norm,
        kda_segscan,
    )
    from vllm.v1.attention.backends.utils import (
        NULL_BLOCK_ID,
        compute_causal_conv1d_metadata,
    )

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    conv_state, rec_state = kv_cache[0], kv_cache[1]
    if not isinstance(conv_state, torch.Tensor) or not conv_state.numel():
        return
    device = conv_state.device
    dtype = worker.model_config.dtype
    P = int(layer.local_projection_size)
    H = int(layer.local_num_heads)
    D = int(layer.head_dim)
    # Row stride of the prefill projection output (KDA forward's split sizes).
    width = 4 * P + D + H + int(getattr(layer, "in_proj_padding", 0) or 0)
    logger.info("Warming up Kimi-K3 prefill glue kernels.")

    # KDA short conv: q / k / v column slices of the projection (row stride
    # width), the layer's conv cache views, one 16-token sequence.
    T = 16
    proj = torch.zeros((T, width), dtype=dtype, device=device)
    qsl_cpu = torch.tensor([0, T], dtype=torch.int32)
    nums_dict, batch_ptr, offs = compute_causal_conv1d_metadata(qsl_cpu, device=device)
    md = SimpleNamespace(
        nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offs
    )
    qsl = qsl_cpu.to(device)
    null = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    no_init = torch.zeros(1, dtype=torch.bool, device=device)
    w = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    for x, wp, st in zip(
        proj[:, : 3 * P].split(P, dim=-1),
        w.split(P, dim=0),
        conv_state.split(P, dim=-2),
    ):
        xt = x.transpose(0, 1)
        causal_conv1d_fn(
            xt,
            wp,
            None,
            activation="silu",
            conv_states=st,
            has_initial_state=no_init,
            cache_indices=null,
            query_start_loc=qsl,
            metadata=md,
        )
        if kda_conv.supported(xt, wp, st, md):
            kda_conv.kda_prefill_conv(xt, wp, st, qsl, null, no_init, md)
    # The fused Q|K|V conv (CUDA): builds / loads the extension here, not in
    # the first prefill.
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    cs = conv_state if is_conv_state_dim_first() else conv_state.transpose(-1, -2)
    if kda_conv.qkv_conv_supported(proj[:, : 3 * P], w, cs, P):
        kda_conv.qkv_conv(proj[:, : 3 * P], w, cs, qsl, null, no_init, P)

    # Output gate norm: the gate is a view of the prefill projection (row
    # stride width) or of the decode projection-overlap output (4 P).
    for stride in sorted({width, 4 * P}):
        kda_norm.warmup(H, stride, layer.o_norm.weight, float(layer.o_norm.eps))

    gather_initial_states(rec_state, null, no_init)
    exporter = getattr(layer, "_checkpoint_exporter", None)
    if exporter is not None:
        ck = torch.empty((1, H, D, D), dtype=rec_state.dtype, device=device)
        exporter.export(
            SimpleNamespace(
                checkpoint_offsets=torch.zeros(1, dtype=torch.int32, device=device),
                state_indices=null,
            ),
            raw_qkv=proj[:, : 3 * P],
            conv_state=conv_state,
            recurrent_checkpoint=ck,
            recurrent_state=rec_state,
            cu_seqlens=qsl,
        )
    bufs = getattr(layer, "_segscan_bufs", None)
    if bufs is not None:
        init = torch.zeros((1, H, D, D), dtype=rec_state.dtype, device=device)
        plan = SimpleNamespace(first=null * 0, count=null * 0 + 1, num_seqs=1)
        kda_segscan.chain_scan(init, bufs.M, bufs.C, bufs.X, plan)  # type: ignore[arg-type]

    torch.accelerator.synchronize()


class _KeyOnly(Exception):
    pass


def _warm_stock_mla_decode(worker: Worker) -> None:
    """Compile tokenspeed's generic MLA decode kernel (TOKENSPEED_MLA ``forward_mqa``)
    for every compile key an eager step can reach. The specialized decode covers the
    captured steady steps; the stock path still serves the decode rows of eager and
    mixed steps (target), the drafter's uncaptured steps, and prompts or chunk tails
    of at most q tokens (MLA treats them as decodes, at their own q_len). Its key
    (``mla_decode._get_compiled_mla_kernel``) depends on the batch and max_seq_len
    through the split count (workspace or not, the reducer's D tiles and max splits),
    so each layer shape walks batch x split-defining lengths with the compile
    stubbed to record the key, then compiles each distinct key once."""
    from tokenspeed_mla import mla_decode as md
    from tokenspeed_mla import tokenspeed_mla_decode

    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLAImpl,
        _get_workspace,
        sm107_warmup,
    )

    config = worker.vllm_config
    context = config.compilation_config.static_forward_context
    spec = config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    # past ~SMs / 2 requests tokenspeed's split is 1 and the reducer's tiles settle
    max_reqs = min(config.scheduler_config.max_num_seqs, 128)
    speculator = getattr(worker.model_runner, "speculator", None)
    draft_names = set(getattr(speculator, "draft_attn_layer_names", ()) or ())
    draft_q = getattr(speculator, "num_query_per_req", num_spec + 1)
    draft_causal = not getattr(speculator, "requires_non_causal", False)
    interleave = config.parallel_config.cp_kv_cache_interleave_size
    device = torch.device("cuda", torch.accelerator.current_device_index())

    # layer shape -> (q_len -> request counts), the impl and its cache
    shapes: dict = {}
    for name, layer in context.items():
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, TokenspeedMLAImpl):
            continue
        read = getattr(layer, "_attn_read_kv_cache", None)
        cache = read() if callable(read) else getattr(layer, "kv_cache", None)
        if isinstance(cache, (list, tuple)):
            cache = cache[0] if cache else None
        if cache is None or cache.numel() == 0:
            continue
        if cache.dtype == torch.uint8:
            cache = cache.view(torch.float8_e4m3fn)
        heads = impl.num_heads * max(impl.dcp_world_size, 1)  # DCP gathers q heads
        if name in draft_names:
            causal = (draft_causal,)
            qlens = {q: range(1, max_reqs + 1) for q in {draft_q - 1, draft_q} if q > 0}
        else:
            causal = (True,)
            # steady width at every batch; shorter "decodes" (short prompts, chunk
            # tails) at a few requests
            qlens = {q: range(1, 9) for q in range(1, num_spec + 1)}
            qlens[num_spec + 1] = range(1, max_reqs + 1)
        key: tuple = (heads, cache.shape[1], causal, impl.need_to_return_lse_for_decode)
        key += (impl.dcp_world_size, impl.dcp_rank, impl._min_split_kv)
        shapes.setdefault(key, (qlens, impl, cache))

    keys: dict = {}

    def record(**kwargs):
        keys.setdefault(tuple(kwargs.items()), kwargs)
        raise _KeyOnly

    compile_fn = md._get_compiled_mla_kernel
    md._get_compiled_mla_kernel = record
    try:
        for (heads, _, causal, lse, cp, _, min_split), (
            qlens,
            impl,
            cache,
        ) in shapes.items():
            workspace = _get_workspace(
                device, heads, impl.kv_lora_rank, min_split, impl._max_decode_tokens
            )
            i32 = dict(dtype=torch.int32, device=device)
            for q_len, batches in qlens.items():
                top = max(batches)
                query = torch.zeros(
                    (top, q_len, heads, cache.shape[-1]),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                tables = torch.zeros((top, 1), **i32)
                lens = torch.ones(top, **i32)
                for b in batches:
                    # max_seq_len sets the split through its 128-key tiles: 1..64
                    # tiles reach every split count tokenspeed can pick
                    for tiles in range(1, 65):
                        for c in causal:
                            try:
                                tokenspeed_mla_decode(
                                    query=query[:b],
                                    kv_cache=cache,
                                    workspace_buffer=workspace,
                                    kv_lora_rank=impl.kv_lora_rank,
                                    qk_rope_head_dim=impl.qk_rope_head_dim,
                                    block_tables=tables[:b],
                                    seq_lens=lens[:b],
                                    max_seq_len=128 * tiles,
                                    softmax_scale=1.0,
                                    output_scale=1.0,
                                    enable_pdl=False,
                                    return_lse=lse,
                                    causal_mask=c,
                                    causal_seqs=lens[:b] if cp > 1 else None,
                                    cp_world=cp,
                                    cp_rank=impl.dcp_rank,
                                    cp_interleave_size=interleave,
                                    min_split_kv=min_split,
                                )
                            except _KeyOnly:
                                pass
                            except (AssertionError, ValueError):
                                pass  # beyond the serving workspace: unreachable
    finally:
        md._get_compiled_mla_kernel = compile_fn
    for kwargs in keys.values():
        compile_fn(**kwargs)
    if keys:
        logger.info(
            "Kimi-K3: compiled %d tokenspeed MLA decode keys for %d layer shapes.",
            len(keys),
            len(shapes),
        )
    # SM107 M256 decode keys (TokenspeedMLAImpl._forward_sm107), per layer shape
    sm107 = 0
    for (heads, page, causal, lse, cp, _, _), (qlens, impl, cache) in shapes.items():
        if not lse and cp == 1:
            sm107 += sm107_warmup(
                impl, page, set(qlens), causal, max_reqs, config.model_config.max_model_len
            )
    if sm107:
        logger.info("Kimi-K3: compiled %d tokenspeed SM107 MLA decode keys.", sm107)


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if current_platform.is_rocm():
        layer = _get_kda_layer(worker, "vllm.models.kimi_k3.amd.kda")
        if layer is not None:
            _warm_gluon_chunk_kda(layer, worker.model_config.dtype)
        return
    if not current_platform.is_cuda():
        return

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    _warm_attn_res(worker)
    _warm_recurrent_kda(layer, worker.model_config.dtype)
    try:
        _warm_prefill_glue(worker, layer)
    except Exception:  # never block startup; the JIT monitor will flag misses
        logger.warning("Kimi-K3 prefill warmup failed.", exc_info=True)
    try:
        _warm_stock_mla_decode(worker)
    except Exception:  # never block startup; the JIT monitor will flag misses
        logger.warning("Kimi-K3 stock MLA decode warmup failed.", exc_info=True)
