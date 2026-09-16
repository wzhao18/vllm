#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke/latency harness for TRT-LLM MR 10576 FP4 MLA decode."""

import argparse
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import torch


def swizzled_sf_offset(row: int, col: int, sf_per_row: int) -> int:
    padded_cols = ((sf_per_row + 3) // 4) * 4
    return (
        col % 4
        + (col // 4) * (4 * 128)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded_cols)
    )


def dequant_fp4(fp4_tensor, sf_tensor, logical_dim: int, global_scale: float):
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=fp4_tensor.device,
    )
    packed_rows = fp4_tensor.view(torch.uint8).reshape(-1, logical_dim // 2)
    sf_per_row = logical_dim // 16
    sf_flat = sf_tensor.view(torch.float8_e4m3fn).reshape(-1)
    out = torch.empty((packed_rows.shape[0], logical_dim), device=fp4_tensor.device)
    for row in range(packed_rows.shape[0]):
        for col in range(sf_per_row):
            packed = packed_rows[row, col * 8 : (col + 1) * 8]
            low, high = packed & 0x0F, (packed >> 4) & 0x0F
            vals = torch.empty(16, device=fp4_tensor.device)
            vals[0::2] = table[(low & 7).long()] * torch.where(
                (low & 8) != 0, -1.0, 1.0
            )
            vals[1::2] = table[(high & 7).long()] * torch.where(
                (high & 8) != 0, -1.0, 1.0
            )
            scale = sf_flat[swizzled_sf_offset(row, col, sf_per_row)].float()
            out[row, col * 16 : (col + 1) * 16] = vals * scale / global_scale
    return out


def validate_output(
    fp4, metadata, packed_q, q_sf, output, context: int, sm_scale: float
):
    if metadata.num_seqs != 1 or context != 128:
        raise ValueError("validation currently requires batch=1 and context=128")
    manager = metadata.kv_cache_manager
    storage = dequant_fp4(
        manager.kv[0, 0, :, 0, :], manager.sf[0], 640, fp4.FP4_MLA_KV_GLOBAL_SCALE
    )
    q = dequant_fp4(packed_q, q_sf, 640, fp4.FP4_MLA_Q_GLOBAL_SCALE)
    q_tail = q[:, 512:].reshape(q.shape[0], 4, 2, 16)
    q_main = q_tail[:, :, 0, :].reshape(q.shape[0], 64)
    q_residual = q_tail[:, :, 1, :].reshape(q.shape[0], 64)
    logical_q = torch.cat((q[:, :512], q_main, q_residual, q_main), dim=-1)
    logical_k = torch.cat(
        (
            storage[:, :512],
            storage[:, 512:576],
            storage[:, 512:576],
            storage[:, 576:640],
        ),
        dim=-1,
    )
    probs = torch.softmax(logical_q @ logical_k.T * sm_scale, dim=-1)
    p_dequant = dequant_fp4(
        metadata._fp4_mla_attention_p_buf,
        metadata._fp4_mla_attention_p_sf_buf,
        128,
        fp4.FP4_MLA_P_GLOBAL_SCALE,
    )
    reference = p_dequant @ storage[:, :512]
    actual = output[0].float()
    error = actual - reference
    return {
        "p_max_abs": float((p_dequant - probs).abs().max().item()),
        "output_max_abs": float(error.abs().max().item()),
        "output_rel_l2": float(
            torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)
        ),
        "output_cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            )
        ),
    }


def load_backend(source_dir: str):
    pkg_name = "tensorrt_llm._torch.attention_backend.fp4_mla"
    for name in (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.attention_backend",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    utils = types.ModuleType("tensorrt_llm._utils")
    utils.get_sm_version = (
        lambda: torch.cuda.get_device_capability()[0] * 10
        + torch.cuda.get_device_capability()[1]
    )
    utils.prefer_pinned = lambda: False
    sys.modules[utils.__name__] = utils

    bindings = types.ModuleType("tensorrt_llm.bindings")
    bindings.DataType = SimpleNamespace(NVFP4="nvfp4")
    sys.modules[bindings.__name__] = bindings

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        os.path.join(source_dir, "__init__.py"),
        submodule_search_locations=[source_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_vllm_adapter(source_dir: str):
    import triton

    package_name = "vllm_native_fp4_mla"
    package = types.ModuleType(package_name)
    package.__path__ = [source_dir]
    sys.modules[package_name] = package
    vllm = types.ModuleType("vllm")
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = triton
    sys.modules["vllm"] = vllm
    sys.modules["vllm.triton_utils"] = triton_utils
    module_name = f"{package_name}.kimi_k3_nvfp4_native"
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(source_dir, "kimi_k3_nvfp4_native.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CacheManager:
    def __init__(self, kv, sf, v_sf):
        self.kv = kv
        self.sf = sf
        self.v_sf = v_sf

    def get_fp4_mla_cache_buffers(self, _layer_idx):
        return self.kv, self.sf

    def get_mla_v_packed_pool(self, _local_layer):
        return None

    def get_mla_v_packed_pool_base(self):
        return None

    def get_mla_v_scale_pool_base(self):
        return self.v_sf.view(torch.uint8).flatten(0, 1)


def build_case(fp4, batch: int, heads: int, context: int, *, opaque_vllm_cache: bool):
    device = torch.device("cuda")
    page = fp4.FP4_MLA_TOKENS_PER_BLOCK
    pages_per_seq = (context + page - 1) // page
    total_pages = batch * pages_per_seq
    storage_dim = 640

    # Synthetic but structurally valid native FP4 cache. Nibbles 0..7 avoid
    # sign-heavy pathological data; positive unit scales keep outputs finite.
    low = torch.randint(
        0,
        8,
        (total_pages, 1, page, 1, storage_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    high = torch.randint(0, 8, low.shape, device=device, dtype=torch.uint8)
    packed = low | (high << 4)
    v_sf_page_bytes = fp4.get_fp4_mla_v_scale_pool_size(512, page)
    if opaque_vllm_cache:
        # vLLM allocates one opaque [B, H=1, N=128, C=392] byte page.
        # Native consumers reinterpret three page-major regions without copies.
        bytes_per_token = storage_dim // 2 + storage_dim // 16 + 32
        raw = torch.empty(
            (total_pages, 1, page, bytes_per_token),
            device=device,
            dtype=torch.uint8,
        )
        page_stride = raw.stride(0)
        base = raw.storage_offset()
        data_bytes = page * (storage_dim // 2)
        sf_bytes = page * (storage_dim // 16)
        kv = torch.as_strided(
            raw,
            (total_pages, 1, page, 1, storage_dim // 2),
            (page_stride, data_bytes, storage_dim // 2, storage_dim // 2, 1),
            base,
        )
        sf = torch.as_strided(
            raw,
            (total_pages, page, storage_dim // 16),
            (page_stride, storage_dim // 16, 1),
            base + data_bytes,
        ).view(torch.float8_e4m3fn)
        v_sf = torch.as_strided(
            raw,
            (1, total_pages, v_sf_page_bytes),
            (total_pages * page_stride, page_stride, 1),
            base + data_bytes + sf_bytes,
        ).view(torch.float8_e4m3fn)
        kv.copy_(packed)
        sf.fill_(1.0)
        v_sf.fill_(1.0)
    else:
        kv = packed
        sf = torch.ones(
            (total_pages, page, storage_dim // 16),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        v_sf = torch.ones(
            (1, total_pages, v_sf_page_bytes),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
    manager = CacheManager(kv, sf, v_sf)

    page_ids = torch.arange(total_pages, device=device, dtype=torch.int32)
    indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * pages_per_seq
    kv_lens = torch.full((batch,), context, device=device, dtype=torch.int32)
    append_lens = torch.ones((batch,), device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        fp4_mla_v_scale_pool=v_sf,
        page_size=page,
        num_contexts=0,
        num_seqs=batch,
        num_tokens=batch,
        num_ctx_tokens=0,
        fp4_mla_page_table_stride=pages_per_seq,
        _fp4_mla_device_page_table=True,
        _fp4_mla_device_page_table_valid=True,
        _paged_kv_indices=page_ids,
        paged_kv_indices=page_ids,
        paged_kv_indptr_decode=indptr,
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=append_lens,
        fp4_mla_generation_kv_lens=kv_lens.clone(),
        fp4_mla_generation_append_lens=append_lens.clone(),
        fp4_mla_generation_lengths_num_tokens=batch,
        fp4_mla_generation_lengths_num_seqs=batch,
        fp4_mla_generation_lengths_num_contexts=0,
        _fp4_mla_generation_lengths_capture_recorded=False,
        _fp4_mla_kv_global_scale=torch.tensor(
            [fp4.FP4_MLA_KV_GLOBAL_SCALE], device=device
        ),
        _fp4_mla_q_global_scale=torch.tensor(
            [fp4.FP4_MLA_Q_GLOBAL_SCALE], device=device
        ),
        is_cuda_graph=False,
    )

    q = torch.zeros((batch, heads, 576), device=device, dtype=torch.bfloat16)
    packed_q = torch.randint(
        0, 256, (batch * heads, 320), device=device, dtype=torch.uint8
    )
    q_sf_size = fp4._get_fp4_mla_swizzled_scale_size(batch * heads, 640)
    q_sf = torch.ones((q_sf_size,), device=device, dtype=torch.float8_e4m3fn)
    output = torch.empty((batch, heads, 512), device=device, dtype=torch.bfloat16)
    return metadata, q, packed_q, q_sf, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--opaque-vllm-cache", action="store_true")
    parser.add_argument("--validate-vllm-adapter", action="store_true")
    parser.add_argument("--validate-cache-update", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "triton")
    os.environ.setdefault("TRTLLM_FP4_MLA_TRITON_PREPACK_V", "0")
    fp4 = load_backend(args.source_dir)
    if os.environ["TRTLLM_FP4_MLA_ATTENTION_BACKEND"] == "cutedsl":
        # The MR checks for a legacy top-level ``ctm`` module even though its
        # kernel imports CUTLASS DSL as ``ctm``. The standalone environment
        # intentionally installs only the current nvidia-cutlass-dsl package.
        fp4._cutedsl_backend_available = lambda: True
        import cutlass.cute as cute

        if not hasattr(cute.nvgpu, "cfence"):
            # Diagnostic compatibility shim for public CUTLASS DSL 4.8. The
            # MR was authored against an internal CTM exposing this compiler
            # fence. Numerical validation is mandatory before this can be
            # considered a valid SM103 port.
            cute.nvgpu.cfence = lambda: None
        if not hasattr(cute.nvgpu, "warp_switch"):
            # Another Rubin CTM scheduling hint absent from the public DSL.
            cute.nvgpu.warp_switch = lambda: None
    metadata, q, packed_q, q_sf, output = build_case(
        fp4,
        args.batch,
        args.heads,
        args.context,
        opaque_vllm_cache=args.opaque_vllm_cache,
    )

    def run():
        fp4.run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q=q,
            output=output,
            sm_scale=0.1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            prequantized_q=packed_q,
            prequantized_q_sf=q_sf,
            q_batch_capacity=args.batch,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / args.iters
    result = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "batch": args.batch,
        "heads": args.heads,
        "layout": "vllm-opaque" if args.opaque_vllm_cache else "split",
        "context": args.context,
        "latency_ms": ms,
        "finite": bool(torch.isfinite(output).all().item()),
        "output_norm": float(torch.linalg.vector_norm(output.float()).item()),
    }
    if args.validate:
        result.update(
            validate_output(fp4, metadata, packed_q, q_sf, output, args.context, 0.1)
        )
    if args.validate_vllm_adapter:
        if not args.opaque_vllm_cache:
            raise ValueError("--validate-vllm-adapter requires --opaque-vllm-cache")
        adapter_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "vllm",
            "v1",
            "attention",
            "backends",
            "mla",
        )
        adapter = load_vllm_adapter(adapter_dir)
        query = torch.randn(
            (args.batch, args.heads, 576),
            dtype=torch.bfloat16,
            device=output.device,
        )
        adapter_q, adapter_q_sf = adapter.quantize_kimi_k3_nvfp4_query(
            SimpleNamespace(), query
        )
        dequant_q = dequant_fp4(
            adapter_q,
            adapter_q_sf,
            640,
            adapter.FP4_MLA_Q_GLOBAL_SCALE,
        )
        physical_tail = dequant_q[:, 512:].reshape(-1, 4, 2, 16)
        tail_main = physical_tail[:, :, 0, :].reshape(-1, 64)
        tail_residual = physical_tail[:, :, 1, :].reshape(-1, 64)
        query_2d = query.float().view(-1, 576)
        restored_q = torch.cat((dequant_q[:, :512], tail_main + tail_residual), dim=-1)
        query_error = restored_q - query_2d
        result["q_quant_rel_l2"] = float(
            torch.linalg.vector_norm(query_error)
            / torch.linalg.vector_norm(query.float())
        )
        result["q_prefix_quant_rel_l2"] = float(
            torch.linalg.vector_norm(query_error[:, :512])
            / torch.linalg.vector_norm(query_2d[:, :512])
        )
        result["q_rope_main_rel_l2"] = float(
            torch.linalg.vector_norm(tail_main - query_2d[:, 512:])
            / torch.linalg.vector_norm(query_2d[:, 512:])
        )
        result["q_rope_residual_norm"] = float(
            torch.linalg.vector_norm(tail_residual)
            / torch.linalg.vector_norm(query_2d[:, 512:])
        )
        result["q_rope_quant_rel_l2"] = float(
            torch.linalg.vector_norm(query_error[:, 512:])
            / torch.linalg.vector_norm(query.float().view(-1, 576)[:, 512:])
        )
        manager = metadata.kv_cache_manager
        raw = torch.as_strided(
            manager.kv,
            (
                manager.kv.shape[0],
                1,
                fp4.FP4_MLA_TOKENS_PER_BLOCK,
                adapter.FP4_MLA_BYTES_PER_TOKEN,
            ),
            (
                manager.kv.stride(0),
                manager.kv.stride(1),
                adapter.FP4_MLA_BYTES_PER_TOKEN,
                1,
            ),
            manager.kv.storage_offset(),
        )
        adapter_output = torch.empty_like(output)
        adapter_owner = SimpleNamespace()
        pages_per_seq = (args.context + fp4.FP4_MLA_TOKENS_PER_BLOCK - 1) // (
            fp4.FP4_MLA_TOKENS_PER_BLOCK
        )
        adapter.run_kimi_k3_nvfp4_attention(
            adapter_owner,
            q_fp4=packed_q,
            q_sf=q_sf,
            cache=raw,
            block_table=metadata._paged_kv_indices.view(args.batch, pages_per_seq),
            seq_lens=metadata.kv_lens_cuda_runtime,
            output=adapter_output,
            query_len_per_seq=1,
            sm_scale=0.1,
        )
        torch.cuda.synchronize()
        adapter_error = adapter_output.float() - output.float()
        result["adapter_max_abs"] = float(adapter_error.abs().max().item())
        result["adapter_rel_l2"] = float(
            torch.linalg.vector_norm(adapter_error)
            / torch.linalg.vector_norm(output.float())
        )
        if args.validate_cache_update:
            update_cache = torch.zeros(
                (
                    1,
                    1,
                    fp4.FP4_MLA_TOKENS_PER_BLOCK,
                    adapter.FP4_MLA_BYTES_PER_TOKEN,
                ),
                dtype=torch.uint8,
                device=output.device,
            )
            update_latent = torch.randn(
                (1, 576), dtype=torch.bfloat16, device=output.device
            )
            update_q_pe = torch.randn(
                (1, args.heads, 64), dtype=torch.bfloat16, device=output.device
            )
            update_owner = SimpleNamespace()
            updated_q_pe = adapter.update_kimi_k3_nvfp4_decode_cache(
                update_owner,
                latent=update_latent,
                q_pe=update_q_pe,
                cache=update_cache,
                block_table=torch.zeros(
                    (1, 1), dtype=torch.int32, device=output.device
                ),
                seq_lens=torch.ones(1, dtype=torch.int32, device=output.device),
                state_indices=torch.zeros(1, dtype=torch.int32, device=output.device),
                slot_mapping=torch.zeros(1, dtype=torch.int64, device=output.device),
                positions=None,
                query_len_per_seq=1,
                rotary_cos_sin=None,
                max_state_slots=1,
                max_rewind=0,
            )
            torch.cuda.synchronize()
            update_kv, update_sf, _ = adapter.split_kimi_k3_nvfp4_cache(update_cache)
            update_dequant = dequant_fp4(
                update_kv[0, 0, 0],
                update_sf[0],
                640,
                adapter.FP4_MLA_KV_GLOBAL_SCALE,
            )[0]
            restored_k = torch.cat(
                (
                    update_dequant[:512],
                    update_dequant[512:576] + update_dequant[576:],
                )
            )
            update_error = restored_k - update_latent[0].float()
            result["cache_update_rel_l2"] = float(
                torch.linalg.vector_norm(update_error)
                / torch.linalg.vector_norm(update_latent.float())
            )
            result["cache_update_norm_ratio"] = float(
                torch.linalg.vector_norm(restored_k)
                / torch.linalg.vector_norm(update_latent.float())
            )
            result["cache_update_nonzero_bytes"] = int(
                torch.count_nonzero(update_cache).item()
            )
            result["cache_update_q_exact"] = bool(
                torch.equal(updated_q_pe, update_q_pe)
            )
            dcp_cache = torch.zeros_like(update_cache)
            dcp_latent = torch.randn(
                (6, 576), dtype=torch.bfloat16, device=output.device
            )
            adapter.update_kimi_k3_nvfp4_decode_cache(
                SimpleNamespace(),
                latent=dcp_latent,
                q_pe=torch.randn(
                    (6, args.heads, 64),
                    dtype=torch.bfloat16,
                    device=output.device,
                ),
                cache=dcp_cache,
                block_table=torch.zeros(
                    (1, 1), dtype=torch.int32, device=output.device
                ),
                seq_lens=torch.tensor([3], dtype=torch.int32, device=output.device),
                state_indices=torch.zeros(1, dtype=torch.int32, device=output.device),
                slot_mapping=torch.tensor(
                    [-1, -1, -1, 0, 1, 2],
                    dtype=torch.int64,
                    device=output.device,
                ),
                positions=None,
                query_len_per_seq=6,
                rotary_cos_sin=None,
                max_state_slots=1,
                max_rewind=5,
            )
            torch.cuda.synchronize()
            dcp_kv, dcp_sf, _ = adapter.split_kimi_k3_nvfp4_cache(dcp_cache)
            dcp_dequant = dequant_fp4(
                dcp_kv[0, 0, 0],
                dcp_sf[0],
                640,
                adapter.FP4_MLA_KV_GLOBAL_SCALE,
            )[0]
            restored_dcp = torch.cat(
                (dcp_dequant[:512], dcp_dequant[512:576] + dcp_dequant[576:])
            )
            result["dcp_offset_update_rel_l2"] = float(
                torch.linalg.vector_norm(restored_dcp - dcp_latent[3].float())
                / torch.linalg.vector_norm(dcp_latent[3].float())
            )
            prefill_cache = torch.zeros_like(update_cache)
            prefill_latent = torch.randn(
                (16, 576), dtype=torch.bfloat16, device=output.device
            )
            prefill_query = torch.randn(
                (16, args.heads, 192), dtype=torch.bfloat16, device=output.device
            )
            adapter.update_kimi_k3_nvfp4_prefill_cache(
                SimpleNamespace(),
                latent=prefill_latent,
                query=prefill_query,
                cache=prefill_cache,
                slot_mapping=torch.arange(16, dtype=torch.int64, device=output.device),
                positions=torch.arange(16, dtype=torch.int64, device=output.device),
                query_start_loc=torch.tensor(
                    [0, 16], dtype=torch.int32, device=output.device
                ),
                state_indices=torch.zeros(1, dtype=torch.int32, device=output.device),
                rotary_cos_sin=None,
                max_state_slots=1,
                max_rewind=0,
            )
            torch.cuda.synchronize()
            prefill_kv, prefill_sf, _ = adapter.split_kimi_k3_nvfp4_cache(prefill_cache)
            prefill_dequant = dequant_fp4(
                prefill_kv[0, 0, 0],
                prefill_sf[0],
                640,
                adapter.FP4_MLA_KV_GLOBAL_SCALE,
            )[0]
            restored_prefill = torch.cat(
                (
                    prefill_dequant[:512],
                    prefill_dequant[512:576] + prefill_dequant[576:],
                )
            )
            result["prefill_update_rel_l2_token0"] = float(
                torch.linalg.vector_norm(restored_prefill - prefill_latent[0].float())
                / torch.linalg.vector_norm(prefill_latent[0].float())
            )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
