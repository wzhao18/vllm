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


def build_case(fp4, batch: int, heads: int, context: int):
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
    kv = low | (high << 4)
    sf = torch.ones(
        (total_pages, page, storage_dim // 16), device=device, dtype=torch.float8_e4m3fn
    )
    v_sf = torch.ones(
        (1, total_pages, fp4.get_fp4_mla_v_scale_pool_size(512, page)),
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
        fp4, args.batch, args.heads, args.context
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
        "context": args.context,
        "latency_ms": ms,
        "finite": bool(torch.isfinite(output).all().item()),
        "output_norm": float(torch.linalg.vector_norm(output.float()).item()),
    }
    if args.validate:
        result.update(
            validate_output(fp4, metadata, packed_q, q_sf, output, args.context, 0.1)
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
