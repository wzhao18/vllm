# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import math
import os
from datetime import datetime

import flashinfer
import torch

from vllm.utils.math_utils import round_up

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


# FP4 (NVFP4 / E2M1) helpers, adapted from flashinfer/tests/test_helpers/utils_fp4.py
_FLOAT4_E2M1_MAX = 6.0
_E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _cast_from_fp4(x):
    """Unpack paired uint8 -> float32. Layout: [high_nibble | low_nibble]."""
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    new_shape = c.shape[:-2] + (c.shape[-2] * c.shape[-1],)
    lut = torch.tensor(_E2M1_TO_FLOAT32, device=c.device)
    return lut[c.to(torch.long)].reshape(new_shape).to(torch.float32)


def _cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _ref_fp4_quant(x, global_scale, block_size):
    """Reference NVFP4 quantization: returns (quantized_float32, scale_fp8_as_float32)."""
    sliced = x.reshape(x.shape[:-1] + (x.shape[-1] // block_size, block_size))
    vec_max = sliced.float().abs().max(dim=-1, keepdim=True)[0]
    scale = (global_scale * vec_max / _FLOAT4_E2M1_MAX).to(torch.float8_e4m3fn).float()
    inv_scale = torch.where(
        scale == 0, torch.zeros_like(scale), 1.0 / scale / global_scale
    )
    scaled = (sliced.float() * inv_scale).clamp(-6.0, 6.0).reshape(x.shape)
    return _cast_to_fp4(scaled), scale.squeeze(-1)


def _recover_swizzled_scales(scale, m, n, block_size, sf_start_index=0):
    """Recover swizzled NVFP4 scale factors to linear layout."""
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    full_m = scale.shape[0]
    tmp = scale.reshape(1, full_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    result = tmp.reshape(full_m, rounded_n).float()
    return result[sf_start_index : sf_start_index + m, :scale_n]


@torch.no_grad()
def benchmark_decode(
    dtype: torch.dtype,
    quant_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None],
    batch_size: int,
    max_seq_len: int,
    num_heads: tuple[int, int] = (64, 8),
    head_size: int = 128,
    kv_layout: str = "HND",
    block_size: int = 16,
    max_q_len: int | None = None,
    pad_to_power_of_2: bool = False,
    warmup: int = 10,
    trials: int = 20,
):
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtypes
    q_quant_dtype = q_quant_dtype or dtype
    kv_quant_dtype = kv_quant_dtype or dtype
    o_quant_dtype = o_quant_dtype or dtype

    num_qo_heads, num_kv_heads = num_heads
    assert num_qo_heads % num_kv_heads == 0

    sm_scale = float(1.0 / (head_size**0.5))

    # large number to reduce kv_cache reuse
    NUM_BLOCKS = int(256000 / block_size)

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    # Always using 1.0 scale to reflect the real perf in benchmarking
    q_scale = 1.0

    # KV sequence lengths (existing context tokens in cache)
    kv_lens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)
    kv_lens[-1] = max_seq_len

    if max_q_len is not None and max_q_len > 1:
        # Speculative decoding: variable query lengths per request
        q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
        q_lens[-1] = max_q_len
        seq_lens = (kv_lens + q_lens).to(torch.int32)
        q_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                q_lens.cumsum(0, dtype=torch.int32),
            ]
        )
        total_q = int(q_lens.sum().item())
    else:
        seq_lens = kv_lens
        q_indptr = None
        total_q = batch_size

    max_seq_len = torch.max(seq_lens).item()

    ref_query = torch.randn(total_q, num_qo_heads, head_size, dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, q_inv_scale = to_float8(ref_query)
    else:
        query, q_inv_scale = ref_query, None

    # Always using 1.0 scale to reflect the real perf in benchmarking
    k_scale = v_scale = 1.0
    ref_kv_cache = torch.randn(kv_cache_shape, dtype=dtype)
    if kv_quant_dtype == FP8_DTYPE:
        kv_cache, kv_inv_scale = to_float8(ref_kv_cache)
    else:
        kv_cache, kv_inv_scale = ref_kv_cache, None

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(batch_size):
        seq_len = seq_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)
    workspace_buffer = torch.zeros(1024 * 1024 * 1024, dtype=torch.int8)
    # Use a separate workspace for the reference wrapper so the prefill
    # wrapper's planning metadata doesn't corrupt trtllm's workspace.
    ref_workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8)

    if max_q_len is not None and max_q_len > 1:
        # Speculative decoding: use causal prefill wrapper as reference
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            ref_workspace_buffer,
            kv_layout,
        )
        wrapper.plan(
            qo_indptr=q_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_size,
            page_size=block_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ref_workspace_buffer,
            kv_layout,
            use_tensor_cores=True,
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_qo_heads,
            num_kv_heads,
            head_size,
            block_size,
            "NONE",
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    def time_fn(fn, warmup=10, trials=20):
        torch.accelerator.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        times = []
        for i in range(warmup):
            fn()
        for i in range(trials):
            start.record()
            fn()
            end.record()
            torch.accelerator.synchronize()
            times.append(start.elapsed_time(end))  # ms
        return sum(times) / len(times), torch.std(torch.tensor(times))

    o_scale = 1.0
    o_sf_scale = None
    o_sf_vec_size = None
    output_baseline = torch.empty(ref_query.shape, dtype=dtype)
    if o_quant_dtype == FP4_DTYPE:
        o_sf_scale = 500.0
        o_sf_vec_size = 16
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2,), dtype=torch.uint8),
            torch.empty(
                (
                    round_up(query.shape[0], 128),
                    round_up(query.shape[1] * query.shape[2] // o_sf_vec_size, 4),
                ),
                dtype=torch.float8_e4m3fn,
            ),
        )
    else:
        output_trtllm = torch.empty(query.shape, dtype=o_quant_dtype)

    def baseline_decode():
        if max_q_len is not None and max_q_len > 1:
            return wrapper.run(ref_query, ref_kv_cache, out=output_baseline)
        else:
            return wrapper.run(
                ref_query,
                ref_kv_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                out=output_baseline,
            )

    # Build trtllm-specific inputs. For spec-decode with padding, we pad only the
    # tensors passed to trtllm; the reference setup (block_tables, seq_lens) is
    # unchanged and used as-is for the reference wrapper.
    trtllm_block_tables = block_tables
    trtllm_seq_lens = seq_lens
    trtllm_q_indptr = q_indptr  # None for standard decode

    if max_q_len is not None and max_q_len > 1 and pad_to_power_of_2:
        # Pad only the trtllm inputs so that padded batch size equals total_q.
        # Padded sequences have 0 Q tokens (repeat q_indptr[-1]) so the kernel skips them.
        padded_batch_size = total_q
        pad = padded_batch_size - batch_size
        if pad > 0:
            trtllm_q_indptr = torch.cat([q_indptr, q_indptr[-1:].expand(pad)])
            trtllm_seq_lens = torch.cat([seq_lens, torch.ones(pad, dtype=torch.int32)])
            trtllm_block_tables = torch.cat(
                [block_tables, block_tables.new_zeros(pad, block_tables.shape[1])]
            )

    # q_len_per_req=None tells the kernel to use max_q_len + cum_seq_lens_q for
    # variable-length spec-decode; leaving it at the default (1) would override
    # max_q_len and compute the wrong batch_size.
    if max_q_len is not None and max_q_len > 1:
        trtllm_extra = {
            "q_len_per_req": None,
            "max_q_len": max_q_len,
            "cum_seq_lens_q": trtllm_q_indptr,
        }
    else:
        trtllm_extra = {}

    def trtllm_decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=trtllm_block_tables,
            seq_lens=trtllm_seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=q_scale * k_scale * sm_scale,
            bmm2_scale=v_scale / o_scale,
            o_sf_scale=o_sf_scale,
            out=output_trtllm,
            **trtllm_extra,
        )

    # Correctness check before timing.
    # Use the actual quantization scales (not the hardcoded 1.0 used for benchmarking)
    # so that the two kernels operate on equivalent inputs.
    actual_q_scale = q_inv_scale.item() if q_inv_scale is not None else q_scale
    actual_k_scale = kv_inv_scale.item() if kv_inv_scale is not None else k_scale
    actual_v_scale = kv_inv_scale.item() if kv_inv_scale is not None else v_scale
    # Baseline uses fake-quantized bf16 reference to match fp8 precision loss
    ref_q_check = (
        query.to(dtype) * actual_q_scale if q_inv_scale is not None else ref_query
    )
    ref_kv_check = (
        kv_cache.to(dtype) * actual_k_scale
        if kv_inv_scale is not None
        else ref_kv_cache
    )
    wrapper.run(ref_q_check, ref_kv_check, out=output_baseline)
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=trtllm_block_tables,
        seq_lens=trtllm_seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=actual_q_scale * actual_k_scale * sm_scale,
        bmm2_scale=actual_v_scale / o_scale,
        o_sf_scale=o_sf_scale,
        out=output_trtllm,
        **trtllm_extra,
    )

    if o_quant_dtype == FP4_DTYPE:
        # Unpack trtllm FP4 output and compare to reference-quantized baseline
        out_unpacked = _cast_from_fp4(output_trtllm.data)
        sf_start = getattr(output_trtllm, "scale_start_index", 0)
        out_scale = _recover_swizzled_scales(
            output_trtllm.scale,
            out_unpacked.shape[0],
            math.prod(out_unpacked.shape[1:]),
            o_sf_vec_size,
            sf_start,
        )
        out_ref_fp4, out_scale_ref = _ref_fp4_quant(
            output_baseline.float(), o_sf_scale, o_sf_vec_size
        )
        assert torch.allclose(
            out_scale.reshape(out_scale_ref.shape), out_scale_ref, rtol=2e-1, atol=2e-1
        ), (
            f"FP4 scale mismatch: max_diff="
            f"{(out_scale.reshape(out_scale_ref.shape) - out_scale_ref).abs().max():.4f}"
        )
        rmse = torch.sqrt(torch.mean((out_unpacked - out_ref_fp4.float()) ** 2))
        assert rmse.item() < 0.3, f"FP4 output RMSE too large: {rmse.item():.4f}"
    else:
        # Use float64 to avoid squaring overflow: bf16 max ~3.4e38, so diffs
        # up to ~1e19 would overflow float32 when squared (max ~3.4e38).
        out_ref = output_baseline.double()
        out_trt = output_trtllm.double()
        assert not torch.isnan(out_ref).any() and not torch.isinf(out_ref).any(), (
            f"baseline output contains NaN/Inf "
            f"(batch_size={batch_size}, max_seq_len={max_seq_len})"
        )
        assert not torch.isnan(out_trt).any() and not torch.isinf(out_trt).any(), (
            f"trtllm output contains NaN/Inf "
            f"(batch_size={batch_size}, max_seq_len={max_seq_len})"
        )
        # Use RMSE to be robust against outlier elements from different kernel
        # accumulation orders; threshold is looser for FP8 quantization.
        rmse_tol = (
            0.1 if (q_quant_dtype == FP8_DTYPE or kv_quant_dtype == FP8_DTYPE) else 0.05
        )
        rmse = torch.sqrt(torch.mean((out_ref - out_trt) ** 2))
        assert rmse.item() < rmse_tol, (
            f"Output RMSE too large: {rmse.item():.4f} (tol={rmse_tol}), "
            f"max_diff={(out_ref - out_trt).abs().max().item():.4f}"
        )

    baseline_mean, baseline_std = time_fn(baseline_decode)
    trtllm_mean, trtllm_std = time_fn(trtllm_decode)

    # Calculate percentage speedup (positive means TRT is faster)
    speedup_percent = (baseline_mean - trtllm_mean) / baseline_mean

    print(
        f"\t{batch_size}\t{max_seq_len}\t{trtllm_mean:.3f}\t{trtllm_std.item():.3f}"
        f"\t{baseline_mean:.3f}\t{baseline_std.item():.3f}\t{speedup_percent:.3f}"
    )

    # Return results for CSV writing
    return {
        "batch_size": batch_size,
        "trtllm_mean": trtllm_mean,
        "trtllm_std": trtllm_std.item(),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std.item(),
        "speedup_percent": speedup_percent,
        "q_dtype": str(q_quant_dtype),
        "kv_cache_dtype": str(kv_quant_dtype),
        "output_dtype": str(o_quant_dtype),
        "block_size": block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "max_seq_len": max_seq_len,
        "max_q_len": max_q_len if max_q_len is not None else 1,
        "padded": pad_to_power_of_2,
    }


def write_results_to_csv(results, filename=None):
    """Write benchmark results to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashinfer_trtllm_benchmark_{timestamp}.csv"

    fieldnames = [
        "batch_size",
        "trtllm_mean",
        "trtllm_std",
        "baseline_mean",
        "baseline_std",
        "speedup_percent",
        "q_dtype",
        "kv_cache_dtype",
        "output_dtype",
        "block_size",
        "num_kv_heads",
        "head_size",
        "max_seq_len",
        "max_q_len",
        "padded",
    ]

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"Results written to {filename}")


if __name__ == "__main__":
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    # max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    # max_seq_lens = [1024]
    # Variable query lengths for speculative decoding benchmark
    spec_max_q_lens = [2, 4, 8]
    all_results = []

    dtype = torch.bfloat16
    quant_dtypes = [
        # (q_quant_dtype, kv_quant_dtype, o_quant_dtype)
        (None, None, None),
        # (None, FP8_DTYPE, None),
        # (FP8_DTYPE, FP8_DTYPE, None),
        # (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
        # (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),
    ]

    # # Standard decode benchmark (q_len = 1 per request)
    # for quant_dtype in quant_dtypes:
    #     q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtype
    #     q_quant_dtype = q_quant_dtype or dtype
    #     kv_quant_dtype = kv_quant_dtype or dtype
    #     o_quant_dtype = o_quant_dtype or dtype

    #     print(
    #         f"Running decode benchmark (q_len=1) for q_dtype = {q_quant_dtype}, "
    #         f"kv_cache_dtype: {kv_quant_dtype}, "
    #         f"output_dtype: {o_quant_dtype}"
    #     )
    #     print(
    #         "\tbatch_size\tmax_seq_len\ttrtllm_mean\ttrtllm_std\tbaseline_mean\t"
    #         "baseline_std\tspeedup_percent"
    #     )
    #     for max_seq_len in max_seq_lens:
    #         for bs in batch_sizes:
    #             result = benchmark_decode(
    #                 dtype=dtype,
    #                 quant_dtypes=quant_dtype,
    #                 batch_size=bs,
    #                 max_seq_len=max_seq_len,
    #             )
    #             all_results.append(result)

    # Speculative decoding benchmark (variable q_len > 1 per request)
    # Note: max_q_len is only supported by the trtllm-gen backend (requires SM100+).
    for max_q_len in spec_max_q_lens:
        for quant_dtype in quant_dtypes:
            q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtype
            q_quant_dtype = q_quant_dtype or dtype
            kv_quant_dtype = kv_quant_dtype or dtype
            o_quant_dtype = o_quant_dtype or dtype

            for pad in [False, True]:
                pad_label = "padded batch to total_q" if pad else "no padding"
                print(
                    f"Running spec-decode benchmark (max_q_len={max_q_len}, {pad_label}) for "
                    f"q_dtype = {q_quant_dtype}, "
                    f"kv_cache_dtype: {kv_quant_dtype}, "
                    f"output_dtype: {o_quant_dtype}"
                )
                print(
                    "\tbatch_size\tmax_seq_len\ttrtllm_mean\ttrtllm_std\tbaseline_mean\t"
                    "baseline_std\tspeedup_percent"
                )
                for max_seq_len in max_seq_lens:
                    for bs in batch_sizes:
                        result = benchmark_decode(
                            dtype=dtype,
                            quant_dtypes=quant_dtype,
                            batch_size=bs,
                            max_seq_len=max_seq_len,
                            max_q_len=max_q_len,
                            pad_to_power_of_2=pad,
                        )
                        all_results.append(result)

    # Write all results to CSV
    write_results_to_csv(all_results)
