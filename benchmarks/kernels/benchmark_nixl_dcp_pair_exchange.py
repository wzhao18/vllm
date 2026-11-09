# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the NIXL DCP pair exchange kernels against the HBM roofline.

Packs and unpacks ``--units`` units of Kimi-K3.1's packed DSA block (24 DSA
layers of fp8 latent and indexer pages plus ``--num-draft`` DSpark latent
pages, laid out by the real KV cache planner) in rounds of ``--chunk-units``,
on one GPU. Pack and unpack each read and write every exchanged byte once, so
each moves 4 x H bytes per unit, H being half of a block's exchanged bytes;
unpacking the last unit, which has no partner, zero-fills its half 1 without
reading it, so moves 3 x H.
Stage 4 of the design asks for at least 70% of the roofline.

``exchange`` times the whole step as the worker runs it, eagerly: unit-table
upload, launches and, for each NCCL round, a device copy of ``send`` into
``recv`` instead.

Example:
    python benchmarks/kernels/benchmark_nixl_dcp_pair_exchange.py \\
        --units 2 5 13 64 --chunk-units 4 8 --num-draft 5

"""

import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from tabulate import tabulate

from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    DcpPairExchanger,
    PairRegion,
    PairUnit,
    build_pair_row_table,
    pair_layout,
    reference_pack,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache, customize_attention_spec

# Kimi-K3.1 at fp8: 576-byte latent rows, 132-byte indexer rows (128 fp8 values
# and an fp32 scale) read as 64-row kernel pages.
LATENT_BYTES, INDEX_ROW_BYTES, INDEX_VALUE_BYTES, INDEX_KERNEL_ROWS = 576, 132, 128, 64
# HBM bandwidth in TB/s by device name, first match wins.
HBM_TBPS = (("B300", 8.0), ("B200", 8.0), ("H200", 4.8), ("H100", 3.35))


def k31_cache(num_dsa: int, num_draft: int, block_size: int, num_blocks: int):
    """K3.1's packed DSA KV cache on the GPU, as the planner lays it out, and
    the row table of its NIXL regions."""
    latent = customize_attention_spec(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE.get_class(),
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=LATENT_BYTES,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
            kv_quant_mode=get_kv_quant_mode("fp8"),
        ),
    )
    index = customize_attention_spec(
        DeepseekV32IndexerBackend,
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=INDEX_ROW_BYTES,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        ),
    )
    specs = {}
    for i in range(num_dsa):
        specs[f"model.layers.{i}.self_attn.attn"] = latent
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = index
    for i in range(num_draft):
        specs[f"draft.layers.{i}.self_attn.attn"] = latent

    cache_config = CacheConfig(block_size=block_size)
    cache_config.kv_cache_layout = "BLHNC"
    cache_config.num_gpu_blocks_override = num_blocks
    config = SimpleNamespace(
        cache_config=cache_config,
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size, specs))
    kv_cache_config = get_kv_cache_config_from_groups(config, [group], 1)
    views = allocate_kv_cache(
        kv_cache_config, torch.device("cuda"), KVCacheLayout.BLHNC
    )
    storage = next(iter(views.values())).untyped_storage()
    buf = torch.empty(0, dtype=torch.uint8, device="cuda").set_(storage)
    regions = [
        PairRegion(
            name,
            specs[name],
            view.data_ptr(),
            specs[name].page_size_bytes,
            view.stride(0) * view.element_size(),
        )
        for name, view in views.items()
    ]
    table = build_pair_row_table(
        regions,
        index_kernel_rows=INDEX_KERNEL_ROWS,
        index_value_bytes=INDEX_VALUE_BYTES,
    )
    return buf, table


class Loopback:
    """Stands in for NCCL: ``recv`` gets this rank's own ``send``."""

    def batch_isend_irecv(self, ops) -> None:
        ops[1].tensor.copy_(ops[0].tensor)


def gpu_us(fn, graph: bool, iters: int) -> float:
    """Median GPU time of ``fn``: CUPTI over cold L2 when FlashInfer has it,
    else CUDA events."""
    try:
        from flashinfer.testing import bench_gpu_time_with_cupti
    except ImportError:
        bench_gpu_time_with_cupti = None
    if bench_gpu_time_with_cupti is not None:
        times = bench_gpu_time_with_cupti(
            fn, use_cuda_graph=graph, cold_l2_cache=True, repeat_iters=iters
        )
        return statistics.median(times) * 1e3
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times) * 1e3


def host_us(fn, iters: int) -> float:
    """Median host time of one call, the stream drained between calls."""
    times = []
    for _ in range(iters):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    torch.accelerator.synchronize()
    return statistics.median(times) * 1e6


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    name = torch.cuda.get_device_name()
    hbm_tbps = args.hbm_tbps or next(
        (tbps for key, tbps in HBM_TBPS if key in name), None
    )
    if hbm_tbps is None:
        raise RuntimeError(f"unknown HBM bandwidth of {name}: pass --hbm-tbps")
    max_units = max(args.units)
    buf, table = k31_cache(args.num_dsa, args.num_draft, args.block_size, max_units + 1)
    buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8, device="cuda"))
    half_bytes = table.half_bytes
    layout = pair_layout(0, 16, 8, args.block_size, 1)
    print(
        f"{name}: {len(table.arrays)} arrays, H = {half_bytes / 1e6:.2f} MB, "
        f"block stride {table.block_stride / 1e6:.2f} MB, roofline {hbm_tbps} TB/s"
    )

    rng = np.random.default_rng(0)
    rows = []
    for chunk_units in args.chunk_units:
        exchanger = DcpPairExchanger(
            table,
            layout,
            torch.device("cuda"),
            chunk_units=chunk_units,
            communicator=Loopback(),
        )
        plan = exchanger._plan
        for num_units in args.units:
            block_ids = (rng.permutation(max_units) + 1)[:num_units].tolist()
            units = [PairUnit(0, b, True) for b in block_ids]
            units[-1] = PairUnit(0, block_ids[-1], False)
            unit_rows = async_tensor_h2d(
                [(u.block_id, u.has_partner) for u in units], "cuda", torch.int64
            )
            chunks = [
                (units[i : i + chunk_units], unit_rows[i : i + chunk_units])
                for i in range(0, num_units, chunk_units)
            ]

            def pack(chunks=chunks, exchanger=exchanger, plan=plan):
                for chunk, chunk_rows in chunks:
                    exchanger._pack(chunk, chunk_rows, plan)

            def unpack(chunks=chunks, exchanger=exchanger, plan=plan):
                for chunk, chunk_rows in chunks:
                    exchanger._unpack(chunk, chunk_rows, plan)

            def exchange(units=units, exchanger=exchanger):
                exchanger.exchange(units)

            # The kernels must match the reference before they are timed.
            first, first_rows = chunks[0]
            exchanger._pack(first, first_rows, plan)
            keep, send = reference_pack(buf, table, layout, [u.block_id for u in first])
            n = len(first)
            assert torch.equal(exchanger.keep[:n], keep)
            assert torch.equal(exchanger.send[:n], send)
            for fn in (pack, unpack, exchange):
                fn()
            torch.accelerator.synchronize()

            pack_bytes = 4 * half_bytes * num_units
            unpack_bytes = pack_bytes - half_bytes
            pack_us = gpu_us(pack, graph=True, iters=args.iters)
            unpack_us = gpu_us(unpack, graph=True, iters=args.iters)
            rows.append(
                [
                    num_units,
                    chunk_units,
                    len(chunks),
                    f"{pack_us:.1f}",
                    f"{pack_bytes / pack_us / 1e3:.0f}",
                    f"{pack_bytes / pack_us / 1e6 / hbm_tbps:.0%}",
                    f"{unpack_us:.1f}",
                    f"{unpack_bytes / unpack_us / 1e3:.0f}",
                    f"{unpack_bytes / unpack_us / 1e6 / hbm_tbps:.0%}",
                    f"{gpu_us(exchange, graph=False, iters=args.iters):.1f}",
                    f"{host_us(exchange, iters=args.iters):.1f}",
                ]
            )
    print(
        tabulate(
            rows,
            headers=[
                "units",
                "K",
                "rounds",
                "pack us",
                "pack GB/s",
                "pack roof",
                "unpack us",
                "unpack GB/s",
                "unpack roof",
                "exchange us",
                "exchange host us",
            ],
        )
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, nargs="+", default=[2, 5, 13, 64])
    parser.add_argument("--chunk-units", type=int, nargs="+", default=[4])
    parser.add_argument("--num-dsa", type=int, default=24)
    parser.add_argument("--num-draft", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--hbm-tbps", type=float, default=None, help="HBM roofline; by GPU name"
    )
    parser.add_argument("--iters", type=int, default=50)
    main(parser.parse_args())
