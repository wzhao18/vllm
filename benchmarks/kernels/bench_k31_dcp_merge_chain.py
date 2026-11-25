# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The #63 DCP top-k merge chain as the DSA decode launches it: per layer
``dcp_topk_merge_rows`` (pack -> all-gather -> select -> rescue) and a PDL
consumer that reads the rows after its wait (as the DSA sparse core). One CUDA
graph of ``--layers`` merges per arm; arms ``release`` False / True alternate
(ABAB) on the same inputs. Prints us per merge+consumer per rows, per arm.

    python3 benchmarks/kernels/bench_k31_dcp_merge_chain.py --world 4
"""

import argparse
import os
import sys

import torch
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests")
)
from distributed.test_dcp_topk_merge import (  # noqa: E402
    _BLOCK,
    TOPK,
    _block_table,
    _make_case,
    _pdl_copy,
    _pdl_copy_kernels,
    _shard,
)


def _bench(group, n, rows, args):
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge_rows

    world, rank = group.world_size, group.rank_in_group
    _, late = _pdl_copy_kernels()
    # Every row a 132K-token request (q8 verify rows of B requests share it).
    base, ends = _make_case([args.context] * rows, seed=rows)
    static, local_end = _shard(base, ends, world, rank)
    req_id = torch.arange(rows, dtype=torch.int32, device="cuda")
    block_table = _block_table(rows, int(local_end.max()))
    rows_out = torch.empty((rows, TOPK), dtype=torch.int32, device="cuda")
    valid_out = torch.empty(rows, dtype=torch.int32, device="cuda")
    seen = torch.empty_like(valid_out)

    def step(release):
        for _ in range(args.layers):
            dcp_topk_merge_rows(
                logits=static,
                row_ends=local_end,
                local_topk=n,
                topk=TOPK,
                req_id=req_id,
                block_table=block_table,
                block_size=_BLOCK,
                stride_rows=_BLOCK,
                dcp_rank=rank,
                rows_out=rows_out,
                valid_out=valid_out,
                release=release,
            )
            _pdl_copy(late, valid_out, seen)

    graphs = {}
    for release in (False, True):
        step(release)
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with group.graph_capture() as ctx, torch.cuda.graph(g, stream=ctx.stream):
            step(release)
        graphs[release] = g
    torch.accelerator.synchronize()
    times = {False: [], True: []}
    for i in range(2 * args.reps):
        release = bool(i % 2)
        group.barrier()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graphs[release].replay()
        b.record()
        b.synchronize()
        times[release].append(a.elapsed_time(b) * 1000 / args.layers)
    out = {}
    for release, ts in times.items():
        ts.sort()
        out[release] = (ts[len(ts) // 2], ts[0])
    return out


def _worker(rank, world, init_method, args):
    from types import SimpleNamespace

    os.environ["VLLM_DSA_BPRIME_MAILBOX"] = str(int(args.mailbox))
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        init_dcp_topk_merge,
    )

    torch.accelerator.set_device_index(rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            group = get_dcp_group()
            n = dcp_merge_local_topk(TOPK, world)
            gate_config = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                model_config=SimpleNamespace(max_model_len=world << 17),
                kv_transfer_config=None,
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=None,
                num_speculative_tokens=0,
            )
            assert init_dcp_topk_merge(gate_config, TOPK) == n, "merge disabled"
            for rows in map(int, args.rows.split(",")):
                res = _bench(group, n, rows, args)
                if rank == 0:
                    (a, amin), (b, bmin) = res[False], res[True]
                    print(
                        f"world {world} N {n} rows {rows:4d}: release=False "
                        f"{a:7.2f} us/layer (min {amin:.2f})  release=True "
                        f"{b:7.2f} (min {bmin:.2f})  delta {a - b:+.2f}",
                        flush=True,
                    )
        finally:
            cleanup_dist_env_and_memory()


def main():
    from vllm.utils.network_utils import get_open_port

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--rows", default="8,16,32,64,128,256,512")
    ap.add_argument("--context", type=int, default=135_168)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--mailbox", action="store_true")
    args = ap.parse_args()
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    mp.spawn(_worker, args=(args.world, init_method, args), nprocs=args.world)


if __name__ == "__main__":
    main()
