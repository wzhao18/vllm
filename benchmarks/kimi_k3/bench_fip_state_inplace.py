# SPDX-License-Identifier: Apache-2.0
"""Persistent prefill KDA: gathered state (gather + core + row scatter, the
production path before) vs in-place state I/O on the cache, F16K 3 x 5461."""

import argparse

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.models.kimi_k3.nvidia.ops import prefill_glue
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), 96
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    cache = torch.randn(64, H, D, D, device="cuda") * 0.1
    idx = torch.tensor([5, 17, 40], dtype=torch.int32, device="cuda")[:N]
    has = torch.ones(N, dtype=torch.bool, device="cuda")

    def gathered():
        h0 = gather_initial_states(cache, idx, has)
        _, fs = sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu, cu_cpu
        )
        prefill_glue.write_state_rows(cache, idx, fs)

    def inplace():
        sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
            state_cache=cache, state_indices=idx, has_initial_state=has,
        )

    for name, fn in (("gathered", gathered), ("inplace", inplace)) * 2:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            fn()
        ts = []
        for _ in range(a.iters):
            e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            e0.record()
            gph.replay()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        ts.sort()
        print(f"lens={a.lens} {name:9s} graph us median {ts[len(ts) // 2]:.1f} min {ts[0]:.1f}")


if __name__ == "__main__":
    main()
