# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: stock concat_and_cache_mla vs the specialized prefill FP8 insert
at the P3-16 per-GPU prefill shape (16K tokens), plus a bitwise check."""

import torch

from vllm import _custom_ops as ops
from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    for t in (16384, 8192):
        kv = (torch.randn(t, 576, device="cuda") * 30).to(torch.bfloat16)
        kv_c, k_pe = kv[:, :512], kv[:, 512:]
        blocks, bs = 8192, 64
        slots = torch.randperm(blocks * bs, device="cuda")[:t].sort().values
        cache = torch.zeros(blocks, bs, 576, dtype=torch.uint8, device="cuda")
        ref = cache.clone()
        one = torch.ones(1, device="cuda")
        stock = timed(lambda: ops.concat_and_cache_mla(
            kv_c, k_pe, ref, slots, "fp8", one, True))
        new = timed(lambda: kv_insert.insert(kv_c, k_pe, cache, slots))
        ok = torch.equal(cache, ref)
        mb = t * 576 * 3 / 1e6
        print(f"kv_insert T={t}: stock {stock:.1f} us, new {new:.1f} us "
              f"({mb / new:.2f} TB/s), bitwise {ok}", flush=True)


if __name__ == "__main__":
    main()
