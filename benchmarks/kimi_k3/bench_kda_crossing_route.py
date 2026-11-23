# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill cost of an interior checkpoint (L1 crossing chunks) at deep-step
shapes: one rank's ~16K-token step split over a few chunks, with vs without a
checkpoint offset inside one chunk, on the persistent route (what the layer
takes when no segment plan applies) and the segmented route (when it does).

    python benchmarks/kimi_k3/bench_kda_crossing_route.py [--heads 96]
"""

import argparse

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def _inputs(T, H, n):
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(n, H, D, D, device="cuda") * 0.1
    return q, k, v, g, beta, A_log, dt_bias, h0


def _time(fn, iters=20):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) / iters


def persistent(lens, cuts, H):
    T, n = sum(lens), len(lens)
    q, k, v, g, beta, A_log, dt_bias, h0 = _inputs(T, H, n)
    cu = torch.tensor([0, *lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cache = torch.zeros(2 * n + 3, H, D, D, device="cuda")
    slot = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 1
    cache[slot.long()] = h0
    has = torch.ones(n, dtype=torch.bool, device="cuda")
    ck = torch.empty(n, H, D, D, device="cuda") if cuts else None

    def run():
        sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu.cuda(), cu,
            state_cache=cache, state_indices=slot, has_initial_state=has,
            step_memo=None, checkpoint_offsets=cuts, checkpoint_out=ck,
        )

    return _time(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=96)
    a = ap.parse_args()
    H = a.heads
    shapes = [
        # (lens of the rank's chunks, crossing chunk index, offset of B in it)
        ([16384], 0, 4096),
        ([16384], 0, 12288),
        ([8192, 8192], 1, 2048),
        ([4096, 12288], 1, 6144),
        ([2048, 6144, 8192], 2, 3072),
        ([1024] * 4 + [12288], 4, 8192),
    ]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"H={H} D={D} SMs={sms}")
    print("lens | persistent no-ckpt ms | with interior ckpt ms | delta | segscan plan (no/with ckpt)")
    for lens, i, off in shapes:
        cuts = [0] * len(lens)
        cuts[i] = off
        t0 = persistent(lens, None, H)
        t1 = persistent(lens, cuts, H)
        cu = torch.tensor([0, *lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        mp = kda_segscan.max_pseudo(H, sms, 64)
        p0 = kda_segscan.make_plan(cu, H, "cuda", None, sms=sms, max_parts=mp)
        p1 = kda_segscan.make_plan(cu, H, "cuda", cuts, sms=sms, max_parts=mp)
        print(f"{lens} | {t0:.3f} | {t1:.3f} | {100 * (t1 / t0 - 1):+.1f}% | "
              f"{getattr(p0, 'num_pseudo', None)}/{getattr(p1, 'num_pseudo', None)}")


if __name__ == "__main__":
    main()
