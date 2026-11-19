# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-by-stage bring-up of ``WideGemm``: compile, launch, sync, check (one case)."""

import sys

import torch


def main():
    rows, n, k, split, silu = (int(a) for a in sys.argv[1:6])
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide

    print("import ok", flush=True)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    g = wide.PreparedWide(w, {rows: split}, silu=bool(silu)).prepare()
    print("compiled", flush=True)
    out = torch.zeros(rows, n // 2 if silu else n, device="cuda", dtype=torch.bfloat16)
    g.run(x, out)
    print("launched", flush=True)
    torch.cuda.synchronize()
    print("synced", flush=True)
    ref = (x.float() @ g.weight.float().t())
    for _ in range(3):
        g.run(x, out)
    torch.cuda.synchronize()
    for _ in range(3):
        g.run(x, out)
    torch.cuda.synchronize()
    st = g.state[rows][1][n // 64:].view(torch.int64).view(-1, 8)[:, :6].double()
    st = (st - st[:, 0].min()) / 1000.0
    for i, nm in enumerate(['epi_entry', 'mma_done', 'staged', 'acquired', 'loaded', 'stored']):
        col = st[:, i]
        print(f'{nm:10s} min {col.min():7.2f} med {col.median():7.2f} max {col.max():7.2f}')
    if not silu:
        err = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"rel_err {err:.3e}", flush=True)
        bad = (out.float() - ref).abs() > 0.02 * ref.abs().max()
        print("bad rows/cols", bad.any(1).nonzero().flatten()[:8].tolist(),
              bad.any(0).nonzero().flatten()[:8].tolist(), flush=True)
    else:
        print("silu out absmax", out.float().abs().max().item(), flush=True)


if __name__ == "__main__":
    main()
