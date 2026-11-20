# SPDX-License-Identifier: Apache-2.0
"""Per-step event timeline of persistent-KDA prefill CTA 0 (dev TLINE_ stamps).

Compiles the kernel with TLINE_=1 (plus any extra dev overrides), runs one
F16K-style launch, and prints clock64 deltas of CTA 0's first chain piece:
E0 compute step top, E1 image published, E2 decay done, E3 VFULL, E4 OOUT,
E5 residual published, E6 U2ACC, E7 v_new published, E8 FIN, E9 epi OFIN,
E10 epi image(t) published, E11 epi decay(t) done, E12 MMA1 issue, E13 MMA3
issue, E14 MMA4a issue, E15 prep RFREE(t-5), E16 prep SFREE(t-5), E17 prep
QKRAW, E18 prep TAB, E19 prep GRAM, E20 prep QK.
"""
import argparse
import json

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fwd_mod
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--t0", type=int, default=8)
    ap.add_argument("--t1", type=int, default=40)
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
    idx = torch.tensor([5, 17, 40, 41, 50, 60], dtype=torch.int32, device="cuda")[:N]
    has = torch.ones(N, dtype=torch.bool, device="cuda")
    fwd_mod._DEV = {"TLINE_": 1, **json.loads(a.extra)}
    for _ in range(3):
        sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
            state_cache=cache, state_indices=idx, has_initial_state=has,
        )
    torch.cuda.synchronize()
    tl = fwd_mod.LAST_TPROBE[0][: 48 * 32].view(48, 32).cpu().tolist()
    names = ["top", "img", "dec", "vful", "oout", "res", "u2acc", "vnew", "FIN",
             "eOFIN", "eimg", "edec", "m1", "m3", "m4a", "pRF", "pSF", "pQKR", "pTAB",
             "pGRAM", "pQK", "pGRAW", "pGSCN", "pMASK", "pSOLV", "pCOMB", "pREST"]
    print("extra", a.extra, "| deltas in clk relative to FIN(t-1) [E8 of step t-1]; prep events of item t")
    print("t    " + " ".join(f"{n:>6s}" for n in names) + "   step")
    for t in range(a.t0, a.t1):
        base = tl[t - 1][8]
        row = [tl[t][e] - base if tl[t][e] else -1 for e in range(27)]
        # epilogue image/decay for chunk t are stamped at index t (t+1 written by step t's epi loop)
        print(f"{t:3d} " + " ".join(f"{x:6d}" for x in row) + f"  {tl[t][8] - base:6d}")
    steps = [tl[t][8] - tl[t - 1][8] for t in range(a.t0, a.t1)]
    print("median step clk", sorted(steps)[len(steps) // 2])


if __name__ == "__main__":
    main()
