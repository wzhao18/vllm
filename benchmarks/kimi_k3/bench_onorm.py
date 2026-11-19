# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA gated o_norm + MXFP8 o_proj input: staged (bulk-copy) kernel vs the
direct-load reference, bitwise check and time at the DEP16 prefill shapes."""

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer


def main():
    kda_segscan.load_extension()
    ops = torch.ops.k3_flashkda_split
    H, D = 96, 128
    torch.manual_seed(0)
    for T in (16384, 8192):
        P = H * D
        x = (torch.randn(1, T, H, D, device="cuda") * 3).to(torch.bfloat16)
        proj = (torch.randn(T, 49376, device="cuda") * 2).to(torch.bfloat16)
        g = proj[:, 3 * P : 4 * P]
        w = (torch.rand(D, device="cuda") + 0.5).contiguous()
        res = {}
        for name, op in (("staged", ops.o_norm_mxfp8), ("direct", ops.o_norm_mxfp8_ref)):
            out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
            for _ in range(3):
                op(x, g, w, 1e-6, out.data, out.scale)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                op(x, g, w, 1e-6, out.data, out.scale)
            e1.record()
            torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / 20 * 1000
            gb = (2 * T * P * 2 + T * P + T * P // 32) / 1e9
            print(f"{name} T={T} H={H}: {us:.1f} us ({gb / us * 1e3:.2f} TB/s)", flush=True)
            res[name] = out
        same = torch.equal(res["staged"].data.view(torch.uint8), res["direct"].data.view(torch.uint8))
        same &= torch.equal(res["staged"].scale, res["direct"].scale)
        print(f"T={T}: staged == direct bitwise: {same}", flush=True)


if __name__ == "__main__":
    main()
