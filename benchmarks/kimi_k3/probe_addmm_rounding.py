# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many roundings does stock's BF16 beta-add GEMM (``residual.addmm_(x, W^T)``,
the MoE latent up: 7168 x 3584) apply on this GPU? Compares cuBLAS's output with
one rounding (bf16(acc + residual)) and two (bf16(bf16(acc) + residual)), acc in FP64
(the accumulation order differs from cuBLAS's FP32 only far below a BF16 ulp)."""

import torch


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    for rows in (32, 64, 96):
        x = torch.randn(rows, 3584, device=dev).bfloat16()
        w = (torch.randn(7168, 3584, device=dev) * 0.02).bfloat16()
        res = torch.randn(rows, 7168, device=dev).bfloat16()
        out = res.clone().addmm_(x, w.t())
        acc = x.double() @ w.double().t()
        one = (acc + res.double()).float().bfloat16()
        two = (acc.float().bfloat16().double() + res.double()).float().bfloat16()
        mm = torch.mm(x, w.t())  # the GEMM alone, BF16 out
        two_mm = (mm.double() + res.double()).float().bfloat16()
        n = out.numel()
        print(
            f"rows {rows}: out==one {(out == one).sum().item() / n:.5f} "
            f"out==two {(out == two).sum().item() / n:.5f} "
            f"out==bf16(mm)+res {(out == two_mm).sum().item() / n:.5f} "
            f"one==two {(one == two).sum().item() / n:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
