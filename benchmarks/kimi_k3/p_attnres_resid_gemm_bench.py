# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES: price of the producer-fused residual epilogue (dataflow C).

The producer of an AttnRes delta writes P <- bf16(P + bf16(x @ w.T)) in place into
the prefix buffer instead of y = bf16(x @ w.T) (stock rounding: the GEMM output
rounds to BF16 first, the sum rounds again). Arms per [rows, K] x [K, 7168] BF16
shape, sustained (power cap), L2-cold operands (rotating buffer sets):
  nvjet     torch.mm (cuBLASLt)
  cute      the sm_107 CuTeDSL persistent base (bf16_gemm.Bf16Gemm), plain store
  res_sub   cute + residual epilogue, P loaded one subtile ahead (aux_t2r)
  res_tile  cute + residual epilogue, P loaded one whole tile ahead (aux_tile_ahead)
Check: res_* == bf16(P0 + cute(x, w)) bitwise. Energy: NVML total-energy counter.
"""

import argparse
import json
import statistics
import time

import cutlass
import cutlass.cute as cute
import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.bf16_gemm import Bf16Gemm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.dense_gemm_bf16_sm107 import (
    PersistentDenseGemmKernel,
)

N = 7168


class _ResSubKernel(PersistentDenseGemmKernel):
    """out = BF16(aux + BF16(acc)) with aux = the prefix (in place)."""

    aux_t2r = True

    def epi_convert_regs_aux(self, acc, aux, dst, tiled_copy_r2s):
        out = cute.make_rmem_tensor(acc.shape, self.c_dtype)
        y = acc.load().to(cutlass.BFloat16).to(cutlass.Float32)
        out.store((aux.load().to(cutlass.Float32) + y).to(self.c_dtype))
        dst.store(tiled_copy_r2s.retile(out).load())


class _ResTileKernel(_ResSubKernel):
    aux_tile_ahead = True


class ResSub(Bf16Gemm):
    kernel_cls = _ResSubKernel

    def __call__(self, x, w, prefix):
        return super().__call__(x, w, prefix, aux=prefix)


class ResTile(ResSub):
    kernel_cls = _ResTileKernel


def energy_mj():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--ks", default="3584,4096,8192,12288")
    p.add_argument("--sets", type=int, default=3, help="rotating operand sets (L2-cold)")
    p.add_argument("--secs", type=float, default=2.0, help="sustained seconds per arm round")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--arms", default="nvjet,cute,res_sub,res_tile")
    args = p.parse_args()
    torch.manual_seed(0)
    T = args.rows
    for K in [int(k) for k in args.ks.split(",")]:
        sets = []
        for _ in range(args.sets):
            x = torch.randn(T, K, device="cuda").bfloat16()
            w = (torch.randn(N, K, device="cuda") * K**-0.5).bfloat16()
            pre = (torch.randn(T, N, device="cuda") * 4).bfloat16()
            sets.append((x, w, pre, torch.empty(T, N, device="cuda", dtype=torch.bfloat16)))
        cute_mm, res_sub, res_tile = Bf16Gemm(), ResSub(), ResTile()

        x, w, pre, y = sets[0]
        cute_mm(x, w, y)
        ref = (pre.float() + y.float()).bfloat16()
        check = {}
        for name, g in (("res_sub", res_sub), ("res_tile", res_tile)):
            pp = pre.clone()
            g(x, w, pp)
            check[name] = bool(torch.equal(pp, ref))
        yn = torch.mm(x, w.t())
        check["cute_vs_nvjet_max_abs"] = float((y.float() - yn.float()).abs().max())
        torch.cuda.synchronize()

        fns = {
            "nvjet": lambda s: torch.mm(s[0], s[1].t(), out=s[3]),
            "cute": lambda s: cute_mm(s[0], s[1], s[3]),
            "res_sub": lambda s: res_sub(s[0], s[1], s[2]),
            "res_tile": lambda s: res_tile(s[0], s[1], s[2]),
        }
        arms = args.arms.split(",")
        res = {a: [] for a in arms}
        ej = {a: [] for a in arms}
        for r in range(args.rounds):
            for a in arms if r % 2 == 0 else arms[::-1]:
                f = fns[a]
                for i in range(6):
                    f(sets[i % args.sets])
                torch.cuda.synchronize()
                n = 0
                e0 = energy_mj()
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.time()
                ev0.record()
                while time.time() - t0 < args.secs:
                    for _ in range(10):
                        f(sets[n % args.sets])
                        n += 1
                    torch.cuda.synchronize()
                ev1.record()
                torch.cuda.synchronize()
                e1 = energy_mj()
                res[a].append(ev0.elapsed_time(ev1) * 1e3 / n)
                if e0 is not None and e1 is not None:
                    ej[a].append((e1 - e0) / n)
        out = {"rows": T, "K": K, "check": check}
        for a in arms:
            out[a + "_us"] = round(statistics.median(res[a]), 1)
            if ej[a]:
                out[a + "_mJ"] = round(statistics.median(ej[a]), 2)
        print(json.dumps(out))
        del sets


if __name__ == "__main__":
    main()
