# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES: price of a TMA-STAGED residual epilogue (dataflow C producer).

Uses the CUTLASS CuTeDSL Blackwell example dense_gemm_alpha_beta_persistent.py shipped
in the image (FlashInfer data dir): its epilogue already TMA-loads a C subtile into
shared memory ahead of the accumulator drain (the source-C pipeline). The epilogue
math is patched at load time (bench only) to the stock residual rounding
    D = bf16(f32(C) + f32(bf16(acc)))
with C = D = the prefix buffer (in place). Arms per [rows, K] x [K, 7168] BF16:
  nvjet      torch.mm (cuBLASLt)
  base       sm_107 CuTeDSL persistent base (bf16_gemm.Bf16Gemm), plain store
  ab_sum     example, original math alpha * acc + beta * C (alpha = beta = 1)
  ab_res     example, residual math (TMA-staged C), in place
Check: ab_res == bf16(P0 + base(x, w)) bitwise.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time


import torch

N = 7168


def load_example(residual: bool):
    import flashinfer

    path = os.path.join(
        os.path.dirname(flashinfer.__file__),
        "data/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py",
    )
    src = open(path).read().replace(
        'get_smem_capacity_in_bytes("sm_100")', 'get_smem_capacity_in_bytes("sm_107")'
    )
    if residual:
        pat = re.compile(
            r"d_vec = epilogue_op\(\s*\(\s*alpha\.to\(self\.epi_dtype\) \* acc_vec\.to\(self\.epi_dtype\)"
            r"\s*\+ beta\.to\(self\.epi_dtype\) \* c_vec_load\.to\(self\.epi_dtype\)\s*\)\s*\)\.to\(self\.d_dtype\)"
        )
        new = (
            "d_vec = (c_vec_load.to(cutlass.Float32)"
            " + acc_vec.to(cutlass.BFloat16).to(cutlass.Float32)).to(self.d_dtype)"
        )
        src, n = pat.subn(new, src)
        assert n == 1, f"epilogue math patch matched {n} times"
    # CuTeDSL compiles from the source file: write the patched copy to disk.
    import importlib.util
    import tempfile

    name = "ab_res_example" if residual else "ab_sum_example"
    fn = os.path.join(tempfile.mkdtemp(prefix="pattnres_"), name + ".py")
    with open(fn, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AbGemm:
    def __init__(self, mod):
        import cutlass
        import cutlass.utils as utils

        self.mod, self.cutlass = mod, cutlass
        self.gemm = mod.SM100PersistentDenseGemmAlphaBetaKernel(
            cutlass.Float32, cutlass.Float32, True, (256, 256), (2, 1)
        )
        self.mac = utils.HardwareInfo().get_max_active_clusters(2)
        self.fn = {}

    @staticmethod
    def _t(x):
        from cutlass.cute.runtime import from_dlpack

        return from_dlpack(x.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def __call__(self, x, w, c, d):
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (self._t(x), self._t(w), self._t(c), self._t(d))
        one = self.cutlass.Float32(1.0)
        key = (x.shape, w.shape)
        if key not in self.fn:
            self.fn[key] = cute.compile(self.gemm, *args, one, one, self.mac, stream)
        self.fn[key](*args, one, one, stream)
        return d


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
    p.add_argument("--ks", default="3584,8192")
    p.add_argument("--sets", type=int, default=3)
    p.add_argument("--secs", type=float, default=2.0)
    p.add_argument("--rounds", type=int, default=2)
    args = p.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.bf16_gemm import Bf16Gemm

    torch.manual_seed(0)
    torch.zeros(1, device="cuda")  # CUDA context before the CuTeDSL hardware queries
    T = args.rows
    base = Bf16Gemm()
    ab_sum = AbGemm(load_example(False))
    ab_res = AbGemm(load_example(True))
    for K in [int(k) for k in args.ks.split(",")]:
        sets = []
        for _ in range(args.sets):
            x = torch.randn(T, K, device="cuda").bfloat16()
            w = (torch.randn(N, K, device="cuda") * K**-0.5).bfloat16()
            pre = (torch.randn(T, N, device="cuda") * 4).bfloat16()
            sets.append((x, w, pre, torch.empty(T, N, device="cuda", dtype=torch.bfloat16)))
        x, w, pre, y = sets[0]
        base(x, w, y)
        ref = (pre.float() + y.float()).bfloat16()
        pp = pre.clone()
        ab_res(x, w, pp, pp)
        d2 = torch.empty_like(pre)
        ab_sum(x, w, pre, d2)
        check = {"ab_res_bitwise": bool(torch.equal(pp, ref)),
                 "ab_res_max_abs": float((pp.float() - ref.float()).abs().max()),
                 "ab_sum_max_abs_vs_ref": float((d2.float() - ref.float()).abs().max())}
        torch.cuda.synchronize()
        fns = {
            "nvjet": lambda s: torch.mm(s[0], s[1].t(), out=s[3]),
            # cuBLASLt beta-add into the prefix (ONE rounding: not stock; a
            # precision decision item only)
            "nvjet_beta": lambda s: torch.addmm(s[2], s[0], s[1].t(), out=s[2]),
            "base": lambda s: base(s[0], s[1], s[3]),
            "ab_sum": lambda s: ab_sum(s[0], s[1], s[2], s[3]),
            "ab_res": lambda s: ab_res(s[0], s[1], s[2], s[2]),
        }
        arms = list(fns)
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
