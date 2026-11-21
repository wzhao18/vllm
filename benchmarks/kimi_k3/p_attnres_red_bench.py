# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES dataflow C producer: residual add through a TMA REDUCE store.

The producer GEMM keeps its plain epilogue (acc -> BF16 tile in smem) and only
swaps the TMA store for cp.reduce.async.bulk.tensor .add: L2 performs
P <- bf16(f32(P) + f32(bf16(acc))) (the stock AttnRes prefix rounding) without
staging P in shared memory. Arms per [rows, K] x [K, 7168] (sustained loop with
NVML J and SM clock, and THRASH = one call after a 512 MB L2 flush):
  BF16:  nvjet | nvjet_beta (cuBLASLt addmm in place: one rounding, decision
         support only) | base (sm107 Bf16Gemm) | base_red (same, reduce store
         into P) | ab_res (CUTLASS alpha/beta example, TMA-staged P, patched
         math) | ab_s_red (example D = bf16(acc + S) reduce-stored into P: the
         latent_up form bf16(P + bf16(S + acc)))
  MXFP8: nvjet_mx (torch._scaled_mm) | mx_<tactic> / mx_<tactic>_red (sm107
         PlainGemm, plain / reduce store into P)
Checks (bitwise): reduce arms == bf16(f32(P0) + f32(plain output of the same
host)).
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
MX_TACTICS = {
    "c2x2": ((512, 256), (2, 2), 256, 8, "n", 0, 0, ""),
    "c2x1": ((512, 256), (2, 1), 256, 8, "n", 0, 0, ""),
    "c2x4": ((512, 256), (2, 4), 256, 4, "m", 0, 0, ""),
}


def nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception:
        return None, None


PYNVML, NVH = None, None


def energy_mj():
    if NVH is None:
        return None
    return PYNVML.nvmlDeviceGetTotalEnergyConsumption(NVH)


def sm_clock():
    if NVH is None:
        return None
    return PYNVML.nvmlDeviceGetClockInfo(NVH, PYNVML.NVML_CLOCK_SM)


def load_example(name: str, math_residual: bool, reduce_store: bool):
    import importlib.util
    import tempfile

    import flashinfer

    path = os.path.join(
        os.path.dirname(flashinfer.__file__),
        "data/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py",
    )
    src = open(path).read().replace(
        'get_smem_capacity_in_bytes("sm_100")', 'get_smem_capacity_in_bytes("sm_107")'
    )
    if math_residual:
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
    if reduce_store:
        n = src.count("cpasync.CopyBulkTensorTileS2GOp()")
        assert n == 1, f"S2G store op matched {n} times"
        src = src.replace(
            "cpasync.CopyBulkTensorTileS2GOp()", "cpasync.CopyReduceBulkTensorTileS2GOp()"
        )
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

        self.cutlass = cutlass
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


def time_arms(fns, sets, secs, rounds, thrash_n, flush):
    arms = list(fns)
    res = {a: [] for a in arms}
    ej = {a: [] for a in arms}
    clk = {a: [] for a in arms}
    thr = {a: [] for a in arms}
    for r in range(rounds):
        for a in arms if r % 2 == 0 else arms[::-1]:
            f = fns[a]
            for i in range(4):
                f(sets[i % len(sets)])
            torch.cuda.synchronize()
            n = 0
            e0 = energy_mj()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.time()
            ev0.record()
            while time.time() - t0 < secs:
                for _ in range(10):
                    f(sets[n % len(sets)])
                    n += 1
                torch.cuda.synchronize()
            ev1.record()
            torch.cuda.synchronize()
            e1 = energy_mj()
            clk[a].append(sm_clock())
            res[a].append(ev0.elapsed_time(ev1) * 1e3 / n)
            if e0 is not None and e1 is not None:
                ej[a].append((e1 - e0) / n)
            # THRASH: isolated calls after a 512 MB L2 flush.
            tt = []
            for i in range(thrash_n):
                flush.zero_()
                e_a, e_b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e_a.record()
                f(sets[i % len(sets)])
                e_b.record()
                torch.cuda.synchronize()
                tt.append(e_a.elapsed_time(e_b) * 1e3)
            thr[a].append(statistics.median(tt))
    out = {}
    for a in arms:
        out[a] = {
            "sus_us": round(statistics.median(res[a]), 1),
            "sus_all": [round(v, 1) for v in res[a]],
            "thrash_us": round(statistics.median(thr[a]), 1),
            "mJ": round(statistics.median(ej[a]), 1) if ej[a] else None,
            "clk": clk[a],
        }
    return out


def main():
    global PYNVML, NVH
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", default="16384:3584,8192:3584,16384:12288", help="rows:K list")
    p.add_argument("--mx", default="16384:6144,16384:12288", help="rows:K list")
    p.add_argument("--mx-tactics", default="c2x2,c2x1")
    p.add_argument("--sets", type=int, default=2)
    p.add_argument("--secs", type=float, default=1.5)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--thrash", type=int, default=8)
    p.add_argument("--no-example", action="store_true")
    args = p.parse_args()
    PYNVML, NVH = nvml()
    torch.manual_seed(0)
    torch.zeros(1, device="cuda")
    from cutlass.cute.nvgpu import cpasync

    print(json.dumps({"reduce_op": hasattr(cpasync, "CopyReduceBulkTensorTileS2GOp"),
                      "power_limit_W": PYNVML.nvmlDeviceGetPowerManagementLimit(NVH) / 1000
                      if NVH is not None else None}), flush=True)
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import bf16_gemm
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        dense_gemm_bf16_sm107 as dg,
    )

    class _RedKernel(dg.PersistentDenseGemmKernel):
        c_store_reduce = True

    class Bf16GemmRed(bf16_gemm.Bf16Gemm):
        kernel_cls = _RedKernel

    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    base, base_red = bf16_gemm.Bf16Gemm(), Bf16GemmRed()
    ab_res = ab_sum = ab_s_red = None
    if not args.no_example:
        ab_res = AbGemm(load_example("ab_res_x", True, False))
        ab_sum = AbGemm(load_example("ab_sum_x", False, False))
        ab_s_red = AbGemm(load_example("ab_s_red_x", False, True))

    def chk(name, got, ref):
        d = got.float() - ref.float()
        return {name + "_bitwise": bool(torch.equal(got, ref)),
                name + "_mism": int((got != ref).sum()),
                name + "_maxabs": float(d.abs().max())}

    for spec in [s for s in args.bf16.split(",") if s]:
        T, K = (int(v) for v in spec.split(":"))
        sets = []
        for _ in range(args.sets):
            x = torch.randn(T, K, device="cuda").bfloat16()
            w = (torch.randn(N, K, device="cuda") * K**-0.5).bfloat16()
            pre = (torch.randn(T, N, device="cuda") * 4).bfloat16()
            s = torch.randn(T, N, device="cuda").bfloat16()
            sets.append((x, w, pre, torch.empty(T, N, device="cuda", dtype=torch.bfloat16), s))
        x, w, pre, y, s = sets[0]
        check = {}
        base(x, w, y)
        pp = pre.clone()
        base_red(x, w, pp)
        check.update(chk("base_red", pp, (pre.float() + y.float()).bfloat16()))
        if ab_res is not None:
            pp = pre.clone()
            ab_res(x, w, pp, pp)
            check.update(chk("ab_res", pp, (pre.float() + y.float()).bfloat16()))
            d2 = torch.empty_like(pre)
            ab_sum(x, w, s, d2)
            pp = pre.clone()
            ab_s_red(x, w, s, pp)
            check.update(chk("ab_s_red", pp, (pre.float() + d2.float()).bfloat16()))
        torch.cuda.synchronize()
        fns = {
            "nvjet": lambda st: torch.mm(st[0], st[1].t(), out=st[3]),
            "nvjet_beta": lambda st: torch.addmm(st[2], st[0], st[1].t(), out=st[2]),
            "base": lambda st: base(st[0], st[1], st[3]),
            "base_red": lambda st: base_red(st[0], st[1], st[2]),
        }
        if ab_res is not None:
            fns["ab_res"] = lambda st: ab_res(st[0], st[1], st[2], st[2])
            fns["ab_s_red"] = lambda st: ab_s_red(st[0], st[1], st[4], st[2])
        out = time_arms(fns, sets, args.secs, args.rounds, args.thrash, flush)
        print(json.dumps({"dtype": "bf16", "rows": T, "K": K, "check": check, "arms": out}), flush=True)
        del sets

    if args.mx:
        from flashinfer import mxfp8_quantize

        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
            kda_conv_gemm as kcg,
        )

        for spec in [s for s in args.mx.split(",") if s]:
            T, K = (int(v) for v in spec.split(":"))
            sets = []
            for _ in range(args.sets):
                xb = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
                wb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
                xq, xs = mxfp8_quantize(xb, is_sf_swizzled_layout=True)
                wq, ws = mxfp8_quantize(wb, is_sf_swizzled_layout=True)
                del xb, wb
                pre = (torch.randn(T, N, device="cuda") * 4).bfloat16()
                sets.append((xq, xs.view(torch.uint8).reshape(-1), wq,
                             ws.view(torch.uint8).reshape(-1), pre,
                             torch.empty(T, N, device="cuda", dtype=torch.bfloat16)))
            fns = {
                "nvjet_mx": lambda st: torch._scaled_mm(
                    st[0], st[2].t(), scale_a=st[1].view(torch.float8_e8m0fnu),
                    scale_b=st[3].view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16),
            }  # fmt: skip
            check = {}
            gemms = {}
            for tn in args.mx_tactics.split(","):
                tac = MX_TACTICS[tn]
                red = tac[:7] + ((tac[7] + "+red").lstrip("+"),)
                for nm, t in ((f"mx_{tn}", tac), (f"mx_{tn}_red", red)):
                    gemms[nm] = {i: kcg.PlainGemm(st[2], st[3], t) for i, st in enumerate(sets)}
                xq, xs, wq, ws, pre, y = sets[0]
                gemms[f"mx_{tn}"][0](xq, xs, y)
                pp = pre.clone()
                gemms[f"mx_{tn}_red"][0](xq, xs, pp)
                check.update(chk(f"mx_{tn}_red", pp, (pre.float() + y.float()).bfloat16()))

                def mk(nm, inplace):
                    def f(st, nm=nm, inplace=inplace):
                        i = 0 if st is sets[0] else 1
                        gemms[nm][i](st[0], st[1], st[4] if inplace else st[5])
                    return f

                fns[f"mx_{tn}"] = mk(f"mx_{tn}", False)
                fns[f"mx_{tn}_red"] = mk(f"mx_{tn}_red", True)
            torch.cuda.synchronize()
            out = time_arms(fns, sets, args.secs, args.rounds, args.thrash, flush)
            print(json.dumps({"dtype": "mxfp8", "rows": T, "K": K, "check": check, "arms": out}),
                  flush=True)
            del sets, gemms


if __name__ == "__main__":
    main()
