# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the SP decode MoE front (D-MOEFRONT), one GPU, CUDA graphs.

Per "layer": an AttnRes-like producer (7 x n CTAs of 128 threads, ~2 KB smem,
triggers its dependents at entry, busy ~T us, then writes h) followed by
  A) ltpin router|latent FP32 GEMM + route_stage   (int5 chain at n <= 72)
  B) moe_front (one kernel)
L layers with distinct weights (cold in L2), graph-replayed; reported: per-layer
time minus the producer-only graph, i.e. producer end -> front staged.
Also a stamped moe_front run: per-CTA phase times after the producer's end.

python bench_moe_front.py [--rows 32,64,96] [--layers 8] [--busy-us 6]
       [--cfg C,wst,xst,nprod,xg ...]
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load_inline

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import moe_front, route_stage

E, LATENT, K, CAP, TOPK = 896, 3584, 7168, 128, 16

_PROD = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void __launch_bounds__(128) prod_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
    int n, long long busy_ns, unsigned long long* end_stamp) {
  __shared__ float pad[448];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  pad[threadIdx.x] = 0.f;
  unsigned long long t = t0;
  while ((long long)(t - t0) < busy_ns) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  const int row = blockIdx.y, part = blockIdx.x;
  const int col = (part * 128 + threadIdx.x) * 8;
  if (row < n && col < 7168) {
    *reinterpret_cast<uint4*>(dst + (long long)row * 7168 + col) =
        *reinterpret_cast<const uint4*>(src + (long long)row * 7168 + col);
  }
  if (end_stamp && threadIdx.x == 0) {
    unsigned long long e;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(e));
    atomicMax(end_stamp, e + (unsigned long long)pad[0]);
  }
}
void prod(torch::Tensor src, torch::Tensor dst, int64_t busy_ns, torch::Tensor end, int64_t carve) {
  static int64_t set = -2;
  if (carve != set) { TORCH_CHECK(cudaFuncSetAttribute(prod_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int)carve) == cudaSuccess); set = carve; }
  const int n = src.size(0);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(7, n);
  cfg.blockDim = dim3(128);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, prod_kernel,
      (const __nv_bfloat16*)src.data_ptr(), (__nv_bfloat16*)dst.data_ptr(), n, (long long)busy_ns,
      end.numel() ? (unsigned long long*)end.data_ptr() : nullptr) == cudaSuccess);
}
"""


def _prod_mod():
    root = os.environ.get("VLLM_CACHE_ROOT", "/tmp")
    build = os.path.join(root, "k3_moe_front_bench_prod")
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name="k3_moe_front_bench_prod", cpp_sources="void prod(torch::Tensor src, torch::Tensor dst, int64_t busy_ns, torch::Tensor end, int64_t carve);",
        cuda_sources="#include <cuda_bf16.h>\n" + _PROD, functions=["prod"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        build_directory=build, verbose=False,
    )  # fmt: skip


def _session(dev):
    x = torch.zeros((CAP, LATENT // 2), dtype=torch.uint8, device=dev)
    sf = torch.zeros((CAP, LATENT // 16 + 4), dtype=torch.uint8, device=dev)
    idx = torch.zeros((CAP, TOPK), dtype=torch.int32, device=dev)
    sw = torch.zeros((CAP, TOPK), dtype=torch.float32, device=dev)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    return x, sf, idx, sw, err


def _time_graph(fn, iters=50, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / iters)
    out.sort()
    return out[len(out) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,64,96")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--busy-us", type=float, default=6.0)
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--carveout", type=int, default=-1, help="producer preferred smem carveout (-1 default)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    pm = _prod_mod()
    fm = moe_front._module()
    rm = route_stage._module()
    L = args.layers
    ws = [(torch.randn((E + LATENT, K), device=dev) * 0.02).to(torch.bfloat16) for _ in range(L)]
    bias = (torch.randn(E, device=dev) * 0.1).float()
    sess = _session(dev)
    nostamp = torch.empty(0, dtype=torch.int64, device=dev)
    busy = int(args.busy_us * 1000)
    carve = args.carveout
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"SMs {sms}; layers {L}; producer busy {args.busy_us} us; carveout {carve}")
    print(f"  moe_front registers/local bytes: {fm.moe_front_attrs()}")
    for c in range(2, 9):
        sm_ = fm.moe_front_smem(32, 8, 2, 2)
        print(f"  max co-resident clusters C={c} smem {sm_ // 1024}K: {fm.moe_front_max_clusters(c, sm_)}")
    for n in [int(r) for r in args.rows.split(",")]:
        src = torch.randn((n, K), device=dev).to(torch.bfloat16)
        h = torch.empty_like(src)
        y32 = torch.empty((n, E + LATENT), device=dev)
        yr = torch.empty((n, E), device=dev)
        cnt = torch.zeros(4, dtype=torch.int32, device=dev)

        def producer_only():
            for _ in range(L):
                pm.prod(src, h, busy, nostamp, carve)

        t_p = _time_graph(producer_only)
        line = f"n{n}: producer {t_p / L:.2f} us/layer"
        if not args.no_base:
            try:
                from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.ltpin import LtPin

                lt = LtPin("router_latent", E + LATENT, K, (n,), out32=True, device=dev)

                def base():
                    for i in range(L):
                        pm.prod(src, h, busy, nostamp, carve)
                        lt.run(ws[i], h, y32)
                        rm.route_stage(y32, E, bias, *sess, None, E, True, 2.827, True, 1.0)

                t_b = _time_graph(base)
                line += f" | ltpin+route_stage {(t_b - t_p) / L:.2f}"
            except Exception as e:  # noqa: BLE001
                line += f" | base n/a ({type(e).__name__}: {str(e)[:80]})"
        cfgs = [tuple(int(v) for v in c.split(",")) for c in args.cfg] or [moe_front.config_for(n)]
        for c in cfgs:
            csize, wst, xst, nprod, xg, wg = c
            smem = fm.moe_front_smem((n + 15) // 16 * 16, wst * wg, xst, xg)
            mc = fm.moe_front_max_clusters(csize, smem)

            def front(c=c):
                for i in range(L):
                    pm.prod(src, h, busy, nostamp, carve)
                    fm.moe_front(ws[i], h, yr, cnt, bias, *sess, None, E, E, True, 2.827, True,
                                 1.0, *c, nostamp)  # fmt: skip

            t_f = _time_graph(front)
            line += f" | front{c} smem {smem // 1024}K clusters {mc} {(t_f - t_p) / L:.2f}"
            # Skeleton: same bytes, barriers, reduce and epilogue, no MMAs (probe bit 0).
            skel = torch.zeros(1, dtype=torch.int64, device=dev)

            def skeleton(c=c):
                for i in range(L):
                    pm.prod(src, h, busy, nostamp, carve)
                    fm.moe_front(ws[i], h, yr, cnt, bias, *sess, None, E, E, True, 2.827, True,
                                 1.0, *c, skel)  # fmt: skip

            t_s = _time_graph(skeleton)
            line += f" skeleton {(t_s - t_p) / L:.2f}"
        print(line, flush=True)
        # Stamped single layer (eager, after a warm call): phase times after the
        # producer's end, median over CTAs.
        c = cfgs[0]
        grid = 35 * c[0]
        st = torch.zeros(grid * 328 + args.debug, dtype=torch.int64, device=dev)
        end = torch.zeros(1, dtype=torch.int64, device=dev)
        def stamped():
            pm.prod(src, h, busy, end, carve)
            fm.moe_front(ws[0], h, yr, cnt, bias, *sess, None, E, E, True, 2.827, True, 1.0,
                         *c, st)  # fmt: skip

        stamped()
        g = torch.cuda.CUDAGraph()  # graph launch: no CPU gap between the two kernels
        with torch.cuda.graph(g):
            stamped()
        for _ in range(3):
            end.zero_()
            st.zero_()
            g.replay()
            torch.cuda.synchronize()
        sv = st[: grid * 328].view(grid, 328)
        s = (sv.double() - float(end.item())) / 1000.0
        names = ["entry", "x-wait", "lastMMA", "accum", "reduced", "topk-go", "end"]  # globaltimer
        parts = []
        for i, nm in enumerate(names):
            v = s[:, i]
            v = v[sv[:, i] != 0]
            if v.numel():
                parts.append(f"{nm} {v.median().item():.2f}/{v.max().item():.2f}")
        print(f"  stamps n{n} cfg{c} (us after producer end, median/max): " + ", ".join(parts),
              flush=True)
        for cta in (0, grid - 1):
            raw = sv[cta].double()
            # clock stamps -> us after the producer end via the CTA's entry/end pairs
            hz = (raw[199] - raw[7]) / max(float(raw[6] - raw[0]), 1.0)  # cycles per ns
            row = s[cta].clone()
            for i in list(range(8, 198)) + list(range(200, 328)):
                if sv[cta, i] != 0:
                    row[i] = s[cta, 0] + (raw[i] - raw[7]) / hz / 1000.0
            ok = sv[cta] != 0
            print(f"  cta{cta} clock {hz.item():.3f} GHz; W slots landed at dependency release: {int(sv[cta, 198]):b}")
            row[198] = 0
            xr = [f"{row[8 + i].item():.1f}" for i in range(64) if ok[8 + i]]
            wr = [f"{row[72 + i].item():.1f}" for i in range(64) if ok[72 + i]]
            wi = [f"{row[136 + i].item():.1f}" for i in range(64) if ok[136 + i]]
            mi = [f"{row[200 + i].item():.1f}" for i in range(64) if ok[200 + i]]
            print(f"  cta{cta} mma-issued {' '.join(mi)}")
            xi = [f"{row[264 + i].item():.2f}" for i in range(64) if ok[264 + i]]
            print(f"  cta{cta} x-issue {' '.join(xi)}")
            print(f"  cta{cta} x-ready {' '.join(xr)}")
            print(f"  cta{cta} W-ready {' '.join(wr)}")
            print(f"  cta{cta} W-issue {' '.join(wi)}", flush=True)


if __name__ == "__main__":
    main()
