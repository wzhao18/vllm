# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR-3 decision microbench: SP all-gather pulled by puller CTAs that run on
the same SMs as the KDA in_proj Q|K|V GEMM (TP8+DCP8 SP prefill, VR200).

Question: with the KDA in_proj all-gather pulled over NVLink by puller warps
into the GEMM's operand buffer, does the GEMM slow down by <= 5% while the AG
is fully hidden?

Proxy for the fused kernel (no fused kernel yet): one puller CTA (1 issuing
thread, a ring of NS smem slots) per SM, launched beside the production sm_107
Q|K|V GEMM (``kda_conv_gemm.PlainGemm`` at the production TP8 tactic) on the
same SMs. Each puller TMA-bulk-loads peer MXFP8 shard chunks (data + F8_128x4
scale bytes) into smem and bulk-stores them into this rank's gathered slot (the
existing SP AG slot layout: zero extra memory). It waits for each source
rank's ready flag (epoch, st.release.sys, raised after an optional injected
skew) with a bounded spin that traps (no silent hang). The proxy shares SM
issue, smem ports, the TMA unit, L2/HBM and NVLink with the GEMM exactly as an
in-kernel puller warp would; it does not model the GEMM's per-tile waits (the
hidden condition is checked from the per-chunk landing stamps instead).

Modes per iteration (ABAB-rotated, every mode every iteration):
  today : Triton push (_push_shard2_kernel, production) + symm barrier, then GEMM
  gemm  : GEMM alone (operand already gathered)
  pull  : puller alone
  co    : puller launched, then the GEMM (both after a common start)
  coG   : GEMM launched, then the puller

Emulation on 1 node (4 GPUs): ``--logical 8`` keeps the 8-rank gathered M
(16384 / 32768 rows); GPU g owns logical shards {g, g+4}; it pulls the 6
shards of its 3 peers over NVLink (6/7 of the 8-rank ingress) and copies its
partner shard locally. On 2 nodes x 4 GPUs (``--logical 8`` = world) every
shard comes over NVLink (the real 8-rank form).

Run: torchrun --nproc-per-node 4 bench_pr3_ag_puller.py --rows 2048 --logical 8
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

H = 7168
N_QKV = 4608  # TP8 q|k|v (3 x 1536)
SF_VEC = 32

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smid() { uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ uint32_t ld_acq_sys(const uint32_t* p) { uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel_sys(uint32_t* p, uint32_t v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

template <int NS>
__global__ void __launch_bounds__(32, 1) puller_kernel(
    const int64_t* __restrict__ csrc, const int64_t* __restrict__ cdst, const int32_t* __restrict__ cbytes,
    const int32_t* __restrict__ crank, int nchunks, uint8_t* local_base, const int64_t* __restrict__ flag_addr,
    uint32_t epoch, int chunk_max, uint64_t* stamps, uint64_t* cta, uint64_t spin_limit_ns) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0) return;
  const uint64_t t0 = gtime();
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int G = gridDim.x, b = blockIdx.x;
  const int n = b < nchunks ? (nchunks - b + G - 1) / G : 0;
  uint32_t seen = 0, ph = 0;  // ph: parity bit per slot
  auto issue = [&](int k) {
    const int c = b + k * G;
    const int r = crank[c];
    if (!((seen >> r) & 1u)) {
      const uint32_t* f = (const uint32_t*)flag_addr[r];
      const uint64_t ts = gtime();
      while (ld_acq_sys(f) != epoch) {
        __nanosleep(64);
        if (gtime() - ts > spin_limit_ns) __trap();
      }
      seen |= 1u << r;
    }
    const int s = k % NS;
    const uint32_t nb = (uint32_t)cbytes[c];
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(smem + (size_t)s * chunk_max)), "l"(csrc[c]), "r"(nb), "r"(sa(&bar[s])) : "memory");
  };
  for (int k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int k = 0; k < n; ++k) {
    const int s = k % NS;
    const int c = b + k * G;
    {
      const uint32_t a = sa(&bar[s]), p = (ph >> s) & 1u;
      asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}" :: "r"(a), "r"(p) : "memory");
      ph ^= 1u << s;
    }
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(local_base + cdst[c]), "r"(sa(smem + (size_t)s * chunk_max)), "r"((uint32_t)cbytes[c]) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    // Store k-1 has read its slot once at most 1 group (store k) is pending.
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    if (k > 0) stamps[b + (k - 1) * G] = gtime();
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  const uint64_t t1 = gtime();
  if (n > 0) stamps[b + (n - 1) * G] = t1;
  cta[3 * b + 0] = t0;
  cta[3 * b + 1] = t1;
  cta[3 * b + 2] = smid();
}

__global__ void raise_flag_kernel(uint32_t* f, uint32_t epoch, uint64_t sleep_ns) {
  if (sleep_ns) { const uint64_t t = gtime(); while (gtime() - t < sleep_ns) __nanosleep(256); }
  __threadfence_system();
  st_rel_sys(f, epoch);
}
__global__ void stamp_kernel(uint64_t* p) { *p = gtime(); }
__global__ void sleep_kernel(uint64_t ns) { const uint64_t t = gtime(); while (gtime() - t < ns) __nanosleep(256); }

static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

void puller(torch::Tensor csrc, torch::Tensor cdst, torch::Tensor cbytes, torch::Tensor crank, int64_t local_base,
            torch::Tensor flag_addr, int64_t epoch, int64_t chunk_max, int64_t ns, int64_t grid, torch::Tensor stamps,
            torch::Tensor cta, int64_t spin_limit_ns) {
  const int n = (int)csrc.numel();
  const size_t smem = (size_t)ns * chunk_max;
#define L(NSV) { auto k = puller_kernel<NSV>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
  k<<<(int)grid, 32, smem, cur()>>>(csrc.data_ptr<int64_t>(), cdst.data_ptr<int64_t>(), cbytes.data_ptr<int32_t>(), \
    crank.data_ptr<int32_t>(), n, (uint8_t*)local_base, flag_addr.data_ptr<int64_t>(), (uint32_t)epoch, (int)chunk_max, \
    (uint64_t*)stamps.data_ptr<int64_t>(), (uint64_t*)cta.data_ptr<int64_t>(), (uint64_t)spin_limit_ns); }
  if (ns == 2) L(2) else if (ns == 3) L(3) else if (ns == 4) L(4) else TORCH_CHECK(false, "ns 2..4");
#undef L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void raise_flag(int64_t f, int64_t epoch, int64_t sleep_ns) {
  raise_flag_kernel<<<1, 1, 0, cur()>>>((uint32_t*)f, (uint32_t)epoch, (uint64_t)sleep_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void stamp(torch::Tensor p, int64_t i) { stamp_kernel<<<1, 1, 0, cur()>>>((uint64_t*)p.data_ptr<int64_t>() + i); }
void gpu_sleep(int64_t ns) { if (ns > 0) sleep_kernel<<<1, 1, 0, cur()>>>((uint64_t)ns); }
"""

_CPP = r"""
#include <torch/extension.h>
void puller(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, int64_t, int64_t,
            int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);
void raise_flag(int64_t, int64_t, int64_t);
void stamp(torch::Tensor, int64_t);
void gpu_sleep(int64_t);
"""


def build_ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="pr3_ag_puller_v1",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["puller", "raise_flag", "stamp", "gpu_sleep"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        verbose=False,
    )


def mxfp8(x):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    s = s.view(torch.uint8).reshape(-1)
    assert s.numel() == x.shape[0] * x.shape[1] // SF_VEC, s.numel()
    return q, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048, help="rows per logical rank")
    ap.add_argument("--logical", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--grid", type=int, default=0, help="puller CTAs (0 = all SMs)")
    ap.add_argument("--order", default="rr", choices=["rr", "seq"])
    ap.add_argument("--skew-us", type=float, default=0.0)
    ap.add_argument("--tactic-flags", default="")
    ap.add_argument("--local-src", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--pull-lag-us", type=float, default=15.0)
    ap.add_argument("--late", type=int, default=-1, help="late physical rank (-1 = last)")
    ap.add_argument("--modes", default="today,gemm,pull,co,coG,occ0,occ64")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    W = dist.get_world_size()
    me = dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", me % torch.cuda.device_count()))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    L = args.logical
    assert L % W == 0
    E = L // W  # logical shards per GPU
    R = args.rows
    M = L * R
    data_b = R * H
    sf_b = R * H // SF_VEC
    sf_base = L * R * H  # C1 slot layout: rows from 0, scales from L*max_rows*H
    slot_b = 2 * L * R * H  # the BF16 slot (C1 MXFP8 rows + scales fit inside)
    late = (W - 1) if args.late < 0 else args.late
    ext = build_ext()
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = args.grid or nsm

    def owner(l):
        return l % W

    mine = [l for l in range(L) if owner(l) == me]

    # Symmetric slot (+ 64 B per rank of flags in a second symmetric buffer).
    buf = symm_mem.empty(slot_b, dtype=torch.uint8, device=dev)
    hb = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    flags = symm_mem.empty(64, dtype=torch.uint32, device=dev)
    flags.zero_()
    hf = symm_mem.rendezvous(flags, dist.group.WORLD.group_name)
    bases = [hb.get_buffer(p, [slot_b], torch.uint8, 0).data_ptr() for p in range(W)]
    fbases = [hf.get_buffer(p, [64], torch.uint32, 0).data_ptr() for p in range(W)]
    flag_addr = torch.tensor(fbases, dtype=torch.int64, device=dev)

    # Same random BF16 rows on every rank (seeded per logical shard), quantized
    # per shard: the gathered tensor is bitwise mxfp8(all rows) (C1).
    ref_data = torch.empty(L * R, H, dtype=torch.uint8, device=dev)
    ref_sf = torch.empty(L * sf_b, dtype=torch.uint8, device=dev)
    for l in range(L):
        g = torch.Generator(device=dev).manual_seed(1000 + l)
        x = torch.randn(R, H, generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)
        q, s = mxfp8(x)
        ref_data[l * R : (l + 1) * R].copy_(q.view(torch.uint8))
        ref_sf[l * sf_b : (l + 1) * sf_b].copy_(s)
    raw = buf
    for l in range(L):  # every row real data (timed passes never clear)
        raw[l * data_b : (l + 1) * data_b].copy_(ref_data[l * R : (l + 1) * R].reshape(-1))
        raw[sf_base + l * sf_b : sf_base + (l + 1) * sf_b].copy_(ref_sf[l * sf_b : (l + 1) * sf_b])
    A = raw[: M * H].view(torch.float8_e4m3fn).view(M, H)
    A_sf = raw[sf_base : sf_base + L * sf_b]

    # GEMM: production sm_107 kernel at the TP8 q|k|v tactic.
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import tactic_for
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (
        PlainGemm,
        shape_tactic,
    )

    gw = torch.Generator(device=dev).manual_seed(7)
    Wb = (torch.randn(N_QKV, H, generator=gw, device=dev) * 0.02).to(torch.bfloat16)
    wq, ws = mxfp8(Wb)
    tac = shape_tactic(N_QKV, tactic_for(N_QKV))
    if args.tactic_flags:
        # e.g. "ab4": cap the A/B smem stages (the smem an in-kernel puller
        # ring would take from the GEMM)
        tac = (*tac[:7], "+".join(f for f in (tac[7], args.tactic_flags) if f))
    gemm = PlainGemm(wq.view(torch.float8_e4m3fn), ws, tactic=tac)
    out = torch.empty(M, N_QKV, dtype=torch.bfloat16, device=dev)

    def run_gemm():
        gemm(A, A_sf, out)

    # Pull chunk list: every non-own logical shard (data + scales), sources =
    # the owner's slot; own-GPU partner shards (emulation) copy locally.
    chunks = []  # (src_addr, dst_off, bytes, src_rank, logical)
    # rotation: owners me+1, me+2, ...
    srcs = sorted(
        [l for l in range(L) if l not in mine],
        key=lambda l: ((owner(l) - me) % W, l),
    )
    per_src = []
    for l in srcs:
        q = owner(l)
        if args.local_src:
            q = me  # contention probe: same bytes, HBM/L2 only (no NVLink)
        lst = []
        # scales first (3% of the bytes): a row tile is usable once its data lands
        sl = mine[0] if args.local_src else l  # local probe reads an own shard
        for off in range(0, sf_b, args.chunk):
            nb = min(args.chunk, sf_b - off)
            lst.append((bases[q] + sf_base + sl * sf_b + off, sf_base + l * sf_b + off, nb, q, l))
        for off in range(0, data_b, args.chunk):
            nb = min(args.chunk, data_b - off)
            lst.append((bases[q] + sl * data_b + off, l * data_b + off, nb, q, l))
        per_src.append(lst)
    if args.push:
        # Push form: this GPU's own shards, TMA-loaded locally and bulk-stored
        # into every peer's slot (absolute remote addresses, local_base 0).
        per_src = []
        for k in range(1, W):
            q = (me + k) % W
            for l in mine:
                lst = []
                for off in range(0, sf_b, args.chunk):
                    nb = min(args.chunk, sf_b - off)
                    lst.append((bases[me] + sf_base + l * sf_b + off, bases[q] + sf_base + l * sf_b + off, nb, me, l))
                for off in range(0, data_b, args.chunk):
                    nb = min(args.chunk, data_b - off)
                    lst.append((bases[me] + l * data_b + off, bases[q] + l * data_b + off, nb, me, l))
                per_src.append(lst)
    if args.order == "seq":
        for lst in per_src:
            chunks += lst
    else:
        i = 0
        while any(per_src):
            for lst in per_src:
                if lst:
                    chunks.append(lst.pop(0))
    nch = len(chunks)
    t = lambda k, dt: torch.tensor([c[k] for c in chunks], dtype=dt, device=dev)  # noqa: E731
    csrc, cdst, cbytes, crank = t(0, torch.int64), t(1, torch.int64), t(2, torch.int32), t(3, torch.int32)
    e64 = torch.empty(0, dtype=torch.int64, device=dev)
    e32 = torch.empty(0, dtype=torch.int32, device=dev)
    stamps = torch.zeros(nch, dtype=torch.int64, device=dev)
    cta = torch.zeros(3 * grid, dtype=torch.int64, device=dev)
    ts = torch.zeros(8, dtype=torch.int64, device=dev)
    ingress = sum(c[2] for c in chunks)
    nonown = [(l * data_b, data_b) for l in range(L) if l not in mine] + [
        (sf_base + l * sf_b, sf_b) for l in range(L) if l not in mine
    ]

    # Today's push (production Triton kernel) of this GPU's shards to every peer.
    from vllm.models.kimi_k3.nvidia.ops.sp_all_gather import _push_shard2_kernel

    peer_ptrs = torch.tensor(bases, dtype=torch.int64, device=dev)

    def run_today():
        for l in mine:
            d = raw[l * data_b : (l + 1) * data_b].view(torch.int32)
            s = raw[sf_base + l * sf_b : sf_base + (l + 1) * sf_b].view(torch.int32)
            nblk = (d.numel() + 4095) // 4096 + (s.numel() + 4095) // 4096
            _push_shard2_kernel[(min(nsm, nblk),)](
                d, s, peer_ptrs, l * data_b // 4, (sf_base + l * sf_b) // 4,
                d.numel(), s.numel(), me, WORLD=W, BLOCK=4096, num_warps=4,
            )  # fmt: skip
        hb.barrier()

    sA = torch.cuda.Stream(device=dev)
    sB = torch.cuda.Stream(device=dev)
    epoch = [0]
    SPIN = 2_000_000_000  # 2 s: trap instead of hanging

    def clear_nonown():
        for off, nb in nonown:
            raw[off : off + nb].zero_()

    def run_mode(mode, check=False):
        # Only the correctness pass clears the gathered rows: timed passes
        # keep real (random) operands in every row, as the fused kernel reads
        # landed rows only (zero rows make the MMA cheaper).
        if check:
            clear_nonown()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        epoch[0] += 1
        ep = epoch[0]
        skew_ns = int(args.skew_us * 1000) if me == late else 0
        ev0 = torch.cuda.Event(enable_timing=True)
        evA = torch.cuda.Event(enable_timing=True)
        evB = torch.cuda.Event(enable_timing=True)
        cur = torch.cuda.current_stream()
        # Hold the GPU while the host enqueues every kernel of the mode, so no
        # host launch gap lands inside a timed span (the cute GEMM launch is
        # ~50-100 us of Python).
        ext.gpu_sleep(2_000_000)
        ev0.record(cur)
        sA.wait_stream(cur)
        sB.wait_stream(cur)
        res = {}
        if mode == "today":
            with torch.cuda.stream(sA):
                ext.gpu_sleep(skew_ns)
                ext.stamp(ts, 0)
                run_today()
                ext.stamp(ts, 1)
                run_gemm()
                ext.stamp(ts, 2)
                evA.record(sA)
            cur.wait_stream(sA)
        else:
            def launch_pull(lag_ns=0):
                with torch.cuda.stream(sB):
                    ext.raise_flag(fbases[me], ep, skew_ns)
                    ext.gpu_sleep(lag_ns)
                    ext.puller(csrc, cdst, cbytes, crank, 0 if args.push else raw.data_ptr(), flag_addr, ep,
                               args.chunk, args.ns, grid, stamps, cta, SPIN)  # fmt: skip
                    ext.stamp(ts, 4)
                    evB.record(sB)

            def launch_gemm(lag_ns=0):
                with torch.cuda.stream(sA):
                    ext.gpu_sleep(skew_ns + lag_ns)
                    ext.stamp(ts, 0)
                    run_gemm()
                    ext.stamp(ts, 2)
                    evA.record(sA)

            if mode == "gemm":
                launch_gemm()
            elif mode == "pull":
                launch_pull()
            elif mode == "co":
                launch_pull()
                launch_gemm(3000)  # pullers dispatch first
            elif mode in ("occ0", "occ64"):
                # Occupancy probe: GEMM first, then one empty CTA per SM with
                # 16 B / 64 KiB smem; CTAs that start before the GEMM ends sit
                # on SMs the GEMM leaves room on.
                launch_gemm()
                with torch.cuda.stream(sB):
                    ext.gpu_sleep(20_000)
                    ext.puller(e64, e64, e32, e32, raw.data_ptr(), flag_addr, ep,
                               16 if mode == "occ0" else 32768, 2, grid, stamps, cta, SPIN)  # fmt: skip
                    evB.record(sB)
            elif mode == "coG":
                # GEMM first; the pullers start after every GEMM CTA is
                # resident, so they land on the SMs the GEMM grid leaves idle.
                launch_gemm()
                launch_pull(int(args.pull_lag_us * 1000))
            cur.wait_stream(sA)
            cur.wait_stream(sB)
        torch.cuda.synchronize()
        tsv = ts.tolist()
        if mode in ("occ0", "occ64"):
            c = cta.view(grid, 3).tolist()
            early = [x for x in c if x[0] < tsv[2]]
            res["gemm_us"] = (tsv[2] - tsv[0]) / 1e3
            res["ctas_beside_gemm"] = len(early)
            res["sms_beside_gemm"] = len(set(x[2] for x in early))
            return res
        if mode in ("today", "gemm", "co", "coG"):
            res["gemm_us"] = (tsv[2] - (tsv[1] if mode == "today" else tsv[0])) / 1e3
            res["end_A_us"] = ev0.elapsed_time(evA) * 1e3
        if mode == "today":
            res["ag_us"] = (tsv[1] - tsv[0]) / 1e3
            res["chain_us"] = (tsv[2] - tsv[0]) / 1e3
        if mode in ("pull", "co", "coG"):
            c = cta.view(grid, 3).tolist()
            st = stamps.tolist()
            t_first = min(x[0] for x in c)
            t_last = max(x[1] for x in c)
            res["pull_us"] = (t_last - t_first) / 1e3
            res["pull_TBps"] = ingress / ((t_last - t_first) * 1e-9) / 1e12
            res["pull_cta_start_spread_us"] = (max(x[0] for x in c) - t_first) / 1e3
            res["sms"] = len(set(x[2] for x in c))
            res["end_B_us"] = ev0.elapsed_time(evB) * 1e3
            if mode in ("co", "coG"):
                g0, g1 = tsv[0], tsv[2]
                res["pull_start_minus_gemm_start_us"] = (t_first - g0) / 1e3
                res["pull_end_minus_gemm_end_us"] = (t_last - g1) / 1e3
                # Hidden check (arrival-order M schedule): rows landed vs rows
                # the GEMM needs at its pace (own shards first, at g0).
                rows_land = []
                for k, cc in enumerate(chunks):
                    off = cc[0] - bases[me] if args.push else cc[1]
                    if off < sf_base:  # data chunk
                        rows_land.append((st[k], cc[2] / H))
                rows_land.sort()
                gem = (g1 - g0)
                have = len(mine) * R
                stall = 0.0
                for tl, nr in rows_land:
                    need_t = g0 + gem * have / M  # when the GEMM reaches row #have
                    stall = max(stall, (tl - need_t) / 1e3)
                    have += nr
                res["stall_us_linear"] = stall
        if check and mode in ("today", "pull", "co", "coG") and not args.local_src:
            ok_d = torch.equal(raw[: M * H], ref_data.reshape(-1))
            ok_s = torch.equal(raw[sf_base : sf_base + L * sf_b], ref_sf)
            res["bitwise"] = bool(ok_d and ok_s)
        return res

    modes = args.modes.split(",")
    for _ in range(3):  # clock ramp
        for m in modes:
            run_mode(m)
    for m in modes:  # correctness
        r = run_mode(m, check=True)
        if "bitwise" in r and not r["bitwise"]:
            raise RuntimeError(f"rank {me} mode {m}: gathered bytes differ")
    samples = {m: [] for m in modes}
    for it in range(args.iters):
        order = modes[it % len(modes):] + modes[: it % len(modes)]
        if it % 2:
            order = order[::-1]
        for m in order:
            samples[m].append(run_mode(m))
    # per-rank medians, gathered to rank 0
    med = {}
    for m, lst in samples.items():
        keys = lst[0].keys()
        med[m] = {k: statistics.median(x[k] for x in lst) for k in keys if k != "bitwise"}
    allmed = [None] * W
    dist.all_gather_object(allmed, med)
    if me == 0:
        rec = {
            "tag": args.tag, "W": W, "L": L, "rows": R, "M": M, "chunk": args.chunk, "ns": args.ns,
            "grid": grid, "order": args.order, "skew_us": args.skew_us, "late": late,
            "ingress_MB": ingress / 1e6, "iters": args.iters, "per_rank": allmed,
            "host": os.uname().nodename, "time": time.strftime("%FT%T"),
        }  # fmt: skip
        summ = {}
        for m in modes:
            summ[m] = {}
            for k in allmed[0][m]:
                v = [r[m][k] for r in allmed]
                summ[m][k] = {"max": max(v), "med": statistics.median(v), "min": min(v)}
        rec["summary"] = summ
        g = summ.get("gemm", {}).get("gemm_us", {}).get("med")
        line = [f"PR3 {args.tag} tactic={tac} W{W} L{L} R{R} chunk{args.chunk} ns{args.ns} grid{grid} {args.order} skew{args.skew_us} local{int(args.local_src)} push{int(args.push)} flags={args.tactic_flags}"]
        for m in modes:
            s = summ[m]
            parts = [f"{k}={s[k]['med']:.1f}/{s[k]['max']:.1f}" for k in s]
            if g and "gemm_us" in s and m != "gemm":
                parts.append(f"slow={100 * (s['gemm_us']['med'] / g - 1):+.1f}%")
            line.append(f"  {m}: " + " ".join(parts))
        print("\n".join(line), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
