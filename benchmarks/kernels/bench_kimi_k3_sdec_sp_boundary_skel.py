# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode (D3-32, TP4): data-movement-only skeleton of the fused SP
layer boundary (SOL protocol, no math).

No standalone exchange kernel: per layer the chain is four GEMM-shaped kernels,
each streaming its per-rank BF16 weight bytes (the producer/consumer GEMMs of the
boundary), with the exchange folded into their prologue / epilogue:

  A  o_proj [7168 x 3072]        epilogue pushes the RS partial rows (16 B
                                 st.volatile to the row owner's Lamport slot)
  B  latent_down [3584 x 7168]   prologue: owner CTA per row polls its 4 slots,
                                 re-arms them, post-AttnRes bytes, then mainloop
  C  latent_up [7168 x 3584]     epilogue: pre-AttnRes bytes, multimem.st of the
                                 owned rows into every rank's gather box
  D  kda in_proj [12448 x 7168]  prologue: polls the gathered rows (Lamport),
                                 local grid barrier, then mainloop

Weights stream with 16 B LDG (part before griddepcontrol.wait = the PDL weight
prefetch). Modes: full (weights + exchange), w (weights only), x (exchange only).
The boundary's exposed time per layer = full - w; the SOL gate compares it.

usage: torchrun --nproc-per-node 4 bench_kimi_k3_sdec_sp_boundary_skel.py
       [--rows 128,256,384] [--layers 24] [--replays 20] [--pre-kb 96]
"""

import argparse
import hashlib
import os

import torch
import torch.distributed as dist

H = 7168
WORDS = H // 2  # uint32 words per BF16 row
RMAX = 384
NMAX = RMAX // 4
# per-rank BF16 weight bytes of the boundary GEMMs (gemm.sp_policy)
W_BYTES = {
    "A": 7168 * 3072 * 2,
    "B": 3584 * 7168 * 2,
    "C": 7168 * 3584 * 2,
    "D": 12448 * 7168 * 2,
}
# model ATTNRES t_sol per layer (pre + post, us) at 22 TB/s -> bytes read per GPU
ATTNRES_SOL_US = {128: (4.3 + 3.8) / 24, 256: (8.5 + 7.5) / 24, 384: (12.7 + 11.2) / 24}

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdint.h>

#define SENT 0xFFFFFFFFu
#define WORDS 3584
#define SEGS 56

struct Prm {
  int kind, do_w, do_x, rank, R, n, parity, nmax, rmax, chunks, rsplit;
  const uint4* w; long long w16; long long pre16;
  unsigned long long peer0, peer1, peer2, peer3;
  uint32_t* box;            // my symmetric buffer: recv [2][4][nmax][WORDS], gather [2][rmax][WORDS]
  unsigned long long mc;    // multicast address of box
  const uint4* hist; long long hist16; long long ar16;  // AttnRes bytes per layer (both halves)
  uint4* hidden;            // [nmax][WORDS / 4]
  unsigned int* ctr;
  uint32_t* sink;
};

__device__ __forceinline__ uint32_t stream_w(const uint4* w, long long b, long long e) {
  uint32_t acc = 0;
  const long long step = (long long)blockDim.x * 8;
  for (long long i = b + threadIdx.x; i < e; i += step) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      long long j = i + (long long)u * blockDim.x;
      v[u] = j < e ? __ldcs(w + j) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].w;
  }
  return acc;
}

__device__ __forceinline__ uint4 ld_vol(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_vol(uint32_t* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st_mc(unsigned long long p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)),
                  "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w)) : "memory");
}
__device__ __forceinline__ bool armed(uint4 v) {
  return v.x == SENT || v.y == SENT || v.z == SENT || v.w == SENT;
}
__device__ __forceinline__ long long recv_ix(const Prm& p, int par, int src, int lrow) {
  return ((long long)(par * 4 + src) * p.nmax + lrow) * WORDS;
}
__device__ __forceinline__ long long gather_ix(const Prm& p, int par, int row) {
  return (long long)2 * 4 * p.nmax * WORDS + ((long long)par * p.rmax + row) * WORDS;
}

__global__ void __launch_bounds__(512) skel(Prm p) {
  const int G = gridDim.x, g = blockIdx.x;
  uint32_t acc = 0;
  long long per = (p.w16 + G - 1) / G;
  long long wb = (long long)g * per, we = min(p.w16, wb + per);
  if (wb > we) wb = we;
  long long wm = min(we, wb + p.pre16);
  if (p.do_w) acc ^= stream_w(p.w, wb, wm);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  const uint4 val = make_uint4(0x3F803F80u ^ p.parity, 0x3F803F81u, 0x3F803F82u, 0x3F803F83u);
  const int R = p.R, n = p.n, par = p.parity;
  if (p.kind == 0) {
    if (p.do_x) {  // re-arm gather[par ^ 1] (consumed by the previous layer's D)
      uint32_t* gb = p.box + gather_ix(p, par ^ 1, 0);
      long long tot = (long long)R * (WORDS / 4);
      uint4 s = make_uint4(SENT, SENT, SENT, SENT);
      for (long long i = (long long)g * blockDim.x + threadIdx.x; i < tot; i += (long long)G * blockDim.x)
        reinterpret_cast<uint4*>(gb)[i] = s;
    }
    const int lane = threadIdx.x & 15;
    const long long grp = ((long long)g * blockDim.x + threadIdx.x) >> 4;
    const long long ngrp = ((long long)G * blockDim.x) >> 4;
    const unsigned long long peers[4] = {p.peer0, p.peer1, p.peer2, p.peer3};
    for (int c = 0; c < p.chunks; ++c) {  // output tiles finish in `chunks` waves
      long long cb = wm + (we - wm) * c / p.chunks, ce = wm + (we - wm) * (c + 1) / p.chunks;
      if (p.do_w) acc ^= stream_w(p.w, cb, ce);
      if (!p.do_x) continue;
      long long ub = (long long)R * SEGS * c / p.chunks, ue = (long long)R * SEGS * (c + 1) / p.chunks;
      for (long long u = ub + grp; u < ue; u += ngrp) {
        int seg = (int)(u / R), row = (int)(u % R);
        int owner = row / n, lrow = row % n;
        long long off = recv_ix(p, par, p.rank, lrow) + seg * 64 + lane * 4;
        uint32_t* dst = reinterpret_cast<uint32_t*>(peers[owner]) + off;
        st_vol(dst, val);
      }
    }
  } else if (p.kind == 1) {
    if (p.do_x) {  // owner CTA per row: poll 4 slots, re-arm, write the sum, post-AttnRes bytes
      uint4 s = make_uint4(SENT, SENT, SENT, SENT);
      long long ar_row = p.ar16 / 2 / n / p.rsplit;
      const int W4 = WORDS / 4 / p.rsplit;
      for (int u = g; u < n * p.rsplit; u += G) {
        const int lrow = u / p.rsplit, w0 = (u % p.rsplit) * W4;
        for (int w4 = w0 + threadIdx.x; w4 < w0 + W4; w4 += blockDim.x) {
          uint4 sum = make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int src = 0; src < 4; ++src) {
            uint32_t* q = p.box + recv_ix(p, par, src, lrow) + w4 * 4;
            uint4 v;
            do { v = ld_vol(q); } while (armed(v));
            sum.x ^= v.x; sum.y ^= v.y; sum.z ^= v.z; sum.w ^= v.w;
            st_vol(q, s);
          }
          p.hidden[(long long)lrow * (WORDS / 4) + w4] = sum;
        }
        long long hb = ((long long)u * ar_row) % (p.hist16 - ar_row);
        acc ^= stream_w(p.hist, hb, hb + ar_row);
      }
    }
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
  } else if (p.kind == 2) {
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
    if (p.do_x) {  // epilogue: pre-AttnRes bytes, multicast the owned rows
      long long ar_row = p.ar16 / 2 / n / p.rsplit;
      const int W4 = WORDS / 4 / p.rsplit;
      for (int u = g; u < n * p.rsplit; u += G) {
        const int lrow = u / p.rsplit, w0 = (u % p.rsplit) * W4;
        long long hb = ((long long)(u + n * p.rsplit) * ar_row) % (p.hist16 - ar_row);
        acc ^= stream_w(p.hist, hb, hb + ar_row);
        long long base = gather_ix(p, par, p.rank * n + lrow) * 4;  // bytes
        for (int w4 = w0 + threadIdx.x; w4 < w0 + W4; w4 += blockDim.x)
          st_mc(p.mc + base + (unsigned long long)w4 * 16, val);
      }
    }
  } else {
    if (p.do_x) {  // prologue: poll the gathered rows, then a local grid barrier
      long long tot = (long long)R * (WORDS / 4);
      const uint32_t* gb = p.box + gather_ix(p, par, 0);
      for (long long i = (long long)g * blockDim.x + threadIdx.x; i < tot; i += (long long)G * blockDim.x) {
        uint4 v;
        do { v = ld_vol(gb + i * 4); } while (armed(v));
        acc ^= v.x;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        unsigned int old = atomicAdd(p.ctr, 1u);
        unsigned int target = (old / G + 1) * G;
        while (*(volatile unsigned int*)p.ctr < target) {}
      }
      __syncthreads();
    }
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
  }
  if (acc == 0x12345678u) p.sink[g] = acc;
}

void launch(int64_t kind, int64_t do_w, int64_t do_x, int64_t rank, int64_t R, int64_t parity,
            int64_t w, int64_t wbytes, int64_t pre_bytes, std::vector<int64_t> peers, int64_t box,
            int64_t mc, int64_t hist, int64_t hist_bytes, int64_t ar_bytes, int64_t hidden,
            int64_t ctr, int64_t sink, int64_t grid, int64_t threads, int64_t chunks, int64_t rsplit) {
  Prm p;
  p.chunks = chunks; p.rsplit = rsplit;
  p.kind = kind; p.do_w = do_w; p.do_x = do_x; p.rank = rank; p.R = R; p.n = R / 4;
  p.parity = parity; p.nmax = 96; p.rmax = 384;
  p.w = reinterpret_cast<const uint4*>(w); p.w16 = wbytes / 16; p.pre16 = pre_bytes / 16;
  p.peer0 = peers[0]; p.peer1 = peers[1]; p.peer2 = peers[2]; p.peer3 = peers[3];
  p.box = reinterpret_cast<uint32_t*>(box); p.mc = mc;
  p.hist = reinterpret_cast<const uint4*>(hist); p.hist16 = hist_bytes / 16; p.ar16 = ar_bytes / 16;
  p.hidden = reinterpret_cast<uint4*>(hidden);
  p.ctr = reinterpret_cast<unsigned int*>(ctr); p.sink = reinterpret_cast<uint32_t*>(sink);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = 0;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, skel, p));
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void launch(int64_t kind, int64_t do_w, int64_t do_x, int64_t rank, int64_t R, int64_t parity,
            int64_t w, int64_t wbytes, int64_t pre_bytes, std::vector<int64_t> peers, int64_t box,
            int64_t mc, int64_t hist, int64_t hist_bytes, int64_t ar_bytes, int64_t hidden,
            int64_t ctr, int64_t sink, int64_t grid, int64_t threads, int64_t chunks, int64_t rsplit);
"""

_mod = None


def _module():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode()).hexdigest()[:16]
        root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        build = os.path.join(root, "k3_sp_boundary_skel", tag)
        os.makedirs(build, exist_ok=True)
        _mod = load_inline(
            name=f"k3_sp_boundary_skel_{tag}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["launch"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
            build_directory=build,
        )
    return _mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--pre-kb", default="0,96,192")
    ap.add_argument("--chunks", default="1,2,4")
    ap.add_argument("--rsplit", default="1,2")
    ap.add_argument("--threads", type=int, default=512)
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--modes", default="full,w,x,sep")
    args = ap.parse_args()
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(dev)
    group = dist.group.WORLD
    import torch.distributed._symmetric_memory as symm_mem

    try:
        symm_mem.enable_symm_mem_for_group(group.group_name)
    except (AttributeError, ImportError):
        pass
    if rank == 0:
        _module()
    dist.barrier()
    mod = _module()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = args.grid or sms
    words = 2 * 4 * NMAX * WORDS + 2 * RMAX * WORDS
    box = symm_mem.empty((words,), dtype=torch.int32, device=dev)
    box.fill_(-1)
    handle = symm_mem.rendezvous(box, group.group_name)
    peers = [int(x) for x in handle.buffer_ptrs]
    mc = int(handle.multicast_ptr or 0)
    assert mc, "no multicast object"
    ncopy = 2
    wts = {k: [torch.empty(v, dtype=torch.uint8, device=dev).random_(0, 255)
               for _ in range(ncopy)] for k, v in W_BYTES.items()}
    hist_bytes = 64 << 20
    hist = torch.ones(hist_bytes, dtype=torch.uint8, device=dev)
    hidden = torch.empty(NMAX * H * 2, dtype=torch.uint8, device=dev)
    ctr = torch.zeros(args.layers, dtype=torch.int32, device=dev)
    sink = torch.zeros(grid, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    dist.barrier()
    res = {}
    for R in [int(r) for r in args.rows.split(",")]:
        ar_bytes = int(ATTNRES_SOL_US[R] * 1e-6 * 22e12) // 16 * 16
        cfgs = []
        for pre in [int(x) for x in args.pre_kb.split(",")]:
            for mode in args.modes.split(","):
                for ch in ([int(x) for x in args.chunks.split(",")] if mode == "full" else [1]):
                    if mode == "sep" and pre != 0:
                        continue
                    for rs in ([int(x) for x in args.rsplit.split(",")] if mode != "w" else [1]):
                        if mode == "x" and pre != int(args.pre_kb.split(",")[0]):
                            continue
                        cfgs.append((pre, mode, ch, rs))
        for pre, mode, ch, rs in cfgs:
            do_w = int(mode in ("full", "w"))
            do_x = int(mode in ("full", "x"))

            def chain():
                for i in range(args.layers):
                    par = i & 1
                    seq = ([(kd, do_w, do_x) for kd in range(4)] if mode != "sep" else
                           [(0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0),
                            (2, 1, 0), (2, 0, 1), (3, 0, 1), (3, 1, 0)])
                    for kind, dw, dx in seq:
                        k = "ABCD"[kind]
                        w = wts[k][i % ncopy]
                        mod.launch(kind, dw, dx, rank, R, par, w.data_ptr(), W_BYTES[k],
                                   pre * 1024, peers, box.data_ptr(), mc,
                                   hist.data_ptr(), hist_bytes, ar_bytes, hidden.data_ptr(),
                                   ctr.data_ptr() + 4 * i, sink.data_ptr(), grid, args.threads, ch, rs)

            box.fill_(-1)
            torch.cuda.synchronize()
            dist.barrier()
            chain()
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                chain()
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            times = []
            for _ in range(args.replays):
                dist.barrier()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                g.replay()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1e3 / args.layers)
            times.sort()
            t = torch.tensor([times[len(times) // 2]], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            res[(R, pre, mode, ch, rs)] = t.item()
            del g
        if rank == 0:
            for pre, mode, ch, rs in cfgs:
                t = res[(R, pre, mode, ch, rs)]
                w = res.get((R, pre, "w", 1, 1))
                ex = f" exposed {t - w:.2f}" if mode in ("full", "sep") and w else ""
                print(f"R{R} r{R // 16} pre{pre}K {mode} c{ch} rs{rs}: {t:.2f} us/layer{ex}",
                      flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
