# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 release DSpark drafter (TP4, D3-32 decode): data-movement-only skeleton
of the sliding-window GQA attention with DIE-LOCAL KV (SOL protocol, no math).

The bytes are the deployed ones: per layer every request reads its 2048-key window
of its 4 KV heads from the BF16 draft cache in the "rows" layout (one draft block
per BLHNC row: [heads 4, page 4096, K|V 128] per layer, the five layers' pages
inside a row of stride 5 x 4 MiB + 128 B, first page at +128 B), 512 KiB per
(request, head), ~201 MiB per layer at 96 requests.

Kernel (persistent, one CTA per SM): P issuers per CTA, each one producer lane with
its own ring of S stages of SB bytes (1D cp.async.bulk into shared memory, mbarrier
complete_tx, the form a tcgen05 attention mainloop consumes) and one consumer warp.
Work items = (request, head, byte range of at most U bytes), split at 2 MiB address
boundaries so each item is homed on one die. An issuer claims items from its die's
list (die = %smid -> sm_domain_table), claims the next item while it streams the
current one, and steals from the other die's list once its own is empty. At an
item's last stage the consumer warp writes the item's partial (18 query rows x 64
FP32 + LSE) and the segment's last item merges the partials (split-KV merge), as
the real kernel must.

Placement arms: striped (cudaMalloc, one global list), local (2 MiB chunks
alternating dies, die-matched claim, +/- stealing), swapped (local placement, map
inverted: all-remote; must be clearly slower than striped, else the map/claim is
wrong), local-unmatched (local placement, one global list).

usage: python bench_kimi_k3_sdec_gqa_dielocal_skel.py [--requests 32,64,96]
"""

import argparse
import hashlib
import os
import random

import torch

HEADS = 4  # KV heads per rank (TP4)
PAGE = 4096  # tokens per draft page ("rows" layout at TP4)
TOKEN = 256  # bytes per token per head: K 64 + V 64 BF16
WINDOW = 2048
LAYERS = 5
LAYER_PAGE = HEADS * PAGE * TOKEN  # 4 MiB
ROW_STRIDE = LAYERS * LAYER_PAGE + 256  # deployed: + 128 (rows 256 B aligned here for the 3-D map)
FIRST = 0
CHUNK = 2 << 20
PART = 1184  # floats per partial: 18 rows x 64 + 18 LSE, padded
MAXJ = 16
SOL_TBS = 22e12  # model t_sol rate
GATE_US = {32: None, 64: None, 96: 13.6}

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdint.h>

#define PART 1184
#define MAXJ 16

struct Item { unsigned long long addr; unsigned bytes; unsigned segj; };

struct Args {
  const Item* items;      // [2][cap] (this layer)
  const int* nitems;      // [2] (this layer)
  int cap;
  int* ctr;               // [2] claim counters (this layer)
  int* seg_ctr;           // [nseg] (this layer)
  const int* seg_nj;      // [nseg]
  float* partial;         // [nseg * MAXJ][PART]
  float* out;             // [nseg][PART]
  const int* sm_die;      // [1024]
  int match, swap, steal, P, S, SB, statik;
  const int* sm_rank;     // [1024] rank of the SM among its die (match) or all SMs
  int n_on0, n_on1;       // SMs per die (match) or all SMs in n_on0
  unsigned long long* stamps;  // [layers][grid * 4][3]: producer start, producer end, consumer end
  const void* tmap;       // 3-D BF16 map (64, K|V, rows of 256 B) of the cache, 128 B swizzle
  unsigned long long base; // cache base (row = (addr - base) / 256)
  int form, layer;        // form 0: 1-D bulk, 1: tensor TMA boxes of SB / 256 rows
  unsigned* sink;
};

__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, unsigned n, unsigned b) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(b) : "memory");
}

__device__ __forceinline__ int claim(const Args& a, int& list, int die) {
  if (list < 0) return -1;
  int i = atomicAdd(&a.ctr[list], 1);
  if (i < a.nitems[list]) return list * a.cap + i;
  if (a.steal && list == die) {
    list = 1 - die;
    i = atomicAdd(&a.ctr[list], 1);
    if (i < a.nitems[list]) return list * a.cap + i;
  }
  list = -1;
  return -1;
}

__global__ void __launch_bounds__(256) skel(Args a) {
  extern __shared__ __align__(1024) unsigned char ring[];
  __shared__ __align__(8) unsigned long long full[64], empty[64];
  __shared__ int meta[64];
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  int iss = w >> 1;
  bool producer = !(w & 1);
  int die = 0, rdie = 0;
  unsigned sm = smid() & 1023;
  if (a.match) { rdie = a.sm_die[sm] & 1; die = rdie ^ (a.swap & 1); }
  const int G = (rdie ? a.n_on1 : a.n_on0) * a.P;
  const int gi = a.sm_rank[sm] * a.P + iss;
  unsigned long long* st = a.stamps + ((size_t)(a.layer * gridDim.x + blockIdx.x) * 4 + iss) * 3;
  if (producer && lane == 0 && iss < a.P) for (int i = 0; i < a.S; ++i) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[iss * a.S + i])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[iss * a.S + i])));
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (iss >= a.P) return;
  unsigned char* myring = ring + (size_t)iss * a.S * a.SB;
  unsigned long long *myfull = full + iss * a.S, *myempty = empty + iss * a.S;
  int* mymeta = meta + iss * a.S;
  if (producer) {
    if (lane != 0) return;
    int list = a.match ? die : 0;
    long long i = 0;
    int k = 0;
    st[0] = gtime();
    int it;
    if (a.statik) { it = gi < a.nitems[list] ? list * a.cap + gi : -1; ++k; } else it = claim(a, list, die);
    while (true) {
      int nxt = -1;
      if (it >= 0) {
        if (a.statik) { int idx = gi + k * G; ++k; nxt = idx < a.nitems[list] ? list * a.cap + idx : -1; }
        else nxt = claim(a, list, die);
      }
      unsigned bytes = 0; const unsigned char* src = nullptr;
      if (it >= 0) { Item I = a.items[it]; bytes = I.bytes; src = (const unsigned char*)I.addr; }
      int nst = it >= 0 ? (int)((bytes + a.SB - 1) / a.SB) : 1;
      for (int q = 0; q < nst; ++q, ++i) {
        int sl = (int)(i % a.S); long long round = i / a.S;
        if (round > 0) wait_par(sa(&myempty[sl]), (unsigned)((round - 1) & 1));
        unsigned b = sa(&myfull[sl]);
        if (it < 0) {
          mymeta[sl] = -1;
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
        } else {
          unsigned n = a.form ? (unsigned)a.SB : min((unsigned)a.SB, bytes - (unsigned)q * a.SB);
          mymeta[sl] = it * 2 + (q == nst - 1);
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory");
          unsigned dst = sa(myring + (size_t)sl * a.SB);
          if (a.form) {
            int row = (int)(((unsigned long long)src - a.base) >> 8) + q * (a.SB >> 8);
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(dst), "l"(a.tmap), "r"(0), "r"(0), "r"(row), "r"(b) : "memory");
          } else {
            bulk(dst, src + (size_t)q * a.SB, n, b);
          }
        }
      }
      if (it < 0) break;
      it = nxt;
    }
    st[1] = gtime();
  } else {
    unsigned acc = 0;
    long long i = 0;
    while (true) {
      int sl = (int)(i % a.S);
      wait_par(sa(&myfull[sl]), (unsigned)((i / a.S) & 1));
      int m = mymeta[sl];
      acc += myring[(size_t)sl * a.SB + 4 * lane];
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&myempty[sl])) : "memory");
      ++i;
      if (m < 0) break;
      if (m & 1) {
        unsigned segj = a.items[m >> 1].segj;
        int seg = segj / MAXJ;
        float* p = a.partial + (size_t)segj * PART;
        for (int x = lane; x < PART; x += 32) p[x] = (float)(acc + x);
        __threadfence();
        int old = 0;
        if (lane == 0) old = atomicAdd(&a.seg_ctr[seg], 1);
        old = __shfl_sync(0xffffffffu, old, 0);
        if (old == a.seg_nj[seg] - 1) {
          __threadfence();
          int nj = a.seg_nj[seg];
          float* o = a.out + (size_t)seg * PART;
          for (int x = lane; x < PART; x += 32) {
            float s = 0.f;
            for (int j = 0; j < nj; ++j) s += __ldcg(a.partial + ((size_t)seg * MAXJ + j) * PART + x);
            o[x] = s;
          }
          if (lane == 0) o[0] = (float)nj;
        }
      }
    }
    if (lane == 0) st[2] = gtime();
    if (acc == 0x12345678u) a.sink[0] = acc;
  }
}

void launch(int64_t items, int64_t nitems, int64_t cap, int64_t ctr, int64_t seg_ctr, int64_t seg_nj,
            int64_t partial, int64_t out, int64_t sm_die, int64_t match, int64_t swap, int64_t steal,
            int64_t P, int64_t S, int64_t SB, int64_t grid, int64_t smem, int64_t sink,
            int64_t statik, int64_t sm_rank, int64_t n_on0, int64_t n_on1, int64_t stamps,
            int64_t tmap, int64_t base, int64_t form, int64_t layer) {
  Args a;
  a.items = (const Item*)items; a.nitems = (const int*)nitems; a.cap = (int)cap;
  a.ctr = (int*)ctr; a.seg_ctr = (int*)seg_ctr; a.seg_nj = (const int*)seg_nj;
  a.partial = (float*)partial; a.out = (float*)out; a.sm_die = (const int*)sm_die;
  a.match = (int)match; a.swap = (int)swap; a.steal = (int)steal;
  a.P = (int)P; a.S = (int)S; a.SB = (int)SB; a.sink = (unsigned*)sink;
  a.statik = (int)statik; a.sm_rank = (const int*)sm_rank; a.n_on0 = (int)n_on0; a.n_on1 = (int)n_on1;
  a.stamps = (unsigned long long*)stamps;
  a.tmap = (const void*)tmap; a.base = (unsigned long long)base; a.form = (int)form; a.layer = (int)layer;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(skel, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024); init = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, skel, a);
  TORCH_CHECK(e == cudaSuccess, "skel launch: ", cudaGetErrorString(e));
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void launch(int64_t items, int64_t nitems, int64_t cap, int64_t ctr, int64_t seg_ctr, int64_t seg_nj,
            int64_t partial, int64_t out, int64_t sm_die, int64_t match, int64_t swap, int64_t steal,
            int64_t P, int64_t S, int64_t SB, int64_t grid, int64_t smem, int64_t sink,
            int64_t statik, int64_t sm_rank, int64_t n_on0, int64_t n_on1, int64_t stamps,
            int64_t tmap, int64_t base, int64_t form, int64_t layer);
"""

_mod = None


def _module():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode()).hexdigest()[:16]
        root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        build = os.path.join(root, "k3_gqa_dielocal_skel", tag)
        os.makedirs(build, exist_ok=True)
        _mod = load_inline(
            name=f"k3_gqa_dielocal_skel_{tag}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["launch"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
            build_directory=build,
        )
    return _mod


def segments(n_req, n_blocks, rng):
    """(address offset, bytes) of every (request, head) window per layer."""
    blocks = rng.sample(range(n_blocks), n_req)
    out = []
    for layer in range(LAYERS):
        segs = []
        for r in range(n_req):
            start = rng.randrange(0, PAGE - WINDOW + 1)
            row = FIRST + blocks[r] * ROW_STRIDE + layer * LAYER_PAGE
            for h in range(HEADS):
                segs.append((row + h * PAGE * TOKEN + start * TOKEN, WINDOW * TOKEN))
        out.append(segs)
    return out


def items_for(segs, base, unit, match):
    """Per-die item lists [(addr, bytes, segj)] and items per segment."""
    lists, nj = ([], []), []
    for s, (off, n) in enumerate(segs):
        j, pos, end = 0, off, off + n
        while pos < end:
            stop = min(end, pos + unit, (pos // CHUNK + 1) * CHUNK)
            die = (pos // CHUNK) & 1 if match else 0
            lists[die].append((base + pos, stop - pos, s * MAXJ + j))
            j, pos = j + 1, stop
        assert j <= MAXJ
        nj.append(j)
    return lists, nj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="32,64,96")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--units", default="256,512")
    ap.add_argument("--forms", default="t1x64x3,t2x32x3,t4x16x3,t2x48x2,b2x32x3")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    mod = _module()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    table = dies.sm_domain_table(dev)
    die_sms = [int((table[:sms] == d).sum()) for d in (0, 1)]
    bad = dies.stream_check(dev.index)
    print(f"GPU sms={sms} per-die {die_sms} stream_check_disagree={list(bad)}", flush=True)
    sm_die = torch.zeros(1024, dtype=torch.int32, device=dev)
    sm_die[: table.numel()] = table
    n_blocks = 128
    nbytes = FIRST + n_blocks * ROW_STRIDE
    bufs = {
        "striped": torch.empty(nbytes, dtype=torch.uint8, device=dev),
        "local": dies.localized_empty(nbytes, [0, 1], dev),
    }
    for b in bufs.values():
        b.random_(0, 255)
    assert bufs["local"].data_ptr() % CHUNK == 0
    arms = [  # name, buffer, match, swap, steal, static (rank-in-die schedule, no atomics)
        ("striped-static", "striped", 0, 0, 0, 1),
        ("local-static", "local", 1, 0, 0, 1),
        ("swapped-static", "local", 1, 1, 0, 1),
        ("local-dyn", "local", 1, 0, 1, 0),
        ("striped-dyn", "striped", 0, 0, 0, 0),
    ]
    smids = list(range(sms))
    dies_of = [int(x) for x in table[:sms].tolist()]
    rank_match = torch.zeros(1024, dtype=torch.int32)
    seen = [0, 0]
    for s in smids:
        rank_match[s] = seen[dies_of[s]]
        seen[dies_of[s]] += 1
    rank_match = rank_match.to(dev)
    rank_all = torch.arange(1024, dtype=torch.int32, device=dev)
    stamps = torch.zeros(LAYERS * sms * 4 * 3, dtype=torch.int64, device=dev)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import tensor_maps
    import cuda.bindings.driver as cuda

    maps = {}
    for bname, b in bufs.items():
        for sbk in (16, 32, 48, 64):
            maps[(bname, sbk)] = tensor_maps([(
                b.data_ptr(), cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                (64, 2, b.numel() // 256), (128, 256), (64, 2, (sbk << 10) // 256),
                cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
                cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B)], dev)
    rng = random.Random(1234)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    for n_req in [int(x) for x in args.requests.split(",")]:
        segs = segments(n_req, n_blocks, rng)
        nseg = n_req * HEADS
        total = sum(n for _, n in segs[0])
        sol_us = total / SOL_TBS * 1e6
        partial = torch.zeros(nseg * MAXJ * PART, dtype=torch.float32, device=dev)
        out = torch.zeros(nseg * PART, dtype=torch.float32, device=dev)
        best = {}
        for unit_k in [int(x) for x in args.units.split(",")]:
            for arm, bname, match, swap, steal, statik in arms:
                base = bufs[bname].data_ptr()
                per_layer = [items_for(s, base, unit_k << 10, match) for s in segs]
                cap = max(max(len(l[0][0]), len(l[0][1])) for l in per_layer)
                rows = [[[[addr, n | (segj << 32)] for addr, n, segj in lists[d]]
                         + [[0, 0]] * (cap - len(lists[d])) for d in (0, 1)]
                        for lists, _ in per_layer]
                it = torch.tensor(rows, dtype=torch.int64, device=dev)
                nit = torch.tensor([[len(lists[d]) for d in (0, 1)] for lists, _ in per_layer],
                                   dtype=torch.int32, device=dev)
                njs = torch.tensor([p[1] for p in per_layer], dtype=torch.int32, device=dev)
                ctrs = torch.zeros(LAYERS * 2 + LAYERS * nseg, dtype=torch.int32, device=dev)
                imbalance = max(int(nit[:, 0].sum()), int(nit[:, 1].sum())) / max(
                    1, int(nit.sum()) / 2) if match else 1.0
                for form in args.forms.split(","):
                    tform = int(form[0] == "t")
                    P, sbk, S = (int(x) for x in form[1:].split("x"))
                    tmap = maps[(bname, sbk)].data_ptr() if tform else 0
                    SB = sbk << 10
                    smem = max(P * S * SB, 116 << 10)
                    if P * S * SB > (200 << 10) or P * S > 64:
                        continue

                    def chain():
                        ctrs.zero_()
                        for li in range(LAYERS):
                            mod.launch(
                                it[li].data_ptr(), nit[li].data_ptr(), cap,
                                ctrs[li * 2:].data_ptr(),
                                ctrs[LAYERS * 2 + li * nseg:].data_ptr(),
                                njs[li].data_ptr(), partial.data_ptr(), out.data_ptr(),
                                sm_die.data_ptr(), match, swap, steal, P, S, SB, sms, smem,
                                sink.data_ptr(), statik,
                                (rank_match if match else rank_all).data_ptr(),
                                seen[0] if match else sms, seen[1] if match else 0,
                                stamps.data_ptr(), tmap, bufs[bname].data_ptr(), tform, li)

                    out.zero_()
                    chain()
                    torch.cuda.synchronize()
                    done = out.view(nseg, PART)[:, 0]
                    ok = bool((done == njs[LAYERS - 1].float()).all())
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        chain()
                    for _ in range(3):
                        g.replay()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ts = []
                    for _ in range(5):
                        e0.record()
                        for _ in range(args.replays):
                            g.replay()
                        e1.record()
                        torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1e3 / (args.replays * LAYERS))
                    t = sorted(ts)[len(ts) // 2]
                    tbs = total / t / 1e6
                    key = arm
                    if key not in best or t < best[key][0]:
                        best[key] = (t, f"U{unit_k}K {form}")
                    sva = stamps.view(LAYERS, sms, 4, 3)[:, :, :P].reshape(LAYERS, -1, 3).cpu()
                    spans, gaps = [], []
                    for li in range(LAYERS):
                        v = sva[li][sva[li][:, 0] > 0]
                        spans.append((int(v[:, 2].max()) - int(v[:, 0].min())) / 1e3)
                        if li:
                            gaps.append((int(v[:, 0].min()) - prev_end) / 1e3)
                        prev_end = int(v[:, 2].max())
                    sv = sva[LAYERS - 1][sva[LAYERS - 1][:, 0] > 0]
                    t0 = int(sv[:, 0].min())
                    dur = ((sv[:, 2] - sv[:, 0]).float() / 1e3).sort().values
                    stamps.zero_()
                    print(f"N{n_req} R{n_req * 4} {arm} U{unit_k}K {form}: "
                          f"{t:.2f} us/layer {tbs:.1f} TB/s imb {imbalance:.3f} "
                          f"{'ok' if ok else 'FAIL'} | last layer: span "
                          f"{(int(sv[:, 2].max()) - t0) / 1e3:.2f} start-spread "
                          f"{(int(sv[:, 0].max()) - t0) / 1e3:.2f} issuer p50 "
                          f"{float(dur[len(dur) // 2]):.2f} max {float(dur[-1]):.2f} us",
                          flush=True)
        gate = GATE_US.get(n_req)
        for arm, (t, cfg) in best.items():
            print(f"BEST N{n_req} R{n_req * 4} {arm}: {t:.2f} us/layer ({cfg}) "
                  f"{total / t / 1e6:.1f} TB/s; sol {sol_us:.2f} us (bytes {total / 1e6:.1f} MB)"
                  + (f"; gate {gate}" if gate else ""), flush=True)


if __name__ == "__main__":
    main()
