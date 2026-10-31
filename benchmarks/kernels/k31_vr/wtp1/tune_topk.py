"""Variants of the balanced top-k filter pass (vllm/v1/attention/ops/balanced_topk_cuda.py) on the owner shapes.

The stock filter_kernel serializes, per work item, a 32 B/thread load -> warp scan -> one 64-bit atomic -> scatter,
so each warp has one item in flight. Variant `ilp<N>`: the loads of N work items are issued before any of them is
processed (N x 32 B in flight per thread). usage: python3 tune_topk.py --shapes <shapes.json>"""
import argparse, hashlib, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(__file__))
from bench_topk import time_graph  # noqa: E402

from vllm.v1.attention.ops import balanced_topk_cuda as B  # noqa: E402

FILTER2 = r"""
// Rows longer than CAP, as a flattened (row, block) work list. ITEMS work items per iteration: all their loads
// are issued first, then each is compacted (scan + one 64-bit atomic per warp + scatter).
__device__ __forceinline__ void filter_item(const uint16_t (&v)[16], int col0, int n, int r, uint32_t t0, int lane,
                                            unsigned long long* __restrict__ counters,
                                            unsigned long long* __restrict__ cand, int* __restrict__ ties,
                                            int tie_cap) {
  constexpr int PER = 16;
  uint32_t gt = 0, eq = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if (col0 + i < n) {
      const uint32_t k = key16(v[i]);
      gt |= (uint32_t)(k > t0) << i;
      eq |= (uint32_t)(k == t0) << i;
    }
  }
  const unsigned long long mine =
      ((unsigned long long)__popc(eq) << 32) | (unsigned long long)__popc(gt);
  unsigned long long incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(FULL, incl, o);
    if (lane >= o) incl += y;
  }
  const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
  if (warp_total == 0ull) return;
  unsigned long long base = 0;
  if (lane == 31) base = atomicAdd(counters + r, warp_total);
  base = __shfl_sync(FULL, base, 31);
  const unsigned long long excl = base + incl - mine;
  int pos_gt = (int)(excl & 0xFFFFFFFFull);
  int pos_eq = (int)(excl >> 32);
  unsigned long long* crow = cand + (int64_t)r * CAP;
  int* trow = ties + (int64_t)r * tie_cap;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if ((gt >> i) & 1u) {
      if (pos_gt < CAP)
        crow[pos_gt] = ((unsigned long long)key16(v[i]) << 32) | (unsigned long long)(uint32_t)(col0 + i);
      ++pos_gt;
    } else if ((eq >> i) & 1u) {
      if (pos_eq < tie_cap) trow[pos_eq] = col0 + i;
      ++pos_eq;
    }
  }
}

__global__ void __launch_bounds__(FILTER_THREADS, MINB)
filter_kernel(const uint16_t* __restrict__ logits, int64_t stride,
              const int* __restrict__ seq_lens, int next_n,
              int width, int num_rows, bool per_row,
              const int* __restrict__ thresh,
              unsigned long long* __restrict__ counters,
              unsigned long long* __restrict__ cand,
              int* __restrict__ ties, int tie_cap,
              bool vec_ok) {
  extern __shared__ int prefix[];  // num_rows + 1
  for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
    const int n = row_len(seq_lens, r, next_n, width, per_row);
    prefix[r] = n > CAP ? (n + BLOCK - 1) / BLOCK : 0;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int carry = 0;
    for (int r0 = 0; r0 < num_rows; r0 += 32) {
      const int r = r0 + lane;
      const int c = r < num_rows ? prefix[r] : 0;
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, incl, o);
        if (lane >= o) incl += y;
      }
      if (r < num_rows) prefix[r] = carry + incl - c;
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) prefix[num_rows] = carry;
  }
  __syncthreads();
  const int total = prefix[num_rows];
  const int lane = threadIdx.x & 31;
  constexpr int PER = BLOCK / FILTER_THREADS;  // 16
  static_assert(PER == 16, "16 keys per thread per item");
  for (int w0 = blockIdx.x; w0 < total; w0 += ITEMS * gridDim.x) {
    uint16_t v[ITEMS][16];
    int rr[ITEMS], nn[ITEMS], cc[ITEMS];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      const int w = w0 + it * gridDim.x;
      rr[it] = -1;
      if (w >= total) continue;
      int lo = 0, hi = num_rows - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (prefix[mid] <= w) lo = mid; else hi = mid - 1;
      }
      const int r = lo;
      rr[it] = r;
      nn[it] = row_len(seq_lens, r, next_n, width, per_row);
      const uint16_t* row = logits + (int64_t)r * stride;
      const int col0 = (w - prefix[r]) * BLOCK + threadIdx.x * PER;
      cc[it] = col0;
      if (vec_ok && col0 + PER <= nn[it]) {
        const uint4* p = reinterpret_cast<const uint4*>(row + col0);
        const uint4 a = __ldcs(p), b = __ldcs(p + 1);
        const uint16_t* pa = reinterpret_cast<const uint16_t*>(&a);
        const uint16_t* pb = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
        for (int i = 0; i < 8; ++i) { v[it][i] = pa[i]; v[it][8 + i] = pb[i]; }
      } else {
#pragma unroll
        for (int i = 0; i < PER; ++i) v[it][i] = col0 + i < nn[it] ? row[col0 + i] : 0;
      }
    }
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      if (rr[it] < 0) continue;  // uniform per CTA (w depends on the block only)
      filter_item(v[it], cc[it], nn[it], rr[it], (uint32_t)thresh[rr[it]], lane, counters, cand, ties, tie_cap);
    }
  }
}

"""


def variant_src(items, minb):
    s = B._CUDA_SRC
    a = s.index("// Rows longer than CAP")
    b = s.index("// Writes a selected column")
    body = FILTER2.replace("ITEMS", str(items)).replace("MINB", str(minb))
    return s[:a] + body + s[b:]


CONST_VARIANTS = {
    "tgt4096": [("constexpr int TARGET = 8192;", "constexpr int TARGET = 4096;")],
    "tgt4096s8k": [("constexpr int TARGET = 8192;", "constexpr int TARGET = 4096;"),
                   ("constexpr int SAMPLES = 4096;", "constexpr int SAMPLES = 8192;"),
                   ("constexpr int SAMPLE_THREADS = 256;  // 16 samples per thread", "constexpr int SAMPLE_THREADS = 512;")],
    "ft512": [("constexpr int BLOCK = 4096;", "constexpr int BLOCK = 8192;"),
              ("constexpr int FILTER_THREADS = 256;", "constexpr int FILTER_THREADS = 512;")],
    "ft128": [("constexpr int BLOCK = 4096;", "constexpr int BLOCK = 2048;"),
              ("constexpr int FILTER_THREADS = 256;", "constexpr int FILTER_THREADS = 128;")],
    # one 64-bit atomic per CTA work item (block scan of the 8 warp totals) instead of one per warp
    "cta_atomic": [("""    const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
    if (warp_total == 0ull) continue;
    unsigned long long base = 0;
    if (lane == 31) base = atomicAdd(counters + r, warp_total);
    base = __shfl_sync(FULL, base, 31);
    const unsigned long long excl = base + incl - mine;""", """    const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
    __shared__ unsigned long long wsum[FILTER_THREADS / 32 + 1];
    const int warp = threadIdx.x >> 5;
    if (lane == 31) wsum[warp] = warp_total;
    __syncthreads();
    if (threadIdx.x < 32) {
      unsigned long long x = lane < FILTER_THREADS / 32 ? wsum[lane] : 0ull;
      unsigned long long ix = x;
#pragma unroll
      for (int o = 1; o < FILTER_THREADS / 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(FULL, ix, o);
        if (lane >= o) ix += y;
      }
      const unsigned long long tot = __shfl_sync(FULL, ix, FILTER_THREADS / 32 - 1);
      unsigned long long b = 0;
      if (lane == 0 && tot) b = atomicAdd(counters + r, tot);
      b = __shfl_sync(FULL, b, 0);
      if (lane < FILTER_THREADS / 32) wsum[lane] = b + ix - x;
    }
    __syncthreads();
    const unsigned long long base = wsum[warp];
    __syncthreads();
    if (warp_total == 0ull) continue;
    const unsigned long long excl = base + incl - mine;""")],
    # timing bound only (wrong output): the filter loads + compares but never compacts
    "diag_nocompact": [("    if (warp_total == 0ull) continue;\n    unsigned long long base = 0;",
                        "    if (warp_total != 0xFFFFFFFFFFFFull) continue;\n    unsigned long long base = 0;")],
}


GRID2D = r"""
// Rows longer than CAP: one CTA per (4096-key block, row) on a 2D grid; blocks past a row's end exit at once.
// No per-CTA prefix scan / binary search (the persistent work list's fixed cost).
__global__ void filter_kernel(const uint16_t* __restrict__ logits, int64_t stride,
                              const int* __restrict__ seq_lens, int next_n,
                              int width, int num_rows, bool per_row,
                              const int* __restrict__ thresh,
                              unsigned long long* __restrict__ counters,
                              unsigned long long* __restrict__ cand,
                              int* __restrict__ ties, int tie_cap,
                              bool vec_ok) {
  const int r = blockIdx.y;
  const int n = row_len(seq_lens, r, next_n, width, per_row);
  if (n <= CAP || (int)blockIdx.x * BLOCK >= n) return;
  const int lane = threadIdx.x & 31;
  constexpr int PER = BLOCK / FILTER_THREADS;  // 16
  const uint32_t t0 = (uint32_t)thresh[r];
  const uint16_t* row = logits + (int64_t)r * stride;
  const int col0 = blockIdx.x * BLOCK + threadIdx.x * PER;
  uint16_t v[PER];
  if (vec_ok && col0 + PER <= n) {
    const uint4* p = reinterpret_cast<const uint4*>(row + col0);
    const uint4 a = __ldg(p), b = __ldg(p + 1);
    const uint16_t* pa = reinterpret_cast<const uint16_t*>(&a);
    const uint16_t* pb = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = pa[i]; v[8 + i] = pb[i]; }
  } else {
#pragma unroll
    for (int i = 0; i < PER; ++i) v[i] = col0 + i < n ? row[col0 + i] : 0;
  }
  uint32_t gt = 0, eq = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if (col0 + i < n) {
      const uint32_t k = key16(v[i]);
      gt |= (uint32_t)(k > t0) << i;
      eq |= (uint32_t)(k == t0) << i;
    }
  }
  const unsigned long long mine =
      ((unsigned long long)__popc(eq) << 32) | (unsigned long long)__popc(gt);
  unsigned long long incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(FULL, incl, o);
    if (lane >= o) incl += y;
  }
  const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
  if (warp_total == 0ull) return;
  unsigned long long base = 0;
  if (lane == 31) base = atomicAdd(counters + r, warp_total);
  base = __shfl_sync(FULL, base, 31);
  const unsigned long long excl = base + incl - mine;
  int pos_gt = (int)(excl & 0xFFFFFFFFull);
  int pos_eq = (int)(excl >> 32);
  unsigned long long* crow = cand + (int64_t)r * CAP;
  int* trow = ties + (int64_t)r * tie_cap;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if ((gt >> i) & 1u) {
      if (pos_gt < CAP)
        crow[pos_gt] = ((unsigned long long)key16(v[i]) << 32) | (unsigned long long)(uint32_t)(col0 + i);
      ++pos_gt;
    } else if ((eq >> i) & 1u) {
      if (pos_eq < tie_cap) trow[pos_eq] = col0 + i;
      ++pos_eq;
    }
  }
}

"""


STAGED = r"""
// Rows longer than CAP: one CTA per (CHUNK x 4096-key span, row) on a 2D grid. Candidates (and ties) are
// compacted into shared memory with shared atomics, then the CTA reserves its global slots with ONE 64-bit
// atomic and copies them out: the global atomic's round trip is paid once per CTA, not once per warp item.
// A CTA whose candidates overflow its shared buffers adds CAP + 1 to the row's count, which sends the row to
// select_kernel's exact streamed fallback (the same path a misjudged sample takes).
constexpr int CHUNK = CHUNK_ITEMS;
constexpr int SCAP = SCAP_N;   // staged candidates per CTA
constexpr int STIE = 1024;     // staged ties per CTA
__global__ void __launch_bounds__(FILTER_THREADS)
filter_kernel(const uint16_t* __restrict__ logits, int64_t stride,
              const int* __restrict__ seq_lens, int next_n,
              int width, int num_rows, bool per_row,
              const int* __restrict__ thresh,
              unsigned long long* __restrict__ counters,
              unsigned long long* __restrict__ cand,
              int* __restrict__ ties, int tie_cap,
              bool vec_ok) {
  __shared__ unsigned long long sc[SCAP];
  __shared__ int st[STIE];
  __shared__ unsigned long long sctr;
  __shared__ unsigned long long sbase;
  const int r = blockIdx.y;
  const int n = row_len(seq_lens, r, next_n, width, per_row);
  const int span0 = blockIdx.x * (CHUNK * BLOCK);
  if (n <= CAP || span0 >= n) return;
  if (threadIdx.x == 0) sctr = 0ull;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  constexpr int PER = BLOCK / FILTER_THREADS;  // 16
  const uint32_t t0 = (uint32_t)thresh[r];
  const uint16_t* row = logits + (int64_t)r * stride;
  uint4 na, nb;
  auto load = [&](int col0, uint4& a, uint4& b) {
    if (vec_ok && col0 + PER <= n) {
      const uint4* p = reinterpret_cast<const uint4*>(row + col0);
      a = __ldcs(p); b = __ldcs(p + 1);
    } else {
      uint16_t t[PER];
#pragma unroll
      for (int i = 0; i < PER; ++i) t[i] = col0 + i < n ? row[col0 + i] : 0;
      a = *reinterpret_cast<uint4*>(t); b = *reinterpret_cast<uint4*>(t + 8);
    }
  };
  int col0 = span0 + threadIdx.x * PER;
  if (col0 < n) load(col0, na, nb);
#pragma unroll 1
  for (int it = 0; it < CHUNK; ++it) {
    if (span0 + it * BLOCK >= n) break;  // uniform per CTA
    const uint4 a = na, b = nb;
    const int c0 = col0;
    col0 += BLOCK;
    if (it + 1 < CHUNK && col0 < n) load(col0, na, nb);  // prefetch the next item
    uint16_t v[PER];
    const uint16_t* pa = reinterpret_cast<const uint16_t*>(&a);
    const uint16_t* pb = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = pa[i]; v[8 + i] = pb[i]; }
    uint32_t gt = 0, eq = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if (c0 + i < n) {
        const uint32_t k = key16(v[i]);
        gt |= (uint32_t)(k > t0) << i;
        eq |= (uint32_t)(k == t0) << i;
      }
    }
    const unsigned long long mine =
        ((unsigned long long)__popc(eq) << 32) | (unsigned long long)__popc(gt);
    unsigned long long incl = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(FULL, incl, o);
      if (lane >= o) incl += y;
    }
    const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
    if (warp_total == 0ull) continue;
    unsigned long long base = 0;
    if (lane == 31) base = atomicAdd(&sctr, warp_total);
    base = __shfl_sync(FULL, base, 31);
    const unsigned long long excl = base + incl - mine;
    int pos_gt = (int)(excl & 0xFFFFFFFFull);
    int pos_eq = (int)(excl >> 32);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if ((gt >> i) & 1u) {
        if (pos_gt < SCAP) sc[pos_gt] = ((unsigned long long)key16(v[i]) << 32) | (unsigned long long)(uint32_t)(c0 + i);
        ++pos_gt;
      } else if ((eq >> i) & 1u) {
        if (pos_eq < STIE) st[pos_eq] = c0 + i;
        ++pos_eq;
      }
    }
  }
  __syncthreads();
  const unsigned long long tot = sctr;
  const int ngt = (int)(tot & 0xFFFFFFFFull), neq = (int)(tot >> 32);
  if (tot == 0ull) return;
  if (ngt > SCAP || neq > STIE) {  // overflow: force the exact streamed fallback for this row
    if (threadIdx.x == 0) atomicAdd(counters + r, (unsigned long long)(CAP + 1));
    return;
  }
  if (threadIdx.x == 0) sbase = atomicAdd(counters + r, tot);
  __syncthreads();
  const int bg = (int)(sbase & 0xFFFFFFFFull), be = (int)(sbase >> 32);
  unsigned long long* crow = cand + (int64_t)r * CAP;
  int* trow = ties + (int64_t)r * tie_cap;
  for (int i = threadIdx.x; i < ngt; i += FILTER_THREADS)
    if (bg + i < CAP) crow[bg + i] = sc[i];
  for (int i = threadIdx.x; i < neq; i += FILTER_THREADS)
    if (be + i < tie_cap) trow[be + i] = st[i];
}

"""


def build_staged(chunk, scap):
    src = B._CUDA_SRC
    a = src.index("// Rows longer than CAP"); b = src.index("// Writes a selected column")
    src = src[:a] + STAGED.replace("CHUNK_ITEMS", str(chunk)).replace("SCAP_N", str(scap)) + src[b:]
    old = "filter_kernel<<<(int)filter_ctas, FILTER_THREADS, (num_rows + 1) * sizeof(int), stream>>>("
    assert old in src
    src = src.replace(old, f"filter_kernel<<<dim3((width + BLOCK * {chunk} - 1) / (BLOCK * {chunk}), num_rows), "
                           "FILTER_THREADS, 0, stream>>>(")
    return _load(src)


def build_grid2d(threads, readonly=False):
    src = B._CUDA_SRC
    a = src.index("// Rows longer than CAP"); b = src.index("// Writes a selected column")
    src = src[:a] + GRID2D + src[b:]
    old = "filter_kernel<<<(int)filter_ctas, FILTER_THREADS, (num_rows + 1) * sizeof(int), stream>>>("
    assert old in src
    src = src.replace(old, "filter_kernel<<<dim3((width + BLOCK - 1) / BLOCK, num_rows), FILTER_THREADS, 0, stream>>>(")
    if readonly:  # timing bound only: loads, no compaction (wrong output)
        o = "  uint32_t gt = 0, eq = 0;\n#pragma unroll\n  for (int i = 0; i < PER; ++i) {\n    if (col0 + i < n) {"
        assert o in src
        src = src.replace(o, "  { uint32_t x = 0; for (int i = 0; i < PER; ++i) x += v[i]; if (x == 0x7FFFFFFFu) counters[r] = 1; return; }\n" + o, 1)
    if threads != 256:
        src = src.replace("constexpr int BLOCK = 4096;", f"constexpr int BLOCK = {threads * 16};")
        src = src.replace("constexpr int FILTER_THREADS = 256;", f"constexpr int FILTER_THREADS = {threads};")
    return _load(src)


def build_const(name):
    src = B._CUDA_SRC
    for o, n in CONST_VARIANTS[name]:
        assert o in src, (name, o)
        src = src.replace(o, n)
    return _load(src)


def _load(src):
    d = hashlib.sha256(src.encode()).hexdigest()[:12]
    bd = os.path.join(os.environ.get("VLLM_CACHE_ROOT", "/tmp"), "wtp1_topk", d)
    os.makedirs(bd, exist_ok=True)
    return load_inline(name=f"wtp1_topk_{d}", cpp_sources="", cuda_sources=src,
                       extra_cuda_cflags=["-O3", "-lineinfo"], build_directory=bd, verbose=False)


def build(items, minb):
    src = variant_src(items, minb)
    d = hashlib.sha256(src.encode()).hexdigest()[:12]
    bd = os.path.join(os.environ.get("VLLM_CACHE_ROOT", "/tmp"), "wtp1_topk", d)
    os.makedirs(bd, exist_ok=True)
    return load_inline(name=f"wtp1_topk_{d}", cpp_sources="", cuda_sources=src,
                       extra_cuda_cflags=["-O3", "-lineinfo"], build_directory=bd, verbose=False)


def run(mod, logits, seq, q, out, topk, ctas_per_sm):
    num_rows = logits.shape[0]
    dev = logits.device
    tie_cap = 1 << (topk - 1).bit_length()
    cand = torch.empty(num_rows * B.CAP, dtype=torch.int64, device=dev)
    scratch = torch.empty(num_rows * (tie_cap + 3), dtype=torch.int32, device=dev)
    counters, thresh, ties = scratch.split([2 * num_rows, num_rows, num_rows * tie_cap])
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    mod.balanced_topk_cuda(logits, seq, q, out, topk, thresh, counters, cand, ties, ctas_per_sm * sms)
    return counters.view(torch.int64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shapes", required=True)
    ap.add_argument("--variants", default="stock:8,stock:4,stock:16,ilp1x8:8,ilp2x4:4,ilp2x8:8,ilp4x4:4,ilp4x2:2,ilp2x2:2")
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--json"); a = ap.parse_args()
    dev = torch.device("cuda"); TOPK = 2048
    res = []
    for c in json.load(open(a.shapes))["cases"]:
        q = c["requests"][0]["query_len"]; ctx = [r["kv_len"] for r in c["requests"]]
        rows = len(ctx) * q; width = (max(ctx) + 127) // 128 * 128
        g = torch.Generator(device=dev).manual_seed(0)
        logits = torch.randn(rows, width, device=dev, generator=g, dtype=torch.bfloat16)
        seq = torch.tensor(ctx, dtype=torch.int32, device=dev)
        ref = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
        valid = sum(ctx) * q * 2
        tr = time_graph(lambda: logits.sum(dtype=torch.float32))
        print(json.dumps(dict(case=c["name"], torch_full_read_us=tr, full_mb=logits.numel() * 2 / 1e6,
                              valid_mb=valid / 1e6)), flush=True)
        B.balanced_topk_cuda(logits, seq, q, ref, TOPK)
        refv = torch.sort(torch.gather(logits, 1, ref.clamp_min(0).long()).float(), dim=1).values
        for v in a.variants.split(","):
            name, cps = v.split(":"); cps = int(cps)
            if name == "stock":
                mod = B._module()
            elif name.startswith("staged"):
                ch, sc = name[6:].split("_"); mod = build_staged(int(ch), int(sc))
            elif name == "readonly":
                mod = build_grid2d(256, readonly=True)
            elif name.startswith("grid2d"):
                mod = build_grid2d(int(name[6:] or 256))
            elif name in CONST_VARIANTS:
                mod = build_const(name)
            else:
                items, minb = name[3:].split("x"); mod = build(int(items), int(minb))
            out = torch.empty_like(ref)
            t = time_graph(lambda: run(mod, logits, seq, q, out, TOPK, cps))
            run(mod, logits, seq, q, out, TOPK, cps); torch.cuda.synchronize()
            outv = torch.sort(torch.gather(logits, 1, out.clamp_min(0).long()).float(), dim=1).values
            uniq = all(out[i].unique().numel() == out.shape[1] for i in range(0, rows, 17))
            same = bool(torch.equal(outv, refv)) and uniq
            row = dict(case=c["name"], variant=v, rows=rows, us=t, same_values_as_stock=same)
            ctr = run(mod, logits, seq, q, out, TOPK, cps)
            torch.cuda.synchronize()
            ngt = (ctr & 0xFFFFFFFF).float(); neq = (ctr >> 32).float()
            row["cand_per_row"] = dict(mean=ngt.mean().item(), max=ngt.max().item(), min=ngt.min().item(),
                                       ties_mean=neq.mean().item(), over_cap=int((ngt > B.CAP).sum().item()),
                                       below_topk=int((ngt < TOPK).sum().item()))
            if a.kernels:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        run(mod, logits, seq, q, out, TOPK, cps)
                    torch.cuda.synchronize()
                row["kernels_us"] = {e.key[:40]: round(e.device_time_total / e.count, 1)
                                     for e in prof.key_averages() if e.device_time_total > 0}
            print(json.dumps(row), flush=True); res.append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
