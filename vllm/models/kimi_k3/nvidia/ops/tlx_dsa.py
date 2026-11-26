# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TLX-S for the DSA layers of a TP8 + DCP8 ShardQ prefill: token-local
projections with weight exchange.

Today a DSA layer all-gathers its BF16 input ([R, 7168] per rank -> [8R, 7168]),
runs the replicated q_a | kv_a | wk | weights GEMMs on all 8R rows, the
column-parallel q_b / g_proj for its 12 of 96 heads, exchanges the query
(head shards -> token shards) and the attention output back (ShardQ), and
reduce-scatters the row-parallel o_proj (GEMM-RS). At R >= ~700 rows per rank a
layer's weights are smaller than those activation exchanges, so here the
parallel boundary moves to the weights: each rank keeps its own R rows end to
end, and one push per layer carries

  * every rank's W_qb | W_g | W_o shard into every rank's staging region (the
    full q_b / g_proj / o_proj weights, 409 MB per GPU, of which 7/8 cross
    NVLink), and
  * the rank's new rows (normalized kv_c, k_pe and the indexer wk | weights
    projection, BF16) into every rank's receive buffer, so each rank inserts all
    8R new MLA / indexer rows into its paged cache and ShardQ workspace exactly
    as today's replicated path does (same kernels, same values).

Then q_b, the indexer, the sparse MLA (ShardQ with a token-local split: no
query or output exchange), g_proj and o_proj run on the rank's own R rows with
all 96 heads. o_proj accumulates its full K = 12288 in FP32 (cuBLAS), which is
more precise than the stock per-rank BF16 partials + reduce.

Push: one grid over all SMs, 2-stage 32 KiB TMA ring (the measured 1.17-1.19
TB/s law, PR-3b / PX in-step probe). Contiguous segments are 1D bulk copies;
W_o's shard is a [7168, 1536] column block of the staged [7168, 12288] (K-major)
weight, written with 2D tensor stores through one tensor map per destination.
The kernel returns once every rank's bytes for this rank landed (counted
completion, one release add per source and rank).

Memory (no new allocation): the staging region and the row buffers alias the
GEMM-RS ``partial`` workspace and the dead tails of the SP all-gather slots
(``TlxDsa``: lifetimes and layout). Fits at MNBT >= 16384 at TP8; raises at
init otherwise.

WAR safety: a peer pushes into my staging / row buffers for DSA layer D' only
after it started D', i.e. finished every layer before, including the MoE of
the previous DSA layer D, which consumes the tokens I dispatch only after my
o_proj of D (the last reader of the staging region). The guard that keeps the
layout correct even if that argument failed: before its first store to a
destination, a pusher waits until that destination released (``release``)
as many DSA layers as the pusher has pushed (device-side counters in the SP
all-gather's signal pad: graph-capture safe).
"""

import hashlib
import os
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CHUNK = 32 << 10
_BR = 8  # W_o rows per 2D unit (8 x 3072 B = 24 KiB)
_SPIN_NS = 30_000_000_000
QB_N, QB_K = 18432, 1536  # 96 heads x (128 nope + 64 rope), q_lora
G_N, G_K = 12288, 7168
O_N, O_K = 7168, 12288
SHARD_QB, SHARD_G, SHARD_O = QB_N // 8, G_N // 8, O_K // 8
W_BYTES = (QB_N * QB_K + G_N * G_K + O_N * O_K) * 2
# Bytes per new row pushed: kv_c (512) + k_pe (64) + wkv_index (index head 128 + 32 index-head weights), BF16.
KV_C, K_PE, WKV = 512, 64, 160
ROW_BYTES = (KV_C + K_PE + WKV) * 2

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <map>
#include <utility>

__device__ __forceinline__ uint64_t tx_now() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t tx_sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t tx_ld_acq(uint64_t a) {
  uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
static cudaStream_t tx_cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

#define TX_MAXSEG 8
// 1D segments: source, bytes, byte offset from each destination's base. 2D segment (W_o): o_rows x 3072 B rows
// (contiguous source) into a [o_rows, 12288] BF16 K-major weight at column block `rank` (two 192 x u64 boxes per row).
struct __align__(64) TxArgs {
  CUtensorMap m[8];
  int64_t src[TX_MAXSEG], nb[TX_MAXSEG], off[TX_MAXSEG], cum[TX_MAXSEG + 1];
  int64_t base[16];  // destination base addresses of buffer b at [b * 8 + d] (1D segments)
  int bi[TX_MAXSEG];  // destination buffer of each 1D segment
  int64_t o_src;
  int n, o_rows;
};

template <int NS>
__global__ void __launch_bounds__(32, 1) tlx_push_kernel(
    const __grid_constant__ TxArgs a, const int64_t* __restrict__ ctr, uint64_t done, uint64_t freew, uint64_t pushed,
    int rank, int W, int chunk, uint64_t limit) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0) return;
  const int G = gridDim.x, g = blockIdx.x;
  const int64_t n1 = a.cum[a.n];                     // 1D chunks per destination
  const int64_t n2 = a.o_rows > 0 ? a.o_rows / 8 : 0;  // 2D units per destination (8 rows each)
  const int64_t total = (n1 + n2) * W;
  if (total == 0) return;  // the init-time load launch: no completion, no epoch
  const int64_t n = g < total ? (total - g + G - 1) / G : 0;
  // Staging-free guard: destination d released as many DSA layers as this rank pushed (local words, written by d).
  const uint32_t epoch = *(volatile uint32_t*)pushed;
  uint32_t polled = 0;
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tx_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  int kind[NS], dd[NS];
  int64_t aux[NS];
  uint32_t nbs[NS];
  uint32_t ph = 0;
  auto issue = [&](int64_t k) {
    const int64_t u = g + k * (int64_t)G, c = u / W;
    const int d = (rank + 1 + (int)(u % W)) % W, s = (int)(k % NS);
    const uint8_t* p;
    uint32_t nb;
    if (c < n1) {
      int si = 0;
      while (si + 1 < a.n && c >= a.cum[si + 1]) ++si;
      const int64_t off = (c - a.cum[si]) * chunk;
      nb = (uint32_t)min((int64_t)chunk, a.nb[si] - off);
      p = (const uint8_t*)a.src[si] + off;
      kind[s] = 0; aux[s] = a.base[a.bi[si] * 8 + d] + a.off[si] + off;
    } else {
      const int64_t j = c - n1;
      nb = 8u * 3072u;
      p = (const uint8_t*)a.o_src + j * 8 * 3072;
      kind[s] = 1; aux[s] = j * 8;
    }
    dd[s] = d; nbs[s] = nb;
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
                 :: "r"(tx_sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(tx_sa(smem + (size_t)s * chunk)), "l"(p), "r"(nb), "r"(tx_sa(&bar[s])) : "memory");
  };
  for (int64_t k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t ab = tx_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}"
                 :: "r"(ab), "r"(p) : "memory");
    ph ^= 1u << s;
    const int d = dd[s];
    if (!((polled >> d) & 1u)) {
      const uint64_t w = freew + 4ull * (uint64_t)d, t0 = tx_now();
      uint32_t v;
      while ((int32_t)((v = tx_ld_acq(w)) - epoch) < 0) {
        __nanosleep(64);
        if (tx_now() - t0 > limit) {
          printf("K31_SPIN_TIMEOUT kernel=tlx_push free rank=%d dst=%d epoch=%u seen=%u\n", rank, d, epoch, v);
          __trap();
        }
      }
      polled |= 1u << d;
    }
    if (kind[s] == 0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"((uint64_t)aux[s]), "r"(tx_sa(smem + (size_t)s * chunk)), "r"(nbs[s]) : "memory");
    } else {
      // The 8 x 3072 B unit is [8 rows][2 halves][192 u64] in smem: one 3D box {192, 2, 8} at (0, 2 rank, row).
      asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                   :: "l"(reinterpret_cast<uint64_t>(&a.m[d])), "r"(0), "r"(rank * 2), "r"((int)aux[s]),
                      "r"(tx_sa(smem + (size_t)s * chunk)) : "memory");
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence_system();
  uint32_t prev;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(done) : "memory");
  if (prev != (uint32_t)(G - 1)) return;
  *(volatile uint32_t*)done = 0u;
  __threadfence_system();
  for (int i = 0; i < W; ++i) {
    const uint64_t c = (uint64_t)ctr[(rank + 1 + i) % W];
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(c) : "memory");
  }
  const uint64_t own = (uint64_t)ctr[rank];
  const uint32_t target = (uint32_t)W;
  const uint64_t t0 = tx_now();
  uint32_t v;
  while ((int32_t)((v = tx_ld_acq(own)) - target) < 0) {
    __nanosleep(32);
    if (tx_now() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=tlx_push rank=%d target=%u seen=%u\n", rank, target, v);
      __trap();
    }
  }
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(own), "r"(0u - target) : "memory");
  *(volatile uint32_t*)pushed = epoch + 1u;
}

// After the last reader of the staging region (o_proj): tell every source this rank is free for its next push.
__global__ void tlx_release_kernel(const int64_t* __restrict__ freew_at_src) {
  const int i = threadIdx.x;
  if (i < 8 && freew_at_src[i] != 0) {
    __threadfence_system();
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"((uint64_t)freew_at_src[i]) : "memory");
  }
}

typedef CUresult (*TxEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                               const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                               CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static TxEncodeFn tx_encode() {
  static TxEncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr, "tlx: cuTensorMapEncodeTiled not found");
    fn = (TxEncodeFn)p;
  }
  return fn;
}

void tlx_push(std::vector<int64_t> src, std::vector<int64_t> nb, std::vector<int64_t> off, std::vector<int64_t> bi,
              std::vector<int64_t> base,
              int64_t o_src, int64_t o_rows, std::vector<int64_t> o_dst, torch::Tensor ctr, int64_t done, int64_t freew,
              int64_t pushed, int64_t rank, int64_t W, int64_t grid, int64_t limit) {
  TORCH_CHECK(src.size() == nb.size() && nb.size() == off.size() && src.size() <= TX_MAXSEG, "tlx_push: segments");
  TORCH_CHECK(base.size() % W == 0 && base.size() <= 16 && W <= 8 && ctr.numel() == W, "tlx_push: W");
  TORCH_CHECK(bi.size() == src.size(), "tlx_push: segment buffers");
  TORCH_CHECK(o_rows % 8 == 0 && (o_rows == 0 || (int64_t)o_dst.size() == W), "tlx_push: W_o rows");
  static std::map<std::pair<int64_t, int64_t>, std::vector<CUtensorMap>> maps;
  TxArgs a;
  memset(&a, 0, sizeof(a));
  a.n = (int)src.size();
  for (int i = 0; i < a.n; ++i) {
    TORCH_CHECK(nb[i] > 0 && nb[i] % 16 == 0 && off[i] % 16 == 0 && src[i] % 16 == 0, "tlx_push: 16-byte aligned");
    a.src[i] = src[i]; a.nb[i] = nb[i]; a.off[i] = off[i]; a.bi[i] = (int)bi[i];
    TORCH_CHECK(bi[i] >= 0 && (bi[i] + 1) * W <= (int64_t)base.size(), "tlx_push: segment buffer index");
    a.cum[i + 1] = a.cum[i] + (nb[i] + _CHUNK_ - 1) / _CHUNK_;
  }
  for (size_t j = 0; j < base.size(); ++j) a.base[(j / W) * 8 + j % W] = base[j];
  a.o_src = o_src; a.o_rows = (int)o_rows;
  if (o_rows > 0) {
    TORCH_CHECK(o_src % 16 == 0, "tlx_push: W_o source alignment");
    const auto key = std::make_pair(o_rows, o_dst[0]);
    auto it = maps.find(key);
    if (it == maps.end()) {
      std::vector<CUtensorMap> v(W);
      for (int d = 0; d < W; ++d) {
        TORCH_CHECK(o_dst[d] % 16 == 0, "tlx_push: W_o destination alignment");
        // [o_rows, 16 halves of 1536 B, 192 u64]: rank r's 3072 B column block = halves 2r, 2r + 1.
        const cuuint64_t dims[3] = {192u, 16u, (cuuint64_t)o_rows};
        const cuuint64_t strides[2] = {1536u, (cuuint64_t)(12288 * 2)};
        const cuuint32_t box[3] = {192u, 2u, 8u};
        const cuuint32_t es[3] = {1u, 1u, 1u};
        CUresult r = tx_encode()(&v[d], CU_TENSOR_MAP_DATA_TYPE_UINT64, 3, (void*)o_dst[d], dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(r == CUDA_SUCCESS, "tlx_push: cuTensorMapEncodeTiled failed ", (int)r);
      }
      it = maps.emplace(key, v).first;
    }
    for (int d = 0; d < W; ++d) a.m[d] = it->second[d];
  }
  auto k = tlx_push_kernel<2>;
  const int smem = 2 * _CHUNK_;
  static bool smem_set = false;
  if (!smem_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set = true;
  }
  k<<<(unsigned)grid, 32, smem, tx_cur()>>>(a, ctr.data_ptr<int64_t>(), (uint64_t)done, (uint64_t)freew,
                                             (uint64_t)pushed, (int)rank, (int)W, _CHUNK_, (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tlx_release(torch::Tensor freew_at_src) {
  tlx_release_kernel<<<1, 32, 0, tx_cur()>>>(freew_at_src.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
""".replace("_CHUNK_", str(_CHUNK))

_CPP = r"""
#include <torch/extension.h>
void tlx_push(std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
              int64_t, int64_t,
              std::vector<int64_t>, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void tlx_release(torch::Tensor);
"""

_lock = threading.Lock()
_ext = None


def _module():
    """The TLX push extension (built once per source into VLLM_CACHE_ROOT)."""
    global _ext
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            h = hashlib.sha256((_CUDA + _CPP).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_tlx_dsa", tag)
            os.makedirs(build, exist_ok=True)
            _ext = load_inline(
                name=f"k3_tlx_dsa_{tag}",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                functions=["tlx_push", "tlx_release"],
                extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
                build_directory=build,
                verbose=False,
            )
    return _ext


class _Region:
    """A dead byte range of a symmetric buffer: buffer index (0 = SP all-gather storage, 1 = GEMM-RS partial),
    byte offset in the buffer, size; ``used`` bytes allocated from its start."""

    def __init__(self, buf: int, off: int, size: int, name: str) -> None:
        self.buf, self.off, self.size, self.used, self.name = buf, off, size, 0, name

    def take(self, nbytes: int) -> int | None:
        nbytes = (nbytes + 255) // 256 * 256
        if self.used + nbytes > self.size:
            return None
        o = self.off + self.used
        self.used += nbytes
        return o


class TlxDsa:
    """Staging, row buffers and words of the DSA TLX-S path, aliased (no new memory) in

    * the GEMM-RS ``partial`` workspace (symm [MNBT, 7168] BF16): between two GEMM-RS calls nothing reads or writes
      it. A peer pushes into my partial for DSA layer L only after its own GEMM-RS of the layer before returned,
      whose per-CTA exit barrier waits for every rank's same CTA, i.e. after every rank finished pulling my partial;
      my next GEMM-RS writes it after my o_proj (the last reader) on the same stream;
    * the SP all-gather slot tails (no gather of a TLX step writes past 51.6% of a slot: the DSA gathers are gone,
      the KDA gathers are MXFP8, the model's final gather takes the stock path).

    Layout (``_pack``): W_o (contiguous, K-major, written by the 3D tensor-store push), W_qb, W_g (split along N at
    shard boundaries when no region holds it whole; each part is its own GEMM writing a column block of the output,
    cuBLAS ldc), and the three row planes, first fit over the regions. Fits at MNBT >= 16384 (partial 234.9 MB +
    two tails of 113.7 MB at TP8); raises at init otherwise (no silent fallback)."""

    def __init__(self, spag, rs) -> None:
        self.spag = spag
        self.world, self.rank = spag.world, spag.rank
        W = self.world
        esz = spag.storage.element_size()
        self.slot_bytes = spag.slot_numel * esz
        tail = (spag.world * spag.max_rows * spag.hidden_size * 33 // 32 + 255) // 256 * 256
        self.max_rows = spag.max_rows
        dev = spag.storage.device
        numel = spag.storage.numel()
        sp_bases = [spag.handle.get_buffer(p, [numel], spag.storage.dtype, 0).data_ptr() for p in range(W)]
        pr_bases = [
            rs.partial_handle.get_buffer(p, list(rs.partial.shape), rs.partial.dtype, 0).data_ptr() for p in range(W)
        ]
        self.bases = sp_bases + pr_bases
        self.raws = (spag.storage.view(torch.uint8), rs.partial.view(torch.uint8))
        regions = [
            _Region(1, 0, rs.partial.numel() * rs.partial.element_size(), "gemm_rs_partial"),
            _Region(0, tail, self.slot_bytes - tail, "sp_ag_slot0_tail"),
            _Region(0, self.slot_bytes + tail, self.slot_bytes - tail, "sp_ag_slot1_tail"),
        ]
        self.layout = self._pack(regions)
        # Words at the end of each rank's SP all-gather signal pad, below its 4 (2 counters + 2 local done counts):
        # [16 free words][ctr][done][pushed].
        pads, pad = spag.handle.signal_pad_ptrs, spag.handle.signal_pad_size
        free_off = pad - 4 * 4 - 4 * 32
        self.ctr = torch.tensor([p + free_off + 4 * 16 for p in pads], dtype=torch.int64, device=dev)
        self.done = pads[self.rank] + free_off + 4 * 17
        self.pushed = pads[self.rank] + free_off + 4 * 18
        self.freew = pads[self.rank] + free_off  # [W] words: word d = releases by destination d
        self.release_to = torch.tensor(
            [pads[p] + free_off + 4 * self.rank for p in range(W)], dtype=torch.int64, device=dev
        )
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        m = _module()
        m.tlx_push([], [], [], [], self.bases, 0, 0, [], self.ctr, self.done, self.freew, self.pushed,
                   self.rank, W, 1, _SPIN_NS)
        m.tlx_release(torch.zeros(W, dtype=torch.int64, device=dev))
        logger.info(
            "Kimi K3.1 DSA TLX-S staging (no new memory, max_rows %d): %s", self.max_rows,
            "; ".join(f"{r.name} {r.used / 1e6:.1f} of {r.size / 1e6:.1f} MB" for r in regions),
        )

    def _view(self, buf: int, off: int, nbytes: int) -> torch.Tensor:
        return self.raws[buf][off : off + nbytes]

    def _pack(self, regions: list) -> dict:
        """First-fit layout (module docstring); raises with the shortfall when it does not fit."""
        bf = torch.bfloat16
        out: dict = {}

        def place(name: str, nbytes: int):
            for r in sorted(regions, key=lambda r: r.size - r.used):
                o = r.take(nbytes)
                if o is not None:
                    return r.buf, o
            free = sum(r.size - r.used for r in regions)
            raise RuntimeError(
                f"Kimi K3.1 DSA TLX-S: {name} ({nbytes / 1e6:.1f} MB) fits no dead region "
                f"(free {[(r.name, round((r.size - r.used) / 1e6, 1)) for r in regions]} MB, {free / 1e6:.1f} in "
                f"all) at max_rows {self.max_rows}; TLX-S needs max_num_batched_tokens >= 16384 at TP8"
            )

        def matrix(name: str, n: int, k: int, shard: int) -> list:
            """N-parts [(n0, n1, buf, off, view)] of an [n, k] BF16 matrix, split at shard boundaries."""
            for parts in range(1, n // shard + 1):
                if (n // shard) % parts:
                    continue
                step = n // parts
                saved = [(r.used) for r in regions]
                try:
                    res = []
                    for j in range(parts):
                        buf, off = place(f"{name}[{j}/{parts}]", step * k * 2)
                        res.append((j * step, (j + 1) * step, buf, off,
                                    self._view(buf, off, step * k * 2).view(bf).view(step, k)))
                    return res
                except RuntimeError:
                    for r, u in zip(regions, saved):
                        r.used = u
                    if parts == n // shard:
                        raise
            raise AssertionError

        buf, off = place("W_o", O_N * O_K * 2)
        out["o"] = (buf, off, self._view(buf, off, O_N * O_K * 2).view(bf).view(O_N, O_K))
        # W_g before W_qb: at MNBT 16384 W_o + W_qb fill the partial (232.8 of 234.9 MB) and W_g's two N halves the
        # slot tails (88.1 of 113.7 MB each), the fewest GEMM parts.
        out["g"] = matrix("W_g", G_N, G_K, SHARD_G)
        out["qb"] = matrix("W_qb", QB_N, QB_K, SHARD_QB)
        out["planes"] = []
        for i, w in enumerate((KV_C, K_PE, WKV)):
            out["planes"].append(place(f"rows plane {i}", (self.world + 1) * self.max_rows * w * 2) + (w,))
        return out

    # Row planes: per plane (kv_c 512, k_pe 64, wkv 160 BF16 columns) a receive region of world x max_rows rows,
    # rank r's rows at r * R (so the step's rows of all ranks are one [world R, w] tensor), then the send rows.
    def _plane(self, i: int, R: int, which: int) -> tuple[int, int, torch.Tensor]:
        buf, base, w = self.layout["planes"][i]
        rb = w * 2
        assert R <= self.max_rows
        if which == self.world:
            o, n = base + self.world * self.max_rows * rb, R
        elif which >= 0:
            o, n = base + which * R * rb, R
        else:
            o, n = base, self.world * R
        return buf, o, self._view(buf, o, n * rb).view(torch.bfloat16).view(n, w)

    def send_rows(self, R: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This rank's send rows: (kv_c [R, 512], k_pe [R, 64], wkv [R, 160]) BF16."""
        return tuple(self._plane(i, R, self.world)[2] for i in range(3))  # type: ignore[return-value]

    def recv_rows(self, R: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every rank's rows in rank order (= the step's token order): ([W R, 512], [W R, 64], [W R, 160])."""
        return tuple(self._plane(i, R, -1)[2] for i in range(3))  # type: ignore[return-value]

    @property
    def w_o(self) -> torch.Tensor:
        return self.layout["o"][2]

    def gemm(self, which: str, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out [R, N] = x @ W_full.T for the staged q_b / g weight (one cuBLAS GEMM per N part; FP32 accumulate)."""
        for n0, n1, _, _, w in self.layout[which]:
            torch.mm(x, w.t(), out=out[:, n0:n1])
        return out

    def push(self, w_qb: torch.Tensor, w_g: torch.Tensor, w_o: torch.Tensor, R: int) -> None:
        """One launch: this rank's W_qb / W_g / W_o shards into every rank's staging and its R send rows into every
        rank's receive planes; returns (stream order) once every rank's bytes for this rank landed."""
        W, r = self.world, self.rank
        assert w_qb.shape == (SHARD_QB, QB_K) and w_g.shape == (SHARD_G, G_K) and w_o.shape == (O_N, SHARD_O)
        assert w_qb.is_contiguous() and w_g.is_contiguous() and w_o.is_contiguous()
        segs = []
        for i in range(3):
            _, _, send = self._plane(i, R, W)
            buf, o, _ = self._plane(i, R, r)
            segs.append((send.data_ptr(), send.numel() * 2, o, buf))
        for which, w, shard in (("qb", w_qb, SHARD_QB), ("g", w_g, SHARD_G)):
            row0 = r * shard
            n0, n1, buf, off, _ = next(p for p in self.layout[which] if p[0] <= row0 < p[1])
            assert row0 + shard <= n1
            segs.append((w.data_ptr(), w.numel() * 2, off + (row0 - n0) * w.shape[1] * 2, buf))
        obuf, ooff, _ = self.layout["o"]
        units = (sum(-(-nb // _CHUNK) for _, nb, _, _ in segs) + O_N // _BR) * W
        _module().tlx_push(
            [a for a, _, _, _ in segs], [nb for _, nb, _, _ in segs], [o for _, _, o, _ in segs],
            [b for _, _, _, b in segs], self.bases, w_o.data_ptr(), O_N,
            [self.bases[obuf * W + d] + ooff for d in range(W)], self.ctr, self.done, self.freew, self.pushed,
            r, W, min(self.num_sms, units), _SPIN_NS,
        )  # fmt: skip

    def release(self) -> None:
        """After the layer's last read of the staging region and the received rows."""
        _module().tlx_release(self.release_to)


_tlx: TlxDsa | None = None
_tlx_init_done = False
# Rows per rank from which a step goes token-local (the ShardQ split's
# TOKEN_LOCAL_MIN_ROWS): at R >= ~700 a DSA layer's weights are fewer bytes
# than its activation exchanges (designs/tp8comm-sol.md sec 1).
MIN_ROWS = 1024


def layer_ok(attn) -> bool:
    """A DSA attention module TLX-S can run: BF16 q_b / g / o shards of the
    TP8 shapes, the fused indexer insert, NoPE."""
    try:
        ws = (attn.q_b_proj.weight, attn.g_proj.weight, attn.o_proj.weight)
        shapes = ((SHARD_QB, QB_K), (SHARD_G, G_K), (O_N, SHARD_O))
        return (
            all(w.dtype == torch.bfloat16 and w.is_contiguous() and tuple(w.shape) == sh for w, sh in zip(ws, shapes))
            and attn.dsa_indexer.use_fused_k_insert
            and attn.dsa_indexer.wkv_index.weight.shape[0] == WKV
            and attn.mla_attn.rotary_emb is None
            and attn.mla_attn.mla_attn.use_shardq
        )
    except AttributeError:
        return False


def enable(dsa_layers: list) -> bool:
    """Model init (after the SP all-gather buffer): enable the token-local
    split for steps of >= MIN_ROWS rows per rank when every DSA layer and the
    buffers support it; logs why not otherwise."""
    from vllm.v1.attention.ops import shardq

    t = get()
    if t is None or not dsa_layers:
        shardq.TOKEN_LOCAL_MIN_ROWS = None
        return False
    bad = [i for i, a in enumerate(dsa_layers) if not layer_ok(a)]
    if bad:
        raise RuntimeError(
            f"Kimi K3.1 DSA TLX-S: {len(bad)} DSA layers lack the supported form (BF16 q_b / g / o shards of the TP8 "
            "shapes, fused indexer insert, 160-wide wkv_index, NoPE, ShardQ)"
        )
    shardq.TOKEN_LOCAL_MIN_ROWS = MIN_ROWS
    logger.info("Kimi K3.1 DSA TLX-S enabled for prefill steps of >= %d rows per rank.", MIN_ROWS)
    return True


def step_token_local(attn) -> bool:
    """This forward's DSA layers run token-local (TLX-S): a DCP ShardQ step
    whose split the planner made token-local."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if _tlx is None or not is_forward_context_available():
        return False
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        return False
    shardq = getattr(md.get(attn.mla_attn.mla_attn.layer_name), "shardq", None)
    return shardq is not None and shardq.split.token_local


def get() -> TlxDsa | None:
    """The process's TLX-S DSA state (None when the instance has no SP all-gather buffer / GEMM-RS workspace, or
    steps never reach MIN_ROWS rows per rank). Raises when TLX-S applies but its staging fits no dead region."""
    global _tlx, _tlx_init_done
    if not _tlx_init_done:
        _tlx_init_done = True
        from vllm.model_executor.kernels.linear.cute_dsl import gemm_rs_ar
        from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

        spag = sp_ag._sp_all_gather
        rs = getattr(gemm_rs_ar, "_gemm_rs_ar", None)
        if (
            spag is not None
            and rs is not None
            and spag.tma
            and spag.world == 8
            and spag.hidden_size == 7168
            and rs.N == 7168
            and spag.max_rows >= MIN_ROWS
        ):
            _tlx = TlxDsa(spag, rs)
    return _tlx
