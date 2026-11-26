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
dead tails of the SP all-gather slots (``SymmSpAllGather``): under TLX no BF16
gather runs (the DSA gathers are gone; the KDA gathers are MXFP8, rows + scales
<= 51.6% of a slot; the model's final gather takes the stock path), so bytes
[world * max_rows * hidden * (1 + 1/32), slot end) of each slot are written and
read by TLX only. Slot 0's tail holds the weights, slot 1's the rows. The
init check below asserts the fit; when it fails (MNBT below 65536 for the
weights region) TLX is not enabled and says so once.

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
# Bytes per new row pushed: kv_c (512) + k_pe (64) + wkv_index (128 + 64), BF16.
KV_C, K_PE, WKV = 512, 64, 192
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
  int64_t base[8];  // destination base addresses (1D segments)
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
      kind[s] = 0; aux[s] = a.base[d] + a.off[si] + off;
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

void tlx_push(std::vector<int64_t> src, std::vector<int64_t> nb, std::vector<int64_t> off, std::vector<int64_t> base,
              int64_t o_src, int64_t o_rows, std::vector<int64_t> o_dst, torch::Tensor ctr, int64_t done, int64_t freew,
              int64_t pushed, int64_t rank, int64_t W, int64_t grid, int64_t limit) {
  TORCH_CHECK(src.size() == nb.size() && nb.size() == off.size() && src.size() <= TX_MAXSEG, "tlx_push: segments");
  TORCH_CHECK((int64_t)base.size() == W && W <= 8 && ctr.numel() == W, "tlx_push: W");
  TORCH_CHECK(o_rows % 8 == 0 && (o_rows == 0 || (int64_t)o_dst.size() == W), "tlx_push: W_o rows");
  static std::map<std::pair<int64_t, int64_t>, std::vector<CUtensorMap>> maps;
  TxArgs a;
  memset(&a, 0, sizeof(a));
  a.n = (int)src.size();
  for (int i = 0; i < a.n; ++i) {
    TORCH_CHECK(nb[i] > 0 && nb[i] % 16 == 0 && off[i] % 16 == 0 && src[i] % 16 == 0, "tlx_push: 16-byte aligned");
    a.src[i] = src[i]; a.nb[i] = nb[i]; a.off[i] = off[i];
    a.cum[i + 1] = a.cum[i] + (nb[i] + _CHUNK_ - 1) / _CHUNK_;
  }
  for (int d = 0; d < W; ++d) a.base[d] = base[d];
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
void tlx_push(std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t,
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


class TlxDsa:
    """Staging, row buffers and words of the DSA TLX-S path over one
    ``SymmSpAllGather`` (its slot tails and signal pads)."""

    def __init__(self, spag) -> None:
        self.spag = spag
        self.world, self.rank = spag.world, spag.rank
        esz = spag.storage.element_size()
        self.slot_bytes = spag.slot_numel * esz
        tail = spag.world * spag.max_rows * spag.hidden_size * 33 // 32
        self.tail = (tail + 255) // 256 * 256
        self.tail_bytes = self.slot_bytes - self.tail
        self.max_rows = spag.max_rows
        rows_need = (self.world + 1) * spag.max_rows * ROW_BYTES  # 3 planes x (W recv + 1 send) x max_rows
        self.ok = (
            spag.tma
            and self.world == 8
            and spag.hidden_size == 7168
            and W_BYTES <= self.tail_bytes
            and rows_need <= self.tail_bytes
        )
        if not self.ok:
            logger.warning(
                "Kimi K3.1 DSA TLX-S not enabled: needs TP8, the TMA push and "
                "%d B (weights) / %d B (rows) in the SP all-gather slot tails, "
                "which have %d B at max_rows %d (raise max_num_batched_tokens).",
                W_BYTES, rows_need, self.tail_bytes, spag.max_rows,
            )
            return
        numel = spag.storage.numel()
        self.bases = [
            spag.handle.get_buffer(p, [numel], spag.storage.dtype, 0).data_ptr() for p in range(self.world)
        ]
        dev = spag.storage.device
        # Words at the end of each rank's signal pad, below the SP all-gather's
        # 4 (2 counters + 2 local done counts): [16 free words][ctr][done][pushed].
        pads, pad = spag.handle.signal_pad_ptrs, spag.handle.signal_pad_size
        top = pad - 4 * 4
        free_off = top - 4 * 32
        self.ctr = torch.tensor([p + free_off + 4 * 16 for p in pads], dtype=torch.int64, device=dev)
        self.done = pads[self.rank] + free_off + 4 * 17
        self.pushed = pads[self.rank] + free_off + 4 * 18
        self.freew = pads[self.rank] + free_off  # [W] words: word d = releases by destination d
        # Release targets: my word in each source's free array.
        self.release_to = torch.tensor(
            [pads[p] + free_off + 4 * self.rank for p in range(self.world)], dtype=torch.int64, device=dev
        )
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # Weight staging (slot 0 tail): [q_b | g | o] full weights; row buffers (slot 1 tail): recv [8 x max_rows]
        # then this rank's send rows.
        self.w_off = self.tail
        self.qb_off = self.w_off
        self.g_off = self.qb_off + QB_N * QB_K * 2
        self.o_off = self.g_off + G_N * G_K * 2
        self.rows_off = self.slot_bytes + self.tail
        raw = spag.storage.view(torch.uint8)
        self.raw = raw
        bf = torch.bfloat16
        self.w_qb = raw[self.qb_off : self.qb_off + QB_N * QB_K * 2].view(bf).view(QB_N, QB_K)
        self.w_g = raw[self.g_off : self.g_off + G_N * G_K * 2].view(bf).view(G_N, G_K)
        self.w_o = raw[self.o_off : self.o_off + O_N * O_K * 2].view(bf).view(O_N, O_K)
        # Load the kernels now (empty launches), not inside a step.
        m = _module()
        m.tlx_push([], [], [], [0] * self.world, 0, 0, [], self.ctr, self.done, self.freew, self.pushed,
                   self.rank, self.world, 1, _SPIN_NS)
        m.tlx_release(torch.zeros(self.world, dtype=torch.int64, device=dev))
        logger.info(
            "Kimi K3.1 DSA TLX-S: staging %.1f MB + rows %.1f MB in the SP all-gather slot tails "
            "(%.1f MB each at max_rows %d); no new memory.",
            W_BYTES / 1e6, rows_need / 1e6, self.tail_bytes / 1e6, spag.max_rows,
        )

    # Row planes (slot 1 tail): per plane (kv_c 512, k_pe 64, wkv 192 BF16 columns) a receive region of
    # world x max_rows rows, rank r's rows at r * R (so the step's rows of all ranks are one [world R, w] tensor),
    # then this rank's send rows.
    def _plane(self, i: int, R: int, which: int) -> torch.Tensor:
        widths = (KV_C, K_PE, WKV)
        base = self.rows_off + (self.world + 1) * self.max_rows * 2 * sum(widths[:i])
        rb = widths[i] * 2
        if which == self.world:
            o, n = base + self.world * self.max_rows * rb, R
        elif which >= 0:
            o, n = base + which * R * rb, R
        else:
            o, n = base, self.world * R
        assert R <= self.max_rows
        return self.raw[o : o + n * rb].view(torch.bfloat16).view(n, widths[i])

    def send_rows(self, R: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This rank's send rows: (kv_c [R, 512], k_pe [R, 64], wkv [R, 192]) BF16."""
        return tuple(self._plane(i, R, self.world) for i in range(3))  # type: ignore[return-value]

    def recv_rows(self, R: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every rank's rows in rank order (= the step's token order): ([W R, 512], [W R, 64], [W R, 192])."""
        return tuple(self._plane(i, R, -1) for i in range(3))  # type: ignore[return-value]

    def push(self, w_qb: torch.Tensor, w_g: torch.Tensor, w_o: torch.Tensor, R: int) -> None:
        """One launch: this rank's W_qb / W_g / W_o shards into every rank's staging and its R send rows into every
        rank's receive block; returns (stream order) once every rank's bytes for this rank landed."""
        W, r = self.world, self.rank
        assert w_qb.shape == (SHARD_QB, QB_K) and w_g.shape == (SHARD_G, G_K) and w_o.shape == (O_N, SHARD_O)
        assert w_qb.is_contiguous() and w_g.is_contiguous() and w_o.is_contiguous()
        segs = []
        for i in range(3):
            send, dst = self._plane(i, R, W), self._plane(i, R, r)
            segs.append((send.data_ptr(), send.numel() * 2, dst.data_ptr() - self.raw.data_ptr()))
        segs += [
            (w_qb.data_ptr(), w_qb.numel() * 2, self.qb_off + r * SHARD_QB * QB_K * 2),
            (w_g.data_ptr(), w_g.numel() * 2, self.g_off + r * SHARD_G * G_K * 2),
        ]
        units = (sum(-(-nb // _CHUNK) for _, nb, _ in segs) + O_N // _BR) * W
        _module().tlx_push(
            [a for a, _, _ in segs], [nb for _, nb, _ in segs], [o for _, _, o in segs], self.bases,
            w_o.data_ptr(), O_N, [b + self.o_off for b in self.bases], self.ctr, self.done, self.freew, self.pushed,
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
    bad = [i for i, a in enumerate(dsa_layers) if not layer_ok(a)]
    if t is None or bad or not dsa_layers:
        if dsa_layers:
            logger.warning(
                "Kimi K3.1 DSA TLX-S not enabled (%s).",
                "buffers" if t is None else f"{len(bad)} DSA layers unsupported",
            )
        shardq.TOKEN_LOCAL_MIN_ROWS = None
        return False
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
    """The process's TLX-S DSA state (None when not enabled)."""
    global _tlx, _tlx_init_done
    if not _tlx_init_done:
        _tlx_init_done = True
        from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

        spag = sp_ag._sp_all_gather
        if spag is not None:
            t = TlxDsa(spag)
            _tlx = t if t.ok else None
    return _tlx
