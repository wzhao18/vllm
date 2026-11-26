# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXPERIMENT ONLY (never for merge): MM-SOL P1-IS push probe, in step.

MM-SOL design rev 3 (prefill/designs/megamoe-sol.md sec 4) says its riskiest unit is
K1: the in-step delivery rate of MM-SOL's own NVLink pushes (dispatch R x 7 x 2080 B
plus the dedup combine R x 7 x 7168 B per GPU per MoE layer), window-ordered, all-peer
interleaved and never gated on one peer. This module launches a persistent push-only
kernel right after every Rubin MegaMoE launch (stock router + main + reduce, which
still produce the step's real output), on all EP ranks, in every MoE layer, on the
same stream, with the step's real neighbours (KVP CE copies, pack / signal, routing).

Placement B of the spec (after the stock main kernel; here also after its reduce, which
runs inside the same compiled launch) and its lifetime proof:
- Destination = each receiver's ``pre_reduced_activation`` region of the bucket that
  just ran (symmetric arena). The stock receiver reads only the slots its
  ``valid_rank_mask`` names, every one rewritten in the launch that reads it (the
  backend asserts ``masks_invalid_combine_slots`` / no per-launch pre-clear), so the
  region holds no live value between launches.
- No rank pushes before every rank entered the probe (entry count), and a rank enters
  only after its own reduce finished (same stream, no PDL), so no stock read is in
  flight on any receiver. A rank's probe exits only after every rank's pushes to it
  completed (done count), and peers' next writes into this region (layer L+1 token_back)
  come after this rank's next router round, which is after its probe exit.
- Sources (read only): this rank's ``x`` rows (dispatch bytes) and its local
  ``fc2_order_output`` (combine bytes).
- Counters: three words at the end of the arena handle's torch symmetric-memory signal
  pad (allocated with the arena, unused by MegaMoE: the kernel uses the peer mappings
  only and ``group_barrier`` is an NCCL all-reduce). Monotone, epoch-tagged.
- New device memory: the stamp ring below (both arms allocate it: the control arm only
  skips the launch), < 4 MiB.

Env: ``K31_MMSOL_PROBE=1`` launches the probe (otherwise built and warm-loaded only);
``K31_MMSOL_STAMPS_DIR`` receives ``rank<r>.bin`` (per call: 8 int64 header + grid x 8
int64 per-CTA stamps); ``K31_MMSOL_PROBE_MODE`` = ``push`` (default) | ``local``
(same bytes stored to this rank's own region: no NVLink; a control).
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PRE_REDUCED = "nvlink.token_comm_dedup.pre_reduced_activation"
_ORDER = "nvlink.token_comm_dedup.fc2_order_output"
_DISPATCH_ROW = 2080  # NVFP4 row 1792 B + SF 224 B + route/score metadata (design sec 1)
_COMBINE_ROW = 7168  # one BF16 row of hidden 3584
_STAGE = 32 << 10
_RING = 256  # calls kept on device
_SMEM_REQ = 296 << 10  # MM-SOL per-CTA smem (design sec 2); clamped to the opt-in max
_SPIN_NS = 20_000_000_000

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t mp_now() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t mp_clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t mp_sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mp_ld_acq(uint64_t a) {
  uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ void mp_spin(uint64_t a, uint32_t target, uint64_t limit, int rank, int what) {
  const uint64_t t0 = mp_now(); uint32_t v;
  while ((int32_t)((v = mp_ld_acq(a)) - target) < 0) {
    __nanosleep(32);
    if (mp_now() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=mmsol_push_probe rank=%d what=%d target=%u seen=%u\n", rank, what, target, v);
      __trap();
    }
  }
}

struct Args {
  const uint8_t* dsrc; int64_t dsrc_span;   // dispatch source and its usable span (bytes, >= STAGE)
  const uint8_t* csrc; int64_t csrc_span;   // combine source
  int64_t dst_local;                         // local address of the receiver region
  const int64_t* offs;                       // [W] peer displacement (arena offsets)
  int64_t slice; int64_t slice_span;         // per-source slice in a receiver; wrap span (slice - STAGE)
  int64_t disp_per_dest; int nwin;           // dispatch bytes per destination, windows
  int64_t rows_per_dest; int row_bytes;      // combine rows per destination
  const int64_t* entry; const int64_t* done; // [W] absolute addresses of the counters on every rank
  uint64_t cta_ctr;                          // local CTA-done word
  uint32_t epoch; int rank; int W; int local_only; uint64_t limit;
  int64_t* stamps; int64_t* hdr; int n_tokens;
};

// Units: phase 1 = dispatch pieces (window-major, destination rotated from rank + 1, up to 32 KiB each); phase 2 =
// combine stages of 4 rows, each row to the next destination (all-peer interleaved, row order = window order). Unit q
// runs on CTA q % G. One issuing thread: TMA load into an NS-stage ring (L = NS / 2 loads ahead), bulk store(s) out.
template <int NS>
__global__ void __launch_bounds__(32, 1) mmsol_push_probe_kernel(Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0) return;
  const int G = gridDim.x, g = blockIdx.x;
  if (a.epoch == 0) {  // init: zero this rank's counters (warm load, before the group barrier)
    if (g == 0) { *(volatile uint32_t*)a.entry[a.rank] = 0u; *(volatile uint32_t*)a.done[a.rank] = 0u;
                  *(volatile uint32_t*)a.cta_ctr = 0u; __threadfence_system(); }
    return;
  }
  int64_t* st = a.stamps + ((int64_t)((a.epoch - 1) % RING) * G + g) * 8;
  st[0] = (int64_t)mp_now(); st[5] = (int64_t)mp_clk();
  uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); st[7] = smid;
  const int W = a.W, Wr = W - 1;
  if (g == 0) {
    for (int i = 0; i < W; ++i)
      asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"((uint64_t)a.entry[(a.rank + i) % W]) : "memory");
  }
  mp_spin((uint64_t)a.entry[a.rank], a.epoch * (uint32_t)W, a.limit, a.rank, 1);
  st[1] = (int64_t)mp_now();
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mp_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

  const int64_t cb = ((a.disp_per_dest / a.nwin) + 15) / 16 * 16;      // bytes per (window, destination)
  const int64_t npc = (cb + STAGE - 1) / STAGE;                         // pieces per (window, destination)
  const int64_t Q1 = (int64_t)a.nwin * Wr * npc;
  const int64_t nrows = a.rows_per_dest * Wr;
  const int64_t Q2 = (nrows + 3) / 4;
  const int64_t Q = Q1 + Q2;
  const int64_t n = g < Q ? (Q - g + G - 1) / G : 0;
  const int64_t comb_base = (a.disp_per_dest + 255) / 256 * 256;        // combine area offset inside a slice
  int64_t dst[NS][4]; uint32_t nb[NS][4]; int ns[NS];
  uint32_t ph = 0;
  auto dest_addr = [&](int d, int64_t pos) -> uint64_t {
    const int64_t p = pos % a.slice_span;
    const int64_t peer = a.local_only ? 0 : a.offs[d];
    return (uint64_t)(a.dst_local + peer + (int64_t)(a.local_only ? d : a.rank) * a.slice + p);
  };
  auto issue = [&](int64_t k) {
    const int64_t q = g + k * (int64_t)G;
    const int s = (int)(k % NS);
    uint32_t ld; const uint8_t* src;
    if (q < Q1) {
      const int64_t w = q / (Wr * npc), r = q % (Wr * npc), i = r / npc, p = r % npc;
      const int d = (a.rank + 1 + (int)i) % W;
      const int64_t off = p * STAGE;
      ld = (uint32_t)min((int64_t)STAGE, cb - off);
      const int64_t pos = w * cb + off;
      src = a.dsrc + ((((int64_t)a.rank * 977 + w * 131 + i * 7) * STAGE + off) % a.dsrc_span) / 16 * 16;
      dst[s][0] = dest_addr(d, pos); nb[s][0] = ld; ns[s] = 1;
    } else {
      const int64_t j0 = (q - Q1) * 4;
      const int m = (int)min((int64_t)4, nrows - j0);
      ld = (uint32_t)(m * a.row_bytes);
      src = a.csrc + ((j0 * a.row_bytes) % a.csrc_span) / 16 * 16;
      for (int t = 0; t < m; ++t) {
        const int64_t j = j0 + t, i = j % Wr, row = j / Wr;
        const int d = (a.rank + 1 + (int)i) % W;
        dst[s][t] = dest_addr(d, comb_base + row * a.row_bytes); nb[s][t] = (uint32_t)a.row_bytes;
      }
      ns[s] = m;
    }
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
                 :: "r"(mp_sa(&bar[s])), "r"(ld) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(mp_sa(smem + (size_t)s * STAGE)), "l"(src), "r"(ld), "r"(mp_sa(&bar[s])) : "memory");
  };
  constexpr int L = NS / 2;
  for (int64_t k = 0; k < L && k < n; ++k) issue(k);
  st[2] = (int64_t)mp_now();
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t ba = mp_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}"
                 :: "r"(ba), "r"(p) : "memory");
    ph ^= 1u << s;
    uint32_t so = 0;
    for (int t = 0; t < ns[s]; ++t) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"(dst[s][t]), "r"(mp_sa(smem + (size_t)s * STAGE + so)), "r"(nb[s][t]) : "memory");
      so += nb[s][t];
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    // The load of unit k + L reuses the stage of unit k + L - NS: its store group must have read smem.
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NS - L - 1) : "memory");
    if (k + L < n) issue(k + L);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  st[3] = (int64_t)mp_now();
  __threadfence_system();
  uint32_t prev;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(a.cta_ctr) : "memory");
  if (prev + 1u == a.epoch * (uint32_t)G) {  // last CTA of this rank's grid in this call
    __threadfence_system();
    for (int i = 0; i < W; ++i)
      asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"((uint64_t)a.done[(a.rank + 1 + i) % W]) : "memory");
    mp_spin((uint64_t)a.done[a.rank], a.epoch * (uint32_t)W, a.limit, a.rank, 2);
    int64_t* h = a.hdr + (int64_t)((a.epoch - 1) % RING) * 8;
    h[0] = a.epoch; h[1] = a.n_tokens; h[2] = (Q1 > 0 ? cb * a.nwin * Wr : 0) + nrows * a.row_bytes;
    h[3] = (int64_t)mp_now(); h[4] = G; h[5] = NS; h[6] = a.nwin; h[7] = g;
  }
  st[4] = (int64_t)mp_now(); st[6] = (int64_t)mp_clk();
}

static cudaStream_t mp_cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

void mmsol_push_probe(torch::Tensor dsrc, int64_t dsrc_span, torch::Tensor csrc, int64_t csrc_span, int64_t dst_local,
                      torch::Tensor offs, int64_t slice, int64_t disp_per_dest, int64_t nwin, int64_t rows_per_dest,
                      int64_t row_bytes, torch::Tensor entry, torch::Tensor done, int64_t cta_ctr, int64_t epoch,
                      int64_t rank, int64_t W, int64_t local_only, int64_t limit, torch::Tensor stamps,
                      torch::Tensor hdr, int64_t n_tokens, int64_t grid, int64_t smem) {
  Args a;
  a.dsrc = (const uint8_t*)dsrc.data_ptr(); a.dsrc_span = dsrc_span;
  a.csrc = (const uint8_t*)csrc.data_ptr(); a.csrc_span = csrc_span;
  a.dst_local = dst_local; a.offs = offs.data_ptr<int64_t>();
  a.slice = slice; a.slice_span = slice - STAGE;
  a.disp_per_dest = disp_per_dest; a.nwin = (int)nwin; a.rows_per_dest = rows_per_dest; a.row_bytes = (int)row_bytes;
  a.entry = entry.data_ptr<int64_t>(); a.done = done.data_ptr<int64_t>(); a.cta_ctr = (uint64_t)cta_ctr;
  a.epoch = (uint32_t)epoch; a.rank = (int)rank; a.W = (int)W; a.local_only = (int)local_only; a.limit = (uint64_t)limit;
  a.stamps = stamps.data_ptr<int64_t>(); a.hdr = hdr.data_ptr<int64_t>(); a.n_tokens = (int)n_tokens;
  auto k = mmsol_push_probe_kernel<NSTAGES>;
  static int smem_set = 0;
  if (smem_set < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    smem_set = (int)smem;
  }
  k<<<(unsigned)grid, 32, (size_t)smem, mp_cur()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = r"""
#include <torch/extension.h>
void mmsol_push_probe(torch::Tensor, int64_t, torch::Tensor, int64_t, int64_t, torch::Tensor, int64_t, int64_t,
                      int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t,
                      int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
"""

_lock = threading.Lock()
_mods: dict[int, object] = {}


def _module(ns: int):
    with _lock:
        if ns not in _mods:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            src = f"#define STAGE {_STAGE}\n#define RING {_RING}\n#define NSTAGES {ns}\n" + _CUDA
            h = hashlib.sha256((src + _CPP).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_mmsol_probe", tag)
            os.makedirs(build, exist_ok=True)
            _mods[ns] = load_inline(
                name=f"k3_mmsol_probe_{tag}",
                cpp_sources=_CPP,
                cuda_sources=src,
                functions=["mmsol_push_probe"],
                extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
                build_directory=build,
                verbose=False,
            )
    return _mods[ns]


class MmsolPushProbe:
    """One per RubinMegaMoE runtime (EP group). Built at construction (collective)."""

    def __init__(self, arena, rank: int, world: int):
        self.launch_on = os.environ.get("K31_MMSOL_PROBE", "0") == "1"
        self.local_only = int(os.environ.get("K31_MMSOL_PROBE_MODE", "push") == "local")
        self.rank, self.world = rank, world
        self.ok = world > 1 and arena.handle is not None
        if not self.ok:
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(dev)
        self.grid = int(props.multi_processor_count)
        optin = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
        if optin <= 0:
            optin = 227 << 10
        self.smem = min(_SMEM_REQ, optin - 2048)
        self.ns = max(2, min(8, self.smem // _STAGE))
        self.ns -= self.ns % 2
        pads, size = list(arena.handle.signal_pad_ptrs), int(arena.handle.signal_pad_size)
        self.entry = torch.tensor([p + size - 64 for p in pads], dtype=torch.int64, device=dev)
        self.done = torch.tensor([p + size - 60 for p in pads], dtype=torch.int64, device=dev)
        self.cta_ctr = int(pads[rank]) + size - 56
        self.offs = torch.tensor(list(arena.offsets), dtype=torch.int64, device=dev)
        self.stamps = torch.zeros((_RING, self.grid, 8), dtype=torch.int64, device=dev)
        self.hdr = torch.zeros((_RING, 8), dtype=torch.int64, device=dev)
        self.host_hdr = torch.zeros((_RING // 2, 8), dtype=torch.int64, pin_memory=True)
        self.host_st = torch.zeros((_RING // 2, self.grid, 8), dtype=torch.int64, pin_memory=True)
        self.dump_stream = torch.cuda.Stream(device=dev)
        self.dump_event: torch.cuda.Event | None = None
        self.dump_pending: tuple[int, int] | None = None
        self.stamps_dir = os.environ.get("K31_MMSOL_STAMPS_DIR", "")
        self.epoch = 0
        self.mod = _module(self.ns)
        # Warm load + counter zero (epoch 0: no barrier, no push), then a group barrier.
        z = self.stamps.view(-1)[:16]
        self.mod.mmsol_push_probe(z, 16, z, 16, 0, self.offs, _STAGE * 2, 0, 1, 0, _COMBINE_ROW, self.entry,
                                  self.done, self.cta_ctr, 0, rank, world, 0, _SPIN_NS, self.stamps, self.hdr, 0,
                                  1, self.smem)  # fmt: skip
        torch.cuda.synchronize()
        if rank == 0:
            logger.info(
                "MMSOL probe built: launch=%s mode=%s grid=%d smem=%d KiB (optin %d KiB) stages=%d pad=%d B "
                "stamps=%s",
                self.launch_on, "local" if self.local_only else "push", self.grid, self.smem >> 10,
                optin >> 10, self.ns, size, self.stamps_dir or "-",
            )  # fmt: skip

    def __call__(self, session, n: int) -> None:
        if not (self.ok and self.launch_on) or torch.cuda.is_current_stream_capturing():
            return
        ws = session.kernel._device_workspace
        names = {sp: {r.name for r in ws.regions(sp)} for sp in ("local", "shared", "tmp")}
        if _PRE_REDUCED not in names["shared"]:
            raise RuntimeError(f"MMSOL probe: no {_PRE_REDUCED} in shared regions {sorted(names['shared'])}")
        pr_off, pr_bytes = ws.offset(_PRE_REDUCED), ws.nbytes(_PRE_REDUCED)
        dst_local = session.shared_workspace.data_ptr() + pr_off
        slice_b = (pr_bytes // self.world) // 256 * 256
        if slice_b < 2 * _STAGE:
            raise RuntimeError(f"MMSOL probe: pre_reduced slice {slice_b} B too small")
        # The order buffer is a local 'data' region: in the separate tmp buffer when the kernel has one.
        if _ORDER in names["tmp"] and session.tmp_buffer is not None:
            base = session.tmp_buffer
        elif _ORDER in names["local"]:
            base = session.local_workspace
        else:
            raise RuntimeError(f"MMSOL probe: no {_ORDER} (local {sorted(names['local'])}, tmp {sorted(names['tmp'])})")
        ord_off, ord_bytes = ws.offset(_ORDER), ws.nbytes(_ORDER)
        csrc = base[ord_off : ord_off + ord_bytes]
        dsrc = session.x.view(torch.uint8).view(-1)
        dspan = max(16, dsrc.numel() - _STAGE)
        cspan = max(16, csrc.numel() - _STAGE)
        nwin = 16 if n <= 2048 else 28  # 112 local experts / W = 7 (p2k) or 4 (p4k)
        self.epoch += 1
        self.mod.mmsol_push_probe(dsrc, dspan, csrc, cspan, dst_local, self.offs, slice_b, n * _DISPATCH_ROW, nwin,
                                  n, _COMBINE_ROW, self.entry, self.done, self.cta_ctr, self.epoch, self.rank,
                                  self.world, self.local_only, _SPIN_NS, self.stamps, self.hdr, n, self.grid,
                                  self.smem)  # fmt: skip
        if self.epoch == 1 and self.rank == 0:
            logger.info(
                "MMSOL probe first launch: n=%d pre_reduced %.1f MiB slice %.1f MiB need/source %.1f MiB "
                "order %.1f MiB x %.1f MiB",
                n, pr_bytes / 2**20, slice_b / 2**20, n * (_DISPATCH_ROW + _COMBINE_ROW) / 2**20,
                ord_bytes / 2**20, dsrc.numel() / 2**20,
            )  # fmt: skip
        if self.stamps_dir and self.epoch % (_RING // 2) == 0:
            self._dump()

    def _dump(self) -> None:
        """Every RING/2 calls: append the previous half (copy long done) and start the copy of the half just
        filled (calls epoch-RING/2+1 .. epoch = slots lo .. lo+RING/2-1) on a side stream. No allocation."""
        half = _RING // 2
        if self.dump_pending is not None:
            self.dump_event.synchronize()
            os.makedirs(self.stamps_dir, exist_ok=True)
            with open(os.path.join(self.stamps_dir, f"rank{self.rank}.bin"), "ab") as f:
                f.write(np.concatenate([self.host_hdr.numpy(), self.host_st.numpy().reshape(half, -1)], axis=1).tobytes())
        lo = (self.epoch - half) % _RING
        self.dump_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.dump_stream):
            self.host_hdr.copy_(self.hdr[lo : lo + half], non_blocking=True)
            self.host_st.copy_(self.stamps[lo : lo + half], non_blocking=True)
            self.dump_event = torch.cuda.Event()
            self.dump_event.record(self.dump_stream)
        self.dump_pending = (lo, half)
