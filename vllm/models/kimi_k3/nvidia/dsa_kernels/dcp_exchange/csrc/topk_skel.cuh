// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Data-movement-only skeleton of the DCP top-k (SOL protocol; benchmark only, never used
// by the model). Same launch shape, mailboxes and epochs as topk_v3, no selection math:
//   PDL wait; (mode & 1) read the local row once into registers (16-byte loads);
//   (mode & 2) one fixed 768-byte Lamport message multicast into this row's slot + one poll
//   of every rank's 768 bytes; (mode & 4) write K ids + the count. mode 0 = the kernel
//   boundary alone (PDL wait, epoch bump, exit).
// Included by dcp_exchange.cu after topk_v3.cuh.

namespace k31dcpx {

constexpr int kSkMsgV = 48;  // 16-byte vectors per message (768 B)

template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_skel_kernel(V2Params p, int mode) {
  __shared__ uint32_t s_rx;
  constexpr int kNR = V2Cfg<T>::kNR;
  const int tid = threadIdx.x;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    int* orow = p.out + (int64_t)row * p.ostride;
    uint32_t acc = 0;
    if (mode & 1) {
      const int L = p.local_len[row];
      const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
      const uintptr_t la = reinterpret_cast<uintptr_t>(lrow);
      const int head0 = (int)(((16u - (uint32_t)(la & 15u)) & 15u) >> 1);
      const int sh0 = (8 - head0) & 7;
      const int gs_hi = min(min((p.lcols + sh0) >> 3, (L + sh0 + 7) >> 3), kNR * T);
      const uint4* gb = reinterpret_cast<const uint4*>(lrow - sh0);
      uint4 v[kNR];
#pragma unroll
      for (int g = 0; g < kNR; ++g) {
        const int i = g * T + tid;
        v[g] = (i >= (sh0 ? 1 : 0) && i < gs_hi) ? __ldg(gb + i) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int g = 0; g < kNR; ++g) acc ^= v[g].x ^ v[g].y ^ v[g].z ^ v[g].w;
    }
    if (mode & 2) {
      const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * p.W) * kV2Msg;
      if (tid < kSkMsgV) {
        const uint4 m = make_uint4(tagw(tag, acc), tagw(tag, (uint32_t)tid), tagw(tag, (uint32_t)row),
                                   tagw(tag, 0u));
        const int64_t off = r1row + (int64_t)p.rank * kV2Msg + tid * 4;
        if (p.mc) mc_st_v4(p.mc + off * 4, m);
        else
          for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, m);
      }
      if (tid == 0) s_rx = 0;
      __syncthreads();
      if (tid < p.W * kSkMsgV) {
        const int r = tid / kSkMsgV, j = (tid % kSkMsgV) * 4;
        const uint32_t* src = p.box + r1row + (int64_t)r * kV2Msg + j;
        const long long t0 = now_ns();
        uint4 v = ld_sys_v4(src);
        while (!tagged(v, tag)) {
          if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 31); break; }
          v = ld_sys_v4(src);
        }
        if (v.w & 0xFFFFFFu) atomicOr(&s_rx, v.w);  // never: keeps the poll's value live
      }
      __syncthreads();
      acc ^= s_rx;
    } else if (acc == 0x9E3779B9u) {
      orow[0] = (int)acc;  // never in practice: keeps the row loads live
    }
    if (mode & 4) {
      const int base = 0;
      for (int i = tid * 4; i < p.K; i += T * 4)
        *reinterpret_cast<int4*>(orow + i) = make_int4(base + i, base + i + 1, base + i + 2, base + i + 3);
      if (tid == 0) p.out_count[row] = p.K;
    } else if (acc == 0x7F4A7C15u) {
      orow[1] = (int)acc;
    }
    if (tid == 0) p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
    if (p.T > (int)gridDim.x) __syncthreads();
  }
}


// ---- fused (a) skeleton: a logits-producer-like kernel with the top-k folded into its
// epilogue (benchmark only). Grid = rows x C CTAs of 256 threads; CTA c of a row writes
// keys [c*L/C, (c+1)*L/C) of the row (synthetic bf16 values, the producer's own store).
//   mode & 1: SMEM 256-bin coarse histogram of its keys + red.add into the row's global
//             histogram, then an arrival counter (the producer's epilogue cost);
//   mode & 2: the row's last CTA reads the histogram and sends the fixed 768 B multicast
//             message + polls every rank's copy (the exchange in the producer's tail);
//   mode & 4: the last CTA re-reads its row once from L2 (the candidate filter) and
//             writes K ids + the count.
// mode 0 = the producer alone (the reference the fused cost is measured against).
constexpr int kFsT = 256;
__global__ void __launch_bounds__(kFsT) topk_fskel_kernel(V2Params p, uint32_t* hist, int* arr,
                                                          int C, int mode) {
  __shared__ uint32_t sh[256];
  __shared__ int s_last;
  const int tid = threadIdx.x;
  const int row = blockIdx.x / C, c = blockIdx.x % C;
  if (row >= p.T) return;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int L = p.local_len[row];
  const int a = (int)((int64_t)L * c / C) & ~7, b = c == C - 1 ? L : (int)((int64_t)L * (c + 1) / C) & ~7;
  uint16_t* lrow = const_cast<uint16_t*>(p.logits) + (int64_t)row * p.lstride;
  if (mode & 1)
    for (int i = tid; i < 256; i += kFsT) sh[i] = 0u;
  __syncthreads();
  for (int g = a + 8 * tid; g < b; g += 8 * kFsT) {  // 8 keys per thread per step
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t h0 = (uint32_t)(g + 2 * j) * 2654435761u ^ (uint32_t)row * 40503u;
      const uint32_t h1 = (uint32_t)(g + 2 * j + 1) * 2654435761u ^ (uint32_t)row * 40503u;
      const uint32_t k0 = 0x3C00u | ((h0 >> 13) & 0x3FFu), k1 = 0x3C00u | ((h1 >> 13) & 0x3FFu);
      w[j] = k0 | (k1 << 16);
      if (mode & 1) {
        atomicAdd(&sh[bf16_key(k0) >> 8], 1u);
        atomicAdd(&sh[bf16_key(k1) >> 8], 1u);
      }
    }
    if (g + 8 <= b) *reinterpret_cast<uint4*>(lrow + g) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  if (!(mode & 1)) return;
  __syncthreads();
  uint32_t* const rh = hist + (int64_t)row * 256;
  for (int i = tid; i < 256; i += kFsT)
    if (sh[i]) atomicAdd(rh + i, sh[i]);
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(arr + row, 1) == C - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // the row's last CTA: local threshold from the histogram (a suffix scan stands in)
  uint32_t hv = __ldcg(rh + tid);
  uint32_t acc = hv;
  rh[tid] = 0u;  // re-armed for the next call
  if (tid == 0) arr[row] = 0;
  if (mode & 2) {
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * p.W) * kV2Msg;
    if (tid < kSkMsgV) {
      const uint4 m = make_uint4(tagw(tag, acc), tagw(tag, (uint32_t)tid), tagw(tag, 0u), tagw(tag, 0u));
      const int64_t off = r1row + (int64_t)p.rank * kV2Msg + tid * 4;
      if (p.mc) mc_st_v4(p.mc + off * 4, m);
      else
        for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, m);
    }
    if (tid < p.W * kSkMsgV) {
      const int r = tid / kSkMsgV, j = (tid % kSkMsgV) * 4;
      const uint32_t* src = p.box + r1row + (int64_t)r * kV2Msg + j;
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 31); break; }
        v = ld_sys_v4(src);
      }
      acc ^= v.y;
    }
    __syncthreads();
    if (tid == 0) p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
  }
  if (mode & 4) {
    uint32_t m = 0;
    for (int g = 8 * tid; g + 8 <= L; g += 8 * kFsT) {
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(lrow + g));
      m += (v.x > acc) + (v.y > acc) + (v.z > acc) + (v.w > acc);
    }
    int* orow = p.out + (int64_t)row * p.ostride;
    for (int i = tid * 4; i < p.K; i += kFsT * 4)
      *reinterpret_cast<int4*>(orow + i) = make_int4(i, i + 1, i + 2, (int)(i + 3 + (m == 0xFFFFFFFu)));
    if (tid == 0) p.out_count[row] = p.K;
  }
}

}  // namespace k31dcpx
