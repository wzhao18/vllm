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

}  // namespace k31dcpx
