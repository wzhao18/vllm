// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Data-movement-only skeleton of the DCP top-k (SOL protocol; benchmark only, never used
// by the model). Same launch shape, mailboxes and epochs as topk_v2, no selection math:
//   PDL wait; (mode & 1) read the local row once into registers (16-byte loads);
//   (mode & 2) one fixed 768-byte Lamport message multicast into this row's slot + one poll
//   of every rank's 768 bytes; (mode & 4) write K ids + the count. mode 0 = the kernel
//   boundary alone (PDL wait, epoch bump, exit).
// Included by dcp_exchange.cu after topk_v2.cuh.

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


// ---------------------------------------------------------------------------------------
// RUNG 0 of the data-independent exact top-k (benchmark only): FIXED rounds, no sample,
// no windows. Per row: land the row in SMEM (as topk_v2), exact 256-bin histogram of the
// TOP byte of every key (warp-private bins), one Lamport round of the 256 counts (r1
// slot) -> tau's top byte b1 and the count above it (the same on every rank); exact
// 256-bin histogram of the LOW byte of the keys in b1, one more round (radix slot 0) ->
// tau and need exactly; threshold write (keys > tau appended, unordered; ties not cut).
// out_count = this rank's keys > tau, plus need on rank 0: the counts sum to K over the
// ranks exactly when tau and need are right (the bench checks it).
// mode: 10 + a: land + top-byte pass + pad only; 20 + a: the full path; a = 1: warp-
// aggregated bin increments (leader adds the ballot count), a = 0: plain SMEM atomics.
template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_fx_kernel(V2Params p, int mode) {
  V2_KERNEL_PROLOGUE(T)
  constexpr int kW = T / 32, kH = kW < 16 ? kW : 16;  // warp-private histograms (16 KB in hist)
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, hh = wid * kH / kW;
  const bool agg = (mode % 10) == 1, full = mode >= 20;
  uint32_t* const wh = hist;                           // kH x 256
  uint32_t* const rx = cpos;                           // W x 256 received words (round 1, 2)
  int* const s = s_i;                                  // 0 b1, 1 a1, 2 tau, 3 need, 4 cnt
  (void)ckey; (void)warp_sums; (void)s_u; (void)s_pref; (void)s_dbg;
  if (p.W > 4 && tid == 0) spin_fail(p.err, 62);  // rx + counts in cpos: W <= 4
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
    const uintptr_t la = reinterpret_cast<uintptr_t>(lrow);
    const int head0 = (int)(((16u - (uint32_t)(la & 15u)) & 15u) >> 1);
    const int sh0 = (8 - head0) & 7;
    const int gs_lo = sh0 ? 1 : 0, gs_hi = min((p.lcols + sh0) >> 3, V2Cfg<T>::kNR * T);
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int n = max(0, gs_hi - gs_lo), c = tid >> 5;
      const int a = gs_lo + (int)((int64_t)n * c / kV2Cp), b = gs_lo + (int)((int64_t)n * (c + 1) / kV2Cp);
      const uint4* gb = reinterpret_cast<const uint4*>(lrow - sh0);
      if (b > a) bulk_g2s(srow + a, gb + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    const int L = p.local_len[row];
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    int* orow = p.out + (int64_t)row * p.ostride;
    const int sh = sh0;  // rows >= 8 keys (the benchmark's)
    const int ng = (L + sh + 7) >> 3, g_hi = (L + sh) >> 3;
    for (int b = tid; b < kH * 256; b += T) wh[b] = 0;
    if (tid == 0) { s[0] = -1; s[1] = 0; s[2] = -1; s[3] = 0; s[4] = 0; }
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    // partial head / tail groups: keys outside [0, L) read as NaN (key 0)
    if (tid == 0 && sh > 0) {
      uint32_t w[4];
      for (int j = 0; j < 4; ++j) {
        const int a = 2 * j - sh, b = a + 1;
        w[j] = ((a >= 0 && a < L) ? lrow[a] : 0xFFFFu) | (((b >= 0 && b < L) ? lrow[b] : 0xFFFFu) << 16);
      }
      srow[0] = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (tid == 32 && g_hi < ng) {
      uint32_t w[4];
      for (int j = 0; j < 4; ++j) {
        const int a = 8 * g_hi - sh + 2 * j, b = a + 1;
        w[j] = ((a >= 0 && a < L) ? lrow[a] : 0xFFFFu) | (((b >= 0 && b < L) ? lrow[b] : 0xFFFFu) << 16);
      }
      srow[g_hi] = make_uint4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();
    // one bin increment per valid key: plain SMEM atomics, or warp-aggregated (the leader
    // of the lanes holding the same bin adds their count)
    auto bump = [&](uint32_t bin, bool ok) {
      if (!agg) {
        if (ok) atomicAdd(&wh[hh * 256 + bin], 1u);
        return;
      }
      const uint32_t v = ok ? bin : 0x100u;
      uint32_t pend = __ballot_sync(0xffffffffu, ok);
      while (pend) {
        const int ld = __ffs(pend) - 1;
        const uint32_t lv = __shfl_sync(0xffffffffu, v, ld);
        const uint32_t m = __ballot_sync(0xffffffffu, v == lv);
        if (lane == ld) atomicAdd(&wh[hh * 256 + lv], (uint32_t)__popc(m));
        pend &= ~m;
      }
    };
    const int ngr = (ng + T - 1) / T * T;  // whole warps in the loop (ballots)
    for (int g = tid; g < ngr; g += T) {
      const uint4 r = g < ng ? srow[g] : make_uint4(~0u, ~0u, ~0u, ~0u);
      const Keys8 k8(r);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t k = k8.key(j);
        bump(k >> 8, k != 0u);
      }
    }
    __syncthreads();
    // one round of the 256 top-byte counts; tau's top byte from every rank's counts
    auto round = [&](int64_t rrow, uint32_t* dst) {
      uint32_t* const st = cpos + 2048;  // the summed counts (rx holds 2 x W x 256 words, W <= 4)
      if (tid < 256) {
        uint32_t c = 0;
#pragma unroll
        for (int h = 0; h < kH; ++h) c += wh[h * 256 + tid];
        st[tid] = c;
      }
      __syncthreads();
      if (tid < 64) {
        const int j = tid * 4;
        const uint4 v = make_uint4(tagw(tag, st[j]), tagw(tag, st[j + 1]), tagw(tag, st[j + 2]), tagw(tag, st[j + 3]));
        const int64_t off = rrow + (int64_t)p.rank * 256 + j;
        if (p.mc) mc_st_v4(p.mc + off * 4, v);
        else
          for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
      }
      if (tid < p.W * 64) {
        const uint32_t* src = p.box + rrow + (int64_t)tid * 4;
        const long long t0 = now_ns();
        uint4 v = ld_sys_v4(src);
        while (!tagged(v, tag)) {
          if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 61); break; }
          v = ld_sys_v4(src);
        }
        *reinterpret_cast<uint4*>(dst + tid * 4) = v;
      }
      __syncthreads();
    };
    if (full) {
      // round-1 words in the r1 slot: rank r's 256 counts at r1row + r * kV2Msg
      const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * p.W) * kV2Msg;
      {
        uint32_t* const st = cpos + 2048;
        if (tid < 256) {
          uint32_t c = 0;
#pragma unroll
          for (int h = 0; h < kH; ++h) c += wh[h * 256 + tid];
          st[tid] = c;
        }
        __syncthreads();
        if (tid < 64) {
          const int j = tid * 4;
          const uint4 v = make_uint4(tagw(tag, st[j]), tagw(tag, st[j + 1]), tagw(tag, st[j + 2]), tagw(tag, st[j + 3]));
          const int64_t off = r1row + (int64_t)p.rank * kV2Msg + j;
          if (p.mc) mc_st_v4(p.mc + off * 4, v);
          else
            for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
        }
        if (tid < p.W * 64) {
          const int r = tid >> 6, j = (tid & 63) * 4;
          const uint32_t* src = p.box + r1row + (int64_t)r * kV2Msg + j;
          const long long t0 = now_ns();
          uint4 v = ld_sys_v4(src);
          while (!tagged(v, tag)) {
            if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 60); break; }
            v = ld_sys_v4(src);
          }
          *reinterpret_cast<uint4*>(rx + r * 256 + j) = v;
        }
        __syncthreads();
      }
      // b1: bins from the top (thread t <-> bin 255 - t)
      {
        uint32_t hb = 0;
        if (tid < 256)
          for (int r = 0; r < p.W; ++r) hb += rx[r * 256 + 255 - tid] & 0xFFFFFFu;
        uint32_t tot;
        const uint32_t ex = v2_scan<T>(hb, warp_sums, &tot);
        if (tid < 256 && ex < (uint32_t)p.K && ex + hb >= (uint32_t)p.K) { s[0] = 255 - tid; s[1] = (int)ex; }
        for (int b = tid; b < kH * 256; b += T) wh[b] = 0;
        __syncthreads();
      }
      const int b1 = s[0];
      if (b1 >= 0) {
        // low-byte counts of the keys in b1, the second round (radix slot 0)
        for (int g = tid; g < ngr; g += T) {
          const uint4 r = g < ng ? srow[g] : make_uint4(~0u, ~0u, ~0u, ~0u);
          const Keys8 k8(r);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = k8.key(j);
            bump(k & 0xFFu, k != 0u && (int)(k >> 8) == b1);
          }
        }
        __syncthreads();
        const int64_t rrow = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + 0) * p.W) * 256;
        round(rrow, rx + p.W * 256);
        uint32_t hb = 0;
        if (tid < 256)
          for (int r = 0; r < p.W; ++r) hb += rx[p.W * 256 + r * 256 + 255 - tid] & 0xFFFFFFu;
        uint32_t tot;
        const uint32_t want = (uint32_t)(p.K - s[1]);
        const uint32_t ex = v2_scan<T>(hb, warp_sums + 32, &tot);
        if (tid < 256 && ex < want && ex + hb >= want) { s[2] = (b1 << 8) | (255 - tid); s[3] = (int)(want - ex); }
        __syncthreads();
      }
      // threshold write: keys > tau (unordered), -1 pad; the count (+ need on rank 0)
      const int tau = s[2];
      for (int g = tid; g < ngr; g += T) {
        const uint4 r = g < ng ? srow[g] : make_uint4(~0u, ~0u, ~0u, ~0u);
        const Keys8 k8(r);
        uint32_t m = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) m |= ((int)k8.key(j) > tau ? 1u : 0u) << j;
        const uint32_t n = __popc(m);
        uint32_t incl = n;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lane >= o) incl += y;
        }
        int base = 0;
        if (lane == 31 && incl) base = atomicAdd(&s[4], (int)incl);
        int c = __shfl_sync(0xffffffffu, base, 31) + (int)(incl - n);
        while (m) {
          const int j = __ffs(m) - 1;
          m &= m - 1;
          if (c < p.K) orow[c] = 8 * g - sh + j;
          ++c;
        }
      }
      __syncthreads();
      const int cnt = min(s[4], p.K);
      for (int i = cnt + tid; i < p.K; i += T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = s[4] + (p.rank == 0 ? s[3] : 0);
    } else {
      for (int i = tid; i < p.K; i += T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = wh[tid] == 0x7FFFFFFFu ? 1 : 0;  // keeps the pass live
    }
    if (tid == 0) p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
    __syncthreads();  // SMEM reused by the next row
  }
}

}  // namespace k31dcpx
