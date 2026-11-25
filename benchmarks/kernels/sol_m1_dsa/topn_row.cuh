// M1-b variant: one 1024-thread CTA per row, the row's NK BF16 logits in shared memory, exact top-N (ties to the
// lowest key) by an 11-bit + 5-bit radix (DeepSelect's digit split), emission, rank-by-counting sort.
// st [rows][6]: c0, c_pass1, c_pass2, c_emit, c1, (g1 - g0)
constexpr int TR_T = 1024;
__device__ __forceinline__ int block_find_from_top(const int* h, int nb, int need, int* s_part, int* out_bin,
                                                   int* out_above) {
  // h: nb (multiple of TR_T) bins; thread t owns bins [t*per, t*per+per) counted from the TOP (bin nb-1 first)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = nb / TR_T;
  int loc = 0;
  for (int j = 0; j < per; ++j) loc += h[nb - 1 - (tid * per + j)];
  int inc = loc;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
  if (lane == 31) s_part[warp] = inc;
  __syncthreads();
  if (warp == 0) {
    int w = s_part[lane], wi = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, wi, o); if (lane >= o) wi += v; }
    s_part[32 + lane] = wi - w;  // exclusive prefix of warps
  }
  __syncthreads();
  const int excl = s_part[32 + warp] + inc - loc;
  if (excl < need && excl + loc >= need) {
    int acc = excl;
    for (int j = 0; j < per; ++j) {
      const int b = nb - 1 - (tid * per + j);
      if (acc + h[b] >= need) { *out_bin = b; *out_above = acc; break; }
      acc += h[b];
    }
  }
  __syncthreads();
  return 0;
}
__global__ void __launch_bounds__(TR_T, 1) topn_row_kernel(const uint16_t* __restrict__ logits, int NK, int TOPN,
                                                          int* __restrict__ out_key, uint16_t* __restrict__ out_sc,
                                                          long long* __restrict__ st) {
  extern __shared__ __align__(16) unsigned char smr[];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
  int* h1 = reinterpret_cast<int*>(smr);       // 2048
  int* h2 = h1 + 2048;                         // 1024 (32 used)
  int* ck = h2 + 1024;                         // [TOPN]
  uint32_t* co = reinterpret_cast<uint32_t*>(ck + TOPN);
  uint16_t* lg = reinterpret_cast<uint16_t*>(co + TOPN);
  __shared__ int s_part[64], s_bin, s_above, s_lo, s_gt, s_ctr, s_tie;
  for (int i = tid; i < NK; i += TR_T) lg[i] = logits[(size_t)row * NK + i];
  for (int i = tid; i < 2048; i += TR_T) h1[i] = 0;
  for (int i = tid; i < 1024; i += TR_T) h2[i] = 0;
  if (tid == 0) { s_ctr = 0; s_tie = 0; }
  __syncthreads();
  long long c0 = clock64(), g0 = gtime();
  for (int i = tid; i < NK; i += TR_T) atomicAdd(&h1[ord16(lg[i]) >> 5], 1);
  __syncthreads();
  long long cp1 = clock64();
  block_find_from_top(h1, 2048, TOPN, s_part, &s_bin, &s_above);
  const int bin = s_bin;
  for (int i = tid; i < NK; i += TR_T) {
    const uint32_t u = ord16(lg[i]);
    if ((int)(u >> 5) == bin) atomicAdd(&h2[u & 31], 1);
  }
  __syncthreads();
  if (tid < 32) {  // 32 low-digit bins: warp 0 finds the cut from the top
    const int need = TOPN - s_above;
    const int v = h2[31 - lane];
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += x; }
    const unsigned b = __ballot_sync(~0u, inc >= need);
    const int L = __ffs(b) - 1;
    if (lane == L) { s_lo = 31 - L; s_gt = s_above + inc - v; }
  }
  __syncthreads();
  long long cp2 = clock64();
  const uint32_t thr = ((uint32_t)bin << 5) | (uint32_t)s_lo;
  const int gt = s_gt, quota = TOPN - gt;
  // emission: > thr in any order (sorted below); ties: the `quota` lowest keys (warp 0 scans in key order)
  const int E = (NK + TR_T - 1) / TR_T, k0 = tid * E, k1 = min(NK, k0 + E);
  int nt = 0;
  for (int i = k0; i < k1; ++i) {
    const uint32_t u = ord16(lg[i]);
    if (u > thr) { const int p = atomicAdd(&s_ctr, 1); ck[p] = i; co[p] = u; }
    nt += u == thr;
  }
  // exclusive scan of tie counts in key order (thread order = key order)
  int inc = nt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += x; }
  if (lane == 31) s_part[tid >> 5] = inc;
  __syncthreads();
  if (tid < 32) {
    int w = s_part[lane], wi = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(~0u, wi, o); if (lane >= o) wi += x; }
    s_part[32 + lane] = wi - w;
  }
  __syncthreads();
  int pos = s_part[32 + (tid >> 5)] + inc - nt;
  if (nt && pos < quota)
    for (int i = k0; i < k1 && pos < quota; ++i)
      if (ord16(lg[i]) == thr) { ck[gt + pos] = i; co[gt + pos] = thr; ++pos; }
  __syncthreads();
  long long cem = clock64();
  if (tid < TOPN) {
    const int myk = ck[tid]; const uint32_t myo = co[tid];
    int rank = 0;
    for (int j = 0; j < TOPN; ++j) { const uint32_t oj = co[j]; const int kj = ck[j]; rank += (oj > myo) | ((oj == myo) & (kj < myk)); }
    out_key[(size_t)row * TOPN + rank] = myk;
    out_sc[(size_t)row * TOPN + rank] = (myo & 0x8000) ? (uint16_t)(myo & 0x7FFF) : (uint16_t)(~myo & 0xFFFF);
  }
  __syncthreads();
  long long c1 = clock64(), g1 = gtime();
  if (tid == 0) { long long* s = st + (size_t)row * 6; s[0] = c0; s[1] = cp1; s[2] = cp2; s[3] = cem; s[4] = c1; s[5] = g1 - g0; }
}
void topn_row_run(torch::Tensor logits, int64_t topn, torch::Tensor out_key, torch::Tensor out_sc, torch::Tensor st) {
  const int rows = logits.size(0) * logits.size(1), NK = logits.size(2);
  size_t smem = (2048 + 1024) * 4 + (size_t)topn * 8 + (size_t)NK * 2 + 64;
  auto s = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(topn_row_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  topn_row_kernel<<<rows, TR_T, smem, s>>>((const uint16_t*)logits.data_ptr(), NK, (int)topn, out_key.data_ptr<int>(),
                                           (uint16_t*)out_sc.data_ptr(), (long long*)st.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
