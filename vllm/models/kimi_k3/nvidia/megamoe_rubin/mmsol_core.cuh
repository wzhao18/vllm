// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (never for merge): MM-SOL skeleton core (prefill/designs/megamoe-sol.md sec 2 / 4).
// sm_107a device helpers shared by the A0 TMEM layout gate (mmsol_a0.cu) and the in-step skel (mmsol_skel.cu):
// 2-CTA mxf4nvf4 (NVFP4, K128 2x-K) and mxf8f6f4 (MXFP8, K64 2x-K) MMAs, the 576-column exclusive TMEM carve,
// tcgen05.cp SF copies (32x128b.warpx4, the production Cp4x32x128bOp), cta_group::2 TMA loads that complete on the
// leader CTA's barrier, mbarrier / cluster primitives.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MMD __device__ __forceinline__

namespace mm {

// ---------------- TMEM plan (576 columns, exclusive): design sec 2 "TMEM plan" ----------------
constexpr int NA = 208, NB = 96;                 // routed pieces a (double-buffered) and b
constexpr int COL_A0 = 0, COL_A1 = 208, COL_B = 416;
constexpr int COL_SFA = 512, COL_SFBA = 528, COL_SFBB = 560;   // 16 + 32 + 16 = 64 SF columns
constexpr int TMEM_COLS = 576;
// SE (MXFP8): 2 x 256 accumulators + SFA 8 (2 MMAs x 1 chunk x 4 cols) + SFB 8 (N256: 2 atoms x 4 cols)
constexpr int COL_SE0 = 0, COL_SE1 = 256, COL_SE_SFA = 512, COL_SE_SFB = 528;

MMD uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
MMD uint64_t now_ns() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
MMD uint64_t clk64() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
MMD uint32_t cta_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
MMD uint32_t smid() { uint32_t r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
MMD uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
MMD void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// ---------------- mbarrier ----------------
MMD void bar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
MMD void bar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
MMD void bar_expect(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
MMD void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
// arrive on a barrier of any CTA in the cluster (b = shared::cluster address from mapa)
MMD void bar_arrive_cluster(uint32_t b) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(b) : "memory");
}
MMD void bar_wait(uint32_t b, uint32_t ph) {
  asm volatile("{.reg .pred P; W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W%=;}"
               :: "r"(b), "r"(ph) : "memory");
}
MMD bool bar_test(uint32_t b, uint32_t ph) {
  uint32_t ok;
  asm volatile("{.reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
               : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  return ok != 0;
}

// ---------------- TMA (cta_group::2: completes on the LEADER CTA's barrier) ----------------
MMD void tma2d_cg2(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t leader_bar) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(leader_bar) : "memory");
}
MMD void tma2d(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
MMD void prefetch_tmap(const CUtensorMap* m) { asm volatile("prefetch.tensormap [%0];" :: "l"(m) : "memory"); }
// 1D bulk smem -> global (local or peer address), bulk-group completion
MMD void bulk_store(uint64_t dst, uint32_t src, uint32_t n) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(n) : "memory");
}
MMD void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> MMD void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> MMD void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }

// ---------------- smem descriptors ----------------
// K-major SWIZZLE_128B operand (rows of 128 B, 8-row atoms 1024 B apart): LBO 16 B (ignored), SBO 1024 B, version 1.
MMD uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x7FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32)  /* 15-bit addr (>256 KiB) */ | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
// tcgen05.cp 32x128b source: one 512 B chunk = 32 rows x 16 B, no swizzle (core matrices 8 x 16 B, 128 B apart).
MMD uint64_t desc_sf(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(128 >> 4) << 32) |
         ((uint64_t)1 << 46);
}
// instruction descriptors
// mxf4nvf4 block16 (NVFP4, UE4M3 scales), K128 = bit 3 (2x-K), A/B E2M1 (1), K-major, M256 (cta_group::2)
MMD uint32_t idesc_nvf4(int n) { return (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | ((256u >> 4) << 24); }
// mxf8f6f4 (MXFP8 E4M3, UE8M0 vec32), K64 = bit 31, M256
MMD uint32_t idesc_mx8(int n) { return (1u << 23) | ((uint32_t)(n >> 3) << 17) | ((256u >> 4) << 24) | (1u << 31); }

// ---------------- tcgen05 ----------------
MMD void tmem_alloc576(uint32_t dst_smem) {
  asm volatile("tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(TMEM_COLS) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
MMD void tmem_dealloc576(uint32_t t) {
  asm volatile("tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(t), "r"(TMEM_COLS) : "memory");
}
MMD void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
MMD void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// warp-uniform issue with elect.sync inside the asm (hw28c rule)
MMD void mma_nvf4(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0;\n"
               "@P tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
MMD void mma_mx8(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0;\n"
               "@P tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
MMD void sf_cp(uint32_t taddr, uint64_t sdesc) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff;\n"
               "@P tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}" :: "r"(taddr), "l"(sdesc) : "memory");
}
// commit all prior tcgen05 ops of this thread -> arrive once on the barrier at the same smem offset in every CTA of mask
MMD void commit_mc(uint32_t bar, uint16_t mask) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff;\n"
               "@P tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;}"
               :: "r"(bar), "h"(mask) : "memory");
}
#define MM_LD16(taddr, r)                                                                                           \
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),      \
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) \
               : "r"(taddr))
MMD void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// bounded spin on a global counter (monotone, epoch-tagged); traps after limit_ns
MMD uint32_t ld_acq_sys(uint64_t a) {
  uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
MMD uint32_t ld_acq_gpu(uint64_t a) {
  uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
MMD void red_add_sys(uint64_t a, uint32_t v) { asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }
MMD void red_add_gpu(uint64_t a, uint32_t v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }
MMD uint32_t atom_add_gpu(uint64_t a, uint32_t v) {
  uint32_t r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(a), "r"(v) : "memory"); return r;
}

}  // namespace mm
