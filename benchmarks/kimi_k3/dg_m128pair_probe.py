# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""prefill-port probe: cta_group::2 tcgen05 building blocks for the route-E fp8 kernel.

One 2-CTA cluster, 128 threads per CTA. Each CTA copies a host-built SMEM byte image
(operands already in their swizzled layouts) and a TMEM image (scale factors, written
with tcgen05.st 32x32b by the 4 warps), the leader CTA issues a short program of
2-CTA MMAs, the commit multicasts to both CTAs, and both CTAs dump 128 lanes x NCOL
columns of TMEM. The host deduces / checks layouts against torch.

  python prefill_port_probe.py        runs all checks and prints the deduced layouts
"""

import sys

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm

SMEM = 64 * 1024
NCOL = 256  # TMEM columns dumped per lane
SF_COL = 256  # TMEM image written at columns [256, 256 + NSF)
NSF = 16


@asm()
def ld_global_v4(ptr):
    return "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l", (Int64(ptr),), (Uint32,) * 4


@asm()
def st_shared_v4(addr, w0, w1, w2, w3):
    return "st.shared.v4.b32 [$0], {$1,$2,$3,$4};", "r,r,r,r,r,~{memory}", (
        Uint32(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))


@asm()
def st_global_u32(ptr, v):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(ptr), Uint32(v))


@asm(Uint32, pure=False)
def ld_global_u32(ptr):
    return "ld.global.nc.b32 $0, [$1];", "=r,l", (Int64(ptr),)


@asm(Uint32, pure=False)
def ld_shared_u32(addr):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Uint32(addr),)


@asm(Uint32, pure=True)
def cta_rank():
    return "mov.u32 $0, %cluster_ctarank;", "=r", ()


@asm()
def cluster_sync():
    return "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}", ()


@asm()
def tmem_alloc2(dst):
    code = "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [$0], 512; "
    code += "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def tmem_free2(taddr):
    return "tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, 512;", "r,~{memory}", (Uint32(taddr),)


@asm()
def tmem_st1(addr, w):
    return ("tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1}; tcgen05.wait::st.sync.aligned;",
            "r,r,~{memory}", (Uint32(addr), Uint32(w)))


@asm(Uint32, pure=False)
def tmem_ld1(addr):
    return ("tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1]; tcgen05.wait::ld.sync.aligned;",
            "=r,r,~{memory}", (Uint32(addr),))


@asm()
def tc_fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def tc_fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def fence_async_smem():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def mbar_init(bar, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", (Uint32(bar), Uint32(count))


@asm()
def mbar_wait(bar, parity):
    code = ("{ .reg .pred p; WAIT_${:uid}: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; "
            "@!p bra WAIT_${:uid}; }")
    return code, "r,r,~{memory}", (Uint32(bar), Uint32(parity))


@asm()
def mbar_arrive_local(bar):
    return "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", (Uint32(bar),)


@asm()
def mma2_mx(d, a, b, idesc, sfa, sfb, acc):
    code = "{ .reg .pred p; setp.ne.b32 p, $6, 0; "
    code += "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$4], [$5], p; }"
    return code, "r,l,l,r,r,r,r,~{memory}", (
        Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(sfa), Uint32(sfb), Uint32(acc))


@asm()
def mma2_f16(d, a, b, idesc, acc):
    code = "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
    code += "tcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, p; }"
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(acc))


@asm()
def commit2_mc(bar):
    code = "{ .reg .b16 m; mov.b16 m, 3; "
    code += "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], m; }"
    return code, "r,~{memory}", (Uint32(bar),)


@asm(Int64, pure=False)
def clk64():
    return "mov.u64 $0, %clock64;", "=l", ()


@asm()
def mma1_mx(d, a, b, idesc, sfa, sfb, acc):
    code = "{ .reg .pred p; setp.ne.b32 p, $6, 0; "
    code += "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$4], [$5], p; }"
    return code, "r,l,l,r,r,r,r,~{memory}", (
        Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(sfa), Uint32(sfb), Uint32(acc))


@asm()
def mma2_f8(d, a, b, idesc, acc):
    code = "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
    code += "tcgen05.mma.cta_group::2.kind::f8f6f4 [$0], $1, $2, $3, p; }"
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(acc))


@asm()
def tc_cp2_64x128(taddr, desc):
    return ("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [$0], $1;", "r,l,~{memory}", (Uint32(taddr), Uint64(desc)))


@asm()
def mma2_mx_ts(d, a_tmem, b, idesc, sfa, sfb, acc):
    code = "{ .reg .pred p; setp.ne.b32 p, $6, 0; "
    code += "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], [$1], $2, $3, [$4], [$5], p; }"
    return code, "r,r,l,r,r,r,r,~{memory}", (
        Uint32(d), Uint32(a_tmem), Uint64(b), Uint32(idesc), Uint32(sfa), Uint32(sfb), Uint32(acc))


@asm()
def commit1(bar):
    return ("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}", (Uint32(bar),))


class Probe:
    """prog: list of (kind 'mx'|'f16'|'mx1'|'f8', a_off, a_bits, b_off, b_bits, idesc, sfa_col, sfb_col, acc, d_col).
    reps > 1: the program is issued ``reps`` times and the leader records clock64 from the first
    issue to the commit landing into out[0, 0, 0:2] (as int bits)."""

    def __init__(self, prog, reps=1):
        self.prog = prog
        self.reps = reps

    @cute.jit
    def __call__(self, simg, timg, out, stream):
        self.kernel(simg, timg, out).launch(grid=(2, 1, 1), block=(128, 1, 1), cluster=(2, 1, 1),
                                            smem=SMEM + 2048, stream=stream)

    @cute.kernel
    def kernel(self, simg, timg, out):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, SMEM + 2048, byte_alignment=1024)
        base = (Uint32(raw.toint()) + Uint32(1023)) & Uint32(0xFFFFFC00)
        bar = base + Uint32(SMEM)
        tslot = bar + Uint32(64)
        rank = cta_rank()
        if tid == 0:
            mbar_init(bar, Uint32(1))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc2(tslot)
        sb = simg.iterator.toint() + Int64(rank) * Int64(SMEM)
        for i in range(tid, SMEM // 16, 128):
            w = ld_global_v4(sb + Int64(i * 16))
            st_shared_v4(base + Uint32(i * 16), w[0], w[1], w[2], w[3])
        fence_async_smem()
        cute.arch.sync_threads()
        taddr = ld_shared_u32(tslot)
        lane_addr = taddr + (Uint32(warp * 32) << Uint32(16))
        tb = timg.iterator.toint() + Int64(rank) * Int64(128 * NSF * 4) + Int64(tid * NSF * 4)
        for c in cutlass.range_constexpr(NSF):
            tmem_st1(lane_addr + Uint32(SF_COL + c), ld_global_u32(tb + Int64(c * 4)))
        # zero D region
        for c in range(NCOL):
            tmem_st1(lane_addr + Uint32(c), Uint32(0))
        tc_fence_before()
        cluster_sync()
        tc_fence_after()
        t0 = clk64()
        one = cutlass.const_expr(self.prog[0][0] == "mx1")
        if rank == 0 and tid == 0:
            for _r in range(self.reps):
                for (kind, a_off, a_bits, b_off, b_bits, idesc, sfa, sfb, acc, dcol) in self.prog:
                    ad = Uint64(a_bits) | Uint64((base + Uint32(a_off)) >> Uint32(4))
                    bd = Uint64(b_bits) | Uint64((base + Uint32(b_off)) >> Uint32(4))
                    if cutlass.const_expr(kind == "mx"):
                        mma2_mx(taddr + Uint32(dcol), ad, bd, Uint32(idesc), taddr + Uint32(SF_COL + sfa),
                                taddr + Uint32(SF_COL + sfb), Uint32(acc))
                    elif cutlass.const_expr(kind == "mx1"):
                        mma1_mx(taddr + Uint32(dcol), ad, bd, Uint32(idesc), taddr + Uint32(SF_COL + sfa),
                                taddr + Uint32(SF_COL + sfb), Uint32(acc))
                    elif cutlass.const_expr(kind == "cp"):
                        for sub in cutlass.range_constexpr(8):
                            tc_cp2_64x128(taddr + Uint32(idesc + 4 * sub), ad + Uint64(sub))
                    elif cutlass.const_expr(kind == "mxts"):
                        mma2_mx_ts(taddr + Uint32(dcol), taddr + Uint32(a_off), bd, Uint32(idesc),
                                   taddr + Uint32(SF_COL + sfa), taddr + Uint32(SF_COL + sfb), Uint32(acc))
                    elif cutlass.const_expr(kind == "f8"):
                        mma2_f8(taddr + Uint32(dcol), ad, bd, Uint32(idesc), Uint32(acc))
                    else:
                        mma2_f16(taddr + Uint32(dcol), ad, bd, Uint32(idesc), Uint32(acc))
            if cutlass.const_expr(one):
                commit1(bar)
            else:
                commit2_mc(bar)
        if cutlass.const_expr(one):
            if rank == 1 and tid == 0:
                mbar_arrive_local(bar)
        mbar_wait(bar, Uint32(0))
        t1 = clk64()
        tc_fence_after()
        ob = out.iterator.toint() + Int64(rank) * Int64(128 * NCOL * 4) + Int64(tid * NCOL * 4)
        for c in range(NCOL):
            st_global_u32(ob + Int64(c * 4), tmem_ld1(lane_addr + Uint32(c)))
        if cutlass.const_expr(self.reps > 1):
            if tid == 0:
                st_global_u32(ob, Uint32(t1 - t0))
        tc_fence_before()
        cluster_sync()
        if warp == 0:
            tc_fence_after()
            tmem_free2(taddr)


_CACHE = {}


def run(prog, simg, timg, reps=1):
    """simg [2, SMEM] u8, timg [2, 128, NSF] u32 -> D [2, 128 lanes, NCOL] f32."""
    import cuda.bindings.driver as cu
    from cutlass.cute.runtime import from_dlpack

    out = torch.zeros((2, 128, NCOL), dtype=torch.float32, device="cuda")
    args = [simg.contiguous().view(-1), timg.contiguous().view(torch.uint8).view(-1), out.view(-1)]
    cargs = [from_dlpack(a, assumed_align=16) for a in args]
    st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    key = str(prog) + str(reps)
    fn = _CACHE.get(key)
    if fn is None:
        fn = _CACHE[key] = cute.compile(Probe(prog, reps), *cargs, st)
    fn(*cargs, st)
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# host layouts
# ---------------------------------------------------------------------------
def desc_bits(lbo, sbo, swz):  # swz: 0 none, 2 = 128B, 4 = 64B, 6 = 32B
    return ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (swz << 61)


SW128_K = desc_bits(16, 1024, 2)


def idesc_mx(m, n, a_mn=False, b_mn=False, a_sf=0, b_sf=0):
    return ((n >> 3) << 17) | (1 << 23) | ((m >> 7) << 27) | (int(a_mn) << 15) | (int(b_mn) << 16) \
        | (a_sf << 29) | (b_sf << 4)


def idesc_bf16(m, n, a_mn=False, b_mn=False):
    return (1 << 4) | (1 << 7) | (1 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24) | (int(a_mn) << 15) | (int(b_mn) << 16)


def sw128_kmajor(x):
    """x [rows, 128] u8 -> SW128 K-major image (rows x 128 B; chunk c of row r at c ^ (r % 8))."""
    rows = x.shape[0]
    out = torch.empty_like(x)
    xc = x.view(rows, 8, 16)
    oc = out.view(rows, 8, 16)
    r = torch.arange(rows, device=x.device)
    for c in range(8):
        oc[r, (c ^ (r % 8))] = xc[:, c]
    return out


def sw64_rows(x):
    """x [rows, 64] u8 -> SW64 image (chunk c of row r at c ^ ((r >> 1) % 4))."""
    rows = x.shape[0]
    out = torch.empty_like(x)
    xc = x.view(rows, 4, 16)
    oc = out.view(rows, 4, 16)
    r = torch.arange(rows, device=x.device)
    for c in range(4):
        oc[r, (c ^ ((r >> 1) % 4))] = xc[:, c]
    return out


def e4m3(t):
    return t.to(torch.float8_e4m3fn).view(torch.uint8)


def dev():
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def d_layout_check(D, ref):
    """D [2, 128, ncols] vs ref [128 rows, N] under the hypothesis
    lane L of CTA c -> row 64c + L % 64, column j -> n = j + (N/2) * (L // 64)."""
    n = ref.shape[1]
    got = torch.empty_like(ref)
    for c in range(2):
        for L in range(128):
            got[64 * c + L % 64, (n // 2) * (L // 64): (n // 2) * (L // 64) + n // 2] = D[c, L, : n // 2]
    return ((got - ref).abs().max() / ref.abs().max()).item()


def test_f16():
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(128, 64, generator=g, device="cuda").bfloat16()  # M x K
    B = torch.randn(128, 64, generator=g, device="cuda").bfloat16()  # N x K
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(A[64 * c: 64 * c + 64].contiguous().view(torch.uint8).view(64, 128)).view(-1)
        simg[c, 8192:16384] = sw128_kmajor(B[64 * c: 64 * c + 64].contiguous().view(torch.uint8).view(64, 128)).view(-1)
    prog = [("f16", 32 * k, SW128_K, 8192 + 32 * k, SW128_K, idesc_bf16(128, 128), 0, 0, int(k > 0), 0) for k in range(4)]
    D = run(prog, simg, torch.zeros(2, 128, NSF, dtype=torch.int32, device="cuda"))
    ref = A.float() @ B.float().T
    print(f"f16 2cta M128 N128 K64, D layout hypothesis rel err {d_layout_check(D, ref):.2e}")


def sf_bytes_to_words(b):
    """b [..] int64 byte values -> int32 words with the byte replicated 4x."""
    w = b * 0x01010101
    w = torch.where(w >= (1 << 31), w - (1 << 32), w)
    return w.to(torch.int32)


def sf_image(fill=127):
    return torch.full((2, 128, NSF), fill, dtype=torch.int64, device="cuda")


def code_of(L):
    return L + 1 if L < 64 else -(L - 63)


def deduce_sf(which):
    """A = B = 1.0 e4m3, K = 32: D = 32 * 2^(ea + eb). For each CTA and TMEM column of the
    SF region, mark it with a lane code and see which D rows/cols pick it up."""
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    one = e4m3(torch.ones(64, 128, device="cuda"))
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(one).view(-1)
        simg[c, 8192:16384] = sw128_kmajor(one).view(-1)
    sfa, sfb = (0, 8) if which == "A" else (8, 0)
    prog = [("mx", 0, SW128_K, 8192, SW128_K, idesc_mx(128, 128), sfa, sfb, 0, 0)]
    res = {}
    for cta in range(2):
        for col in range(8):
            t = sf_image()
            for L in range(128):
                t[cta, L, col] = 127 + code_of(L)
            D = run(prog, simg, sf_bytes_to_words(t))
            e = torch.log2(D[:, :, :64] / 32.0).round().long()  # [2, 128, 64]
            hits = (e != 0).nonzero().tolist()
            if hits:
                res[(cta, col)] = e
    # translate: for each (cta, col) with hits, map D element -> (logical row/col) -> source lane
    for (cta, col), e in res.items():
        inv = {code_of(L): L for L in range(128)}
        lines = {}
        for dc in range(2):
            for L in range(128):
                for j in range(64):
                    v = int(e[dc, L, j])
                    if v == 0:
                        continue
                    row = 64 * dc + L % 64
                    n = j + 64 * (L // 64)
                    key = row if which == "A" else n
                    lines.setdefault(key, set()).add(inv.get(v, -999))
        ks = sorted(lines)
        summ = {k: sorted(lines[k]) for k in ks}
        # compress: print a few
        sample = [(k, summ[k]) for k in ks[:: max(1, len(ks) // 8)]]
        print(f"SF{which}: marked cta {cta} col {col}: {len(ks)} {'rows' if which == 'A' else 'cols'} hit, "
              f"(logical idx -> source lanes) sample {sample}")


def test_pv():
    """O[h, d] = P'[h, kv] V[kv, d]: A = P' K-major SW128 (64 heads x 128 kv per CTA),
    B = V MN-major SW64 (per CTA 64 dims x 128 kv = 128 kv rows of 64 B), mxf8, K = 128."""
    g = torch.Generator(device="cuda").manual_seed(1)
    P = torch.rand(128, 128, generator=g, device="cuda") * 4
    V = torch.randn(128, 128, generator=g, device="cuda")  # kv x dims (N = 128 over both CTAs)
    P8, V8 = e4m3(P), e4m3(V)
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(P8[64 * c: 64 * c + 64]).view(-1)
        simg[c, 16384:16384 + 8192] = sw64_rows(V8[:, 64 * c: 64 * c + 64].contiguous()).view(-1)
    ref = P8.view(torch.float8_e4m3fn).float() @ V8.view(torch.float8_e4m3fn).float()
    for lbo, sbo in ((16, 512), (512, 2048), (2048, 512), (8192, 512)):
        bits = desc_bits(lbo, sbo, 4)
        prog = [("mx", 32 * k, SW128_K, 16384 + 2048 * k, bits, idesc_mx(128, 128, b_mn=True), 0, 0, int(k > 0), 0)
                for k in range(4)]
        D = run(prog, simg, sf_bytes_to_words(sf_image()))
        print(f"PV mxf8 2cta A K-major SW128, B MN-major SW64 lbo {lbo} sbo {sbo}: rel err {d_layout_check(D, ref):.2e}")
    # QK with scale factors per the deduced layouts + bf16 rope into the same accumulator
    Q = torch.randn(128, 128, generator=g, device="cuda") * 50
    K = torch.randn(128, 128, generator=g, device="cuda") * 50
    ea = torch.randint(-3, 4, (128,), generator=g, device="cuda")
    eb = torch.randint(-3, 4, (128,), generator=g, device="cuda")
    Q8 = e4m3(Q / torch.exp2(ea.float()).view(-1, 1))
    K8 = e4m3(K / torch.exp2(eb.float()).view(-1, 1))
    Qr = torch.randn(128, 64, generator=g, device="cuda").bfloat16()
    Kr = torch.randn(128, 64, generator=g, device="cuda").bfloat16()
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(Q8[64 * c: 64 * c + 64]).view(-1)
        simg[c, 8192:16384] = sw128_kmajor(K8[64 * c: 64 * c + 64]).view(-1)
        simg[c, 16384:24576] = sw128_kmajor(Qr[64 * c: 64 * c + 64].contiguous().view(torch.uint8).view(64, 128)).view(-1)
        simg[c, 24576:32768] = sw128_kmajor(Kr[64 * c: 64 * c + 64].contiguous().view(torch.uint8).view(64, 128)).view(-1)
    t = sf_image()
    L = torch.arange(128, device="cuda")
    for c in range(2):
        for col in range(2):
            t[c, :, col] = 127 + ea[64 * c + L % 64]  # SFA: lane L -> row L % 64 (cols 0, 1)
            t[c, :, 8 + col] = 127 + eb[64 * (L // 64) + 32 * col + L % 32]  # SFB
    prog = [("mx", 32 * k, SW128_K, 8192 + 32 * k, SW128_K, idesc_mx(128, 128), 0, 8, int(k > 0), 0) for k in range(4)]
    prog += [("f16", 16384 + 32 * k, SW128_K, 24576 + 32 * k, SW128_K, idesc_bf16(128, 128), 0, 0, 1, 0) for k in range(4)]
    D = run(prog, simg, sf_bytes_to_words(t))
    ref = (Q8.view(torch.float8_e4m3fn).float() * torch.exp2(ea.float()).view(-1, 1)) @ \
        (K8.view(torch.float8_e4m3fn).float() * torch.exp2(eb.float()).view(-1, 1)).T + Qr.float() @ Kr.float().T
    print(f"QK mxf8 2cta with SFA/SFB layouts + bf16 rope: rel err {d_layout_check(D, ref):.2e}")


def test_rate():
    """Clock per MMA for a long stream of MMAs (independent of data)."""
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    simg[:, :32768] = e4m3(torch.ones(2, 32768, device="cuda") * 0.5)
    t = sf_bytes_to_words(sf_image())
    R = 64
    cases = {
        "2cta mxf8 M128 N128 K32 (QK)": [("mx", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, idesc_mx(128, 128), 0, 8, 1, 0) for k in range(4)],
        "2cta mxf8 M128 N256 K32": [("mx", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, idesc_mx(128, 256), 0, 8, 1, 0) for k in range(4)],
        "2cta mxf8 M256 N128 K32": [("mx", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, idesc_mx(256, 128), 0, 8, 1, 0) for k in range(4)],
        "2cta f8f6f4 (no scale) M128 N128 K32": [("f8", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, (1 << 4) * 0 | ((128 >> 3) << 17) | ((128 >> 4) << 24), 0, 0, 1, 0) for k in range(4)],
        "2cta bf16 M128 N128 K16": [("f16", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, idesc_bf16(128, 128), 0, 0, 1, 0) for k in range(4)],
        "2cta mxf8 PV M128 N128 K32 B MN SW64": [("mx", 32 * k, SW128_K, 16384 + 2048 * k, desc_bits(16, 512, 4), idesc_mx(128, 128, b_mn=True), 0, 8, 1, 0) for k in range(4)],
        "2cta mxf8 4 accumulators (PV order)": [("mx", 32 * k, SW128_K, 16384 + 2048 * k, desc_bits(16, 512, 4), idesc_mx(128, 128, b_mn=True), 0, 8, 1, 32 * b) for b in range(4) for k in range(4)],
        "1cta mxf8 M128 N128 K32": [("mx1", 32 * k, SW128_K, 16384 + 32 * k, SW128_K, idesc_mx(128, 128), 0, 8, 1, 0) for k in range(4)],
    }
    for name, prog in cases.items():
        D = run(prog, simg, t, reps=R)
        clk = D[0, 0, 0].view(torch.int32).item()
        n = R * len(prog)
        print(f"{name:42s}: {clk / n:7.1f} clk/MMA ({n} MMAs, {clk} clk)")


def test_pv256():
    """PV with FlashMLA's V split: CTA c holds dims 256c..256c+255 as two MN-major SW128 tiles
    [128 kv x 128 B]; MMA m (N=256): CTA0 dims 128m.., CTA1 dims 256+128m..; D lanes 0-63 = CTA0's
    half, 64-127 = CTA1's half."""
    g = torch.Generator(device="cuda").manual_seed(2)
    P = torch.rand(128, 128, generator=g, device="cuda") * 4
    V = torch.randn(128, 512, generator=g, device="cuda")
    P8, V8 = e4m3(P), e4m3(V)
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(P8[64 * c: 64 * c + 64]).view(-1)
        for m in range(2):
            d0 = 256 * c + 128 * m
            simg[c, 16384 + 16384 * m: 32768 + 16384 * m] = sw128_kmajor(V8[:, d0: d0 + 128].contiguous()).view(-1)
    Pf, Vf = P8.view(torch.float8_e4m3fn).float(), V8.view(torch.float8_e4m3fn).float()
    ref = Pf @ Vf  # [128 heads, 512]
    t = sf_image()
    t[:, :, 8:16] = 127
    for lbo in (16, 8192, 16384):
        bits = desc_bits(lbo, 1024, 2)
        prog = [("mx", 32 * k, SW128_K, 16384 + 16384 * m + 4096 * k, bits, idesc_mx(128, 256, b_mn=True), 0, 8,
                 int(k > 0), 128 * m) for m in range(2) for k in range(4)]
        D = run(prog, simg, sf_bytes_to_words(t))
        got = torch.empty_like(ref)
        for c in range(2):
            for L in range(128):
                hf = L // 64
                got[64 * c + L % 64, 256 * hf: 256 * hf + 256] = torch.cat([D[c, L, 0:128], D[c, L, 128:256]]) if NCOL >= 256 else 0
        print(f"PV N256 MN-major SW128 lbo {lbo}: rel err {((got - ref).norm() / ref.norm()).item():.2e}")


def test_ats():
    """QK with A (Q, 64 rows x 128 B per CTA) copied smem->TMEM by tcgen05.cp 64x128b.warpx2::02_13
    (8 x 16-B subtiles -> 4 cols each) and used as the TMEM A operand (8 cols per K32 step)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    Q = torch.randn(128, 128, generator=g, device="cuda") * 10
    K = torch.randn(128, 128, generator=g, device="cuda") * 10
    Q8, K8 = e4m3(Q), e4m3(K)
    simg = torch.zeros(2, SMEM, dtype=torch.uint8, device="cuda")
    for c in range(2):
        simg[c, 0:8192] = sw128_kmajor(Q8[64 * c: 64 * c + 64]).view(-1)
        simg[c, 8192:16384] = sw128_kmajor(K8[64 * c: 64 * c + 64]).view(-1)
    ref = Q8.view(torch.float8_e4m3fn).float() @ K8.view(torch.float8_e4m3fn).float().T
    t = sf_bytes_to_words(sf_image())
    QT = 160  # tmem column of A
    prog = [("cp", 0, SW128_K, 0, SW128_K, QT, 0, 0, 0, 0)]
    prog += [("mxts", QT + 8 * k, SW128_K, 8192 + 32 * k, SW128_K, idesc_mx(128, 128), 0, 8, int(k > 0), 0) for k in range(4)]
    D = run(prog, simg, t)
    print(f"QK A-from-TMEM (cp 64x128b warpx2::02_13): rel err {d_layout_check(D, ref):.2e}")


def main():
    torch.manual_seed(0)
    tests = sys.argv[1:] or ["f16", "sfa", "sfb"]
    if "f16" in tests:
        test_f16()
    if "sfa" in tests:
        deduce_sf("A")
    if "sfb" in tests:
        deduce_sf("B")
    if "pv" in tests:
        test_pv()
    if "pv256" in tests:
        test_pv256()
    if "ats" in tests:
        test_ats()
    if "rate" in tests:
        test_rate()


if __name__ == "__main__":
    main()
