# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC ONLY (never merge): the DSTAMP arm (16-rank DSA chain + MoE route chain,
clocks aligned across ranks). Combines

* the STAMP-1125a untraced anchor: every PDL-wait return of a CuTe kernel traced from
  specialized_decode/ or dcp_topk_merge/ stores {%globaltimer, site << 40 | gridid} into a
  2^16-entry gridid-keyed ring (CTA (0,0,0) thread 0, one plain store);
* per-CTA %globaltimer rings (routediag layout: [RING, CTAS, 8] int64 + [CTAS] int32
  call counters, slot 6 = %smid) for the DSA chain's CuTe kernels (``ring_begin`` /
  ``ring_stamp``: DcpValue(pushed) "dcpv", the #63 pack "pack");
* device-side call rings for the dsac (csrc dsa_core.cu ``dbg``: [RING, MAXC, 24]) and
  the #63 select ("X1": DCPTopkSelectKernel ``stamps``: [RING, MAXR, 8]), each CTA's
  call = atomicAdd(counter[cta], 1) % RING (every CTA of a grid runs once per call);
* a cross-rank clock sync (``clocksync``: NTP-style ping-pong over symmetric memory,
  rank 0 <-> every rank, min-RTT sample; run twice per dump for the drift).

``dump(case)`` (forward bench, after each case) writes
``$OA_DIAG_DIR/<job>.<step>/<case>.rank<R>.raw.pt`` (+ .sites.json; STAMP format: anchor)
and ``<case>.rank<R>.dsa.pt`` (rings + clock sync); rings are zeroed after each dump.
Every buffer address the CuTe kernels bake in is per process, so the specialized cute
cache is private (``VLLM_KIMI_K3_CUTE_CACHE_DIR``)."""

import json
import os
import sys
import time
import zlib

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int64, Uint32, Uint64

from .primitives import asm

os.environ["VLLM_KIMI_K3_CUTE_CACHE_DIR"] = f"/tmp/dsadiag-cutecache-{os.getpid()}"

# dsac (dsa_core.cu) and X1 select (dcp_topk_merge) device rings: keep in sync with
# DSAC_RING / DSAC_MAXC in dsa_core.cu and _SEL_RING / _SEL_MAXR in dcp_topk_merge.
DSAC_RING, DSAC_MAXC, DSAC_SLOTS = 64, 2048, 24
SEL_RING, SEL_MAXR, SEL_SLOTS = 64, 256, 8
RING, SLOTS = 64, 8
ANCHOR_LOG2 = 16

_RINGS: dict = {}  # (tag, ctas) -> (ring [RING, ctas, SLOTS], ctr [ctas])
_DEV: dict = {}  # kind -> flat int64 buffer (dsac / sel)
_RAW: dict = {}
_SITES: dict = {}
_SIG: dict = {}


@asm(Int64)
def gtimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint32)
def smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Uint32)
def atom_inc(addr):
    return "atom.global.add.u32 $0, [$1], 1;", "=r,l,~{memory}", (addr,)


@asm()
def st64(addr, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (addr, value)


# ------------------------------------------------------------------ CuTe per-CTA rings
def ring_buffer(tag, ctas):
    """(ring address, counter address) of ``tag`` as Python ints (trace time)."""
    key = (tag, int(ctas))
    if key not in _RINGS:
        dev = torch.cuda.current_device()
        ring = torch.zeros(RING, key[1], SLOTS, dtype=torch.int64, device=dev)
        ctr = torch.zeros(key[1], dtype=torch.int32, device=dev)
        _RINGS[key] = (ring, ctr)
    ring, ctr = _RINGS[key]
    return ring.data_ptr(), ctr.data_ptr()


@cute.jit
def ring_begin(base: cutlass.Constexpr, ctr: cutlass.Constexpr, ctas: cutlass.Constexpr, cta, leader):
    """Slot 0 (time) + slot 6 (%smid) by ``leader`` of CTAs < ``ctas``; returns the row
    address (0 elsewhere)."""
    addr = Int64(0)
    if leader & (cta < ctas):
        call = atom_inc(Int64(ctr) + Int64(cta) * 4) % RING
        addr = Int64(base) + (Int64(call) * ctas + Int64(cta)) * (SLOTS * 8)
        st64(addr, gtimer())
        st64(addr + 48, Int64(smid()))
    return addr


@cute.jit
def ring_stamp(addr, slot: cutlass.Constexpr):
    if addr != Int64(0):
        st64(addr + slot * 8, gtimer())


# ------------------------------------------------------------------ device rings
def _dev(kind):
    if kind not in _DEV:
        ring, maxc, slots = {"dsac": (DSAC_RING, DSAC_MAXC, DSAC_SLOTS),
                             "sel": (SEL_RING, SEL_MAXR, SEL_SLOTS)}[kind]
        dev = torch.cuda.current_device()
        # ring [ring, maxc, slots] int64 then [maxc] u32 counters (maxc / 2 int64)
        _DEV[kind] = (torch.zeros(ring * maxc * slots + maxc // 2, dtype=torch.int64, device=dev),
                      ring, maxc, slots)
    return _DEV[kind][0]


def dsac_dbg():
    """The dsac ``dbg`` buffer (DsaCore.run): device call ring, see dsa_core.cu."""
    return _dev("dsac")


def select_stamps():
    """The #63 select's ``stamps`` buffer (dcp_topk_select): device call ring."""
    return _dev("sel")


# ------------------------------------------------------------------ anchor (STAMP-1125a)
@asm(Uint64)
def gridid():
    return "mov.u64 $0, %gridid;", "=l", ()


def raw_buffer(tag, rows, nbytes):
    key = (tag, int(rows))
    if key not in _RAW:
        dev = torch.cuda.current_device()
        _RAW[key] = torch.zeros(nbytes // 8, dtype=torch.int64, device=dev)
    return _RAW[key].data_ptr()


def _anchor_site(frame):
    name = frame.f_code.co_filename
    rel = name.split("specialized_decode/", 1)[-1]
    if rel == name:
        rel = name.split("vllm/", 1)[-1]
    text = f"{rel}:{frame.f_lineno}:{frame.f_code.co_name}"
    site = zlib.crc32(text.encode()) & 0x7FFFFF
    _SITES[site] = text
    return site


def _anchor_code(site):
    base = raw_buffer("anchor", 0, 16 << ANCHOR_LOG2)
    mask = (1 << ANCHOR_LOG2) - 1
    return (
        "{ .reg .pred p; .reg .u32 a0, a1, a2, a3, a4, a5, s; .reg .u64 t, g, ad, k; "
        "mov.u32 a0, %tid.x; mov.u32 a1, %tid.y; mov.u32 a2, %tid.z; "
        "mov.u32 a3, %ctaid.x; mov.u32 a4, %ctaid.y; mov.u32 a5, %ctaid.z; "
        "or.b32 s, a0, a1; or.b32 s, s, a2; or.b32 s, s, a3; or.b32 s, s, a4; "
        "or.b32 s, s, a5; setp.eq.u32 p, s, 0; "
        "mov.u64 t, %globaltimer; mov.u64 g, %gridid; "
        f"and.b64 ad, g, {mask}; shl.b64 ad, ad, 4; add.u64 ad, ad, {base}; "
        f"shl.b64 k, g, 24; shr.b64 k, k, 24; or.b64 k, k, {site << 40}; "
        "@p st.global.v2.u64 [ad], {t, k}; }"
    )


@asm()
def _anchor(site: int):
    return _anchor_code(site), "~{memory}", ()


_ORIG_WAIT = cute.arch.griddepcontrol_wait
_ANCHORED = ("specialized_decode", "dcp_topk_merge")


def _wait_anchored(*args, **kwargs):
    """``griddepcontrol_wait`` + the anchor stamp for kernels traced from the
    specialized-decode sources or the #63 merge (whose cubins compile per process)."""
    result = _ORIG_WAIT(*args, **kwargs)
    frame = sys._getframe(1)
    if any(k in frame.f_code.co_filename for k in _ANCHORED):
        _anchor(_anchor_site(frame))
    return result


cute.arch.griddepcontrol_wait = _wait_anchored


# ------------------------------------------------------------------ clock sync
_CLK_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ long long ld_acq(const long long* p) {
  long long v; asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(long long* p, long long v) {
  asm volatile("st.release.sys.global.s64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
// own/peer buffers (int64): [0] ping seq (rank 0 -> r), [64 + r] pong seq, [128 + r] pong time
__global__ void k31_dsadiag_pp(const long long* peers, long long* own, long long* out, int rank, int W, int K,
                               long long base, long long timeout) {
  if (threadIdx.x != 0) return;
  if (rank == 0) {
    for (int r = 1; r < W; ++r) {
      long long* pr = reinterpret_cast<long long*>(peers[r]);
      for (int k = 0; k < K; ++k) {
        const long long seq = base + k + 1;
        long long* o = out + ((long long)r * K + k) * 3;
        const long long t0 = gt();
        st_rel(pr, seq);
        bool ok = true;
        while (ld_acq(own + 64 + r) != seq) { if (gt() - t0 > timeout) { ok = false; break; } }
        const long long t2 = gt();
        if (!ok) { o[0] = -1; continue; }
        o[0] = t0; o[1] = ld_acq(own + 128 + r); o[2] = t2;
      }
    }
  } else {
    long long* p0 = reinterpret_cast<long long*>(peers[0]);
    const long long ts = gt();
    for (int k = 0; k < K; ++k) {
      const long long seq = base + k + 1;
      while (ld_acq(own) < seq) { if (gt() - ts > timeout) return; }
      const long long t1 = gt();
      st_rel(p0 + 128 + rank, t1);
      st_rel(p0 + 64 + rank, seq);
      out[((long long)rank * K + k) * 3 + 1] = t1;
    }
  }
}
void pingpong(torch::Tensor peers, torch::Tensor own, torch::Tensor out, int64_t rank, int64_t W, int64_t K,
              int64_t base, int64_t timeout) {
  auto s = c10::cuda::getCurrentCUDAStream();
  k31_dsadiag_pp<<<1, 32, 0, s>>>(reinterpret_cast<const long long*>(peers.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(own.data_ptr<int64_t>()), reinterpret_cast<long long*>(out.data_ptr<int64_t>()),
      (int)rank, (int)W, (int)K, base, timeout);
}
"""
_CLK: dict = {}
CLK_ROUNDS = 32


def _clk_state():
    if "ext" in _CLK:
        return _CLK
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from torch.utils.cpp_extension import load_inline

    try:
        from vllm.distributed.parallel_state import get_dcp_group, get_world_group

        group = get_dcp_group()
        if group.world_size != get_world_group().world_size:
            group = get_world_group()  # (all ranks of the step)
        pg = group.device_group
    except Exception:  # (no vLLM groups: a standalone torch.distributed job)
        pg = dist.group.WORLD
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    build = f"/tmp/dsadiag-clk-{os.getpid()}"
    os.makedirs(build, exist_ok=True)
    _CLK["ext"] = load_inline(
        name="k31_dsadiag_clk",
        cpp_sources=["void pingpong(torch::Tensor peers, torch::Tensor own, torch::Tensor out, int64_t rank, "
                     "int64_t W, int64_t K, int64_t base, int64_t timeout);"],
        cuda_sources=[_CLK_SRC],
        functions=["pingpong"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        build_directory=build,
    )
    dev = torch.cuda.current_device()
    box = symm_mem.empty(256, dtype=torch.int64, device=dev)
    box.zero_()
    h = symm_mem.rendezvous(box, pg)
    _CLK.update(
        box=box,
        peers=torch.tensor([int(x) for x in h.buffer_ptrs], dtype=torch.int64, device=dev),
        rank=dist.get_rank(pg),
        W=dist.get_world_size(pg),
        pg=pg,
        calls=0,
        global_rank=dist.get_rank(),
    )
    torch.cuda.synchronize()
    dist.barrier(pg)
    return _CLK


def clocksync():
    """One ping-pong pass: int64 [W, K, 3] (t0 rank-0 send, t1 rank-r receive, t2 rank-0
    receive; rank 0 holds t0/t2, rank r its own t1), the host wall time and the group rank."""
    import torch.distributed as dist

    s = _clk_state()
    out = torch.zeros(s["W"], CLK_ROUNDS, 3, dtype=torch.int64, device=s["box"].device)
    torch.cuda.synchronize()
    dist.barrier(s["pg"])
    base = 1000 * (s["calls"] + 1)
    s["calls"] += 1
    s["ext"].pingpong(s["peers"], s["box"], out, s["rank"], s["W"], CLK_ROUNDS, base, 2_000_000_000)
    torch.cuda.synchronize()
    return out.cpu()


def presync():
    """Clock sync BEFORE a bench case (forward bench, all ranks): with the dump's passes
    the stamps lie between two syncs, so the per-rank offset is interpolated, never
    extrapolated (a drift estimate from passes 0.5 s apart is too noisy to extrapolate)."""
    try:
        _CLK["pre"] = clocksync()
    except Exception as e:  # (diagnostic: never fail the bench)
        _CLK.pop("pre", None)
        print(f"dsadiag presync failed: {e!r}", flush=True)


# ------------------------------------------------------------------ dump
def _run_dir():
    out = os.environ.get(
        "OA_DIAG_DIR", "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/DSTAMP/diag"
    )
    run = os.environ.get("SLURM_JOB_ID", "j") + "." + os.environ.get("SLURM_STEP_ID", "s")
    os.makedirs(f"{out}/{run}", exist_ok=True)
    return f"{out}/{run}"


def dump(case):
    """After each bench case (all ranks, in lockstep): clock sync x2 (0.5 s apart), then
    the anchor (STAMP raw format) and every ring; rings are zeroed afterwards."""
    d = _run_dir()
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    torch.cuda.synchronize()
    data = {}
    try:
        c0 = clocksync()
        time.sleep(0.5)
        c1 = clocksync()
        data["clk:0"], data["clk:1"] = c0, c1
        if "pre" in _CLK:
            data["clk:pre"] = _CLK.pop("pre")
        data["clk:rank"] = torch.tensor([_CLK["rank"], _CLK["W"], _CLK["global_rank"]], dtype=torch.int64)
    except Exception as e:  # (diagnostic: never fail the bench)
        print(f"dsadiag clocksync failed: {e!r}", flush=True)
    for (tag, ctas), (ring, ctr) in _RINGS.items():
        data[f"ring:{tag}:{ctas}"] = (ring.cpu(), ctr.cpu())
    for kind, (buf, ring, maxc, slots) in _DEV.items():
        n = ring * maxc * slots
        data[f"dev:{kind}"] = (buf[:n].view(ring, maxc, slots).cpu(), buf[n:].view(torch.int32)[:maxc].cpu())
    torch.save(data, f"{d}/{case}.rank{rank}.dsa.pt")
    raw = {}
    for k, b in _RAW.items():  # gridid-keyed: only when changed since the last dump
        sig = int(b.sum().item())
        if _SIG.get(k) == sig:
            continue
        _SIG[k] = sig
        raw[f"{k[0]}:{k[1]}"] = b.cpu()
    torch.save(raw, f"{d}/{case}.rank{rank}.raw.pt")
    with open(f"{d}/{case}.rank{rank}.raw.pt.sites.json", "w") as f:
        json.dump({str(k): v for k, v in _SITES.items()}, f)
    for ring, ctr in _RINGS.values():
        ring.zero_()
        ctr.zero_()
    for buf, *_ in _DEV.values():
        buf.zero_()
    torch.cuda.synchronize()
