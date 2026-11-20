"""K3.1 SOL lane 'comm': NVLink collective / push primitives at the optables comm-row message sizes.

Launch: srun -N<nodes> --ntasks-per-node=4 python3 commbench.py <rdv file> <cases.json> <out.jsonl>
Every rank belongs to one group of N consecutive ranks; all groups of a size run concurrently (as DP x TP deployments do).
Kinds / variants (bytes = per-op message parameter, see cases):
  ar  (M = full tensor bytes):   nvls_tri (multimem.ld_reduce acc f32 + multimem.st on own 1/N, barrier before/after =
                                 the gemm_rs_ar epilogue algorithm), torch_multimem, torch_twoshot, torch_oneshot, nccl
  rs  (M = full tensor bytes):   nvls_tri (ld_reduce own 1/N -> local), push_sum (push chunks + barrier + local sum), nccl
  ag  (bytes = per-rank shard):  push (the SymmSpAllGather kernel: persistent 16B stores to every peer + symm barrier),
                                 mc (multimem.st of the shard to the multicast address + barrier), ce (copy engine), nccl
  a2a (bytes = per-peer chunk):  push (SM stores), pull (barrier + SM remote loads), ce (copy engine), nccl
  bar : symm barrier only
hops > 1 adds (hops - 1) symm barriers (dependent hand-offs) to push/pull/ce/mc variants.
Times: CUDA graph of `iters` calls, median of 5 replays / iters, max over all ranks.
"""
import datetime
import faulthandler
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"])
local = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
dist.init_process_group("nccl", init_method="file://" + sys.argv[1], rank=rank, world_size=world,
                        timeout=datetime.timedelta(minutes=20), device_id=dev)
CASES = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
ONLY_N = [int(x) for x in os.environ.get("ONLY_N", "").split(",") if x]
SKIP_VARIANTS = set(x for x in os.environ.get("SKIP_VARIANTS", "").split(",") if x)
BARRIER = os.environ.get("BARRIER", "flag")
SMS = torch.cuda.get_device_properties(dev).multi_processor_count
T0 = time.time()


def log(d):
    if rank == 0:
        d["t_wall_s"] = round(time.time() - T0, 1)
        line = json.dumps(d)
        print(line, flush=True)
        with open(OUT, "a") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------------------------- kernels
@triton.jit
def push_kernel(src, dst_addrs, dst_off, src_stride, numel, rank, WORLD: tl.constexpr, BLOCK: tl.constexpr,
                SAME_SRC: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        if SAME_SRC:
            x = tl.load(src + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            if not SAME_SRC:
                x = tl.load(src + peer.to(tl.int64) * src_stride + offs, mask=mask)
            dst = tl.load(dst_addrs + peer).to(tl.pointer_type(src.dtype.element_ty))
            tl.store(dst + dst_off + offs, x, mask=mask)


@triton.jit
def push2d_kernel(src, dst_addrs, dst_off, src_stride, numel, rank, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """One program column per peer (grid [tiles, WORLD]): the peer address is loaded once per program."""
    peer = (rank + 1 + tl.program_id(1)) % WORLD
    dst = tl.load(dst_addrs + peer).to(tl.pointer_type(src.dtype.element_ty))
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(src + peer.to(tl.int64) * src_stride + offs, mask=mask)
        tl.store(dst + dst_off + offs, x, mask=mask)


SENT = tl.constexpr(-2147450880)  # int32 0x80008000: two bf16 -0.0 (the Lamport empty marker)


@triton.jit
def lamport_kernel(src, dst_addrs, my_buf, cnt_ptr, out, chunk, rank, slot_elems, src_stride,
                   WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """One-kernel Lamport exchange (int32 words): program (t, j) pushes its tiles of the chunk for peer j into that
    peer's slot (epoch % 3), then spins on its own slot until peer j's words are non-sentinel, copies them out and
    re-arms the slot of epoch + 2 (last used at epoch - 1, next written at epoch + 2 only after this rank's next push)."""
    e = tl.load(cnt_ptr)
    slot = e % 3
    peer = (rank + 1 + tl.program_id(1)) % WORLD
    dst = tl.load(dst_addrs + peer).to(tl.pointer_type(tl.int32))
    base_out = slot.to(tl.int64) * slot_elems + (slot * 0 + rank).to(tl.int64) * chunk
    for tile in range(tl.program_id(0), tl.cdiv(chunk, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < chunk
        x = tl.load(src + peer.to(tl.int64) * src_stride + offs, mask=mask)
        tl.store(dst + base_out + offs, x, mask=mask)
    base_in = slot.to(tl.int64) * slot_elems + peer.to(tl.int64) * chunk
    base_clr = ((e + 2) % 3).to(tl.int64) * slot_elems + peer.to(tl.int64) * chunk
    for tile in range(tl.program_id(0), tl.cdiv(chunk, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < chunk
        v = tl.load(my_buf + base_in + offs, mask=mask, other=1, volatile=True)
        while tl.min(tl.where(mask, (v != SENT).to(tl.int32), 1), axis=0) == 0:
            v = tl.load(my_buf + base_in + offs, mask=mask, other=1, volatile=True)
        tl.store(out + peer.to(tl.int64) * chunk + offs, v, mask=mask)
        tl.store(my_buf + base_clr + offs, tl.full([BLOCK], SENT, tl.int32), mask=mask)


@triton.jit
def inc_kernel(cnt_ptr):
    tl.store(cnt_ptr, tl.load(cnt_ptr) + 1)


LAMPORT_MAX = 4 << 20  # bytes per peer chunk


@triton.jit
def pull_kernel(src_addrs, dst, src_off, numel, rank, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    peer = (rank + 1 + tl.program_id(1)) % WORLD
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        if True:
            s = tl.load(src_addrs + peer).to(tl.pointer_type(dst.dtype.element_ty))
            x = tl.load(s + src_off + offs, mask=mask)
            tl.store(dst + peer.to(tl.int64) * numel + offs, x, mask=mask)


@triton.jit
def mm_reduce_kernel(mc_base, out, start16, n16, BLOCK: tl.constexpr, STORE_MC: tl.constexpr):
    """multimem.ld_reduce (acc f32) of n16 16-byte vectors from start16; STORE_MC: multimem.st back (all-reduce),
    else store to local out (int32 view, reduce-scatter). n16 is a multiple of BLOCK (no masks: asm runs on all lanes)."""
    for tile in range(tl.program_id(0), n16 // BLOCK, tl.num_programs(0)):
        i = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        addr = mc_base + (start16 + i) * 16
        a, b, c, d = tl.inline_asm_elementwise(
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l", [addr], dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=False, pack=1)
        if STORE_MC:
            tl.inline_asm_elementwise(
                "multimem.st.relaxed.sys.global.v4.bf16x2 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0;",
                "=r,l,r,r,r,r", [addr, a, b, c, d], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.store(out + i * 4 + 0, a)
            tl.store(out + i * 4 + 1, b)
            tl.store(out + i * 4 + 2, c)
            tl.store(out + i * 4 + 3, d)


@triton.jit
def mm_store_kernel(src, mc_base, start16, n16, BLOCK: tl.constexpr):
    """multicast all-gather: store local src (int32 view, n16 x 16 B) to the multicast address at start16."""
    for tile in range(tl.program_id(0), n16 // BLOCK, tl.num_programs(0)):
        i = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(src + i * 4 + 0)
        b = tl.load(src + i * 4 + 1)
        c = tl.load(src + i * 4 + 2)
        d = tl.load(src + i * 4 + 3)
        addr = mc_base + (start16 + i) * 16
        tl.inline_asm_elementwise(
            "multimem.st.relaxed.sys.global.v4.bf16x2 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0;",
            "=r,l,r,r,r,r", [addr, a, b, c, d], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def flag_barrier_kernel(cnt_ptr, flag_addrs, my_flags, rank, WORLD: tl.constexpr, WP2: tl.constexpr):
    """Lamport-style flag barrier: write epoch into my slot on every peer (release, sys), spin until every slot of mine
    holds >= epoch (acquire, sys). One CTA, one lane per peer. Epoch = per-rank call counter (graph-replay safe)."""
    e = tl.load(cnt_ptr) + 1
    tl.store(cnt_ptr, e)
    p = tl.arange(0, WP2)
    m = p < WORLD
    a = tl.load(flag_addrs + p, mask=m, other=0)
    dst = a.to(tl.pointer_type(tl.int32)) + rank
    tl.atomic_xchg(dst, tl.zeros_like(p) + e, mask=m, sem="release", scope="sys")
    v = tl.atomic_add(my_flags + p, 0, mask=m, sem="acquire", scope="sys")
    done = tl.min(tl.where(m, (v >= e).to(tl.int32), 1), axis=0)
    while done == 0:
        v = tl.atomic_add(my_flags + p, 0, mask=m, sem="acquire", scope="sys")
        done = tl.min(tl.where(m, (v >= e).to(tl.int32), 1), axis=0)


MMB = 256          # lanes per multimem tile (4 KiB)
PUSH_BLOCK = 8192  # bf16 elements per push tile (16 KiB), as SymmSpAllGather
PULL_BLOCK = 4096


# ----------------------------------------------------------------------------------------------- group state
class G:
    def __init__(self, N, max_bytes):
        self.N = N
        groups = [list(range(g * N, (g + 1) * N)) for g in range(world // N)]
        mine = None
        for gr in groups:
            pg = dist.new_group(gr)
            if rank in gr:
                mine, self.pg = gr, pg
        self.me = mine.index(rank)
        self.gname = self.pg.group_name
        quant = N * MMB * 16
        self.half = ((max_bytes + quant - 1) // quant) * quant  # bytes per half
        self.buf = symm_mem.empty(2 * self.half // 2, dtype=torch.bfloat16, device=dev)
        self.hdl = symm_mem.rendezvous(self.buf, self.gname)
        self.A = self.buf[: self.half // 2]          # receive / in-place tensor
        self.B = self.buf[self.half // 2:]           # symmetric send buffer (pull)
        self.src = torch.zeros(self.half // 2, dtype=torch.bfloat16, device=dev)  # local send buffer
        self.out = torch.zeros(self.half // 2, dtype=torch.bfloat16, device=dev)
        self.buf.zero_()
        bases = [self.hdl.get_buffer(p, [2 * self.half // 2], torch.bfloat16, 0).data_ptr() for p in range(N)]
        self.addrA = torch.tensor(bases, dtype=torch.int64, device=dev)
        self.addrB = torch.tensor([b + self.half for b in bases], dtype=torch.int64, device=dev)
        try:
            self.mc = int(self.hdl.multicast_ptr)
        except Exception:
            self.mc = 0
        self.flags = symm_mem.empty(64, dtype=torch.int32, device=dev)
        self.fh = symm_mem.rendezvous(self.flags, self.gname)
        self.flags.zero_()
        self.flag_addrs = torch.tensor([self.fh.get_buffer(p, [64], torch.int32, 0).data_ptr() for p in range(N)],
                                       dtype=torch.int64, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.lslot = N * LAMPORT_MAX // 4  # int32 words per slot
        self.L = symm_mem.empty(3 * self.lslot, dtype=torch.int32, device=dev)
        self.lh = symm_mem.rendezvous(self.L, self.gname)
        self.L.fill_(-2147450880)
        self.laddr = torch.tensor([self.lh.get_buffer(p, [3 * self.lslot], torch.int32, 0).data_ptr() for p in range(N)],
                                  dtype=torch.int64, device=dev)
        self.lcnt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.lout = torch.zeros(N * LAMPORT_MAX // 4, dtype=torch.int32, device=dev)
        self.lsrc = torch.ones(N * LAMPORT_MAX // 4, dtype=torch.int32, device=dev)
        self.wp2 = triton.next_power_of_2(max(N, 2))
        self.peerA = [self.hdl.get_buffer(p, [self.half // 2], torch.bfloat16, 0) for p in range(N)]
        torch.cuda.synchronize()
        self.hdl.barrier()
        torch.cuda.synchronize()

    def lamport(self, chunk_words, same_src, extra):
        grid_t = max(1, min(SMS // self.N, triton.cdiv(chunk_words, 2048)))
        lamport_kernel[(grid_t, self.N)](self.lsrc, self.laddr, self.L, self.lcnt, self.lout, chunk_words, self.me,
                                         self.lslot, 0 if same_src else chunk_words, WORLD=self.N, BLOCK=2048,
                                         num_warps=4)
        inc_kernel[(1,)](self.lcnt, num_warps=1)
        for _ in range(extra):  # dependent extra hand-offs: one more Lamport round each (1 word per peer)
            lamport_kernel[(1, self.N)](self.lsrc, self.laddr, self.L, self.lcnt, self.lout, 4, self.me,
                                        self.lslot, 0, WORLD=self.N, BLOCK=2048, num_warps=4)
            inc_kernel[(1,)](self.lcnt, num_warps=1)

    def bar(self, n=1):
        for _ in range(n):
            if BARRIER == "torch":
                self.hdl.barrier()
            else:
                flag_barrier_kernel[(1,)](self.cnt, self.flag_addrs, self.flags, self.me, WORLD=self.N, WP2=self.wp2,
                                          num_warps=1)


def rnd(b, q):
    return max(q, ((int(b) + q - 1) // q) * q)


def make_ops(g, kind, nbytes, hops):
    """Returns (actual_bytes, {variant: fn})."""
    N, me = g.N, g.me
    extra = max(hops - 1, 0)
    ops = {}
    if kind == "bar":
        return 0, {"barrier": lambda: g.bar(1), "torch_barrier": lambda: g.hdl.barrier()}
    if kind in ("ar", "rs"):
        M = rnd(nbytes, N * MMB * 16)
        el = M // 2
        t = g.A[:el]
        per16 = M // 16 // N
        if g.mc:
            ops["nvls_tri"] = (lambda: (g.bar(), mm_reduce_kernel[(min(SMS * 2, max(1, per16 // MMB)),)](
                g.mc, g.out.view(torch.int32), me * per16, per16, BLOCK=MMB, STORE_MC=(kind == "ar"), num_warps=4), g.bar()))
        if kind == "ar" and M <= LAMPORT_MAX:
            def lam_one():  # one-shot: Lamport all-gather of the full tensor + local FP32-accumulated sum
                g.lamport(M // 4, True, 0)
                g.lout[:N * (M // 4)].view(torch.bfloat16).view(N, -1).sum(0, dtype=torch.float32)
            ops["lamport_oneshot"] = lam_one
        if M // N <= LAMPORT_MAX:
            w = M // N // 4

            def lam_two():  # two-shot: Lamport scatter of 1/N chunks + local sum (+ Lamport all-gather of the sums)
                g.lamport(w, False, 0)
                g.lout[:N * w].view(torch.bfloat16).view(N, -1).sum(0, dtype=torch.float32)
                if kind == "ar":
                    g.lamport(w, True, 0)
            ops["lamport_twoshot" if kind == "ar" else "lamport_sum"] = lam_two
        if kind == "ar":
            ops["torch_multimem"] = lambda: torch.ops.symm_mem.multimem_all_reduce_(t, "sum", g.gname)
            ops["torch_twoshot"] = lambda: torch.ops.symm_mem.two_shot_all_reduce_(t, "sum", g.gname)
            if M <= (16 << 20):
                ops["torch_oneshot"] = lambda: torch.ops.symm_mem.one_shot_all_reduce(t, "sum", g.gname)
            ops["nccl"] = lambda: dist.all_reduce(t, group=g.pg)
        else:
            chunk = el // N
            src = g.src[:el]
            outv = g.out[:chunk]
            grid = min(SMS, max(1, triton.cdiv(chunk, PUSH_BLOCK)))

            def push_sum():
                push_kernel[(grid,)](src, g.addrA, me * chunk, chunk, chunk, me, WORLD=N, BLOCK=PUSH_BLOCK,
                                     SAME_SRC=False, num_warps=4)
                g.bar()
                torch.sum(g.A[:el].view(N, chunk), dim=0, out=outv)
            ops["push_sum"] = push_sum
            ops["dbg_push_only"] = lambda: (push_kernel[(grid,)](src, g.addrA, me * chunk, chunk, chunk, me, WORLD=N,
                                                                  BLOCK=PUSH_BLOCK, SAME_SRC=False, num_warps=4), g.bar())
            ops["dbg_sum_only"] = lambda: torch.sum(g.A[:el].view(N, chunk), dim=0, out=outv)
            ops["nccl"] = lambda: dist.reduce_scatter_tensor(outv, src, group=g.pg)
        return M, ops
    if kind == "ag":
        S = rnd(nbytes, MMB * 16)
        el = S // 2
        src = g.src[:el]
        grid = min(SMS, max(1, triton.cdiv(el, PUSH_BLOCK)))
        ops["push"] = lambda: (push_kernel[(grid,)](src, g.addrA, me * el, 0, el, me, WORLD=N, BLOCK=PUSH_BLOCK,
                                                    SAME_SRC=True, num_warps=4), g.bar(1 + extra))
        if S <= LAMPORT_MAX:
            ops["lamport"] = lambda: g.lamport(S // 4, True, extra)
        g2 = max(1, min(SMS // N, triton.cdiv(el, PUSH_BLOCK)))
        ops["push2d"] = lambda: (push2d_kernel[(g2, N)](src, g.addrA, me * el, 0, el, me, WORLD=N, BLOCK=PUSH_BLOCK,
                                                        num_warps=4), g.bar(1 + extra))
        if g.mc:
            n16 = S // 16
            src32 = src.view(torch.int32)
            ops["mc"] = lambda: (mm_store_kernel[(min(SMS * 2, max(1, n16 // MMB)),)](
                src32, g.mc, me * n16, n16, BLOCK=MMB, num_warps=4), g.bar(1 + extra))

        def ce():
            for p in range(N):
                if p != me:
                    g.peerA[p][me * el:(me + 1) * el].copy_(src, non_blocking=True)
            g.bar(1 + extra)
        ops["ce"] = ce
        ops["nccl"] = lambda: dist.all_gather_into_tensor(g.A[:el * N], src, group=g.pg)
        return S, ops
    if kind in ("a2a", "a2a_ce"):
        S = rnd(nbytes, 64)
        el = S // 2
        src = g.src[:el * N]
        grid = min(SMS, max(1, triton.cdiv(el, PUSH_BLOCK)))
        ops["push"] = lambda: (push_kernel[(grid,)](src, g.addrA, me * el, el, el, me, WORLD=N, BLOCK=PUSH_BLOCK,
                                                    SAME_SRC=False, num_warps=4), g.bar(1 + extra))
        if S <= LAMPORT_MAX:
            ops["lamport"] = lambda: g.lamport(S // 4, False, extra)
        g2 = max(1, min(SMS // N, triton.cdiv(el, PUSH_BLOCK)))
        ops["push2d"] = lambda: (push2d_kernel[(g2, N)](src, g.addrA, me * el, el, el, me, WORLD=N, BLOCK=PUSH_BLOCK,
                                                        num_warps=4), g.bar(1 + extra))
        pgrid = max(1, min(SMS * 4 // N, triton.cdiv(el, PULL_BLOCK)))
        ops["pull"] = lambda: (g.bar(1 + extra), pull_kernel[(pgrid, N)](g.addrB, g.out, me * el, el, me, WORLD=N,
                                                                     BLOCK=PULL_BLOCK, num_warps=8))

        def ce():
            for i in range(1, N):
                p = (me + i) % N
                g.peerA[p][me * el:(me + 1) * el].copy_(src[p * el:(p + 1) * el], non_blocking=True)
            g.bar(1 + extra)
        ops["ce"] = ce
        if kind == "a2a":
            ops["nccl"] = lambda: dist.all_to_all_single(g.A[:el * N], src, group=g.pg)
        return S, ops
    raise ValueError(kind)


def traffic(kind, N, b):
    if kind in ("ar", "rs"):
        return b
    return b * max(N - 1, 1)


def timeit(fn, est_us, allow_graph=True):
    iters = int(max(3, min(50, 3000.0 / max(est_us, 1.0))))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = allow_graph
    try:
        if not allow_graph:
            raise RuntimeError("eager")
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(iters):
                fn()
        run = gr.replay
        run(); torch.cuda.synchronize()
    except Exception as ex:
        graph = False
        torch.cuda.synchronize()

        def run():
            for _ in range(iters):
                fn()
    ts = []
    for _ in range(5):
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    med = torch.tensor([statistics.median(ts), min(ts)], device=dev, dtype=torch.float64)
    dist.all_reduce(med, op=dist.ReduceOp.MAX)
    if graph:
        del gr
    return med[0].item(), med[1].item(), iters, graph


def check(g):
    """Correctness of the hand-written variants at a small size (values = group rank + 1)."""
    N, me = g.N, g.me
    res = {}
    el = N * MMB * 8
    try:
        g.A[:el].fill_(me + 1); torch.cuda.synchronize(); g.bar(); torch.cuda.synchronize()
        if g.mc:
            per16 = el * 2 // 16 // N
            g.bar(); mm_reduce_kernel[(4,)](g.mc, g.out.view(torch.int32), me * per16, per16, BLOCK=MMB, STORE_MC=True, num_warps=4)
            g.bar(); torch.cuda.synchronize()
            res["nvls_ar_ok"] = bool((g.A[:el] == N * (N + 1) / 2).all().item())
            g.A[:el].fill_(me + 1); torch.cuda.synchronize(); g.bar()
            mm_reduce_kernel[(4,)](g.mc, g.out.view(torch.int32), me * per16, per16, BLOCK=MMB, STORE_MC=False,
                                   num_warps=4)
            g.bar(); torch.cuda.synchronize()
            res["nvls_rs_ok"] = bool((g.out[:el // N] == N * (N + 1) / 2).all().item())
        src = g.src[:el // N]; src.fill_(me + 1)
        push_kernel[(4,)](src, g.addrA, me * (el // N), 0, el // N, me, WORLD=N, BLOCK=PUSH_BLOCK, SAME_SRC=True,
                          num_warps=4)
        g.bar(); torch.cuda.synchronize()
        exp = torch.arange(1, N + 1, device=dev, dtype=torch.bfloat16).repeat_interleave(el // N)
        res["push_ag_ok"] = bool((g.A[:el] == exp).all().item())
        if g.mc:
            g.A[:el].zero_(); torch.cuda.synchronize(); g.bar()
            n16 = el // N * 2 // 16
            mm_store_kernel[(4,)](src.view(torch.int32), g.mc, me * n16, n16, BLOCK=MMB, num_warps=4)
            g.bar(); torch.cuda.synchronize()
            res["mc_ag_ok"] = bool((g.A[:el] == exp).all().item())
        g.lsrc.fill_(me + 1); torch.cuda.synchronize(); g.bar(); torch.cuda.synchronize()
        for _ in range(4):
            g.lamport(1000, True, 0)
        torch.cuda.synchronize()
        expl = torch.arange(1, N + 1, device=dev, dtype=torch.int32).repeat_interleave(1000)
        res["lamport_ok"] = bool((g.lout[:N * 1000] == expl).all().item())
        g.lsrc.fill_(1)
    except Exception as ex:
        res["check_err"] = repr(ex)[-600:]
    ok = torch.tensor([0 if any(v is False for v in res.values()) else 1], device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    res["all_ranks_ok"] = bool(ok.item())
    g.buf.zero_(); torch.cuda.synchronize(); g.bar(); torch.cuda.synchronize()
    return res


log(dict(meta=True, barrier=BARRIER, torch=torch.__version__, triton=triton.__version__, nccl=".".join(map(str, torch.cuda.nccl.version())),
         world=world, sms=SMS, device=torch.cuda.get_device_name(dev), host=os.uname().nodename,
         symm_ops=sorted(x for x in dir(torch.ops.symm_mem) if not x.startswith("_"))[:60],
         env={k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k.startswith("TORCH_SYMM")}))

Ns = sorted(set(c["N"] for c in CASES))
if ONLY_N:
    Ns = [n for n in Ns if n in ONLY_N]
for N in Ns:
    if N > world:
        continue
    cs = [c for c in CASES if c["N"] == N]
    cs.sort(key=lambda c: (c["kind"], c["bytes"]))
    maxb = 0
    for c in cs:
        k, b = c["kind"], c["bytes"]
        need = b if k in ("ar", "rs") else (b * N if k != "bar" else 0)
        maxb = max(maxb, need)
    maxb = max(maxb, 1 << 20)
    try:
        g = G(N, maxb)
    except Exception as ex:
        log(dict(N=N, group_err=repr(ex)[:400]))
        continue
    log(dict(N=N, group=True, half_bytes=g.half, multicast=bool(g.mc), check=check(g)))
    for c in cs:
        # re-arm the Lamport slots for this case's layout (stale words of an older layout would end a poll early)
        g.L.fill_(-2147450880); torch.cuda.synchronize(); g.bar(); torch.cuda.synchronize(); dist.barrier()
        try:
            actual, ops = make_ops(g, c["kind"], c["bytes"], c.get("hops", 1))
        except Exception as ex:
            log(dict(case=c, err=repr(ex)[:300]))
            continue
        est = 4.0 + traffic(c["kind"], N, actual) / 0.8e6
        row = dict(kind=c["kind"], N=N, bytes=c["bytes"], bytes_actual=actual, hops=c.get("hops", 1),
                   grid=bool(c.get("grid")), us={}, min_us={}, err={})
        for v, fn in ops.items():
            if v in SKIP_VARIANTS:
                continue
            faulthandler.dump_traceback_later(150, exit=False)
            sys.stderr.write("rank %d case %s N%d %d %s\n" % (rank, c["kind"], N, c["bytes"], v)) if rank == 0 else None
            # every rank must agree a variant works before timing it (a local exception would desync the collectives)
            okt = torch.ones(1, device=dev)
            try:
                fn(); torch.cuda.synchronize()
            except Exception as ex:
                okt.zero_(); row["err"][v] = repr(ex)[-400:]
            dist.all_reduce(okt, op=dist.ReduceOp.MIN)
            if okt.item() < 1:
                row["err"].setdefault(v, "failed on another rank")
                faulthandler.cancel_dump_traceback_later()
                continue
            try:
                med, mn, iters, graph = timeit(fn, est, allow_graph=(v != "nccl"))
                row["us"][v] = round(med, 3)
                row["min_us"][v] = round(mn, 3)
                row.setdefault("graph", {})[v] = graph
            except Exception as ex:
                row["err"][v] = repr(ex)[:200]
                sys.stderr.write("rank %d timeit error %s\n" % (rank, repr(ex)[:300]))
            faulthandler.cancel_dump_traceback_later()
        if row["us"]:
            best = min(row["us"], key=row["us"].get)
            row["best"] = best
            row["best_us"] = row["us"][best]
        log(row)
    del g
    torch.cuda.synchronize()
    dist.barrier()
log(dict(done=True))
dist.destroy_process_group()
