# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_k31_dcp_exchange.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Multi-GPU correctness of the K3.1 DCP exchanges (dsa_kernels/dcp_exchange).

Run under torchrun (one process per GPU; W = world size, 4 on one node, 16 on
4 nodes):

    torchrun --nproc-per-node 4 tests/models/kimi_k3/nvidia/specialized_decode/sm_107/test_k3_sm107_dcp_exchange.py

``K31_DCPX_GROUP_SIZE=g`` (g divides the world size) splits the world into
consecutive DCP groups of g ranks, as in DCPg x DP(W/g) (for example DCP4 x DP4
on 16 GPUs). Every group runs its own exchange on its own mailboxes at the same
time, with group-specific data, and is checked against its own reference.

Top-k: every rank holds its interleaved shard of the same global bf16 logits.
The union of the ranks' selections must EQUAL the reference set: the K
largest 16-bit keys, ties broken by ascending global token id (the stock
stable merge's rule). Each rank's output must be its tokens above the K-th
key in position order, then its selected ties in position order, padded with
-1; it must be bitwise identical across repeated calls and between the 1- and
2-round modes. Every case runs at interleave 896 and 1. Where the stock merge
(fp32 upcast + persistent_topk + pack + all_gather + stable select) runs, its
global set is compared too (it can differ from the reference only when a
rank's local top-K cut itself falls inside the tie group).
O exchange: the owner's merged heads must equal the FP32 LSE merge of all
ranks' partials (bf16 rounding tolerance).
Spin limits: with the serving limit (600 s) or none (0), a peer that launches 7 s
after the others (past the 5 s test default) must not trap.
"""

import os
import sys

import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.dsa_kernels.dcp_exchange.ops import (
    DcpExchange,
    local_len_of,
    local_to_global,
)
import pytest  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

K = 2048
INTERLEAVE = int(os.environ.get("K31_DCPX_INTERLEAVE", "896"))
INTERLEAVE_CUR = [INTERLEAVE]  # interleave of the case being generated
WORLD = [1]
GROUP = [None]  # the DCP group under test (None: the whole world)
SEED_OFF = [0]  # per-group data offset (DCPg x DP: groups see different data)
W_LATE_SKEW_RANK = 1  # the O exchange's late peer (rank 0 is late for the top-k)


def keys_of(x_bf16: torch.Tensor) -> torch.Tensor:
    u = x_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    nan = (u & 0x7FFF) > 0x7F80
    k = torch.where((u & 0x8000) != 0, (~u) & 0xFFFF, u | 0x8000)
    return torch.where(nan, torch.zeros_like(k), k)


def make_logits(case: str, T: int, G: torch.Tensor, gen: torch.Generator, device):
    Gmax = int(G.max())
    x = torch.randn(T, Gmax, generator=gen, device=device)
    if case == "normal":
        pass
    elif case == "relu":  # most scores exactly zero
        x = torch.relu(x - 1.2) * 3
    elif case == "hot_recent":  # last chunk(s) score much higher
        pos = torch.arange(Gmax, device=device)
        hot = (pos[None, :] >= (G[:, None] - 3 * INTERLEAVE)) & (pos[None, :] < G[:, None])
        x = x + hot * 6.0
    elif case == "wide":  # huge dynamic range
        x = torch.exp(x * 12)
    elif case == "sink":  # a few huge tokens, the rest tiny/zero
        x = torch.relu(x) * 1e-6
        x[:, :4] = 1e6
    elif case == "ties":  # few distinct values
        x = torch.randint(0, 6, (T, Gmax), generator=gen, device=device).float()
    elif case == "flat_rank0":  # every token of DCP rank 0 tied at the top
        pos = torch.arange(Gmax, device=device)
        x = torch.where(((pos // INTERLEAVE_CUR[0]) % WORLD[0]) == 0, 1.0, -1.0)
        x = x[None, :].expand(T, Gmax).clone()
    elif case in ("skew_a", "skew_b"):
        # DCP rank 1 holds every top key, spread over many octaves: its share
        # estimate lies octaves above tau, the other ranks' windows far below, so
        # the centred 1R window misses and the row takes the refine round (two
        # variants with different bins)
        pos = torch.arange(Gmax, device=device)
        owner = ((pos // INTERLEAVE_CUR[0]) % WORLD[0]) == 1 % WORLD[0]
        u = torch.rand(T, Gmax, generator=gen, device=device)
        hi = torch.exp2(u * 60 - 30) if case == "skew_a" else torch.exp2(u * 40 - 5)
        x = torch.where(owner[None, :], hi, torch.exp2(u * 10 - 70))
    elif case == "neg":
        x = -torch.abs(x) * 100
    else:
        raise ValueError(case)
    return x.to(torch.bfloat16)


def _stock_global_mask(full, G, Lr, local, rank, W, il, T, Gmax, dev):
    """Stock VR merge (local persistent_topk + pack + all_gather + stable
    select) -> [T, Gmax] bool of the selected global ids, or None."""
    try:
        from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
            dcp_topk_id_bits,
            pack_dcp_topk_candidates_cutedsl,
            stable_topk_from_gathered_candidates_cutedsl,
        )
    except Exception:  # noqa: BLE001
        return None
    lg32 = local.float().contiguous()
    lg32 = torch.nan_to_num(lg32, nan=float("-inf"))
    loc = torch.empty(T, K, dtype=torch.int32, device=dev)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)
    torch.ops._C.persistent_topk(lg32, Lr, loc, ws, K, int(Lr.max()))
    packed = torch.empty(T, K, 2, dtype=torch.float32, device=dev)
    pack_dcp_topk_candidates_cutedsl(lg32, loc, packed, rank, W, il, None)
    buf = torch.empty(W, T, K, 2, dtype=torch.float32, device=dev)
    dist.all_gather_into_tensor(buf, packed, group=GROUP[0])
    sel = torch.empty(T, K, dtype=torch.int32, device=dev)
    # The base reads the rank-major [W, T, K, 2] all-gather in place.
    stable_topk_from_gathered_candidates_cutedsl(
        buf, K, out=sel, score_bits=16, id_bits=dcp_topk_id_bits(1 << 22))
    m = torch.zeros(T, Gmax + 1, dtype=torch.bool, device=dev)
    s = sel.long()
    s = torch.where((s >= 0) & (s < Gmax), s, torch.full_like(s, Gmax))
    m.scatter_(1, s, True)
    return m[:, :Gmax]


_SIX_MODES = ((True, "top"), (True, "top"), (True, "center"),
              (False, "top"), (True, "center"), (False, "top"))


def check_topk(ex: DcpExchange, case: str, B: int, q: int, ctx: int, seed: int, il: int,
               stats: torch.Tensor, stock: bool, modes=_SIX_MODES, before_call=None):
    W, rank, dev = ex.W, ex.rank, ex.device
    T = B * q
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + SEED_OFF[0])
    ctxs = torch.randint(max(1, ctx // 4), ctx + 1, (B,), generator=gen, device=dev)
    if case in ("short",):
        ctxs = torch.randint(1, K + 1, (B,), generator=gen, device=dev)
    if case == "edge":
        ctxs = torch.full((B,), K + 1, device=dev)
    ctxs[0] = ctx
    # spec rows: row j of request b sees ctx_b - q + 1 + j tokens
    G = (ctxs[:, None] - q + 1 + torch.arange(q, device=dev)[None, :]).clamp(min=1).reshape(-1)
    G = G.to(torch.int32)
    lcase = "normal" if case in ("short", "edge") else case
    INTERLEAVE_CUR[0], WORLD[0] = il, W
    full = make_logits(lcase, T, G, gen, dev)
    Gmax = full.shape[1]
    Lr = local_len_of(G.to(torch.int64), rank, W, il).to(torch.int32)
    Lmax_all = int(local_len_of(torch.tensor([Gmax]), 0, W, il))
    pos = torch.arange(Lmax_all, device=dev)
    gid = local_to_global(pos, rank, W, il)
    stride = (Lmax_all + 7) // 8 * 8 + 8
    local = torch.full((T, stride), float("nan"), dtype=torch.bfloat16, device=dev)
    ok = gid < Gmax
    local[:, : Lmax_all][:, ok] = full[:, gid[ok]]

    # 1R (window at the local max, twice; centred at the local tau estimate,
    # twice) and 2R twice: all six outputs must be bitwise identical
    outs = []
    for one_round, window in modes:
        if before_call is not None:
            before_call()
        out = torch.full((T, K + 64), -7, dtype=torch.int32, device=dev)
        cnt = torch.full((T,), -7, dtype=torch.int32, device=dev)
        ex.topk_merge(local, Lr, G, out, cnt, max_local_len=int(Lr.max()),
                      one_round=one_round, window=window, interleave=il, stats=stats)
        outs.append((out, cnt))
    torch.cuda.synchronize()
    ex.check()
    errs = []
    out, cnt = outs[0]
    for j, (o2, c2) in enumerate(outs[1:], 1):
        if not (torch.equal(o2[:, :K], out[:, :K]) and torch.equal(c2, cnt)):
            nd = int(((o2[:, :K] != out[:, :K]).any(1) | (c2 != cnt)).sum())
            errs.append(f"nondeterministic: call {j} differs in {nd} rows")

    # ---- reference: K largest keys, ties by ascending global id
    fk = keys_of(full)
    colmask = torch.arange(Gmax, device=dev)[None, :] < G[:, None]
    fk = torch.where(colmask, fk, torch.zeros_like(fk))
    gids = torch.arange(Gmax, device=dev, dtype=torch.int64)
    comb = (fk.long() << 24) | (0xFFFFFF - gids)[None, :]
    comb = torch.where(fk > 0, comb, torch.full_like(comb, -1))
    kk = min(K, Gmax)
    top = comb.topk(kk, dim=1).values
    ref = torch.zeros(T, Gmax + 1, dtype=torch.bool, device=dev)
    rg = torch.where(top >= 0, 0xFFFFFF - (top & 0xFFFFFF), torch.full_like(top, Gmax))
    ref.scatter_(1, rg, True)
    ref = ref[:, :Gmax]
    del comb, top
    nvalid = (fk > 0).sum(1)
    tau = torch.where(nvalid > K, fk.sort(dim=1, descending=True).values[:, K - 1],
                      torch.zeros_like(nvalid, dtype=fk.dtype))
    del fk

    # ---- ours: union over ranks (local positions -> global ids)
    c = cnt.long()
    ar = torch.arange(K, device=dev)[None, :]
    valid = ar < c[:, None]
    sel = out[:, :K].long()
    if bool(((sel < 0) & valid).any()) or bool(((sel >= Lr.long()[:, None]) & valid).any()):
        errs.append("bad positions")
        return errs
    if not bool((out[:, :K][~valid] == -1).all()):
        errs.append("padding not -1")
    selg = local_to_global(sel.clamp(min=0), rank, W, il)
    selg = torch.where(valid, selg, torch.full_like(selg, Gmax))
    mine = torch.zeros(T, Gmax + 1, dtype=torch.bool, device=dev)
    mine.scatter_(1, selg, True)
    mine = mine[:, :Gmax]
    if int(mine.sum()) != int(c.sum()):
        errs.append("duplicates")
    # order: above-tau tokens (increasing position), then ties (increasing position)
    lk = keys_of(local[:, :Lmax_all]).long()
    sk = lk.gather(1, sel.clamp(min=0, max=Lmax_all - 1))
    gt = (sk > tau.long()[:, None]) | (nvalid <= K)[:, None]
    nxt = torch.arange(K - 1, device=dev)
    both = valid[:, 1:]
    bad_grp = (~gt[:, :-1] & gt[:, 1:]) & both
    same = gt[:, :-1] == gt[:, 1:]
    bad_ord = same & both & (sel[:, 1:] <= sel[:, :-1])
    if bool(bad_grp.any()) or bool(bad_ord.any()):
        errs.append(f"output order: {int(bad_grp.any(1).sum())} group / "
                    f"{int(bad_ord.any(1).sum())} position rows")
    del nxt
    allm = mine.to(torch.uint8)
    dist.all_reduce(allm, group=GROUP[0])
    if int(allm.max()) > 1:
        errs.append("token selected on two ranks")
    ours = allm > 0
    bad = (ours != ref).any(1)
    if bool(bad.any()):
        rows = bad.nonzero()[:5, 0].tolist()
        errs.append(f"set != reference in {int(bad.sum())} rows {rows}")
    if stock:
        sm = _stock_global_mask(full, G, Lr, local, rank, W, il, T, Gmax, dev)
        if sm is not None:
            srow = (sm != ref).any(1)
            nd = int(srow.sum())
            nmo = int(((sm != ours).any(1) & ~srow).sum())
            if rank == 0:
                print(f"     stock merge: differs from the reference in {nd}/{T} rows; "
                      f"ours differs from stock in {nmo} of the others", flush=True)
            if nmo:
                errs.append(f"set != stock merge in {nmo} rows")
    return errs


def check_stale_refine(ex: DcpExchange, il: int, stats: torch.Tensor):
    """Centred 1R refine messages are written only on the rows that take the refine
    round, while a row's (tag, parity) returns every 510 uses. A refine-round use
    at epoch e0 (logits skew_a), 509 one-round uses (every row's window and bin
    words rewritten, the refine words not), then different logits (skew_b) that take
    the refine round at e0 + 510 (e0's tag and parity) with DCP rank 1 (which holds
    the keys at tau) delayed, so the other ranks poll its refine message before it
    is sent: a stale refine message must not be accepted. The 3 x 511 uses per row
    also cross the epoch wrap (epochs stay in [0, 510))."""
    errs = []
    W, dev = ex.W, ex.device
    sleep = (lambda: torch.cuda._sleep(2_000_000)) if ex.rank == 1 else None
    one = ((True, "center"),)
    gen = torch.Generator(device=dev)
    gen.manual_seed(77)
    for k, (B, q, ctx) in enumerate([(4, 4, 100_000), (2, 8, 60_000), (1, 8, 139_000)]):
        T = B * q
        st = [torch.zeros_like(stats) for _ in range(3)]
        e = check_topk(ex, "skew_a", B, q, ctx, 900 + k, il, st[0], False, modes=one)
        errs += [f"prime T{T}: {x}" for x in e]
        lg = torch.randn(T, 4096, generator=gen, device=dev).to(torch.bfloat16)
        Gl = torch.full((T,), 4096 * W, dtype=torch.int32, device=dev)
        Ll = torch.full((T,), 4096, dtype=torch.int32, device=dev)
        o = torch.empty(T, K, dtype=torch.int32, device=dev)
        c = torch.empty(T, dtype=torch.int32, device=dev)
        for _ in range(509):
            ex.topk_merge(lg, Ll, Gl, o, c, max_local_len=4096, one_round=True,
                          window="center", interleave=il, stats=st[1])
        torch.cuda.synchronize()
        ex.check()
        e = check_topk(ex, "skew_b", B, q, ctx, 950 + k, il, st[2], False, modes=one,
                       before_call=sleep)
        errs += [f"stale T{T}: {x}" for x in e]
        fb = torch.stack([x[2] + x[3] for x in st])  # refine rows (centred: fallback_r8)
        dist.all_reduce(fb)
        fb = fb.tolist()
        if ex.rank == 0:
            print(f"     stale refine T{T}: refine row-ranks prime {fb[0]}, between {fb[1]}, "
                  f"victim {fb[2]}", flush=True)
        if fb[0] == 0 or fb[2] == 0:
            errs.append(f"T{T}: no refine rows (test does not exercise the refine slot)")
        for x in st:
            stats += x
    torch.cuda.synchronize()
    lo, hi = int(ex.tk_epoch.min()), int(ex.tk_epoch.max())
    if lo < 0 or hi >= 510:
        errs.append(f"epoch out of [0, 510): {lo}..{hi}")
    return errs


def check_tie_rank(ex: DcpExchange, case: str, B: int, q: int, ctx: int, seed: int):
    """tie_order="rank" (the decode wiring's default): the same tokens above the K-th
    key and the same number of ties at it as the global-id order, the ties filling
    the ranks in rank order (each rank before the last one with selected ties gives
    all of its ties), each rank's in position order."""
    W, rank, dev = ex.W, ex.rank, ex.device
    T = B * q
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + SEED_OFF[0])
    ctxs = torch.randint(max(1, ctx // 4), ctx + 1, (B,), generator=gen, device=dev)
    G = (ctxs[:, None] - q + 1 + torch.arange(q, device=dev)[None, :]).reshape(-1)
    G = G.to(torch.int32)
    full = make_logits(case, T, G, gen, dev)
    Lr = local_len_of(G.to(torch.int64), rank, W, INTERLEAVE).to(torch.int32)
    Lmax = int(local_len_of(torch.tensor([full.shape[1]]), 0, W, INTERLEAVE))
    pos = torch.arange(Lmax, device=dev)
    gid = local_to_global(pos, rank, W, INTERLEAVE)
    local = torch.full((T, Lmax + 8), float("nan"), dtype=torch.bfloat16, device=dev)
    ok = gid < full.shape[1]
    local[:, :Lmax][:, ok] = full[:, gid[ok]]
    res = {}
    for name, order, one_round in (("global", "global", False), ("rank", "rank", False),
                                   ("rank1c", "rank", True)):
        out = torch.full((T, K), -7, dtype=torch.int32, device=dev)
        cnt = torch.zeros(T, dtype=torch.int32, device=dev)
        ex.topk_merge(local, Lr, G, out, cnt, max_local_len=int(Lr.max()),
                      one_round=one_round, window="center", tie_order=order,
                      small_ctx=0)
        res[name] = (out, cnt)
    torch.cuda.synchronize()
    ex.check()
    keys = keys_of(local[:, :Lmax])
    keys = torch.where(pos[None, :] < Lr[:, None], keys, torch.zeros_like(keys))
    errs = []
    if not (torch.equal(res["rank1c"][0], res["rank"][0])
            and torch.equal(res["rank1c"][1], res["rank"][1])):
        errs.append("tie_order=rank: 1-round centred window differs from 2 rounds")
    # the K-th key: the smallest selected key over all ranks (global-id run)
    big = 1 << 20
    taus = torch.full((T,), big, dtype=torch.int64, device=dev)
    for r in range(T):
        c = int(res["global"][1][r])
        if c:
            taus[r] = int(keys[r, res["global"][0][r, :c].long()].min())
    dist.all_reduce(taus, op=dist.ReduceOp.MIN, group=GROUP[0])
    sel_eq = torch.zeros(T, W, dtype=torch.int64, device=dev)
    ties_r = torch.zeros(T, W, dtype=torch.int64, device=dev)
    for r in range(T):
        sg = res["global"][0][r, : int(res["global"][1][r])].long()
        sr = res["rank"][0][r, : int(res["rank"][1][r])].long()
        tau = int(taus[r])
        kg, kr = keys[r, sg], keys[r, sr]
        if not torch.equal(sg[kg > tau].sort().values, sr[kr > tau].sort().values):
            errs.append(f"row {r}: tokens above the K-th key differ")
        if bool((kr < tau).any()):
            errs.append(f"row {r}: rank order selected below the K-th key")
        sel_eq[r, rank] = int((kr == tau).sum())
        ties_r[r, rank] = int((keys[r] == tau).sum())
    dist.all_reduce(sel_eq, group=GROUP[0])
    dist.all_reduce(ties_r, group=GROUP[0])
    cnt_g = res["global"][1].clone()
    cnt_r = res["rank"][1].clone()
    dist.all_reduce(cnt_g, group=GROUP[0])
    dist.all_reduce(cnt_r, group=GROUP[0])
    if not torch.equal(cnt_g, cnt_r):
        errs.append("total counts differ between tie orders")
    for r in range(T):
        last = int((sel_eq[r] > 0).nonzero().max()) if bool((sel_eq[r] > 0).any()) else -1
        if last > 0 and not torch.equal(sel_eq[r, :last], ties_r[r, :last]):
            errs.append(f"row {r}: ties not filled in rank order")
    return errs[:5]


def check_slots(ex: DcpExchange, B: int, q: int, ctx: int, seed: int):
    """The decode wiring's form (dcp_exchange.decode): KV slots through a block table
    with a block stride above the block size (other layers' pages in between) and an
    explicit row -> request map must be exactly the position-mode selection mapped
    through that table."""
    W, rank, dev = ex.W, ex.rank, ex.device
    T, bs, bstride = B * q, 64, 96
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + SEED_OFF[0])
    ctxs = torch.randint(ctx // 2, ctx + 1, (B,), generator=gen, device=dev)
    G = (ctxs[:, None] - q + 1 + torch.arange(q, device=dev)[None, :]).reshape(-1)
    G = G.to(torch.int32)
    Lr = local_len_of(G.to(torch.int64), rank, W, INTERLEAVE).to(torch.int32)
    Lmax = int(Lr.max())
    logits = torch.randn(T, Lmax + 64, generator=gen, device=dev).to(torch.bfloat16)
    nblk = (Lmax + bs - 1) // bs
    # request order permuted against the rows: row r -> request perm[r // q]
    perm = torch.randperm(B, generator=gen, device=dev).to(torch.int32)
    req_of_row = perm.repeat_interleave(q)
    table = torch.randperm(B * nblk + 7, generator=gen, device=dev)[: B * nblk]
    table = table.view(B, nblk).to(torch.int32)
    table = torch.cat([table, torch.full((B, 3), -5, dtype=torch.int32, device=dev)], 1)
    pos = torch.full((T, K), -7, dtype=torch.int32, device=dev)
    pcnt = torch.zeros(T, dtype=torch.int32, device=dev)
    ex.topk_merge(logits, Lr, G, pos, pcnt, max_local_len=Lmax, one_round=False)
    slots = torch.full((T, K), -7, dtype=torch.int32, device=dev)
    scnt = torch.zeros(T, dtype=torch.int32, device=dev)
    ex.err.zero_()
    ex.topk_merge(
        logits, Lr, G, slots, scnt, max_local_len=Lmax, block_table=table,
        block_size=bs, block_stride=bstride, req_of_row=req_of_row, one_round=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    if int(ex.err.item()):
        return [f"kernel error {int(ex.err.item())}"]
    if not torch.equal(pcnt, scnt):
        return ["slot counts differ from position counts"]
    # Invert the table: slot -> local position of the row's request.
    errs = []
    for r in range(T):
        c = int(pcnt[r])
        req = int(req_of_row[r])
        got = slots[r, :c].long()
        blk, off = got // bstride, got % bstride
        inv = torch.full((B * nblk + 7,), -1, dtype=torch.int64, device=dev)
        inv[table[req, :nblk].long()] = torch.arange(nblk, device=dev)
        p_got = inv[blk] * bs + off
        if (off >= bs).any() or (inv[blk] < 0).any() or (p_got >= int(Lr[r])).any():
            errs.append(f"row {r}: slot outside the request's blocks"); continue
        if p_got.unique().numel() != c:
            errs.append(f"row {r}: duplicate slots"); continue
        # deterministic output (position order): exactly the mapped positions
        if not torch.equal(p_got, pos[r, :c].long()):
            errs.append(f"row {r}: slots are not the mapped positions")
    return errs


def check_o(ex: DcpExchange, T: int, D: int, seed: int):
    W, rank, dev, H, hpr = ex.W, ex.rank, ex.device, ex.heads, ex.hpr
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed * 100 + rank + SEED_OFF[0])
    y = torch.randn(T, H, D, generator=gen, device=dev).to(torch.bfloat16)
    lse = torch.randn(T, H, generator=gen, device=dev) * 3
    empty = torch.rand(T, H, generator=gen, device=dev) < 0.1
    lse[empty] = float("-inf")
    y[empty] = float("nan")  # an empty partial's payload is garbage
    out = torch.empty(T, hpr, D, dtype=torch.bfloat16, device=dev)
    out_lse = torch.empty(T, hpr, dtype=torch.float32, device=dev)
    ex.err.zero_()
    ex.o_exchange(y, lse, out, lse_base_e=True, out_lse=out_lse)
    torch.cuda.synchronize()
    assert int(ex.err.item()) == 0
    ys = [torch.empty_like(y) for _ in range(W)]
    ls = [torch.empty_like(lse) for _ in range(W)]
    dist.all_gather(ys, y, group=GROUP[0])
    dist.all_gather(ls, lse, group=GROUP[0])
    Y = torch.stack(ys)[:, :, rank * hpr : (rank + 1) * hpr].float()  # [W, T, hpr, D]
    Y = torch.nan_to_num(Y, nan=0.0)
    L = torch.stack(ls)[:, :, rank * hpr : (rank + 1) * hpr]
    Lm = L.max(0).values
    w = torch.exp(L - Lm)
    w = torch.nan_to_num(w, nan=0.0)
    s = w.sum(0)
    ref = (w[..., None] * Y).sum(0) / s.clamp(min=1e-30)[..., None]
    ref = torch.where((s > 0)[..., None], ref, torch.zeros_like(ref))
    err = (out.float() - ref).abs().max().item()
    lref = torch.where(s > 0, Lm + torch.log(s), torch.full_like(s, float("-inf")))
    lerr = (torch.nan_to_num(out_lse, neginf=0) - torch.nan_to_num(lref, neginf=0)).abs().max().item()
    return [] if (err < 2e-2 and lerr < 1e-4) else [f"O mismatch T{T} D{D}: {err} lse {lerr}"]


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    world = dist.get_world_size()
    g = int(os.environ.get("K31_DCPX_GROUP_SIZE", str(world)))
    assert world % g == 0, (world, g)
    group = dist.group.WORLD
    if g < world:
        # every rank creates every group (new_group is collective over the world)
        groups = [dist.new_group(list(range(i, i + g))) for i in range(0, world, g)]
        group = groups[rank // g]
        GROUP[0] = group
        SEED_OFF[0] = 7919 * (rank // g)
    # K31_DCPX_TEST_NW1: the 1-round window (default 256 = the decode wiring, whose
    # data the stale-refine / epoch-wrap check is built to drive; 512 is the wider
    # variant, where that check's T8 case takes no refine round)
    nw1 = int(os.environ.get("K31_DCPX_TEST_NW1", "256"))
    ex = DcpExchange(group, dev, cap=1536, cap1=256, one_round_window=nw1,
                     interleave=INTERLEAVE)
    assert ex.W == g and ex.rank == rank % g, (ex.W, ex.rank)
    if rank == 0:
        print(f"DCP groups of {g} ranks x {world // g}", flush=True)
    stats = torch.zeros(8, dtype=torch.int32, device=dev)
    failures = []
    cases = [
        ("normal", 1, 8, 200_000),
        ("normal", 4, 4, 60_000),
        ("hot_recent", 8, 4, 190_000),
        ("relu", 4, 8, 120_000),
        ("wide", 4, 4, 100_000),
        ("sink", 2, 4, 80_000),
        ("ties", 4, 4, 50_000),
        ("neg", 2, 8, 30_000),
        ("short", 8, 4, 4000),
        ("edge", 4, 4, 4000),
        ("normal", 64, 4, 190_000),
        ("hot_recent", 128, 4, 120_000),
        ("relu", 192, 4, 60_000),       # T = 768
        ("normal", 128, 8, 190_000),    # T = 1024 (> one wave at 1 CTA/row)
        ("hot_recent", 192, 8, 150_000),  # T = 1536
        ("short", 192, 8, 4000),        # T = 1536, select-all rows
        ("flat_rank0", 2, 4, 60_000),   # one rank holds > kTieStore ties to select
        # K < context < W * 896 at IL 896: ranks without keys, tau low in the context
        ("normal", 1, 8, 2600),
        ("relu", 1, 8, 3000),
    ]
    if os.environ.get("K31_DCPX_QUICK"):
        cases = cases[:3]
    ils = [int(x) for x in os.environ.get("K31_DCPX_ILS", f"{INTERLEAVE},1").split(",")]
    for il in ils:
        for i, (case, B, q, ctx) in enumerate(cases):
            errs = check_topk(ex, case, B, q, ctx, 1000 + i, il, stats, stock=B * q <= 1024)
            tag = f"topk {case} B{B} q{q} ctx{ctx} IL{il}"
            if errs:
                failures.append(tag + ": " + "; ".join(errs[:3]))
            if rank == 0:
                print(("FAIL " if errs else "ok   ") + tag, flush=True)
    errs = check_stale_refine(ex, INTERLEAVE, stats)
    if errs:
        failures.append("stale refine / epoch wrap: " + "; ".join(errs[:3]))
    if rank == 0:
        print(("FAIL " if errs else "ok   ") + "stale refine / epoch wrap", flush=True)
    for i, (case, B, q, ctx) in enumerate(
        [("normal", 4, 4, 150_000), ("ties", 8, 4, 50_000), ("relu", 16, 8, 60_000),
         # more rows than resident CTAs: the 1R rows' direct write in multi-row CTAs
         ("hot_recent", 192, 8, 150_000)]
    ):
        errs = check_tie_rank(ex, case, B, q, ctx, 700 + i)
        tag = f"tie_order=rank {case} B{B} q{q} ctx{ctx}"
        if errs:
            failures.append(tag + ": " + "; ".join(errs[:3]))
        if rank == 0:
            print(("FAIL " if errs else "ok   ") + tag, flush=True)
    for i, (B, q, ctx) in enumerate([(4, 4, 150_000), (16, 8, 60_000), (64, 4, 30_000)]):
        errs = check_slots(ex, B, q, ctx, 500 + i)
        tag = f"slots B{B} q{q} ctx{ctx}"
        if errs:
            failures.append(tag + ": " + "; ".join(errs[:3]))
        if rank == 0:
            print(("FAIL " if errs else "ok   ") + tag, flush=True)
    # repeated calls (epoch / parity / tag reuse), incl. launches with no merge row
    for it in range(300):
        if it % 50 == 0:
            errs = check_topk(ex, "normal", 2, 4, 30_000, 7 + it % 3, INTERLEAVE, stats, False)
        else:
            errs = []
            T = 8
            short = it % 7 == 3  # every row global_len <= K: only the row-0 token
            lg = torch.randn(T, 4096, device=dev).to(torch.bfloat16)
            Gl = torch.full((T,), 512 * ex.W if short else 4096 * ex.W, dtype=torch.int32, device=dev)
            Ll = torch.full((T,), 512 if short else 4096, dtype=torch.int32, device=dev)
            o = torch.empty(T, K, dtype=torch.int32, device=dev)
            c = torch.empty(T, dtype=torch.int32, device=dev)
            ex.topk_merge(lg, Ll, Gl, o, c, max_local_len=4096, one_round=it % 3 == 0,
                          window="center" if it % 2 else "top")
        if errs:
            failures.append(f"repeat {it}: {errs[:2]}")
    torch.cuda.synchronize()
    ex.check()
    if rank == 0:
        print("ok   repeated calls" if not failures else "", flush=True)
    # Serving spin limits: a peer that arrives 7 s late (past the kernels' 5 s
    # test/warmup default) must not trap under the serving limit or with none.
    import time

    for limit in (600.0, 0.0):
        ex.set_spin_timeout(limit)
        torch.cuda.synchronize()
        dist.barrier()
        # top-k: no collective before the kernel, so the peers spin while rank 0
        # sleeps on the host.
        T = 8
        lg = torch.randn(T, 4096, device=dev).to(torch.bfloat16)
        Gl = torch.full((T,), 4096 * ex.W, dtype=torch.int32, device=dev)
        Ll = torch.full((T,), 4096, dtype=torch.int32, device=dev)
        o = torch.empty(T, K, dtype=torch.int32, device=dev)
        c = torch.empty(T, dtype=torch.int32, device=dev)
        if rank == 0:
            time.sleep(7.0)
        ex.topk_merge(lg, Ll, Gl, o, c, max_local_len=4096)
        torch.cuda.synchronize()
        ex.check()
        total = c.sum().clone()
        dist.all_reduce(total, group=GROUP[0])
        errs = [] if int(total) == T * K else [f"late top-k count {int(total)}"]
        # O exchange: its data is local, so the late rank delays the kernel itself.
        if rank == W_LATE_SKEW_RANK:
            time.sleep(7.0)
        errs += check_o(ex, 8, 128, 77)
        torch.cuda.synchronize()
        ex.check()
        tag = f"late peer (7 s) under spin limit {limit:g} s"
        if errs:
            failures.append(tag + ": " + "; ".join(errs[:3]))
        if rank == 0:
            print(("FAIL " if errs else "ok   ") + tag, flush=True)
    ex.set_spin_timeout(5.0)
    for T in (1, 8, 64, 512, 1536):
        for D in (128, 512):
            errs = check_o(ex, T, D, T + D)
            if errs:
                failures.extend(errs)
            if rank == 0:
                print(("FAIL " if errs else "ok   ") + f"o_exchange T{T} D{D}", flush=True)
    dist.all_reduce(stats)
    if rank == 0:
        from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.dsa_kernels.dcp_exchange.ops import STAT_NAMES

        print("stats (rows, summed over ranks): "
              + ", ".join(f"{n}={int(v)}" for n, v in zip(STAT_NAMES, stats.tolist())),
              flush=True)
    fl = torch.tensor([len(failures)], device=dev)
    dist.all_reduce(fl)
    if failures:
        print(f"[rank {rank}] failures:\n  " + "\n  ".join(failures), flush=True)
    dist.barrier()
    if rank == 0:
        print("ALL PASS" if int(fl) == 0 else f"FAILURES {int(fl)}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if int(fl) == 0 else 1)


if __name__ == "__main__":
    main()
