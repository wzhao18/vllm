# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation of the DcpExchange decode in the real model path
(``VLLM_K31_DCP_EXCHANGE_VERIFY=1``; eager steps only, collective, slow).

* ``verify_topk``: all-gathers the step's bf16 indexer logits over the DCP group,
  rebuilds every row's global context (interleaved DCP layout) and its exact top-K
  threshold, maps the expected local positions through the MLA block table, and checks
  this rank's exchanged slots: every token above the K-th key is selected, every
  selected token is above or at the K-th key, no duplicates, and the ranks' counts sum
  to min(K, context). That is "the same set as the exact reference up to ties".
* ``verify_o``: all-gathers the partial outputs and LSEs, merges this rank's heads in
  FP32 and applies W_UV in FP32 (the ideal), and in the stock order (FP32 merge, one
  bf16 rounding, bf16 W_UV); reports both errors against the ideal.

Counters land in ``dcpx.vstats`` (logged at exit, JSON per rank when
``VLLM_K31_DCP_EXCHANGE_VERIFY_OUT`` names a directory).
"""

from __future__ import annotations

import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _add(stats: dict, key: str, value: float) -> None:
    stats[key] = stats.get(key, 0) + value


def _max(stats: dict, key: str, value: float) -> None:
    stats[key] = max(stats.get(key, 0.0), value)


@torch.no_grad()
def verify_topk(dx, logits, local, glob, out, rows, md, stride, max_local) -> None:
    from vllm.distributed import get_dcp_group

    group = get_dcp_group()
    W, rank, IL, K = dx.W, dx.rank, dx.interleave, dx.K
    dev = logits.device
    lg = logits[:rows, :max_local].contiguous()
    all_l = group.all_gather(lg.unsqueeze(0), dim=0)  # [W, T, max_local]
    loc = local[:rows].reshape(1, -1).to(torch.int32).contiguous()
    all_loc = group.all_gather(loc, dim=0)  # [W, T]
    cnt = dx.counts[:rows].reshape(1, -1).clone()
    all_cnt = group.all_gather(cnt, dim=0).sum(0)  # [T]
    glob = glob[:rows].to(torch.int64)
    gmax = int(glob.max())
    bs = int(md.block_size)
    bt = md.block_table
    req = md.req_id_per_token[:rows].to(torch.int64)
    stats = dx.vstats
    tag = f"topk.T{rows}"
    bad_rows: list[tuple] = []
    if gmax <= 0:
        return
    gpos = torch.arange(gmax, device=dev, dtype=torch.int64)
    owner = (gpos // IL) % W
    lp = (gpos // (IL * W)) * IL + gpos % IL
    lp_c = lp.clamp(max=max_local - 1)
    mine = owner == rank
    chunk = max(1, (1 << 24) // gmax)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        rr = torch.arange(r0, r1, device=dev)
        vals = all_l[owner[None, :], rr[:, None], lp_c[None, :]].float()
        in_ctx = gpos[None, :] < glob[r0:r1, None]
        held = lp[None, :] < all_loc[owner[None, :], rr[:, None]].to(torch.int64)
        _add(stats, "topk.layout_mismatch", int((in_ctx & ~held).sum()))
        valid = in_ctx & held & ~torch.isnan(vals)
        n_valid = valid.sum(1)
        masked = torch.where(valid, vals, torch.full_like(vals, -math.inf))
        kk = min(K, gmax)
        tau = masked.topk(kk, dim=1).values[:, -1]
        sel_all = n_valid <= K
        strict = valid & ((masked > tau[:, None]) | sel_all[:, None])
        ties = valid & (masked == tau[:, None]) & ~sel_all[:, None]
        expect_total = torch.minimum(n_valid, torch.full_like(n_valid, K))
        need = expect_total - strict.sum(1)
        tie_choice = (ties.sum(1) > need) & (need > 0)
        bad_total = all_cnt[r0:r1].to(torch.int64) != expect_total
        blk = (lp_c // bs).clamp(max=bt.shape[1] - 1)
        slots = bt[req[r0:r1, None], blk[None, :]].to(torch.int64) * stride + lp_c % bs
        n_bad = 0
        for i in range(r1 - r0):
            r = r0 + i
            c = int(dx.counts[r])
            sel = out[r, :c].to(torch.int64)
            dup = sel.unique().numel() != c
            s_mine = slots[i][strict[i] & mine]
            t_mine = slots[i][ties[i] & mine]
            missing = int((~torch.isin(s_mine, sel)).sum())
            extra = int((~torch.isin(sel, torch.cat([s_mine, t_mine]))).sum())
            bad = dup or missing or extra or bool(bad_total[i])
            if bad:
                n_bad += 1
                if len(bad_rows) < 4:
                    bad_rows.append(
                        (
                            r,
                            c,
                            int(all_cnt[r]),
                            int(expect_total[i]),
                            missing,
                            extra,
                            dup,
                        )
                    )
        _add(stats, f"{tag}.rows", r1 - r0)
        _add(stats, f"{tag}.bad", n_bad)
        _add(stats, f"{tag}.tie_choice_rows", int(tie_choice.sum()))
        _add(stats, f"{tag}.ties_at_kth", int(ties.sum()))
        _add(stats, "topk.rows", r1 - r0)
        _add(stats, "topk.bad", n_bad)
    _add(stats, "topk.calls", 1)
    if bad_rows:
        logger.warning(
            "DcpExchange verify rank %d T%d: bad rows (row, count, ranks' count, "
            "expected, missing, extra, dup): %s",
            rank,
            rows,
            bad_rows,
        )


@torch.no_grad()
def verify_o(dx, layer, attn_out, lse, out) -> None:
    from vllm.distributed import get_dcp_group

    group = get_dcp_group()
    rank, hpr = dx.rank, dx.hpr
    rows = attn_out.shape[0]
    a_all = group.all_gather(attn_out.unsqueeze(0).contiguous(), dim=0)
    l_all = group.all_gather(lse.unsqueeze(0).float().contiguous(), dim=0)
    hs = slice(rank * hpr, (rank + 1) * hpr)
    a = a_all[:, :, hs].float()  # [W, T, hpr, Dl]
    lw = l_all[:, :, hs]  # [W, T, hpr]
    if not bool(layer.impl.lse_base_on_e):
        lw = lw * math.log(2.0)
    m = lw.amax(0)
    live = torch.isfinite(m)
    w = torch.exp(lw - torch.where(live, m, torch.zeros_like(m)))
    w = torch.where(live[None], w, torch.zeros_like(w))
    s = w.sum(0)
    w = w / torch.where(s > 0, s, torch.ones_like(s))
    merged = (w[..., None] * a).sum(0)  # [T, hpr, Dl] fp32
    w_uv = layer.W_UV  # [hpr, Dl, 128]
    ideal = torch.bmm(merged.transpose(0, 1), w_uv.float()).transpose(0, 1)
    stock = torch.bmm(merged.to(w_uv.dtype).transpose(0, 1), w_uv).transpose(0, 1)
    got = out.float()

    def rel(x: torch.Tensor) -> float:
        if not bool(live.any()):
            return 0.0
        d = (x - ideal)[live]
        ref = ideal[live]
        scale = ref.square().mean().sqrt().clamp(min=1e-30)
        return float(d.square().mean().sqrt() / scale)

    stats = dx.vstats
    tag = f"o.T{rows}"
    e_x, e_s = rel(got), rel(stock.float())
    _add(stats, f"{tag}.calls", 1)
    _max(stats, f"{tag}.exchange_vs_fp32_max", e_x)
    _max(stats, f"{tag}.stock_vs_fp32_max", e_s)
    _add(stats, f"{tag}.exchange_vs_fp32_sum", e_x)
    _add(stats, f"{tag}.stock_vs_fp32_sum", e_s)
    _max(
        stats,
        f"{tag}.exchange_vs_stock_maxabs",
        float((got - stock.float())[live].abs().max()) if bool(live.any()) else 0.0,
    )
    _add(stats, f"{tag}.dead_heads", int((~live).sum()))
    _add(stats, f"{tag}.dead_nonzero", int((got[~live] != 0).sum()))
    _add(stats, f"{tag}.nonfinite", int((~torch.isfinite(got)).sum()))
