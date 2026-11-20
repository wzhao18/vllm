"""KV / state capacity per GPU.

    free_per_GPU  = HBM budget (272.48 GiB, measured free at vLLM start) - resident weights per GPU - reserve
    bytes_group   = per-request bytes stored by the whole TP/DCP group (DSA KV x replication + KDA state + conv +
                    RecoverSSM records + drafter SWA), SOL minimal state (one FP32 KDA state per request, FP32 conv)
    r_max         = free_per_GPU / bytes_group        (requests per GPU; group of N holds N x r requests)

Replication: 'rep' DSA stores the full DSA KV on every rank (x N); DCP / DSA-DP / DEP store it once.
DSA-CP (D9) and DEP hold the same bytes per GPU on average (DCP only balances long contexts).

Two capacity views (2026-11-18, microbench decode lane "Capacity finding"):
  * col "sol"        - byte-exact minimal state (above).
  * col "achievable" - the deployed packed hybrid block pool (vLLM kv_cache_planning + kimi_k3 dsa_cache): every
    request holds whole block rows. A row holds bs tokens of all 24 DSA layers (bs x 15,456 B per GPU). Per request
    and per GPU of the group:
        DSA rows   ceil((L + 1 + k) / (bs x W))          W = DCP width (DCP N, DSA-CP G, else 1; DSA-DP: owner only)
        KDA rows   G x (1 + k)                            G >= 3 state groups (69 KDA layers in the 23-per-row repeat
                                                          pattern of the 24 DSA layers; more if a row fits fewer
                                                          states), 1 + num_speculative_blocks state slots each
        SWA rows   ceil(2048 / bs) + 1                    drafter sliding window ring
    The block size can be pinned to any multiple of 64 (VLLM_KIMI_K3_PACKED_BLOCK_SIZE; the planner re-splits the
    states), so the achievable column takes the minimum per-request bytes over bs = 64..32768 (packed_min_bytes).
    Reproduced deployments: D1 unpinned bs 1024, G 3: 24 / 36 rows at k3 / k7 (measured 24 / 36); D2 pinned bs 1024,
    G 4, BF16 conv: 36 / 52 rows (measured 36 / 52). The fixed floor is the G x (1 + k) spec-decode state slots
    padded to full rows. The SP path (D3-type) measured 13 rows at bs 4096 = 9 DSA + 2 SWA + 2: its states do not
    take G x (1 + k) rows (layout not reproduced here), so residual 'sp' is charged the byte-exact state instead,
    which is below those 2 rows (achievable <= measured bytes per request, test_kv_block_floor).
"""
import math

import model as Mdl
from cost import resident_weights
from hw import GiB, HWS

RESERVE_GIB = 16.0   # activations + CUDA graphs + comm/symmetric buffers + MegaMoE arena (assumption)


def per_request_group_bytes(t, k, L):
    s = Mdl.per_request_state_bytes(k, L)
    rep = t.N if t.dsa == "rep" else 1
    tot = s["dsa"] * rep + s["kda"] + s["conv"] + s["recover"] + s["swa"]
    return tot, s


def afd_capacity(t, ck_name, k, L):
    """AFD A-pool capacity (the A pool holds no experts). r_max per total GPU depends on the A:F split; the curve
    uses the split with the most A GPUs (largest r_max) and run_afd skips any split whose A pool would overflow."""
    w, by = resident_weights(t, "decode", ck_name)
    b, parts = per_request_group_bytes(t, k, L)
    free = HWS["sol"].hbm_budget - w - RESERVE_GIB * GiB
    rA = max(free, 0.0) / b
    by_split = {"%d:%d" % (nA, nF): rA * nA / (nA + nF) for (nA, nF) in t.afd["choices"]}
    return dict(weights_GB=w / 1e9, weights_by={kk: v / 1e9 for kk, v in by.items()}, free_GB=free / 1e9,
                per_req_GB=b / 1e9, per_req_parts_GB={kk: v / 1e9 for kk, v in parts.items()},
                rmax=max(by_split.values()), rmax_A=rA, rmax_by_split=by_split, feasible=free > 0)


ALIGN = 64            # block-size alignment (lcm of the kernel block sizes, FlashMLA / indexer 64)
STATE_GROUPS = 3      # cdiv(69 KDA layers, 23 states per row): repeat pattern of the 24 DSA layers
DRAFT_SWA = 2048      # DSpark sliding window (tokens)
BS_MAX = 512          # pinned block sizes searched: 64 x {1..512} tokens


def _npow2(x):
    return 1 << max(0, math.ceil(math.log2(max(x, 1))))


def packed_layout(t, k, bs=None, G=None, conv_bytes=4):
    """Packed hybrid block layout of topology t: block size bs (tokens; None = the unpinned default of
    packed_dsa_block_size), row bytes per GPU, KDA state page per GPU, state groups G (>= 3 repeat pattern and >= the
    groups needed for the states to fit one row; a pinned VLLM_PACKED_KV_STATE_GROUPS may only raise it).
    conv_bytes: 4 = FP32 conv (user rule, the model's state), 2 = BF16 conv (deployed mamba_cache_dtype auto)."""
    C = Mdl.C
    tp = max(t.N, 1)
    page = (C["Hk"] * C["dk"] * C["dv"] * 4 + 3 * C["P"] * (C["CONV"] - 1 + k) * conv_bytes) / tp
    row_tok = C["N_DSA"] * (C["MLA_ROW"] + C["IDX_ROW"])
    if bs is None:
        per_row = math.ceil(C["N_KDA"] / math.ceil(C["N_KDA"] / C["N_DSA"]))
        bs = ALIGN * _npow2(math.ceil(per_row * page / (ALIGN * row_tok)))
    row_B = bs * row_tok
    fit = int(row_B // page)
    if fit < 1:
        return None                                   # a state does not fit one row at this block size
    G_min = max(STATE_GROUPS, math.ceil(C["N_KDA"] / fit))
    if G is not None and G < G_min:
        return None                                   # the planner refuses too few pinned groups
    return dict(bs=bs, row_B=row_B, page_B=page, G=max(G_min, G or 0))


def packed_request_group_bytes(t, k, L, bs=None, G=None, conv_bytes=4):
    """Per-request bytes on the whole group for one packed layout (see module docstring); None if infeasible."""
    lay = packed_layout(t, k, bs, G, conv_bytes)
    if lay is None:
        return None, None
    bs, row_B, G = lay["bs"], lay["row_B"], lay["G"]
    N = max(t.N, 1)
    Lq = L + 1 + k
    if t.dsa == "dcp":
        dsa_rows = N * math.ceil(Lq / (bs * N))
    elif t.dsa == "dsacp":
        dsa_rows = t.G * math.ceil(Lq / (bs * t.G))
    elif t.dsa == "rep":
        dsa_rows = N * math.ceil(Lq / bs)
    else:                                    # DSA-DP owner / DEP local
        dsa_rows = math.ceil(Lq / bs)
    swa_rows = N * (math.ceil(DRAFT_SWA / bs) + 1)
    s = Mdl.per_request_state_bytes(k, L)
    if t.residual == "sp":
        state_rows, state_B = 0, s["kda"] + s["conv"] + s["recover"]
    else:
        state_rows, state_B = N * G * (1 + k), 0.0
    rows_gpu = (dsa_rows + swa_rows + state_rows) / N
    tot = (dsa_rows + swa_rows + state_rows) * row_B + state_B
    return tot, dict(lay, rows_per_gpu=rows_gpu, dsa_rows=dsa_rows / N, swa_rows=swa_rows / N,
                     state_rows=state_rows / N, state_outside_pool_GB=state_B / 1e9, per_gpu_B=tot / N)


_MIN_CACHE = {}


def packed_min_bytes(t, k, L, conv_bytes=4):
    """Achievable: the smallest per-request bytes over every supported pinned block size (VLLM_KIMI_K3_PACKED_BLOCK_SIZE,
    any positive multiple of 64; the planner then splits the KDA states into as many groups as the row needs)."""
    key = (t.id, t.N, t.dsa, t.residual, t.G, k, L, conv_bytes)
    if key not in _MIN_CACHE:
        best = None
        for m in range(1, BS_MAX + 1):
            b, d = packed_request_group_bytes(t, k, L, bs=ALIGN * m, conv_bytes=conv_bytes)
            if b is not None and (best is None or b < best[0]):
                best = (b, d)
        _MIN_CACHE[key] = best
    return _MIN_CACHE[key]


def capacity(t, ck_name, k, L, phase="decode", col="sol"):
    if t.afd:
        return afd_capacity(t, ck_name, k, L)
    w, by = resident_weights(t, phase, ck_name)
    budget = HWS["sol"].hbm_budget
    free = budget - w - RESERVE_GIB * GiB
    b, parts = per_request_group_bytes(t, k, L)
    blocks = None
    if col != "sol":
        b, blocks = packed_min_bytes(t, k, L)
        blocks = dict(blocks, default=packed_request_group_bytes(t, k, L)[1])
    rmax = max(free, 0.0) / b
    return dict(weights_GB=w / 1e9, weights_by={kk: v / 1e9 for kk, v in by.items()}, free_GB=free / 1e9,
                per_req_GB=b / 1e9, per_req_parts_GB={kk: v / 1e9 for kk, v in parts.items()}, rmax=rmax,
                feasible=free > 0, col=col, blocks=blocks)
