"""KV / state capacity per GPU.

    free_per_GPU  = HBM budget (272.48 GiB, measured free at vLLM start) - resident weights per GPU - reserve
    bytes_group   = per-request bytes stored by the whole TP/DCP group (DSA KV x replication + KDA state + conv +
                    RecoverSSM records + drafter SWA), SOL minimal state (one FP32 KDA state per request, FP32 conv)
    r_max         = free_per_GPU / bytes_group        (requests per GPU; group of N holds N x r requests)

Replication: 'rep' DSA stores the full DSA KV on every rank (x N); DCP / DSA-DP / DEP store it once.
DSA-CP (D9) and DEP hold the same bytes per GPU on average (DCP only balances long contexts).
"""
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


def capacity(t, ck_name, k, L, phase="decode"):
    if t.afd:
        return afd_capacity(t, ck_name, k, L)
    w, by = resident_weights(t, phase, ck_name)
    budget = HWS["sol"].hbm_budget
    free = budget - w - RESERVE_GIB * GiB
    b, parts = per_request_group_bytes(t, k, L)
    rmax = max(free, 0.0) / b
    return dict(weights_GB=w / 1e9, weights_by={kk: v / 1e9 for kk, v in by.items()}, free_GB=free / 1e9,
                per_req_GB=b / 1e9, per_req_parts_GB={kk: v / 1e9 for kk, v in parts.items()}, rmax=rmax,
                feasible=free > 0)
