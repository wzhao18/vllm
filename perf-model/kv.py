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


def capacity(t, ck_name, k, L, phase="decode"):
    w, by = resident_weights(t, phase, ck_name)
    budget = HWS["sol"].hbm_budget
    free = budget - w - RESERVE_GIB * GiB
    b, parts = per_request_group_bytes(t, k, L)
    if t.afd:
        nA, nF = t.afd["choices"][1]
        # A-pool holds no experts; capacity per total GPU scales with nA/(nA+nF)
        rmax = max(free, 0) / b * nA / (nA + nF)
    else:
        rmax = max(free, 0.0) / b
    return dict(weights_GB=w / 1e9, weights_by={kk: v / 1e9 for kk, v in by.items()}, free_GB=free / 1e9,
                per_req_GB=b / 1e9, per_req_parts_GB={kk: v / 1e9 for kk, v in parts.items()}, rmax=rmax,
                feasible=free > 0)
