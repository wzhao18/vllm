"""EP <= 16 targets for the microbench lanes: model t_SOL / t_achievable per op for every in-scope topology and band.

    python3 perf-model/run.py && python3 perf-model/microbench/results/model/make_targets.py

Reads out/optables.json (written by run.py) and writes targets.json next to this file.
Bands = the op-table regimes: prefill F=16384 over L=78.4K; decode r in {1, 8, 32} requests per GPU, k=3, L=135K
(r above r_max is absent). Per-op times are per instance (one kernel of one layer); `count` instances per step.
derate for a lane = measured_us / t_ach_us of the matching row.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, PM)

from topologies import ALL, SCOPE  # noqa: E402

BAND = {"prefill": "prefill_F16K_L78K", "decode r1": "decode_r1", "decode r8": "decode_r8", "decode r32": "decode_r32"}


def main():
    with open(os.path.join(PM, "out", "optables.json")) as f:
        opt = json.load(f)
    with open(os.path.join(PM, "out", "meta.json")) as f:
        meta = json.load(f)
    cols, ccols = opt["cols"], opt["comm_cols"]
    out = dict(
        generated=meta["generated"], source="perf-model/out/optables.json (run.py), scope = topologies.SCOPE",
        scope_rule="EP <= 16 and <= 16 GPUs per engine/EP domain (MoE-TP D1 included; AFD, PP4, EP32-72 dropped)",
        units="t_sol_us / t_ach_us per op instance; count = instances per step; step_* = whole step in ms",
        bands={"prefill_F16K_L78K": "prefill chunk F=16384 fresh tokens per engine over L=78.4K, 5,456 fresh/request",
               "decode_r1": "decode r=1 request/GPU, k=3 (4 rows/request), L=135K",
               "decode_r8": "decode r=8", "decode_r32": "decode r=32"},
        topologies={}, targets={})
    for key, tb in opt["tables"].items():
        ck, tid, reg = key.split("|")
        if tid not in SCOPE:
            continue
        t = ALL[tid]
        out["topologies"][tid] = dict(name=t.name, phase=t.phase, N=t.N, E=t.E, gpus=t.gpus, dsa=t.dsa,
                                      moe_dense=t.moe_dense, experts=t.experts)
        band = BAND[reg]
        ops = []
        for r in tb["ops"]:
            d = dict(zip(cols, r))
            ops.append(dict(op_id=d["id"], fam=d["fam"], count=d["count"], dtype=d["dtype"],
                            t_sol_us=round(d["t_sol_s"] * 1e6, 3), t_ach_us=round(d["t_ach_s"] * 1e6, 3),
                            bound_ach=d["bound_ach"], flops=d["flops"], w_B=d["w_B"], act_B=d["act_B"],
                            kv_B=d["kv_B"], state_B=d["state_B"]))
        comm = []
        for r in tb["comm"]:
            d = dict(zip(ccols, r))
            comm.append(dict(layer=d["layer"], prim=d["prim"], name=d["name"], in_B=d["in_B"], out_B=d["out_B"],
                             hops=d["hops"], t_sol_us=round(d["t_sol_s"] * 1e6, 3),
                             t_ach_us=round(d["t_ach_s"] * 1e6, 3), async_=bool(d["async"])))
        out["targets"]["%s|%s|%s" % (ck, tid, band)] = dict(
            ckpt=ck, topology=tid, band=band, regime=tb["regime"], step_sol_ms=tb["t_sol_ms"],
            step_ach_ms=tb["t_ach_ms"], ops=ops, comm=comm)
    with open(os.path.join(HERE, "targets.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("targets: %d (topology, ckpt, band) tables, %d topologies" % (len(out["targets"]), len(out["topologies"])))


if __name__ == "__main__":
    main()
