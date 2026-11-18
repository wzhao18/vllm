"""Map pf_bench profile op rows (opmap.json) onto the SOL model's op ids (out/optables.json) and write rows.json.
usage: python3 build_rows.py <optables.json> <out rows.json> <topo>=<opmap.json>:<step_us>:<case> ...
measured_us per row = per-forward GPU busy time of the mapped kernels summed over all layers (rank 0, eager,
torch.profiler), t_ach_us/t_sol_us = model count x per-occurrence time (+ the comm rows a fused kernel absorbs)."""
import json
import re
import sys

# (kind regex, module path regex, kernel regex) -> model op id; first match wins
RULES = [
    # layer 0: KDA attention + dense MLP
    (r"L0", r"mlp\.gate_up_proj", r".", "DENSE.gate_up"),
    (r"L0", r"mlp\.down_proj", r".", "DENSE.down"),
    (r"L0", r"mlp\.act_fn|^mlp$", r".", "DENSE.act"),
    # KDA
    (r"KimiK3DeltaAttention|L0", r"in_proj", r".", "KDA.in_proj"),
    (r"KimiK3DeltaAttention|L0", r"f_b_proj|f_a_proj|b_proj", r".", "KDA.f_b_proj"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r"GemmRsAr|gemm_rs_ar", "KDA.o_proj+AR"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r"o_proj|nvjet", "KDA.o_proj+AR"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r"conv", "KDA.conv"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r"flash_kda|gather_initial|scatter_rows|segscan|chunk|recur", "KDA.core_prefill"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r"rmsnorm_gated", "KDA.o_norm"),
    (r"KimiK3DeltaAttention|L0", r"self_attn", r".", "KDA.gate_beta_norm"),
    # DSA
    (r"DSA", r"fused_qkv_a_proj|wkv_index|kv_a_proj|q_a_proj", r".", "DSA.qkv_a_packed"),
    (r"DSA", r"q_b_proj|wq_index", r".", "DSA.qb_wq_packed"),
    (r"DSA", r"g_proj", r".", "DSA.g_proj"),
    (r"DSA", r"self_attn$|o_proj", r"GemmRsAr|gemm_rs_ar|nvjet", "DSA.o_proj+AR"),
    (r"DSA", r"indexer", r"mqa_logits|device_kernel", "DSA.idx_score"),
    (r"DSA", r"indexer", r"topk|gvr", "DSA.topk"),
    (r"DSA", r"indexer", r"barrier|signal|nccl|push|multimem", "comm.dsa"),
    (r"DSA", r"indexer", r".", "DSA.idx_q_quant+insert"),
    (r"DSA", r"mla_attn", r"fmha|sparse_mla|flash_mla", "DSA.sparse_mla"),
    (r"DSA", r"mla_attn", r"nvjet|gemm", "DSA.w_uk+w_uv"),
    (r"DSA", r"mla_attn|self_attn", r"gate_sigmoid_mul", "DSA.gate_mul"),
    (r"DSA", r"mla_attn", r"barrier|signal|nccl|push|multimem|gather", "comm.dsa"),
    (r"DSA", r"mla_attn|layernorm|self_attn", r".", "DSA.prologue+aux"),
    # MoE
    (r".", r"block_sparse_moe\.gate$", r".", "MOE.router"),
    (r".", r"block_sparse_moe$", r"topk", "MOE.topk"),
    (r".", r"block_sparse_moe\.experts", r".", "MOE.megamoe(dispatch+fc1+act+fc2+combine)"),
    (r".", r"routed_expert_down_proj", r".", "MOE.latent_down"),
    (r".", r"routed_output_transform", r".", "MOE.latent_up"),
    (r".", r"routed_expert_norm", r".", "MOE.latent_norm"),
    (r".", r"shared_experts\.gate_up_proj", r".", "MOE.shared_gate_up"),
    (r".", r"shared_experts\.act_fn", r".", "MOE.shared_act"),
    (r".", r"shared_experts\.down_proj", r".", "MOE.shared_down"),
    (r".", r"block_sparse_moe", r"barrier|nccl|push|multimem", "comm.moe"),
    (r".", r"block_sparse_moe", r".", "MOE.other"),
    # layer level
    (r"Kimi|L0", r"^-$", r"attn_res", "L.attnres_pre+post"),
    (r"Kimi|L0", r"^-$", r"push_shard|barrier|nccl|multimem|Memcpy", "comm.layer(SP/token-shard AG)"),
    (r"Kimi|L0", r".", r"rms_norm|norm", "L.norms"),
    (r"draft", r".", r".", "DS.drafter"),
    (r"top", r"draft|propose", r".", "DS.drafter"),
    (r"top", r".", r"nccl|Memcpy|barrier|push", "comm.top(embed/final/bench restore)"),
    (r"top", r".", r"attn_res", "F.attnres_final"),
    (r"top", r".", r"embedding", "G.embed"),
    (r".", r".", r".", "other"),
]

# model op ids per mapped row (summed); "+AR": add the unit's synchronous comm rows the fused GEMM absorbs
MODEL = {
    "KDA.in_proj": ["KDA.in_proj"], "KDA.f_b_proj": ["KDA.f_b_proj"], "KDA.conv": ["KDA.conv"],
    "KDA.core_prefill": ["KDA.core_prefill"], "KDA.o_norm": ["KDA.o_norm"], "KDA.gate_beta_norm": ["KDA.gate_beta_norm"],
    "KDA.o_proj+AR": ["KDA.o_proj"], "DSA.qkv_a_packed": ["DSA.qkv_a_packed"], "DSA.qb_wq_packed": ["DSA.qb_wq_packed"],
    "DSA.g_proj": ["DSA.g_proj"], "DSA.o_proj+AR": ["DSA.o_proj"], "DSA.idx_score": ["DSA.idx_score"],
    "DSA.topk": ["DSA.topk"], "DSA.idx_q_quant+insert": ["DSA.idx_q_quant"], "DSA.sparse_mla": ["DSA.sparse_mla"],
    "DSA.w_uk+w_uv": ["DSA.w_uk", "DSA.w_uv"], "DSA.gate_mul": ["DSA.gate_mul"], "DSA.prologue+aux": ["DSA.prologue"],
    "MOE.router": ["MOE.router"], "MOE.topk": ["MOE.topk"],
    "MOE.megamoe(dispatch+fc1+act+fc2+combine)": ["MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine"],
    "MOE.latent_down": ["MOE.latent_down"], "MOE.latent_up": ["MOE.latent_up"], "MOE.latent_norm": ["MOE.latent_norm"],
    "MOE.shared_gate_up": ["MOE.shared_gate_up"], "MOE.shared_act": ["MOE.shared_act"], "MOE.shared_down": ["MOE.shared_down"],
    "L.attnres_pre+post": ["L.attnres_pre", "L.attnres_post"], "DENSE.gate_up": ["DENSE.gate_up"],
    "DENSE.down": ["DENSE.down"], "DENSE.act": ["DENSE.act"], "DS.drafter": ["DS.aux_capture", "DS.fc", "DS.hidden_norm", "DS.ctx_kv_proj", "DS.ctx_kv_insert"],
    "G.embed": ["G.embed"], "F.attnres_final": ["F.attnres_final"],
}
# comm rows a fused kernel absorbs: (unit, name regex)
FUSED_COMM = {"KDA.o_proj+AR": ("kda", r"AR|RS|o_proj"), "DSA.o_proj+AR": ("dsa", r"o_proj"),
              "MOE.megamoe(dispatch+fc1+act+fc2+combine)": ("moe", r"MegaMoE")}
UNIT_N = {"kda": 69, "dsa": 24, "moe": 92, "dense": 1, "embed": 1, "final": 1, "drafter": 1}


def classify(r):
    for k, p, n, op in RULES:
        if re.search(k, r["kind"]) and re.search(p, r["path"]) and re.search(n, r["kernel"]):
            return op
    return "other"


def main():
    opt = json.load(open(sys.argv[1]))
    out = []
    for spec in sys.argv[3:]:
        topo, rest = spec.split("=", 1)
        path, step_us, case = rest.split(":")
        key = topo
        tab = opt["tables"][key] if key in opt["tables"] else opt["tables"][f"mxfp4_release|{key}|prefill"]
        topo = tab.get("topo", key)
        case = f"{case} [model {key}]"
        cols = opt["cols"]
        ops = {r[0]: dict(zip(cols, r)) for r in tab["ops"]}
        ccols = opt["comm_cols"]
        comm = [dict(zip(ccols, r)) for r in tab["comm"]]
        rows = json.load(open(path))
        meas, excl, kern = {}, {}, {}
        for r in rows:
            op = classify(r)
            if op.startswith("comm.layer"):
                op = "comm.layer.dsa" if "DSA" in r["kind"] else "comm.layer.kda"
            meas[op] = meas.get(op, 0.0) + r["busy_ms"] * 1e3
            excl[op] = excl.get(op, 0.0) + r["excl_ms"] * 1e3
            kern.setdefault(op, {})
            kern[op][r["kernel"][:60]] = kern[op].get(r["kernel"][:60], 0) + r["busy_ms"] * 1e3
        for op in sorted(meas, key=lambda o: -meas[o]):
            ids = MODEL.get(op, [])
            t_ach = sum(ops[i]["count"] * ops[i]["t_ach_s"] for i in ids if i in ops) * 1e6
            t_sol = sum(ops[i]["count"] * ops[i]["t_sol_s"] for i in ids if i in ops) * 1e6
            fused = []
            if op in FUSED_COMM:
                unit, rx = FUSED_COMM[op]
                for c in comm:
                    if c["layer"] == unit and not c["async"] and re.search(rx, c["prim"] + " " + c["name"]):
                        t_ach += UNIT_N[unit] * c["t_ach_s"] * 1e6
                        t_sol += UNIT_N[unit] * c["t_sol_s"] * 1e6
                        fused.append(f"{c['prim']} ({c['name']})")
            top = sorted(kern[op].items(), key=lambda x: -x[1])[:3]
            d = meas[op] / t_ach if t_ach > 0 else None
            note = f"busy {meas[op] / 1e3:.2f} ms/step, critical-path excl {excl[op] / 1e3:.2f} ms; kernels: " + \
                   "; ".join(f"{k} {v / 1e3:.2f} ms" for k, v in top)
            if fused:
                note += "; model comm folded in: " + ", ".join(fused)
            if d is not None and d > 2.0:
                note = "KERNEL-DEBUG; " + note
            out.append(dict(case=f"{topo} {case} (step median {float(step_us) / 1e3:.1f} ms)", topology=topo,
                            op_ids=ids or [op], row=op, measured_us=round(meas[op], 1), t_sol_us=round(t_sol, 1),
                            t_ach_us=round(t_ach, 1), derate=round(d, 3) if d else None, notes=note))
        # synchronous comm the model prices but the profile shows as separate comm kernels
        # block rows on the critical path (exclusive time: overlapped streams counted once)
        def fam(prefixes):
            return [o for o in excl if o.startswith(prefixes)]
        for name, pref, model_ids, units in [
            ("block: KDA attention + its SP/AR comm (excl)", ("KDA.", "comm.layer.kda"), [i for i in ops if i.startswith("KDA.")], ("kda",)),
            ("block: DSA attention + its SP/AR comm (excl)", ("DSA.", "comm.dsa", "comm.layer.dsa"), [i for i in ops if i.startswith("DSA.")], ("dsa",)),
            ("block: MoE (excl)", ("MOE.", "comm.moe"), [i for i in ops if i.startswith("MOE.")], ("moe",)),
            ("block: AttnRes (excl)", ("L.",), ["L.attnres_pre", "L.attnres_post"], ()),
        ]:
            m = sum(excl[o] for o in fam(pref))
            ta = sum(ops[i]["count"] * ops[i]["t_ach_s"] for i in model_ids) * 1e6
            ts = sum(ops[i]["count"] * ops[i]["t_sol_s"] for i in model_ids) * 1e6
            for c in comm:
                if c["layer"] in units and not c["async"]:
                    ta += UNIT_N[c["layer"]] * c["t_ach_s"] * 1e6
                    ts += UNIT_N[c["layer"]] * c["t_sol_s"] * 1e6
            out.append(dict(case=f"{topo} {case}", topology=topo, op_ids=model_ids + [f"comm:{u}" for u in units], row=name,
                            measured_us=round(m, 1), t_sol_us=round(ts, 1), t_ach_us=round(ta, 1),
                            derate=round(m / ta, 3) if ta else None,
                            notes="critical-path exclusive GPU time of the block's kernels (concurrent streams counted once) vs "
                                  "model block = its ops + the unit's synchronous comm rows"))
        mc = sum(meas.get(o, 0) for o in meas if o.startswith("comm"))
        absorbed = set()
        for op, (unit, rx) in FUSED_COMM.items():
            for c in comm:
                if c["layer"] == unit and not c["async"] and re.search(rx, c["prim"] + " " + c["name"]):
                    absorbed.add((c["layer"], c["prim"], c["name"]))
        rest = [c for c in comm if not c["async"] and (c["layer"], c["prim"], c["name"]) not in absorbed]
        t_ach = sum(UNIT_N.get(c["layer"], 1) * c["t_ach_s"] for c in rest) * 1e6
        t_sol = sum(UNIT_N.get(c["layer"], 1) * c["t_sol_s"] for c in rest) * 1e6
        out.append(dict(case=f"{topo} {case}", topology=topo, op_ids=[f"comm:{c['prim']} ({c['layer']} {c['name']})" for c in rest],
                        row="comm (separate kernels)", measured_us=round(mc, 1), t_sol_us=round(t_sol, 1),
                        t_ach_us=round(t_ach, 1), derate=round(mc / t_ach, 3) if t_ach else None,
                        notes="measured = all comm-classified kernels (incl. the per-layer _push_shard AG counted in the blocks) not fused into a GEMM / MegaMoE; model = synchronous comm rows "
                              "not folded into a fused row above"))
        out.append(dict(case=f"{topo} {case}", topology=topo, op_ids=["step"], row="full step",
                        measured_us=float(step_us), t_sol_us=tab["t_sol_ms"] * 1e3, t_ach_us=tab["t_ach_ms"] * 1e3,
                        derate=round(float(step_us) / (tab["t_ach_ms"] * 1e3), 3),
                        notes="vllm bench forward eager median (target + DSpark drafter, rank max) vs model step"))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for r in out:
        print(f"{r['topology']:6s} {r['row'][:44]:44s} meas {r['measured_us'] / 1e3:8.2f} ach {r['t_ach_us'] / 1e3:8.2f} "
              f"sol {r['t_sol_us'] / 1e3:8.2f} derate {r['derate']}")


if __name__ == "__main__":
    main()
