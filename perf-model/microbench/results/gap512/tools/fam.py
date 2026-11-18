"""fam.py <parsed.json>: per-case family table (busy / exposed / ach / derate / exposed-ach)."""
import json
import sys

FAM = {
    "BF16 dense GEMMs (cuBLAS nvjet + CuTe split-K, +split-K reduce)": [
        "KDA.in_proj", "KDA.o_proj", "DSA.qkv_a", "DSA.g_proj", "DSA.o_proj", "DSA.w_uv", "MOE.router+latent_down",
        "MOE.shared_gate_up", "MOE.shared_down", "MOE.latent_up", "DENSE"],
    "MegaMoE (dispatch+fc1+act+fc2+combine, P11/P12)": ["MOE.megamoe", "MOE.prepare"],
    "DSA DCP exact top-k (pp hist/window/win + DCP merge P8)": ["DSA.topk", "DSA.topk_merge"],
    "KDA core (KdaWide: conv+gate+recurrence+o_norm)": ["KDA.core"],
    "DSA q path + sparse MLA + DCP value (P6/P7b)": ["DSA.q_path", "DSA.sparse_mla", "DSA.dcp_value", "DSA.prologue"],
    "DSA indexer logits (paged_mqa_logits)": ["DSA.idx_score"],
    "small elementwise (AttnRes mix, SiLU, latent RMSNorm, router top-k)": [
        "ATTNRES", "MOE.shared_act", "MOE.latent_norm", "MOE.topk", "EMBED", "OTHER"],
    "SP comm (AllGather / ReduceScatter push)": ["COMM.SP_AG", "COMM.SP_RS"],
    "drafter + final (DSpark k=3 propose)": ["TAIL.drafter+final"],
    "spin wait (QueryWait)": ["WAIT"],
}


def fam_table(c):
    by = {r["group"]: r for r in c["rows"]}
    out = []
    for f, gs in FAM.items():
        b = sum(by[g]["measured_us"] for g in gs if g in by)
        e = sum(by[g]["exposed_us"] for g in gs if g in by)
        a = sum(by[g]["t_ach_us"] for g in gs if g in by)
        s = sum(by[g]["t_sol_us"] for g in gs if g in by)
        out.append(dict(family=f, groups=gs, busy_us=b, exposed_us=e, t_ach_us=a, t_sol_us=s,
                        derate=b / a if a else None, exposed_gap_us=e - a))
    out.sort(key=lambda x: -x["exposed_gap_us"])
    return out


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    for case, c in d.items():
        ach = c["model_t_ach_us"] - c["model_not_in_bench_ach_us"]
        print("== %s step %.0f us vs model ach %.0f (x%.2f), sol %.0f; gap %.0f us" % (
            case, c["step_median_us"], ach, c["step_median_us"] / ach, c["model_t_sol_us"], c["step_median_us"] - ach))
        for r in fam_table(c):
            print("  %-70s busy %7.0f expo %7.0f ach %7.0f  x%-6s gap %6.0f" % (
                r["family"], r["busy_us"], r["exposed_us"], r["t_ach_us"],
                "%.2f" % r["derate"] if r["derate"] else "-", r["exposed_gap_us"]))
