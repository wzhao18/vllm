"""Wave-2 rows (same schema as wave-1 lanes). Measured = vllm bench forward median (rank max / step); model t_ach/t_sol from
out/optables.json (prefill F16384 L78K) or, for exact-shape prefill/decode points, the wave-1 rows' model values.
python3 perf-model/microbench/results/wave2/mkrows.py"""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(H, "../../.."))
sys.path.insert(0, PM); os.chdir(PM)
import decode as Dec  # noqa: E402
import prefill as Pre  # noqa: E402
from topologies import DECODE, PREFILL  # noqa: E402
TD = {t.id: t for t in DECODE}; TPF = {t.id: t for t in PREFILL}
def pexact(tid, f):  # model at the exact measured prefill shape (F16384 per instance, f_req)
    return tuple(Pre.run_point(TPF[tid], "mxfp4_release", c, 16384, Pre.L_REF, f_req=f)["t_ms"] * 1e3 for c in ("sol", "achievable"))
def dexact(r):  # D3 decode model, exact r, k=3, L 135K
    return tuple(round(Dec.run_point(TD["D3"], "mxfp4_release", c, r, 3)["t_ms"] * 1e3, 1) for c in ("sol", "achievable"))
OPT = json.load(open(os.path.join(H, "../../../out/optables.json")))["tables"]
LP = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/results"
LD = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/decode/results"
VEND = "moe_kernel=vendored (vLLM megamoe_rubin)"

def opt(t):
    v = OPT[f"mxfp4_release|{t}|prefill"]; return v["t_sol_ms"] * 1e3, v["t_ach_ms"] * 1e3

rows = []
def pf(topo, case, meas_us, gpus_inst, sol, ach, quant, log, extra=""):
    tok = 16384 / (meas_us * 1e-6 * gpus_inst)
    mtok = 16384 / (ach * 1e-6 * gpus_inst)
    llf = "" if quant == "mxfp8" else " NOT like-for-like (model is mxfp4_release / MXFP8 MoE);"
    rows.append(dict(case=case, topology=topo, op_ids=["step"], row="full step", measured_us=meas_us,
                     t_sol_us=round(sol, 1), t_ach_us=round(ach, 1), derate=round(meas_us / ach, 3),
                     notes=f"{tok:.0f} tok/s/GPU measured vs model achievable {mtok:.0f} (MXFP8 MoE);{llf} MegaMoE quant "
                           f"{'mxfp4_mxfp8 (pinned)' if quant == 'mxfp8' else 'nvfp4 (release policy)'} (asserted in log); "
                           f"{VEND}; raw {log}. {extra}".strip(), moe_kernel="vendored", wave=2))

s, a = opt("P2b")
pf("P2b", "P2b pf16k_L78k MXFP8 [model P2b F16384]", 212809.6, 8, s, a, "mxfp8", f"{LP}/w2p2b/p2b_mx.log",
   "DP2 x (TP8=DCP8) ShardQ (log: ShardQ prefill), EP16; torch profile in p2b_mx.prof")
pf("P2b", "P2b pf16k_L78k release MegaMoE policy (NVFP4) [model P2b F16384]", 207930.0, 8, s, a, "release", f"{LP}/w2p2b/p2b_rel.log")
s, a = opt("P5-4")
pf("P5-4", "P5-4 pf16k_L78k MXFP8 [model P5-4 F16384]", 421591.7, 4, s, a, "mxfp8", f"{LP}/w2p5/p54_mx.log",
   "TP4 x DP4, no DCP (replicated DSA KV), EP16; torch profile in p54_mx.prof")
pf("P5-4", "P5-4 pf16k_L78k release MegaMoE policy (NVFP4) [model P5-4 F16384]", 386280.0, 4, s, a, "release", f"{LP}/w2p5/p54_rel.log")
s, a = opt("P5-8")
pf("P5-8", "P5-8 pf16k_L78k MXFP8 [model P5-8 F16384]", 345103.4, 8, s, a, "mxfp8", f"{LP}/w2p5/p58_mx.log",
   "TP8, no DCP (replicated DSA KV), EP8, k=7 drafter as P1")
pf("P5-8", "P5-8 pf16k_L78k release MegaMoE policy (NVFP4) [model P5-8 F16384]", 321030.2, 8, s, a, "release", f"{LP}/w2p5/p58_rel.log")
# P4 release arms: wave-1 exact-shape model (f_req 4096) from prefill/rows.json
pf("P4-4", "P4-4 pf16k_L78k(4x4096) release MegaMoE policy (NVFP4) [model P4-4 F16384 f4096]", 285614.8, 4,
   *pexact("P4-4", 4096), "release", f"{LP}/w2p4rel/p4_4_rel.log", "PR #20 DSA-DP tree (log: DSA-DP outputs pushed)")
pf("P4-2", "P4-2 pf16k_L78k(4x4096) release MegaMoE policy (NVFP4) [model P4-2 F16384 f4096]", 536312.0, 2,
   *pexact("P4-2", 4096), "release", f"{LP}/w2p4rel/p4_2_rel.log", "PR #20 DSA-DP tree")

D3M = {r: dexact(r) for r in (1, 4, 8, 16, 24)}  # current model, D3 EP16 exact r
def dec(topo, tag, B, meas, ach, log, notes):
    sol = D3M[B // 4][0] if ach is not None else None
    rows.append(dict(case=f"{tag}:q4_B{B}_M{4*B}_u132k", topology=topo, k=3, B_per_dp_rank=B, r_per_gpu=B / 4,
                     op_ids=["step (target verify 93 layers + DSpark k-step propose; bench scope)"], measured_us=meas,
                     t_sol_us=sol, t_ach_us=ach, derate=None if ach is None else round(meas / ach, 3),
                     ctx_mean=135168, ctx_scale=1.0, profiled=False, notes=f"{notes}; {VEND}; log {log}",
                     kind="step", moe_kernel="vendored", wave=2))
for B, m, tag in [(4, 19274.7, "d332lo"), (16, 20888.3, "d332lo"), (32, 20918.1, "d332lo"), (64, 26984.3, "d332b64"),
                  (96, 30482.9, "d332b96")]:
    dec("D3-32", tag, B, m, None, f"{LD}/w2d332/{tag}.log",
        "DP8 x (TP4=DCP4) SP, EP32 (moe-backend deep_gemm_mega_moe as wave-1 D3); MXFP8 MoE act; t_ach null: the model "
        "has no D3-32 topology (EP<=16 scope)")
for B, m, tag in [(4, 21194.5, "d3baselo"), (16, 24491.9, "d3baselo"), (32, 24366.7, "d3baselo"), (64, 30365.4, "d3baseb64")]:
    dec("D3", tag, B, m, D3M[B // 4][1], f"{LD}/w2d3base/{tag}.log",
        "same-session control: exact wave-1 D3 EP16 config (point c512); model exact")
for B, m, tag in [(4, 19773.7, "d3rello"), (16, 22178.3, "d3rello"), (32, 21756.8, "d3rello"), (64, 27656.8, "d3relb64"),
                  (96, 30194.0, "d3relb96")]:
    dec("D3", tag, B, m, D3M[B // 4][1], f"{LD}/w2d3rel/{tag}.log",
        "release-MoE arm: moe-backend kimi_k3_mega_moe (release C512 decode role), MegaMoE NVFP4 W4A4 (log: 92 layers in "
        "the nvfp4 kernel layout); NOT like-for-like with the mxfp4_release model")
json.dump(rows, open(os.path.join(H, "rows.json"), "w"), indent=1)
print(len(rows), "rows")
for r in rows: print(r["topology"], r["case"][:70], r["measured_us"], r["t_ach_us"], r["derate"])
