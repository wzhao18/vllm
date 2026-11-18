"""D-GEMM: SplitKGemmG (split-K with the K reduction through L2, no smem mailbox) correctness + tactic sweep at the
D3-32 SP shapes. usage: python g_sweep.py out.json [quick]"""
import json
import sys

import torch

sys.path.insert(0, "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM")
from gemm_bench import DEV, copies, timed_chain  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_g import (  # noqa: E402
    G_N_TILES, PreparedG, check_tactic)

CASES = [  # family, N, K, rows
    ("latent_down", 3584, 7168, (4, 32, 96)),
    ("latent_up", 7168, 3584, (32, 96)),
    ("router", 896, 7168, (32, 96)),
    ("shared_down", 7168, 6144, (32, 96)),
    ("shared_gu", 12288, 7168, (32, 96)),
    ("dsa_o", 7168, 3072, (128,)),
    ("wq_b", 4096, 1536, (128,)),
    ("q_b", 4608, 1536, (128,)),
    ("dsa_in", 5344, 7168, (128,)),
    ("kda_in", 12448, 7168, (128,)),
]
SMEM = 220 * 1024


def tactics(n, k, r):
    nt = next(t for t in G_N_TILES if t >= r)
    stage = 128 * 128 * 2 + nt * 128 * 2
    stages = max(2, min(8, SMEM // stage))
    out = []
    for split in (1, 2, 3, 4, 6, 7, 8):
        t = (128, nt, split, stages)
        ctas = -(-n // 128) * split
        if check_tactic(t, k) is None and ctas <= 2 * 212 and (ctas >= 60 or split == 8):
            out.append(t)
    return out


def main():
    res = []
    quick = len(sys.argv) > 2
    for fam, n, k, rows in CASES:
        c = copies(n * k * 2)
        ws = [(torch.randn(n, k, device=DEV) * 0.02).bfloat16() for _ in range(c)]
        for r in rows:
            x = torch.randn(r, k, device=DEV).bfloat16()
            out = torch.empty(r, n, device=DEV, dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            for tac in tactics(n, k, r):
                try:
                    projs = [PreparedG(w, {r: tac}) for w in (ws[:1] if quick else ws)]
                    out.fill_(float("nan"))
                    projs[0].run(x, out)
                    torch.cuda.synchronize()
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    ok = bool(torch.isfinite(out).all()) and err < 1e-2
                    us = None if quick else timed_chain([lambda p=p: p.run(x, out) for p in projs])
                    res.append(dict(fam=fam, rows=r, tactic=tac, us=us, relerr=err, ok=ok,
                                    tbps=(n * k * 2 / us / 1e6) if us else None))
                    print(f"{fam:12s} r={r:4d} {str(tac):22s} ok={ok} err {err:.2g} "
                          + (f"{us:8.2f} us {n*k*2/us/1e6:6.2f} TB/s" if us else ""), flush=True)
                except Exception as e:  # noqa: BLE001
                    res.append(dict(fam=fam, rows=r, tactic=tac, err=repr(e)[:400]))
                    print(f"{fam} r={r} {tac} ERR {e!r}"[:600], flush=True)
                    if quick:
                        import traceback
                        traceback.print_exc()
                        json.dump(res, open(sys.argv[1], "w"), indent=1)
                        return
            json.dump(res, open(sys.argv[1], "w"), indent=1)
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
