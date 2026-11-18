"""D-GEMM: SplitKTmemGemm64 (weights staged in TMEM before the PDL wait, one-hop DSMEM reduce-scatter) tactic sweep at
the D3-32 SP shapes, vs the current sp_policy tactic. Chained cold weights in a CUDA graph (gemm_bench.timed_chain).
usage: python tmem_sweep.py out.json"""
import json
import sys

import torch

sys.path.insert(0, "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM")
from gemm_bench import DEV, copies, timed_chain  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import PreparedProjection  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_tmem64 import acc_columns  # noqa: E402

CASES = [  # family, N, K, rows
    ("latent_down", 3584, 7168, (32, 64, 96)),
    ("shared_down", 7168, 6144, (32, 96)),
    ("latent_up", 7168, 3584, (32, 96)),
    ("shared_gu", 12288, 7168, (32, 96)),
    ("dsa_in", 5344, 7168, (128,)),
    ("kda_in", 12448, 7168, (128,)),
    ("o_proj", 7168, 3072, (128,)),
    ("wq_b", 4096, 1536, (128,)),
]
SPLITS = (2, 4, 6, 8)


def tile_for(r):
    for t in (64, 80, 96, 128):
        if r <= t:
            return t
    return None


def main():
    res = []
    for fam, n, k, rows in CASES:
        c = copies(n * k * 2)
        ws = [(torch.randn(n, k, device=DEV) * 0.02).bfloat16() for _ in range(c)]
        for r in rows:
            x = torch.randn(r, k, device=DEV).bfloat16()
            out = torch.empty(r, n, device=DEV, dtype=torch.bfloat16)
            ref = (x.float() @ ws[0].float().t())
            nt = tile_for(r)
            for split in SPLITS:
                if (n // 128) * split > 2 * 212 or n % 128:
                    continue
                tac = (128, nt, split, 8, (512 - acc_columns(nt)) // 32)
                try:
                    projs = [PreparedProjection(w, {r: tac}).prepare() for w in ws]
                    projs[0].run(x, out)
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    us = timed_chain([lambda p=p: p.run(x, out) for p in projs])
                    res.append(dict(fam=fam, rows=r, tactic=tac, us=us, tbps=n * k * 2 / us / 1e6, relerr=err))
                    print(f"{fam:12s} r={r:4d} {str(tac):28s} {us:8.2f} us {n*k*2/us/1e6:6.2f} TB/s err {err:.2g}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    res.append(dict(fam=fam, rows=r, tactic=tac, err=repr(e)[:300]))
                    print(f"{fam} r={r} {tac} ERR {e!r}"[:300], flush=True)
            json.dump(res, open(sys.argv[1], "w"), indent=1)
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
