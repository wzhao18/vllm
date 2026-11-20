"""Timing-only ablation generator (P-KDA2-S6 r9): drop the prep q/k L2 norm (sum of squares, 4-level butterfly,
2 rsqrt per row slice) in phase 3 of an n4-lineage kernel (st7 / pf7 / m1). Numerics wrong by design; equals
P-KDA2's PREP_ABL_ bit 1 on the b kernel. Used to test whether a fixed prep instruction cut transfers more on the
BT64 pair kernel (prep-throughput bound per E1) than on the BT32 kernel.

usage: mkabl.py IN OUT
"""
import sys

src = open(sys.argv[1]).read()
old = """                        sq = cutlass.Float32(0.0)
                        sk = cutlass.Float32(0.0)
                        for u in cutlass.range_constexpr(8):
                            fq = cutlass.Float32(qr[u])
                            fk = cutlass.Float32(kr[u])
                            sq += fq * fq
                            sk += fk * fk
                        for off in [1, 2, 4, 8]:
                            sq += cute.arch.shuffle_sync_bfly(sq, off)
                            sk += cute.arch.shuffle_sync_bfly(sk, off)
                        # scale folded into rqn: one FMUL replaces 8 per slice
                        rqn = cute.math.rsqrt(sq + 1e-6, fastmath=True) * scale
                        rkn = cute.math.rsqrt(sk + 1e-6, fastmath=True)
                        g8s = v_gcs8[(rw2, None, sg2, stg)]"""
new = """                        rqn = scale
                        rkn = cutlass.Float32(1.0)
                        g8s = v_gcs8[(rw2, None, sg2, stg)]"""
assert src.count(old) == 1, src.count(old)
open(sys.argv[2], "w").write(src.replace(old, new))
print("mkabl ok", sys.argv[2])
