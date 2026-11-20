"""Variant generator: pair-step chain EMULATION (timing probe, numerics wrong).

One state image pass per two BT32 chunks (BT64 image) and one K64 state update
per pair; the odd chunk's KdS comes from a 2-MMA correction instead of the 8-MMA
image product.  Emulated on the BT32 loop:
  - compute / epilogue skip the image + decay pass for odd chunks,
  - MMA1a issues 2 of its 8 K-block MMAs on odd chunks,
  - MMA4a is skipped on even chunks and issued twice (K64) on odd chunks.
The odd chunk keeps its FIN + state_inp_bar hops (2 extra hops vs the real
pair step, so the emulation is conservative).

usage: mkpair.py IN OUT
"""
import re
import sys

src = open(sys.argv[1]).read()
lines = src.split("\n")


def wrap(text, indent, cond, expect):
    n = 0
    for i, ln in enumerate(lines):
        if ln == " " * indent + text:
            lines[i] = " " * indent + f"if {cond}:\n" + " " * (indent + 4) + text
            n += 1
    assert n == expect, (text, indent, n)


ce = "(t & 1) == 0"
wrap("cute.copy(mst_ld, mst_thr.partition_S(m0), r_mst)", 16, ce, 1)
wrap("cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)", 24, ce, 1)
wrap("cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))", 20, ce, 1)
wrap("cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))", 20, ce, 1)
co = "(t & 1) == 1"
wrap("cute.copy(h_ld, h_thr.partition_S(h0q), r_e1)", 28, co, 1)
wrap("cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)", 36, co, 1)
wrap("cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))", 32, co, 1)
wrap("cute.copy(h_st, rcur, h_sthr.partition_D(hmst))", 32, co, 1)
src = "\n".join(lines)

# MMA1a: 2 of 8 K blocks on odd chunks
old = """                            cute.gemm(
                                mma_1,
                                t_u,
                                t_inp[None, None, kb],
                                b_kd[None, None, kb, csm],
                                t_u,
                            )
"""
new = """                            if cutlass.const_expr(kb < 2):
                                cute.gemm(
                                    mma_1,
                                    t_u,
                                    t_inp[None, None, kb],
                                    b_kd[None, None, kb, csm],
                                    t_u,
                                )
                            else:
                                if (_t & 1) == 0:
                                    cute.gemm(
                                        mma_1,
                                        t_u,
                                        t_inp[None, None, kb],
                                        b_kd[None, None, kb, csm],
                                        t_u,
                                    )
"""
assert src.count(old) == 1
src = src.replace(old, new)

# MMA4a: skip on even chunks, twice (K64) on odd chunks
pat = re.compile(r"( +)cute\.gemm\(\n\1    mma_4a,\n(?:\1    .*\n)+?\1\)\n")
n4 = 0


def rep(m):
    global n4
    n4 += 1
    ind = m.group(1)
    blk = m.group(0)
    blk2 = "".join("    " + ln + "\n" for ln in blk.rstrip("\n").split("\n"))
    return f"{ind}if (_t & 1) == 1:\n" + blk2 + blk2


src = pat.sub(rep, src)
assert n4 == 3, n4
open(sys.argv[2], "w").write(src)
print("ok pair")
