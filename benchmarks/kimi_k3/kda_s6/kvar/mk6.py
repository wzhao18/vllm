"""Variant generator: NST-stage ring with the five four-warp prep instances
decoupled from the stages (instance i takes CTA-stream chunks pos = i mod 5,
stage pos mod NST, per-stage barrier parity (pos // NST) & 1).  The Gram TMEM
window, MB_GRAM and the instance barrier stay per instance.  Packed
(partial-TMA) routes only: every chunk arms GRAW/QKRAW.

usage: mk6.py IN OUT [NST=6]
"""
import re
import sys

src = open(sys.argv[1]).read()
NST = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def sub(old, new, s, count=1):
    n = s.count(old)
    assert n == count, (n, old[:90])
    return s.replace(old, new)


src = sub("STAGES = 5\n", f"STAGES = {NST}\nNINST = 5\n", src)
mb = ["QK", "GRAW", "QKRAW", "VFULL", "VFREE", "SFREE", "RFREE"]
blocks = {}
off = 0
for n in mb:
    blocks[n] = off
    off += NST
blocks["INP"] = off
off += 1
for n in ["OOUT", "TAB", "U2ACC", "OFIN", "FIN"]:
    blocks[n] = off
    off += NST
blocks["OEMPTY"] = off
off += 1
for n in ["BRAW", "GRAM", "GREADY"]:
    blocks[n] = off
    off += NST
for n, v in blocks.items():
    src, k = re.subn(rf"^MB_{n} = \d+$", f"MB_{n} = {v}", src, count=1, flags=re.M)
    assert k == 1, n
src = re.sub(r"MemRange\[cutlass.Int64, \d+\]", f"MemRange[cutlass.Int64, {off}]", src)

L = src.split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("    if warp >= 12 + 52 * DP_PREP_:"))
b = next(i for i, l in enumerate(L) if l.startswith("    if cutlass.const_expr(DP_PREP_ >= 1):"))
prep = "\n".join(L[a:b])
prep = re.sub(r"\binst\b", "stg", prep)
prep = sub("inst = (warp - 12) >> 2" .replace("inst", "stg"), "inst = (warp - 12) >> 2", prep)
prep = sub("p_g64 + stg * (C // 2)", "p_g64 + inst * (C // 2)", prep)
prep = sub("barrier_id=7 + stg", "barrier_id=7 + inst", prep)
prep = sub("MB_GRAM + stg", "MB_GRAM + inst", prep, 2)
prep = sub("li0 = stg - ph0", "li0 = inst - ph0", prep)
# per-chunk stage views
prep = sub("        s_gram_st = gm_st_thr.partition_D(t_sgram[None, None, None, stg])\n", "", prep)
prep = sub("cute.copy(gram_st, r_gram_b, s_gram_st)",
           "cute.copy(gram_st, r_gram_b, gm_st_thr.partition_D(t_sgram[None, None, None, stg]))", prep)
fb = """                f_b = v_braw[None, None, stg]
                b_d, b_s = cpasync.tma_partition(
                    tma_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(f_b, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
"""
prep = sub(fb, "", prep)
use = """                            cute.copy(
                                tma_b,
                                b_s[(None, ci, hidx >> 3)],"""
prep = sub(use, """                            f_b = v_braw[None, None, stg]
                            b_d, b_s = cpasync.tma_partition(
                                tma_b,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_b, 0, 2),
                                cute.group_modes(gB, 0, 2),
                            )
""" + use, prep)
# scheduling over NINST instances, stage from the CTA stream position
prep = sub("        ph0 = cutlass.Int32(0)\n", "        ph0 = cutlass.Int32(0)\n        pbase = cutlass.Int32(0)\n", prep)
prep = sub("                li0 += STAGES\n", "                li0 += NINST\n", prep)
prep = sub("            n_iters = (t_tiles - li0 + STAGES - 1) // STAGES\n            ph0 += t_tiles\n            ph0 = ph0 % STAGES\n",
           "            n_iters = (t_tiles - li0 + NINST - 1) // NINST\n            pb0 = pbase\n            pbase += t_tiles\n            ph0 += t_tiles\n            ph0 = ph0 % NINST\n", prep)
prep = sub("last_ci = li0 + (n_iters - 1) * STAGES", "last_ci = li0 + (n_iters - 1) * NINST", prep)
prep = sub("                    ci = li0 + it * STAGES\n",
           """                    ci = li0 + it * NINST
                    pos = pb0 + ci
                    stg = pos % STAGES
                    kk = (pos // STAGES) & 1
                    pp_rfree = 1 ^ kk
                    pp_sfree = 1 ^ kk
                    pp_graw = kk
                    pp_qkraw = kk
                    pp_braw = kk
""", prep)
prep = sub("                    pp_rfree ^= 1\n                    pp_sfree ^= 1\n", "", prep)
prep = sub("                        pp_braw ^= 1\n", "                        pass\n", prep)
prep = sub("                        pp_graw ^= 1\n                        pp_qkraw ^= 1\n", "                        pass\n", prep)
L = L[:a] + prep.split("\n") + L[b:]
open(sys.argv[2], "w").write("\n".join(L))
print("NST", NST, "mbar slots", off)
