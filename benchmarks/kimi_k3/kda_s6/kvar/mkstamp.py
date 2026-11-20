"""Add a prep phase-stamp probe to a kernel variant (PREP_ONLY variants).

usage: mkstamp.py SRC DST
Stamps (clock64) by prep warp 0 of every instance, chain piece kx == 0, local
iteration it == 4, into tprobe[slot * 64 + inst * 12 + k] (needs the forward
module's _TPROBE = 1 so the buffer is real):
 0 iteration start      1 after MB_RFREE      2 after MB_SFREE + ibar
 3 after gate ibar       4 after decoration ibar 5 after MB_GRAM
 6 after Gram store ibar 7 warp 0 solve done     8 after final ibar"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
HELPER = '''
@dsl_user_op
def _clk64(*, loc=None, ip=None):
    v = llvm.inline_asm(
        T.i64(), [], "mov.u64 $0, %clock64;", "=l", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(v)


@cute.struct'''


def stamp(ind, k):
    p = " " * ind
    return (f"{p}if (kx == 0) & (it == 4) & (plw == 0):\n"
            f"{p}    with cute.arch.elect_one():\n"
            f"{p}        tprobe[slot * 64 + inst * 12 + {k}] = _clk64()\n")


sites = [
    # (anchor text, insert after anchor?, indent, k)
    ("                    it = phase_base + phase_it\n", True, 20, 0),
    ("                            cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)\n",
     True, 28, 1),
    ("                        cute.arch.mbarrier_wait(mb + MB_SFREE + inst, pp_sfree)\n"
     "                    ibar.arrive_and_wait()\n", True, 20, 2),
    ("                        ibar.arrive_and_wait()\n"
     "                    # -- phase 3: q/k load", "mid3", 20, 3),
    ("                        v_rf[D, inst] = _exp2f(anch)\n"
     "                    ibar.arrive_and_wait()\n", True, 20, 4),
    ("                    cute.arch.mbarrier_wait(mb + MB_GRAM + inst, pp_gram)\n", True, 20, 5),
    ("                    cute.copy(gram_st, r_gram_b, s_gram_st)\n"
     "                    ibar.arrive_and_wait()\n", True, 20, 6),
    ("                    # -- phase 7: off-diag combine (warp 0) || restore (warps 1-3) --\n",
     False, 20, 7),
    ("                    cute.arch.fence_proxy(\"async.shared\", space=\"cta\")\n"
     "                    ibar.arrive_and_wait()\n"
     "                    if plw == 0:\n"
     "                        with cute.arch.elect_one():\n"
     "                            cute.arch.mbarrier_arrive(mb + MB_QK + inst)\n", False, 20, 8),
]
for anchor, after, ind, k in sites:
    assert s.count(anchor) == 1, (k, anchor[:70])
    if after == "mid3":
        head = "                        ibar.arrive_and_wait()\n"
        s = s.replace(anchor, head + stamp(ind, k) + anchor[len(head):])
    elif after:
        s = s.replace(anchor, anchor + stamp(ind, k))
    else:
        # stamp 8 sits after the final ibar, before the QK arrive
        if k == 8:
            cut = anchor.index("                    if plw == 0:\n")
            s = s.replace(anchor, anchor[:cut] + stamp(ind, k) + anchor[cut:])
        else:
            s = s.replace(anchor, stamp(ind, k) + anchor)
s = s.replace("\n\n@cute.struct", "\n" + HELPER, 1)
open(dst, "w").write(s)
print(dst, "stamped")
