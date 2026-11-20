"""Variant generator: split the chain SIMT term of the ladder (timing probes).

usage: mksub.py IN(lad1.py) OUT WHAT
  WHAT = img : state image + decay pass off (compute windows 0-1, epilogue 2-3)
         res : residual off (KdS ld, V/beta, residual image st)
         vn  : V_new image off (MMA3 result ld, bf16 st)
         out : epilogue output off (t_out ld, output STS)
"""
import sys

src = open(sys.argv[1]).read()
what = sys.argv[3]
src = src.replace("PREP_ONLY = 0\n", "PREP_ONLY = 0\n\n\n" "def _nop(*args, **kwargs):\n    return None\n", 1)
assert "def _nop" in src

R = {
    "img": [
        "cute.copy(mst_ld, mst_thr.partition_S(m0), r_mst)",
        "cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)",
        "cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))",
        "cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))",
        "cute.copy(h_ld, h_thr.partition_S(h0p), r_e1)",
        "cute.copy(h_ld, h_thr.partition_S(h0q), r_e1)",
        "cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)",
        "cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))",
        "cute.copy(h_st, rcur, h_sthr.partition_D(hmst))",
    ],
    "res": [
        "cute.copy(u_wide_ld, uw_ldthr.partition_S(u2f), r_u_wide)",
        "cute.copy(u_wide_st, rb_wide, uw_sthr.partition_D(twri_dv_w))",
        "cute.copy(mst_ld, mst_thr.partition_S(t_u), r_u_old)",
        "cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twri_dv))",
    ],
    "vn": [
        "cute.copy(h16c_ld, h16c_thr.partition_S(t_vst_h), r_u2h)",
        "cute.copy(w8b_st, rb_u2h, w8b_thrc.partition_D(twri_h))",
        "cute.copy(mst_ld, mst_thr.partition_S(t_vst), r_u_old)",
        "cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twri))",
    ],
    "out": [
        "cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)",
        "cute.copy(stm_np, s8, dst)",
    ],
}


def wrapc(text, indent, cond, expect):
    global src
    lines = src.split("\n")
    n = 0
    for i, ln in enumerate(lines):
        if ln == " " * indent + text:
            lines[i] = " " * indent + f"if cutlass.const_expr({cond}):\n" + " " * (indent + 4) + text
            n += 1
    assert n == expect, (text, indent, n)
    src = "\n".join(lines)


if what == "half":
    # half of the image + decay pass every chunk (static; = one pass per 64 tokens on average)
    wrapc("cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)", 24, "False", 1)
    wrapc("cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))", 20, "win == 0", 1)
    wrapc("cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))", 20, "win == 0", 1)
    for ind in (20, 36):
        wrapc("cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)", ind, "hh < 1", 1)
    for ind in (16, 32):
        wrapc("cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))", ind, "hh < 2", 1)
        wrapc("cute.copy(h_st, rcur, h_sthr.partition_D(hmst))", ind, "hh < 2", 1)
    R[what] = []
for s in R[what]:
    n = src.count(s)
    assert n >= 1, s
    src = src.replace(s, "_nop" + s[len("cute.copy"):])
open(sys.argv[2], "w").write(src)
print("ok", what)
