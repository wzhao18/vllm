"""Variant generator: chain delta ladder (timing probes, numerically wrong).

usage: mklad.py IN OUT LEVEL
  LEVEL 1: prep instances run only their barrier protocol (RFREE/SFREE wait,
           instance barrier, QK arrive): no raw loads, no gate/decoration/
           Gram/solve.  The recurrence chain runs alone (V loads kept).
  LEVEL 2: + chain SIMT off: every TMEM ld/st of compute and epilogue (state
           image + decay, residual, V_new image, output) and the output STS
           are skipped; all mbarrier waits/arrives, named barriers, fences and
           commits stay.
  LEVEL 3: + no chain MMAs (the MMA warp still commits every stage).
  LEVEL 4: + no V TMA (plain VFULL arrive) and no output TMA store.
"""
import sys

src = open(sys.argv[1]).read()
lev = int(sys.argv[3])


def sub(old, new, count=None):
    global src
    n = src.count(old)
    assert n > 0, old
    if count is not None:
        assert n == count, (old, n)
    src = src.replace(old, new)


# module-level switches
sub(
    "PREP_ONLY = 0\n",
    "PREP_ONLY = 0\n"
    f"LAD = {lev}\n\n\n"
    "def _skip(*args, **kwargs):\n"
    "    return None\n\n\n"
    "_c2 = _skip if LAD >= 2 else cute.copy\n"
    "_g3 = _skip if LAD >= 3 else cute.gemm\n"
    "_c4 = _skip if LAD >= 4 else cute.copy\n",
    1,
)

# ---- level 1: protocol-only prep ----
head = "    if warp >= 12:\n        cute.arch.warpgroup_reg_dealloc(48)\n"
tail = "\n    cute.arch.sync_threads()\n    tmem.free(tmem_ptr)\n"
i = src.index(head) + len(head)
j = src.index(tail, i)
body = src[i:j]
lines = body.split("\n")
ind = "\n".join(("    " + ln) if ln.strip() else ln for ln in lines)
null = """        if cutlass.const_expr(LAD >= 1):
            inst = (warp - 12) >> 2
            plw = (warp - 12) & 3
            pp_rfree = cutlass.Int32(1)
            pp_sfree = cutlass.Int32(1)
            ibar = pipeline.NamedBarrier(barrier_id=7 + inst, num_threads=128)
            ph0 = cutlass.Int32(0)
            for kx in cutlass.range(k1 - k0):
                seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                t_tiles = (seq_len + C - 1) // C
                li0 = inst - ph0
                if li0 < 0:
                    li0 += STAGES
                n_iters = (t_tiles - li0 + STAGES - 1) // STAGES
                ph0 += t_tiles
                ph0 = ph0 % STAGES
                for _it in cutlass.range(n_iters):
                    if plw == 0:
                        cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)
                        cute.arch.mbarrier_wait(mb + MB_SFREE + inst, pp_sfree)
                    ibar.arrive_and_wait()
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_QK + inst)
                    pp_rfree ^= 1
                    pp_sfree ^= 1
        else:
"""
if lev >= 1:
    src = src[:i] + null + ind + src[j:]

# ---- level 2: chain SIMT off (compute + epilogue TMEM traffic, output STS) ----
if lev >= 2:
    for atom in ("mst_ld", "mst_st", "w16b_st", "u_wide_ld", "u_wide_st",
                 "h16c_ld", "w8b_st", "h_ld", "h_st", "o16_cp", "stm_np"):
        sub(f"cute.copy({atom},", f"_c2({atom},")

# ---- level 3: no chain MMAs ----
if lev >= 3:
    import re

    src, n3 = re.subn(r"cute\.gemm\((\s*)(mma_1|mma_3|mma_4a|mma_4b),", r"_g3(\1\2,", src)
    assert n3 >= 7, n3

# ---- level 4: no V TMA, no output TMA store ----
if lev >= 4:
    sub(
        "cute.arch.mbarrier_arrive_and_expect_tx(\n"
        "                            mb + MB_VFULL + csl, C * D * 2\n"
        "                        )",
        "cute.arch.mbarrier_arrive(mb + MB_VFULL + csl)",
        1,
    )
    sub("cute.copy(\n                            tma_v,", "_c4(\n                            tma_v,")
    sub("cute.copy(tma_o,", "_c4(tma_o,")

open(sys.argv[2], "w").write(src)
print("ok", lev)
