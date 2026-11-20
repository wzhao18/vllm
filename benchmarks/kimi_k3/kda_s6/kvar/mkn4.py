"""Transform kernel.py: 4 prep instances over 5 stages + a raw-loader warp (28).

usage: mkn4.py IN OUT
Instances no longer own a stage: instance i takes CTA-global chunks G = i mod 4,
using stage G mod 5 (parity (G // 5) & 1).  Warp 28 waits RFREE/SFREE of each
stage and issues the raw beta/g/k/q TMA loads as soon as the stage frees, so
raw-load latency overlaps the instance's previous item instead of following
an idle wait for its own stage.
"""
import re
import sys

src = open(sys.argv[1]).read()


def sub1(old, new, s, count=1):
    n = s.count(old)
    assert n == count, (n, old[:80])
    return s.replace(old, new)


# ---- module constant
src = sub1("STAGES = 5\n", "STAGES = 5\n# prep instances (4 warps each, warps 12..11+4*NINST); warp 12+4*NINST loads\nNINST = 4\n", src)

lines = src.split("\n")
# locate prep section
i_prep = next(i for i, l in enumerate(lines) if l.startswith("    if warp >= 12:"))
i_end = next(i for i in range(i_prep, len(lines)) if lines[i].startswith("    cute.arch.sync_threads()"))
prep = "\n".join(lines[i_prep:i_end])
head = "\n".join(lines[:i_prep])
tail = "\n".join(lines[i_end:])

# ---- grab the raw-load code blocks (to move into the loader) before editing
beta_blk = """                    if cutlass.const_expr(BETA_TMA_ == 1):
                        if plw == 2:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mb + MB_BRAW + inst, C * 8 * 2
                                )
                            cute.copy(
                                tma_b,
                                b_s[(None, ci, hidx >> 3)],
                                b_d,
                                tma_bar_ptr=mb + MB_BRAW + inst,
                            )
"""
prep = sub1(beta_blk, "", prep)

# g/k TMA block: from "if tma:\n  if plw == 0:\n  wait RFREE" up to the k copy end
i0 = prep.index("                    if tma:\n                        if plw == 0:\n                            cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)")
i1 = prep.index("tma_bar_ptr=mb + MB_QKRAW + inst,\n                            )\n", i0) + len("tma_bar_ptr=mb + MB_QKRAW + inst,\n                            )\n")
gk_blk = prep[i0:i1]
prep = prep[:i0] + "#@GK@\n" + prep[i1:]

# q TMA block after SFREE
j0 = prep.index("                    if tma:\n                        if plw == 0:\n                            f_qr = cute.make_tensor(")
j1 = prep.index("tma_bar_ptr=mb + MB_QKRAW + inst,\n                            )\n", j0) + len("tma_bar_ptr=mb + MB_QKRAW + inst,\n                            )\n")
q_blk = prep[j0:j1]
prep = prep[:j0] + "#@Q@\n" + prep[j1:]

# beta partition at chain start (per stage now: the loader does it)
bpart = """                f_b = v_braw[None, None, inst]
                b_d, b_s = cpasync.tma_partition(
                    tma_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(f_b, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
"""
prep = sub1(bpart, "", prep)

# ---- prep block condition (instances only)
prep = sub1("    if warp >= 12:\n", "    if (warp >= 12) & (warp < 12 + 4 * NINST):\n", prep)

# ---- chunk assignment
prep = sub1("        ph0 = cutlass.Int32(0)\n", "        gbase = cutlass.Int32(0)\n", prep)
old_as = """            li0 = inst - ph0
            if li0 < 0:
                li0 += STAGES
            n_iters = (t_tiles - li0 + STAGES - 1) // STAGES
            ph0 += t_tiles
            ph0 = ph0 % STAGES
"""
new_as = """            li0 = inst - gbase % NINST
            if li0 < 0:
                li0 += NINST
            n_iters = (t_tiles - li0 + NINST - 1) // NINST
            gbase_c = gbase
            gbase += t_tiles
"""
prep = sub1(old_as, new_as, prep)
prep = sub1("                last_ci = li0 + (n_iters - 1) * STAGES\n", "                last_ci = li0 + (n_iters - 1) * NINST\n", prep)
old_ci = "                    ci = li0 + it * STAGES\n"
new_ci = """                    ci = li0 + it * NINST
                    gidx = gbase_c + ci
                    stg = gidx % STAGES
                    par = (gidx // STAGES) & 1
"""
prep = sub1(old_ci, new_ci, prep)

# ---- per-item: stage resources follow stg; Gram window / GRAM barrier / ibar stay per instance
out = []
in_item = False
for l in prep.split("\n"):
    if "par = (gidx // STAGES) & 1" in l:
        in_item = True
        out.append(l)
        continue
    if in_item and l.strip().startswith("#") is False and "inst" in l and "MB_GRAM" not in l:
        l = re.sub(r"\binst\b", "stg", l)
    out.append(l)
prep = "\n".join(out)

# Gram image store view per stage (was built once per instance)
prep = sub1("        s_gram_st = gm_st_thr.partition_D(t_sgram[None, None, None, inst])\n", "", prep)
prep = sub1("                    cute.copy(gram_st, r_gram_b, s_gram_st)\n",
            "                    s_gram_st = gm_st_thr.partition_D(t_sgram[None, None, None, stg])\n"
            "                    cute.copy(gram_st, r_gram_b, s_gram_st)\n", prep)

# ---- parities: stage barriers from the global chunk index
prep = re.sub(r"mb \+ MB_SFREE \+ stg, pp_sfree\)", "mb + MB_SFREE + stg, par ^ 1)", prep)
prep = re.sub(r"mb \+ MB_GRAW \+ stg, pp_graw\)", "mb + MB_GRAW + stg, par)", prep)
prep = re.sub(r"mb \+ MB_QKRAW \+ stg, pp_qkraw\)", "mb + MB_QKRAW + stg, par)", prep)
prep = re.sub(r"mb \+ MB_BRAW \+ stg, pp_braw\)", "mb + MB_BRAW + stg, par)", prep)
for v in ("pp_rfree", "pp_sfree", "pp_braw", "pp_graw", "pp_qkraw"):
    assert v + ")" not in prep, v
old_tog = """                    pp_rfree ^= 1
                    pp_sfree ^= 1
                    pp_gram ^= 1
                    if cutlass.const_expr(BETA_TMA_ == 1):
                        pp_braw ^= 1
"""
prep = sub1(old_tog, "                    pp_gram ^= 1\n", prep)
i_t = prep.index("                    # GRAW/QKRAW are armed+waited only on full chunks")
i_t2 = prep.index("                        pp_qkraw ^= 1\n", i_t) + len("                        pp_qkraw ^= 1\n")
prep = prep[:i_t] + prep[i_t2:]

# ---- loader warp: same chain walk, every chunk, stage = global chunk mod 5
def reindent(blk, d):
    return "\n".join((" " * d + l[16:]) if l.strip() else l for l in blk.split("\n"))

gk = gk_blk.replace("inst", "stg").replace("pp_rfree", "par ^ 1")
gk = gk.replace("                    if tma:\n                        if plw == 0:\n", "")
gk = "\n".join(l[4:] if l.startswith(" " * 4) else l for l in gk.split("\n") if "MB_RFREE" not in l)
q = q_blk.replace("inst", "stg")
q = q.replace("                    if tma:\n                        if plw == 0:\n", "")
q = "\n".join(l[4:] if l.startswith(" " * 4) else l for l in q.split("\n"))

loader = f'''    # =====================================================================
    # RAW LOADERS (warps 12 + 4 * NINST + j): beta/g/k at RFREE, q at SFREE.
    # Loader j serves the chunks of instance j (one issuing thread per chunk
    # stream: a TMA issuer completes one barrier stage at a time).
    # =====================================================================
    if warp >= 12 + 4 * NINST:
        cute.arch.warpgroup_reg_dealloc(24)
        jl = warp - (12 + 4 * NINST)
        if jl < NINST:
            gl = cutlass.Int32(0)
            for kx in cutlass.range(k1 - k0):
                chain = cute.arch.make_warp_uniform(schain[k0 + kx])
                seq_idx = chain // H_
                hidx = chain % H_
                bos = cutlass.Int32(cu[seq_idx]) + cute.arch.make_warp_uniform(
                    spt0[k0 + kx]
                )
                seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                t_tiles = (seq_len + C - 1) // C
                gQ = cute.flat_divide(cute.domain_offset((bos, 0, 0), mQ), (C, 1, D))
                gK = cute.flat_divide(cute.domain_offset((bos, 0, 0), mK), (C, 1, D))
                gG = cute.flat_divide(cute.domain_offset((bos, 0, 0), mG), (C, 1, D))
                if cutlass.const_expr(BETA_TMA_ == 1):
                    gB = cute.flat_divide(cute.domain_offset((bos, 0), mB), (C, 8))
                l0 = jl - gl % NINST
                if l0 < 0:
                    l0 += NINST
                for it in cutlass.range((t_tiles - l0 + NINST - 1) // NINST):
                    ci = l0 + it * NINST
                    gidx = gl + ci
                    stg = gidx % STAGES
                    par = (gidx // STAGES) & 1
                    if cutlass.const_expr(PARTIAL_TMA_ == 1 or FULL_CHUNKS_ == 1):
                        tma = True
                    else:
                        tma = seq_len >= (ci + 1) * C
                    cute.arch.mbarrier_wait(mb + MB_RFREE + stg, par ^ 1)
                    if cutlass.const_expr(BETA_TMA_ == 1):
                        f_b = v_braw[None, None, stg]
                        b_d, b_s = cpasync.tma_partition(
                            tma_b,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(f_b, 0, 2),
                            cute.group_modes(gB, 0, 2),
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mb + MB_BRAW + stg, C * 8 * 2
                            )
                        cute.copy(
                            tma_b,
                            b_s[(None, ci, hidx >> 3)],
                            b_d,
                            tma_bar_ptr=mb + MB_BRAW + stg,
                        )
                    if tma:
{gk}
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_GRAW + stg)
                    cute.arch.mbarrier_wait(mb + MB_SFREE + stg, par ^ 1)
                    if tma:
{q}
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_QKRAW + stg)
                gl += t_tiles
'''
mode = sys.argv[3] if len(sys.argv) > 3 else "loader"
poll_only = None
if mode == "selfpoll":
    # probe: self-issued N4 plus loader warps that only poll RFREE/SFREE
    w_r = "                    cute.arch.mbarrier_wait(mb + MB_RFREE + stg, par ^ 1)\n"
    w_s = "                    cute.arch.mbarrier_wait(mb + MB_SFREE + stg, par ^ 1)\n"
    poll_only = loader.split(w_r, 1)[0] + w_r + w_s + "                gl += t_tiles\n"
    mode = "self"
if mode == "self":
    # instances issue their own raw loads (warp 0, after RFREE / SFREE of stg)
    body = loader.split("                    cute.arch.mbarrier_wait(mb + MB_RFREE + stg, par ^ 1)\n", 1)[1]
    part_a, part_b = body.split("                    cute.arch.mbarrier_wait(mb + MB_SFREE + stg, par ^ 1)\n", 1)
    part_b = part_b.split("                gl += t_tiles\n", 1)[0]
    part_a = "                    cute.arch.mbarrier_wait(mb + MB_RFREE + stg, par ^ 1)\n" + part_a
    ind = lambda b: "".join("    " + l if l.strip() else l for l in b.splitlines(True))
    prep = prep.replace("#@GK@\n", "                    if plw == 0:\n" + ind(part_a))
    prep = prep.replace("#@Q@\n", "                    if plw == 0:\n" + ind(part_b))
    loader = poll_only or ("    if warp >= 12 + 4 * NINST:\n"
                           "        cute.arch.warpgroup_reg_dealloc(24)\n")
else:
    prep = prep.replace("#@GK@\n", "").replace("#@Q@\n", "")
src = head + "\n" + loader + prep + "\n" + tail
open(sys.argv[2], "w").write(src)
print("ok", len(src))
