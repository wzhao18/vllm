"""Variant generator: BT64 recurrence chain over BT32 stage pairs (timing
prototype on the L1 chain-only kernel; numerics wrong by design: the cross
blocks of the 64-token inverse / intra-output matrix use stand-in operands).

Per 64-token step (stages A = chunk 2p, B = chunk 2p+1, any slot pair):
  one state image + decay pass (compute dk 0-63, epilogue dk 64-127),
  MMA1a  X_h = img @ Kd_h       (2 x 8 N32, commit OOUT per half)
  MMA1b  O_h = img @ Qd_h       (2 x 8 N32, commit RFREE A,B)
  residual for both halves      (compute, BF16 images at TMEM 384/400)
  MMA3   V_new = r64 @ inv64    (lower block-triangular, 3 blocks = 6 MMAs)
  V_new BF16 images             (TMEM 416/432)
  MMA4a  S += V_new @ Ki64      (K64, 4 MMAs, commit FIN A,B)
  MMA4b  O += V_new @ M64       (3 blocks = 6 MMAs, commit OFIN/SFREE A,B)
Chunk counts are rounded up to even in every role (extra chunk is padding).
TMEM: img 0-63, state 64-191, O 192/256, X 224/288, V_new f32 320/352.

usage: mkp64.py IN(lad1.py) OUT
"""
import sys

src = open(sys.argv[1]).read()
pass

n = src.count("t_tiles = (seq_len + C - 1) // C")
assert n >= 5, n
src = src.replace(
    "t_tiles = (seq_len + C - 1) // C", "t_tiles = ((seq_len + 2 * C - 1) // (2 * C)) * 2"
)
lines = src.split("\n")


def find(text, start=0, indent=None):
    for i in range(start, len(lines)):
        if lines[i] == text:
            return i
    raise AssertionError(text)


SPLIT_END = "            if cutlass.const_expr((SPLIT_ == 1) or (FINAL_ == 1)):"

# ---------------- compute ----------------
COMPUTE = '''            for tp in cutlass.range(t_tiles // 2):  # noqa: B007
                csb = csc + 1
                phb = p_qk
                if csb == STAGES:
                    csb = 0
                    phb = p_qk ^ 1
                cute.arch.mbarrier_wait(mb + MB_QK + csc, p_qk)
                cute.arch.mbarrier_wait(mb + MB_QK + csb, phb)
                m0 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32),
                    mst_base.layout,
                )
                cute.copy(mst_ld, mst_thr.partition_S(m0), r_mst)
                for win in cutlass.range_constexpr(2):
                    rcur = r_mst if win % 2 == 0 else r_mst2
                    rnxt = r_mst2 if win % 2 == 0 else r_mst
                    msth = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 64 + win * 32, dtype=cutlass.Float32
                        ),
                        mst_base.layout,
                    )
                    twin = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + win * 16, dtype=cutlass.BFloat16),
                        w32b_base.layout,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(win < 1):
                        mnxt = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + 64 + (win + 1) * 32, dtype=cutlass.Float32
                            ),
                            mst_base.layout,
                        )
                        cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)
                    for gv in cutlass.range_constexpr(4):
                        g8d = cute.make_tensor(
                            r_gt.iterator + 8 * gv, cute.make_layout((8,))
                        )
                        cute.autovec_copy(v_gt8[(win * 4 + gv, None, csc)], g8d)
                    mv = rcur.load()
                    rb32f.store(mv.to(cutlass.BFloat16) * cutlass.BFloat16(INP_SCALE))
                    rcur.store(mv * r_gt.load())
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))
                    cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))
                cute.arch.fence_view_async_tmem_store()
                state_inp_bar.arrive()
                for hh2 in cutlass.range_constexpr(2):
                    csh = csc if hh2 == 0 else csb
                    phh = p_qk if hh2 == 0 else phb
                    for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                        r_bt_f_wide[ee] = v_bt[uw_uid[ee][1], csh]
                    cute.arch.mbarrier_wait(mb + MB_VFULL + csh, phh)
                    cute.arch.mbarrier_wait(mb + MB_OOUT + csh, phh)
                    xh = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + (224 if hh2 == 0 else 288), dtype=cutlass.Float32
                        ),
                        u2f.layout,
                    )
                    cute.copy(u_wide_ld, uw_ldthr.partition_S(xh), r_u_wide)
                    cute.autovec_copy(uw_v[None, None, None, csh], r_vv_wide)
                    cute.arch.fence_view_async_tmem_load()
                    dvv = (
                        r_vv_wide.load().to(cutlass.Float32)
                        - r_u_wide.load() * deanchor
                    ) * r_bt_f_wide.load()
                    rb_wide.store(dvv.to(cutlass.BFloat16))
                    r3h = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 384 + hh2 * 16, dtype=cutlass.BFloat16
                        ),
                        u2b.layout,
                    )
                    cute.copy(u_wide_st, rb_wide, uw_sthr.partition_D(r3h))
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_VFREE + csc)
                    cute.arch.mbarrier_arrive(mb + MB_VFREE + csb)
                u_inp_bar.arrive()

                cute.arch.mbarrier_wait(mb + MB_U2ACC + csc, p_u2a)
                for hh2 in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        t_vst_h = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + 320 + hh2 * 32 + half * 16,
                                dtype=cutlass.Float32,
                            ),
                            h16c_base.layout,
                        )
                        cute.copy(h16c_ld, h16c_thr.partition_S(t_vst_h), r_u2h)
                        cute.arch.fence_view_async_tmem_load()
                        rb_u2h.store(r_u2h.load().to(cutlass.BFloat16))
                        twri_h = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + 416 + hh2 * 16 + half * 8,
                                dtype=cutlass.BFloat16,
                            ),
                            w8b_base.layout,
                        )
                        cute.copy(w8b_st, rb_u2h, w8b_thrc.partition_D(twri_h))
                cute.arch.fence_view_async_tmem_store()
                u2_inp_bar.arrive()

                cute.arch.mbarrier_wait(mb + MB_FIN + csc, p_fin)
                ofin_stage = csc
                ofin_phase = p_fin
                csc += 2
                if csc >= STAGES:
                    csc -= STAGES
                    p_qk ^= 1
                    p_vf ^= 1
                    p_oo ^= 1
                    p_u2a ^= 1
                    p_fin ^= 1
'''
s = find("            for t in cutlass.range(t_tiles):  # noqa: B007")
e = find(SPLIT_END, s)
lines[s:e] = COMPUTE.rstrip("\n").split("\n") + [""]

# ---------------- epilogue ----------------
pre = find("            cute.arch.mbarrier_wait(mb + MB_QK + csn, pqn)")
assert lines[pre + 1] == "            h0p = cute.make_tensor("
arr = find("            state_inp_bar.arrive()", pre)
assert lines[arr + 1] == "            csn += 1", lines[arr + 1]
img = lines[pre:arr + 1]
img = ["        " + ln if ln.strip() else ln for ln in img]
img = [ln.replace("h0p", "h0q") for ln in img]
e = find(SPLIT_END, arr)
EPI_HEAD = '''            csn += 2
            if csn >= STAGES:
                csn -= STAGES
                pqn ^= 1
            for tp in cutlass.range(t_tiles // 2):
                if tp + 1 < t_tiles // 2:
                    cute.arch.mbarrier_wait(mb + MB_FIN + cse, pe_state)'''.split("\n")
EPI_MID = '''                    csn += 2
                    if csn >= STAGES:
                        csn -= STAGES
                        pqn ^= 1
                cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)
                for hh2 in cutlass.range_constexpr(2):
                    t = tp * 2 + hh2
                    t_oh = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + (192 if hh2 == 0 else 256), dtype=cutlass.Float32
                        ),
                        t_out.layout,
                    )
                    cute.copy(o16_cp, out_thr.partition_S(t_oh), r_o)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(hh2 == 1):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_OEMPTY)
                    if warp == 4:
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                    ep_bar.arrive_and_wait()
                    r_ob.store((r_o.load() * out_deanchor).to(cutlass.BFloat16))
                    for dh in cutlass.range_constexpr(2):
                        for tg in cutlass.range_constexpr(2):
                            dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                            ta = tg * 16 + (mtx >> 1) * 8 + row8
                            tp2 = ta >> 1
                            par = ta & 1
                            raw_row = tp2 + (dim_base >> 6) * 16
                            raw_col = (
                                (dim_base & 63) ^ ((tp2 & 3) << 4) ^ (par << 3)
                            ) + par * 64
                            addr = so_base + ping * 8192 + (raw_row * 128 + raw_col) * 2
                            e0 = dh * 16 + tg * 8
                            s8 = cute.make_tensor(
                                r_ob.iterator + e0, cute.make_layout((8,))
                            )
                            dst = cute.make_tensor(
                                cute.make_ptr(
                                    cutlass.BFloat16,
                                    addr,
                                    cute.AddressSpace.smem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((8,)),
                            )
                            cute.copy(stm_np, s8, dst)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    ep_bar.arrive_and_wait()
                    if warp == 4:
                        if t >= warm_chunks:
                            f_o = cute.make_tensor(
                                p_o + ping * (C * D),
                                cute.make_layout(
                                    (C, 1, (64, 2)), stride=(64, 0, (1, 2048))
                                ),
                            )
                            o_s, o_g = cpasync.tma_partition(
                                tma_o,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_o, 0, 3),
                                cute.group_modes(gO, 0, 3),
                            )
                            cute.copy(tma_o, o_s, o_g[(None, t, hidx, 0)])
                            cute.arch.cp_async_bulk_commit_group()
                    ping ^= 1
                cse += 2
                if cse >= STAGES:
                    cse -= STAGES
                    pe_state ^= 1
                    pe_fin ^= 1
'''.rstrip("\n").split("\n")
lines[arr + 1:e] = EPI_HEAD + img + EPI_MID + [""]

# ---------------- MMA warp ----------------
s = find("                    for _t in cutlass.range(t_tiles):", find("                pm_oe = cutlass.Int32(1)"))
e = find("                        pm_oe ^= 1", s) + 1
MMA = '''                    for _tp in cutlass.range(t_tiles // 2):
                        csb = csm + 1
                        if csb == STAGES:
                            csb = 0
                        state_inp_bar.arrive_and_wait()
                        for hh2 in cutlass.range_constexpr(2):
                            csh = csm if hh2 == 0 else csb
                            t_uh = cute.make_tensor(
                                cute.recast_ptr(
                                    tmem_ptr + (224 if hh2 == 0 else 288),
                                    dtype=cutlass.Float32,
                                ),
                                t_u.layout,
                            )
                            mma_1.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(D // 16):
                                cute.gemm(
                                    mma_1,
                                    t_uh,
                                    t_inp[None, None, kb],
                                    b_kd[None, None, kb, csh],
                                    t_uh,
                                )
                                mma_1.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(mb + MB_OOUT + csh)
                        cute.arch.mbarrier_wait(mb + MB_OEMPTY, pm_oe)
                        for hh2 in cutlass.range_constexpr(2):
                            csh = csm if hh2 == 0 else csb
                            t_oh = cute.make_tensor(
                                cute.recast_ptr(
                                    tmem_ptr + (192 if hh2 == 0 else 256),
                                    dtype=cutlass.Float32,
                                ),
                                t_out.layout,
                            )
                            mma_1.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(D // 16):
                                cute.gemm(
                                    mma_1,
                                    t_oh,
                                    t_inp[None, None, kb],
                                    b_qd[None, None, kb, csh],
                                    t_oh,
                                )
                                mma_1.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_RFREE + csm)
                            tcgen05.commit(mb + MB_RFREE + csb)
                        u_inp_bar.arrive_and_wait()
                        for jj in cutlass.range_constexpr(2):
                            t_vh = cute.make_tensor(
                                cute.recast_ptr(
                                    tmem_ptr + 320 + jj * 32, dtype=cutlass.Float32
                                ),
                                t_vst.layout,
                            )
                            mma_3.set(tcgen05.Field.ACCUMULATE, False)
                            for ii in cutlass.range_constexpr(jj + 1):
                                r3i = cute.make_tensor(
                                    cute.recast_ptr(
                                        tmem_ptr + 384 + ii * 16, dtype=cutlass.BFloat16
                                    ),
                                    t_ri3.layout,
                                )
                                csi = csb if (jj == 1 and ii == 1) else csm
                                for kb in cutlass.range_constexpr(C // 16):
                                    cute.gemm(
                                        mma_3,
                                        t_vh,
                                        r3i[None, None, kb],
                                        b_inv[None, None, kb, csi],
                                        t_vh,
                                    )
                                    mma_3.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_U2ACC + csm)
                            tcgen05.commit(mb + MB_U2ACC + csb)
                        u2_inp_bar.arrive_and_wait()
                        mma_4a.set(tcgen05.Field.ACCUMULATE, True)
                        for hh2 in cutlass.range_constexpr(2):
                            csh = csm if hh2 == 0 else csb
                            r4h = cute.make_tensor(
                                cute.recast_ptr(
                                    tmem_ptr + 416 + hh2 * 16, dtype=cutlass.BFloat16
                                ),
                                t_ri4.layout,
                            )
                            for kb in cutlass.range_constexpr(C // 16):
                                cute.gemm(
                                    mma_4a,
                                    t_d4a,
                                    r4h[None, None, kb],
                                    b_fta[None, None, kb, csh],
                                    t_d4a,
                                )
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_FIN + csm)
                            tcgen05.commit(mb + MB_FIN + csb)
                        mma_4b.set(tcgen05.Field.ACCUMULATE, True)
                        for jj in cutlass.range_constexpr(2):
                            t_oh = cute.make_tensor(
                                cute.recast_ptr(
                                    tmem_ptr + (192 if jj == 0 else 256),
                                    dtype=cutlass.Float32,
                                ),
                                t_d4b.layout,
                            )
                            for ii in cutlass.range_constexpr(jj + 1):
                                r4i = cute.make_tensor(
                                    cute.recast_ptr(
                                        tmem_ptr + 416 + ii * 16, dtype=cutlass.BFloat16
                                    ),
                                    t_ri4.layout,
                                )
                                csi = csb if (jj == 1 and ii == 1) else csm
                                for kb in cutlass.range_constexpr(C // 16):
                                    cute.gemm(
                                        mma_4b,
                                        t_oh,
                                        r4i[None, None, kb],
                                        b_ftb[None, None, kb, csi],
                                        t_oh,
                                    )
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_OFIN + csm)
                            tcgen05.commit(mb + MB_OFIN + csb)
                            tcgen05.commit(mb + MB_SFREE + csm)
                            tcgen05.commit(mb + MB_SFREE + csb)
                        csm += 2
                        if csm >= STAGES:
                            csm -= STAGES
                        pm_oe ^= 1'''.split("\n")
lines[s:e] = MMA
open(sys.argv[2], "w").write("\n".join(lines))
print("ok p64")
