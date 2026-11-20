"""GS variant: conflict-free gate-prefix (GCS) layout, bitwise identical math.

usage: mkgs.py SRC DST
The decoration reads G[row, 8 channels] as two LDS.128 while its two
half-warps read rows r and r+4; with the plain [32,128] FP32 layout the 16
lanes of a half-warp hit only 4 of the 8 bank groups (2x excess wavefronts,
128 per chunk). Channels 32-63 and 96-127 store their 4-float halves swapped
(physical channel = ch ^ (((ch >> 5) & 1) << 2)), which spreads every half-warp
over all 8 groups and leaves the per-row gate stores in one 128 B line.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
reps = [
    # physical channel of this prep thread's gate column
    ("        ptl = plw * 32 + lane\n",
     "        ptl = plw * 32 + lane\n"
     "        ptls = ptl ^ (((ptl >> 5) & 1) << 2)\n", 1),
    # 4-float view of the gate prefix rows
    ("    v_gcs2 = cute.make_tensor(\n",
     "    v_gcs4 = cute.make_tensor(\n"
     "        p_gcs, cute.make_layout((C, 4, 32, STAGES), stride=(D, 1, 4, gcs_stage))\n"
     "    )\n"
     "    v_gcs2 = cute.make_tensor(\n", 1),
    ("v_gcs[rw0 + u, ptl, inst] = accg", "v_gcs[rw0 + u, ptls, inst] = accg", 2),
    ("v_gcs[rw0 + 8 + u, ptl, inst] = accg", "v_gcs[rw0 + 8 + u, ptls, inst] = accg", 1),
    ("v_gcs[rw, ptl, inst] = accg", "v_gcs[rw, ptls, inst] = accg", 1),
    ("gl_t = cutlass.Float32(v_gcs[C - 1, ptl, inst])",
     "gl_t = cutlass.Float32(v_gcs[C - 1, ptls, inst])", 1),
    ("                        g8s = v_gcs8[(rw2, None, sg2, inst)]\n"
     "                        cute.autovec_copy(g8s, g8)\n",
     "                        gxs = (sg2 >> 2) & 1\n"
     "                        cute.autovec_copy(\n"
     "                            v_gcs4[(rw2, None, (2 * sg2) ^ gxs, inst)], g4a\n"
     "                        )\n"
     "                        cute.autovec_copy(\n"
     "                            v_gcs4[(rw2, None, (2 * sg2 + 1) ^ gxs, inst)], g4b\n"
     "                        )\n"
     "                        for u in cutlass.range_constexpr(4):\n"
     "                            g8[u] = g4a[u]\n"
     "                            g8[4 + u] = g4b[u]\n", 1),
    ("        rf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)\n",
     "        rf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)\n"
     "        g4a = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)\n"
     "        g4b = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)\n", 1),
]
for a, b, n in reps:
    c = s.count(a)
    if c != n:
        sys.exit(f"pattern count {c} != {n}: {a[:70]!r}")
    s = s.replace(a, b)
open(dst, "w").write(s)
print("wrote", dst)
