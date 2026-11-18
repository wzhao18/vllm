"""Consistency checks for the SOL model (stdlib only): python3 k31vr/solmodel/test_solmodel.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cost  # noqa: E402
import decode as Dec  # noqa: E402
import kv  # noqa: E402
import model as Mdl  # noqa: E402
from hw import HWS  # noqa: E402
from topologies import ALL, comm_events  # noqa: E402


class TestModel(unittest.TestCase):
    def test_sparse_mla_dtype_follows_kv_row(self):
        self.assertEqual(Mdl.sparse_mla_dtype(576), "fp8")      # per-tensor FP8 row -> FP8 Q x FP8 KV
        self.assertEqual(Mdl.sparse_mla_dtype(656), "bf16")     # fp8_ds_mla row -> dequantized BF16
        ck = Mdl.ckpt_cfg("mxfp4_release")
        self.assertEqual(Mdl.op_dtype(Mdl.OPS_BY_ID["DSA.sparse_mla"], ck), Mdl.sparse_mla_dtype(Mdl.C["MLA_ROW"]))

    def test_recoverssm_record_traffic_matches_capacity(self):
        k, B = 3, 1.0
        env = Mdl.env_decode(B, k, 135000)
        r = Mdl.evaluate(Mdl.OPS_BY_ID["KDA.core_decode"], env, Mdl.ckpt_cfg("mxfp4_release"))
        state = Mdl.C["Hk"] * Mdl.C["dk"] * Mdl.C["dv"] * 4
        rec_write = (r["st_w"] - state) * Mdl.C["N_KDA"]
        self.assertAlmostEqual(rec_write, Mdl.per_request_state_bytes(k, 135000)["recover"], delta=1.0)

    def test_c512_anchor_maps_b_per_dp4_rank(self):
        import run
        a = run.run_anchors()[0]
        self.assertEqual([r["label"].split(" ")[0] for r in a["rows"]][:2], ["B4", "B16"])
        self.assertIn("r=1)", a["rows"][0]["label"])

    def test_afd_capacity_single_source(self):
        t = ALL["D11"]
        a = kv.capacity(t, "mxfp4_release", 3, 135000)
        b = kv.afd_capacity(t, "mxfp4_release", 3, 135000)
        self.assertEqual(a["rmax"], b["rmax"])
        self.assertEqual(len(b["rmax_by_split"]), 3)

    def test_integer_expert_partition(self):
        self.assertEqual(cost.experts_per_gpu(72), 13)
        self.assertEqual(cost.experts_per_gpu(64), 14)
        _, by = cost.resident_weights(ALL["D4-72"], "decode", "mxfp4_release")
        per_slot = Mdl.expert_bytes(Mdl.ckpt_cfg("mxfp4_release")) * Mdl.C["N_MOE"]
        self.assertAlmostEqual(by["experts"], 13 * per_slot, delta=1.0)     # not 896/72 = 12.44

    def test_tp_moe_dense_has_transposes(self):
        env = Mdl.env_decode(16, 3, 135000)
        ck = Mdl.ckpt_cfg("mxfp4_release")
        for tid in ("D7-4", "D8"):
            prims = [e.get("prim") or e["alts"][0]["prim"] for e in comm_events(ALL[tid], "decode", env, ck)["moe"]]
            self.assertEqual(sum(p.startswith("P14 TP-MoE transpose") for p in prims), 2, tid)

    def test_scope_32_gpus(self):
        from topologies import SCOPE, in_scope
        for tid in ("D7-4-32", "D7-2-32", "D7-8-32", "D8-32", "D9-32", "D3-32", "D4-32", "P3-32", "P6-32", "P9-32",
                    "D7-4-16", "D7-2-16", "D7-8-16", "D8-16", "D9-16", "P6-16", "P9", "P9-16", "D1", "P1"):
            self.assertIn(tid, SCOPE)
        for tid in ("D7-4", "D8", "D9", "D10", "D11", "P6", "P7", "P3-72", "D4-72"):
            self.assertFalse(in_scope(ALL[tid]), tid)
        for tid in SCOPE:
            self.assertLessEqual(ALL[tid].gpus, 32, tid)
            self.assertTrue(ALL[tid].experts == "tp" or ALL[tid].E <= 32, tid)
        env = Mdl.env_decode(16, 3, 135000)
        ck = Mdl.ckpt_cfg("mxfp4_release")
        for tid in ("D7-4-16", "D8-16", "D7-4-32", "D8-32"):
            prims = [e.get("prim") or e["alts"][0]["prim"] for e in comm_events(ALL[tid], "decode", env, ck)["moe"]]
            self.assertEqual(sum(p.startswith("P14 TP-MoE transpose") for p in prims), 2, tid)
            self.assertTrue(any(p.startswith("P11") for p in prims), tid)
        per_slot = Mdl.expert_bytes(ck) * Mdl.C["N_MOE"]
        for tid, slots in (("D8-16", 56), ("D8-32", 28), ("D3-32", 28)):
            _, by = cost.resident_weights(ALL[tid], "decode", "mxfp4_release")
            self.assertAlmostEqual(by["experts"], slots * per_slot, delta=1.0)
        self.assertEqual((ALL["D3-32"].N, ALL["D3-32"].residual, ALL["D3-32"].dsa, ALL["D3-32"].E), (4, "sp", "dcp", 32))
        self.assertEqual((ALL["P3-32"].N, ALL["P3-32"].E), (1, 32))

    # ---- wave-1 microbench model fixes (microbench/RESULTS.md "Model fixes")
    def test_comm_fixes(self):
        a, s = HWS["achievable"], HWS["sol"]
        self.assertAlmostEqual(a.nvl, 1.06e12)                                 # measured push / mc / RS ceiling
        self.assertEqual({n: round(a.nvls[1][n] / 1e12, 2) for n in (2, 4, 8, 16)}, {2: 0.64, 4: 0.77, 8: 0.84, 16: 0.88})
        M = 256 * 2 ** 20
        self.assertAlmostEqual(a.nvls_ar(M, 8, rs=True) - a.nvls[0][8], M / 1.06e12)          # comm lane rec. 2
        self.assertAlmostEqual(a.nvls_ar(M, 2, rs=True) - a.nvls[0][2], M / 0.88e12)
        tiny = dict(prim="P2 SP-AG", name="x", in_B=64.0, out_B=64.0, hops=1, kind="push", N=4)
        self.assertAlmostEqual(cost.comm_time(tiny, a)[0], 4.22e-6)              # per-N standalone floor
        self.assertLess(cost.comm_time(tiny, a, fused=True)[0], 4.22e-6)
        self.assertLess(cost.comm_time(tiny, s)[0], 4.22e-6)                     # SOL has no launch floor

    def test_comm_rows_upper_bound(self):
        """Every comm-lane row: the achievable comm time is not above the measured standalone kernel."""
        import json
        from topologies import comm_row_event
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "microbench/results/comm/rows.json")) as f:
            rows = json.load(f)
        for x in rows:
            t = cost.comm_time(comm_row_event(x), HWS["achievable"])[0] * 1e6
            self.assertLessEqual(t, x["measured_us"] * 1.0001, x["case"])

    def test_kernel_floor(self):
        env = Mdl.env_decode(1, 3, 135000)
        a = cost.evaluate(ALL["D1"], "decode", "mxfp4_release", "achievable", env)
        s = cost.evaluate(ALL["D1"], "decode", "mxfp4_release", "sol", env)
        self.assertAlmostEqual(HWS["achievable"].floor_s, 2.0e-6)
        self.assertGreaterEqual(a["unit"]["kda"]["comp"] + a["unit"]["kda"]["kernels"], 2.0e-6)
        self.assertEqual(s["unit"]["kda"]["kernels"], 0.0)
        self.assertIn("DSA.idx_score", cost.KERNEL_FLOOR_OPS)

    def test_anchors_upper_bound(self):
        """The achievable column is never slower than a shipped measurement (anchor ratio >= 1). Allow-list: the C512
        P2 prefill marginal slope is a lower-bound fit (165 ms + 11 ms/1K), not a measured step."""
        import run
        allow = ("marginal step ms per 1K chunk tokens",)
        for a in run.run_anchors():
            for r in a["rows"]:
                if any(x in r["label"] for x in allow):
                    continue
                self.assertGreaterEqual(r["ratio_ach"], 1.0, (a["name"], r["label"]))

    def test_megamoe_rows_upper_bound(self):
        """Every moe-lane row: the model's MegaMoE layer is not above the measured fused kernel."""
        import calibrate as Cal
        for x in Cal.MOE_ROWS:
            lay, _ = Cal._row_layer(x)
            self.assertLessEqual(lay * 1e6, x["measured_kernel_us"] * 1.0001, x["case"])

    def test_megamoe_overlap(self):
        for tid, phase, env in (("P3-16", "prefill", Mdl.env_prefill(16384, 78400)),
                                ("D3", "decode", Mdl.env_decode(32, 3, 135000))):
            for col in ("sol", "achievable"):
                m = cost.evaluate(ALL[tid], phase, "mxfp4_release", col, env)["megamoe"]
                self.assertGreaterEqual(m["layer"], max(m["ops"], m["link_bw"]) - 1e-12, tid)
                self.assertLess(m["layer"], m["serial"], tid)
                self.assertAlmostEqual(m["exposed"], m["layer"] - m["ops"], places=12)
        # D1 (MoE-TP, no EP) has no MegaMoE term
        self.assertIsNone(cost.evaluate(ALL["D1"], "decode", "mxfp4_release", "sol", Mdl.env_decode(16, 3, 135000))["megamoe"])

    def test_p1_attnres_on_shard_plus_ag(self):
        from topologies import op_map
        self.assertEqual(op_map("L.attnres_pre", ALL["P1"], "prefill")["row"], 8)
        self.assertEqual(op_map("L.attnres_pre", ALL["D1"], "decode")["row"], 1)
        env = Mdl.env_prefill(16384, 78400)
        ck = Mdl.ckpt_cfg("mxfp4_release")
        kda = comm_events(ALL["P1"], "prefill", env, ck)["kda"]
        names = [e.get("name") or e["alts"][0]["name"] for e in kda]
        self.assertIn("AttnRes shard AG", names)
        self.assertFalse(any(n.endswith(" AR") for n in names))
        dec = comm_events(ALL["D1"], "decode", Mdl.env_decode(16, 3, 135000), ck)["kda"]
        self.assertTrue(any((e.get("name") or e["alts"][0]["name"]).endswith(" AR") for e in dec))

    def test_kv_block_floor(self):
        import calibrate as Cal
        # deployed layouts reproduce the measured block rows per request: D1 unpinned (bs 1024, G 3), D2 pinned
        # bs 1024 with VLLM_PACKED_KV_STATE_GROUPS=4 and the deployed BF16 conv state
        for k, rows in ((3, 24), (7, 36)):
            b = kv.capacity(ALL["D1"], "mxfp4_release", k, 135000, col="achievable")["blocks"]["default"]
            self.assertEqual((b["bs"], b["G"]), (1024, 3))
            self.assertAlmostEqual(b["rows_per_gpu"], rows)
        for k, rows in ((3, 36), (7, 52)):
            _, d = kv.packed_request_group_bytes(ALL["D2"], k, 135000, bs=1024, G=4, conv_bytes=2)
            self.assertAlmostEqual(d["rows_per_gpu"], rows)
        self.assertIsNone(kv.packed_request_group_bytes(ALL["D2"], 3, 135000, bs=1024, G=3, conv_bytes=2)[0])
        row_tok = Mdl.C["N_DSA"] * (Mdl.C["MLA_ROW"] + Mdl.C["IDX_ROW"])
        for (tid, k), (rows, bs) in Cal.MEASURED_ROWS.items():
            # achievable = min over pinned block sizes: never more bytes per request than the measured deployment
            # (like for like: the deployments ran BF16 conv state; the model's FP32 conv rule adds up to ~2% at D2 k7)
            _, d = kv.packed_min_bytes(ALL[tid], k, 135000, conv_bytes=2)
            self.assertLessEqual(d["per_gpu_B"], rows * bs * row_tok, (tid, k))
        # upper bound: SOL >= achievable >= measured r_max on every measured path
        for (tid, k), meas in Cal.MEASURED_RMAX.items():
            s = kv.capacity(ALL[tid], "mxfp4_release", k, 135000)["rmax"]
            a = kv.capacity(ALL[tid], "mxfp4_release", k, 135000, col="achievable")["rmax"]
            self.assertGreaterEqual(s, a, tid)
            self.assertGreaterEqual(a, meas, (tid, k))
        # calibrated capacity: MoE-TP padding only on MoE-TP layouts; D8 is not charged D1's padding
        self.assertGreater(Cal._extra_weights_GiB(ALL["D1"], "mxfp4_release"), 20.0)
        self.assertLess(Cal._extra_weights_GiB(ALL["D8-32"], "mxfp4_release"), 12.0)

    def test_achievable_upper_bounds_measured_steps(self):
        import calibrate as Cal
        for tid in Cal.DEC_MEASURED:
            for p in Cal.DER["by_topology"][tid]["points"]:
                if p.get("moe_quant", "mxfp4_mxfp8") != "mxfp4_mxfp8":
                    continue
                a = Dec.run_point(ALL[tid], "mxfp4_release", "achievable", p["r_per_gpu"], p["k"])["t_ms"]
                self.assertLessEqual(a, p["measured_ms"] * 1.0001, (tid, p["k"], p["r_per_gpu"]))

    def test_prefill_achievable_upper_bounds_measured(self):
        """Every measured L78K prefill step (waves 1-2, MXFP8 arms vs the model, NVFP4-MoE release arms vs O4)."""
        import calibrate as Cal
        import prefill as Pre
        for path in Cal.PRE_MEASURED:
            for nv in (False, True):
                for F, f, ms in Cal._prefill_points(path, nv):
                    a = Pre.run_point(ALL[path], "mxfp4_release", "achievable", F, Pre.L_REF, f_req=f,
                                      moe_act="nvfp4" if nv else None)["t_ms"]
                    self.assertLessEqual(a, ms, (path, nv, F))

    def test_calibrated_reproduces_measured(self):
        import calibrate as Cal
        import prefill as Pre
        for tid in Cal.DEC_MEASURED:
            for r, ms in Cal._measured_points(tid):
                p = Dec.run_point(ALL[tid], "mxfp4_release", "achievable", r, 3)
                c = Cal.decode_calibrated(ALL[tid], "mxfp4_release", p)
                self.assertAlmostEqual(c["t_ms"], ms, places=6)
                self.assertEqual(c["tag"], "measured", (tid, r))
        # unmeasured layouts are tagged; r beyond the front path's measured range is extrapolated
        p = Dec.run_point(ALL["D8-32"], "mxfp4_release", "achievable", 2, 3)
        self.assertEqual(Cal.decode_calibrated(ALL["D8-32"], "mxfp4_release", p)["tag"], "projected")
        p = Dec.run_point(ALL["D7-4-32"], "mxfp4_release", "achievable", 32, 3)
        c = Cal.decode_calibrated(ALL["D7-4-32"], "mxfp4_release", p)
        self.assertEqual((c["tag"], c["path"]), ("extrapolated", "D3-32"))          # EP32 front = measured D3-32
        p = Dec.run_point(ALL["D3"], "nvfp4_modelopt", "achievable", 8, 3)
        self.assertNotEqual(Cal.decode_calibrated(ALL["D3"], "nvfp4_modelopt", p)["tag"], "measured")
        # prefill: calibrated >= measured throughput at every measured (path, F, f) (equal by construction)
        for path in Cal.PRE_MEASURED:
            for F, f, ms in Cal._prefill_points(path):
                c = Cal.prefill_calibrated(ALL[path], "mxfp4_release", F, Pre.L_REF, f_req=f)
                meas_tok = F / (ALL[path].N * ms * 1e-3)
                self.assertGreaterEqual(c["tok_s_gpu"], meas_tok * 0.9999, (path, F))
                self.assertEqual(c["tag"], "measured")
            self.assertEqual(Cal.prefill_calibrated(ALL[path], "mxfp4_release", 65536, Pre.L_REF)["tag"], "extrapolated")
        c6 = Cal.prefill_calibrated(ALL["P6-16"], "mxfp4_release", 16384, Pre.L_REF)
        self.assertTrue(c6["dedup_removed"])
        self.assertEqual(c6["path"], "P4-2")


    def test_dsadp_decode_push_is_zero_bytes(self):
        env = Mdl.env_decode(16, 3, 135000)
        evs = comm_events(ALL["D7-4"], "decode", env, Mdl.ckpt_cfg("mxfp4_release"))["dsa"]
        self.assertTrue(all(e["in_B"] == 0 for e in evs if e.get("prim", "").startswith("P13")))

    def test_achievable_not_faster_than_sol(self):
        for tid in ("D1", "D7-4", "D4-16"):
            s = Dec.run_point(ALL[tid], "mxfp4_release", "sol", 1, 3)
            a = Dec.run_point(ALL[tid], "mxfp4_release", "achievable", 1, 3)
            self.assertLess(s["t_ms"], a["t_ms"], tid)

    def test_sol_hop_not_below_measured(self):
        self.assertGreaterEqual(HWS["sol"].hop_s, 2.86e-6 - 1e-12)


if __name__ == "__main__":
    unittest.main()
