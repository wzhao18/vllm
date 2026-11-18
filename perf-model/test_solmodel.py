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
