"""VR200 hardware profiles for the K3.1 SOL model.

Two columns, never mixed and never fitted:
  * "sol"        - public spec peaks (dense tensor rates, HBM4 22 TB/s, NVLink6 1.8 TB/s per direction).
  * "achievable" - our own Hecate measurements (best GEMM per dtype, full-chip HBM read, symmetric-memory
                   NVLink push, flag ping-pong hop, CUDA-graph node). No derating beyond the measured peaks.

All values are read from inputs/hw.json (built by inputs/hw_src/build_hw.py from the bench logs).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GiB = 2 ** 30


def _load():
    with open(os.path.join(HERE, "inputs", "hw.json")) as f:
        return json.load(f)


HWJ = _load()


def _fp32_simt_derived():
    """FP32 CUDA-core rate derived from OUR measured device attributes (SM count x clock).

    212 SMs x 128 FP32 lanes x 2 FLOP (FMA) x 2.424 GHz.  Used for both columns (no FFMA bench yet)."""
    sms = HWJ["topology"]["sms_per_gpu"]["value"]
    clk = HWJ["clocks"]["sm_clock_max_MHz"]["value"] * 1e6
    return sms * 128 * 2 * clk


class HW:
    """One hardware column. Rates in FLOP/s and B/s, latencies in s, capacities in B."""

    def __init__(self, name, peak, hbm, hbm_die_local, nvl, hop_s, kernel_s, l2_bw, hbm_budget, source):
        self.name = name
        self.peak = peak            # dict dtype -> FLOP/s ('bf16','fp8','fp4','tf32','simt')
        self.hbm = hbm              # full-chip HBM B/s
        self.hbm_die_local = hbm_die_local  # HBM B/s when every byte is read by the die that homes it (O3)
        self.nvl = nvl              # NVLink B/s per GPU per direction
        self.hop_s = hop_s          # one dependent NVLink hand-off (flag) latency
        self.kernel_s = kernel_s    # fixed per-kernel cost inside a CUDA graph
        self.l2_bw = l2_bw
        self.hbm_budget = hbm_budget  # bytes usable by weights + KV + workspace per GPU
        self.source = source

    def as_dict(self):
        return dict(name=self.name, peak_PF={k: v / 1e15 for k, v in self.peak.items()}, hbm_TBps=self.hbm / 1e12,
                    hbm_die_local_TBps=self.hbm_die_local / 1e12, nvl_TBps_dir=self.nvl / 1e12,
                    hop_us=self.hop_s * 1e6, kernel_us=self.kernel_s * 1e6, l2_TBps=self.l2_bw / 1e12,
                    hbm_budget_GiB=self.hbm_budget / GiB, source=self.source)


def profiles():
    d = HWJ["sol_model_defaults"]
    simt = _fp32_simt_derived()
    budget = d["hbm_budget_GiB"]["value"] * GiB
    node = d["graph_kernel_overhead_us"]["derated"] * 1e-6
    sol = HW(
        "sol",
        peak={"bf16": d["bf16_tc_PF"]["sol"] * 1e15, "fp8": d["fp8_mxfp8_tc_PF"]["sol"] * 1e15,
              "fp4": d["fp4_tc_PF"]["sol"] * 1e15, "tf32": HWJ["compute"]["tf32_dense_PF"]["sol"] * 1e15,
              "simt": simt},
        hbm=d["hbm_TBps"]["sol"] * 1e12,
        hbm_die_local=d["hbm_TBps"]["sol"] * 1e12,   # spec is full-chip; die-local cannot exceed it
        nvl=d["nvlink_TBps_per_dir"]["sol"] * 1e12,
        hop_s=node,        # SOL: a dependent hand-off costs one graph node, no extra flag latency
        kernel_s=0.0,      # SOL: kernels perfectly chained (persistent / PDL), no per-kernel gap
        l2_bw=d["l2_TBps"]["derated"] * 1e12,
        hbm_budget=budget,
        source="public VR200 spec: BF16 4 / FP8 17.5 / FP4 35 PF dense, HBM4 22 TB/s, NVLink6 3.6 TB/s bidir "
               "(1.8 per direction); FP32 SIMT derived from measured SMs x clock; hop = 1 graph node (measured)",
    )
    ach = HW(
        "achievable",
        peak={"bf16": d["bf16_tc_PF"]["derated"] * 1e15, "fp8": d["fp8_mxfp8_tc_PF"]["derated"] * 1e15,
              "fp4": d["fp4_tc_PF"]["derated"] * 1e15, "tf32": HWJ["compute"]["tf32_dense_PF"]["achievable"] * 1e15,
              "simt": simt},
        hbm=d["hbm_TBps"]["derated_default"] * 1e12,
        hbm_die_local=d["hbm_TBps"]["derated_die_local"] * 1e12,
        nvl=d["nvlink_TBps_per_dir"]["derated"] * 1e12,
        hop_s=d["nvlink_hop_latency_us"]["derated"] * 1e-6,
        kernel_s=node,
        l2_bw=d["l2_TBps"]["derated"] * 1e12,
        hbm_budget=budget,
        source="Hecate measurements 2026-10-30: best GEMM BF16 3.65 / FP8 10.9 / FP4 21.1 PF, HBM read 13.9 TB/s "
               "(17.1 die-local, code docstring), NVLink ingress 1.25 TB/s, flag hop 2.86 us, graph node 0.78 us",
    )
    return {"sol": sol, "achievable": ach}


HWS = profiles()
COLUMNS = ("sol", "achievable")


def constants_table():
    """Rows for the HTML hardware table (public spec + our measurements only)."""
    s, a = HWS["sol"], HWS["achievable"]
    rows = [
        ("BF16 dense tensor", "PFLOP/s", s.peak["bf16"] / 1e15, a.peak["bf16"] / 1e15, "spec table / torch GEMM 8192^3"),
        ("FP8 / MXFP8 dense tensor", "PFLOP/s", s.peak["fp8"] / 1e15, a.peak["fp8"] / 1e15, "spec table / torch._scaled_mm 16384^3"),
        ("NVFP4 / MXFP4 dense tensor", "PFLOP/s", s.peak["fp4"] / 1e15, a.peak["fp4"] / 1e15, "spec table (dense 'training' row) / flashinfer mm_fp4 16384^3"),
        ("TF32 dense tensor", "PFLOP/s", s.peak["tf32"] / 1e15, a.peak["tf32"] / 1e15, "spec table / torch tf32 GEMM"),
        ("FP32 SIMT (KDA recurrence, conv)", "TFLOP/s", s.peak["simt"] / 1e12, a.peak["simt"] / 1e12, "derived: 212 SMs x 128 lanes x 2 x 2.424 GHz (measured attributes)"),
        ("HBM4 bandwidth", "TB/s", s.hbm / 1e12, a.hbm / 1e12, "spec 22 TB/s / triton full-chip read 13.93 TB/s"),
        ("HBM die-local (O3)", "TB/s", s.hbm_die_local / 1e12, a.hbm_die_local / 1e12, "spec cap / die_local.py docstring 17.1 TB/s"),
        ("NVLink per GPU per direction", "TB/s", s.nvl / 1e12, a.nvl / 1e12, "spec 3.6 bidir / sp_all_gather.py ingress 1.25 TB/s (pair push 1.09 measured)"),
        ("Dependent NVLink hop", "us", s.hop_s * 1e6, a.hop_s * 1e6, "SOL = 1 graph node / symm-mem flag ping-pong one-way 2.86 us"),
        ("Per-kernel overhead in graph", "us", s.kernel_s * 1e6, a.kernel_s * 1e6, "SOL = 0 (persistent) / 1000-node CUDA graph 0.78 us"),
        ("L2 read bandwidth (<=64 MiB)", "TB/s", s.l2_bw / 1e12, a.l2_bw / 1e12, "l2bench 30-35 TB/s"),
        ("HBM budget for weights+KV+workspace", "GiB", s.hbm_budget / GiB, a.hbm_budget / GiB, "vLLM startup free memory 272.48 GiB"),
    ]
    return [dict(name=r[0], unit=r[1], sol=r[2], achievable=r[3], source=r[4]) for r in rows]
