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


def _sfu_derived():
    """MUFU (exp2) rate derived from the measured SM count and clock: 16 MUFU ops / clk / SM."""
    sms = HWJ["topology"]["sms_per_gpu"]["value"]
    clk = HWJ["clocks"]["sm_clock_max_MHz"]["value"] * 1e6
    return sms * 16 * clk


def _nvls_measured():
    """Achievable NVLS (multimem) all-reduce per N from OUR Hecate benches.

    alpha(N): symmbench multimem AR at 16-64 KiB (eager) minus the measured eager launch (2.27 us).
    algbw(N): ncclbench NVLS all-reduce at 256 MiB, busbw / (2 (N-1) / N).  N=2 uses the N=4 values."""
    lat = HWJ["latency"]
    launch = lat["kernel_launch_eager_us"]["measured"]
    ar = lat["symm_mem_allreduce_us"]["data"]
    bus = HWJ["nvlink"]["nccl_large_msg_busbw_GBps"]
    alpha, bw = {}, {}
    for n in (4, 8, 16):
        small = [r["multimem_us"] for r in ar[str(n)] if r["bytes"] <= 65536]
        alpha[n] = (sum(small) / len(small) - launch) * 1e-6
        bw[n] = bus["n%d" % n]["all_reduce"] * 1e9 / (2.0 * (n - 1) / n)
    alpha[2], bw[2] = alpha[4], bw[4]
    return alpha, bw


NVLS_ALPHA, NVLS_BW = _nvls_measured()
# Lamport-mailbox push latency grows with the number of slots polled: 2.86 us one-way flag (N=2, symm2.py) vs
# 3.39 us owner-reduce-done after the last push at 16 ranks (specialized_decode/cutedsl/common/twoshot.py).
PUSH_POLL_PER_PEER_S = (3.39 - 2.86) / 15 * 1e-6


class HW:
    """One hardware column. Rates in FLOP/s and B/s, latencies in s, capacities in B."""

    def __init__(self, name, peak, hbm, hbm_die_local, nvl, hop_s, kernel_s, l2_bw, hbm_budget, source,
                 sfu=None, poll_s=0.0, nvls=None):
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
        self.sfu = sfu              # MUFU ops/s (exp in softmax)
        self.poll_s = poll_s        # extra push latency per additional peer slot polled (Lamport mailbox)
        self.nvls = nvls            # None: NVLS AR = 2 hops + M(1+1/N)/nvl; else (alpha(N), algbw(N)) measured

    def nvls_ar(self, M, N, rs=False):
        """NVLS (multimem) all-reduce (or reduce-scatter, rs=True) of M bytes over N GPUs."""
        if self.nvls is None:
            if rs:
                return 2 * self.hop_s + M / self.nvl
            return 2 * self.hop_s + M * (1 + 1.0 / N) / self.nvl
        alpha, bw = self.nvls
        n = min((k for k in alpha if k >= N), default=max(alpha))
        return alpha[n] + M / bw[n] * (0.5 if rs else 1.0)

    def push_hop(self, N):
        """One dependent push hand-off into a mailbox written by N-1 peers."""
        return self.hop_s + self.poll_s * max(N - 2, 0)

    def as_dict(self):
        return dict(name=self.name, peak_PF={k: v / 1e15 for k, v in self.peak.items()}, hbm_TBps=self.hbm / 1e12,
                    hbm_die_local_TBps=self.hbm_die_local / 1e12, nvl_TBps_dir=self.nvl / 1e12,
                    hop_us=self.hop_s * 1e6, kernel_us=self.kernel_s * 1e6, l2_TBps=self.l2_bw / 1e12,
                    sfu_Tops=(self.sfu or 0) / 1e12, poll_us_per_peer=self.poll_s * 1e6,
                    nvls_alpha_us={str(k): v * 1e6 for k, v in (self.nvls[0] if self.nvls else {}).items()},
                    nvls_algbw_TBps={str(k): v / 1e12 for k, v in (self.nvls[1] if self.nvls else {}).items()},
                    hbm_budget_GiB=self.hbm_budget / GiB, source=self.source)


def profiles():
    d = HWJ["sol_model_defaults"]
    simt = _fp32_simt_derived()
    sfu = _sfu_derived()
    flag = d["nvlink_hop_latency_us"]["derated"] * 1e-6
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
        hop_s=flag,        # SOL: no public NVLink latency figure; never below our measured one-way flag hand-off
        kernel_s=0.0,      # SOL: kernels perfectly chained (persistent / PDL), no per-kernel gap
        sfu=sfu,
        l2_bw=d["l2_TBps"]["derated"] * 1e12,
        hbm_budget=budget,
        source="public VR200 spec: BF16 4 / FP8 17.5 / FP4 35 PF dense, HBM4 22 TB/s, NVLink6 3.6 TB/s bidir "
               "(1.8 per direction); FP32 SIMT + MUFU derived from measured SMs x clock; hop = our measured one-way "
               "flag (no public latency figure; a SOL hop is never set below a measured hand-off)",
    )
    ach = HW(
        "achievable",
        peak={"bf16": d["bf16_tc_PF"]["derated"] * 1e15, "fp8": d["fp8_mxfp8_tc_PF"]["derated"] * 1e15,
              "fp4": d["fp4_tc_PF"]["derated"] * 1e15, "tf32": HWJ["compute"]["tf32_dense_PF"]["achievable"] * 1e15,
              "simt": simt},
        hbm=d["hbm_TBps"]["derated_default"] * 1e12,
        hbm_die_local=d["hbm_TBps"]["derated_die_local"] * 1e12,
        nvl=d["nvlink_TBps_per_dir"]["derated"] * 1e12,
        hop_s=flag,
        kernel_s=node,
        sfu=sfu, poll_s=PUSH_POLL_PER_PEER_S, nvls=(NVLS_ALPHA, NVLS_BW),
        l2_bw=d["l2_TBps"]["derated"] * 1e12,
        hbm_budget=budget,
        source="Hecate measurements 2026-10-30: best GEMM BF16 3.65 / FP8 10.9 / FP4 21.1 PF, HBM read 13.9 TB/s "
               "(17.1 die-local, code docstring), NVLink ingress 1.25 TB/s, flag hop 2.86 us (+0.035 us per polled peer), "
               "graph node 0.78 us overlapped under PDL, NVLS AR alpha/algbw from symmbench + ncclbench",
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
        ("Dependent NVLink push hop", "us", s.hop_s * 1e6, a.hop_s * 1e6, "symm-mem flag ping-pong one-way 2.86 us in both columns (no public latency); achievable adds 0.035 us per polled peer (16-rank Lamport reduce 3.39 us, twoshot.py)"),
        ("NVLS all-reduce alpha (N=4/8/16)", "us", 2 * s.hop_s * 1e6, a.nvls[0][16] * 1e6, "SOL = 2 hops / symmbench multimem AR 16-64 KiB minus eager launch: %s us" % " / ".join("%.1f" % (a.nvls[0][n] * 1e6) for n in (4, 8, 16))),
        ("NVLS all-reduce algbw (N=4/8/16)", "TB/s", s.nvl / 1e12, a.nvls[1][16] / 1e12, "SOL = M(1+1/N) at NVLink / ncclbench NVLS AR 256 MiB: %s TB/s" % " / ".join("%.2f" % (a.nvls[1][n] / 1e12) for n in (4, 8, 16))),
        ("MUFU exp rate", "Top/s", s.sfu / 1e12, a.sfu / 1e12, "derived: 212 SMs x 16 / clk x 2.424 GHz"),
        ("Per-kernel gap in graph", "us", s.kernel_s * 1e6, a.kernel_s * 1e6, "SOL = 0 (persistent) / 0.78 us graph node, overlapped under PDL: a kernel costs max(roofline, 0.78 us)"),
        ("L2 read bandwidth (<=64 MiB)", "TB/s", s.l2_bw / 1e12, a.l2_bw / 1e12, "l2bench 30-35 TB/s"),
        ("HBM budget for weights+KV+workspace", "GiB", s.hbm_budget / GiB, a.hbm_budget / GiB, "vLLM startup free memory 272.48 GiB"),
    ]
    return [dict(name=r[0], unit=r[1], sol=r[2], achievable=r[3], source=r[4]) for r in rows]
