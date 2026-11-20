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


NVLS_ALPHA, _NVLS_BW_NCCL = _nvls_measured()

# ---- wave-1 microbench fixes (microbench/RESULTS.md "Model fixes"; comm lane run7, decode / gap512 lanes) ----------
# Large-message NVLS all-reduce: the multimem ld_reduce + multimem.st kernel (comm lane `nvls_tri`, fits.json slope over
# 16-256 MiB) reaches 0.64 / 0.77 / 0.84 / 0.88 TB/s algbw at N = 2/4/8/16; the NCCL-derived values above (0.29-0.46)
# are an NCCL artefact, not the multimem kernel. alpha stays the symmbench small-message value (~13 us, fused kernels).
NVLS_BW = {2: 0.64e12, 4: 0.77e12, 8: 0.84e12, 16: 0.88e12}
# NVLS reduce-scatter (comm lane recommendation 2): ld_reduce of the whole tensor through the switch runs at
# M / t = 1.00-1.06 TB/s at N >= 4 (~0.75 x the AR cost, not 0.5x) and 0.88 TB/s at N = 2:
#   t_RS = alpha(N) + M / min(1.06 TB/s, push ceiling)   (N >= 4),   alpha(2) + M / 0.88 TB/s   (N = 2)
NVLS_RS_BW = {2: 0.88e12, 4: 1.06e12, 8: 1.06e12, 16: 1.06e12}
# SM push / push2d / multicast AG per-GPU ceiling: 0.91-1.06 TB/s at every N (push 1.04, multicast AG 1.05, NVLS RS
# ld_reduce 1.06; comm-153 P6 q publish N=16 moves 106 MB in 104.3 us incl. one hop = 1.05+ TB/s). The 1.25 TB/s
# ingress is never reached. The measured maximum (1.06) is the achievable bound.
PUSH_BW_MEASURED = 1.06e12
# A standalone launched exchange cannot beat the fastest measured standalone exchange at that N: the minimum
# measured_us over the comm lane's a2a / AG rows (one-kernel Lamport exchange, comm rows.json): 3.86 / 4.22 / 4.49 /
# 4.89 us at N = 2 / 4 / 8 / 16 (N = 32 unmeasured: the N = 16 value). Exchanges fused into a compute kernel pay the
# 2.86 us in-kernel hop only. test_comm_rows_upper_bound checks comm_time <= measured on every comm row.
EXCH_FLOOR_S = {2: 3.86e-6, 4: 4.22e-6, 8: 4.49e-6, 16: 4.89e-6}
# Per-kernel floor of the grid-dependent kernels (KDA recurrence, indexer logits + top-k, sparse MLA, MegaMoE; the
# ops that need a grid-wide dependency and stay separate kernels in any implementation, KERNEL_FLOOR_OPS in cost.py).
# Launched decode kernels measure >= ~4 us end to end (decode / gap512 lanes: AttnRes mix 5.8 us per call, SP AG/RS
# 6.5 us, exchanges 3.9-4.9 us; KDA core / MlaQuery / indexer 10-50 us per layer), but under PDL the next kernel's
# launch and prologue overlap the previous kernel's tail, so the non-overlappable part is about half of that: 2 us.
# Bound check: with 4 us the achievable D1 k7 step at r = 1/16 (4.69 ms) exceeds the shipped D1 e2e steps (4.48-4.76
# ms incl. host time); 2 us keeps every microbench row and e2e anchor above the model (tests assert it). GEMM /
# elementwise chains are persistent- or PDL-fused (sdec D1) and keep the 0.78 us graph-node gap. The 10-50 us
# excess of today's kernels is kernel-specific (KERNEL-DEBUG list), not charged.
KERNEL_FLOOR_S = 2.0e-6

# Lamport-mailbox push latency grows with the number of slots polled: 2.86 us one-way flag (N=2, symm2.py) vs
# 3.39 us owner-reduce-done after the last push at 16 ranks (specialized_decode/cutedsl/common/twoshot.py).
PUSH_POLL_PER_PEER_S = (3.39 - 2.86) / 15 * 1e-6

class HW:
    """One hardware column. Rates in FLOP/s and B/s, latencies in s, capacities in B."""

    def __init__(self, name, peak, hbm, hbm_die_local, nvl, hop_s, kernel_s, l2_bw, hbm_budget, source,
                 sfu=None, poll_s=0.0, nvls=None, rs_bw=None, exch_floor_s=None, floor_s=0.0):
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
        self.rs_bw = rs_bw          # None: RS = half the AR bandwidth term; else measured NVLS RS B/s per N
        self.exch_floor_s = exch_floor_s or {}  # N -> minimum time of one launched exchange (push event)
        self.floor_s = floor_s      # per-kernel floor of the grid-dependent ops (cost.KERNEL_FLOOR_OPS)

    def nvls_ar(self, M, N, rs=False):
        """NVLS (multimem) all-reduce (or reduce-scatter, rs=True) of M bytes over N GPUs."""
        if self.nvls is None:
            if rs:
                return 2 * self.hop_s + M / self.nvl
            return 2 * self.hop_s + M * (1 + 1.0 / N) / self.nvl
        alpha, bw = self.nvls
        n = min((k for k in alpha if k >= N), default=max(alpha))
        if rs:
            if self.rs_bw is None:
                return alpha[n] + 0.5 * M / bw[n]
            return alpha[n] + M / min(self.rs_bw[n], self.nvl if n > 2 else self.rs_bw[n])
        return alpha[n] + M / bw[n]

    def exch_floor(self, N):
        """Minimum time of one standalone launched exchange over N ranks (0 in SOL)."""
        if not self.exch_floor_s:
            return 0.0
        n = min((k for k in self.exch_floor_s if k >= N), default=max(self.exch_floor_s))
        return self.exch_floor_s[n]

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
                    hbm_budget_GiB=self.hbm_budget / GiB,
                    nvls_rs_TBps={str(k): v / 1e12 for k, v in (self.rs_bw or {}).items()},
                    exch_floor_us={str(k): v * 1e6 for k, v in self.exch_floor_s.items()}, kernel_floor_us=self.floor_s * 1e6, source=self.source)


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
        nvl=PUSH_BW_MEASURED,
        hop_s=flag,
        kernel_s=node, floor_s=KERNEL_FLOOR_S,
        sfu=sfu, poll_s=PUSH_POLL_PER_PEER_S, nvls=(NVLS_ALPHA, NVLS_BW), rs_bw=NVLS_RS_BW,
        exch_floor_s=EXCH_FLOOR_S,
        l2_bw=d["l2_TBps"]["derated"] * 1e12,
        hbm_budget=budget,
        source="Hecate measurements 2026-10-30: best GEMM BF16 3.65 / FP8 10.9 / FP4 21.1 PF, HBM read 13.9 TB/s "
               "(17.1 die-local, code docstring), flag hop 2.86 us (+0.035 us per polled peer); 2026-11-18 microbench fixes: "
               "push 1.06 TB/s, NVLS AR algbw 0.64/0.77/0.84/0.88 TB/s (multimem kernel), NVLS RS 1.06 / 0.88 (N=2) "
               "TB/s, standalone-exchange floor 3.9-4.9 us by N, 2 us floor on grid-dependent kernels",
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
        ("NVLink per GPU per direction", "TB/s", s.nvl / 1e12, a.nvl / 1e12, "spec 3.6 bidir / comm lane: SM push, push2d and multicast AG top out at 0.91-1.05 TB/s (1.25 ingress never reached)"),
        ("Dependent NVLink push hop", "us", s.hop_s * 1e6, a.hop_s * 1e6, "symm-mem flag ping-pong one-way 2.86 us in both columns (no public latency); achievable adds 0.035 us per polled peer (16-rank Lamport reduce 3.39 us, twoshot.py)"),
        ("NVLS all-reduce alpha (N=4/8/16)", "us", 2 * s.hop_s * 1e6, a.nvls[0][16] * 1e6, "SOL = 2 hops / symmbench multimem AR 16-64 KiB minus eager launch: %s us" % " / ".join("%.1f" % (a.nvls[0][n] * 1e6) for n in (4, 8, 16))),
        ("NVLS all-reduce algbw (N=4/8/16)", "TB/s", s.nvl / 1e12, a.nvls[1][16] / 1e12, "SOL = M(1+1/N) at NVLink / comm lane multimem nvls_tri fit 16-256 MiB: %s TB/s (N=4/8/16); NVLS RS = alpha + M / %s TB/s (N=2/4/8/16)" % (" / ".join("%.2f" % (a.nvls[1][n] / 1e12) for n in (4, 8, 16)), " / ".join("%.2f" % (v / 1e12) for v in a.rs_bw.values()))),
        ("Standalone exchange floor (N=16)", "us", s.exch_floor(16) * 1e6, a.exch_floor(16) * 1e6, "SOL 0 / fastest measured standalone exchange per N (comm lane a2a / AG rows): %s us at N=2/4/8/16; fused in-kernel hops keep 2.86 us" % " / ".join("%.2f" % (v * 1e6) for v in a.exch_floor_s.values())),
        ("MUFU exp rate", "Top/s", s.sfu / 1e12, a.sfu / 1e12, "derived: 212 SMs x 16 / clk x 2.424 GHz"),
        ("Per-kernel gap in graph", "us", s.kernel_s * 1e6, a.kernel_s * 1e6, "SOL = 0 (persistent) / 0.78 us graph node, overlapped under PDL: a launched GEMM / elementwise kernel costs max(roofline, 0.78 us)"),
        ("Per-kernel floor (KDA core, indexer, sparse MLA, MegaMoE)", "us", s.floor_s * 1e6, a.floor_s * 1e6, "SOL = 0 / ~4 us measured launched decode kernel minus the PDL-overlapped launch + prologue (bounded by the D1 e2e anchors)"),
        ("L2 read bandwidth (<=64 MiB)", "TB/s", s.l2_bw / 1e12, a.l2_bw / 1e12, "l2bench 30-35 TB/s"),
        ("HBM budget for weights+KV+workspace", "GiB", s.hbm_budget / GiB, a.hbm_budget / GiB, "vLLM startup free memory 272.48 GiB"),
    ]
    return [dict(name=r[0], unit=r[1], sol=r[2], achievable=r[3], source=r[4]) for r in rows]
