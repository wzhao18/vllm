"""Build k31vr/solmodel/inputs/hw.json from the VR200 bench logs + spec sheet + code docstrings."""
import json, sys, os, collections

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = "/Users/weizha/k31vr-dev/k31vr/solmodel/inputs/hw.json"
HECATE = "hecate:/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/hw"


def lines(f):
    for ln in open(os.path.join(RAW, f)):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                yield json.loads(ln)
            except Exception:
                pass


c1 = {k: v for d in lines("c1-913068.log") for k, v in d.items()}

# ---------- NCCL tables (c4: ncclbench.py, NVLS on, default proto; c5: symmbench.py MODE=nccl variants) ----------
nccl = collections.defaultdict(list)
for d in lines("c4-913069.log"):
    if "n" in d and "bytes" in d:
        row = {"bytes": d["bytes"]}
        for op in ("all_gather", "all_to_all", "all_reduce"):
            row[op + "_us"] = d.get(op + "_us")
            row[op + "_busbw_GBps"] = d.get(op + "_busbw_GBps")
            if op + "_graph_us" in d:
                row[op + "_graph_us"] = d[op + "_graph_us"]
                row[op + "_graph_busbw_GBps"] = d[op + "_graph_busbw_GBps"]
        nccl[str(d["n"])].append(row)
nccl_variants = collections.defaultdict(lambda: collections.defaultdict(list))
symm_ar = collections.defaultdict(list)
for d in lines("c5-913098.log"):
    if "variant" in d and d.get("variant", "").startswith("nccl") and "n" in d:
        nccl_variants[d["variant"]][str(d["n"])].append(
            {k: d[k] for k in ("bytes", "all_gather_us", "all_to_all_us", "all_reduce_us")})
    if "symm_allreduce_n" in d:
        symm_ar[str(d["symm_allreduce_n"])].append(
            {"bytes": d["bytes"], "one_shot_us": d["one_shot_us"], "two_shot_us": d["two_shot__us"],
             "multimem_us": d["multimem__us"], "one_shot_graph_us": d["one_shot_graph_us"]})
p2p = [d for d in lines("c4-913069.log") if "p2p_peer" in d]
l2_read = [d for d in lines("c5-913098.log") if "read_MiB" in d]
chase = [d for d in lines("c5-913098.log") if "chase_footprint_MiB" in d]
c6 = list(lines("c6-913203.log"))
pingpong = [d for d in c6 if "pingpong_peer" in d]
remote_bw = [d for d in c6 if "pattern" in d]

steady_oneway = [d["rep2_oneway_us"] for d in pingpong]

S = "spec"
M = "measured"
hw = {
    "_meta": {
        "device": "NVIDIA VR200 (Rubin, compute capability 10.7, 'sm_107'), Vera Rubin NVL72, 4 GPUs/node (2 Vera Rubin superchips)",
        "measured_on": "Hecate batch-xdr, 2026-10-30, jobs 913068 (1 node), 913069/913098/913203 (4 nodes, --segment=4, 16 GPUs one NVL72 rack); "
                       "container vllm-openai-cu134-nightly-3f1bcfd82b, torch 2.16.0.dev20261009+cu134, NCCL 2.32.3, triton 3.8, flashinfer 0.7.1",
        "bench_scripts": HECATE + "/{hwbench.py,l2bench.py,ncclbench.py,symmbench.py,symm2.py}; logs c1-913068.log c4-913069.log c5-913098.log c6-913203.log "
                         "(local copies: k31vr/solmodel/inputs/hw_src/raw/, flat dump: scratchpad/vr200_hw.jsonl)",
        "spec_source": "NVIDIA Vera Rubin NVL72 product page spec table (https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/), "
                       "'Preliminary information. All values are up to and subject to change.'; rows marked dense where NVIDIA says so",
        "convention": "every leaf {spec, measured, unit, source, note}; 'sol' = value the SOL model should use as the hardware bound; "
                      "'achievable' = best measured value (use for derating / microbench matching). TB = 1e12 B, GB = 1e9 B, GiB = 2^30 B, PF = 1e15 FLOP/s.",
    },
    "topology": {
        "gpus_per_node": {"value": 4, "source": "nvidia-smi topo (c1-913033.log: GPU0-3 NV36 to each other)", "kind": M},
        "nvlink_domain_gpus": {"value": 72, "source": "NVL72 product spec; Hecate segments nvlblk16 feature = 16-GPU (4-node) scheduling blocks, rack = 18 nodes", "kind": S},
        "nvlinks_per_gpu_pair_reported": {"value": 36, "source": "nvidia-smi topo 'NV36' (36 NVLink6 links per GPU through the switch)", "kind": M},
        "dies_per_gpu": {"value": 2, "source": "specialized_decode/cutedsl/common/dies.py docstring (CUDA 13.4 locality domains)", "kind": M},
        "sms_per_gpu": {"value": 212, "source": "cudaDevAttrMultiProcessorCount (c1)", "kind": M},
        "sms_per_die_driver": {"value": 100, "source": "dies.py: green contexts attribute 100 SMs to each die of a 212-SM part (12 SMs not in either green-context set)", "kind": M},
        "intra_vs_inter_node": "No difference measured: symm-mem ping-pong one-way 2.82-2.93 us to all 15 peers (same node or other nodes); NCCL p2p 1 MiB 24.0-24.6 GB/s to peers 1/4/8/12; "
                               "remote read/write BW pair r0-r1 (same node) == r0-r4 (other node). Model the rack as a flat NVLink switch.",
    },
    "clocks": {
        "sm_clock_max_MHz": {"value": 2424, "source": "cudaDevAttrClockRate / nvidia-smi Max Clocks (c1, smi-q-913033.txt)", "kind": M},
        "sm_clock_observed_MHz": {"value": 2280, "source": "nvidia-smi -q Clocks.SM at idle sample (smi-q-913033.txt)", "kind": M},
        "mem_clock_MHz": {"value": 4752, "source": "cudaDevAttrMemoryClockRate", "kind": M},
        "power_limit_W": {"value": 2300, "source": "nvidia-smi GPU Ceiling Power Limit (default=max=2300 W)", "kind": M},
    },
    "compute": {
        "_note": "Dense tensor-core rates. 'measured' = best square GEMM (8192^3 or 16384^3) with BF16 output from hwbench.py (c1). "
                 "Spec FP8 = NVIDIA 'FP8/FP6 Training' (dense); spec NVFP4 dense = 'NVFP4 Training' 35 PF (the 50 PF 'NVFP4 Inference' row is not marked dense; "
                 "treat as non-dense/adaptive-compression figure, do NOT use in SOL). MXFP4 x MXFP4 and MXFP8 x MXFP4 run at the FP4 / FP8 tensor rate respectively "
                 "(block-scaled MMA kinds mxf4nvf4 / mxf8f6f4).",
        "bf16_dense_PF": {"spec": 4.0, "measured": 3.648, "unit": "PFLOP/s",
                          "source": "spec: NVL72 table FP16/BF16 (dense) 4 PF/GPU; measured: torch bf16 GEMM 8192^3 3647.7 TF, 16384^3 3587.8 TF (c1)",
                          "sol": 4.0, "achievable": 3.65, "achievable_frac_of_spec": 0.91},
        "tf32_dense_PF": {"spec": 2.0, "measured": 1.949, "unit": "PFLOP/s",
                          "source": "spec 2 PF dense; measured torch tf32 8192^3 1949 TF (c1)", "sol": 2.0, "achievable": 1.95},
        "fp8_dense_PF": {"spec": 17.5, "measured": 10.87, "unit": "PFLOP/s",
                         "source": "spec: 'FP8/FP6 Training' 17.5 PF dense; measured: torch._scaled_mm FP8 e4m3 tensorwise 16384^3 10867.8 TF, 8192^3 10369 TF "
                                   "(rowwise-scaled cuBLAS path only 4.2-5.0 PF = a library path issue, not hardware)",
                         "sol": 17.5, "achievable": 10.9, "achievable_frac_of_spec": 0.62,
                         "note": "Large gap spec vs measured; likely power/clock-limited at 2300 W or immature cuBLAS sm_107 kernels. SOL uses spec; derate to 10.9."},
        "mxfp8_dense_PF": {"spec": 17.5, "measured": 10.90, "unit": "PFLOP/s",
                           "source": "spec = FP8 row; measured torch._scaled_mm MXFP8 (e8m0 per-32) 16384^3 10899.6 TF, 8192^3 10125 TF (c1)",
                           "sol": 17.5, "achievable": 10.9, "note": "applies to MXFP8 act x MXFP4 weight routed experts too (mxf8f6f4 kind runs at the FP8 rate)"},
        "nvfp4_mxfp4_dense_PF": {"spec": 35.0, "spec_inference_not_dense": 50.0, "measured": 21.12, "unit": "PFLOP/s",
                                 "source": "spec: 'NVFP4 Training' 35 PF dense ('NVFP4 Inference' 50 PF, not marked dense); measured: flashinfer mm_fp4 auto 16384^3 21120 TF, "
                                           "torch NVFP4 8192^3 17549 TF, cudnn 13.2-13.5 PF, cutlass 7.4-11.1 PF, trtllm 5.6-6.5 PF (c1); torch MXFP4 _scaled_mm unsupported (error)",
                                 "sol": 35.0, "achievable": 21.1, "achievable_frac_of_spec": 0.60,
                                 "note": "MXFP4 (e8m0 / 32) assumed equal to NVFP4 rate (same mxf4nvf4 MMA kind); not separately measured."},
        "fp32_cuda_core_TF": {"spec": 130.0, "derived": 131.6, "unit": "TFLOP/s",
                              "source": "spec: NVL72 table FP32 130 TF/GPU; derived 212 SM x 128 FP32 lanes x 2 FLOP (FMA) x 2.424 GHz = 131.6 TF. Not measured.",
                              "sol": 130.0, "achievable": None,
                              "note": "Use for KDA FP32 recurrence / conv (non-tensor-core). FP32 SGEMM via tensor-core emulation exists on the spec sheet but KDA state math is FFMA."},
        "fp64_TF": {"spec": 33.0, "unit": "TFLOP/s", "source": "spec table", "sol": 33.0},
        "per_sm_per_clk_derived": {
            "note": "derived from spec / (212 SM x clock): BF16 4.0 PF -> 8192 FLOP/clk/SM at 2.30 GHz; FP8 17.5 PF -> 32768 FLOP/clk/SM at 2.52 GHz; "
                    "NVFP4 35 PF -> 65536 FLOP/clk/SM at 2.52 GHz. The spec clock implied (~2.3-2.5 GHz) brackets the measured max 2.424 GHz; treat these as consistency checks only."},
    },
    "memory": {
        "hbm_capacity_spec_GB": {"spec": 288, "unit": "GB", "source": "NVL72 spec table: 288 GB HBM4 per Rubin GPU", "kind": S},
        "hbm_total_GiB": {"measured": 279.81, "unit": "GiB", "source": "nvidia-smi -q FB Total 286524 MiB (Reserved 1297 MiB, Free 285228 MiB) on all 4 GPUs (smi-q-913033.txt)"},
        "hbm_torch_total_GiB": {"measured": 278.54, "unit": "GiB",
                                "source": "vLLM startup 'Free memory on device (272.48-272.51/278.54 GiB) on startup' in every engine log of results/k31vr-frozen-sweep-20261114/runs"},
        "hbm_free_at_vllm_startup_GiB": {"measured": 272.48, "unit": "GiB",
                                         "source": "same log line; ~6.06 GiB taken by CUDA context / NCCL / other processes before the engine profiles",
                                         "note": "Use 272.48 GiB as the per-GPU budget for weights + activations + KV in the capacity model (then apply gpu_memory_utilization)."},
        "hbm_bw_TBps": {"spec": 22.0, "unit": "TB/s",
                        "source": "spec: NVL72 table 22 TB/s HBM4 per GPU (rack 1,580 TB/s / 72 = 21.9); derived from attributes 16128-bit bus x 4.752 GHz x 2 / 8 = 19.2 TB/s (attribute-based, unverified meaning)",
                        "measured_full_chip_read": 13.93,
                        "measured_full_chip_read_source": "hwbench triton read 8 GiB, 4 CTA/SM blk 4096: 13.927 TB/s; 8 CTA/SM 13.86 (c1); l2bench 8 GiB 13.75 (c5); "
                                                          "megamoe_rubin/die_local.py '~13.9 TB/s'; dies.py '~12.6-13.6 TB/s'",
                        "measured_copy_read_plus_write": 13.88, "measured_copy_source": "torch copy_ 8 GiB, (read+write)/t (c1); kda/verify_loop.py '~12.5-13 TB/s copy rate'",
                        "measured_write_only": 8.40, "measured_write_source": "torch fill_ 8 GiB (c1) - fill kernel, likely not a write-BW ceiling",
                        "measured_die_local_read": 17.1,
                        "measured_die_local_source": "megamoe_rubin/die_local.py: '~17.1 TB/s when every SM reads its own die (ledger row 10, measured on VR200)'; dies.py says ~16.5 TB/s",
                        "sol": 22.0, "achievable_default_alloc": 13.9, "achievable_die_local": 17.1,
                        "note": "Default allocations interleave the two dies every 4 KiB, so a die-oblivious stream pulls ~half its bytes over the die-to-die link -> 13.9 TB/s. "
                                "Die-local placement (cuMemCreate locality domain) -> 16.5-17.1 TB/s (78% of spec). SOL model: use 22 TB/s as spec bound, "
                                "17.1 (die-local) as the achievable ceiling for kernels that can place data per die (weights, experts, KV pages), 13.9 otherwise."},
        "die_to_die": {"cold_miss_local_ns": 290, "cold_miss_remote_die_ns": 455,
                       "source": "specialized_decode/cutedsl/common/dies.py docstring (latency probe)",
                       "note": "die-to-die bandwidth not separately measured; implied by 13.9 vs 17.1 TB/s"},
        "l2_size_MiB": {"measured": 126.0, "bytes": 132120576, "per_die_MiB": 63.0,
                        "source": "cudaDevAttrL2CacheSize = 132120576 (c1); per die from dies.py ('each die has its own L2') assuming an even split",
                        "persisting_max_MiB": 78.75, "persisting_source": "cudaDevAttrMaxPersistingL2CacheSize = 82575360"},
        "l2_read_bw_TBps": {"measured": 35.0, "unit": "TB/s",
                            "source": "l2bench.py repeated-read (int32, blk 1024, 8 CTA/SM): 16-64 MiB footprint 30.3-35.4 TB/s; 96 MiB 26.3; 128 MiB 18.4; >=192 MiB HBM-bound (c5)",
                            "note": "Effective L2-resident footprint for full-rate reuse ~64 MiB (one die's L2 worth, default die interleave), not 126 MiB."},
        "l2_round_trip_ns": {"measured": 169, "source": "common/primitives.py docstring: one warp, L2-resident dependent round trip relaxed.gpu 169.0 ns, relaxed.sys 168.2 ns; 16 in flight 380 ns"},
        "smem_per_sm_KiB": {"measured": 228, "bytes": 233472, "per_block_optin_bytes": 232448, "source": "cudaDevAttrMaxSharedMemoryPerMultiprocessor / PerBlockOptin (c1)"},
        "regs_per_sm": {"measured": 65536, "source": "c1 cuda_attrs"},
        "lpddr5x_per_superchip_TB": {"spec": 1.5, "source": "NVL72 spec table: 1.5 TB LPDDR5X per Vera CPU (2 GPUs); NVLink-C2C 1.8 TB/s per superchip",
                                     "note": "relevant only for KV offload/Mooncake host tier"},
        "skinny_gemm_weight_stream": {
            "source": "hwbench.py skinny GEMMs K=7168 N=16384 (c1)",
            "bf16_M16_M64_M128_us": [24.11, 24.06, 24.02], "bf16_weight_TBps": 9.76,
            "fp8_rowwise_M16_M64_M128_us": [16.10, 16.16, 16.33], "fp8_weight_TBps": 7.27,
            "note": "cuBLAS decode-shape GEMMs stream weights at only 9.8 TB/s (70% of 13.9). In-tree CuTe kernels do better; the SOL model should use HBM SOL, not these."},
        "pointer_chase_unreliable": {"data": chase, "note": "c5 l2bench pointer chase: 4 MiB/4 KiB stride 833 ns, 64 MiB 332 ns, 2 GiB/1 MiB stride 88 ns - inconsistent "
                                                          "(TLB effects / short chain); do not use. Use dies.py 290/455 ns cold-miss and primitives.py 169 ns L2 numbers."},
    },
    "nvlink": {
        "per_gpu_bw_spec_TBps_bidir": {"spec": 3.6, "unit": "TB/s", "source": "NVL72 spec table: NVLink 6 3.6 TB/s per GPU (rack 260 TB/s switch bandwidth); bidirectional"},
        "per_gpu_bw_spec_TBps_per_direction": {"spec": 1.8, "unit": "TB/s", "source": "3.6 / 2 (36 links x 50 GB/s per direction)", "sol": 1.8},
        "symm_mem_remote_write_TBps": {"measured": 1.094, "unit": "TB/s",
                                        "source": "symm2.py (c6): one GPU (rank 0) SM-driven 16-B stores to one peer's symmetric buffer, 256 MiB: pair r0-r1 1.094, r0-r4 (other node) 1.092 TB/s at 8 CTA/SM; "
                                                  "ring_all (every GPU writes its neighbour) 1.07 TB/s",
                                        "achievable": 1.09, "frac_of_spec_dir": 0.61},
        "symm_mem_remote_read_TBps": {"measured": 1.072, "unit": "TB/s",
                                       "source": "symm2.py (c6): pair 1.06-1.07 TB/s at >=4 CTA/SM (1 CTA/SM: 0.28); ring_all 0.97 TB/s",
                                       "note": "reads need >=4 CTA/SM of in-flight loads to saturate"},
        "nvlink_ingress_achievable_TBps": {"measured": 1.25, "unit": "TB/s",
                                            "source": "nvidia/ops/sp_all_gather.py docstring ('the 1.25 TB/s NVLink ingress'); provenance of 1.25 not in the bench logs here",
                                            "note": "Use 1.25 TB/s as achievable per-GPU NVLink ingress (many-to-one pushes); 1.09 measured for a single pair. "
                                                    "SP all-gather push kernel reaches 0.98 TB/s (17.4K tokens TP4, 191 us) vs NCCL ring 0.40 TB/s."},
        "nccl_large_msg_busbw_GBps": {
            "source": "ncclbench.py (c4), NVLS enabled, MNNVL, eager, 256 MiB",
            "n4": {"all_gather": 406.3, "all_to_all": 310.4, "all_reduce": 489.0},
            "n8": {"all_gather": 449.6, "all_to_all": 361.1, "all_reduce": 512.4},
            "n16": {"all_gather": 445.1, "all_to_all": 386.3, "all_reduce": 864.4},
            "note": "NCCL busbw (nccl-tests convention: AG/A2A x(n-1)/n, AR x2(n-1)/n) tops out at 0.31-0.45 TB/s for AG/A2A - only ~25-35% of the 1.25 TB/s achievable "
                    "ingress; NCCL is far from SOL for bulk collectives. AR n16 864 GB/s uses NVLS (switch reduction). SOL model should model comm at the "
                    "symmetric-memory push rate (1.09-1.25 TB/s per GPU) and keep NCCL numbers only as the 'stock' reference."},
        "p2p_nccl_send_recv": {"data": p2p, "source": "ncclbench.py p2p ping-pong (c4)",
                               "note": "64 MiB one-way 178 GB/s (NCCL send/recv, few channels); 4 KiB round trip 41-44 us."},
    },
    "latency": {
        "kernel_launch_eager_us": {"measured": 2.27, "source": "hwbench.py: eager tiny kernel (z.add_) back-to-back, 2000 iters (c1)"},
        "cuda_graph_node_us": {"measured": 0.78, "source": "hwbench.py: 1000 tiny kernels in one CUDA graph, per node (c1)",
                               "note": "Per-kernel floor inside a graph without PDL. Use as the per-kernel fixed overhead in the SOL model's launch term."},
        "pdl_dependent_release_us": {"measured": [0.73, 0.75], "with_unfenced_nvlink_stores": [0.90, 0.96],
                                      "source": "specialized_decode/cutedsl/oproj/prepared.py docstring (1-GPU microbench: dependent released 0.73-0.75 us after last CTA exit; 0.90-0.96 with multimem stores in flight)"},
        "symm_mem_flag_oneway_us": {"measured": 2.86, "range": [min(steady_oneway), max(steady_oneway)],
                                     "source": "symm2.py ping-pong (c6), sys-scope release xchg / acquire poll, 1 warp, rank0 <-> peers 1..15 (intra- and inter-node), steady reps 1-2; "
                                               "rep0 134-247 us is first-touch / mapping warmup, ignore",
                                     "note": "Per-hop signal latency for push_in/push_out style handoffs over NVLink (one-way, flag only, no payload). Same for all peers in the 16-GPU block."},
        "symm_mem_allreduce_us": {"data": dict(symm_ar),
                                  "source": "symmbench.py (c5) torch.ops.symm_mem one_shot / two_shot / multimem all-reduce, eager (includes ~2.3 us launch), bf16",
                                  "note": "multimem (NVLS) AR n16 16 KiB-512 KiB: 15.1-16.3 us flat; n4: 14.7-19.2 us. Lower bound for an in-graph fused AR: "
                                          "drafter Lamport AR+RMSNorm ~13 us per site at B1 TP16 stock (specialized_decode/cutedsl/draft/allreduce.py)."},
        "nccl_small_msg_us": {
            "source": "ncclbench.py (c4, NVLS on, default proto) eager + CUDA-graph; symmbench MODE=nccl variants nvls0/LL/dbg (c5) eager",
            "summary": {
                "n4": {"all_gather": "21.5-24.4 us (16 KiB-1 MiB)", "all_to_all": "22.3-30.8 us", "all_reduce": "21.5-26.7 us"},
                "n8": {"all_gather": "21.7-31.0 us", "all_to_all": "22.8-42.5 us", "all_reduce": "33.3-38.1 us"},
                "n16": {"all_gather": "36.8-41.5 us", "all_to_all": "23.8-88.7 us (64-89 us at >=512 KiB)", "all_reduce": "47.6-66.7 us"},
                "graph_overhead": "CUDA-graph-captured NCCL is slower than eager here: n4 AG 36.5 vs 23 us, AR 41.5-45 vs 22-24 us; n16 AG 55-58 vs 37-40, AR 66-82 vs 48-64",
            },
            "note": "NCCL small-message latency (~22 us n4, ~40-65 us n16) is 8-20x the 2.9 us NVLink flag latency; the SOL model should use symmetric-memory push "
                    "latency (alpha ~3 us per dependent hop + bytes / 1.09-1.25 TB/s) for new methods, NCCL only for stock paths."},
    },
    "tables": {"nccl_c4_nvls": dict(nccl), "nccl_c5_variants": {k: dict(v) for k, v in nccl_variants.items()},
               "l2_read_c5": l2_read, "symm_pingpong_c6": pingpong, "symm_remote_bw_c6": remote_bw},
    "sol_model_defaults": {
        "_note": "What the first-principles SOL model should plug in (spec bounds) and the measured derate per resource for matching microbenchmarks.",
        "bf16_tc_PF": {"sol": 4.0, "derated": 3.65},
        "fp8_mxfp8_tc_PF": {"sol": 17.5, "derated": 10.9},
        "fp4_tc_PF": {"sol": 35.0, "derated": 21.1},
        "fp32_simt_TF": {"sol": 130.0, "derated": None},
        "hbm_TBps": {"sol": 22.0, "derated_die_local": 17.1, "derated_default": 13.9},
        "l2_TBps": {"derated": 35.0},
        "nvlink_TBps_per_dir": {"sol": 1.8, "derated": 1.25, "derated_single_pair": 1.09},
        "nvlink_hop_latency_us": {"derated": 2.86},
        "graph_kernel_overhead_us": {"derated": 0.78},
        "pdl_release_us": {"derated": 0.74},
        "hbm_budget_GiB": {"value": 272.48},
        "nvlink_domain_gpus": 72,
    },
}
json.dump(hw, open(OUT, "w"), indent=1)
print("wrote", OUT, os.path.getsize(OUT))
