# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive the Kimi K3.1 wide-EP MegaMoE header from DeepGEMM's stock
``sm100_fp8_fp4_mega_moe_situ.cuh``. Each change is behind a macro (default on)
so variants can be A/B'd from one source:

K31_COMBINE_V2      combine: one 512-byte chunk per warp task, every top-k slot
                    loaded at once (16 B per lane per slot, L2 loads), so a chunk
                    costs one load latency instead of a 2-deep serial chain over
                    16 slots; same accumulation order (bitwise identical).
K31_PUSH_RED        expert-count push to peers: fire-and-forget red.relaxed.sys
                    instead of a returning atom.sys.
K31_NO_PREPULL_BAR  drop the pre-pull NVLink barrier (+ its grid sync): the pull
                    waits for every rank's per-expert counts (it already did), then
                    acquires them; SM 0's pushing threads fence.acq_rel.sys before
                    pushing, so the source-index writes of all SMs (ordered by the
                    grid sync) happen before the counts (fence -> relaxed red ->
                    ld.acquire.sys).
K31_W_PREFETCH      weight warp: prefetch the first task's whole weight + SF tiles
                    into L2 (TMA tensor prefetch), so weight bytes stream while
                    tokens are still being dispatched (the smem pipeline only holds
                    kNumStages k-blocks). Prefetching every task thrashed L2 (+16%).
K31_W_PREFETCH_AHEAD=D  weight warp: keep D k-blocks of L2 prefetch ahead of the
                    smem pipeline (more HBM bytes in flight per SM than kNumStages
                    smem stages allow; VR needs ~128 KiB in flight per SM).
K31_SKIP_ZERO_SEND  dispatch: an SM adds its per-expert token count to the global send counter
                    only when it is non-zero (stock: every SM x every expert, one returning atomic
                    each, kNumExperts / 128 serial rounds of 212-way contended atomics); SM 0's
                    count push then states the SM arrival count (kNumSMs) itself.
K31_BAR_ALL_CTAS    pre-pull NVLink barrier: every SM polls the local signal (acquire.sys)
                    instead of SM 0 polling and a grid sync broadcasting it (one grid sync less
                    on the dispatch critical path). The barrier phase is read at dispatch entry,
                    before the dispatch grid sync that precedes SM 0's counter update.
K31_DECODE_SHAPED   scheduler: DeepGEMM's decode-shaped order (all L1 tasks before L2
                    unless the ring would wrap, for BLOCK_M <= 64).
Usage: make_wep_header.py <stock .cuh> <out .cuh> [MACRO=0 ...] [--trace]
"""
import subprocess
import sys

src, dst = sys.argv[1], sys.argv[2]
flags = {"K31_COMBINE_V2": 1, "K31_PUSH_RED": 1, "K31_NO_PREPULL_BAR": 0, "K31_W_PREFETCH": 0,
         "K31_W_PREFETCH_AHEAD": 0, "K31_DECODE_SHAPED": 0,
         "K31_SKIP_ZERO_SEND": 1, "K31_BAR_ALL_CTAS": 1}
trace = False
for a in sys.argv[3:]:
    if a == "--trace":
        trace = True
    else:
        k, v = a.split("=")
        assert k in flags, k
        flags[k] = int(v)

s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:200], s.count(old))
    s = s.replace(old, new)


# ---- macros
defs = "".join(f"#ifndef {k}\n#define {k} {v}\n#endif\n" for k, v in flags.items())
sub("namespace deep_gemm {\n", "// Kimi K3.1 wide-EP MegaMoE (vLLM, derived from DeepGEMM's stock header; see make_wep_header.py)\n"
    + defs + "\nnamespace deep_gemm {\n")

# ---- K31_PUSH_RED
sub("""                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);""", """#if K31_PUSH_RED
                // Fire-and-forget (relaxed, like the stock atom); ordering comes from the barrier signal or,
                // without the barrier, from the fence.acq_rel.sys before this loop
                asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::
                    "l"(sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx)),
                    "l"(expert_status) : "memory");
#else
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
#endif""")

# ---- K31_NO_PREPULL_BAR
sub("""            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);""", """#if K31_NO_PREPULL_BAR
            asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);""")
sub("""        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
""", """#if not K31_NO_PREPULL_BAR
        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
#endif
""")
sub("""        // Wait token data arrival
        scheduler.fetch_expert_recv_count();
""", """        // Wait token data arrival
        scheduler.fetch_expert_recv_count();
#if K31_NO_PREPULL_BAR
        // Every rank's count push is now visible: acquire it (each lane its experts), so the source indices
        // and the senders' input buffers ordered before the pushes are visible to the pulls below. Every rank
        // pushes a count for every expert, so all ranks have entered this launch (no combine-buffer reuse hazard).
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExpertsPerRank; i += 32)
            (void) ptx::ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(i));
        __syncwarp();
#endif
""")
sub("\nnamespace deep_gemm {\n", "\nnamespace deep_gemm {\n")

# ---- K31_SKIP_ZERO_SEND
sub("""        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }""", """#if K31_SKIP_ZERO_SEND
        // Only experts this SM routes tokens to need an offset; SM 0 states the SM arrival count when pushing
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint32_t num_sm_tokens = shared_storage.expert_token_count[i];
            if (num_sm_tokens > 0) {
                shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(workspace.get_expert_send_count_ptr(i), static_cast<uint64_t>(num_sm_tokens)));
            }
        }
#else
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
#endif""")
sub("""                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(""", """#if K31_SKIP_ZERO_SEND
                const auto expert_status = (static_cast<uint64_t>(kNumSMs) << 32) |
                    (*workspace.get_expert_send_count_ptr(i) & 0xffffffffull);
#else
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
#endif
                *sym_buffer.map(""")

# ---- K31_BAR_ALL_CTAS
sub("""        // Count experts' tokens
        read_topk_idx(""", """#if K31_BAR_ALL_CTAS
        // Pre-pull barrier phase, read before the dispatch grid sync (SM 0 advances the counter only after it)
        const uint32_t k31_nvl_status = *reinterpret_cast<volatile uint32_t*>(workspace.get_nvl_barrier_counter_ptr()) & 3;
#endif
        // Count experts' tokens
        read_topk_idx(""")
sub("""#if not K31_NO_PREPULL_BAR
        // Barrier before pulling
""", """#if not K31_NO_PREPULL_BAR
#if K31_BAR_ALL_CTAS
        // Barrier before pulling: SM 0 signals every rank (release.sys, after its count pushes); every SM then
        // acquires the local signal itself, so no grid sync is needed to broadcast the arrival
        {
            const auto signal_phase = k31_nvl_status & 1, signal_sign = k31_nvl_status >> 1;
            auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);
            if (sm_idx == 0) {
                if (thread_idx < kNumRanks)
                    ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                if (thread_idx == 0)
                    ptx::red_add(workspace.get_nvl_barrier_counter_ptr(), 1);
            }
            if (thread_idx == 0) {
                const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                comm::wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                    printf("DeepGEMM K31 pre-pull barrier timeout: rank=%d, sm=%u, signal=%d, target=%d\\n",
                           sym_buffer.rank_idx, sm_idx, ptx::ld_acq_sys(signal_ptr), target);
                });
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        }
#else
        // Barrier before pulling
""")
sub("""            /* After the NVLink barrier, there is a grid sync */ true
        );
#endif
""", """            /* After the NVLink barrier, there is a grid sync */ true
        );
#endif
#endif
        // K31: pre-pull barrier done
""")

# ---- K31_W_PREFETCH
sub("""            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;
""", """            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;
#if K31_W_PREFETCH
            // First task only: prefetch its whole weight and SF tiles into L2 (same boxes as the loads below)
            // while the tokens are still being dispatched (the smem pipeline holds only kNumStages of them)
            if (__k31_first_task and not task_info.is_shared() and cute::elect_one_sync()) {
                constexpr uint32_t kPackFactor = get_smem_pack_factor<weight_dtype_t>();
                constexpr uint32_t kAtomStorage = tma::get_inner_block_atom_size<BLOCK_K, kSwizzleBMode, kPackFactor, weight_dtype_t>();
                constexpr uint32_t kAtomLogical = kAtomStorage * kPackFactor;
                for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_K / kAtomLogical; ++ i) {
                        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
                                     :: "l"(tensor_map_b_ptr), "r"(kb * BLOCK_K + i * kAtomLogical), "r"(half_n_idx),
                                        "r"(half_idx), "r"(expert_idx) : "memory");
                    }
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 :: "l"(tensor_map_sfb_ptr), "r"(n_block_idx * BLOCK_N),
                                    "r"(task_info.local_expert_idx * shape_sfb_k + kb * (BLOCK_K / 128)) : "memory");
                }
            }
            __syncwarp();
            __k31_first_task = false;
#endif
""")
sub("""        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
""", """        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
#if K31_W_PREFETCH
        bool __k31_first_task = true;
#endif
""")

# ---- K31_W_PREFETCH_AHEAD
sub("""            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset""", """#if K31_W_PREFETCH_AHEAD > 0
            const auto prefetch_k_block = [&](const uint32_t& kb) {
                constexpr uint32_t kPackFactor = get_smem_pack_factor<weight_dtype_t>();
                constexpr uint32_t kAtomStorage = tma::get_inner_block_atom_size<BLOCK_K, kSwizzleBMode, kPackFactor, weight_dtype_t>();
                constexpr uint32_t kAtomLogical = kAtomStorage * kPackFactor;
                if (kb < num_k_blocks and not task_info.is_shared() and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_K / kAtomLogical; ++ i)
                        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
                                     :: "l"(tensor_map_b_ptr), "r"(kb * BLOCK_K + i * kAtomLogical), "r"(half_n_idx),
                                        "r"(half_idx), "r"(expert_idx) : "memory");
                }
                __syncwarp();
            };
            #pragma unroll
            for (uint32_t kb = 0; kb < K31_W_PREFETCH_AHEAD; ++ kb)
                prefetch_k_block(kb);
#endif
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
#if K31_W_PREFETCH_AHEAD > 0
                prefetch_k_block(k_block_idx + K31_W_PREFETCH_AHEAD);
#endif
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset""")

# ---- K31_DECODE_SHAPED
sub("""        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts>(""", """        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts, static_cast<bool>(K31_DECODE_SHAPED)>(""")

# ---- K31_COMBINE_V2
start = s.index("        // Combine: reduce top-k results and write back")
mid_anchor = "        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us\n"
mid = s.index(mid_anchor)
end_anchor = """                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else"""
end = s.index(end_anchor)
pre = s[start:mid]          # stock chunk setup + grid sync + notify + dispatch barrier
loop = s[mid:end] + """                cute::tma_store_arrive();
            }
            __syncwarp();
        }
"""
v2 = """#if K31_COMBINE_V2
        // One task = one 512-byte chunk (16 bytes per lane) of one token; every selected slot's vector is
        // loaded at once, then reduced in the stock order (shared first, then routed slots in top-k order)
        {
            constexpr uint32_t kV2HiddenBytes = kHidden * sizeof(nv_bfloat16);
            constexpr uint32_t kV2ChunkBytes = 32 * sizeof(uint4);
            constexpr uint32_t kV2NumChunks = kV2HiddenBytes / kV2ChunkBytes;
            constexpr uint32_t kV2NumSlots = kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u);
            DG_STATIC_ASSERT(kV2HiddenBytes % kV2ChunkBytes == 0, "Hidden must be a multiple of 256 BF16 elements");
            const auto grid_idx = ptx::get_grid_idx() + 1;
            for (uint32_t task_idx = epilogue_warp_idx * kNumSMs + sm_idx; task_idx < num_tokens * kV2NumChunks;
                 task_idx += kNumSMs * kNumEpilogueWarps) {
                const uint32_t token_idx = task_idx / kV2NumChunks;
                const uint32_t byte_offset = (task_idx % kV2NumChunks) * kV2ChunkBytes + lane_idx * static_cast<uint32_t>(sizeof(uint4));

                const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                    static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                    (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

                const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
                const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                    is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
                comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                                 [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\\n", sym_buffer.rank_idx, token_idx); });

                uint4 values[kV2NumSlots];
                uint32_t mask = total_mask;
                #pragma unroll
                for (uint32_t i = 0; i < kV2NumSlots; ++ i) {
                    if (mask) {
                        const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                        mask ^= 1u << slot_idx;
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx).get_data_buffer(token_idx).get_base_ptr(),
                            byte_offset);
                        values[i] = __ldcg(reinterpret_cast<const uint4*>(src_ptr));
                    }
                }
                const uint32_t num_selected = __popc(total_mask);
                float2 reduced[kNumElemsPerUint4] = {};
                #pragma unroll
                for (uint32_t i = 0; i < kV2NumSlots; ++ i) {
                    if (i < num_selected) {
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&values[i]);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[l], bf16_values[l]);
                    }
                }
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[l]);
                *reinterpret_cast<uint4*>(math::advance_ptr<uint8_t>(y, static_cast<uint64_t>(token_idx) * kV2HiddenBytes + byte_offset)) = casted;
            }
        }
#else
""" + loop + "#endif\n"
s = s[:start] + pre + v2 + s[end + len("""                cute::tma_store_arrive();
            }
            __syncwarp();
        }
"""):]
open(dst, "w").write(s)
if trace:
    here = __file__.rsplit("/", 1)[0]
    subprocess.check_call([sys.executable, f"{here}/make_trace_header.py", dst, dst])
print("wrote", dst, flags, "trace" if trace else "")
