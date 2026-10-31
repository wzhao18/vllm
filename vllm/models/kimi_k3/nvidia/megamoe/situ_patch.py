# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 wide-EP decode changes to DeepGEMM's FP8 x FP4 SiTU MegaMoE kernel.

``patch_situ_header`` derives a header from DeepGEMM's stock
``deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh`` (installed with ``overlay``).
Every change keeps the kernel's arithmetic and accumulation order, so the output
is bitwise identical to the stock kernel; each is behind a macro (default 1):

K31_COMBINE_V2      combine: one 512-byte chunk of one token per warp task, every
                    selected top-k slot loaded at once (16 B per lane per slot), then
                    reduced in the stock order (shared first, routed slots in top-k
                    order). The stock loop is a 2-deep serial load chain over the slots.
K31_PUSH_RED        expert-count push to peers: fire-and-forget ``red.relaxed.sys``
                    instead of a returning ``atom.sys`` (kNumExperts / 128 NVLink round
                    trips on SM 0 before the pre-pull barrier).
K31_SKIP_ZERO_SEND  dispatch offsets: an SM adds its per-expert token count to the
                    global send counter only when it is non-zero. Stock: every SM does
                    one returning, 212-way contended atomic per expert (kNumExperts =
                    all 896 experts at EP16/EP32). SM 0's count push then states the SM
                    arrival count (kNumSMs) that the receivers wait for.
K31_BAR_ALL_CTAS    pre-pull NVLink barrier: every SM acquires the local signal itself
                    instead of SM 0 polling it and a grid sync broadcasting it. The
                    barrier phase is read at dispatch entry, before the dispatch grid
                    sync that precedes SM 0's counter update.
K31_DECODE_SHAPED   scheduler: DeepGEMM's decode-shaped task order (claim every L1 task
                    before any L2 task while each pool block has its own ring slot,
                    BLOCK_M <= 64), so no L1 task waits behind an L2 consumer and the
                    tail is short, nearly-ready L2 tasks. Applied only when the installed
                    scheduler has the ``kDecodeShaped`` parameter.

``PatchError`` is raised when the stock header does not have the expected text
(another DeepGEMM version): the caller keeps the stock kernel.
"""

from __future__ import annotations

FLAGS = ("K31_COMBINE_V2", "K31_PUSH_RED", "K31_SKIP_ZERO_SEND", "K31_BAR_ALL_CTAS",
         "K31_DECODE_SHAPED")
STOCK_HEADER = "deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh"
SCHEDULER_HEADER = "deep_gemm/scheduler/mega_moe.cuh"
MARKER = "Kimi K3.1 wide-EP MegaMoE"


class PatchError(RuntimeError):
    pass


class _Patcher:
    def __init__(self, text: str) -> None:
        self.s = text

    def sub(self, old: str, new: str) -> None:
        n = self.s.count(old)
        if n != 1:
            raise PatchError(f"expected 1 match, found {n}: {old[:120]!r}")
        self.s = self.s.replace(old, new)


def _push_red(p: _Patcher) -> None:
    p.sub(
        """                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);""",
        """#if K31_PUSH_RED
                // Fire-and-forget (relaxed, like the stock atom); the pre-pull barrier signal (release.sys) orders it
                asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::
                    "l"(sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx)),
                    "l"(expert_status) : "memory");
#else
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
#endif""",
    )


def _skip_zero_send(p: _Patcher) -> None:
    p.sub(
        """        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }""",
        """#if K31_SKIP_ZERO_SEND
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
#endif""",
    )
    p.sub(
        """                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(""",
        """#if K31_SKIP_ZERO_SEND
                const auto expert_status = (static_cast<uint64_t>(kNumSMs) << 32) |
                    (*workspace.get_expert_send_count_ptr(i) & 0xffffffffull);
#else
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
#endif
                *sym_buffer.map(""",
    )


def _bar_all_ctas(p: _Patcher) -> None:
    p.sub(
        """        // Count experts' tokens
        read_topk_idx(""",
        """#if K31_BAR_ALL_CTAS
        // Pre-pull barrier phase, read before the dispatch grid sync (SM 0 advances the counter only after it)
        const uint32_t k31_nvl_status = *reinterpret_cast<volatile uint32_t*>(workspace.get_nvl_barrier_counter_ptr()) & 3;
#endif
        // Count experts' tokens
        read_topk_idx(""",
    )
    p.sub(
        """        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
""",
        """#if K31_BAR_ALL_CTAS
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
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
#endif
        // K31: pre-pull barrier done
""",
    )


_COMBINE_V2 = """#if K31_COMBINE_V2
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
"""


def _decode_shaped(p: _Patcher) -> None:
    p.sub(
        """        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts>(""",
        """        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts, static_cast<bool>(K31_DECODE_SHAPED)>(""",
    )


def _combine_v2(p: _Patcher) -> None:
    s = p.s
    start_anchor = "        // Combine: reduce top-k results and write back"
    mid_anchor = "        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us\n"
    end_anchor = """                cute::tma_store_arrive();
            }
            __syncwarp();
        }
"""
    for a in (start_anchor, mid_anchor, end_anchor):
        if s.count(a) != 1:
            raise PatchError(f"combine anchor not unique: {a[:80]!r}")
    mid, end = s.index(mid_anchor), s.index(end_anchor) + len(end_anchor)
    if not s.index(start_anchor) < mid < end:
        raise PatchError("combine anchors out of order")
    # The stock token loop stays under #else
    p.s = s[:mid] + _COMBINE_V2 + s[mid:end] + "#endif\n" + s[end:]


def patch_situ_header(
    text: str, flags: dict[str, int] | None = None, scheduler_text: str | None = None
) -> str:
    """The Kimi K3.1 header derived from the stock ``text`` (see module doc).
    ``flags`` overrides the macro defaults (all 1). ``scheduler_text`` is the
    installed ``SCHEDULER_HEADER``; without ``kDecodeShaped`` in it (or without it)
    the scheduler order stays stock."""
    if MARKER in text:
        raise PatchError("header is already patched")
    vals = {k: 1 for k in FLAGS}
    for k, v in (flags or {}).items():
        if k not in vals:
            raise KeyError(k)
        vals[k] = int(v)
    p = _Patcher(text)
    _push_red(p)
    _skip_zero_send(p)
    _bar_all_ctas(p)
    _combine_v2(p)
    if scheduler_text is not None and "bool kDecodeShaped" in scheduler_text:
        _decode_shaped(p)
    else:
        vals["K31_DECODE_SHAPED"] = 0
    defs = "".join(f"#ifndef {k}\n#define {k} {v}\n#endif\n" for k, v in vals.items())
    p.sub(
        "\nnamespace deep_gemm {\n",
        f"\n// {MARKER} (vLLM vllm/models/kimi_k3/nvidia/megamoe/situ_patch.py, "
        "derived from DeepGEMM's stock header)\n" + defs + "\nnamespace deep_gemm {\n",
    )
    return p.s
