# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instrument a DeepGEMM FP8xFP4 SiTU MegaMoE header with per-SM %globaltimer stamps.

The stamps go to the last 8 token rows of the input token buffer (unused when
num_tokens <= num_max_tokens_per_rank - 8): uint64 [kNumSMs][16]. Slots:
 0 entry  1 after PDL wait  2 dispatch counts+offsets  3 src-idx + grid sync
 4 after pre-pull NVLink barrier  5 dispatch pull end (warp 0)  6 recv counts seen
 7 weight warp first task  8 weight warp done  9 MMA warp done  10 epilogue loop done
 11 after epilogue grid sync  12 combine done  13 cleanup barrier done
 14 token warp first L1 tokens ready  15 epilogue first accumulator ready
Usage: make_trace_header.py <stock .cuh> <out .cuh>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

TS = ("{ uint64_t __t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(__t)); "
      "buffer.input_token_buffer.get_data_buffer(kNumMaxTokensPerRank - 8).template get_base_ptr<uint64_t>()"
      "[sm_idx * 16 + {slot}] = __t; }")


def ts(slot, cond="thread_idx == 0"):
    return f"if ({cond}) " + TS.replace("{slot}", str(slot))


def ins_after(anchor, text, count=1):
    global s
    assert s.count(anchor) == count, (anchor, s.count(anchor))
    sep = "" if anchor.endswith("\n") else "\n"
    s = s.replace(anchor, anchor + sep + text + "\n")


def ins_before(anchor, text, count=1):
    global s
    assert s.count(anchor) == count, (anchor, s.count(anchor))
    s = s.replace(anchor, text + "\n" + anchor)


# buffer is declared after the TMA prefetch; put the entry stamp right after it
ins_after("    const auto workspace = buffer.workspace;\n", "    " + ts(0))
ins_after("    cudaGridDependencySynchronize();\n", "    " + ts(1))
ins_before("""        // Write source indices (~2 us with 512 tokens)""", "        " + ts(2))
ins_after("""            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
""", "        " + ts(3))
if "        // K31: pre-pull barrier done\n" in s:
    ins_after("        // K31: pre-pull barrier done\n", "        " + ts(4))
else:
    ins_after("""            /* After the NVLink barrier, there is a grid sync */ true
        );
""", "        " + ts(4))
ins_after("""        // Wait token data arrival
        scheduler.fetch_expert_recv_count();
""", "        " + ts(6))
ins_before("""        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
""", "        " + ts(5))
ins_after("""            /* At the end of kernel does not need to sync */ false
        );
""", "        " + ts(13))
# weight warp
ins_after("""        // GEMM TMA load warp for weights with SF
        task_info_t task_info;""", "        bool __first = true;")
ins_after("""            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
""", "            if (__first) { __first = false; " + ts(7, "cute::elect_one_sync()") + " }")
ins_before("""    } else if (warp_idx == kNumDispatchWarps + 2) {""", "        " + ts(8, "cute::elect_one_sync()"))
# token warp: first L1 full
ins_after("""                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
""", "                if (__tok_first) { __tok_first = false; " + ts(14, "cute::elect_one_sync()") + " }")
ins_after("""        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;""", "        bool __tok_first = true;")
# MMA warp done
ins_before("""            // To safely deconstruct barriers, we need another round of waits""", "            " + ts(9, "cute::elect_one_sync()"))
# epilogue
ins_after("""        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
""", "            if (current_iter_idx == 1) " + ts(15, "epilogue_thread_idx == 0").replace("if (", "{ if (") + " }")
ins_before("""        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs""", "        " + ts(10, "epilogue_thread_idx == 0"))
ins_before("""        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below""",
           "        " + ts(11, "epilogue_thread_idx == 0"))
# combine end: at the end of the epilogue branch
anchor = """    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)"""
assert s.count(anchor) == 1
s = s.replace(anchor, "        " + ts(12, "epilogue_thread_idx == 0") + "\n" + anchor)
open(dst, "w").write(s)
print("wrote", dst)
