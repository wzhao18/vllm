# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES probe: is the prefill AttnRes (specialized_prefill attn_res_mxfp8)
load-bound or consumer-bound? Builds probe copies of the real kernel source:
  base    the kernel as shipped
  nomath  consumers release each chunk right after it lands (loads + epilogue
          stores only, no per-source math)
  noload  the producer issues no copies (arrive with 0 tx): math + stores only
CUPTI kernel time per variant and shape at F16K rows."""

import argparse
import hashlib
import json
import os

import torch

from pgemm_bench import kernel_times  # noqa: E402

H = 7168
EPS = 1e-6


def build(variant: str, depth: int, src_path: str):
    from torch.utils.cpp_extension import load

    import vllm.envs as envs

    src = open(src_path).read()
    if variant in ("nomath",):
        a = "mbarrier_wait(plan.bar_ready[chunk_slot], pr);"
        assert src.count(a) == 1
        src = src.replace(a, a + "\n if (lane == 0) mbarrier_arrive("
                          "plan.bar_consumed[chunk_slot]); continue;")
    if variant in ("noload",):
        a = "mbarrier_expect_tx(plan.bar_ready[chunk_slot], transaction_bytes);"
        assert src.count(a) == 1
        src = src.replace(a, "mbarrier_expect_tx(plan.bar_ready[chunk_slot], 0);")
        for b in ("cp_async_bulk(buf_ptr(slot),",
                  "cp_async_bulk(delta_buf_ptr(chunk_slot),"):
            assert src.count(b) == 1, b
            src = src.replace(b, "if (false) " + b)
    tag = hashlib.sha256((src + variant + str(depth) + torch.__version__)
                         .encode()).hexdigest()[:16]
    d = os.path.join(envs.VLLM_CACHE_ROOT, "k31pf_attnres_probe", tag)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "k.cu")
    if not os.path.exists(p):
        with open(p + ".tmp", "w") as f:
            f.write(src)
        os.replace(p + ".tmp", p)
    return load(
        name=f"k31pf_attnres_probe_{tag}", sources=[p], extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", f"-DK31PF_CHUNK_DEPTH={depth}",
            "-gencode", "arch=compute_100f,code=sm_100f"],
        build_directory=d, verbose=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--variants", default="base,nomath,noload")
    p.add_argument("--depths", default="2,3")
    args = p.parse_args()
    import vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 as am

    T = args.rows
    blocks = torch.randn(T, 8, H, device="cuda").bfloat16()
    prefix = torch.randn(T, H, device="cuda").bfloat16()
    delta = torch.randn(T, H, device="cuda").bfloat16()
    w = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
    qk = (torch.randn(H, device="cuda") * H**-0.5).bfloat16()
    q = torch.empty(T, H, dtype=torch.uint8, device="cuda")
    sf = torch.empty((T + 127) // 128 * 128 * (H // 32), dtype=torch.uint8,
                     device="cuda")
    out = torch.empty_like(prefix)
    for depth in [int(x) for x in args.depths.split(",")]:
        mods = {v: build(v, depth, am._SRC) for v in args.variants.split(",")}
        for nb, bf in ((8, True), (5, True), (2, True), (5, False)):
            row = {"depth": depth, "nb": nb, "bf16_out": bf}
            rd = (nb + 2) * T * H * 2
            wr = T * H * 2 * (1 + int(bf)) + T * H * 33 // 32
            row["MB"] = round((rd + wr) / 1e6)
            for v, m in mods.items():
                us = kernel_times(lambda: m.attn_res_mxfp8(
                    prefix, delta, blocks, w, qk, w, out if bf else None, q, sf,
                    nb, -1, EPS, EPS))[0]
                row[v] = round(us, 1)
                row[v + "_TBps"] = round((rd + wr) / us / 1e6, 2)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
