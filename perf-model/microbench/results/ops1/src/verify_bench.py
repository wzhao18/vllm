"""ops1 follow-up checks (one GPU each shard):
shard 0: BF16 cuBLAS at unaligned N (6284, 3206: row stride not 16 B aligned) vs N padded to /8 (6288, 3208).
shard 1: gemm-003 (16384 x 49376 x 7168 BF16) repeated 6x with idle gaps, clocks logged (run-to-run 3.7 vs 5.0 ms).
shard 2: same as 1 for gemm-022 FP8 and gemm-029 BF16.
usage: verify_bench.py out.jsonl i/n
"""
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from mbc import Out, env_info, time_us

out = Out(sys.argv[1])
i = int(sys.argv[2].split("/")[0])
dev = torch.device("cuda")
out.row(kind="env", **env_info())


def clocks():
    try:
        return subprocess.run(["nvidia-smi", "-i", str(torch.cuda.current_device()), "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)[:60]


if i == 0:
    for M in (16384, 2048, 1024, 512, 256, 64, 32):
        for N in (6284, 6288, 3206, 3208, 12440):
            x = torch.randn(M, 7168, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, 7168, device=dev, dtype=torch.bfloat16) * 0.02
            t, m = time_us(lambda: F.linear(x, w))
            out.row(kind="verify", test="align", M=M, N=N, K=7168, us=round(t, 3), method=m,
                    tflops=round(2 * M * N * 7168 / t / 1e6, 1))
            del x, w
            torch.cuda.empty_cache()
elif i in (1, 2):
    shapes = [(16384, 49376, 7168, "bf16")] if i == 1 else [(16384, 49376, 7168, "fp8"), (16384, 7168, 12288, "bf16")]
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    for M, N, K, dt in shapes:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        if dt == "fp8":
            xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
            sa, sb, wt = xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), wq.t()
            f = lambda: torch._scaled_mm(xq, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)  # noqa: E731
        else:
            f = lambda: F.linear(x, w)  # noqa: E731
        for rep in range(6):
            time.sleep(5)
            c0 = clocks()
            t, m = time_us(f, repeat_ms=200)
            out.row(kind="verify", test="repeat", M=M, N=N, K=K, dtype=dt, rep=rep, us=round(t, 3), method=m,
                    clocks_before=c0, clocks_after=clocks(), tflops=round(2 * M * N * K / t / 1e6, 1))
