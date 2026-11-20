"""D-GEMM round 7: per-CTA resources of the in-step Mxfp8PreGemm tactics (registers, threads,
smem stages / bytes) and the co-residency headroom next to SpAttnResGather CTAs. Compiles
without tvm-ffi so the driver function is reachable; same kernel code and tactic as prewait.py."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    _checked, resolve_kernel)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import prewait as pw

TACTICS = [
    ("kda_in R128", ((128, 64), (1, 1), False, 128)),
    ("kda_in R256", ((256, 128), (2, 1), False, 256)),
    ("kda_in R384", ((256, 192), (2, 1), True, 256)),
    ("kda_o R384", ((128, 128), (1, 1), False, 128)),
]
dev = 0
torch.cuda.set_device(dev)
A = cuda.CUfunction_attribute
for name, (tile, cluster, swap_ab, inst_m) in TACTICS:
    gemm = pw.Sm107BlockScaledPersistentDenseGemmKernel(
        pw.SF_VEC, (inst_m, tile[1], pw.INST_K), (tile[0], tile[1], pw.TILER_K), cluster,
        prefetch_dist=0, use_pdl=True, prewait_stages=-1, weight_is_a=swap_ab)
    gemm.max_ab_stages = 6 if name == "kda_in R384" else 0
    sm, sk, sn = cute.sym_int(), cute.sym_int(), cute.sym_int()
    f8 = cutlass.Float8E4M3FN
    a = cute.runtime.make_fake_compact_tensor(f8, (sm, sk), stride_order=(1, 0), assumed_align=16)
    b = cute.runtime.make_fake_compact_tensor(f8, (sn, sk), stride_order=(1, 0), assumed_align=16)
    if swap_ab:
        c = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (sn, sm), stride_order=(0, 1), assumed_align=16)
    else:
        c = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (sm, sn), stride_order=(1, 0), assumed_align=16)
    sfa = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    sfb = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=8)
    mac = pw._max_active_clusters(cluster[0] * cluster[1])
    try:
        code = cute.compile(gemm.wrapper, a, b, c, cutlass.Int64(1), cutlass.Int64(1), cutlass.Int64(1), 1,
                            sfa, sfb, alpha, mac, make_fake_stream(), swap_ab,
                            options=f"--opt-level 2 --gpu-arch={pw._arch(dev)}")
        _, fn = resolve_kernel(code, dev)
        regs = _checked(cuda.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, fn))[0]
        sh = _checked(cuda.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn))[0]
        mx = _checked(cuda.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fn))[0]
        thr = gemm.threads_per_cta
        print(f"{name}: regs {regs} x {thr} thr = {regs * thr} / 65536 per SM; static smem {sh}; "
              f"max dyn attr {mx}; ab stages {getattr(gemm, 'num_ab_stage', None)} prewait {gemm.prewait_stages} "
              f"smem_capacity {gemm.smem_capacity}", flush=True)
    except Exception as error:
        print(f"{name}: FAILED {type(error).__name__}: {str(error)[:300]}", flush=True)
