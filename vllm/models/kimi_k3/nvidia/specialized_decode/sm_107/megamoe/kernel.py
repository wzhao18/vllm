# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM107 specialized decode's own FP8 x FP4 SiTU MegaMoE kernel.

``SituMegaMoE`` builds and launches the wide-EP MegaMoE kernel from the header that
``situ_patch`` derives from DeepGEMM's stock ``sm100_fp8_fp4_mega_moe_situ.cuh``.
DeepGEMM is used read-only: its headers are an include path (behind this
module's own directory, which holds the patched kernel header), and its public
API supplies the symmetric buffer, the SM count, the SM locality domains, the PDL
setting and the BLOCK_M query. DeepGEMM's JIT, include root and kernels stay
untouched: every stock DeepGEMM kernel in the process still compiles from the
installed tree.

The host side mirrors DeepGEMM's for this one kernel (pinned to the vendored
DeepGEMM; ``check`` raises when the installed DeepGEMM disagrees):

- ``csrc/jit_kernels/heuristics/mega_moe.hpp`` ``get_mega_moe_config`` (MXFP8FP4):
  tile sizes, pipeline stages, shared memory;
- ``csrc/jit_kernels/impls/runtime_utils.hpp``: the TMA descriptors;
- ``csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_situ.hpp``: the template
  instantiation, the kernel arguments and the launch (2-CTA clusters, one CTA per
  SM, PDL as DeepGEMM's runtime setting);
- deep_jit's NVCC command (``backend/cuda/options.hpp``, ``backend.hpp``; SM107
  compiles for ``sm_100f`` as DeepGEMM's ``init_jit`` does).

Same template arguments, same NVCC command and the same launch give the same
kernel as DeepGEMM would JIT from the patched header: the output is bitwise equal
to the stock kernel's (the patch keeps the arithmetic and its order).
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cuda.bindings.driver as cuda
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STOCK_HEADER = "deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh"
SCHEDULER_HEADER = "deep_gemm/scheduler/mega_moe.cuh"
KERNEL = "sm100_fp8_fp4_mega_moe_situ_impl"

# DeepGEMM constants this host code mirrors (vendored DeepGEMM).
_SMEM_CAPACITY = 232448  # SM100ArchSpec::smem_capacity
_CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192, 240)  # layout::kCandidateBlockM
_TASK_INFO_BYTES = 32  # sizeof(sched::TaskInfo<true>)
_MAX_RANKS = 72  # layout::kNumMaxRanks
_GRAN_K = 32  # MX scale group along K
_DISPATCH_THREADS, _NON_EPILOGUE_THREADS, _EPILOGUE_THREADS = 128, 128, 256
_CLUSTER = 2

# deep_jit CompilerOptions::default_options + DeepGEMM init_jit's additions, in order
_NVCC_FLAGS = (
    "-O3",
    "--compiler-options=-O3",
    "--ptxas-options=--register-usage-level=10",
    "-std=c++20",
    "--compiler-options=-fPIC",
    "--compiler-options=-fconcepts",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--diag-suppress=39,161,174,177,186,940",
    "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
)

# sm100_fp8_fp4_mega_moe_situ.hpp's instantiation source, verbatim
_SOURCE = """
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_situ_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {}
    >);
}};
"""


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _ceil_div(x: int, a: int) -> int:
    return (x + a - 1) // a


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def float_literal(v: float) -> str:
    """DeepGEMM's ``to_string(float)``: ``{sign}0x{:a}f`` of the FP32 value
    (shortest hexadecimal form, as std::format's ``a``)."""
    v = _f32(v)
    if not math.isfinite(v):
        raise ValueError(f"MegaMoE: non-finite activation scale {v}")
    mantissa, exponent = abs(v).hex()[2:].split("p")
    whole, _, fraction = mantissa.partition(".")
    fraction = fraction.rstrip("0")
    digits = whole + ("." + fraction if fraction else "")
    return f"{'-' if math.copysign(1.0, v) < 0 else ''}0x{digits}p{exponent}f"


@dataclass(frozen=True)
class SituConfig:
    """``MegaMoEConfig`` of DeepGEMM's ``get_mega_moe_config`` (MXFP8FP4)."""

    block_m: int
    block_n: int
    block_k: int
    load_block_m: int
    load_block_n: int
    store_block_m_l1: int
    store_block_m_l2: int
    sf_block_m: int
    sf_block_n: int
    num_ring_tokens: int
    num_sf_ring_tokens: int
    swizzle_acts_mode: int
    swizzle_weights_mode: int
    num_stages: int
    smem_size: int
    num_bytes_per_pull: int


def block_m_for(num_ranks: int, num_experts: int, num_topk: int, num_tokens: int) -> int:
    """``get_block_config_for_mega_moe``'s BLOCK_M (MXFP8FP4), in FP32 as the C++."""
    expected = _f32(_f32(_f32(float(num_tokens)) * num_ranks) * num_topk)
    expected = _f32(expected / num_experts)
    covered = _f32(expected + _f32(math.sqrt(expected)))
    block_m = 16 if expected <= 10 else 32
    if expected > 24:
        num_blocks = math.ceil(_f32(covered / 240))
        if num_blocks == 1 and covered > 192 and num_experts // num_ranks < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if num_blocks * candidate >= covered:
                break
    assert block_m in _CANDIDATE_BLOCK_M
    return block_m


def situ_config(
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    num_ring_tokens: int,
    num_sf_ring_tokens: int,
) -> SituConfig:
    """DeepGEMM's ``get_mega_moe_config`` for MXFP8FP4 (FP8 activations, FP4 or
    FP8 weights unpacked to 8-bit shared memory)."""
    block_m = block_m_for(num_ranks, num_experts, num_topk, num_tokens)
    store_l1 = 8 if block_m <= 16 else 16 if block_m <= 64 else 32 if block_m <= 192 else 24
    store_l2 = 8 if block_m == 240 else store_l1
    block_n, block_k = 128, 1024 // 8
    if num_ring_tokens % block_m:
        raise RuntimeError(f"MegaMoE: ring of {num_ring_tokens} tokens, BLOCK_M {block_m}")
    sf_block_m, sf_block_n = _align(block_m, 128), _align(block_n, 128)
    num_bytes_per_pull = hidden
    while num_bytes_per_pull > 8192:
        if num_bytes_per_pull % 2:
            raise RuntimeError(f"MegaMoE: odd pull size {num_bytes_per_pull}")
        num_bytes_per_pull //= 2
    # get_pipeline_config_for_mega_moe
    dispatch_warps, epilogue_warps = _DISPATCH_THREADS // 32, _EPILOGUE_THREADS // 32
    load_block_m = block_m // 2
    smem_dispatch = _align(num_experts * 4, 1024) + _align(
        num_bytes_per_pull * dispatch_warps, 1024
    )
    epilogue_warpgroups = epilogue_warps // 4
    cd_l1 = epilogue_warpgroups * store_l1 * (block_n // 2) * 2
    cd_l2 = epilogue_warpgroups * store_l2 * block_n * 2
    smem_cd = _align(max(cd_l1, cd_l2), 1024)
    smem_task_info = 2 * _TASK_INFO_BYTES
    smem_barriers = (dispatch_warps + 2 * 2 + epilogue_warps * 2 + 2 * 2) * 8
    smem_amax = store_l1 * epilogue_warps * 4
    per_stage = load_block_m * block_k + block_n * block_k
    per_stage += sf_block_m * (block_k // _GRAN_K) + sf_block_n * (block_k // _GRAN_K)
    per_stage += 2 * 8
    fixed = smem_dispatch + smem_cd + smem_amax + smem_barriers + smem_task_info + 4
    num_stages = (_SMEM_CAPACITY - fixed) // per_stage
    if num_stages < 2:
        raise RuntimeError(f"MegaMoE: {num_stages} pipeline stages")
    return SituConfig(
        block_m, block_n, block_k, load_block_m, block_n, store_l1, store_l2,
        sf_block_m, sf_block_n, num_ring_tokens, num_sf_ring_tokens, 128, 128,
        num_stages, fixed + num_stages * per_stage, num_bytes_per_pull,
    )  # fmt: skip


def instantiation(
    cfg: SituConfig,
    num_max_tokens_per_rank: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_shared_experts: int,
    num_topk: int,
    num_sms: int,
    num_ranks: int,
    alpha: float,
    beta: float,
    fast_math: bool,
    weight_dtype: torch.dtype,
) -> str:
    """The kernel source DeepGEMM's launcher would JIT for these arguments."""
    dtypes = {
        torch.int8: "cutlass::detail::float_e2m1_unpacksmem_t",  # packed FP4
        torch.float8_e4m3fn: "cutlass::float_e4m3_t",
    }
    if weight_dtype not in dtypes:
        raise RuntimeError(f"MegaMoE: unsupported routed weight dtype {weight_dtype}")
    return _SOURCE.format(
        num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        cfg.block_m, cfg.block_n, cfg.block_k,
        cfg.store_block_m_l1, cfg.store_block_m_l2,
        cfg.sf_block_m, cfg.sf_block_n,
        cfg.num_ring_tokens,
        cfg.num_sf_ring_tokens,
        cfg.num_stages,
        cfg.num_bytes_per_pull,
        _DISPATCH_THREADS, _NON_EPILOGUE_THREADS, _EPILOGUE_THREADS,
        num_sms, num_ranks,
        float_literal(alpha),
        float_literal(beta),
        "true" if fast_math else "false",
        dtypes[weight_dtype],
    )  # fmt: skip


# ----------------------------------------------------------------- build / load
def _nvcc() -> Path:
    """deep_jit's ``find_cuda_toolkit``: CUDA_HOME / CUDA_PATH, ``which nvcc``,
    /usr/local/cuda."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        found = shutil.which("nvcc")
        home = str(Path(found).parent.parent) if found else "/usr/local/cuda"
    nvcc = Path(home) / "bin" / "nvcc"
    if not os.access(nvcc, os.X_OK):
        raise RuntimeError(f"MegaMoE: NVCC is not executable: {nvcc}")
    return nvcc


def _arch(device: int) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 7):
        raise RuntimeError(f"MegaMoE (SM107): device capability {major}.{minor}")
    # DeepGEMM init_jit: the SM100 family target on SM107
    return "100f"


@cache
def _dg_include(dg_file: str) -> Path:
    include = Path(dg_file).resolve().parent / "include"
    if not (include / STOCK_HEADER).is_file():
        raise RuntimeError(f"MegaMoE: DeepGEMM has no {STOCK_HEADER} under {include}")
    return include


@cache
def _dg_digest(include: Path) -> str:
    """Content digest of DeepGEMM's ``deep_gemm/`` headers (the kernel's includes)."""
    h = hashlib.sha256(str(include).encode())
    for p in sorted((include / "deep_gemm").rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(include)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@cache
def patched_include(dg_file: str, cache_root: str) -> Path:
    """``<cache>/situ_<digest>/include``: the patched kernel header at its stock
    path; the kernel's other includes resolve to DeepGEMM's tree behind it."""
    from .situ_patch import PatchError, patch_situ_header

    include = _dg_include(dg_file)
    sched = include / SCHEDULER_HEADER
    try:
        text = patch_situ_header(
            (include / STOCK_HEADER).read_text(),
            scheduler_text=sched.read_text() if sched.is_file() else None,
        )
    except PatchError as e:
        raise RuntimeError(
            f"Kimi K3 MegaMoE (SM107): cannot derive the wide-EP kernel from "
            f"{include / STOCK_HEADER}: {e}"
        ) from e
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    root = Path(cache_root) / f"situ_{digest}" / "include"
    header = root / STOCK_HEADER
    if not header.is_file():
        header.parent.mkdir(parents=True, exist_ok=True)
        tmp = header.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(header)
    return root


def build(source: str, include_dirs: tuple[Path, ...], device: int, cache_root: str) -> Path:
    """The cubin of ``source`` (deep_jit's NVCC command), cached by content."""
    nvcc = _nvcc()
    flags = (f"--gpu-architecture=sm_{_arch(device)}", *_NVCC_FLAGS)
    version = subprocess.run(
        [str(nvcc), "--version"], check=True, capture_output=True, text=True
    ).stdout
    h = hashlib.sha256()
    for part in (source, " ".join(flags), version, *map(str, include_dirs)):
        h.update(part.encode())
        h.update(b"\0")
    h.update(_dg_digest(include_dirs[-1]).encode())
    for d in include_dirs[:-1]:
        h.update((d / STOCK_HEADER).read_bytes())
    out = Path(cache_root) / f"kernel.{KERNEL}.{h.hexdigest()[:24]}"
    cubin = out / "kernel.cubin"
    if cubin.is_file():
        return cubin
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=cache_root))
    try:
        (tmp / "kernel.cu").write_text(source)
        cmd = [str(nvcc), str(tmp / "kernel.cu"), "--cubin", "--output-file",
               str(tmp / "kernel.cubin"), *flags]  # fmt: skip
        for d in include_dirs:
            cmd += ["--include-path", str(d)]
        logger.info("Kimi K3 MegaMoE (SM107): compiling %s", out.name)
        # (deep_jit compiles from an empty directory: no same-name include picks)
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        built = tmp / "kernel.cubin"
        if res.returncode or not built.is_file() or not built.stat().st_size:
            raise RuntimeError(
                f"Kimi K3 MegaMoE (SM107): NVCC failed ({res.returncode}):\n"
                f"{' '.join(cmd)}\n{res.stdout}\n{res.stderr}"
            )
        try:
            tmp.rename(out)
        except OSError:  # another rank installed the same kernel first
            if not cubin.is_file():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cubin


def _checked(result, what: str):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Kimi K3 MegaMoE (SM107): {what} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


class _Kernel:
    """One loaded cubin (one kernel): the CUfunction of the current context."""

    def __init__(self, cubin: Path, smem: int) -> None:
        self.library = _checked(
            cuda.cuLibraryLoadData(cubin.read_bytes(), None, None, 0, None, None, 0),
            "cuLibraryLoadData",
        )
        count = _checked(cuda.cuLibraryGetKernelCount(self.library), "kernel count")
        if count != 1:
            raise RuntimeError(f"Kimi K3 MegaMoE (SM107): {count} kernels in {cubin}")
        kernels = _checked(cuda.cuLibraryEnumerateKernels(1, self.library), "kernels")
        self.function = _checked(cuda.cuKernelGetFunction(kernels[0]), "function")
        attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _checked(cuda.cuFuncSetAttribute(self.function, attr, smem), "smem attribute")


# ----------------------------------------------------------------- TMA descriptors
_DT = cuda.CUtensorMapDataType
_TMA_DTYPES = {
    torch.int32: _DT.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.float32: _DT.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.bfloat16: _DT.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: _DT.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint8: _DT.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int8: _DT.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,  # packed FP4, unpacked smem
}
_SWIZZLE = {
    0: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    16: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def _encode(t: torch.Tensor, dims, strides, box, swizzle: int) -> bytes:
    """``make_tma_{2,4}d_desc``'s cuTensorMapEncodeTiled (strides in elements)."""
    elem = t.element_size()
    if t.data_ptr() % 16:
        raise RuntimeError("MegaMoE: TMA base not 16-byte aligned")
    if any(s * elem % 16 for s in strides):
        raise RuntimeError(f"MegaMoE: TMA strides {strides} x {elem} B not 16-byte aligned")
    box = list(box)
    if swizzle:
        box[0] = swizzle // elem
    if t.dtype == torch.int8 and dims[0] % 128:
        raise RuntimeError("MegaMoE: packed FP4 inner dim not a multiple of 128")
    tensor_map = _checked(
        cuda.cuTensorMapEncodeTiled(
            _TMA_DTYPES[t.dtype], len(dims), t.data_ptr(),
            [cuda.cuuint64_t(d) for d in dims],
            [cuda.cuuint64_t(s * elem) for s in strides],
            [cuda.cuuint32_t(b) for b in box],
            [cuda.cuuint32_t(1)] * len(dims),
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            _SWIZZLE[swizzle],
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )  # fmt: skip
    return bytes((ctypes.c_ubyte * 128).from_address(tensor_map.getPtr()))


def _tma_2d(t, inner, outer, box_inner, box_outer, outer_stride, swizzle):
    return _encode(t, (inner, outer), (outer_stride,), (box_inner, box_outer), swizzle)


def _tma_sf(t, shape_mn, shape_k, block_mn, num_groups, smem_outer, sf_k_stride):
    """``make_tma_sf_desc`` (MN-major, no swizzle, packed UE8M0 ints)."""
    shape_mn = _align(shape_mn, 16 // t.element_size())  # get_tma_aligned_size
    stride = shape_mn if sf_k_stride == 0 else sf_k_stride
    if stride < shape_mn:
        raise RuntimeError("MegaMoE: SF K stride below the MN extent")
    per = _GRAN_K * (1 if t.dtype == torch.float32 else 4)
    return _tma_2d(t, shape_mn, _ceil_div(shape_k, per) * num_groups, block_mn,
                   smem_outer, stride, 0)  # fmt: skip


def _tma_weights(t, block_k, load_block_n, domains, batched):
    """``make_tma_weights_{3,2}d_desc``: (B, N, K) / (N, K) -> 4D
    (K, N / domains, domains, B). The decode never localizes the routed weights
    (the stock ``(2, B, N / 2, K)`` MLOPart layout is not served)."""
    if t.dim() != (3 if batched else 2):
        raise RuntimeError(
            f"MegaMoE: weights of shape {tuple(t.shape)}: the SM107 launcher takes "
            "unlocalized (B, N, K) / (N, K) weights only"
        )
    k = t.size(-1) * (2 if t.dtype == torch.int8 else 1)
    batches = t.size(0) if batched else 1
    n = t.size(-2) // domains
    slice_stride = n * t.stride(-2)
    outer = t.stride(0) if batched else slice_stride * domains
    return _encode(t, (k, n, domains, batches), (t.stride(-2), slice_stride, outer),
                   (block_k, load_block_n, 1, 1), 128)  # fmt: skip


class _SymBuffer(ctypes.Structure):
    """``layout::SymBuffer<>``."""

    _fields_ = [
        ("base", ctypes.c_int64),
        ("offsets", ctypes.c_int64 * _MAX_RANKS),
        ("rank_idx", ctypes.c_uint32),
    ]


_Desc = ctypes.c_ubyte * 128


class _Launch:
    """Arguments and config of one (shape, weights, buffer) launch; per call only
    ``y``, the stats pointer, ``num_tokens`` and the stream change."""

    def __init__(self, kernel: _Kernel, cfg: SituConfig, num_sms: int, pdl: bool,
                 sym: _SymBuffer, descs: list[bytes], domains: torch.Tensor) -> None:  # fmt: skip
        self.kernel, self.domains = kernel, domains
        self.y, self.stats = ctypes.c_void_p(0), ctypes.c_void_p(0)
        self.num_tokens, self.sym = ctypes.c_uint32(0), sym
        self.descs = [_Desc.from_buffer_copy(d) for d in descs]
        self.domains_ptr = ctypes.c_void_p(domains.data_ptr())
        args = [self.y, self.stats, self.num_tokens, self.sym, *self.descs, self.domains_ptr]
        self.args = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
        self.address = ctypes.addressof(self.args)
        ids = cuda.CUlaunchAttributeID
        cluster = cuda.CUlaunchAttribute()
        cluster.id = ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        cluster.value.clusterDim.x, cluster.value.clusterDim.y = _CLUSTER, 1
        cluster.value.clusterDim.z = 1
        self.attributes = [cluster]
        if pdl:
            serial = cuda.CUlaunchAttribute()
            serial.id = ids.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            serial.value.programmaticStreamSerializationAllowed = 1
            self.attributes.append(serial)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = num_sms, 1, 1
        config.blockDimX = _DISPATCH_THREADS + _NON_EPILOGUE_THREADS + _EPILOGUE_THREADS
        config.blockDimY = config.blockDimZ = 1
        config.sharedMemBytes = cfg.smem_size
        config.attrs, config.numAttrs = self.attributes, len(self.attributes)

    def __call__(self, y: torch.Tensor, stats: torch.Tensor | None, num_tokens: int) -> None:
        self.y.value = y.data_ptr()
        self.stats.value = None if stats is None else stats.data_ptr()
        self.num_tokens.value = num_tokens
        self.config.hStream = torch.cuda.current_stream(y.device).cuda_stream
        _checked(
            cuda.cuLaunchKernelEx(self.config, self.kernel.function, self.address, 0),
            "cuLaunchKernelEx",
        )


class SituMegaMoE:
    """The SM107 decode's FP8 x FP4 SiTU MegaMoE launcher for one process: the
    patched kernel, built per shape on first use (as DeepGEMM's JIT would, at the
    runner's warmup) and launched with DeepGEMM's host arguments."""

    def __init__(self, deep_gemm, cache_root: Path | None = None) -> None:
        self.dg = deep_gemm
        if cache_root is None:
            from vllm import envs

            cache_root = Path(envs.VLLM_CACHE_ROOT) / "kimi_k3_megamoe"
        self.cache_root = str(cache_root)
        dg_file = deep_gemm.__file__
        self.include = (patched_include(dg_file, self.cache_root), _dg_include(dg_file))
        self.kernels: dict[str, _Kernel] = {}
        self.launches: dict[tuple, _Launch] = {}

    def _kernel(self, source: str, smem: int, device: int) -> _Kernel:
        kernel = self.kernels.get(source)
        if kernel is None:
            cubin = build(source, self.include, device, self.cache_root)
            kernel = self.kernels[source] = _Kernel(cubin, smem)
        return kernel

    def __call__(
        self,
        y: torch.Tensor,
        l1_weights: tuple[torch.Tensor, torch.Tensor],
        l2_weights: tuple[torch.Tensor, torch.Tensor],
        sym_buffer,
        shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        cumulative_local_expert_recv_stats: torch.Tensor | None = None,
        recipe: tuple[int, int, int] = (1, 1, 32),
        activation: str = "situ",
        activation_clamp: float | None = None,
        fast_math: bool = True,
        activation_alpha: float = 1.0,
        activation_beta: float = 0.0,
    ) -> None:
        """``deep_gemm.fp8_fp4_mega_moe`` for ``activation="situ"``, on the patched
        kernel."""
        num_tokens = y.size(0)
        key = (
            num_tokens, sym_buffer.buffer.data_ptr(), fast_math,
            activation_alpha, activation_beta,
            *(t.data_ptr() for t in (*l1_weights, *l2_weights)),
            *(t.data_ptr() for t in (*(shared_l1_weights or ()), *(shared_l2_weights or ()))),
        )  # fmt: skip
        launch = self.launches.get(key)
        if launch is None:
            launch = self.launches[key] = self._prepare(
                y, l1_weights, l2_weights, sym_buffer, shared_l1_weights,
                shared_l2_weights, recipe, activation, activation_clamp, fast_math,
                activation_alpha, activation_beta,
            )  # fmt: skip
        launch(y, cumulative_local_expert_recv_stats, num_tokens)

    def _prepare(self, y, l1_weights, l2_weights, buf, shared_l1, shared_l2, recipe,
                 activation, clamp, fast_math, alpha, beta) -> _Launch:  # fmt: skip
        dg = self.dg
        if tuple(recipe) != (1, 1, 32) or activation != "situ" or clamp is not None:
            raise RuntimeError(
                f"Kimi K3 MegaMoE (SM107): SiTU with recipe (1, 1, 32) and no clamp "
                f"only; got recipe {recipe}, activation {activation}, clamp {clamp}"
            )
        if not (math.isfinite(alpha) and math.isfinite(beta) and alpha > 0 and beta >= 0):
            raise RuntimeError(f"MegaMoE: SiTU scales alpha {alpha} beta {beta}")
        (w1, w1_sf), (w2, w2_sf) = l1_weights, l2_weights
        if (shared_l1 is None) != (shared_l2 is None):
            raise RuntimeError("MegaMoE: shared L1 and L2 weights go together")
        num_tokens, device = y.size(0), y.device.index
        domains_n = dg.get_num_locality_domains()
        # check_weights_layout_3d: logical (E, N, K) (never localized here)
        if w1.dim() != 3 or w2.dim() != 3:
            raise RuntimeError(
                f"MegaMoE: weights {tuple(w1.shape)} / {tuple(w2.shape)}: the SM107 "
                "launcher takes unlocalized (E, N, K) weights only"
            )
        epr = w1.size(-3)
        inter2 = w1.size(-2)
        hidden = w1.size(-1) * (2 if w1.dtype == torch.int8 else 1)
        inter = w2.size(-1) * (2 if w2.dtype == torch.int8 else 1)
        if inter2 != 2 * inter or w1.dtype != w2.dtype:
            raise RuntimeError(f"MegaMoE: weights {tuple(w1.shape)} / {tuple(w2.shape)}")
        ptrs = list(buf.handle.buffer_ptrs)
        num_ranks, rank = len(ptrs), buf.group.rank()
        num_experts = epr * num_ranks
        if num_experts != buf.num_experts or num_tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"MegaMoE: {num_experts} experts / {num_tokens} tokens against a buffer "
                f"for {buf.num_experts} / {buf.num_max_tokens_per_rank}"
            )
        num_topk, mtpr = buf.num_topk, buf.num_max_tokens_per_rank
        num_shared, shared_inter = 0, 0
        if shared_l1 is not None:
            shared_inter = shared_l2[0].size(-1)
            if shared_inter % inter:
                raise RuntimeError("MegaMoE: shared width not a multiple of the expert's")
            num_shared = shared_inter // inter
        ring, sf_ring = buf.l1_acts.size(0), buf.l1_acts_sf.size(0)
        cfg = situ_config(num_ranks, num_experts, num_tokens, num_topk, hidden, ring, sf_ring)
        stock_block_m = dg.get_block_m_for_mega_moe(
            num_ranks, num_experts, mtpr, num_tokens, num_topk, "fp8xfp4"
        )
        if stock_block_m != cfg.block_m:
            raise RuntimeError(
                f"Kimi K3 MegaMoE (SM107): BLOCK_M {cfg.block_m} != DeepGEMM's "
                f"{stock_block_m} ({num_tokens} tokens): the installed DeepGEMM's "
                "heuristics differ from the ones this launcher mirrors"
            )
        num_sms = dg.get_num_sms()
        source = instantiation(
            cfg, mtpr, hidden, inter, num_experts, num_shared, num_topk, num_sms,
            num_ranks, alpha, beta, fast_math, w1.dtype,
        )  # fmt: skip
        kernel = self._kernel(source, cfg.smem_size, device)
        sf_outer = cfg.block_k // (_GRAN_K * 4)
        lb_m, lb_n, bk, bn = cfg.load_block_m, cfg.load_block_n, cfg.block_k, cfg.block_n
        a = (buf.l1_acts, buf.l1_acts_sf, buf.l2_acts, buf.l2_acts_sf)
        d = [
            _tma_2d(a[0], hidden, ring, bk, lb_m, a[0].stride(-2), 128),
            _tma_sf(a[1], sf_ring, hidden, cfg.sf_block_m, 1, sf_outer, a[1].stride(-1)),
            _tma_weights(w1, bk, lb_n, domains_n, True),
            _tma_sf(w1_sf, inter * 2, hidden, bn, epr, sf_outer, w1_sf.stride(-1)),
            _tma_2d(a[2], inter, ring, bn // 2, cfg.store_block_m_l1, a[2].stride(-2), 64),
            _tma_2d(a[2], inter, ring, bk, lb_m, a[2].stride(-2), 128),
            _tma_sf(a[3], sf_ring, inter, cfg.sf_block_m, 1, sf_outer, a[3].stride(-1)),
            _tma_weights(w2, bk, lb_n, domains_n, True),
            _tma_sf(w2_sf, hidden, inter, bn, epr, sf_outer, w2_sf.stride(-1)),
        ]
        if num_shared:
            s = (buf.shared_l1_acts, buf.shared_l1_acts_sf, buf.shared_l2_acts,
                 buf.shared_l2_acts_sf)  # fmt: skip
            (sw1, sw1_sf), (sw2, sw2_sf) = shared_l1, shared_l2
            d += [
                _tma_2d(s[0], hidden, mtpr, bk, lb_m, s[0].stride(-2), 128),
                _tma_sf(s[1], s[1].size(0), hidden, cfg.sf_block_m, 1, sf_outer, s[1].stride(-1)),
                _tma_weights(sw1, bk, lb_n, domains_n, False),
                _tma_sf(sw1_sf, shared_inter * 2, hidden, bn, 1, sf_outer, sw1_sf.stride(-1)),
                _tma_2d(s[2], shared_inter, mtpr, bn // 2, cfg.store_block_m_l1, s[2].stride(-2), 64),
                _tma_2d(s[2], shared_inter, mtpr, bk, lb_m, s[2].stride(-2), 128),
                _tma_sf(s[3], s[3].size(0), shared_inter, cfg.sf_block_m, 1, sf_outer, s[3].stride(-1)),
                _tma_weights(sw2, bk, lb_n, domains_n, False),
                _tma_sf(sw2_sf, hidden, shared_inter, bn, 1, sf_outer, sw2_sf.stride(-1)),
            ]  # fmt: skip
        else:  # the routed descriptors stand in (never read)
            d += [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]
        sym = _SymBuffer()
        sym.base, sym.rank_idx = ptrs[rank], rank
        for i in range(_MAX_RANKS):
            sym.offsets[i] = ptrs[i] - ptrs[rank] if i < num_ranks else 0
        domains = dg.locality_domain.get_balanced_sm_locality_domains()
        return _Launch(kernel, cfg, num_sms, bool(dg.get_pdl()), sym, d, domains)
