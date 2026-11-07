# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A CUDA-graph IF/ELSE node inside a stream capture this code does not own (vLLM's
CUDA graph capture of the decode step), so one captured step can take one of two
bodies per replay, chosen on the device.

``GraphSwitch.begin(lengths, bound)`` launches a one-warp predicate kernel on the
capturing stream (max(lengths) <= bound, written into a conditional handle with
cudaGraphSetConditional), appends an IF node with two bodies (CUDA 12.8+) after it and
makes the node the capturing stream's dependency; ``with switch.body(i):`` then
captures body i on a body stream that is the current torch stream meanwhile. Body 0
runs when the predicate holds, body 1 otherwise. Work captured after the bodies
depends on the node (the whole taken body).

- PDL: programmatic edges stay inside each body; the edges into and out of the node
  are full dependencies (the first kernel of a body starts after the predicate
  kernel, the first kernel after the node after the taken body).
- Streams forked inside a body (event waits on the body stream) join the body's
  capture and must be joined back before the body ends.
- Memory: allocations on the body stream are routed to a private pool of the switch
  (never released) while a body is captured; the caller should not allocate in
  bodies at all (``allocations`` counts the allocator's allocations per body).
"""

import contextlib
import ctypes
import glob
import os
from functools import cache

import torch

SOURCE = r"""
#include <cuda_runtime.h>
extern "C" __global__ void switch_max_le(cudaGraphConditionalHandle handle,
                                         const int* lengths, int count, int bound) {
    int top = 0;
    for (int i = threadIdx.x; i < count; i += 32) top = max(top, lengths[i]);
    for (int o = 16; o > 0; o >>= 1)
        top = max(top, __shfl_xor_sync(0xffffffffu, top, o));
    if (threadIdx.x == 0) cudaGraphSetConditional(handle, top <= bound ? 1u : 0u);
}
"""


def _check(result):
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"Kimi K3 graph switch: CUDA error {err}")
    return rest[0] if len(rest) == 1 else tuple(rest)


def _cuda_home():
    for root in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda"):
        if root and os.path.isdir(os.path.join(root, "include")):
            return root
    raise RuntimeError("Kimi K3 graph switch: no CUDA toolkit headers")


@cache
def _predicate_kernel(major, minor):
    """The predicate kernel as a CUfunction: NVRTC (-rdc) + nvJitLink with
    libcudadevrt (cudaGraphSetConditional is a device runtime call)."""
    from cuda.bindings import driver, nvjitlink, nvrtc

    home = _cuda_home()
    arch = f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}"
    program = _check(nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"switch.cu", 0, [], []))
    options = [b"-rdc=true", f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
    options += [f"--include-path={home}/include".encode()]
    if int(nvrtc.nvrtcCompileProgram(program, len(options), options)[0]) != 0:
        size = _check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(f"Kimi K3 graph switch: NVRTC failed:\n{log.decode()}")
    size = _check(nvrtc.nvrtcGetCUBINSize(program))
    cubin = b" " * size
    _check(nvrtc.nvrtcGetCUBIN(program, cubin))
    devrt = sorted(glob.glob(f"{home}/lib64/libcudadevrt.a"))
    devrt += sorted(glob.glob(f"{home}/targets/*/lib/libcudadevrt.a"))
    if not devrt:
        raise RuntimeError("Kimi K3 graph switch: libcudadevrt.a not found")
    link = nvjitlink.create(2, [f"-arch={arch}", "-O3"])
    try:
        nvjitlink.add_data(link, nvjitlink.InputType.CUBIN, cubin, len(cubin), "switch")
        nvjitlink.add_file(link, nvjitlink.InputType.LIBRARY, devrt[0])
        nvjitlink.complete(link)
        image = bytearray(nvjitlink.get_linked_cubin_size(link))
        nvjitlink.get_linked_cubin(link, image)
    finally:
        nvjitlink.destroy(link)
    library = _check(
        driver.cuLibraryLoadData(bytes(image), None, None, 0, None, None, 0)
    )
    kernel = _check(driver.cuLibraryGetKernel(library, b"switch_max_le"))
    return library, kernel


class GraphSwitch:
    """IF/ELSE nodes appended to the current capture of ``device`` (see module doc)."""

    def __init__(self, device):
        self.device = torch.device(device)
        cc = torch.cuda.get_device_capability(self.device)
        self.library, self.kernel = _predicate_kernel(*cc)
        self.body_stream = torch.cuda.Stream(device=self.device)
        # Body allocations (there should be none): a private pool kept for the
        # process lifetime, so replays never see the blocks reused.
        self.pool = torch.cuda.graph_pool_handle()
        self.allocations: list[int] = []
        self.bodies: list = []

    def _predicate(self, stream, handle, lengths, count, bound):
        from cuda.bindings import driver

        function = _check(driver.cuKernelGetFunction(self.kernel))
        args = (
            ctypes.c_uint64(int(handle)),
            ctypes.c_void_p(lengths.data_ptr()),
            ctypes.c_int(count),
            ctypes.c_int(bound),
        )
        pointers = (ctypes.c_void_p * 4)(*(ctypes.addressof(a) for a in args))
        _check(
            driver.cuLaunchKernel(
                function, 1, 1, 1, 32, 1, 1, 0, stream, ctypes.addressof(pointers), 0
            )
        )

    def begin(self, lengths, bound):
        """Append the predicate kernel and the IF/ELSE node to the current stream's
        capture; ``lengths``: int32 [n] on the device (read at replay). Then capture
        body 0 (max(lengths) <= ``bound``) and body 1 with ``body(0)``, ``body(1)``."""
        from cuda.bindings import runtime

        raw = torch.cuda.current_stream(self.device).cuda_stream
        status, _, graph, *_ = _check(runtime.cudaStreamGetCaptureInfo(raw))
        active = runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
        if status != active:
            raise RuntimeError("Kimi K3 graph switch: the stream is not capturing")
        handle = _check(runtime.cudaGraphConditionalHandleCreate(graph, 0, 0))
        assert lengths.dtype == torch.int32 and lengths.is_contiguous()
        self._predicate(raw, handle, lengths, lengths.numel(), bound)
        info = _check(runtime.cudaStreamGetCaptureInfo(raw))
        graph, dependencies, edges, count = info[2], info[3], info[4], info[5]
        params = runtime.cudaGraphNodeParams()
        params.type = runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = (
            runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
        )
        params.conditional.size = 2
        node = _check(
            runtime.cudaGraphAddNode(graph, dependencies, edges, count, params)
        )
        self.bodies = list(params.conditional.phGraph_out)[:2]
        flags = runtime.cudaStreamUpdateCaptureDependenciesFlags
        _check(
            runtime.cudaStreamUpdateCaptureDependencies(
                raw, [node], None, 1, flags.cudaStreamSetCaptureDependencies
            )
        )

    @contextlib.contextmanager
    def body(self, index):
        """Capture into body ``index`` with the body stream current; allocations on it
        go to this switch's private pool (the capture's own pool is already recording
        for the capturing stream) and are counted in ``allocations``."""
        from cuda.bindings import runtime

        mode = runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
        side = self.body_stream.cuda_stream
        target = self.bodies[index]
        _check(runtime.cudaStreamBeginCaptureToGraph(side, target, None, None, 0, mode))
        stats = torch.cuda.memory_stats(self.device)
        before = stats.get("allocation.all.allocated", 0)
        try:
            with torch.cuda.stream(self.body_stream):
                begin_pool = torch._C._cuda_beginAllocateCurrentStreamToPool
                begin_pool(self.device.index, self.pool)
                try:
                    yield
                finally:
                    torch._C._cuda_endAllocateToPool(self.device.index, self.pool)
        finally:
            _check(runtime.cudaStreamEndCapture(side))
        stats = torch.cuda.memory_stats(self.device)
        self.allocations.append(stats.get("allocation.all.allocated", 0) - before)
