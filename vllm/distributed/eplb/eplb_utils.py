# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import os
import threading

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CpuGpuEvent:
    """
    Combines a CUDA event with a CPU threading event to enforce record->wait
    ordering across two threads.

    This class is designed for exactly two threads: one producer that calls
    record() and one consumer that calls wait(). Using it with more than two
    threads is not supported and will produce undefined behavior.

    CUDA events alone are insufficient for cross-thread synchronization because
    waiting on an unrecorded CUDA event is a no-op. The wait will return
    immediately instead of blocking. This class adds a threading.Event so
    that the waiting thread blocks on the CPU side until record() is called, at
    which point the CUDA event is guaranteed to be in-flight and event.wait() will
    correctly synchronize the GPU stream.
    """

    def __init__(self):
        self._event = torch.cuda.Event()
        self._recorded = threading.Event()

    def wait(self, stream: torch.cuda.Stream | None = None):
        """
        Blocks the calling thread until record finishes. Used to guarantee that the
        record kernel is called before wait.

        Should only be called by the Async Eplb thread.
        """
        self._recorded.wait()
        self._event.wait(stream)
        self._recorded.clear()

    def record(self, stream: torch.cuda.Stream | None = None):
        """
        Unblocks the waiting thread after calling event.record().

        Should only be called by the main thread.
        """
        if self._recorded.is_set():
            raise RuntimeError(
                "CpuGpuEvent.record() called before the previous event was "
                "consumed by wait()"
            )
        self._event = torch.cuda.Event()
        self._event.record(stream)
        self._recorded.set()


def override_envs_for_eplb(
    parallel_config: ParallelConfig,
    moe_backend: str | None = None,
) -> None:
    """
    Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
        moe_backend: The configured MoE backend (e.g. ``deep_gemm_mega_moe``).
            Used to detect cooperative-launch kernels in the same hang class
            as DeepEP low-latency.
    """
    is_data_parallel = parallel_config.data_parallel_size > 1
    is_eplb_enabled = parallel_config.enable_eplb
    is_deepep_ll = parallel_config.all2all_backend == "deepep_low_latency"
    is_nccl_based_eplb_communicator = parallel_config.eplb_config.communicator in (
        "torch_nccl",
        "pynccl",
    )
    # DeepGEMM MegaMoE dispatches via PyTorch symmetric memory and launches
    # cooperative kernels that try to grab most of the GPU's SMs, the same
    # class of kernel as DeepEP LL. Both deadlock against a concurrent NCCL
    # collective that occupies those SMs.
    is_cooperative_moe = moe_backend == "deep_gemm_mega_moe"

    # Override NCCL_MAX_CTAS to avoid hangs when EPLB's NCCL weight exchange
    # collides with a cooperative-launch MoE backend.
    #
    # The MoE kernel uses a cooperative launch and tries to reserve a large
    # fraction of the GPU's SMs; if those SMs are currently occupied by NCCL,
    # the MoE launch blocks until enough SMs are freed. Conversely NCCL P2P
    # only completes when all peers participate -- so if any peer is stalled
    # waiting for SMs the entire collective hangs.
    #
    # Originally observed with async EPLB + DeepEP LL (NCCL on background
    # thread, DeepEP on main); also reproduces with sync EPLB + DeepGEMM
    # MegaMoE, because NCCL P2P runs on its own internal stream and can
    # overlap with the next forward's cooperative kernel on the default
    # stream.
    #
    # Limiting NCCL occupancy via NCCL_MAX_CTAS leaves SMs available for the
    # cooperative kernel, breaking the cycle.
    # See: https://github.com/deepseek-ai/DeepEP/issues/496
    if (
        is_data_parallel
        and is_eplb_enabled
        and (is_deepep_ll or is_cooperative_moe)
        and is_nccl_based_eplb_communicator
    ):
        current_value_str = os.getenv("NCCL_MAX_CTAS")

        if current_value_str and current_value_str.isdigit():
            return

        override_value = 8
        os.environ["NCCL_MAX_CTAS"] = str(override_value)
        trigger = "deepep_low_latency" if is_deepep_ll else "deep_gemm_mega_moe"
        logger.info_once(
            f"EPLB: Setting NCCL_MAX_CTAS={override_value} "
            f"for expert parallel with NCCL-based EPLB communicator and "
            f"cooperative MoE backend ({trigger})",
            scope="global",
        )
