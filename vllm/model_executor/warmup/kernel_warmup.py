# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    Only rank 0 runs the profiling; other ranks participate in the
    dummy_run (so collectives don't deadlock) but enter the autotune
    context with ``tune_mode=False`` and fall through to the default
    tactic on every kernel call.

    Setting ``VLLM_FLASHINFER_AUTOTUNE_SAVE_DIR=<dir>`` saves rank 0's
    profiling cache to ``<dir>/rank_0.json`` after autotune. No save
    happens unless the env var is set.
    """
    import os

    import torch.distributed as dist

    import vllm.utils.flashinfer as fi_utils

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_rank_zero = rank == 0

    with torch.inference_mode(), fi_utils.autotune(tune_mode=is_rank_zero):
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process. Set per-rank so
        # only rank 0's MoE wrapper opens its inner autotune(True) context.
        fi_utils._is_fi_autotuning = is_rank_zero

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False

    save_dir = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_SAVE_DIR")
    if save_dir and is_rank_zero:
        from flashinfer.autotuner import AutoTuner

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"rank_{rank}.json")
        AutoTuner.get().save_configs(save_path)
        logger.info(
            "[FlashInfer autotune] rank %d cache saved to %s", rank, save_path
        )
