# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@triton.jit
def _scatter_draft_tokens_kernel(
    dst_ptr,  # [max_num_reqs, dst_stride]
    dst_stride,
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    src_ptr,  # [num_reqs, src_stride]
    src_stride,
    num_tokens,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_tokens
    tokens = tl.load(src_ptr + row * src_stride + offsets, mask=mask)
    tl.store(dst_ptr + req_state_idx * dst_stride + offsets, tokens, mask=mask)


def scatter_draft_tokens(
    dst: torch.Tensor, idx_mapping: torch.Tensor, draft_tokens: torch.Tensor
) -> None:
    """``dst[idx_mapping] = draft_tokens`` in one small kernel (rows with a -1
    sentinel in ``idx_mapping`` are skipped)."""
    num_reqs, num_tokens = draft_tokens.shape
    if num_reqs == 0:
        return
    if draft_tokens.stride(1) != 1 or dst.stride(1) != 1:
        dst[idx_mapping] = draft_tokens
        return
    _scatter_draft_tokens_kernel[(num_reqs,)](
        dst,
        dst.stride(0),
        idx_mapping,
        draft_tokens,
        draft_tokens.stride(0),
        num_tokens,
        BLOCK=triton.next_power_of_2(num_tokens),
        launch_pdl=current_platform.is_arch_support_pdl(),
    )


def get_drafter_hidden_states(
    model: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Hidden states fed to the drafter.

    Targets such as DeepSeek V4 expose the pre-hc_head residual through
    get_mtp_target_hidden_states(). The buffer is sized at
    max_num_batched_tokens and only allocated for drafters that consume
    target hidden states, so None means "use the regular hidden states".
    """
    get_target_hidden_states = getattr(model, "get_mtp_target_hidden_states", None)
    if get_target_hidden_states is None:
        return hidden_states
    target_hidden_states = get_target_hidden_states()
    if target_hidden_states is None:
        return hidden_states
    return target_hidden_states[: hidden_states.shape[0]]


def get_pp_safe_draft_load_config(load_config: LoadConfig) -> LoadConfig:
    """Avoid collectives that include PP ranks without a draft model."""
    if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
        logger.warning_once(
            "fastsafetensors cannot load a draft model instantiated on only "
            "one pipeline stage; falling back to the standard safetensors "
            "loader for the draft model. The target model still uses "
            "fastsafetensors."
        )
        return replace(load_config, load_format="auto")
    return load_config


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor | None,
        num_draft_tokens: int,
    ) -> None:
        """draft_tokens: [num_reqs, num_draft_tokens], needed only when the batch
        has structured-output requests."""
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = num_draft_tokens
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        assert draft_tokens is not None
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
