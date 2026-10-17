# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from enum import IntEnum
from typing import cast

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, _unpack
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

# SP-v2 overlaps collectives with expert computation on a side stream.
_PF_SP_CARVEOUT = int(os.environ.get("ARIA_PF_SP_CARVEOUT", "32"))
_pf_sp_stream: torch.cuda.Stream | None = None


def _pf_sp_comm_stream() -> torch.cuda.Stream:
    global _pf_sp_stream
    if _pf_sp_stream is None:
        _pf_sp_stream = torch.cuda.Stream(priority=-1)
    return _pf_sp_stream


_PF_SP_KEEP = object()


def _pf_sp_set_carveout(n: int):
    setter = getattr(torch._C, "_set_sm_carveout_experimental", None)
    getter = getattr(torch._C, "_get_sm_carveout_experimental", None)
    if n <= 0 or setter is None:
        return _PF_SP_KEEP
    prev = getter() if getter is not None else None
    setter(n)
    return prev


def _pf_sp_restore_carveout(prev) -> None:
    if prev is not _PF_SP_KEEP:
        torch._C._set_sm_carveout_experimental(prev)


# ARIA pf-sp (ARIA_PF_SP=1, prefill workers only). On an eager forward above
# ARIA_PF_SP_MIN_TOKENS tokens, KimiLinearModel keeps the residual stream
# sequence-sharded over TP: rank r owns token rows [r*c, (r+1)*c) with
# c = ceil(T / tp), and the rows past T are zeros. Each TP-sharded block gets
# an all-gather before it, and its output is reduce-scattered instead of
# all-reduced. This runner's latent tail then reduce-scatters the latent and
# the shared partial and up-projects its own tokens with the replicated weight.
# This module owns the per-forward state and the collectives, because both the
# model and this runner use them. Unset, nothing below runs and no state is set.
_PF_SP = os.environ.get("ARIA_PF_SP", "0") == "1"
# Default 9362: sequence-shard only where the stock path's latent all-reduce
# (T x 3584 bf16) is above FlashInfer's 64 MiB MNNVL cap and so already runs
# as a PYNCCL ring (the hidden all-reduces switch at 4681 tokens). Below it
# the stock reductions accumulate in fp32 and NCCL reduce-scatter would be
# noisier.
_PF_SP_MIN_TOKENS = int(os.environ.get("ARIA_PF_SP_MIN_TOKENS", "9362"))
# "pynccl": NCCL all-gather / reduce-scatter on the TP communicator.
# "auto": vLLM's sequence-parallel ops (MNNVL Lamport kernels when the size
# fits their buffers, else NCCL).
_PF_SP_BACKEND = os.environ.get("ARIA_PF_SP_BACKEND", "pynccl")
# Debug: check the pad rows and finiteness of the first N real pf-sp forwards
# (one device sync each). 0 = off.
_PF_SP_CHECK = int(os.environ.get("ARIA_PF_SP_CHECK", "0"))

# ARIA pf-sp v2 (ARIA_PF_SP_V2=1, read only with ARIA_PF_SP=1). Inside a pf-sp
# forward:
# - V2_MOE: the MoE router gate and latent down-projection run on this rank's
#   token shard and their outputs are all-gathered; every MoE-block collective
#   runs on the SP side stream, overlapped with the shared and routed
#   experts (shared expert first, its reduce-scatter under the routed experts).
# - V2_FP8AG: the attention input is quantized on the shard by the attention's
#   own first-GEMM quantizer (per-token 1x128 FP8, packed UE8M0 scales) and
#   all-gathered in FP8; that GEMM then runs on the gathered operands.
# Unset, nothing below runs and the pf-sp path is v1's, call for call.
_PF_SP_V2 = _PF_SP and os.environ.get("ARIA_PF_SP_V2", "0") == "1"
_PF_SP_V2_MOE = _PF_SP_V2 and os.environ.get("ARIA_PF_SP_V2_MOE", "1") == "1"
_PF_SP_V2_FP8AG = _PF_SP_V2 and os.environ.get("ARIA_PF_SP_V2_FP8AG", "1") == "1"


class PfSpState:
    """Per-forward sequence-parallel layout; ``active`` only inside a forward."""

    active = False
    num_tokens = 0  # T
    num_padded = 0  # Tp = c * world
    shard = 0  # c
    rank = 0
    world = 1


pf_sp_state = PfSpState()


def pf_sp_begin(num_tokens: int) -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    st = pf_sp_state
    st.world = get_tensor_model_parallel_world_size()
    st.rank = get_tensor_model_parallel_rank()
    st.num_tokens = num_tokens
    st.shard = -(-num_tokens // st.world)
    st.num_padded = st.shard * st.world
    st.active = True


def pf_sp_end() -> None:
    pf_sp_state.active = False


def pf_sp_take_shard(x: torch.Tensor) -> torch.Tensor:
    """This rank's rows of a replicated ``[T, ...]`` tensor, zero-padded to c."""
    st = pf_sp_state
    lo = st.rank * st.shard
    hi = min(lo + st.shard, x.shape[0])
    if hi - lo == st.shard:
        return x[lo:hi]
    out = x.new_zeros((st.shard,) + tuple(x.shape[1:]))
    if hi > lo:
        out[: hi - lo].copy_(x[lo:hi])
    return out


def _pf_sp_pynccl():
    from vllm.distributed import get_tp_group

    comm = get_tp_group().device_communicator.pynccl_comm
    assert comm is not None and not comm.disabled, "ARIA pf-sp needs pynccl"
    return comm


def pf_sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """``[c, ...]`` shard -> ``[Tp, ...]`` on every rank (rank-major rows)."""
    st = pf_sp_state
    assert x.shape[0] == st.shard, (tuple(x.shape), st.shard)
    x = x.contiguous()
    if _PF_SP_BACKEND == "auto":
        from vllm.models.common.ops.sequence_parallel import sp_all_gather

        return sp_all_gather(x)
    out = torch.empty(
        (st.num_padded,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device
    )
    _pf_sp_pynccl().all_gather(out, x)
    return out


def pf_sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """``[Tp, ...]`` TP partial -> this rank's ``[c, ...]`` rows of the sum."""
    st = pf_sp_state
    assert x.shape[0] == st.num_padded, (tuple(x.shape), st.num_padded)
    x = x.contiguous()
    if _PF_SP_BACKEND == "auto":
        from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

        return sp_reduce_scatter(x)
    out = torch.empty((st.shard,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    _pf_sp_pynccl().reduce_scatter(out, x)
    return out


# ---- pf-sp v2 ----------------------------------------------------------------
# Every v2 side-stream collective is ordered after the main-stream work issued
# before it (an event the side stream waits on) and joined by an event the main
# stream waits on before it reads the result, so all NCCL calls on the TP
# communicator stay in one total order. Outputs are allocated on the main
# stream before the ordering event; inputs are record_stream'ed to the side
# stream.


class PfSp2State:
    """v2 per-forward state; only set inside a v2 forward."""

    fp8_marker: torch.Tensor | None = None  # the stand-in attention input
    fp8_input: tuple[torch.Tensor, torch.Tensor] | None = None  # gathered (q, s)
    moe_layers = 0  # this forward's MoE blocks on the token shard
    fp8_layers = 0  # this forward's attention inputs gathered in FP8
    logged: set = set()


pf_sp2_state = PfSp2State()


def pf_sp2_log_once(key: str, msg: str, *args) -> None:
    if key not in pf_sp2_state.logged:
        pf_sp2_state.logged.add(key)
        logger.info("ARIA pf-sp v2: %s", msg % args if args else msg)


def pf_sp2_begin() -> None:
    pf_sp2_state.moe_layers = 0
    pf_sp2_state.fp8_layers = 0


def pf_sp2_reset() -> None:
    pf_sp2_state.fp8_marker = None
    pf_sp2_state.fp8_input = None


def _pf_sp2_gather_group(pairs, stream) -> None:
    """All-gather each ``(out [Tp, ...], x [c, ...])`` pair; one NCCL group."""
    comm = _pf_sp_pynccl()
    if len(pairs) > 1:
        comm.group_start()
    try:
        for out, x in pairs:
            comm.all_gather(out, x, stream=stream)
    finally:
        if len(pairs) > 1:
            comm.group_end()


def pf_sp2_all_gather_async(
    xs: list[torch.Tensor], ev_in: torch.cuda.Event, ev_out: torch.cuda.Event
) -> list[torch.Tensor]:
    """Side-stream all-gather of the ``[c, ...]`` shards ``xs`` (one NCCL
    group); the main stream must wait on ``ev_out`` before reading the
    returned ``[Tp, ...]`` tensors."""
    from vllm.utils.torch_utils import current_stream

    st = pf_sp_state
    xs = [x.contiguous() for x in xs]
    for x in xs:
        assert x.shape[0] == st.shard, (tuple(x.shape), st.shard)
    outs = [x.new_empty((st.num_padded,) + tuple(x.shape[1:])) for x in xs]
    main, side = current_stream(), _pf_sp_comm_stream()
    ev_in.record(main)
    with torch.cuda.stream(side):
        ev_in.wait(side)
        _pf_sp2_gather_group(list(zip(outs, xs)), side)
        ev_out.record(side)
    for x in xs:
        x.record_stream(side)
    return outs


def pf_sp2_reduce_scatter_async(
    x: torch.Tensor, ev_in: torch.cuda.Event, ev_out: torch.cuda.Event
) -> torch.Tensor:
    """Side-stream reduce-scatter of the ``[Tp, ...]`` TP partial ``x``; the
    main stream must wait on ``ev_out`` before reading the ``[c, ...]`` rows."""
    from vllm.utils.torch_utils import current_stream

    st = pf_sp_state
    assert x.shape[0] == st.num_padded, (tuple(x.shape), st.num_padded)
    x = x.contiguous()
    out = x.new_empty((st.shard,) + tuple(x.shape[1:]))
    main, side = current_stream(), _pf_sp_comm_stream()
    ev_in.record(main)
    with torch.cuda.stream(side):
        ev_in.wait(side)
        _pf_sp_pynccl().reduce_scatter(out, x, stream=side)
        ev_out.record(side)
    x.record_stream(side)
    return out


def pf_sp2_fp8_input_reason(lin) -> str | None:
    """Why ``lin`` cannot take the FP8 all-gathered input (None if it can):
    it must be a column-parallel FP8_PB_WO linear on the DeepGEMM block kernel
    whose quantizer is dynamic per-token 1x128 groups, so each token's FP8
    values and scales depend only on that token: quantizing the shard and
    gathering gives the operands its own apply would build."""
    qm = getattr(lin, "quant_method", None)
    kernel = getattr(qm, "kernel", None)
    fmt = getattr(qm, "fmt", None)
    if kernel is None or fmt is None or not hasattr(fmt, "apply"):
        return f"quant method {type(qm).__name__} has no kernel / format scheme"
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
            DeepGemmFp8BlockScaledMMKernel,
        )
    except Exception as exc:  # noqa: BLE001
        return f"import failed: {exc!r}"
    if type(kernel) is not DeepGemmFp8BlockScaledMMKernel:
        return f"kernel is {type(kernel).__name__}"
    q = getattr(kernel, "quant_fp8", None)
    if not (
        kernel.apply_input_quant
        and q is not None
        and q.is_group_quant
        and not q.static
        and q.group_size == 128
    ):
        return "the quantizer is not dynamic per-token 1x128"
    if getattr(lin, "gather_output", False):
        return "gather_output is set"
    return None


def _pf_sp2_scale_like(
    s: torch.Tensor, rows: int, full_rows: torch.Tensor
) -> torch.Tensor:
    """``full_rows`` (token-major ``[>= rows, G]`` scales) laid out as the
    quantizer lays out ``rows`` tokens, judged from its ``[c, G]`` shard output
    ``s``: token-major, or MN-major with the packed-UE8M0 int32 TMA alignment
    (4 rows) or the shard's own row stride rule."""
    if s.stride(-1) == 1:
        return full_rows[:rows]
    c = s.shape[0]
    stride = -(-rows // 4) * 4 if s.dtype == torch.int32 or s.stride(1) != c else rows
    out = torch.empty_strided(
        (rows, s.shape[1]), (1, stride), dtype=s.dtype, device=s.device
    )
    out.copy_(full_rows[:rows])
    return out


def pf_sp2_fp8_gather(lin, x: torch.Tensor) -> torch.Tensor:
    """Quantize this rank's ``[c, K]`` shard with ``lin``'s own quantizer and
    all-gather the FP8 values and the packed scales (main stream, one NCCL
    group). Returns the stand-in input for the attention: a ``[T, K]``
    zero-stride tensor the wrapped ``lin`` recognizes; any other input goes to
    the stock forward."""
    st = pf_sp_state
    kernel = lin.quant_method.kernel
    params = kernel._get_layer_params(lin)
    x = x.contiguous()
    q, s = kernel.quant_fp8(
        x, params.input_scale, params.input_scale_ub, use_triton=kernel.use_triton
    )
    assert q.dim() == 2 and s.dim() == 2 and s.shape[0] == x.shape[0]
    tp, k, t = st.num_padded, x.shape[1], st.num_tokens
    q_full = torch.empty((tp, k), dtype=q.dtype, device=x.device)
    s_rows = s.contiguous()  # [c, G], token-major
    s_full_rows = s_rows.new_empty((tp, s_rows.shape[1]))
    _pf_sp2_gather_group(
        [(q_full.view(torch.uint8), q.view(torch.uint8)), (s_full_rows, s_rows)], None
    )
    # The GEMM runs on the first T rows (the pad rows are dropped), with the
    # scales in the layout the quantizer gives T tokens: the same operands as
    # quantizing the gathered bf16 rows [:T].
    marker = x.new_zeros((1, 1)).expand(t, k)
    pf_sp2_state.fp8_input = (q_full[:t], _pf_sp2_scale_like(s, t, s_full_rows))
    pf_sp2_state.fp8_marker = marker
    return marker


def pf_sp2_wrap_fp8_linear(lin) -> None:
    """ARIA pf-sp v2: ``lin`` (the attention's first GEMM) consumes the FP8
    operands ``pf_sp2_fp8_gather`` stashed when it is called with that call's
    stand-in input; any other call is the module's own forward."""
    stock_forward = lin.forward

    def forward(input_):
        st2 = pf_sp2_state
        if st2.fp8_marker is None or input_ is not st2.fp8_marker:
            return stock_forward(input_)
        q, s = st2.fp8_input
        st2.fp8_input = None
        st2.fp8_marker = None
        qm = lin.quant_method
        kernel = qm.kernel

        def kernel_apply(layer, _x, bias):
            # Fp8BlockScaledMMLinearKernel.apply_weights minus the input quant.
            params = kernel._get_layer_params(layer)
            out = kernel.apply_block_scaled_mm(
                A=q, B=params.weight, As=s, Bs=params.block_scale
            )
            if bias is not None:
                out = out + bias
            return out.to(dtype=kernel.config.out_dtype)

        bias = lin.bias if not lin.skip_bias_add else None
        output = qm.fmt.apply(lin, q, bias, kernel_apply)
        if not lin.return_bias:
            return output
        return output, lin.bias if lin.skip_bias_add else None

    lin.forward = forward


class LatentTailTier(IntEnum):
    """Which tail implementation the fused path runs, by token count.

    The tiers share the same replicated up-projection weight, so the choice is
    per batch and needs no weight relayout: tiers 0 and 2 read it as a
    column-parallel row-shard, tier 1 reads it whole.
    """

    # ``_small_batch_tail``. Decode-sized (<= the op's max_num_tokens): one
    # CuTeDSL collective fuses the latent reduce, RMSNorm and the shared
    # reduce-scatter, then a sharded up-projection multicasts through a Lamport
    # copy. Requires SM100, TP 8/16, BF16
    TAIL_FUSION = 0

    # ``_overlap_allreduce_tail``. The portable default, up to
    # VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens: reduce the latent,
    # up-project the full hidden dim from the replicated weight, and add the
    # separately reduced shared output, hiding that shared all-reduce behind the
    # up-projection GEMM on the aux stream.
    ALLREDUCE_OVERLAP = 1

    # ``_shard_up_proj_tail``. Prefill-sized: each rank up-projects only its
    # hidden shard and accumulates into the shared partial, so the shared
    # all-reduce also stitches the routed shards. Same two all-reduces as tier 1
    # at 1/tp of the up-projection FLOPs; it gives up tier 1's aux-stream
    # overlap, since the reduce now has to follow the accumulate.
    COLUMN_PARALLEL = 2


class LatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    The fused path (tp>1, un-reduced combine output, shared expert, no SP)
    dispatches over ``LatentTailTier`` by token count; see that enum for what
    each tier does and when it applies.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # The tail-fusion kernels are tcgen05-based, so they require an
        # SM100 NVIDIA device; the runner falls back to the default latent
        # MoE path everywhere else.
        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        )
        # Overlap the shared-expert all-reduce with the tier-1 up-projection.
        self._shared_ar_events = (torch.cuda.Event(), torch.cuda.Event())
        use_fused_path = self._use_fused_path()
        if (
            self.enable_k3_latent_moe_tail_fusion
            and use_fused_path
            and self.moe_config.tp_size not in (8, 16)
        ):
            logger.warning_once(
                "K3 latent-MoE tail fusion currently supports TP=8 and TP=16, "
                "but TP=%d is configured. Falling back to the default path.",
                self.moe_config.tp_size,
            )
            self.enable_k3_latent_moe_tail_fusion = False

        if self.enable_k3_latent_moe_tail_fusion and use_fused_path:
            vllm_config = get_current_vllm_config()
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support DBO or ubatching."
                )
            if vllm_config.model_config.enable_sleep_mode:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support sleep mode."
                )
            transform = self.routed_output_transform
            assert transform is not None
            norm = transform.norm
            assert norm is not None
            from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
                TrtLlmMxfp4ExpertsMonolithic,
            )
            from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
                TrtLlmNvFp4ExpertsMonolithic,
            )

            experts_cls = getattr(self._quant_method, "experts_cls", None)
            # The kernel instance is built after weight loading, while the tail
            # must register its CuTeDSL warmup units during runner construction.
            if (
                experts_cls is TrtLlmMxfp4ExpertsMonolithic
                or experts_cls is TrtLlmNvFp4ExpertsMonolithic
            ) and self.moe_config.hidden_dim == self.moe_config.hidden_dim_unpadded:
                self.moe_config.defer_moe_finalize()
            from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import (
                KimiK3LatentMoETailOp,
            )

            op = KimiK3LatentMoETailOp.initialize(
                hidden_size=transform.up_proj.weight.shape[0],
                latent_size=norm.weight.shape[0],
                experts_per_token=(
                    self.moe_config.experts_per_token
                    if self.moe_config.use_deferred_moe_finalize
                    else 0
                ),
                dtype=norm.weight.dtype,
                device=current_platform.current_device(),
                rms_eps=norm.variance_epsilon,
            )
            self._k3_latent_moe_tail_op = op
            if self.moe_config.use_deferred_moe_finalize:
                self.moe_config.limit_deferred_moe_finalize(op.contract.max_num_tokens)
                logger.info_once(
                    "K3 latent-MoE tail fusion with deferred top-k finalization "
                    "is enabled for up to %d tokens.",
                    op.contract.max_num_tokens,
                )

    def _get_zero_residual(
        self,
        hidden_states: torch.Tensor,
        max_token_num: int,
    ) -> torch.Tensor:
        """Read-only zero ``residual_in`` for the fused AR+RMSNorm kernel.

        flashinfer requires a residual buffer even when there is no residual to
        add.
        """
        buf = getattr(self, "_zero_residual", None)
        if buf is None:
            buf = torch.zeros(
                max_token_num * hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._zero_residual = buf

        assert buf.dtype == hidden_states.dtype
        assert buf.device == hidden_states.device
        assert hidden_states.numel() <= buf.numel()

        return buf[: hidden_states.numel()].view_as(hidden_states)

    def _use_fused_path(self) -> bool:
        # The fused path merges the latent and shared reductions into one
        # all-reduce, so it needs actual TP parallelism, a shared expert (to
        # concat), an un-reduced combine output, and no sequence parallelism.
        return (
            self.moe_config.tp_size > 1
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and not self.moe_config.is_sequence_parallel
            # Tail kernels read the weight directly, without quantization scales.
            and self.routed_output_transform is not None
            and isinstance(
                self.routed_output_transform.up_proj.quant_method,
                UnquantizedLinearMethod,
            )
        )

    def _select_tail_tier(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> LatentTailTier:
        num_tokens = fused_output.shape[0]
        # tier 0
        if self.enable_k3_latent_moe_tail_fusion and (
            0 < num_tokens <= self._k3_latent_moe_tail_op.contract.max_num_tokens
        ):
            return LatentTailTier.TAIL_FUSION

        transform = self.routed_output_transform
        assert transform is not None
        # tier 1
        if (
            num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
        ):
            return LatentTailTier.ALLREDUCE_OVERLAP
        # tier 2
        return LatentTailTier.COLUMN_PARALLEL

    def _small_batch_tail(
        self,
        fused_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 0: the CuTeDSL operator fuses the whole tail."""
        transform = self.routed_output_transform
        assert transform is not None
        norm = transform.norm
        assert norm is not None

        result = self._k3_latent_moe_tail_op(
            fused_output,
            shared_output,
            norm.weight,
            transform.up_proj.weight,
        )
        # The operator already reduced; this only strips padding.
        return self._maybe_reduce_final_output(
            result, trunc_size, output_is_reduced=True
        )

    def _overlap_allreduce_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 1: reduce the latent, up-project the full hidden dim from the
        replicated weight, and add the separately reduced shared output.

        Small enough batches hide that shared all-reduce behind the up-projection
        GEMM on the aux stream.
        """
        transform = self.routed_output_transform
        assert transform is not None
        assert shared_output.size(0) <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        if transform.norm is not None:
            fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            fused_latent = tensor_model_parallel_all_reduce(fused_output)

        # Overlap the shared-expert all-reduce with the up-projection GEMM while
        # the batch is small enough for it to pay off.
        result, shared_output = maybe_execute_in_parallel(
            lambda: torch.mm(fused_latent, transform.up_proj.weight.t()),
            lambda: tensor_model_parallel_all_reduce(shared_output),
            self._shared_ar_events[0],
            self._shared_ar_events[1],
            aux_stream(),
        )
        result.add_(shared_output)

        # Output is already fully reduced; this only strips padding.
        return self._maybe_reduce_final_output(
            result, trunc_size, output_is_reduced=True
        )

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 2: column-parallel up-projection folded into the final reduce."""
        transform = self.routed_output_transform
        assert transform is not None

        if transform.norm is not None:
            latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            latent = tensor_model_parallel_all_reduce(fused_output)

        weight = transform.up_proj.weight
        shard_size = weight.shape[0] // self.moe_config.tp_size
        shard_start = get_tensor_model_parallel_rank() * shard_size

        # column-parallel
        up_proj_shard = weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def _pf_sp_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """pf-sp: reduce-scatter the latent and the shared partial, then
        up-project this rank's tokens over the full hidden dim with the
        replicated weight, accumulated into its reduced shared rows. Returns
        this rank's ``[c, hidden]`` token shard."""
        from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
            ZeroExpertRouter,
        )

        transform = self.routed_output_transform
        assert transform is not None
        assert not isinstance(self.router, ZeroExpertRouter)
        latent = pf_sp_reduce_scatter(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)
        out = pf_sp_reduce_scatter(shared_output)
        out.addmm_(latent, transform.up_proj.weight.t())
        return out[..., :trunc_size] if trunc_size is not None else out

    # ARIA pf-sp v2 state; only used when ARIA_PF_SP_V2=1.
    _pf_sp2_ev: tuple | None = None

    def pf_sp2_reason(self) -> str | None:
        """Why this runner cannot take the v2 MoE block (None if it can)."""
        from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
            ZeroExpertRouter,
        )

        transform = self.routed_output_transform
        if not self._use_fused_path():
            return "not the fused latent path"
        if transform is None or transform.norm is None:
            return "no latent RMSNorm"
        if self._shared_experts is None:
            return "no shared expert"
        if isinstance(self.router, ZeroExpertRouter):
            return "zero-expert router"
        if self.gate is not None:
            return "the runner holds the gate"
        if not self.routed_experts.quant_method.is_monolithic:
            return "the routed experts are not monolithic"
        if self.do_naive_dispatch_combine or self.moe_config.pcp_size > 1:
            return "naive dispatch / PCP"
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )

        layer = self._shared_experts._layer
        if getattr(layer, "gemm_rs_ar", None) is not None or getattr(
            layer, "shard_sequence_parallel", False
        ):
            return "the shared expert reduces itself"
        # Only the row count is read: v2 runs above 9362 tokens.
        rows = torch.empty((_PF_SP_MIN_TOKENS + 1, 1), device="meta")
        if (
            self._shared_experts._determine_shared_experts_order(rows)
            != SharedExpertsOrder.NO_OVERLAP
        ):
            return "the shared expert would not run in the main-stream order"
        return None

    def pf_sp2_forward(
        self, hidden_shard: torch.Tensor, router_and_down
    ) -> torch.Tensor:
        """pf-sp v2 MoE block. ``hidden_shard`` is this rank's ``[c, hidden]``
        rows; ``router_and_down(x) -> (latent, router_logits)`` is the model's
        router gate + latent down-projection, run here on the shard. Returns
        this rank's ``[c, hidden]`` MoE output.

        Side stream: AG(hidden) | AG(latent, logits) | RS(shared partial) |
        RS(latent). Main stream: gate + down on the shard; the shared expert
        once AG(hidden) lands (under AG(latent, logits)); the routed experts
        once AG(latent, logits) lands (under RS(shared)); then RMSNorm and the
        up-projection of this rank's tokens into its reduced shared rows. The
        kernels are v1's; only the gate / down GEMMs run on c rows, not Tp."""
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )
        from vllm.utils.torch_utils import current_stream

        transform = self.routed_output_transform
        assert transform is not None and transform.norm is not None
        if self._pf_sp2_ev is None:
            self._pf_sp2_ev = tuple(torch.cuda.Event() for _ in range(8))
        ev = self._pf_sp2_ev
        main = current_stream()
        self.routed_experts._ensure_moe_quant_config_init()

        (hidden_full,) = pf_sp2_all_gather_async([hidden_shard], ev[0], ev[1])
        latent_shard, logits_shard = router_and_down(hidden_shard)
        latent_full, logits_full = pf_sp2_all_gather_async(
            [latent_shard, logits_shard], ev[2], ev[3]
        )

        ev[1].wait(main)
        prev = _pf_sp_set_carveout(_PF_SP_CARVEOUT)
        try:
            self._maybe_apply_shared_experts(hidden_full, SharedExpertsOrder.NO_OVERLAP)
        finally:
            _pf_sp_restore_carveout(prev)
        del hidden_full
        shared_rows = pf_sp2_reduce_scatter_async(
            self._shared_experts.output, ev[4], ev[5]
        )

        ev[3].wait(main)
        latent_full, og_pre, og_post = self._maybe_pad_hidden_states(
            hidden_shard, latent_full
        )
        fused = self.routed_experts.forward_monolithic(
            x=latent_full, router_logits=logits_full, input_ids=None
        )
        assert not isinstance(fused, UnfinalizedMoEOutput)
        if og_pre is not None:
            fused = fused[..., :og_pre]
        latent_rows = pf_sp2_reduce_scatter_async(fused, ev[6], ev[7])

        ev[7].wait(main)  # the side stream ran RS(shared) before RS(latent)
        out = shared_rows
        out.addmm_(transform.norm(latent_rows), transform.up_proj.weight.t())
        pf_sp2_log_once(
            "moe",
            "MoE block on the token shard (first at %d tokens, %d rows per rank; "
            "router gate + latent down on the shard, collectives on the side stream)",
            pf_sp_state.num_tokens,
            pf_sp_state.shard,
        )
        return out[..., :og_post] if og_post is not None else out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_fused_path():
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        if _PF_SP and pf_sp_state.active:
            raise RuntimeError("ARIA pf-sp needs the fused latent MoE path")
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        if (
            og_hidden_dim_pre_xform is None
            and self.moe_config.should_defer_moe_finalize(hidden_states.shape[0])
        ):
            assert not (_PF_SP and pf_sp_state.active)
            unfinalized = self._forward_impl(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )
            shared_output, fused_output = _unpack(unfinalized)
            assert shared_output is not None
            assert isinstance(fused_output, UnfinalizedMoEOutput)
            result = self._small_batch_tail(fused_output, shared_output, None)
            return self._maybe_add_zero_expert_output(result)

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        tier = self._select_tail_tier(fused_output, shared_output)
        if _PF_SP and pf_sp_state.active:
            latent_tail = self._pf_sp_tail
        elif tier is LatentTailTier.TAIL_FUSION:
            latent_tail = self._small_batch_tail
        elif tier is LatentTailTier.ALLREDUCE_OVERLAP:
            latent_tail = self._overlap_allreduce_tail
        else:
            latent_tail = self._shard_up_proj_tail

        result = latent_tail(fused_output, shared_output, og_hidden_dim_post_xform)
        return self._maybe_add_zero_expert_output(result)

    def allreduce_norm_latent_out(
        self,
        hidden_states: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer."""
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _AR_RESIDUAL_RMS_NORM,
            _can_use_flashinfer,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        if self.moe_config.tp_size == 1:
            return norm(hidden_states)

        if flashinfer_trtllm_fused_allreduce_norm is not None:
            ok, max_token_num = _can_use_flashinfer(
                hidden_states, self.moe_config.tp_size
            )
            if ok:
                norm_out = torch.empty_like(hidden_states)
                # With norm_out provided, the kernel writes the new residual
                # (all_reduce(hidden_states) + residual) into the hidden_states
                # buffer and the normalized result into norm_out.
                flashinfer_trtllm_fused_allreduce_norm(
                    allreduce_in=hidden_states,
                    residual=self._get_zero_residual(hidden_states, max_token_num),
                    rms_gamma=norm.weight,
                    rms_eps=norm.variance_epsilon,
                    world_size=self.moe_config.tp_size,
                    weight_bias=0.0,
                    launch_with_pdl=True,
                    fp32_acc=True,
                    max_token_num=max_token_num,
                    pattern_code=_AR_RESIDUAL_RMS_NORM,
                    norm_out=norm_out,
                )
                return norm_out

        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced)
