# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuDNN FROST KDA prefill with the KDA short conv fused into its q/k/v loads.

The plan is cuDNN's own (``CompiledKda`` of cudnn-frontend 1.33 over a graph node
of this call's shapes); only its uncut launch is swapped for the vendored one in
this package, whose TMA loads read the raw Q|K|V columns of the in_proj output
and whose CG0 warps convolve them (stock conv arithmetic, bit for bit) before the
q/k L2 norm and the V operand. A tail launch (the (sequence, head) tiles past the
full waves: cuDNN's d_v split or chain kernels) reads conv'd planes of its
sequences from qkv_conv. Prep-path and piece-chain plans report supported
False; the caller then runs the conv and FROST unfused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

_DTYPE_NAMES = {
    torch.bfloat16: "BFLOAT16",
    torch.float16: "HALF",
    torch.float32: "FLOAT",
    torch.int32: "INT32",
    torch.int64: "INT64",
}


@dataclass
class _Dtype:
    name: str


@dataclass
class _Port:
    dim: tuple
    dtype: torch.dtype

    def get_data_type(self) -> _Dtype:
        return _Dtype(_DTYPE_NAMES[self.dtype])


@dataclass
class _Node:
    inputs: dict
    outputs: dict
    params: dict = field(default_factory=dict)


_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "cu_seqlens",
    "initial_state",
    "O",
    "final_state",
    "a_log",
    "dt_bias",
)


class FrostConvKda:
    """One FROST plan with the fused conv at (tokens, sequences, heads)."""

    def __init__(
        self,
        tokens: int,
        num_seqs: int,
        heads: int,
        dim: int,
        lower_bound: float,
        device: torch.device,
    ) -> None:
        from cudnn.linear_attention.frost.kda_engine import CompiledKda

        from . import kda_prefill_f16, kda_uncut_forward_f16

        bf, f32 = torch.bfloat16, torch.float32
        T, N, H, D = tokens, num_seqs, heads, dim
        node = _Node(
            inputs={
                "q": _Port((T, H, D), bf),
                "k": _Port((T, H, D), bf),
                "v": _Port((T, H, D), bf),
                "g": _Port((T, H, D), bf),
                "beta": _Port((T, H), bf),
                "cu_seqlens": _Port((N + 1,), torch.int32),
                "initial_state": _Port((N, H, D, D), f32),
                "a_log": _Port((H,), f32),
                "dt_bias": _Port((H, D), f32),
            },
            outputs={"O": _Port((T, H, D), bf), "final_state": _Port((N, H, D, D), f32)},
            params=dict(
                scale=D**-0.5,
                use_qk_l2norm=True,
                use_beta_sigmoid=True,
                safe_gate=True,
                gate_lower_bound=float(lower_bound),
            ),
        )
        plan = CompiledKda(node, kda_prefill_f16)
        self.supported = not (plan.chain or plan.prep or plan.dv_split)
        # A tail launch (the tiles past the full waves) runs cuDNN's own kernels on
        # conv'd planes of its sequences (qkv_conv over those sequences only).
        self.tail_seq = plan.tail.batch_base if plan.tail is not None else None
        if not self.supported:
            return
        # Ragged pieces split long sequences into chain pieces at run time; those
        # kernels read conv'd planes, so the fused plan runs every tile uncut.
        plan.ragged_pieces = 0
        plan.bind(list(_NAMES))
        self._conv: dict = {}
        plan.build_uncut_forward = lambda **kw: kda_uncut_forward_f16.build_uncut_forward(
            **kw, **self._conv
        )
        plan.run_uncut_forward = lambda compiled, **kw: (
            kda_uncut_forward_f16.run_uncut_forward(compiled, **kw, **self._conv)
        )
        self.plan = plan
        self.shape = (T, N, H, D)
        self.device = device
        self.workspace = torch.empty(
            max(plan.workspace_size, 1), dtype=torch.uint8, device=device
        )

    def __call__(
        self,
        qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state: torch.Tensor,
        conv_slot: torch.Tensor,
        conv_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """qkv: raw in_proj Q|K|V columns (T, 3 H D) bf16, rows 16 B aligned;
        g (T, H, D) raw gate logits; beta (T, H) logits (row-strided OK);
        conv_weight (3 H D, 4) FP32; conv_state (lines, 3 H D, >= 3);
        conv_slot (N,) int32 = the sequence's conv-state line, -1 for none;
        conv_indices / has_initial_state: the qkv_conv metadata (tail planes).
        Reads the conv state; qkv_conv_state updates it afterwards."""
        from cudnn.frost.workspace import Workspace

        T, N, H, D = self.shape
        q, k, v = (qkv[:, i * H * D : (i + 1) * H * D].view(T, H, D) for i in range(3))
        o = torch.empty(T, H, D, dtype=qkv.dtype, device=qkv.device)
        final_state = torch.empty(N, H, D, D, dtype=torch.float32, device=qkv.device)
        tail = (None, None, None)
        if self.tail_seq is not None:
            from vllm.models.kimi_k3.nvidia.ops import kda_conv

            b = self.tail_seq
            planes = kda_conv.qkv_conv(
                qkv,
                conv_weight,
                conv_state,
                cu_seqlens[b:],
                conv_indices[b:],
                has_initial_state[b:],
                H * D,
                write_state=False,
            )
            tail = tuple(t.view(T, H, D) for t in planes.unbind(0))
        self._conv = dict(
            conv_w=conv_weight,
            conv_state=conv_state,
            conv_slot=conv_slot,
            conv_tail_q=tail[0],
            conv_tail_k=tail[1],
            conv_tail_v=tail[2],
        )
        views = [
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            initial_state,
            o,
            final_state,
            A_log,
            dt_bias.view(H, D),
        ]
        ws = Workspace(
            self.workspace,
            self.plan.workspace_size,
            "frost_kda_conv",
            device=self.workspace.device.index,
        )
        self.plan.run(views, ws, torch.cuda.current_stream(self.device).cuda_stream)
        return o, final_state


_plans: dict = {}


def frost_conv_plan(
    tokens: int, num_seqs: int, heads: int, dim: int, lower_bound: float, device
) -> FrostConvKda:
    key = (tokens, num_seqs, heads, dim, float(lower_bound), device)
    plan = _plans.get(key)
    if plan is None:
        plan = _plans[key] = FrostConvKda(
            tokens, num_seqs, heads, dim, lower_bound, device
        )
    return plan
