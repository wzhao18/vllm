# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM107 specialized decode's deployment topology, computed once at load.

``Topology`` describes the parallel layout the decode step has to serve:

- ``g``: the attention group, the TP size inside one DP replica.
- ``dcp``: decode context parallel over the group (1 or ``g``).
- ``dp``: data-parallel replicas.
- ``ep``: expert-parallel ranks of the MoE (``g x dp`` under expert parallelism,
  else 1).
- ``moe_backend``: the kernel config's MoE backend.
- ``q_set``: the verify lengths (tokens per request) of the decode graphs.
- ``stock_sp``: the stock decoder's ``use_sequence_parallel`` (PP1, expert
  parallelism, ``g > 1`` and native MegaMoE or ``dp > 1``): the stock residual
  stream is row-sharded over the group.
- ``moe_tokens``: the MegaMoE token capacity per rank the stock model builds
  (``max_num_batched_tokens``, divided over the group under stock SP; 0 = unknown).

It selects one of three modes (``Mode``):

- ``TP_EXPERTS``: no expert parallelism; TP-sharded experts (TP8 / TP16, DP1:
  stock shards the experts over DP x TP when DP > 1, refused).
- ``EP_GROUP``: expert parallelism over attention groups of ``g > 1`` ranks (4, 8
  or 16), DCP 1 or ``g``, any DP. Two residual layouts (``step_layout``): C512's
  sequence-parallel step (TP4 = DCP4, ``sp_step``: stock's row shard, layout (c),
  ``runtime.sp_attention``) and the replicated step everywhere else (layout (b),
  ``runtime.group_ep``).
- ``DEP``: expert parallelism with ``g = 1`` (data-parallel attention).

``refusal`` gives the reason a topology is not served (None when it is). It is the
only topology gate of the decode: a refused topology raises at load, never a silent
fallback to the stock forward. DEP is refused ("mode DEP not implemented on sm_107
yet"); the EP_GROUP layouts' own deployment checks (KV dtype, rows, speculation)
follow in ``runtime.sp_unsupported`` / ``runtime.ep_group_unsupported``.

CPU-only: nothing here imports CuTe DSL or a device API (the Rubin MegaMoE record
checks read the shipped JSON file).
"""

import enum
import functools
import json
from dataclasses import dataclass


class Mode(enum.Enum):
    TP_EXPERTS = "tp_experts"
    EP_GROUP = "ep_group"
    DEP = "dep"


class Layout(enum.Enum):
    """Where a residual stream lives between layers."""

    REPLICATED = "replicated"  # every group rank holds every row
    ROW_SHARDED = "row_sharded"  # stock sequence parallelism (``sp_shard``)
    LOCAL = "local"  # g = 1: the rank's own rows


# Attention-group sizes served without expert parallelism (TP-sharded experts).
TP_EXPERTS_GROUPS = (8, 16)
# Attention-group sizes of the EP modes: h = 96 / g heads per rank (96, 24, 12, 6).
# TP2 is out of scope (no TP2 kernels; vocab / TP2 exceeds the argmax's 65536).
EP_GROUPS = (1, 4, 8, 16)
# The EP_GROUP topology the SP step serves (C512's decode role).
SP_STEP_GROUP = 4


@dataclass(frozen=True)
class Topology:
    g: int
    dcp: int
    dp: int
    ep: int
    moe_backend: str
    q_set: tuple[int, ...]
    stock_sp: bool
    moe_tokens: int = 0

    @property
    def expert_parallel(self) -> bool:
        return self.ep > 1

    @property
    def mode(self) -> Mode:
        if not self.expert_parallel:
            return Mode.TP_EXPERTS
        return Mode.DEP if self.g == 1 else Mode.EP_GROUP

    @property
    def layout(self) -> Layout:
        """The stock decoder's residual layout."""
        if self.g == 1:
            return Layout.LOCAL
        return Layout.ROW_SHARDED if self.stock_sp else Layout.REPLICATED

    @property
    def sp_step(self) -> bool:
        """The EP_GROUP case the SP step (``sp_attention.SpStep``) serves: TP4 = DCP4
        attention groups over native MegaMoE, the stock residual row-sharded."""
        from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS

        return (
            self.mode is Mode.EP_GROUP
            and self.stock_sp
            and self.moe_backend in NATIVE_MEGA_MOE_BACKENDS
            and self.g == SP_STEP_GROUP
            and self.dcp == SP_STEP_GROUP
        )

    @property
    def step_layout(self) -> Layout:
        """Where the specialized step keeps the residual stream: EP_GROUP's SP step
        keeps stock's row shard (layout (c), ``runtime.sp_attention``), every other
        EP_GROUP topology replicates it over the group (layout (b),
        ``runtime.group_ep``: o_proj partial -> stock reduce-scatter + all-gather ->
        replicated AttnRes, the MoE on stock's row shard); TP_EXPERTS replicated,
        DEP local (the stock layout)."""
        if self.mode is Mode.EP_GROUP:
            return Layout.ROW_SHARDED if self.sp_step else Layout.REPLICATED
        return self.layout

    def layout_at(self, rows: int) -> Layout:
        """The step layout of the captured graph of ``rows`` token rows (the
        coverage table's "layout"). One per deployment today: each layout's step
        owns its own mailboxes, so mixing them per R would build both; a measured
        per-R choice is W8's."""
        if rows <= 0:
            raise ValueError(f"layout_at: {rows} rows")
        return self.step_layout

    def __str__(self) -> str:
        return (
            f"{self.mode.value} TP{self.g} DCP{self.dcp} x DP{self.dp} EP{self.ep} "
            f"({self.moe_backend}, q{list(self.q_set)}, {self.layout.value}, step "
            f"{self.step_layout.value})"
        )

    @classmethod
    def from_config(cls, vllm_config) -> "Topology":
        """The deployment's topology (never raises for an unsupported one:
        ``refusal`` decides)."""
        from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS

        parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
        g, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
        ep = g * dp if parallel.enable_expert_parallel else 1
        backend = vllm_config.kernel_config.moe_backend
        q = 1 + spec.num_speculative_tokens if spec is not None else 1
        query_lens = getattr(vllm_config, "get_decode_cudagraph_query_lens", None)
        q_set = tuple(sorted(query_lens(q))) if query_lens is not None else (q,)
        # The stock K3 decoder's ``use_sequence_parallel`` (model.py).
        stock_sp = (
            parallel.pipeline_parallel_size == 1
            and parallel.enable_expert_parallel
            and g > 1
            and (backend in NATIVE_MEGA_MOE_BACKENDS or dp > 1)
        )
        # The stock KimiMoE's MegaMoE ``max_num_tokens``: max_num_batched_tokens,
        # cdiv over the TP group under stock SP (each rank sees its row shard).
        scheduler = getattr(vllm_config, "scheduler_config", None)
        tokens = getattr(scheduler, "max_num_batched_tokens", None)
        moe_tokens = 0
        if isinstance(tokens, int) and tokens > 0 and ep > 1:
            moe_tokens = -(-tokens // g) if stock_sp else tokens
        return cls(
            g=g,
            dcp=parallel.decode_context_parallel_size,
            dp=dp,
            ep=ep,
            moe_backend=str(backend),
            q_set=q_set,
            stock_sp=stock_sp,
            moe_tokens=moe_tokens,
        )


@functools.cache
def rubin_megamoe_records() -> tuple[tuple, ...]:
    """(world size, dtype, kernel variant, max_tokens, live_tokens) of every shipped
    ``kimi_k3_mega_moe`` tuning record of the vendored kernel revision."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.config import (
        BACKEND_REVISION,
        SHIPPED_KNOBS,
    )

    with open(SHIPPED_KNOBS) as f:
        entries = json.load(f).get("entries", ())
    return tuple(
        (
            e["world_size"],
            e.get("dtype"),
            e.get("kernel_variant"),
            e.get("max_tokens"),
            e.get("live_tokens"),
        )
        for e in entries
        if isinstance(e, dict)
        and e.get("backend_revision") == BACKEND_REVISION
        and isinstance(e.get("world_size"), int)
        and isinstance(e.get("max_tokens"), int)
    )


@functools.cache
def rubin_megamoe_world_sizes() -> frozenset[int]:
    """EP world sizes with a shipped ``kimi_k3_mega_moe`` tuning record of the
    vendored kernel revision (``megamoe_rubin.config.resolve_knobs`` raises on a
    miss at run time; the decode refuses it at load instead)."""
    return frozenset(record[0] for record in rubin_megamoe_records())


def rubin_megamoe_bucket(capacity: int) -> tuple[str, str]:
    """(dtype, kernel variant) the Rubin MegaMoE experts pick for ``capacity``
    tokens per rank (``megamoe_rubin.experts.rubin_quant_kind`` /
    ``rubin_kernel_variant``)."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.experts import (
        rubin_kernel_variant,
        rubin_quant_kind,
    )

    return rubin_quant_kind(capacity), rubin_kernel_variant(capacity)


def rubin_megamoe_capacity_gap(world_size: int, capacity: int) -> str | None:
    """Why ``resolve_knobs`` would find no record for EP ``world_size`` at
    ``capacity`` tokens per rank (its exact match on world size, dtype and kernel
    variant, then the token-bucket selection of ``_select_token_entries``), None
    when it finds one."""
    dtype, variant = rubin_megamoe_bucket(capacity)
    matches = [
        r
        for r in rubin_megamoe_records()
        if r[0] == world_size and r[1] == dtype and r[2] == variant
    ]
    exact = any(r[3] == capacity for r in matches)
    plain = any(r[4] is None or r[4] == r[3] for r in matches)
    if exact or plain:
        return None
    tuned = sorted({(r[1], r[2], r[3]) for r in rubin_megamoe_records()
                    if r[0] == world_size})  # fmt: skip
    listed = ", ".join(f"{d} {v} {m}" for d, v, m in tuned) or "none"
    return (
        f"no Rubin MegaMoE tuning record for (world size {world_size}, capacity "
        f"{capacity} tokens per rank: {dtype} {variant}) (records at EP{world_size}: "
        f"{listed})"
    )


def refusal(topology: Topology, *, dsa: bool, pp: int = 1) -> str | None:
    """Why the SM107 decode does not serve ``topology`` (completes "Kimi K3
    specialized decode does not support ..."), None when it does. ``dsa``: the model
    has DSA layers (K3.1); ``pp``: the pipeline-parallel size."""
    from vllm.config.kernel import KIMI_K3_MEGA_MOE, NATIVE_MEGA_MOE_BACKENDS

    t = topology
    if pp != 1:
        return "pipeline parallelism"
    if t.dcp not in (1, t.g):
        return (
            f"decode context parallel other than 1 or TP (DCP{t.dcp} at TP{t.g}; "
            "DCP strictly between 1 and TP is out of scope)"
        )
    if t.mode is Mode.TP_EXPERTS:
        if t.g not in TP_EXPERTS_GROUPS:
            return (
                f"TP other than 8/16 without expert parallelism (TP{t.g}; the "
                "TP-sharded experts serve TP8 / TP16 only)"
            )
        if t.dp > 1:
            # Stock FusedMoE flattens TP across DP without expert parallelism
            # (``flatten_tp_across_dp_and_pcp``: experts sharded over DP x TP ranks,
            # a DP-wide dispatch / combine); the TP-sharded experts reduce over the TP
            # group only.
            return (
                f"data parallelism without expert parallelism (DP{t.dp} x TP{t.g}: "
                "stock shards the experts over DP x TP; the TP-sharded experts "
                "reduce over the TP group only)"
            )
        return None
    # Expert parallelism (EP_GROUP / DEP).
    if t.moe_backend not in NATIVE_MEGA_MOE_BACKENDS:
        return (
            f"expert parallelism on MoE backend {t.moe_backend} (the EP modes run "
            f"native MegaMoE only: {', '.join(sorted(NATIVE_MEGA_MOE_BACKENDS))})"
        )
    if not dsa:
        return (
            "expert parallelism on models without DSA layers (Kimi K3 MLA; the EP "
            "modes serve Kimi K3.1 only)"
        )
    if t.g not in EP_GROUPS:
        return (
            f"TP{t.g} under expert parallelism (attention groups of "
            f"{'/'.join(map(str, EP_GROUPS))} ranks only)"
        )
    if t.moe_backend == KIMI_K3_MEGA_MOE:
        tuned = rubin_megamoe_world_sizes()
        if t.ep not in tuned:
            listed = ", ".join(f"EP{w}" for w in sorted(tuned))
            return (
                f"{KIMI_K3_MEGA_MOE} at EP{t.ep}: no Rubin MegaMoE tuning record for "
                f"world size {t.ep} (tuned: {listed})"
            )
        gap = rubin_megamoe_capacity_gap(t.ep, t.moe_tokens) if t.moe_tokens else None
        if gap is not None:
            return f"{KIMI_K3_MEGA_MOE} at EP{t.ep}: {gap}"
    if t.mode is Mode.DEP:
        return f"mode DEP not implemented on sm_107 yet (TP1 x DP{t.dp}, EP{t.ep})"
    # EP_GROUP: TP4 / TP8 / TP16, DCP 1 or TP, any DP; the layout's own checks follow
    # (``runtime.sp_unsupported`` / ``runtime.ep_group_unsupported``).
    return None
