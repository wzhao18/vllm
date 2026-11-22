# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic MoE routing for forward benchmarks.

Dummy gate weights give degenerate routing, which skews MoE and all-to-all
cost. The routing still runs; its result is replaced by the fixture at
whichever boundary the layer exposes:

- Gate output: fixed logits whose top-k is the fixture. Covers every backend
  that routes from logits, including monolithic kernels that run top-k
  internally (e.g. FlashInfer TRTLLM).
- Experts input: models that compute top-k themselves and pass
  `topk_ids` to their experts (e.g. MegaMoE with a fused gate) get the
  fixture's IDs. The routing weights are kept; they don't affect cost.

Hash-routed layers ignore logits and keep their tables unless their experts
take `topk_ids`.
"""

import inspect
import zlib

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

logger = init_logger(__name__)

# Fixture logits: chosen experts score far above the rest under any monotonic
# scoring function, so top-k picks exactly them.
_CHOSEN_LOGIT = 4.0
_OTHER_LOGIT = -4.0


def balanced_topk_ids(
    num_tokens: int, num_experts: int, top_k: int, ep_size: int, rng
) -> np.ndarray:
    """Expert IDs [num_tokens, top_k] giving every EP rank equal load.

    Each token's slots go round-robin over EP ranks, to distinct experts.
    When ``num_experts`` is not a multiple of ``ep_size`` (redundant-slot
    padding, linear placement), rank ``r`` owns logical experts
    ``[r * local, min((r + 1) * local, num_experts))``.
    """
    local = -(-num_experts // ep_size)
    owned = np.clip(num_experts - np.arange(ep_size) * local, 0, local)
    slots = np.arange(num_tokens)[:, None] * top_k + np.arange(top_k)
    ranks = slots % ep_size
    # Slots of one token that land on the same rank take consecutive experts.
    repeat = np.zeros_like(ranks)
    for j in range(1, top_k):
        repeat[:, j] = (ranks[:, :j] == ranks[:, j : j + 1]).sum(axis=1)
    assert owned.min() > 0 and repeat.max() < owned.min()
    base = rng.integers(local, size=(num_tokens, ep_size))
    offset = np.take_along_axis(base, ranks, axis=1)
    return ranks * local + (offset + repeat) % owned[ranks]


def zipf_topk_ids(
    num_tokens: int, num_experts: int, top_k: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample distinct experts per token with Zipf(alpha=1.25) popularity."""
    expert_ids = rng.permutation(num_experts)
    inverse_weights = np.arange(1, num_experts + 1, dtype=np.float64) ** 1.25
    ids = np.empty((num_tokens, top_k), dtype=np.int64)
    for start in range(0, num_tokens, 1024):
        count = min(1024, num_tokens - start)
        # Exponential races sample weighted top-k without replacement.
        scores = rng.exponential(size=(count, num_experts)) * inverse_weights
        selected = np.argpartition(scores, top_k - 1, axis=1)[:, :top_k]
        ids[start : start + count] = expert_ids[selected]
    return ids


def _fixture_rows(fixture: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Select this rank's rows when the model shards tokens across TP ranks."""
    descriptor = get_forward_context().batch_descriptor
    offset = 0
    if descriptor is not None:
        tp = get_tp_group()
        full = descriptor.num_tokens
        if num_tokens < full and num_tokens == -(-full // tp.world_size):
            offset = tp.rank_in_group * num_tokens
    return fixture[offset : offset + num_tokens]


def _moe_layers(model: torch.nn.Module) -> list[tuple[torch.nn.Module, GateLinear]]:
    """(MoE module, gate) per layer, in model order."""
    layers: dict[GateLinear, torch.nn.Module] = {}
    for module in model.modules():
        gate = getattr(module, "gate", None)
        if not isinstance(gate, GateLinear):
            gate = getattr(module, "router", None)
        if not isinstance(gate, GateLinear) or gate in layers:
            continue
        if _top_k(module) is not None:
            layers[gate] = module
    return [(module, gate) for gate, module in layers.items()]


def _top_k(module: torch.nn.Module) -> int | None:
    """The MoE layer's top-k, from the module or its experts submodules."""
    for sub in module.modules():
        for name in ("top_k", "n_activated_experts"):
            top_k = getattr(sub, name, None)
            if isinstance(top_k, int):
                return top_k
    return None


class RoutingFixture:
    """Installs synthetic routing for one case; `remove()` restores the model."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_tokens: int,
        dp_rank: int,
        dp_size: int,
        ep_size: int,
        seed: int,
        device: torch.device,
        mode: str,
    ):
        layers = _moe_layers(model)
        if not layers:
            raise ValueError("No MoE gates found; use --routing real.")
        self._hooks = []
        self._biases: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._fired = [0] * len(layers)
        self._layers = layers
        # Per-rank rows, with headroom for padding; every rank generates the
        # same global fixture and takes its slice.
        rows = -(-num_tokens // 256) * 256 + 256
        for layer, (module, gate) in enumerate(layers):
            if (getattr(module, "num_expert_group", None) or 1) > 1:
                raise NotImplementedError("Synthetic routing with grouped top-k.")
            num_experts = gate.weight.shape[0]
            top_k = _top_k(module)
            assert top_k is not None
            layer_seed = layer
            if mode == "zipf" and gate.prefix:
                # Preserve layer popularity when pipeline stages change.
                layer_seed = zlib.crc32(gate.prefix.encode())
            rng = np.random.default_rng([seed, layer_seed])
            if mode == "zipf":
                ids = zipf_topk_ids(rows * dp_size, num_experts, top_k, rng)
            else:
                ids = balanced_topk_ids(
                    rows * dp_size, num_experts, top_k, ep_size, rng
                )
            ids = ids[dp_rank * rows : (dp_rank + 1) * rows]
            logits = torch.full((rows, num_experts), _OTHER_LOGIT, dtype=torch.float32)
            logits.scatter_(1, torch.from_numpy(ids), _CHOSEN_LOGIT)
            # Distinct values: no ties for any top-k implementation.
            logits += torch.from_numpy(rng.uniform(0, 0.5, logits.shape)).float()
            self._hooks.append(
                gate.register_forward_hook(
                    self._substitute(logits.to(device), self._fired, layer)
                )
            )
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is None:
                bias = getattr(module, "e_score_correction_bias", None)
            if bias is None:
                bias = getattr(module, "expert_bias", None)
            if isinstance(bias, torch.Tensor):
                self._biases.append((bias, bias.detach().clone()))
                bias.data.zero_()
            experts = getattr(module, "experts", None)
            if isinstance(experts, torch.nn.Module):
                params = list(inspect.signature(experts.forward).parameters)
                if "topk_ids" in params:
                    self._hooks.append(
                        experts.register_forward_pre_hook(
                            self._substitute_ids(
                                torch.from_numpy(ids).to(device),
                                params.index("topk_ids"),
                                self._fired,
                                layer,
                            ),
                            with_kwargs=True,
                        )
                    )

    @staticmethod
    def _substitute(fixture: torch.Tensor, fired: list[int], layer: int):
        cast = [fixture]

        def hook(module, inputs, output):
            fired[layer] += 1
            logits = output[0] if isinstance(output, tuple) else output
            if cast[0].dtype != logits.dtype:
                cast[0] = fixture.to(logits.dtype)  # Once, at warmup.
            replaced = _fixture_rows(cast[0], logits.shape[0])
            return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

        return hook

    @staticmethod
    def _substitute_ids(fixture: torch.Tensor, index: int, fired, layer: int):
        cast = [fixture]

        def hook(module, args, kwargs):
            fired[layer] += 1
            ids = kwargs["topk_ids"] if "topk_ids" in kwargs else args[index]
            if cast[0].dtype != ids.dtype:
                cast[0] = fixture.to(ids.dtype)  # Once, at warmup.
            replaced = _fixture_rows(cast[0], ids.shape[0])
            if "topk_ids" in kwargs:
                return args, {**kwargs, "topk_ids": replaced}
            return (*args[:index], replaced, *args[index + 1 :]), kwargs

        return hook

    def check_applied(self) -> None:
        """Fail if a layer bypassed both the gate and the experts hooks."""
        missed = [
            type(module).__name__
            for (module, _), fired in zip(self._layers, self._fired)
            if not fired
        ]
        if missed:
            raise RuntimeError(
                f"Synthetic routing didn't reach {len(missed)} MoE layers "
                f"({missed[0]}): their gate is fused away. Use --routing real."
            )

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        for bias, saved in self._biases:
            bias.data.copy_(saved)


# --- EPLB DIAGNOSTIC (never merge): sdec route_stage routing modes ----------------------------------------
# With EPLB_ROUTE_FIX=1 (sp_mlp.EPLB_FIX buffers captured in the graph) a case named "<shape>~<mode>~s<seed>" runs:
#   real     real router on the bench activations, real correction bias (= --routing real)
#   real0    the same with e_score_correction_bias zeroed (= what --routing balanced did in the sdec graph)
#   bal      every token's 16 slots on the 16 EP ranks, one each, uniform expert within the rank (true balance)
#   uni      16 distinct experts per token, uniform over all experts (multinomial noise only)
#   rep      served decode routes: blocks of 4 consecutive shard rows drawn from EPLB_REPLAY_POOL (npz: blocks
#            [N, 4, layers, top_k] int16, logical ids, linear placement = no EPLB)
#   perm     rep relabelled per layer by EPLB_REPLAY_POOL's perm [layers, E] (an R=0 EPLB placement fitted on
#            other served steps: out of sample)
#   permin   the same with perm_in, fitted on the replayed steps themselves (in sample: the static-popularity bound)
#   permr    rep under an R=16 EPLB placement emulated without extra memory: the 16 hottest experts get a replica
#            hosted in one of the 16 coldest experts' slots, odd tokens go to the replica (pool permr/permr_rep,
#            fitted out of sample); permrin = the same fitted in sample
#   permact  rep under an R=0 placement balancing expected ACTIVE experts per rank (LPT on P(expert active in a
#            768-token step)), out of sample; permactin = in sample
# Every GPU builds every GPU's ids from (seed, layer, rank), so rank 0 can print the global skew; after the case
# each rank's seen buffers (the ids MegaMoE read on the last replay) are all-reduced into per-layer expert counts
# and rank 0 logs "EPLBFIX <case> ..." skew lines.
_POOL = {}


def _eplb_pool():
    import os

    path = os.environ["EPLB_REPLAY_POOL"]
    if path not in _POOL:
        d = np.load(path)
        perms = {"perm": (d["perm"], None), "permin": (d["perm_in"], None),
                 "permr": (d["permr"], d["permr_rep"]), "permrin": (d["permr_in"], d["permr_in_rep"])}
        if "perm_act" in d:
            perms.update(permact=(d["perm_act"], None), permactin=(d["perm_act_in"], None))
        _POOL[path] = (d["blocks"], perms)
    return _POOL[path]


def _eplb_ids(mode, rows, num_experts, top_k, ep_size, seed, layer, layer_col, rank):
    rng = np.random.default_rng([seed, layer, rank, 0xE91B])
    if mode == "bal":
        return balanced_topk_ids(rows, num_experts, top_k, ep_size, rng)
    if mode == "uni":
        return np.argsort(rng.random((rows, num_experts)), axis=1)[:, :top_k]
    blocks, perms = _eplb_pool()
    # one draw of whole blocks per (seed, rank), shared by every layer: a token keeps its routes across layers
    tok_rng = np.random.default_rng([seed, rank, 0xB10C])
    pick = tok_rng.integers(blocks.shape[0], size=-(-rows // blocks.shape[1]))
    ids = blocks[pick][:, :, layer_col, :].reshape(-1, top_k)[:rows].astype(np.int64)
    if mode in perms:
        prim, rep = perms[mode]
        phys = prim[layer_col][ids]
        if rep is not None:
            # replicas (EPLB redundant experts, emulated in place: a replica shares a cold expert's slot): odd
            # tokens of a replicated expert go to its replica
            alt = rep[layer_col][ids]
            odd = (np.arange(ids.shape[0]) % 2 == 1)[:, None]
            phys = np.where((alt >= 0) & odd, alt, phys)
        ids = phys
    return ids


class _EplbFixture:
    def __init__(self, runner, case_name, mode, seed):
        import torch.distributed as dist

        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_mlp

        reg = sp_mlp.EPLB_FIX
        if reg["flag"] is None:
            raise RuntimeError("EPLB_ROUTE_FIX: no route_stage layer registered a fixture buffer")
        self.reg, self.case, self.mode = reg, case_name, mode
        self.layers = sorted(reg["layers"])
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        par = runner.parallel_config
        self.ep = par.data_parallel_size * par.tensor_parallel_size
        self.saved = []
        if mode == "real0":
            for _, gate in _moe_layers(runner.model):
                b = getattr(gate, "e_score_correction_bias", None)
                if isinstance(b, torch.Tensor):
                    self.saved.append((b, b.detach().clone()))
                    b.data.zero_()
            if not self.saved:
                raise RuntimeError("EPLB_ROUTE_FIX real0: no e_score_correction_bias found")
        if mode in ("bal", "uni", "rep", "perm", "permin", "permr", "permrin", "permact", "permactin"):
            for col, li in enumerate(self.layers):
                fix, _ = reg["layers"][li]
                ids = _eplb_ids(mode, fix.shape[0], reg["E"], fix.shape[1], self.ep, seed, col, col, self.rank)
                fix.copy_(torch.from_numpy(ids).to(fix.device, torch.int32))
            reg["flag"].fill_(True)
        else:
            reg["flag"].fill_(False)
        for _, seen in reg["layers"].values():
            seen.fill_(-1)
        torch.cuda.synchronize()

    def check_applied(self) -> None:
        import torch.distributed as dist

        E = self.reg["E"]
        counts = torch.zeros(len(self.layers), E + 1, dtype=torch.int64, device=self.reg["flag"].device)
        for col, li in enumerate(self.layers):
            seen = self.reg["layers"][li][1].to(torch.int64)
            seen = torch.where(seen < 0, E, seen)
            counts[col] += torch.bincount(seen.flatten(), minlength=E + 1)[: E + 1]
        dead = int((counts[:, :E].sum(1) == 0).sum())
        if dead:  # a layer whose route_stage never ran this case: the fixture did not reach it
            raise RuntimeError(f"EPLB_ROUTE_FIX {self.case}: {dead} layers saw no routes on rank {self.rank}")
        dist.all_reduce(counts)
        if self.rank != 0:
            return
        c = counts[:, :E].double().cpu().numpy()
        local = E // self.ep
        rk = c.reshape(len(self.layers), self.ep, local).sum(2)
        mm = rk.max(1) / np.maximum(rk.mean(1), 1e-9)
        per_e = c.sum(1) / E
        hot = c.max(1) / np.maximum(per_e, 1e-9)
        act = (c > 0).mean(1)
        tok = c.sum(1)
        q = lambda a: f"mean {a.mean():.3f} p50 {np.median(a):.3f} p90 {np.quantile(a, 0.9):.3f} max {a.max():.3f}"
        logger.info(
            "EPLBFIX %s mode %s pairs/layer %.0f | EP%d rank max/mean %s | hottest expert/mean %s | "
            "active experts %s | busiest-rank pairs mean %.1f",
            self.case, self.mode, tok.mean(), self.ep, q(mm), q(hot), q(act), rk.max(1).mean(),
        )
        # active experts per EP rank (weight bytes streamed) and the largest per-expert token count
        act_rk = (c > 0).reshape(len(self.layers), self.ep, local).sum(2)
        print(f"EPLBFIX {self.case} mode {self.mode} pairs/layer {tok.mean():.0f} EP{self.ep} rank_mm {q(mm)} "
              f"hot {q(hot)} active {q(act)} busiest_rank_pairs {rk.max(1).mean():.1f} "
              f"active_per_rank mean {act_rk.mean():.1f} max {act_rk.max(1).mean():.1f} "
              f"max_tokens_per_expert {c.max(1).mean():.1f}", flush=True)
        import os

        out = os.environ.get("EPLB_FIX_DUMP")
        if out:  # per-layer [EP rank, local expert] counts of the routes MegaMoE read, for the per-layer cost fit
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, f"{self.case}.counts.npy"), c.reshape(len(self.layers), self.ep, local))

    def remove(self) -> None:
        self.reg["flag"].fill_(False)
        for b, s in self.saved:
            b.data.copy_(s)


def install_routing(
    runner,
    requested: str,
    dummy_weights: bool,
    num_tokens: int,
    seed: int,
    case_name: str | None = None,
) -> tuple[str, RoutingFixture | None]:
    """Resolve `--routing` and install a synthetic fixture if selected.

    `auto` is Zipf with dummy weights and real otherwise; it falls back to
    real routing when the model doesn't support the fixture.
    """
    import os

    if os.environ.get("EPLB_ROUTE_FIX") == "1":
        parts = (case_name or "").split("~")
        if len(parts) != 3 or not parts[2].startswith("s"):
            raise ValueError(f"EPLB_ROUTE_FIX: case {case_name!r} is not <shape>~<mode>~s<seed>")
        mode = parts[1]
        if mode not in ("real", "real0", "bal", "uni", "rep", "perm", "permin", "permr", "permrin", "permact", "permactin"):
            raise ValueError(f"EPLB_ROUTE_FIX: unknown mode {mode!r}")
        return f"eplb-{mode}", _EplbFixture(runner, case_name, mode, int(parts[2][1:]))
    mode = requested
    if mode == "auto":
        mode = "zipf" if dummy_weights else "real"
    if mode == "real":
        return mode, None
    parallel = runner.parallel_config
    ep_size = (
        parallel.data_parallel_size
        * parallel.tensor_parallel_size
        * parallel.prefill_context_parallel_size
        if parallel.enable_expert_parallel
        else 1
    )
    # Without EP, MoE gathers PCP rows, including replicated decode queries.
    if not parallel.enable_expert_parallel:
        num_tokens *= parallel.prefill_context_parallel_size
    try:
        if mode == "zipf" and parallel.prefill_context_parallel_size > 1:
            raise NotImplementedError("Zipf routing with prefill context parallelism.")
        routing = RoutingFixture(
            runner.model,
            num_tokens,
            runner.dp_rank,
            runner.dp_size,
            ep_size,
            seed,
            runner.device,
            mode,
        )
    except (ValueError, NotImplementedError) as e:
        if requested != "auto":
            raise
        logger.warning("Using real routing: %s", e)
        return "real", None
    return mode, routing
