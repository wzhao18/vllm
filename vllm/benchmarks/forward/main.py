# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the model forward of exact batches as one-shot CUDA graphs.

Each case is a list of requests `(query_len, kv_len, num_draft_tokens)`
per DP rank. The bench builds the batch without the scheduler, runs it through
the real `execute_model` (input prep, attention metadata, DP sync), and times
the target forward and configured drafter captured as a CUDA graph. Target
sampling, rejection sampling and CPU step overhead aren't measured.

Single GPU: `vllm bench forward --model M --case decode=32*1@8192`.
Multi-GPU: `torchrun --nproc-per-node=N -m vllm.entrypoints.cli.main bench
forward ...`, one process per GPU.
"""

import argparse
import json
import math
import os
import socket
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from vllm.benchmarks.forward.shapes import (
    BenchCase,
    jitter_kv_lens,
    load_cases,
    parse_case,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


def add_cli_args(parser: FlexibleArgumentParser) -> None:
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="`NAME=REQS`, where REQS is a comma list of `[N*]Q[@K][c][:D]` "
        "(query length Q, KV length K including the query (default Q), D draft "
        "tokens, N requests). A prefill's prefix is a prefix-cache hit unless c "
        "marks a chunked prefill. E.g. `decode=32*1@8192`, "
        "`verify=16*4@4096:3`, `prefill=8192`, `hit=2048@10240`, "
        "`mix=8192,32*1@8192`. Repeatable.",
    )
    parser.add_argument("--shapes", help="JSON or YAML file with cases.")
    parser.add_argument(
        "--kv-len-jitter",
        type=float,
        default=0.5,
        help="Draw each request's prefix (KV length minus query length) "
        "uniformly within +-this fraction of its value, like the varied lengths "
        "of a real batch, capped at the max model length. Seeded by --seed only, "
        "so every DP rank gets the same lengths. 0 keeps the exact lengths.",
    )
    parser.add_argument(
        "--kv-init",
        choices=["zeros"],
        default="zeros",
        help="Initial KV cache contents of each case's blocks.",
    )
    parser.add_argument(
        "--routing",
        choices=["auto", "real", "balanced", "zipf"],
        default="auto",
        help="MoE routing. balanced: fixed routing with equal load per EP rank. "
        "zipf: fixed weighted sampling without replacement (alpha=1.25), "
        "with expert popularity shuffled per layer. "
        "auto: zipf with dummy weights, else real; falls back to real when "
        "the synthetic fixture is unsupported.",
    )
    parser.add_argument(
        "--exclude-drafter",
        action="store_true",
        help="Time only the target model. By default, a configured MTP, EAGLE, "
        "DSpark, or DFlash drafter is included, assuming all draft tokens "
        "are accepted. Target sampling and rejection sampling are excluded.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Replays per sample.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-json", help="Write results to this file.")
    parser.add_argument(
        "--profile-dir",
        help="After timing, trace a few replays of each case and write "
        "`<case>.rank<r>.trace.json.gz` (Perfetto) and `<case>.rank<r>.kernels.json` "
        "(GPU time per kernel per forward) here. Summarize or compare with "
        "`python -m vllm.benchmarks.forward.profile A.kernels.json [B.kernels.json]`.",
    )
    parser.add_argument(
        "--eplb-load",
        help="With --enable-eplb: a `.npy` of logical expert loads, shape (MoE "
        "layers, routed experts), e.g. token counts of real traffic. Before the "
        "cases, EPLB rebalances once from this load (the placement serving's "
        "dynamic EPLB computes from its load window), then stays fixed. Needs "
        "synchronous EPLB (`--eplb-config '{\"use_async\": false}'`).",
    )
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(load_format="dummy", enable_prefix_caching=False)


def _setup_env() -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Each case's eager pass compiles exactly the shapes it needs.
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    if "RANK" in os.environ:
        return  # Launched by torchrun.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )


def _make_engine_args(args: argparse.Namespace, cases: list[BenchCase]):
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.distributed_executor_backend = "external_launcher"
    if engine_args.pipeline_parallel_size > 1:
        from vllm.benchmarks.forward.pipeline import ForwardPipelineExecutor

        engine_args.distributed_executor_backend = ForwardPipelineExecutor
    engine_args.enable_prefix_caching = False
    engine_args.compilation_config.mode = CompilationMode.NONE
    engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if engine_args.max_num_seqs is None:
        engine_args.max_num_seqs = max(len(c.requests) for c in cases)
    if engine_args.max_num_batched_tokens is None:
        # Prefix hits may replay up to a window more per request.
        engine_args.max_num_batched_tokens = max(
            8192,
            *(
                c.num_tokens + 1024 * sum(r.prefix_hit for r in c.requests)
                for c in cases
            ),
        )
    return engine_args


def _row_sums(model_state) -> torch.Tensor:
    """(layers, local experts, tensors) int64 sums of each expert row's 32-bit words
    of every EPLB-moved tensor (bench-only weight-movement check)."""
    out = []
    for layer in model_state.model.expert_weights:
        cols = []
        for t in layer:
            u = t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)
            assert u.shape[1] % 4 == 0, u.shape
            cols.append(u.view(torch.int32).sum(dim=1, dtype=torch.int64))
        out.append(torch.stack(cols, dim=1))
    return torch.stack(out)


def _gather_rows(local: torch.Tensor, ep) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(ep.world_size)]
    torch.distributed.all_gather(parts, local, group=ep.device_group)
    return torch.cat(parts, dim=1)  # (layers, physical, tensors)


def apply_eplb_load(runner, path: str) -> dict[str, Any]:
    """Rebalance the experts once from the logical loads in ``path``.

    The load goes into EP rank 0's load window at each logical expert's primary
    physical slot (the initial EPLB map puts logical ``e`` in slot ``e``); the
    other ranks contribute zeros to the rearrangement's all-reduce. Returns the
    busiest-rank over mean-rank load, per layer, before and after.
    """
    from vllm.distributed.parallel_state import get_ep_group

    state = runner.eplb.state
    if state is None:
        raise ValueError("--eplb-load needs --enable-eplb.")
    if state.is_async:
        raise ValueError(
            "--eplb-load rebalances synchronously: pass "
            '--eplb-config \'{"use_async": false}\'.'
        )
    load = torch.from_numpy(np.load(path)).to(torch.int64)
    ep = get_ep_group()
    stats: dict[str, Any] = {}
    matched = 0
    for key, model_state in state.model_states.items():
        model = model_state.model
        window = model_state.expert_load_window
        window.zero_()
        shape = (model.num_moe_layers, model.num_logical_experts)
        if tuple(load.shape) != shape:
            continue
        matched += 1
        if not torch.equal(
            model_state.physical_to_logical_map[:, : shape[1]].cpu().long(),
            torch.arange(shape[1]).expand(shape[0], -1),
        ):
            raise ValueError("--eplb-load expects the initial EPLB placement.")
        if ep.rank_in_group == 0:
            window[0, :, : shape[1]] = load.to(window)
        stats[key] = {"before": _rank_balance(load, model_state, ep.world_size)}
    if not matched:
        raise ValueError(f"--eplb-load {path}: shape {tuple(load.shape)} matches no MoE model.")
    import time as _time

    before = {
        k: m.physical_to_logical_map.cpu().clone() for k, m in state.model_states.items()
    }
    sums0 = {
        k: (_gather_rows(_row_sums(m), ep), m.physical_to_logical_map.cpu().clone())
        for k, m in state.model_states.items()
    }
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.experts import KimiK3MegaMoEExperts

    stg = [
        m.scale_staging
        for m in runner.get_model().modules()
        if isinstance(m, KimiK3MegaMoEExperts) and m.scale_staging is not None
    ][:1]
    if stg:
        stats["staging_before"] = {"held": list(stg[0].held), "pending": stg[0].pending}
    free0 = torch.cuda.mem_get_info()[0]
    torch.accelerator.synchronize()
    t0 = _time.perf_counter()
    state.rearrange()
    torch.accelerator.synchronize()
    wall = _time.perf_counter() - t0
    free1 = torch.cuda.mem_get_info()[0]
    for key, model_state in state.model_states.items():
        if key in stats:
            stats[key]["after"] = _rank_balance(load, model_state, ep.world_size)
            model_state.expert_load_window.zero_()
            after = model_state.physical_to_logical_map.cpu()
            changed = before[key] != after
            per_rank = changed.reshape(changed.shape[0], ep.world_size, -1).sum(-1)
            stats[key]["moved_slots"] = {
                "total": int(changed.sum()),
                "per_layer_max": int(changed.sum(-1).max()),
                "per_rank_layer_max": int(per_rank.max()),
                "busiest_rank_total": int(per_rank.sum(0).max()),
            }
    for key, model_state in state.model_states.items():
        before_sums, before_map = sums0[key]
        after_sums = _gather_rows(_row_sums(model_state), ep).cpu()
        after_map = model_state.physical_to_logical_map.cpu().long()
        slot_of = torch.empty_like(before_map.long())
        n = before_map.shape[1]
        layers = torch.arange(before_map.shape[0]).unsqueeze(1)
        slot_of[layers, before_map.long()] = torch.arange(n).expand_as(before_map)
        expected = before_sums.cpu()[layers, slot_of[layers, after_map]]
        bad = (expected != after_sums).any(-1)
        stats.setdefault(key, {})["weight_rows_wrong"] = {
            "rows": int(bad.sum()),
            "layers": int(bad.any(-1).sum()),
            "first": [(int(a), int(b)) for a, b in bad.nonzero()[:8].tolist()],
            "per_tensor_wrong": (expected != after_sums).sum((0, 1)).tolist(),
        }
    if stg:
        stats["staging_after"] = {"held": list(stg[0].held), "pending": stg[0].pending}
    stats["rearrange_wall_s"] = round(wall, 4)
    stats["rearrange_free_mem_delta_mib"] = round((free1 - free0) / 2**20, 1)
    logger.info("EPLB load %s applied: busiest/mean rank load %s", path, stats)
    return stats


def _rank_balance(load: torch.Tensor, model_state, ep_size: int) -> dict[str, float]:
    """Busiest over mean rank load per layer (mean and max over layers) of the
    current placement, a logical expert's load split evenly over its replicas."""
    p2l = model_state.physical_to_logical_map.cpu().long()
    replicas = torch.zeros_like(load).scatter_add_(1, p2l, torch.ones_like(p2l))
    per_slot = torch.gather(load.double() / replicas.clamp_min(1), 1, p2l)
    per_rank = per_slot.reshape(load.shape[0], ep_size, -1).sum(-1)
    ratio = per_rank.max(-1).values / per_rank.mean(-1).clamp_min(1e-9)
    return {"mean": round(ratio.mean().item(), 4), "max": round(ratio.max().item(), 4)}


_SAVE_CASE: list[str] = []
_SAVE_MODEL: list[Any] = []


def _install_save_hidden() -> None:
    """Bench-only (EPLB-ZM output check): with VLLM_BENCH_FWD_SAVE_HIDDEN=<dir>,
    EP rank 0 of each DP group saves the first model output of every case (the
    eager reference) as <dir>/<case>.rank<r>.pt, so arms can be compared."""
    out = os.environ.get("VLLM_BENCH_FWD_SAVE_HIDDEN")
    if not out:
        return
    from vllm.benchmarks.forward import capture
    from vllm.distributed.parallel_state import get_world_group

    os.makedirs(out, exist_ok=True)
    out_dir = out
    first = capture._first_tensor
    seen: set[str] = set()

    def save_first(output):
        t = first(output)
        name = _SAVE_CASE[-1] if _SAVE_CASE else "case"
        rank = get_world_group().rank
        if name not in seen and rank in (0, 4):
            seen.add(name)
            torch.save(t.detach().cpu(), os.path.join(out, f"{name}.rank{rank}.pt"))
        return t

    capture._first_tensor = save_first
    # Per-layer MoE outputs (first sub-call of each case) at a few depths: tells a
    # wrong-expert bug (differs at layer 0) from amplified combine-order noise.
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.experts import KimiK3MegaMoEExperts

    model = _SAVE_MODEL[-1] if _SAVE_MODEL else None
    if model is None:
        return
    layers = [m for m in model.modules() if isinstance(m, KimiK3MegaMoEExperts)]
    picks = sorted({0, 1, 2, len(layers) // 2, len(layers) - 1})
    for i in picks:

        def hook(mod, args, out, i=i):
            name = _SAVE_CASE[-1] if _SAVE_CASE else "case"
            key = f"{name}.moe{i}"
            rank = get_world_group().rank
            if key in seen or rank != 0:
                return
            seen.add(key)
            t = out[0] if isinstance(out, tuple) else out
            torch.save(t.detach().cpu(), os.path.join(out_dir, f"{key}.rank{rank}.pt"))

        layers[i].register_forward_hook(hook)
        if i == 0 and os.environ.get("VLLM_BENCH_PERTURB_MOE0"):
            # Control: +-1 bf16 ulp on a fixed 8.7% of MoE layer 0's output elements
            # (the measured EPLB-vs-no-EPLB layer-0 rounding difference), to measure
            # how far rounding-level noise is amplified downstream.
            frac = float(os.environ["VLLM_BENCH_PERTURB_MOE0"])

            def perturb(mod, args, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.dtype != torch.bfloat16 or t.numel() == 0:
                    return None
                g = torch.Generator(device=t.device).manual_seed(1234)
                u = torch.rand(t.shape, generator=g, device=t.device)
                step = torch.where(u < frac / 2, 1, torch.where(u < frac, -1, 0)).to(torch.int16)
                bits = t.view(torch.int16)
                bits.add_(step * (t != 0).to(torch.int16))  # one ulp up/down, zeros kept
                return None

            layers[i].register_forward_hook(perturb)

        def pre_hook(mod, args, kwargs, i=i):
            # Inputs of the same first sub-call: hidden, topk weights, topk ids.
            name = _SAVE_CASE[-1] if _SAVE_CASE else "case"
            key = f"{name}.moein{i}"
            rank = get_world_group().rank
            if key in seen or rank != 0 or i > 2:
                return
            seen.add(key)
            for tag, t in zip(("x", "w", "ids"), args[:3]):
                torch.save(
                    t.detach().cpu().contiguous(),
                    os.path.join(out_dir, f"{key}.{tag}.rank{rank}.pt"),
                )

        layers[i].register_forward_pre_hook(pre_hook, with_kwargs=True)


def run_case(runner, case: BenchCase, args: argparse.Namespace) -> dict[str, Any]:
    _SAVE_CASE.append(case.name)
    required_drafts = max((r.num_draft_tokens for r in case.requests), default=0)
    if required_drafts > runner.num_speculative_steps:
        raise ValueError(
            f"Case {case.name!r} needs {required_drafts} draft tokens per request, "
            f"but the runner supports {runner.num_speculative_steps}. "
            "Set --speculative-config with a sufficient num_speculative_tokens."
        )

    from vllm.benchmarks.forward.batch import build_batch
    from vllm.benchmarks.forward.capture import ForwardBench
    from vllm.benchmarks.forward.kv_init import touched_kv_views, zero_kv_blocks
    from vllm.benchmarks.forward.pipeline import PipelineBench
    from vllm.benchmarks.forward.routing import install_routing

    context = runner.vllm_config.compilation_config.static_forward_context
    kv_caches = {
        name: getattr(layer, "kv_cache", None) for name, layer in context.items()
    }
    attn_groups = [group for groups in runner.attn_groups for group in groups]
    kv_config = runner.kv_cache_config
    dcp_size = runner.parallel_config.decode_context_parallel_size
    rng = np.random.default_rng([args.seed, runner.dp_rank])
    include_drafter = not args.exclude_drafter
    lookahead = (
        runner.num_speculative_steps + 1
        if include_drafter and runner.vllm_config.speculative_config is not None
        else 0
    )
    if lookahead and any(
        r.kv_len + lookahead > runner.model_config.max_model_len for r in case.requests
    ):
        raise ValueError("Increase --max-model-len to leave room for draft lookahead.")
    batch = build_batch(
        case,
        kv_config.kv_cache_groups,
        kv_config.num_blocks,
        runner.model_config.get_vocab_size(),
        rng,
        dcp_size,
        lookahead,
        kv_config=kv_config,
    )
    zero_kv_blocks(kv_caches, attn_groups, batch, kv_config.num_blocks)
    profile_prefix = None
    if args.profile_dir:
        from vllm.distributed.parallel_state import get_world_group

        os.makedirs(args.profile_dir, exist_ok=True)
        rank = get_world_group().rank
        profile_prefix = os.path.join(args.profile_dir, f"{case.name}.rank{rank}")
    bench_cls = (
        PipelineBench
        if runner.parallel_config.pipeline_parallel_size > 1
        else ForwardBench
    )
    bench = bench_cls(
        runner,
        batch,
        args.iters,
        args.samples,
        touched_kv_views(
            kv_caches,
            attn_groups,
            batch,
            kv_config.num_blocks,
            dcp_size,
            lookahead,
        ),
        case.name,
        profile_prefix,
        include_drafter=include_drafter,
    )
    routing, fixture = install_routing(
        runner, args.routing, args.load_format == "dummy", batch.num_tokens, args.seed
    )
    try:
        result = bench.run()
        if fixture is not None:
            fixture.check_applied()
    finally:
        runner.execute_model(batch.finish())
        if fixture is not None:
            fixture.remove()
    return {
        "num_tokens": batch.num_tokens,
        "kv_blocks": batch.num_blocks,
        "routing": routing,
        **result,
    }


def main(args: argparse.Namespace) -> None:
    cases = [parse_case(c) for c in args.case]
    if args.shapes:
        cases += load_cases(args.shapes)
    if not cases:
        raise ValueError("Pass at least one --case or --shapes.")
    if not 0 <= args.kv_len_jitter < 1:
        raise ValueError("--kv-len-jitter must be in [0, 1).")
    _setup_env()

    from vllm import LLM
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import get_world_group

    llm = LLM.from_engine_args(_make_engine_args(args, cases))
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    # Same draw on every rank: shapes are balanced across DP ranks.
    max_kv_len = runner.model_config.max_model_len
    cases = [
        jitter_kv_lens(c, args.kv_len_jitter, max_kv_len, args.seed) for c in cases
    ]
    world = get_world_group()
    results = []
    with torch.inference_mode(), set_current_vllm_config(runner.vllm_config):
        _SAVE_MODEL.append(runner.get_model())
        _install_save_hidden()
        if args.eplb_load:
            apply_eplb_load(runner, args.eplb_load)
        for case in cases:
            result = run_case(runner, case, args)
            gathered: list[Any] = [None] * world.world_size
            torch.distributed.all_gather_object(gathered, result, group=world.cpu_group)
            if world.rank != 0:
                continue
            rank0 = gathered[0]
            latency = rank0["latency_us"]  # Each replay is the max over ranks.
            results.append(
                {
                    "case": case.name,
                    "requests": [asdict(r) for r in case.requests],
                    "median_us": latency["median"],
                    "ranks": gathered,
                }
            )
            print(
                f"[forward] {case.name}: {len(case.requests)} reqs, "
                f"{rank0['num_tokens']} tokens, median {latency['median']:.1f} us, "
                f"sample spread {latency['sample_spread_pct']:.2f}%, graph vs "
                f"eager rel rms {rank0['graph_vs_eager']['relative_rms']:.1e}",
                flush=True,
            )
            rel_rms = rank0["graph_vs_eager"]["relative_rms"]
            if not math.isfinite(rel_rms):
                # Non-finite outputs change data-dependent kernel timing (routing,
                # top-k): this case's latency is not a valid measurement.
                print(f"[forward] {case.name}: INVALID non-finite output", flush=True)
            if profile := rank0.get("profile"):
                from vllm.benchmarks.forward.profile import format_summary

                print(
                    f"[forward] {case.name} rank 0 kernels ({profile['summary']}):\n"
                    + format_summary({**profile, "kernels": profile["top_kernels"]}),
                    flush=True,
                )
            if latency["sample_spread_pct"] > 0.5:
                logger.warning(
                    "%s: unstable timing (sample medians spread %.2f%%).",
                    case.name,
                    latency["sample_spread_pct"],
                )
    if world.rank == 0 and args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
