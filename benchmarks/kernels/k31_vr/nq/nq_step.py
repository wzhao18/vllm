"""Whole Kimi K3.1 decode step (target verify + DSpark drafter) under `vllm bench forward`, for
the nq stock-vs-specialized A/B: 16 ranks under srun (--init-file rendezvous), one process
per arm. A wrapper only: vllm/benchmarks is untouched.

Derived from $W/c1/dsacore/tools/bench_b1_step.py (= b1/r2x/tools/bench_b1_step.py, the
merge-gate / oracle harness, with the 2026-11-01 MX-aware indexer fill), minus its arm and
fault hooks, plus:

* FULL decode graphs where the specialized decode reads them. bench forward sets
  cudagraph_mode NONE and the config post-init then clears the capture sizes
  (vllm/benchmarks/forward/main.py _make_engine_args; config/vllm.py _set_cudagraph_sizes),
  so the specialized decode would refuse the deployment (runtime.unsupported: no FULL
  decode graphs) and the drafter would see no batches (draft.draft_batches() == (): stock
  drafter, ~0.9 ms at B1). Instead of re-deriving the captured rows (bench_b1_step used
  ``captured_rows(sizes, q, 1 << 20)`` and ``s // q`` drafter batches, neither the
  production rounding), the wrapper calls the real ``specialized_decode_config`` and
  ``draft_batches`` with the production compilation fields (FULL_AND_PIECEWISE, the
  config's capture sizes and max capture size) set on the config for the duration of the
  call: limit = min(max_num_seqs * q, max capture) and round_up(s, K) / K come from the
  code under test, whatever it becomes. The stock arm (decode_backend auto) is untouched.
* --overlay DIR: the code under test. DIR is a whole vllm tree (stage_overlay.sh: the base
  env vllm-jh-242fb49cd4, hard-linked, with every file of the checkout that differs from
  242fb49cd4 under vllm/ shipped over it, listed in DIR/SOURCE_REV); nq_inner.sh puts DIR
  first on PYTHONPATH in place of the base env. The wrapper checks that vllm and kimi_k3
  resolve inside DIR and rank 0 logs the revision and the shipped files.
* Random indexer keys after each case's KV zeroing (on by default; --no-fill-index): the
  bench never prefills, and an all-zero indexer cache makes the DCP top-k one 100K-token tie
  (not what serving sees). Scale word = 1.0 in the reader's format: packed UE8M0 0x7F7F7F7F
  where the MQA logits read MX scales (SM10x incl. SM107), else FP32 1.0. The MXFP4 keys of
  the nq configs (68 B/token: 64 B of e2m1 pairs + 4 UE8M0 scales) take random bytes as
  e2m1 pairs, as the C256 recipe did (merge/tools/t10/mt_n4_m10.sh). The MLA cache stays
  zero unless --fill-mla (the sparse core's cost does not depend on its values).
* Bench KV pool sized to the case set (patch_kv_pool), not to gpu-memory-utilization: the
  bench's per-case state snapshot (~158 MiB per request at q4) and graph/eager transients
  need the memory the server's KV would take; both arms get the same pool. Cases the server
  would not admit (more blocks than its KV capacity) are refused. Per case, rank 0 logs the
  context set, its KV blocks vs pool and production capacity, the snapshot size and peak memory.

What the median is: one CUDA graph per case of the target forward plus the drafter's K-step
propose with every draft accepted, rank-max per replay, median of iters x samples
(capture.py). Not included in either arm: target logits / LM head, rejection or sampled
verify (incl. specialized runtime/sampled_verify.py) and CPU step overhead. The 'graph vs
eager rel rms' of the ls arm compares the specialized graph with the stock eager pass and
saturates at ~1.4 with dummy weights over 93 layers: it is not a correctness signal.
"""
import argparse
import os
import socket
import sys
import time

RANK0 = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")) == "0"


def log(*a):
    if RANK0:
        print("NQ:", *a, flush=True)


def parse_sizes(spec):
    sizes = set()
    for item in spec.split(","):
        if item.strip():
            lo, _, hi = item.partition("-")
            sizes.update(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(sizes))


def rendezvous(init_file):
    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    rank = int(os.environ["SLURM_PROCID"])
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=os.environ["SLURM_NTASKS"],
        LOCAL_RANK=os.environ["SLURM_LOCALID"],
    )
    if rank == 0:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        address = f"{socket.gethostbyname(socket.gethostname())}:{port}"
        with open(init_file + ".tmp", "w") as f:
            f.write(address)
        os.rename(init_file + ".tmp", init_file)
    else:
        while not os.path.exists(init_file):
            time.sleep(0.2)
        with open(init_file) as f:
            address = f.read()
    host, port = address.split(":")
    os.environ.update(MASTER_ADDR=host, MASTER_PORT=port)


def apply_overlay(root):
    """Check that vLLM is imported from the overlay tree ``root`` (stage_overlay.sh: the
    base env vllm-jh-242fb49cd4 with every file of the checkout that differs from
    242fb49cd4 under vllm/ shipped over it; nq_inner.sh puts ``root`` first on
    PYTHONPATH in place of the base env) and log what it carries."""
    if not root:
        log("overlay: none (base env vllm-jh-242fb49cd4)")
        return
    if not os.path.isfile(os.path.join(root, "vllm", "__init__.py")):
        raise RuntimeError(
            f"overlay {root} is not a whole vllm tree (no vllm/__init__.py): restage it "
            "with stage_overlay.sh"
        )
    import vllm
    import vllm.models.kimi_k3 as k3

    real = os.path.realpath(root) + os.sep
    for mod in (vllm, k3):
        if not os.path.realpath(mod.__file__).startswith(real):
            raise RuntimeError(f"overlay not applied: {mod.__name__} from {mod.__file__}")
    rev, shipped = "", []
    p = os.path.join(root, "SOURCE_REV")
    if os.path.isfile(p):
        with open(p) as f:
            lines = f.read().splitlines()
        rev = lines[0] if lines else ""
        shipped = [x.split(None, 1)[1] for x in lines if x.startswith(("M ", "A ", "D "))]
    log(f"overlay: vllm from {os.path.dirname(vllm.__file__)} rev {rev or '?'}")
    log(f"overlay: {len(shipped)} files differ from the base under vllm/: {shipped}")


def patch_specialized_decode(sizes, max_capture):
    """Run the real specialized_decode_config / draft_batches with the production
    compilation fields (see the module docstring)."""
    import vllm.models.kimi_k3.nvidia.model as model
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
    from vllm.config import CUDAGraphMode
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    def specialized(vllm_config):
        extra = vllm_config.additional_config
        return isinstance(extra, dict) and extra.get("decode_backend", "auto") != "auto"

    def production(fn):
        def call(vllm_config, *args):
            if not specialized(vllm_config):
                return fn(vllm_config, *args)
            c = vllm_config.compilation_config
            saved = c.cudagraph_mode, c.cudagraph_capture_sizes, c.max_cudagraph_capture_size
            c.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
            c.cudagraph_capture_sizes = list(sizes)
            c.max_cudagraph_capture_size = max_capture
            try:
                return fn(vllm_config, *args)
            finally:
                (c.cudagraph_mode, c.cudagraph_capture_sizes,
                 c.max_cudagraph_capture_size) = saved  # fmt: skip

        return call

    build = production(runtime.specialized_decode_config)

    def specialized_decode_config(vllm_config):
        config = build(vllm_config)
        if config is not None:
            rows = config.captured
            log(
                f"specialized decode: q {config.q} dcp {config.dcp} bf16_state "
                f"{config.bf16_state} captured {len(rows)} rows "
                f"{rows[:1]}..{rows[-1:]} (capture sizes {len(sizes)}, max {max_capture})"
            )
        return config

    runtime.specialized_decode_config = specialized_decode_config
    model.specialized_decode_config = specialized_decode_config
    batches = production(draft.draft_batches)

    def draft_batches(vllm_config, steps):
        out = batches(vllm_config, steps)
        if specialized(vllm_config):
            log(f"drafter fast-path candidate batches (draft_batches, K={steps}): {list(out)}")
        return out

    draft.draft_batches = draft_batches


KERNEL_PAGE = 64


def patch_fill(fill_index, fill_mla):
    """$W/c1/dsacore/tools/bench_b1_step.py patch_fill_index (2026-11-01 fix): random index
    keys per 64-key kernel page with an MX-aware 1.0 scale word; optional MLA rows."""
    if not fill_index:
        log("indexer fill: off (all-zero keys: one-tie top-k)")
        return
    import torch

    from vllm.benchmarks.forward import kv_init

    zero = kv_init.zero_kv_blocks

    @torch.inference_mode()
    def zero_kv_blocks(kv_caches, attn_groups, batch, num_blocks):
        zero(kv_caches, attn_groups, batch, num_blocks)
        from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

        mx = mqa_logits_use_mx_scales()
        word = 0x7F7F7F7F if mx else 0x3F800000
        rank = int(os.environ.get("RANK", "0"))
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + rank)
        counts = {}
        for group in attn_groups:
            names = group.layer_names
            if not names:
                continue
            indexer = all("indexer" in n for n in names)
            if not indexer and (
                not fill_mla
                or any("indexer" in n for n in names)
                or isinstance(group.kv_cache_spec, kv_init.MambaSpec)
            ):
                continue
            ranges = batch.group_block_ranges[group.kv_cache_group_id]
            for tensor in kv_init._group_tensors(kv_caches, group, num_blocks):
                if tensor.dim() < 3:
                    continue
                dim, factor = kv_init._block_dim(tensor, num_blocks)
                assert dim == 0, tensor.shape
                for begin, end in ranges:
                    pages = tensor.narrow(0, begin * factor, (end - begin) * factor)
                    tokens = pages.shape[1]
                    flat = pages.view(torch.uint8).view(pages.shape[0], -1)
                    per_tok = flat.shape[1] // tokens
                    if indexer:
                        if tokens > KERNEL_PAGE:
                            assert tokens % KERNEL_PAGE == 0, tokens
                            flat = flat.unflatten(1, (tokens // KERNEL_PAGE, -1))
                            tokens = KERNEL_PAGE
                        dim_k = per_tok - 4
                        vals = torch.randn(
                            (*flat.shape[:-1], tokens * dim_k), device=flat.device, generator=gen
                        )
                        flat[..., : tokens * dim_k] = (
                            (vals * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
                        )
                        flat[..., tokens * dim_k :].view(torch.int32).fill_(word)
                        key = f"indexer:{per_tok}B/token+{word:#x}"
                    elif per_tok == 576:  # per-tensor FP8 latent rows (k_scale 1.0)
                        rec = flat.view(flat.shape[0], tokens, per_tok)
                        step = max(1, (1 << 22) // (tokens * per_tok))
                        for r0 in range(0, rec.shape[0], step):
                            r = rec[r0 : r0 + step]
                            v = torch.randn(r.shape, device=r.device, generator=gen) * 0.5
                            r.copy_(v.to(torch.float8_e4m3fn).view(torch.uint8))
                        key = "mla:fp8"
                    else:
                        key = f"skipped({per_tok}B/token)"
                    counts[key] = counts.get(key, 0) + pages.shape[0]
        if rank == 0:
            print(f"NQ: filled KV pages {counts} (mx_scales={mx})", flush=True)

    # NQ-TUNE: whether random FP8 bytes read as e2m1 pairs (MXFP4 keys) give the serving top-k cost: compare the
    # deep_select / DCP-exchange kernel times of a profiled case with an e2e decode iteration profile at the same B.
    kv_init.zero_kv_blocks = zero_kv_blocks
    log(f"indexer fill: random keys per 64-key page{' + MLA rows' if fill_mla else ''}")


GIB = 1 << 30
POOL = {}  # patch_kv_pool -> report_cases: production capacity, bench pool, per-case KV blocks


def case_blocks(case, groups, dcp, lookahead):
    """KV blocks build_batch allocates for ``case`` (its ``cursor``: null block + every group's
    live blocks per request), from the same ``_num_blocks`` it calls."""
    from vllm.benchmarks.forward.batch import _num_blocks

    total = 1
    for r in case.requests:
        for g in groups:
            total += _num_blocks(g.kv_cache_spec, r.prefix_len, r.kv_len + lookahead, dcp)[1]
    return total


def ctx_stats(case):
    """Per-request context (prefix = kv_len - query_len) summary of one case."""
    lens = sorted(r.prefix_len for r in case.requests)
    n = len(lens)
    return (f"B {n} ctx sum {sum(lens)} min {lens[0]} p50 {lens[n // 2]} mean {sum(lens) // n} "
            f"max {lens[-1]}")


def patch_kv_pool(shapes, include_drafter, floor_gib):
    """Bench KV pool = the blocks the case set needs, identical in both arms; production KV
    capacity is checked, not used as the pool.

    Why (n4-nq-f5b39b1 c256 stock, 2026-11-06): the bench sizes KV like the server
    (gpu-memory-utilization 0.9, 100.2 GiB = 4.9M tokens/rank) but profiles with cudagraph
    NONE, and every case then clones every KV / state block its step writes
    (ForwardBench._capture_and_time snapshots = state_reset_bytes, ~158 MiB per request at q4:
    20 GiB at B=128, 30 GiB at B=192, 40 GiB at B=256, independent of context length) and
    captures a graph next to the eager pass. That bench-only memory does not fit in the 10%
    the server leaves: M512 OOM'd on the 1 GiB [512, max_model_len] indexer logits, M768
    in the snapshot clone itself. The cases' KV (agentx B=128: 24.3M of 79M tokens) was not
    the problem; the pool only needs to hold the largest case, so the rest of the memory
    goes to the bench's snapshot / graph / eager transients.

    The pool is sized from the case set and the KV config (block counts are deterministic
    and do not depend on the arm), so both arms get the same pool. A case that needs more
    blocks than the server would have at the configured gpu-memory-utilization (``N0``,
    an upper bound: no CUDA graph memory reserved) is refused at startup: the scheduler
    would never run it. Scale its contexts with nq_shapes.py --max-kv-tokens."""
    import torch

    import vllm.v1.engine.core as core
    from vllm.benchmarks.forward.shapes import BenchCase, BenchRequest, load_cases

    cases = load_cases(shapes)
    get_configs = core.get_kv_cache_configs

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available, *args, **kwargs):
        configs = get_configs(vllm_config, kv_cache_specs, available, *args, **kwargs)
        n0 = min(c.num_blocks for c in configs)
        groups = configs[0].kv_cache_groups
        dcp = vllm_config.parallel_config.decode_context_parallel_size
        lookahead = (
            vllm_config.num_speculative_tokens + 1
            if include_drafter and vllm_config.speculative_config is not None
            else 0
        )
        need = {c.name: case_blocks(c, groups, dcp, lookahead) for c in cases}
        max_len = vllm_config.model_config.max_model_len
        one = BenchCase("max_model_len", (BenchRequest(1, max_len - lookahead),))
        floor = case_blocks(one, groups, dcp, lookahead)
        log(f"KV: production capacity at gpu-memory-utilization "
            f"{vllm_config.cache_config.gpu_memory_utilization}: {n0} blocks "
            f"({available[0] / GIB:.2f} GiB/rank; upper bound, no CUDA graph memory reserved)")
        over = {k: v for k, v in need.items() if v > n0}
        if over:
            raise RuntimeError(
                f"cases need more KV blocks than the server holds ({n0}): {over}. The scheduler "
                "would not admit them; scale their contexts (nq_shapes.py --max-kv-tokens, "
                "NQ_MAX_KV_TOKENS in nq_run1.sh)"
            )
        target = max(max(need.values()), floor) + 8
        per_block = available[0] / n0
        size = int(per_block * target) + 1
        for _ in range(4):
            before = vllm_config.model_config.max_model_len
            configs = get_configs(vllm_config, kv_cache_specs, [size] * len(available), *args,
                                  **kwargs)
            assert vllm_config.model_config.max_model_len == before == max_len, (
                "bench KV pool changed max_model_len", before, max_len)
            got = min(c.num_blocks for c in configs)
            if got >= target:
                break
            size = int(size * target / got) + int(per_block) + 1
        else:
            raise RuntimeError(f"bench KV pool: {got} blocks < {target} needed")
        free = torch.tensor([torch.cuda.mem_get_info()[0]], dtype=torch.int64)
        from vllm.distributed.parallel_state import get_world_group

        torch.distributed.all_reduce(free, op=torch.distributed.ReduceOp.MIN,
                                     group=get_world_group().cpu_group)
        free = int(free.item())
        if free - size < floor_gib * GIB:
            raise RuntimeError(
                f"bench KV pool {size / GIB:.2f} GiB leaves {(free - size) / GIB:.2f} GiB "
                f"(< {floor_gib} GiB) for the snapshot / graph / eager transients"
            )
        POOL.update(n0=n0, pool=got, need=need)
        log(f"KV: bench pool {got} blocks ({size / GIB:.2f} GiB/rank, largest case "
            f"{max(need.values())} blocks + 8, identical in both arms); "
            f"{(free - size) / GIB:.2f} GiB/rank left for the bench's snapshot, graph and "
            f"eager transients (min over ranks)")
        return configs

    core.get_kv_cache_configs = get_kv_cache_configs


def report_cases():
    """Per case, rank 0: the context set (identical in both arms: same shapes file), its KV
    blocks vs the bench pool and the production capacity, then what the step used."""
    import torch

    import vllm.benchmarks.forward.kv_init as kv_init
    import vllm.benchmarks.forward.main as fmain

    run = fmain.run_case
    touched = kv_init.touched_kv_views

    def touched_kv_views(*args, **kwargs):
        views = touched(*args, **kwargs)
        snap = sum(v.numel() * v.element_size() for v in views)
        log(f"  state snapshot {snap / GIB:.2f} GiB (bench-only clone), "
            f"free {torch.cuda.mem_get_info()[0] / GIB:.2f} GiB before it")
        return views

    def run_case(runner, case, args):
        need, n0, pool = POOL.get("need", {}).get(case.name), POOL.get("n0"), POOL.get("pool")
        kv = (f"KV {need} blocks = {need / n0:.0%} of production capacity {n0}, pool {pool}"
              if need else "KV blocks: ?")
        log(f"case {case.name}: {ctx_stats(case)} | {kv}")
        result = run(runner, case, args)
        check = "" if need in (None, result["kv_blocks"]) else f" MISMATCH(predicted {need})"
        log(f"  done {case.name}: kv_blocks {result['kv_blocks']}{check} state_reset "
            f"{result.get('state_reset_bytes', 0) / GIB:.2f} GiB, peak allocated "
            f"{torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()
        return result

    kv_init.touched_kv_views = touched_kv_views
    fmain.run_case = run_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-file", default="")
    ap.add_argument("--capture-sizes", required=True, help="production list (+ measured rows)")
    ap.add_argument("--max-capture", type=int, required=True)
    ap.add_argument("--overlay", default="")
    ap.add_argument("--no-fill-index", action="store_true")
    ap.add_argument("--fill-mla", action="store_true")
    ap.add_argument("--free-floor-gib", type=float, default=16.0,
                    help="memory the bench KV pool must leave for snapshot / graph / transients")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    rendezvous(args.init_file)
    apply_overlay(args.overlay)
    sizes = parse_sizes(args.capture_sizes)
    patch_specialized_decode(sizes, args.max_capture)
    patch_fill(not args.no_fill_index, args.fill_mla)
    if "--shapes" in rest:
        patch_kv_pool(rest[rest.index("--shapes") + 1], "--exclude-drafter" not in rest,
                      args.free_floor_gib)
    else:
        log("KV: no --shapes: bench pool = production gpu-memory-utilization sizing")
    report_cases()
    import torch

    import vllm

    log(f"vllm {vllm.__version__} ({os.path.dirname(vllm.__file__)}) torch {torch.__version__}")
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.benchmarks.forward.main import main as forward_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    forward_main(parser.parse_args(rest))


if __name__ == "__main__":
    main()
