#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank benchmark of the standalone Rubin MegaMoE backend
(vllm/models/kimi_k3/nvidia/megamoe_rubin) for the Kimi K3.1 routed MoE on VR200.

Getting the backend importable (overlay)
----------------------------------------
``vllm`` is a regular package, so the kmoe worktree cannot be merged into the env's
vLLM through PYTHONPATH order alone: build an overlay env = the K3.1 VR vLLM env (with
its compiled extensions) plus the kmoe backend directory, then point
``k31vr_env.sh`` at it (it puts ``$W/env/vllm-$K31VR_VLLM_TAG`` first on
PYTHONPATH and reads FlashInfer's version from its ``K31VR_FI`` file):

  W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
  KMOE=<kmoe worktree, on Lustre>
  OV=$W/env/vllm-ov-rubin
  rsync -a --delete $W/env/vllm-jh-242fb49cd4/ $OV/
  rsync -a --delete $KMOE/vllm/models/kimi_k3/nvidia/megamoe_rubin/ \
      $OV/vllm/models/kimi_k3/nvidia/megamoe_rubin/
  # in the container: export K31VR_VLLM_TAG=ov-rubin; source $W/env/k31vr_env.sh

Re-run the second rsync after every backend change. The JSON ``env.backend_dir`` /
``env.backend_git`` fields record which copy was imported.

Launch
------
torchrun (RANK / WORLD_SIZE / LOCAL_RANK) or srun (SLURM_PROCID / SLURM_NTASKS /
SLURM_LOCALID with MASTER_ADDR / MASTER_PORT exported), one process per GPU, all EP
ranks in one NVL72 rack. Device = local rank mod visible devices, so
``--gpus-per-task=1`` works too. On an allocation ``$J`` with hosts ``$HS``:

  K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
  IMG=$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh
  M=/lustre/fsw/coreai_comparch_inferencex/weizha/models
  B=$W/moe4/tools/bench_megamoe_rubin.py
  N=$(echo ${HS//,/ } | wc -w)
  export MASTER_ADDR=$(scontrol show hostnames $HS | head -1) MASTER_PORT=29711
  run() { srun --overlap --jobid $J -N$N -w $HS --ntasks-per-node=4 \
      --container-image=$IMG --no-container-mount-home \
      --container-mounts=$W:$W,$K:$K,$M:$M:ro \
      bash -c "export K31VR_VLLM_TAG=ov-rubin; source $W/env/k31vr_env.sh; \
               python3 $B $*"; }

  # 1 node  = EP4  (HS=h1):          shipped EP4 smoke knobs (2048 tokens)
  run --quant-kind nvfp4 --tokens 2048
  # 2 nodes = EP8  (HS=h1,h2):       tuned EP8 point (capacity 5760 from the record)
  run --quant-kind nvfp4 --tokens 5760 --json $W/moe4/results/rubin_ep8.jsonl
  # 4 nodes = EP16 (HS=h1,h2,h3,h4): tuned EP16 point, real layer-5 experts
  run --quant-kind mxfp4_mxfp8 --tokens 5925 \
      --weights ckpt:$M/kimi-k3.1/rainhollow-test-weights-1024_v1:5

Timing (``timer`` in the JSON)
------------------------------
Every sample: device-synchronized group barrier, a GPU delay kernel
(``torch.cuda._sleep``, ``--delay-us``) so the timed launches are already queued when
the start event fires, start event, the timed work, end event; per-sample time /
replays, then the matched samples are MAX-reduced over ranks and p10/p50/p90 of the
max are reported (FlashInfer ``core/tuning_measurement.py`` + ``shim/autotune.py``).

- ``kernel`` / ``stage_and_kernel``: ``session.launch`` (pre-staged, kernel only) and
  ``moe.forward`` (Triton staging + kernel) captured in a ``torch.cuda.CUDAGraph``
  after warmup, ``--graph-replays`` back-to-back replays per sample. Graph capture is
  also the CUDA-graph-safety check (``graph_vs_eager_max_abs`` compares a replayed
  forward with an eager one; 0 expected).
- ``kernel_eager`` / ``stage_and_kernel_eager``: one eager call per sample behind
  the same delay kernel. ``kernel_eager`` is the FlashInfer tuner's own protocol
  for the shipped knob records (``profiling_policy``: native_events, no CUDA graph,
  warm L2), i.e. the like-for-like comparison with the numbers below.

Compare with (VR, FlashInfer tuner, pre-staged, BF16 combine, SiTU 4.0/25.0):
  EP16 5925 tok/rank: NVFP4 1188 us, MXFP4xMXFP8 1361 us, DeepGEMM 1603 us
  EP8  5760 tok/rank: NVFP4  924 us, MXFP4xMXFP8 1121 us, DeepGEMM 1599 us

Correctness: relative L2 of the output on ``--ref-samples`` sampled tokens against
(a) the quantization-faithful oracle (ported FlashInfer reference from the staged
bytes; expect <= ~2e-3) and (b) a full-precision reference with BF16 activations and
the exact MXFP4 weights (activation and FC1->FC2 requantization error). Staging is
A/B-checked byte for byte against the torch stager (0 expected).
``--sweep-fc1-norm`` (NVFP4) re-runs the point with ``fc1_norm_const`` = 2^0..2^7
and reports rel-L2 vs (b) for each; ``suggested_input_norm_const`` is the
power-of-two constant derived from this point's input amax
(``config.nvfp4_norm_const_from_amax``).

Capacity: ``--max-tokens`` if given, else the ``max_tokens`` of the knob record
tuned at exactly that live token count (e.g. EP8 5760 -> 5760), else the tuner's
rounding (128 below 128 tokens, else a multiple of 256).

Outputs one JSON line per point on rank 0 (and appends them to ``--json`` if given).
"""

import os

# The CuTe DSL captures its target at first import and vLLM imports it early on VR.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _env_int(*names):
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    return None


def init_distributed():
    rank = _env_int("RANK", "SLURM_PROCID")
    world = _env_int("WORLD_SIZE", "SLURM_NTASKS")
    local = _env_int("LOCAL_RANK", "SLURM_LOCALID")
    if rank is None or world is None or local is None:
        raise SystemExit("run under torchrun or srun (rank / world / local rank env).")
    if world > 1 and (
        "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ
    ):
        raise SystemExit("export MASTER_ADDR and MASTER_PORT (srun does not set them).")
    visible = torch.cuda.device_count()
    if visible < 1:
        raise SystemExit("no visible CUDA device.")
    index = local % visible
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    if world > 1:
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, device_id=device, init_method="env://"
        )
    return rank, world, device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quant-kind", choices=("nvfp4", "mxfp4_mxfp8"), default="nvfp4")
    p.add_argument("--variant", choices=("inference", "genphase"), default="inference")
    p.add_argument("--tokens", default="5925", help="live tokens per rank, comma list")
    p.add_argument(
        "--max-tokens",
        default=None,
        help="bucket capacity per point (comma list or one value); default: the "
        "tuned record's max_tokens for that live count, else the tuner's rounding",
    )
    p.add_argument(
        "--knobs",
        default=None,
        help="knob JSON path, inline JSON (v2 file / record / bare knob dict); "
        "default: the backend's shipped knobs/k31_vr.json",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=896,
        help="global experts (EP-size emulation: 224 on EP4 = 56 experts/rank like EP16)",
    )
    p.add_argument(
        "--dedup",
        default=None,
        choices=("standalone_warps", "reuse_dispatch_warps", "wg2", "wg2s"),
        help="run the inference record on the refreshed dedup-combine kernel "
        "(variant inference_dedup) with this token-back transport; wg2: two epilogue "
        "warpgroups (inference_dedup_wg2, reuse_dispatch_warps)",
    )
    p.add_argument("--routing", choices=("random", "balanced"), default="random")
    p.add_argument("--epbal-percall", type=int, default=0,
                   help="int64.3 item 3 per-call A/B of the served EP-balance log: N timed eager calls per point "
                        "with the log on/off in alternating 64-call blocks aligned to the session call ids "
                        "(pre-registered in MEGAMOEFIX.md 12:20); prints EPBAL_PERCALL lines and skips the timing")
    p.add_argument(
        "--active-experts",
        type=int,
        default=None,
        help="random routing over A active experts per rank, split over the two die "
        "halves of the local experts (the step slice's EP-size emulation: 896 experts "
        "at EP4 with A=28 is 224 local slots, 28 active)",
    )
    p.add_argument(
        "--die-local",
        action="store_true",
        help="decode-server placement: die_split schedule + experts homed per die half "
        "(die_local.place_experts), as KimiK3MegaMoEExperts does on decode servers",
    )
    p.add_argument(
        "--shared",
        default=None,
        help="HIDDEN,INTER: fuse a random MXFP8 E4M3 shared expert of that shape "
        "(Kimi K3.1: 7168,6144) into the GenPhase kernel; reports its rel-L2 vs "
        "the stock MXFP8 chain (FP32 GEMMs, SiTU, MXFP8 requant)",
    )
    p.add_argument(
        "--rmsnorm",
        action="store_true",
        help="fuse the consumer RMSNorm (random BF16 weight, Kimi K3.1 latent norm) "
        "into the top-k reduce, as the decode path deploys it",
    )
    p.add_argument(
        "--shared-l2-position",
        choices=("early", "tail"),
        default=None,
        help="fused shared FC2 placement (default: the session's)",
    )
    p.add_argument("--ids-dtype", choices=("int32", "int64"), default="int32")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50, help="timed samples per metric")
    p.add_argument(
        "--graph-replays",
        type=int,
        default=10,
        help="back-to-back CUDA graph replays per timed sample",
    )
    p.add_argument(
        "--delay-us",
        type=float,
        default=1000.0,
        help="GPU delay kernel before each sample's start event (0 disables)",
    )
    p.add_argument("--ref-samples", type=int, default=64, help="0 disables references")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="append result lines to this file")
    p.add_argument(
        "--weights",
        default="random-mxfp4",
        help="random-mxfp4 | random-mxfp4-spread17 | ckpt:<model dir>:<layer>; the "
        "model dir holds model.safetensors.index.json, e.g. "
        "/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1/"
        "rainhollow-test-weights-1024_v1 (MoE layers 1..92)",
    )
    p.add_argument(
        "--input-norm-const",
        type=float,
        default=1.0,
        help="NVFP4 activation staging norm constant (power of two)",
    )
    p.add_argument(
        "--fc1-norm-const",
        type=float,
        default=1.0,
        help="NVFP4 FC1->FC2 requantization norm constant (power of two)",
    )
    p.add_argument(
        "--sweep-fc1-norm",
        action="store_true",
        help="NVFP4: report rel-L2 vs the full-precision reference for "
        "fc1_norm_const = 2^0..2^7 per point (needs --ref-samples > 0)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------------


def random_mxfp4_layer(W, experts, offset, hidden, inter, seed, device, spread17):
    """Canonical MXFP4 local-expert weights, deterministic per GLOBAL expert id."""
    u8 = dict(dtype=torch.uint8, device=device)
    w13 = torch.empty((experts, 2 * inter, hidden // 2), **u8)
    s13 = torch.empty((experts, 2 * inter, hidden // 32), **u8)
    w2 = torch.empty((experts, hidden, inter // 2), **u8)
    s2 = torch.empty((experts, hidden, inter // 32), **u8)
    for e in range(experts):
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1_000_003 + offset + e)
        if spread17:
            # random codes; per-block exponents uniform over exactly 18 values
            # (spread 17: the widest exact NVFP4 conversion, subnormal E4M3 scales)
            legs = ((w13, s13, 2 * inter, hidden), (w2, s2, hidden, inter))
            for codes, scales, rows, cols in legs:
                codes[e] = torch.randint(0, 256, (rows, cols // 2), generator=g, **u8)
                k = torch.randint(
                    -24, -6, (rows, cols // 32), generator=g, device=device
                )
                k[0, 0], k[0, 1] = -7, -24
                scales[e] = (k + 127).to(torch.uint8)
            continue
        a = torch.randn((2 * inter, hidden), generator=g, device=device) * hidden**-0.5
        w13[e], s13[e] = W.quantize_mxfp4_block32(a)
        b = torch.randn((hidden, inter), generator=g, device=device) * inter**-0.5
        w2[e], s2[e] = W.quantize_mxfp4_block32(b)
    return w13, s13, w2, s2


def ckpt_mxfp4_layer(W, path, layer, experts, offset, hidden, inter, device):
    """One layer's local experts from a Kimi K3.1 MXFP4 checkpoint: w1/w3/w2
    ``weight_packed`` (codes) + ``weight_scale`` (UE8M0), keys as
    ``language_model.model.layers.L.block_sparse_moe.experts.N.w1.weight_packed``
    (``weights.parse_expert_key``)."""
    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    want = {}
    for key, fname in index.items():
        parsed = W.parse_expert_key(key)
        if parsed is None:
            continue
        lay, exp, proj, part = parsed
        if lay == layer and offset <= exp < offset + experts:
            slot = (exp - offset, proj, part)
            if slot in want:
                raise SystemExit(f"duplicate checkpoint tensors for {slot}: {key}.")
            want[slot] = (key, fname)
    if len(want) != experts * 6:
        raise SystemExit(
            f"layer {layer}: found {len(want)} of {experts * 6} expert tensors "
            f"(experts [{offset}, {offset + experts})) in {path}."
        )
    tensors = {}
    for fname in sorted({f for _, f in want.values()}):
        with safe_open(os.path.join(path, fname), framework="pt") as f:
            for k, (key, fn) in want.items():
                if fn == fname:
                    t = f.get_tensor(key)
                    if t.element_size() != 1:
                        raise SystemExit(
                            f"{key}: expected 1-byte MXFP4 storage, got {t.dtype}."
                        )
                    tensors[k] = t.view(torch.uint8)

    def stack(name, part, shape):
        out = torch.stack([tensors[(e, name, part)] for e in range(experts)])
        if tuple(out.shape[1:]) != shape:
            raise SystemExit(f"{name}.{part} shape {tuple(out.shape[1:])} != {shape}.")
        return out.to(device)

    w13 = torch.cat(
        (
            stack("w1", "weight", (inter, hidden // 2)),
            stack("w3", "weight", (inter, hidden // 2)),
        ),
        1,
    )
    s13 = torch.cat(
        (
            stack("w1", "weight_scale", (inter, hidden // 32)),
            stack("w3", "weight_scale", (inter, hidden // 32)),
        ),
        1,
    )
    w2 = stack("w2", "weight", (hidden, inter // 2))
    s2 = stack("w2", "weight_scale", (hidden, inter // 32))
    return w13, s13, w2, s2


# ---------------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------------


def make_shared(W, shape, rows, seed, device):
    """Random MXFP8 E4M3 shared expert in the stock linears' storage (row-major FP8,
    F8_128x4 scales) + a capacity MXFP8 input; returns (FusedSharedArgs, ref(n))."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.reference import _situ
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.session import FusedSharedArgs

    hidden, inter = shape
    gen = torch.Generator(device=device).manual_seed(seed)

    def q(t):
        data, sf = W.quantize_mxfp8_block32(t)
        return data, sf

    def dq(data, sf):
        f = data.float().reshape(*data.shape[:-1], -1, 32)
        return (f * W.e8m0_to_f32(sf).unsqueeze(-1)).reshape(data.shape)

    w13 = torch.randn((2 * inter, hidden), generator=gen, device=device) * hidden**-0.5
    w2 = torch.randn((hidden, inter), generator=gen, device=device) * inter**-0.5
    x = torch.randn((max(rows, 128), hidden), generator=gen, device=device)
    (d13, s13), (d2, s2), (dx, sx) = q(w13), q(w2), q(x)
    fc1, fc2 = W.prepare_mxfp8_shared_weights(
        d13, W.to_blocked(s13), d2, W.to_blocked(s2), hidden=hidden, intermediate=inter
    )
    args = FusedSharedArgs(dx, W.to_blocked(sx), fc1, fc2)
    w13f, w2f, xf = dq(d13, s13), dq(d2, s2), dq(dx, sx)

    def ref(n, exact_act=False):
        gu = xf[:n] @ w13f.t()
        act = _situ(gu[:, :inter], gu[:, inter:], 4.0, 25.0).float()
        if not exact_act:  # stock: BF16 SiTU output requantized to MXFP8
            act = dq(*q(act.to(torch.bfloat16)))
        return (act @ w2f.t()).to(torch.bfloat16).float()

    return args, ref


def make_routes(
    n, num_experts, topk, world, rank, kind, seed, device, ids_dtype, active=None
):
    g = torch.Generator(device=device)
    g.manual_seed(seed * 7919 + rank)
    if kind == "random":
        scores = torch.rand((n, num_experts), generator=g, device=device)
        if active is not None:
            local = torch.arange(num_experts, device=device) % (num_experts // world)
            half = num_experts // world // 2
            on = (local < active // 2) | ((local >= half) & (local < half + active // 2))
            scores = scores.masked_fill(~on, -1.0)
        ids = scores.topk(topk, dim=1).indices
    else:
        # balanced: token t's j-th route goes to rank (t + j) % W, distinct local
        # experts per rank; every rank receives n * topk / W routes from every rank.
        e_local = num_experts // world
        t = torch.arange(n, device=device).view(-1, 1) + rank * n
        j = torch.arange(topk, device=device).view(1, -1)
        dst = (t + j) % world
        local = ((t * topk + j) // world) % e_local
        ids = dst * e_local + local
    weights = torch.rand((n, topk), generator=g, device=device) + 0.05
    weights = weights / weights.sum(dim=1, keepdim=True)
    return ids.to(getattr(torch, ids_dtype)).contiguous(), weights.float().contiguous()


def check_routes(ids, num_experts):
    ids64 = ids.to(torch.int64)
    if bool(((ids64 < -1) | (ids64 >= num_experts)).any()):
        raise SystemExit("route ids out of range.")
    s = ids64.sort(dim=1).values
    if bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any()):
        raise SystemExit("duplicate expert id within a token row.")


# ---------------------------------------------------------------------------------
# timing (FlashInfer core/tuning_measurement.py protocol)
# ---------------------------------------------------------------------------------


def sleep_cycles_per_us():
    """Calibrate ``torch.cuda._sleep`` (cycles) against CUDA events on this GPU."""
    cycles = 2_000_000
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda._sleep(cycles)  # first call / clock ramp
    torch.cuda.synchronize()
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles / max(start.elapsed_time(end) * 1000.0, 1e-3)


def _gather_objects(value, group):
    if group is None:
        return [value]
    out = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, value, group=group)
    return out


def capture_graph(fn, group, what):
    """``fn`` captured in a CUDA graph; raises unless capture succeeded on every rank
    (capture launches nothing, so lock step is unaffected)."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.symm_arena import group_barrier

    graph = torch.cuda.CUDAGraph()
    group_barrier(group)
    error = None
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception as exc:  # reported collectively below
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather_objects(error, group)
    if any(errors):
        raise SystemExit(f"CUDA graph capture of {what} failed: {errors}")
    return graph


def measure(run, *, samples, replays, delay_cycles, group):
    """FlashInfer events protocol: per sample barrier, delay kernel, start event,
    ``replays`` back-to-back ``run()``, end event, elapsed / replays; matched samples
    MAX-reduced over ranks; p10/p50/p90 of the max."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.symm_arena import group_barrier

    local = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(samples):
        group_barrier(group)
        if delay_cycles > 0:
            torch.cuda._sleep(delay_cycles)
        start.record()
        for _ in range(replays):
            run()
        end.record()
        end.synchronize()
        local.append(start.elapsed_time(end) * 1000.0 / replays)
    t = torch.tensor(local, dtype=torch.float64, device="cuda")
    if group is not None:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    vals = t.tolist()
    q = statistics.quantiles(vals, n=10) if len(vals) >= 2 else vals * 9
    return {"p10_us": q[0], "p50_us": statistics.median(vals), "p90_us": q[-1]}


def env_identity(pkg_dir):
    ident = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "backend_dir": pkg_dir,
    }
    try:
        import cutlass

        ident["cutlass_dsl"] = str(cutlass.__version__)
    except Exception as exc:  # pragma: no cover - informational
        ident["cutlass_dsl"] = f"unavailable: {exc}"
    try:
        sha = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", pkg_dir, "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
        ).stdout.strip()
        ident["backend_git"] = (sha or "not-a-git-tree") + ("+dirty" if dirty else "")
    except Exception:
        ident["backend_git"] = "unknown"
    return ident


# ---------------------------------------------------------------------------------


_EPBAL_VERDICTS: list = []


def _epbal_percall(args, session, moe, hidden, ids, wts, weights, bucket, out, fused, rms, variant, n,
                   rank, world, group):
    """Per-call A/B of the served EP-balance log (MEGAMOEFIX.md pre-registration, MAIN 12:14). Blocks of 64 calls
    aligned to the session call ids: the ring word is set to the ring ("on", even block) or 0 ("off") at each block
    start, so each on-block logs exactly once (its first call, call id % 64 == 0)."""
    import random
    import statistics as st

    ws = session.kernel._device_workspace
    names = {r.name for r in ws.regions("local")}
    epoch_name = next(r for r in ("nvlink.token_comm_dedup.call_epoch", "rubin.genphase_tail.epoch") if r in names)
    ring_names = [r for r in session._EPBAL_REGIONS if r in names]
    ring_addr = session._ep_balance_ring.address
    lw = session.local_workspace

    def set_word(on: bool) -> None:
        for r in ring_names:
            o = ws.offset(r)
            lw[o : o + 8].view(torch.int64).fill_(ring_addr if on else 0)

    for _ in range(8):  # warm: no timing
        moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out, shared=fused, rmsnorm=rms)
    torch.cuda.synchronize()
    eo = ws.offset(epoch_name)
    e0 = int(lw[eo : eo + 4].view(torch.int32).item())
    N = args.epbal_percall
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
    calls = []
    for k in range(N):
        c = e0 + 1 + k
        if c % 64 == 0:
            set_word((c // 64) % 2 == 0)
        ev[k][0].record()
        moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out, shared=fused, rmsnorm=rms)
        ev[k][1].record()
        calls.append(c)
    torch.cuda.synchronize()
    set_word(True)  # leave the log on, as served
    mine = [ev[k][0].elapsed_time(ev[k][1]) * 1000.0 for k in range(N)]  # us
    allr = [mine]
    if world > 1 and group is not None:
        allr = [None] * world
        dist.all_gather_object(allr, mine, group=group)
    t = [max(r[k] for r in allr) for k in range(N)]
    if rank != 0:
        return
    on = [t[k] for k in range(N) if (calls[k] // 64) % 2 == 0]
    off = [t[k] for k in range(N) if (calls[k] // 64) % 2 == 1]
    first = {calls[k] // 64: t[k] for k in range(N) if calls[k] % 64 == 0}
    d = [first[b] - first[b + 1] for b in sorted(first) if b % 2 == 0 and b + 1 in first]
    rng = random.Random(7)
    boots = sorted(st.median(rng.choices(d, k=len(d))) for _ in range(10000)) if d else [0.0]
    med, lo, hi = (st.median(d) if d else 0.0), boots[249], boots[9749]
    mean_on, mean_off = st.mean(on), st.mean(off)
    rel = 100.0 * (mean_on - mean_off) / mean_off
    se = (st.variance(on) / len(on) + st.variance(off) / len(off)) ** 0.5
    rel_hi = 100.0 * (mean_on - mean_off + 1.96 * se) / mean_off
    amort = 100.0 * med / (64.0 * mean_off)
    p99 = lambda xs: sorted(xs)[int(0.99 * (len(xs) - 1))]  # noqa: E731
    p99_on, p99_off = p99(on), p99(off)
    pass1 = hi <= 2.0
    pass2 = amort <= 0.1 and p99_on <= p99_off * 1.01
    verdict = pass1 and pass2
    _EPBAL_VERDICTS.append(verdict)
    print(f"EPBAL_PERCALL {variant}/{n} pairs={len(d)} fire_median_d={med:+.3f}us CI95=[{lo:+.3f}, {hi:+.3f}] "
          f"(PASS-1 upper<=+2us: {pass1}) | mean on={mean_on:.2f}us off={mean_off:.2f}us rel={rel:+.3f}% "
          f"(upper {rel_hi:+.3f}%) amortised={amort:+.4f}% p99 on={p99_on:.2f} off={p99_off:.2f} "
          f"(PASS-2 amortised<=0.1% and p99<=+1%: {pass2}) -> {'PASS' if verdict else 'FAIL'}", flush=True)


def main():
    args = parse_args()
    rank, world, device = init_distributed()
    group = dist.group.WORLD if world > 1 else None

    import vllm.models.kimi_k3.nvidia.megamoe_rubin as mr
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import reference as R
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.config import (
        SHIPPED_KNOBS,
        RubinMoeProblem,
        bench_capacity,
        make_buckets,
        nvfp4_norm_const_from_amax,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

    problem = RubinMoeProblem(
        quant_kind=args.quant_kind,
        num_experts=args.num_experts,
        input_norm_const=args.input_norm_const,
        fc1_norm_const=args.fc1_norm_const,
    )
    nvfp4 = problem.quant_kind == "nvfp4"
    if args.sweep_fc1_norm and (not nvfp4 or args.ref_samples <= 0):
        raise SystemExit("--sweep-fc1-norm needs nvfp4 and --ref-samples > 0.")
    if args.graph_replays < 1 or args.iters < 1:
        raise SystemExit("--graph-replays and --iters must be >= 1.")
    knob_source = args.knobs or SHIPPED_KNOBS
    tokens = [int(t) for t in args.tokens.split(",") if t]
    if args.max_tokens:
        caps = [int(t) for t in args.max_tokens.split(",")]
        caps = caps * len(tokens) if len(caps) == 1 else caps
        if len(caps) != len(tokens):
            raise SystemExit("--max-tokens needs one value or one per --tokens entry.")
    else:
        caps = [
            bench_capacity(
                knob_source,
                problem=problem,
                variant=args.variant,
                world_size=world,
                live_tokens=t,
            )
            for t in tokens
        ]
    if any(t > c for t, c in zip(tokens, caps)):
        raise SystemExit("a point's live tokens exceed its capacity.")
    points = sorted(
        {(args.variant, c, t) for t, c in zip(tokens, caps)}, key=lambda p: (p[1], p[2])
    )
    # one bucket per capacity: the smallest live point of that capacity picks the knobs
    by_cap = {}
    for v, c, t in points:
        by_cap.setdefault(c, (v, c, t))
    buckets, notes = make_buckets(
        knob_source,
        problem=problem,
        world_size=world,
        points=list(by_cap.values()),
        device_name=torch.cuda.get_device_name(),
    )
    if args.die_local:
        import dataclasses

        from vllm.models.kimi_k3.nvidia.megamoe_rubin import die_local

        mask = die_local.current_die_sm_mask32()
        if mask is None:
            raise SystemExit("--die-local: no trusted SM -> die map on this GPU")
        # as KimiK3MegaMoEExperts on decode servers: one cluster shape, full grid
        buckets = [
            dataclasses.replace(
                b,
                knobs=dataclasses.replace(b.knobs, fallback_cluster_shape_mn=None),
                max_sm_count=None,
                die_sm_mask32=mask,
            )
            for b in buckets
        ]
    if args.dedup:
        import dataclasses

        buckets = [
            dataclasses.replace(
                b,
                variant=(
                    "inference_dedup_wg2"
                    if args.dedup in ("wg2", "wg2s")
                    else "inference_dedup"
                ),
                knobs=dataclasses.replace(
                    b.knobs,
                    token_back_mode=(
                        {"wg2": "reuse_dispatch_warps", "wg2s": "standalone_warps"}.get(
                            args.dedup, args.dedup
                        )
                    ),
                ),
            )
            for b in buckets
        ]
    shared_shape = None
    if args.shared:
        import dataclasses

        shared_shape = tuple(int(v) for v in args.shared.split(","))
        if args.shared_l2_position is not None:
            from vllm.models.kimi_k3.nvidia.megamoe_rubin import session as rubin_session

            rubin_session.RUBIN_FUSED_SHARED_L2_POSITION = args.shared_l2_position
        buckets = [dataclasses.replace(b, shared=shared_shape) for b in buckets]
    bucket_of = {b.max_tokens: b for b in buckets}
    if rank == 0:
        for note in notes:
            print(note, flush=True)

    moe = RubinMegaMoE(problem, group, buckets)
    e_local = moe.experts_per_rank
    offset = rank * e_local
    if args.weights.startswith("ckpt:"):
        _, path, layer = args.weights.split(":", 2)
        canonical = ckpt_mxfp4_layer(
            W, path, int(layer), e_local, offset, problem.hidden, problem.intermediate,
            device,
        )
    else:
        canonical = random_mxfp4_layer(
            W,
            e_local,
            offset,
            problem.hidden,
            problem.intermediate,
            args.seed,
            device,
            spread17=args.weights == "random-mxfp4-spread17",
        )
    try:
        weights = moe.prepare_weights(*canonical, layer_id=0)
    except W.NonRepresentableError as exc:
        print(f"[rank {rank}] {exc}", flush=True)
        raise
    if args.die_local:
        weights = die_local.place_experts(weights)
    if weights.report is not None and rank == 0:
        print(weights.report.summary().splitlines()[0], flush=True)
    fused = None
    if shared_shape is not None:
        fused, shared_ref = make_shared(W, shared_shape, max(caps), args.seed + rank, device)
        if args.die_local:
            fc1 = die_local.place_halves(fused.fc1[0].permute(0, 2, 1)).permute(0, 2, 1)
            fused = fused._replace(fc1=(fc1, fused.fc1[1]))
    moe.warmup(weights, ids_dtypes=(getattr(torch, args.ids_dtype),), shared=fused)
    rms = None
    if args.rmsnorm:
        gen = torch.Generator(device=device).manual_seed(args.seed + 7)
        norm_w = (1 + 0.1 * torch.randn(problem.hidden, generator=gen, device=device)).to(torch.bfloat16)
        rms = (norm_w, 1e-6)
        moe.prepare_rmsnorm(weights, *rms)
    ident = env_identity(os.path.dirname(mr.__file__))
    cycles_per_us = sleep_cycles_per_us()
    delay_cycles = int(args.delay_us * cycles_per_us)
    timer = {
        "protocol": "flashinfer-events: barrier, delay kernel, start event, replays, "
        "end event; per-sample time / replays; MAX over ranks per sample; p10/p50/p90",
        "kernel": "CUDA graph of session.launch (pre-staged), graph replays",
        "stage_and_kernel": "CUDA graph of moe.forward (staging + kernel)",
        "eager": "one eager call per sample (FlashInfer tuner native_events "
        "protocol of the shipped knob records)",
        "graph_replays": args.graph_replays,
        "samples": args.iters,
        "warmup": args.warmup,
        "delay_us": args.delay_us,
        "delay_cycles": delay_cycles,
        "cold_l2": False,
    }

    for variant, cap, n in points:
        bucket = bucket_of[cap]
        session = moe.sessions[bucket]
        gen = torch.Generator(device=device).manual_seed(args.seed * 31 + rank + n)
        hidden = torch.randn(
            (n, problem.hidden), generator=gen, device=device
        ).to(torch.bfloat16)
        ids, wts = make_routes(
            n, problem.num_experts, problem.top_k, world, rank, args.routing,
            args.seed + n, device, args.ids_dtype, args.active_experts,
        )
        check_routes(ids, problem.num_experts)
        out = torch.empty(
            (max(n, 1), problem.hidden), dtype=torch.bfloat16, device=device
        )
        if args.epbal_percall:
            _epbal_percall(args, session, moe, hidden, ids, wts, weights, bucket, out, fused, rms, variant, n,
                           rank, world, group)
            continue
        res = moe.forward(
            hidden, ids, wts, weights, bucket=bucket, out=out, shared=fused, rmsnorm=rms
        )
        torch.cuda.synchronize()
        moe.check_route_errors()
        # Bitwise fingerprint of this rank's first eager output (A/B of ordering-only kernel changes).
        import hashlib

        out_hash = hashlib.sha256(
            out[:n].contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()[:16]
        out_hashes = [out_hash]
        if group is not None:
            out_hashes = [None] * dist.get_world_size(group)
            dist.all_gather_object(out_hashes, out_hash, group=group)

        # staging A/B vs the torch stager (live rows)
        rx, rsf, ridx, rw = R.stage_inputs_torch(
            hidden, ids, wts, quant_kind=problem.quant_kind,
            capacity=session.x.shape[0], sf_cols=session.sf_cols,
            num_experts=problem.num_experts, input_norm_const=problem.input_norm_const,
        )
        stage_mismatch = int(
            (session.x.view(torch.uint8)[:n] != rx[:n]).sum()
            + (session.x_sf.view(torch.uint8)[:n] != rsf[:n]).sum()
            + (session.topk_idx != ridx).sum()
            + (session.topk_weights[:n] != rw[:n]).sum()
        )

        result = {"out_sha256_per_rank": out_hashes}
        amax = torch.tensor(
            [float(hidden.float().abs().max()) if n else 0.0], device=device
        )
        if group is not None:
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
        if nvfp4 and float(amax) > 0:
            result["input_amax"] = float(amax)
            result["suggested_input_norm_const"] = nvfp4_norm_const_from_amax(
                float(amax)
            )
        bf16_ref = None
        if args.ref_samples > 0 and n > 0:
            xs = session.x.view(torch.uint8)[:n]
            sfs = session.x_sf.view(torch.uint8)[:n]
            idx, expected = R.sampled_reference(
                [xs, sfs, session.topk_idx[:n], session.topk_weights[:n]],
                n,
                lambda p: R.megamoe_oracle(
                    p[0], p[1], p[2], p[3], weights, quant_kind=problem.quant_kind,
                    local_expert_offset=offset, hidden=problem.hidden,
                    intermediate=problem.intermediate, situ_beta=problem.situ_beta,
                    situ_linear_beta=problem.situ_linear_beta,
                ),
                group=group,
                sample_size=args.ref_samples,
            )
            result["rel_l2_oracle"] = R.output_error(out[:n], idx, expected)
            bf16_ref = R.sampled_reference(
                [hidden, ids.to(torch.int64), wts],
                n,
                lambda p: R.bf16_partial(
                    p[0], p[1], p[2], canonical, local_expert_offset=offset,
                    intermediate=problem.intermediate, situ_beta=problem.situ_beta,
                    situ_linear_beta=problem.situ_linear_beta,
                ),
                group=group,
                sample_size=args.ref_samples,
            )
            result["rel_l2_bf16"] = R.output_error(out[:n], *bf16_ref)
            errs = torch.tensor(
                [result["rel_l2_oracle"], result["rel_l2_bf16"]], device=device
            )
            if group is not None:
                dist.all_reduce(errs, op=dist.ReduceOp.MAX, group=group)
            result["rel_l2_oracle"], result["rel_l2_bf16"] = errs.tolist()

        target = out[:n] if n else session.empty_output

        if fused is not None and n:
            got = res[1].float()
            want = shared_ref(n)
            exact = shared_ref(n, exact_act=True)

            def rel(a, b):
                return float((a - b).norm() / b.norm().clamp_min(1e-30))

            result["shared_rel_l2"] = rel(got, want)
            # noise floor: the stock chain and the kernel vs an unquantized activation
            result["shared_rel_l2_stock_vs_exact_act"] = rel(want, exact)
            result["shared_rel_l2_kernel_vs_exact_act"] = rel(got, exact)

        def launch():
            session.launch(weights, target, rmsnorm=rms, shared=fused)

        def forward():
            moe.forward(
                hidden, ids, wts, weights, bucket=bucket, out=out, shared=fused, rmsnorm=rms
            )

        # eager (FlashInfer tuner protocol of the knob records), then CUDA graphs
        for _ in range(args.warmup):
            launch()
        timing = dict(samples=args.iters, delay_cycles=delay_cycles, group=group)
        kernel_eager = measure(launch, replays=1, **timing)
        for _ in range(args.warmup):
            forward()
        staged_eager = measure(forward, replays=1, **timing)
        g_kernel = capture_graph(launch, group, "session.launch")
        g_forward = capture_graph(forward, group, "moe.forward")
        for _ in range(args.warmup):
            g_kernel.replay()
            g_forward.replay()
        torch.cuda.synchronize()
        kernel = measure(g_kernel.replay, replays=args.graph_replays, **timing)
        staged = measure(g_forward.replay, replays=args.graph_replays, **timing)
        # CUDA graph safety: a replayed forward equals an eager one
        forward()
        torch.cuda.synchronize()
        eager_out = out[:n].clone()
        g_forward.replay()
        torch.cuda.synchronize()
        diff = torch.tensor(
            [float((out[:n].float() - eager_out.float()).abs().max()) if n else 0.0],
            device=device,
        )
        del g_kernel, g_forward
        moe.check_route_errors()

        sweep = None
        if args.sweep_fc1_norm and bf16_ref is not None:
            sweep = {}
            for e in range(8):
                const = 2.0**e
                w_c = W.prepare_weights(
                    problem.quant_kind,
                    *canonical,
                    hidden=problem.hidden,
                    intermediate=problem.intermediate,
                    input_norm_const=problem.input_norm_const,
                    fc1_norm_const=const,
                    layer_id=0,
                    expert_offset=offset,
                )
                moe.forward(hidden, ids, wts, w_c, bucket=bucket, out=out)
                torch.cuda.synchronize()
                err = torch.tensor([R.output_error(out[:n], *bf16_ref)], device=device)
                if group is not None:
                    dist.all_reduce(err, op=dist.ReduceOp.MAX, group=group)
                sweep[f"2^{e}"] = float(err)
            moe.check_route_errors()

        mism = torch.tensor([stage_mismatch], device=device, dtype=torch.int64)
        if group is not None:
            dist.all_reduce(mism, group=group)
            dist.all_reduce(diff, op=dist.ReduceOp.MAX, group=group)
        line = {
            "quant_kind": problem.quant_kind,
            "variant": bucket_of[cap].variant,
            "world_size": world,
            "tokens_per_rank": n,
            "max_tokens": cap,
            "routing": args.routing,
            "weights": args.weights,
            "input_norm_const": problem.input_norm_const,
            "fc1_norm_const": problem.fc1_norm_const,
            "kernel": kernel,
            "stage_and_kernel": staged,
            "kernel_eager": kernel_eager,
            "stage_and_kernel_eager": staged_eager,
            "timer": timer,
            "graph_vs_eager_max_abs": float(diff),
            "stage_mismatch_bytes_all_ranks": int(mism),
            **result,
            "fc1_norm_sweep_rel_l2_bf16": sweep,
            "knobs": bucket.knobs.as_dict(),
            "kernel_name": session.name,
            "arena_bytes": moe.arena.nbytes,
            "local_workspace_bytes": sum(s.local_bytes for s in moe.sessions.values()),
            "nvfp4_max_spread": (
                None if weights.report is None else weights.report.max_spread
            ),
            "nvfp4_subnormal_scale_blocks": (
                None if weights.report is None else weights.report.subnormal_blocks
            ),
            "env": ident,
        }
        if rank == 0:
            text = json.dumps(line, default=str)
            print(text, flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(text + "\n")

    torch.cuda.synchronize()
    # Served EP-balance ring (int64.3 item 3): every sampled call's global routes must be one point's n x world x topk
    # (random routing: every route valid), and the per-rank loads must add up to it.
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance

    if args.epbal_percall and rank == 0:
        print(f"EPBAL_PERCALL_ALL {'PASS' if _EPBAL_VERDICTS and all(_EPBAL_VERDICTS) else 'FAIL'} "
              f"shapes={len(_EPBAL_VERDICTS)} passed={sum(_EPBAL_VERDICTS)}", flush=True)
    recs = ep_balance.drain_all()
    expected = {n * world * problem.top_k for _, _, n in points}
    glob = [r for r in recs if r["local_rank"] < 0]
    ok = sum(1 for r in glob if r["routes"] in expected and sum(r["rank_loads"]) == r["routes"] and r["world"] == world)
    # local (decode genphase) records: every rank logs its own experts; merge by call across ranks
    gathered = [recs]
    if world > 1 and group is not None:
        gathered = [None] * world
        dist.all_gather_object(gathered, recs, group=group)
    merged = ep_balance.merge_local(gathered)
    ok += sum(1 for m in merged if m["complete"] and m["routes"] in expected)
    n_rec = len(glob) + len(merged)
    if rank == 0:
        mm = [round(r["max_over_mean"], 3) for r in glob + merged]
        print(f"EPBAL records={n_rec} routes_ok={ok}/{n_rec} max_over_mean={mm[:8]}", flush=True)
    moe.destroy()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
