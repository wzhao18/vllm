"""cfg2bench.py: a production no-quant decode config -> the step bench's engine args and env.

    cfg2bench.py <point|cfg.yaml|cfg.json> --arm stock|ls [--drafter old|new|<dir>] [--rows 8,16,...]
                 [--format sh|json]
    cfg2bench.py <point> --split --rows 8,...,1024   # '<rows <= production limit> <rows above>' 
    cfg2bench.py --snapshot <dir>        # JSON copies of every nq-*-xdr.yaml (login node, PyYAML)

The decode role of ``jh/noquant/cfg/nq-<point>-xdr.yaml`` is the source of truth: its args
and env (plus the global ``environment`` keys the engine reads) become the bench's, so the
A/B cannot drift from the e2e configs. Every decode arg is either kept, dropped with a
reason (serving/P/D/infra only) or rewritten; an arg in neither list is an error (a new
production knob must be classified here, not silently lost). Both arms get the same args
and env; only ``additional-config`` differs (stock = ``decode_backend`` auto, ls =
``latency_specialized``), as audit 'stock baseline validity' requires (no in-process
fallback as the stock arm).

Bench-only changes (printed as comments in the output):
  * --load-format dummy (both arms), target = the e2e model view (dummy weights, real
    config), drafter path = the Lustre checkpoint behind /raid/common-models (old) or the
    new kimi-k3.1-pro-dspark-preview (--drafter new);
  * capture sizes: the production list plus every measured row B*q and its drafter rows
    B*K (so each measured shape is a captured FULL graph in production terms: the verify
    graph runs it unpadded and draft_batches() contains B); max-cudagraph-capture-size and
    max-num-seqs grow to the largest measured row when --rows exceeds the production
    limit min(max-num-seqs * q, max capture) (the arbitrary-batch goal: rows the
    production config never captured). nq_ab.sh runs the rows <= that limit and the
    rows above it as separate processes (--part lo|hi), so the rows inside the limit
    always run under the production max-num-seqs and capture cap;
  * env path rewrites: the node-local /raid/scratch JIT dirs of the e2e run move to the
    bench's per-overlay build root ($NQ_BUILD, set by nq_inner.sh) or to the inner
    script's cache layout.

Reads only json/yaml/argparse: runs on the login node or a batch host (PyYAML optional
when a --snapshot JSON copy exists).
"""
import argparse
import hashlib
import json
import os
import shlex
import sys

W = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr"
M = "/lustre/fsw/coreai_comparch_inferencex/weizha/models"
NQ_CFG = f"{W}/jh/noquant/cfg"
NB = f"{W}/nqbench"
# Target: the e2e model key hf:moonshotai/Kimi-K3 (srtslurm-jh.yaml / k31r srtslurm.yaml)
# is this view (K3.1 config with model_type kimi_k3); weights are dummy in the bench.
TARGET = f"{W}/e2e/model-view/kimi-k3.1-k3type"
DRAFTERS = {
    "old": f"{M}/kimi-k3-dspark",  # 64 heads, markov_rank 256 (all nq configs today)
    "new": f"{M}/kimi-k3.1-pro-dspark-preview",  # 96 heads (6/rank at TP16), markov_rank 512
}
# The configs mount the old drafter here (mk_nq.py extra_mount).
DRAFT_MOUNT = "/raid/common-models/kimi-k3-dspark"

# decode args the bench passes through unchanged. NQ-TUNE: gpu-memory-utilization stays at the production 0.9 even
# where --rows doubles max-num-seqs (q4 B256 / q8 B128 at u192k): check the KV pool fits those cases (no 'blocks'
# error in the 1024-row cases) before reading anything into a missing row.
KEEP = {
    "kv-cache-dtype", "mamba-ssm-cache-dtype", "gpu-memory-utilization",
    "tensor-parallel-size", "decode-context-parallel-size", "block-size",
    "cp-kv-cache-interleave-size", "dcp-comm-backend", "trust-remote-code",
    "language-model-only", "moe-backend", "no-enable-flashinfer-autotune",
    "attention-backend", "sparse-indexer-topk-backend", "quantization-config",
    "max-model-len", "max-num-batched-tokens", "attention-config",
    "distributed-timeout-seconds", "cpu-distributed-timeout-seconds",
    "data-parallel-size", "enable-expert-parallel", "enable-ep-weight-filter",
    "use-replayssm", "kernel-config",
}
# decode args the bench rewrites (handled below)
REWRITE = {
    "speculative-config", "additional-config", "compilation-config",
    "max-num-seqs", "max-cudagraph-capture-size",
}
# decode args with no meaning in a single-process step bench
DROP = {
    "served-model-name": "serving only",
    "load-format": "bench uses dummy weights (both arms)",
    "enable-prefix-caching": "bench forward forces it off",
    "prefix-match-unit": "prefix caching only",
    "stream-interval": "serving output only",
    "kv-transfer-config": "P/D connector (no transfer in the bench)",
    "enable-cumem-allocator": "allocator pool for sleep/offload; bench replays one graph per case",
    "numa-bind": "host CPU binding (graph replay timing is GPU-only)",
    "numa-bind-nodes": "host CPU binding",
    "data-parallel-rpc-port": "DP serving RPC",
}
# global `environment` keys the engine reads (all others are serving/transport infra)
ENV_PREFIXES = ("VLLM_", "NCCL_", "K31_", "K3_", "DG_", "FLASHINFER_", "TVM_FFI_")
ENV_DROP = {
    "PYTHONPATH": "nq_inner.sh composes it (base env + code-under-test overlay)",
    "VLLM_USE_RUST_FRONTEND": "frontend",
    "VLLM_SERVER_DEV_MODE": "server",
    "VLLM_ENGINE_READY_TIMEOUT_S": "server",
    "VLLM_WORKER_MULTIPROC_METHOD": "external launcher: no worker processes",
    "VLLM_ITERLOG_DIR": "serving iteration log",
    "VLLM_MOONCAKE_LOAD_RECV_THREADS": "Mooncake store",
    "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE": "Mooncake store",
    "K3_PIN_CPUS": "pinhook (host CPU pinning) is not on the bench path",
    "VLLM_CACHE_ROOT": "nq_inner.sh cache layout",
    "FLASHINFER_WORKSPACE_BASE": "nq_inner.sh cache layout (node-local in e2e)",
    "DG_JIT_CACHE_DIR": "nq_inner.sh cache layout (node-local in e2e)",
    # Removed from the code (automatic per-shape policy in the specialized decode): an
    # older config may still carry them; the code under test no longer reads them.
    "VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS": "removed: WideDsa rows are an automatic per-shape policy",
    "VLLM_K31_DSA_PARTIALS": "removed: the DSA partials format is an automatic per-shape policy",
}
# reasons for the global `environment` keys outside ENV_PREFIXES (every dropped key is listed)
ENV_DROP_PREFIX = {
    "UCX_": "UCX transport (P/D NIXL), not on the bench path",
    "NIXL_": "P/D KV transfer",
    "MC_": "Mooncake store",
    "MOONCAKE": "Mooncake store",
    "DYN_": "dynamo serving frontend",
    "ETCD_": "dynamo service discovery",
    "NATS_": "dynamo messaging",
    "WITH_NVIDIA_PEERMEM": "P/D transport (GPUDirect RDMA)",
    "CUTE_DSL_": "nq_inner.sh cache layout",
    "TORCHINDUCTOR_": "nq_inner.sh cache layout",
    "TRITON_": "nq_inner.sh cache layout",
    "CUDA_CACHE_PATH": "nq_inner.sh cache layout (node-local)",
    "XDG_": "nq_inner.sh cache layout",
    "HF_HOME": "nq_inner.sh cache layout",
    "HOME": "nq_inner.sh cache layout",
}
# node-local e2e JIT build dirs -> the bench's per-overlay build root (nq_inner.sh: $NQ_BUILD)
BUILD_DIRS = {"K31_DCPX_BUILD_DIR": "dcpx", "K31_DSAC_BUILD_DIR": "dsac"}


def parse_sizes(spec):
    sizes = set()
    for item in str(spec).split(","):
        if item.strip():
            lo, _, hi = item.partition("-")
            sizes.update(range(int(lo), int(hi or lo) + 1))
    return sorted(sizes)


def resolve(cfg):
    """A point name -> the nq decode config (the --snapshot JSON copy when present)."""
    if os.path.exists(cfg):
        return cfg
    snap = f"{NB}/cfg/nq-{cfg}-xdr.json"
    if os.path.exists(snap):
        return snap
    return f"{NQ_CFG}/nq-{cfg}-xdr.yaml"


def load(path):
    if path.endswith(".json"):
        with open(path) as f:
            doc = json.load(f)
        src = doc.get("_snapshot", {}).get("source")
        if src and os.path.exists(src):
            with open(src, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()[:12]
            if sha != doc["_snapshot"]["sha256"]:
                sys.exit(f"cfg2bench: snapshot {path} is stale ({src} changed): rerun --snapshot")
        return doc
    import yaml  # login node / snapshot

    with open(path) as f:
        return yaml.safe_load(f)


def snapshot(out):
    import glob

    import yaml

    os.makedirs(out, exist_ok=True)
    for src in sorted(glob.glob(f"{NQ_CFG}/nq-*-xdr.yaml")):
        with open(src, "rb") as f:
            raw = f.read()
        doc = yaml.safe_load(raw)
        doc["_snapshot"] = {"source": src, "sha256": hashlib.sha256(raw).hexdigest()[:12]}
        dst = os.path.join(out, os.path.basename(src)[:-5] + ".json")
        with open(dst + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.rename(dst + ".tmp", dst)
        print("wrote", dst)


def flag(key, value):
    """One engine arg as argv items (store_true flags carry no value)."""
    if value is True:
        return [f"--{key}"]
    if value is False or value is None:
        return []
    if isinstance(value, (list, dict)):
        value = json.dumps(value)
    return [f"--{key}", str(value)]


def env_drop_reason(key):
    for prefix, why in ENV_DROP_PREFIX.items():
        if key.startswith(prefix):
            return why
    return "not read by the engine (serving / transport / host infra)"


def split_rows(doc, rows):
    """(rows <= the production limit, rows above it): limit = min(max-num-seqs * q,
    max capture size), the largest verify row the production config captures."""
    lim = production_limit(doc)
    return [m for m in rows if m <= lim], [m for m in rows if m > lim]


def production_limit(doc):
    args = doc["roles"]["decode"]["args"]
    k = int(json.loads(args["speculative-config"])["num_speculative_tokens"])
    comp = json.loads(args.get("compilation-config") or "{}")
    sizes = comp.get("cudagraph_capture_sizes") or ()
    maxcap = int(args.get("max-cudagraph-capture-size") or max(sizes))
    return min(int(args["max-num-seqs"]) * (k + 1), maxcap)


def build(doc, arm, drafter, rows):
    dec = doc["roles"]["decode"]
    args, renv = dict(dec["args"]), dict(dec.get("env") or {})
    unknown = sorted(set(args) - KEEP - REWRITE - set(DROP))
    if unknown:
        sys.exit(f"cfg2bench: unclassified decode args {unknown}: add them to KEEP/DROP")
    notes = []
    tp = int(args["tensor-parallel-size"])
    dcp = int(args.get("decode-context-parallel-size", 1))
    if (tp, dcp) != (16, 16):
        notes.append(f"WARNING: decode is TP{tp}/DCP{dcp}, not the TP16=DCP16 A/B layout")
    spec = json.loads(args["speculative-config"])
    k = int(spec["num_speculative_tokens"])
    q = k + 1
    draft = DRAFTERS.get(drafter, drafter)
    if not os.path.isabs(draft):
        sys.exit(f"cfg2bench: drafter {drafter!r} is neither old/new nor an absolute path")
    if spec.get("model") not in (DRAFT_MOUNT, DRAFTERS["old"], DRAFTERS["new"]):
        notes.append(f"NOTE: config drafter {spec.get('model')} replaced by {draft}")
    spec["model"] = draft
    comp = json.loads(args.get("compilation-config") or "{}")
    sizes = sorted(set(comp.get("cudagraph_capture_sizes") or ()))
    mns = int(args["max-num-seqs"])
    maxcap = int(args.get("max-cudagraph-capture-size") or max(sizes))
    added = []
    for m in rows:
        if m % q:
            sys.exit(f"cfg2bench: row {m} is not a multiple of q={q}")
        for s in (m, m // q * k):  # verify rows, drafter rows (draft.draft_batches: B*K)
            if s not in sizes:
                sizes.append(s)
                added.append(s)
    sizes.sort()
    limit = min(mns * q, maxcap)  # production: the largest captured verify row
    if rows and max(rows) > limit:
        top = max(rows)
        notes.append(
            f"NOT PRODUCTION: capture cap {maxcap} -> {max(maxcap, top)} and max-num-seqs "
            f"{mns} -> {max(mns, -(-top // q))}: rows {[m for m in rows if m > limit]} are "
            f"above the production limit {limit} (min(max-num-seqs * q, capture cap))"
        )
        maxcap = max(maxcap, top)
        mns = max(mns, -(-top // q))
    sizes = [s for s in sizes if s <= maxcap]
    if added:
        notes.append(f"capture sizes added for measured rows: {sorted(added)}")
    comp["cudagraph_capture_sizes"] = sizes
    extra = json.loads(args.get("additional-config") or "{}")
    extra["decode_backend"] = "latency_specialized" if arm == "ls" else "auto"
    argv = ["--model", TARGET, "--load-format", "dummy"]
    for key in sorted(KEEP & set(args)):
        argv += flag(key, args[key])
    argv += flag("max-num-seqs", mns)
    argv += flag("max-cudagraph-capture-size", maxcap)
    # bench forward overrides cudagraph_mode (NONE: it captures each case itself) and
    # the post-init then clears the sizes; nq_step.py restores them where the
    # specialized decode reads them (--capture-sizes / --max-capture).
    argv += flag("compilation-config", comp)
    argv += flag("speculative-config", spec)
    argv += flag("additional-config", extra)
    env, dropped = {}, {}
    for scope in (doc.get("environment") or {}, renv):  # role env overrides global
        for key, value in scope.items():
            if key in ENV_DROP:
                dropped[key] = ENV_DROP[key]
            elif key in BUILD_DIRS:
                env[key] = "${NQ_BUILD}/" + BUILD_DIRS[key]
            elif key.startswith(ENV_PREFIXES) or key == "HF_HUB_OFFLINE":
                if "/raid/scratch" in str(value):
                    dropped[key] = f"node-local e2e path {value}"
                else:
                    env[key] = str(value)
            elif key == "CUDA_CACHE_MAXSIZE":
                env[key] = str(value)
            else:
                dropped[key] = env_drop_reason(key)
    return {
        "q": q,
        "k": k,
        "mns": mns,
        "max_capture": maxcap,
        "production_limit": limit,
        "capture_sizes": sizes,
        "argv": argv,
        "env": env,
        "dropped_env": dropped,
        "dropped_args": {a: DROP[a] for a in sorted(set(args) & set(DROP))},
        "notes": notes,
        "drafter": draft,
        "arm": arm,
    }


def as_sh(out, src):
    lines = [f"# cfg2bench {src} arm={out['arm']} drafter={out['drafter']} q={out['q']}"]
    lines += [f"# {n}" for n in out["notes"]]
    lines += [f"# dropped arg {a}: {why}" for a, why in out["dropped_args"].items()]
    lines += [f"# dropped env {e}: {why}" for e, why in sorted(out["dropped_env"].items())]
    for key, value in sorted(out["env"].items()):
        # ${NQ_BUILD} stays an expansion; every other value is literal
        val = value if value.startswith("${NQ_BUILD}") else shlex.quote(value)
        lines.append(f"export {key}={val}")
    lines.append(f"NQ_Q={out['q']}")
    lines.append(f"NQ_MNS={out['mns']}")
    lines.append(f"NQ_MAXCAP={out['max_capture']}")
    lines.append(f"NQ_PRODLIMIT={out['production_limit']}")
    lines.append("NQ_CAP=" + ",".join(map(str, out["capture_sizes"])))
    lines.append("NQ_ARGS=(" + " ".join(shlex.quote(a) for a in out["argv"]) + ")")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cfg", nargs="?")
    ap.add_argument("--arm", choices=["stock", "ls"])
    ap.add_argument("--drafter", default="old")
    ap.add_argument("--rows", default="", help="measured verify rows (B*q), e.g. 8,16,32")
    ap.add_argument("--format", choices=["sh", "json"], default="sh")
    ap.add_argument(
        "--split", action="store_true",
        help="print '<rows <= production limit> <rows above it>' (comma lists, '-' = none)",
    )  # fmt: skip
    ap.add_argument("--snapshot", metavar="DIR")
    a = ap.parse_args()
    if a.snapshot:
        return snapshot(a.snapshot)
    if a.split:
        if not a.cfg:
            ap.error("cfg is required")
        lo, hi = split_rows(load(resolve(a.cfg)), parse_sizes(a.rows) if a.rows else [])
        print(",".join(map(str, lo)) or "-", ",".join(map(str, hi)) or "-")
        return
    if not a.cfg or not a.arm:
        ap.error("cfg and --arm are required")
    src = resolve(a.cfg)
    out = build(load(src), a.arm, a.drafter, parse_sizes(a.rows) if a.rows else [])
    print(json.dumps(out, indent=1) if a.format == "json" else as_sh(out, src))


if __name__ == "__main__":
    main()
