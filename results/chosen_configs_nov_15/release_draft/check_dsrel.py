#!/usr/bin/env python3
"""check_dsrel.py: GPU-free checks of the release-draft renders written by gen_dsrel.py, run on the Hecate login node with
the srtctl venv python (PyYAML + srtctl):
    /lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/srtctl-venv/bin/python check_dsrel.py [--points ...] [--fabrics ...]
        [--kinds ...] [--prefix dsrel] [--draft-attn-backend FLASHINFER] [--scratch DIR] [--no-render]
Nothing is submitted: `srtctl render` and srtctl's load_config / RuntimeContext only read the config and write the render
into --scratch. Exit 0 = every check PASS.

Per render (dsrel-<base>[-ev]-<fab> vs its source frz-<base>[-ev]-<fab>):
  1. files: the source's files + DSREL.txt; same modes; no symlinks; no *.tmp-* / *.prev-* next to it.
  2. byte checks: every runtime file == the source with frz-<base>[-ev]-<fab> renamed (sbatch_script.sh, render.json,
     workflow.env.resolved, point.env, srtslurm.yaml, kmc.args, nq4.drop); provenance records (FRZ.txt, INTEG.txt, render.log)
     verbatim; git_state.txt = source with the draft mount line swapped; golden-acceptance.json / AL_FIX.txt = release AL
     (AgentX) or null / absent (gsm8k). No runtime file names a frozen render (the stale-name pitfall: sbatch_script.sh stages
     <render dir>/render/*.yaml at job start).
  3. parsed YAML (render/config.yaml = what runs, and hybrid.yaml): flattened diff vs the source = the two speculative-configs
     (JSON: only model / attention_backend / synthetic_acceptance_length) + the draft extra_mount (+ the role PYTHONPATHs
     when the tree moved). Nothing else (topology, k, CONC, capture sizes, Mooncake, precision, env) differs.
  4. speculative-config semantics on prefill = decode: dspark, the release draft mount, the source's k, the backend knob,
     probabilistic drafting; AgentX: synthetic at the release golden AL (and its per-position rates); gsm8k: block, no AL.
  5. AgentX / gsm8k pair: the agentx-vs-gsm8k parsed diff of the new pair == that of the frozen pair, except the
     speculative-config, whose agentx -> gsm8k change is exactly rejection synthetic -> block and the AL dropped.
  6. srtctl load_config + RuntimeContext (as do_sweep, with the render's srtslurm.yaml): container alias -> existing .sqsh,
     /model, draft mount <deps>/kimi-k3.1-pro-dspark-e4e6e626 -> /raid/common-models/kimi-k3.1-pro-dspark (ro), no preview
     mount, k31vr at its own path, /raid/scratch, /logs, nothing mounted over /opt/k3.
  7. `srtctl render` (preflight on: model path + container) of render/config.yaml into --scratch: it must render, and the
     rendered sbatch_script.sh / render.json / git_state.txt / config.yaml must differ from the render's files exactly as the
     frozen source's re-render differs from the frozen source's files (paths and Generated stamps only).
Once: the release draft dir (config.json md5, safetensors size, sha256 record, the fields the configs rely on) and the
vLLM tree on the PYTHONPATH (present, full, release-draft code present = launch-ready).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dsrel as G  # noqa: E402

import yaml  # noqa: E402

K = "/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr"
SRTCTL = f"{K}/srtctl-venv/bin/srtctl"
SRT_ROOT = f"{K}/srt-slurm"
MODEL_VIEW = f"{G.W}/e2e/model-view/kimi-k3.1-k3type"
RELEASE_CONFIG_MD5 = "688531896a5bc74c362b8e85ca619385"
RELEASE_SAFETENSORS_BYTES = 4959959810
TREE_FILES = ("vllm/__init__.py", "vllm/_C_stable_libtorch.abi3.so", "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
              "vllm/third_party/trtllm_gvr/__init__.py", "vllm/models/kimi_k3/nvidia/ops/sp_all_gather.py",
              "vllm/model_executor/models/qwen3_dspark.py")
# release-draft code that the frozen tree lacks (tracks M and K of the dsrel workflow): launch readiness markers
READY_MARKERS = ("vllm/models/kimi_k3/nvidia/dspark_gqa.py", "vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/draft/gqa_*.py")

FAILS: list[str] = []
WARNS: list[str] = []


def ok(cond: bool, msg: str) -> bool:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def warn(msg: str) -> None:
    print("WARN " + msg)
    WARNS.append(msg)


def files_of(d: str) -> set[str]:
    return {os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs}


def flat_yaml(text: str) -> dict:
    return dict(G.flat(yaml.safe_load(text)))


def spec_of(cfg: dict, role: str) -> dict:
    return json.loads(cfg["roles"][role]["args"]["speculative-config"])


def check_render(src: str, dst: str, base: str, kind: str, prefix: str, backend: str, tree: str | None,
                 weights: tuple[str, str] | None = None) -> dict:
    name = os.path.basename(dst)
    sname = os.path.basename(src)
    print(f"\n== {name}  (source {sname})")
    fs, fd = files_of(src), files_of(dst)
    ok(fd == fs | {"DSREL.txt"}, f"{name}: file set = source + DSREL.txt (extra {sorted(fd - fs - {'DSREL.txt'})}, missing {sorted(fs - fd)})")
    ok(not any(os.path.islink(os.path.join(dst, f)) for f in fd), f"{name}: no symlinks")
    ok(all(os.stat(os.path.join(src, f)).st_mode == os.stat(os.path.join(dst, f)).st_mode for f in fs & fd),
       f"{name}: file modes kept (sbatch_script.sh executable)")
    side = [x for x in os.listdir(os.path.dirname(dst)) if x.startswith(name + ".tmp-")]
    ok(not side, f"{name}: no leftover temp dirs {side}")
    frz_re = G.full_name_re(base)
    for f in sorted(fs & fd):
        a = open(os.path.join(src, f)).read()
        b = open(os.path.join(dst, f)).read()
        if f in ("render/config.yaml", "hybrid.yaml"):
            continue
        if f == "render/git_state.txt":
            exp = a.replace(f"  - extra_mount:{G.PREVIEW_MNT}:ro: {G.PREVIEW_HOST}\n",
                            f"  - extra_mount:{G.RELEASE_MNT}:ro: {G.RELEASE_HOST}\n")
            if weights is not None:
                exp = exp.replace(f"  - extra_mount:{G.TEST_WEIGHTS}:ro: {G.TEST_WEIGHTS}\n",
                                  f"  - extra_mount:{weights[0]}:ro: {weights[0]}\n")
            ok(b == exp and exp != a, f"{name}: {f} = source with the draft (and target weights) mount lines swapped")
        elif f == "srtslurm.yaml" and weights is not None:
            exp = G.rename(a, base, prefix)[0].replace(G.TEST_VIEW, weights[1])
            ok(b == exp and a.count(G.TEST_VIEW) == 3, f"{name}: {f} = source with the model view -> {weights[1]}")
        elif f == "golden-acceptance.json":
            continue
        elif f == "AL_FIX.txt":
            continue
        elif f.endswith(G.PROVENANCE_SUFFIXES):
            ok(b == a, f"{name}: provenance record {f} verbatim")
        else:
            ok(b == G.rename(a, base, prefix)[0], f"{name}: {f} = source with the render name changed")
            ok(not frz_re.search(b), f"{name}: {f} names no frozen render")
    sb = open(f"{dst}/render/sbatch_script.sh").read()
    ok(f'cp "{G.RS}/{name}/render"/*.yaml "${{OUTPUT_DIR}}/"' in sb, f"{name}: sbatch_script.sh stages its own render dir")
    ok(f"#SBATCH --job-name=coreai_comparch_inferencex-k31e2e.{name}\n" in sb, f"{name}: job name")
    rj = json.load(open(f"{dst}/render/render.json"))
    ok(rj["script"] == f"{G.RS}/{name}/render/sbatch_script.sh" and rj["name"].endswith("." + name), f"{name}: render.json name/script")
    # parsed YAMLs
    cfg_new = yaml.safe_load(open(f"{dst}/render/config.yaml"))
    cfg_old = yaml.safe_load(open(f"{src}/render/config.yaml"))
    for f in ("render/config.yaml", "hybrid.yaml"):
        diffs = G.yaml_diff(open(f"{src}/{f}").read(), open(f"{dst}/{f}").read(), base, prefix)
        bad = G.check_diff(diffs, tree or G.FROZEN_TREE, weights)
        ok(not bad, f"{name}: {f} parsed diff = {len(diffs)} keys, all draft keys ({'; '.join(bad)[:400]})")
        t = open(f"{dst}/{f}").read()
        ok("kimi-k3.1-pro-dspark-preview" not in t and "TOKENSPEED_MLA" not in t, f"{name}: {f} has no preview draft / TOKENSPEED_MLA")
    sp, sd = spec_of(cfg_new, "prefill"), spec_of(cfg_new, "decode")
    k = spec_of(cfg_old, "decode")["num_speculative_tokens"]
    ok(sp == sd, f"{name}: prefill and decode speculative-config identical")
    exp = {"method": "dspark", "model": G.RELEASE_MNT, "num_speculative_tokens": k}
    if backend != "auto":
        exp["attention_backend"] = backend
    exp["draft_sample_method"] = "probabilistic"
    if kind == "agentx":
        exp.update(rejection_sample_method="synthetic", synthetic_acceptance_length=G.RELEASE_AL[k])
    else:
        exp.update(rejection_sample_method="block")
    ok(sd == exp, f"{name}: speculative-config == {json.dumps(exp)}")
    al = sd.get("synthetic_acceptance_length")
    if al is not None:
        r = G.rates(al, k)
        ok(1 <= al <= k + 1 and abs(1 + sum(r) - al) < 1e-9, f"{name}: AL {al} -> per-position rates {r} (sum + 1 = AL)")
    ga = json.load(open(f"{dst}/golden-acceptance.json"))
    if kind == "agentx":
        ok(ga == {"golden_acceptance_length": G.RELEASE_AL[k]}, f"{name}: golden-acceptance.json {ga}")
        alfix = open(f"{dst}/AL_FIX.txt").read()
        ok(f"AL -> {G.RELEASE_AL[k]} (k={k}," in alfix and alfix.count("\n") == 1, f"{name}: AL_FIX.txt records the release AL")
    else:
        ok(ga == {"golden_acceptance_length": None}, f"{name}: golden-acceptance.json {ga}")
    ds = open(f"{dst}/DSREL.txt").read()
    ok(ds.startswith(f"{name} = {sname} + release draft"), f"{name}: DSREL.txt names its source")
    pp = {r: cfg_new["roles"][r]["env"]["PYTHONPATH"].split(":") for r in ("prefill", "decode")}
    trees = {e for v in pp.values() for e in v if "/k31vr/" in e}
    ok(trees == {tree or G.FROZEN_TREE}, f"{name}: the vLLM tree on both role PYTHONPATHs = {sorted(trees)}")
    return dict(name=name, cfg=cfg_new, k=k, spec=sd, tree=next(iter(trees)) if len(trees) == 1 else None,
                nodes=rj["total_nodes"])


def check_pair(src_a: str, src_e: str, dst_a: str, dst_e: str, base: str, prefix: str) -> None:
    def d(a, b):
        fa, fb = flat_yaml(G.rename(open(a).read(), base, prefix)[0]), flat_yaml(G.rename(open(b).read(), base, prefix)[0])
        return {k: (fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
    old = d(f"{src_a}/render/config.yaml", f"{src_e}/render/config.yaml")
    new = d(f"{dst_a}/render/config.yaml", f"{dst_e}/render/config.yaml")
    name = os.path.basename(dst_a)
    spec_keys = {".roles.prefill.args.speculative-config", ".roles.decode.args.speculative-config"}
    ok(set(old) == set(new), f"{name} / -ev: agentx-vs-gsm8k diff keys = the frozen pair's ({len(new)} keys)")
    same = all(new[k] == old[k] for k in set(old) & set(new) - spec_keys if not k == ".name")
    ok(same, f"{name} / -ev: agentx-vs-gsm8k values = the frozen pair's (outside the speculative-configs)")
    for k in spec_keys & set(new):
        ja, je = json.loads(new[k][0]), json.loads(new[k][1])
        moved = {x: (ja.get(x), je.get(x)) for x in set(ja) | set(je) if ja.get(x) != je.get(x)}
        ok(set(moved) == {"rejection_sample_method", "synthetic_acceptance_length"} and moved["rejection_sample_method"] ==
           ("synthetic", "block") and moved["synthetic_acceptance_length"][1] is None,
           f"{name} / -ev {k.split('.')[2]}: agentx -> gsm8k = synthetic -> block, AL dropped ({moved})")


def check_mounts(dst: str, weights: tuple[str, str] | None = None) -> None:
    from pathlib import Path

    from srtctl.core.config import load_config
    from srtctl.core.runtime import RuntimeContext

    name = os.path.basename(dst)
    os.environ["SRTSLURM_CONFIG"] = f"{dst}/srtslurm.yaml"
    # a node list for the role placement, but NO job id: with SLURM_JOB_ID set, srtctl resolves node IPs with
    # `srun --jobid`; without it, socket resolution (no Slurm call). The slurm stubs on PATH (main) back this up.
    for k in ("SLURM_JOB_ID", "SLURM_JOBID"):
        os.environ.pop(k, None)
    os.environ["SLURM_NODELIST"] = os.environ["SLURM_JOB_NODELIST"] = "hecate[0001-0014]"
    c = load_config(f"{dst}/render/config.yaml")
    img = str(c.model.container)
    ok(img.endswith(".sqsh") and os.path.exists(img), f"{name}: container alias -> {img}")
    rt = RuntimeContext.from_config(c, "1", Path(f"/tmp/chk-dsrel-{os.getuid()}/{name}"))
    m = {str(h): str(t) for h, t in rt.container_mounts.items()}
    tgt = {t.split(":")[0]: (h, t) for h, t in m.items()}
    view = weights[1] if weights is not None else MODEL_VIEW
    ok(tgt.get("/model", ("",))[0] == view, f"{name}: /model <- {tgt.get('/model', ('-',))[0]} (want {view})")
    if weights is not None:
        w = tgt.get(weights[0])
        ok(w is not None and w[0] == weights[0] and w[1].endswith(":ro"), f"{name}: target weights {weights[0]} mounted ro")
        ok(G.TEST_WEIGHTS not in m, f"{name}: test weights not mounted")
    rel = tgt.get(G.RELEASE_MNT)
    ok(rel is not None and rel[0] == G.RELEASE_HOST and rel[1].endswith(":ro"), f"{name}: draft {G.RELEASE_HOST} -> {rel and rel[1]}")
    ok(G.PREVIEW_MNT not in tgt and G.PREVIEW_HOST not in m, f"{name}: preview draft not mounted")
    ok(tgt.get(G.W, ("",))[0] == G.W, f"{name}: k31vr mounted at its own path (tree, JIT dirs, AgentX runtime)")
    ok(tgt.get("/raid/scratch", ("",))[0] == "/raid/scratch", f"{name}: /raid/scratch mounted")
    ok("/logs" in tgt, f"{name}: /logs mounted")
    ok(not any(t.startswith("/opt/k3") for t in tgt), f"{name}: nothing mounted over /opt/k3")


def norm_render(text: str, scratch_dir: str, render_dir: str) -> str:
    text = text.replace(scratch_dir, render_dir)
    text = re.sub(r"^# Generated: \d{8}_\d{6}$", "# Generated: X", text, flags=re.M)
    return re.sub(r"^Generated at: .*$", "Generated at: X", text, flags=re.M)


def rerender(render_dir: str, scratch: str) -> tuple[int, dict]:
    """srtctl render of <render>/render/config.yaml into scratch (as <scratch>/<name>/hybrid.yaml so the script's Config
    path maps back); returns rc and {file: normalized(rendered) == normalized(render's)}."""
    name = os.path.basename(render_dir)
    sd = f"{scratch}/{name}"
    shutil.rmtree(sd, ignore_errors=True)
    os.makedirs(sd)
    shutil.copy(f"{render_dir}/render/config.yaml", f"{sd}/hybrid.yaml")
    env = dict(os.environ, SRTSLURM_CONFIG=f"{render_dir}/srtslurm.yaml", HOME=f"{G.W}/e2e/home",
               XDG_CACHE_HOME=f"{G.W}/e2e/cache/xdg-login", UV_CACHE_DIR=f"{G.W}/e2e/cache/uv-login")
    for k in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_NODELIST", "SLURM_JOB_NODELIST"):
        env.pop(k, None)
    p = subprocess.run([SRTCTL, "render", "--to", f"{sd}/render", "-f", f"{sd}/hybrid.yaml"], cwd=SRT_ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    open(f"{sd}/render.log", "w").write(p.stdout)
    res = {}
    for f in ("sbatch_script.sh", "render.json", "git_state.txt", "config.yaml"):
        a = f"{sd}/render/{f}"
        if not os.path.exists(a):
            res[f] = None
            continue
        ra = norm_render(open(a).read(), sd, render_dir)
        rb = norm_render(open(f"{render_dir}/render/{f}").read(), sd, render_dir)
        res[f] = ra == rb if ra == rb else [l for l in set(ra.splitlines()) ^ set(rb.splitlines())]
    return p.returncode, res


def check_render_cli(src: str, dst: str, base: str, prefix: str, scratch: str) -> None:
    name = os.path.basename(dst)
    rc_d, res_d = rerender(dst, scratch)
    ok(rc_d == 0, f"{name}: srtctl render (preflight: model path + container) rc={rc_d} ({scratch}/{name}/render.log)")
    rc_s, res_s = rerender(src, scratch)
    ok(rc_s == 0, f"{name}: srtctl render of the frozen source rc={rc_s}")
    for f, got in res_d.items():
        if got is True:
            ok(True, f"{name}: srtctl-rendered {f} == the render's (paths, Generated stamps aside)")
            continue
        # drift the frozen source already had (its render/config.yaml edited after its render): accept exactly that drift
        exp = res_s[f]
        exp_m = sorted(G.rename(x, base, prefix)[0] for x in exp) if isinstance(exp, list) else exp
        ok(isinstance(got, list) and isinstance(exp, list) and sorted(got) == exp_m,
           f"{name}: srtctl-rendered {f} differs from the render's exactly as the frozen source's does "
           f"({len(got) if isinstance(got, list) else got} lines: {sorted(got)[:4] if isinstance(got, list) else ''})")


def check_draft_dir() -> None:
    d = G.RELEASE_HOST
    print(f"\n== release draft {d}")
    ok(hashlib.md5(open(f"{d}/config.json", "rb").read()).hexdigest() == RELEASE_CONFIG_MD5, "config.json md5 = e4e6e626's")
    ok(os.path.getsize(f"{d}/model.safetensors") == RELEASE_SAFETENSORS_BYTES, "model.safetensors size = e4e6e626's LFS size")
    ok(open(f"{d}/model.sha256").read().split()[0] == G.RELEASE_SHA256, "model.sha256 record = e4e6e626's LFS oid")
    c = json.load(open(f"{d}/config.json"))
    t = c["transformer_layer_config"]
    ok(c["speculators_model_type"] == "dspark" and c["architectures"] == ["DSparkDraftModel"], "speculators dspark checkpoint")
    ok(t["layer_types"] == ["sliding_attention"] * 5 and t["sliding_window"] == 2048, "5 sliding-window layers, window 2048")
    ok((t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]) == (96, 16, 64), "GQA 96 q / 16 kv heads, head_dim 64")
    ok(t["max_position_embeddings"] >= 1048576, "max_position_embeddings >= the renders' max-model-len 1048576")
    ok(c["sample_from_anchor"] is True and c["block_size"] == 7 and c["markov_rank"] == 512, "anchor sampling, block 7, Markov rank 512")
    ok(c["aux_hidden_state_layer_ids"] == [76, 80, 84, 88, 92] and c["draft_vocab_size"] == 163840, "aux layers / vocab = preview's")


def check_tree(tree: str) -> bool:
    print(f"\n== vLLM tree {tree}")
    for f in TREE_FILES:
        ok(os.path.isfile(f"{tree}/{f}") and not os.path.islink(f"{tree}/{f}"), f"tree has {f}")
    import glob
    ready = [bool(glob.glob(f"{tree}/{m}")) for m in READY_MARKERS]
    for m, r in zip(READY_MARKERS, ready):
        (print if r else warn)(f"{'PRESENT' if r else 'MISSING'} release-draft code {m}")
    return all(ready)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=",".join(p for p, _, _ in G.POINTS))
    ap.add_argument("--fabrics", default=",".join(G.FABRICS))
    ap.add_argument("--kinds", default=",".join(G.KINDS))
    ap.add_argument("--prefix", default="dsrel")
    ap.add_argument("--draft-attn-backend", default="FLASH_ATTN", choices=G.BACKENDS)
    ap.add_argument("--tree", default=G.FROZEN_TREE, help="the tree gen_dsrel.py was given")
    ap.add_argument("--scratch", default=f"{G.W}/dsrel/cfg/scratch")
    ap.add_argument("--no-render", action="store_true", help="skip the srtctl render step")
    ap.add_argument("--target-weights", default=None, metavar="WEIGHTS_DIR,MODEL_VIEW",
                    help="the --target-weights gen_dsrel.py was given")
    a = ap.parse_args(argv)
    weights = tuple(os.path.normpath(x) for x in a.target_weights.split(",")) if a.target_weights else None
    # refusing stubs first on PATH: nothing this checker runs (srtctl render, RuntimeContext) may call Slurm
    stubs = f"{a.scratch}/slurm-stubs"
    os.makedirs(stubs, exist_ok=True)
    real_scontrol = shutil.which("scontrol")
    for cmd in ("srun", "sbatch", "salloc", "scancel", "scontrol"):
        body = f'echo "check_dsrel: {cmd} is disabled during render checks" >&2\nexit 1\n'
        if cmd == "scontrol" and real_scontrol:  # only the local, read-only host list expansion srtctl uses
            body = f'if [ "$1" = show ] && [ "$2" = hostnames ] && [ $# -eq 3 ]; then exec {real_scontrol} "$@"; fi\n' + body
        with open(f"{stubs}/{cmd}", "w") as fh:
            fh.write("#!/bin/sh\n" + body)
        os.chmod(f"{stubs}/{cmd}", 0o755)
    os.environ["PATH"] = stubs + ":" + os.environ.get("PATH", "")
    pts = {p: b for p, b, _ in G.POINTS}
    rows, trees = [], set()
    check_draft_dir()
    for label in a.points.split(","):
        base = pts[label]
        for fab in a.fabrics.split(","):
            got = {}
            for kind in a.kinds.split(","):
                src = f"{G.RS}/{G.frz_name(base, kind, fab)}"
                dst = f"{G.RS}/{G.dst_name(a.prefix, base, kind, fab)}"
                if not ok(os.path.isdir(dst), f"{dst} exists"):
                    continue
                info = check_render(src, dst, base, kind, a.prefix, a.draft_attn_backend, os.path.normpath(a.tree),
                                    weights)
                trees.add(info["tree"])
                check_mounts(dst, weights)
                if not a.no_render:
                    check_render_cli(src, dst, base, a.prefix, a.scratch)
                got[kind] = (src, dst)
                rows.append((label, fab, kind, info))
            if len(got) == 2:
                check_pair(got["agentx"][0], got["gsm8k"][0], got["agentx"][1], got["gsm8k"][1], base, a.prefix)
    ready = {t: check_tree(t) for t in sorted(x for x in trees if x)}
    print("\n| render | nodes | k | rejection | AL | draft backend | tree | launch-ready |")
    print("|---|---|---|---|---|---|---|---|")
    for label, fab, kind, i in rows:
        s = i["spec"]
        print(f"| {i['name']} | {i['nodes']} | {i['k']} | {s['rejection_sample_method']} | "
              f"{s.get('synthetic_acceptance_length', '-')} | {s.get('attention_backend', 'auto')} | "
              f"{os.path.basename(i['tree'] or '?')} | {'yes' if ready.get(i['tree']) else 'NO (tree lacks release-draft code)'} |")
    print(f"\nCHECK_DSREL {'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s), {len(WARNS)} warning(s), {len(rows)} renders")
    for f in FAILS:
        print("  FAIL " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
