#!/usr/bin/env python3
"""gen_dsrel.py: release-draft renders of the 8 chosen Kimi K3.1 VR200 points (Inferact/Kimi-K3.1-Pro-DSpark @ e4e6e626).

Each chosen point has a frozen AgentX render frz-<base>-<fab> and a gsm8k twin frz-<base>-ev-<fab> per fabric (xdr = IB,
spx = RoCE). This script derives
    dsrel-<base>-<fab>      from  frz-<base>-<fab>        (AgentX: synthetic rejection at the release golden AL)
    dsrel-<base>-ev-<fab>   from  frz-<base>-ev-<fab>     (gsm8k: real (block) rejection, AL dropped)
and changes ONLY:
  1. Draft: the extra_mount  <deps>/kimi-k3.1-pro-dspark-preview:/raid/common-models/kimi-k3.1-pro-dspark-preview:ro
     becomes                 <deps>/kimi-k3.1-pro-dspark-e4e6e626:/raid/common-models/kimi-k3.1-pro-dspark:ro
     and "model" in the speculative-config of the prefill AND the decode role becomes /raid/common-models/kimi-k3.1-pro-dspark.
  2. Draft attention backend: "attention_backend": "TOKENSPEED_MLA" (an MLA backend: it raises for the GQA sliding-window
     draft) becomes --draft-attn-backend (default FLASH_ATTN: 1024-token draft blocks; FLASHINFER forces 64-token
     blocks under BLHNC and fails startup with 128-token prefix matching, DSREL_NOTES.md; "auto" drops the key, then configure_dense_draft picks
     FLASH_ATTN). One value for both roles: P writes the draft KV, NIXL moves it, D reads it, so both use one KV layout.
  3. AgentX renders: synthetic_acceptance_length becomes the release golden AL for the render's k (RELEASE_AL), and
     golden-acceptance.json and AL_FIX.txt record it. gsm8k twins keep real rejection and golden-acceptance null.
  4. Names: every full render name frz-<base>[-ev]-<fab> becomes dsrel-<base>[-ev]-<fab> in every runtime file. That covers
     the job name, the config name, and the render dir in sbatch_script.sh / render.json. sbatch_script.sh copies
     <render dir>/render/*.yaml into the run dir at job start, so a stale name would run the OLD render's config.yaml.
     Provenance records (*.txt, *.log) stay verbatim; DSREL.txt is added.
  5. render/git_state.txt: the draft line of srtctl's extra-mount snapshot (exactly what `srtctl render` writes).
     hybrid.yaml (srtctl's recipe record) gets the same draft edits as render/config.yaml (the file that runs). In most
     frozen renders hybrid.yaml predates later edits of render/config.yaml (Mooncake 0.3.14 wrapper, gsm8k-twin keys), and
     those differences are kept. Any synthetic AL in it becomes the release AL of its k.
  6. --tree (optional): the vLLM tree on the prefill and decode PYTHONPATH. The default keeps the frozen tree, so nothing
     changes. The release draft needs the integrated dsrel code (NIXL draft-KV transfer, the K3 GQA drafter fast path,
     the KV planner): regenerate with --tree <integrated tree> --replace before a launch (README.md).
  7. --k-override POINT=K[,POINT=K] (optional; k-variant renders, prefix dsrelk<K> by default): the draft length K
     instead of the frozen k, and only what K changes (k_variant):
       - "num_speculative_tokens" K in the prefill AND decode speculative-config (one draft length: P drafts the first
         block, D verifies K + 1 rows per request); AgentX: synthetic_acceptance_length = the release golden AL of K
         (gsm8k twins keep real block rejection, no AL);
       - the DECODE role's compilation-config cudagraph_capture_sizes and max-cudagraph-capture-size, re-derived for
         q = K + 1 from the frozen list (derive_capture_sizes): the frozen list holds verify rows B*(k+1) of its verify
         batches and drafter rows B*k of its drafter batches; each batch keeps its rows at the new length (B*(K+1),
         B*K), the cap becomes max-num-seqs*(K+1). Re-deriving at K = k returns the frozen list unchanged (checked);
       - nothing else: the prefill role's capture list does not depend on k (the frozen k7 and k3 points share it; P
         never captures a verify step), and max-num-batched-tokens already holds max-num-seqs x (K+1) verify rows on D
         and the draft pass's max-num-seqs x K query rows on both roles (k_requirements, vllm/config/vllm.py);
       - only points whose decode is TP16 = DCP16 (C512's sequence-parallel decode accepts k3 only: sp_unsupported).
Everything else stays byte-identical: topology, k, CONC, capture sizes, Mooncake, precision, env, JIT/cache dirs and the
AgentX client runtime dir. The script checks this itself: it parses both YAMLs and diffs the flattened trees, and it
re-checks the JSON of both speculative-configs. check_dsrel.py adds the srtctl render/preflight checks on Hecate.

Modes:
  Hecate (default): python3 gen_dsrel.py [--points C1,C64,...] [--fabrics xdr,spx] [--kinds agentx,gsm8k]
                        [--draft-attn-backend FLASHINFER] [--tree DIR] [--prefix dsrel] [--replace] [--stamp ISO]
      reads $RS/frz-*, writes $RS/<prefix>-* (RS = /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders). It
      refuses to overwrite an existing render without --replace. --replace moves the old one to <name>.prev-<stamp>, and it
      refuses if a run of that render is still in progress.
  Mirror: python3 gen_dsrel.py --mirror <chosen_configs_nov_15 dir>
      reads configs/<point>/{agentx,gsm8k} (the frozen renders of the chosen fabric, byte copies of Hecate's) and writes
      release_draft/configs/<point>/{agentx,gsm8k}. These are byte copies of what Hecate mode writes for that fabric with
      the same --stamp, so the repo diff vs ../configs is the exact render change. With --k-override the output goes to
      release_draft/k<K>/configs (--mirror-out DIR elsewhere).
  --renders DIR (testing): read frz-* from and write the renders to DIR instead of RS (e.g. a copy of the frozen renders
      of both fabrics); the runtime files still name RS (they only run from there).
Python 3.9+, standard library only; PyYAML (when importable) adds the parsed-tree self-check.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

W = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr"
RS = f"{W}/e2e/renders"
OUTPUTS = f"{W}/e2e/outputs"
DEPS = f"{W}/jh/noquant/cfg2/deps"
PREVIEW_HOST = f"{DEPS}/kimi-k3.1-pro-dspark-preview"
PREVIEW_MNT = "/raid/common-models/kimi-k3.1-pro-dspark-preview"
RELEASE_HOST = f"{DEPS}/kimi-k3.1-pro-dspark-e4e6e626"
RELEASE_MNT = "/raid/common-models/kimi-k3.1-pro-dspark"
RELEASE_ID = "Inferact/Kimi-K3.1-Pro-DSpark @ e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f"
RELEASE_SHA256 = "83a4f35d3a68fe69275b548a0658f687b5bddaf48f8201f999ef81b538f1fb5f"
FROZEN_TREE = f"{W}/hti/lowlat/trees/ll-integ-dea4ebd06f"
# Target weights of the frozen sweep: the K3.1 test checkpoint behind the Dynamo model view (model_type kimi_k3) that
# srtslurm.yaml maps the model aliases to (mounted at /model); the view symlinks into the checkpoint dir, which the
# configs mount read-only at its own path. --target-weights moves all four references to another checkpoint + view.
TEST_WEIGHTS = "/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1/rainhollow-test-weights-1024_v1"
TEST_VIEW = f"{W}/e2e/model-view/kimi-k3.1-k3type"

# InferenceX golden AL tables (probabilistic draft sampling, block verification, thinking_on, SPEED-Bench coding).
RELEASE_AL = {1: 1.86, 2: 2.56, 3: 3.08, 4: 3.46, 5: 3.81, 6: 4.00, 7: 4.16}
RELEASE_AL_SRC = ("InferenceX 5146af7ab7 inferencex-e2e/infx/golden_al_distribution/"
                  "kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark__probabilistic-block.yaml")
PREVIEW_AL = {1: 1.82, 2: 2.44, 3: 2.88, 4: 3.19, 5: 3.41, 6: 3.57, 7: 3.69, 8: 3.68}  # hti/lowlat/fixal.sh (frozen sweep)

# label (results/chosen_configs_nov_15/configs/<label>), frozen render base (frz-<base>), fabric of the chosen run
POINTS = (
    ("C1", "c1", "xdr"),
    ("C16-1P2D", "c16-1p2d", "xdr"),
    ("C72-1P3D", "c72-1p3d", "spx"),
    ("C48-1P1D", "c48-1p1d", "spx"),
    ("C64", "c64", "spx"),
    ("C128-1P1D", "c128-1p1d", "spx"),
    ("C160-1P1D-llh", "c160-1p1d-llh", "xdr"),
    ("C512", "c512", "spx"),
)
FABRICS = ("xdr", "spx")
KINDS = ("agentx", "gsm8k")
# Valid for the release draft's layers (GQA, head_dim 64, sliding window 2048, BF16 KV): AttentionBackendEnum names, or
# "auto" = drop the key (configure_dense_draft -> FLASH_ATTN, which is FA2 via PTX on sm_107 in the frozen tree).
BACKENDS = ("FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "auto")
PROVENANCE_SUFFIXES = (".txt", ".log")  # provenance records: kept verbatim (never read by a run)
SPEC_RE = re.compile(r"^( +)speculative-config: '(\{[^']*\})'$", re.M)
PYPATH_RE = re.compile(r"^( +PYTHONPATH: )(.*)$", re.M)
SPEC_CHANGED = ("model", "attention_backend", "synthetic_acceptance_length")
# --k-override (change 7): a role's args (roles.<role>.args.<name>, six spaces in srtctl's layout) and the decode keys a
# k-variant render re-derives
ARG_RE = {
    name: re.compile(rf"^(      {re.escape(name)}: )(.*)$", re.M)
    for name in ("max-cudagraph-capture-size", "compilation-config", "max-num-seqs", "max-num-batched-tokens",
                 "tensor-parallel-size", "decode-context-parallel-size")
}
K_CAPTURE_KEYS = (".roles.decode.args.compilation-config", ".roles.decode.args.max-cudagraph-capture-size")


class GenError(RuntimeError):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise GenError(msg)


def frz_name(base: str, kind: str, fab: str) -> str:
    return f"frz-{base}-ev-{fab}" if kind == "gsm8k" else f"frz-{base}-{fab}"


def dst_name(prefix: str, base: str, kind: str, fab: str) -> str:
    return f"{prefix}-{base}-ev-{fab}" if kind == "gsm8k" else f"{prefix}-{base}-{fab}"


def full_name_re(base: str) -> re.Pattern:
    """Full frozen render names of one point (both kinds, both fabrics); never a prefix of another point's name."""
    return re.compile(r"(?<![\w-])frz-" + re.escape(base) + r"(-ev)?-(xdr|spx)(?![\w-])")


def rename(text: str, base: str, prefix: str) -> tuple[str, int]:
    return full_name_re(base).subn(lambda m: f"{prefix}-{base}{m.group(1) or ''}-{m.group(2)}", text)


def rates(al: float, k: int) -> list[float]:
    """vllm/config/speculative.py SpeculativeConfig._acceptance_length_to_rates (minimum-variance schedule)."""
    n = al - 1
    full = int(n)
    return ([1.0] * full + [round(n - full, 6)] + [0.0] * (k - full - 1))[:k]


def role_span(text: str, role: str) -> tuple[int, int]:
    """Character span of the body of roles.<role> in a render YAML (srtctl's layout: 'roles:' at column 0, each role at
    two spaces, its keys deeper; the body ends at the next non-blank line indented by at most two spaces)."""
    top = re.search(r"^roles:\n", text, re.M)
    need(top is not None, "no top-level 'roles:'")
    heads = list(re.finditer(rf"^  {re.escape(role)}:\n", text[top.end():], re.M))
    need(len(heads) == 1, f"roles.{role} found {len(heads)}x")
    start = end = top.end() + heads[0].end()
    for line in text[start:].splitlines(keepends=True):
        if line.strip() and not line.startswith("   "):
            break
        end += len(line)
    return start, end


def role_arg(body: str, name: str) -> str:
    found = ARG_RE[name].findall(body)
    need(len(found) == 1, f"args {name} found {len(found)}x in the role")
    return found[0][1]


def derive_capture_sizes(sizes: list[int], k: int, k_new: int, mns: int) -> tuple[list[int], list[int], list[int]]:
    """(capture list for draft length k_new, verify batches, drafter batches) from a decode capture list for draft length
    k (change 7). Each frozen size is a verify row B*(k+1) of a verify batch, a drafter row B*k of a drafter batch (B <=
    max-num-seqs: runtime/draft.draft_batches captures no larger drafter batch), or both (multiples of k(k+1)); every
    batch keeps its rows at the new length: B*(k_new+1) for the verify batches, B*k_new for the drafter batches. A size
    that is neither is an error. At k_new = k the frozen list comes back unchanged (k_variant checks it)."""
    q = k + 1
    verify = sorted({s // q for s in sizes if s % q == 0})
    draft = sorted({s // k for s in sizes if s % k == 0 and s // k <= mns})
    stray = sorted(s for s in sizes if s % q and not (s % k == 0 and s // k <= mns))
    need(not stray, f"capture sizes {stray} are neither verify rows B*{q} nor drafter rows B*{k} (B <= {mns})")
    return sorted({b * (k_new + 1) for b in verify} | {b * k_new for b in draft}), verify, draft


def captured_graphs(sizes: list[int], k: int, mns: int, cap: int) -> tuple[list[int], list[int]]:
    """(verify batches, drafter batches) of the FULL graphs a decode capture list gives at draft length k: Model Runner
    V2 rounds a size up to whole (k+1)-row verify requests and drops it above max-num-seqs x (k+1) or the cap
    (cudagraph_utils), the drafter up to whole k-row draft blocks, dropped above the cap or max-num-seqs
    (runtime/draft.draft_batches)."""
    q = k + 1
    verify = {-(-s // q) for s in sizes if -(-s // q) * q <= min(cap, mns * q)}
    draft = {-(-s // k) for s in sizes if -(-s // k) * k <= cap and -(-s // k) <= mns}
    return sorted(verify), sorted(draft)


def k_limits(roles: dict[str, tuple[int, int]], k_new: int) -> None:
    """Why nothing but the speculative-configs and the decode capture keys changes with k (change 7). roles: {role:
    (max-num-seqs, max-num-batched-tokens)}. Both roles' max-num-batched-tokens hold the DSpark draft pass of
    max-num-seqs requests x k_new query rows (else vllm/config/vllm.py lowers max_num_active_seqs), the decode's also
    max-num-seqs x (k_new + 1) verify rows (the widest uniform verify batch)."""
    for role, (mns, mnbt) in roles.items():
        need(mnbt > k_new and mnbt // k_new >= mns, f"{role}: max-num-batched-tokens {mnbt} holds the draft pass of "
             f"{mnbt // k_new} requests at K={k_new}, below max-num-seqs {mns}")
        if role == "decode":
            need(mnbt >= mns * (k_new + 1), f"decode max-num-batched-tokens {mnbt} < max-num-seqs {mns} x q{k_new + 1}")


def k_requirements(text: str, k_new: int) -> dict[str, tuple[int, int]]:
    """k_limits on a render YAML's prefill and decode args; returns their {role: (max-num-seqs,
    max-num-batched-tokens)}."""
    roles = {}
    for role in ("prefill", "decode"):
        start, end = role_span(text, role)
        body = text[start:end]
        roles[role] = (int(role_arg(body, "max-num-seqs")), int(role_arg(body, "max-num-batched-tokens")))
    k_limits(roles, k_new)
    return roles


def k_variant(text: str, k: int, k_new: int) -> tuple[str, dict]:
    """Change 7 on one render YAML: the decode role's cudagraph_capture_sizes and max-cudagraph-capture-size re-derived
    for q = k_new + 1 (derive_capture_sizes), after the checks that nothing else changes with k (k_requirements).
    The compilation-config JSON keeps its other keys, their order and its separators."""
    start, end = role_span(text, "decode")
    body = text[start:end]
    tp, dcp = (int(role_arg(body, a)) for a in ("tensor-parallel-size", "decode-context-parallel-size"))
    need((tp, dcp) == (16, 16), f"decode TP{tp} / DCP{dcp}: k overrides are for TP16 = DCP16 decode roles only (C512's "
         "sequence-parallel decode accepts k3 only: sp_unsupported)")
    need(k_new in RELEASE_AL, f"K={k_new} has no release golden AL (k1..k7)")
    mns, cap = int(role_arg(body, "max-num-seqs")), int(role_arg(body, "max-cudagraph-capture-size"))
    raw = role_arg(body, "compilation-config")
    need(len(raw) > 2 and raw[0] == raw[-1] == "'", "decode compilation-config is not a single-quoted JSON string")
    comp = json.loads(raw[1:-1])
    fmt = next((s for s in ((",", ":"), (", ", ": ")) if json.dumps(comp, separators=s) == raw[1:-1]), None)
    need(fmt is not None, "decode compilation-config is not in a layout json.dumps reproduces")
    sizes = comp.get("cudagraph_capture_sizes")
    need(isinstance(sizes, list) and sizes and sizes == sorted(set(sizes)), "decode capture sizes: not a sorted list")
    need(cap == mns * (k + 1) == sizes[-1], f"frozen decode cap {cap}: not max-num-seqs {mns} x q{k + 1} = the largest "
         f"capture size {sizes[-1]}")
    need(derive_capture_sizes(sizes, k, k, mns)[0] == sizes, "the frozen list does not re-derive to itself at its k")
    new, verify, draft = derive_capture_sizes(sizes, k, k_new, mns)
    new_cap = mns * (k_new + 1)
    need(new[-1] == new_cap, f"largest new capture size {new[-1]} != max-num-seqs {mns} x q{k_new + 1}")
    v_old, d_old = captured_graphs(sizes, k, mns, cap)
    v_new, d_new = captured_graphs(new, k_new, mns, new_cap)
    need(set(verify) <= set(v_new) and set(draft) <= set(d_new),
         f"k{k_new} graphs miss verify batches {sorted(set(verify) - set(v_new))} / drafter batches "
         f"{sorted(set(draft) - set(d_new))}")
    need((v_new[-1], d_new[-1]) == (v_old[-1], d_old[-1]), f"largest verify / drafter batch (the capacity anchors) "
         f"{v_old[-1]} / {d_old[-1]} -> {v_new[-1]} / {d_new[-1]}")
    roles = k_requirements(text, k_new)
    new_raw = "'" + json.dumps({**comp, "cudagraph_capture_sizes": new}, separators=fmt) + "'"
    body = ARG_RE["max-cudagraph-capture-size"].sub(lambda m: m.group(1) + str(new_cap), body)
    body = ARG_RE["compilation-config"].sub(lambda m: m.group(1) + new_raw, body)
    return text[:start] + body + text[end:], dict(
        k=k, k_new=k_new, mns=mns, cap=cap, new_cap=new_cap, sizes=sizes, new=new, verify=verify, draft=draft,
        v_old=v_old, d_old=d_old, v_new=v_new, d_new=d_new, roles=roles)


def edit_spec(js: str, kind: str, backend: str, record: bool = False, k_new: int | None = None) -> tuple[str, dict]:
    """record=False: render/config.yaml (what runs), strict per kind. record=True: hybrid.yaml, srtctl's recipe record,
    which in most frozen renders predates later edits of render/config.yaml (gsm8k twins keep the AgentX recipe there,
    some with the old K3 AL): same draft edits, and any synthetic AL becomes the release AL of its k. k_new: the
    --k-override draft length (change 7; the AL follows it)."""
    d = json.loads(js)
    k = d.get("num_speculative_tokens")
    need(d.get("method") == "dspark", f"speculative-config method {d.get('method')!r} != dspark")
    need(d.get("model") == PREVIEW_MNT, f"speculative-config model {d.get('model')!r} != {PREVIEW_MNT}")
    need(d.get("attention_backend") == "TOKENSPEED_MLA", f"attention_backend {d.get('attention_backend')!r}")
    need(d.get("draft_sample_method") == "probabilistic", f"draft_sample_method {d.get('draft_sample_method')!r}")
    need(isinstance(k, int) and k in RELEASE_AL, f"num_speculative_tokens {k!r} has no release golden AL")
    need("synthetic_acceptance_rates" not in d, "speculative-config carries synthetic_acceptance_rates")
    old_al, rej = d.get("synthetic_acceptance_length"), d.get("rejection_sample_method")
    if record:
        need(rej in ("synthetic", "block"), f"recipe record rejection {rej!r}")
        need(((rej == "synthetic") == isinstance(old_al, float)) and (old_al is None or 1 <= old_al <= k + 1),
             f"recipe record rejection {rej!r} with AL {old_al!r}")
    elif kind == "agentx":
        need(rej == "synthetic", f"agentx rejection {rej!r}")
        need(old_al == PREVIEW_AL[k], f"agentx AL {old_al!r} != preview golden {PREVIEW_AL[k]} for k={k}")
    else:
        need(rej == "block", f"gsm8k rejection {rej!r}")
        need(old_al is None, "gsm8k twin carries a synthetic AL")
    d["model"] = RELEASE_MNT
    if backend == "auto":
        del d["attention_backend"]
    else:
        d["attention_backend"] = backend  # same key position
    k_out = k if k_new is None else k_new
    need(k_out in RELEASE_AL, f"K={k_out} has no release golden AL")
    d["num_speculative_tokens"] = k_out  # same key position
    if rej == "synthetic":
        d["synthetic_acceptance_length"] = RELEASE_AL[k_out]
    sep = (",", ":") if '":"' in js or '","' in js else (", ", ": ")
    new = json.dumps(d, separators=sep)
    need("'" not in new, "single quote in speculative-config JSON")
    return new, dict(k=k_out, k_src=k, old_al=old_al, al=d.get("synthetic_acceptance_length"), spec=d,
                     rejection=d["rejection_sample_method"])


def edit_yaml(text: str, kind: str, backend: str, tree: str, record: bool = False,
              weights: tuple[str, str] | None = None, k_new: int | None = None) -> tuple[str, dict]:
    specs = SPEC_RE.findall(text)
    need(len(specs) == 2, f"{len(specs)} speculative-config lines (want prefill + decode)")
    need(specs[0][1] == specs[1][1], "prefill and decode speculative-configs differ in the source")
    info = {}

    def fix(m: re.Match) -> str:
        new, inf = edit_spec(m.group(2), kind, backend, record, k_new)
        info.update(inf)
        return f"{m.group(1)}speculative-config: '{new}'"

    text, n = SPEC_RE.subn(fix, text)
    need(n == 2, f"edited {n} speculative-configs")
    if k_new is not None:  # change 7: the decode capture keys for q = K + 1
        text, info["capture"] = k_variant(text, info["k_src"], k_new)
    old_mount = f"- {PREVIEW_HOST}:{PREVIEW_MNT}:ro"
    need(text.count(old_mount + "\n") == 1, f"draft extra_mount line found {text.count(old_mount)}x")
    text = text.replace(old_mount + "\n", f"- {RELEASE_HOST}:{RELEASE_MNT}:ro\n")
    need(text.count(FROZEN_TREE) == 2, f"frozen tree on {text.count(FROZEN_TREE)} lines (want prefill + decode PYTHONPATH)")
    pp = PYPATH_RE.findall(text)
    need(sum(FROZEN_TREE in v.split(":") for _, v in pp) == 2, "frozen tree not an element of exactly 2 PYTHONPATHs")
    if tree != FROZEN_TREE:
        text, n = PYPATH_RE.subn(
            lambda m: m.group(1) + ":".join(tree if e == FROZEN_TREE else e for e in m.group(2).split(":")), text)
        need(text.count(tree) == 2 and FROZEN_TREE not in text, "tree swap did not hit exactly the 2 role PYTHONPATHs")
    if weights is not None:
        old_w = f"- {TEST_WEIGHTS}:{TEST_WEIGHTS}:ro\n"
        need(text.count(old_w) == 1, f"target weights extra_mount line found {text.count(old_w)}x")
        text = text.replace(old_w, f"- {weights[0]}:{weights[0]}:ro\n")
        need(TEST_WEIGHTS not in text, "test weights still referenced in the config")
    for bad in ("kimi-k3.1-pro-dspark-preview", "TOKENSPEED_MLA"):
        need(bad not in text, f"{bad} still in the config")
    info["tree"] = tree
    return text, info


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{p}[{i}]")
    else:
        yield p, o


def yaml_diff(src_text: str, dst_text: str, base: str, prefix: str) -> list[tuple[str, object, object]] | None:
    """Flattened parsed-tree diff (source names mapped to the new names first); None without PyYAML."""
    try:
        import yaml
    except ImportError:
        return None
    a = dict(flat(yaml.safe_load(rename(src_text, base, prefix)[0])))
    b = dict(flat(yaml.safe_load(dst_text)))
    return [(k, a.get(k, "<absent>"), b.get(k, "<absent>")) for k in sorted(set(a) | set(b))
            if a.get(k, "<absent>") != b.get(k, "<absent>")]


def check_diff(diffs, tree: str, weights: tuple[str, str] | None = None, k_change: bool = False) -> list[str]:
    """The parsed diff may contain only the draft keys (and the role PYTHONPATHs when the tree moves; with k_change, a
    k-variant render's num_speculative_tokens and the decode capture keys, change 7)."""
    ok_keys = {".roles.prefill.args.speculative-config", ".roles.decode.args.speculative-config"}
    if tree != FROZEN_TREE:
        ok_keys |= {".roles.prefill.env.PYTHONPATH", ".roles.decode.env.PYTHONPATH"}
    if k_change:
        ok_keys |= set(K_CAPTURE_KEYS)
    spec_ok = set(SPEC_CHANGED) | ({"num_speculative_tokens"} if k_change else set())
    seen, bad = set(), []
    for k, a, b in diffs:
        if k in ok_keys:
            seen.add(k)
            if k.endswith("speculative-config"):
                ja, jb = json.loads(a), json.loads(b)
                moved = {x for x in set(ja) | set(jb) if ja.get(x) != jb.get(x)}
                if not moved <= spec_ok:
                    bad.append(f"{k}: JSON keys {sorted(moved)} changed")
            elif k.endswith("compilation-config"):
                ja, jb = json.loads(a), json.loads(b)
                moved = {x for x in set(ja) | set(jb) if ja.get(x) != jb.get(x)}
                if moved != {"cudagraph_capture_sizes"}:
                    bad.append(f"{k}: JSON keys {sorted(moved)} changed (want cudagraph_capture_sizes only)")
        elif re.fullmatch(r"\.extra_mount\[\d+\]", k) and (a, b) == (
                f"{PREVIEW_HOST}:{PREVIEW_MNT}:ro", f"{RELEASE_HOST}:{RELEASE_MNT}:ro"):
            seen.add(".extra_mount")
        elif weights is not None and re.fullmatch(r"\.extra_mount\[\d+\]", k) and (a, b) == (
                f"{TEST_WEIGHTS}:{TEST_WEIGHTS}:ro", f"{weights[0]}:{weights[0]}:ro"):
            pass
        else:
            bad.append(f"{k}: {str(a)[:120]!r} -> {str(b)[:120]!r}")
    for k in sorted((ok_keys | {".extra_mount"}) - seen):
        bad.append(f"{k}: expected to change, did not")
    return bad


def in_progress_runs(name: str) -> list[str]:
    """run_in_alloc.sh writes in_alloc.txt at start and appends rc= at the end: no rc= and < 8 h old = live."""
    live = []
    for f in glob.glob(f"{OUTPUTS}/a*-{name}-*/in_alloc.txt"):
        try:
            if "rc=" not in open(f).read() and time.time() - os.path.getmtime(f) < 8 * 3600:
                live.append(os.path.dirname(f))
        except OSError:
            continue
    return live


def derive(src: str, dst: str, *, base: str, prefix: str, kind: str, fab: str, backend: str, tree: str,
           stamp: str, gen_md5: str, replace: bool, hecate: bool, weights: tuple[str, str] | None = None,
           k_new: int | None = None) -> dict:
    need(os.path.isdir(src), f"missing source render {src}")
    sname, dname = frz_name(base, kind, fab), dst_name(prefix, base, kind, fab)
    if os.path.exists(dst):
        need(replace, f"{dst} exists (use --replace)")
        if hecate:
            live = in_progress_runs(dname)
            need(not live, f"{dname} has runs in progress: {live}")
    tmp = f"{dst}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=True)
    try:
        info = _derive_in_place(tmp, src, base=base, prefix=prefix, kind=kind, sname=sname, dname=dname,
                                backend=backend, tree=tree, stamp=stamp, gen_md5=gen_md5, weights=weights,
                                k_new=k_new)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if os.path.exists(dst):
        if hecate:  # keep the replaced render next to the new one
            os.rename(dst, f"{dst}.prev-{stamp.replace(':', '')}")
        else:  # mirror: the repo keeps history
            shutil.rmtree(dst)
    os.rename(tmp, dst)
    info.update(src=src, dst=dst, name=dname)
    return info


def _derive_in_place(d: str, src: str, *, base, prefix, kind, sname, dname, backend, tree, stamp, gen_md5,
                     weights=None, k_new=None) -> dict:
    files = sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)
    need(not any(os.path.islink(os.path.join(d, f)) for f in files), "symlink in the source render")
    for f in ("hybrid.yaml", "render/config.yaml", "render/sbatch_script.sh", "render/render.json",
              "render/git_state.txt", "golden-acceptance.json", "workflow.env.resolved", "point.env"):
        need(f in files, f"source render lacks {f}")
    need(("AL_FIX.txt" in files) == (kind == "agentx"), "AL_FIX.txt presence does not match the render kind")
    need("DSREL.txt" not in files, "source already has DSREL.txt")
    need(sname in open(f"{d}/render/config.yaml").read(), f"source config does not name {sname}")
    renamed = {}
    for f in files:
        if f.endswith(PROVENANCE_SUFFIXES):
            continue
        p = os.path.join(d, f)
        t = open(p).read()
        t2, n = rename(t, base, prefix)
        need(not full_name_re(base).search(t2), f"{f}: frozen name left after rename")
        if n:
            with open(p, "w") as fh:
                fh.write(t2)
            renamed[f] = n
    # the job copies <render dir>/render/*.yaml at start: it must be THIS render's dir
    sb = open(f"{d}/render/sbatch_script.sh").read()
    need(f'cp "{RS}/{dname}/render"/*.yaml "${{OUTPUT_DIR}}/"' in sb, "sbatch_script.sh does not stage its own render dir")
    need(json.load(open(f"{d}/render/render.json"))["script"] == f"{RS}/{dname}/render/sbatch_script.sh",
         "render.json script is not this render's")
    # configs: render/config.yaml is what runs (sbatch_script.sh stages render/*.yaml; run_in_alloc.sh copies it);
    # hybrid.yaml is srtctl's recipe record, which in most frozen renders predates later render/config.yaml edits
    new, info = edit_yaml(open(f"{d}/render/config.yaml").read(), kind, backend, tree, weights=weights, k_new=k_new)
    with open(f"{d}/render/config.yaml", "w") as fh:
        fh.write(new)
    new_hy, info_hy = edit_yaml(open(f"{d}/hybrid.yaml").read(), kind, backend, tree, record=True, weights=weights,
                                k_new=k_new)
    need(info_hy["k"] == info["k"], f"hybrid.yaml k={info_hy['k']} != render/config.yaml k={info['k']}")
    if k_new is not None:
        cap, cap_hy = info["capture"], info_hy["capture"]
        need((cap_hy["new"], cap_hy["new_cap"]) == (cap["new"], cap["new_cap"]),
             "hybrid.yaml re-derives another decode capture list than render/config.yaml")
    with open(f"{d}/hybrid.yaml", "w") as fh:
        fh.write(new_hy)
    # git state snapshot: srtctl lists every extra mount as "  - extra_mount:<target>:ro: <host>"
    gp = f"{d}/render/git_state.txt"
    g = open(gp).read()
    old_l, new_l = f"  - extra_mount:{PREVIEW_MNT}:ro: {PREVIEW_HOST}\n", f"  - extra_mount:{RELEASE_MNT}:ro: {RELEASE_HOST}\n"
    need(g.count(old_l) == 1, "git_state.txt draft mount line not found once")
    g = g.replace(old_l, new_l)
    if weights is not None:
        old_w = f"  - extra_mount:{TEST_WEIGHTS}:ro: {TEST_WEIGHTS}\n"
        need(g.count(old_w) == 1, "git_state.txt target weights mount line not found once")
        g = g.replace(old_w, f"  - extra_mount:{weights[0]}:ro: {weights[0]}\n")
    with open(gp, "w") as fh:
        fh.write(g)
    # srtslurm.yaml: the model aliases and the /model mount name the target's model view
    if weights is not None:
        sp = f"{d}/srtslurm.yaml"
        need(os.path.exists(sp), "render lacks srtslurm.yaml")
        t = open(sp).read()
        need(t.count(TEST_VIEW) == 3, f"srtslurm.yaml names the test model view {t.count(TEST_VIEW)}x (want 2 aliases + mount)")
        with open(sp, "w") as fh:
            fh.write(t.replace(TEST_VIEW, weights[1]))
    # AL records
    k, k_src, al = info["k"], info["k_src"], info["al"]
    ga = json.load(open(f"{d}/golden-acceptance.json"))
    if kind == "agentx":
        need(ga == {"golden_acceptance_length": PREVIEW_AL[k_src]}, f"source golden-acceptance.json {ga}")
        with open(f"{d}/golden-acceptance.json", "w") as fh:
            fh.write(f'{{"golden_acceptance_length": {al}}}\n')
        with open(f"{d}/AL_FIX.txt", "w") as fh:
            fh.write(f"{stamp} AL -> {al} (k={k}, InferenceX K3.1 RELEASE golden table: {RELEASE_AL_SRC}; "
                     f"was {info['old_al']} = preview table"
                     + (f" at k={k_src}; k {k_src} -> {k} by --k-override" if k != k_src else "") + ")\n")
    else:
        need(ga == {"golden_acceptance_length": None}, f"gsm8k source golden-acceptance.json {ga}")
    # parsed self-check of both YAMLs against their own sources
    diffs = yaml_diff(open(f"{src}/render/config.yaml").read(), new, base, prefix)
    diffs_hy = yaml_diff(open(f"{src}/hybrid.yaml").read(), new_hy, base, prefix)
    for f, dd in (("render/config.yaml", diffs), ("hybrid.yaml", diffs_hy)):
        bad = check_diff(dd, tree, weights, k != k_src) if dd is not None else []
        need(not bad, f"{f}: unexpected parsed differences: " + "; ".join(bad))
    k_line = ""
    if k_new is not None:
        c = info["capture"]
        by_v, by_d = sorted(set(c["v_new"]) - set(c["verify"])), sorted(set(c["d_new"]) - set(c["draft"]))
        lost_v = sorted(set(c["v_old"]) - set(c["v_new"]))
        lost_d = sorted(set(c["d_old"]) - set(c["d_new"]))
        k_line = (
            f"  k override (--k-override, change 7): num_speculative_tokens {k_src} -> {k} on prefill + decode"
            + (f", synthetic AL -> {al} (release golden of k{k})" if kind == "agentx" else ", real (block) rejection")
            + f"; decode max-cudagraph-capture-size {c['cap']} -> {c['new_cap']} (max-num-seqs {c['mns']} x q{k + 1}), "
            f"cudagraph_capture_sizes {len(c['sizes'])} -> {len(c['new'])} sizes: the frozen list's verify batches "
            f"({len(c['verify'])}) x {k + 1} and drafter batches ({len(c['draft'])}) x {k}\n"
            f"    FULL graphs at k{k}: {len(c['v_new'])} verify batches (k{k_src}: {len(c['v_old'])}), "
            f"{len(c['d_new'])} drafter batches (k{k_src}: {len(c['d_old'])}); largest {c['v_new'][-1]} / "
            f"{c['d_new'][-1]} (unchanged); by-products of the other rounding {by_v} / {by_d}; the frozen list's own "
            f"by-products not re-made {lost_v} / {lost_d} (they pad to the next graph)\n"
            f"    unchanged with k: prefill / decode max-num-seqs and max-num-batched-tokens "
            + ", ".join(f"{r} {v[0]} / {v[1]}" for r, v in c["roles"].items())
            + f" (both hold the draft pass's max-num-seqs x {k} rows, decode its {c['mns']} x {k + 1} verify rows); "
            "the prefill capture list (k-independent)\n")
    rt = re.findall(r"^AIPERF_RUNTIME_DIR=(.*)$", open(f"{d}/workflow.env.resolved").read(), re.M)
    bare = sorted({f for f in files if not f.endswith(PROVENANCE_SUFFIXES)
                   and f"frz-{base}" in open(os.path.join(d, f)).read()})
    spec_line = json.dumps(info["spec"])
    with open(f"{d}/DSREL.txt", "w") as fh:
        fh.write(
            f"{dname} = {sname} + release draft (gen_dsrel.py md5 {gen_md5}, {stamp})\n"
            f"  source: {RS}/{sname} (frozen sweep render; its provenance: FRZ.txt, INTEG.txt)\n"
            f"  draft: {RELEASE_ID} (model.safetensors sha256 {RELEASE_SHA256})\n"
            f"         extra_mount {RELEASE_HOST} -> {RELEASE_MNT} (ro); was {PREVIEW_HOST} -> {PREVIEW_MNT}\n"
            f"  speculative-config (prefill = decode): model -> {RELEASE_MNT}; attention_backend TOKENSPEED_MLA -> "
            f"{'(dropped: auto = FLASH_ATTN)' if backend == 'auto' else backend}"
            + (f"; synthetic_acceptance_length {info['old_al']} -> {al} (k={k}, release golden, rates "
               f"{rates(al, k)})" if kind == "agentx" else "; real (block) rejection, no synthetic AL") + "\n"
            f"  now: {spec_line}\n"
            + k_line
            + f"  hybrid.yaml (srtctl's recipe record; render/config.yaml is what runs): the same draft edits; its "
            f"speculative-config is {info_hy['rejection']}"
            + (f", AL {info_hy['old_al']} -> {info_hy['al']}" if info_hy["rejection"] == "synthetic" else "") + "\n"
            + (f"  target weights: {weights[0]} (model view {weights[1]} at /model); was {TEST_WEIGHTS} "
               f"(view {TEST_VIEW})\n" if weights is not None else "")
            + f"  tree (prefill + decode PYTHONPATH): {tree}"
            + (" (frozen tree, unchanged: no release-draft support; regenerate with --tree <integrated tree> before a "
               "launch)" if tree == FROZEN_TREE else f" (was {FROZEN_TREE})") + "\n"
            f"  renamed {sname} -> {dname} in: "
            + ", ".join(f"{f} x{n}" for f, n in sorted(renamed.items())) + "\n"
            f"  kept: every other key and file (parsed-YAML diff vs the source = the keys above"
            f"{'' if diffs is not None else ' [PyYAML missing: not self-checked]'}); AIPERF_RUNTIME_DIR "
            f"{rt[0] if rt else '-'} (the source's AgentX client venv, shared)"
            + (f"; bare source base frz-{base} still named in {bare} (that runtime dir only)" if bare else "") + "\n")
    info.update(renamed=renamed, diffs=len(diffs) if diffs is not None else None,
                diffs_hy=len(diffs_hy) if diffs_hy is not None else None, bare=bare)
    return info


def summary(info: dict) -> str:
    d = info["dst"]
    nodes = json.load(open(f"{d}/render/render.json"))["total_nodes"]
    s = info["spec"]
    return (f"{info['name']:<30} nodes={nodes:<2} k={info['k']} {s['rejection_sample_method']:<9} "
            f"AL={str(s.get('synthetic_acceptance_length', '-')):<5} backend={s.get('attention_backend', 'auto(FLASH_ATTN)'):<11} "
            f"draft={s['model']} tree={os.path.basename(info['tree'])} parsed-diffs config={info['diffs']} "
            f"hybrid={info['diffs_hy']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default=",".join(p for p, _, _ in POINTS), help="labels, comma-separated")
    ap.add_argument("--fabrics", default=",".join(FABRICS))
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--draft-attn-backend", default="FLASH_ATTN", choices=BACKENDS)
    ap.add_argument("--tree", default=FROZEN_TREE, help="vLLM tree for the prefill + decode PYTHONPATH")
    ap.add_argument("--prefix", default="dsrel")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--stamp", default=None, help="ISO time recorded in DSREL.txt / AL_FIX.txt (default: now)")
    ap.add_argument("--mirror", default=None, help="chosen_configs_nov_15 dir: write release_draft/configs from configs/")
    ap.add_argument("--target-weights", default=None, metavar="WEIGHTS_DIR,MODEL_VIEW",
                    help="serve another target checkpoint: its dir (mounted read-only at its path) and its Dynamo "
                         "model view (srtslurm.yaml model aliases + /model mount); default: the frozen sweep's test "
                         "weights")
    ap.add_argument("--k-override", default=None, metavar="POINT=K[,POINT=K]",
                    help="k-variant renders (change 7) of these points; --points defaults to them and the prefix to "
                         "dsrelk<K>")
    ap.add_argument("--mirror-out", default=None,
                    help="mirror mode: output dir (default release_draft/configs, release_draft/k<K>/configs with "
                         "--k-override)")
    ap.add_argument("--renders", default=RS, help="testing: the dir holding the frz-* sources and the outputs")
    a = ap.parse_args(argv)
    stamp = a.stamp or datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    gen_md5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    pts = {p: (b, f) for p, b, f in POINTS}
    overrides = parse_k_override(a.k_override, pts)
    if overrides and a.points == ap.get_default("points"):
        a.points = ",".join(overrides)
    if overrides and a.prefix == ap.get_default("prefix"):
        need(len(set(overrides.values())) == 1, "--k-override with several K: name the renders with --prefix")
        a.prefix = f"dsrelk{next(iter(overrides.values()))}"
    labels = [x for x in a.points.split(",") if x]
    for x in labels:
        need(x in pts, f"unknown point {x} (known: {', '.join(pts)})")
        need(not overrides or x in overrides, f"point {x} has no k override: a k-variant render set holds only the "
             "overridden points")
    need(re.fullmatch(r"[a-z][a-z0-9]*", a.prefix) is not None and a.prefix != "frz", f"bad prefix {a.prefix!r}")
    need(not overrides or a.prefix != "dsrel", "k-variant renders need their own prefix (dsrel is the frozen-k set)")
    ks = sorted(set(overrides.values()))
    mirror_out = a.mirror_out or (f"release_draft/k{ks[0]}/configs" if overrides else "release_draft/configs")
    need(not (a.mirror and len(ks) > 1 and a.mirror_out is None), "--k-override with several K: give --mirror-out")
    renders = os.path.normpath(a.renders)
    tree = os.path.normpath(a.tree)
    weights = None
    if a.target_weights:
        parts = [os.path.normpath(x) for x in a.target_weights.split(",")]
        need(len(parts) == 2, "--target-weights takes WEIGHTS_DIR,MODEL_VIEW")
        weights = (parts[0], parts[1])
        if a.mirror is None:  # on Hecate: both must exist (the view's config.json is what the engine reads)
            need(os.path.isfile(os.path.join(weights[0], "config.json")), f"{weights[0]} has no config.json")
            need(os.path.isfile(os.path.join(weights[1], "config.json")), f"{weights[1]} has no config.json")
    if a.mirror is None:
        for f in ("vllm/__init__.py", "vllm/v1/attention/backends/registry.py"):
            need(os.path.isfile(f"{tree}/{f}"), f"--tree {tree} has no {f}")
        reg_path = f"{tree}/vllm/v1/attention/backends/registry.py"
    else:  # the checkout this script lives in (results/chosen_configs_nov_15/release_draft -> repo root)
        reg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../vllm/v1/attention/backends/registry.py")
    if a.draft_attn_backend != "auto" and os.path.isfile(reg_path):
        need(re.search(rf"^    {a.draft_attn_backend} = ", open(reg_path).read(), re.M) is not None,
             f"{a.draft_attn_backend} is not an AttentionBackendEnum member in {reg_path}")
    jobs = []
    for label in labels:
        base, chosen = pts[label]
        for kind in [x for x in a.kinds.split(",") if x]:
            need(kind in KINDS, f"bad kind {kind}")
            if a.mirror:
                src = f"{a.mirror}/configs/{label}/{kind}"
                dst = f"{a.mirror}/{mirror_out}/{label}/{kind}"
                jobs.append((src, dst, base, kind, chosen, overrides.get(label)))
            else:
                for fab in [x for x in a.fabrics.split(",") if x]:
                    need(fab in FABRICS, f"bad fabric {fab}")
                    jobs.append((f"{renders}/{frz_name(base, kind, fab)}",
                                 f"{renders}/{dst_name(a.prefix, base, kind, fab)}", base, kind, fab,
                                 overrides.get(label)))
    rc = 0
    for src, dst, base, kind, fab, k_new in jobs:
        try:
            if a.mirror:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            info = derive(src, dst, base=base, prefix=a.prefix, kind=kind, fab=fab, backend=a.draft_attn_backend,
                          tree=tree, stamp=stamp, gen_md5=gen_md5, replace=a.replace, hecate=a.mirror is None,
                          weights=weights, k_new=k_new)
            print("made " + summary(info))
        except GenError as e:
            print(f"FAIL {dst}: {e}")
            rc = 1
    return rc


def parse_k_override(spec: str | None, points: dict) -> dict[str, int]:
    """--k-override POINT=K[,POINT=K] -> {label: K} (change 7)."""
    out: dict[str, int] = {}
    for item in filter(None, (spec or "").split(",")):
        label, sep, k = item.partition("=")
        need(sep == "=" and k.isdigit(), f"--k-override item {item!r} is not POINT=K")
        need(label in points, f"--k-override: unknown point {label} (known: {', '.join(points)})")
        need(label not in out, f"--k-override names {label} twice")
        need(int(k) in RELEASE_AL, f"--k-override {item}: K must be one of {sorted(RELEASE_AL)} (release golden AL)")
        out[label] = int(k)
    return out


if __name__ == "__main__":
    sys.exit(main())
