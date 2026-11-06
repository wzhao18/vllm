"""Pure-Python checks of the nq bench harness (no torch / GPU): python3 test_nq_harness.py

* cfg2bench.build on a synthetic decode role: arg classification, capture extension, arms, env rewrites
* nq_step.patch_specialized_decode with stand-in vllm modules: the real functions see the production
  compilation fields only for the duration of the call, the stock arm is untouched
* nq_step.apply_overlay on a temp package tree: vllm / kimi_k3 resolve from the whole-tree overlay
* nq_shapes.py / nq_summary.py / nq_junit.py end to end on synthetic files
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def doc(rows_cap=512, mns=128, k=3):
    sizes = [6, 8, 15, 16, 18, 20, 21, 24, 30, 32, 64, 128, 256, 384, 512]
    return {
        "environment": {"VLLM_USE_V2_MODEL_RUNNER": "1", "UCX_TLS": "rc", "NCCL_NVLS_ENABLE": "1",
                        "CUTE_DSL_CACHE_DIR": "/raid/scratch/x/cute", "VLLM_SERVER_DEV_MODE": "1"},
        "roles": {"decode": {
            "env": {"VLLM_K31_DCP_EXCHANGE": "1", "K31_DCPX_BUILD_DIR": "/raid/scratch/x/dcpx",
                    "PYTHONPATH": "/x", "K3_PIN_CPUS": "1", "VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS": "512",
                    "VLLM_K31_DSA_PARTIALS": "proj"},
            "args": {
                "tensor-parallel-size": 16, "decode-context-parallel-size": 16, "kv-cache-dtype": "fp8",
                "mamba-ssm-cache-dtype": "float32", "trust-remote-code": True, "served-model-name": "m",
                "load-format": "fastsafetensors", "max-num-seqs": mns, "max-cudagraph-capture-size": rows_cap,
                "compilation-config": json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE",
                                                  "cudagraph_capture_sizes": sizes}),
                "speculative-config": json.dumps({"method": "dspark", "model": "/raid/common-models/kimi-k3-dspark",
                                                  "num_speculative_tokens": k}),
                "additional-config": '{"decode_backend":"latency_specialized"}',
                "kv-transfer-config": "{}", "attention-config": '{"indexer_kv_dtype":"mxfp4"}',
            }}},
    }


def test_cfg2bench():
    import cfg2bench

    ls = cfg2bench.build(doc(), "ls", "new", [8, 16, 1024])
    st = cfg2bench.build(doc(), "stock", "new", [8, 16, 1024])
    assert ls["q"] == 4 and ls["mns"] == 256 and ls["max_capture"] == 1024
    # measured rows and their drafter rows (B*K) are captured: 16 -> 12, 1024 -> 768
    for s in (8, 6, 16, 12, 1024, 768):
        assert s in ls["capture_sizes"], s
    a = ls["argv"]
    assert a[a.index("--additional-config") + 1] == '{"decode_backend": "latency_specialized"}'
    b = st["argv"]
    assert b[b.index("--additional-config") + 1] == '{"decode_backend": "auto"}'
    # both arms identical apart from decode_backend
    strip = lambda v: [x for x in v if "decode_backend" not in x]
    assert strip(a) == strip(b) and ls["env"] == st["env"]
    spec = json.loads(a[a.index("--speculative-config") + 1])
    assert spec["model"] == cfg2bench.DRAFTERS["new"]
    assert "--trust-remote-code" in a and "--served-model-name" not in a and "--kv-transfer-config" not in a
    assert a[a.index("--load-format") + 1] == "dummy"
    env = ls["env"]
    assert env["K31_DCPX_BUILD_DIR"] == "${NQ_BUILD}/dcpx" and "UCX_TLS" not in env
    assert "PYTHONPATH" not in env and "K3_PIN_CPUS" not in env and "VLLM_SERVER_DEV_MODE" not in env
    assert "CUTE_DSL_CACHE_DIR" not in env
    # removed env switches (automatic per-shape policy now) are dropped, with the reason
    assert "VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS" not in env and "VLLM_K31_DSA_PARTIALS" not in env
    assert "removed" in ls["dropped_env"]["VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS"]
    assert "removed" in ls["dropped_env"]["VLLM_K31_DSA_PARTIALS"]
    # every dropped key is listed with a reason (nothing silently lost)
    assert "UCX_TLS" in ls["dropped_env"] and "CUTE_DSL_CACHE_DIR" in ls["dropped_env"]
    assert ls["production_limit"] == 512 and any("NOT PRODUCTION" in n for n in ls["notes"])
    # inside the production limit nothing grows
    small = cfg2bench.build(doc(), "ls", "old", [8, 64, 512])
    assert small["mns"] == 128 and small["max_capture"] == 512 and small["production_limit"] == 512
    assert not any("NOT PRODUCTION" in n for n in small["notes"])
    # the production limit is min(max-num-seqs * q, capture cap): rows above mns*q but <= the cap still grow mns
    tight = cfg2bench.build(doc(mns=64), "ls", "old", [8, 512])
    assert tight["production_limit"] == 256 and tight["mns"] == 128 and tight["max_capture"] == 512
    # --split: rows <= the limit, rows above it
    assert cfg2bench.split_rows(doc(), [8, 512, 768, 1024]) == ([8, 512], [768, 1024])
    assert cfg2bench.split_rows(doc(mns=64), [8, 256, 512]) == ([8, 256], [512])
    # an unclassified production arg is an error, not silently dropped
    d = doc()
    d["roles"]["decode"]["args"]["new-knob"] = 1
    try:
        cfg2bench.build(d, "ls", "old", [])
    except SystemExit as e:
        assert "new-knob" in str(e)
    else:
        raise AssertionError("unclassified arg accepted")
    sh = cfg2bench.as_sh(ls, "x")
    assert "export K31_DCPX_BUILD_DIR=${NQ_BUILD}/dcpx" in sh and "NQ_CAP=" in sh and "NQ_PRODLIMIT=512" in sh
    assert "# dropped env UCX_TLS:" in sh


def fake_vllm(tmp):
    """vllm / vllm.config / vllm.models.kimi_k3... stand-ins (real package dirs for the overlay test)."""
    for mod in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[mod]
    base = os.path.join(tmp, "base")
    files = {
        "vllm/__init__.py": "",
        "vllm/config.py": "import enum\nclass CUDAGraphMode(enum.Enum):\n    NONE = 0\n    FULL_AND_PIECEWISE = 3\n",
        "vllm/models/__init__.py": "",
        "vllm/models/kimi_k3/__init__.py": "WHERE = 'base'\n",
        "vllm/models/kimi_k3/nvidia/__init__.py": "",
        "vllm/models/kimi_k3/nvidia/model.py":
            "from .specialized_decode.runtime import specialized_decode_config\n",
        "vllm/models/kimi_k3/nvidia/specialized_decode/__init__.py": "",
        "vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py": (
            "from vllm.config import CUDAGraphMode\n"
            "SEEN = []\n"
            "def specialized_decode_config(c):\n"
            "    extra = c.additional_config\n"
            "    if extra.get('decode_backend', 'auto') == 'auto':\n"
            "        return None\n"
            "    cc = c.compilation_config\n"
            "    SEEN.append((cc.cudagraph_mode, list(cc.cudagraph_capture_sizes), cc.max_cudagraph_capture_size))\n"
            "    assert cc.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE\n"
            "    from types import SimpleNamespace as N\n"
            "    q = 4\n"
            "    limit = min(c.max_num_seqs * q, cc.max_cudagraph_capture_size)\n"
            "    rows = tuple(sorted({-(-s // q) * q for s in cc.cudagraph_capture_sizes} - set()))\n"
            "    return N(q=q, dcp=16, bf16_state=False, captured=tuple(r for r in rows if r <= limit))\n"
        ),
        "vllm/models/kimi_k3/nvidia/specialized_decode/runtime/draft.py": (
            "def draft_batches(c, steps):\n"
            "    cc = c.compilation_config\n"
            "    sizes = cc.cudagraph_capture_sizes or ()\n"
            "    cap = cc.max_cudagraph_capture_size or max(sizes, default=0)\n"
            "    rounded = {-(-s // steps) * steps for s in sizes}\n"
            "    return sorted(r // steps for r in rounded if r <= cap)\n"
        ),
    }
    for rel, text in files.items():
        p = os.path.join(base, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    sys.path.insert(0, base)
    return base


def test_patch_and_overlay():
    import nq_step

    with tempfile.TemporaryDirectory() as tmp:
        base = fake_vllm(tmp)
        # a kimi_k3-only tree (the old overlay format) is refused
        old_ov = os.path.join(tmp, "old")
        shutil.copytree(os.path.join(base, "vllm/models/kimi_k3"), os.path.join(old_ov, "vllm/models/kimi_k3"))
        try:
            nq_step.apply_overlay(old_ov)
        except RuntimeError as e:
            assert "whole vllm tree" in str(e)
        else:
            raise AssertionError("kimi_k3-only overlay accepted")
        # overlay: the whole base tree with changed files over it (stage_overlay.sh), first on sys.path
        ov = os.path.join(tmp, "ov")
        shutil.copytree(os.path.join(base, "vllm"), os.path.join(ov, "vllm"))
        open(os.path.join(ov, "vllm/models/kimi_k3/__init__.py"), "w").write("WHERE = 'overlay'\n")
        open(os.path.join(ov, "SOURCE_REV"), "w").write("abc wzhao/x + 2 files\nM vllm/envs.py\nM vllm/models/kimi_k3/__init__.py\n"
                                                        "T tests/x.py\n")
        # vllm imported from the base: refused (overlay not first on the path)
        try:
            nq_step.apply_overlay(ov)
        except RuntimeError as e:
            assert "overlay not applied" in str(e)
        else:
            raise AssertionError("base vllm accepted as the overlay")
        for mod in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
            del sys.modules[mod]
        sys.path.insert(0, ov)
        nq_step.apply_overlay(ov)
        import vllm.models.kimi_k3 as k3

        assert k3.WHERE == "overlay", k3.__file__
        import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
        from vllm.config import CUDAGraphMode

        assert runtime.__file__.startswith(ov)
        sizes = (6, 8, 15, 16, 768, 1024)
        nq_step.patch_specialized_decode(sizes, 1024)
        import vllm.models.kimi_k3.nvidia.model as model
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

        def config(backend):
            cc = types.SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[],
                                       max_cudagraph_capture_size=0)
            return types.SimpleNamespace(additional_config={"decode_backend": backend},
                                         compilation_config=cc, max_num_seqs=256)

        c = config("latency_specialized")
        out = model.specialized_decode_config(c)
        assert out.captured == (8, 16, 768, 1024), out.captured
        assert runtime.SEEN[-1] == (CUDAGraphMode.FULL_AND_PIECEWISE, list(sizes), 1024)
        # restored after the call
        cc = c.compilation_config
        assert (cc.cudagraph_mode, cc.cudagraph_capture_sizes, cc.max_cudagraph_capture_size) == (
            CUDAGraphMode.NONE, [], 0)
        assert draft.draft_batches(c, 3) == [2, 3, 5, 6, 256]  # 1024 -> 1026 > cap
        assert c.compilation_config.cudagraph_capture_sizes == []
        # stock arm: untouched (the real function returns None before reading anything)
        s = config("auto")
        n = len(runtime.SEEN)
        assert model.specialized_decode_config(s) is None and len(runtime.SEEN) == n
        assert draft.draft_batches(s, 3) == []
        sys.path.remove(base)
        sys.path.remove(ov)
    for mod in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[mod]


def test_shapes_and_summary():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.json")
        subprocess.check_call([sys.executable, os.path.join(HERE, "nq_shapes.py"), "8", out, "--rows", "64,8",
                               "--ctx", "agentx,u192k", "--point", "c1"], stdout=subprocess.DEVNULL)
        cases = json.load(open(out))["cases"]
        assert [c["name"] for c in cases] == ["q8_B1_M8_agentx", "q8_B8_M64_agentx", "q8_B1_M8_u192k",
                                              "q8_B8_M64_u192k"]
        assert all(r["num_draft_tokens"] == 7 and r["query_len"] == 8 for c in cases for r in c["requests"])
        cov = ("(EngineCore pid=1) INFO [decode.py:533] Kimi K3 decode coverage M{m}(q8): "
               "embed=tuned(ring) kda=generic(splitk + kdasplit 8x8) dsa=stock(splitk + S2-DSA)\n")
        for i, arm, med in ((1, "ls", (5000.0, 11000.0)), (2, "stock", (6000.0, 10000.0))):
            with open(os.path.join(tmp, f"c64-new-agentx-{i}{arm}.log"), "w") as f:
                f.write(f"NQRUN point=c64 arm={arm} drafter=new q=8 ctx=agentx rows=8,64 extra= ov=x\n")
                if arm == "ls":
                    f.write(cov.format(m=8) + cov.format(m=64))
                    f.write("INFO Kimi K3 DSpark drafter: fused probabilistic Markov sampling for draft batches "
                            "(1, 2, 8)\n")
                    # runtime/draft.py's current format (mask annotation between)
                    f.write("INFO Kimi K3 DSpark drafter: Lamport DCP attention (non-causal) for (7, 56) rows\n")
                    # context rows are B*(K+1): B=1 -> 8; B=8 must not be flagged by its B value
                    f.write("INFO Kimi K3 DSpark drafter: sharded context step for (8,) rows\n")
                for m, us in zip((8, 64), med):
                    f.write(f"[forward] q8_B{m // 8}_M{m}_agentx: {m // 8} reqs, {m} tokens, median {us} us, "
                            "sample spread 0.01%, graph vs eager rel rms 1.4e+00\n")
                f.write("NQEND rc=0\n")
        text = subprocess.check_output([sys.executable, os.path.join(HERE, "nq_summary.py"), tmp,
                                        "--csv", os.path.join(tmp, "o.csv")], text=True)
        assert "0.833" in text and "1.100 WORSE" in text, text
        assert "markov,dcp,ctx" in text and "M8: dsa=stock(splitk + S2-DSA)" in text, text
        m64 = next(x for x in text.splitlines() if x.strip().startswith("M64 "))
        assert m64.endswith("| markov,dcp"), m64  # B=8: no ctx false positive (its context row is 64)
        assert "MISSING (requested, not measured): none" in text and "UNCOVERED (no ls coverage line): none" in text
        # an 'hi' process (rows above the production limit) whose ls arm crashed after M512: MISSING, UNCOVERED, XCAP
        for i, arm in ((1, "ls"), (2, "stock")):
            with open(os.path.join(tmp, f"c64-new-agentx-hi-{i}{arm}.log"), "w") as f:
                f.write(f"NQRUN point=c64 arm={arm} drafter=new q=8 ctx=agentx rows=512,1024 prodlimit=256 "
                        "maxcap=1024 mns=128 extra= ov=x\n")
                if arm == "ls":
                    f.write(cov.format(m=512))
                for m in ((512,) if arm == "ls" else (512, 1024)):
                    f.write(f"[forward] q8_B{m // 8}_M{m}_agentx: x, median 20000.0 us, sample spread 0.01%\n")
                f.write("NQEND rc=1\n" if arm == "ls" else "NQEND rc=0\n")
        # a different line sharing the tag prefix must not join this point's summary
        open(os.path.join(tmp, "c64-new-agentx-x_exclude_drafter-1ls.log"), "w").write("NQRUN point=c64 q=8\n")
        text = subprocess.check_output([sys.executable, os.path.join(HERE, "nq_summary.py"), tmp,
                                        "--tag", "c64-new-agentx"], text=True)
        assert "exclude_drafter" not in text, text
        m1024 = next(x for x in text.splitlines() if x.strip().startswith("M1024 "))
        assert "MISSING(ls)" in m1024 and "UNCOVERED" in m1024 and "XCAP" in m1024, m1024
        assert "XCAP: rows above the production limit 256" in text, text


def test_junit_summary():
    with tempfile.TemporaryDirectory() as tmp:
        g4, g2 = os.path.join(tmp, "gpu4"), os.path.join(tmp, "gpu2")
        os.makedirs(g4)
        os.makedirs(g2)
        open(os.path.join(g4, "status.txt"), "w").write(
            "NQTEST plan pytest tests/models/kimi_k3/test_a.py\n"
            "NQTEST plan pytest tests/models/kimi_k3/test_b.py\n"
            "NQTEST plan pytest tests/models/kimi_k3/test_c.py\n"
            "NQTEST plan pytest tests/models/kimi_k3/test_d.py\n"
            "NQTEST plan torchrun tests/models/kimi_k3/test_a.py\n"
            "NQTEST pytest tests/models/kimi_k3/test_a.py rc=1 secs=10\n"
            "NQTEST pytest tests/models/kimi_k3/test_b.py rc=0 secs=5\n"
            "NQTEST pytest tests/models/kimi_k3/test_c.py rc=124 secs=1800\n"
            "NQTEST torchrun tests/models/kimi_k3/test_a.py rc=0 np=4 secs=60\n"
            "NQTEST stage gpu4 rc=124\n")
        open(os.path.join(g4, "test_a.xml"), "w").write(
            '<testsuites><testsuite><testcase classname="t" name="ok"/>'
            '<testcase classname="t" name="bad"><failure message="mismatch 3 rows">x</failure></testcase>'
            '<testcase classname="t" name="sk"><skipped message="needs Rubin"/></testcase></testsuite></testsuites>')
        open(os.path.join(g4, "test_b.xml"), "w").write(
            '<testsuites><testsuite><testcase classname="t" name="s1"><skipped message="no CUDA"/></testcase>'
            '</testsuite></testsuites>')
        open(os.path.join(g2, "status.txt"), "w").write("NQTEST none - rc=NONE secs=0 (none)\n")
        text = subprocess.check_output([sys.executable, os.path.join(HERE, "nq_junit.py"), g4, g2,
                                        os.path.join(tmp, "missing")], text=True)
        assert "1 passed, 1 failed, 0 errors, 1 skipped" in text and "failing t::bad: mismatch 3 rows" in text, text
        assert "ALL-SKIPPED" in text and "TIMEOUT" in text and "NO-JUNIT" in text, text
        assert "UNCOVERED (no test files)" in text and "missing: NOT RUN" in text, text
        # a planned file the stage timeout cut off before it started: NOT RUN, and the stage TIMEOUT
        assert "pytest tests/models/kimi_k3/test_d.py: NOT RUN" in text and "gpu4: stage TIMEOUT" in text, text
        assert "torchrun   tests/models/kimi_k3/test_a.py rc=0" in text, text
        assert text.strip().splitlines()[-1].startswith("GPUTEST FLAGGED"), text


def test_ab_split_and_dead_marker():
    """nq_ab.sh with stub cfg2bench / nq_run1: rows split at the production limit into two processes per arm; an arm
    that dies at startup skips only its later runs of the same process tag (not other lines / parts)."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = os.path.join(tmp, "tools")
        os.makedirs(tools)
        for f in ("nq_ab.sh", "nq_summary.py"):
            shutil.copy(os.path.join(HERE, f), tools)
        open(os.path.join(tools, "cfg2bench.py"), "w").write(
            "import sys\nrows = [int(x) for x in sys.argv[sys.argv.index('--rows') + 1].split(',')]\n"
            "lo = [r for r in rows if r <= 256]; hi = [r for r in rows if r > 256]\n"
            "print(','.join(map(str, lo)) or '-', ','.join(map(str, hi)) or '-')\n")
        # stub run: every ls process of the 'lo' part dies at startup; everything else measures its rows
        open(os.path.join(tools, "nq_run1.sh"), "w").write(
            'OUT=$3; ARM=$5; ROWS=$7; echo "$(basename $OUT) $ROWS" >> $(dirname $OUT)/RUNS\n'
            'echo "NQRUN point=$4 arm=$ARM drafter=$6 q=8 ctx=$8 rows=$ROWS prodlimit=256" > $OUT.log\n'
            'if [ $ARM = ls ] && [ "${ROWS%%,*}" -le 256 ]; then echo "NQEND rc=1" >> $OUT.log; exit 1; fi\n'
            'for m in ${ROWS//,/ }; do echo "[forward] q8_B$((m / 8))_M${m}_$8: x, median 100.0 us" >> $OUT.log; done\n'
            'echo "NQEND rc=0" >> $OUT.log\n')
        r = os.path.join(tmp, "R")
        ab = os.path.join(tools, "nq_ab.sh")
        subprocess.check_call(["bash", ab, "1", "h1,h2,h3,h4", r, "c64", "new", "8,64,512,1024", "ls,stock,stock,ls",
                               "agentx"], stdout=subprocess.DEVNULL)
        subprocess.check_call(["bash", ab, "1", "h1,h2,h3,h4", r, "c64", "new", "8,64", "ls,stock", "agentx", "--",
                               "--exclude-drafter"], stdout=subprocess.DEVNULL)
        runs = open(os.path.join(r, "RUNS")).read().split("\n")
        assert runs[:7] == [
            "c64-new-agentx-1ls 8,64", "c64-new-agentx-2stock 8,64", "c64-new-agentx-3stock 8,64",
            # (4ls skipped: the lo ls arm died at startup)
            "c64-new-agentx-hi-1ls 512,1024", "c64-new-agentx-hi-2stock 512,1024", "c64-new-agentx-hi-3stock 512,1024",
            "c64-new-agentx-hi-4ls 512,1024"], runs
        # the other line's ls arm still runs (its own tag)
        assert runs[7:9] == ["c64-new-agentx-xexclude_drafter-1ls 8,64",
                             "c64-new-agentx-xexclude_drafter-2stock 8,64"], runs
        assert os.path.exists(os.path.join(r, "DEAD-c64-new-agentx-ls"))
        assert not os.path.exists(os.path.join(r, "DEAD-c64-new-agentx-hi-ls"))
        text = open(os.path.join(r, "c64-new-agentx.summary.txt")).read()
        assert "MISSING(ls)" in text and "c64-new-agentx-hi" in text and "exclude" not in text, text


def test_ctx_cap_and_case_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.json")
        args = [sys.executable, os.path.join(HERE, "nq_shapes.py"), "4", out, "--rows", "8,1024"]
        subprocess.check_call(args, stdout=subprocess.DEVNULL)
        full = json.load(open(out))["cases"]
        txt = subprocess.check_output(args + ["--max-kv-tokens", "30000000"], text=True)
        capped = json.load(open(out))["cases"]
        assert "ctx q4_B256_M1024_agentx: B 256" in txt, txt
        # B=2 is below the cap and unchanged; B=256 is scaled to fit, with the distribution's shape kept
        assert capped[0] == full[0] and full[0]["ctx"]["scale"] == 1.0
        big = capped[1]
        assert big["ctx"]["scale"] < 1 and big["ctx"]["sum"] <= 30000000
        assert sum(r["kv_len"] - r["query_len"] for r in big["requests"]) == big["ctx"]["sum"]
        assert big["ctx"]["max"] < full[1]["ctx"]["max"] and big["ctx"]["min"] >= 8
    # case_blocks mirrors build_batch's cursor: 1 + every group's live blocks per request
    import nq_step

    fake = types.ModuleType("vllm.benchmarks.forward.batch")
    fake._num_blocks = lambda spec, computed, kv_len, dcp: (0, 4 if spec == "mamba" else -(-kv_len // (1024 * dcp)))
    saved = {m: sys.modules.get(m) for m in ("vllm", "vllm.benchmarks", "vllm.benchmarks.forward",
                                             "vllm.benchmarks.forward.batch")}
    for m in saved:
        sys.modules[m] = fake if m.endswith("batch") else types.ModuleType(m)
    try:
        req = types.SimpleNamespace(prefix_len=40000, kv_len=40004)
        case = types.SimpleNamespace(requests=(req, req))
        groups = [types.SimpleNamespace(kv_cache_spec=s) for s in ("mla", "mla", "mamba")]
        assert nq_step.case_blocks(case, groups, 16, 4) == 1 + 2 * (3 + 3 + 4)
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
