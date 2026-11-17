# Release draft renders for the 8 chosen points (Kimi K3.1 VR200)

These are the configs that run the released draft **Inferact/Kimi-K3.1-Pro-DSpark @ e4e6e626** at the 8 chosen points
(`../README.md`). They are rendered and validated, and nothing has been launched. Each point gets an AgentX render and a
gsm8k twin for both fabrics: 32 renders on Hecate under `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders/`.
`configs/` is a byte-identical repo mirror of the fabric each chosen run used, so `diff -r ../configs/<point> configs/<point>`
is the exact change.

**Launch readiness.** The renders still point at the frozen vLLM tree (`hti/lowlat/trees/ll-integ-dea4ebd06f`). That tree
cannot serve the release draft:
- its NIXL DCP pair exchange rejects the GQA draft's KV;
- it has no GQA drafter fast path;
- its sliding-window draft group turns off 128-token prefix hits.

The other dsrel tracks fix these. Before a launch, regenerate the renders on the integrated tree, as in
[Launching later](#launching-later). `check_dsrel.py` reports `launch-ready: NO` until the tree contains the release-draft
code.

## Renders

| point | AgentX (synthetic AL) | gsm8k twin (real block rejection) | nodes | decode | k | release AL | chosen run's fabric |
|---|---|---|---|---|---|---|---|
| C1 | `dsrel-c1-{xdr,spx}` | `dsrel-c1-ev-{xdr,spx}` | 6 | TP16+DCP16 | 7 | 4.16 | xdr |
| C16 1P2D | `dsrel-c16-1p2d-{xdr,spx}` | `dsrel-c16-1p2d-ev-{xdr,spx}` | 10 | 2 x TP16+DCP16 | 7 | 4.16 | xdr |
| C72 1P3D | `dsrel-c72-1p3d-{xdr,spx}` | `dsrel-c72-1p3d-ev-{xdr,spx}` | 14 | 3 x TP16+DCP16 | 7 | 4.16 | spx |
| C48 1P1D | `dsrel-c48-1p1d-{xdr,spx}` | `dsrel-c48-1p1d-ev-{xdr,spx}` | 6 | TP16+DCP16 | 7 | 4.16 | spx |
| C64 | `dsrel-c64-{xdr,spx}` | `dsrel-c64-ev-{xdr,spx}` | 6 | TP16+DCP16 | 7 | 4.16 | spx |
| C128 1P1D | `dsrel-c128-1p1d-{xdr,spx}` | `dsrel-c128-1p1d-ev-{xdr,spx}` | 6 | TP16+DCP16 | 3 | 3.08 | spx |
| C160 1P1D llh | `dsrel-c160-1p1d-llh-{xdr,spx}` | `dsrel-c160-1p1d-llh-ev-{xdr,spx}` | 6 | TP16+DCP16 | 3 | 3.08 | xdr |
| C512 | `dsrel-c512-{xdr,spx}` | `dsrel-c512-ev-{xdr,spx}` | 8 | DP4 x (TP4+DCP4), EP16 | 3 | 3.08 | spx |

The prefill role is unchanged: TP8+DCP8, or C512's DP4 x (TP4+DCP4). Pick the render whose fabric matches the
allocation's partition. xdr is IB (batch-xdr) and spx is RoCE (batch-spx); P/D split across the two fabrics transfers
nothing.

## What changed vs the frozen render (and nothing else)

`gen_dsrel.py` derives `dsrel-<base>[-ev]-<fab>` from `frz-<base>[-ev]-<fab>` and changes only the following:

1. **Draft mount.**
   - The extra_mount `<deps>/kimi-k3.1-pro-dspark-preview:/raid/common-models/kimi-k3.1-pro-dspark-preview:ro` becomes
     `<deps>/kimi-k3.1-pro-dspark-e4e6e626:/raid/common-models/kimi-k3.1-pro-dspark:ro`, where
     `<deps>` = `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/jh/noquant/cfg2/deps`.
   - The release dir holds the e4e6e626 checkpoint: model.safetensors sha256 83a4f35d...1fb5f, 4,959,959,810 B.
   - `"model"` in the speculative-config of the prefill **and** decode roles becomes `/raid/common-models/kimi-k3.1-pro-dspark`.
2. **Draft attention backend.**
   - `"attention_backend": "TOKENSPEED_MLA"` becomes the `--draft-attn-backend` knob, default **FLASHINFER**.
   - One value covers both roles: P writes the draft KV, NIXL moves it and D reads it, so both must use one layout.
   - TOKENSPEED_MLA is MLA-only. vLLM rejects it for the draft layers (head_size, kv_cache_dtype, non-MLA, sliding
     window; see `checks/vllm_check-20261116.out`), so the frozen value cannot start.
   - `auto` drops the key; configure_dense_draft then picks FLASH_ATTN, which is FA2 via PTX on sm_107.
   - The draft KV dtype stays the stock `auto` (BF16).
3. **Synthetic AL** (AgentX renders only). `synthetic_acceptance_length` becomes the release golden AL for the render's k:
   4.16 at k7 and 3.08 at k3, from InferenceX 5146af7ab7
   `golden_al_distribution/kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark__probabilistic-block.yaml`
   (k1..k7 = 1.86 2.56 3.08 3.46 3.81 4.00 4.16). The preview values were 3.69 and 2.88.
   - vLLM resolves these to the per-position rates [1, 1, 1, 0.16, 0, 0, 0] (k7) and [1, 1, 0.08] (k3).
   - `golden-acceptance.json` and a fresh `AL_FIX.txt` record the value.
   - gsm8k twins keep real (block) rejection, no AL, and `golden-acceptance.json` = null.
4. **Names.** Every full frozen render name `frz-<base>[-ev]-<fab>` becomes `dsrel-...` in every runtime file. That covers
   the job name, the config `name`, the sbatch script's `Config:` / `cp <render dir>/render/*.yaml` lines and
   `render.json`. Provenance records (`FRZ.txt`, `INTEG.txt`, `render.log`) stay verbatim, and `DSREL.txt` records the
   derivation. One exception: `AIPERF_RUNTIME_DIR` of C16 1P2D and C72 1P3D still points at the frozen sweep's
   `agentx-runtime/frz-c16-1p2d` / `frz-c72-1p3d` client venv, the same shared dependency as before. C128 and C160 llh
   share `nq4-c160`, as they did in the frozen sweep.
5. **`render/git_state.txt`.** The draft line of srtctl's extra-mount snapshot, exactly what `srtctl render` now writes.
6. **`hybrid.yaml`** (srtctl's recipe record). It gets the same draft edits. In most frozen renders it predates later
   `render/config.yaml` edits (the Mooncake 0.3.14 wrapper, the gsm8k-twin keys). The file that runs is
   `render/config.yaml`.

Everything else is byte-identical to the frozen render: topology, k, CONC, max-num-seqs, capture sizes, Mooncake / NIXL,
`dcp_pair_exchange`, precision, env, JIT/cache dirs, tree, image and identity.

Exact diff for C1 (AgentX, `diff -r ../configs/C1/agentx configs/C1/agentx`, AL_FIX.txt and DSREL.txt aside):
```
hybrid.yaml / render/config.yaml (same 4 lines in each)
1c1
< name: coreai_comparch_inferencex-k31e2e.frz-c1-xdr
> name: coreai_comparch_inferencex-k31e2e.dsrel-c1-xdr
161c161 (prefill) and 205c205 (decode)
<       speculative-config: '{"method": "dspark", "model": "/raid/common-models/kimi-k3.1-pro-dspark-preview", "num_speculative_tokens": 7, "attention_backend": "TOKENSPEED_MLA", "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.69}'
>       speculative-config: '{"method": "dspark", "model": "/raid/common-models/kimi-k3.1-pro-dspark", "num_speculative_tokens": 7, "attention_backend": "FLASHINFER", "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 4.16}'
216c216
< - /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-preview:/raid/common-models/kimi-k3.1-pro-dspark-preview:ro
> - /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626:/raid/common-models/kimi-k3.1-pro-dspark:ro
golden-acceptance.json: {"golden_acceptance_length": 3.69} -> 4.16
render/git_state.txt:7  extra_mount:/raid/common-models/kimi-k3.1-pro-dspark-preview:ro: <deps>/kimi-k3.1-pro-dspark-preview
                     -> extra_mount:/raid/common-models/kimi-k3.1-pro-dspark:ro: <deps>/kimi-k3.1-pro-dspark-e4e6e626
render/render.json:3,12  name / script: frz-c1-xdr -> dsrel-c1-xdr
render/sbatch_script.sh:4,41,55,68  job name, Config path, cp "<renders>/dsrel-c1-xdr/render"/*.yaml, echo Config
```

## Validation (GPU-free, nothing submitted)

- **`gen_dsrel.py` self-check.** For every render, the parsed YAML of both `render/config.yaml` and `hybrid.yaml` differs
  from its source in exactly 3 keys: the 2 speculative-configs and the draft extra_mount. Log: `checks/gen-20261116T1900.out`.
- **`check_dsrel.py`** (Hecate, srtctl venv): `CHECK_DSREL PASS: 0 failure(s), 2 warning(s), 32 renders`, 1,663 checks
  (`checks/check-20261116T1900.out`). It checks:
  - byte-level file checks against the renamed source;
  - the parsed-YAML diff;
  - speculative-config semantics on prefill = decode;
  - that the AgentX/gsm8k pair differs exactly as the frozen pair does (only synthetic -> block, AL dropped);
  - srtctl `load_config` + `RuntimeContext` mounts: release draft at `/raid/common-models/kimi-k3.1-pro-dspark` ro, no
    preview mount, `/model`, k31vr, `/raid/scratch`, nothing over `/opt/k3`;
  - **`srtctl render`** of every render's `render/config.yaml` with preflight (model path + container). The re-rendered
    `sbatch_script.sh`, `render.json`, `git_state.txt` and `config.yaml` equal the render's files except for paths and
    Generated stamps;
  - the release draft dir (config md5, safetensors size, sha256 record, the fields the configs rely on).

  The 2 warnings say the frozen tree lacks the release-draft code. The checker refuses any Slurm call: it unsets
  `SLURM_JOB_ID` and puts refusing `srun` / `sbatch` / `scancel` stubs first on PATH. An earlier run of it had let
  srtctl attempt `srun --jobid 1` IP lookups, which failed immediately; `checks/` has the clean rerun.
- **`vllm_check.py`** (production vLLM tree, CPU-only on the dsrel dev node via `gpurun.sh cfg ""`): `VLLM_CHECK PASS`.
  - vLLM parses all 64 speculative-configs (32 renders x prefill/decode) and resolves the synthetic AL to rates.
  - The checkpoint loads through vLLM's config path as `Qwen3DSparkModel`: 5 x sliding window 2048, 96 q / 16 kv heads x 64,
    aux layers 76..92, Markov rank 512, anchor sampling. Heads per rank: 6/1 at TP16, 12/2 at TP8, 24/4 at TP4.
  - **FLASHINFER** passes `validate_configuration` for the draft layers on DeviceCapability(10, 7): head 64, BF16 KV,
    block 1024, sliding window, KV connector, DCP 1. FLASH_ATTN and TRITON_ATTN also pass. The frozen TOKENSPEED_MLA fails.
- **Repo mirror** (`configs/`, from `gen_dsrel.py --mirror ..`): its 232 files are md5-identical to the Hecate renders of
  the chosen fabric, both written with `--stamp 2026-11-16T19:00:00-08:00`.

## Launching later

1. Stage the integrated code: the tracks' changes merged on `wzhao/k31-dspark-release`.
   - Run `mkov.sh <checkout> <tag>` to get `$W/dsrel/ov/<tag>`, then freeze a snapshot that nothing refreshes in place:
     `cp -al $W/dsrel/ov/<tag> $W/dsrel/trees/<tag>`.
   - A refresh of `ov/<tag>` writes new inodes, so a hard-linked snapshot keeps the old files.
2. On the Hecate login node, from `$W/dsrel/cfg`:
   - `python3 tools/gen_dsrel.py --tree $W/dsrel/trees/<tag> --replace`. It moves each old render to
     `<name>.prev-<stamp>` and refuses while a run of that render is in progress.
   - Then `.../srtctl-venv/bin/python tools/check_dsrel.py --tree $W/dsrel/trees/<tag>`. It must print
     `CHECK_DSREL PASS` with launch-ready yes.
   - Optionally run `gpurun.sh <tag> "" python3 tools/vllm_check.py` with the new tree.
   - JIT/cache dirs stay the frozen ones. They are content-keyed: unchanged kernels hit and new drafter kernels compile
     on first start, so budget the first start or pre-warm the CuTe cache.
3. Run each point as in the frozen sweep (`../README.md`, Reproduce a point):
   - `run_in_alloc.sh <render>` inside an allocation of `render.json`'s `total_nodes` nodes on the matching fabric, then
     the gsm8k twin on the same allocation.
   - Startup checks:
     - `Resolved architecture: Qwen3DSparkModel` (or the Kimi subclass that track M adds);
     - the draft layers on `FLASHINFER` with a 1024-token kernel block. trtllm-gen needs flashinfer-cubin, which the image
       has; 64 means no trtllm-gen and eager draft graphs;
     - `Capturing dspark CUDA graphs (FULL)`;
     - the specialized-decode drafter coverage lines;
     - NIXL successful transfers > 0 and no handshake failures in the first minutes;
     - no `Disabling fine-grained prefix-cache hits`.

## k-choice note (recommendation, not applied)

**Question.** Which draft length k maximizes throughput at each point, given the release AL curve and the drafter cost?

**Constraints.**
- On TP16=DCP16 decode, k can be at most 7.
- C512's sequence-parallel decode accepts **only k3** (`sp_unsupported`: DSpark q4, `SP_MAX_ROWS` 512 = 128 x 4), so
  C512 stays k3.

**Inputs.** `kchoice.py` (output in `checks/kchoice.out`) is calibrated on the 15 TP16 frozen-sweep runs. For each k it
needs the verify step at q = k + 1 rows per request, evaluated at the batch while decoding. It builds that step from three
parts:
- **Target term.** Each run's measured server step, minus its preview drafter, fitted as
  `a + b E_d(qB) + c qB + d B`.
- **q-sensitivity.** The fit's dependence on q is scaled by kappa = 0.755. kappa comes from the one clean measured k
  contrast: the C96 1P2D A/B, k7 a1093473 vs k3 a1097000. A second estimate from the server step curves at B 25.5 gives
  kappa 0.751.
- **Drafter term.** The release drafter cost with the fast path, 0.19-0.38 ms.

The closed-loop model (xpyd form, per-point eta, chi, tau) then gives x and y at the point's concurrency.

**Break-even.** At equal B, k beats k0 only if step(k)/step(k0) < AL(k)/AL(k0). The release ratios against k7 are
k6 0.962, k5 0.916, k4 0.832 and k3 0.740. Against k3 they are k4 1.123, k5 1.237, k6 1.299 and k7 1.351. The release
curve gains more at high k than the preview (k5/k3 1.237 vs 1.184), and the release drafter costs nearly the same at
every k because its MLA attention is gone. Both favour a longer draft at high batch.

**Projected change of interactivity x vs the frozen k at the same concurrency.** Values are calibrated, fast-path drafter;
brackets give the uncalibrated kappa 1 bound.

| point | B | frozen k | projected release at frozen k vs measured preview | k3 | k4 | k5 | k6 | k7 | recommendation |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.24 | 7 | x +15%, y +3% | -20% | -12% | -5% | -2% | 0 | **k7** |
| C16 1P2D | 0.55 | 7 | x +15%, y +1% | -20% | -12% | -5% | -2% | 0 | **k7** |
| C72 1P3D | 2.8 | 7 | x +23%, y +2% | -19% | -11% | -3% | -1% | 0 | **k7** |
| C48 1P1D | 6.7 | 7 | x +27%, y +2% | -14% | -6% | +0.5% [+5%] | +0.7% [+3%] | 0 | **k7** (k5/k6 within model error) |
| C64 | 12.3 | 7 | x +30%, y +3% | -11% | -4% | +2.2% [+8%] | +1.4% [+4%] | 0 | **k7**; k5 optional A/B |
| C128 1P1D | 27.6 | 3 | x +17%, y +2% | 0 | +6.7% [+0.7%] | **+12.2%** [+1.3%] | +10.3% [-4%] | +7.5% [-10%] | **k5** (k4 lower-risk) |
| C160 1P1D llh | 37.7 | 3 | x +19%, y +3% | 0 | +6.3% [-0.2%] | **+11.3%** [-0.9%] | +8.3% [-8%] | +4.3% [-15%] | **k5** (k4 lower-risk) |
| C512 | 51 per DP rank | 3 | AL 2.88 -> 3.08 | k3 only (SP decode) | - | - | - | - | **k3** |

**Reading the table.**
- At a fixed concurrency, y moves about 5x less than x: AgentX users spend about 16 s per cycle idle against 3-6 s in
  decode. For k5 at C128 / C160 that is +1.6% y at the same C.
- The x gain can be traded for throughput by raising C. The front's slope between neighbouring chosen points is
  d ln y / d ln x = -1.2 to -1.6.
- With the stock release drafter the choice is the same: C128 k5 +12.6%, C160 k5 +11.6%; k7 stays best at C1-C72.

**Validation and error.**
- Projecting the C96 A/B from one arm to the other: x within 3.4-3.7% with kappa 0.755, and 8.7-9.1% with kappa 1.
- The residual is mostly the p90/mean ITL tail, held at the source run's value: 1.215 at k7 vs 1.189 at k3.
- Differences below about 3% are therefore noise. That is why C48 and C64 stay at k7.
- The high-batch k5 gain depends on kappa. The uncalibrated bound makes k3, k4 and k5 a wash at C128 / C160.

**Recommendation.**
- **C1, C16 1P2D, C72 1P3D, C48 1P1D, C64:** keep k7.
- **C128 1P1D and C160 1P1D llh:** move from k3 to **k5**, projected +11-12% x (k4: +6-7%). First measure the decode step
  with the release drafter at q6 vs q4 for B 24-40, same context vectors: switch if step(q6)/step(q4) <= 1.237 (k4:
  q5/q4 <= 1.123).
- **C512:** stays k3.

**Applying a k change to a render.** `gen_dsrel.py --k-override C128-1P1D=5,C160-1P1D-llh=5` writes the k5 renders
`dsrelk5-*` (and `check_dsrel.py --k-override` checks them). It changes:
- prefill and decode speculative-config: `num_speculative_tokens` 5 and synthetic AL 3.81 (gsm8k twin: k only);
- decode `max-cudagraph-capture-size` (k+1) x max-num-seqs = 960;
- the decode `cudagraph_capture_sizes`, re-derived for the q6 verify rows and the k5 drafter rows.

The release drafter's specialized decode must cover those shapes: NQ-NOFALLBACK refuses uncovered shapes at capture.
`k5/README.md` has the derivation, the checks, the q6 coverage audit and the measured k5 vs k3 steps.
`capture_shapes.json` lists every role's current shapes for those coverage tests.

## Interface for the other tracks

- **Render paths.** `$RS/dsrel-<base>[-ev]-<fab>` for base in `c1`, `c16-1p2d`, `c72-1p3d`, `c48-1p1d`, `c64`, `c128-1p1d`,
  `c160-1p1d-llh`, `c512` and fab in `xdr`, `spx`.
- **Draft in the container.** `/raid/common-models/kimi-k3.1-pro-dspark`, read-only. The host path is
  `<deps>/kimi-k3.1-pro-dspark-e4e6e626`. Bench and test configs should use the same paths.
- **speculative-config, identical on prefill and decode.**
  `{"method": "dspark", "model": "/raid/common-models/kimi-k3.1-pro-dspark", "num_speculative_tokens": K, "attention_backend": "FLASHINFER", "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "synthetic_acceptance_length": AL}`
  with (K, AL) = (7, 4.16) for C1/C16/C72/C48/C64 and (3, 3.08) for C128/C160/C512. gsm8k twins use
  `"rejection_sample_method": "block"` and no AL.
  - The draft KV that P writes (stock path) and D reads therefore uses **FLASHINFER's layout**, BF16, with the draft at
    DCP 1 and heads split by TP. Tracks K (attention kernel) and N (NIXL head mapping) must match it.
  - Changing the knob changes that contract: `gen_dsrel.py --draft-attn-backend {FLASHINFER,FLASH_ATTN,TRITON_ATTN,auto}`.
- **Shapes to cover.** `capture_shapes.json` has, per point and role: TP/DCP/DP/EP, k, q, max-num-seqs, capture sizes,
  `dcp_pair_exchange` (on for the 7 TP16 points) and decode_backend.
- **Code tree.** `gen_dsrel.py --tree <tree>` swaps the frozen tree on both roles' PYTHONPATH. `check_dsrel.py --tree <tree>`
  checks the result and reports launch readiness from these markers: `vllm/models/kimi_k3/nvidia/dspark_gqa.py` and
  `.../cutedsl/draft/gqa_*.py`.

## Files

- `gen_dsrel.py`: the generator. Hecate mode writes the 32 renders; `--mirror <chosen_configs_nov_15>` writes
  `configs/`. Knobs: `--draft-attn-backend`, `--tree`, `--target-weights`, `--k-override` (k-variant renders,
  `k5/`), `--prefix`, `--points/--fabrics/--kinds`, `--replace`, `--stamp`, `--mirror-out`, `--renders` (testing).
- `k5/`: the k5 renders of C128 1P1D / C160 1P1D llh (`k5/README.md`): repo mirror, checks, the q6 coverage audit
  (`coverage_k5.py`) and the k5 vs k3 step measurements.
- `check_dsrel.py`: the GPU-free render checks (srtctl venv on the Hecate login node).
- `vllm_check.py`: the vLLM-level speculative-config, checkpoint and draft-backend check (image, CPU-only).
- `kchoice.py`: the k-choice model (standard library); `checks/kchoice.out` holds its output.
- `shapes_from_renders.py` writes `capture_shapes.json`.
- `configs/<point>/{agentx,gsm8k}`: the repo mirror of the chosen-fabric renders.
- `checks/`: generation, validation and model logs.
