# Vendoring record: megamoe_rubin

## Upstream

- **Source**: FlashInfer `0.7.3.dev20261031`,
  `flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources` (41 files): the
  upstream Rubin CuTe DSL MegaMoE export, `cutedsl_megamoe` commit
  `1667b47a3c911ecade464ab524baf192a0bf5962`, plus the FlashInfer-local device
  patches recorded in that drop's VENDOR.md ("Local differences": PR #6447 and the
  `fc12_scheduler.py` phase-interleave claim fix). These patches are part of the bytes
  below; a re-export from upstream must re-apply them.
- **Entry points**: `RubinInferenceMegaMoE` (`BlockScaledSwapAbMegaMoeKernel`) and
  `RubinInferenceGenphaseMegaMoE` (`BlockScaledSwapAbGenphaseMoeKernel`).
- **Check**: apart from the files listed under "Local differences" below, every file
  is byte-identical to the installed `fi-0.7.3.dev20261031` tree on Hecate
  (`$W/env/fi-0.7.3.dev20261031`, `W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr`).
  The CPU unit test `tests/models/kimi_k3/test_megamoe_rubin_conversion.py` recomputes
  every per-file hash below.

## Local differences

- **S1 (with S0)** in `sources/communication/nvlink_domain/token_comm.py`
  (FlashInfer 0.7.3.dev20261031 sha256
  `4731d31ad1871b24e6a22c1d6fad3bee33794a54ae0ec192632dc13d6f8d4134`, local sha256 in
  the manifest below). Source: FlashInfer commit `3a47d4d53` on github
  `wzhao18/flashinfer`, branch `wzhao/kimi-k3-megamoe-sm107`; copied byte for byte.
  - S0: token_in claims tokens dynamically (`token_in_dynamic_claim`) instead of static
    striping when `max_tokens_per_rank * topk <= 4096`.
  - S1: decode-gated metadata-with-sizes. When `max_tokens_per_rank * topk <= 4096`,
    `world_size <= 32` and one router data CTA (trace-time `decode_landing`, no env
    var), the router is one CTA: it publishes sizes, its expert-major source prefix
    and every route's `TokenSrcMetadata` (and score) straight into the destination's
    landing rows in the sizes round, so token_in waits on `sizes_ready` only (no sort,
    no push CTAs, no `metadata_ready`; scheduler poll 200 instead of 1000 cycles).
    Otherwise (prefill) the stock router, regions and expert walks are traced
    verbatim.
  - Outputs are bit-identical to stock (the GEMMs, SiTU and the top-k reduction order
    are unchanged); measured -9..-12 us per launch at decode buckets on VR.
  - The kernel file (`block_scaled_swap_ab_mega_moe_kernel.py`) is unchanged; it uses
    only `token_comm` entry points whose signatures S1 keeps.

## Rewrite rule

**R0 (applied): identity.** Every file is byte-identical to the FlashInfer original
(except the "Local differences" above, which are whole-file copies, not rewrites). All
internal imports of the export are relative (the deepest, `.....` from
`kernel_src/rubin/inference/mega/*`, resolves to `sources`); the only absolute imports
are `cutlass*`, `cuda.bindings.driver`, `packaging.version` and the standard library.
The package is imported as `vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources`
instead of FlashInfer's top-level `sources` on a bootstrapped `sys.path` (FlashInfer
`shim/_paths.py` is not needed). `src/__init__.py` is ours (empty) and is not part of
the manifest.

**R1 (reserved for future drops):** an export containing an absolute `sources.` import
must be rewritten by
`^(\s*)from sources(\.|\s)` -> `\1from vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources\2`
(and `import sources` likewise), with every rewritten file listed here with its original
and rewritten sha256. No `flashinfer` / `nvshmem` import may remain in `src/` or in the
runtime modules.

Do not hand-edit, format or lint-fix `src/`. The repository's ruff / format / mypy hooks
need an exclude for `vllm/models/kimi_k3/nvidia/megamoe_rubin/src/` (a stock
`pyproject.toml` / `.pre-commit-config.yaml` change, left to the integration PR).

## Environment variables read inside `src/` (diagnostic only)

- `MEGA_USE_NCU` (`token_comm.py`, genphase kernel): NCU replay mode, changes the reset
  ownership and the generated kernel name.
- `MEGA_F4CVT_USE_MANUAL` (`topk_reduce.py`): FP4 conversion path of the quantized
  combine (not used with BF16 combine).

This backend never sets them; they must stay unset in serving.
`tests/models/kimi_k3/test_k31_no_env_switches.py` excludes `src/` (vendored) from its
scan; the backend's own modules stay covered.

## sha256 manifest of `src/sources`

```
d4c180fe55b9a054811c189f118f536b040ce5c3cd424587ab68deea3491aeb5  sources/__init__.py
dc585a31e5810fc4dda00e55b07c6bfde2fd75c34985b7dbe618378642bdc4b8  sources/api.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/communication/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/communication/nvlink_domain/__init__.py
315da23c5bf66deb796b5837f7b8c4beb8d20cdd1a7ec158b2e75d64100719aa  sources/communication/nvlink_domain/symmetric_buffer.py
725b855edb022d5e3a3ee7c077342012d3aa9eb9c0169e8c9753b3df2dde0b81  sources/communication/nvlink_domain/token_comm.py
5a342ca0a2a89761ee7d0ad3cd07b6bf2a2b54e714d475a75a95c22a90e30b72  sources/communication/token_protocol.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/helpers/__init__.py
4af7ec124e769b0db2839ac42f90cbfc41fbfe693723f79c471f4b4fff8d0cb8  sources/helpers/constants.py
868871ef468436bcfbfb638c4b8778805f05d6772ee2dea0ba9a5c9b4901d6cc  sources/helpers/cute_py_helpers.py
d92c61b501a7544e15c06de94d66177ebd060ee31194d3b8ba3dab4d73433aa0  sources/helpers/device_workspace.py
d531c11140576dcb22d3b53624b216bbcc8f090283259bcad74d6a0544a24581  sources/helpers/dsl_helpers.py
c9c129755b57ec5bb105e58b301c0a9ece44142b59fabb69613e41aecf076ca3  sources/helpers/flag_batch.py
4b8a3c3e231793f87d6b1bcd5dffee8b5c19c384c3a719fbb4ed895d9929ba29  sources/helpers/iket_compat.py
11bd36ff08a6ee6fa8ad3379c616b5ef7e223f6d967486c7e0af0000e06ef28b  sources/helpers/ptx_helpers.py
91478bee142314688ec56aec5129e46c392597a24fa88cb4d1031b612a031d7c  sources/helpers/smem_workspace.py
f12c4a4df541d4e80346e4cb256f788fc825283696ed69670df13c72c2811525  sources/helpers/software_sync.py
39a95432034901e309c9da1b815153ba85eeaa0b17c3e0386a66067f41ee6b22  sources/helpers/utils.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/__init__.py
81689f9850af024d9ec22416d4c2a1535d73946d455b7e22042d321534918642  sources/kernel_src/function_mapping.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/__init__.py
fbea697f91fe5c5f1623568cb3ed45951905c2428bf2a49f330943ed910df4bc  sources/kernel_src/rubin/custom_mix_cga_helpers.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/inference/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/inference/mega/__init__.py
b4e8bf2729618064678aa11cc5c38476626f13cd97d40f675a9fbe3ced71dd02  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py
6355b04c8672b40e96d2ae1a3309650e1e06eadaf3172e3ccaf2a2b51a88f740  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_extension.py
601a5a53da27d04f6dd98cbe9eb408159bce8b0a5f26f8611b1d39339c22cf24  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop.py
e7f3aa713f90170f3f2c4b616af7a338ee4cb3cdef34621cb21e6ad8a90aabf6  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py
a578368593bed9cf91b8dacf16fa65f8ab822a3001bb457e053f2b4b179013a0  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py
292cd6c6d62b9d6cb7a15e0f43363756941124a645c34be6947b5542282c3ee5  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py
d85b41f1f605c8ce4ae97696e4c8aad5f02eb95412eb42e20d0cfca2dcc4e54a  sources/kernel_src/rubin/inference/mega/dynamic_mainloop.py
0fbc10e05282241250de08ccfaebd533076f29abdd3bb949c063cf80daf54496  sources/kernel_src/rubin/inference/mega/tma_gather.py
f8640f21703099acbf74064368d2d1a6d410760e358bb4341a8b26402b0a8a1f  sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py
56efd874b9859abfcafb2e0cdc39c2690404a274b07590f5c1630f57f57e3f9b  sources/kernel_src/rubin/inference/mega/topk_reduce.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/schedulers/__init__.py
22a68aede7b609cd7f770954d428098207bcb504c12cb71b0bb2198ca339cc04  sources/kernel_src/schedulers/base.py
bf754355af676f767eca31dee76dade21bd9e440a755c159861c943ab45b57c6  sources/kernel_src/schedulers/fc12_mapping.py
ce86b6234fe181b31dc6bbfafe0ed9093a39a600a8e0d3f4dae25983a8f72f97  sources/kernel_src/schedulers/fc12_scheduler.py
c43fc6e1b0858191776e82e702b2f1c373d3984b14f58a8d5011764d20ce7f47  sources/kernel_src/schedulers/non_clc_mixed_cga.py
7b116a8adfecd85197f90e81f14d16d55704210fa3e583ba9f71d80c96b2af79  sources/kernel_src/schedulers/work_id_claim.py
b88290f0cc90713b16e0fd8c60d12d5168027e42497b629251cd47c05ceecfe5  sources/quant_def.py
```

## Ported FlashInfer shim logic (not vendored; re-diff on every FlashInfer bump)

Nothing from FlashInfer's `shim/`, `backends/` or `core/` is imported. The files below
were read (sha256 of the 0.7.3.dev20261031 originals) and their logic ported:

```
f96b97d65934be5186270aef8d7ab8ab756a2c9f8bed8201199d5d82359a040a  shim/block_scaled.py   -> config.py (Sm107BlockScaledMoeConfig rules), session.py (_build_kernel, _runtime_kwargs, prepare/launch)
8d640d2149306fc7d11609e272b9552d4ed8228c2a257551a53db5c92dd4c88f  shim/kernel_helpers.py -> weights.py (FP4/scale helpers, to_blocked, gate/up interleave, prequantized transform), reference.py (oracle)
5475350601b74c33b0eca406d9bbda31e1fb6cd4aa02bef71fc439df8d194602  shim/comm.py           -> symm_arena.py (re-implemented on torch symmetric memory)
e8ff3f29e8a1d7b7ced468f080fcfc24f86e77bc5f6330409fac396c78c8ec94  shim/knob_cache.py     -> config.py (read-only v2 reader; BACKEND_REVISION, key fields)
bb51648766e34e7cbc9d581106ca8a913fb46180eb059114719097b375663135  shim/dependencies.py   -> arch.py
779b509737ad45d6aefb84b3a7d245c8a36f9b7031b4ec12e6456482521a492a  shim/correctness.py    -> reference.py (sampled oracle, output_error)
```

Also ported: `core/tuning.py` `select_token_entries` / `live_buckets` (config.py) and
`flashinfer/cute_dsl/availability.py` `_dsl_captured_arch` (arch.py). Not taken: the
autotuner (`shim/autotune.py`, `tune.py`), NVSHMEM bootstrap (`core/runtime`,
`core/comm`), the per-call routing validation, and the sm100 fused stager (staging.py is
our Triton kernel with the same numerics as FlashInfer's torch staging).

Deliberate differences from the FlashInfer shim:

- one symmetric arena per EP group, carved at identical offsets on every rank, instead of
  per-tensor NVSHMEM allocations (the kernel's single offset table holds for all of them);
- weight bindings cached per layer (FlashInfer keeps one binding and rebinds on every
  weight change), output bindings in a 4-entry LRU;
- no quantized combine, no in-kernel reduction, no masked-slot pre-clear (asserted via
  `kernel.masks_invalid_combine_slots`);
- knob-cache misses raise (`KnobLookupError`) instead of falling back to
  `default_knobs()`; equal-capacity duplicates resolve to the lowest recorded p50;
- routed expert ids other than -1 outside `[0, num_experts)` raise (`RouteIdError`,
  via a sticky device flag set by the stager) like FlashInfer's
  `validate_routing_values`, but without a per-call host sync; `num_experts` is the
  physical expert count with linear placement;
- the compiled-kernel cache key adds quant kind, SiTU betas and device to the kernel
  name (the genphase name omits the betas), and the arena layout hash includes every
  session's kernel name.

## Data: `knobs/k31_vr.json`

Merged FlashInfer v2 knob records of ALL VR tuner runs
(`$W/moe4/results/fi107/knobs_*.json`: 44 files, 44 records; EP4 smoke, EP8 and EP16,
inference and genphase, NVFP4 and MXFP4xMXFP8, SiTU 4.0/25.0, BF16 combine), produced
by `config.merge_knob_files`, which raises unless every source file contributes all of
its records (`source_files` = 44). Each record carries `provenance.file`, that file's
sha256 and the FlashInfer / CUDA versions. Data, not a switch.

Timer protocol of the recorded `p50_us` (`profiling_policy` per record:
`timer=native_events`, `cuda_graph_profile_replays=null`, `cold_l2=false`):
FlashInfer `shim/autotune.py` + `core/tuning_measurement.py` with no measurement
policy, i.e. pre-staged eager `trial.launch` per sample behind a group barrier and a
stream delay kernel, CUDA events, warm L2, per-sample MAX over ranks, median. The
EP16 5925 (NVFP4 1188 us, MXFP4xMXFP8 1361 us) and EP8 5760 (924 / 1121 us) figures
come from this protocol; `bench_megamoe_rubin.py` reports the same protocol as
`kernel_eager` next to its CUDA-graph `kernel` number.
