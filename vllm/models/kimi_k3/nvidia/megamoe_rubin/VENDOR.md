# Vendoring record: megamoe_rubin

## Upstream

- **Source**: FlashInfer tag `wzhao/kimi-k3-megamoe-sm107-ep32-20261118` (merge commit
  `a368c5f66d2976abec8c9beb652fb681098ba416` on github `wzhao18/flashinfer`
  `wzhao/kimi-k3-megamoe-sm107-rebase`: the SM107 MegaMoE SOL work 6e438852b plus EP 4 / 28 / 32
  support 89fafd77e), `flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources`
  (41 files): the upstream Rubin CuTe DSL MegaMoE export (`cutedsl_megamoe` commit
  `1667b47a3c911ecade464ab524baf192a0bf5962`) plus that FlashInfer branch's device patches
  (S0/S1 decode landing and dynamic token_in claim, die_split two-ended expert claim,
  NVFP4 decode weight-stream L2 prefetch + evict-first hints, EP28/EP32 geometry). Copied byte
  for byte except the R1 import rewrite of two files (section "FlashInfer numerics helpers").
- **Entry points**: `RubinInferenceMegaMoE` (`BlockScaledSwapAbMegaMoeKernel`) and
  `RubinInferenceGenphaseMegaMoE` (`BlockScaledSwapAbGenphaseMoeKernel`).
- **Check**: every file is byte-identical to the tag (`git show a368c5f66:<path>`). The CPU unit
  test `tests/models/kimi_k3/test_megamoe_rubin_conversion.py` recomputes every per-file hash
  below.

## Local differences

Only the R1 import rewrite (below). Dropped relative to the previous (0.7.3.dev20261031) drop:

- **S2** (second epilogue warpgroup, vLLM-local): not carried over; the a368c5f66 kernel has one
  epilogue warpgroup and no `epilogue_warpgroups` impl field. `session.build_kernels` builds one
  kernel per session.
- **DL1 / DL2** (vLLM-local die-split scheduler `die_split_fc12_scheduler.py`, impl field
  `die_sm_mask32`): replaced by FlashInfer's `die_split` (`PhaseInterleavedFc12Scheduler`, same
  idea: die 0 claims from the lowest local expert up, die 1 from the highest down, weights homed
  per expert half by `die_local.place_experts`). The host maps the trusted die-1 SM mask to the
  kernel's `sm_locality_domain` launch argument (uint8 per `%smid`, FlashInfer layout).

## FlashInfer numerics helpers: `src/fi_numerics` and the R1 rewrite

The a368c5f66 epilogue and top-k reduce import FlashInfer's batch-invariant numerics helpers
(`flashinfer.moe_ep.batch_invariant`, which imports `flashinfer.batch_invariant.fp_interface.term`). They are
vendored under `src/fi_numerics` (`moe_ep_batch_invariant/`: the 16 non-Lean modules; `fp_interface/term.py`),
with one rewrite: `from ...batch_invariant.fp_interface.term import` -> `from ..fp_interface.term import`. The two
`src/sources` files that import them get rule R1 (absolute `flashinfer.moe_ep.batch_invariant.` ->
`vllm.models.kimi_k3.nvidia.megamoe_rubin.src.fi_numerics.moe_ep_batch_invariant.`); nothing else changes.

```
vendored path | FlashInfer source | original sha256 | vendored sha256
fi_numerics/fp_interface/term.py | flashinfer/batch_invariant/fp_interface/term.py | 28552340a68b34d0e3fe6f3ca17e04e71b6bfaa0801faaa9565d8c4f7e57186c | 28552340a68b34d0e3fe6f3ca17e04e71b6bfaa0801faaa9565d8c4f7e57186c
fi_numerics/moe_ep_batch_invariant/__init__.py | flashinfer/moe_ep/batch_invariant/__init__.py | 08e7f00cff876a51408741a2439c59764f5995b1cfff17f3dd406a2c85f43165 | 08e7f00cff876a51408741a2439c59764f5995b1cfff17f3dd406a2c85f43165
fi_numerics/moe_ep_batch_invariant/spec.py | flashinfer/moe_ep/batch_invariant/spec.py | 989928d2153aed69e0fad8105d5ae82d6d3a6e1e8b30f044dc2ce71b61ee8d99 | 0bd1797527a53ae5c829c5bc2daa9af1ec2dd2d69f50fbfec97bb08c0c4e4e4d
fi_numerics/moe_ep_batch_invariant/a16_spec.py | flashinfer/moe_ep/batch_invariant/a16_spec.py | ba11db083b4cd5bf3d8565b9bc2abe22ca30105d45c6410942880b24a44d0a7b | b7e81a1336ddc4f5619005242726250a1a1ca7257e1763c6d6aa650282504e99
fi_numerics/moe_ep_batch_invariant/activation_variants.py | flashinfer/moe_ep/batch_invariant/activation_variants.py | 90343dc1ee1adc0f0eb903ec1c9486dec605c38eb4ae6a866aed7cda9223c281 | 90343dc1ee1adc0f0eb903ec1c9486dec605c38eb4ae6a866aed7cda9223c281
fi_numerics/moe_ep_batch_invariant/device.py | flashinfer/moe_ep/batch_invariant/device.py | 955c082f1852d195873d1f9fe181e94116821008373fe5e3cfba5dff439c327b | 955c082f1852d195873d1f9fe181e94116821008373fe5e3cfba5dff439c327b
fi_numerics/moe_ep_batch_invariant/mxfp4_device.py | flashinfer/moe_ep/batch_invariant/mxfp4_device.py | 7dc5b7f268d400fd3fa83e88a978eafd0954ca12c9efb276111944382947889f | 7dc5b7f268d400fd3fa83e88a978eafd0954ca12c9efb276111944382947889f
fi_numerics/moe_ep_batch_invariant/mxfp4_spec.py | flashinfer/moe_ep/batch_invariant/mxfp4_spec.py | be8fc4187c944f2a09bfd73ff4125b91a6b0893b0782d70012c6d4c368196aa2 | 46e4768347059907423c35638a201ff658742e94be0387a2f8d3c03e9b80287b
fi_numerics/moe_ep_batch_invariant/mxfp8_device.py | flashinfer/moe_ep/batch_invariant/mxfp8_device.py | c0bdc14d04e3ead57a67552b819166d4aeb720d64d4481fb908f83c189fad22f | c0bdc14d04e3ead57a67552b819166d4aeb720d64d4481fb908f83c189fad22f
fi_numerics/moe_ep_batch_invariant/mxfp8_spec.py | flashinfer/moe_ep/batch_invariant/mxfp8_spec.py | 64742d9c52594df60b702c18c6d0e0abcd2e76125b529422fc15703071d2cb18 | 460b761b2818f36dd9d3032ec2cf8ef025dd2522ecd800572b57f423d959dbbf
fi_numerics/moe_ep_batch_invariant/numerical_variants.py | flashinfer/moe_ep/batch_invariant/numerical_variants.py | 29b5035ae0071bef0e737fd14a7f68e7e860cb1ae520a8f5dd134d0ccc183b0b | 29b5035ae0071bef0e737fd14a7f68e7e860cb1ae520a8f5dd134d0ccc183b0b
fi_numerics/moe_ep_batch_invariant/nvfp4_device.py | flashinfer/moe_ep/batch_invariant/nvfp4_device.py | 48c6af6d43f2ce265074cd2250bd1ab30577e5f9b2b1783741f41417158c6292 | 48c6af6d43f2ce265074cd2250bd1ab30577e5f9b2b1783741f41417158c6292
fi_numerics/moe_ep_batch_invariant/nvfp4_spec.py | flashinfer/moe_ep/batch_invariant/nvfp4_spec.py | e7517ea9afb780268847d1f3f922b6ee8f6441a76da96957c8c94afb66fcadc4 | e879d71a40a7162c8a4753657ee92f4ea473f0e24122df069afd839f0dcd479d
fi_numerics/moe_ep_batch_invariant/quantized_combine_device.py | flashinfer/moe_ep/batch_invariant/quantized_combine_device.py | 69642a57331ea9c400a7b49c2da61ccbddce106d20363015040275b50f72a586 | 69642a57331ea9c400a7b49c2da61ccbddce106d20363015040275b50f72a586
fi_numerics/moe_ep_batch_invariant/stage_partition.py | flashinfer/moe_ep/batch_invariant/stage_partition.py | 7f81538ed96c5618c0bb9e775ec23bffb07c76dbb216c786d576f25ab691cfef | 7f81538ed96c5618c0bb9e775ec23bffb07c76dbb216c786d576f25ab691cfef
fi_numerics/moe_ep_batch_invariant/wire_device.py | flashinfer/moe_ep/batch_invariant/wire_device.py | d12079dd9473da6dab22651f3f5931d0813b33c61cfeddb9df07071ed2fa012f | d12079dd9473da6dab22651f3f5931d0813b33c61cfeddb9df07071ed2fa012f
fi_numerics/moe_ep_batch_invariant/wire_spec.py | flashinfer/moe_ep/batch_invariant/wire_spec.py | f21445052dc0d7579df6f425ba53fc4891b5b9cebfe22e9791fcd218613ad6bf | b9ae9960f9bfee084f2c61fb32bd2a27d40e7c5f9a99c32a77dce3f964f40aab
sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py | (R1 | rewrite) | c6ec37d63026c83e85a69eb14c0234c8e74f34a999882739a241f611dd751560 | e3ab102c3ac93e5cb6a81eec0707d078b02f9295674f24122bb5fae64d43dabb
sources/kernel_src/rubin/inference/mega/topk_reduce.py | (R1 | rewrite) | 44316f7e872fe1ca7ba3d233ea3505069aececfd18a6423c151c77b1916d6358 | f63a32f0379fea7d3c88cb88f3e26bdb875bdb24745416185ed9b7f76c1de66f
```

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
8736c9dd051ee56496c1c55b064d9e73a45dcaa247ee4cadabc197edd767c449  sources/communication/nvlink_domain/token_comm.py
5a342ca0a2a89761ee7d0ad3cd07b6bf2a2b54e714d475a75a95c22a90e30b72  sources/communication/token_protocol.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/helpers/__init__.py
4af7ec124e769b0db2839ac42f90cbfc41fbfe693723f79c471f4b4fff8d0cb8  sources/helpers/constants.py
868871ef468436bcfbfb638c4b8778805f05d6772ee2dea0ba9a5c9b4901d6cc  sources/helpers/cute_py_helpers.py
8e60cb15729e91245dcafb8a9a9c17466884e6b68983307a2b9c213b71ef047a  sources/helpers/device_workspace.py
d531c11140576dcb22d3b53624b216bbcc8f090283259bcad74d6a0544a24581  sources/helpers/dsl_helpers.py
c9c129755b57ec5bb105e58b301c0a9ece44142b59fabb69613e41aecf076ca3  sources/helpers/flag_batch.py
4b8a3c3e231793f87d6b1bcd5dffee8b5c19c384c3a719fbb4ed895d9929ba29  sources/helpers/iket_compat.py
215675b7f0def6526bb3c75b385cf12ccf0a5a35e8816c68bffbda03eaeabd9c  sources/helpers/ptx_helpers.py
91478bee142314688ec56aec5129e46c392597a24fa88cb4d1031b612a031d7c  sources/helpers/smem_workspace.py
f12c4a4df541d4e80346e4cb256f788fc825283696ed69670df13c72c2811525  sources/helpers/software_sync.py
39a95432034901e309c9da1b815153ba85eeaa0b17c3e0386a66067f41ee6b22  sources/helpers/utils.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/__init__.py
81689f9850af024d9ec22416d4c2a1535d73946d455b7e22042d321534918642  sources/kernel_src/function_mapping.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/__init__.py
fbea697f91fe5c5f1623568cb3ed45951905c2428bf2a49f330943ed910df4bc  sources/kernel_src/rubin/custom_mix_cga_helpers.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/inference/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/rubin/inference/mega/__init__.py
e3ab102c3ac93e5cb6a81eec0707d078b02f9295674f24122bb5fae64d43dabb  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py
6355b04c8672b40e96d2ae1a3309650e1e06eadaf3172e3ccaf2a2b51a88f740  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_extension.py
850caf943d137eeeeb99e997a6debce7aea14d3db2cd1e1992601b4e3b711af2  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop.py
60963f01d755f52d5f7b6e1e7c76242bc583b4d2e26192e3639f5aaee9caa22c  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py
413acdb503943619a72e5decbfc2fadd99981c1bb9684808b9420141886c71e5  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py
9179cb6a1f53f55fb6a0f03a269f3a1702567cdf9e2798ec33c50834259e6e42  sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py
d85b41f1f605c8ce4ae97696e4c8aad5f02eb95412eb42e20d0cfca2dcc4e54a  sources/kernel_src/rubin/inference/mega/dynamic_mainloop.py
0fbc10e05282241250de08ccfaebd533076f29abdd3bb949c063cf80daf54496  sources/kernel_src/rubin/inference/mega/tma_gather.py
f8640f21703099acbf74064368d2d1a6d410760e358bb4341a8b26402b0a8a1f  sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py
f63a32f0379fea7d3c88cb88f3e26bdb875bdb24745416185ed9b7f76c1de66f  sources/kernel_src/rubin/inference/mega/topk_reduce.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sources/kernel_src/schedulers/__init__.py
22a68aede7b609cd7f770954d428098207bcb504c12cb71b0bb2198ca339cc04  sources/kernel_src/schedulers/base.py
c853ecafc65add0d5aac0e52b65bec89966c6d52d568afa32e58e46d73fdb8fa  sources/kernel_src/schedulers/fc12_mapping.py
5f59fb9d47bd67e829351d2de9e65f289ec13201844539948db7688e1dce917e  sources/kernel_src/schedulers/fc12_scheduler.py
d6f380b805ea4c432cea09a6d81ea2156e645f7fed0af1b575de2d2145feedd3  sources/kernel_src/schedulers/non_clc_mixed_cga.py
88ca9f947ab88ab703c0f2b1375e5aef6bec890967c34aae0fde7c27845e1af6  sources/kernel_src/schedulers/work_id_claim.py
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

Two records are local, not FlashInfer tuner output: EP16 NVFP4 inference i4096 and
i8192 with tile N=256 (`mma_tiler_mnk` 256x256x256, otherwise the w16c i4096 record's
knobs), tuned for the S2 kernel at real EP16 (4 nodes, 896 experts, random routing) with
the lane bench's `kernel_eager` (the protocol above). Source files
`$W/hti/p2/megamoe-s2/ep16/knobs_s2_ep16_nvfp4_i{4096,8192}.json`. On the S2 kernel N=256
beats the N=128 record at every measured point (4350 tok: 729 vs 810 us on the 148-SM
shared-overlap grid, 693 vs 781 us full grid). The i8192 record is the C512 prefill bucket,
which fell back to the i4096 record before. The i4096 record wins the equal-capacity tie on
p50 (682 vs 731 us).
