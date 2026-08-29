# Kimi-K3 Mooncake Storage Accounting

Status: compact-I/O root cause and validation complete, 2026-08-29. Numerical
results are labeled as measured, reconstructed, or modeled. Optional PMU
restore-closure and ReplaySSM connector work remains explicitly gated below.

## Question

The DCP8 + DEP16 AgentX run scales through concurrency 64 but loses throughput
and external-prefix-cache effectiveness at concurrency 80. The GPU KV capacity
reported by vLLM appears large enough for the active prompts, so this analysis
separates four quantities that should not be conflated:

1. active logical prompt tokens;
2. resident GPU KV blocks;
3. Mooncake values retained from current and historical turn boundaries; and
4. bytes transferred by PUT and GET operations.

The leading finding is that Mooncake capacity is not described by the active
GPU token count. Prefix-match-unit (PMU) tail storage retains many historical
KDA checkpoints, and every checkpoint is serialized using the full shared
physical-pool page. This combination explains the observed Mooncake footprint
to within 0.3%. A default-off compact group-I/O implementation now omits the
unused physical regions. At c80 it reduced terminal occupancy from roughly
3.23 TB with eviction to 1.98 TB without eviction, while external-prefix hit
reached 92.7%. A two-pass standard-rejection GSM8K run then restored 271.57
GiB of compact values with no GET errors and scored 0.944 on pass two.
At c120, the same compact implementation completed all 1,332 warmup requests
with a 94.5% external-prefix hit rate and zero eviction. Its 300-second profile
ended at 2.74 TB / 3.52 TB (78.0%) and 142,756 keys, still with zero eviction;
the server-side external-hit metric remained 90.2%. Compact group I/O therefore
moves the demonstrated zero-eviction operating point beyond c80 to at least
c120 for this workload.

Numerically, the user's capacity intuition is correct for the normal aligned
state: it accounts for only about 358.29 GiB. The deployment supplies 3,600 GiB
raw and admits about 3,240 GiB at the 90% high watermark, so normal state alone
would fit about nine times. The reconstructed PMU population adds 2,877.44 GiB,
bringing the total to 3,235.73 GiB—almost exactly the admission limit. The
problem is therefore retained checkpoint multiplicity and value layout, not a
misinterpretation of the 24 x 150 GiB configured capacity.

## Workload and source evidence

The principal c80 source is:

`srt-slurm/outputs/kimi-k3-mxfp4-gb300-disagg-dcp8-dep16-dspark4-nixl-push-mooncake-slice1m-load-subbatch2-agentx-conc80/609430`

Its prefill log reports 39.73 GiB of available KV memory per rank and
16,287,227 logical tokens across the DCP8 engine. Reconstructing concurrent
requests from the profile gives a peak of about 15.20 million active prompt
tokens. At the engine's observed density, that is about 296.5 GiB, or 93.3% of
the 317.84 GiB aggregate GPU KV budget. This is a GPU-residency estimate, not a
Mooncake-store estimate.

## Physical layout

### Model groups

Kimi-K3 has 24 MLA layers and 69 KDA layers. They are represented by four
hybrid cache groups: one MLA group and three KDA groups of 23 real layers each.
DSpark contributes five additional MLA cache layers. Consequently the common
physical pool stride is 29 layer segments for every group.

For the current KDA state shape at speculative length four, one real KDA layer
contains:

| Component | Shape and dtype | Raw bytes |
| --- | ---: | ---: |
| Convolution state | `[7, 4608]`, BF16 | 64,512 |
| Recurrent state | `[12, 128, 128]`, FP32 | 786,432 |
| Total | | 850,944 |
| Padded layer page | | 884,736 |

Each Mooncake value therefore spans:

`29 * 884,736 = 25,657,344 bytes = 24.46875 MiB`.

This is directly confirmed by the c80 transfer metrics: `save_put_total_bytes /
save_put_total_keys` is exactly 25,657,344 bytes per key. The logged metrics
aggregate all eight DCP workers; they must not be interpreted as per-rank
counts.

The c16 accounting canary (`612971`) independently emitted the live runtime
geometry on every prefill rank. Each KDA group has 23 layers, 19,571,712 raw
bytes, 20,348,928 padded-content bytes, and a 25,657,344-byte physical payload;
the remaining 5,308,416 bytes are cross-group physical padding. The MLA group
has 29 layers and the same 25,657,344-byte payload. The compact-I/O canary
(`613025`) initialized all ranks with 19,571,712-byte KDA values and the
unchanged 25,657,344-byte MLA value, matching the content-only projection
exactly. It subsequently performed real Mooncake restores: one ten-second
interval loaded 512 values / 12,844,449,792 bytes with zero failed keys or RPC
errors. PUT traffic averaged about 22.1 MB/value instead of the legacy
25.657 MB/value as expected from the MLA/KDA mixture. This establishes that the
compact address vectors work for both PUT and GET on the live Kimi layout. It
was subsequently confirmed by model-accuracy validation and a c80 capacity
run, documented below.

### Cross-group padding cost

The four token databases do receive the same `addrs` and `block_lens` lists,
and every PUT/GET expands through all 29 entries. This is not four copies of
four independent groups, however. vLLM allocates one backing buffer, overlays
all cache groups from byte zero, and uses one global physical-block pool. A key
serializes one physical pool slot and `@group:N` determines how those slots are
interpreted. For a KDA key, slots 0 through 22 contain that group's real
layers; slots 23 through 28 are unused padding rather than live MLA or another
KDA group's contents. Hybrid Mamba cache initialization zeroes these slots.

It is nevertheless inefficient for KDA groups: each KDA group has 23 real
layers but serializes a 29-segment physical page. The six unused segments are
5.0625 MiB per KDA key: 20.69% of transferred bytes, or 26.09% overhead versus
the 23 active padded pages. Including the 3.97% padding within each KDA layer,
5.80371 MiB per KDA key is not model state, or 23.72% of its stored value.
DSpark also raises the common stride from 24 to 29 segments, increasing every
stored value by 20.83%.

For the modeled c80 live population of 4,181 MLA keys and 131,232 KDA keys,
group-specific padded region lists would reduce the store from 3,235.73 GiB to
about 2,586.94 GiB, saving 648.79 GiB (20.05%). Content-only KDA regions would
reduce it to about 2,491.95 GiB, saving 743.78 GiB (22.99%). These savings are
substantial but smaller than the PMU historical-state amplification.

This is physical-layout padding, not literal inclusion of another group's live
block contents. The clean general optimization is to give each token database
group-specific I/O regions while keeping the whole backing storage registered.
That requires separating physical block stride from transfer length, which the
current `block_len` representation conflates. Region construction must support
block-compact, layer-compact, and scattered-head layouts, and the external key
namespace needs a layout version or fingerprint so old full-page and new
compact values cannot be confused. This avoids a staging copy while restoring
directly into the correct overlaid slots.

There is an existing implementation to reuse rather than inventing this
mapping from scratch. `kv_connector/v1/offloading/worker.py` already builds
per-layer `(num_blocks, page)` views, computes unpadded Mamba sizes, deduplicates
overlaid aliases, and constructs group-specific references from
`group.layer_names`. Its canonical mapping also versions the external format.
The c80 run uses LBNHC, for which this representation applies directly. Its
packed-layout fallback still exports the whole block for every group, so a
general Mooncake implementation must add stride-aware group I/O vectors or a
staging fallback for packed layouts rather than claiming the same saving there.

PMU boundary MLA values have another, smaller inefficiency. The current
partial-tail path serializes the whole 1,536-token physical page even when the
boundary is only 128-token aligned. Across the c80 requests, the useful residue
averages about 844.9 of 1,536 tokens, so a length-aware MLA representation could
save roughly 45% of the 53.6 GiB boundary-MLA class, about 24 GiB. KDA boundary
values are fixed-size recurrent states and cannot be truncated this way. The
current key/value format has no valid-length metadata, so this also requires a
versioned external layout.

## What is retained in Mooncake

### Normal aligned storage

With DCP8, MLA external keys cover 12,288 global tokens: a 1,536-token local
block on each of eight DCP ranks. KDA recurrent state remains a boundary state
per 1,536 tokens and is namespaced across the eight ranks. With prefix-cache
retention interval zero, the normal path stores dense MLA blocks plus reachable
KDA boundary states rather than every intermediate KDA chunk.

For a fresh 131,072-token request, a useful upper bound is 11 MLA values per
rank plus one current boundary value for each of three KDA groups: 14 values
per rank, 112 deployment-wide values, or about 2.676 GiB. Deduplication and
reachability masks can reduce actual new PUT traffic.

### PMU partial-tail storage

PMU stores reusable checkpoints at sub-block turn boundaries. In a hybrid
MLA/KDA cache this may require intermediate full KDA chunks plus the final
partial boundary checkpoint. Old checkpoints from earlier turns remain valid
keys and are retained even after a descendant turn creates a later checkpoint.
Thus the external store accumulates reusable history that is not part of the
currently active GPU state.

The intuition that a partial block can be discarded once a continuation fills
that block is correct for a single linear conversation, but the current cache
has no such replacement operation. Keys are content-addressed at each
128-token boundary. The old boundary remains a valid branch point, while the
new full-block endpoint has a different key; writing the latter neither
overwrites nor removes the former. Moreover, `_maybe_offload_partial_tail()`
does not save only the final partial block. For every cache group it walks from
the normal save's LCM-aligned floor through the turn boundary, storing uncovered
full blocks as `pmu_gap` values and the final partial state as `pmu_boundary`.
This is why PMU can amplify storage far beyond one stale tail per request.

A uniform "store only the final endpoint" policy is not a valid production
replacement. With DSpark/EAGLE, `FullAttentionManager` verifies the endpoint
and then drops one 128-token PMU unit before execution. Hybrid convergence can
therefore select a Mamba/KDA checkpoint below the nominal endpoint. If the
intermediate KDA checkpoint immediately below that dropped boundary was not
stored, even an identical later prompt can fall back to the previous normal
1,536-token checkpoint (or zero) and recompute the gap. The diagnostic
`partial_tail_endpoint_only` mode remains output-correct through recomputation,
but it can reduce the fine-grained hit it is intended to preserve. It must not
be promoted as-is.

There is a second, independent requirement: when a FullAttention physical
block is smaller than the normal-save LCM, its keys form a contiguous chain.
Lookup stops at the first missing chained block. Omitting an intermediate full
attention block can therefore make the retained final tail unreachable, even
without EAGLE. Any reduced admission policy must store the complete
manager-specific restore closure: contiguous attention ancestry through the
verification/lookahead point, the post-EAGLE-drop frontier, and the newest KDA
checkpoint each recurrent manager can actually select.

The clean general interface is a coordinator-produced partial-tail restore
plan, analogous to the existing load mask, rather than Mooncake inferring
retention from physical block sizes. Each cache manager should return its
required key boundaries and ancestry for a semantic reuse endpoint. Global
history should remain governed by capacity eviction, with staged non-leasing
lookup and leases granted only to the selected restore plan. Explicit deletion
of content-addressed ancestors is unsafe because other branches can share
them. The modeled 1.637 TiB `pmu_gap` class is consequently an upper bound on
possible savings, not a safe deletion target; selected-load telemetry must
measure how many predecessor checkpoints are actually needed.

Reconstructing the c80 key population from the request trace and exact key
generation rules gives:

| Retained value class | Keys | Bytes |
| --- | ---: | ---: |
| Normal dense MLA | 1,938 | 46.31 GiB |
| Normal reachable KDA | 13,056 | 311.98 GiB |
| **Normal total** | **14,994** | **358.29 GiB** |
| PMU intermediate full KDA | 68,520 | 1,637.30 GiB |
| PMU boundary MLA | 2,243 | 53.60 GiB |
| PMU boundary KDA | 49,656 | 1,186.54 GiB |
| **PMU total** | **120,419** | **2,877.44 GiB** |
| **Modeled live total** | **135,413** | **3,235.73 GiB** |
| **Observed final Mooncake total** | **135,042** | **3,226.86 GiB** |

The model differs by only 371 keys, or 0.27%. PMU-originated values account for
88.9% of modeled live keys and 8.03 times the normal-path bytes. At least 48,888
boundary keys, about 1,168 GiB, are older turn endpoints rather than the latest
unique endpoint for their source. This is a conservative obsolete-history
estimate because it does not classify intermediate PMU KDA states by ancestry.

This boundary result is exact at the trace level. Deduplicating the c80
`profile_export.jsonl` by `(source_trace_id, turn_index)` gives 2,265 semantic
prompt endpoints across 132 source traces. The PMU rules reproduce 2,243 MLA
partial endpoints and 2,069 KDA partial endpoints; the latter expand to
`2,069 * 3 groups * 8 DCP namespaces = 49,656` KDA boundary keys. Keeping only
the final observed endpoint of each source leaves 131 MLA keys plus
`120 * 3 * 8 = 2,880` KDA keys. Thus 48,888 of 51,899 PMU boundary keys,
94.2%, are historical relative to the final observed turns, totaling
1,168.19 GiB in the legacy format. Historical does not mean globally dead:
branched continuations and replayed earlier turns can still reference those
content-addressed ancestors.

The Mooncake master independently confirms that the workload reaches the
capacity ceiling, not merely incidental eviction. The deployment has 24
clients, each mounting a
150 GiB segment, for 3.52 TiB total. At the end of c80 it reports 3.15 TiB /
3.52 TiB (89.6%), essentially the configured 0.90 high watermark, with 135,042
live keys. It had cumulatively evicted 121,722 keys totaling 2.84 TiB. The
decode consumers run in capacity-only mode, so their segments contribute
storage capacity but do not create an additional decode-side key population.
This explains both the large aggregate capacity and why eviction begins near
the observed resident footprint.

The matched c64 run also eventually reaches the ceiling: 3.16 TiB / 3.52 TiB
(89.9%), with 135,505 live keys and 2.44 TiB cumulatively evicted. Therefore
"first reaches capacity at c80" would be too strong. The c80 distinction is
greater churn and a larger concurrency-dependent working set: 2.84 TiB was
evicted, about 0.40 TiB more than c64, while useful prefixes are needed by more
simultaneous sessions. The accounting run must determine whether the reduced
hit rate is specifically eviction/rewrite churn rather than occupancy alone.

This is the strongest current explanation for why 15.2 million active prompt
tokens can coexist with a roughly 3.2 TiB external store: active GPU state and
retained reusable prefix history are different populations.

The time series directly ties the falling external hit rate to eviction churn.
The store remains pinned at the 90% high watermark while the live key count
oscillates around 135,000 and evicted keys increase monotonically:

| Local time | Store occupancy | Live keys | Cumulative evicted keys | External hit rate |
| --- | ---: | ---: | ---: | ---: |
| 16:04:29 | 86.0% | 129,511 | 6,799 | 90.4% |
| 16:11:19 | 88.8% | 133,802 | 48,322 | 89.3% |
| 16:13:39 | 89.8% | 135,254 | 60,854 | 88.2% |
| 16:17:59 | 89.9% | 135,415 | 82,044 | 86.0% |
| 16:20:59 | 90.0% | 135,539 | 96,269 | 83.7% |
| 16:23:59 | 90.0% | 135,554 | 110,116 | 82.3% |
| 16:26:09 | 89.6% | 135,026 | 120,074 | 81.0% |

This rules out GPU KV capacity as the direct cause of the hit-rate decline.
The storage reconstruction above supplies the missing class decomposition:
PMU-related values account for 88.9% of live keys, while KDA values also carry
the cross-group physical padding quantified earlier. The matched classifier
controls are now measuring rewrite-after-eviction traffic directly as they
approach the watermark.

The matched 150-versus-170 GB/rank capacity control subsequently reached its
first eviction in both arms. The 150 GB/rank store first evicted at 128,995
live keys, while the 170 GB/rank store first evicted at 146,420 live keys. The
larger store therefore retained 13.5% more keys at its capacity knee, almost
exactly matching its 13.3% increase in configured capacity. Both arms sustained
roughly 92.5% external-prefix hit before eviction. The elapsed time from the
first stored key to eviction was similar because the larger-capacity arm also
completed requests and admitted new history faster; time-to-eviction alone is
not a valid capacity comparison. The key population at the knee is the cleaner
control variable.

This result made the expected role of compact group I/O precise: increase the
number of useful histories retained per byte and delay eviction while leaving
admission semantics unchanged. The completed c120 validation confirms that
effect. A fixed-capacity store can still fill under a longer or larger workload;
when compact I/O eventually reaches its new knee, the next general optimization
is manager-derived restore-closure admission rather than exhaustive PMU
candidate retention.

The miss classifier also changes exactly at eviction. Before the first 150 GB
eviction, 48.0 million missing lookup candidates had never been stored and
zero had been stored earlier. In the first five minutes after eviction, 688
missing candidates were classified as previously stored. The 170 GB arm had
only 10 such keys before its first eviction and 286 in the following 3.3
minutes. These counts are small relative to all fine-grained lookup candidates,
but their onset provides direct evidence that eviction removes histories that
the workload later requests. It closes the causal chain from capacity pressure
to a real lost-prefix mechanism; the longer run will measure whether those
losses accumulate enough to reproduce the retained c80 hit-rate decline.

The early post-eviction normalized loss rate is not yet different between the
capacity arms: 2,247 / 9,236,972 missing candidates in the 150 GB arm and
1,111 / 4,382,668 in the 170 GB arm were previously stored, about 0.024--0.025%
in both. Both arms are already evicting and their admission/work rates differ,
so their end-to-end performance cannot be attributed to capacity by direct
arm-to-arm comparison. This control establishes that capacity scales retained
key count and that eviction creates later-requested misses; the compact run is
the cleaner test because it changes bytes per value at fixed topology and
capacity.

The completed 30-minute controls reinforce that limitation. The 150 GB arm
finished at 3.16 / 3.52 TB with 135,564 keys and 71 eviction passes; the 170 GB
arm finished at 3.57 / 3.98 TB with 153,049 keys and 31 passes. Both had zero
request and transfer errors. The larger arm completed 2,410 requests versus
2,075 and admitted a different closed-loop access stream. Its final normalized
previously-stored-missing fraction was actually higher: 50,465 / 13,458,984
(0.375%) versus 28,065 / 13,409,219 (0.209%). Therefore the controls prove
proportional retained-key capacity and that evicted keys are later requested,
but they do not estimate the performance gain of capacity alone. The compact
run preserves the topology, workload duration, and capacity and is the needed
causal A/B.

For reference, the 150 GB arm measured 122,886 input tokens/s, 46.40 s p50
TTFT, 116.09 s p90 TTFT, 14.87 ms p50 ITL, and 20.61 ms p90 ITL. The 170 GB arm
measured 144,671 input tokens/s, 29.03 s p50 TTFT, 76.25 s p90 TTFT, 24.33 ms
p50 ITL, and 28.74 ms p90 ITL. These end-to-end differences are descriptive,
not a capacity effect, because the completed request populations differ.

### Existence probes alter eviction lifetime

The installed Mooncake 0.3.12.post1 source confirms that `ExistKey` and
`BatchExistKey` are not read-only metadata probes. For every existing object,
`MasterService::BatchExistKey` calls `GrantLeaseForGroup`; an ungrouped object
receives the same lease directly. The c80 master runs with
`default_kv_lease_ttl=3600000` ms (one hour). New PUTs initially receive a
zero-duration hard lease, but any successful existence probe promotes them to
the one-hour lease.

The current AgentX recipe does not enable Mooncake group semantics, so this
effect is presently per object. If group semantics are enabled later, the
current group ID combines every rank and cache group at one hash boundary;
probing one member could then lease MLA and KDA members together. Lifecycle
groups should instead preserve atomic rank shards while separating attention
and recurrent-state families whose usefulness differs.

This interacts badly with fine-grained lookup. `MooncakeStoreWorker.lookup()`
constructs candidates for every 128-token hash boundary, every cache group, and
every rank namespace before asking `batch_is_exist`. It therefore grants long
leases to historical PMU states along the full ancestry even though the Mamba
load plan searches from the right and normally consumes only the newest
recurrent checkpoint. The save-side deduplication probe can also lease existing
candidates.

This provides a mechanism for cache pollution: repeatedly probed historical
PMU keys become protected while newly written but not yet reused keys remain
immediately evictable. It also explains why the store can report substantial
eviction while frequently traversed history persists. The intended correction
is not explicit deletion of content-addressed history, which is unsafe for
shared prefixes and branches. Prefer either:

1. a Mooncake non-leasing existence API for planning, followed by lease
   acquisition only for the final selected load keys; or
2. a vLLM spec-aware lookup that checks required dense MLA blocks normally but
   probes KDA checkpoints from newest to oldest and stops at the selected
   complete boundary.

The second option can be prototyped without changing Mooncake, but it must
preserve hybrid hit convergence and revalidate the selected keys before load.
The accounting run should additionally count which PMU-gap keys are actually
selected for GET, not merely found by the broad existence probe.

A shorter lease is also a useful control now that GETs are sub-batched, but it
is not the ideal final fix. The lease only needs to cover lookup-to-GET queueing
and the selected transfer, with margin; one hour is tied to benchmark duration
rather than transfer correctness. A matched c80 A/B at a conservative few
minutes can test how much scan-induced protection drives churn. It must be
checked for `LEASE_EXPIRED` failures before being considered safe.

### Endpoint-only PMU diagnostic

Stopping all intermediate PMU-gap KDA checkpoints is a useful storage
diagnostic but not a production admission policy. DSpark/EAGLE verifies a hit
at boundary H and then drops one 128-token PMU unit; the recurrent manager may
therefore need the newest checkpoint at or below H-128 rather than the stored
endpoint H. Full-attention blocks also require contiguous ancestry through the
saved frontier. Endpoint-only storage remains output-correct only because a
missing restore point falls back to an older checkpoint or recomputation; it
can destroy the intended prefix reuse.

In c80, these gap states alone are 68,520 values / 1,637.30 GiB, or 50.6% of
the modeled live store. The c16 diagnostic confirms that gap stores can be
eliminated, but it also shows the expected loss of reusable restore points. The
replacement must retain a manager-derived restore closure, including the EAGLE
predecessor and full-attention ancestry, rather than assuming that the final
endpoint alone is sufficient.

The broad probe also does much more metadata work than either manager needs.
For full attention, `FullAttentionManager` first checks only full 12,288-token
block hashes and then searches at most the interior 128-token boundaries of the
first missing block. For KDA, `MambaManager` searches newest to oldest and stops
at the first complete recurrent state. Mooncake currently asks existence for
all 128-token hashes before invoking either algorithm. A two-phase spec-aware
planner can therefore reduce both metadata traffic and lease pollution:

1. query MLA full-block endpoints, then only the first partial block's interior;
2. query KDA endpoints newest-first in bounded windows until one complete
   boundary is found; and
3. acquire/revalidate leases only for the final load plan.

This remains general across request lengths and avoids a fixed KV-size cap.

## DSpark/EAGLE effects

The source and retained-log audit rules out a fourfold persistent KDA-store
multiplier from speculative length four. DSpark affects storage in two other,
distinct ways.

### Registered target and draft state

The target registers 93 cache-bearing layers: 24 MLA and 69 KDA. The DSpark
checkpoint declares five decoder layers, and `K3DSparkDecoderLayer` contains an
MLA cache but no KDA cache. `K3DSparkForCausalLM` assigns those caches logical
layer IDs 93 through 97. They are ordinary members of the MLA cache group, so
the final groups are:

| Group | Real cached layers | Padded slots in the shared window |
| --- | ---: | ---: |
| MLA | 24 target MLA + 5 draft MLA = 29 | 0 |
| KDA 0 | 23 target KDA | 6 |
| KDA 1 | 23 target KDA | 6 |
| KDA 2 | 23 target KDA | 6 |

This is independently visible in retained logs. The c80 DSpark producer reports
`num_regions=29` in its hybrid-SSM registration and
`num_groups=4, num_segments=29` in Mooncake registration. Retained Kimi-K3
runs without DSpark report 24 segments. Thus the five draft layers widen the
common physical value from 24 to 29 slots, a 20.83% increase over the
non-DSpark value, or 17.24% of the current value.

For DCP8, one local MLA page is 1,536 tokens times 576 FP8 bytes, exactly
884,736 bytes. The k=4 KDA page is also padded to 884,736 bytes. Consequently
each key still transfers 29 pages, or 24.46875 MiB. For an MLA key all 29 pages
are meaningful target/draft MLA state. For a KDA key only 23 pages are KDA
state; six pages are padding. The five draft MLA pages are not copied into a
KDA value as live data, but their increase of the shared group width creates
five of those six unused KDA slots.

For the modeled c80 population (4,181 MLA keys and 131,232 KDA keys), the five
extra slots account for about 557.88 GiB relative to a 24-slot layout. Only
17.23 GiB is necessary draft-MLA content in MLA keys; about 540.66 GiB is the
incidental widening of KDA values. Group-specific I/O removes the latter while
preserving the former.

### Speculative KDA blocks are transient, not separate Mooncake values

At speculative length four, each target KDA layer's convolution state grows
from `[3, 4608]` to `[7, 4608]` BF16. Its raw per-layer state grows from
814,080 to 850,944 bytes, an increase of 36,864 bytes (4.53%). Both sizes fit
inside the same 884,736-byte padded page, so this changes no bytes in the
current full-page Mooncake format. A future content-only compact format would
expose this small raw-state increase: 847,872 bytes per 23-layer KDA value,
or about 103.63 GiB across the modeled 131,232 KDA keys. It is measurable but
still not a speculative-length multiplier.

`MambaSpec.num_speculative_blocks=4` also allocates four physical scratch
blocks per KDA group. Across three KDA groups this is 12 additional shared-pool
blocks per active request:

`12 * 29 * 884,736 = 307,888,128 bytes = 293.625 MiB per rank/request`.

The c80 producer log reaches at most 15 simultaneously running requests, so
the corresponding active scratch upper bound is about 4.30 GiB per rank, not
hundreds of GiB in Mooncake.

The save path makes the persistence distinction explicit. `process_tokens()`
emits only logical token chunks, and a KDA value selects
`group_blocks[start // 1536]`. The four speculative blocks are appended after
the request's logical chunk range, so no store key can select them. Each KDA
key therefore persists one boundary-state block, not one value per speculative
position. Partial-tail saves similarly select the KDA copy-on-write boundary
block, not the speculative tail slots.

There is still a transient interaction worth measuring. The scheduler's
async-save reference path pins every allocated request block, rather than only
the blocks selected by that save. It therefore pins the speculative scratch
slots until all ranks report save completion. Store backlog can lengthen their
lifetime and force later allocations, but it does not add persistent keys or
bytes to the Mooncake master. The accounting telemetry should report unique
speculative blocks pinned and their pin duration separately from stored values.

Finally, this disaggregated recipe does not persist decode-side draft state:
the prefill producer leaves `save_decode_cache` disabled, and the decode
workers are capacity-only Mooncake consumers. DSpark's EAGLE-style hit rule
does deliberately recompute the last 128-token hash unit in fine-grained PMU
mode to regenerate target hidden states. That is a fixed correctness cost
present at both c64 and c80, not a concurrency-dependent store-capacity
amplifier.

The audit conclusion is therefore: speculative length four is important for
transient GPU residency and save pinning, but it does not create four
persistent KDA values. The large external DSpark inefficiency is the shared
29-slot physical layout, especially its approximately 540.66 GiB modeled
cross-group widening of KDA values.

## ReplaySSM / RecoverSSM

[vLLM PR 51855](https://github.com/vllm-project/vllm/pull/51855) replaces the
per-speculative-position recurrent-state copies with one checkpoint and compact
per-token recovery records. Its reported fixed-budget effective KV capacity
gain is 10.97%. For this workload, modeled steady per-request KDA/MLA residency
drops from about 29 physical blocks to 16, despite a larger page format.

The larger format is exact rather than qualitative. RecoverSSM adds a
`[12, 5, 128]` FP32 correction record and a `[12, 5, 256]` BF16 key/gate
record, 30,720 bytes each. The raw KDA layer becomes 912,384 bytes, and the
platform raises the hybrid block from 1,536 to 1,664 tokens, making its page
958,464 bytes. A naive 29-segment Mooncake value would therefore be
27,795,456 bytes, 2,138,112 bytes or 8.33% larger than the current value.

[vLLM PR 52993](https://github.com/vllm-project/vllm/pull/52993) optimizes the
recovery kernels and should be included in any future performance A/B.

ReplaySSM is not currently safe to enable with Mooncake: vLLM explicitly
rejects ReplaySSM with a KV connector. Naively removing that validation would
serialize the larger physical stride, including transient recovery-record
space, increasing each external value by about 8.33%. Connector support must
first distinguish persistent convolution history and recurrent checkpoint from
transient recovery records, scatter a restore into the larger runtime page,
initialize the omitted transient fields, and version the external namespace.
PR 52993 improves the verify/commit kernels but does not change these storage
semantics. ReplaySSM is therefore promising for GPU residency, but it is not
an explanation or immediate fix for the current external-store footprint.

## Transfer-control evidence

Matched real-transfer and combined Mooncake-GET/NIXL-payload no-op controls show
that transfer is material but not the only c80 limit:

| Run | Input TPS | QPS | p50 TTFT | p50 ITL |
| --- | ---: | ---: | ---: | ---: |
| c64 real | 174,536 | 1.3983 | 8.54 s | 20.68 ms |
| c64 combined no-op | 173,184 | 1.3972 | 4.82 s | 25.48 ms |
| c80 real | 132,310 | 1.2098 | 28.87 s | 22.45 ms |
| c80 combined no-op | 143,850 | 1.3106 | 16.68 s | 23.70 ms |

At c80, removing the transfer payload improves input throughput by 8.72% and
reduces p50 TTFT by 42.2%. But even the no-op control loses 16.8% input
throughput from c64 to c80, so transfer optimization alone cannot restore ideal
scaling. The real c80 run also contains two GET and one PMU PUT hard 60-second
failures, and GET/PUT latency is roughly 2.4 times c64.

## Diagnostic instrumentation

An isolated branch, `wzhao/mooncake-storage-accounting`, contains default-off
telemetry. It records candidate, deduplicated, missing, successfully stored, and
failed values and bytes, split by normal/PMU-boundary/PMU-gap class and cache
group. Failed GETs are rechecked to distinguish transfer failure while a key is
still present from lookup-to-GET eviction races. It also logs raw content,
padded content, physical payload, block span, speculative blocks, and TP
replication for each group.

The full Mooncake worker unit suite passes: 139 tests. No diagnostic code has
been promoted to the main branch yet.

An independent compact-I/O prototype is in
`wzhao/mooncake-compact-group-io` at commit `f48bdbcf1d`. Behind a default-off
`compact_group_io` flag, it versions the key namespace, separates physical
block stride from transfer length, builds group-specific regions, strips known
attention/Mamba padding, and uses the actual physical block ID for each value.
LBNHC, LBHNC, and LHBNC have explicit test coverage; block-outer BLHNC, BLNHC,
and BHLNC currently fail closed rather than silently serialize invalid data.
Compact multi-block values also fail closed because the load-error API currently
has a one-key to one-invalid-block contract. The complete affected two-file
unit suite passes 151 tests. A real Mooncake + NixlPush c16 runtime canary
completed as job `613025`; its
artifacts are isolated under `vllm-investigation-runs/mooncake-compact-group-io`
rather than the retained benchmark output tree.

A follow-up code audit found two blockers before enabling it beyond the current
Kimi canary: the compact namespace initially lacked a schema fingerprint, and
the multi-block preparation path assumed consecutive physical block IDs. The
working tree now includes a 128-bit canonical schema fingerprint over ordered
group/layer/spec geometry, dtype, layout, effective DCP-promoted chunk size,
and hash-block size. The last two fields prevent deployments with different
token spans per value from sharing a compact namespace. The implementation
consumes each supplied block ID, preserves a coalesced fast path for consecutive
dense blocks, and rejects partially overlapping non-identical regions. A
byte-sentinel round trip verifies exact restoration of meaningful bytes while
stripped padding remains untouched.
The multi-block path now explicitly fails closed, preserving the one-key to
one-invalidation-ID contract; current Kimi `process_tokens()` emits one
database block per key. The complete affected two-file suite passes 151 tests.
The live canary has completed real external GETs without connector errors and
reached a 34.5% external-prefix hit during warmup. Restore-side GSM8K, matched
c80, and higher-concurrency c120 validations subsequently passed as documented
below.

The completed c16 profile returned 88/88 requests without request or connector
errors. It measured 23,187.9 input tokens/s, 0.2890 QPS, 0.928 s p50 TTFT,
1.760 s p90 TTFT, and 10.27 ms p90 ITL. The frontend cache-hit metric was
90.01%. At shutdown Mooncake held 406.83 GB across 20,424 keys with zero
evictions, or about 19.92 MB per live key versus exactly 25.657 MB/key in the
legacy format. This is a 22.4% observed residency reduction for the canary's
actual MLA/KDA key mixture.

A separate default-off staged-lookup prototype is checkpointed on
`wzhao/mooncake-spec-aware-lookup` at commit `12b3ecb542`. It preserves the
legacy FullAttention/Mamba hit result while querying full-attention endpoints
and recurrent checkpoints in bounded manager-specific windows. The full
Mooncake worker test file passes 135 tests after the final windowing changes.
For the observed 128-unit c80
requests and window size 16, the common probe count is modeled to fall from
4,096 keys/request to 512–640 (84–88% fewer). This reduces accidental lease
refresh but does not eliminate it; a non-leasing planning API remains the clean
Mooncake-side follow-up.

A correctness-first restore-closure prototype is isolated on
`wzhao/mooncake-restore-closure`. It asks the existing cache managers to
evaluate candidate boundary sets, retains FullAttention ancestry, and selects
the KDA predecessor that remains usable after DSpark/EAGLE drops one PMU unit.
It also converges multiple KDA groups onto a common usable predecessor. The
planner now enables access recording on a hypothetical external pool, performs
one normal hybrid-manager convergence to collect the candidates actually
consumed, and performs a second convergence to verify the reduced closure.
This is O(manager-probed boundaries), avoids duplicating cache-type rules, and
fails safe to all candidates if verification differs. Additional tests require
all recurrent groups to restore atomically, preserve older and newer
multi-turn endpoints, and use an already-saved frontier as the DSpark
predecessor without redundant admission. The complete coordinator and worker
suites pass 183 tests.

An independent review found that this is not yet sufficient for a runtime
capacity claim. The prototype treats `_saved_offset` as proof that the seeded
FullAttention chain and KDA frontier exist externally. In the production save
path `_record_saved(token_len)` can advance even when `store_mask` emits no
keys, so `_saved_offset` is a progress watermark rather than an existence
certificate; later Mooncake eviction can also remove a once-written frontier.
The planner must either query the bounded prerequisite set or conservatively
admit the seeded predecessor. The current tests also need three genuinely
distinct recurrent manager groups, DCP8 rank expansion with one missing shard,
two divergent hash-chain branches, an evicted saved frontier, and exact
filtered PUT-to-GET tensor round trips. The prototype remains isolated until
these gates and compact-I/O composition pass GSM8K.

## Follow-up optimizations

The general compact group-specific I/O optimization and its c80/c120 and
accuracy validation are complete. The remaining opportunities are separate
follow-ups rather than requirements for the established root cause:

1. Repair and validate manager-derived PMU restore-closure admission, starting
   from actual external prerequisite existence rather than `_saved_offset`.
2. Add save-pin lifetime telemetry for current-boundary, speculative-scratch,
   checkpoint, and copy-on-write blocks; this targets transient GPU pressure,
   not Mooncake master capacity.
3. Add a non-leasing planning/existence API so broad historical probes do not
   extend candidate lifetime before the selected restore plan is known.
4. Define a versioned ReplaySSM connector projection that persists only base
   convolution/recurrent state and reconstructs transient recovery records.

## Success criteria

This investigation is not complete when an inefficiency is merely identified.
The intended endpoint is a general implementation change followed by matched
c64/c80 and higher-concurrency runs. A successful change must show all of:

1. lower live Mooncake key/byte residency and fewer eviction rewrites;
2. sustained or improved external-prefix-cache hit rate as concurrency grows;
3. no increase in failed GET/PUT operations or recomputation;
4. preserved model accuracy and exact KV restore behavior; and
5. improved or preserved input throughput, QPS, TTFT, and decode ITL.

The first production candidate is group-specific compact I/O. It removes bytes
that are provably unrelated to a key's cache group, does not change cache
admission or replacement semantics, and is modeled to move the observed c80
residency from 3,226.86 GiB to roughly 2,485 GiB if the key population and its
observed MLA/KDA mixture are held constant. That is below the 3,240 GiB high
watermark, so compact I/O alone should provide a decisive c80 test of the
capacity hypothesis.

A restore-closure-aware PMU admission policy is the second candidate.
The simpler endpoint-only diagnostic is now ruled out for production because
it omits a KDA checkpoint needed after DSpark's one-unit hit reduction. The
replacement must derive its retained checkpoint set from the cache managers'
actual lookup and convergence semantics, then pass hit-rate, recompute,
branching, and accuracy validation. Compact I/O has now passed c80 and c120
alone. Restore closure should only be combined with it after the prototype's
existence, branching, rank-atomicity, round-trip, and accuracy gaps are closed.

The restore-closure prototype has a conservative capacity bound. The modeled
population contains 118,176 PMU KDA keys: 68,520 gap states plus 49,656
boundary states. The 2,243 PMU MLA boundary keys imply at most 2,243 endpoints
times three KDA groups times eight rank namespaces, or 53,832 manager-selected
KDA predecessors. Replacing the exhaustive PMU KDA population with that
closure removes at least 64,344 compact KDA values, or about 1,172.84 GiB. A
central estimate that replaces the gap population while retaining the observed
boundary-sized predecessor population removes 68,520 values / 1,248.95 GiB.
Combined with compact group I/O, the modeled store becomes about 1.24--1.32
TiB. These are bounds derived from the retained key population, not runtime
measurements.

Before this policy is used in a sweep, its minimum validation matrix is:

1. FullAttention ancestry with short and nonzero frontiers, including the
   partial/lookahead tail.
2. Three-group KDA hybrid restore without speculation.
3. Three-group KDA plus DSpark, verifying the attention endpoint and the
   predecessor below the one-PMU speculative drop.
4. Multi-turn and branched histories, retaining every admitted semantic
   endpoint rather than deleting old branches.
5. An unadmitted gap branch that recomputes from the seeded normal frontier
   with identical output.
6. A missing KDA rank/group that fails the hybrid restore atomically instead of
   loading mixed state.
7. Exact mock-Mooncake tensor round trips for the non-speculative and DSpark
   closures.
8. DSpark plus Mooncake GSM8K parity with the exhaustive admission baseline.

The compact-only c80 capacity validation completed as job `613183`, with artifacts under
`vllm-investigation-runs/mooncake-compact-group-io-c80/613183`. It uses the
same c80 workload and 150 GB/rank Mooncake capacity as the retained baseline;
the only connector-format change is compact group I/O.

After the initial trajectory primers, this run began restoring compact values
from Mooncake. One 10-second interval completed 208 GET batches covering 408
keys / 9,737,920,512 bytes; another completed 176 batches covering 328 keys /
7,831,388,160 bytes. Both reported zero failed keys and zero GET errors. At
470.82 GB / 22,248 keys the master still reported zero eviction. The external
prefix hit metric had begun rising from 0% to 4.4% as replayed turns entered
the pressure phase.

The final result is the decisive capacity validation. Mooncake held 1.98 TB
across 102,736 keys, only 56.2% of its 3.52 TB capacity, with zero eviction.
The external-prefix hit rate reached 92.7% and the benchmark completed all 884
warmup requests plus 531 profiled requests with zero request or connector
errors. Its reported frontend cached-token rate was 95.136%. The retained
legacy c80 run instead reached about 3.23 TB, evicted state, and ended near an
81% external-prefix hit rate. Compact group I/O therefore removes the c80
capacity cliff rather than merely changing accounting.

The 300-second compact profile reported 119,524.9 input tokens/s, 1.6358 QPS,
2.469 s p50 TTFT, 5.511 s p90 TTFT, and 14.03 ms p90 ITL. The retained legacy
profile used a 1,800-second measurement window, so its throughput mix is not
a strict matched-duration performance control; storage occupancy, eviction,
connector errors, and hit-rate retention are the causal comparison here.
The first compact c100 allocation, job `613540`, failed before serving traffic:
DeepGEMM MegaMoE's startup profile asserted `Grid sync timeout` on one decode
node, poisoning its CUDA context. It allocated no KV store state and is not a
compact-I/O or capacity result. The identical c100 rerun is job `613568`, under
`vllm-investigation-runs/mooncake-compact-group-io-c100/613568`.
Compact c120 validation completed as job `613559`, under
`vllm-investigation-runs/mooncake-compact-group-io-c120/613559`; neither job
had a dependency on the other.

The c120 warmup completed 1,332/1,332 requests with zero errors. External
prefix hit rose from 0% during the initial 132 root primers to 94.5% at warmup
completion. Mooncake then held 2.02 TB / 3.52 TB and 104,292 keys with zero
eviction. The fixed 300-second profile completed 595 requests with zero request
errors; 14 remaining accepted requests were cancelled only after the fixed
grace-period cutoff. At the terminal measurement Mooncake held 2.74 TB
(78.0%) and 142,756 keys, still with zero eviction and no compact connector
failure. The server-side external-prefix metric ended at 90.2%.

The standard srt aggregated result reports 1.853 requests/s, 115,218 input
tokens/s, 1,211 output tokens/s, 4,851 total tokens/s/GPU, 14.81/28.93 s
p50/p90 TTFT, and 15.02/16.94 ms p50/p90 ITL. AIPerf's separate effective
timeslice aggregation reports 1.750 requests/s, 108,994 input tokens/s, and
1,146 output tokens/s; the definitions must not be mixed. The result JSON's
frontend cache-hit share is 84.797%, while the raw usage-derived prompt-cache
read share is 85.15%. Both use a different denominator from the server's
external-prefix metric. The c120 result demonstrates that compact group I/O
alone keeps this workload below the capacity/eviction cliff through at least
c120.

Standard-rejection GSM8K write-side accuracy validation completed as job
`613217`, with
artifacts under
`vllm-investigation-runs/mooncake-compact-group-io-gsm8k/613217`. It uses the
same DCP8+DEP16 topology and compact format, but replaces synthetic DSpark
acceptance with normal target verification.
The matched exhaustive-format baseline is job `600457`: 0.951 accuracy and
zero invalid responses on all 1,319 questions. Compact I/O must match that
result within normal deterministic evaluation tolerance.

The compact run scored 0.942 accuracy with 0.001 invalid responses, versus
0.951/0.000 for the exhaustive-format baseline. It completed all 1,319
questions with zero lookup, PUT, or request errors. One interval stored 1,420
keys / 29,952,230,400 bytes, exactly
21,093,120 bytes per key. This equals one 25,657,344-byte MLA value plus three
19,571,712-byte content-only KDA values divided by four. It is direct runtime
evidence that the six unrelated KDA segments are no longer serialized.

This one-pass evaluation is only a write-side check. It recorded no
`load_get_count` samples and a 0% external-prefix hit rate because the unique
GSM8K prompts remained resident in the prefill GPU cache. Consequently its
accuracy cannot validate bytes restored from the compact Mooncake format. A
valid restore-side evaluation must populate Mooncake, evict or reset only the
prefill-local cache while preserving the external store, replay the same
questions, demonstrate nonzero successful compact GETs, and then compare the
second-pass accuracy. Reducing prefill KV capacity and repeating the fixed
corpus is the simplest no-code control if a local-only reset cannot be invoked
reliably through Dynamo.

Dynamo exposes a potentially stronger deterministic control on the prefill
worker's system port: `POST /engine/flush_cache` calls
`engine_client.reset_prefix_cache()` without `reset_connector=True`. This
would clear the local vLLM prefix cache while preserving Mooncake; it is
distinct from `dynamo.prefill.clear_kv_blocks`, which explicitly clears the
connector. The benchmark environment variable intended to invoke this route
was present in job `613454`'s recipe, but it was not propagated to the built-in
GSM8K wrapper. There is no flush message or newly logged `Successfully reset
prefix cache` line, so the run must not be described as a deterministic-flush
control.

Job `613454` nevertheless provides a real restore-side accuracy validation.
It ran the same 1,319-question corpus twice. Pass one scored 0.942 accuracy
with 0.001 invalid responses; pass two scored 0.944 with 0.001 invalid
responses. During pass two, the prefill worker completed 3,456 Mooncake GET
batches covering 13,824 compact keys and 291,591,290,880 bytes (271.57 GiB).
Every GET completed with zero failed keys and zero errors. Mooncake retained
271.57 GB / 13,824 keys with zero eviction. Thus the second pass did not merely
exercise local-cache hits: it restored the complete compact external working
set and preserved model accuracy. The results are under
`vllm-investigation-runs/mooncake-compact-group-io-gsm8k-repeat2-read-validation/613454`.

The validated first production solution is not merely increasing Mooncake
capacity. It avoids serializing unused physical padding while preserving the
existing multi-turn admission semantics. Reducing retained historical PMU
state remains a separate opportunity that requires the restore-closure gates
above.

The compact implementation is committed on `wzhao/k3-nvfp4-perf` as
`a911b24e60` and `ed04c4dbec`. Its focused Mooncake worker, scheduler, and HMA
suite passes 179 tests. The feature remains default-off and fail-closed for
unknown layouts, overlapping physical segments, incompatible schema
fingerprints, or ambiguous one-key-per-block assumptions.

### Rejected terminal-prefill PMU-tail handoff

One follow-up prototype handed the final partial KDA block directly from a
remote-prefill request to Mooncake after its last forward pass. The lifetime
argument was sound: unlike a continuing local request, a remote-prefill request
does not overwrite that source block in a subsequent step, and the connector's
normal store reference can keep it alive through the asynchronous PUT. Focused
core and connector tests passed, and a c16 runtime canary completed 87 profiling
requests without request, CUDA, or transfer errors.

The cache result did not justify the change. Against the matched compact c16
baseline, realized reuse relative to the trace-derived 128-token PMU ceiling
changed only from 93.13% to 93.91%. The same structured 128--1408-token deficit
remained. The prototype was therefore reverted and was not included in the
production commits.

The experiment also clarifies the lifecycle. The prefill endpoint is not the
general next-turn restore point: decode subsequently appends generated tokens,
while this deployment deliberately has `save_decode_cache=false`. Persisting
the prefill endpoint cannot materialize the post-decode KDA state. In addition,
the trace-derived ceiling uses the source conversation, whereas the live next
turn can contain model-generated assistant text, so per-request differences may
be negative and should not be interpreted as exact missing-token counts. Future
work should address decode-history reconstruction or retention semantics rather
than repeating this handoff.

### Modeled and measured compact-format concurrency knee

Two independent estimates agree on compact c80 residency. Applying the exact
group mixture gives about 2,485 GiB; applying the measured c16 compact-to-legacy
bytes/key ratio gives about 2,505 GiB. Both are well below the 3,240 GiB
admission watermark. A deliberately conservative linear scaling with
concurrency predicts about 3,106 GiB at c100 and 3,728 GiB at c120, placing the
knee near c104. That is an upper-pressure model: c64 and c80 traverse the same
corpus and end with almost the same capped legacy live-key population, so the
working set can grow sublinearly. The c120 measurement disproved the simple
linear knee estimate: it ended at 2.74 TB / 3.52 TB (78.0%), 142,756 keys, and
zero eviction, while retaining a 90.2% server-side external-prefix hit rate.
The demonstrated compact-format knee is therefore above c120 for this workload.

The compact c80 result should not be judged by throughput alone. The decisive
capacity signature is approximately 2.5 TiB occupancy, a live-key population
that can exceed the legacy 135,500-key ceiling without eviction, stable
external-prefix hit rate, and collapse of rewrite-after-eviction PUTs. If these
storage metrics improve while throughput remains capped, the remaining limit
is scheduling or P-to-D transfer rather than Mooncake capacity.
