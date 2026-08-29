# DS4 hybrid dispatch research — measured results

Date: 2026-08-29

Local-only branch: `research/v013-ds4-hybrid-dispatch`

Base checkpoint: `738c1804a88e9f99742714d7fbc354ecf7e0b279` (`Prepare v0.1.3 multi-model MoE runtime checkpoint`)

Scope is deliberately narrow. DirectStorage is excluded. FRONT keeps the restored 2.7-GiB registered host store. The campaign measures only:

1. all-hit memory-only L1/L2 split;
2. all/partial L2 miss admission on a full 8-GiB cache;
3. L2-hit -> P -> GPU same-use promotion economics;
4. shared-expert CPU overlap on a dedicated worker against routed H2D.

The exact DS4 M03 work unit is used for compute tests: step 65, layer 0, six routed experts, 7,096,320 B per expert slot. CPU/GPU routed endpoint max-abs difference is 0.01320755, matching retained M03 evidence.

## 1. All-hit memory-only split

No H2D and no storage are timed. All six selected experts are already resident in either L1/GPU or L2/CPU. CPU and GPU subsets execute concurrently.

| L2 / CPU | L1 / GPU | Critical wall ms/layer | CPU branch ms | GPU branch ms |
|---:|---:|---:|---:|---:|
| 0 | 6 | 0.5051 | 0.0000 | 0.5050 |
| 1 | 5 | 0.7590 | 0.7172 | 0.4416 |
| 2 | 4 | 1.1890 | 1.1436 | 0.3616 |
| 3 | 3 | 1.7006 | 1.6425 | 0.2941 |
| 4 | 2 | 2.0870 | 2.0184 | 0.2136 |
| 5 | 1 | 3.7916 | 3.7175 | 0.1361 |
| 6 | 0 | 2.8347 | 2.8273 | 0.0000 |

The 5/1 CPU-heavy cell shows larger CPU variability than neighbouring cells; the independent promotion sweep below gives a more stable 5-CPU branch around 2.34 ms. The qualitative result is unchanged: with zero movement, every expert moved from GPU residency to CPU increases the critical branch. Hybrid dispatch is interesting only when CPU execution avoids or hides a source/promotion cost.

## 2. Real 8-GiB L2 admission: all/partial miss

This uses the current 0.1.3 SiliangEM cache engine against the real expert-major DS4 GGUF on C:. The cache is 8 GiB / 1210 slots, LRU, unbuffered + overlapped, deferred wait enabled. It is filled completely once before measurement. Every timed miss therefore incurs a real full-cache eviction. Three repetitions use distinct routed layers so the requested miss geometry remains exact.

Layer-0 selected expert bytes copied from L2 were byte-exact against the frozen M03 package.

| Selected L2 hits | Misses | prepare_async ms | I/O wait ms | Total admission ms | Bytes read | Evictions |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 0.0110 | 0.0002 | **0.0112** | 0 | 0 |
| 5 | 1 | 0.3993 | 2.0565 | **2.4480** | 7,096,320 | 1 |
| 4 | 2 | 0.7871 | 3.5737 | **4.3944** | 14,192,640 | 2 |
| 3 | 3 | 1.1735 | 5.1109 | **6.2844** | 21,288,960 | 3 |
| 2 | 4 | 1.5172 | 6.6669 | **8.1679** | 28,385,280 | 4 |
| 1 | 5 | 1.9570 | 8.1180 | **10.1618** | 35,481,600 | 5 |
| 0 | 6 | 2.3549 | 9.6698 | **12.0445** | 42,577,920 | 6 |

The slope is close to ~2 ms of additional admission wall per extra cold expert in this full-cache state. The submission component itself is ~0.39 ms/miss and includes current victim/table work; the remainder is the overlapped read completion wall.

### Partial-miss route with the other selected experts already in L1

The following is an assembled critical-path projection from measured components, not an additional synthetic benchmark. For `N>=1`, the measured L2 admission wall is much longer than the already-resident K/GPU subset compute, so the K-hit branch can fit under the admission window. After the misses arrive, the current best same-use choice from section 3 is to compute them on CPU rather than promote them through P.

Stable CPU subset medians are taken from the independent p=0 promotion cells.

| Misses loaded to L2 | Existing L1 hits | L2 admission ms | Post-arrival CPU subset ms | Routed/source critical ms/layer |
|---:|---:|---:|---:|---:|
| 0 | 6 | 0 | 0 | **0.505** GPU resident endpoint |
| 1 | 5 | 2.448 | 0.697 | **~3.145** |
| 2 | 4 | 4.394 | 1.211 | **~5.606** |
| 3 | 3 | 6.284 | 1.484 | **~7.769** |
| 4 | 2 | 8.168 | 1.906 | **~10.074** |
| 5 | 1 | 10.162 | 2.335 | **~12.497** |
| 6 | 0 | 12.045 | 2.687 | **~14.732** |

The all-miss value is the requested worst-case routed/source path for this experiment: about 12.04 ms to admit six experts into a full 8-GiB L2 plus about 2.69 ms to compute the six routed experts on CPU. It is not whole-model token latency.

## 3. L2 hit competing for L1 admission

This assay starts with `N=1..6` selected experts available in ordinary/pageable host memory and the remaining selected experts already resident in L1/GPU. For each N it sweeps how many L2 candidates are promoted for this same use through the current product topology:

`L2/pageable -> P pinned memcpy -> private-stream H2D -> GPU compute`.

The remaining L2 candidates compute on CPU concurrently.

### Best same-use choice

For every N from 1 through 6, **promote=0 is the fastest measured same-use choice**.

| Selected L2 hits N | p=0 wall ms | p=1 wall ms | all N promoted wall ms |
|---:|---:|---:|---:|
| 1 | **0.765** | 1.448 | 1.448 |
| 2 | **1.257** | 1.545 | 2.466 |
| 3 | **1.527** | 1.579 | 3.309 |
| 4 | **1.947** | 2.272 | 3.782 |
| 5 | **2.382** | 2.682 | 4.510 |
| 6 | **2.702** | 3.123 | 5.241 |

The closest cell is N=3, where promoting one L2 expert costs only ~0.052 ms more than keeping all three CPU. This is still not a same-use win.

### Why current promotion loses

The dominant cost is not GPU compute. It is `L2 -> P` host memcpy.

Representative source-to-P memcpy medians:

- one promoted expert: ~0.57-0.98 ms depending on concurrent CPU load;
- two promoted: ~1.57-2.07 ms;
- three promoted: ~2.35-3.01 ms;
- six promoted: ~4.37 ms.

When a CPU routed branch is active, the host memcpy generally gets worse because both consumers use host memory bandwidth. This independently explains the retained D5 observation of ~160-245 ms/token of L2->P worker-copy work.

Therefore current same-layer policy should not promote an L2 hit merely because K has a free/reclaimable slot. Promotion needs future reuse value, or a cheaper source path, to amortize the one-time transport.

## 4. Shared expert on a dedicated CPU worker overlapping routed H2D

This assay deliberately follows the requested topology:

- shared expert on a persistent dedicated CPU worker;
- CPU backend uses one worker thread pinned to logical CPU 0;
- routed source is hot/pinned before timing;
- routed H2D uses the production-style Siliang private CUDA copy stream, three H2D submissions per expert;
- 0..6 expert transfers are measured.

| Routed experts moved | H2D alone ms | Shared CPU alone ms | H2D concurrent ms | Shared CPU concurrent ms | H2D tail after shared ms | Critical wall ms | H2D slowdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.965 | 0.000 | 1.967 | 0.000 | 1.980 | 1.00x |
| 1 | 0.583 | 1.975 | 0.586 | 2.154 | 0.000 | 2.228 | 1.01x |
| 2 | 1.129 | 1.961 | 1.134 | 2.286 | 0.000 | 2.372 | 1.00x |
| 3 | 1.677 | 1.966 | 1.688 | 2.443 | 0.000 | 2.553 | 1.01x |
| 4 | 2.228 | 1.979 | 2.237 | 2.633 | 0.000 | 2.798 | 1.00x |
| 5 | 2.768 | 1.959 | 2.783 | 2.705 | 0.000 | 2.873 | 1.01x |
| 6 | 3.310 | 1.963 | 3.333 | 2.747 | 0.396 | 3.333 | 1.01x |

This changes the interpretation of the older shared-CPU overlap warning. With a dedicated worker, **DMA itself is almost perfectly protected**: only ~0-1% H2D slowdown through six experts. Contention is paid mostly by the CPU shared branch, which slows from ~1.96 ms alone to ~2.75 ms with six simultaneous H2Ds.

For six routed transfers, shared CPU covers about 2.75 ms of the 3.33-ms concurrent transfer path and leaves ~0.40 ms after the shared branch completes.

### Correctness caveat

The current CUDA shared-expert graph is byte/numerically exact to the frozen M03 shared reference (`max_abs=0`). The current 0.1.3 CPU backend produces a materially different shared output (`max_abs ~1.50647` vs frozen/current CUDA) on the same input/weights/graph. The old M03 and current CPU backend implementations are not identical.

Therefore the overlap mechanism is measured and promising, but **shared-CPU placement must not be promoted into the natural runtime until this current-CPU numerical difference is understood or accepted under an explicit correctness criterion**.

## 5. FRONT remains unchanged

The restored product FRONT keeps its full registered host store:

- host store: 2,829,703,168 B (~2.635 GiB);
- rolling GPU bank: 81,431,552 B (~77.66 MiB);
- no `cudaMallocHost(2.7 GiB)`; backing is ordinary committed RAM registered with CUDA.

FRONT rolling is expensive in traffic (~2.635 GiB/token), but retained phase-slab evidence shows the one-layer issue->need window is normally long enough to hide the transfer. This campaign does not attempt to replace FRONT with DirectStorage or a different host backing.

## 6. Immediate conclusions

1. **L1 hit remains the dominant state.** All-hit memory-only is fastest with all six routed experts on GPU (~0.505 ms/layer for this work unit).
2. **A full 8-GiB L2 hit is cheap to discover**: six hits cost only ~0.011 ms of cache lookup/admission bookkeeping.
3. **Cold misses are expensive before compute**: six misses into a full 8-GiB L2 cost ~12.04 ms even though reads are overlapped.
4. **Current P-based same-use promotion is not profitable.** For N=1..6 L2 hits, p=0 wins every cell. L2->P memcpy is the dominant promotion tax.
5. **Adaptive policy should classify after the router**: K hits execute GPU immediately; already-L2 hits should normally remain CPU for the current use unless admission is justified by future reuse; cold misses first pay their L2 admission cost.
6. **Dedicated-thread shared CPU overlap is mechanically much better than the older contention result suggested.** It barely perturbs DMA, but the current CPU shared-expert numerical mismatch is a correctness blocker.

These are component/mechanism results, not a whole-model tok/s claim. Natural integration should only follow after selecting the policy from these measured cells.

## 7. Follow-up clarification: direct-registered transient GPU upper bound

The section-3 result is specific to the current product path `L2 -> P memcpy -> H2D`. A second sweep removes that host copy and assumes the selected L2 source is already CUDA-registered/pinned. The destination is interpreted as transient GPU/R-style execution, not persistent K admission:

`registered L2/source -> private-stream H2D -> transient GPU slot -> routed GPU compute`.

No `cudaHostRegister` lifecycle is timed here; this is the already-registered transport upper bound. Per-use register/unregister was previously measured separately and rejected as too expensive.

| Selected L2 hits | Best moved to transient GPU | Kept CPU | Best critical wall ms | Keep-all-CPU wall ms |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | **0.655** | 0.655 |
| 2 | 1 | 1 | **0.923** | 1.386 |
| 3 | 1 | 2 | **1.367** | 1.518 |
| 4 | 2 | 2 | **1.474** | 1.957 |
| 5 | 3 | 2 | **1.957** | 2.441 |
| 6 | 3 | 3 | **2.218** | 2.759 |

This reverses the section-3 conclusion once the P memcpy is removed. For 4-6 L2-resident selected experts, the optimum converges near a balanced CPU/GPU split. In particular, with six L2-resident experts, `3 CPU + 3 transient GPU` beats `6 CPU` by about 0.54 ms in this current-binary mechanism assay.

The result does **not** require a K eviction or persistent K admission. R=12 already provides two six-slot transient GPU banks in the product topology. A production direct-register design would therefore target `registered L2 -> R`, preserving K policy independently.

## 8. Follow-up clarification: current shared GPU compute cost

On the same 0.1.3 checkpoint and the same frozen shared-expert work unit:

- shared GPU compute-only wall: **0.1139 ms/layer**;
- shared CPU compute-only with one dedicated backend thread: **2.0027 ms/layer**;
- current CUDA output vs frozen shared reference: `max_abs=0`;
- current CPU output vs current CUDA/frozen: `max_abs~1.5093`.

With six expert H2Ds and one dedicated shared-CPU worker:

- H2D alone: **3.3018 ms**;
- H2D while shared CPU runs: **3.3080 ms**;
- shared CPU duration in that concurrent run: **2.6969 ms**;
- observed H2D tail after shared CPU finishes: **0.4634 ms**;
- shared+H2D critical wall: **3.3080 ms**.

That tail stops at H2D completion; it does **not** include routed GPU compute. Six already-resident routed experts take about **0.505 ms/layer** in the current all-L1 assay. Therefore a simple `shared CPU || six H2D`, followed by six-expert GPU compute, projects to roughly `3.308 + 0.505 ~= 3.81 ms/layer`, or about `0.463 + 0.505 ~= 0.97 ms` remaining after the shared CPU branch finishes. This is a component projection, not an integrated natural-graph measurement.

The shared-GPU alternative is dramatically cheaper when no routed transfer exists: about 0.114 ms/layer instead of about 2.00 ms/layer on one CPU thread. Shared-CPU placement therefore only becomes attractive if its VRAM release and overlap with routed transport compensate that exposed cost on hit-heavy layers/tokens.

## 9. Terminology correction for full-L2 admission

The `0-miss / 6-hit` ~0.011 ms figure is not a data admission. It is the fixed `prepare_async` lookup/classification/bookkeeping path that is called even when all selected experts are already resident in L2.

For six real misses into a full 8-GiB L2:

- `prepare_async`: **2.3549 ms total**, about 0.39 ms/miss, covering hit/miss classification, victim/table work and async read submission;
- blocking `wait_experts`: **9.6698 ms total**, waiting for the six outstanding storage reads to complete;
- total L2 arrival: **12.0445 ms**;
- if all six are then executed on CPU, routed CPU compute adds **2.6872 ms**;
- resulting routed/source component path: about **14.73 ms/layer**.

The 2.6872 ms is actual routed expert matrix compute after the expert bytes are resident. It is not scheduling, copying or cache preparation.

## 10. Natural non-frozen route telemetry

A research-only console flag was added on this branch:

`--expert-cache-route-stats`

It is default-off, requires the L1 K/R/P runtime, emits only aggregate shutdown statistics (no per-token/per-layer spam), and writes the explicitly requested telemetry to stderr independently of ordinary INFO log filtering. It separates two different axes:

- **residency before dispatch:** L1/K hit, L2 hit, or uncached/cold L2 miss;
- **execution in the current runtime:** existing K, newly admitted K, transient R, or CPU.

It also reports route-composition histograms, so costs can later be weighted by actual route states instead of only aggregate hit percentages.

Three fresh natural runs were executed with no route replay/oracle and different prompts (factual, technical, creative). Configuration was held fixed: DS4 expert-major, L2=8 GiB LRU, K216/R12/P12 cumulative-LFU, full registered FRONT, bounded-prefill arena disabled, 32 requested decode tokens. Each retained 1,333 decode route calls / 7,998 selections.

| Workload | L1 | L2 | Cold | Generation |
|---|---:|---:|---:|---:|
| factual | 30.06% | 36.47% | 33.47% | 2.0 tok/s |
| technical | 32.30% | 37.97% | 29.73% | 2.1 tok/s |
| creative | 29.18% | 34.93% | 35.88% | 1.9 tok/s |

Across all three runs (3,999 route calls / 23,994 selections):

- L1/K resident: **7,321 = 30.51%**;
- L2 resident but not K: **8,748 = 36.46%**;
- uncached/cold at route time: **7,925 = 33.03%**;
- unknown classifications: **0**.

Current execution placement for the same selections was:

- existing K hit: **7,321 = 30.51%**;
- newly admitted to K: **1,778 = 7.41%**;
- transient R execution: **14,895 = 62.08%**;
- CPU routed execution: **0**;
- unknown execution: **0**.

The most common pre-dispatch route composition was `L1=2, L2=2, cold=2` (361 routes, **9.03%**). Other common mixed states were `3/2/1` (6.83%), `2/3/1` (6.80%), `1/3/2` (6.78%), and `3/1/2` (5.50%). This directly shows that mixed same-layer states are central rather than edge cases.

The current execution policy is correspondingly R-heavy: the most frequent execution compositions were `2 K-hit + 4 R` (20.51%), `1 K-hit + 5 R` (16.43%), `3 K-hit + 3 R` (15.40%), and `0 K-hit + 6 R` (13.93%). This makes the direct-register/hybrid CPU-GPU question quantitatively relevant to natural decode, not just to frozen micro-assays.

## 11. SLFU cold-admission and K-hot demotion variants

Follow-up policy work introduced `slfu` as the user-facing name for the existing lifetime-frequency cumulative-LFU admission policy (`cumulative-lfu` remains a legacy alias). L1 LRU is retired from qualification at K216: DS4's 43 routed layers x top-6 yield a 258-key cyclic route working set per token, larger than global K216. An always-admit recency scan therefore evicts early-layer entries before their next-token reuse. With exclusive L2->K admission it also removes those entries from L2, converting the next access into a cold miss.

Two orthogonal SLFU controls are now research-visible:

- `--admit-k-cold on|off`
  - `on`: preserves the old first-use behavior; a cold miss may enter K immediately if SLFU admits it;
  - `off`: a first-use cold expert is admitted to L2 but executed transiently via R; only a later L2 hit can be considered for K.
- `--demote-k-hot on|off`
  - `off`: preserves the immediate exclusive L2->K behavior;
  - `on`: if SLFU selects a K victim, the candidate executes from R for the current layer. After routed compute completes, the runtime performs an exclusive tier swap: K victim D2H -> bounded pinned demotion staging, candidate R -> K by D2D, then the candidate's L2 slot is released and reused for the victim. K metadata changes only after this transition completes. No persistent K/L2 duplication exists.

This does **not** define or alter the L2 eviction policy. L2 remains independently configured as LRU, LFU, or W-TinyLFU. A demoted victim is inserted into the exact L2 slot freed by the promoted candidate and starts with normal L2-local insertion state; K lifetime-frequency is not injected into L2 policy state.

### Mechanical demotion gate — corrected exclusive swap

The earlier leased-shadow prototype was rejected because it duplicated K residents in L2 and reduced effective L2 capacity. Its `demote=on` measurements are superseded and must not be used for qualification.

The corrected zero-duplication mechanism was then smoke-tested with `SLFU + admit-k-cold=off + demote-k-hot=on`. In a 4-token run it produced:

- 58 deferred K promotions;
- 58 transition commits;
- 58 K->L2 demotions;
- 0 transition cancels / 0 demotion failures;
- D2H victim traffic: 174 part copies, 410,517,504 B total;
- D2D R->K traffic: 174 part copies, 410,517,504 B total.

With the L2 commit still performed synchronously at the next mapper, host commit work was ~95 ms total across 58 swaps. Moving that L2 release/store work to one persistent worker reduced the next-router exposed wait to **12.895 ms total**, while the worker performed ~33.553 ms of device-event wait and ~55.190 ms of L2 host-copy work in parallel with the graph. This is about **0.22 ms exposed per swap** in the smoke cell. These are mechanism timings, not a whole-model throughput claim.

### Natural 32-token policy matrix

Protocol held fixed: DS4 expert-major, L2=8 GiB LRU, K216/R12/P12, FRONT full registered, routed prefill arena off, deterministic prompt/sampling. Only L1 policy / SLFU toggles changed. Each arm observed 1,333 decode routes / 7,998 expert selections.

| L1 policy | admit cold | demote hot | L1 hit | L2 hit | cold | K admits | R transient |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFU always-admit | n/a | n/a | **0.00%** | 9.23% | **90.77%** | 7,998 | 0 |
| W-TinyLFU | n/a | n/a | 24.68% | 9.21% | 66.10% | 6,024 | 0 |
| SLFU | on | off | 26.71% | **37.28%** | **36.01%** | 514 | 5,348 |
| SLFU | off | off | **26.77%** | 37.05% | 36.18% | 492 | 5,365 |
| SLFU | on | on | *superseded* | *superseded* | *superseded* | — | — |
| SLFU | off | on | *superseded* | *superseded* | *superseded* | — | — |

Immediate interpretation at 32 tokens:

1. LFU's always-admit behavior is scan-thrashing at K216 just like LRU, despite its different victim ranking. It has not yet earned a global retirement conclusion, but it is non-competitive in this DS4/K216 cell.
2. W-TinyLFU retains meaningful K locality but remains too admission-aggressive for an exclusive L2->K hierarchy in this early checkpoint: L2 breadth collapses and cold traffic remains high.
3. SLFU's rejection/bypass behavior is the dominant structural improvement: it preserves a large warm L2 population while still learning a hot K set.
4. `admit-k-cold=off` works mechanically (2,894 first-use cold bypasses in this run) but changes the 32-token aggregate only slightly because baseline SLFU already rejects most low-value cold candidates.
5. The original `demote-k-hot=on` rows are invalid for the corrected design because they came from the rejected leased-shadow prototype. The corrected exclusive swap must be re-run naturally before any hit-rate conclusion is drawn.

The intended long-horizon checkpoints remain cumulative generated-token marks 32/64/128/256/512/1024/2048. The runtime now emits these checkpoints automatically under `--expert-cache-route-stats`. The full 2048-token matrix has not been physically completed in this pass; the 32-token matrix above is the completed natural comparison.

## 12. Natural 64-token three-replica screening

A second screening pass removed frozen routes and deterministic sampling overrides. Each cell used three fresh processes with the same prompt and ordinary/default sampler behavior. Each process generated 64 requested tokens and emitted cumulative 32/64 route-stat checkpoints from the same continuously evolving cache state. K216/R12/P12, 8-GiB L2 capacity, full registered FRONT, and routed prefill arena off were held fixed. No runs were concurrent.

L1 LRU was excluded by the already-established K216 scan-thrash mechanism. L2 LRU was not physically repeated in this screening; its retained historical DS4 cells remain the external control. L2 LFU and W-TinyLFU were screened physically.

### L2 policy screen with SLFU `cold=off, demote=off`

| L2 policy | L1 median | L2 median | cold median | combined hit | generation median |
|---|---:|---:|---:|---:|---:|
| LFU | 26.49% | 21.58% | **51.35%** | 48.07% | ~1.65 tok/s (2 completed timing rows) |
| W-TinyLFU | **28.71%** | **32.32%** | **38.97%** | **61.03%** | **1.9 tok/s** |

The LFU L2 result was stable across its three 64-token checkpoints (cold 51.33-53.85%) and is rejected for the longer pass in this topology. W-TinyLFU materially improves warm-L2 breadth and reduces cold traffic.

### W-TinyLFU L2 x L1/admission screen

| L1 policy / controls | L1 median | L2 median | cold median | combined hit | generation median | swap exposed median |
|---|---:|---:|---:|---:|---:|---:|
| SLFU cold=off, demote=off | 28.71% | 32.32% | 38.97% | 61.03% | 1.9 tok/s | — |
| **SLFU cold=on, demote=off** | 28.81% | **34.26%** | **37.10%** | **63.07%** | 1.9 tok/s | — |
| **SLFU cold=off, demote=on** | **29.26%** | 33.27% | 38.15% | 62.53% | 1.9 tok/s | **~0.040 ms/swap** |
| **SLFU cold=on, demote=on** | 28.36% | 34.06% | 37.58% | 62.42% | 1.9 tok/s | ~0.129 ms/swap |
| W-TinyLFU L1 | 22.68% | 5.39% | **71.93%** | 28.07% | **1.4 tok/s** | — |

The W-TinyLFU-L1 arm remains admission-aggressive: at 64 tokens its three runs produced roughly 12.3k-12.7k K admissions out of 16,254 selections, R=0, and only ~5% L2 residency. It is rejected for longer qualification. L1 LFU is likewise not repeated because its always-admit K216 scan-thrash is structural and had already produced zero/near-zero useful K locality in the earlier natural cell.

The corrected exclusive `demote-k-hot=on` path does not reproduce the rejected shadow-capacity penalty. Across the three `cold=off, demote=on` runs it committed 364-404 real K<->L2 swaps per run with no cancels/failures. The persistent worker hid the D2H/D2D plus host-L2 commit work so that next-router exposed wait was about 0.036-0.064 ms per swap. The `cold=on, demote=on` cell showed higher variance in exposed wait (~0.029-0.148 ms/swap), another reason not to rank it from hit-rate alone.

The three configurations retained for longer natural runs are therefore:

1. `L2=W-TinyLFU, L1=SLFU, admit-k-cold=on, demote-k-hot=off`;
2. `L2=W-TinyLFU, L1=SLFU, admit-k-cold=off, demote-k-hot=on`;
3. `L2=W-TinyLFU, L1=SLFU, admit-k-cold=on, demote-k-hot=on`.

The next stage extends these cells with fresh three-replica natural runs and reads cumulative checkpoints from each uninterrupted generation rather than restarting the cache at each checkpoint.

## 13. Natural 256-token three-replica extension

The three SLFU/W-TinyLFU-L2 finalists from the 64-token screen were extended with **three new fresh-process natural runs each** to 256 requested decode tokens. Each uninterrupted generation emitted cumulative 32/64/128/256 checkpoints from one continuously evolving cache. Default sampler behavior was retained; no frozen route, replay/oracle, temperature override, top-k override, or fixed seed was used.

### Cumulative state at 256 tokens

| Configuration | Rep 1 L1/L2/cold | Rep 2 | Rep 3 | Median L1 | Median L2 | Median cold | Generation median |
|---|---|---|---|---:|---:|---:|---:|
| **SLFU cold=on, demote=off** | 29.07 / 37.05 / 33.88 | 28.59 / 36.70 / 34.71 | 29.31 / 35.49 / 35.19 | **29.07%** | **36.70%** | **34.71%** | **1.9 tok/s** |
| SLFU cold=off, demote=on | 27.26 / 35.87 / 36.87 | 29.40 / 34.97 / 35.63 | 29.40 / 36.82 / 33.78 | **29.40%** | 35.87% | 35.63% | 1.9 tok/s |
| SLFU cold=on, demote=on | 29.21 / 34.10 / 36.68 | 28.02 / 36.32 / 35.66 | 27.98 / 36.07 / 35.95 | 28.02% | 36.07% | 35.95% | 1.9 tok/s |

The median of each column is reported independently, so L1+L2 medians should not be treated as the exact median of a per-run combined-hit scalar. The robust ordering is driven mainly by cold share and confirmed again by the late-window view below.

### Window-only state: generated tokens 129-256

Subtracting the cumulative 128 checkpoint from 256 isolates the later 128-token window and removes most of the cold-start weight:

| Configuration | Rep 1 L1/L2/cold | Rep 2 | Rep 3 | Median L1 | Median L2 | Median cold |
|---|---|---|---|---:|---:|---:|
| **SLFU cold=on, demote=off** | 30.76 / 38.26 / 30.98 | 30.73 / 38.79 / 30.48 | 31.21 / 37.78 / 31.00 | **30.76%** | **38.26%** | **30.98%** |
| SLFU cold=off, demote=on | 27.79 / 37.38 / 34.83 | 31.56 / 36.15 / 32.28 | 30.47 / 38.38 / 31.15 | 30.47% | 37.38% | 32.28% |
| SLFU cold=on, demote=on | 28.80 / 35.35 / 35.85 | 29.45 / 38.02 / 32.53 | 28.07 / 37.05 / 34.89 | 28.80% | 37.05% | 34.89% |

The simpler **cold=on / demote=off** cell is the clear 256-token winner: its late window converges around 31% L1, 38% L2, and 31% cold. `cold=off / demote=on` remains second; the corrected demotion mechanism is cheap enough that its loss is policy-driven rather than transport-driven.

For `cold=off / demote=on`, the three 256-token runs committed 620/663/640 real swaps with zero transition failures. Next-router exposed wait was approximately **0.0338 / 0.1026 / 0.0185 ms per swap** (median ~0.0338 ms/swap), while D2H/D2D and host-L2 commit work ran behind the persistent worker. Thus real K<->L2 demotion is mechanically viable, but in this workload delaying first-use cold admission sacrifices more locality than the demotion recovers by 256 tokens.

The three cells are retained for one fresh natural 1024-token run each to test whether their ranking changes at a longer horizon.

### Demotion reuse lifecycle telemetry

Before long-horizon qualification, `--expert-cache-route-stats` was extended to measure whether a real K->L2 demotion is useful before the demoted expert is needed again. Each demoted `(layer, expert)` opens one lifecycle episode. On its first later selection, the existing L2 prepare result classifies the episode without any extra cache query:

- `reuse_L2`: the demoted expert is still an L2 hit at its next selection;
- `reuse_cold`: it has been evicted from L2 before its next selection and is cold again;
- `reuse_pending`: it has not been selected again yet by the current checkpoint/end of run;
- `reuse_unknown`: the existing L2 classification was not exact.

Reuse distance is measured in subsequent visits to the victim's own layer, not raw global route calls, with buckets `1`, `2-4`, `5-8`, `9-16`, and `17+`. This handles global-K victims from both earlier and later layers correctly. No per-expert logging or additional L2 lookup is performed; the metric reuses `prepare_l2`'s existing per-route hit/miss classification.

A 4-token mechanical smoke on L2 W-TinyLFU + L1 SLFU, `admit-k-cold=off`, `demote-k-hot=on` closed exactly: 44 demotions = 3 `reuse_L2` + 0 `reuse_cold` + 41 pending + 0 unknown. All three resolved useful reuses were in the first same-layer revisit bucket. This smoke is a telemetry gate only, not a performance result.
