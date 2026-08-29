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
