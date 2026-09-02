# DS4 decode locality: asynchronous expert execution + dynamic CPU/GPU load balancing

**Research date:** 2026-09-02  
**Branch:** `research/v016-expert-arrival`  
**Base:** `132ff512e2`  
**Model:** DeepSeek-V4-Flash 0731 Siliang expert-major  
**Reference machine:** Ryzen 5 2600 / RTX 2070 8 GiB / 24 GiB DDR4 / Windows WDDM / Gen3 NVMe.

This report records not only the final throughput, but the causal chain: baseline, observed bottlenecks, failed assumptions, fixes, correctness constraints, and why the result exists.

## 1. Frozen maintainer continuity baseline

Matched configuration before the new decode path:

- `llama-server`, K216 / R12 / P12
- L2 = 8192 MiB, **LRU**
- L1/K = **SLFU**
- `admit-k-cold=on`, `demote-k-hot=off`
- DeepSeek4 FRONT rolling
- routed-expert prefill OFF for the continuity decode cell
- managed/no-mmap backing
- t2/tb2
- 256 decode tokens, 3 fresh processes

| metric | median | range |
|---|---:|---:|
| decode | **1.880 tok/s** | 1.872–1.899 |
| TPOT | **~532 ms/token** | — |
| cumulative @256 | 1.873 | 1.865–1.896 |
| window 128→256 | 1.818 | 1.809–1.849 |
| K @256 | 25.56% | 25.14–26.49% |
| L2 @256 | 36.77% | 35.34–39.50% |
| cold @256 | 36.74% | 34.94–39.52% |

Receipt: `baseline-current-20260902.json`.

The historical ~2.3 tok/s result remains evidence but is not the current baseline: it had a substantially more favorable locality trajectory.

## 2. Storage arrival: wait-all was the wrong execution boundary

L2=8 GiB assays measured real NVMe→host completion for top-6 misses. Representative 6-miss medians:

| ready expert | LRU | SLFU |
|---:|---:|---:|
| 1 | **2.50 ms** | **2.60 ms** |
| 2 | 4.39 | 4.58 |
| 3 | 6.72 | 6.86 |
| 4 | 8.95 | 9.02 |
| 5 | 10.93 | 10.99 |
| 6 | **11.78** | **11.86** |

The first expert is usable ~9.3 ms before the last. Raw I/O timing is almost policy-independent; LRU/SLFU mainly change ownership/admission.

Old critical path:

```text
issue N reads -> wait all -> compute all N
```

New critical path:

```text
issue N reads
ready E1 -> full CPU expert compute
ready E2 -> full CPU expert compute
...
ready EN -> final full expert compute
```

With a real 6-cold progressive assay, final read was ~11.8 ms, final CPU expert ~0.7 ms, total wall ~12.5 ms. This experimentally confirmed the desired **last-arrival + tail-compute** behavior.

## 3. CPU compute fits between storage arrivals

With the research build correctly compiled AVX/AVX2 ON, one DS4 expert on the 4-thread CPU path is roughly 0.63–0.75 ms; the natural progressive executor is around ~0.7 ms/expert including orchestration. This is below the typical ~1.5–2.2 ms spacing between NVMe completions.

Therefore most cold-expert CPU compute can be hidden behind later storage reads.

## 4. Important implementation traps

### In-flight `OVERLAPPED` objects cannot move

The first ready-queue prototype compacted pending entries. Windows keeps the pointer to the original `OVERLAPPED`; moving it while I/O is active is invalid. Correct design: fixed slots + active bitmap + temporary handle view.

### Win32 wait-any has a 64-handle limit

Decode top-6 was fine, but prefill may have >64 outstanding reads. `WaitForMultipleObjects` must drain in windows <=64. Before this fix, prefill could leave outstanding I/O at the prefill→decode boundary.

### CPU research build must match production ISA

A misconfigured research build had AVX/AVX2 OFF and made one expert appear ~4.3 ms instead of ~0.63 ms. Cost-model data is invalid unless build features match the qualified binary.

## 5. Execution placement and persistent residency are separate decisions

For the current token/layer/top-6:

```text
K hit    -> GPU, no movement
L2 hit   -> CPU or transient GPU according to break-even/backlog
cold     -> NVMe async -> CPU progressive
```

Independently, SLFU decides whether a non-K expert deserves future K residency. If yes, L2/P→K is prepared asynchronously and K ownership becomes visible only after CUDA completion.

This separation is central: **compute NOW != residency NEXT**.

## 6. Dynamic load balancer grounding

Planner inputs are intended to come from pre-bake/LUT calibration:

- K/L2/cold composition for the current top-6
- CPU full-expert cost vs thread count
- NVMe arrival curve 0→6 misses
- L2→P memcpy cost
- P→GPU H2D cost
- resident GPU compute cost
- current CPU/GPU backlog

A critical correction from historical transport assays: ~0.55 ms/expert was an idealized direct/pinned H2D figure. The actual pageable L2→P→GPU product promotion path was closer to ~1.4–1.5 ms for one expert. The planner therefore uses product-path economics and remains conservative.

For multiple cold misses it often chooses CPU for L2 hits, because CPU work can finish before/between NVMe arrivals. GPU is used heavily for already-K-resident experts and only selectively for L2 hits.

## 7. Correctness contract

Completion order is irrelevant. Every expert writes to its **original router dispatch slot**, and final accumulation is always slot `0 -> 1 -> 2 -> 3 -> 4 -> 5`.

Current endpoint evidence:

- CPU current vs frozen M03: exact / roundoff-level (~0 to 1e-7)
- GPU current vs frozen M03: **max_abs ~0.01320755**
- CPU vs GPU: ~0.01320753

The CUDA delta predates dynamic scheduling. The hybrid gate must demonstrate no new divergence beyond this known backend-specific difference.

## 8. Causal performance chain

### Progressive CPU-only sustained proof

Before reintroducing K/GPU locality, the natural `llama-server` custom routed op was tested with K deliberately unused. Over 256 decode tokens it reached:

- **2.537 tok/s**
- **~394.1 ms/token**

This is not a matched production topology, but it is strong causal evidence that progressive NVMe→CPU execution alone can move the system from the ~532 ms/token baseline regime toward ~400 ms/token.

### First full hybrid: improvement, but unexpectedly slow

After adding K-hit GPU execution, dynamic L2 CPU/GPU choice, SLFU future admission and cold progressive CPU, the first matched K216 runs were only ~**2.16–2.17 tok/s**. This proved there was a new implementation overhead hiding the gain.

Profiler at 64 tokens / 2709 routed-layer calls before the fix:

```text
hybrid callback total       ~23.62 s
CPU full-expert compute      ~8.86 s
wait_next                    ~1.33 s
L2/P/GPU copy                ~1.33 s
prepare                      ~1.24 s
GPU submit                  ~10.49 s   <-- pathological
GPU result read              ~0.18 s
join                         ~0.05 s
```

`GPU submit` was ~3.87 ms per routed layer, equivalent to roughly **166 ms/token** over 43 routed layers. That was far too large to be GPU arithmetic.

## 9. Why `tensor_set` was changed

The first out-of-band hybrid GPU executor did three synchronous `ggml_backend_tensor_set()` operations per routed layer for tiny inputs:

- activation ~16 KiB
- top-6 physical IDs ~24 B
- router weights ~24 B

The payload is tiny, so multi-millisecond cost cannot be explained by bandwidth. The exposed host/GPU synchronization/submission behavior was the problem.

The path was changed to:

```text
CPU values
  -> memcpy to small dedicated pinned host block
  -> activation H2D async
  -> IDs H2D async
  -> weights H2D async
  -> CUDA event dependency
  -> GPU routed graph
```

After this change, at 64 tokens:

```text
hybrid callback total       ~16.25 s
CPU full-expert compute      ~9.62 s
wait_next                    ~3.33 s
L2/P/GPU copy                ~1.53 s
prepare                      ~1.15 s
GPU submit                   ~0.195 s
GPU result read              ~0.218 s
join                         ~0.053 s
```

GPU submit dropped from ~3.87 ms/layer to ~0.07 ms/layer, about a 50x reduction.

**Attribution:** the overall 1.88→2.56 gain is not “just pinned inputs”. Progressive read/compute had already shown 2.537 tok/s with K excluded. The pinned-input fix removed an artificial ~160 ms/token overhead introduced by the first hybrid GPU implementation, allowing the progressive-execution gain to survive in the full hybrid topology.

## 10. Deferred K admission

Persistent K preparation was moved off the current-token critical path:

```text
current routed compute completes
K admission continues on private copy stream
next work proceeds
later finalize event -> expose K ownership -> release L2 owner
```

Residual exposed finalize wait is approximately **0.0022–0.0024 ms per finalize**, only ~1–1.4 ms total across hundreds of finalizations in a 256-token run. K admission H2D is therefore almost completely hidden.

## 11. Final matched 256-token result

Same topology as baseline: K216/R12/P12, L2 8 GiB LRU, L1 SLFU, cold-on, demote-off, FRONT rolling, managed/no-mmap, 256 decode, fresh process per replica.

| replica | tok/s | ms/token | K @256 | L2 @256 | cold @256 |
|---|---:|---:|---:|---:|---:|
| R1 | **2.5609** | 390.48 | 29.51% | 36.11% | 34.38% |
| R2 | **2.5625** | 390.24 | 31.87% | 33.47% | 34.67% |
| R3 | **2.5851** | 386.83 | 25.91% | 43.05% | 31.04% |
| **median** | **2.5625** | **390.24** | — | — | — |

Compared with baseline median 1.880 tok/s / ~532 ms-token:

- throughput: **~+36.3%**
- TPOT reduction: **~142 ms/token**

The locality trajectories differ materially, while throughput is tightly clustered. This argues strongly against a single lucky route trajectory.

Late-window throughput remains >2.5 tok/s; the gain is sustained rather than a startup transient.

## 12. Final profiler after async GPU inputs

Representative 256-token profiles:

```text
R1: total 69.67s | prepare 5.32 | CPU 41.33 | wait_next 16.76 | copy 3.69 | GPU submit .79 | GPU read .96 | join .22
R2: total 69.38s | prepare 5.34 | CPU 40.58 | wait_next 17.56 | copy 3.17 | GPU submit .81 | GPU read 1.11 | join .21
R3: total 68.48s | prepare 4.82 | CPU 41.89 | wait_next 13.57 | copy 5.70 | GPU submit .77 | GPU read .90 | join .22
```

The dominant costs are now real work: CPU expert arithmetic, storage-arrival waiting, and source/promotion copies. Multi-millisecond per-layer host-side GPU submission serialization is gone.

## 13. Relationship with the historical ~140 ms/token Nsight gap

Historical Nsight analysis had identified an exposed host/source wait bucket on the order of ~140 ms/token in a ~527 ms/token regime. The current frozen baseline is ~532 ms/token; the final hybrid result is ~390 ms/token: approximately **142 ms/token recovered**.

We must not claim these are the identical bucket until a fresh Nsight capture proves it. However, the scale matches closely, and the callback profiler independently found a concrete ~166 ms/token-equivalent synchronous GPU-input submission artifact before the pinned-input fix.

Fresh Nsight profiling is therefore a priority: identify which old wait/synchronize region disappeared and what residual stalls remain.

## 14. Interpretation

The mechanism is better described as **locality-aware asynchronous expert execution** than simply “CPU/GPU hybrid”:

1. exploit staggered NVMe completion;
2. compute cold experts immediately on CPU;
3. execute K hits on GPU without movement;
4. promote L2 hits to GPU only when calibrated break-even/backlog justifies it;
5. prepare future K residency asynchronously;
6. commit ownership only after completion;
7. reduce outputs in original router order;
8. avoid tiny synchronous GPU transfers in per-layer control loops.

## 15. Next campaign opened by this result

Three topology questions should now be tested under the new asynchronous execution model.

### A. Historical L2-only / CPU routed compute

Goal: revisit the historical large-L2 regime with the new progressive execution path.

Intent:

- L2 ~18 GiB historical target
- routed expert compute on CPU
- static components on GPU
- concurrent NVMe reads + progressive CPU compute
- hybrid GPU execution only if the calibrated planner has a reason to use it; do not force GPU promotion

Question: how close can a large host expert cache get when the old wait-all penalty is removed?

### B. Static-on-CPU / maximum-K regime

Goal: remove FRONT rolling entirely for this experiment and spend the freed GPU budget on the largest historically safe K envelope.

Intent:

- static components computed on CPU
- maximize K without crossing the known WDDM failure regime
- reduce L2 as needed for the topology
- use locality-aware dynamic routed execution

Question: over **2K decode**, does persistent expert locality from a much larger K outweigh the CPU debt of static components?

This is the clean experiment for the trade:

```text
less static GPU residency / no FRONT logistics
           vs
more persistent routed-expert GPU locality
```

### C. Current FRONT rolling + K256 + L2/L1

Goal: test the current static architecture with a larger K under the new decode pipeline.

Intent:

- DeepSeek4 FRONT rolling stays active
- K256
- L2 LRU
- L1 SLFU
- progressive cold CPU + dynamic L2 CPU/GPU placement

Question: what is the best throughput when the current FRONT design and the new routed execution model are combined with K256?

## 16. Campaign protocol

For **each topology**:

- measure **prefill and decode**
- decode depth: **2K tokens**
- **3 fresh replicas**
- L2 eviction: **LRU**
- L1/K admission: **SLFU**
- use at least **6 CPU threads** for the performance campaign
- preserve exact configuration receipts, route-state histograms and timing checkpoints
- deterministic correctness gate remains separate from throughput/natural workload runs

The historical continuity prompt should remain available as one arm, but the repository-level benchmark should grow to fixed **prose / code / hybrid** workload fixtures so variance is explicit rather than hidden.

## 17. Nsight order

Recommended order before spending the full 3x2K campaign time:

1. one representative run per topology with Nsight Systems;
2. verify no WDDM spill/paging pathology;
3. inspect remaining wait/synchronize buckets and CPU/GPU/NVMe overlap;
4. only then run the full 3-rep 2K qualification.

The Nsight pass should explicitly answer:

- is NVMe wait now overlapped with CPU full-expert work as designed?
- is GPU routed compute overlapping CPU/cold work, or introducing a new serialization?
- how much FRONT H2D is still exposed vs hidden?
- are K admission H2D copies still almost entirely hidden?
- which part of the old ~140 ms/token exposed region has disappeared?
- what residual host-side synchronization remains?
- does 6-thread CPU routed execution steal enough CPU from I/O submission to hurt storage overlap?

## 18. Pre-bake integration

The new runtime policy should not retain workstation-specific constants. Startup/pre-bake should calibrate and publish the cost table used by the layer-local planner:

- CPU expert compute LUT by thread count and expert count
- GPU resident expert compute LUT
- L2→P memcpy LUT
- P→GPU H2D LUT
- NVMe→L2 completion curve 0→6 misses
- optional contention-adjusted/EWMA runtime corrections

At each layer/token/top-6 the planner can then evaluate the measured current state and choose execution placement dynamically.

## 19. Evidence / receipts

Primary current evidence:

- `research/v016-hybrid-locality/baseline-current-20260902.json`
- `research/v016-hybrid-locality/hybrid-locality-calibration-20260902.json`
- `D:\hybrid-asyncinput-continuity-256-r1.json`
- `D:\hybrid-asyncinput-continuity-256-r2.json`
- `D:\hybrid-asyncinput-continuity-256-r3.json`
- corresponding `.err` route/profiler logs
- `D:\siliang-v016-progressive-cpu-*`
- `D:\siliang-v016-l2-arrival-*`

Before release/publication these should be copied into a frozen research-results bundle or immutable branch/artifact so the result can be reconstructed from the evidence rather than from chat history.
