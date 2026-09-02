# Siliang v0.1.6 — Async Locality / Maintainer Strategy Qualification Plan

Date: 2026-09-02

Execution handoff for Codex. Read this file completely before changing code or running expensive tests.

## 0. Mission

Continue the v0.1.6 research from the current working tree as-is and qualify the new locality-aware asynchronous expert execution design.

Immediate goals:

1. preserve and verify the current dynamic-hybrid prototype;
2. use **elevated Nsight Systems** to close the remaining synchronization / overlap attribution;
3. qualify three distinct maintainer strategy profiles rather than one universal baseline;
4. measure **prefill and decode at 2K depth**, 3 fresh-process repetitions per qualified strategy;
5. keep correctness as a hard gate, especially around mixed CPU/GPU execution;
6. turn measured hardware costs into pre-bake inputs instead of hard-coded workstation constants.

Current result: matched 256-token decode moved from ~1.880 tok/s to a 3-rep dynamic-hybrid median of **2.563 tok/s (~390.24 ms/token)**. Explain and qualify this before chasing ungrounded tweaks.

---

## 1. Repository / branch / safety

Worktree: `D:\siliang-v016-expert-arrival`

Branch: `research/v016-expert-arrival`

Committed HEAD when written: `c3348998da`

Remote: `origin/research/v016-expert-arrival`

The worktree is intentionally **dirty**. Uncommitted changes are the functioning hybrid/progressive prototype and are valuable research state.

### Git rules

- **DO NOT** reset, clean, checkout-over, rebase, or discard the current dirty worktree.
- Do not modify the original/main Siliang worktree to solve conflicts.
- Before broad edits, inspect `git diff` and preserve WIP via local checkpoint commit on a research sub-branch or external patch/snapshot with exact HEAD/diff identity.
- Do not force-push.
- Do not push experimental runtime code to release branches.

Documentation/evidence checkpoints:

- `d0fc7a590b` — async hybrid locality findings + calibration + baseline
- `6a1def0bb` — maintainer profiles + first max-K projection
- `c3348998da` — corrected max-K projection including avoided FRONT traffic

Current dirty code is expected in GGML CPU/SiliangEM, llama context/graph, `src/siliang-moe-runtime.cpp`, and `research/ds4-hybrid/`. Do not assume uncommitted means disposable.

---

## 2. Runtime / model / machine

Use **`llama-server` only**, not `llama-cli`.

Research server build:

`D:\siliang-v016-hybrid-server-build\bin\Release\llama-server.exe`

Model:

`C:\behemoth\0731\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf`

Maintainer machine:

- RTX 2070 8 GB, WDDM
- Ryzen 5 2600, 6C/12T
- 24 GB DDR4
- Gen3 NVMe
- Windows 11

The machine is not guaranteed clean. PiLink and other apps may remain active. Record RAM and WDDM state for every probe.

---

## 3. Read these findings first

- `research/v016-hybrid-locality/HYBRID_ASYNC_LOCALITY_FINDINGS_20260902.md`
- `research/v016-hybrid-locality/NEXT_TOPOLOGY_CAMPAIGN_2K_20260902.md`
- `research/v016-hybrid-locality/MAINTAINER_BASELINE.md`
- `research/v016-hybrid-locality/hybrid-locality-calibration-20260902.json`
- `research/v016-hybrid-locality/baseline-current-20260902.json`

Historical static/K frontier:

`C:\Users\ciao7\OneDrive\Desktop\progetti\claude\behemoth-loader\siliang-engine-zero-miss-0731\research\moe-arena\DS4_TARGETED_CAMPAIGN_RESULTS_20260824.md`

Historical pre-bake:

`C:\Users\ciao7\OneDrive\Desktop\progetti\claude\behemoth-loader\siliang-engine-zero-miss-0731\research\moe-arena\DS4_TARGETED_PREBAKE_VALIDATION_CAMPAIGN_20260824.md`

250-token policy evidence:

`C:\Users\ciao7\OneDrive\Desktop\progetti\claude\behemoth-loader\siliang-engine-zero-miss-0731\research\moe-arena\DS4_TRACE250_POLICY_PREBAKE_20260824.json`

Route trace:

`C:\Users\ciao7\OneDrive\Desktop\progetti\claude\behemoth-loader\siliang-engine-zero-miss-0731\.agent-local\ds4-activation-arena-250-20260824.route.jsonl`

---

## 4. Proven evidence — do not rediscover blindly

### 4.1 NVMe -> host arrivals are staggered

L2=8 GiB, representative LRU cumulative completion times for six real cold experts:

- 1st ~2.50 ms
- 2nd ~4.39 ms
- 3rd ~6.72 ms
- 4th ~8.95 ms
- 5th ~10.93 ms
- 6th ~11.78 ms

Completion order was consistently submission order in measured runs.

Backend now has `wait_next_expert`.

Implementation lessons:

- never move an in-flight Windows `OVERLAPPED`; stable slots + active bitmap;
- `WaitForMultipleObjects` max is 64; larger prefill/wait-all batches drain in <=64 windows;
- decode top-6 remains true wait-any.

### 4.2 Progressive CPU changes the critical path

Real 6-cold assay, correct AVX/AVX2 build, 4 threads:

- last storage arrival ~11.8 ms
- last expert CPU compute ~0.67-0.70 ms
- progressive wall ~12.5 ms
- frozen M03 correctness exact/roundoff-level

Validated transformation:

`wait-all + compute-all` -> `last-arrival + tail-compute`

Natural CPU-progressive prototype with K intentionally frozen/empty sustained **~2.537 tok/s / 394.1 ms/token over 256 decode tokens**.

### 4.3 Dynamic CPU/GPU locality semantics

Per `(token, layer, top-k route)`:

- K resident -> GPU immediately, no movement;
- L2 resident -> CPU by default, GPU only when calibrated finish-time math reduces critical path;
- cold -> concurrent storage reads, full-expert CPU compute on each completion;
- persistent K admission is separate from current execution placement;
- future K admission copies async, owner becomes K only after completion;
- reduction is always original router dispatch order, completion order irrelevant.

Never regress:

`execution placement NOW != persistent residency for NEXT reuse`.

### 4.4 Correctness status

Frozen M03 routed work unit:

- current CPU vs frozen: ~1e-7 / effectively exact
- current GPU vs frozen: `max_abs ~= 0.01320755`
- CPU vs GPU: same ~0.0132 backend-specific delta

Dynamic split must not add error from arrival/completion/reduction order.

### 4.5 Synchronous GPU input overhead was a major bug

First hybrid implementation reached ~2.16-2.17 tok/s because activation/IDs/weights for the small GPU expert graph used synchronous CUDA `tensor_set` per routed layer.

Profile before fix:

- GPU submit ~10.49 s / 2709 routed-layer calls
- ~3.87 ms/layer
- ~166 ms/token at 43 layers

Inputs are tiny: this was synchronization/host exposure, not bandwidth.

Fix: tiny pinned-host staging + private-stream async H2D + event dependency into GPU graph.

After fix:

- GPU submit ~0.07 ms/layer
- 64-token run ~2.65 tok/s
- 256-token fresh runs: 2.56094 / 2.56253 / 2.58512 tok/s
- median **2.56253 tok/s / 390.24 ms/token**

Old current matched baseline: **1.880 tok/s / ~532 ms/token**.

### 4.6 Deferred K admission is effectively hidden

One-layer deferred K commit residual finalize wait:

`~0.0022-0.0024 ms/finalize`

Do not optimize this tail again unless new evidence contradicts it.

---

## 5. Thread policy

The hybrid full-expert CPU executor is no longer hard-coded to 4 threads. It follows `llama_n_threads(ctx)`, with research override if needed.

Smoke `-t 6 -tb 6` confirmed:

`RESEARCH hybrid CPU threads=6`

Use **at least 6 threads** for this campaign. Future pre-bake chooses worker width from measured LUT/contestion, not constants.

---

## 6. Maintainer baseline = three strategy profiles

Do not collapse these into one score; they correspond to different hardware envelopes.

### Profile A — RAM-dominant / CPU-first

Intent:

- large host expert residency;
- routed compute primarily CPU;
- optional small GPU used as transient accelerator;
- no requirement for large persistent K.

Historical workstation control topology:

- L2=18432 MiB
- L2 policy LRU
- true L2-only: K0/R0/P0
- static components GPU-resident in the historical workstation setup
- FRONT roll off

Historical native-host evidence around 2.18 tok/s at 12 threads exists, but is not current qualification.

Need two A subarms:

**A1** — L2-only + progressive CPU routed compute, no persistent K, no transient GPU routed compute.

**A2** — same L2 source, no persistent K, but dynamic CPU + transient GPU execution. GPU only when finish-time planner justifies it. L2 ownership stays L2; transient GPU compute must not accidentally become K admission.

This is the RAM/small-GPU comparison.

### Profile B — static CPU / reduced L2 / large K

Intent:

- move static weights to host CPU placement;
- disable FRONT rolling;
- use freed VRAM for persistent K;
- reduce L2 so host RAM stays within machine budget;
- shift locality from RAM to GPU.

This requires explicit static placement; `--expert-cache-roll off` alone is not enough.

Current analytical target:

- arithmetic K envelope near **K973**;
- K788 is already near locality plateau and is a safer first probe;
- for K~950-973, L2 budget projection is ~516 slots / ~3.4 GiB if host static payload replaces current GPU static payload.

### Profile C — balanced RAM + GPU

Target:

- K256
- R12/P12
- L2 8 GiB initially
- L2 LRU
- L1 SLFU
- `admit-k-cold=on`
- `demote-k-hot=off`
- `--expert-cache-roll deepseek4`
- bounded routed prefill ON
- `--ctx-checkpoints 0`
- managed `--no-mmap`
- fresh `llama-server`
- `-t 6 -tb 6`

---

## 7. Profile B analytical grounding

Retained static LUT:

- measured static families CPU compute: **413.56 ms/token**
- same families GPU-wall: **66.37 ms/token**
- gross CPU-vs-GPU compute debt: **+347.19 ms/token**

Important correction: all-static-CPU also removes FRONT rolling traffic.

Current C Nsight probe identifies a large H2D stream consistent with ~2.699 GiB FRONT payload and roughly **~232 ms/token H2D device time**.

Current static CUDA model footprint ~4685 MiB; removing ~155 MiB FRONT banks gives ~4840 MiB arithmetic VRAM to repurpose. At ~6.75 MiB/expert slot, K256 -> roughly **K973** by payload arithmetic.

Moving ~4685 MiB static payload to host means reducing L2. From current 1210-slot 8-GiB L2, equal-budget projection leaves ~**516 L2 slots (~3.4 GiB)**.

250-token natural trace simulation with cumulative-frequency K + exclusive LRU L2:

| topology | K | L2 slots | K hit | L2 hit | cold |
|---|---:|---:|---:|---:|---:|
| C-like | 256 | 1210 | ~33.34% | ~33.91% | ~32.76% |
| B | 788 | 678 | ~52.43% | ~15.90% | ~31.67% |
| B | 950 | 516 | ~56.30% | ~12.57% | ~31.13% |
| B | 973 | 516 | ~56.81% | ~12.43% | ~30.76% |
| K curve ref | 1060 | n/a | ~58.61% | n/a | n/a |

Key result: moving capacity from L2 to K does **not** create a cold-miss explosion in this trace.

K973 captures ~60.6 more K hits/token than K256. At retained ~0.41147 ms CPU->GPU selected-expert value, this is ~24.9 ms/token extra routed benefit.

Corrected first-order delta vs current ~390.24-ms C reference:

`+347.19 static CPU debt - ~232 FRONT H2D removed - ~24.9 extra-K benefit ~= +90.3 ms/token`

Central estimate:

- ~480.5 ms/token
- ~2.08 tok/s

Optimistic if most shared-expert CPU debt overlaps routed work:

- ~418.5 ms/token
- ~2.39 tok/s

B deserves a probe. K788 -> K973 buys only ~4.4 K-hit points / ~4.7 ms-token compute-only; do not chase absolute K maximum unless nearly free.

---

## 8. Static host semantics / future static scheduler

With `--no-op-offload`, operations with weights normally prefer the backend of those weights. A static weight in ordinary CPU memory normally means CPU compute.

Do not assume:

`host resident == automatic cheap GPU compute`.

A CUDA-compatible host buffer / explicit offload path is a separate topology.

General rule now established by routed work:

`residency != execution placement`.

Future static scheduler should compare finish time, not isolated latency:

`finish_cpu = max(dependency_ready, cpu_queue_ready) + cpu_compute`

`finish_gpu = max(dependency_ready, copy_queue_ready) + transfer + max(gpu_queue_ready, copy_done) + gpu_compute`

Critical-path exposure after overlap with sibling branches is what matters.

Static planning is architecture-sensitive because model DAG/components differ. Routed expert scheduling is much more architecture-agnostic; pre-bake supplies top-k, shapes and calibrated costs.

---

## 9. Nsight — current non-elevated C evidence

Preferred Nsight:

`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.4.1\target-windows-x64\nsys.exe`

Existing artifacts:

- `D:\nsys-c-k256-hybrid-20260902.nsys-rep`
- `D:\nsys-c-k256-hybrid-20260902.sqlite`
- `D:\nsys-c-k256-hybrid-20260902.stats.txt`
- `D:\nsys-c-k256-hybrid-20260902.json`

Probe C:

- K256
- L2 8 GiB LRU
- L1 SLFU
- R12/P12
- FRONT roll
- routed prefill ON
- ctx-checkpoints 0
- t6/tb6
- 128 decode tokens

Observed:

- prompt ~3.27 tok/s
- decode **~2.545 tok/s / 392.97 ms/token**

Non-elevated warnings:

- WDDM tracing disabled
- CPU context switches disabled
- CPU sampling disabled

Rerun elevated before strong WDDM/CPU-scheduler claims.

### 9.1 CUDA findings to revalidate elevated

Large H2D stream = stream 17:

- ~351 GB H2D in approximate 48-s decode window
- ~28.3 s H2D device-time sum
- consistent with ~2.8-2.9 GB/token FRONT-scale traffic

CUDA Graph execution primarily stream 14.

Approximate decode-window union:

- H2D union ~30.0 s / 48 s
- CUDA Graph execution union ~7.47 s / 48 s
- measured H2D/Graph-Trace intersection: **0**

Do **not** assert “FRONT never overlaps compute” until elevated WDDM/timeline confirms Graph Trace semantics.

### 9.2 Residual synchronization

Non-elevated decode-window CUPTI synchronization:

- stream 14 `STREAM_SYNCHRONIZE`: ~9.655 s / 48 s, 103207 calls
- stream 16 `STREAM_SYNCHRONIZE`: ~0.771 s, 77984 calls
- event synchronize time tiny

At ~2.54 tok/s, stream-14 sync is roughly **~79 ms/token** host-side synchronization tax.

`ggml_backend_cuda_synchronize()` maps directly to `cudaStreamSynchronize(cuda_ctx->stream())`.

Normal server has `cb_eval == nullptr`, so this is not debug per-node eval sync.

Primary attribution hypothesis:

> CPU/GPU split boundaries introduced by the CPU-resident custom hybrid op may force GGML scheduler cross-backend synchronization/copies on the main CUDA stream.

Hypothesis only.

### Elevated Nsight tasks

Rerun C as Administrator with:

- `--trace=cuda,nvtx,wddm`
- CPU sampling/context switches if enabled elevated
- same binary/config/fixture
- 128-256 decode tokens, not 2K

Research-only NVTX ranges are useful if convenient:

- request prefill
- decode token
- per-layer hybrid callback
- FRONT L+1 copy
- K future admission
- CPU progressive wait/compute

If NVTX is inconvenient, use logged monotonic timestamps + SQLite correlation; do not add production dependency only for profiling.

Answer:

1. Is FRONT-sized H2D actually overlapping main GPU graph execution under WDDM?
2. What calls account for ~79 ms/token stream-14 synchronization?
3. Are they scheduler split/copy boundaries from CPU hybrid op?
4. How much GPU idle remains while CPU progressive/NVMe work occurs?
5. Does WDDM show spill, budget contention, preemption or serialization hidden from CUPTI?
6. Is K256 above budget but stable, or throughput-capped by WDDM?

Do not optimize until attribution is solid.

---

## 10. Probe order

### P0 — preserve WIP and reproduce

Before architecture edits:

1. snapshot dirty WIP safely;
2. rebuild product-like research `llama-server`;
3. smoke dynamic hybrid t6;
4. short C control without Nsight, confirm ~2.5+ tok/s class behavior;
5. record binary SHA256 + git/diff identity.

If reproduction fails, stop and debug first.

### P1 — elevated Nsight C

Do elevated C probe first.

Deliver:

- `.nsys-rep`
- `.sqlite`
- stats export
- short attribution markdown
- exact command
- binary hash
- K/L2/cold composition
- WDDM budget/usage observations

### P2 — Profile A plumbing

True A1 must not fake K with K6.

Extend research path so progressive full-expert CPU can run with:

- K0
- R0
- P0 unless transient GPU requires bounded staging
- L2=18432 MiB LRU

A1:

- progressive CPU routed compute
- no persistent K
- no transient GPU routed compute

A2:

- same L2 source
- no persistent K
- dynamic CPU + transient GPU
- GPU only when calibrated finish-time planner selects it
- L2 ownership preserved

For A2 use bounded pinned staging; never reintroduce synchronous `tensor_set`.

18-GiB RAM watchdog:

- capture RAM Available/commit/process WS before startup, model-ready, during request;
- if Available < ~1.5 GiB or pagefile churn becomes severe, stop and classify memory-pressure/non-comparable;
- do not attribute paging slowdown to intrinsic L2-only performance.

Short probes first, 64-256 decode. If viable, one elevated Nsight probe per A subarm.

### P3 — Profile B placement + capacity

Do not start with 2K K1060.

1. Build explicit static placement receipt from actual DS4 inventory/LUT.
2. Verify static CPU means host-backed weights and intended CPU ops.
3. FRONT roll off.
4. Probe candidate points:
   - K788 + corresponding L2 first
   - K950-ish + corresponding L2 if justified
   - K973 only if allocation/headroom sane
5. Startup/capacity + short decode probes only initially.
6. Record CUDA workspace + WDDM budget/usage; static placement can alter workspace.
7. Stop K growth when marginal locality value is smaller than placement/memory cost.

If short B probe is plausibly ~2.1-2.4 tok/s and memory sane, run elevated Nsight.

### P4 — Profile C K256

After residual-sync attribution, re-run short t6 C:

- L2 8 GiB LRU
- L1 SLFU
- K256/R12/P12
- FRONT roll
- routed prefill ON
- ctx checkpoints off

Any sync fix must pass correctness and show it removes, not relocates, the stall.

---

## 11. 2K qualification campaign

Only profiles passing short probe + correctness + memory sanity proceed.

### Policies

Always:

- L2 eviction = **LRU**
- L1/K admission = **SLFU** when K exists

Never substitute LFU for L2 in this campaign.

### Threads

Primary qualification:

- `-t 6`
- `-tb 6`
- hybrid executor must report 6 threads

Thread sweep is separate calibration, not mixed into topology comparison.

### Fresh process

Every repetition starts a fresh `llama-server`.

No warm-up route replay/frozen-route performance tricks.

No server left active between arms.

### Measure prefill and decode separately

Use isolated workloads so large prefill does not unintentionally seed decode locality.

#### P2K — prefill

- fixed 2048-token prompt fixture
- `n_predict=1`
- collect prompt eval tok/s/wall + H2D/cache/path telemetry
- context large enough for no truncation/checkpoint artifacts
- bounded K prefill: disable server context checkpoints when applicable

#### D2K — decode

- fixed short continuity prompt
- `n_predict=2048`
- collect cumulative/windowed tok/s at 256/512/1024/1512/2048 where feasible
- collect locality composition over time

A 32-token prompt is not a “2K prefill”. If a profile structurally cannot support 2K prompt prefill, document it and use largest matched supported depth.

### Repetitions

For every qualified profile:

- P2K x3 fresh processes
- D2K x3 fresh processes

Report median/min/max and route/memory state for every rep, not best run.

Primary matrix:

- A1 — RAM/L2, CPU progressive
- A2 — RAM/L2, CPU + transient GPU dynamic
- B — static CPU + reduced L2 + large K, if probe qualifies
- C — K256 + L2 + FRONT balanced

The maintainer baselines are strategy families; A1/A2 decide the RAM/small-GPU family behavior.

---

## 12. Required metrics

### Identity

- git HEAD
- dirty diff identity/checkpoint
- binary SHA256
- model identity
- exact command/environment
- date/time

### Performance

- prompt tokens/ms/tok-s
- decode tokens/ms-token/tok-s
- windowed decode speed

### Locality

- K hit %
- L2 hit %
- cold %
- K admissions/evictions
- L2 evictions/rejections
- R executions
- CPU executions
- planner composition if available

### I/O / transport

- storage bytes read
- storage blocked/wait
- per-expert completion where relevant
- H2D ops/bytes
- FRONT H2D bytes
- expert H2D bytes
- copy-stream exposed wait

### CPU

- actual hybrid threads
- CPU compute time
- wait-next time
- ready queue depth if instrumented
- utilization/scheduling under elevated Nsight

### GPU / WDDM

- local VRAM budget/usage
- shared/spill
- CUDA workspace
- H2D stream occupancy
- main-stream graph execution
- stream/event synchronizations
- GPU idle gaps

### RAM

- physical Available
- commit
- process private bytes
- working set
- page faults/pagefile pressure

---

## 13. Correctness gates

Performance never qualifies a topology alone.

### Frozen micro gate

Use retained M03 routed work unit/frozen output.

For planner execution compositions:

- CPU-only remains frozen/roundoff-equivalent;
- GPU may reproduce known ~0.01320755 endpoint delta;
- mixed CPU/GPU must not exceed delta explained by GPU branch;
- completion order must not alter result;
- reduction always original router dispatch order.

Prefer an exhaustive/representative sweep of feasible top-6 execution compositions.

### Natural deterministic gate

Temp=0 + fixed seed. Compare:

- reference
- progressive CPU
- dynamic hybrid

Distinguish expected CUDA numerical divergence from race/order nondeterminism.

Require repeated fixed-plan output/hash determinism.

### Fail closed

- no silent managed-source fallback to resident model bytes;
- no K ownership before copy completion;
- no persistent K/L2 duplication except transient P/R semantics.

---

## 14. Pre-bake direction

Do not productize workstation thresholds.

### Routed measurements

- CPU full-expert cost vs thread count
- GPU resident expert compute
- L2 -> pinned staging
- pinned -> GPU H2D
- cold arrival curve 0..top-k misses
- storage concurrency
- current CPU/GPU queue pressure

### Static measurements per actual architecture/tensor family

- bytes
- frequency/token/layer
- CPU latency
- GPU latency
- transfer latency
- backend compatibility
- DAG predecessors/successors
- sibling overlap windows
- residency memory cost

Runtime chooses predicted finish time/locality; model-specific pre-bake supplies graph-aware costs. This is how the mechanism generalizes beyond DS4.

---

## 15. Stop rules

Stop / do not qualify if:

- WDDM spill or paging invalidates comparison;
- RAM Available collapses;
- topology silently falls back;
- managed-source correctness fails;
- K/L2 ownership invariant fails;
- nondeterminism exceeds known backend numerical behavior;
- K requires unsafe WDDM headroom just to start;
- profiler changes the topology enough to make speed meaningless.

Do not run 2K repetitions after a short probe already proves structural no-go.

---

## 16. Deliverables

Create under:

`research/v016-hybrid-locality/qualification-20260902/`

Suggested:

```text
manifest.json
README.md
nsys/
  C-balanced/
  A1-l2-cpu/
  A2-l2-hybrid/
  B-static-cpu-maxk/
probes/
p2k/
d2k/
correctness/
receipts/
```

Every result gets machine-readable receipt + short human interpretation.

Update `HYBRID_ASYNC_LOCALITY_FINDINGS_20260902.md` only with obtained evidence.

Checkpoint causal documentation separately from experimental code where practical.

---

## 17. Immediate Codex sequence

1. Read all retained evidence listed in section 3.
2. Inspect the dirty diff and preserve it safely before broad edits.
3. Rebuild the product-like research `llama-server`; reproduce a short C t6 dynamic-hybrid control in the ~2.5+ tok/s class.
4. Rerun **C under elevated Nsight Systems 2026.4.1** with WDDM and CPU tracing enabled.
5. Attribute the ~79 ms/token main-stream synchronization tax and verify/refute FRONT-H2D vs CUDA-Graph overlap.
6. Do not optimize until attribution is solid.
7. Implement the true K0 progressive path for A1.
8. Implement and qualify K0 transient-GPU dynamic execution for A2 without persistent K.
9. Run A1/A2 short probes with RAM watchdog; run elevated Nsight if viable.
10. Build the explicit B static-placement receipt; probe K788 first, then ~K950 only if evidence justifies it.
11. Run elevated Nsight on B if viable.
12. Run correctness gates across planner execution compositions.
13. Run P2K/D2K x3 only for profiles that pass probe, correctness, and memory gates.
14. Produce a final cross-profile report. **Do not declare one universal winner; report the best strategy per hardware envelope.**

## 18. Definition of done

This phase is done only when we have:

- a preserved/reproducible hybrid WIP identity;
- elevated Nsight attribution for C residual synchronization and FRONT overlap;
- valid A1 and A2 short probes, or explicit no-go evidence;
- an explicit/calculated B placement plus at least one grounded short probe, or a justified analytical no-go;
- correctness evidence for mixed CPU/GPU execution and canonical reduction;
- 3x P2K + 3x D2K for every strategy that qualifies;
- machine-readable receipts and a concise causal report explaining **why** each result happened.
