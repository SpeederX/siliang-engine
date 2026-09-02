# Next topology campaign — 2K prefill + decode

Date: 2026-09-02
Status: planned after async-locality 2.5625 tok/s matched result.

## Common protocol

- DeepSeek-V4-Flash 0731 Siliang expert-major GGUF
- `llama-server`, fresh process per replica
- 3 replicas per topology
- decode depth: 2048 tokens
- record prompt/prefill throughput and decode checkpoints 32/64/128/256/512/1024/1512/2048
- L2 eviction policy: **LRU** whenever L2 is present
- L1/K policy: **SLFU** whenever K is present
- `admit-k-cold=on`, `demote-k-hot=off` unless a topology explicitly has no K
- at least **6 CPU threads** for the campaign (`-t 6 -tb 6` as initial fixed performance point)
- `-ngl 99 -ncmoe 43 -nkvo --no-op-offload`
- `--parallel 1`
- managed/no-mmap backing
- route statistics enabled
- async locality runtime enabled
- deterministic correctness gate is separate from the natural performance campaign
- one Nsight Systems profile per topology before full 3x2K qualification

## A — historical L2-only extreme

Purpose: measure the best host-resident expert regime after removing wait-all.

Target topology:

- L2 = **18432 MiB**
- L2 policy = **LRU**
- K=0 / R=0 / P=0
- routed experts computed on CPU
- static model components remain GPU-resident through normal `-ngl 99 -ncmoe 43`
- FRONT roll = `off`
- progressive NVMe->L2 `wait_next` + full-expert CPU compute
- no forced L2->GPU promotion; if a future K-less transient-GPU planner arm is tested, record it as a separate A2 arm

Historical anchor: native-host 18-GiB evidence reported ~2.18463 tok/s at 12 threads. This is only an anchor; the new run must be fresh and matched.

Implementation note: current research dynamic hybrid hook is armed through the K/R/P arena and therefore does not yet represent a pure K=0 topology. A dedicated K=0 progressive L2 graph path is required before this arm is valid.

## B — all-static CPU / maximum K extreme

Purpose: test the opposite resource allocation: remove static GPU logistics/residency and spend the freed VRAM on persistent routed-expert locality.

Target topology:

- **all intended static components explicitly placed/computed on CPU**
- FRONT roll = `off`
- K = maximum capacity that passes the WDDM/VRAM safety gate for this placement
- R/P remain route aligned (start R12/P12)
- L2 reduced to the smallest capacity that avoids pathological cold/page pressure for this topology; do not assume 18 GiB
- L2 LRU + L1 SLFU
- progressive cold CPU + dynamic locality execution

Important: `--expert-cache-roll off` alone does **not** mean “all static CPU”. This arm requires an explicit placement override and a startup receipt listing every static family and its device. Without that receipt the run is invalid.

Historical context: the old static/K frontier showed PB3-output_b + Khot288 as the best partial-displacement point, while q_b/output_a/shared CPU were costly. This campaign intentionally tests a more extreme topology under the new scheduler; it must not be described as the historical winner.

Before 2K, run a VRAM capacity probe under this exact static placement to find the actual K ceiling. Do not infer K from raw freed bytes because WDDM and CUDA workspace change with placement.

## C — current FRONT rolling + K256

Purpose: combine the current static architecture with the new async locality scheduler and a larger K.

Target topology:

- FRONT roll = `deepseek4`
- K256 / R12 / P12
- L2 LRU
- L1 SLFU
- `admit-k-cold=on`, `demote-k-hot=off`
- progressive cold CPU + dynamic L2 CPU/GPU break-even planner
- asynchronous pinned GPU input path
- deferred future K admission

L2 capacity should be explicitly chosen and recorded before qualification. Start from the currently proven 8 GiB arm for clean comparison; a second capacity arm is a separate experiment, not silently folded into C.

## Nsight pre-pass

Run one representative replica for A/B/C before the 3x2K campaign.

Capture questions:

1. residual host/API synchronization after removal of per-layer synchronous `tensor_set`;
2. CPU full-expert compute overlap with NVMe reads;
3. GPU routed work overlap with CPU/cold path;
4. exposed vs hidden FRONT H2D;
5. exposed vs hidden K-admission H2D;
6. WDDM local/shared memory and spill behavior;
7. disk submission latency and whether 6 CPU threads interfere with I/O issue/completion;
8. residual idle gaps between routed layer phases.

A topology proceeds to 3x2K only if correctness passes and the Nsight run is not dominated by paging/WDDM spill or an obvious configuration error.

## Workload suite

For the first topology campaign, preserve the historical technical continuity prompt so the result can be compared with prior evidence. In parallel create three repository fixtures:

- **prose**: long technical explanation / reasoning prose
- **code**: software generation/modification workload
- **hybrid**: code + explanation + technical reasoning

The 3x2K topology comparison should use the same fixed fixture per arm. The broader prose/code/hybrid variance suite can be run after the topology winner is identified, otherwise the experiment count explodes.


## 2026-09-02 update — maintainer profiles and Arm B analytical projection

The maintainer baseline is not one global number. Siliang should retain separate strategy profiles because hardware envelopes differ:

1. **RAM / CPU-oriented** — routed experts primarily host-resident and CPU-computed; GPU may be absent or irrelevant.
2. **Small-GPU hybrid** — RAM remains the main expert store, but transient GPU execution is used when measured finish-time economics justify it; persistent K may be zero/minimal.
3. **Balanced RAM + GPU** — persistent K plus L2 and architecture-specific static logistics, represented on the maintainer RTX 2070 system by the K256/L2/FRONT family.

These are different strategies, not rankings of the same configuration. Performance documentation must report which envelope a number belongs to.

### Arm B: analytical all-static-CPU / max-K projection — corrected after Nsight C probe

**This section supersedes the earlier pre-Nsight projection.** The first calculation added the full CPU-static debt to the current ~390 ms/token hybrid TPOT but failed to subtract the FRONT H2D traffic that disappears when those static weights remain on CPU. That made Arm B incorrectly look much worse than it is.

The retained DS4 static LUT projects the measured static families at:

- CPU static compute: **413.56 ms/token**;
- corresponding GPU-wall compute: **66.37 ms/token**;
- gross CPU-vs-GPU compute debt: **+347.19 ms/token**.

The C Nsight probe on the current K256/L2/FRONT topology identifies the large H2D stream at about **2.8-2.9 GB/token** and ~**232 ms/token of device H2D time**, matching the ~2.699-GiB FRONT payload. Arm B with all static weights CPU-resident and FRONT rolling disabled removes this transfer from the current critical path.

The current v0.1.6 K256 profile accounts ~4685.05 MiB of static CUDA model allocation. Removing the ~155 MiB FRONT double banks as well gives ~4840 MiB that could theoretically be repurposed. At 6.75 MiB/expert-slot payload this buys ~717 additional slots, putting the arithmetic K envelope near **K973**. This is a capacity projection, not a WDDM qualification.

Moving the ~4685 MiB currently GPU-resident static payload into host memory requires giving back roughly the same amount of L2 capacity on the 24-GiB maintainer machine. Starting from the measured 8-GiB / 1210-slot L2, this yields roughly **516 L2 slots (~3.4 GiB)** while K grows from 256 to ~973. Total expert-capacity slot-equivalents therefore stay nearly constant rather than collapsing.

A 250-token natural-route simulation using cumulative-frequency K admission plus exclusive LRU L2 gives:

| topology proxy | K | L2 slots | K hit | L2 hit | cold |
|---|---:|---:|---:|---:|---:|
| current C | 256 | 1210 | ~33.34% | ~33.91% | ~32.76% |
| B K788 | 788 | 678 | ~52.43% | ~15.90% | ~31.67% |
| B K950 | 950 | 516 | ~56.30% | ~12.57% | ~31.13% |
| **B K973** | **973** | **516** | **~56.81%** | **~12.43%** | **~30.76%** |

This is the important result: the hierarchy can shift a large amount of locality from L2 to K **without increasing cold miss rate** in this trace. It directly supports the user's proposed strategy of reducing L2 to buy host room for static weights while using the freed VRAM to maximize K.

K973 captures about 60.6 more routed selections/token than K256. Using the retained ~0.41147 ms compute-only CPU->GPU value per selected expert, this is about **24.9 ms/token** of additional routed-compute benefit.

The corrected first-order Arm-B delta relative to the current ~390.24-ms hybrid reference is therefore:

`+347.19 static CPU-vs-GPU compute debt - ~232 FRONT H2D removed - ~24.9 extra-K benefit ~= +90.3 ms/token`.

That projects to roughly **480.5 ms/token / 2.08 tok/s** before dependency overlap and secondary source/workspace effects. The shared expert is a sibling FFN branch; if most of its ~62.1-ms CPU-vs-GPU debt overlaps routed work, an optimistic bound is about **418.5 ms/token / 2.39 tok/s**.

Therefore Arm B is **not** a throwaway falsification cell. Its plausible range is now approximately **2.1-2.4 tok/s**, and profiling it is justified. The remaining uncertainty is dominated by graph dependency/overlap of the CPU static chain, not K hit-rate arithmetic.

The K curve still flattens: K788 -> K973 buys only ~4.4 percentage points of K hit, around 4.7 ms/token compute-only. Exact K1060 capacity is therefore not strategically important unless the pre-bake finds additional near-zero-debt VRAM victims.

### Static host placement semantics

With `--no-op-offload`, GGML preferentially assigns an operation to the backend of its weight. A static weight placed in an ordinary CPU buffer therefore normally causes that operation to execute on CPU. Host storage is not equivalent to "GPU compute after an automatic cheap transfer". A CUDA-compatible host buffer / explicit offload path is a different topology and must be modeled separately.

A future dynamic static scheduler should separate **weight residency** from **execution placement**, exactly as the routed hybrid now separates persistent K admission from current execution. For each static operation, pre-bake should estimate completion time on CPU versus copy-stream+GPU using the actual dependency window, rather than comparing isolated latencies. This is architecture-sensitive because the graph position, tensor families and overlap windows differ by model.
