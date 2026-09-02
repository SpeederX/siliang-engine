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
