# Runtime configuration

Siliang Engine v0.1.3 configures its expert memory hierarchy through typed
command-line options. The same options appear in `llama-cli --help` and
`llama-server --help`. The expert cache is opt-in: omit the options, or pass
`--no-expert-cache`, to use the ordinary model path.

There is no process-environment setup step. Put the complete configuration on
each command line so a log or benchmark receipt can identify the requested
path without hidden session state.

## Options

| Option | Meaning |
| --- | --- |
| `--expert-cache` / `--no-expert-cache` | Enable or disable the typed hierarchy. Enabling requires a nonzero L2 or L1 tier. |
| `--expert-cache-l2-mib N` | Managed host L2 capacity in MiB. The requested value is exact and is not silently resized. |
| `--expert-cache-l2-policy POLICY` | L2 policy: `lru`, `lfu`, `slfu` (`cumulative-lfu` legacy alias), or `wtinylfu-w10-slru-p80`. L2 SLFU admits a candidate only when its lifetime frequency beats the coldest resident victim; rejected candidates are served from bounded current-request host scratch and do not become persistent L2 residency. `wtinylfu` is an accepted short spelling. |
| `--expert-cache-l1-k N` | Total persistent CUDA L1 policy budget K, in expert slots. Homogeneous models share it globally; heterogeneous models partition it across routed layers. |
| `--expert-cache-exchange-r N` | CUDA exchange capacity R per schema arena, in expert slots. Homogeneous physical capacity is K + R. |
| `--expert-cache-elevator-p N` | One global pinned-host elevator ring P, in expert slots, sized to the largest expert schema. |
| `--expert-cache-l1-policy POLICY` | L1 policy: `slfu` (Siliang lifetime-frequency admission/bypass; `cumulative-lfu` is a legacy alias), always-admit `lfu`, or W-TinyLFU W10/SLRU-P80. L1 LRU is retired because it scan-thrashes when routed reuse distance exceeds K. |
| `--admit-k-cold on|off` | SLFU only. `on` permits first-use cold experts to enter K; `off` keeps first-use cold experts in L2/R and only considers them for K on a later L2 hit. Default: `on`. |
| `--demote-k-hot on|off` | SLFU only. `on` defers K replacement until routed compute finishes, then swaps the L2 candidate into K and demotes the displaced K victim into the candidate's released L2 slot. No persistent K/L2 duplication. Default: `off`. |
| `--expert-cache-roll MODE` | Static rolling mode: `off` or `deepseek4`. `deepseek4` controls only the architecture-specific FRONT slab; it is independent of the generic routed-expert K/R/P arena. |
| `--expert-cache-prefill` / `--no-expert-cache-prefill` | Enable or disable experimental routed-MoE batch-union prompt processing in K. The current bitmap supports up to 256 experts per layer and the worst-case union must fit every layer-local K slice. Disabled by default. |
| `--expert-cache-memory-report` / `--no-expert-cache-memory-report` | Enable or suppress periodic host-memory reporting. |
| `--expert-cache-route-stats` / `--no-expert-cache-route-stats` | Emit aggregate decode-route residency/execution statistics at shutdown, including L1/L2/uncached and K/R/CPU execution composition histograms. Requires L1 K/R/P. The explicit telemetry is written to stderr even at normal CLI verbosity. Disabled by default. |
| `--expert-cache-deferred-wait` / `--no-expert-cache-deferred-wait` | Enable or disable deferred L2 I/O waits. |

L1 does not exist as a standalone K allocation. If K is nonzero, both R and P
must be nonzero. K must be at least the model top-k; R and P must each be at
least twice top-k and must be exact multiples of top-k. L1 currently requires
one serial, non-speculative sequence. Use `--parallel 1` with `llama-server`.
The current route callback has no sequence identity and treats more than one
decode route in a graph as prompt work; K, R, P, phase, and event state are
also context-global. Multiple server slots therefore fail closed instead of
sharing unsafe residency state. L2-only configurations do not have this limit.
LoRA adapters are not supported while L1 K/R/P is enabled; startup and dynamic
adapter requests fail closed instead of silently serving the base model.

For a heterogeneous model, K is divided into balanced per-routed-layer slices;
it must therefore be at least routed-layer-count times top-k. If the model has
B distinct expert schemas, the slot-equivalent device count is K + B*R, not
K + R. Actual bytes are schema-dependent: each bank allocates its local K plus
R using that schema's expert size. The resolved runtime logs every bank before
serving a request.

`--expert-cache-roll deepseek4` is accepted only for the validated DeepSeek4
shape: 43 routed layers, 256 experts per layer, and top-k 6. v0.1.3 rolls the
DeepSeek4 FRONT set only. This is deliberately separate from the routed-expert
arena: Gemma/Qwen/Ornith use the same generic K/R/P mechanism with roll `off`.
The FRONT source store uses ordinary committed host memory that is registered
read-only with CUDA after population, matching the qualified research topology.
It does not require one multi-gigabyte `cudaMallocHost` allocation. The small
P elevator remains a separate pinned-host allocation.

`--expert-cache-prefill` is topology-gated rather than architecture-name-gated.
Startup requires a routed MoE with at most 256 experts per layer and
`min(n_ubatch * top_k, expert_count)` must fit the K slice available to every
routed layer after schema-bank partitioning. Unsupported capacities fail closed.
This does not make prefill a performance-qualified preset for every supported
model; it only removes the DeepSeek-only implementation constraint.

## DeepSeek4 requalification prototype

This is the configuration transferred from the bounded P12 research path:

```powershell
& "<build-directory>\bin\Release\llama-server.exe" `
    -m "<deepseek4-expert-major.gguf>" `
    -ngl 99 -ncmoe 43 -nkvo --no-op-offload `
    -c 8192 -b 512 -ub 512 -t 2 -tb 2 `
    --parallel 1 `
    --expert-cache `
    --expert-cache-l2-mib 12288 `
    --expert-cache-l2-policy lfu `
    --expert-cache-l1-k 216 `
    --expert-cache-exchange-r 12 `
    --expert-cache-elevator-p 12 `
    --expert-cache-l1-policy cumulative-lfu `
    --expert-cache-roll deepseek4 `
    --reasoning off --reasoning-format deepseek `
    --host 127.0.0.1 --port 8080
```

The 12, 14, and 16 GiB L2 choices are `12288`, `14336`, and `16384` MiB.
Under the source research runtime, one complete natural 2,000-token start per
capacity measured 2.12587, 2.20073, and 2.28674 tok/s respectively. Those are
reliable completed observations, but not a newly qualified v0.1.3 ranking:
each arm has one start and the generated work differed. The port has passed the
minimal wiring smoke below, but still needs longer deterministic validation and
fresh interleaved timing before those capacities are compared.

### Current v0.1.3 port smoke

On 2026-08-26, a local Windows CUDA control/candidate smoke used greedy
two-token generation, `-c 2048`, L2=2048 MiB, K216/R12/P12, L2 LFU, the L1
cumulative-LFU admission/bypass policy then spelled `lfu`, and `deepseek4`
rolling. The candidate produced the same rendered two-token
continuation as `--no-expert-cache`. Its telemetry resolved the typed
configuration, armed all 43 routed layers and the FRONT slab, served route
uploads and a compute wait, recorded FRONT copy/wait activity, and reported
zero path failures.

The runtime log reached the configured token limit. A controller disconnect
prevented retention of the CLI wrapper exit code. This is wiring and
correctness-path smoke evidence only: the two-token timings are not a
benchmark, 2048 MiB is not a recommended capacity, and this run does not
requalify the 12, 14, or 16 GiB research observations.

For a CLI smoke, use the same cache and placement options with `llama-cli`:

```powershell
& "<build-directory>\bin\Release\llama-cli.exe" `
    -m "<deepseek4-expert-major.gguf>" `
    -ngl 99 -ncmoe 43 -nkvo --no-op-offload `
    -c 8192 -b 512 -ub 512 -t 2 -tb 2 `
    --expert-cache `
    --expert-cache-l2-mib 12288 `
    --expert-cache-l2-policy lfu `
    --expert-cache-l1-k 216 `
    --expert-cache-exchange-r 12 `
    --expert-cache-elevator-p 12 `
    --expert-cache-l1-policy cumulative-lfu `
    --expert-cache-roll deepseek4 `
    -p "<prompt>" -n 128
```

Use `--no-expert-cache` for the explicit control. Do not add tier options to a
disabled command.

## Experimental routed-MoE microbatch prefill

This path is opt-in and model-structure-derived. It is an adaptation to the bounded
K/P/R hierarchy, not the full-layer double buffering described by the
[FreeToken paper](https://arxiv.org/abs/2608.16157). That method needs two whole
expert layers resident at once. On the 8 GB RTX 2070 test system, two DS4 expert
layers would require about 3.38 GiB before other model allocations, which does
not fit the observed headroom.

For each routed layer and prompt microbatch, Siliang deduplicates the selected
experts and gives every member of that union a stable transient slot in K for
the gate, up, and down operations. P stages admissions in bounded waves.
L2-to-K promotion is exclusive: once the P-staged copy reaches K, its managed
L2 slot is released. P staging and R bypass are inclusive with L2; R remains a
decode-only exchange tier. The transient prefill K state is invalidated across
the prompt/decode transition, after which decode repopulates K under the
selected L1 policy.

Router scoring and the selected mixture weights remain on the GPU. The CPU
mapper receives only the contiguous selected-expert IDs needed to form the
union, manage K/L2, and translate logical experts to physical slots. Removing
the weight tensor from that callback does not change top-k selection,
normalization, or the GPU weighted sum of expert outputs.

Each fully mapped routed-MoE sweep also produces one 256-bit expert bitmap per routed
layer. The info summary reports `prefill_bitmap` completed sweeps, sweep tokens,
adjacent-sweep comparisons per layer, seeded experts, needed experts, overlap,
new experts, unused seed experts, coverage, precision, resets, and incomplete
sweep sequences. The legacy `prefill_tokens` field is summed once per mapped
layer; `sweep_tokens` is summed once per completed routed-layer sweep. Debug
logging emits the four raw 64-bit words for every mapped layer and sweep. The
common startup graph-reservation and optional warmup traces, together with
their prefill counters, are discarded before serving begins. `llama-server`
establishes the boundary again after its capability and slot probes. Raw reset,
layer, completion, and incomplete records carry a
monotonic context epoch and attempt number so a parser can distinguish warmup,
serving, retries, and later phase-change scopes even when the completed-sweep
ordinal restarts. Summary counters aggregate from the serving reset across
later phase epochs, while `current_epoch` identifies the last active epoch. A
partial mapped sweep is counted as incomplete on an order mismatch, reset, or
runtime shutdown. These bitmaps are context-sweep telemetry, not HTTP request
IDs.

v0.1.3 does not use the previous bitmap to prefetch, admit, evict, or fill L1.
Rolling L2 lookahead remains conditional on this telemetry showing that saved
wait exceeds wrong-read and eviction cost. The ordinary L1/L2 exclusivity and
the P/R overlap exception are unchanged.

Admission is deliberately conservative. Startup uses the worst-case union
`min(n_ubatch * top_k, expert_count)` even though real token routes may overlap,
and requires that union to fit each routed layer's local K window. For a
homogeneous schema the full K budget is shared; for heterogeneous schema banks,
K is partitioned across routed layers before this check. DS4 K216/top-k 6 still
permits at most `-ub 36`; `-ub 32` leaves a 24-slot margin. The same rule applies
to Gemma/Qwen/Ornith without an architecture-name allowlist. The current bitmap
caps this path at 256 experts per layer. Unsupported geometry or capacity fails
closed.

Use this bounded diagnostic command on the current 8 GB test machine:

```powershell
& "<build-directory>\bin\Release\llama-server.exe" `
    -m "<deepseek4-expert-major.gguf>" `
    -ngl 99 -ncmoe 43 -nkvo --no-op-offload `
    -c 2048 -b 512 -ub 32 -t 2 -tb 12 `
    --parallel 1 `
    --expert-cache `
    --expert-cache-l2-mib 2048 `
    --expert-cache-l2-policy lfu `
    --expert-cache-l1-k 216 `
    --expert-cache-exchange-r 12 `
    --expert-cache-elevator-p 12 `
    --expert-cache-l1-policy cumulative-lfu `
    --expert-cache-roll deepseek4 `
    --expert-cache-prefill `
    --no-warmup `
    --reasoning off --reasoning-format deepseek `
    --host 127.0.0.1 --port 18081 --no-webui
```

There are two distinct controls. Replace `--expert-cache-prefill` with
`--no-expert-cache-prefill` while retaining `-ub 32` to isolate the new path.
Then compare both with the operational `-ub 512` configuration with prefill
disabled. Keep `-tb 12` in every arm: a separate completed prompt reached 4.57
tok/s with twelve batch threads versus 1.87 tok/s with two on a different
prompt, making batch-thread count a critical control. Smaller microbatches may
repeat expert admissions and increase total storage traffic, so this mechanism
is not a throughput claim until fixed-prompt output equivalence and fresh-start
A/B timing pass.

## Pi and the OpenAI-compatible server

`llama-server` exposes an OpenAI-compatible API. Keep the DS4 server command
above running, then point Pi at:

```text
Base URL: http://127.0.0.1:18081/v1
API key:  any nonempty local placeholder if the client requires one
Model:    the model id returned by GET /v1/models
```

A local one-token streaming request completed `POST /v1/chat/completions` with
HTTP 200 and clean content while using `--reasoning off --reasoning-format
deepseek`. This specifically covers a length limit immediately after the
generation prompt: without `--reasoning off`, DS4 can stop after an incomplete
`<think>` prefix and the structured parser can turn that limit into HTTP 500.
Keep the DeepSeek response parser enabled so Pi can receive structured tool
calls. `--skip-chat-parsing` is a content-only fallback, not the recommended
agent configuration. The debug log can still identify the working parser as
`peg-native`; that label is expected and is not the previous parse failure.
The controller stopped its owned server process after the request.

Pi 0.84.3 reserves 4,096 tokens when selecting an output allowance. With an
honest 2,048-token model window, that calculation clamps ordinary requests to
`max_tokens: 1` even when `maxTokens` is larger. For bounded fresh-session
testing, override the emitted request value in the model entry:

```json
"contextWindow": 2048,
"maxTokens": 512,
"samplingParams": {
  "max_tokens": 128
}
```

`samplingParams` is merged after Pi's calculated fields. This workaround does
not create context capacity: prompt plus generated tokens must still fit 2,048.
For a normal growing agent session, raise both llama-server `-c` and Pi's
`contextWindow` to the same value above Pi's 4,096-token reserve instead of
claiming capacity the server does not have.

The hierarchy is context-owned, so `--parallel 1` is part of the v0.1.3 L1
contract rather than only a benchmark preference. The server rejects auto or
multi-slot parallelism before model loading when L1 is requested. Separate
contexts or processes each allocate their own L2/K/R/P hierarchy; budget them
independently.

## Cross-model trial configurations

These commands transfer measured research settings into the new interface.
They are starting points for correctness and allocation checks, not universal
presets. Run a `--no-expert-cache` control first, preserve the exact model hash
and generated tokens, and do not publish a speed claim without at least three
fresh interleaved starts per arm.

Gemma4 26B-A4B QAT used resident host tensors, top-k 8, 30 routed layers, and a
homogeneous schema. Do not add a redundant host L2:

```powershell
& "<llama-server.exe>" -m "<gemma4-26b-a4b.gguf>" `
    -ngl 99 -ncmoe 30 -c 2048 -b 512 -ub 512 -t 4 -tb 4 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1440 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --no-expert-cache-prefill
```

Qwen3-30B-A3B used resident host tensors, top-k 8, 48 routed layers, and
heterogeneous schema banks. K1440 is the conservative transfer point; the
research-only bank-local K1968 result must not be assumed to fit or perform the
same way in v0.1.3:

```powershell
& "<llama-server.exe>" -m "<qwen3-30b-a3b.gguf>" `
    -ngl 99 -ncmoe 48 -c 4096 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1440 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --no-expert-cache-prefill
```

Ornith-1.0-35B uses the same `qwen35moe` architecture family as Qwen3.6 but had
a distinct positive resident-source result. Its retained decode bring-up point
was K1920 (48 slots per routed layer); it is a requalification starting point,
not a universal preset:

```powershell
& "<llama-server.exe>" -m "<ornith-1.0-35b.gguf>" `
    -ngl 99 -ncmoe 40 -c 4096 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1920 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --no-expert-cache-prefill
```

Qwen3.6-35B-A3B is a no-go for the transferred K1440 topology. The retained
three-start observation was slower than its stock control, and the model has
heterogeneous expert schemas. Keep it as a control in v0.1.3:

```powershell
& "<llama-server.exe>" -m "<qwen3.6-35b-a3b.gguf>" `
    -ngl 99 -ncmoe 40 -c 4096 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --no-expert-cache
```

GPT-OSS 120B uses the bounded managed host source rather than a redundant copy
of resident tensors. The v0.1.3 runtime gate keeps it as an L2-only diagnostic;
the prior K216 transport study does not qualify the generic L1 path. Prompt
union and rolling are disabled because both mechanisms are DeepSeek4-specific:

```powershell
& "<llama-server.exe>" -m "<gpt-oss-120b-expert-major.gguf>" `
    -ngl 99 -ncmoe 36 -nkvo --no-op-offload `
    -c 4096 -b 512 -ub 512 -t 12 -tb 12 `
    --expert-cache `
    --expert-cache-l2-mib 18432 `
    --expert-cache-l2-policy lru `
    --expert-cache-roll off `
    --no-expert-cache-prefill
```

Keep `--no-expert-cache-prefill` in the retained GPT-OSS gate command. The prior
GPT-OSS result selected pinned staged overlap in the research binary; it does
not qualify v0.1.3 until
the same artifact passes output equivalence, source-armed telemetry, allocation
checks, and fresh-start timing. Generic prefill is now capability-gated, but a
GPT-OSS prefill trial still needs to satisfy the current expert-count and
layer-local K bounds before it can arm.

The generic schema-based L1 runtime does not architecture-block GPT-OSS. A
manual K/R/P bring-up may therefore use serial mode, keep the DS4 FRONT feature
off, and leave generic prefill disabled until separately qualified:
`--parallel 1`, `--expert-cache-roll off`, and `--no-expert-cache-prefill`.
That is an experimental two-tier trial, not the
frozen L2-only gate profile or a qualified performance preset.

M4 remains inconclusive and is a no-go for v0.1.3: there is no useful integrated
implementation of that architecture to transfer. DirectStorage rolling,
whole-L2 CUDA registration, route replay/oracles as product inputs, and pre-bake
also remain outside this release.

## Memory sizing

L2 reserves and commits real system memory. Homogeneous models reserve K + R
CUDA slots; heterogeneous models reserve K plus one R tail per schema, with
schema-dependent byte sizes. P reserves one global pinned-host ring sized to the
largest expert schema. Leave headroom for Windows, non-expert model
weights, KV cache, CUDA workspaces, other applications, and WDDM pressure.
Larger values are not automatically faster. If the requested path falls back,
the run is void for performance comparison.
