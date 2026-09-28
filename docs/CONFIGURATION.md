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
| `--expert-cache-hybrid` | Experimental, decode only. Experts the L1 policy would not admit to K may run on CPU from L2 instead of crossing P and R. A cost table chooses, for every route, how many L2-resident and uncached bypass experts stay on CPU. Predicted K admissions always go to GPU. Requires K and a nonzero L2. Disabled by default. |
| `--expert-cache-hybrid-cost CPU_L2,CPU_MISS,STAGE_L2,STAGE_PINNED,STAGE_MISS,GPU` | Per-expert cost table coefficients in microseconds for `--expert-cache-hybrid`: CPU compute of an L2 hit, CPU compute of an uncached expert, host staging of an L2 hit through P, host submission of a pinned L2 hit, host staging of an uncached expert, and H2D plus GPU compute of a staged expert. Default: `350,650,400,40,900,170`, measured on Qwen3.8 Flash Next with a Ryzen 5 2600 and an RTX 2070. |
| `--expert-cache-l2-pinned-mib N` | Experimental. Registers the leading N MiB of the L2 arena with CUDA. Decode and bounded-prefill copies of experts resident there go straight to K or R without the P memcpy (a 900-token prompt increment on Qwen3.8 Flash Next went from about 17.6 to 15.8 s); each such L2 slot stays leased until its copy completes. Startup fails if CUDA refuses the registration. Requires K and at most the L2 size. Default: 0. |
| `--expert-cache-l1-k-decode N` | Experimental. Total K slots during decode, above `--expert-cache-l1-k`. A large prefill ubatch needs a large CUDA compute buffer, but a decode graph uses only a few MiB of it; the extra slots are mapped onto the end of that buffer with CUDA virtual memory. Bounded prefill evicts them and runs with the `--expert-cache-l1-k` slots, so that value must still hold one prefill route union. Each decode graph is checked before an extra slot is used: if its plan would reach the lent region, the extra slots are dropped with a warning. Requires `--expert-cache-prefill`, one expert schema, and CUDA VMM. Default: 0 (same as K). |
| `--expert-cache-prefill-layer-major` | Experimental, qwen4exp. A prompt longer than one ubatch runs every ubatch through a layer before the next layer starts, so each layer's routed experts cross the host-to-GPU link once per prompt instead of once per ubatch. The residual stream of the whole prompt is held in host memory (40 KB per token on Qwen3.8 Flash Next). The output is identical to the ubatch-major path for the same ubatch split. A prompt reaches it only when one decode call carries it, so set `-b` to at least the prompt length. Requires `--expert-cache-prefill`. Disabled by default. |
| `--expert-cache-staging-threads N` | Host threads, the mapping thread included, that copy each expert from L2 into the pinned P ring before its H2D copy. Requires K. Range 1-16, default: 1. |
| `--expert-cache-prefill-l2-retain` | Bounded prefill keeps each expert it copies into K in L2 as well, instead of the exclusive release decode uses. K is transient during prefill, so without it the next ubatch reads the expert from disk again. Pair it with the `slfu` L2 policy so one prefill pass does not flush L2. Requires `--expert-cache-prefill` and a nonzero L2. Disabled by default. |
| `--expert-cache-verify` | Diagnostic and slow. After each decode route, samples the device bytes of every routed K/R slot, compares them with the model bytes of the expected expert, and logs mismatches with the expert whose bytes the slot actually holds. Requires K and model-mapped experts. Disabled by default. |
| `--expert-cache-exchange-r N` | CUDA exchange capacity R per schema arena, in expert slots. Homogeneous physical capacity is K + R. |
| `--expert-cache-elevator-p N` | One global pinned-host elevator ring P, in expert slots, sized to the largest expert schema. |
| `--expert-cache-l1-policy POLICY` | L1 policy: `slfu` (Siliang lifetime-frequency admission/bypass; `cumulative-lfu` is a legacy alias), always-admit `lfu`, or W-TinyLFU W10/SLRU-P80. L1 LRU is retired because it scan-thrashes when routed reuse distance exceeds K. |
| `--admit-k-cold on|off` | SLFU only. `on` permits first-use cold experts to enter K; `off` keeps first-use cold experts in L2/R and only considers them for K on a later L2 hit. Default: `on`. |
| `--demote-k-hot on|off` | SLFU only. `on` defers K replacement until routed compute finishes, then swaps the L2 candidate into K and demotes the displaced K victim into the candidate's released L2 slot. The demotion path avoids creating a second L2 copy for an already-resident victim. Default: `off`. |
| `--expert-cache-roll MODE` | Static rolling mode: `off` or `deepseek4`. `deepseek4` controls only the architecture-specific FRONT slab; it is independent of the generic routed-expert K/R/P arena. |
| `--expert-cache-prefill` / `--no-expert-cache-prefill` | Enable or disable experimental routed-MoE batch-union prompt processing in K. The current bitmap supports up to 512 experts per layer and the worst-case union must fit every layer-local K slice. Disabled by default. |
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

Per-layer K slices are usually too small for bounded prefill, whose route union
can reach the full expert count. A file whose routed layers all use one expert
schema gets one K shared by every layer. Qwen3.8 Flash Next UD-Q2_K_XL
quantizes layer 2 gate and up as IQ3_XXS and every other layer as IQ2_XS; the
expert-major converter's `--replace-part` option can swap in layer-2 parts
requantized to IQ2_XS, which gives a single-schema file.

`--expert-cache-hybrid` changes where decode experts run, not which experts run.
For each route of each routed layer, K hits stay on GPU and predicted K
admissions cross to GPU as before. The remaining bypass experts are split by
residency (L2 hit or uncached), and a precomputed cost table indexed by those
two counts and the forced-GPU count gives how many of each class stay on CPU.
The predicted layer time is the host staging of every expert that crosses to
GPU plus the larger of CPU compute and H2D-plus-GPU compute: the CPU branch
runs in the host step that maps the route, while the H2D copies it started
proceed on the private copy stream. Ties keep the expert on GPU. The lowest-ranked candidates are
the ones moved to CPU, and at least one route slot always stays on GPU.

CPU experts read L2 directly, and uncached ones are read with the same
overlapped I/O as an L2-only configuration. Their K, P, and R state is
untouched. The graph computes both branches over full route rows. Each branch
yields exact zeros for the other's slots, and they are added before the
ordinary weighted reduction, so the reduction order is unchanged. CPU and GPU
kernels quantize activations differently, so hybrid output is not bit-identical
to the all-GPU path. Evaluate divergence before relying on it. Prompt
processing never uses the CPU route. The shutdown `decode_host` line reports
host staging time per decode map and per staged expert, which is the
measurement the staging coefficients should come from.

`--expert-cache-l1-k-decode` separates the prefill and decode K sizes. Prefill
throughput grows with the ubatch, and a 4096-token ubatch needs about 2.2 GB of
CUDA compute buffer at 32k context, which leaves room for only a small K next to
it. Decode wants a large K but runs one token at a time, and its graph plan
uses only a few MiB of that buffer. With this option the compute buffer is
allocated with CUDA virtual memory and its end is a separate physical block per
expert part; the arena maps the same blocks as extra K slots in front of the
base slots. Every phase change already invalidates K, so the extra slots never
carry prefill state into decode; entering decode zeroes them, and a decode
admission into an extra slot keeps the expert's L2 copy because the next prefill
will drop the slot. When the scheduler reallocates the compute buffer for a
larger graph, the new buffer maps the same end blocks, so the extra slots stay
valid. On Qwen3.8 Flash Next with an RTX 2070 (32k context, 29k-token prompt,
f16 KV cache on GPU): ubatch 3072 with `--expert-cache-l1-k 512
--expert-cache-l1-k-decode 1440` prefilled at 88 tok/s and decoded at 7.0 tok/s
after the prompt, against 102.8 and 2.8 tok/s for ubatch 4096 with K 512 and
`-nkvo`, whose attention on the host costs decode 364 graph splits per token
instead of 98.

`--expert-cache-prefill-layer-major` targets the other half of that cost. With
bounded prefill every ubatch streams nearly all routed experts of every layer
through K, so a long prompt pays the expert traffic once per ubatch. Layer-major
prefill places every ubatch in the memory modules once, then runs layers
0..n-2 for all ubatches before moving to the next layer, keeping the residual
stream in host memory; the memory modules replay each ubatch's placement per
layer (a second placement would skip the state reset of a new recurrent
sequence). The last layer runs in the regular ubatch loop, so outputs are
produced as before. With the same ubatch split the output is identical to the
ubatch-major path. On the same machine and prompt: 88 -> 211 tok/s with
`-b 32768 --no-checkpoint-ubatch` (the server otherwise ends the prompt with an
extra ubatch-sized chunk for a context checkpoint, which costs one more full
pass over the experts).

With layer-major prefill the server also stops splitting prompts for context
checkpoints. A recurrent model needs a checkpoint a few tokens before the end of
each prompt (and at user messages) so the next request can roll back to it; the
server used to end a decode call at each such position, and every extra call is
one more pass over the routed experts. Instead the server passes the positions
to the context, which ends a ubatch at each of them and copies every layer's
recurrent state row as that layer passes the position. The captured checkpoint
is byte-identical to the one the split path creates (verified on Qwen3.8 Flash
Next); a 7.4k-token prompt took 43 s instead of 65 s. Draft-model and multimodal
requests keep the split path.

Within each layer, bounded prefill issues the disk reads for the experts that
are not in L2 and copies the L2-resident experts first; each missing expert is
copied as soon as its own read has landed, while the later reads are still in
flight. Reading and copying therefore overlap instead of running one after the
other: on Qwen3.8 Flash Next a 930-token prompt increment went from about 24 s
to about 16 s, and ubatch-major prefill of a 3.6k-token prompt from 38.7 to
57.4 tok/s, with bit-identical output.

`--expert-cache-l2-pinned-mib` removes the host memcpy from the GPU route for
experts resident in the registered part of L2. The registration is split into
chunks of at most 2 GiB, and every chunk is touched once at startup because the
driver pins lazily on the first copy. A direct copy leases its L2 slot; the
lease is returned only after the P bank event of the route that issued the copy
has completed, and an exclusive L2-to-K release waits for the same point. A
leased slot is never an L2 eviction victim. Windows limits how much a process
can register: on a 24 GB machine with an 8 GB RTX 2070 the limit was about
10 GiB in total, so a 14 GiB L2 can only be partly pinned. Pinned pages cannot
be trimmed, which leaves less room for the file cache under memory pressure.
With `--expert-cache-hybrid`, pinned L2 hits are a separate class in the cost
table.

`--expert-cache-roll deepseek4` is accepted only for the validated DeepSeek4
shape: 43 routed layers, 256 experts per layer, and top-k 6. v0.1.3 rolls the
DeepSeek4 FRONT set only. This is deliberately separate from the routed-expert
arena: Gemma/Qwen/Ornith use the same generic K/R/P mechanism with roll `off`.
The FRONT source store uses ordinary committed host memory that is registered
read-only with CUDA after population, matching the qualified research topology.
It does not require one multi-gigabyte `cudaMallocHost` allocation. The small
P elevator remains a separate pinned-host allocation.

`--expert-cache-prefill` is topology-gated rather than architecture-name-gated.
Startup requires a routed MoE with at most 512 experts per layer and
`min(n_ubatch * top_k, expert_count)` must fit the K slice available to every
routed layer after schema-bank partitioning. Unsupported capacities fail closed.
This does not make prefill a performance-qualified preset for every supported
model; it only removes the DeepSeek-only implementation constraint.

## Managed no-mmap backing (v0.1.6 experimental)

On Windows, a monolithic expert-major GGUF can use `--load-mode none` together
with `--expert-cache` and a nonzero L2. Routed expert **weight** tensors then use
a `CPU_SILIANG_MANAGED` metadata/proxy buffer: it reserves virtual address
space but commits no weight pages, and the model loader does not materialize
their bytes. SiliangEM direct GGUF I/O is the mandatory source for CPU execution
and K staging; if it cannot serve a managed expert, the runtime fails closed
instead of falling back to model-resident or mmap-backed bytes. Expert sidecars
remain ordinary CPU tensors.

With `--expert-cache-roll deepseek4`, the DeepSeek4 FRONT tensors use the same
managed proxy contract. The roller materializes its single committed host store
directly from exact model-owned GGUF offsets, then serves every graph from the
double GPU bank. FRONT ownership is independent from routed-expert prefill:
`--expert-cache-prefill` controls only the routed-expert K batch-union path; it
does not enable or disable FRONT rolling.

The DS4 token embedding table is also file-owned in this profile. The reference
GGUF stores `token_embd.weight` as a 129,280 x 4,096 F16 table (1,010 MiB), but
one token needs only one 8 KiB row. The model therefore keeps only a managed
proxy for the table and materializes the requested rows into the existing F32
ubatch embedding input through a persistent file reader. This removes the
1,010 MiB `CUDA_Host` model buffer created by ordinary no-mmap loading. A
matched greedy mmap/managed run produced byte-identical output; 32-token decode
throughput differed by less than 1% in that receipt.

Matched fresh-start smoke on the RTX 2070 workstation produced byte-identical
greedy output for mmap and fully managed no-mmap. At model-ready state, mmap
used about 12.51 GiB working set and 13.30 GiB private memory; managed no-mmap
used about 6.14 GiB working set and 14.15 GiB private memory, leaving about
10.93 GiB physical memory available versus 4.51 GiB for mmap in that run. A
separate K256/R12/P12, `-ub 512`, `-tb 12` bounded-prefill smoke processed 991
prompt tokens at 24.81 tok/s. These are architecture/correctness receipts, not
general performance guarantees.

## DeepSeek4 v0.1.3 profile

The release profile uses the smaller 8 GiB managed L2 together with K216/R12/P12
and SLFU turnover:

```powershell
& "<build-directory>in\Release\llama-server.exe" `
    -m "<deepseek4-expert-major.gguf>" `
    -ngl 99 -ncmoe 43 -nkvo --no-op-offload `
    -c 4096 -b 512 -ub 512 -t 2 -tb 2 `
    --parallel 1 `
    --expert-cache `
    --expert-cache-l2-mib 8192 `
    --expert-cache-l2-policy lru `
    --expert-cache-l1-k 216 `
    --expert-cache-exchange-r 12 `
    --expert-cache-elevator-p 12 `
    --expert-cache-l1-policy slfu `
    --admit-k-cold on `
    --demote-k-hot on `
    --expert-cache-roll deepseek4 `
    --no-expert-cache-prefill `
    --host 127.0.0.1 --port 8080
```

`--expert-cache-route-stats` may be added for aggregate qualification telemetry;
it is not required for normal use.

### FRONT determinism gate

The earlier asynchronous single-bank FRONT overwrite could change greedy output
across fresh starts. v0.1.4 closed that race with a full CUDA-backend fence but
that fence serialized prompt processing. v0.1.5 replaces it with two FRONT
banks and device-side all-stream completion events: layer N+2 reuses N's bank
only after every CUDA consumer has completed, without a per-layer host-wide
backend synchronize. Three fresh 64-token starts produced the same token hash,
and the bounded K256/`-ub 512` prompt smoke reached 24.90 tok/s over 1,101
prompt tokens.

### Historical DS4 capacity observations

Earlier source-research runs used a different runtime revision and larger L2
capacities. One complete natural 2,000-token start per capacity measured:

| L2 | Decode throughput |
| ---: | ---: |
| 12 GiB | 2.12587 tok/s |
| 14 GiB | 2.20073 tok/s |
| 16 GiB | 2.28674 tok/s |

These remain historical research evidence; they are not v0.1.3 performance
claims and should not be compared directly with the current 8 GiB profile.

For a CLI smoke, use the same cache and placement options with `llama-cli`. Use
`--no-expert-cache` for the explicit control; do not add tier options to a
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

Each fully mapped routed-MoE sweep also produces one 512-bit expert bitmap per routed
layer. The info summary reports `prefill_bitmap` completed sweeps, sweep tokens,
adjacent-sweep comparisons per layer, seeded experts, needed experts, overlap,
new experts, unused seed experts, coverage, precision, resets, and incomplete
sweep sequences. The legacy `prefill_tokens` field is summed once per mapped
layer; `sweep_tokens` is summed once per completed routed-layer sweep. Debug
logging emits the eight raw 64-bit words for every mapped layer and sweep (record `v=2`). The
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
caps this path at 512 experts per layer. Unsupported geometry or capacity fails
closed.

**Prefill sizing rule:** size K for the intended ubatch, not only for startup.
On DS4 (top-k 6, 256 experts), K216 permits at most `-ub 36` (`-ub 32` is
safe). K256 reaches the full 256-expert universe, so larger ubatches remain
geometrically valid: `min(512 * 6, 256) = min(1024 * 6, 256) = 256`. Above this
point the practical ubatch limit is GPU workspace/VRAM rather than K capacity.
On the RTX 2070 reference machine, a diverse 1,024-token prompt improved from
18.46 tok/s with two `-ub 512` sweeps to 31.33 tok/s with one `-ub 1024` sweep;
expert H2D fell from about 115.5 GB to 65.4 GB. Treat the larger ubatch as a
qualified reference-machine result, not a universal preset.

Larger is not automatically faster. A three-start interleaved A/B on a
7,369-token code prompt measured 17.79, 28.65, and 23.84 prompt tok/s median
for `-ub 512`, `-ub 1024`, and `-ub 2048`. `-ub 2048` fit in the 8 GB GPU,
but host time between routed layers grew about 4.5x per token. Details are in
[`PERFORMANCE.md`](PERFORMANCE.md#deepseek4-bounded-prefill-ubatch-ab-2026-09-25).

`llama-server` context checkpoints deliberately split the last four prompt
tokens into a separate decode call so a checkpoint can be created. That is
normally useful server behavior, but with bounded MoE prefill it causes a second
43-layer expert sweep. On the reference DS4 profile, `--ctx-checkpoints 0`
removed that N-4/4 split and improved fresh 32/64/128/256-token prompt throughput
by roughly 18-25%. Disable checkpoints only when that server feature is not
needed; this is a throughput/feature tradeoff, not an unconditional default.

With `--expert-cache-route-stats`, every completed bounded-prefill sweep also
emits one `SILIANG_PREFILL_SWEEP` record containing token count, aggregate
unique-expert union statistics, K hits/misses/admissions/evictions, P waves, and
expert H2D operations/bytes. The record is intended for sweep attribution and is
silent when route statistics are disabled.

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
    --expert-cache-l1-policy slfu `
    --admit-k-cold on `
    --demote-k-hot on `
    --expert-cache-roll deepseek4 `
    --expert-cache-prefill `
    --no-warmup `
    --reasoning off --reasoning-format deepseek `
    --host 127.0.0.1 --port 18081 --no-webui
```

The command above is the low-K diagnostic profile. For DS4 prompt throughput,
use K256 with `-ub 1024`, the qualified RTX 2070 reference setting. Measure
before using a larger ubatch; `-ub 2048` was slower on that machine. For server throughput
measurements where context checkpoints are not required, add
`--ctx-checkpoints 0`. To isolate bounded prefill, toggle
`--expert-cache-prefill` / `--no-expert-cache-prefill` while keeping K, ubatch,
checkpoint policy, and `-tb 12` unchanged.

## Qwen3.8 Flash Next agent profiles (v0.1.8)

These profiles target a coding agent on the reference workstation: Ryzen 5
2600, 24 GB RAM, 8 GB RTX 2070, model on an NVMe SSD. They use the
single-schema expert-major file (UD-Q2_K_XL with layer 2 gate/up requantized to
IQ2_XS, 78.9 GB) and an f16 KV cache on the GPU. All numbers are
machine-specific measurements, not guarantees.

32k context, the default agent profile:

```powershell
& "<llama-server.exe>" -m "<qwen3.8-flash-next-em.gguf>" `
  -c 32768 -b 32768 -ub 3072 --parallel 1 -ngl 99 -ncmoe 48 --no-op-offload -t 8 -tb 8 `
  --expert-cache --expert-cache-l2-mib 12288 --expert-cache-l2-policy lfu `
  --expert-cache-l1-k 512 --expert-cache-l1-k-decode 1440 `
  --expert-cache-exchange-r 20 --expert-cache-elevator-p 20 --expert-cache-l1-policy slfu `
  --expert-cache-prefill --expert-cache-prefill-layer-major --no-checkpoint-ubatch `
  --expert-cache-hybrid --expert-cache-staging-threads 2 --expert-cache-l2-pinned-mib 8192 `
  --ctx-checkpoints 4 --cache-ram 0 --slot-save-path "<slot-dir>" `
  --reasoning-budget 16384 --reasoning-budget-message " Time is up: I must stop thinking now and give the final answer." `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 18081
```

On a 29k-token prompt: prefill 229 tok/s, decode after it 7.9 tok/s; decode on
a short prompt 8.9 tok/s. Through a real agent session (Pi, sampling on,
5k-25k context) decode was 5.1-7.7 tok/s.

Short prompt increments are bounded by disk reads, not by prompt length. A
20-token follow-up routes to about 4,400 distinct (layer, expert) pairs and a
900-token tool result to about 19,300 of the 24,576, while the 12 GiB L2 holds
6,870 experts; the missing ones (about 5 GB and 30 GB) are read from the SSD in
every prefill pass. Measured: a 20-token follow-up takes 2.0-2.5 s of prefill,
a 900-token increment about 16 s. More RAM for L2 is what removes this cost.

64k context: replace `-c 32768 -b 32768 -ub 3072` with `-c 65536 -b 65536 -ub
1024` and use `--expert-cache-l2-mib 10240`, which leaves host RAM for the
layer-major residual stream (about 2.3 GB on a 59k-token prompt). On a
59k-token prompt: prefill 195 tok/s, decode after it 5.2 tok/s, 3.4 GB of RAM
still free. A long agent task that reads many files can fill 32k, so prefer 64k
for agent work.

A quantized KV cache is an opt-in alternative for 64k, not the default: `-ctk
q8_0 -ctv q8_0` frees enough VRAM for `-ub 2048` and `--expert-cache-l2-mib
12288`. On the same 59k-token prompt: prefill 196 tok/s, decode 6.3 tok/s, but
only 1.5 GB of RAM free. Against the f16 KV cache (all experts on GPU, 65
teacher-forced positions after the prompt) the next token agreed on 61, with a
mean top-10 KL divergence of 0.043 (max 0.79).

Thinking: the Qwen3.8 chat template supports it, and the request chooses it.
`chat_template_kwargs.enable_thinking` turns it on or off, and
`thinking_budget_tokens` caps it; when the cap is reached the server inserts
`--reasoning-budget-message` and the end-of-thinking tag, and the model
answers. `--reasoning-budget` is the cap for requests that do not send one.
The message must be one line when it comes from a `.cmd` file: the option
does not unescape `\n`. Pi 0.84.3 sends both fields with `thinkingFormat:
"qwen-chat-template"` and `thinkingTokenBudgetField: "thinking_budget_tokens"`,
taking the budgets from its `thinkingBudgets` setting; it maps `xhigh` and
`max` to the `high` budget.

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

## Cross-model v0.1.3 qualification configurations

These are the configurations used by the final Windows CUDA release-candidate
qualification. The throughput values are medians of three fresh server starts
with greedy 256-token decode. They are machine-specific evidence, not universal
presets.

| Model / path | Median decode | Range |
| --- | ---: | ---: |
| Gemma4 26B-A4B K1440 | 21.651 tok/s | 21.253-21.672 |
| Qwen3 30B-A3B K1440 | 19.261 tok/s | 17.360-20.164 |
| Qwen3.6 35B-A3B K1440 | 9.279 tok/s | 7.549-9.534 |
| Qwen3.6 35B-A3B no-cache | 11.011 tok/s | 10.210-11.392 |
| Ornith 1.0 35B K1920 | 13.967 tok/s | 13.948-14.004 |
| GPT-OSS 120B managed L2 | 3.344 tok/s | 3.335-3.356 |

Gemma4 26B-A4B QAT:

```powershell
& "<llama-server.exe>" -m "<gemma4-26b-a4b.gguf>" `
    -ngl 99 -ncmoe 30 -c 32768 -b 512 -ub 512 -t 4 -tb 4 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1440 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --expert-cache-prefill
```

Qwen3-30B-A3B:

```powershell
& "<llama-server.exe>" -m "<qwen3-30b-a3b.gguf>" `
    -ngl 99 -ncmoe 48 -c 32768 -b 512 -ub 3 -t 12 -tb 12 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1440 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --expert-cache-prefill
```

Ornith-1.0-35B:

```powershell
& "<llama-server.exe>" -m "<ornith-1.0-35b.gguf>" `
    -ngl 99 -ncmoe 40 -c 32768 -b 512 -ub 6 -t 12 -tb 12 `
    --parallel 1 --expert-cache `
    --expert-cache-l1-k 1920 `
    --expert-cache-exchange-r 16 `
    --expert-cache-elevator-p 16 `
    --expert-cache-l1-policy wtinylfu-w10-slru-p80 `
    --expert-cache-roll off `
    --expert-cache-prefill
```

Qwen3.6-35B-A3B remains supported, but K1440 is not the recommended v0.1.3
performance path. The matched current control was about 18.7% faster at the
median, so use the explicit no-cache configuration unless re-tuning on another
machine:

```powershell
& "<llama-server.exe>" -m "<qwen3.6-35b-a3b.gguf>" `
    -ngl 99 -ncmoe 40 -c 32768 -b 512 -ub 4 -t 12 -tb 12 `
    --parallel 1 --no-expert-cache
```

GPT-OSS 120B uses the bounded managed host source rather than a redundant copy
of resident tensors:

```powershell
& "<llama-server.exe>" -m "<gpt-oss-120b-expert-major.gguf>" `
    -ngl 99 -ncmoe 36 -nkvo --no-op-offload `
    -c 8192 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --expert-cache `
    --expert-cache-l2-mib 18432 `
    --expert-cache-l2-policy lru `
    --expert-cache-roll off `
    --no-expert-cache-prefill
```

All three GPT-OSS qualification runs completed with stable throughput but low
host-memory headroom. Keep the historical GPT-OSS benchmark in PERFORMANCE.md
as historical evidence rather than replacing it with the pressure-qualified
3.344 tok/s release receipt.

M4 remains inconclusive/no-go for v0.1.3 because there is no useful integrated
architecture path to claim.

## Memory sizing

L2 reserves and commits real system memory. Homogeneous models reserve K + R
CUDA slots; heterogeneous models reserve K plus one R tail per schema, with
schema-dependent byte sizes. P reserves one global pinned-host ring sized to the
largest expert schema. Leave headroom for Windows, non-expert model
weights, KV cache, CUDA workspaces, other applications, and WDDM pressure.
Larger values are not automatically faster. If the requested path falls back,
the run is void for performance comparison.
