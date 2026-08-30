# Performance evidence

This page contains the sanitized measurements behind the performance table in
the root [`README.md`](../README.md). Values are decode throughput unless a
section is explicitly labeled prompt processing. They describe one Windows
workstation and the named model and runtime configurations; they are not general
performance guarantees.

The environment-variable commands retained on this page are historical
v0.1.2 evidence, not the v0.1.3 operator interface. v0.1.3 uses the typed
`--expert-cache*` CLI options documented in [`CONFIGURATION.md`](CONFIGURATION.md);
the recorded commands below are intentionally not rewritten retroactively.

The measurements do not have equal evidence depth:

| Model and path | Evidence retained | Public evidence tier |
| --- | --- | --- |
| DeepSeek V4 Flash 0731, stock GGUF arena versus expert-major GGUF arena | Every raw timing repetition, cold-cache event, artifact identity, command, request, output identity, engine counter, and process I/O counter | Current fully cold layout benchmark |
| DeepSeek V4 Flash 0731, stock GGUF mmap versus stock GGUF arena | Every raw timing repetition, schedule, model identity, command, request, output identity, and path counters | Current same-file benchmark |
| DeepSeek V4 Flash 0731, stock GGUF mmap versus expert-major GGUF arena | Every raw timing repetition, schedule, artifact hashes, runtime hash, command, request, and result summary | Historical precursor; superseded for the direct layout claim |
| DeepSeek V4 pre-0731 | Every raw timing repetition for three matched arms; partial timing configuration and artifact identity | Historical matched run |
| gpt-oss-120B | Every raw timing repetition for two matched arms; partial timing configuration and artifact identity | Historical matched run |
| DeepSeek V4 Flash 0731 prompt processing, pre-prefill-port runtime | Two completed interactive requests and cumulative runtime telemetry; prompts differed and no matched arm was frozen | Observational prefill controls only |

## Workstation

- Windows 11, 64-bit.
- AMD Ryzen 5 2600, 6 cores and 12 threads.
- 24 GB system RAM (23.93 GiB observed).
- NVIDIA GeForce RTX 2070 with 8 GB VRAM.
- ASUS ROG Strix B450-F Gaming motherboard.
- 1 TB WD_BLACK SN850X NVMe SSD. The platform exposes the M.2 link as PCIe
  3.0 x4, about 3.94 GB/s of theoretical payload bandwidth per direction
  before protocol and workload overhead.

The current 2026-08-10 runs used Siliang's build of llama.cpp 10270
(`071327508`), a Release CUDA build targeting compute capability 7.5, and MSVC
19.44.35225.0 x64. The fully retained precursor run used a Release CUDA build
targeting compute capability 7.5, CUDA 13.2.51, MSVC 19.44.35225.0 x64, and the
same physical NVMe device for both GGUF files. The older rows were measured on
this workstation, but their retained summaries do not contain an equally
complete build packet.

## v0.1.3 release-candidate qualification (2026-08-31)

This section records the final multi-model Windows CUDA release-candidate gate.
It is separate from the historical benchmark sections below. The server cells
used fresh processes, greedy 256-token decode (`temperature=0`, `top-k=1`, seed
42), and three repetitions per path.

| Model / path | Median decode | Range | Output consistency |
| --- | ---: | ---: | --- |
| Gemma4 26B-A4B, K1440/R16/P16 | **21.651 tok/s** | 21.253-21.672 | one token hash across 3/3 |
| Qwen3 30B-A3B, K1440/R16/P16 | **19.261 tok/s** | 17.360-20.164 | one token hash across 3/3 |
| Qwen3.6 35B-A3B, K1440/R16/P16 | **9.279 tok/s** | 7.549-9.534 | one token hash across 3/3 |
| Qwen3.6 35B-A3B, no expert cache | **11.011 tok/s** | 10.210-11.392 | one token hash across 3/3 |
| Ornith 1.0 35B, K1920/R16/P16 | **13.967 tok/s** | 13.948-14.004 | one token hash across 3/3 |
| GPT-OSS 120B, 18 GiB managed L2 | **3.344 tok/s** | 3.335-3.356 | one token hash across 3/3; host-memory pressure in every rep |

The Qwen3.6 matched control is the important decision result: the no-cache path
was about 18.7% faster at the median than K1440, so v0.1.3 does not recommend
K1440 for that model even though the path is correct.

### DeepSeek4 FRONT determinism closure

The initial v0.1.3 FRONT rolling candidate produced different greedy outputs
across fresh starts. Isolation showed that the model/backend, managed L2, and
K216/L2/R/P were deterministic when FRONT rolling was disabled. The single
rolling FRONT bank therefore received a completion fence before overwrite.
With the fence, three fresh 64-token runs produced one identical token hash at
**1.936-1.983 tok/s**.

**2,048-token release gate:** one complete current-profile run generated all 2,048 tokens at **1.94436 tok/s** (514.31 ms/token), after reaching route-stat checkpoints at 32, 64, 128, 256, 512, 1,024, 1,512, and 2,048 tokens with no runtime failure. Host-memory headroom was low during the run, so the number is retained as depth/stability evidence rather than an isolated performance ceiling.

The historical DS4 and GPT-OSS results below remain valid evidence for their
original runtime revisions and protocols. They are not silently replaced by
this release-candidate matrix.

## DeepSeek4 prompt-processing observation (2026-08-27)

One interactive request on the same workstation completed 1,854 prompt tokens
in 992,584.21 ms: 535.37 ms/token, or 1.87 prompt tok/s. The server used the
0731 expert-major model, context 2,048, batch/ubatch 512, two prompt threads,
L2=2,048 MiB, K216/R12/P12, L2 LFU, the L1 cumulative-LFU admission/bypass
policy then spelled `lfu`, and DeepSeek4 FRONT rolling. The machine remained in
ordinary interactive use during the request.

A separate fresh request using twelve batch threads completed 1,570 prompt
tokens in 343,328.27 ms: 218.68 ms/token, or 4.57 prompt tok/s. The operator
reported `-tb 12` as the relevant configuration change. This is 2.44x the first
observed rate, but the prompt lengths and contents differed, so it is not a
matched thread-count speedup claim. It establishes twelve batch threads as the
control setting for the prefill experiment.

The cumulative host-cache report reached 102,142 projection lookups, 68,007
hits, 34,135 misses, and 224.28 GiB read from the expert slab. Its 66.6% lookup
hit rate is mostly the packed expert's three projections sharing one admission;
it must not be read as 66.6% expert residency reuse. The old binary's
`expert-level hit` field overflowed and its cross-layer statistic included
prefill, so those two printed values are invalid. v0.1.3 now counts explicit
expert requests/hits and labels cross-layer overlap as decode-only.

Late in the first prompt, only 20.0% of cache calls used the `cne1 == 1` fast
path, and both requests reported zero graph reuses. This makes bounded
microbatching a reasonable experiment, not a demonstrated optimization. These
were unfrozen interactive workloads with no output-equivalence arm and no
interleaved schedule. They are pre-port observational controls only; no product
speedup claim follows from them.

The current v0.1.3 candidate removes the selected mixture-weight tensor from
the CPU route-mapping dependency while leaving GPU top-k scoring and expert
weighting unchanged. It also records four-word, per-layer route bitmaps for
strictly ordered prefill sweeps and reports adjacent-sweep coverage and
precision. These are implementation and telemetry changes, not measured
throughput gains. Speculative bitmap-driven L2 lookahead remains disabled.

## DeepSeek V4 Flash 0731

### Current fully cold expert-major layout benchmark (2026-08-10)

This benchmark directly compares the monolithic stock layout with the finalized
expert-major layout while keeping the Siliang arena active in both arms. The
binary, physical disk, 18 GiB arena, request, and server arguments are fixed.
The declared variable is the GGUF expert-tensor layout and its corresponding
Siliang source path. This is the current direct evidence for the repack's
experimental contribution; it does not establish an upper bound for later
layout or expert-routing work.

#### Identity and method

- Model variant:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731`.
- Stock GGUF: 86,720,111,488 bytes; retained SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
- Finalized expert-major GGUF: 86,923,096,064 bytes; retained SHA-256
  `83d09412bbfbbcc0fa9f77afeee763d5f72da1ad055be50675111f585ab8b99a`.
- Runtime: the same llama.cpp build 10270 (`071327508`) Release CUDA binary in
  every cell. The retained executable-and-adjacent-DLL composite SHA-256 is
  `eb7dd512e3812f1465f44d488343a699a9f291804949b5ad21ddab38301e83de`.
- Storage: both files were on the same volume and physical WD_BLACK SN850X
  NVMe device.

Header checks identified 43 contiguous zero-based MoE layers and 256 experts
in both files. The expert-major file contained 43 packed tensors with adjacent
`gate | up | down` parts and 512-byte alignment.

Each cell used a fresh server process. The two arms followed the predeclared
schedule `stock/expert-major`, `expert-major/stock`,
`stock/expert-major`. Before every process start, the Windows standby list was
purged. The post-purge standby footprint was 1.352-2.895 MiB across the six
cells. The first pair was the integrated correctness gate; the remaining four
cells ran only after its outputs matched byte-for-byte.

Every request generated exactly 256 tokens with prompt caching disabled, EOS
ignored, temperature 0, top-k 1, and seed 42. Context size was 2,048 tokens.
The exact prompt was:

```text
The memory hierarchy of a modern workstation spans five orders of magnitude in latency. Registers answer in a fraction of a nanosecond while
```

The server command shape was:

```powershell
llama-server -m "<stock-or-expert-major.gguf>" -ngl 99 -ncmoe 43 -nkvo `
    --no-op-offload -lv 4 -c 2048 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --host 127.0.0.1 --port "<per-cell-port>" --no-webui
```

Both arms used `GGML_MOE_PREFETCH=0`, `SILIANGEM_CACHE_MIB=18432`,
`SILIANGEM_DEFER=1`, `SILIANGEM_MEM_REPORT=0`, and
`SILIANGEM_VERBOSE=1`. Neither arm disabled the arena.

#### Raw repetitions

Values below are retained at their stored precision and ordered by repetition
number within each arm:

| Repetition | Stock GGUF, 18 GiB arena | Expert-major GGUF, 18 GiB arena |
| ---: | ---: | ---: |
| 1 | 2.269094224031289 | 2.8502146723796251 |
| 2 | 2.407391843995291 | 2.7741166722168868 |
| 3 | 2.2736884430272695 | 2.6893616291715725 |

| Arm | Median | Observed range |
| --- | ---: | ---: |
| Stock GGUF, 18 GiB arena | 2.2736884430272695 | 2.269094224031289-2.407391843995291 |
| Expert-major GGUF, 18 GiB arena | 2.7741166722168868 | 2.6893616291715725-2.8502146723796251 |

The README rounds these values to three decimals. Its direct layout result is:

```text
speedup = 2.7741166722168868 / 2.2736884430272695
        = 1.2200953392380045x
change  = (1.2200953392380045 - 1) * 100
        = 22.009533923800451%
```

Every expert-major repetition exceeded every stock repetition. All six outputs
were byte-identical, stopped at the fixed 256-token limit, and have UTF-8
SHA-256
`3f16e5c22a19a5acc07fd2ea1b5dd4c4132d3426d468b096577785f1d4316f8f`.
No cell fell back to mmap.

#### I/O evidence

The stock source completed 11,622 expert fetches per cell. Each fetch required
separate `gate`, `up`, and `down` reads, for 34,866 engine-level read requests
and 76.61 GiB counted. The expert-major source recorded 11,653 misses and
submitted one contiguous read per miss, for 11,653 engine-level read requests
and 77.01 GiB counted. It recorded 205,209 lookups, 193,556 hits, zero
synchronous submissions, and 11,653 pending submissions per cell.

The operating-system counters include all reads by the owned server process,
not only expert reads:

| Arm | Process read operations per cell | Process read bytes per cell | Page-fault median (range) |
| --- | ---: | ---: | ---: |
| Stock GGUF, 18 GiB arena | 37,945 | 83,522,523,137 (77.786 GiB) | 7,846,563 (7,124,645-8,127,270) |
| Expert-major GGUF, 18 GiB arena | 14,418 | 83,814,864,897 (78.059 GiB) | 7,349,148 (7,257,653-7,422,209) |

The current repack therefore reduced engine read requests by 66.6% and total
process read operations by 62.0%, while process read bytes differed by only
0.35%. The page-fault counter combines soft and hard faults, so its 6.3% median
reduction is supporting context, not proof of fewer physical disk page-ins.

The controlled result is a +22.0% median decode-throughput improvement from the
current expert-major layout on this workstation. It complements the separate
arena-versus-mmap result below. It is an experimental single-system result, not
a general guarantee, and the 18 GiB arena remains an operator-selected setting
rather than a safe default.

### Current stock-GGUF arena benchmark (2026-08-10)

This current-namespace benchmark compares ordinary mmap with the Siliang arena
using the same monolithic stock GGUF and the same binary. It therefore isolates
the arena path on the stock file; it does not measure the effect of repacking.

#### Identity and method

- Model variant:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731`.
- Stock GGUF: 86,720,111,488 bytes; retained SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
  The benchmark recorded the current size and reused this retained hash instead
  of rehashing the 81 GiB file during the run.
- Runtime: llama.cpp build 10270 (`071327508`) with the current
  `SILIANGEM_*` implementation, MSVC 19.44.35225.0 x64, CUDA compute
  capability 7.5.
- Post-run runtime composite SHA-256, covering `llama-server.exe` and its 15
  adjacent DLLs with the repository's
  `server-executable-and-adjacent-dlls-v1` identity scheme:
  `eb7dd512e3812f1465f44d488343a699a9f291804949b5ad21ddab38301e83de`.

Header preflight identified a single-file `deepseek4` model with 43 contiguous
zero-based MoE layers, 256 experts, and `gate`, `up`, and `down` expert parts.
No split-GGUF metadata was present.

Each cell used a fresh server process. The two arms followed the predeclared
schedule `mmap/arena`, `arena/mmap`, `mmap/arena`. Every request generated
exactly 256 tokens with prompt caching disabled, EOS ignored, temperature 0,
top-k 1, and seed 42. Context size was 2,048 tokens.

The exact prompt was:

```text
The memory hierarchy of a modern workstation spans five orders of magnitude in latency. Registers answer in a fraction of a nanosecond while
```

The server command shape was:

```powershell
llama-server -m "<stock.gguf>" -ngl 99 -ncmoe 43 -nkvo `
    --no-op-offload -lv 4 -c 2048 -b 512 -ub 512 -t 12 -tb 12 `
    --parallel 1 --host 127.0.0.1 --port "<per-cell-port>" --no-webui
```

Both arms used `GGML_MOE_PREFETCH=0`, `SILIANGEM_CACHE_MIB=18432`,
`SILIANGEM_DEFER=1`, `SILIANGEM_MEM_REPORT=0`, and
`SILIANGEM_VERBOSE=1`. The mmap arm additionally set
`SILIANGEM_DISABLE=1`; the arena arm left it unset.

#### Raw repetitions

Values below are retained at their stored precision and ordered by repetition
number within each arm:

| Repetition | Stock GGUF, mmap | Same stock GGUF, 18 GiB arena |
| ---: | ---: | ---: |
| 1 | 1.3753221477236957 | 2.2909080606512897 |
| 2 | 1.4137696235227584 | 2.2172097307618839 |
| 3 | 1.3749005660900655 | 2.3845864801025818 |

| Arm | Median | Observed range |
| --- | ---: | ---: |
| Stock GGUF, mmap | 1.3753221477236957 | 1.3749005660900655-1.4137696235227584 |
| Same stock GGUF, 18 GiB arena | 2.2909080606512897 | 2.2172097307618839-2.3845864801025818 |

The README rounds these values to three decimals. Its arena-on-stock result is:

```text
speedup = 2.2909080606512897 / 1.3753221477236957
        = 1.6657246918061969x
change  = (1.6657246918061969 - 1) * 100
        = 66.57246918061969%
```

All six outputs were byte-identical, generated exactly 256 tokens, and have
UTF-8 SHA-256
`3f16e5c22a19a5acc07fd2ea1b5dd4c4132d3426d468b096577785f1d4316f8f`.
In every arena repetition, the runtime armed the stock scattered-source path,
reported 11,622 successful fetches out of 11,622 calls, served 43/43 layers,
substituted arena pointers for 43/43 layers, and counted the same 76.61 GiB as
the expected read volume. No arena cell fell back to mmap.

This proves that a compatible monolithic stock GGUF can use the arena and that
the arena produced +66.6% median decode throughput in this same-file test.
Expert-major preparation remains the core and recommended workflow. The fully
cold benchmark above separately holds the arena active in both arms and directly
measures the current expert-major layout at +22.0%. Do not infer a layout effect
by comparing this section's 2.291 tok/s with the precursor's 2.667 tok/s because
those two values come from runs with different runtime revisions and request
lengths.

The 18 GiB arena caused high memory pressure on this 24 GB workstation. This
benchmark does not establish 18 GiB as a safe default for this or another
machine.

### Expert-major precursor benchmark (2026-08-08)

This benchmark ran immediately before an identifier-only namespace migration.
It remains fully retained as historical precursor evidence, but the fully cold
benchmark above supersedes it as the direct measurement of the current
expert-major layout.

#### Identities

- Model variant:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731`.
- Stock GGUF: 86,720,111,488 bytes; SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
- Measured expert-major GGUF: 86,923,096,064 bytes; SHA-256
  `cf6a6cc0c893ec4bc18c1d3b0bf55a37e1230e01cbeda1cb616bad4be93b2481`.
- Measured runtime composite SHA-256, covering `llama-server.exe` and its
  adjacent runtime DLLs:
  `20b5fc0cd1429318840f162ceccf2540e402f12c95b3bd49020ece9499cab7da`.

The repack gate found 43 contiguous zero-based MoE layers, 256 experts,
adjacent `gate | up | down` parts, and 512-byte alignment. It passed 1,024
sampled byte comparisons against the stock source.

#### Method

Each cell used a fresh server process. The three arms were run in this
predeclared balanced schedule:

1. stock mmap, expert-major mmap, expert-major arena;
2. expert-major arena, stock mmap, expert-major mmap;
3. expert-major mmap, expert-major arena, stock mmap.

Every measurement had its own 48-token warmup. The measured request generated
exactly 128 tokens with prompt caching disabled, EOS ignored, temperature 0,
top-k 1, and seed 42. Context size was 2,048 tokens.

The exact prompt was:

```text
The memory hierarchy of a modern workstation spans five orders of magnitude in latency. Registers answer in a fraction of a nanosecond while
```

The server command shape was:

```powershell
llama-server -m "<stock-or-expert-major.gguf>" -ngl 99 -ncmoe 43 -nkvo `
    --no-op-offload -c 2048 -b 512 -ub 512 -t 12 -tb 12 `
    --host 127.0.0.1 --port "<per-cell-port>" --no-webui
```

The measured binary predates the namespace-only rename. In the historical
v0.1.2 public names, the mmap arms are equivalent to:

```powershell
$env:SILIANGEM_DISABLE = "1"
$env:SILIANGEM_VERBOSE = "1"
$env:GGML_MOE_PREFETCH = "0"
```

The arena arm is equivalent to:

```powershell
$env:SILIANGEM_CACHE_MIB = "18432"
$env:SILIANGEM_DEFER = "1"
$env:SILIANGEM_VERBOSE = "1"
$env:GGML_MOE_PREFETCH = "0"
```

#### Raw repetitions

Values below are retained at their stored precision and ordered by repetition
number within each arm:

| Repetition | Stock GGUF, mmap | Expert-major GGUF, mmap | Expert-major GGUF, 18 GiB arena |
| ---: | ---: | ---: | ---: |
| 1 | 1.4415437293585438 | 1.4524943344777628 | 2.6673920861727685 |
| 2 | 1.4771836805317011 | 1.4259443965682377 | 2.6924623066320374 |
| 3 | 1.4557804857721115 | 1.4021580263644269 | 2.6533579758427508 |

| Arm | Median | Observed range |
| --- | ---: | ---: |
| Stock GGUF, mmap | 1.4557804857721115 | 1.4415437293585438-1.4771836805317011 |
| Expert-major GGUF, mmap | 1.4259443965682377 | 1.4021580263644269-1.4524943344777628 |
| Expert-major GGUF, 18 GiB arena | 2.6673920861727685 | 2.6533579758427508-2.6924623066320374 |

The README rounds these values to three decimals. Its stock-to-arena result is:

```text
speedup = 2.6673920861727685 / 1.4557804857721115
        = 1.83227630280952x
change  = (1.83227630280952 - 1) * 100
        = 83.2276302809518%
```

For the same expert-major file, the arena was 1.87061437500107x faster than
the mmap arm (+87.0614375001067%). Repacking alone was -2.04949094286351%
relative to the stock mmap median in this run.

All nine measured outputs were byte-identical and stopped at the exact
128-token limit. The arena cells proved active direct, unbuffered, overlapped
I/O with deferred waits and no observed mmap fallback.

The operator explicitly allowed the 18 GiB minimum-free-memory gate to be
bypassed, and memory pressure was observed. This benchmark does not establish
that an 18 GiB arena is a safe default for this or another machine.

The later current-namespace smoke used a newly finalized expert-major artifact
with SHA-256
`83d09412bbfbbcc0fa9f77afeee763d5f72da1ad055be50675111f585ab8b99a`.
Its stock, repacked-mmap, and repacked-arena outputs were identical. That smoke
was a correctness and path-activation check, not another performance run. The
fully cold current-layout benchmark above subsequently measured this finalized
artifact.

## DeepSeek V4 pre-0731

This historical source revision was identified as `98819d55b`. The matched
experiment used three fresh starts per arm and produced byte-identical outputs
in all nine cells. The model variant was
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix`.

Each repetition used a fresh server process, a 48-token warmup, and 256
measured tokens. The schedule was stock/repacked-mmap/repacked-arena,
repacked-mmap/repacked-arena/stock, repacked-arena/stock/repacked-mmap. The
request used the same exact prompt shown in the 0731 section, a 2,048-token
context, temperature 0, top-k 1, seed 42, and prompt caching enabled.

The server command shape was:

```powershell
llama-server -m "<stock-or-expert-major.gguf>" -ngl 99 -ncmoe 43 -nkvo `
    --no-op-offload -c 2048 -b 512 -ub 512 `
    --host 127.0.0.1 --port "<per-cell-port>" --no-webui
```

In the historical v0.1.2 public names, both mmap arms are equivalent to
`SILIANGEM_DISABLE=1`; the arena arm is equivalent to
`SILIANGEM_CACHE_MIB=12288` and `SILIANGEM_DEFER=1`. All arms used verbose
statistics and `GGML_MOE_PREFETCH=0`.

| Repetition | Stock GGUF, mmap | Expert-major GGUF, mmap | Expert-major GGUF, 12 GiB arena |
| ---: | ---: | ---: | ---: |
| 1 | 1.1042890158325669 | 1.097256405938271 | 2.2565604315918639 |
| 2 | 1.0828779542486264 | 1.1061915402887119 | 2.2463628424067923 |
| 3 | 1.0980332727157489 | 1.1073456408947382 | 2.17088331919376 |

| Arm | Median | Observed range |
| --- | ---: | ---: |
| Stock GGUF, mmap | 1.0980332727157489 | 1.0828779542486264-1.1042890158325669 |
| Expert-major GGUF, mmap | 1.1061915402887119 | 1.097256405938271-1.1073456408947382 |
| Expert-major GGUF, 12 GiB arena | 2.2463628424067923 | 2.17088331919376-2.2565604315918639 |

The calculation used by the README is:

```text
speedup = 2.2463628424067923 / 1.0980332727157489
        = 2.04580580409089x
change  = (2.04580580409089 - 1) * 100
        = 104.580580409089%
```

The historical harness did not clear the deferred-read setting after an arena
cell. This has no effect when `SILIANGEM_DISABLE=1` selects the explicit mmap
path, but it is retained as a harness limitation.

All nine cells produced 256 tokens and the same retained decoded text. That
text has SHA-256
`81f3456e0a72679343ba75c6042d66ea970245632045e3634b181823fe416061`.
Complete model and runtime composite hashes were not retained, so this remains
historical evidence rather than current release validation. A later
current-namespace correctness smoke confirmed identical mmap and arena output
for the migrated expert-major model, but did not retime it. That current
artifact is 86,923,096,064 bytes with 43 layers, 256 experts, adjacent
`gate | up | down` parts, and tensor-payload SHA-256
`c9378c080d44087b8d55129ad1263c4a150d92466400dfa2f2e99b7e2da7c673`.

## gpt-oss-120B

This historical source revision was identified as `3a60ee5b6`. Both arms used
the same expert-major GGUF. The baseline was that repacked file running through
mmap with the arena disabled; it was not a stock GGUF. The model variant was
gpt-oss-120B MXFP4, and the measured expert-major artifact was about 59.04 GiB.

Each repetition used a fresh server process, a 48-token warmup, and 256
measured tokens. The schedule was mmap/arena, arena/mmap, mmap/arena. The
request used the same exact prompt shown in the 0731 section, a 4,096-token
context, temperature 0, top-k 1, seed 42, and prompt caching disabled.

The server command shape was:

```powershell
llama-server -m "<expert-major.gguf>" -ngl 99 -ncmoe 36 -nkvo `
    --no-op-offload -c 4096 -b 512 -ub 512 -t 12 -tb 12 `
    --host 127.0.0.1 --port "<per-cell-port>" --no-webui
```

In the historical v0.1.2 public names, the mmap arm is equivalent to
`SILIANGEM_DISABLE=1` with verbose statistics. The arena arm is equivalent to
`SILIANGEM_CACHE_MIB=18432`, `SILIANGEM_DEFER=1`, and verbose statistics.
`GGML_MOE_PREFETCH` was unset in both arms.

| Repetition | Expert-major GGUF, mmap | Same expert-major GGUF, 18 GiB arena |
| ---: | ---: | ---: |
| 1 | 1.9715534867250719 | 3.9532783039052428 |
| 2 | 2.0303500854928074 | 4.0939993771363756 |
| 3 | 1.9641837688185779 | 4.052138293591697 |

| Arm | Median | Observed range | Mean |
| --- | ---: | ---: | ---: |
| Expert-major GGUF, mmap | 1.9715534867250719 | 1.9641837688185779-2.0303500854928074 | 1.9886957803454858 |
| Same expert-major GGUF, 18 GiB arena | 4.052138293591697 | 3.9532783039052428-4.0939993771363756 | 4.033138658211105 |

The median calculation used by the README is:

```text
speedup = 4.052138293591697 / 1.9715534867250719
        = 2.0553022379944x
change  = (2.0553022379944 - 1) * 100
        = 105.53022379944%
```

Earlier summaries reported the means rounded to 1.989 and 4.033 tok/s. Their
ratio is 2.02803x, or +102.803%. Those values were correct means, but the README
now follows the current median-and-range reporting contract.

All six cells produced 256 nonempty tokens and the same retained decoded text.
That decoded text has SHA-256
`2c2c49328dc31ea070eacbb1a400fed1639a2c1b837467dc03f2df78282d413e`.
The historical JSON writer substituted `?` for some punctuation, so this is a
hash of the retained JSON-decoded text, not a claim about original wire bytes.
The arena armed in all three arena cells and remained unarmed in all three mmap
cells.

Complete model and runtime composite hashes were not retained. This row
demonstrates the arena effect on one repacked model; it does not measure the
cost or benefit of repacking against a stock gpt-oss GGUF.

The current-namespace equivalent is a 63,397,407,232-byte expert-major file
with 36 layers, 128 experts, and adjacent `gate | up | down` parts. Its retained
tensor-payload SHA-256 is
`f88a27e8889feb69ea4b165f631ad2b79d8bbb121fa75ac56cd098422f72cab6`.
A later deterministic smoke used 32 generated tokens, a 4,096-token context,
12 threads, temperature 0, top-k 1, seed 42, `-ngl 99 -ncmoe 36 -nkvo
--no-op-offload -b 512 -ub 512`, and an 18 GiB deferred arena. Its mmap and
arena outputs were identical, but its timings were intentionally excluded from
the historical throughput claim.

## Interpretation and new measurements

The README deliberately labels these evidence tiers. Comparisons across rows
should not be treated as a model leaderboard because the retained packets have
different completeness and model-specific settings.

For a new claim, use [`../scripts/runtime-gate.ps1`](../scripts/runtime-gate.ps1)
and follow the performance contract in [`../SILIANG_AGENTS.md`](../SILIANG_AGENTS.md):
run on an idle machine, use at least three independent starts per arm,
interleave the arms, retain every valid repetition, and report medians and
ranges together with artifact hashes and effective path evidence.
