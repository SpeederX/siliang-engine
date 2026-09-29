# Siliang Engine

> Four ounces can move a thousand pounds.

![Siliang Engine](assets/siliang-engine.png)

Siliang Engine is an experimental inference engine maintained as a fork of
[`llama.cpp`](https://github.com/ggml-org/llama.cpp). It applies small,
high-leverage changes to the Windows Mixture-of-Experts (MoE) data path so very
large models can make better use of limited workstation memory.

Its core workflow combines model-owned expert sources with a typed L2/L1 memory
hierarchy. A bounded system-RAM L2 can serve out-of-core experts, while a CUDA
L1 keeps K persistent experts plus R exchange slots and a bounded pinned P
  elevator. The v0.1.8 DeepSeek4 path can also roll its architecture-specific
  FRONT set. Separately, the generic routed-MoE arena can, as an explicit
  experiment, reuse K for bounded prompt microbatches when the topology and
  layer-local capacity checks pass. The GPU retains router weights; only selected expert IDs enter
  the CPU cache-control path. Per-sweep route bitmaps measure reuse without
  enabling speculative admission. Expert-major GGUF remains the recommended source layout for DS4 and
GPT-OSS; compatible stock MoE models can promote experts from their existing
resident host tensors without allocating a redundant L2. v0.1.8 qualifies the
arena for Qwen3.8 Flash Next as a local coding-agent backend on an 8 GB GPU and
24 GB of RAM.

The Siliang expert arena is supported on Windows today. The fork preserves the
upstream backend architecture, and release CI also builds Linux CPU and macOS
Metal configurations so Siliang changes cannot silently make the fork
Windows-only. Porting the Siliang arena itself beyond Windows remains planned.

The expert cache is opt-in. Without `--expert-cache`, the engine keeps the
ordinary path and allocates no Siliang L2 or L1 arena. Configuration is carried
on the `llama-cli` or `llama-server` command line; there is no environment setup
helper.

## Quickstart

1. Build Siliang Engine using the [build instructions](#build) below.
2. For the recommended path, prepare the source model as an expert-major GGUF
   by following [`tools/README.md`](tools/README.md). A compatible monolithic
   stock GGUF can also use the arena without repacking.
3. Start `llama-cli` or `llama-server` with an explicit measured configuration.
   The options are shared by both binaries and are listed in `--help`.

```powershell
& "<build-directory>\bin\Release\llama-cli.exe" `
    -m "<expert-major-model.gguf>" `
    -ngl 99 -ncmoe <all-routed-layers> `
    --expert-cache `
    --expert-cache-l2-mib <measured-MiB> `
    --expert-cache-l2-policy lfu `
    -p "<prompt>" `
    -n 32
```

Managed L2 is consumed only by CPU-backed routed experts, so the placement
flags must keep every intended routed layer on CPU; verify nonzero lookup
telemetry rather than assuming the allocation is active. Use a cache budget
that leaves room for Windows, the model's non-expert
weights, KV cache, and GPU shared-memory pressure. Larger is not automatically
better. Use `--no-expert-cache` for a deliberate control. The full DS4
K216/R12/P12 server command, Pi endpoint, policy options, and conservative
  routed-MoE prefill-microbatch experiment, Gemma4/Qwen3/Qwen3.6/Ornith/GPT-OSS trial recipes, and the Qwen3.8 Flash Next agent profiles are in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Performance

Reference-workstation observations (hardware below), not universal throughput
claims. Rates are server-reported tokens per second; benchmarks use greedy
sampling (temperature 0, top-k 1, seed 42) and a fresh server process per run.
Rows were not re-measured on later engine versions; the Engine column links to
the release notes or benchmark record. In the configurations, K is the number of
GPU slots that keep experts resident, R the GPU exchange slots, P the pinned
host staging slots, and the RAM cache is the L2 expert cache in system memory.

| Model | Result | Configuration | Engine |
| --- | --- | --- | --- |
| Qwen3.8 Flash Next, UD-Q2_K_XL expert-major ([model](https://huggingface.co/SpeederX/Qwen3.8-Flash-Next-UD-Q2_K_XL-EM-GGUF)) | Prefill of 28,908 tok: 229.4 tok/s<br>Decode of 256 tok: 6.9 tok/s (7.9 after token 64)<br>Runs: 1<br>Warmup: 0 tok | [32k agent profile](docs/CONFIGURATION.md#qwen38-flash-next-agent-profiles-v018): 12 GiB RAM cache (8 GiB pinned), K 512 in prefill and 1,440 in decode, hybrid CPU/GPU decode, layer-major prefill, ubatch 3072, f16 KV | [v0.1.8](docs/releases/v0.1.8.md) |
| Qwen3.8 Flash Next | Prefill of 26 tok: not measured<br>Decode of 512 tok: 8.92 tok/s<br>Runs: 1<br>Warmup: 48 tok | [32k agent profile](docs/CONFIGURATION.md#qwen38-flash-next-agent-profiles-v018) | [v0.1.8](docs/releases/v0.1.8.md) |
| Qwen3.8 Flash Next | Prefill of 5k-25k tok context: not measured<br>Decode of agent replies: 5.1-7.7 tok/s (range over turns)<br>Runs: real Pi agent session, sampling on<br>Warmup: none | [32k agent profile](docs/CONFIGURATION.md#qwen38-flash-next-agent-profiles-v018) | [v0.1.8](docs/releases/v0.1.8.md) |
| Qwen3.8 Flash Next | Prefill of 58,777 tok: 194.8 / 195.0 tok/s<br>Decode of 256 tok: 5.3 / 5.2 tok/s after token 64<br>Runs: 2 (both values shown)<br>Warmup: 0 tok | [64k agent profile](docs/CONFIGURATION.md#qwen38-flash-next-agent-profiles-v018): as 32k, with a 10 GiB RAM cache and ubatch 1024 | [v0.1.8](docs/releases/v0.1.8.md) |
| DeepSeek V4 Flash 0731, expert-major | Prefill of 7,369 tok: 37.97 tok/s (v0.1.7 in the same session: 26.36)<br>Decode of 32 tok: not reported<br>Runs: 3, median (range 37.83-38.47)<br>Warmup: 0 tok | Bounded prefill on GPU: K256/R12/P12, 2 GiB RAM cache (LFU), FRONT rolling, ubatch 1024, 12 prompt threads | [v0.1.8](docs/releases/v0.1.8.md) |
| DeepSeek V4 Flash 0731, expert-major | Prefill of 26 tok: not measured<br>Decode of 128 tok: 1.666 tok/s (v0.1.7 in the same session: 1.672); desktop in use<br>Runs: 3, median (range 1.643-1.702)<br>Warmup: 48 tok | v0.1.3 DS4 profile: K216/R12/P12, 8 GiB RAM cache (LRU), FRONT rolling, prefill off, 2 threads | [v0.1.8](docs/releases/v0.1.8.md) |
| DeepSeek V4 Flash 0731, expert-major | Prefill of 26 tok: not measured<br>Decode of 128 tok: 2.044 tok/s; idle machine<br>Runs: 3, median (range 1.973-2.129)<br>Warmup: 48 tok | v0.1.3 DS4 profile, as above | [v0.1.7](docs/releases/v0.1.7.md) |
| Gemma4 26B-A4B | Prefill of short prompt: not measured<br>Decode of 256 tok: 21.651 tok/s<br>Runs: 3, median (range 21.253-21.672)<br>Warmup: not recorded | Stock GGUF, experts promoted from host memory (no separate RAM cache): K1440/R16/P16, W-TinyLFU GPU policy | [v0.1.3](docs/releases/v0.1.3.md) |
| Qwen3 30B-A3B | Prefill of short prompt: not measured<br>Decode of 256 tok: 19.261 tok/s<br>Runs: 3, median (range 17.360-20.164)<br>Warmup: not recorded | Stock GGUF, experts promoted from host memory: K1440/R16/P16, W-TinyLFU GPU policy | [v0.1.3](docs/releases/v0.1.3.md) |
| Ornith 1.0 35B | Prefill of short prompt: not measured<br>Decode of 256 tok: 13.967 tok/s<br>Runs: 3, median (range 13.948-14.004)<br>Warmup: not recorded | Stock GGUF, experts promoted from host memory: K1920/R16/P16, W-TinyLFU GPU policy | [v0.1.3](docs/releases/v0.1.3.md) |
| Qwen3.6 35B-A3B | Prefill of short prompt: not measured<br>Decode of 256 tok: 11.011 tok/s<br>Runs: 3, median (range 10.210-11.392)<br>Warmup: not recorded | Expert cache off (plain CPU-MoE path). With K1440/R16/P16 it was correct but slower: 9.279 tok/s | [v0.1.3](docs/releases/v0.1.3.md) |
| GPT-OSS 120B | Prefill of short prompt: not measured<br>Decode of 256 tok: 3.344 tok/s; low host-memory headroom<br>Runs: 3, median (range 3.335-3.356)<br>Warmup: not recorded | Expert-major GGUF, 18 GiB managed RAM cache (LRU), no GPU expert slots, prefill off | [v0.1.3](docs/releases/v0.1.3.md) |
| DeepSeek V4 Flash | Prefill of short prompt: not measured<br>Decode of 2,048 tok: 1.944 tok/s; depth and stability check<br>Runs: 1<br>Warmup: not recorded | v0.1.3 DS4 profile: K216/R12/P12, 8 GiB RAM cache, FRONT rolling | [v0.1.3](docs/releases/v0.1.3.md) |
| DeepSeek V4 Flash 0731 | Prefill of short prompt: not measured<br>Decode of 256 tok: 2.774 tok/s vs 2.274 on the stock file (1.22x)<br>Runs: 3 per arm, median (range 2.689-2.850)<br>Warmup: 0 tok, file cache purged | Layout A/B: expert-major vs stock GGUF, both with the same 18 GiB RAM cache | [pre-v0.1.3](docs/HISTORICAL_RESULTS.md#arena-and-layout-benchmarks-2026-08) |
| DeepSeek V4 Flash 0731 | Prefill of short prompt: not measured<br>Decode of 256 tok: 2.291 tok/s vs 1.375 on mmap (1.67x)<br>Runs: 3 per arm, median (range 2.217-2.385)<br>Warmup: not recorded | Cache A/B on the stock GGUF: 18 GiB RAM cache vs plain mmap | [pre-v0.1.3](docs/HISTORICAL_RESULTS.md#arena-and-layout-benchmarks-2026-08) |
| DeepSeek V4 (pre-0731) | Prefill of short prompt: not measured<br>Decode of 256 tok: 2.246 tok/s vs 1.098 on stock mmap (2.05x)<br>Runs: 3 per arm, median (range 2.171-2.257)<br>Warmup: 48 tok | Expert-major file with a 12 GiB RAM cache vs the stock file on plain mmap | [pre-v0.1.3](docs/HISTORICAL_RESULTS.md#arena-and-layout-benchmarks-2026-08) |
| gpt-oss-120B | Prefill of short prompt: not measured<br>Decode of 256 tok: 4.052 tok/s vs 1.972 on mmap (2.06x)<br>Runs: 3 per arm, median (range 3.953-4.094)<br>Warmup: 48 tok | Cache A/B on the same expert-major file: RAM cache vs plain mmap | [pre-v0.1.3](docs/HISTORICAL_RESULTS.md#arena-and-layout-benchmarks-2026-08) |

### Reference workstation

The benchmark workstation runs Windows 11 on an
[ASUS ROG Strix B450-F Gaming](https://rog.asus.com/motherboards/rog-strix/rog-strix-b450-f-gaming-model/spec/)
motherboard with an AMD Ryzen 5 2600, 24 GB of system RAM, an NVIDIA GeForce
RTX 2070 with 8 GB of VRAM, and a 1 TB
[WD_BLACK SN850X](https://www.sandisk.com/en-us/products/ssd/internal-ssd/wd-black-sn850x-nvme-ssd)
NVMe SSD. The drive supports PCIe Gen4 x4, but this Ryzen 2000-series B450
platform exposes its M.2 link as PCIe 3.0 x4. Using PCIe 3.0's 8 GT/s rate and
128b/130b encoding, that is about 3.94 GB/s of theoretical payload bandwidth
per direction before protocol, storage, and workload overhead; the calculation
is consistent with the [PCI-SIG bandwidth table](https://pcisig.com/how-does-pcie-30-8gts-double-pcie-20-5gts-bit-rate).

## What is included

- A typed, opt-in MoE hierarchy with managed host L2, a CUDA K policy budget,
  per-schema R exchange banks, and bounded global P staging.
- An architecture-guarded DeepSeek4 FRONT rolling path for serial decode.
- A separate topology-gated routed-MoE bounded-prefill experiment, currently
  limited to at most 512 experts per layer and layer-local K capacity.
- The expert-major GGUF preparation workflow for the core and recommended
  Siliang path, plus validated arena support for compatible monolithic stock
  GGUFs.
- CPU and CUDA build entry points for this `llama.cpp` fork.
- Reproducible model-free checks, runtime validation tooling, and source
  provenance records.

See [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for the complete
repository map.

## Build

Run the build script from PowerShell at the repository root on 64-bit Windows.
Build directories are supplied by the caller.

CPU:

```powershell
.\scripts\build.ps1 -Backend Cpu -BuildRoot "<build-root>"
```

CUDA builds use the upstream `llama.cpp` multi-architecture selection for the
installed CUDA toolkit when no architecture is supplied:

```powershell
.\scripts\build.ps1 `
    -Backend Cuda `
    -BuildRoot "<build-root>"
```

For a local diagnostic build, `-CudaArchitecture` can narrow the binary to one
compute capability. For example, use `75` only for a GPU whose CUDA compute
capability is 7.5:

```powershell
.\scripts\build.ps1 `
    -Backend Cuda `
    -CudaArchitecture 75 `
    -BuildRoot "<build-root>"
```

Release builds do not pin one GPU generation or duplicate an architecture list
inside Siliang. The effective CUDA architecture set selected by the pinned
`llama.cpp`/CUDA toolchain is recorded in `provenance/BUILD-INFO.txt`.

The scripts produce portable Release builds with runtime-selected CPU backend
variants. `GGML_NATIVE` stays off so a package is not tied to the build host;
compatible systems select an optimized variant such as Haswell/AVX2 when the
process starts. The scripts do not choose a model or inference settings for
you. More details are in [`scripts/README.md`](scripts/README.md).

## Credits

Siliang Engine is built on [`llama.cpp`](https://github.com/ggml-org/llama.cpp),
created by [Georgi Gerganov](https://github.com/ggerganov) and developed by the
ggml and llama.cpp contributors. Their work made this experimental spin-off
possible.

The exact upstream revision, the Siliang delta, and the reconstruction checks
are recorded in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

Siliang's SSD expert-streaming and resident-cache design was inspired by
[DwarfStar](https://github.com/antirez/ds4) by
[Salvatore Sanfilippo](https://github.com/antirez). Siliang is an independent
Windows/llama.cpp implementation: it uses a committed system-RAM arena and an
expert-major GGUF so one routed expert can be fetched with one contiguous
read. No DwarfStar code is included.

Its measure-first, reads-per-token memory-tiering method was also inspired by
[ESP32-AI](https://github.com/slvDev/esp32-ai) by
[Viacheslav Sierbov](https://github.com/slvDev). ESP32-AI demonstrated
access-pattern-aware placement and load-time staging across flash, PSRAM, and
SRAM. Siliang applies that engineering discipline to routed MoE experts; its
fixed-slot arena and overlapped Windows I/O path are separate implementations.

## Documentation

- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) - runtime configuration and
  reset workflow.
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) - raw benchmark evidence,
  calculations, and evidence limits.
- [`docs/HISTORICAL_RESULTS.md`](docs/HISTORICAL_RESULTS.md) - results of
  earlier releases and benchmarks.
- [`docs/releases/`](docs/releases/) - per-release changes, qualification, and
  known boundaries.
- [`tools/README.md`](tools/README.md) - expert-major model preparation.
- [`scripts/README.md`](scripts/README.md) - build, checks, and runtime-gate
  commands.
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) - upstream identity and patch
  provenance.

## Contributing

Open a bug report using the
[`bug report template`](.github/ISSUE_TEMPLATE/bug_report.md). Include the
operating system, RAM, VRAM, storage class, issue kind, reproduction steps, and
any additional notes.

Contributors and coding agents should follow both the upstream
[`AGENTS.md`](AGENTS.md) guidance and the Siliang-specific
[`SILIANG_AGENTS.md`](SILIANG_AGENTS.md) rules.

### Tagged artifacts

Pushing a `v*` tag runs Windows, Linux, and macOS compatibility validation, then
builds the release packages on Windows. The downloadable artifacts are:

- `siliang-engine-<tag>-windows-x64-cpu.zip`
- `siliang-engine-<tag>-windows-x64-cuda-13.2.zip`
- `SHA256SUMS`

Actions retains the verified packages for 30 days. A `v*` tag runs the
qualification CI. CI is the sole producer of Windows CPU/CUDA archives; when a
tag run completes successfully, it triggers the release publisher, which checks
that the run built the tagged commit, verifies the checksums, and publishes those
exact artifacts as a prerelease. The publisher remains manually dispatchable for
an existing tag whose CI run has already succeeded, without rebuilding the
packages.
For v0.1.8, see the [release notes](docs/releases/v0.1.8.md); results of
earlier releases are in [`docs/HISTORICAL_RESULTS.md`](docs/HISTORICAL_RESULTS.md).

## License

The upstream `llama.cpp` source at the repository root remains under its MIT
License in [`LICENSE`](LICENSE). Siliang Engine additions are available under
the separate [`licenses/SILIANG-ENGINE-MIT.txt`](licenses/SILIANG-ENGINE-MIT.txt).
Bundled third-party components retain their own notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
