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
  elevator. The v0.1.3 DeepSeek4 path can also roll its architecture-specific
  FRONT set. Separately, the generic routed-MoE arena can, as an explicit
  experiment, reuse K for bounded prompt microbatches when the topology and
  layer-local capacity checks pass. The GPU retains router weights; only selected expert IDs enter
  the CPU cache-control path. Per-sweep route bitmaps measure reuse without
  enabling speculative admission. Expert-major GGUF remains the recommended source layout for DS4 and
GPT-OSS; compatible stock MoE models can promote experts from their existing
resident host tensors without allocating a redundant L2.

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
  routed-MoE prefill-microbatch experiment, Gemma4/Qwen3/Qwen3.6/Ornith/GPT-OSS trial recipes are in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Performance

Values below are experimental decode throughput. Each row names its paired
control; the current layout comparison keeps the arena enabled in both arms,
while the arena comparisons use an explicit mmap control. Raw measurements,
settings, calculations, and evidence limitations are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

These rows describe earlier runtime revisions and remain **historical evidence**.
They are not v0.1.3 throughput claims. The current release-candidate qualification
for Gemma4, Qwen3, Qwen3.6, Ornith, GPT-OSS, and DeepSeek4 is recorded in
[`docs/releases/v0.1.3.md`](docs/releases/v0.1.3.md) and
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#v013-release-candidate-qualification-2026-08-31).
The historical DS4 2,000-token capacity observations remain documented separately
in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#historical-ds4-capacity-observations).

### v0.1.3 qualification snapshot

Fresh `llama-server` release-candidate qualification on the reference Windows
CUDA workstation produced the following 3-start 256-token decode medians. These
are release evidence, not universal presets:

| Model / v0.1.3 path | Median decode | Range |
| --- | ---: | ---: |
| Gemma4 26B-A4B, K1440/R16/P16 | **21.651 tok/s** | 21.253-21.672 |
| Qwen3 30B-A3B, K1440/R16/P16 | **19.261 tok/s** | 17.360-20.164 |
| Qwen3.6 35B-A3B, no expert cache | **11.011 tok/s** | 10.210-11.392 |
| Ornith 1.0 35B, K1920/R16/P16 | **13.967 tok/s** | 13.948-14.004 |
| GPT-OSS 120B, 18 GiB managed L2 | **3.344 tok/s** | 3.335-3.356 (host-memory pressure) |
| DeepSeek V4 Flash, 8 GiB L2 + K216/R12/P12 + FRONT | **1.944 tok/s** | one complete 2,048-token decode; low host-memory headroom |

Qwen3.6 K1440 was also correct but slower (9.279 tok/s median), so the release
recommendation remains the matched no-cache path. DeepSeek4 also passed a separate 3-start 64-token determinism gate after the FRONT completion fence was added; all three runs produced the same token hash. The 2,048-token row above is a depth/stability result, not a replacement for the historical 18 GiB benchmark.

### Historical performance evidence

| Model | Baseline path | Siliang path | Speedup | Evidence |
| --- | --- | --- | ---: | --- |
| DeepSeek V4 Flash 0731, expert-major layout | Stock GGUF, 18 GiB arena: 2.274 tok/s median (2.269-2.407) | Expert-major GGUF, same 18 GiB arena: 2.774 tok/s median (2.689-2.850) | 1.22x (+22.0%) | [Current fully cold layout benchmark](docs/PERFORMANCE.md#current-fully-cold-expert-major-layout-benchmark-2026-08-10), n=3 per arm |
| DeepSeek V4 Flash 0731, stock GGUF | Stock GGUF mmap: 1.375 tok/s median (1.375-1.414) | Same stock GGUF, 18 GiB arena: 2.291 tok/s median (2.217-2.385) | 1.67x (+66.6%) | [Current same-file benchmark](docs/PERFORMANCE.md#current-stock-gguf-arena-benchmark-2026-08-10), n=3 per arm |
| DeepSeek V4 (pre-0731) | Stock GGUF mmap: 1.098 tok/s median (1.083-1.104) | 2.246 tok/s median (2.171-2.257) | 2.05x (+104.6%) | [Historical matched run](docs/PERFORMANCE.md#deepseek-v4-pre-0731), n=3 per arm |
| gpt-oss-120B | Same repacked GGUF on mmap: 1.972 tok/s median (1.964-2.030) | 4.052 tok/s median (3.953-4.094) | 2.06x (+105.5%) | [Historical matched run](docs/PERFORMANCE.md#gpt-oss-120b), n=3 per arm |

The current 0731 layout row directly isolates the present repack: both arms use
the same binary, disk, 18 GiB arena, and request, with the standby list purged
before every process start.
All six 256-token outputs were byte-identical, and every expert-major repetition
was faster than every stock repetition. The repack moved nearly the same bytes
but reduced engine expert-read requests from 34,866 to 11,653 and total process
read operations from 37,945 to 14,418. This +22.0% result describes the current
experimental layout, not a guaranteed gain or a ceiling for future layout and
routing work.

The separate current stock-GGUF row isolates the arena against mmap. The older
DeepSeek row is a matched historical experiment. A comparable gpt-oss stock
GGUF control was not retained, so that row isolates the arena using the same
repacked file on both paths; it must not be presented as a stock-model
comparison. The earlier 0731 expert-major precursor remains documented in the
detailed evidence, but it is no longer the basis for the direct layout claim.
See the evidence labels when comparing results from different tiers.

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
  limited to at most 256 experts per layer and layer-local K capacity.
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

Actions retains the verified packages for 30 days. Maintainers review them before
attaching them to the corresponding [GitHub Release](https://github.com/SpeederX/siliang-engine/releases).
For v0.1.3, see the [release notes](docs/releases/v0.1.3.md).

## License

The upstream `llama.cpp` source at the repository root remains under its MIT
License in [`LICENSE`](LICENSE). Siliang Engine additions are available under
the separate [`licenses/SILIANG-ENGINE-MIT.txt`](licenses/SILIANG-ENGINE-MIT.txt).
Bundled third-party components retain their own notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
