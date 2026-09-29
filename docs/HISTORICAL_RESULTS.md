# Historical results

This page keeps the performance results of earlier Siliang runtime revisions.
They remain valid evidence for the revision and configuration they name, but
they are not claims about the current release. Current results are in the root
[`README.md`](../README.md#performance), and each release records its own gate
in [`releases/`](releases/). Raw measurements, commands and evidence limits for
the benchmarks below are in [`PERFORMANCE.md`](PERFORMANCE.md).

All results come from the reference workstation described in the README.

## Release history

| Release | Focus | Headline result on the reference workstation | Notes |
| --- | --- | --- | --- |
| v0.1.3 | Typed MoE hierarchy, first cross-model qualification | Gemma4, Qwen3, Qwen3.6, Ornith, GPT-OSS and DS4 decode medians (table below) | [v0.1.3](releases/v0.1.3.md) |
| v0.1.4 | K/L2 duplicate-residency fix, embedded Web UI | DS4 two-request regression passed; no new performance claim | [v0.1.4](releases/v0.1.4.md) |
| v0.1.5 | Double-buffered DS4 FRONT | DS4 bounded prefill 29.91 tok/s on 512 prompt tokens; 3/3 identical 64-token decodes | [v0.1.5](releases/v0.1.5.md) |
| v0.1.6 | Managed no-mmap source for DS4 | Same output as mmap. Working set at ready is 3.5 GiB instead of 10.7 GiB (bounded prefill) and 5.2 GiB instead of 12.4 GiB (K/R/P + FRONT decode). `-ub 1024` is the fastest DS4 prefill ubatch at 28.65 tok/s | [v0.1.6](releases/v0.1.6.md) |
| v0.1.7 | Upstream `b10270` to `b11188` | DS4 decode 2.044 tok/s and bounded prefill 28.66 tok/s, matching v0.1.6 | [v0.1.7](releases/v0.1.7.md) |

## v0.1.3 qualification snapshot

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
recommendation remains the matched no-cache path. DeepSeek4 also passed a
separate 3-start 64-token determinism gate after the FRONT completion fence was
added; all three runs produced the same token hash. The 2,048-token row above is
a depth/stability result, not a replacement for the historical 18 GiB
benchmark. See [`PERFORMANCE.md`](PERFORMANCE.md#v013-release-candidate-qualification-2026-08-31).

## Arena and layout benchmarks (2026-08)

| Model | Baseline path | Siliang path | Speedup | Evidence |
| --- | --- | --- | ---: | --- |
| DeepSeek V4 Flash 0731, expert-major layout | Stock GGUF, 18 GiB arena: 2.274 tok/s median (2.269-2.407) | Expert-major GGUF, same 18 GiB arena: 2.774 tok/s median (2.689-2.850) | 1.22x (+22.0%) | [Fully cold layout benchmark](PERFORMANCE.md#current-fully-cold-expert-major-layout-benchmark-2026-08-10), n=3 per arm |
| DeepSeek V4 Flash 0731, stock GGUF | Stock GGUF mmap: 1.375 tok/s median (1.375-1.414) | Same stock GGUF, 18 GiB arena: 2.291 tok/s median (2.217-2.385) | 1.67x (+66.6%) | [Same-file benchmark](PERFORMANCE.md#current-stock-gguf-arena-benchmark-2026-08-10), n=3 per arm |
| DeepSeek V4 (pre-0731) | Stock GGUF mmap: 1.098 tok/s median (1.083-1.104) | 2.246 tok/s median (2.171-2.257) | 2.05x (+104.6%) | [Historical matched run](PERFORMANCE.md#deepseek-v4-pre-0731), n=3 per arm |
| gpt-oss-120B | Same repacked GGUF on mmap: 1.972 tok/s median (1.964-2.030) | 4.052 tok/s median (3.953-4.094) | 2.06x (+105.5%) | [Historical matched run](PERFORMANCE.md#gpt-oss-120b), n=3 per arm |

The 0731 layout row directly isolates the expert-major repack: both arms use
the same binary, disk, 18 GiB arena, and request, with the standby list purged
before every process start. All six 256-token outputs were byte-identical, and
every expert-major repetition was faster than every stock repetition. The
repack moved nearly the same bytes but reduced engine expert-read requests from
34,866 to 11,653 and total process read operations from 37,945 to 14,418. This
+22.0% result describes that experimental layout, not a guaranteed gain or a
ceiling for future layout and routing work.

The stock-GGUF row isolates the arena against mmap. The older DeepSeek row is a
matched historical experiment. A comparable gpt-oss stock GGUF control was not
retained, so that row isolates the arena using the same repacked file on both
paths; it must not be presented as a stock-model comparison. The earlier 0731
expert-major precursor remains documented in the detailed evidence, but it is
no longer the basis for the direct layout claim. See the evidence labels in
[`PERFORMANCE.md`](PERFORMANCE.md) when comparing results from different tiers.

The historical DS4 2,000-token capacity observations are documented in
[`CONFIGURATION.md`](CONFIGURATION.md#historical-ds4-capacity-observations).
