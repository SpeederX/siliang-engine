# Scripts

Run these entry points from PowerShell at the repository root. Use paths outside
the repository for generated builds, models, logs, and benchmark results.

## Build

```powershell
.\scripts\build.ps1 -Backend Cpu -BuildRoot "<build-root>"

.\scripts\build.ps1 `
    -Backend Cuda `
    -BuildRoot "<build-root>"
```

When `-CudaArchitecture` is omitted, CUDA delegates architecture selection to
the pinned upstream `llama.cpp` source and installed CUDA toolkit. A local
diagnostic build can still select a
single target, for example `-CudaArchitecture 75` for compute capability 7.5.
The script configures a Release build with network-dependent upstream features
disabled. CPU code uses the same portable dispatch model as official
`llama.cpp` packages:
`GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`, and
`GGML_CPU_ALL_VARIANTS=ON`. The package selects the best compatible CPU backend
at runtime instead of inheriting the instruction set of the build machine.

The helper above is the supported Windows release entry point. Linux and macOS
currently use the ordinary upstream CMake build surface; release CI compiles
representative Linux CPU and macOS Metal configurations to preserve fork
compatibility. The Siliang expert arena itself remains Windows-only.

## Model preparation

`repack-model.ps1`, `preflight_repack.py`, and `check_expert_major.py` implement
the supported expert-major preparation workflow. Follow
[`../tools/README.md`](../tools/README.md) rather than invoking the converter
directly.

## Runtime configuration

The runtime is configured directly through the shared `llama-cli` and
`llama-server` options:

```powershell
& "<llama-server.exe>" -m "<model.gguf>" `
    -ngl 99 -ncmoe <all-routed-layers> `
    --parallel 1 `
    --expert-cache `
    --expert-cache-l2-mib <measured-MiB> `
    --expert-cache-l2-policy lfu
```

Use `--no-expert-cache` for a control. See
[`../docs/CONFIGURATION.md`](../docs/CONFIGURATION.md) for the K/R/P rules, DS4
prototype command, policies, and cross-model diagnostic recipes.

## Model-free checks

Verify the pinned upstream base and Siliang delta, then run the Python and
PowerShell checks:

```powershell
.\scripts\verify-snapshot.ps1
.\scripts\test.ps1 -PythonExecutable "<python>"
```

When a build directory is available, selected upstream CTest cases can be
included:

```powershell
.\scripts\test.ps1 `
    -PythonExecutable "<python>" `
    -BuildDirectory "<cpu-build-directory>" `
    -CTestRegex "<upstream-test-regex>"
```

These checks validate the public fork and tooling. They do not establish
model correctness or throughput.

## Runtime gate

`runtime-gate.ps1` performs controlled reference/candidate model checks and
retains a complete evidence directory. It requires distinct reference and
candidate runtime identities and owns the primary model paths, typed cache
arguments, serial server mode, prompt, sampling configuration, and cell
lifecycle.

Both executables must be typed v0.1.3-compatible builds whose `--help` exposes
the expert-cache flags. The script intentionally has no v0.1.2 environment
fallback; use a separate legacy harness for historical binaries.

```powershell
.\scripts\runtime-gate.ps1 `
    -ReferenceServer "<reference-llama-server.exe>" `
    -CandidateServer "<candidate-llama-server.exe>" `
    -DeepSeekModel "<deepseek-expert-major.gguf>" `
    -GptOssModel "<gpt-oss-expert-major.gguf>" `
    -GptOssMaxDeviceModelBufferMiB "<measured-limit>" `
    -DeepSeekExpertCacheL2MiB 12288 `
    -GptOssExpertCacheL2MiB 18432 `
    -ResultsRoot "<results-root>"
```

The gate records executable and adjacent-DLL identities, effective runtime
configuration, exact CLI arguments, output evidence, cache activity, teardown
state, and memory pressure. Its DeepSeek4 arena cells use K216/R12/P12,
cumulative-LFU admission/bypass, FRONT rolling, and `--parallel 1`; GPT-OSS remains a separate L2-only diagnostic
profile. If `-AllowMemoryPressure` is deliberately used, the pressure counts
remain in `summary.json` and must be disclosed with the result.

A passing process launch or model load is not enough. Publish a performance
cell only when deterministic correctness passes, the intended path is proven
active, no fallback occurred, and the machine met the declared isolation
requirements.

## Script inventory

| Script | Purpose |
| --- | --- |
| `build.ps1` | Configure and build the CPU or CUDA runtime. |
| `repack-model.ps1` | Safely orchestrate expert-major conversion. |
| `preflight_repack.py` | Reject unsupported source geometry before writing. |
| `check_expert_major.py` | Validate the finalized expert-major structure. |
| `verify-snapshot.ps1` | Verify the pinned upstream base and Siliang source delta. |
| `test.ps1` | Run model-free checks and optional selected CTests. |
| `runtime-gate.ps1` | Produce controlled reference/candidate runtime evidence. |

Long-running model gates should always write to a unique external result
directory and be judged from their final status marker, not from a quiet log or
partially written JSON.
