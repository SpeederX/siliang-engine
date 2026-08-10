# Runtime configuration

Siliang Engine is configured per process. Set the intended values in the same
PowerShell session that launches `llama-cli` or `llama-server`, and reset them
before changing benchmark arms or models.

The arena is opt-in. If `SILIANGEM_CACHE_MIB` is not set, Siliang uses the
ordinary mmap path and does not reserve arena memory. Empty, zero, negative,
non-integer, and out-of-range values are rejected and also fall back to mmap;
there is no implicit arena size. `SILIANGEM_DISABLE` takes precedence when it is
set to a nonzero value.

## Environment helper

`scripts/siliang-env.ps1` is the supported entry point for replacing,
inspecting, disabling, and clearing the process-local Siliang settings. Run it
from the same PowerShell session that will launch the engine:

```powershell
# Enable the arena with a workload-specific budget.
.\scripts\siliang-env.ps1 -CacheMiB <measured-MiB>

# Optional diagnostics for an enabled arena.
.\scripts\siliang-env.ps1 -CacheMiB <measured-MiB> -Verbose
.\scripts\siliang-env.ps1 -CacheMiB <measured-MiB> -NoMemoryReport

# Inspect, select a deliberate mmap control, or clear all managed settings.
.\scripts\siliang-env.ps1 -Show
.\scripts\siliang-env.ps1 -Disable
.\scripts\siliang-env.ps1 -Reset
```

Enabling the arena also enables deferred waits and disables mmap prefetch.
`-Verbose` enables cache statistics. `-NoMemoryReport` suppresses periodic
system-memory reporting. `-Disable` clears the arena settings and selects the
mmap control. `-Reset` clears every variable managed by the helper, while
`-Show` changes nothing and displays their current process values.

## Settings

| Name                     | Intended behavior                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `SILIANGEM_CACHE_MIB`  | Positive integer arena budget in MiB. Setting it explicitly opts into the arena; when absent or invalid, mmap is used.                       |
| `SILIANGEM_DISABLE`    | Disables the arena and uses mmap. This is useful for a deliberate control run.                                                             |
| `SILIANGEM_VERBOSE`    | Emits periodic and final cache statistics when enabled.                                                                                    |
| `SILIANGEM_MEM_REPORT` | Set to `0` to suppress the periodic system-memory query and its output.                                                                  |
| `SILIANGEM_DEFER`      | Set to `0` to disable deferred waits; other values leave them enabled.                                                                   |
| `GGML_MOE_PREFETCH`    | Opts into the separate mmap-page prefetch hint. It is off by default and is not arena prefetch.                                            |

The arena reserves and commits real memory. Leave headroom for the operating
system, non-expert model weights, KV cache, applications, and GPU shared-memory
pressure. Paging the arena can remove the intended throughput benefit.

Telemetry collection and report emission are different controls. Verbose mode
changes what is printed, while the engine can continue updating its diagnostic
counters. The memory-report switch controls only the periodic system-memory
query and its output.

`GGML_MOE_PREFETCH` applies only after the arena path declines an access. It is
therefore not a way to warm or retain the Siliang RAM arena. Leave it disabled
unless mmap prefetch is the one declared variable in a dedicated experiment.

## Deliberate mmap control

Use an arena-disabled process only as a correctness or performance baseline,
then reset the environment before the next arm:

```powershell
.\scripts\siliang-env.ps1 -Disable
& "<path-to-llama-cli.exe>" -m "<model.gguf>" -p "<prompt>" -n 32
.\scripts\siliang-env.ps1 -Reset
```

For controlled comparisons, use the workflow in
[`../scripts/README.md`](../scripts/README.md).
