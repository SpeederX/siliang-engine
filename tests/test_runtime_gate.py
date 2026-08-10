from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "runtime-gate.ps1"
POWERSHELL_FUNCTIONS = (
    "Assert-NoModelOverrideArguments",
    "Assert-DistinctServerPaths",
    "Get-ManagedEnvironment",
    "Clear-ManagedEnvironment",
    "Set-CellEnvironment",
    "Assert-ExpertMajorEvidence",
    "Assert-ArenaEvidence",
    "Get-ArtifactIdentity",
    "Get-StringSha256",
    "Get-RuntimeIdentity",
    "Assert-DistinctRuntimeIdentities",
    "Get-DeviceModelBufferEvidence",
    "Assert-ExpectedCellTermination",
    "Get-MemoryPressureSummary",
)


class RuntimeGateContractTests(unittest.TestCase):
    def run_powershell(self, body: str, *, environment: dict[str, str] | None = None) -> None:
        function_names = ", ".join(f"'{name}'" for name in POWERSHELL_FUNCTIONS)
        prelude = f"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$sourcePath = $env:SILIANG_TEST_RUNTIME_GATE_SOURCE
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $sourcePath, [ref]$tokens, [ref]$parseErrors
)
if (@($parseErrors).Count -ne 0) {{
    throw ('runtime-gate.ps1 parse failure: ' + ((@($parseErrors | ForEach-Object {{ $_.Message }})) -join '; '))
}}
foreach ($functionName in @({function_names})) {{
    $definition = $ast.Find({{
        param($candidate)
        $candidate -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $candidate.Name -ceq $functionName
    }}, $true)
    if ($null -eq $definition) {{
        throw "Function not found: $functionName"
    }}
    Invoke-Expression $definition.Extent.Text
}}
"""
        encoded = base64.b64encode((prelude + body).encode("utf-16-le")).decode("ascii")
        process_environment = os.environ.copy()
        process_environment["SILIANG_TEST_RUNTIME_GATE_SOURCE"] = str(RUNTIME_GATE)
        if environment:
            process_environment.update(environment)
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            cwd=REPOSITORY_ROOT,
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"PowerShell contract test failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_rejects_model_overrides_and_same_server_path(self) -> None:
        self.run_powershell(
            r"""
function Assert-Rejected {
    param([string[]]$Values)
    $rejected = $false
    try {
        Assert-NoModelOverrideArguments -Label test -Arguments $Values
    } catch {
        $rejected = $true
    }
    if (-not $rejected) {
        throw ('Model override was accepted: ' + ($Values -join ' '))
    }
}
Assert-Rejected @('-m', 'other.gguf')
Assert-Rejected @('--model', 'other.gguf')
Assert-Rejected @('-m=other.gguf')
Assert-Rejected @('--model=other.gguf')
Assert-Rejected @('-mu', 'https://example.invalid/model.gguf')
Assert-Rejected @('--model-url=https://example.invalid/model.gguf')
Assert-Rejected @('-hf', 'owner/model')
Assert-Rejected @('--hf-repo=owner/model')
Assert-Rejected @('-hff', 'model.gguf')
Assert-Rejected @('--hf-file=model.gguf')
Assert-NoModelOverrideArguments -Label test -Arguments @('-mmproj', 'vision.gguf', '--model-draft', 'draft.gguf')

$samePathRejected = $false
try {
    Assert-DistinctServerPaths -ReferencePath 'C:\runtime\server.exe' -CandidatePath 'c:\RUNTIME\server.exe'
} catch {
    $samePathRejected = $true
}
if (-not $samePathRejected) {
    throw 'Case-insensitive identical server paths were accepted.'
}
"""
        )

    def test_siliangem_environment_and_log_contracts(self) -> None:
        self.run_powershell(
            r"""
$script:managedEnvironmentPattern = '^(SILIANGEM_|GGML_MOE_)'
$script:ArenaMiB = 4096
$script:AllowMemoryPressure = $false

$env:BEHEMOTH_LEGACY_SENTINEL = 'untouched'
$env:SILIANGEM_STALE = 'remove-me'
$env:GGML_MOE_PREFETCH = '1'
Clear-ManagedEnvironment
if (Test-Path Env:SILIANGEM_STALE) {
    throw 'Clear-ManagedEnvironment retained a SILIANGEM variable.'
}
if (Test-Path Env:GGML_MOE_PREFETCH) {
    throw 'Clear-ManagedEnvironment retained a GGML_MOE variable.'
}
if ($env:BEHEMOTH_LEGACY_SENTINEL -cne 'untouched') {
    throw 'The retired namespace is still managed as an alias.'
}

Set-CellEnvironment -ArenaState Disabled
if ($env:SILIANGEM_VERBOSE -cne '1' -or $env:SILIANGEM_DISABLE -cne '1' -or
    $env:GGML_MOE_PREFETCH -cne '0') {
    throw 'The disabled cell did not use the SILIANGEM environment contract.'
}
if (Test-Path Env:BEHEMOTH_DISABLE) {
    throw 'The disabled cell emitted a retired environment variable.'
}
$disabled = Assert-ArenaEvidence -ArenaState Disabled `
    -LogText 'siliangem: disabled by SILIANGEM_DISABLE - using mmap'
if ($disabled.armed -or -not $disabled.explicitlyDisabled) {
    throw 'The SILIANGEM disabled marker was not accepted.'
}

Set-CellEnvironment -ArenaState Arena
if ($env:SILIANGEM_VERBOSE -cne '1' -or $env:SILIANGEM_CACHE_MIB -cne '4096' -or
    $env:SILIANGEM_DEFER -cne '1' -or (Test-Path Env:SILIANGEM_DISABLE)) {
    throw 'The arena cell did not use the SILIANGEM environment contract.'
}

$metadata = [pscustomobject]@{ status = 'expert-major-metadata-ok' }
$loader = Assert-ExpertMajorEvidence `
    -LogText 'load_model: siliangem expert-major: 2 layers x 2 experts' `
    -MetadataEvidence $metadata
if (-not $loader.loaderInfoMarkerPresent) {
    throw 'The SILIANGEM loader marker was not recognized.'
}

$arenaLog = @'
siliangem: slab 2x2 experts, 1.000 MiB each, cache 4 slots (unbuffered+overlapped)
siliangem[decode]: 10 lookups, 6 hits (60.0%), 4 misses, 1.250 GiB read from slab
siliangem[submit]: 1 sync (25.0%), 3 pending (75.0%)
siliangem[mem]: available 8192 MiB | file cache 4096 MiB | commit charged 1000 MiB
'@
$arena = Assert-ArenaEvidence -ArenaState Arena -LogText $arenaLog
if (-not $arena.armed -or $arena.lookups -ne 10 -or $arena.hits -ne 6 -or
    $arena.misses -ne 4 -or $arena.synchronousSubmissions -ne 1 -or
    $arena.pendingSubmissions -ne 3) {
    throw 'The SILIANGEM telemetry contract was not parsed correctly.'
}

$retiredMarkersRejected = $false
try {
    $legacyLog = $arenaLog.Replace('siliangem', 'behemoth')
    Assert-ArenaEvidence -ArenaState Arena -LogText $legacyLog
} catch {
    $retiredMarkersRejected = $true
}
if (-not $retiredMarkersRejected) {
    throw 'Retired log markers were accepted as current evidence.'
}

Clear-ManagedEnvironment
Remove-Item Env:BEHEMOTH_LEGACY_SENTINEL -ErrorAction SilentlyContinue
"""
        )

    def test_runtime_identity_covers_executable_and_adjacent_dlls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            copy = root / "copy"
            for directory in (reference, candidate, copy):
                directory.mkdir()
                (directory / "runtime.dll").write_bytes(b"same-dll")
            (reference / "llama-server.exe").write_bytes(b"reference-exe")
            (candidate / "llama-server.exe").write_bytes(b"reference-exe")
            (candidate / "runtime.dll").write_bytes(b"candidate-dll")
            (copy / "llama-server.exe").write_bytes(b"reference-exe")

            self.run_powershell(
                r"""
$reference = Get-RuntimeIdentity -ServerPath $env:SILIANG_TEST_REFERENCE -Label reference
$candidate = Get-RuntimeIdentity -ServerPath $env:SILIANG_TEST_CANDIDATE -Label candidate
$copy = Get-RuntimeIdentity -ServerPath $env:SILIANG_TEST_COPY -Label copy
if ($reference.adjacentDllCount -ne 1) {
    throw "Expected one adjacent DLL, got $($reference.adjacentDllCount)."
}
Assert-DistinctRuntimeIdentities -ReferenceIdentity $reference -CandidateIdentity $candidate
if ($reference.fullRuntimeSha256 -cne $copy.fullRuntimeSha256) {
    throw 'Content-identical copied runtimes did not receive the same full identity.'
}
$identicalRejected = $false
try {
    Assert-DistinctRuntimeIdentities -ReferenceIdentity $reference -CandidateIdentity $copy
} catch {
    $identicalRejected = $true
}
if (-not $identicalRejected) {
    throw 'Content-identical copied runtimes were accepted as a reference/candidate pair.'
}
""",
                environment={
                    "SILIANG_TEST_REFERENCE": str(reference / "llama-server.exe"),
                    "SILIANG_TEST_CANDIDATE": str(candidate / "llama-server.exe"),
                    "SILIANG_TEST_COPY": str(copy / "llama-server.exe"),
                },
            )

    def test_exit_pressure_and_gpt_allocation_evidence_are_fail_closed(self) -> None:
        self.run_powershell(
            r"""
$terminationEvidence = Assert-ExpectedCellTermination -Termination ([pscustomobject]@{
    processWasRunningBeforeTermination = $true
    stopProcessForceRequested = $true
    processExited = $true
    termination = 'forced-after-cell'
    exitCode = 0
})
if ($terminationEvidence.status -cne 'expected-harness-forced-termination') {
    throw 'Expected termination evidence was not retained.'
}
if (-not $terminationEvidence.processWasRunningBeforeTermination -or
    -not $terminationEvidence.stopProcessForceRequested -or
    -not $terminationEvidence.processExited -or
    $terminationEvidence.exitCode -ne 0) {
    throw 'Complete harness termination evidence was not retained.'
}
foreach ($validExitCode in @(-1, 1)) {
    $alternateEvidence = Assert-ExpectedCellTermination -Termination ([pscustomobject]@{
        processWasRunningBeforeTermination = $true
        stopProcessForceRequested = $true
        processExited = $true
        termination = 'forced-after-cell'
        exitCode = $validExitCode
    })
    if ($alternateEvidence.exitCode -ne $validExitCode) {
        throw "Observed exit code $validExitCode was not retained."
    }
}
foreach ($badTermination in @(
    [pscustomobject]@{ processWasRunningBeforeTermination = $false; stopProcessForceRequested = $false; processExited = $true; termination = 'server-exited'; exitCode = 0 },
    [pscustomobject]@{ processWasRunningBeforeTermination = $true; stopProcessForceRequested = $false; processExited = $true; termination = 'forced-after-cell'; exitCode = 0 },
    [pscustomobject]@{ processWasRunningBeforeTermination = $true; stopProcessForceRequested = $true; processExited = $true; termination = 'server-exited'; exitCode = 0 },
    [pscustomobject]@{ processWasRunningBeforeTermination = $true; stopProcessForceRequested = $true; processExited = $false; termination = 'forced-after-cell'; exitCode = 0 },
    [pscustomobject]@{ processWasRunningBeforeTermination = $true; stopProcessForceRequested = $true; processExited = $true; termination = 'forced-after-cell'; exitCode = $null },
    [pscustomobject]@{ processWasRunningBeforeTermination = $true; processExited = $true; termination = 'forced-after-cell'; exitCode = 0 }
)) {
    $rejected = $false
    try {
        Assert-ExpectedCellTermination -Termination $badTermination
    } catch {
        $rejected = $true
    }
    if (-not $rejected) {
        throw 'An unexpected termination outcome was accepted.'
    }
}

$cells = @(
    [pscustomobject]@{
        name = 'arena-pressure'
        arenaEvidence = [pscustomobject]@{ memoryPressureReportCount = 2; memoryPressureObserved = $true }
    },
    [pscustomobject]@{
        name = 'arena-clear'
        arenaEvidence = [pscustomobject]@{ memoryPressureReportCount = 0; memoryPressureObserved = $false }
    }
)
$pressure = Get-MemoryPressureSummary -Cells $cells -Allowed $true
if (-not $pressure.allowMemoryPressure -or $pressure.gateEnforced -or
    $pressure.reportCount -ne 2 -or $pressure.observedCellCount -ne 1 -or
    $pressure.observedCells[0] -cne 'arena-pressure') {
    throw 'Memory-pressure summary did not preserve the override and report counts.'
}

$allocationLog = @'
load_tensors: offloading output layer to GPU
load_tensors: offloading 35 repeating layers to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors: CUDA0 model buffer size = 6461.28 MiB
'@
$allocation = Get-DeviceModelBufferEvidence -LogText $allocationLog -MaximumMiB 7000
if ($allocation.status -cne 'observed-allocation-and-offload-within-predeclared-ceiling' -or
    $allocation.totalMiB -ne 6461.28 -or $allocation.offload.offloadedLayers -ne 37 -or
    $allocation.offload.totalLayers -ne 37 -or $allocation.offload.lines.Count -ne 3) {
    throw 'GPT allocation/offload evidence was not retained as expected.'
}
$missingOffloadRejected = $false
try {
    Get-DeviceModelBufferEvidence -LogText 'CUDA0 model buffer size = 6461.28 MiB' -MaximumMiB 7000
} catch {
    $missingOffloadRejected = $true
}
if (-not $missingOffloadRejected) {
    throw 'A GPT allocation record without offload evidence was accepted.'
}
"""
        )


if __name__ == "__main__":
    unittest.main()
