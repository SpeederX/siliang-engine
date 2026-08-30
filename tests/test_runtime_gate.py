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
    "Assert-TypedExpertCacheHelp",
    "Assert-NoModelOverrideArguments",
    "Assert-DistinctServerPaths",
    "Get-CellExpertCacheArguments",
    "Assert-ExpertMajorEvidence",
    "Assert-ArenaEvidence",
    "Get-FileSha256",
    "Get-ArtifactIdentity",
    "Get-StringSha256",
    "Get-RuntimeIdentity",
    "Assert-DistinctRuntimeIdentities",
    "Get-DeviceModelBufferEvidence",
    "Assert-ExpectedCellTermination",
    "Get-MemoryPressureSummary",
)


class RuntimeGateContractTests(unittest.TestCase):
    def test_declared_l1_policy_matches_deepseek_profile(self) -> None:
        source = RUNTIME_GATE.read_text(encoding="utf-8")
        self.assertIn("l1Policy = 'slfu'", source)
        self.assertNotIn("l1Policy = 'lfu'", source)

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
Assert-Rejected @('--parallel', '2')
Assert-Rejected @('--expert-cache')
Assert-Rejected @('--expert-cache-l2-mib=8192')
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

    def test_typed_help_preflight_rejects_legacy_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            typed = root / "typed-server.cmd"
            legacy = root / "legacy-server.cmd"
            deceptive = root / "deceptive-server.cmd"
            typed.write_text(
                "@echo off\n"
                "echo --expert-cache --no-expert-cache --expert-cache-l2-mib --expert-cache-l2-policy\n"
                "echo --expert-cache-l1-k --expert-cache-exchange-r --expert-cache-elevator-p --expert-cache-l1-policy\n"
                "echo --expert-cache-roll --expert-cache-prefill --no-expert-cache-prefill\n"
                "echo --expert-cache-memory-report --no-expert-cache-memory-report\n"
                "echo --expert-cache-deferred-wait --no-expert-cache-deferred-wait\n"
                "exit /b 0\n",
                encoding="ascii",
            )
            legacy.write_text("@echo off\necho legacy help\nexit /b 0\n", encoding="ascii")
            deceptive.write_text(
                "@echo off\n"
                "echo --expert-cache --no-expert-cache --expert-cache-l2-mib --expert-cache-l2-policy\n"
                "echo --expert-cache-l1-k --expert-cache-exchange-r --expert-cache-elevator-p --expert-cache-l1-policy\n"
                "echo --expert-cache-roll --no-expert-cache-prefill --no-expert-cache-memory-report\n"
                "echo --expert-cache-deferred-wait --no-expert-cache-deferred-wait\n"
                "exit /b 0\n",
                encoding="ascii",
            )
            self.run_powershell(
                r"""
Assert-TypedExpertCacheHelp -ServerPath $env:SILIANG_TEST_TYPED_SERVER -Label typed
$legacyRejected = $false
try {
    Assert-TypedExpertCacheHelp -ServerPath $env:SILIANG_TEST_LEGACY_SERVER -Label legacy
} catch {
    $legacyRejected = $true
}
if (-not $legacyRejected) {
    throw 'A legacy runtime without typed expert-cache help was accepted.'
}
$deceptiveRejected = $false
try {
    Assert-TypedExpertCacheHelp -ServerPath $env:SILIANG_TEST_DECEPTIVE_SERVER -Label deceptive
} catch {
    $deceptiveRejected = $true
}
if (-not $deceptiveRejected) {
    throw 'The negative memory-report spelling falsely satisfied the positive help option.'
}
""",
                environment={
                    "SILIANG_TEST_TYPED_SERVER": str(typed),
                    "SILIANG_TEST_LEGACY_SERVER": str(legacy),
                    "SILIANG_TEST_DECEPTIVE_SERVER": str(deceptive),
                },
            )

    def test_typed_expert_cache_argument_and_log_contracts(self) -> None:
        self.run_powershell(
            r"""
$script:DeepSeekExpertCacheL2MiB = 8192
$script:GptOssExpertCacheL2MiB = 18432
$script:AllowMemoryPressure = $false

$disabledArguments = @(Get-CellExpertCacheArguments -ArenaState Disabled -Profile DeepSeek4)
if (($disabledArguments -join ' ') -cne '--no-expert-cache') {
    throw 'The disabled cell did not select the typed CLI control.'
}
$disabled = Assert-ArenaEvidence -ArenaState Disabled `
    -Profile DeepSeek4 -LogText 'ordinary mmap path' -Arguments $disabledArguments
if ($disabled.armed -or -not $disabled.explicitlyDisabled) {
    throw 'The typed CLI disabled control was not accepted.'
}

$deepSeekArguments = @(Get-CellExpertCacheArguments -ArenaState Arena -Profile DeepSeek4)
$deepSeekText = $deepSeekArguments -join ' '
foreach ($expected in @(
    '--expert-cache', '--expert-cache-l2-mib 8192', '--expert-cache-l2-policy lru',
    '--expert-cache-l1-k 216', '--expert-cache-exchange-r 12',
    '--expert-cache-elevator-p 12', '--expert-cache-l1-policy slfu',
    '--admit-k-cold on', '--demote-k-hot on', '--expert-cache-roll deepseek4', '--expert-cache-memory-report',
    '--expert-cache-deferred-wait'
)) {
    if (-not $deepSeekText.Contains($expected)) {
        throw "The DeepSeek4 profile is missing: $expected"
    }
}

$gptArguments = @(Get-CellExpertCacheArguments -ArenaState Arena -Profile GptOss)
$gptText = $gptArguments -join ' '
if (-not $gptText.Contains('--expert-cache-l2-mib 18432') -or
    -not $gptText.Contains('--expert-cache-l2-policy lru') -or
    -not $gptText.Contains('--expert-cache-roll off') -or
    -not $gptText.Contains('--no-expert-cache-prefill') -or
    $gptText.Contains('--expert-cache-l1-k')) {
    throw 'The GPT-OSS profile did not remain an L2-only diagnostic.'
}

$metadata = [pscustomobject]@{ status = 'expert-major-metadata-ok' }
$loader = Assert-ExpertMajorEvidence `
    -LogText 'load_model: siliangem expert-major: 2 layers x 2 experts' `
    -MetadataEvidence $metadata
if (-not $loader.loaderInfoMarkerPresent) {
    throw 'The expert-major loader marker was not recognized.'
}

$arenaLog = @'
llama_context: Siliang expert cache enabled: L2=8192 MiB K=216 R=12 P=12 roll=1
siliangem: source 2x2 experts, 1.000 MiB each, cache 4 slots (0.004 GiB), policy 1, unbuffered+overlapped, deferred-wait ON
siliang_moe_runtime: armed decode-only arena layers=43/43 schemas=1 layout=global experts=256 top_k=6 K=216 R=12 P=12 policy=slfu source=host-l2 transport=private-stream-staged arena=1536 MiB pinned=81 MiB
expert cache: DeepSeek-V4 FRONT slab armed bank=64 MiB host=2752 MiB resident_layer=0
siliang_moe_runtime: serving decode route map=1 layer=0 K_hits=0 K_misses=6 admissions=6 R_bypass=0 H2D_ops=6 failure=0
siliang_moe_runtime: serving decode compute_wait=1 layer=0 failure=0
siliang_ds4_front_slab: serving decode tokens=1 copies=1 waits=2 H2D_bytes=67108864 failure=0
siliangem[periodic]: 10 lookups, 6 hits (60.0%), 4 misses, expert-level hit 40.0%, 1.250 GiB read from slab
siliangem[submit]: 1 sync (25.0%), 3 pending (75.0%)
siliangem[mem]: available 8192 MiB | file cache 4096 MiB | commit charged 1000 MiB
'@
$arena = Assert-ArenaEvidence -ArenaState Arena -Profile DeepSeek4 -LogText $arenaLog -Arguments $deepSeekArguments
if (-not $arena.armed -or $arena.lookups -ne 10 -or $arena.hits -ne 6 -or
    $arena.misses -ne 4 -or $arena.synchronousSubmissions -ne 1 -or
    $arena.pendingSubmissions -ne 3 -or -not $arena.typedConfigurationPresent -or
    $arena.moeRuntime.firstRouteH2dOperations -ne 6 -or $arena.frontSlab.copies -ne 1) {
    throw 'The typed expert-cache telemetry contract was not parsed correctly.'
}

$gptLog = @'
llama_context: Siliang expert cache enabled: L2=18432 MiB K=0 R=0 P=0 roll=0
siliangem: source 36x128 experts, 4.000 MiB each, cache 4608 slots (18.000 GiB), policy 0, unbuffered+overlapped, deferred-wait ON
siliangem[periodic]: 12 lookups, 8 hits (66.7%), 4 misses, expert-level hit 33.3%, 1.000 GiB read from slab
siliangem[submit]: 1 sync (25.0%), 3 pending (75.0%)
siliangem[mem]: available 4096 MiB | file cache 2048 MiB | commit charged 12000 MiB
'@
$gptArena = Assert-ArenaEvidence -ArenaState Arena -Profile GptOss -LogText $gptLog -Arguments $gptArguments
if (-not $gptArena.armed -or $gptArena.moeRuntime -ne $null -or $gptArena.frontSlab -ne $null) {
    throw 'The GPT-OSS L2-only telemetry contract was not accepted.'
}

$wrongResolvedRejected = $false
try {
    $wrongResolved = $gptLog.Replace('L2=18432 MiB K=0 R=0 P=0 roll=0', 'L2=12288 MiB K=0 R=0 P=0 roll=0')
    Assert-ArenaEvidence -ArenaState Arena -Profile GptOss -LogText $wrongResolved -Arguments $gptArguments
} catch {
    $wrongResolvedRejected = $true
}
if (-not $wrongResolvedRejected) {
    throw 'A GPT-OSS log with the wrong resolved L2 capacity was accepted.'
}

$missingFrontRejected = $false
try {
    $missingFront = $arenaLog.Replace(
        'siliang_ds4_front_slab: serving decode tokens=1 copies=1 waits=2 H2D_bytes=67108864 failure=0', '')
    Assert-ArenaEvidence -ArenaState Arena -Profile DeepSeek4 -LogText $missingFront `
        -Arguments $deepSeekArguments
} catch {
    $missingFrontRejected = $true
}
if (-not $missingFrontRejected) {
    throw 'A DeepSeek4 cell without FRONT activity evidence was accepted.'
}

$retiredMarkersRejected = $false
try {
    $legacyLog = $arenaLog.Replace('siliangem', 'behemoth')
    Assert-ArenaEvidence -ArenaState Arena -Profile DeepSeek4 -LogText $legacyLog -Arguments $deepSeekArguments
} catch {
    $retiredMarkersRejected = $true
}
if (-not $retiredMarkersRejected) {
    throw 'Retired log markers were accepted as current evidence.'
}
"""
        )

    def test_runtime_gate_does_not_configure_expert_cache_through_environment(self) -> None:
        source = RUNTIME_GATE.read_text(encoding="utf-8")
        retired_prefix = "SILIANG" + "EM_"
        retired_prefetch_prefix = "GGML_" + "MOE_"

        self.assertNotIn(f"$env:{retired_prefix}", source)
        self.assertNotIn(f"$env:{retired_prefetch_prefix}", source)
        self.assertIn("Get-CellExpertCacheArguments", source)
        self.assertIn("'--parallel', '1'", source)
        self.assertIn("'-c', '4096', '-b', '512', '-ub', '512', '-t', '2', '-tb', '2'", source)
        self.assertIn("'-ngl', '99', '-ncmoe', '36'", source)

    def test_runtime_identity_covers_executable_and_adjacent_dlls(self) -> None:
        source = RUNTIME_GATE.read_text(encoding="utf-8")
        self.assertNotIn("Get-FileHash", source)
        self.assertIn("[Security.Cryptography.SHA256]::Create()", source)

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
function Get-FileHash { throw 'Get-FileHash must not be required by the runtime identity path.' }
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
