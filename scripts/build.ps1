[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Cpu', 'Cuda')]
    [string]$Backend,

    [string]$CudaArchitecture,
    [string]$BuildRoot,
    [string]$BuildDirectory,
    [string]$Generator,
    [ValidateRange(1, 1024)]
    [int]$Parallel = [Environment]::ProcessorCount,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceDirectory = $repositoryRoot
if (-not (Test-Path -LiteralPath (Join-Path $sourceDirectory 'CMakeLists.txt') -PathType Leaf)) {
    throw "llama.cpp source was not found at the repository root: $sourceDirectory"
}

if ($Backend -eq 'Cpu' -and -not [string]::IsNullOrWhiteSpace($CudaArchitecture)) {
    throw '-CudaArchitecture is only valid with -Backend Cuda.'
}
if (-not [string]::IsNullOrWhiteSpace($BuildRoot) -and
    -not [string]::IsNullOrWhiteSpace($BuildDirectory)) {
    throw 'Use either -BuildRoot or -BuildDirectory, not both.'
}

function Resolve-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    if ([IO.Path]::IsPathRooted($PathValue)) {
        return [IO.Path]::GetFullPath($PathValue)
    }
    return [IO.Path]::GetFullPath((Join-Path $repositoryRoot $PathValue))
}

if (-not [string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $resolvedBuildDirectory = Resolve-AbsolutePath $BuildDirectory
    if (Test-Path -LiteralPath $resolvedBuildDirectory) {
        throw "Fresh-build policy: -BuildDirectory already exists: $resolvedBuildDirectory"
    }
} else {
    if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
        $BuildRoot = Join-Path 'artifacts' 'build'
    }
    $resolvedBuildRoot = Resolve-AbsolutePath $BuildRoot
    [void](New-Item -ItemType Directory -Force -Path $resolvedBuildRoot)

    $backendSlug = $Backend.ToLowerInvariant()
    do {
        $leaf = '{0}-{1}-{2}' -f $backendSlug, (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID
        $resolvedBuildDirectory = Join-Path $resolvedBuildRoot $leaf
        if (Test-Path -LiteralPath $resolvedBuildDirectory) {
            Start-Sleep -Milliseconds 1000
        }
    } while (Test-Path -LiteralPath $resolvedBuildDirectory)
}

$repoFull = [IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\')
$sourceFull = [IO.Path]::GetFullPath($sourceDirectory).TrimEnd('\')
$buildFull = [IO.Path]::GetFullPath($resolvedBuildDirectory).TrimEnd('\')
if ($buildFull -ieq $repoFull -or $buildFull -ieq $sourceFull) {
    throw 'The build directory must not be the repository root or the source directory.'
}

[void](New-Item -ItemType Directory -Path $resolvedBuildDirectory)

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ('> {0} {1}' -f $Program, ($Arguments -join ' '))
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

$cmakeArguments = @(
    '-S', $sourceDirectory,
    '-B', $resolvedBuildDirectory,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CONFIGURATION_TYPES=Release',
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_BACKEND_DL=ON',
    '-DGGML_CPU_ALL_VARIANTS=ON',
    '-DLLAMA_CURL=OFF',
    '-DLLAMA_BUILD_TESTS=ON',
    '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_UI=ON',
    '-DLLAMA_USE_PREBUILT_UI=ON'
)

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $cmakeArguments += @('-G', $Generator)
}

if ($Backend -eq 'Cuda') {
    $cmakeArguments += '-DGGML_CUDA=ON'
    if (-not [string]::IsNullOrWhiteSpace($CudaArchitecture)) {
        $cmakeArguments += ('-DCMAKE_CUDA_ARCHITECTURES={0}' -f $CudaArchitecture)
    }
} else {
    $cmakeArguments += '-DGGML_CUDA=OFF'
}

$cudaArchitectureDisplay = if ($Backend -ne 'Cuda') {
    '<not applicable>'
} elseif ([string]::IsNullOrWhiteSpace($CudaArchitecture)) {
    '<llama.cpp upstream default>'
} else {
    $CudaArchitecture
}

Write-Host 'Effective build configuration:'
Write-Host ('  source              : {0}' -f $sourceDirectory)
Write-Host ('  build               : {0}' -f $resolvedBuildDirectory)
Write-Host ('  backend             : {0}' -f $Backend)
Write-Host ('  CUDA architecture   : {0}' -f $cudaArchitectureDisplay)
Write-Host ('  generator           : {0}' -f $(if ($Generator) { $Generator } else { '<CMake default>' }))
Write-Host '  configuration       : Release'
Write-Host '  BUILD_SHARED_LIBS    : ON'
Write-Host '  GGML_NATIVE          : OFF'
Write-Host '  GGML_BACKEND_DL      : ON'
Write-Host '  GGML_CPU_ALL_VARIANTS: ON'
Write-Host '  LLAMA_CURL           : OFF'
Write-Host '  LLAMA_BUILD_TESTS    : ON'
Write-Host '  LLAMA_BUILD_TOOLS    : ON'
Write-Host '  LLAMA_BUILD_SERVER   : ON'
Write-Host '  LLAMA_BUILD_UI       : ON'
Write-Host '  LLAMA_USE_PREBUILT_UI: ON'

Invoke-Native -Program 'cmake' -Arguments $cmakeArguments

$effectiveCudaArchitectures = 'none'
$cudaArchitectureSource = 'not-applicable'
if ($Backend -eq 'Cuda') {
    if (-not [string]::IsNullOrWhiteSpace($CudaArchitecture)) {
        $effectiveCudaArchitectures = $CudaArchitecture
        $cudaArchitectureSource = 'explicit-override'
    } else {
        $configureLog = Join-Path $resolvedBuildDirectory 'CMakeFiles\CMakeConfigureLog.yaml'
        if (-not (Test-Path -LiteralPath $configureLog -PathType Leaf)) {
            throw "CMake configure log was not found: $configureLog"
        }
        $architectureMatches = @(
            Select-String -LiteralPath $configureLog -Pattern '^\s*CMAKE_CUDA_ARCHITECTURES:\s*"([^"]+)"\s*$'
        )
        $architectureValues = @(
            $architectureMatches |
                ForEach-Object { $_.Matches[0].Groups[1].Value } |
                Sort-Object -Unique
        )
        if ($architectureValues.Count -ne 1 -or [string]::IsNullOrWhiteSpace($architectureValues[0])) {
            throw "Could not determine one effective upstream CMAKE_CUDA_ARCHITECTURES value from $configureLog"
        }
        $effectiveCudaArchitectures = $architectureValues[0]
        $cudaArchitectureSource = 'llama.cpp-upstream'
    }
}

$buildReceipt = [ordered]@{
    schema = 'siliang-build-config-v1'
    backend = $Backend
    build_shared_libs = $true
    ggml_native = $false
    ggml_backend_dl = $true
    ggml_cpu_all_variants = $true
    llama_build_ui = $true
    llama_use_prebuilt_ui = $true
    cuda_architectures = $effectiveCudaArchitectures
    cuda_architecture_source = $cudaArchitectureSource
}
$buildReceiptPath = Join-Path $resolvedBuildDirectory 'SILIANG-BUILD-CONFIG.json'
[IO.File]::WriteAllText(
    $buildReceiptPath,
    (($buildReceipt | ConvertTo-Json -Depth 4) + "`n"),
    [Text.UTF8Encoding]::new($false)
)
Write-Host ('  CUDA effective       : {0}' -f $effectiveCudaArchitectures)
Write-Host ('  CUDA source          : {0}' -f $cudaArchitectureSource)
Write-Host ('  build receipt        : {0}' -f $buildReceiptPath)

if (-not $ConfigureOnly) {
    Invoke-Native -Program 'cmake' -Arguments @(
        '--build', $resolvedBuildDirectory,
        '--config', 'Release',
        '--parallel', $Parallel.ToString([Globalization.CultureInfo]::InvariantCulture)
    )

    $uiHeaderPath = Join-Path $resolvedBuildDirectory 'tools\ui\ui.h'
    if (-not (Test-Path -LiteralPath $uiHeaderPath -PathType Leaf)) {
        throw "Embedded llama.cpp Web UI header was not generated: $uiHeaderPath"
    }
    $uiHeader = Get-Content -LiteralPath $uiHeaderPath -Raw
    if (-not $uiHeader.Contains('#define LLAMA_UI_HAS_ASSETS 1')) {
        throw 'Release build completed without embedded llama.cpp Web UI assets.'
    }
    Write-Host '  embedded Web UI      : verified'
}

Write-Host ('SILIANG_BUILD_DIRECTORY={0}' -f $resolvedBuildDirectory)
