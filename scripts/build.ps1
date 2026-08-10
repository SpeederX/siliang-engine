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

if ($Backend -eq 'Cuda' -and [string]::IsNullOrWhiteSpace($CudaArchitecture)) {
    throw '-CudaArchitecture is required when -Backend Cuda is selected (for example: 75).'
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
    '-DLLAMA_CURL=OFF',
    '-DLLAMA_BUILD_TESTS=ON',
    '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_UI=OFF',
    '-DLLAMA_USE_PREBUILT_UI=OFF'
)

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $cmakeArguments += @('-G', $Generator)
}

if ($Backend -eq 'Cuda') {
    $cmakeArguments += @(
        '-DGGML_CUDA=ON',
        ('-DCMAKE_CUDA_ARCHITECTURES={0}' -f $CudaArchitecture)
    )
} else {
    $cmakeArguments += '-DGGML_CUDA=OFF'
}

Write-Host 'Effective build configuration:'
Write-Host ('  source              : {0}' -f $sourceDirectory)
Write-Host ('  build               : {0}' -f $resolvedBuildDirectory)
Write-Host ('  backend             : {0}' -f $Backend)
Write-Host ('  CUDA architecture   : {0}' -f $(if ($Backend -eq 'Cuda') { $CudaArchitecture } else { '<not applicable>' }))
Write-Host ('  generator           : {0}' -f $(if ($Generator) { $Generator } else { '<CMake default>' }))
Write-Host '  configuration       : Release'
Write-Host '  LLAMA_CURL           : OFF'
Write-Host '  LLAMA_BUILD_TESTS    : ON'
Write-Host '  LLAMA_BUILD_TOOLS    : ON'
Write-Host '  LLAMA_BUILD_SERVER   : ON'
Write-Host '  LLAMA_BUILD_UI       : OFF'

Invoke-Native -Program 'cmake' -Arguments $cmakeArguments

if (-not $ConfigureOnly) {
    Invoke-Native -Program 'cmake' -Arguments @(
        '--build', $resolvedBuildDirectory,
        '--config', 'Release',
        '--parallel', $Parallel.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}

Write-Host ('SILIANG_BUILD_DIRECTORY={0}' -f $resolvedBuildDirectory)
