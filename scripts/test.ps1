[CmdletBinding()]
param(
    [string]$PythonExecutable = 'python',
    [string]$BuildDirectory,
    [string]$CTestRegex = '^(test-gguf|test-log|test-arg-parser)$'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

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

Write-Host 'Checking PowerShell syntax...'
$powerShellFiles = @(Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'scripts') -Filter '*.ps1' -File)
foreach ($file in $powerShellFiles) {
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $file.FullName,
        [ref]$tokens,
        [ref]$parseErrors
    )
    if (@($parseErrors).Count -ne 0) {
        $messages = @($parseErrors | ForEach-Object { $_.Message }) -join '; '
        throw "PowerShell parse failure in $($file.Name): $messages"
    }
}

Push-Location $repositoryRoot
try {
    Write-Host 'Checking Python syntax...'
    Invoke-Native -Program $PythonExecutable -Arguments @(
        '-m', 'py_compile',
        'tools/gguf_reader.py',
        'tools/make_expert_major_gguf.py',
        'scripts/check_expert_major.py',
        'scripts/preflight_repack.py',
        'tests/test_converter.py',
        'tests/test_hygiene.py'
    )

    Write-Host 'Checking converter CLI help...'
    Invoke-Native -Program $PythonExecutable -Arguments @(
        'tools/make_expert_major_gguf.py', '--help'
    )

    Write-Host 'Running Python unit tests...'
    Invoke-Native -Program $PythonExecutable -Arguments @(
        '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'
    )

    if (-not [string]::IsNullOrWhiteSpace($BuildDirectory)) {
        $resolvedBuildDirectory = if ([IO.Path]::IsPathRooted($BuildDirectory)) {
            [IO.Path]::GetFullPath($BuildDirectory)
        } else {
            [IO.Path]::GetFullPath((Join-Path $repositoryRoot $BuildDirectory))
        }
        if (-not (Test-Path -LiteralPath (Join-Path $resolvedBuildDirectory 'CTestTestfile.cmake') -PathType Leaf)) {
            throw "CTest metadata was not found in $resolvedBuildDirectory"
        }

        Write-Host ('Running selected model-free CTests: {0}' -f $CTestRegex)
        Invoke-Native -Program 'ctest' -Arguments @(
            '--test-dir', $resolvedBuildDirectory,
            '-C', 'Release',
            '--output-on-failure',
            '-R', $CTestRegex
        )
    }
} finally {
    Pop-Location
}

Write-Host 'All requested tests passed.'
