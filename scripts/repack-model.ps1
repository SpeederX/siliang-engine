[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination,
    [ValidateRange(1, 2147483647)][int]$Samples = 64,
    [string]$PythonExecutable = 'python'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$converter = Join-Path $repositoryRoot 'tools\make_expert_major_gguf.py'
$preflight = Join-Path $repositoryRoot 'scripts\preflight_repack.py'
$expertMajorProbe = Join-Path $repositoryRoot 'scripts\check_expert_major.py'

if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Source GGUF does not exist: $Source"
}
$Source = (Resolve-Path -LiteralPath $Source).Path
$Destination = [IO.Path]::GetFullPath($Destination)
$destinationExtension = [IO.Path]::GetExtension($Destination)
$partialDestination = if ([string]::IsNullOrEmpty($destinationExtension)) {
    $Destination + '.partial'
} else {
    $Destination.Substring(0, $Destination.Length - $destinationExtension.Length) +
        '.partial' + $destinationExtension
}
$destinationParent = Split-Path -Parent $Destination
if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
    throw "Destination directory does not exist: $destinationParent"
}
if ($Source -ieq $Destination -or $Source -ieq $partialDestination) {
    throw 'Source and destination paths must be different.'
}
foreach ($path in @($Destination, $partialDestination)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing output: $path"
    }
}

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

function Invoke-NativeJson {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ('> {0} {1}' -f $Program, ($Arguments -join ' '))
    $outputLines = @(& $Program @Arguments)
    $exitCode = $LASTEXITCODE
    foreach ($line in $outputLines) {
        Write-Host ([string]$line)
    }
    if ($exitCode -ne 0) {
        throw "$Program failed with exit code $exitCode"
    }
    if ($outputLines.Count -eq 0) {
        throw "$Program completed without emitting the required JSON evidence"
    }

    $jsonText = [string]::Join([Environment]::NewLine, [string[]]$outputLines)
    try {
        return ($jsonText | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        throw "$Program emitted invalid JSON evidence: $($_.Exception.Message)"
    }
}

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue
    )

    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = $null
    try {
        $stream = [IO.File]::OpenRead($PathValue)
        $hash = $algorithm.ComputeHash($stream)
        return ([BitConverter]::ToString($hash)).Replace('-', '')
    } finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
        $algorithm.Dispose()
    }
}

function Get-ArtifactRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    $item = Get-Item -LiteralPath $Path
    $digest = Get-FileSha256 -PathValue $Path
    return [PSCustomObject]@{
        Path = $item.FullName
        Bytes = [Int64]$item.Length
        SHA256 = $digest.ToLowerInvariant()
    }
}

function Write-ArtifactRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)]$Record
    )

    Write-Host ('{0}:' -f $Label)
    Write-Host ('  path        : {0}' -f $Record.Path)
    Write-Host ('  bytes       : {0}' -f $Record.Bytes)
    Write-Host ('  sha256      : {0}' -f $Record.SHA256)
}

Write-Host 'Effective repack configuration:'
Write-Host ('  source      : {0}' -f $Source)
Write-Host ('  partial     : {0}' -f $partialDestination)
Write-Host ('  final       : {0}' -f $Destination)
Write-Host ('  samples     : {0}' -f $Samples)

$sourceBefore = Get-ArtifactRecord -Path $Source
Write-ArtifactRecord -Label 'Source before repack' -Record $sourceBefore

$preflightRecord = Invoke-NativeJson -Program $PythonExecutable -Arguments @($preflight, '--src', $Source)
$preflightStatusProperty = $preflightRecord.PSObject.Properties['status']
if ($null -eq $preflightStatusProperty -or
    -not ($preflightStatusProperty.Value -is [string]) -or
    $preflightStatusProperty.Value -cne 'preflight-ok') {
    throw 'Preflight JSON must contain status exactly equal to preflight-ok.'
}
$sourceArchitectureProperty = $preflightRecord.PSObject.Properties['sourceArchitecture']
if ($null -eq $sourceArchitectureProperty -or
    -not ($sourceArchitectureProperty.Value -is [string])) {
    throw 'Preflight JSON must contain sourceArchitecture as a string.'
}
$sourceArchitecture = [string]$sourceArchitectureProperty.Value
if ([string]::IsNullOrEmpty($sourceArchitecture)) {
    throw 'Preflight JSON sourceArchitecture must not be empty.'
}
if ($sourceArchitecture -cne $sourceArchitecture.Trim()) {
    throw 'Preflight JSON sourceArchitecture must not contain surrounding whitespace.'
}
Write-Host ('  source arch : {0}' -f $sourceArchitecture)

Invoke-Native -Program $PythonExecutable -Arguments @(
    $converter, '--src', $Source, '--dst', $partialDestination, '--dry-run'
)
if (Test-Path -LiteralPath $partialDestination) {
    throw "Dry-run unexpectedly created the partial output: $partialDestination"
}
Invoke-Native -Program $PythonExecutable -Arguments @(
    $converter, '--src', $Source, '--dst', $partialDestination
)
if (-not (Test-Path -LiteralPath $partialDestination -PathType Leaf)) {
    throw "Converter completed without creating the partial output: $partialDestination"
}
if (Test-Path -LiteralPath $Destination) {
    throw "Final output appeared while conversion was running: $Destination"
}
Invoke-Native -Program $PythonExecutable -Arguments @(
    $converter, '--src', $Source, '--dst', $partialDestination,
    '--verify', '--samples', $Samples.ToString([Globalization.CultureInfo]::InvariantCulture)
)
Invoke-Native -Program $PythonExecutable -Arguments @(
    $expertMajorProbe, '--model', $partialDestination
)

$sourceAfter = Get-ArtifactRecord -Path $Source
$partialRecord = Get-ArtifactRecord -Path $partialDestination
Write-ArtifactRecord -Label 'Source after repack' -Record $sourceAfter
Write-ArtifactRecord -Label 'Verified partial output' -Record $partialRecord

if ($sourceBefore.Bytes -ne $sourceAfter.Bytes -or
    $sourceBefore.SHA256 -cne $sourceAfter.SHA256) {
    throw (
        'Source GGUF changed during repack; refusing to finalize. ' +
        "Before=$($sourceBefore.SHA256)/$($sourceBefore.Bytes), " +
        "after=$($sourceAfter.SHA256)/$($sourceAfter.Bytes). " +
        "Verified partial retained at $partialDestination"
    )
}
if (Test-Path -LiteralPath $Destination) {
    throw "Refusing to overwrite final output that appeared during validation: $Destination"
}

$movedToFinal = $false
try {
    Move-Item -LiteralPath $partialDestination -Destination $Destination
    $movedToFinal = $true

    $finalRecord = Get-ArtifactRecord -Path $Destination
    Write-ArtifactRecord -Label 'Final output' -Record $finalRecord
    if ($partialRecord.Bytes -ne $finalRecord.Bytes -or
        $partialRecord.SHA256 -cne $finalRecord.SHA256) {
        throw 'Final output does not match the verified partial output.'
    }
} catch {
    if ($movedToFinal -and
        (Test-Path -LiteralPath $Destination -PathType Leaf) -and
        -not (Test-Path -LiteralPath $partialDestination)) {
        try {
            Move-Item -LiteralPath $Destination -Destination $partialDestination
            Write-Warning "Finalization failed; output restored to $partialDestination"
        } catch {
            Write-Warning (
                'Finalization failed and the output could not be restored to the partial path: ' +
                $_.Exception.Message
            )
        }
    }
    throw
}

$receipt = [ordered]@{
    status = 'repack-finalized'
    source = $Source
    sourceArchitecture = $sourceArchitecture
    sourceBytesBefore = $sourceBefore.Bytes
    sourceSHA256Before = $sourceBefore.SHA256
    sourceBytesAfter = $sourceAfter.Bytes
    sourceSHA256After = $sourceAfter.SHA256
    partial = $partialDestination
    partialBytes = $partialRecord.Bytes
    partialSHA256 = $partialRecord.SHA256
    destination = $Destination
    destinationBytes = $finalRecord.Bytes
    destinationSHA256 = $finalRecord.SHA256
    samples = $Samples
}
Write-Output ($receipt | ConvertTo-Json -Compress)
Write-Host ('Repack verified and finalized: {0}' -f $Destination)
