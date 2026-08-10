[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReferenceServer,
    [Parameter(Mandatory = $true)][string]$CandidateServer,
    [Parameter(Mandatory = $true)][string]$DeepSeekModel,
    [Parameter(Mandatory = $true)][string]$GptOssModel,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1.0, 1048576.0)]
    [double]$GptOssMaxDeviceModelBufferMiB,

    [string]$PythonExecutable = 'python',
    [string]$ResultsRoot,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1048576)][int]$ArenaMiB,
    [ValidateRange(1024, 65535)][int]$DeepSeekPort = 8140,
    [ValidateRange(1024, 65535)][int]$GptOssPort = 8141,
    [string[]]$DeepSeekServerArguments = @(
        '-ngl', '99', '-ncmoe', '43', '-nkvo', '--no-op-offload',
        '-c', '2048', '-b', '512', '-ub', '512', '-t', '12', '-tb', '12'
    ),
    [string[]]$GptOssServerArguments = @(
        '-ngl', '99', '-ncmoe', '33', '-nkvo', '--no-op-offload',
        '-c', '4096', '-b', '512', '-ub', '512', '-t', '12', '-tb', '12',
        '-lv', '4'
    ),
    [ValidateRange(1, 4096)][int]$NPredict = 128,
    [ValidateRange(1, 4096)][int]$WarmupTokens = 48,
    [ValidateRange(30, 86400)][int]$ReadyTimeoutSeconds = 900,
    [ValidateRange(30, 86400)][int]$RequestTimeoutSeconds = 7200,
    [ValidateRange(1, 1048576)][int]$MinimumFreeMiB = 8192,
    [ValidateRange(0, 100)][int]$MaxCpuLoadPercent = 20,
    [ValidateRange(0, 100)][int]$MaxGpuUtilizationPercent = 5,
    [ValidateRange(0, 100)][int]$MaxDiskUtilizationPercent = 15,
    [ValidateRange(0, 1048576)][int]$MaxFreeMemoryDriftMiB = 512,
    [switch]$AllowMemoryPressure,
    [ValidateRange(0, 600)][int]$SettleSeconds = 15
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$expertMajorProbe = Join-Path $repositoryRoot 'scripts\check_expert_major.py'
$fixedPrompt = 'The memory hierarchy of a modern workstation spans five orders of magnitude in latency. Registers answer in a fraction of a nanosecond while'
$performanceRegressionLimitPercent = 5.0
$managedEnvironmentPattern = '^(SILIANGEM_|GGML_MOE_)'

function Resolve-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) {
        throw "Required file does not exist: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Assert-NoModelOverrideArguments {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $ownedModelOptions = @(
        '-m', '--model',
        '-mu', '--model-url',
        '-hf', '-hfr', '--hf-repo',
        '-hff', '--hf-file'
    )
    foreach ($argument in $Arguments) {
        $normalizedArgument = ([string]$argument).ToLowerInvariant()
        foreach ($option in $ownedModelOptions) {
            if ($normalizedArgument -ceq $option -or $normalizedArgument.StartsWith($option + '=')) {
                throw "$Label must not contain primary-model option $argument; the runtime gate owns the model path."
            }
        }
    }
}

function Assert-DistinctServerPaths {
    param(
        [Parameter(Mandatory = $true)][string]$ReferencePath,
        [Parameter(Mandatory = $true)][string]$CandidatePath
    )

    if ([string]::Equals($ReferencePath, $CandidatePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw '-ReferenceServer and -CandidateServer resolve to the same path.'
    }
}

$ReferenceServer = Resolve-RequiredFile $ReferenceServer
$CandidateServer = Resolve-RequiredFile $CandidateServer
$DeepSeekModel = Resolve-RequiredFile $DeepSeekModel
$GptOssModel = Resolve-RequiredFile $GptOssModel
Assert-DistinctServerPaths -ReferencePath $ReferenceServer -CandidatePath $CandidateServer
Assert-NoModelOverrideArguments -Label '-DeepSeekServerArguments' -Arguments $DeepSeekServerArguments
Assert-NoModelOverrideArguments -Label '-GptOssServerArguments' -Arguments $GptOssServerArguments
if (-not (Test-Path -LiteralPath $expertMajorProbe -PathType Leaf)) {
    throw "Expert-major metadata probe was not found: $expertMajorProbe"
}
if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
    throw "Python executable was not found: $PythonExecutable"
}

if ($DeepSeekPort -eq $GptOssPort) {
    throw '-DeepSeekPort and -GptOssPort must be different.'
}
if (-not (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi is required for the idle-machine and gpt-oss VRAM checks.'
}

if ([string]::IsNullOrWhiteSpace($ResultsRoot)) {
    $ResultsRoot = Join-Path $repositoryRoot 'artifacts\runtime'
} elseif (-not [IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRoot = Join-Path $repositoryRoot $ResultsRoot
}
$ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot)
[void](New-Item -ItemType Directory -Force -Path $ResultsRoot)
$runLeaf = 'gate-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID
$runDirectory = Join-Path $ResultsRoot $runLeaf
if (Test-Path -LiteralPath $runDirectory) {
    throw "Refusing to overwrite an existing result directory: $runDirectory"
}
[void](New-Item -ItemType Directory -Path $runDirectory)

$script:results = New-Object 'System.Collections.Generic.List[object]'
$script:resultsPath = Join-Path $runDirectory 'results.json'

function Save-Results {
    $payload = [ordered]@{
        generatedUtc = [DateTime]::UtcNow.ToString('o')
        complete = $false
        cells = @($script:results.ToArray())
    }
    $payload | ConvertTo-Json -Depth 12 | Out-File -LiteralPath $script:resultsPath -Encoding utf8
}

function Get-ManagedEnvironment {
    $values = [ordered]@{}
    foreach ($item in @(Get-ChildItem Env: | Where-Object { $_.Name -match $managedEnvironmentPattern } | Sort-Object Name)) {
        $values[$item.Name] = $item.Value
    }
    return $values
}

function Clear-ManagedEnvironment {
    foreach ($item in @(Get-ChildItem Env: | Where-Object { $_.Name -match $managedEnvironmentPattern })) {
        Remove-Item -LiteralPath ('Env:{0}' -f $item.Name) -ErrorAction SilentlyContinue
    }
}

function Set-CellEnvironment {
    param([Parameter(Mandatory = $true)][ValidateSet('Disabled', 'Arena')][string]$ArenaState)

    Clear-ManagedEnvironment
    $env:SILIANGEM_VERBOSE = '1'
    $env:GGML_MOE_PREFETCH = '0'
    if ($ArenaState -eq 'Disabled') {
        $env:SILIANGEM_DISABLE = '1'
    } else {
        $env:SILIANGEM_CACHE_MIB = $ArenaMiB.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SILIANGEM_DEFER = '1'
    }
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * (2 * $backslashes)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Assert-PortAvailable {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if (@($listeners | Where-Object { $_.Port -eq $Port }).Count -ne 0) {
        throw "TCP port $Port is already in use. No existing process will be stopped automatically."
    }
}

function Get-GpuSnapshot {
    $lines = @(& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) {
        throw 'nvidia-smi did not return GPU utilization and memory data.'
    }
    $maximumUtilization = 0.0
    $usedMiB = 0.0
    foreach ($line in $lines) {
        $parts = @($line -split ',')
        if ($parts.Count -ne 2) {
            throw "Unexpected nvidia-smi output: $line"
        }
        $utilization = 0.0
        $memory = 0.0
        if (-not [double]::TryParse($parts[0].Trim(), [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture, [ref]$utilization) -or
            -not [double]::TryParse($parts[1].Trim(), [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture, [ref]$memory)) {
            throw "Could not parse nvidia-smi output: $line"
        }
        $maximumUtilization = [Math]::Max($maximumUtilization, $utilization)
        $usedMiB += $memory
    }
    return [pscustomobject]@{
        gpuCount = $lines.Count
        maxUtilizationPercent = $maximumUtilization
        usedMiB = $usedMiB
    }
}

function Get-VramSnapshot {
    $gpu = Get-GpuSnapshot
    try {
        $adapterCounters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory)
    } catch {
        throw "Could not read GPU adapter memory counters: $($_.Exception.Message)"
    }
    if ($adapterCounters.Count -eq 0) {
        throw 'GPU adapter memory counters returned no instances; shared VRAM cannot be recorded.'
    }
    $sharedBytes = ($adapterCounters | Measure-Object -Property SharedUsage -Sum).Sum
    $dedicatedBytes = ($adapterCounters | Measure-Object -Property DedicatedUsage -Sum).Sum
    return [pscustomobject]@{
        nvidiaUsedMiB = [double]$gpu.usedMiB
        adapterDedicatedMiB = [Math]::Round(([double]$dedicatedBytes / 1MB), 1)
        adapterSharedMiB = [Math]::Round(([double]$sharedBytes / 1MB), 1)
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

function Get-ArtifactIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Write-Host ("Hashing {0}: {1}" -f $Label, $PathValue)
    $item = Get-Item -LiteralPath $PathValue
    $sha256 = Get-FileSha256 -PathValue $PathValue
    return [pscustomobject]@{
        path = $item.FullName
        bytes = [int64]$item.Length
        sha256 = $sha256
        lastWriteUtc = $item.LastWriteTimeUtc.ToString('o')
    }
}

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        $hash = $algorithm.ComputeHash($bytes)
        return ([BitConverter]::ToString($hash)).Replace('-', '')
    } finally {
        $algorithm.Dispose()
    }
}

function Get-RuntimeIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$ServerPath,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $serverIdentity = Get-ArtifactIdentity -PathValue $ServerPath -Label ("{0} executable" -f $Label)
    $serverDirectory = Split-Path -Parent $ServerPath
    $dllIdentities = @()
    foreach ($dll in @(Get-ChildItem -LiteralPath $serverDirectory -Filter '*.dll' -File | Sort-Object Name)) {
        $dllIdentities += Get-ArtifactIdentity -PathValue $dll.FullName `
            -Label ("{0} adjacent DLL {1}" -f $Label, $dll.Name)
    }

    # Paths and timestamps are evidence, but are deliberately excluded from the
    # content identity so copied reference/candidate runtimes still compare equal.
    $identityRecords = @(
        'exe|{0}|{1}' -f $serverIdentity.bytes, $serverIdentity.sha256
    )
    foreach ($dllIdentity in $dllIdentities) {
        $identityRecords += 'dll|{0}|{1}|{2}' -f `
            ([IO.Path]::GetFileName([string]$dllIdentity.path)).ToLowerInvariant(), `
            $dllIdentity.bytes, $dllIdentity.sha256
    }

    return [pscustomobject]@{
        schema = 'server-executable-and-adjacent-dlls-v1'
        directory = $serverDirectory
        server = $serverIdentity
        adjacentDlls = $dllIdentities
        adjacentDllCount = $dllIdentities.Count
        fullRuntimeSha256 = Get-StringSha256 -Value ($identityRecords -join "`n")
    }
}

function Assert-DistinctRuntimeIdentities {
    param(
        [Parameter(Mandatory = $true)]$ReferenceIdentity,
        [Parameter(Mandatory = $true)]$CandidateIdentity
    )

    if ([string]$ReferenceIdentity.fullRuntimeSha256 -ceq [string]$CandidateIdentity.fullRuntimeSha256) {
        throw 'Reference and candidate have identical executable-plus-adjacent-DLL runtime identities.'
    }
}

function Get-ExpertMajorMetadataEvidence {
    param([Parameter(Mandatory = $true)][string]$ModelPath)

    Write-Host ("Checking expert-major metadata: {0}" -f $ModelPath)
    $output = @(& $PythonExecutable $expertMajorProbe --model $ModelPath 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ("Expert-major metadata probe failed with exit code {0}: {1}" -f
            $exitCode, ($output -join "`n"))
    }
    try {
        $evidence = ($output -join "`n") | ConvertFrom-Json
    } catch {
        throw "Expert-major metadata probe returned invalid JSON: $($_.Exception.Message)"
    }
    if ([string]$evidence.status -cne 'expert-major-metadata-ok') {
        throw "Expert-major metadata probe returned an unexpected status for $ModelPath"
    }
    return $evidence
}

function Get-StorageIdentity {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    $root = [IO.Path]::GetPathRoot($PathValue)
    if ($root -notmatch '^(?<letter>[A-Za-z]):\\$') {
        throw "A local drive-letter model path is required to record physical storage identity: $PathValue"
    }
    $driveLetter = $Matches['letter']
    try {
        $volume = Get-Volume -DriveLetter $driveLetter -ErrorAction Stop
        $partition = Get-Partition -DriveLetter $driveLetter -ErrorAction Stop
        $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
    } catch {
        throw "Could not resolve storage identity for ${driveLetter}: $($_.Exception.Message)"
    }
    # Get-Disk does not expose MediaType on every Windows/Storage provider.
    # Get-PhysicalDisk does, and DeviceId normally matches the Get-Disk number.
    # Keep the already-recorded serial/unique ID authoritative if that optional
    # enrichment is unavailable.
    $physicalDisk = $null
    try {
        $physicalDisk = Get-PhysicalDisk -ErrorAction Stop |
            Where-Object { [string]$_.DeviceId -ceq [string]$disk.Number } |
            Select-Object -First 1
    } catch {
        $physicalDisk = $null
    }
    $mediaType = if ($null -ne $physicalDisk -and
        $physicalDisk.PSObject.Properties.Name -contains 'MediaType') {
        [string]$physicalDisk.MediaType
    } else {
        'Unknown'
    }
    return [pscustomobject]@{
        drive = ('{0}:' -f $driveLetter.ToUpperInvariant())
        volumeUniqueId = [string]$volume.UniqueId
        fileSystem = [string]$volume.FileSystem
        fileSystemLabel = [string]$volume.FileSystemLabel
        diskNumber = [int]$disk.Number
        diskFriendlyName = [string]$disk.FriendlyName
        diskSerialNumber = ([string]$disk.SerialNumber).Trim()
        diskUniqueId = [string]$disk.UniqueId
        busType = [string]$disk.BusType
        mediaType = $mediaType
        diskBytes = [int64]$disk.Size
    }
}

function Get-MachineSample {
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $processor = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor |
        Where-Object { $_.Name -eq '_Total' } |
        Select-Object -First 1
    $disk = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk |
        Where-Object { $_.Name -eq '_Total' } |
        Select-Object -First 1
    if ($null -eq $processor -or $null -eq $disk) {
        throw 'Could not read language-independent CPU/disk performance counters.'
    }
    $gpu = Get-GpuSnapshot
    return [pscustomobject]@{
        utc = [DateTime]::UtcNow.ToString('o')
        freeMiB = [Math]::Round(([double]$operatingSystem.FreePhysicalMemory / 1024.0), 0)
        cpuPercent = [double]$processor.PercentProcessorTime
        diskPercent = [double]$disk.PercentDiskTime
        gpuPercent = [double]$gpu.maxUtilizationPercent
        gpuUsedMiB = [double]$gpu.usedMiB
    }
}

function Assert-MachineIdle {
    param([Parameter(Mandatory = $true)][string]$CellName)

    $samples = @()
    foreach ($sampleNumber in 1..3) {
        if ($sampleNumber -gt 1) {
            Start-Sleep -Seconds 3
        }
        $sample = Get-MachineSample
        $samples += $sample
        Write-Host ('  idle sample {0}/3: free {1} MiB, CPU {2:N1}%, GPU {3:N1}%, disk {4:N1}%' -f
            $sampleNumber, $sample.freeMiB, $sample.cpuPercent, $sample.gpuPercent, $sample.diskPercent)
    }

    $minimumFree = ($samples.freeMiB | Measure-Object -Minimum).Minimum
    $maximumFree = ($samples.freeMiB | Measure-Object -Maximum).Maximum
    $averageCpu = ($samples.cpuPercent | Measure-Object -Average).Average
    $averageGpu = ($samples.gpuPercent | Measure-Object -Average).Average
    $averageDisk = ($samples.diskPercent | Measure-Object -Average).Average
    if ($minimumFree -lt $MinimumFreeMiB) {
        throw "$CellName is void: free RAM $minimumFree MiB is below required $MinimumFreeMiB MiB."
    }
    if (($maximumFree - $minimumFree) -gt $MaxFreeMemoryDriftMiB) {
        throw "$CellName is void: free RAM drift $($maximumFree - $minimumFree) MiB exceeds $MaxFreeMemoryDriftMiB MiB."
    }
    if ($averageCpu -gt $MaxCpuLoadPercent) {
        throw "$CellName is void: average CPU load $averageCpu% exceeds $MaxCpuLoadPercent%."
    }
    if ($averageGpu -gt $MaxGpuUtilizationPercent) {
        throw "$CellName is void: average GPU load $averageGpu% exceeds $MaxGpuUtilizationPercent%."
    }
    if ($averageDisk -gt $MaxDiskUtilizationPercent) {
        throw "$CellName is void: average disk load $averageDisk% exceeds $MaxDiskUtilizationPercent%."
    }
    return $samples
}

function Wait-ServerReady {
    param(
        [Parameter(Mandatory = $true)][Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][int]$Port
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            return $false
        }
        try {
            $health = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/health" -f $Port) `
                -Method Get -TimeoutSec 3 -ErrorAction Stop
            if ($health.status -eq 'ok') {
                return $true
            }
        } catch {
            # Loading can legitimately take many minutes for out-of-core models.
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Read-CellLog {
    param([string]$StandardOutputPath, [string]$StandardErrorPath)

    $chunks = @()
    foreach ($path in @($StandardOutputPath, $StandardErrorPath)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $chunks += (Get-Content -LiteralPath $path -Raw -ErrorAction SilentlyContinue)
        }
    }
    return ($chunks -join "`n")
}

function Assert-ExpertMajorEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$LogText,
        [Parameter(Mandatory = $true)][object]$MetadataEvidence
    )

    if ([string]$MetadataEvidence.status -cne 'expert-major-metadata-ok') {
        throw 'Cell is void: the model does not have validated expert-major metadata.'
    }
    return [pscustomobject]@{
        metadata = $MetadataEvidence
        loaderInfoMarkerPresent = [bool]($LogText -match '(?im)siliangem expert-major')
    }
}

function Assert-ArenaEvidence {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Disabled', 'Arena')][string]$ArenaState,
        [Parameter(Mandatory = $true)][string]$LogText
    )

    $armed = $LogText -match '(?im)siliangem:.*cache\s+\d+\s+slots'
    $directIo = $LogText -match '(?im)unbuffered\+overlapped'
    $fallback = $LogText -match '(?im)siliangem:.*using mmap'
    $explicitlyDisabled = $LogText -match '(?im)siliangem: disabled by SILIANGEM_DISABLE - using mmap'
    if ($ArenaState -eq 'Arena') {
        if (-not $armed) {
            throw 'Cell is void: the requested arena did not report a nonzero slot count.'
        }
        if (-not $directIo) {
            throw 'Cell is void: the log does not prove that unbuffered+overlapped I/O armed.'
        }
        if ($fallback) {
            throw 'Cell is void: the arena log contains a fallback-to-mmap message.'
        }
        $failurePattern = '(?im)^.*siliangem.*(?:read failed|ISSUE failed|cannot open|cannot read|bad slab|not sector aligned|zero expert stride|source declined|alloc failed|metadata is missing).*$'
        $failureMatch = [regex]::Match($LogText, $failurePattern)
        if ($failureMatch.Success) {
            throw "Cell is void: arena read/geometry/memory failure: $($failureMatch.Value.Trim())"
        }

        $workMatches = @([regex]::Matches(
            $LogText,
            '(?im)siliangem\[[^\]]+\]:\s+(?<lookups>\d+) lookups,\s+(?<hits>\d+) hits \([^\)]*\),\s+(?<misses>\d+) misses,.*?(?<gib>[0-9]+(?:\.[0-9]+)?) GiB read from slab'
        ))
        $submitMatches = @([regex]::Matches(
            $LogText,
            '(?im)siliangem\[submit\]:\s+(?<sync>\d+) sync \([^\)]*\),\s+(?<pending>\d+) pending'
        ))
        $memoryMatches = @([regex]::Matches(
            $LogText,
            '(?im)siliangem\[mem\]:\s+available (?<available>\d+) MiB \| file cache (?<cache>\d+) MiB.*?commit charged (?<commit>\d+) MiB(?<tail>[^\r\n]*)'
        ))
        $pressureMatches = @([regex]::Matches(
            $LogText,
            '(?im)^.*siliangem\[mem\].*PRESSURE.*$'
        ))
        if ($workMatches.Count -eq 0 -or $submitMatches.Count -eq 0 -or $memoryMatches.Count -eq 0) {
            throw 'Cell is void: periodic served-work, submission, or siliangem[mem] evidence is absent.'
        }

        $work = $workMatches[$workMatches.Count - 1]
        $submit = $submitMatches[$submitMatches.Count - 1]
        $memory = $memoryMatches[$memoryMatches.Count - 1]
        $lookups = [int64]$work.Groups['lookups'].Value
        $hits = [int64]$work.Groups['hits'].Value
        $misses = [int64]$work.Groups['misses'].Value
        $readGiB = [double]::Parse($work.Groups['gib'].Value, [Globalization.CultureInfo]::InvariantCulture)
        $sync = [int64]$submit.Groups['sync'].Value
        $pending = [int64]$submit.Groups['pending'].Value
        $availableMiB = [int64]$memory.Groups['available'].Value
        $fileCacheMiB = [int64]$memory.Groups['cache'].Value
        $commitMiB = [int64]$memory.Groups['commit'].Value
        $pressureObserved = $pressureMatches.Count -gt 0

        if ($lookups -le 0 -or $hits -le 0 -or $misses -le 0 -or $readGiB -le 0) {
            throw "Cell is void: served-work counters are not all nonzero (lookups=$lookups hits=$hits misses=$misses readGiB=$readGiB)."
        }
        if ($lookups -ne ($hits + $misses)) {
            throw "Cell is void: lookups $lookups do not equal hits+misses $($hits + $misses)."
        }
        if (($sync + $pending) -ne $misses) {
            throw "Cell is void: successful read submissions $($sync + $pending) do not equal misses $misses."
        }
        if (-not $AllowMemoryPressure -and (
            $availableMiB -lt 400 -or $fileCacheMiB -lt 64 -or $pressureObserved
        )) {
            throw "Cell is void: siliangem[mem] reports memory pressure (available=$availableMiB MiB, file-cache=$fileCacheMiB MiB)."
        }
        return [pscustomobject]@{
            armed = $true
            lookups = $lookups
            hits = $hits
            misses = $misses
            completedReadGiB = $readGiB
            synchronousSubmissions = $sync
            pendingSubmissions = $pending
            availableMiB = $availableMiB
            fileCacheMiB = $fileCacheMiB
            commitChargedMiB = $commitMiB
            memoryPressureObserved = $pressureObserved
            memoryPressureReportCount = $pressureMatches.Count
            memoryPressureGateEnforced = -not [bool]$AllowMemoryPressure
        }
    } else {
        if (-not $explicitlyDisabled) {
            throw 'Cell is void: the log does not prove SILIANGEM_DISABLE selected mmap.'
        }
        if ($armed) {
            throw 'Cell is void: the arena armed in an arena-disabled control.'
        }
        return [pscustomobject]@{
            armed = $false
            explicitlyDisabled = $true
        }
    }
}

function Get-DeviceModelBufferEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$LogText,
        [Parameter(Mandatory = $true)][double]$MaximumMiB
    )

    $matches = @([regex]::Matches(
        $LogText,
        '(?im)(?<backend>CUDA[^\s:]*)\s+model buffer size\s*=\s*(?<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB'
    ))
    if ($matches.Count -eq 0) {
        throw 'gpt-oss allocation check is inconclusive: no CUDA model-buffer allocation evidence was logged.'
    }
    $allocations = @()
    $totalMiB = 0.0
    foreach ($match in $matches) {
        $mib = [double]::Parse($match.Groups['mib'].Value, [Globalization.CultureInfo]::InvariantCulture)
        $totalMiB += $mib
        $allocations += [pscustomobject]@{
            backend = $match.Groups['backend'].Value
            modelBufferMiB = $mib
        }
    }
    if ($totalMiB -gt $MaximumMiB) {
        throw ("gpt-oss allocation check failed: CUDA model buffers total {0:N1} MiB, above the predeclared {1:N1} MiB ceiling." -f
            $totalMiB, $MaximumMiB)
    }
    $offloadLineMatches = @([regex]::Matches(
        $LogText,
        '(?im)^.*load_tensors:\s+offload(?:ing|ed)\b[^\r\n]*$'
    ))
    $offloadCompletionMatches = @([regex]::Matches(
        $LogText,
        '(?im)load_tensors:\s+offloaded\s+(?<offloaded>\d+)\/(?<total>\d+)\s+layers\s+to\s+GPU'
    ))
    if ($offloadLineMatches.Count -eq 0 -or $offloadCompletionMatches.Count -eq 0) {
        throw 'gpt-oss allocation check is inconclusive: complete layer-offload log evidence is absent.'
    }
    $offloadCompletion = $offloadCompletionMatches[$offloadCompletionMatches.Count - 1]
    $offloadedLayers = [int]$offloadCompletion.Groups['offloaded'].Value
    $totalLayers = [int]$offloadCompletion.Groups['total'].Value
    if ($offloadedLayers -le 0 -or $totalLayers -le 0 -or $offloadedLayers -gt $totalLayers) {
        throw "gpt-oss allocation check logged invalid layer-offload counts: $offloadedLayers/$totalLayers."
    }
    $offloadLines = @($offloadLineMatches | ForEach-Object { $_.Value.Trim() })
    return [pscustomobject]@{
        allocations = $allocations
        totalMiB = $totalMiB
        maximumMiB = $MaximumMiB
        offload = [pscustomobject]@{
            lines = $offloadLines
            offloadedLayers = $offloadedLayers
            totalLayers = $totalLayers
        }
        status = 'observed-allocation-and-offload-within-predeclared-ceiling'
    }
}

function Invoke-Completion {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][int]$Tokens,
        [switch]$MeasureVram
    )

    $body = [ordered]@{
        prompt = $fixedPrompt
        n_predict = $Tokens
        temperature = 0
        top_k = 1
        seed = 42
        cache_prompt = $false
        ignore_eos = $true
    }
    $uri = "http://127.0.0.1:{0}/completion" -f $Port
    $bodyJson = $body | ConvertTo-Json
    $peakDedicatedMiB = $null
    $peakSharedMiB = $null

    if ($MeasureVram) {
        $initialVram = Get-VramSnapshot
        $peakDedicatedMiB = [double]$initialVram.nvidiaUsedMiB
        $peakSharedMiB = [double]$initialVram.adapterSharedMiB
        $job = Start-Job -ScriptBlock {
            param($RequestUri, $RequestBody, $TimeoutSeconds)
            $ErrorActionPreference = 'Stop'
            Invoke-RestMethod -Uri $RequestUri -Method Post -TimeoutSec $TimeoutSeconds `
                -ContentType 'application/json' -Body $RequestBody
        } -ArgumentList $uri, $bodyJson, $RequestTimeoutSeconds
        try {
            while ($job.State -in @('NotStarted', 'Running')) {
                $snapshot = Get-VramSnapshot
                $peakDedicatedMiB = [Math]::Max($peakDedicatedMiB, [double]$snapshot.nvidiaUsedMiB)
                $peakSharedMiB = [Math]::Max($peakSharedMiB, [double]$snapshot.adapterSharedMiB)
                [void](Wait-Job -Job $job -Timeout 1)
            }
            if ($job.State -ne 'Completed') {
                $reason = if ($null -ne $job.ChildJobs[0].JobStateInfo.Reason) {
                    $job.ChildJobs[0].JobStateInfo.Reason.Message
                } else {
                    "job state $($job.State)"
                }
                throw "Completion request failed: $reason"
            }
            $response = Receive-Job -Job $job -ErrorAction Stop
            $finalVram = Get-VramSnapshot
            $peakDedicatedMiB = [Math]::Max($peakDedicatedMiB, [double]$finalVram.nvidiaUsedMiB)
            $peakSharedMiB = [Math]::Max($peakSharedMiB, [double]$finalVram.adapterSharedMiB)
        } finally {
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
    } else {
        $response = Invoke-RestMethod -Uri $uri -Method Post `
            -TimeoutSec $RequestTimeoutSeconds -ContentType 'application/json' -Body $bodyJson
    }

    $content = [string]$response.content
    if ([string]::IsNullOrEmpty($content)) {
        throw 'Completion returned empty output.'
    }
    $predictedCount = if ($response.PSObject.Properties.Name -contains 'tokens_predicted') {
        [int]$response.tokens_predicted
    } else {
        [int]$response.timings.predicted_n
    }
    if ($predictedCount -ne $Tokens) {
        throw "Completion returned $predictedCount tokens; fixed request required $Tokens."
    }
    $tokensPerSecond = [double]$response.timings.predicted_per_second
    if ($tokensPerSecond -le 0) {
        throw "Completion returned invalid decode throughput: $tokensPerSecond"
    }
    return [pscustomobject]@{
        content = $content
        predictedTokens = $predictedCount
        decodeTokensPerSecond = $tokensPerSecond
        promptTokensPerSecond = [double]$response.timings.prompt_per_second
        peakDedicatedVramMiB = $peakDedicatedMiB
        peakSharedVramMiB = $peakSharedMiB
    }
}

function Stop-CellProcess {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)

    $Process.Refresh()
    $processWasRunningBeforeTermination = -not $Process.HasExited
    $stopProcessForceRequested = $false
    if ($processWasRunningBeforeTermination) {
        $stopProcessForceRequested = $true
        Stop-Process -Id $Process.Id -Force -ErrorAction Stop
        if (-not $Process.WaitForExit(30000)) {
            throw "Server process $($Process.Id) did not exit within 30 seconds."
        }
        $Process.Refresh()
    }
    if (-not $Process.HasExited) {
        throw "Server process $($Process.Id) did not exit after termination."
    }
    $exitCode = $null
    try {
        $exitCode = [int]$Process.ExitCode
    } catch {
        # ExitCode can be unavailable for a process that ended before its handle opened.
    }
    return [pscustomobject]@{
        exitCode = $exitCode
        processExited = [bool]$Process.HasExited
        processWasRunningBeforeTermination = [bool]$processWasRunningBeforeTermination
        stopProcessForceRequested = [bool]$stopProcessForceRequested
        termination = if ($stopProcessForceRequested) { 'forced-after-cell' } else { 'server-exited' }
    }
}

function Assert-ExpectedCellTermination {
    param([Parameter(Mandatory = $true)]$Termination)

    $requiredProperties = @(
        'processWasRunningBeforeTermination',
        'stopProcessForceRequested',
        'processExited',
        'termination',
        'exitCode'
    )
    foreach ($propertyName in $requiredProperties) {
        if ($Termination.PSObject.Properties.Name -cnotcontains $propertyName) {
            throw "Cell cannot pass: termination evidence is missing '$propertyName'."
        }
    }
    if (-not [bool]$Termination.processWasRunningBeforeTermination) {
        throw 'Cell cannot pass: the server exited before the harness requested end-of-cell termination.'
    }
    if (-not [bool]$Termination.stopProcessForceRequested -or
        [string]$Termination.termination -cne 'forced-after-cell') {
        throw 'Cell cannot pass: harness-controlled Stop-Process -Force evidence is missing.'
    }
    if (-not [bool]$Termination.processExited) {
        throw 'Cell cannot pass: the server process did not terminate.'
    }
    if ($null -eq $Termination.exitCode) {
        throw 'Cell cannot pass: the terminated server exit code is unavailable.'
    }
    return [pscustomobject]@{
        status = 'expected-harness-forced-termination'
        processWasRunningBeforeTermination = [bool]$Termination.processWasRunningBeforeTermination
        termination = [string]$Termination.termination
        stopProcessForceRequested = [bool]$Termination.stopProcessForceRequested
        processExited = [bool]$Termination.processExited
        exitCode = [int]$Termination.exitCode
    }
}

function Invoke-Cell {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][ValidateSet('Reference', 'Candidate')][string]$Build,
        [Parameter(Mandatory = $true)][string]$ServerPath,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][string[]]$ServerArguments,
        [Parameter(Mandatory = $true)][ValidateSet('Disabled', 'Arena')][string]$ArenaState,
        [ValidateRange(0.0, 1048576.0)][double]$MaximumDeviceModelBufferMiB = 0.0,
        [switch]$Warmup,
        [switch]$RequireIdle
    )

    Write-Host ''
    Write-Host ("=== {0} ===" -f $Name)
    $idleSamples = @()
    if ($RequireIdle) {
        $idleSamples = @(Assert-MachineIdle -CellName $Name)
    }
    Assert-PortAvailable -Port $Port
    Set-CellEnvironment -ArenaState $ArenaState
    $environment = Get-ManagedEnvironment

    $allArguments = @('-m', $ModelPath) + $ServerArguments + @(
        '--host', '127.0.0.1', '--port', $Port.ToString([Globalization.CultureInfo]::InvariantCulture), '--no-webui'
    )
    $commandLineArguments = @($allArguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument ([string]$_) })
    $stdoutPath = Join-Path $runDirectory ($Name + '.stdout.log')
    $stderrPath = Join-Path $runDirectory ($Name + '.stderr.log')

    Write-Host ('  executable : {0}' -f $ServerPath)
    Write-Host ('  model      : {0}' -f $ModelPath)
    Write-Host ('  flags      : {0}' -f ($commandLineArguments -join ' '))
    Write-Host '  environment:'
    foreach ($key in $environment.Keys) {
        Write-Host ('    {0}={1}' -f $key, $environment[$key])
    }

    $process = $null
    $termination = $null
    $record = [ordered]@{
        name = $Name
        build = $Build
        arenaState = $ArenaState
        server = $ServerPath
        model = $ModelPath
        serverIdentity = $script:artifactIdentityByPath[$ServerPath]
        runtimeIdentity = $script:runtimeIdentityByPath[$ServerPath]
        modelIdentity = $script:artifactIdentityByPath[$ModelPath]
        modelStorage = $script:storageIdentityByPath[$ModelPath]
        arguments = $allArguments
        environment = $environment
        idleSamples = $idleSamples
        stdout = $stdoutPath
        stderr = $stderrPath
        startedUtc = [DateTime]::UtcNow.ToString('o')
        success = $false
    }

    try {
        $gpuBefore = Get-GpuSnapshot
        $process = Start-Process -FilePath $ServerPath -ArgumentList $commandLineArguments `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        $record.processId = $process.Id

        if (-not (Wait-ServerReady -Process $process -Port $Port)) {
            $tail = (Read-CellLog -StandardOutputPath $stdoutPath -StandardErrorPath $stderrPath)
            if ($tail.Length -gt 800) {
                $tail = $tail.Substring($tail.Length - 800)
            }
            throw "Server did not become ready. Log tail: $tail"
        }

        Start-Sleep -Seconds 1
        $loadLog = Read-CellLog -StandardOutputPath $stdoutPath -StandardErrorPath $stderrPath
        $expertMajorEvidence = Assert-ExpertMajorEvidence -LogText $loadLog `
            -MetadataEvidence $script:expertMajorMetadataByPath[$ModelPath]
        $record.expertMajorEvidence = $expertMajorEvidence

        $warmupResult = $null
        if ($Warmup) {
            $warmupResult = Invoke-Completion -Port $Port -Tokens $WarmupTokens
            Start-Sleep -Seconds 1
            $warmupLog = Read-CellLog -StandardOutputPath $stdoutPath -StandardErrorPath $stderrPath
            $warmupEvidence = Assert-ArenaEvidence -ArenaState $ArenaState -LogText $warmupLog
        }

        $gpuBeforeMeasurement = Get-GpuSnapshot
        $serverProcess = Get-Process -Id $process.Id -ErrorAction Stop
        $workingSetMiB = [Math]::Round(([double]$serverProcess.WorkingSet64 / 1MB), 1)
        $completion = Invoke-Completion -Port $Port -Tokens $NPredict -MeasureVram
        Start-Sleep -Seconds 1
        $finalLog = Read-CellLog -StandardOutputPath $stdoutPath -StandardErrorPath $stderrPath
        $arenaEvidence = Assert-ArenaEvidence -ArenaState $ArenaState -LogText $finalLog
        $deviceModelBufferEvidence = $null
        if ($MaximumDeviceModelBufferMiB -gt 0) {
            $deviceModelBufferEvidence = Get-DeviceModelBufferEvidence `
                -LogText $finalLog -MaximumMiB $MaximumDeviceModelBufferMiB
        }
        $gpuAfterMeasurement = Get-GpuSnapshot

        $termination = Stop-CellProcess -Process $process
        $record.serverExitCode = $termination.exitCode
        $record.termination = $termination.termination
        $record.terminationEvidence = Assert-ExpectedCellTermination -Termination $termination
        $doneMarkerPath = Join-Path $runDirectory ($Name + '.done')
        ('PASS {0} exit={1} termination={2}' -f
            [DateTime]::UtcNow.ToString('o'), $termination.exitCode, $termination.termination) |
            Out-File -LiteralPath $doneMarkerPath -Encoding ascii

        $record.success = $true
        $record.completedUtc = [DateTime]::UtcNow.ToString('o')
        $record.output = $completion.content
        $record.predictedTokens = $completion.predictedTokens
        $record.decodeTokensPerSecond = $completion.decodeTokensPerSecond
        $record.promptTokensPerSecond = $completion.promptTokensPerSecond
        $record.warmupTokensPerSecond = if ($null -ne $warmupResult) { $warmupResult.decodeTokensPerSecond } else { $null }
        $record.workingSetMiB = $workingSetMiB
        $record.gpuUsedMiBBeforeStart = $gpuBefore.usedMiB
        $record.gpuUsedMiBBeforeMeasurement = $gpuBeforeMeasurement.usedMiB
        $record.gpuUsedMiBAfterMeasurement = $gpuAfterMeasurement.usedMiB
        $record.gpuUsedMiBDelta = $gpuAfterMeasurement.usedMiB - $gpuBefore.usedMiB
        $record.peakDedicatedVramMiB = $completion.peakDedicatedVramMiB
        $record.peakSharedVramMiB = $completion.peakSharedVramMiB
        $record.arenaEvidence = $arenaEvidence
        $record.deviceModelBufferEvidence = $deviceModelBufferEvidence
        $record.doneMarker = $doneMarkerPath
        [void]$script:results.Add([pscustomobject]$record)
        Save-Results

        Write-Host ('  output     : nonempty, {0} fixed greedy tokens' -f $completion.predictedTokens)
        Write-Host ('  decode     : {0:N3} tok/s' -f $completion.decodeTokensPerSecond)
        Write-Host ('  process WS : {0:N1} MiB' -f $workingSetMiB)
        Write-Host ('  GPU used   : {0:N0} -> {1:N0} MiB (total across visible GPUs)' -f
            $gpuBefore.usedMiB, $gpuAfterMeasurement.usedMiB)
        Write-Host ('  peak VRAM  : dedicated {0:N1} MiB, shared {1:N1} MiB' -f
            $completion.peakDedicatedVramMiB, $completion.peakSharedVramMiB)
        return [pscustomobject]$record
    } catch {
        $cellError = $_
        if ($null -ne $process -and $null -eq $termination) {
            try {
                $termination = Stop-CellProcess -Process $process
                $record.serverExitCode = $termination.exitCode
                $record.termination = $termination.termination
            } catch {
                $record.terminationError = $_.Exception.Message
            }
        }
        $record.completedUtc = [DateTime]::UtcNow.ToString('o')
        $record.error = $cellError.Exception.Message
        [void]$script:results.Add([pscustomobject]$record)
        Save-Results
        throw
    } finally {
        if ($null -ne $process) {
            $process.Refresh()
            if (-not $process.HasExited) {
                try {
                    [void](Stop-CellProcess -Process $process)
                } catch {
                    Write-Warning ("Final server cleanup failed: {0}" -f $_.Exception.Message)
                }
            }
            $process.Dispose()
        }
        Clear-ManagedEnvironment
        if ($SettleSeconds -gt 0) {
            Start-Sleep -Seconds $SettleSeconds
        }
    }
}

function Assert-IdenticalOutput {
    param(
        [Parameter(Mandatory = $true)][string]$GateName,
        [Parameter(Mandatory = $true)][object[]]$Cells
    )

    if ($Cells.Count -lt 2) {
        throw "$GateName needs at least two cells."
    }
    $expected = [string]$Cells[0].output
    foreach ($cell in $Cells | Select-Object -Skip 1) {
        if ([string]$cell.output -cne $expected) {
            throw "$GateName failed: deterministic outputs differ ($($Cells[0].name) versus $($cell.name))."
        }
    }
    Write-Host ("PASS: {0} - {1} byte-identical nonempty outputs." -f $GateName, $Cells.Count)
}

function Get-Median {
    param([Parameter(Mandatory = $true)][double[]]$Values)

    if ($Values.Count -eq 0) {
        throw 'Cannot calculate a median of zero values.'
    }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$middle]
    }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Get-MemoryPressureSummary {
    param(
        [Parameter(Mandatory = $true)][object[]]$Cells,
        [Parameter(Mandatory = $true)][bool]$Allowed
    )

    $arenaEvidenceCellCount = 0
    $observedCellNames = @()
    $reportCount = 0
    foreach ($cell in $Cells) {
        if ($null -eq $cell.arenaEvidence -or
            -not ($cell.arenaEvidence.PSObject.Properties.Name -contains 'memoryPressureReportCount')) {
            continue
        }
        $arenaEvidenceCellCount++
        $cellReportCount = [int]$cell.arenaEvidence.memoryPressureReportCount
        $reportCount += $cellReportCount
        if ($cellReportCount -gt 0 -or [bool]$cell.arenaEvidence.memoryPressureObserved) {
            $observedCellNames += [string]$cell.name
        }
    }
    return [pscustomobject]@{
        allowMemoryPressure = $Allowed
        gateEnforced = -not $Allowed
        observed = $reportCount -gt 0
        reportCount = $reportCount
        observedCellCount = $observedCellNames.Count
        observedCells = $observedCellNames
        arenaEvidenceCellCount = $arenaEvidenceCellCount
    }
}

$script:runtimeIdentityByPath = @{}
$script:runtimeIdentityByPath[$ReferenceServer] = Get-RuntimeIdentity `
    -ServerPath $ReferenceServer -Label 'reference runtime'
$script:runtimeIdentityByPath[$CandidateServer] = Get-RuntimeIdentity `
    -ServerPath $CandidateServer -Label 'candidate runtime'
Assert-DistinctRuntimeIdentities `
    -ReferenceIdentity $script:runtimeIdentityByPath[$ReferenceServer] `
    -CandidateIdentity $script:runtimeIdentityByPath[$CandidateServer]

$script:expertMajorMetadataByPath = @{}
foreach ($modelPath in @($DeepSeekModel, $GptOssModel)) {
    if (-not $script:expertMajorMetadataByPath.ContainsKey($modelPath)) {
        $script:expertMajorMetadataByPath[$modelPath] = Get-ExpertMajorMetadataEvidence `
            -ModelPath $modelPath
    }
}

$script:artifactIdentityByPath = @{}
$script:artifactIdentityByPath[$ReferenceServer] = $script:runtimeIdentityByPath[$ReferenceServer].server
$script:artifactIdentityByPath[$CandidateServer] = $script:runtimeIdentityByPath[$CandidateServer].server
$artifactRequests = @(
    @($DeepSeekModel, 'DeepSeek model'),
    @($GptOssModel, 'gpt-oss model')
)
foreach ($request in $artifactRequests) {
    if (-not $script:artifactIdentityByPath.ContainsKey($request[0])) {
        $script:artifactIdentityByPath[$request[0]] = Get-ArtifactIdentity `
            -PathValue $request[0] -Label $request[1]
    }
}
$script:storageIdentityByPath = @{}
foreach ($modelPath in @($DeepSeekModel, $GptOssModel)) {
    if (-not $script:storageIdentityByPath.ContainsKey($modelPath)) {
        $script:storageIdentityByPath[$modelPath] = Get-StorageIdentity -PathValue $modelPath
    }
}

$originalManagedEnvironment = Get-ManagedEnvironment
$configurationPath = Join-Path $runDirectory 'configuration.json'
$configuration = [ordered]@{
    generatedUtc = [DateTime]::UtcNow.ToString('o')
    referenceServer = $ReferenceServer
    candidateServer = $CandidateServer
    deepSeekModel = $DeepSeekModel
    gptOssModel = $GptOssModel
    artifactIdentities = @($script:artifactIdentityByPath.Values)
    runtimeIdentities = @($script:runtimeIdentityByPath.Values)
    expertMajorMetadata = @($script:expertMajorMetadataByPath.Values)
    modelStorage = @($script:storageIdentityByPath.Values)
    deepSeekServerArguments = $DeepSeekServerArguments
    gptOssServerArguments = $GptOssServerArguments
    prompt = $fixedPrompt
    request = [ordered]@{
        nPredict = $NPredict
        warmupTokens = $WarmupTokens
        temperature = 0
        topK = 1
        seed = 42
        cachePrompt = $false
        ignoreEos = $true
    }
    arenaMiB = $ArenaMiB
    allowMemoryPressure = [bool]$AllowMemoryPressure
    gptOssMaxDeviceModelBufferMiB = $GptOssMaxDeviceModelBufferMiB
    performanceStartsPerBuild = 3
    performanceRegressionLimitPercent = $performanceRegressionLimitPercent
    idleGate = [ordered]@{
        minimumFreeMiB = $MinimumFreeMiB
        maximumCpuPercent = $MaxCpuLoadPercent
        maximumGpuPercent = $MaxGpuUtilizationPercent
        maximumDiskPercent = $MaxDiskUtilizationPercent
        maximumFreeMemoryDriftMiB = $MaxFreeMemoryDriftMiB
    }
}
$configuration | ConvertTo-Json -Depth 10 | Out-File -LiteralPath $configurationPath -Encoding utf8
Save-Results

Write-Host ('Raw result directory: {0}' -f $runDirectory)
Write-Host 'Fixed completion request: temperature=0, top_k=1, seed=42, cache_prompt=false, ignore_eos=true'
Write-Host ('Fixed output length: {0} tokens; warmup: {1} tokens' -f $NPredict, $WarmupTokens)
Write-Host ('Regression rule: fail when candidate median is more than {0:N1}% below reference.' -f
    $performanceRegressionLimitPercent)
if ($AllowMemoryPressure) {
    Write-Warning 'Memory-pressure rejection is disabled by explicit request; pressure telemetry is still recorded.'
}

try {
    Clear-ManagedEnvironment

    Write-Host ''
    Write-Host '--- DeepSeek deterministic correctness ---'
    $deepReferenceDisabled = Invoke-Cell -Name 'deepseek-correctness-reference-disabled' `
        -Build Reference -ServerPath $ReferenceServer -ModelPath $DeepSeekModel `
        -Port $DeepSeekPort -ServerArguments $DeepSeekServerArguments -ArenaState Disabled
    $deepCandidateDisabled = Invoke-Cell -Name 'deepseek-correctness-candidate-disabled' `
        -Build Candidate -ServerPath $CandidateServer -ModelPath $DeepSeekModel `
        -Port $DeepSeekPort -ServerArguments $DeepSeekServerArguments -ArenaState Disabled
    $deepCandidateArena = Invoke-Cell -Name 'deepseek-correctness-candidate-arena' `
        -Build Candidate -ServerPath $CandidateServer -ModelPath $DeepSeekModel `
        -Port $DeepSeekPort -ServerArguments $DeepSeekServerArguments -ArenaState Arena
    Assert-IdenticalOutput -GateName 'DeepSeek reference/candidate/arena-disabled equivalence' `
        -Cells @($deepReferenceDisabled, $deepCandidateDisabled, $deepCandidateArena)

    Write-Host ''
    Write-Host '--- DeepSeek three-start interleaved performance ---'
    $performanceRows = @()
    $orders = @(
        @('Reference', 'Candidate'),
        @('Candidate', 'Reference'),
        @('Reference', 'Candidate')
    )
    for ($repetition = 1; $repetition -le 3; $repetition++) {
        foreach ($buildName in $orders[$repetition - 1]) {
            $server = if ($buildName -eq 'Reference') { $ReferenceServer } else { $CandidateServer }
            $cell = Invoke-Cell -Name ('deepseek-performance-{0}-r{1}' -f $buildName.ToLowerInvariant(), $repetition) `
                -Build $buildName -ServerPath $server -ModelPath $DeepSeekModel `
                -Port $DeepSeekPort -ServerArguments $DeepSeekServerArguments -ArenaState Arena `
                -Warmup -RequireIdle
            $performanceRows += $cell
        }
    }

    $referenceRows = @($performanceRows | Where-Object { $_.build -eq 'Reference' -and $_.success })
    $candidateRows = @($performanceRows | Where-Object { $_.build -eq 'Candidate' -and $_.success })
    if ($referenceRows.Count -ne 3 -or $candidateRows.Count -ne 3) {
        throw "Performance gate requires exactly three valid independent starts per build; got reference=$($referenceRows.Count), candidate=$($candidateRows.Count)."
    }
    Assert-IdenticalOutput -GateName 'DeepSeek interleaved performance output equivalence' `
        -Cells @($performanceRows)

    $referenceValues = [double[]]@($referenceRows | ForEach-Object { $_.decodeTokensPerSecond })
    $candidateValues = [double[]]@($candidateRows | ForEach-Object { $_.decodeTokensPerSecond })
    $referenceMedian = Get-Median $referenceValues
    $candidateMedian = Get-Median $candidateValues
    $referenceMinimum = ($referenceValues | Measure-Object -Minimum).Minimum
    $referenceMaximum = ($referenceValues | Measure-Object -Maximum).Maximum
    $candidateMinimum = ($candidateValues | Measure-Object -Minimum).Minimum
    $candidateMaximum = ($candidateValues | Measure-Object -Maximum).Maximum
    $candidateDeltaPercent = 100.0 * ($candidateMedian / $referenceMedian - 1.0)

    Write-Host ('Reference: median {0:N3} tok/s, range {1:N3}-{2:N3}, n=3' -f
        $referenceMedian, $referenceMinimum, $referenceMaximum)
    Write-Host ('Candidate: median {0:N3} tok/s, range {1:N3}-{2:N3}, n=3 ({3:+0.0;-0.0}%)' -f
        $candidateMedian, $candidateMinimum, $candidateMaximum, $candidateDeltaPercent)
    if ($candidateMedian -lt ($referenceMedian * (1.0 - $performanceRegressionLimitPercent / 100.0))) {
        throw ('Performance gate failed: candidate median is {0:N2}% slower than reference (limit {1:N1}%).' -f
            (-$candidateDeltaPercent), $performanceRegressionLimitPercent)
    }

    Write-Host ''
    Write-Host '--- gpt-oss correctness and VRAM smoke ---'
    $gptReference = Invoke-Cell -Name 'gptoss-smoke-reference-arena' `
        -Build Reference -ServerPath $ReferenceServer -ModelPath $GptOssModel `
        -Port $GptOssPort -ServerArguments $GptOssServerArguments -ArenaState Arena `
        -MaximumDeviceModelBufferMiB $GptOssMaxDeviceModelBufferMiB -RequireIdle
    $gptCandidateDisabled = Invoke-Cell -Name 'gptoss-smoke-candidate-disabled' `
        -Build Candidate -ServerPath $CandidateServer -ModelPath $GptOssModel `
        -Port $GptOssPort -ServerArguments $GptOssServerArguments -ArenaState Disabled `
        -MaximumDeviceModelBufferMiB $GptOssMaxDeviceModelBufferMiB -RequireIdle
    $gptCandidate = Invoke-Cell -Name 'gptoss-smoke-candidate-arena' `
        -Build Candidate -ServerPath $CandidateServer -ModelPath $GptOssModel `
        -Port $GptOssPort -ServerArguments $GptOssServerArguments -ArenaState Arena `
        -MaximumDeviceModelBufferMiB $GptOssMaxDeviceModelBufferMiB -RequireIdle
    Assert-IdenticalOutput -GateName 'gpt-oss reference/candidate arena-disabled equivalence' `
        -Cells @($gptReference, $gptCandidateDisabled, $gptCandidate)
    Write-Host ('gpt-oss peak VRAM: dedicated reference/candidate-disabled/candidate-arena {0:N1}/{1:N1}/{2:N1} MiB; shared {3:N1}/{4:N1}/{5:N1} MiB; process WS {6:N1}/{7:N1}/{8:N1} MiB.' -f
        $gptReference.peakDedicatedVramMiB,
        $gptCandidateDisabled.peakDedicatedVramMiB,
        $gptCandidate.peakDedicatedVramMiB,
        $gptReference.peakSharedVramMiB,
        $gptCandidateDisabled.peakSharedVramMiB,
        $gptCandidate.peakSharedVramMiB,
        $gptReference.workingSetMiB,
        $gptCandidateDisabled.workingSetMiB,
        $gptCandidate.workingSetMiB)

    $memoryPressureSummary = Get-MemoryPressureSummary `
        -Cells @($script:results.ToArray()) -Allowed ([bool]$AllowMemoryPressure)
    $summary = [ordered]@{
        passed = $true
        allowMemoryPressure = [bool]$AllowMemoryPressure
        memoryPressure = $memoryPressureSummary
        deepSeekCorrectness = 'byte-identical'
        deepSeekReference = [ordered]@{
            medianTokensPerSecond = $referenceMedian
            minimumTokensPerSecond = $referenceMinimum
            maximumTokensPerSecond = $referenceMaximum
            validIndependentStarts = 3
        }
        deepSeekCandidate = [ordered]@{
            medianTokensPerSecond = $candidateMedian
            minimumTokensPerSecond = $candidateMinimum
            maximumTokensPerSecond = $candidateMaximum
            validIndependentStarts = 3
            deltaPercent = $candidateDeltaPercent
        }
        gptOssCorrectness = 'byte-identical-reference-arena-candidate-disabled-candidate-arena'
        gptOssReferencePeakDedicatedVramMiB = $gptReference.peakDedicatedVramMiB
        gptOssCandidateDisabledPeakDedicatedVramMiB = $gptCandidateDisabled.peakDedicatedVramMiB
        gptOssCandidatePeakDedicatedVramMiB = $gptCandidate.peakDedicatedVramMiB
        gptOssReferencePeakSharedVramMiB = $gptReference.peakSharedVramMiB
        gptOssCandidateDisabledPeakSharedVramMiB = $gptCandidateDisabled.peakSharedVramMiB
        gptOssCandidatePeakSharedVramMiB = $gptCandidate.peakSharedVramMiB
        gptOssReferenceDeviceModelBuffer = $gptReference.deviceModelBufferEvidence
        gptOssCandidateDisabledDeviceModelBuffer = $gptCandidateDisabled.deviceModelBufferEvidence
        gptOssCandidateDeviceModelBuffer = $gptCandidate.deviceModelBufferEvidence
    }
    $summary | ConvertTo-Json -Depth 8 | Out-File -LiteralPath (Join-Path $runDirectory 'summary.json') -Encoding utf8

    $gateDoneMarker = Join-Path $runDirectory 'gate.done'
    $finalPayload = [ordered]@{
        generatedUtc = [DateTime]::UtcNow.ToString('o')
        complete = $true
        passed = $true
        doneMarker = $gateDoneMarker
        cells = @($script:results.ToArray())
    }
    $finalPayload | ConvertTo-Json -Depth 12 | Out-File -LiteralPath $script:resultsPath -Encoding utf8
    ('PASS {0}' -f [DateTime]::UtcNow.ToString('o')) | Out-File -LiteralPath $gateDoneMarker -Encoding ascii
    Write-Host ''
    Write-Host ('PASS: runtime gate complete. Raw evidence: {0}' -f $runDirectory)
} finally {
    Clear-ManagedEnvironment
    foreach ($key in $originalManagedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, [string]$originalManagedEnvironment[$key], 'Process')
    }
}
