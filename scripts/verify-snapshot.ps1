[CmdletBinding()]
param(
    # Use while the engine delta and provenance artifacts are intentionally
    # uncommitted. This mode never changes HEAD or the real index.
    [switch]$Authoring
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$manifestPath = Join-Path $repositoryRoot 'docs\source-manifest.json'
$provenancePath = Join-Path $repositoryRoot 'docs\PROVENANCE.md'
$patchPath = Join-Path $repositoryRoot 'patches\siliang-engine.patch'
$artifactPaths = @(
    'docs/source-manifest.json',
    'docs/PROVENANCE.md',
    'patches/siliang-engine.patch',
    'scripts/verify-snapshot.ps1'
)

$expected = [ordered]@{
    schema = 3
    project = 'siliang-engine'
    layout = 'fork-root'
    originUrl = 'https://github.com/SpeederX/siliang-engine'
    upstreamUrl = 'https://github.com/ggml-org/llama.cpp'
    upstreamBase = '07132750825a4f2d27a547cd9cdde1c6f6001885'
    upstreamTag = 'b10270'
    upstreamRootTree = '46f77bf060878e2b3b9d7c43b4d8a3a566ba3384'
    patchPath = 'patches/siliang-engine.patch'
    patchSha256 = 'A4FE4D79DCBF0E17F04979ECECEA08C3C9DC7B7FF90AA959DE44774ADD127FF6'
    patchGitBlob = '81744cd073906cee6819312e8acd03270f745b65'
    patchInsertions = 2303
    patchDeletions = 4
}

$expectedFiles = @(
    [ordered]@{
        path = 'ggml/include/ggml-cpu.h'
        status = 'modified'
        mode = '100644'
        baseBlob = 'dc6453c6eaa16667f720f987659ad42d03a403a2'
        finalBlob = '326ac4abc9333328f03e2bb0e67668c4c797df08'
    },
    [ordered]@{
        path = 'ggml/src/ggml-cpu/ggml-cpu.c'
        status = 'modified'
        mode = '100644'
        baseBlob = '491316f7491252248d6f74a60440d3efa7aa6177'
        finalBlob = '9c18ea720354a1e82be5aa67c286ada4dea96f8d'
    },
    [ordered]@{
        path = 'ggml/src/ggml-cpu/siliangem_moe_cache.h'
        status = 'added'
        mode = '100644'
        baseBlob = $null
        finalBlob = '7b151a326decaec47585bdadf8ca567b616ab868'
    },
    [ordered]@{
        path = 'src/llama-model-loader.cpp'
        status = 'modified'
        mode = '100644'
        baseBlob = 'b31e92e2da7ef42eabbb47173bb1f2088c952f39'
        finalBlob = '7241fd79312f7ff6812cd8492df361b3e637e7ae'
    },
    [ordered]@{
        path = 'src/llama-model-loader.h'
        status = 'modified'
        mode = '100644'
        baseBlob = 'd6b31c2311186608f48e88d1a37c23adc7e1b0c7'
        finalBlob = '47da6c71417e9934f7c3ed40824119bfd9970c0e'
    }
)

function Normalize-GitHubUrl {
    param([Parameter(Mandatory = $true)][string]$Url)

    $normalized = $Url.Trim().Replace('\', '/').TrimEnd('/')
    if ($normalized -match '^git@github\.com:(.+)$') {
        $normalized = 'https://github.com/' + $Matches[1]
    } elseif ($normalized -match '^ssh://git@github\.com/(.+)$') {
        $normalized = 'https://github.com/' + $Matches[1]
    }
    $normalized = $normalized -replace '\.git$', ''
    return $normalized.ToLowerInvariant()
}

function Get-IndexEntry {
    param([Parameter(Mandatory = $true)][string]$Path)

    $lines = @(& git -c core.longpaths=true -C $repositoryRoot ls-files --stage -- $Path)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the index entry for $Path."
    }
    if ($lines.Count -eq 0) {
        return $null
    }
    if ($lines.Count -ne 1 -or $lines[0] -notmatch '^(\d{6}) ([0-9a-f]{40}) (\d+)\t(.+)$') {
        throw "Could not parse the index entry for ${Path}: $($lines -join '; ')"
    }
    if ($Matches[3] -ne '0') {
        throw "Unmerged index entry for ${Path}: stage $($Matches[3])."
    }
    return [pscustomobject]@{
        Mode = $Matches[1]
        Blob = $Matches[2]
    }
}

function Get-HeadEntry {
    param([Parameter(Mandatory = $true)][string]$Path)

    $lines = @(& git -c core.longpaths=true -C $repositoryRoot ls-tree HEAD -- $Path)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect HEAD for $Path."
    }
    if ($lines.Count -eq 0) {
        return $null
    }
    if ($lines.Count -ne 1 -or $lines[0] -notmatch '^(\d{6}) blob ([0-9a-f]{40})\t(.+)$') {
        throw "Could not parse the HEAD entry for ${Path}: $($lines -join '; ')"
    }
    return [pscustomobject]@{
        Mode = $Matches[1]
        Blob = $Matches[2]
    }
}

foreach ($requiredFile in @($manifestPath, $provenancePath, $patchPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required provenance artifact is missing: $requiredFile"
    }
}

& git -c core.longpaths=true -C $repositoryRoot rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Siliang provenance must be verified from a Git worktree.'
}

$originUrl = (& git -C $repositoryRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($originUrl)) {
    throw 'The fork-root repository must have an origin remote.'
}
if ((Normalize-GitHubUrl $originUrl) -cne (Normalize-GitHubUrl $expected.originUrl)) {
    throw "origin points to '$originUrl'; expected '$($expected.originUrl)'."
}

$remoteNames = @(& git -C $repositoryRoot remote)
if ($LASTEXITCODE -ne 0) {
    throw 'Could not list Git remotes.'
}
if ($remoteNames -contains 'upstream') {
    $upstreamUrl = (& git -C $repositoryRoot remote get-url upstream).Trim()
    if ($LASTEXITCODE -ne 0 -or
        (Normalize-GitHubUrl $upstreamUrl) -cne (Normalize-GitHubUrl $expected.upstreamUrl)) {
        throw "upstream points to '$upstreamUrl'; expected '$($expected.upstreamUrl)'."
    }
} else {
    Write-Host '  upstream remote is not configured; validating the recorded upstream object and ancestry.'
}

& git -C $repositoryRoot cat-file -e "$($expected.upstreamBase)^{commit}"
if ($LASTEXITCODE -ne 0) {
    throw "Pinned upstream commit is unavailable: $($expected.upstreamBase). Use a full-history checkout."
}
$baseTree = (& git -C $repositoryRoot rev-parse "$($expected.upstreamBase)^{tree}").Trim()
if ($LASTEXITCODE -ne 0 -or $baseTree -cne $expected.upstreamRootTree) {
    throw "Pinned upstream tree mismatch: expected $($expected.upstreamRootTree), found $baseTree."
}
& git -C $repositoryRoot merge-base --is-ancestor $expected.upstreamBase HEAD
if ($LASTEXITCODE -ne 0) {
    throw "HEAD does not descend from pinned upstream base $($expected.upstreamBase)."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$provenance = Get-Content -LiteralPath $provenancePath -Raw
$metadataChecks = @(
    @('schema', [string]$manifest.schema, [string]$expected.schema),
    @('project', [string]$manifest.project, [string]$expected.project),
    @('layout', [string]$manifest.layout, [string]$expected.layout),
    @('repository.origin', [string]$manifest.repository.origin, [string]$expected.originUrl),
    @('repository.upstream', [string]$manifest.repository.upstream, [string]$expected.upstreamUrl),
    @('upstream.base', [string]$manifest.upstream.base, [string]$expected.upstreamBase),
    @('upstream.tag', [string]$manifest.upstream.tag, [string]$expected.upstreamTag),
    @('upstream.rootTree', [string]$manifest.upstream.rootTree, [string]$expected.upstreamRootTree),
    @('source.canonicalPatch.path', [string]$manifest.source.canonicalPatch.path, [string]$expected.patchPath),
    @('source.canonicalPatch.sha256', [string]$manifest.source.canonicalPatch.sha256, [string]$expected.patchSha256),
    @('source.canonicalPatch.gitBlob', [string]$manifest.source.canonicalPatch.gitBlob, [string]$expected.patchGitBlob)
)
foreach ($check in $metadataChecks) {
    if ($check[1] -cne $check[2]) {
        throw "Manifest $($check[0]) mismatch: expected $($check[2]), found $($check[1])."
    }
}
if ([int]$manifest.source.engineDelta.insertions -ne $expected.patchInsertions -or
    [int]$manifest.source.engineDelta.deletions -ne $expected.patchDeletions) {
    throw ('Manifest engine-delta counts mismatch: expected +{0}/-{1}, found +{2}/-{3}.' -f
        $expected.patchInsertions,
        $expected.patchDeletions,
        $manifest.source.engineDelta.insertions,
        $manifest.source.engineDelta.deletions)
}

$manifestFiles = @($manifest.source.engineDelta.files)
if ($manifestFiles.Count -ne $expectedFiles.Count) {
    throw "Manifest engine file count mismatch: expected $($expectedFiles.Count), found $($manifestFiles.Count)."
}
$manifestByPath = @{}
foreach ($entry in $manifestFiles) {
    $path = [string]$entry.path
    if ([string]::IsNullOrWhiteSpace($path) -or $manifestByPath.ContainsKey($path)) {
        throw "Invalid or duplicate manifest path: $path"
    }
    $manifestByPath[$path] = $entry
}
foreach ($file in $expectedFiles) {
    if (-not $manifestByPath.ContainsKey($file.path)) {
        throw "Manifest is missing engine path: $($file.path)"
    }
    $entry = $manifestByPath[$file.path]
    $actualBaseBlob = if ($null -eq $entry.baseBlob) { $null } else { [string]$entry.baseBlob }
    if ([string]$entry.status -cne $file.status -or
        [string]$entry.mode -cne $file.mode -or
        $actualBaseBlob -cne $file.baseBlob -or
        [string]$entry.finalBlob -cne $file.finalBlob) {
        throw "Manifest blob or mode contract mismatch for $($file.path)."
    }
}

foreach ($identity in @(
    $expected.originUrl,
    $expected.upstreamUrl,
    $expected.upstreamBase,
    $expected.upstreamRootTree,
    $expected.patchSha256,
    $expected.patchGitBlob
)) {
    if (-not $provenance.Contains([string]$identity)) {
        throw "docs/PROVENANCE.md is missing identity: $identity"
    }
}
foreach ($file in $expectedFiles) {
    foreach ($identity in @($file.path, $file.baseBlob, $file.finalBlob)) {
        if ($null -ne $identity -and -not $provenance.Contains([string]$identity)) {
            throw "docs/PROVENANCE.md is missing engine identity: $identity"
        }
    }
}

$actualPatchHash = (Get-FileHash -LiteralPath $patchPath -Algorithm SHA256).Hash
if ($actualPatchHash -ine $expected.patchSha256) {
    throw "Canonical patch SHA-256 mismatch: expected $($expected.patchSha256), found $actualPatchHash."
}
$actualPatchBlob = (& git -C $repositoryRoot hash-object --no-filters -- $patchPath).Trim()
if ($LASTEXITCODE -ne 0 -or $actualPatchBlob -cne $expected.patchGitBlob) {
    throw "Canonical patch Git blob mismatch: expected $($expected.patchGitBlob), found $actualPatchBlob."
}

$patchChangedPaths = @()
$patchNewPaths = @()
$currentPatchPath = $null
$insertions = 0
$deletions = 0
foreach ($line in @(Get-Content -LiteralPath $patchPath)) {
    if ($line -match '^diff --git a/(.+) b/(.+)$') {
        if ($Matches[1] -cne $Matches[2]) {
            throw "Canonical patch contains a rename or mismatched path: $line"
        }
        $currentPatchPath = $Matches[1]
        $patchChangedPaths += $currentPatchPath
    } elseif ($line -match '^new file mode (\d{6})$') {
        if ([string]::IsNullOrWhiteSpace($currentPatchPath) -or $Matches[1] -cne '100644') {
            throw "Unexpected new-file metadata in canonical patch: $line"
        }
        $patchNewPaths += $currentPatchPath
    } elseif ($line.StartsWith('+') -and -not $line.StartsWith('+++')) {
        $insertions++
    } elseif ($line.StartsWith('-') -and -not $line.StartsWith('---')) {
        $deletions++
    }
}
$expectedPaths = @($expectedFiles | ForEach-Object { $_.path } | Sort-Object)
$actualPaths = @($patchChangedPaths | Sort-Object -Unique)
if ($patchChangedPaths.Count -ne $expectedFiles.Count -or
    ($actualPaths -join "`n") -cne ($expectedPaths -join "`n")) {
    throw "Canonical patch path boundary mismatch: $($actualPaths -join ', ')."
}
$expectedNewPaths = @($expectedFiles | Where-Object { $_.status -eq 'added' } | ForEach-Object { $_.path } | Sort-Object)
$actualNewPaths = @($patchNewPaths | Sort-Object -Unique)
if (($actualNewPaths -join "`n") -cne ($expectedNewPaths -join "`n")) {
    throw "Canonical patch new-path boundary mismatch: $($actualNewPaths -join ', ')."
}
if ($insertions -ne $expected.patchInsertions -or $deletions -ne $expected.patchDeletions) {
    throw ('Canonical patch line-count mismatch: expected +{0}/-{1}, found +{2}/-{3}.' -f
        $expected.patchInsertions, $expected.patchDeletions, $insertions, $deletions)
}

# Apply the canonical patch to the pinned base in an isolated temporary index.
# GIT_INDEX_FILE keeps the real worktree and index untouched.
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$temporaryIndex = Join-Path $temporaryRoot ('siliang-index-' + [Guid]::NewGuid().ToString('N'))
$hadIndexOverride = Test-Path Env:GIT_INDEX_FILE
$previousIndexOverride = $env:GIT_INDEX_FILE
try {
    $env:GIT_INDEX_FILE = $temporaryIndex
    & git -c core.longpaths=true -C $repositoryRoot read-tree $expected.upstreamBase
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not populate the isolated index from the pinned upstream base.'
    }
    & git -c core.longpaths=true -c core.autocrlf=false -C $repositoryRoot apply --cached --check --whitespace=nowarn $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Canonical patch does not apply cleanly to the pinned upstream base.'
    }
    & git -c core.longpaths=true -c core.autocrlf=false -C $repositoryRoot apply --cached --whitespace=nowarn $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Canonical patch application failed in the isolated index.'
    }
    $isolatedChangedPaths = @(& git -C $repositoryRoot diff --cached --name-only $expected.upstreamBase -- | Sort-Object)
    if ($LASTEXITCODE -ne 0 -or
        ($isolatedChangedPaths -join "`n") -cne ($expectedPaths -join "`n")) {
        throw "Isolated patch path boundary mismatch: $($isolatedChangedPaths -join ', ')."
    }
    foreach ($file in $expectedFiles) {
        $entry = Get-IndexEntry $file.path
        if ($null -eq $entry -or $entry.Mode -cne $file.mode -or $entry.Blob -cne $file.finalBlob) {
            throw "Isolated patch produced the wrong final entry for $($file.path)."
        }
    }
} finally {
    if ($hadIndexOverride) {
        $env:GIT_INDEX_FILE = $previousIndexOverride
    } else {
        Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $temporaryIndex) {
        $resolvedTemporaryIndex = [IO.Path]::GetFullPath($temporaryIndex)
        $requiredPrefix = $temporaryRoot + [IO.Path]::DirectorySeparatorChar
        if (-not $resolvedTemporaryIndex.StartsWith($requiredPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            (Split-Path -Leaf $resolvedTemporaryIndex) -notmatch '^siliang-index-[0-9a-f]{32}$') {
            throw "Refusing unsafe temporary-index cleanup target: $resolvedTemporaryIndex"
        }
        Remove-Item -LiteralPath $resolvedTemporaryIndex -Force
    }
}

# Validate the actual fork checkout. Authoring mode permits base or final index
# and HEAD entries, while always requiring exact final worktree bytes.
foreach ($file in $expectedFiles) {
    $absolutePath = Join-Path $repositoryRoot ($file.path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        throw "Engine path is missing from the worktree: $($file.path)"
    }
    $worktreeBlob = (& git -C $repositoryRoot hash-object "--path=$($file.path)" -- $file.path).Trim()
    if ($LASTEXITCODE -ne 0 -or $worktreeBlob -cne $file.finalBlob) {
        throw "Worktree blob mismatch for $($file.path): expected $($file.finalBlob), found $worktreeBlob."
    }

    $indexEntry = Get-IndexEntry $file.path
    $headEntry = Get-HeadEntry $file.path
    if ($Authoring) {
        $allowedBlobs = @($file.baseBlob, $file.finalBlob) | Where-Object { $null -ne $_ }
        if ($null -ne $indexEntry -and
            ($indexEntry.Mode -cne $file.mode -or $indexEntry.Blob -notin $allowedBlobs)) {
            throw "Authoring index has an invalid entry for $($file.path): $($indexEntry.Mode) $($indexEntry.Blob)."
        }
        if ($file.status -eq 'modified' -and $null -eq $indexEntry) {
            throw "Authoring index unexpectedly lacks the upstream tracked path $($file.path)."
        }
        if ($null -ne $headEntry -and
            ($headEntry.Mode -cne $file.mode -or $headEntry.Blob -notin $allowedBlobs)) {
            throw "Authoring HEAD has an invalid entry for $($file.path): $($headEntry.Mode) $($headEntry.Blob)."
        }
        if ($file.status -eq 'modified' -and $null -eq $headEntry) {
            throw "Authoring HEAD unexpectedly lacks the upstream tracked path $($file.path)."
        }
    } else {
        if ($null -eq $indexEntry -or $indexEntry.Mode -cne $file.mode -or $indexEntry.Blob -cne $file.finalBlob) {
            throw "Strict index mismatch for $($file.path)."
        }
        if ($null -eq $headEntry -or $headEntry.Mode -cne $file.mode -or $headEntry.Blob -cne $file.finalBlob) {
            throw "Strict HEAD mismatch for $($file.path)."
        }
    }
}

if (-not $Authoring) {
    foreach ($artifactPath in $artifactPaths) {
        $absolutePath = Join-Path $repositoryRoot ($artifactPath.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
            throw "Strict provenance artifact is missing: $artifactPath"
        }
        $worktreeBlob = (& git -C $repositoryRoot hash-object "--path=$artifactPath" -- $artifactPath).Trim()
        $indexEntry = Get-IndexEntry $artifactPath
        $headEntry = Get-HeadEntry $artifactPath
        if ($LASTEXITCODE -ne 0 -or $null -eq $indexEntry -or $null -eq $headEntry -or
            $worktreeBlob -cne $indexEntry.Blob -or $indexEntry.Blob -cne $headEntry.Blob) {
            throw "Strict provenance artifact differs across HEAD, index, and worktree: $artifactPath"
        }
    }
}

$mode = if ($Authoring) { 'authoring' } else { 'strict' }
Write-Host ("Fork-root provenance verified ({0}): base {1}; tree {2}; five paths; patch +{3}/-{4}; SHA-256 {5}." -f
    $mode,
    $expected.upstreamBase,
    $expected.upstreamRootTree,
    $insertions,
    $deletions,
    $actualPatchHash)
