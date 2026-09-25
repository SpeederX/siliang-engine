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
    upstreamBase = 'e85e15cf6d810cd1268498c2e5b657bb3ece47bc'
    upstreamTag = 'b11188'
    upstreamRootTree = 'cd9cd8e6821229813a9b5a35673e4d1e2f876d56'
    patchPath = 'patches/siliang-engine.patch'
    patchSha256 = '4CA4C3881011682906F83AB83F618045186EB3F012A7507A9577AE5F3BA8B518'
    patchGitBlob = '1aca5140a0862cb482dc57287d9840fbb022b2be'
    patchInsertions = 11230
    patchDeletions = 38
}

function New-ExpectedEngineFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()]$BaseBlob,
        [Parameter(Mandatory = $true)][string]$FinalBlob,
        [string]$PreviousBlob
    )

    $entry = [ordered]@{
        path = $Path
        status = if ($null -eq $BaseBlob) { 'added' } else { 'modified' }
        mode = '100644'
        baseBlob = $BaseBlob
        finalBlob = $FinalBlob
    }
    if (-not [string]::IsNullOrWhiteSpace($PreviousBlob)) {
        $entry.previousBlob = $PreviousBlob
    }
    return $entry
}

$expectedFiles = @(
    New-ExpectedEngineFile 'common/arg.cpp' '63e342776d549b6abaf9c4aef5543a729e6dfd97' '1f685113f8228efa57445c8e16531dc55dd256ca'
    New-ExpectedEngineFile 'common/common.cpp' '364688ef466dab454e278e4e7c8f85864e39f432' '7e5ff09b17a5e7f0016d9b91c6edba556e751583'
    New-ExpectedEngineFile 'common/common.h' '9194d3dcb650a37c10dade1f10b7f1430936db9c' '9102beb9fe390a49365d2e9016d1d0649fbf44ea'
    New-ExpectedEngineFile 'common/speculative.cpp' '6fdfa4dc33f0514462f8e8d40cdf093a0029a73e' '2eb9ce9c722052ef40fdc986e4687ac1354473aa'
    New-ExpectedEngineFile 'ggml/include/ggml-backend.h' 'cc3f8cd36e3549c4f3ed3d6ad5f6b577af4b837b' '7f5cf0f4f39d27998cbcf9c2ac1dc62fd52dc3dc'
    New-ExpectedEngineFile 'ggml/include/ggml-cpu.h' 'dc6453c6eaa16667f720f987659ad42d03a403a2' '8f1634660f907e8e2d52ad8c20c122e576544184'
    New-ExpectedEngineFile 'ggml/include/ggml-cuda.h' '1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b' '6c4776a86af28a6485d8cfb8b4242b2080e7bd16'
    New-ExpectedEngineFile 'ggml/src/ggml-backend.cpp' '20bf965017e31338f579c1d8d1a64b0ab56c78ee' 'fa80801b1ed78088f5b04d426a40931e1df87161'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu-impl.h' '5dd9ec8e628acad32537e87eccfdeb01c1c1dc46' '699e60a2680883f6872dad7cf26ac72c3eb587e1'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.c' '8bb0ff7bc3366be957dd20aba3fbe0f50dd249d7' '0be59cf484cc81ba3da5c5b0d6c066d1a593487f'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.cpp' '1df0f2bb926894eba96beef691ff2bf520ab70b3' '1dc3a427112d12dd268388fbfc519dd56ccba6d1'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/siliangem_moe_cache.h' $null 'fe88c90c6583eeb1a67f018e04f53c57123bc8ea'
    New-ExpectedEngineFile 'ggml/src/ggml-cuda/ggml-cuda.cu' 'e76ff3128fddd7bf248e16b8469d27c2f441b8e1' '011f69f075910448498ed2fbe0c6d2f7e5b346d0'
    New-ExpectedEngineFile 'include/llama.h' '1805ed0559f92a818dfe951b012608ee2be3115c' '22c232bb3d86d746747514d04ff213400846c68f'
    New-ExpectedEngineFile 'src/CMakeLists.txt' 'afdaddc79de81bc03dadee2707e67d2af4b814c0' '4ff52fe171c6f3f525997cb3156d095792821b14'
    New-ExpectedEngineFile 'src/llama-context.cpp' '8675f6087336069c43dffff0435002bb4358c6f5' '6c2f288db48faabb9c1d33664b68229036bd4440'
    New-ExpectedEngineFile 'src/llama-context.h' 'b403b099b76fefea6f3d8a4959bb12287b633876' '5a147f5459407f34c7a37804ef8c2481ea0e9f71'
    New-ExpectedEngineFile 'src/llama-cparams.h' 'b592de18c79470243ece446cb7db1ca00b0b803b' 'd9bfa8ca9c167e1fdbc56c09299924e607e8054f'
    New-ExpectedEngineFile 'src/llama-graph.cpp' '0b3bab612375b27b357971287857826dcc8b66e8' '158cf8b74993536ea05ebd7a37b2c26fa73a51d1'
    New-ExpectedEngineFile 'src/llama-graph.h' '3daa425bc07bdb2ee9b618124bfa7dfcebc6094f' 'e03a1b59a38ed55d5556a3e457b93d21e9a3b531'
    New-ExpectedEngineFile 'src/llama-model-loader.cpp' '43c396f15af3475e405da8dd73e6a8010ce3d7a5' '9707a9ada9b7b21cf2fa402554a0df394ab29ad6'
    New-ExpectedEngineFile 'src/llama-model-loader.h' '9e51d0ce750503788d2236b4892b8fa3e409b478' 'da89930317474ed8a6ad0f103056d9fd09d49786'
    New-ExpectedEngineFile 'src/llama-model.cpp' 'ab5e744b5d1076c2ad9c4700051ae3bdb9edb024' '371fa9c9750400746f0d02633168194e42ee02a9'
    New-ExpectedEngineFile 'src/llama-model.h' 'a0f9f11423e36699996d1d5f8e56514a0d19b7e8' 'dbc5b6102da3242d25e859cd1c163e3e7cc47a36'
    New-ExpectedEngineFile 'src/llama.cpp' 'ad8e443882ad66e8a68c028f970a0fffd03b92d4' '42440f6ebedebdd3c8da8eb4651ebf0618772cb9'
    New-ExpectedEngineFile 'src/models/deepseek4.cpp' '6bf9d34449426504cea436531118eb9b13941f20' 'bf61c024c1cf005a85c9f7e3c8f21c0abb2514b3'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.cpp' $null '05636ef8d52cbde133dff107f53bd8c35e05c6da'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.h' $null 'a3c1052be580ed7ceecbd3d64d01350e439a5a74'
    New-ExpectedEngineFile 'src/siliang-expert-source.h' $null '72b1d213be6b1d40f687f040e8913e648aeaefd3'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.cpp' $null '33132c85bcc7630c6761fbe18995047dbe05a048'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.h' $null '5bf98bd7fdaa26a64c66f8c4231ea133d6136cf6'
    New-ExpectedEngineFile 'tests/CMakeLists.txt' '9b3a4fcc4bbf6afc77cc8554fa14412722b887b4' '8244dcbb99f2fb414f49a9a4aaf2e6f29bc5906f'
    New-ExpectedEngineFile 'tests/test-arg-parser.cpp' 'e0907631abd8a89e5b6dbadf10d28b65ed483b9a' 'ea164bfe12351845c9bad0e1e3953daafc834017'
    New-ExpectedEngineFile 'tests/test-siliang-prefill.cpp' $null '7d740fe1e709cf8a7985ff418124a732c3f281c8'
    New-ExpectedEngineFile 'tools/server/server-context.cpp' 'e95fb63ab3cca4352f627738edb077ce8413b31f' '2f4325a704f82a07174369d1f3e861cfc67deb8a'
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
        $allowedBlobs = @(@($file.baseBlob, $file.finalBlob) | Where-Object { $null -ne $_ })
        if ($file.Contains('previousBlob')) {
            $allowedBlobs += $file.previousBlob
        }
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
Write-Host ("Fork-root provenance verified ({0}): base {1}; tree {2}; {3} paths; patch +{4}/-{5}; SHA-256 {6}." -f
    $mode,
    $expected.upstreamBase,
    $expected.upstreamRootTree,
    $expectedFiles.Count,
    $insertions,
    $deletions,
    $actualPatchHash)
