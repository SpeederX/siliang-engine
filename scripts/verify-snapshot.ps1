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
    patchSha256 = 'D8F6141278E3A37B09B25AF79A6AD7DF3A11A4472AF6CFAD955CC7E0FD50DFB3'
    patchGitBlob = 'c611d5db2ff4b05b54c70c9847317f2498d5bd0d'
    patchInsertions = 9070
    patchDeletions = 24
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
    New-ExpectedEngineFile 'common/arg.cpp' '86af0ba10a327283f2500f0bb8e48095df547017' 'eb65102b7f7edd66f27aaaf683721bb19fa1d55e' '750a8ebc623dbc2eeb529c59dc28cf45ba1b23fe'
    New-ExpectedEngineFile 'common/common.cpp' 'd9ce5755161edc853dbcf629819ee75971ba413d' '65c2222ce3d65869c9301946580ed8bf937a77c6'
    New-ExpectedEngineFile 'common/common.h' '3444aa157e9b73727ea2ca6107eb0dc9f9b36a74' 'e785435b2ad7921e243aaf7d2ffb1c2a5ce9fb9b'
    New-ExpectedEngineFile 'common/speculative.cpp' '70dc0ac3b1b74fdd5f08b470308786c3f12411e7' '3f7c5f87559b224c038b86f958557abbad2c13a4'
    New-ExpectedEngineFile 'ggml/include/ggml-backend.h' '2924fdbe9884df40abf505fd89d277f5281a835b' 'd04327dbc6ca76b1831ccd1fd8571a107e4982a9'
    New-ExpectedEngineFile 'ggml/include/ggml-cpu.h' 'dc6453c6eaa16667f720f987659ad42d03a403a2' '11b2df581c908f8f2109e3de53e5e5bcd3a18dfb' '326ac4abc9333328f03e2bb0e67668c4c797df08'
    New-ExpectedEngineFile 'ggml/include/ggml-cuda.h' '1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b' '74ea73681303061a0eacc332a640dd3c8ffd6b44' '626d752466b32d33e3629985e574ed6c42b06900'
    New-ExpectedEngineFile 'ggml/src/ggml-backend.cpp' 'f6fb91798ca484fd1298d7012be3ae8d73cb0ea4' '4e3d46d30679d3097395409f0382bcd3ddaea2a3'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu-impl.h' '5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4' '509b24e675768c03d1120abb9198906f465d064e'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.c' '491316f7491252248d6f74a60440d3efa7aa6177' 'a15513e7715743f5a352ef542145b51eef2a17ae' '9c18ea720354a1e82be5aa67c286ada4dea96f8d'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.cpp' '16cc5116c5451787c6a1dd1988e38b761f20ef12' '527506464c7209742ea256f09096b319370f4f61' '4ab2467b5bbe1a95801c65a3b46e0454ec679bbc'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/siliangem_moe_cache.h' $null 'a44e39f00926be71dbd8273d24374c04711fb2cc' '7b151a326decaec47585bdadf8ca567b616ab868'
    New-ExpectedEngineFile 'ggml/src/ggml-cuda/ggml-cuda.cu' '561ab7ac599f9e285d2a0296caee0ab0a14ea5c8' '23a1ae2f12586e2cab240f068bbbb49095700fda' '944daa425e23aa6f2883bb883a797f4f4709d106'
    New-ExpectedEngineFile 'include/llama.h' 'fb2ca38cee4f8ba84bb6178f1e345e066b0d07e5' 'd19e6c29c54af4a1d9dbb6aadfc2e0214f02d865'
    New-ExpectedEngineFile 'src/CMakeLists.txt' '24f05cc91673217726b919229e1626b7f74a7bcb' '4b620618f1304164881ac97692d397fa8152688f'
    New-ExpectedEngineFile 'src/llama-context.cpp' '19cca7df1e9deaafc1e8ee50d0c78ae5ffbc6cfb' 'c53e9f973241138d34757fbca17d21371d0518b7' 'e41f09245ec35a3f7266b73cbfc9d0b0c473e6af'
    New-ExpectedEngineFile 'src/llama-context.h' 'bf91daa8b562aa66d15b08ca559b6baa09ab7855' '8aae1bde38941c24120a9140cc3559c597f24689'
    New-ExpectedEngineFile 'src/llama-cparams.h' '5018170ed85e3b82abad65e6a3c71859067c9f71' 'bc8a0bdaccb2e9183fc667a779cefd032dd784fe' '9105c1a4556b72208a0db5d902d26dd894bce50c'
    New-ExpectedEngineFile 'src/llama-graph.cpp' '2be3b75fb9825ccc9aa08cda294f46d6422c61ea' 'f83f0c4e7b61aab53567eae30e34a583bb92d45b'
    New-ExpectedEngineFile 'src/llama-graph.h' '32d8d395aa4546ed7e90e7d26d24218fcd37547a' '08c5d6968e6fde56d207c6607fdc6834e97287ca'
    New-ExpectedEngineFile 'src/llama-model-loader.cpp' 'b31e92e2da7ef42eabbb47173bb1f2088c952f39' '709db618f656f22e02fbcea1101db185341a8afb' '80994ea1e4d0cc1fb0e3eb8db2dcb5238d106c97'
    New-ExpectedEngineFile 'src/llama-model-loader.h' 'd6b31c2311186608f48e88d1a37c23adc7e1b0c7' 'aa1772d1377724450c2f3b9b3013e771607f8c3f' '47da6c71417e9934f7c3ed40824119bfd9970c0e'
    New-ExpectedEngineFile 'src/llama-model.h' '6b9e94a0a6921745fd20f58aba38490480c36a38' 'f14b607ded2574b98290f9eb5e9237989e9313d6'
    New-ExpectedEngineFile 'src/llama.cpp' 'd6e0bbfefa729329fe6b83e46e603a85dab0f2e3' '3113ca8d1cacc2a8c42ef815e4aba9a6c22a88bb'
    New-ExpectedEngineFile 'src/models/deepseek4.cpp' '89cd461765adfe8c32fa2e6c6b6d2e962de4b0ac' 'e0ae251cb500c5eb55d6b2c8b36a24a6f6005b84'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.cpp' $null '35ad425cc24f181969aee089a65a37f54a1f6122' 'd0e3ae4e38c584871ae274b78e4bf58dcbde4034'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.h' $null 'a3c1052be580ed7ceecbd3d64d01350e439a5a74'
    New-ExpectedEngineFile 'src/siliang-expert-source.h' $null '01b630212a342f9eb0770985995098a049734064'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.cpp' $null '0890eda725e5e9ca457500f9bbcf4e01814f482b' 'f76550a3ec32def14569d532769ae74182adcec3'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.h' $null '5bf98bd7fdaa26a64c66f8c4231ea133d6136cf6' '70549d1feb0a73d14aec5f1d7c7989c0e55aa713'
    New-ExpectedEngineFile 'tests/CMakeLists.txt' '419e1eba4c2cdb465d20453004eeeca5af28037f' 'b66f6c8fa62a68fcc95194cf5d5689035056adc7'
    New-ExpectedEngineFile 'tests/test-arg-parser.cpp' 'fd5adb740eab632505cd0a4d999fb55a093a5f84' '303883ced6f6c8704c3e4b75c3391a373af61002'
    New-ExpectedEngineFile 'tests/test-siliang-prefill.cpp' $null '7d740fe1e709cf8a7985ff418124a732c3f281c8'
    New-ExpectedEngineFile 'tools/server/server-context.cpp' '5d2798cc14e9295646bb8e570bbec166c9ecc72c' '1162ccd4b1dc37d203f048d9fbe47f73e0fc997a'
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
