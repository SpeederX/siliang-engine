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
    patchSha256 = '98413ABA53FCD432DE6EB05E0E2D25791F7C26DBBB6888529FFAD2C65B5F4AF1'
    patchGitBlob = '9690f097d6af1c93207f1d7ba151f2b9184a6f5b'
    patchInsertions = 10899
    patchDeletions = 29
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
    New-ExpectedEngineFile 'common/arg.cpp' '86af0ba10a327283f2500f0bb8e48095df547017' '6cbfc21a81c449b46cfa3456f40030161765995b' '21881e865fd7f870b4f25c0cbeeb7b1e707df7dc'
    New-ExpectedEngineFile 'common/common.cpp' 'd9ce5755161edc853dbcf629819ee75971ba413d' '3c25ea9519bca36bffdca6f3ce1230bcce112a38' '702a52c5e1cf0603862c294fe56d82a040de2098'
    New-ExpectedEngineFile 'common/common.h' '3444aa157e9b73727ea2ca6107eb0dc9f9b36a74' '86bbf4a9c35e7abc35e0cd1521a6557d41602ca9'
    New-ExpectedEngineFile 'common/speculative.cpp' '70dc0ac3b1b74fdd5f08b470308786c3f12411e7' '3f7c5f87559b224c038b86f958557abbad2c13a4'
    New-ExpectedEngineFile 'ggml/include/ggml-backend.h' '2924fdbe9884df40abf505fd89d277f5281a835b' '1b45c1ad7cd9618936aaebc243b82e8b48a09b84' 'd04327dbc6ca76b1831ccd1fd8571a107e4982a9'
    New-ExpectedEngineFile 'ggml/include/ggml-cpu.h' 'dc6453c6eaa16667f720f987659ad42d03a403a2' '8f1634660f907e8e2d52ad8c20c122e576544184'
    New-ExpectedEngineFile 'ggml/include/ggml-cuda.h' '1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b' '6c4776a86af28a6485d8cfb8b4242b2080e7bd16' '9377ba5b12d4ce22025852ecef7b5f15b3cab6cc'
    New-ExpectedEngineFile 'ggml/src/ggml-backend.cpp' 'f6fb91798ca484fd1298d7012be3ae8d73cb0ea4' 'f075d0274d7944bb96967f42b8453e1d80e58ec0' '4e3d46d30679d3097395409f0382bcd3ddaea2a3'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu-impl.h' '5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4' '97a2a0b1b72df88a9af3cb6ab825ee4b474a40ae'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.c' '491316f7491252248d6f74a60440d3efa7aa6177' 'ae7c8a129a9f5dc7cd5c16b175261f665ee5e5a8' '83518e829c9f9a72a606514cef178260d3f73a37'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/ggml-cpu.cpp' '16cc5116c5451787c6a1dd1988e38b761f20ef12' 'f39002fc95a5f793e2f2ed68187bfa8b3df8201a'
    New-ExpectedEngineFile 'ggml/src/ggml-cpu/siliangem_moe_cache.h' $null 'fe88c90c6583eeb1a67f018e04f53c57123bc8ea' 'f0d97e499332959d836ec8d97aa4f63f13b5629d'
    New-ExpectedEngineFile 'ggml/src/ggml-cuda/ggml-cuda.cu' '561ab7ac599f9e285d2a0296caee0ab0a14ea5c8' '3ed2efc408af1388904873b03d324c23fa69c6de' 'd3d2cb63b8ec6a9e9621d29a44af510b607423ca'
    New-ExpectedEngineFile 'include/llama.h' 'fb2ca38cee4f8ba84bb6178f1e345e066b0d07e5' 'cdf629e88ebbbff200a44261802e5077b67e7fbd'
    New-ExpectedEngineFile 'src/CMakeLists.txt' '24f05cc91673217726b919229e1626b7f74a7bcb' '4b620618f1304164881ac97692d397fa8152688f'
    New-ExpectedEngineFile 'src/llama-context.cpp' '19cca7df1e9deaafc1e8ee50d0c78ae5ffbc6cfb' 'ecdb8f92ecc2faeee71dee7b460efbf7b194a822'
    New-ExpectedEngineFile 'src/llama-context.h' 'bf91daa8b562aa66d15b08ca559b6baa09ab7855' '7c566f561669941cec1196ee0fc88519c4b173ab'
    New-ExpectedEngineFile 'src/llama-cparams.h' '5018170ed85e3b82abad65e6a3c71859067c9f71' '2de4137861983ca96170eeb71b893ededb9dad29'
    New-ExpectedEngineFile 'src/llama-graph.cpp' '2be3b75fb9825ccc9aa08cda294f46d6422c61ea' '4258d05f5ff0f958dafdb621aeecd7d2530e1f48'
    New-ExpectedEngineFile 'src/llama-graph.h' '32d8d395aa4546ed7e90e7d26d24218fcd37547a' '08c5d6968e6fde56d207c6607fdc6834e97287ca'
    New-ExpectedEngineFile 'src/llama-model-loader.cpp' 'b31e92e2da7ef42eabbb47173bb1f2088c952f39' '630aa17427d80a56559813d67e3d97d25d9a76b9' '709db618f656f22e02fbcea1101db185341a8afb'
    New-ExpectedEngineFile 'src/llama-model-loader.h' 'd6b31c2311186608f48e88d1a37c23adc7e1b0c7' 'fec9ae9ee9550f8a7cb930f9a0334aa3c5c157be' 'aa1772d1377724450c2f3b9b3013e771607f8c3f'
    New-ExpectedEngineFile 'src/llama-model.cpp' 'dda311c47bbf64c0333c2c48b23f16bf24153e42' '1626b09eb0b0c618303c5fe8d583c3b015c7bae0' 'dda311c47bbf64c0333c2c48b23f16bf24153e42'
    New-ExpectedEngineFile 'src/llama-model.h' '6b9e94a0a6921745fd20f58aba38490480c36a38' 'f14b607ded2574b98290f9eb5e9237989e9313d6'
    New-ExpectedEngineFile 'src/llama.cpp' 'd6e0bbfefa729329fe6b83e46e603a85dab0f2e3' '3113ca8d1cacc2a8c42ef815e4aba9a6c22a88bb'
    New-ExpectedEngineFile 'src/models/deepseek4.cpp' '89cd461765adfe8c32fa2e6c6b6d2e962de4b0ac' 'e0ae251cb500c5eb55d6b2c8b36a24a6f6005b84'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.cpp' $null '91d6640557077aa6c2f2afad77d8130df46f2ab9' '8068e77bc74bc1198b79c80deb50431551a03c32'
    New-ExpectedEngineFile 'src/siliang-ds4-front-slab.h' $null 'a3c1052be580ed7ceecbd3d64d01350e439a5a74'
    New-ExpectedEngineFile 'src/siliang-expert-source.h' $null '01b630212a342f9eb0770985995098a049734064'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.cpp' $null 'ab97163e047107215dedbcfa9989d7a97e81b27e' '6cf14959d265c2f97b7ed3846d057bffcead8944'
    New-ExpectedEngineFile 'src/siliang-moe-runtime.h' $null '5bf98bd7fdaa26a64c66f8c4231ea133d6136cf6'
    New-ExpectedEngineFile 'tests/CMakeLists.txt' '419e1eba4c2cdb465d20453004eeeca5af28037f' 'b66f6c8fa62a68fcc95194cf5d5689035056adc7'
    New-ExpectedEngineFile 'tests/test-arg-parser.cpp' 'fd5adb740eab632505cd0a4d999fb55a093a5f84' 'd1c58a368d236b6ded05c7353b6108405e1ec6df'
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
