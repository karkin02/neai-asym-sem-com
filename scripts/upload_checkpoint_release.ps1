param(
    [Parameter(Mandatory = $true)]
    [string]$Repository,

    [string]$Tag = "v1.0.0-checkpoints",

    [string]$Title = "SmolVLA Architecture A checkpoints",

    [switch]$CreateRelease,

    [switch]$Clobber
)

$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$assetRoot = Join-Path $projectRoot "outputs\exports\checkpoints"
$assets = @(
    @{
        Path = Join-Path $assetRoot "smolvla-pickplace-baseline-step1500.zip"
        Sha256 = "22B0BB8E0A1861DBFD5554C9B20A21478B7766815B06F9109E1687A9808324FB"
    },
    @{
        Path = Join-Path $assetRoot "smolvla-warehouse-general200-step1500.zip"
        Sha256 = "27EDFB679720345C577C8F78B3FAC2A24296CC36001A16B50AE6EEA860418BCD"
    }
)

if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw "Repository must use OWNER/REPO format."
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed or is not on PATH. Install it from https://cli.github.com/."
}

& gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run: gh auth login"
}

foreach ($asset in $assets) {
    if (-not (Test-Path -LiteralPath $asset.Path -PathType Leaf)) {
        throw "Missing release asset: $($asset.Path)"
    }

    $actualHash = (Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash
    if ($actualHash -ne $asset.Sha256) {
        throw "SHA256 mismatch for $($asset.Path). Expected $($asset.Sha256), got $actualHash."
    }

    Write-Output "Verified $($asset.Path)"
}

& gh release view $Tag --repo $Repository *> $null
$releaseExists = $LASTEXITCODE -eq 0

if (-not $releaseExists) {
    if (-not $CreateRelease) {
        throw "Release $Tag does not exist. Rerun with -CreateRelease to create it."
    }

    & gh release create $Tag --repo $Repository --title $Title --notes @"
Architecture A SmolVLA checkpoint assets.

- Baseline pick-and-place checkpoint, step 1500
- Balanced warehouse specialist checkpoint, step 1500

These model files are release assets and are intentionally excluded from Git source history.
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create GitHub release $Tag."
    }
}

$uploadArgs = @(
    "release", "upload", $Tag,
    $assets[0].Path,
    $assets[1].Path,
    "--repo", $Repository
)
if ($Clobber) {
    $uploadArgs += "--clobber"
}

& gh @uploadArgs
if ($LASTEXITCODE -ne 0) {
    throw "GitHub release upload failed. Existing assets are not overwritten unless -Clobber is supplied."
}

Write-Output "Uploaded checkpoint assets to $Repository release $Tag."
