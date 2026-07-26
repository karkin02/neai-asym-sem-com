param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$obsoleteNames = @(
    "configs",
    "work",
    ".git",
    ".uv-python"
)

$targets = foreach ($name in $obsoleteNames) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $name))
    if ([System.IO.Path]::GetDirectoryName($target) -ne $projectRoot) {
        throw "Unsafe project cleanup target: $target"
    }
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target -Force
        if (-not $item.PSIsContainer) {
            throw "Cleanup target is not a directory: $target"
        }
        $target
    }
}

if (-not $Execute) {
    Write-Output "Preview only. The following obsolete project folders would be removed:"
    $targets
    Write-Output "Rerun with -Execute to delete them permanently."
    exit 0
}

foreach ($target in $targets) {
    Write-Output "Removing $target"
    Remove-Item -LiteralPath $target -Recurse -Force
}

Write-Output "Obsolete project-folder cleanup complete."
