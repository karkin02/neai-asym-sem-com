param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputs = Join-Path $workspace "outputs"

$patterns = @(
    "architecture_a_*", "architecture_b_*", "architecture_c_*",
    "a_*", "b_*", "c_*", "evaluation_*", "v2_*",
    "integration_validation_*", "single_wrist_*", "active_wrist_*",
    "damage_*", "gif_frame_*", "job011_*", "demo_view_*"
)

$protected = @(
    "train",
    "releases",
    "yolo_warehouse_dataset_v9_damage_balanced",
    "lerobot_dataset_warehouse_v3_150",
    "mujoco_rendered_demo"
)

$targets = foreach ($pattern in $patterns) {
    Get-ChildItem -LiteralPath $outputs -Force -Filter $pattern -ErrorAction SilentlyContinue
}

$obsoleteFiles = @(
    (Join-Path $workspace "yolov8n.pt"),
    (Join-Path $workspace "yolo26n.pt"),
    (Join-Path $outputs "releases\fol-abc-integration-20260802-191632.zip"),
    (Join-Path $outputs "releases\fol-abc-integration-clean-20260802-191713.zip")
)

foreach ($path in $obsoleteFiles) {
    if (Test-Path -LiteralPath $path) {
        $targets += Get-Item -LiteralPath $path -Force
    }
}

$targets = @($targets |
    Where-Object { $_.Name -notin $protected } |
    Sort-Object FullName -Unique)

$totalBytes = 0
foreach ($target in $targets) {
    $resolved = [IO.Path]::GetFullPath($target.FullName)
    if (-not $resolved.StartsWith(
        $workspace + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing unsafe target outside workspace: $resolved"
    }

    if ($target.PSIsContainer) {
        $totalBytes += (Get-ChildItem -LiteralPath $resolved -Recurse -Force -File |
            Measure-Object Length -Sum).Sum
    } else {
        $totalBytes += $target.Length
    }
}

Write-Host "Cleanup targets: $($targets.Count)"
Write-Host "Recoverable size: $([math]::Round($totalBytes / 1MB, 1)) MB"
$targets | ForEach-Object { Write-Host $_.FullName }

if (-not $Execute) {
    Write-Host "Preview only. Re-run with -Execute to delete these paths."
    exit 0
}

foreach ($target in $targets) {
    Remove-Item -LiteralPath $target.FullName -Recurse -Force
}

Write-Host "Cleanup complete."
