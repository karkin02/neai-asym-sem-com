param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "outputs"))
$obsolete = @(
    "demonstrations_spatial_50",
    "demonstrations_diverse_50",
    "demonstrations_hard_40",
    "demonstrations_warehouse_normal_30",
    "demonstrations_smooth",
    "demonstrations_diverse_probe",
    "demonstrations_v2",
    "demonstrations",
    "lerobot_dataset_130_warehouse_replay",
    "lerobot_dataset_100_spatial",
    "lerobot_dataset_100_diverse",
    "lerobot_dataset_90_hard",
    "lerobot_dataset_images",
    "lerobot_dataset_codex",
    "evaluation_90_hard",
    "evaluation_50",
    "evaluation_compare_spatial",
    "evaluation_compare_stage1",
    "evaluation_50_stage2",
    "evaluation_100_diverse",
    "evaluation_100_spatial",
    "evaluation_replan10",
    "evaluation_replan25",
    "screen_v2_warehouse_000100",
    "screen_v2_warehouse_000200",
    "screen_v2_warehouse_000300",
    "screen_v2_warehouse_000400",
    "screen_v2_warehouse_000500",
    "screen_v2_pickplace_000100",
    "screen_v2_pickplace_000200",
    "screen_v2_pickplace_000300",
    "screen_v2_pickplace_000400",
    "screen_v2_pickplace_000500",
    "screen_general200_warehouse_000200",
    "screen_general200_warehouse_000400",
    "screen_general200_warehouse_000600",
    "screen_general200_warehouse_000800",
    "screen_general200_warehouse_001000",
    "screen_general200_pickplace_000200",
    "screen_general200_pickplace_000400",
    "screen_general200_pickplace_000600",
    "screen_general200_pickplace_000800",
    "screen_general200_pickplace_001000",
    "screen_fresh200_warehouse_000500",
    "screen_fresh200_warehouse_001000",
    "screen_fresh200_warehouse_001500",
    "screen_fresh200_warehouse_002000",
    "screen_fresh200_warehouse_002500",
    "screen_fresh200_warehouse_003000",
    "screen_fresh200_pickplace_000500",
    "screen_fresh200_pickplace_001000",
    "screen_fresh200_pickplace_001500",
    "screen_fresh200_pickplace_002000",
    "screen_fresh200_pickplace_002500",
    "screen_fresh200_pickplace_003000",
    "train\smolvla_pickplace\checkpoints",
    "train\smolvla_warehouse_normal_80\checkpoints",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\000500",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\001000",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\002000",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\002500",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\003000",
    "train\smolvla_general_warehouse_200_fresh\checkpoints\last",
    "lerobot_dataset_warehouse_normal_diverse_100",
    "lerobot_dataset_80_warehouse_normal",
    "lerobot_dataset",
    "evaluation_warehouse_normal_80_cpu",
    "evaluation_warehouse80_pickplace_cpu",
    "evaluation_warehouse80_obstacle_cpu",
    "evaluation",
    "python311_recovery",
    "warehouse_demo",
    "confidence_check_normal",
    "confidence_check_normal_121456",
    "confidence_check_obstacle",
    "architecture_split_low_122231072",
    "architecture_split_normal_122230099",
    "incident_report_render"
)

$targets = foreach ($name in $obsolete) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $outputRoot $name))
    if (-not $target.StartsWith(
        $outputRoot + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe output target: $target"
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
    Write-Output "Preview only. The following obsolete folders would be removed:"
    $targets
    Write-Output "Rerun with -Execute to delete them permanently."
    exit 0
}

foreach ($target in $targets) {
    Write-Output "Removing $target"
    Remove-Item -LiteralPath $target -Recurse -Force
}

Write-Output "Obsolete output cleanup complete."
