param(
    [int]$Episodes = 3,
    [int]$Seed = 1010,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$ReplanEvery = 0,
    [double]$ActionSmoothing = 1.0,
    [double]$MaxJointDelta = 0.0,
    [double]$MaxGripperDelta = 0.0,
    [string]$Environment = ".venv312-rocm",
    [ValidateSet("pick_place", "warehouse_normal", "unexpected_obstacle")]
    [string]$Scene = "warehouse_normal",
    [switch]$Gui,
    [switch]$GuiReplay
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "$Environment\Scripts\python.exe"
$manifestPath = Join-Path $projectRoot "config\architecture_a_model.json"
$evaluator = Join-Path $projectRoot "scripts\evaluate_smolvla.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found at $python"
}
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Architecture A model manifest not found at $manifestPath"
}
if ($Episodes -lt 1) {
    throw "Episodes must be at least 1."
}
if ($ReplanEvery -lt 0) {
    throw "ReplanEvery must be zero or greater."
}
if ($ActionSmoothing -le 0.0 -or $ActionSmoothing -gt 1.0) {
    throw "ActionSmoothing must be in (0, 1]."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$checkpointRelative = $manifest.checkpoint
$checkpointRole = "baseline"
if (
    $Scene -eq "warehouse_normal" -and
    $null -ne $manifest.scenario_checkpoints -and
    $null -ne $manifest.scenario_checkpoints.warehouse_normal
) {
    $checkpointRelative = $manifest.scenario_checkpoints.warehouse_normal.checkpoint
    $checkpointRole = $manifest.scenario_checkpoints.warehouse_normal.status
}
$checkpoint = Join-Path $projectRoot $checkpointRelative
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Selected $checkpointRole checkpoint not found at $checkpoint"
}
Write-Output "Checkpoint role: $checkpointRole"
Write-Output "Checkpoint path: $checkpoint"

if ($Device -eq "cuda") {
    $gpuCheck = @'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ROCm cannot see a GPU.")
x = torch.ones(4, device="cuda") + 1
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0), x)
'@
    $gpuCheck | & $python -
    if ($LASTEXITCODE -ne 0) {
        throw "ROCm GPU preflight failed. Retry with -Device cpu."
    }
}

$output = Join-Path $projectRoot "outputs\architecture_a_demo"
$guiArgument = @()
if ($Gui) {
    $guiArgument = @("--gui")
}
if ($GuiReplay) {
    $guiArgument = @("--gui-replay")
}
Push-Location $projectRoot
try {
    & $python $evaluator `
        --checkpoint $checkpoint `
        --episodes $Episodes `
        --seed $Seed `
        --device $Device `
        --scene $Scene `
        --replan-every $ReplanEvery `
        --action-smoothing $ActionSmoothing `
        --max-joint-delta $MaxJointDelta `
        --max-gripper-delta $MaxGripperDelta `
        --output $output `
        @guiArgument
    if ($LASTEXITCODE -ne 0) {
        throw "Architecture A SmolVLA demo failed."
    }
} finally {
    Pop-Location
}
