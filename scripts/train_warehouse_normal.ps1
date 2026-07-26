param(
    [int]$Steps = 1500,
    [int]$BatchSize = 1,
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot "train_smolvla.ps1"
$baseline = Join-Path $projectRoot "outputs\train\smolvla_pickplace_50\checkpoints\last\pretrained_model"
$dataset = Join-Path $projectRoot "outputs\lerobot_dataset_80_warehouse_normal"

if (-not (Test-Path -LiteralPath $baseline)) {
    throw "Selected baseline checkpoint not found at: $baseline"
}
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Warehouse training dataset not found at: $dataset"
}

& $trainer `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -Checkpoint $baseline `
    -DatasetRoot $dataset `
    -DatasetRepoId "local/architecture_a_so101_warehouse_normal_80" `
    -RunName "smolvla_warehouse_normal_80" `
    -Environment $Environment

if ($LASTEXITCODE -ne 0) {
    throw "Warehouse SmolVLA fine-tuning failed."
}
