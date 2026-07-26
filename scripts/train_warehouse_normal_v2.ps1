param(
    [int]$Steps = 500,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.000025,
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot "train_smolvla.ps1"
$baseline = Join-Path $projectRoot "outputs\train\smolvla_pickplace_50\checkpoints\last\pretrained_model"
$dataset = Join-Path $projectRoot "outputs\lerobot_dataset_130_warehouse_replay"

if (-not (Test-Path -LiteralPath $baseline)) {
    throw "Selected baseline checkpoint not found at: $baseline"
}
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Warehouse replay dataset not found at: $dataset"
}

& $trainer `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -Checkpoint $baseline `
    -DatasetRoot $dataset `
    -DatasetRepoId "local/architecture_a_so101_warehouse_replay_130" `
    -RunName "smolvla_warehouse_normal_v2" `
    -LearningRate $LearningRate `
    -WarmupSteps 100 `
    -DecaySteps 1000 `
    -SaveFreq 100 `
    -Environment $Environment

if ($LASTEXITCODE -ne 0) {
    throw "Warehouse SmolVLA v2 fine-tuning failed."
}
