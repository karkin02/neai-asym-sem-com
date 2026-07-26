param(
    [int]$Steps = 1500,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.00005,
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot "train_smolvla.ps1"
$baseline = Join-Path $projectRoot "outputs\train\smolvla_pickplace_50\checkpoints\last\pretrained_model"
$dataset = Join-Path $projectRoot "outputs\lerobot_dataset_warehouse_normal_diverse_100"

if (-not (Test-Path -LiteralPath $baseline)) {
    throw "Selected baseline checkpoint not found at: $baseline"
}
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Diverse warehouse dataset not found at: $dataset"
}

& $trainer `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -Checkpoint $baseline `
    -DatasetRoot $dataset `
    -DatasetRepoId "local/architecture_a_so101_warehouse_normal_diverse_100" `
    -RunName "smolvla_warehouse_normal_100" `
    -LearningRate $LearningRate `
    -WarmupSteps 300 `
    -DecaySteps 1500 `
    -SaveFreq 300 `
    -Environment $Environment

if ($LASTEXITCODE -ne 0) {
    throw "Diverse warehouse SmolVLA fine-tuning failed."
}
