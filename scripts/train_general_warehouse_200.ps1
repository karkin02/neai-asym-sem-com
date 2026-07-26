param(
    [int]$Steps = 1000,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.000025,
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot "train_smolvla.ps1"
$baseline = Join-Path $projectRoot "outputs\train\smolvla_pickplace_50\checkpoints\last\pretrained_model"
$dataset = Join-Path $projectRoot "outputs\lerobot_dataset_general_warehouse_balanced_200"

if (-not (Test-Path -LiteralPath $baseline)) {
    throw "Selected baseline checkpoint not found at: $baseline"
}
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Balanced general/warehouse dataset not found at: $dataset"
}

& $trainer `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -Checkpoint $baseline `
    -DatasetRoot $dataset `
    -DatasetRepoId "local/architecture_a_so101_general_warehouse_balanced_200" `
    -RunName "smolvla_general_warehouse_200" `
    -LearningRate $LearningRate `
    -WarmupSteps 200 `
    -DecaySteps 1000 `
    -SaveFreq 200 `
    -Environment $Environment

if ($LASTEXITCODE -ne 0) {
    throw "Balanced general/warehouse SmolVLA training failed."
}
