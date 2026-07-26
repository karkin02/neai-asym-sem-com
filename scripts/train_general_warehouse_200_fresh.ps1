param(
    [int]$Steps = 3000,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.0001,
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot "train_smolvla.ps1"
$baseCheckpoint = Join-Path $projectRoot ".hf-cache\checkpoints\smolvla_base"
$dataset = Join-Path $projectRoot "outputs\lerobot_dataset_general_warehouse_balanced_200"

if (-not (Test-Path -LiteralPath (Join-Path $baseCheckpoint "config.json"))) {
    throw "Local SmolVLA base checkpoint not found at: $baseCheckpoint"
}
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Balanced general/warehouse dataset not found at: $dataset"
}

& $trainer `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -Checkpoint $baseCheckpoint `
    -DatasetRoot $dataset `
    -DatasetRepoId "local/architecture_a_so101_general_warehouse_balanced_200" `
    -RunName "smolvla_general_warehouse_200_fresh" `
    -LearningRate $LearningRate `
    -WarmupSteps 1000 `
    -DecaySteps 30000 `
    -SaveFreq 500 `
    -Environment $Environment

if ($LASTEXITCODE -ne 0) {
    throw "Fresh balanced general/warehouse SmolVLA training failed."
}
