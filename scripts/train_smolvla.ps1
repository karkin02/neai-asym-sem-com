param(
    [int]$Steps = 10000,
    [int]$BatchSize = 8,
    [string]$Checkpoint = "lerobot/smolvla_base",
    [string]$DatasetRoot = "outputs\lerobot_dataset_50",
    [string]$DatasetRepoId = "local/architecture_a_so101_pickplace_50",
    [string]$RunName = "smolvla_pickplace_50",
    [string]$Environment = ".venv312-rocm"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "$Environment\Scripts\python.exe"
$trainer = Join-Path $projectRoot "$Environment\Scripts\lerobot-train.exe"
if (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $projectRoot $DatasetRoot
}
$datasetRoot = $DatasetRoot
$outputDir = Join-Path $projectRoot "outputs\train\$RunName"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Run scripts\setup_py312.ps1 first."
}
if (-not (Test-Path -LiteralPath $datasetRoot)) {
    throw "LeRobot dataset not found at: $datasetRoot"
}

& $python -c "import truststore"
if ($LASTEXITCODE -ne 0) {
    throw "truststore is missing. Install it with: $python -m pip install truststore"
}

$gpuCheck = @'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA-compatible GPU.")
x = torch.ones(4, device="cuda") + 1
torch.cuda.synchronize()
print(torch.__version__, torch.cuda.get_device_name(0), x)
'@
$gpuCheck | & $python -
if ($LASTEXITCODE -ne 0) {
    throw "The GPU kernel preflight failed. Repair the selected environment before training."
}

$env:HF_HOME = Join-Path $projectRoot ".hf-cache"
$env:PYTHONIOENCODING = "utf-8"

$checkpointPath = $Checkpoint
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    $checkpointPath = Join-Path $env:HF_HOME "checkpoints\smolvla_base"
    $env:SMOLVLA_REPO_ID = $Checkpoint
    $env:SMOLVLA_LOCAL_DIR = $checkpointPath
    $downloadCode = @'
import os
import truststore

truststore.inject_into_ssl()

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["SMOLVLA_REPO_ID"],
    local_dir=os.environ["SMOLVLA_LOCAL_DIR"],
)
'@
    $downloadCode | & $python -
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download the SmolVLA checkpoint."
    }
}

$vlmRepoId = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
$vlmPath = Join-Path $env:HF_HOME "checkpoints\smolvlm2_500m_video_instruct"
$processorConfig = Join-Path $vlmPath "processor_config.json"
if (-not (Test-Path -LiteralPath $processorConfig)) {
    $env:SMOLVLM_REPO_ID = $vlmRepoId
    $env:SMOLVLM_LOCAL_DIR = $vlmPath
    $downloadVlmCode = @'
import os
import truststore

truststore.inject_into_ssl()

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["SMOLVLM_REPO_ID"],
    local_dir=os.environ["SMOLVLM_LOCAL_DIR"],
)
'@
    $downloadVlmCode | & $python -
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download the SmolVLM model and processor."
    }
}

if (Test-Path -LiteralPath $outputDir) {
    $suffix = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputDir = "$outputDir-$suffix"
}

Push-Location $projectRoot
try {
    $env:SMOLVLA_CHECKPOINT = $checkpointPath
    $env:SMOLVLA_DATASET_ROOT = $datasetRoot
    $env:SMOLVLA_OUTPUT_DIR = $outputDir
    $env:SMOLVLM_LOCAL_DIR = $vlmPath
    $env:SMOLVLA_BATCH_SIZE = "$BatchSize"
    $env:SMOLVLA_STEPS = "$Steps"
    $env:SMOLVLA_DATASET_REPO_ID = $DatasetRepoId
    $launchCode = @'
import os
import sys
import truststore

truststore.inject_into_ssl()

from lerobot.scripts.lerobot_train import main

sys.argv = [
    "lerobot-train",
    f"--policy.path={os.environ['SMOLVLA_CHECKPOINT']}",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--policy.empty_cameras=1",
    "--policy.use_amp=true",
    "--policy.resize_imgs_with_padding=[256,256]",
    f"--policy.vlm_model_name={os.environ['SMOLVLM_LOCAL_DIR']}",
    '--rename_map={"observation.images.overhead":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}',
    f"--dataset.repo_id={os.environ['SMOLVLA_DATASET_REPO_ID']}",
    f"--dataset.root={os.environ['SMOLVLA_DATASET_ROOT']}",
    f"--output_dir={os.environ['SMOLVLA_OUTPUT_DIR']}",
    "--job_name=architecture_a_smolvla",
    f"--batch_size={os.environ['SMOLVLA_BATCH_SIZE']}",
    f"--steps={os.environ['SMOLVLA_STEPS']}",
    "--num_workers=0",
    "--wandb.enable=false",
]
main()
'@
    $launchCode | & $python -
    if ($LASTEXITCODE -ne 0) {
        throw "SmolVLA training failed."
    }
} finally {
    Pop-Location
}
