param(
    [string]$Python312 = "",
    [ValidateSet("cu126", "cu128", "cu130")]
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv312-cuda"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not $Python312) {
    $Python312 = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Python312) {
        throw "Python 3.12 was not found. Install it or pass -Python312 <path>."
    }
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python312 -m venv $venvPath
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install `
    "torch==2.11.0" `
    "torchvision==0.26.0" `
    "torchaudio==2.11.0" `
    --index-url "https://download.pytorch.org/whl/$CudaWheel"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the PyTorch 2.11 $CudaWheel stack."
}

$before = @'
import json, torch
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda}))
'@ | & $venvPython - | ConvertFrom-Json
if (-not $before.cuda) {
    throw "The installed PyTorch wheel is not CUDA-enabled."
}

& $venvPython -m pip install -e "$repoRoot[lerobot]"
if ($LASTEXITCODE -ne 0) {
    throw "Project and LeRobot installation failed."
}

$verify = @'
import json, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch is installed, but no NVIDIA GPU is available.")
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "matrix_ok": bool(torch.isfinite(y).all().item()),
}))
'@ | & $venvPython - | ConvertFrom-Json
if ($verify.torch -ne $before.torch -or $verify.cuda -ne $before.cuda) {
    throw "pip replaced the CUDA PyTorch build during LeRobot installation."
}
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The CUDA LeRobot environment has broken package requirements."
}

Write-Host "NVIDIA LeRobot environment is ready."
Write-Host "Torch: $($verify.torch)"
Write-Host "CUDA runtime: $($verify.cuda)"
Write-Host "GPU: $($verify.gpu)"
